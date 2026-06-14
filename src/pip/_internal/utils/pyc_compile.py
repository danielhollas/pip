# -------------------------------------------------------------------------- #
# NOTE: Importing from pip's internals or vendored modules should be AVOIDED
#       so this module remains fast to import, minimizing the overhead of
#       spawning a new bytecode compiler worker.
# -------------------------------------------------------------------------- #
from __future__ import annotations

import compileall
import importlib.util
import os
import sys
import warnings
from collections.abc import Callable, Iterable
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, TypeAlias

if TYPE_CHECKING:
    from pip._vendor.typing_extensions import Self

WorkerSetting: TypeAlias = int | Literal["auto"]

CODE_SIZE_THRESHOLD = 1000 * 1000  # 1 MB of .py code
WORKER_LIMIT = 8


class CompileResult(NamedTuple):
    py_path: str
    pyc_path: str
    is_success: bool
    compile_output: str


def _compile_single(py_path: str | Path) -> CompileResult:
    # compile_file() returns True silently even if the source file is nonexistent.
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Python file '{py_path!s}' does not exist")

    with warnings.catch_warnings(), redirect_stdout(StringIO()) as stdout:
        warnings.filterwarnings("ignore")
        success = compileall.compile_file(py_path, force=True, quiet=True)
    pyc_path = importlib.util.cache_from_source(py_path)
    return CompileResult(str(py_path), pyc_path, success, stdout.getvalue())


class BytecodeCompiler(Protocol):
    """Abstraction for compiling Python modules into bytecode in bulk."""

    def __call__(self, paths: Iterable[str]) -> Iterable[CompileResult]: ...

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return


class SerialCompiler(BytecodeCompiler):
    """Compile a set of Python modules one by one in-process."""

    def __call__(self, paths: Iterable[str | Path]) -> Iterable[CompileResult]:
        for p in paths:
            yield _compile_single(p)


class ParallelCompiler(BytecodeCompiler):
    """Compile a set of Python modules using a pool of workers."""

    def __init__(self, workers: int) -> None:
        assert sys.version_info >= (3, 14)
        # Lazy imports to be able to monkeypatch the module in tests
        # Sub-interpreters use threads which have less overhead than OS processes.
        from concurrent import futures

        self.pool = futures.InterpreterPoolExecutor(workers)  # type: ignore[attr-defined]
        self.workers = workers

    def __call__(self, paths: Iterable[str | Path]) -> Iterable[CompileResult]:
        yield from self.pool.map(_compile_single, paths)

    def __exit__(self, *args: object) -> None:
        self.pool.shutdown(wait=False)


def create_bytecode_compiler(
    max_workers: WorkerSetting = "auto",
    code_size_check: Callable[[int], bool] | None = None,
) -> BytecodeCompiler:
    """Return a bytecode compiler appropriate for the workload and platform.

    Parallelization will only be used if:
      - There are 2 or more CPUs available
      - The maximum # of workers permitted is at least 2
      - There is "enough" code to be compiled to offset the worker startup overhead
          (if it can be determined in advance via code_size_check)

    A maximum worker count of "auto" will use the number of CPUs available to the
    process or system, up to a hard-coded limit (to avoid resource exhaustion).

    code_size_check is a callable that receives the code size threshold (in # of
    bytes) for parallelization and returns whether it will be surpassed or not.
    """
    import logging

    logger = logging.getLogger(__name__)

    if sys.version_info < (3, 14):
        logger.info("Bytecode will be compiled serially")
        if max_workers != "auto" and max_workers != 1:
            logger.warning(
                "Parallel bytecode compilation is only available on Python>=3.14"
            )

        return SerialCompiler()

    # New in Python 3.13, but we return early anyway for older pythons
    cpus: int | None = os.process_cpu_count()  # type: ignore

    logger.debug("Detected CPU count: %s", cpus)
    logger.debug("Configured worker count: %s", max_workers)

    # Case 1: Parallelization is disabled or pointless (there's only one CPU).
    if max_workers == 1 or cpus == 1 or cpus is None:
        logger.info("Bytecode will be compiled serially")
        return SerialCompiler()

    # Case 2: There isn't enough code for parallelization to be worth it.
    if code_size_check is not None and not code_size_check(CODE_SIZE_THRESHOLD):
        logger.info("Bytecode will be compiled serially (not enough .py code)")
        return SerialCompiler()

    # Case 3: Attempt to initialize a parallelized compiler.
    # The concurrent executors will spin up new workers on a "on-demand basis",
    # which helps to avoid wasting time on starting new workers that won't be
    # used. (** This isn't true for the fork start method, but forking is
    # fast enough that it doesn't really matter.)
    workers = min(cpus, WORKER_LIMIT) if max_workers == "auto" else max_workers
    try:
        compiler = ParallelCompiler(workers)
        logger.info("Bytecode will be compiled using at most %s workers", workers)
        return compiler
    except (ImportError, NotImplementedError, OSError) as e:
        # Case 4: InterpreterPool is unavailable
        logger.info("Err! Falling back to serial bytecode compilation", exc_info=e)
        return SerialCompiler()
