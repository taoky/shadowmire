"""Cooperative cancellation at operation boundaries, owned by the CLI."""

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Empty, SimpleQueue
from types import TracebackType
from typing import ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")
_stop_requested = False
_abandoned_workers = False


class StopRequested(BaseException):
    """Control flow, deliberately excluded from ordinary worker error handlers."""


def request_stop() -> None:
    # Signal handlers must not acquire locks (including threading.Event's lock).
    global _stop_requested
    _stop_requested = True


def reset() -> None:
    global _stop_requested, _abandoned_workers
    _stop_requested = False
    _abandoned_workers = False


def checkpoint() -> None:
    if _stop_requested:
        raise StopRequested


def has_abandoned_workers() -> bool:
    return _abandoned_workers


def cancellable_iter(items: Iterable[T]) -> Iterator[T]:
    checkpoint()
    for item in items:
        checkpoint()
        yield item
    checkpoint()


def cancellable_completed(futures: Iterable[Future[T]]) -> Iterator[Future[T]]:
    # Register once: repeatedly calling wait() over millions of pending futures
    # would rescan the whole set each time a single project finishes.
    completed: SimpleQueue[Future[T]] = SimpleQueue()
    unique = set(futures)
    for future in cancellable_iter(unique):
        future.add_done_callback(completed.put)
    remaining = len(unique)
    while remaining:
        checkpoint()
        try:
            future = completed.get(timeout=0.2)
        except Empty:
            continue
        checkpoint()
        yield future
        remaining -= 1
    checkpoint()


class CancellableExecutor(ThreadPoolExecutor):
    """Let cancellation reach the CLI without waiting for blocked IO threads."""

    def submit(
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        checkpoint()
        return super().submit(fn, *args, **kwargs)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        global _abandoned_workers
        if exc_type is not None and issubclass(
            exc_type, (StopRequested, KeyboardInterrupt)
        ):
            request_stop()
            _abandoned_workers = True
            self.shutdown(wait=False, cancel_futures=True)
            return False
        return super().__exit__(exc_type, exc_val, exc_tb)


def run_in_worker(fn: Callable[[], T]) -> T:
    """Run blocking IO without blocking cancellation on the main thread.

    The callable must not use the main thread's SQLite connection.
    """
    with CancellableExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        for completed in cancellable_completed([future]):
            return completed.result()
    raise AssertionError("worker produced no result")
