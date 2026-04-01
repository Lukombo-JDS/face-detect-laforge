from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Callable
from uuid import uuid4

from app.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class TaskResult:
    status: str
    result: Any = None
    error: str | None = None


class BackgroundTaskRunner:
    def __init__(self) -> None:
        self._queue: Queue[tuple[str, Callable[..., Any], tuple[Any, ...], dict[str, Any]]] = Queue()
        self._results: dict[str, TaskResult] = {}
        self._stop_event = Event()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        task_id = str(uuid4())
        self._results[task_id] = TaskResult(status="queued")
        self._queue.put((task_id, func, args, kwargs))
        return task_id

    def status(self, task_id: str) -> TaskResult:
        return self._results.get(task_id, TaskResult(status="unknown", error="task_id inconnu"))

    def shutdown(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                task_id, func, args, kwargs = self._queue.get(timeout=0.1)
            except Empty:
                continue

            self._results[task_id] = TaskResult(status="running")
            try:
                result = func(*args, **kwargs)
                self._results[task_id] = TaskResult(status="done", result=result)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("background_task_failed")
                self._results[task_id] = TaskResult(status="failed", error=str(exc))
            finally:
                self._queue.task_done()
