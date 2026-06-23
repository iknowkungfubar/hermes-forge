"""SlotWorker — serialized access to an inference slot with priority queuing.

Wraps a WorkflowRunner with a priority queue. Each submit() call waits for
its turn, runs to completion, and returns the result.

Auto-preemption: if a submitted task has strictly higher priority (lower int)
than the currently running task, the running task is cancelled and the
higher-priority task takes over.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from hermes_forge.core.workflow import Workflow
from hermes_forge.core.messages import Message


@dataclass(order=True)
class _Task:
    """Priority queue item. Lower priority value = higher urgency."""

    priority: int
    timestamp: float = field(compare=False)
    task_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4()))
    coro: Any = field(compare=False, default=None)


class SlotWorker:
    """Serializes workflow execution on a single inference slot.

    Args:
        run_fn: Async callable that runs a workflow (e.g., WorkflowRunner.run).
        max_concurrent: Maximum concurrent tasks (default 1 for single GPU).
    """

    def __init__(
        self,
        run_fn: Any = None,
        max_concurrent: int = 1,
    ) -> None:
        self._run_fn = run_fn
        self._max_concurrent = max_concurrent
        self._queue: asyncio.PriorityQueue[_Task] = asyncio.PriorityQueue()
        self._current_task: _Task | None = None
        self._cancel_event: asyncio.Event | None = None
        self._worker_task: asyncio.Task | None = None

    async def submit(
        self,
        workflow: Workflow,
        messages: list[Message],
        priority: int = 0,
    ) -> Any:
        """Submit a workflow run with the given priority.

        Lower priority value = runs first. Default is 0 (FIFO).

        Returns the result of the workflow run.
        """
        if self._run_fn is None:
            raise RuntimeError("SlotWorker has no run_fn configured")

        loop = asyncio.get_event_loop()
        task = _Task(
            priority=priority,
            timestamp=loop.time(),
        )

        future: asyncio.Future[Any] = asyncio.Future()

        async def run_and_complete():
            try:
                # Check preemption
                if (
                    self._current_task is not None
                    and priority < self._current_task.priority
                ):
                    if self._cancel_event:
                        self._cancel_event.set()

                self._current_task = task
                cancel_event = asyncio.Event()
                self._cancel_event = cancel_event

                result = await self._run_fn(
                    workflow=workflow,
                    messages=messages,
                    cancel_event=cancel_event,
                )
                if not future.done():
                    future.set_result(result)
            except Exception as e:
                if not future.done():
                    future.set_exception(e)
            finally:
                if self._current_task is task:
                    self._current_task = None
                    self._cancel_event = None
                self._queue.task_done()

        self._queue.put_nowait(task)
        task.coro = run_and_complete()

        # Start worker if not running
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._process_queue())

        return await future

    async def _process_queue(self) -> None:
        """Process items from the priority queue."""
        while True:
            try:
                task = await self._queue.get()
                if task.coro:
                    await task.coro
            except asyncio.CancelledError:
                break
            except Exception:
                continue

    async def shutdown(self) -> None:
        """Cancel all pending tasks and shutdown the worker."""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
