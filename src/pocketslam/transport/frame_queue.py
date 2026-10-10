"""Bounded frame queue that drops the oldest item when full.

Live tracking needs the newest frame, not every frame. When the pipeline falls
behind, old frames are discarded so latency stays bounded.
"""

from __future__ import annotations

import asyncio
from collections import deque


class DropOldestQueue[T]:
    """Single-consumer asyncio queue with a fixed size that never blocks the producer."""

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            msg = f"maxsize must be >= 1, got {maxsize}"
            raise ValueError(msg)
        self._items: deque[T] = deque(maxlen=maxsize)
        self._ready = asyncio.Event()
        self._closed = False
        self.dropped = 0

    def put(self, item: T) -> None:
        """Add an item, discarding the oldest one if the queue is full.

        Raises:
            RuntimeError: If the queue is closed.
        """
        if self._closed:
            msg = "queue is closed"
            raise RuntimeError(msg)
        if len(self._items) == self._items.maxlen:
            self.dropped += 1
        self._items.append(item)
        self._ready.set()

    async def get(self) -> T | None:
        """Wait for the next item.

        Returns:
            The oldest buffered item, or None once the queue is closed and empty.
        """
        while not self._items:
            if self._closed:
                return None
            self._ready.clear()
            await self._ready.wait()
        return self._items.popleft()

    def close(self) -> None:
        """Stop accepting items and wake the consumer."""
        self._closed = True
        self._ready.set()

    def __len__(self) -> int:
        return len(self._items)
