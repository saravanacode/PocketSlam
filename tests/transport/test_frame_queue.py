import asyncio

import pytest

from pocketslam.transport.frame_queue import DropOldestQueue


async def test_drops_oldest_when_full() -> None:
    q: DropOldestQueue[int] = DropOldestQueue(2)
    for i in range(5):
        q.put(i)
    assert q.dropped == 3
    assert [await q.get(), await q.get()] == [3, 4]


async def test_get_waits_for_put() -> None:
    q: DropOldestQueue[str] = DropOldestQueue(1)
    getter = asyncio.create_task(q.get())
    await asyncio.sleep(0)
    q.put("frame")
    assert await asyncio.wait_for(getter, 1.0) == "frame"


async def test_close_drains_then_returns_none() -> None:
    q: DropOldestQueue[int] = DropOldestQueue(3)
    q.put(1)
    q.close()
    assert await q.get() == 1
    assert await q.get() is None
    with pytest.raises(RuntimeError, match="closed"):
        q.put(2)


def test_rejects_zero_size() -> None:
    with pytest.raises(ValueError, match="maxsize"):
        DropOldestQueue[int](0)
