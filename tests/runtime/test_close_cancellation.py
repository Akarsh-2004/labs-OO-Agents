# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owners must not release shared resources before component cleanup finishes."""

import asyncio

from nooa.runtime.event_manager import EventManager


async def test_repeated_cancellation_and_concurrent_close_wait_for_all_callbacks():
    manager = EventManager()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def first():
        calls.append("first")

    async def second():
        entered.set()
        await release.wait()
        calls.append("second")
        await manager.aclose()  # Recursive owner close must not deadlock.

    manager.on_close(first)
    manager.on_close(second)
    closer = asyncio.create_task(manager.aclose())
    await entered.wait()
    concurrent = asyncio.create_task(manager.aclose())
    try:
        for _ in range(2):
            closer.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not closer.done()
            assert not concurrent.done()
            assert calls == []
    finally:
        release.set()
        results = await asyncio.gather(closer, concurrent, return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)
    assert results[1] is None
    assert calls == ["second", "first"]
    await manager.aclose()
    assert calls == ["second", "first"]
