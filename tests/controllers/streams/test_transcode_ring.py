"""Tests for the bounded encoded segment ring."""

from __future__ import annotations

import asyncio

import pytest

from music_assistant.controllers.streams.transcode_sessions import (
    EncodedSegment,
    SegmentRing,
    SessionClosed,
)

SAMPLE_RATE = 1000  # 1 sample = 1 ms, keeps the math readable


def _segment(seq: int, *, size: int = 100, ms: int = 100) -> EncodedSegment:
    return EncodedSegment(
        seq=seq,
        kind="media",
        data=b"\x00" * size,
        start_offset=(seq - 1) * size,
        start_samples=(seq - 1) * ms,
        samples=ms,
    )


def _head() -> EncodedSegment:
    return EncodedSegment(
        seq=0, kind="head", data=b"HEAD", start_offset=0, start_samples=0, samples=0
    )


def _ring(
    *,
    max_seconds: float = 1.0,
    max_bytes: int = 10_000,
    floor: int | None = None,
    aborted: list[None] | None = None,
) -> tuple[SegmentRing, list[int]]:
    floor_samples = [floor or 0]
    abort_calls: list[None] = aborted if aborted is not None else []
    ring = SegmentRing(
        SAMPLE_RATE,
        max_seconds,
        max_bytes,
        retention_floor=lambda: floor_samples[0],
        slow_consumer_abort=lambda: abort_calls.append(None),
    )
    return ring, floor_samples


async def test_put_and_window_mapping() -> None:
    """Segments are addressable by byte offset and media sample position."""
    ring, _ = _ring()
    await ring.put(_head())
    for seq in range(1, 6):
        await ring.put(_segment(seq))
    first = ring.first_media_segment
    assert first is not None
    assert first.seq == 1
    assert ring.seq_for_offset(0) == 1
    assert ring.seq_for_offset(99) == 1
    assert ring.seq_for_offset(100) == 2
    assert ring.seq_for_samples(0) == 1
    assert ring.seq_for_samples(150) == 2


async def test_the_window_slides_with_the_retention_floor() -> None:
    """Segments the floor passed are evicted; the resume window stays bounded."""
    ring, floor = _ring(max_seconds=0.3, max_bytes=1_000_000)
    await ring.put(_segment(1))
    await ring.put(_segment(2))
    await ring.put(_segment(3))
    # a consumer drained through 200ms; the next producer pushes evict that part
    floor[0] = 200
    await ring.put(_segment(4))
    await ring.put(_segment(5))
    seqs = [s.seq for s in ring.iter_segments()]
    assert seqs == [3, 4, 5]
    # the evicted window stays convertible through anchors
    assert ring.samples_for_offset(150) == pytest.approx(150, abs=1)


async def test_byte_cap_slides_the_window() -> None:
    """The byte bound evicts consumed segments and keeps the window bounded."""
    ring, floor = _ring(max_seconds=60, max_bytes=250)
    await ring.put(_segment(1, size=100))
    await ring.put(_segment(2, size=100))
    floor[0] = 200
    await ring.put(_segment(3, size=100))
    # segments at/behind the floor are kept until the byte cap needs the space
    assert [s.seq for s in ring.iter_segments()] == [2, 3]
    assert ring.total_bytes <= 250


async def test_producer_blocks_on_a_full_window_then_resumes() -> None:
    """With no floor movement the producer blocks at the cap, then wakes."""
    ring, floor = _ring(max_seconds=60, max_bytes=100)
    await ring.put(_segment(1, size=100))
    # one segment of bounded overshoot (2x) is tolerated ...
    await ring.put(_segment(2, size=100))
    # ... beyond that the producer waits for the floor to advance
    put_task = asyncio.ensure_future(ring.put(_segment(3, size=100)))
    await asyncio.sleep(0.01)
    assert not put_task.done()
    floor[0] = 200
    ring.notify_space()
    await asyncio.wait_for(put_task, timeout=1)
    assert next(s.seq for s in ring.iter_segments()) in (2, 3)


async def test_closing_the_ring_releases_a_waiting_producer() -> None:
    """Termination wakes a producer blocked on ring space."""
    ring, _ = _ring(max_seconds=60, max_bytes=100)
    await ring.put(_segment(1, size=100))
    await ring.put(_segment(2, size=100))
    put_task = asyncio.ensure_future(ring.put(_segment(3, size=100)))
    await asyncio.sleep(0.01)
    assert not put_task.done()
    ring.close()
    with pytest.raises(SessionClosed):
        await put_task


async def test_iterator_stops_at_eof_and_on_eviction() -> None:
    """A consumer iterator ends at producer EOF and refuses evicted cursors."""
    ring, floor = _ring(max_seconds=0.3, max_bytes=1_000_000)
    await ring.put(_segment(1))
    await ring.put(_segment(2))
    await ring.put(_segment(3))
    floor[0] = 200
    await ring.put(_segment(4))
    await ring.put(_segment(5))
    ring.producer_complete()
    # segments 1-2 slid out of the window
    assert [s.seq async for s in ring.iter_from(1)] == []
    # a live cursor reads the retained window to EOF
    assert [s.seq async for s in ring.iter_from(3)] == [3, 4, 5]


async def test_slow_consumer_is_aborted_past_twice_the_cap() -> None:
    """A consumer pinning the floor past 2x cap gets its transport aborted."""
    aborted: list[None] = []
    ring, floor = _ring(max_seconds=60, max_bytes=100, aborted=aborted)
    # floor pinned at 0: first and second segment accepted, third is over 2x
    await ring.put(_segment(1, size=100))
    await ring.put(_segment(2, size=100))
    put_task = asyncio.ensure_future(ring.put(_segment(3, size=100)))
    await asyncio.sleep(0.01)
    assert aborted
    assert not put_task.done()
    # the abort kills the reader; its floor advances and the producer proceeds
    floor[0] = 200
    ring.notify_space()
    await asyncio.wait_for(put_task, timeout=1)
