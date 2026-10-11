"""Tests for the resumable TranscodeSession lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import ContentType, MediaType
from music_assistant_models.errors import AudioError
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.controllers.streams import transcode_sessions as ts_mod
from music_assistant.controllers.streams.transcode_sessions import (
    FailReason,
    FrozenSessionContext,
    SessionHooks,
    SessionKey,
    SessionState,
    TerminateReason,
    TranscodeSession,
)
from music_assistant.helpers.ffmpeg import get_ffmpeg_stream
from music_assistant.models.music_provider import ProviderStreamLimitError

SAMPLE_RATE = 44100


@dataclass
class FakeResponse:
    """Minimal sink standing in for an aiohttp StreamResponse."""

    payload: bytearray = field(default_factory=bytearray)
    fail_after: int | None = None
    hang_after: int | None = None
    hang_gate: asyncio.Event | None = None
    force_closed: bool = False
    writes: int = 0

    async def write(self, data: bytes) -> None:
        """Record bytes, optionally failing or blocking like a slow/dead socket."""
        self.writes += 1
        if self.fail_after is not None and self.writes > self.fail_after:
            raise ConnectionResetError("client disconnected")
        if self.hang_after is not None and self.writes >= self.hang_after and self.hang_gate:
            await self.hang_gate.wait()
        self.payload.extend(data)

    def force_close(self) -> None:
        """Record the forced close."""
        self.force_closed = True


def _silence_pcm(seconds: int) -> AsyncGenerator[bytes]:
    pcm = AudioFormat(
        content_type=ContentType.from_bit_depth(16),
        sample_rate=SAMPLE_RATE,
        bit_depth=16,
        channels=2,
    )

    async def gen() -> AsyncGenerator[bytes]:
        for _ in range(seconds):
            yield b"\x00" * pcm.pcm_sample_size

    return gen()


async def _encoded_mp3(seconds: int = 3) -> bytes:
    pcm = AudioFormat(
        content_type=ContentType.from_bit_depth(16),
        sample_rate=SAMPLE_RATE,
        bit_depth=16,
        channels=2,
    )
    out = AudioFormat(
        content_type=ContentType.MP3,
        sample_rate=SAMPLE_RATE,
        bit_depth=16,
        channels=2,
    )
    data = bytearray()
    async for chunk in get_ffmpeg_stream(
        audio_input=_silence_pcm(seconds), input_format=pcm, output_format=out
    ):
        data.extend(chunk)
    return bytes(data)


def _streamdetails() -> StreamDetails:
    return StreamDetails(
        provider="test_prov",
        item_id="item-1",
        audio_format=AudioFormat(
            content_type=ContentType.FLAC,
            sample_rate=SAMPLE_RATE,
            bit_depth=16,
            channels=2,
        ),
        media_type=MediaType.TRACK,
        duration=200,
        allow_seek=True,
    )


def _output_format() -> AudioFormat:
    return AudioFormat(
        content_type=ContentType.MP3,
        sample_rate=SAMPLE_RATE,
        bit_depth=16,
        channels=2,
    )


def _make_session(
    raw: bytes,
    *,
    is_current: bool = True,
    first_bytes: list[int] | None = None,
    completed: list[tuple[int, float]] | None = None,
    chain_error: Exception | None = None,
    gate: asyncio.Event | None = None,
    gate_after_bytes: int = 8192,
) -> TranscodeSession:
    streamdetails = _streamdetails()
    streamdetails.buffer = MagicMock(cancelled=False, has_error=False)
    first_bytes = first_bytes if first_bytes is not None else []
    completed = completed if completed is not None else []

    def make_chain(_start_ms: int) -> AsyncGenerator[bytes]:
        async def gen() -> AsyncGenerator[bytes]:
            if chain_error is not None:
                raise chain_error
            sent = 0
            # feed in small chunks so the producer goes through many iterations
            for pos in range(0, len(raw), 4096):
                if gate is not None and sent >= gate_after_bytes:
                    await gate.wait()
                yield raw[pos : pos + 4096]
                sent += 4096

        return gen()

    hooks = SessionHooks(
        is_current=lambda: is_current,
        on_first_bytes=lambda: first_bytes.append(1),
        on_completed=lambda b, s: _record_completion(completed, b, s),
    )
    key = SessionKey("sess-1", "q1", "qi1", "player1", "mp3")
    ctx = FrozenSessionContext(
        key=key,
        fingerprint=MagicMock(),
        queue_item=MagicMock(),
        streamdetails=streamdetails,
        buffer=streamdetails.buffer,
        pcm_format=AudioFormat(
            content_type=ContentType.from_bit_depth(16),
            sample_rate=SAMPLE_RATE,
            bit_depth=16,
            channels=2,
        ),
        output_format=_output_format(),
        filter_params=[],
        base_media_ms=0,
        http_profile="default",
        hooks=hooks,
        make_chain=make_chain,
    )
    session = TranscodeSession(ctx)
    session.start()
    return session


async def _record_completion(
    completed: list[tuple[int, float]], bytes_sent: int, seconds: float
) -> None:
    completed.append((bytes_sent, seconds))


async def _pump_until(session: TranscodeSession, response: FakeResponse) -> None:
    consumer = session.attach(None)
    await session.pump(consumer, response, start_seq=1, replay_head=True)


async def test_a_full_stream_runs_head_to_completion(raw_mp3: bytes) -> None:
    """First connection produces the complete entity and completes once."""
    first_bytes: list[int] = []
    completed: list[tuple[int, float]] = []
    session = _make_session(raw_mp3, first_bytes=first_bytes, completed=completed)
    response = FakeResponse()
    await _pump_until(session, response)

    assert bytes(response.payload) == raw_mp3
    # the complete entity stays parked under the resume lease for late reconnects
    assert session.state == SessionState.WAITING_RESUME
    await session.terminate(TerminateReason.LEASE_EXPIRED)
    assert first_bytes == [1]
    assert len(completed) == 1


async def test_disconnect_waits_for_resume_and_replays_without_restart(
    raw_mp3: bytes,
) -> None:
    """A dropped connection parks the session; a reconnect replays from a checkpoint."""
    first_bytes: list[int] = []
    completed: list[tuple[int, float]] = []
    # producer parks halfway so it cannot EOF while the socket is dead
    producer_gate = asyncio.Event()
    session = _make_session(
        raw_mp3,
        first_bytes=first_bytes,
        completed=completed,
        gate=producer_gate,
        gate_after_bytes=60_000,
    )
    consumer = session.attach(None)
    # the socket blocks mid stream, like a player that vanished without a FIN
    socket_gate = asyncio.Event()
    first = FakeResponse(hang_after=3, hang_gate=socket_gate)
    pump_task = asyncio.ensure_future(session.pump(consumer, first, start_seq=1, replay_head=True))
    await _spin(3)
    assert not pump_task.done()
    pump_task.cancel()
    with suppress(asyncio.CancelledError):
        await pump_task
    await asyncio.sleep(0.02)
    assert session.state == SessionState.WAITING_RESUME

    # the producer and its segments are retained across the disconnect
    assert session.ring.total_bytes > 0
    second = FakeResponse()
    consumer_two = session.attach(None)
    assert session.state.value == "active"
    producer_gate.set()
    socket_gate.set()  # release the dead write in case it is still scheduled
    await session.pump(consumer_two, second, start_seq=1, replay_head=True)

    # a fresh entity head plus media from the checkpoint, no bytes past the end
    assert second.payload[:3] == raw_mp3[:3]
    assert len(second.payload) <= len(raw_mp3)
    assert session.state == SessionState.WAITING_RESUME
    # media-start style bookkeeping fires once per session, not per connection
    assert first_bytes == [1]
    assert len(completed) == 1
    await session.terminate(TerminateReason.LEASE_EXPIRED)


async def test_a_session_rotation_terminates_instead_of_waiting(raw_mp3: bytes) -> None:
    """When the queue session rotated, the disconnect is authoritative."""
    current = {"value": True}
    producer_gate = asyncio.Event()
    session = _make_session(raw_mp3, gate=producer_gate, gate_after_bytes=60_000)
    session.ctx.hooks.is_current = lambda: current["value"]
    # block the socket after a few writes, flip the session, then release
    socket_gate = asyncio.Event()
    response = FakeResponse(hang_after=3, hang_gate=socket_gate)
    consumer = session.attach(None)
    pump_task = asyncio.ensure_future(
        session.pump(consumer, response, start_seq=1, replay_head=True)
    )
    await _spin(3)
    current["value"] = False
    socket_gate.set()
    producer_gate.set()
    await asyncio.wait_for(pump_task, timeout=2)
    assert session.terminate_reason == TerminateReason.SESSION_ROTATION


async def test_producer_audio_error_fails_the_session(raw_mp3: bytes) -> None:
    """Transcoder/chain exits classify as TRANSCODER_EXITED and fail consumers."""
    session = _make_session(raw_mp3, chain_error=AudioError("ffmpeg died"))
    response = FakeResponse()
    await _pump_until(session, response)
    assert session.state == SessionState.FAILED
    assert session.fail_reason == FailReason.TRANSCODER_EXITED
    assert response.force_closed


async def test_producer_provider_error_is_distinguishable(raw_mp3: bytes) -> None:
    """Provider source failures carry their own classification."""
    provider = MagicMock()
    provider.max_concurrent_streams = 1
    session = _make_session(raw_mp3, chain_error=ProviderStreamLimitError(provider, 5.0))
    response = FakeResponse()
    await _pump_until(session, response)
    assert session.state == SessionState.FAILED
    assert session.fail_reason == FailReason.PROVIDER_SOURCE


async def test_lease_expiry_terminates_a_session_without_consumers(
    raw_mp3: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the resume lease the generation is torn down completely."""
    monkeypatch.setattr(ts_mod, "RESUME_LEASE_SECONDS", 0.05)
    session = _make_session(raw_mp3, gate=asyncio.Event(), gate_after_bytes=60_000)
    consumer = session.attach(None)
    socket_gate = asyncio.Event()
    pump_task = asyncio.ensure_future(
        session.pump(
            consumer,
            FakeResponse(hang_after=3, hang_gate=socket_gate),
            start_seq=1,
            replay_head=True,
        )
    )
    await _spin(3)
    pump_task.cancel()
    with suppress(asyncio.CancelledError):
        await pump_task
    await asyncio.sleep(0.02)
    assert session.state == SessionState.WAITING_RESUME
    await _spin(15)
    assert session.state.value == "terminated"
    assert session.terminate_reason == TerminateReason.LEASE_EXPIRED
    assert session.ring.closed


async def test_terminate_cancels_the_producer_and_closes_chain(raw_mp3: bytes) -> None:
    """An external terminate ends a running producer (queue stop / server shutdown)."""
    chain_closed = asyncio.Event()
    streamdetails = _streamdetails()
    streamdetails.buffer = MagicMock(cancelled=False, has_error=False)

    async def make_chain(_start_ms: int) -> AsyncGenerator[bytes]:
        try:
            for pos in range(0, len(raw_mp3), 1024):
                await asyncio.sleep(0)
                yield raw_mp3[pos : pos + 1024]
        finally:
            chain_closed.set()

    ctx = FrozenSessionContext(
        key=SessionKey("s", "q", "i", "p", "mp3"),
        fingerprint=MagicMock(),
        queue_item=MagicMock(),
        streamdetails=streamdetails,
        buffer=streamdetails.buffer,
        pcm_format=AudioFormat(
            content_type=ContentType.from_bit_depth(16),
            sample_rate=SAMPLE_RATE,
            bit_depth=16,
            channels=2,
        ),
        output_format=_output_format(),
        filter_params=[],
        base_media_ms=0,
        http_profile="default",
        hooks=SessionHooks(is_current=lambda: True, on_first_bytes=lambda: None),
        make_chain=make_chain,
    )
    session = TranscodeSession(ctx)
    session.start()
    await asyncio.sleep(0)
    await session.terminate(TerminateReason.QUEUE_STOP)
    assert chain_closed.is_set()
    assert session.terminate_reason == TerminateReason.QUEUE_STOP


async def _spin(ticks: int) -> None:
    """Let scheduled callbacks/tasks run a few event-loop iterations."""
    for _ in range(ticks):
        await asyncio.sleep(0.01)


@pytest.fixture
async def raw_mp3() -> bytes:
    """Encode a short silent MP3 with the real encoder for the session to split."""
    return await _encoded_mp3()
