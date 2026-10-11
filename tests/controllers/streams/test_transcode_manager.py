"""Tests for TranscodeSessionManager negotiation, validation and reclamation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import ContentType, MediaType
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.streamdetails import StreamDetails

if TYPE_CHECKING:
    from aiohttp import web

from music_assistant.controllers.streams import transcode_sessions as ts_mod
from music_assistant.controllers.streams.transcode_sessions import (
    AttachResult,
    ResumeOutcome,
    SessionHooks,
    SessionKey,
    SessionState,
    TerminateReason,
    TranscodeSession,
    TranscodeSessionManager,
)
from music_assistant.models.music_provider import MusicProvider
from tests.controllers.streams.test_transcode_session import _encoded_mp3


@dataclass
class FakeResponse:
    """Minimal response sink."""

    payload: bytearray = field(default_factory=bytearray)
    hang_after: int | None = None
    hang_gate: asyncio.Event | None = None

    async def write(self, data: bytes) -> None:
        """Write or block to simulate a dead socket."""
        if self.hang_after is not None and len(self.payload) >= self.hang_after and self.hang_gate:
            await self.hang_gate.wait()
        self.payload.extend(data)

    def force_close(self) -> None:
        """No-op stand-in."""


SAMPLE_RATE = 44100


def _pcm() -> AudioFormat:
    return AudioFormat(
        content_type=ContentType.from_bit_depth(16),
        sample_rate=SAMPLE_RATE,
        bit_depth=16,
        channels=2,
    )


def _out() -> AudioFormat:
    return AudioFormat(
        content_type=ContentType.MP3,
        sample_rate=SAMPLE_RATE,
        bit_depth=16,
        channels=2,
    )


def _streamdetails() -> StreamDetails:
    details = StreamDetails(
        provider="prov-1",
        item_id="item-1",
        audio_format=_pcm(),
        media_type=MediaType.TRACK,
        duration=200,
        allow_seek=True,
    )
    details.buffer = SimpleNamespace(cancelled=False, has_error=False)
    return details


class Harness:
    """Wires a manager to a fake mass and gated mp3 chains."""

    def __init__(self, raw: bytes) -> None:
        """Initialize the harness."""
        self.raw = raw
        self.mass = MagicMock()
        provider = MagicMock(spec=MusicProvider)
        provider.available = True
        self.mass.get_provider.return_value = provider
        self.manager = TranscodeSessionManager(self.mass)
        self.key = SessionKey("sess-1", "q1", "qi1", "player1", "mp3")
        self.queue_item = MagicMock()
        self.queue_item.uri = "prov-1://track/item-1"
        self.queue_item.streamdetails = _streamdetails()
        self.first_bytes: list[int] = []
        self.completed: list[tuple[int, float]] = []

    def streamdetails(self) -> StreamDetails:
        """Return fresh streamdetails with a healthy buffer."""
        details = _streamdetails()
        self.queue_item.streamdetails = details
        return details

    def hooks(self) -> SessionHooks:
        """Return always-current, counting hooks."""
        return SessionHooks(
            is_current=lambda: True,
            on_first_bytes=lambda: self.first_bytes.append(1),
            on_completed=lambda _b, _s: _record(self.completed, _b, _s),
        )

    def make_chain_factory(
        self, gate: asyncio.Event | None, gate_after: int = 60_000
    ) -> Callable[[int], AsyncGenerator[bytes]]:
        """Build a make_chain emitting the raw encoded bytes behind a gate."""

        def factory(_start_media_ms: int) -> AsyncGenerator[bytes]:
            async def gen() -> AsyncGenerator[bytes]:
                sent = 0
                for pos in range(0, len(self.raw), 4096):
                    if gate is not None and sent >= gate_after:
                        await gate.wait()
                    yield self.raw[pos : pos + 4096]
                    sent += 4096

            return gen()

        return factory

    def request(self, *, range_header: str | None = None, npt: str | None = None) -> web.Request:
        """Build a fake request with the given resume headers."""
        headers = {}
        if range_header is not None:
            headers["Range"] = range_header
        if npt is not None:
            headers["TimeSeekRange.dlna.org"] = npt
        return cast("web.Request", SimpleNamespace(headers=headers, match_info={"fmt": "mp3"}))

    async def resolve(
        self,
        *,
        details: StreamDetails | None = None,
        request: web.Request | None = None,
        gate: asyncio.Event | None = None,
        filter_params: list[str] | None = None,
    ) -> AttachResult:
        """Run one manager.resolve with standard arguments."""
        # stable streamdetails across reconnects, unless a test swaps them
        details = details or self.queue_item.streamdetails
        return await self.manager.resolve(
            key=self.key,
            queue_item=self.queue_item,
            streamdetails=details,
            buffer=details.buffer,
            pcm_format=_pcm(),
            output_format=_out(),
            filter_params=filter_params or [],
            http_profile="default",
            request=request or self.request(),
            duration_s=200.0,
            total_bytes_hint=None,
            make_chain=self.make_chain_factory(gate),
            hooks=self.hooks(),
        )

    async def pump_result(self, result: AttachResult, *, hang: int | None = None) -> FakeResponse:
        """Attach and pump a decision to completion (or a hang/cancel point)."""
        assert result.session is not None
        response = FakeResponse(
            hang_after=hang,
            hang_gate=asyncio.Event() if hang is not None else None,
        )
        consumer = _session_of(result).attach(None)
        await _session_of(result).pump(
            consumer,
            response,
            start_seq=result.start_seq,
            replay_head=result.replay_head,
        )
        return response


async def _record(completed: list[tuple[int, float]], bytes_sent: int, seconds: float) -> None:
    """Stand-in completion recorder."""
    completed.append((bytes_sent, seconds))


def _session_of(result: AttachResult) -> TranscodeSession:
    """Assert and return the session of a decision."""
    assert result.session is not None
    return result.session


async def test_fresh_request_starts_a_new_session(raw_mp3: bytes) -> None:
    """The first GET starts a session and serves the complete entity."""
    harness = Harness(raw_mp3)
    result = await harness.resolve()
    assert result.outcome == ResumeOutcome.FRESH
    assert result.replay_head is True
    assert result.headers["X-MA-Resume"] == "supported"
    response = await harness.pump_result(result)
    assert bytes(response.payload) == raw_mp3
    assert harness.first_bytes == [1]
    assert len(harness.completed) == 1


async def test_rangeless_reconnect_replays_from_the_checkpoint(raw_mp3: bytes) -> None:
    """A second plain GET hits the same session and replays without re-encoding."""
    harness = Harness(raw_mp3)
    gate = asyncio.Event()
    first = await harness.resolve(gate=gate)
    socket_gate = asyncio.Event()
    response_one = FakeResponse(hang_after=20_000, hang_gate=socket_gate)
    consumer = _session_of(first).attach(None)
    pump_task = asyncio.ensure_future(
        _session_of(first).pump(
            consumer,
            response_one,
            start_seq=first.start_seq,
            replay_head=first.replay_head,
        )
    )
    await _spin(3)
    pump_task.cancel()
    with suppress(asyncio.CancelledError):
        await pump_task
    assert _session_of(first).state == SessionState.WAITING_RESUME

    second = await harness.resolve(request=harness.request())
    assert second.outcome == ResumeOutcome.REPLAYED
    assert second.replay_head is True
    assert _session_of(second) is _session_of(first)
    assert second.headers["X-MA-Resume"].startswith("replayed")
    gate.set()
    socket_gate.set()
    response_two = await harness.pump_result(second)
    assert response_two.payload[:3] == raw_mp3[:3]
    assert len(response_two.payload) <= len(raw_mp3)
    assert harness.first_bytes == [1]


async def test_a_byte_range_in_window_continues_with_206(raw_mp3: bytes) -> None:
    """A Range inside the retained window is a 206 without a replayed head."""
    harness = Harness(raw_mp3)
    gate = asyncio.Event()
    first = await harness.resolve(gate=gate)
    # park a consumer so the window is being produced
    socket_gate = asyncio.Event()
    response_one = FakeResponse(hang_after=20_000, hang_gate=socket_gate)
    consumer = _session_of(first).attach(None)
    pump_task = asyncio.ensure_future(
        _session_of(first).pump(consumer, response_one, start_seq=1, replay_head=True)
    )
    await _spin(3)

    # pick a byte offset known to be inside a retained segment
    media = _session_of(first).ring.first_media_segment
    assert media is not None
    target_offset = media.start_offset + 4
    continued = await harness.resolve(
        request=harness.request(range_header=f"bytes={target_offset}-"),
        gate=gate,
    )
    assert continued.outcome == ResumeOutcome.CONTINUED
    assert continued.replay_head is False
    assert "Content-Range" in continued.headers
    assert continued.headers["Accept-Ranges"] == "bytes"
    pump_task.cancel()
    with suppress(asyncio.CancelledError):
        await pump_task
    gate.set()
    socket_gate.set()


async def test_npt_in_window_returns_timeseek_header(raw_mp3: bytes) -> None:
    """DLNA time seeks inside the window answer with timeSeekRange.dlna.org."""
    harness = Harness(raw_mp3)
    gate = asyncio.Event()
    await harness.resolve(gate=gate)
    await _spin(2)
    decision = await harness.resolve(
        request=harness.request(npt="npt=0.5-"),
        gate=gate,
    )
    assert decision.outcome in (ResumeOutcome.CONTINUED, ResumeOutcome.REPLAYED)
    if decision.outcome == ResumeOutcome.CONTINUED:
        assert "timeSeekRange.dlna.org" in decision.headers
    gate.set()


async def test_changed_fingerprint_replaces_the_session(raw_mp3: bytes) -> None:
    """Different DSP/output parameters reject splicing and re-encode fresh."""
    harness = Harness(raw_mp3)
    gate = asyncio.Event()
    first = await harness.resolve(gate=gate)
    await _spin(2)
    details = harness.streamdetails()
    decision = await harness.resolve(
        details=details,
        request=harness.request(),
        gate=asyncio.Event(),
        filter_params=["volume=1.5dB"],
    )
    assert decision.outcome == ResumeOutcome.RE_ENCODED
    assert decision.reencode_reason == "fingerprint"
    assert _session_of(decision) is not _session_of(first)
    assert _session_of(first).state == SessionState.TERMINATED
    assert _session_of(first).terminate_reason == TerminateReason.REPLACED
    gate.set()


async def test_swapped_streamdetails_rejects_resume(raw_mp3: bytes) -> None:
    """A provider capacity reselection (new streamdetails object) means new audio."""
    harness = Harness(raw_mp3)
    gate = asyncio.Event()
    await harness.resolve(gate=gate)
    await _spin(2)
    replacement = _streamdetails()
    decision = await harness.resolve(details=replacement, request=harness.request())
    assert decision.outcome == ResumeOutcome.RE_ENCODED
    assert decision.reencode_reason == "source"
    gate.set()


async def test_range_past_eof_completes(raw_mp3: bytes) -> None:
    """A Range at/after the produced end is answered as a complete, empty entity."""
    harness = Harness(raw_mp3)
    result = await harness.resolve()
    await harness.pump_result(result)  # drains to EOF; session leases the full entity
    await _spin(2)
    decision = await harness.resolve(
        request=harness.request(range_header=f"bytes={len(raw_mp3) + 100}-"),
    )
    assert decision.outcome == ResumeOutcome.COMPLETE


async def test_queue_scoped_termination(raw_mp3: bytes) -> None:
    """terminate_queue_sessions honors the queue/session scope."""
    harness = Harness(raw_mp3)
    gate = asyncio.Event()
    one = await harness.resolve(gate=gate)
    other_key = SessionKey("sess-2", "q1", "qi1", "player1", "mp3")
    other = await harness.manager.resolve(
        key=other_key,
        queue_item=harness.queue_item,
        streamdetails=harness.streamdetails(),
        buffer=harness.queue_item.streamdetails.buffer,
        pcm_format=_pcm(),
        output_format=_out(),
        filter_params=[],
        http_profile="default",
        request=harness.request(),
        duration_s=200.0,
        total_bytes_hint=None,
        make_chain=harness.make_chain_factory(asyncio.Event()),
        hooks=harness.hooks(),
    )
    await harness.manager.terminate_queue_sessions("q1", "sess-1")
    assert _session_of(one).state == SessionState.TERMINATED
    assert _session_of(other).state in (
        SessionState.STARTING,
        SessionState.ACTIVE,
        SessionState.WAITING_RESUME,
    )
    await harness.manager.terminate_queue_sessions("q1", None)
    assert _session_of(other).state == SessionState.TERMINATED
    gate.set()


async def test_terminate_for_item(raw_mp3: bytes) -> None:
    """Swept items release their sessions."""
    harness = Harness(raw_mp3)
    gate = asyncio.Event()
    result = await harness.resolve(gate=gate)
    await harness.manager.terminate_for_item("q1", "qi1")
    assert _session_of(result).state == SessionState.TERMINATED
    assert _session_of(result).terminate_reason == TerminateReason.ITEM_SWEPT
    assert harness.key not in harness.manager._sessions
    gate.set()


async def test_close_terminates_everything(raw_mp3: bytes) -> None:
    """Server shutdown tears down all sessions."""
    harness = Harness(raw_mp3)
    gate = asyncio.Event()
    result = await harness.resolve(gate=gate)
    await harness.manager.close()
    assert _session_of(result).state == SessionState.TERMINATED
    assert _session_of(result).terminate_reason == TerminateReason.SERVER_SHUTDOWN
    gate.set()


async def test_session_count_cap_evicts_waiting_sessions(
    raw_mp3: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-manager cap reaps parked sessions, never active ones."""
    monkeypatch.setattr(ts_mod, "RESUME_MAX_SESSIONS", 2)
    harness = Harness(raw_mp3)
    parked_keys: list[SessionKey] = []
    for index in range(2):
        gate = asyncio.Event()
        key = SessionKey("sess-1", "q1", f"qi{index}", "player1", "mp3")
        parked_keys.append(key)
        await harness.manager.resolve(
            key=key,
            queue_item=harness.queue_item,
            streamdetails=harness.streamdetails(),
            buffer=harness.queue_item.streamdetails.buffer,
            pcm_format=_pcm(),
            output_format=_out(),
            filter_params=[],
            http_profile="default",
            request=harness.request(),
            duration_s=200.0,
            total_bytes_hint=None,
            make_chain=harness.make_chain_factory(gate, gate_after=4000),
            hooks=harness.hooks(),
        )
        # gate is never set: the producer parks in the opening window
    await _spin(2)

    # the third parked session evicts the oldest waiting one to make room
    third_gate = asyncio.Event()
    third_key = SessionKey("sess-1", "q1", "qi-new", "player1", "mp3")
    await harness.manager.resolve(
        key=third_key,
        queue_item=harness.queue_item,
        streamdetails=harness.streamdetails(),
        buffer=harness.queue_item.streamdetails.buffer,
        pcm_format=_pcm(),
        output_format=_out(),
        filter_params=[],
        http_profile="default",
        request=harness.request(),
        duration_s=200.0,
        total_bytes_hint=None,
        make_chain=harness.make_chain_factory(third_gate, gate_after=4000),
        hooks=harness.hooks(),
    )
    assert parked_keys[0] not in harness.manager._sessions
    assert third_key in harness.manager._sessions
    third_gate.set()


async def _spin(ticks: int) -> None:
    """Run the event loop briefly."""
    for _ in range(ticks):
        await asyncio.sleep(0.02)


@pytest.fixture
async def raw_mp3() -> bytes:
    """Provide a short real MP3 encode."""
    return await _encoded_mp3()
