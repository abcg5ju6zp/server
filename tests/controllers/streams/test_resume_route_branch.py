"""Tests for the serve_queue_item_stream fork into resumable sessions."""

from __future__ import annotations

from types import MethodType
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from music_assistant_models.enums import ContentType, CrossfadeMode, MediaType

import music_assistant.controllers.streams.controller as controller_mod
from music_assistant.controllers.streams.controller import StreamsController
from music_assistant.controllers.streams.transcode_sessions import (
    AttachResult,
    ResumeOutcome,
    SessionState,
)

QUEUE_ID = "q1"
SESSION_ID = "sess1"
ITEM_ID = "item1"
PLAYER_ID = "player1"


class FakeStreamResponse:
    """Stand-in recording prepare instead of touching a socket."""

    def __init__(
        self,
        status: int = 200,
        reason: str = "OK",
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the fake response."""
        self.status = status
        self.reason = reason
        self.headers: dict[str, str] = headers or {}
        self.content_type: object = ""
        self.content_length = None
        self.prepared = False
        self.chunked = False
        self.force_closed_value = False

    def enable_chunked_encoding(self) -> None:
        """Record chunked encoding."""
        self.chunked = True

    def force_close(self) -> None:
        """Record a forced close."""
        self.force_closed_value = True

    async def prepare(self, request: object) -> None:
        """Pretend the response was sent."""
        self.prepared = True


def _request(method: str = "GET", fmt: str = "mp3") -> MagicMock:
    """Build a minimal request for the single route."""
    request = MagicMock()
    request.method = method
    request.headers = {}
    request.match_info = {
        "queue_id": QUEUE_ID,
        "player_id": PLAYER_ID,
        "session_id": SESSION_ID,
        "queue_item_id": ITEM_ID,
        "fmt": fmt,
    }
    request.transport = None
    return request


def _audio_format(content_type: ContentType) -> MagicMock:
    """Build a lightweight AudioFormat mock."""
    fmt = MagicMock()
    fmt.content_type = content_type
    fmt.sample_rate = 44100
    fmt.bit_depth = 16
    fmt.channels = 2
    fmt.pcm_sample_size = 44100 * 2 * 2
    fmt.output_format_str = content_type.value
    return fmt


def _controller(media_type: MediaType, *, allow_seek: bool = True) -> MagicMock:
    """Build a controller mock wired through to the resume fork."""
    ctrl = MagicMock()
    ctrl.logger = Mock()
    ctrl._log_request = Mock()
    ctrl._open_item_streams = {}
    ctrl._active_output_streams = 0
    ctrl._raise_if_stale_item_request = MethodType(
        StreamsController._raise_if_stale_item_request, ctrl
    )
    ctrl._serve_resumable_item_stream = MethodType(
        StreamsController._serve_resumable_item_stream, ctrl
    )
    ctrl.get_crossfade_mode = Mock(return_value=CrossfadeMode.DISABLED)

    queue = MagicMock()
    queue.queue_id = QUEUE_ID
    queue.display_name = "Q1"
    ctrl.mass.player_queues.get.return_value = queue
    pq_data = MagicMock()
    pq_data.session_id = SESSION_ID
    ctrl.mass.player_queues.queue_data.return_value = pq_data

    player = MagicMock()
    player.player_id = PLAYER_ID
    player.display_name = "P1"
    player.state.group_members = []
    player.get_config_value.return_value = "default"
    ctrl.mass.players.get_player.return_value = player

    streamdetails = MagicMock()
    streamdetails.allow_seek = allow_seek
    streamdetails.is_realtime = False
    streamdetails.seek_position = 0
    streamdetails.audio_format.bit_depth = 16
    queue_item = MagicMock()
    queue_item.media_item = MagicMock()
    queue_item.media_type = media_type
    queue_item.duration = 200 if media_type != MediaType.RADIO else None
    queue_item.name = "Track"
    queue_item.streamdetails = streamdetails
    queue_item.extra_attributes = {"playback_speed": 1.0}
    ctrl.mass.player_queues.get_item.return_value = queue_item

    ctrl.audio.select_pcm_format = AsyncMock(return_value=_audio_format(ContentType.MP3))
    output_format = _audio_format(
        ContentType.MP3 if media_type != MediaType.RADIO else ContentType.OGG
    )
    ctrl.audio.get_output_format = AsyncMock(return_value=output_format)
    ctrl.audio.get_player_output_plan = Mock(return_value=MagicMock(filter_params=[]))
    ctrl._update_audio_processing_context = Mock()

    ctrl.transcode_sessions = MagicMock()
    ctrl.transcode_sessions.resolve = AsyncMock()
    return ctrl


async def test_eligible_track_is_routed_to_a_resume_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A seekable MP3 track resolves through the session manager and returns 200."""
    monkeypatch.setattr(
        "music_assistant.controllers.streams.controller.web.StreamResponse",
        FakeStreamResponse,
    )
    monkeypatch.setattr(controller_mod, "overlay_active", lambda _queue: False)
    ctrl = _controller(MediaType.TRACK)
    session = MagicMock()
    session.state = SessionState.WAITING_RESUME
    session.attach.return_value = MagicMock()
    session.pump = AsyncMock()
    decision = AttachResult(
        outcome=ResumeOutcome.FRESH,
        session=session,
        start_seq=1,
        replay_head=True,
        content_offset=0,
        content_samples=0,
        total_bytes=None,
        headers={"X-MA-Resume": "supported"},
    )
    ctrl.transcode_sessions.resolve.return_value = decision

    response = await StreamsController.serve_queue_item_stream(ctrl, _request())

    ctrl.transcode_sessions.resolve.assert_awaited_once()
    assert response.prepared
    assert response.status == 200
    assert response.headers["X-MA-Resume"] == "supported"
    session.attach.assert_called_once()
    session.pump.assert_awaited_once()


async def test_a_continued_decision_returns_206(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CONTINUED decision is prepared with status 206 and no head replay."""
    monkeypatch.setattr(
        "music_assistant.controllers.streams.controller.web.StreamResponse",
        FakeStreamResponse,
    )
    monkeypatch.setattr(controller_mod, "overlay_active", lambda _queue: False)
    ctrl = _controller(MediaType.TRACK)
    session = MagicMock()
    session.state = SessionState.ACTIVE
    session.attach.return_value = MagicMock()
    session.pump = AsyncMock()
    decision = AttachResult(
        outcome=ResumeOutcome.CONTINUED,
        session=session,
        start_seq=12,
        replay_head=False,
        content_offset=40_000,
        content_samples=441_000,
        total_bytes=4_000_000,
        headers={
            "X-MA-Resume": "continued; npt=10",
            "Content-Range": "bytes 40000-3999999/4000000",
        },
    )
    ctrl.transcode_sessions.resolve.return_value = decision

    response = await StreamsController.serve_queue_item_stream(ctrl, _request())

    assert response.status == 206
    assert response.headers["Content-Range"] == "bytes 40000-3999999/4000000"
    kwargs = session.pump.await_args.kwargs
    assert kwargs["start_seq"] == 12
    assert kwargs["replay_head"] is False


async def test_radio_never_touches_the_resume_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ineligible streams keep the direct path even if it then fails downstream."""
    monkeypatch.setattr(
        "music_assistant.controllers.streams.controller.web.StreamResponse",
        FakeStreamResponse,
    )
    ctrl = _controller(MediaType.RADIO)
    # make the direct path stop at a recognizable point
    ctrl.audio.get_queue_item_stream = Mock(side_effect=RuntimeError("direct path"))

    with pytest.raises(RuntimeError, match="direct path"):
        await StreamsController.serve_queue_item_stream(ctrl, _request(fmt="ogg"))

    ctrl.transcode_sessions.resolve.assert_not_called()
