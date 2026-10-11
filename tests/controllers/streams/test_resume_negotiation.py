"""Tests for HTTP range/NPT parsing and resumable-session eligibility."""

from __future__ import annotations

from typing import Any

import pytest
from music_assistant_models.enums import ContentType, CrossfadeMode, MediaType

from music_assistant.controllers.streams.transcode_sessions import (
    NPT_NOW,
    TranscodeSessionManager,
    parse_byte_range,
    parse_npt,
)


def test_byte_range_accepts_a_single_open_ended_range() -> None:
    """Only ``bytes=N-`` maps to a resume position."""
    assert parse_byte_range("bytes=0-") == 0
    assert parse_byte_range("bytes=1234-") == 1234
    assert parse_byte_range("BYTES=42-") == 42


@pytest.mark.parametrize(
    "value",
    ["bytes=-500", "bytes=100-200", "bytes=100-200,300-400", "bytes=", "items=1-", "nonsense"],
)
def test_byte_range_rejects_unsupported_forms(value: str) -> None:
    """Suffix, bounded and multipart ranges (and garbage) are ignored."""
    assert parse_byte_range(value) is None


def test_npt_parses_seconds_and_clocktime() -> None:
    """DLNA npt accepts fractional seconds and hh:mm:ss forms."""
    assert parse_npt("npt=12.5-") == 12500
    assert parse_npt("npt=00:01:02-") == 62000
    assert parse_npt("npt=now-") == NPT_NOW


@pytest.mark.parametrize("value", ["npt=abc-", "npt=1:2-", "clock=12.5-", "npt=-"])
def test_npt_rejects_garbage(value: str) -> None:
    """Malformed npt starts are ignored."""
    assert parse_npt(value) is None


def test_npt_ignores_the_end_of_a_bounded_range() -> None:
    """A start-end range resumes at its start; the end is the client's business."""
    assert parse_npt("npt=12.5-13") == 12500


def _eligible_kwargs(**overrides: object) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "method": "GET",
        "media_type": MediaType.TRACK,
        "duration": 200,
        "allow_seek": True,
        "is_realtime": False,
        "crossfade_mode": CrossfadeMode.DISABLED,
        "overlay_is_active": False,
        "playback_speed": 1.0,
        "content_type": ContentType.FLAC,
    }
    kwargs.update(overrides)
    return kwargs


def test_a_regular_seekable_track_is_eligible() -> None:
    """A finite seekable track rendered in splittable codec uses resume sessions."""
    assert TranscodeSessionManager.is_eligible(**_eligible_kwargs()) is True
    assert (
        TranscodeSessionManager.is_eligible(**_eligible_kwargs(content_type=ContentType.MP3))
        is True
    )
    assert (
        TranscodeSessionManager.is_eligible(**_eligible_kwargs(content_type=ContentType.AAC))
        is True
    )
    assert (
        TranscodeSessionManager.is_eligible(**_eligible_kwargs(content_type=ContentType.WAV))
        is True
    )


def test_head_requests_are_eligible_for_header_advertising() -> None:
    """HEAD may advertise ranges; it never creates a session."""
    assert TranscodeSessionManager.is_eligible(**_eligible_kwargs(method="HEAD")) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"method": "OPTIONS"},
        {"media_type": MediaType.RADIO},
        {"media_type": MediaType.AUDIO_SOURCE},
        {"media_type": MediaType.SOUND_EFFECT},
        {"duration": None},
        {"allow_seek": False},
        {"is_realtime": True},
        {"crossfade_mode": CrossfadeMode.SMART_CROSSFADE},
        {"overlay_is_active": True},
        {"playback_speed": 1.5},
        {"content_type": ContentType.OGG},
        {"content_type": ContentType.UNKNOWN},
    ],
)
def test_ineligible_streams_keep_the_direct_path(overrides: dict[str, object]) -> None:
    """Every excluded category falls through to the unchanged direct path."""
    assert TranscodeSessionManager.is_eligible(**_eligible_kwargs(**overrides)) is False
