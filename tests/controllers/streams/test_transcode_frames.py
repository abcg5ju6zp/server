"""Tests for frame-aligned splitting of real ffmpeg encoder output."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from music_assistant_models.enums import ContentType
from music_assistant_models.media_items import AudioFormat

from music_assistant.controllers.streams.transcode_sessions import (
    BaseFrameSplitter,
    CodecFrame,
    FlacSplitter,
    build_splitter,
)
from music_assistant.helpers.ffmpeg import get_ffmpeg_stream

SAMPLE_RATE = 44100
BIT_DEPTH = 16
CHANNELS = 2
DURATION_SECONDS = 3


def _silence_pcm(fmt: AudioFormat) -> AsyncGenerator[bytes]:
    async def gen() -> AsyncGenerator[bytes]:
        block = fmt.pcm_sample_size
        for _ in range(DURATION_SECONDS):
            yield b"\x00" * block

    return gen()


async def _encode(content_type: ContentType) -> bytes:
    pcm_format = AudioFormat(
        content_type=ContentType.from_bit_depth(BIT_DEPTH),
        sample_rate=SAMPLE_RATE,
        bit_depth=BIT_DEPTH,
        channels=CHANNELS,
    )
    output_format = AudioFormat(
        content_type=content_type,
        sample_rate=SAMPLE_RATE,
        bit_depth=BIT_DEPTH,
        channels=CHANNELS,
    )
    out = bytearray()
    async for chunk in get_ffmpeg_stream(
        audio_input=_silence_pcm(pcm_format),
        input_format=pcm_format,
        output_format=output_format,
    ):
        out.extend(chunk)
    return bytes(out)


def _split(
    raw: bytes, output_format: AudioFormat, cut: int
) -> tuple[BaseFrameSplitter, list[CodecFrame]]:
    splitter = build_splitter(output_format)
    assert splitter is not None
    frames: list[CodecFrame] = []
    for pos in range(0, len(raw), cut):
        frames.extend(splitter.feed(raw[pos : pos + cut]))
    if tail := splitter.flush():
        frames.append(tail)
    return splitter, frames


@pytest.mark.parametrize(
    ("content_type", "head_marker"),
    [
        (ContentType.MP3, None),
        (ContentType.AAC, b""),
        (ContentType.FLAC, b"fLaC"),
        (ContentType.WAV, b"RIFF"),
    ],
)
@pytest.mark.parametrize("cut", [1, 7, 128, 4096])
async def test_splitter_reassembles_exactly(
    content_type: ContentType, head_marker: bytes | None, cut: int
) -> None:
    """Any byte feeding pattern reconstructs the encoder output exactly."""
    raw = await _encode(content_type)
    output_format = AudioFormat(
        content_type=content_type,
        sample_rate=SAMPLE_RATE,
        bit_depth=BIT_DEPTH,
        channels=CHANNELS,
    )
    splitter, frames = _split(raw, output_format, cut)
    assert frames, f"{content_type} produced no frames at cut={cut}"
    reassembled = splitter.head + b"".join(frame.data for frame in frames)
    assert reassembled == raw
    total_seconds = sum(frame.samples for frame in frames) / SAMPLE_RATE
    assert total_seconds == pytest.approx(DURATION_SECONDS, abs=0.1)
    if head_marker is not None:
        assert splitter.head.startswith(head_marker)


async def test_no_half_frame_is_published_mid_stream_mp3() -> None:
    """A trailing partial frame stays buffered until the next feed completes it."""
    raw = await _encode(ContentType.MP3)
    output_format = AudioFormat(
        content_type=ContentType.MP3,
        sample_rate=SAMPLE_RATE,
        bit_depth=BIT_DEPTH,
        channels=CHANNELS,
    )
    splitter = build_splitter(output_format)
    assert splitter is not None
    # split the stream at the first frame boundary and feed it minus one byte
    frames = splitter.feed(raw[:44])  # ID3/Xing preamble only
    assert frames == []
    partial_frames = splitter.feed(raw[44:200])
    # everything returned must be a whole frame; the dangling tail is retained
    boundary = 44 + sum(len(f.data) for f in partial_frames)
    assert boundary <= 200
    completed = splitter.feed(raw[200:201])
    assert not completed or all(f.data for f in completed)
    # full reassembly still holds
    rest_frames = []
    rest_frames.extend(splitter.feed(raw[201:]))
    if tail := splitter.flush():
        rest_frames.append(tail)
    all_frames = partial_frames + completed + rest_frames
    assert splitter.head + b"".join(f.data for f in all_frames) == raw


def test_flac_rejects_a_false_sync_with_bad_crc() -> None:
    """A 0xFFF8-looking word that fails the header CRC-8 is not a frame sync."""
    splitter = FlacSplitter(SAMPLE_RATE)
    # valid fLaC + STREAMINFO marked as the last metadata block (4 + 4 + 34 bytes)
    streaminfo = b"\x80\x00\x00\x22" + b"\x00" * 34
    splitter.feed(b"fLaC" + streaminfo)
    # a bogus sync candidate whose CRC does not match, followed by nothing valid
    candidate = bytes([0xFF, 0xF8, 0xC9, 0x18, 0x00, 0x00])
    frames = splitter.feed(candidate)
    assert frames == []
    assert splitter.flush() is None
