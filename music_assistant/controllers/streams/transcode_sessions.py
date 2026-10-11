"""
Resumable transcode sessions for the streams controller.

A resumable session owns the *output* side of a track stream: the final encode
ffmpeg chain, a bounded ring of frame-aligned encoded segments and the media
time safely handed to clients. When a player briefly drops its HTTP connection
and re-requests the same URL, the session is still alive (waiting for a resume)
and the client continues from a retained frame boundary instead of restarting
the encode from the beginning.

Everything here is in-process only. Segments never touch disk (see the project
Usage Policy) and a server restart therefore answers every reconnect as a
fresh stream.

Only finite, seekable, non-realtime singles rendered as MP3, ADTS/AAC, FLAC or
WAV are eligible; every other stream keeps using the direct response path.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import aclosing, suppress
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha1
from json import dumps as json_dumps
from typing import TYPE_CHECKING, Any, Final, Literal
from uuid import uuid4

from music_assistant_models.enums import ContentType, CrossfadeMode, MediaType
from music_assistant_models.errors import (
    AudioError,
    MediaNotFoundError,
    ProviderUnavailableError,
)

from music_assistant.constants import MASS_LOGGER_NAME
from music_assistant.helpers.audio import get_output_format_key
from music_assistant.models.music_provider import MusicProvider, ProviderStreamLimitError

from .constants import (
    RESUME_LEASE_SECONDS,
    RESUME_MAX_SESSIONS,
    RESUME_RING_MAX_BYTES_FLAC,
    RESUME_RING_MAX_BYTES_LOSSY,
    RESUME_RING_MAX_BYTES_PCM,
    RESUME_RING_SECONDS,
    RESUME_SAFETY_REPLAY_MS,
    RESUME_SEGMENT_MAX_PACK_BYTES,
    RESUME_SLOW_CONSUMER_ABORT_FACTOR,
    RESUME_TOTAL_MAX_BYTES,
)

if TYPE_CHECKING:
    from aiohttp import web
    from music_assistant_models.media_items import AudioFormat
    from music_assistant_models.streamdetails import StreamDetails

    from music_assistant.controllers.streams.audio_buffer import AudioBuffer
    from music_assistant.helpers.dsp import ComplexFilter
    from music_assistant.mass import MusicAssistant
    from music_assistant.models.player_queue import QueueItem

# Filter plan entries are either plain ffmpeg filter strings or complex filters.
FilterParams = Sequence["str | ComplexFilter"]

LOGGER = logging.getLogger(f"{MASS_LOGGER_NAME}.transcode_sessions")

# Content types whose encoded output can be split on self-contained frame
# boundaries (or, for WAV, on fixed PCM block boundaries).
SPLITTABLE_CONTENT_TYPES: frozenset[ContentType] = frozenset(
    {ContentType.MP3, ContentType.AAC, ContentType.FLAC, ContentType.WAV}
)

# Media types that are finite items a player may want to resume.
RESUMABLE_MEDIA_TYPES: frozenset[MediaType] = frozenset(
    {MediaType.TRACK, MediaType.AUDIOBOOK, MediaType.PODCAST_EPISODE}
)


class SessionState(StrEnum):
    """Lifecycle state of a resumable transcode session."""

    STARTING = "starting"
    ACTIVE = "active"
    WAITING_RESUME = "waiting_resume"
    FAILED = "failed"
    TERMINATED = "terminated"


class TerminateReason(StrEnum):
    """Why a session was authoritatively terminated."""

    LEASE_EXPIRED = "lease_expired"
    QUEUE_STOP = "queue_stop"
    SESSION_ROTATION = "session_rotation"
    ITEM_SWEPT = "item_swept"
    REPLACED = "replaced"
    EVICTED = "evicted"
    COMPLETED = "completed"
    SERVER_SHUTDOWN = "server_shutdown"


class FailReason(StrEnum):
    """Distinguishable producer failures."""

    TRANSCODER_EXITED = "transcoder_exited"
    PROVIDER_SOURCE = "provider_source"
    BUFFER_WRITE = "buffer_write"


class ResumeOutcome(StrEnum):
    """How a request was answered relative to an existing session."""

    FRESH = "fresh"
    REPLAYED = "replayed"
    CONTINUED = "continued"
    RE_ENCODED = "re_encoded"
    COMPLETE = "complete"


# Sentinel returned by the NPT parser for a ``npt=now`` request.
NPT_NOW: Final[str] = "now"


@dataclass(frozen=True)
class SessionKey:
    """Identity of a session, derived from the stream URL."""

    queue_session_id: str
    queue_id: str
    queue_item_id: str
    player_id: str
    fmt: str


@dataclass(frozen=True)
class TranscodeFingerprint:
    """Frozen audio semantics a session was created with."""

    provider_instance: str
    item_id: str
    base_media_ms: int
    output_key: str
    pcm_signature: str
    filter_signature: str

    def matches(self, other: TranscodeFingerprint) -> bool:
        """Return whether another fingerprint describes the same audio."""
        return self == other


@dataclass(frozen=True)
class CodecFrame:
    """One self-contained encoded frame and the media time it covers."""

    data: bytes
    samples: int


@dataclass(frozen=True)
class EncodedSegment:
    """A published chunk of the encoded entity, aligned to frame boundaries."""

    seq: int
    kind: Literal["head", "media"]
    data: bytes
    # Absolute producer byte offset at which this segment starts.
    start_offset: int
    # Absolute media samples at which this segment starts / how many it covers.
    start_samples: int
    samples: int

    @property
    def end_offset(self) -> int:
        """Return the absolute byte offset just past this segment."""
        return self.start_offset + len(self.data)

    @property
    def end_samples(self) -> int:
        """Return the absolute media sample just past this segment."""
        return self.start_samples + self.samples


@dataclass
class Consumer:
    """One attached HTTP response pumping a session."""

    consumer_id: str
    player_id: str
    generation: int
    request: web.BaseRequest | None
    start_seq: int
    # End media samples of the last segment drained by this consumer.
    drained_samples: int = 0


@dataclass
class SessionHooks:
    """Callbacks into the streams controller, kept narrow for testability."""

    # Return False once the queue session this response belongs to is no longer
    # the playing one (stop, session rotation, item change).
    is_current: Callable[[], bool]
    # Fired once per session, when the first encoded segment reached a client.
    on_first_bytes: Callable[[], None]
    # Fired once per session when the whole item was produced and consumed.
    on_completed: Callable[[int, float], Awaitable[None]] | None = None


class SessionClosed(Exception):
    """Raised inside the producer when the session was terminated."""


# ---------------------------------------------------------------------------
# HTTP range negotiation helpers (pure functions)
# ---------------------------------------------------------------------------


def parse_byte_range(value: str | None) -> int | None:
    """
    Parse a single open-ended HTTP byte range.

    Only ``bytes=<start>-`` is accepted; suffix ranges, multipart ranges and
    garbage return None so the caller treats the request as a plain GET.

    :param value: The raw ``Range`` header value.
    :return: The requested start byte offset, or None.
    """
    if not value:
        return None
    try:
        unit, _, spec = value.partition("=")
        if unit.strip().lower() != "bytes" or not spec:
            return None
        if "," in spec:
            # multipart range: not supported, ignore
            return None
        start, sep, end = spec.partition("-")
        if sep != "-":
            return None
        start = start.strip()
        end = end.strip()
        if not start:
            # suffix range (bytes=-N): unsupported
            return None
        if end:
            return None
        result = int(start)
    except ValueError, AttributeError:
        return None
    return result if result >= 0 else None


def parse_npt(value: str | None) -> int | str | None:
    """
    Parse a DLNA ``TimeSeekRange.dlna.org`` npt start specification.

    :param value: The raw header value, e.g. ``npt=12.5-`` or ``npt=now-``.
    :return: Media time in milliseconds, the NPT_NOW sentinel, or None.
    """
    if not value:
        return None
    header = value.strip().lower()
    if not header.startswith("npt="):
        return None
    spec = header[4:]
    # take the start range part up to the first '-'
    start, sep, _rest = spec.partition("-")
    if sep != "-":
        return None
    start = start.strip()
    if not start:
        return None
    if start == "now":
        return NPT_NOW
    try:
        seconds = _parse_npt_time(start)
    except ValueError:
        return None
    return max(int(seconds * 1000), 0)


def _parse_npt_time(value: str) -> float:
    """Parse an NPT time as seconds (``ss[.frac]`` or ``hh:mm:ss[.frac]``)."""
    if ":" in value:
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError(value)
        hours, minutes, seconds = (float(part) for part in parts)
        return hours * 3600 + minutes * 60 + seconds
    return float(value)


# ---------------------------------------------------------------------------
# Frame splitters
# ---------------------------------------------------------------------------


class BaseFrameSplitter:
    """Split encoded ffmpeg output into self-contained media frames."""

    sample_rate: int = 0

    def __init__(self) -> None:
        """Initialize the splitter."""
        self._buf = bytearray()
        self._head = b""
        self._head_done = False

    @property
    def head(self) -> bytes:
        """Return the container header bytes (empty when the codec has none)."""
        return self._head

    @property
    def head_done(self) -> bool:
        """Return whether the container header has been consumed."""
        return self._head_done

    def feed(self, data: bytes) -> list[CodecFrame]:
        """
        Consume encoder output and return all complete frames found.

        Partial frames are retained until more data arrives.

        :param data: Raw bytes from the encoder.
        """
        raise NotImplementedError

    def flush(self) -> CodecFrame | None:
        """
        Return any trailing bytes once the encoder ended.

        A conformant encoder ends on a frame boundary, so this is normally
        None. A non-empty remainder is published without media duration rather
        than dropped: it is the producer's final data.
        """
        if not self._head_done:
            # never reached the first media frame: nothing safe to hand out
            return None
        if not self._buf:
            return None
        tail = CodecFrame(bytes(self._buf), 0)
        self._buf = bytearray()
        return tail


_MP3_BITRATES: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {
    # version index (MPEG1=3, MPEG2=2/0): (layer3 v1, layer3 v2)
    3: (
        (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0),
        (),
    ),
    2: (
        (),
        (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0),
    ),
    0: (
        (),
        (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0),
    ),
}

_MP3_SAMPLE_RATES: dict[int, tuple[int, ...]] = {
    3: (44100, 48000, 32000),
    2: (22050, 24000, 16000),
    0: (11025, 12000, 8000),
}


class Mp3Splitter(BaseFrameSplitter):
    """Split an MPEG audio (layer III) elementary stream into frames."""

    def feed(self, data: bytes) -> list[CodecFrame]:
        """Consume MP3 encoder output, returning all complete frames."""
        self._buf.extend(data)
        frames: list[CodecFrame] = []
        offset = 0
        if not self._head_done:
            # an optional ID3v2 tag or other preamble precedes the first frame;
            # keep it as the container head
            first_sync = self._find_sync(self._buf, 0)
            if first_sync is None:
                return frames
            if first_sync > 0:
                self._head = bytes(self._buf[:first_sync])
            self._head_done = True
            offset = first_sync
        buf = self._buf
        while True:
            sync = self._find_sync(buf, offset)
            if sync is None or sync + 4 > len(buf):
                break
            frame = self._frame_at(buf, sync)
            if frame is None:
                # header bytes looked like a sync but describe no valid frame:
                # rescan from one byte past the candidate
                offset = sync + 1
                continue
            frame_len, samples, self.sample_rate = frame
            if sync + frame_len > len(buf):
                offset = sync
                break
            frames.append(CodecFrame(bytes(buf[sync : sync + frame_len]), samples))
            offset = sync + frame_len
        # drop a consumed preamble and retain a trailing partial frame
        if offset > 0:
            del self._buf[:offset]
        return frames

    @staticmethod
    def _find_sync(buf: bytes | bytearray, start: int) -> int | None:
        for i in range(start, len(buf) - 1):
            if buf[i] == 0xFF and (buf[i + 1] & 0xE0) == 0xE0:
                return i
        return None

    @staticmethod
    def _frame_at(buf: bytes | bytearray, i: int) -> tuple[int, int, int] | None:
        b1 = buf[i + 1]
        b2 = buf[i + 2]
        version = (b1 >> 3) & 0x03
        layer = (b1 >> 1) & 0x03
        if layer != 1:  # only Layer III
            return None
        tables = _MP3_BITRATES.get(version)
        sample_rates = _MP3_SAMPLE_RATES.get(version)
        if tables is None or sample_rates is None:
            return None
        is_v1 = version == 3
        bitrate_table = tables[0] if is_v1 else tables[1]
        bitrate_index = (b2 >> 4) & 0x0F
        sample_rate_index = (b2 >> 2) & 0x03
        padding = (b2 >> 1) & 0x01
        if bitrate_index in (0, 15) or sample_rate_index == 3:
            return None
        bitrate_k = bitrate_table[bitrate_index]
        sample_rate = sample_rates[sample_rate_index]
        if is_v1:
            frame_len = (144 * bitrate_k * 1000) // sample_rate + padding
            samples = 1152
        else:
            frame_len = (72 * bitrate_k * 1000) // sample_rate + padding
            samples = 576
        if frame_len < 4:
            return None
        return frame_len, samples, sample_rate


class AdtsSplitter(BaseFrameSplitter):
    """Split a raw ADTS (AAC) stream into frames."""

    _SAMPLE_RATES = (
        96000,
        88200,
        64000,
        48000,
        44100,
        32000,
        24000,
        22050,
        16000,
        12000,
        11025,
        8000,
        7350,
    )

    def feed(self, data: bytes) -> list[CodecFrame]:
        """Consume ADTS encoder output, returning all complete frames."""
        self._buf.extend(data)
        frames: list[CodecFrame] = []
        offset = 0
        buf = self._buf
        while True:
            sync = self._find_sync(buf, offset)
            if sync is None or sync + 7 > len(buf):
                break
            b2 = buf[sync + 2]
            b3 = buf[sync + 3]
            b4 = buf[sync + 4]
            b5 = buf[sync + 5]
            profile_object_type = (b2 >> 6) & 0x03
            sample_rate_index = (b2 >> 2) & 0x0F
            frame_len = ((b3 & 0x03) << 11) | (b4 << 3) | ((b5 & 0xE0) >> 5)
            if profile_object_type == 0 or sample_rate_index in (0xC, 0xD, 0xF):
                offset = sync + 1
                continue
            if frame_len < 7 or sync + frame_len > len(buf):
                offset = sync
                break
            sample_rate = self._SAMPLE_RATES[sample_rate_index]
            frames.append(CodecFrame(bytes(buf[sync : sync + frame_len]), 1024))
            if self.sample_rate == 0:
                self.sample_rate = sample_rate
            offset = sync + frame_len
        if not self._head_done and frames:
            self._head_done = True
        if offset > 0:
            del self._buf[:offset]
        return frames

    @staticmethod
    def _find_sync(buf: bytes | bytearray, start: int) -> int | None:
        for i in range(start, len(buf) - 1):
            if buf[i] == 0xFF and (buf[i + 1] & 0xF0) == 0xF0:
                return i
        return None


def _flac_crc8(data: bytes) -> int:
    """Compute the FLAC CRC-8 (poly 0x07, init 0) of data."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (((crc << 1) ^ 0x07) & 0xFF) if crc & 0x80 else ((crc << 1) & 0xFF)
    return crc


_FLAC_BLOCK_SIZES = {1: 192, 2: 576, 3: 1152, 4: 2304, 5: 4608}
_FLAC_SAMPLE_RATES = {
    1: 88200,
    2: 176400,
    3: 192000,
    4: 8000,
    5: 16000,
    6: 22050,
    7: 24000,
    8: 32000,
    9: 44100,
    10: 48000,
    11: 96000,
}


class FlacSplitter(BaseFrameSplitter):
    """
    Split a native FLAC stream into frames.

    FLAC frames do not carry their own length, so a frame is delimited by the
    validated sync of the frame that follows. Every candidate sync is checked
    with the header CRC-8 before it is accepted.
    """

    def __init__(self, fallback_sample_rate: int) -> None:
        """
        Initialize the splitter.

        :param fallback_sample_rate: Sample rate used when the frame header
            references the STREAMINFO value.
        """
        super().__init__()
        self._fallback_sample_rate = fallback_sample_rate
        self._meta_consumed = 0

    def feed(self, data: bytes) -> list[CodecFrame]:
        """Consume FLAC encoder output, returning frames delimited by a next sync."""
        self._buf.extend(data)
        if not self._head_done:
            head_len = self._consume_metadata()
            if head_len is None:
                return []
            self._head = bytes(self._buf[:head_len])
            self._head_done = True
            self._meta_consumed = head_len
        frames: list[CodecFrame] = []
        buf = self._buf
        search = self._meta_consumed
        while True:
            sync = self._find_validated_sync(buf, search)
            if sync is None:
                break
            next_sync = self._find_validated_sync(buf, sync + 2)
            if next_sync is None:
                # wait for the frame that delimits this one; keep one byte of
                # slack so a sync straddling the chunk edge is still found
                search = sync
                break
            samples, _rate = self._frame_header(buf, sync)
            frames.append(CodecFrame(bytes(buf[sync:next_sync]), samples))
            search = next_sync
        consumed = search if frames else self._meta_consumed
        if consumed > self._meta_consumed:
            del self._buf[:consumed]
            self._meta_consumed = 0
        return frames

    def flush(self) -> CodecFrame | None:
        """Publish the final FLAC frame, which runs from the pending sync to EOF."""
        # the last frame runs from the pending sync to end-of-stream
        start = self._meta_consumed
        if self._head_done and len(self._buf) > start:
            sync = self._find_validated_sync(self._buf, start)
            if sync is not None:
                samples, _rate = self._frame_header(self._buf, sync)
                frame = CodecFrame(bytes(self._buf[sync:]), samples)
                self._buf = bytearray()
                self._meta_consumed = 0
                return frame
        return None

    def _consume_metadata(self) -> int | None:
        """Return the length of the ``fLaC`` marker plus all metadata blocks."""
        buf = self._buf
        if len(buf) < 4:
            return None
        if bytes(buf[:4]) != b"fLaC":
            raise AudioError("output not splittable: missing fLaC marker")
        pos = 4
        while True:
            if pos + 4 > len(buf):
                return None
            header = buf[pos]
            block_len = int.from_bytes(buf[pos + 1 : pos + 4], "big")
            pos += 4 + block_len
            if pos > len(buf):
                return None
            if header & 0x80:
                return pos

    def _find_validated_sync(self, buf: bytes | bytearray, start: int) -> int | None:
        i = start
        while True:
            idx = buf.find(b"\xff", i)
            if idx < 0 or idx + 4 >= len(buf):
                return None
            if (buf[idx + 1] & 0xFC) != 0xF8:
                i = idx + 1
                continue
            header_len = self._header_length(buf, idx)
            if header_len is None or idx + header_len > len(buf):
                i = idx + 1
                continue
            if _flac_crc8(bytes(buf[idx : idx + header_len - 1])) != buf[idx + header_len - 1]:
                i = idx + 1
                continue
            return idx

    @staticmethod
    def _sync_sanity(buf: bytes, idx: int) -> bool:
        return idx + 4 < len(buf) and buf[idx] == 0xFF and (buf[idx + 1] & 0xFC) == 0xF8

    def _header_length(self, buf: bytes | bytearray, idx: int) -> int | None:
        """Determine the FLAC frame header length (including its CRC-8)."""
        if idx + 5 > len(buf):
            return None
        b2 = buf[idx + 2]
        block_size_code = (b2 >> 4) & 0x0F
        sample_rate_code = b2 & 0x0F
        b4 = buf[idx + 4]
        # coded frame/sample number: utf-8 style, length = leading ones + 1
        leading = 0
        bit = 0x80
        while bit and b4 & bit:
            leading += 1
            bit >>= 1
        if leading > 6:
            return None
        # sync(2) + b2 + b3 + coded number (leading+1) + crc(1)
        length = 6 + leading
        if block_size_code == 6:
            length += 1
        elif block_size_code == 7:
            length += 2
        if sample_rate_code == 12:
            length += 1
        elif sample_rate_code in (13, 14):
            length += 2
        return length

    def _frame_header(self, buf: bytes | bytearray, idx: int) -> tuple[int, int]:
        """Return (samples, sample_rate) described by a validated frame header."""
        b2 = buf[idx + 2]
        block_size_code = (b2 >> 4) & 0x0F
        sample_rate_code = b2 & 0x0F
        if block_size_code in _FLAC_BLOCK_SIZES:
            samples = _FLAC_BLOCK_SIZES[block_size_code]
        elif block_size_code >= 8:
            samples = 256 << (block_size_code - 8)
        else:
            samples = 0
        sample_rate = _FLAC_SAMPLE_RATES.get(sample_rate_code, self._fallback_sample_rate)
        if sample_rate_code == 0:
            sample_rate = self._fallback_sample_rate
        if self.sample_rate == 0:
            self.sample_rate = sample_rate
        return samples, sample_rate


class WavSplitter(BaseFrameSplitter):
    """Split a WAV (RIFF + PCM) stream into fixed block-aligned frames."""

    def __init__(self) -> None:
        """Initialize the splitter."""
        super().__init__()
        self._block_align = 0
        self._byte_rate = 0
        self._data_start = -1

    def feed(self, data: bytes) -> list[CodecFrame]:
        """Consume WAV encoder output, returning block-aligned PCM frames."""
        self._buf.extend(data)
        if not self._head_done:
            parsed = self._parse_head()
            if parsed is None:
                return []
            head_len, block_align, byte_rate = parsed
            self._head = bytes(self._buf[:head_len])
            self._block_align = block_align
            self._byte_rate = byte_rate
            self._data_start = head_len
            self._head_done = True
        frames: list[CodecFrame] = []
        usable = len(self._buf) - self._data_start
        whole_blocks = (usable // self._block_align) * self._block_align
        end = self._data_start + whole_blocks
        samples_per_block = self._block_align * self.sample_rate // self._byte_rate
        pos = self._data_start
        while pos < end:
            frames.append(
                CodecFrame(bytes(self._buf[pos : pos + self._block_align]), samples_per_block)
            )
            pos += self._block_align
        if end > self._data_start:
            del self._buf[:end]
            # subsequent bytes are pure PCM data
            self._data_start = 0
        return frames

    def flush(self) -> CodecFrame | None:
        """Discard a trailing sub-block PCM remainder; encoder output is block aligned."""
        return None

    def _parse_head(self) -> tuple[int, int, int] | None:
        """Locate the data chunk and parse the format chunk."""
        buf = self._buf
        if len(buf) < 12:
            return None
        if bytes(buf[:4]) != b"RIFF" or bytes(buf[8:12]) != b"WAVE":
            raise AudioError("output not splittable: missing RIFF/WAVE marker")
        pos = 12
        fmt_data: bytes | None = None
        while pos + 8 <= len(buf):
            chunk_id = bytes(buf[pos : pos + 4])
            chunk_size = struct.unpack_from("<I", buf, pos + 4)[0]
            body_start = pos + 8
            if chunk_id == b"data":
                # a streamed WAV declares 0xFFFFFFFF (or a size past EOF): the
                # data simply runs to the end, so the body size is irrelevant here
                if fmt_data is None:
                    raise AudioError("output not splittable: data chunk before fmt")
                sample_rate = struct.unpack_from("<I", fmt_data, 4)[0]
                byte_rate = struct.unpack_from("<I", fmt_data, 8)[0]
                block_align = struct.unpack_from("<H", fmt_data, 12)[0]
                if block_align == 0:
                    raise AudioError("output not splittable: zero block align")
                self.sample_rate = sample_rate
                return body_start, block_align, byte_rate
            body_end = body_start + chunk_size
            if body_end > len(buf):
                return None
            if chunk_id == b"fmt ":
                fmt_data = bytes(buf[body_start:body_end])
            pos = body_end + (chunk_size & 1)
        return None


def build_splitter(output_format: AudioFormat) -> BaseFrameSplitter | None:
    """
    Build the frame splitter for an encoder output format.

    :param output_format: The encoded output format.
    :return: A splitter, or None for formats that can not be split.
    """
    content_type = output_format.content_type
    if content_type == ContentType.MP3:
        return Mp3Splitter()
    if content_type == ContentType.AAC:
        return AdtsSplitter()
    if content_type == ContentType.FLAC:
        return FlacSplitter(output_format.sample_rate)
    if content_type == ContentType.WAV:
        return WavSplitter()
    return None


# ---------------------------------------------------------------------------
# Segment ring
# ---------------------------------------------------------------------------


class SegmentRing:
    """Bounded deque of encoded segments with backpressure and eviction."""

    def __init__(
        self,
        sample_rate: int,
        max_seconds: float,
        max_bytes: int,
        retention_floor: Callable[[], int],
        slow_consumer_abort: Callable[[], None],
    ) -> None:
        """
        Initialize the ring.

        :param sample_rate: Sample rate of the media samples in segments.
        :param max_seconds: Retained media duration cap.
        :param max_bytes: Retained payload byte cap.
        :param retention_floor: Returns the absolute media sample below which
            segments can never be requested again (oldest consumer cursor, or
            the last checkpoint when no consumer is attached).
        :param slow_consumer_abort: Called when the consumer owning the floor
            blocks retention past twice the configured cap.
        """
        self._sample_rate = sample_rate
        self._max_ms = int(max_seconds * 1000)
        self._max_bytes = max_bytes
        self._retention_floor = retention_floor
        self._slow_consumer_abort = slow_consumer_abort
        self._segments: deque[EncodedSegment] = deque()
        # (end_offset, end_samples) for every evicted segment, in order.
        self._anchors: deque[tuple[int, int]] = deque(maxlen=4096)
        self._lock = asyncio.Lock()
        self._data_available = asyncio.Condition(self._lock)
        self._space_available = asyncio.Condition(self._lock)
        self._eof = False
        self._closed = False
        self._total_bytes = 0
        self._total_samples = 0

    @property
    def total_bytes(self) -> int:
        """Return payload bytes currently retained."""
        return self._total_bytes

    @property
    def closed(self) -> bool:
        """Return whether the ring was closed (terminated/failed)."""
        return self._closed

    def producer_reached_eof(self) -> bool:
        """Return whether the producer delivered (or closed at) its end."""
        return self._eof

    def iter_segments(self) -> tuple[EncodedSegment, ...]:
        """Return a snapshot tuple of the currently retained segments."""
        return tuple(self._segments)

    def close(self) -> None:
        """Release every waiter; no further segments are accepted."""
        if self._closed:
            return
        self._closed = True
        self._eof = True
        self._schedule_notify()

    def producer_complete(self) -> None:
        """Mark that the producer delivered its final segment."""
        self._eof = True
        self._schedule_notify()

    async def put(self, segment: EncodedSegment) -> None:
        """
        Append a segment, waiting for retention space when the ring is full.

        When the head start is pinned by a consumer that stopped reading, the
        producer is allowed bounded growth up to the slow-consumer abort factor
        instead of blocking forever; past that, the blocking consumer is aborted
        so eviction can proceed on the following puts.

        :raises SessionClosed: When the session was terminated while waiting.
        """
        async with self._lock:
            while True:
                if self._closed:
                    raise SessionClosed
                if not self._would_exceed(segment):
                    break
                if self._evict_locked():
                    continue
                if self._within_abort_factor(segment):
                    # pinned, but still within the bounded overshoot: accept
                    break
                self._abort_pinned_consumers()
                await self._space_available.wait()
            self._segments.append(segment)
            self._total_bytes += len(segment.data)
            self._total_samples += segment.samples
            self._evict_to_limits_locked()
            self._data_available.notify_all()

    def notify_space(self) -> None:
        """Wake a producer blocked on retention space (consumer detached)."""

        async def _notify() -> None:
            async with self._lock:
                self._space_available.notify_all()

        with suppress(RuntimeError):
            asyncio.get_running_loop().create_task(_notify())

    def samples_for_seq(self, seq: int) -> int | None:
        """Return the start media samples of a retained segment seq."""
        segment = next((s for s in self._segments if s.seq == seq), None)
        return segment.start_samples if segment is not None else None

    @property
    def first_segment(self) -> EncodedSegment | None:
        """Return the oldest retained segment."""
        return self._segments[0] if self._segments else None

    @property
    def last_segment(self) -> EncodedSegment | None:
        """Return the newest retained segment."""
        return self._segments[-1] if self._segments else None

    @property
    def first_media_segment(self) -> EncodedSegment | None:
        """Return the oldest retained media segment (head excluded)."""
        return next((s for s in self._segments if s.kind == "media"), None)

    def seq_for_offset(self, absolute_offset: int) -> int | None:
        """
        Map an absolute producer byte offset to the segment containing it.

        :return: The segment seq, or None when the offset was evicted or has not
            been produced yet.
        """
        for segment in self._segments:
            if segment.kind == "head":
                continue
            # a boundary offset resumes from the segment starting there
            if segment.start_offset <= absolute_offset < segment.end_offset:
                return segment.seq
        return None

    def seq_for_samples(self, absolute_samples: int) -> int | None:
        """Map an absolute media sample position to the segment starting there."""
        for segment in self._segments:
            if segment.kind == "head":
                continue
            if segment.start_samples <= absolute_samples < segment.end_samples:
                return segment.seq
        return None

    def samples_for_offset(self, absolute_offset: int) -> int | None:
        """
        Convert an absolute byte offset to absolute media samples.

        Uses retained segments when possible and eviction anchors (linear rate
        between known points) for evicted regions.
        """
        segment_seq = self.seq_for_offset(absolute_offset)
        if segment_seq is not None:
            segment = next(s for s in self._segments if s.seq == segment_seq)
            return segment.start_samples
        if self._anchors:
            return self._interpolate(absolute_offset)
        return None

    def offset_for_samples(self, absolute_samples: int) -> int | None:
        """Convert absolute media samples to an absolute byte offset."""
        for segment in self._segments:
            if segment.kind == "head":
                continue
            if segment.start_samples <= absolute_samples <= segment.end_samples:
                if segment.samples == 0:
                    return segment.start_offset
                ratio = (absolute_samples - segment.start_samples) / segment.samples
                return int(segment.start_offset + ratio * len(segment.data))
        anchors = list(self._anchors)
        if anchors and absolute_samples <= anchors[-1][1]:
            prev_offset, prev_samples = 0, 0
            for end_offset, end_samples in anchors:
                if absolute_samples <= end_samples:
                    if end_samples == prev_samples:
                        return end_offset
                    ratio = (absolute_samples - prev_samples) / (end_samples - prev_samples)
                    return int(prev_offset + ratio * (end_offset - prev_offset))
                prev_offset, prev_samples = end_offset, end_samples
        return None

    def iter_from(self, seq: int) -> RingIterator:
        """Return an iterator over segments starting at the given seq."""
        return RingIterator(self, seq)

    def _schedule_notify(self) -> None:
        async def _notify() -> None:
            async with self._lock:
                self._data_available.notify_all()
                self._space_available.notify_all()

        with suppress(RuntimeError):
            asyncio.get_running_loop().create_task(_notify())

    def _would_exceed(self, segment: EncodedSegment) -> bool:
        return (
            self._total_bytes + len(segment.data) > self._max_bytes
            or self._total_samples + segment.samples > self._max_ms * self._sample_rate // 1000
        )

    def _within_abort_factor(self, segment: EncodedSegment) -> bool:
        return (
            self._total_bytes + len(segment.data)
            <= self._max_bytes * RESUME_SLOW_CONSUMER_ABORT_FACTOR
            and self._total_samples + segment.samples
            <= self._max_ms * self._sample_rate // 1000 * RESUME_SLOW_CONSUMER_ABORT_FACTOR
        )

    def _abort_pinned_consumers(self) -> None:
        self._slow_consumer_abort()

    def _floor_samples(self) -> int:
        return self._retention_floor()

    def _evict_to_limits_locked(self) -> None:
        while self._segments and (
            self._total_bytes > self._max_bytes
            or self._total_samples > self._max_ms * self._sample_rate // 1000
        ):
            if not self._evict_locked():
                break

    def _evict_locked(self) -> bool:
        """
        Evict the oldest media segment when nobody can request it anymore.

        Segments ending at or before the retention floor (oldest consumer
        position, or the last checkpoint) are gone for good. Anything newer is
        part of the resume window and stays - the producer blocks instead.
        """
        media_index = next((i for i, seg in enumerate(self._segments) if seg.kind == "media"), None)
        if media_index is None:
            return False
        candidate = self._segments[media_index]
        floor = self._floor_samples()
        over_cap = (
            self._total_bytes > self._max_bytes
            or self._total_samples > self._max_ms * self._sample_rate // 1000
        )
        if candidate.end_samples > floor and over_cap:
            # inside the live resume window: retention is pinned
            return False
        if candidate.end_samples > floor:
            # within caps and still ahead of the floor: nothing to reclaim
            return False
        del self._segments[media_index]
        self._total_bytes -= len(candidate.data)
        self._total_samples -= candidate.samples
        self._anchors.append((candidate.end_offset, candidate.end_samples))
        self._space_available.notify_all()
        return True

    def _interpolate(self, absolute_offset: int) -> int | None:
        anchors = list(self._anchors)
        if not anchors:
            return None
        prev_offset, prev_samples = 0, 0
        for end_offset, end_samples in anchors:
            if absolute_offset <= end_offset:
                if end_offset == prev_offset:
                    return end_samples
                ratio = (absolute_offset - prev_offset) / (end_offset - prev_offset)
                return int(prev_samples + ratio * (end_samples - prev_samples))
            prev_offset, prev_samples = end_offset, end_samples
        return anchors[-1][1]


class RingIterator:
    """Async iterator over a ring's segments from a starting seq."""

    def __init__(self, ring: SegmentRing, seq: int) -> None:
        """Initialize the iterator."""
        self._ring = ring
        self._next_seq = seq

    def __aiter__(self) -> RingIterator:
        """Return self."""
        return self

    async def __anext__(self) -> EncodedSegment:
        """Return the next segment or raise StopAsyncIteration at EOF."""
        async with self._ring._data_available:
            while True:
                for segment in self._ring._segments:
                    if segment.seq == self._next_seq:
                        self._next_seq += 1
                        return segment
                first = self._ring._segments[0] if self._ring._segments else None
                # a completed session keeps its segments drainable; closed only
                # means no *new* segments will ever arrive
                if self._ring._eof and (
                    first is None or self._next_seq > self._ring._segments[-1].seq
                ):
                    raise StopAsyncIteration
                if first is not None and self._next_seq < first.seq:
                    raise StopAsyncIteration
                await self._ring._data_available.wait()


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class FrozenSessionContext:
    """Everything a session was created with, frozen for its lifetime."""

    key: SessionKey
    fingerprint: TranscodeFingerprint
    queue_item: QueueItem
    streamdetails: StreamDetails
    buffer: AudioBuffer
    pcm_format: AudioFormat
    output_format: AudioFormat
    filter_params: FilterParams
    base_media_ms: int
    http_profile: str
    hooks: SessionHooks
    # Factory building the full encode chain at the given start media time.
    make_chain: Callable[[int], AsyncGenerator[bytes]]
    on_ended: Callable[[TranscodeSession], None] | None = None


@dataclass
class AttachResult:
    """The manager's decision for one stream request."""

    outcome: ResumeOutcome
    session: TranscodeSession | None
    start_seq: int
    replay_head: bool
    # Absolute producer offset/sample the response body starts at.
    content_offset: int
    content_samples: int
    # Best known total entity size (for Content-Range/Content-Length), if any.
    total_bytes: int | None
    headers: dict[str, str] = field(default_factory=dict)
    reencode_reason: str | None = None


class TranscodeSession:
    """One generation: a producer, its encode chain, its ring and consumers."""

    def __init__(self, context: FrozenSessionContext) -> None:
        """
        Initialize the session.

        :param context: The frozen session context.
        """
        self.ctx = context
        self.key = context.key
        self.fingerprint = context.fingerprint
        self.output_format = context.output_format
        self.sample_rate = context.output_format.sample_rate
        splitter = build_splitter(context.output_format)
        if splitter is None:
            msg = "output format is not splittable"
            raise AudioError(msg)
        self._splitter: BaseFrameSplitter = splitter
        max_bytes = self._ring_byte_cap(context.output_format.content_type)
        self._ring = SegmentRing(
            self.sample_rate,
            RESUME_RING_SECONDS,
            max_bytes,
            self._retention_floor,
            self._abort_slow_consumer,
        )
        self._state = SessionState.STARTING
        self._fail_reason: FailReason | None = None
        self._terminate_reason: TerminateReason | None = None
        self._error: Exception | None = None
        self._generation = 0
        self._consumers: dict[str, Consumer] = {}
        self._producer_task: asyncio.Task[None] | None = None
        self._lease_handle: asyncio.TimerHandle | None = None
        self._next_seq = 1
        self._abs_offset = 0
        self._abs_samples = 0
        self._head_published = False
        self._head = b""
        self._producer_eof = False
        self._first_delivered = False
        self._completed_reported = False
        # conservative end position safely handed to clients (media samples)
        self.checkpoint_samples = 0
        self.created_at = time.monotonic()

    @property
    def state(self) -> SessionState:
        """Return the current lifecycle state."""
        return self._state

    @property
    def fail_reason(self) -> FailReason | None:
        """Return the failure classification when failed."""
        return self._fail_reason

    @property
    def terminate_reason(self) -> TerminateReason | None:
        """Return the termination reason when terminated."""
        return self._terminate_reason

    @property
    def error(self) -> Exception | None:
        """Return the exception that ended the producer, if any."""
        return self._error

    @property
    def generation(self) -> int:
        """Return the current generation; stale consumers compare against it."""
        return self._generation

    @property
    def ring(self) -> SegmentRing:
        """Return the segment ring."""
        return self._ring

    @property
    def has_consumers(self) -> bool:
        """Return whether a consumer is currently attached."""
        return bool(self._consumers)

    @property
    def produced_offset(self) -> int:
        """Return absolute producer bytes produced."""
        return self._abs_offset

    @property
    def produced_samples(self) -> int:
        """Return absolute media samples produced."""
        return self._abs_samples

    @property
    def total_produced_bytes(self) -> int:
        """Return encoded payload bytes produced (excluding a container head)."""
        return max(self._abs_offset - len(self._head), 0)

    @property
    def total_produced_seconds(self) -> float:
        """Return produced media seconds."""
        return self._abs_samples / self.sample_rate if self.sample_rate else 0.0

    @property
    def measured_bytes_per_second(self) -> float | None:
        """Return observed encoded bytes per produced second, once measurable."""
        payload_bytes = self.total_produced_bytes
        if self.total_produced_seconds > 0.5 and payload_bytes > 0:
            return payload_bytes / self.total_produced_seconds
        return None

    def start(self) -> None:
        """Start the producer task."""
        loop = asyncio.get_running_loop()
        self._producer_task = loop.create_task(self._produce())

    def attach(self, request: web.BaseRequest | None) -> Consumer:
        """
        Attach a consumer, superseding a half-open one of the same player.

        :param request: The request of the new consumer (for transport abort).
        """
        for consumer in list(self._consumers.values()):
            # a reconnect from the same player supersedes its half-dead previous
            # response: abort (not graceful close) mirrors close_superseded_item_streams
            if (
                consumer.player_id == self.key.player_id
                and consumer.request is not None
                and consumer.request is not request
                and (transport := consumer.request.transport) is not None
            ):
                transport.abort()
        self._cancel_lease()
        consumer = Consumer(
            consumer_id=uuid4().hex,
            player_id=self.key.player_id,
            generation=self._generation,
            request=request,
            start_seq=1,
        )
        self._consumers[consumer.consumer_id] = consumer
        if self._state == SessionState.WAITING_RESUME:
            self._state = SessionState.ACTIVE
        LOGGER.debug(
            "Resume session %s: consumer %s attached (state: %s)",
            self.key.queue_item_id,
            consumer.consumer_id[:8],
            self._state,
        )
        return consumer

    async def detach(self, consumer: Consumer) -> None:
        """
        Detach a consumer, arming the resume lease when none remain.

        :param consumer: The consumer to remove.
        """
        self._consumers.pop(consumer.consumer_id, None)
        # a detached cursor frees pinned retention, so wake a blocked producer
        self._ring.notify_space()
        if self._consumers:
            return
        if self._state in (SessionState.FAILED, SessionState.TERMINATED):
            return
        if self._producer_eof:
            # the whole item was delivered: bookkeep once, then keep the complete
            # entity under the resume lease so late reconnects still answer correctly
            await self._report_completion_if_any()
            self._state = SessionState.WAITING_RESUME
            self._arm_lease()
            return
        self._state = SessionState.WAITING_RESUME
        self._arm_lease()
        LOGGER.debug(
            "Resume session %s: waiting for resume (%.0fs lease)",
            self.key.queue_item_id,
            RESUME_LEASE_SECONDS,
        )

    async def pump(
        self,
        consumer: Consumer,
        resp: Any,
        start_seq: int,
        replay_head: bool,
    ) -> None:
        """
        Pump segments to an HTTP response until it ends or the session does.

        :param consumer: The attached consumer.
        :param resp: The prepared aiohttp StreamResponse.
        :param start_seq: First segment seq to send.
        :param replay_head: Whether the entity needs the container head first.
        """
        disconnect = False
        consumer.start_seq = start_seq
        # the head segment may not be published yet when the consumer starts:
        # iterate from seq 0 and pick it up as it arrives
        first_seq = 0 if replay_head else start_seq
        try:
            async for segment in self._ring.iter_from(first_seq):
                if consumer.generation != self._generation:
                    break
                if not self.ctx.hooks.is_current():
                    await self.terminate(TerminateReason.SESSION_ROTATION)
                    break
                if segment.kind == "head":
                    if replay_head:
                        try:
                            await resp.write(segment.data)
                        except BrokenPipeError, ConnectionResetError, ConnectionError:
                            disconnect = True
                            break
                    continue
                try:
                    await resp.write(segment.data)
                except BrokenPipeError, ConnectionResetError, ConnectionError:
                    disconnect = True
                    break
                consumer.start_seq = segment.seq + 1
                consumer.drained_samples = segment.end_samples
                self._advance_checkpoint()
                if not self._first_delivered:
                    self._first_delivered = True
                    self.ctx.hooks.on_first_bytes()
            if self._state == SessionState.FAILED:
                # the body ends on a producer failure: tell the client to drop it
                resp.force_close()
        finally:
            LOGGER.debug(
                "Resume session %s: consumer %s pump ended (disconnect: %s, state: %s)",
                self.key.queue_item_id,
                consumer.consumer_id[:8],
                disconnect,
                self._state,
            )
            await self.detach(consumer)

    async def terminate(self, reason: TerminateReason) -> None:
        """
        Authoritatively terminate this session and release everything it owns.

        :param reason: What triggered the termination.
        """
        if self._state in (SessionState.FAILED, SessionState.TERMINATED):
            return
        LOGGER.debug("Resume session %s terminating: %s", self.key.queue_item_id, reason)
        self._terminate_reason = reason
        self._state = SessionState.TERMINATED
        self._generation += 1
        self._cancel_lease()
        self._ring.close()
        if (
            self._producer_task
            and not self._producer_task.done()
            and self._producer_task is not asyncio.current_task()
        ):
            self._producer_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._producer_task
        if reason == TerminateReason.COMPLETED:
            await self._report_completion_if_any()
        if self.ctx.on_ended is not None:
            self.ctx.on_ended(self)

    def resume_target_samples(self) -> int:
        """Return the conservative media sample position a Range-less resume uses."""
        safety = RESUME_SAFETY_REPLAY_MS * self.sample_rate // 1000
        target = max(self.checkpoint_samples - safety, 0)
        # never point before the retained window
        media = self._ring.first_media_segment
        if media is not None:
            target = max(target, media.start_samples)
        return target

    @staticmethod
    def _ring_byte_cap(content_type: ContentType) -> int:
        if content_type == ContentType.WAV:
            return RESUME_RING_MAX_BYTES_PCM
        if content_type == ContentType.FLAC:
            return RESUME_RING_MAX_BYTES_FLAC
        return RESUME_RING_MAX_BYTES_LOSSY

    def _advance_checkpoint(self) -> None:
        drained = [c.drained_samples for c in self._consumers.values() if c.drained_samples]
        if drained:
            self.checkpoint_samples = min(drained)

    def _retention_floor(self) -> int:
        """Return the oldest media position any current consumer may still read."""
        cursors: list[int] = []
        for consumer in self._consumers.values():
            samples = self._ring.samples_for_seq(consumer.start_seq)
            if samples is not None:
                cursors.append(samples)
        if cursors:
            return min(cursors)
        # no consumers: nothing older than the drained checkpoint can be asked for
        return self.checkpoint_samples

    def _abort_slow_consumer(self) -> None:
        """Abort the transport of the consumer owning the retention floor."""
        floor = self._retention_floor()
        for consumer in self._consumers.values():
            if (
                self._ring.samples_for_seq(consumer.start_seq) == floor
                and consumer.request is not None
                and (transport := consumer.request.transport) is not None
            ):
                transport.abort()
                return

    async def _publish_head(self) -> None:
        if self._head_published:
            return
        self._head = self._splitter.head
        head_segment = EncodedSegment(
            seq=0,
            kind="head",
            data=self._head,
            start_offset=0,
            start_samples=0,
            samples=0,
        )
        await self._ring.put(head_segment)
        self._abs_offset += len(self._head)
        self._head_published = True

    async def _publish_frames(self, frames: list[CodecFrame]) -> None:
        packed = bytearray()
        packed_samples = 0
        for frame in frames:
            if packed and len(packed) + len(frame.data) > RESUME_SEGMENT_MAX_PACK_BYTES:
                await self._emit_media_segment(bytes(packed), packed_samples)
                packed = bytearray()
                packed_samples = 0
            packed.extend(frame.data)
            packed_samples += frame.samples
        if packed:
            await self._emit_media_segment(bytes(packed), packed_samples)

    async def _emit_media_segment(self, data: bytes, samples: int) -> None:
        if not self._head_published and self._splitter.head_done:
            await self._publish_head()
        segment = EncodedSegment(
            seq=self._next_seq,
            kind="media",
            data=data,
            start_offset=self._abs_offset,
            start_samples=self._abs_samples,
            samples=samples,
        )
        await self._ring.put(segment)
        self._abs_offset += len(data)
        self._abs_samples += samples
        self._next_seq += 1
        if self._state == SessionState.STARTING:
            self._state = SessionState.ACTIVE if self._consumers else SessionState.WAITING_RESUME

    async def _produce(self) -> None:
        chain = self.ctx.make_chain(self.fingerprint.base_media_ms)
        try:
            async with aclosing(chain):
                async for chunk in chain:
                    frames = self._splitter.feed(chunk)
                    if frames:
                        await self._publish_frames(frames)
                tail = self._splitter.flush()
                if tail is not None and tail.data:
                    await self._publish_frames([tail])
            self._producer_eof = True
            if not self._head_published and self._splitter.head_done:
                await self._publish_head()
            self._ring.producer_complete()
            if not self._consumers and self._state not in (
                SessionState.FAILED,
                SessionState.TERMINATED,
            ):
                # the producer beat the (about to attach) consumer to EOF: keep the
                # complete entity under the resume lease instead of tearing it down
                self._state = SessionState.WAITING_RESUME
                self._arm_lease()
        except asyncio.CancelledError:
            raise
        except SessionClosed:
            pass
        except (ProviderStreamLimitError, ProviderUnavailableError, MediaNotFoundError) as err:
            await self._fail(FailReason.PROVIDER_SOURCE, err)
        except MemoryError as err:
            await self._fail(FailReason.BUFFER_WRITE, err)
        except AudioError as err:
            await self._fail(FailReason.TRANSCODER_EXITED, err)
        except Exception as err:
            await self._fail(FailReason.TRANSCODER_EXITED, err)

    async def _fail(self, reason: FailReason, error: Exception) -> None:
        if self._state in (SessionState.FAILED, SessionState.TERMINATED):
            return
        LOGGER.warning(
            "Resume session %s failed (%s): %s",
            self.key.queue_item_id,
            reason,
            error,
        )
        self._fail_reason = reason
        self._error = error
        self._state = SessionState.FAILED
        self._generation += 1
        self._cancel_lease()
        self._ring.close()
        if self.ctx.on_ended is not None:
            self.ctx.on_ended(self)
        await self._report_completion_if_any()

    async def _report_completion_if_any(self) -> None:
        if self._completed_reported:
            return
        if not self._producer_eof or self._state == SessionState.FAILED:
            return
        # only bookkeep audio that actually reached a client
        if not self._first_delivered:
            return
        self._completed_reported = True
        if self.ctx.hooks.on_completed is not None:
            payload_seconds = self.total_produced_seconds
            try:
                await self.ctx.hooks.on_completed(self.total_produced_bytes, payload_seconds)
            except Exception:
                LOGGER.exception("Resume session completion callback failed")

    def _arm_lease(self) -> None:
        self._cancel_lease()
        loop = asyncio.get_running_loop()
        self._lease_handle = loop.call_later(RESUME_LEASE_SECONDS, self._lease_expired)

    def _cancel_lease(self) -> None:
        if self._lease_handle is not None:
            self._lease_handle.cancel()
            self._lease_handle = None

    def _lease_expired(self) -> None:
        if self._state != SessionState.WAITING_RESUME:
            return
        task = asyncio.get_running_loop().create_task(self.terminate(TerminateReason.LEASE_EXPIRED))
        task.add_done_callback(lambda _t: None)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


_NPT_HEADER = "TimeSeekRange.dlna.org"
# Fixed encoder bitrates of get_ffmpeg_stream (see helpers/ffmpeg.py).
_LOSSY_BYTES_PER_SECOND = {ContentType.MP3: 320_000 // 8, ContentType.AAC: 256_000 // 8}


@dataclass
class ResumeRequest:
    """Parsed position request of one HTTP GET."""

    byte_offset: int | None = None
    npt_ms: int | None = None
    npt_now: bool = False

    @property
    def has_explicit_position(self) -> bool:
        """Return whether the client asked for a specific position."""
        return self.byte_offset is not None or self.npt_ms is not None or self.npt_now


def parse_resume_request(headers: Any) -> ResumeRequest:
    """
    Parse the resume-relevant headers of a stream request.

    :param headers: A mapping of request headers.
    """
    npt = parse_npt(headers.get(_NPT_HEADER))
    return ResumeRequest(
        byte_offset=parse_byte_range(headers.get("Range")),
        npt_ms=npt if isinstance(npt, int) else None,
        npt_now=npt == NPT_NOW,
    )


class TranscodeSessionManager:
    """Registry and negotiation for resumable transcode sessions."""

    def __init__(self, mass: MusicAssistant) -> None:
        """
        Initialize the manager.

        :param mass: The MusicAssistant instance.
        """
        self.mass = mass
        self._sessions: dict[SessionKey, TranscodeSession] = {}
        self._by_queue_session: dict[tuple[str, str], set[SessionKey]] = {}
        self._by_item: dict[tuple[str, str], set[SessionKey]] = {}
        self._eviction_tasks: set[asyncio.Task[None]] = set()

    @staticmethod
    def is_eligible(
        *,
        method: str,
        media_type: MediaType,
        duration: float | None,
        allow_seek: bool,
        is_realtime: bool,
        crossfade_mode: CrossfadeMode,
        overlay_is_active: bool,
        playback_speed: float,
        content_type: ContentType,
    ) -> bool:
        """
        Return whether a request may use the resumable session path.

        Everything ineligible keeps using the direct streaming path unchanged.
        """
        if method not in ("GET", "HEAD"):
            return False
        if media_type not in RESUMABLE_MEDIA_TYPES:
            return False
        if not duration:
            return False
        if not allow_seek or is_realtime:
            return False
        if crossfade_mode != CrossfadeMode.DISABLED or overlay_is_active:
            return False
        if playback_speed != 1.0:
            return False
        return content_type in SPLITTABLE_CONTENT_TYPES

    async def resolve(  # noqa: PLR0913
        self,
        *,
        key: SessionKey,
        queue_item: QueueItem,
        streamdetails: StreamDetails,
        buffer: AudioBuffer,
        pcm_format: AudioFormat,
        output_format: AudioFormat,
        filter_params: FilterParams,
        http_profile: str,
        request: web.Request,
        duration_s: float | None,
        total_bytes_hint: int | None,
        make_chain: Callable[[int], AsyncGenerator[bytes]],
        hooks: SessionHooks,
    ) -> AttachResult:
        """
        Resolve a stream request to a fresh, replayed, continued or new session.

        :param key: Session identity derived from the URL.
        :param queue_item: The queue item being streamed.
        :param streamdetails: The resolved source streamdetails.
        :param buffer: The raw PCM buffer feeding the encode chain.
        :param pcm_format: The PCM format the encode chain consumes.
        :param output_format: The encoded output format.
        :param filter_params: The frozen DSP/filter arguments.
        :param http_profile: The player's configured HTTP profile.
        :param request: The incoming HTTP request.
        :param duration_s: Finite item duration in seconds, if known.
        :param total_bytes_hint: Best available entity byte-size estimate.
        :param make_chain: Factory of the encode chain, taking a start media ms.
        :param hooks: Controller callbacks for the session.
        """
        resume_request = parse_resume_request(request.headers)
        fingerprint = self._fingerprint(
            streamdetails,
            int(streamdetails.seek_position * 1000),
            pcm_format,
            output_format,
            filter_params,
        )
        base_ms = int(streamdetails.seek_position * 1000)
        existing = self._sessions.get(key)
        if existing is not None and existing.state in (
            SessionState.FAILED,
            SessionState.TERMINATED,
        ):
            existing = None

        reencode_reason: str | None = None
        if existing is not None:
            invalid_reason = self._invalid_reason(existing, fingerprint, streamdetails, buffer)
            if invalid_reason is not None:
                reencode_reason = invalid_reason
                await existing.terminate(TerminateReason.REPLACED)
                existing = None

        if existing is not None:
            decision = self._decide_for_existing(
                existing,
                resume_request,
                http_profile,
                duration_s,
                total_bytes_hint,
            )
            if decision is not None:
                return decision
            # the requested position left the retained window: re-encode from it
            reencode_reason = reencode_reason or "window"
            target_ms = self._target_ms_for_reencode(existing, resume_request)
            await existing.terminate(TerminateReason.REPLACED)
            existing = None
        else:
            target_ms = self._target_ms_without_session(
                base_ms, resume_request, output_format, duration_s, total_bytes_hint
            )

        session = await self._create_session(
            key=key,
            queue_item=queue_item,
            streamdetails=streamdetails,
            buffer=buffer,
            pcm_format=pcm_format,
            output_format=output_format,
            filter_params=filter_params,
            http_profile=http_profile,
            start_media_ms=target_ms,
            hooks=hooks,
            make_chain=make_chain,
        )
        outcome = ResumeOutcome.RE_ENCODED if reencode_reason else ResumeOutcome.FRESH
        first_seq = 1
        result = AttachResult(
            outcome=outcome,
            session=session,
            start_seq=first_seq,
            replay_head=True,
            content_offset=0,
            content_samples=0,
            total_bytes=total_bytes_hint,
            reencode_reason=reencode_reason,
        )
        result.headers = self._build_headers(
            result, resume_request=resume_request, duration_s=duration_s
        )
        return result

    async def terminate_queue_sessions(
        self, queue_id: str, queue_session_id: str | None = None
    ) -> None:
        """
        Terminate sessions owned by a queue playback session (stop/clear/replace).

        :param queue_id: The queue whose sessions to terminate.
        :param queue_session_id: When given, only that playback session's
            sessions are terminated; None clears all sessions of the queue.
        """
        if queue_session_id is None:
            keys = {key for key in self._sessions if key.queue_id == queue_id}
        else:
            keys = set(self._by_queue_session.get((queue_id, queue_session_id), set()))
        for key in list(keys):
            session = self._sessions.get(key)
            if session is not None:
                await session.terminate(TerminateReason.QUEUE_STOP)

    async def terminate_for_item(self, queue_id: str, queue_item_id: str) -> None:
        """
        Terminate sessions of a queue item swept out of the playing window.

        :param queue_id: The owning queue.
        :param queue_item_id: The queue item that was swept.
        """
        keys = set(self._by_item.get((queue_id, queue_item_id), set()))
        for key in list(keys):
            session = self._sessions.get(key)
            if session is not None:
                await session.terminate(TerminateReason.ITEM_SWEPT)

    async def close(self) -> None:
        """Terminate every session (server shutdown)."""
        sessions = list(self._sessions.values())
        for session in sessions:
            await session.terminate(TerminateReason.SERVER_SHUTDOWN)

    def diagnostics(self) -> dict[str, Any]:
        """Return session counts for diagnostics."""
        states: dict[str, int] = {}
        total_bytes = 0
        for session in self._sessions.values():
            states[session.state.value] = states.get(session.state.value, 0) + 1
            total_bytes += session.ring.total_bytes
        return {"sessions": states, "retained_bytes": total_bytes}

    def _spawn_eviction(self, session: TranscodeSession) -> None:
        """Terminate a session outside the caller's flow, keeping the task referenced."""
        task = asyncio.get_running_loop().create_task(session.terminate(TerminateReason.EVICTED))
        self._eviction_tasks.add(task)
        task.add_done_callback(self._eviction_tasks.discard)

    def _decide_for_existing(
        self,
        session: TranscodeSession,
        resume_request: ResumeRequest,
        http_profile: str,
        duration_s: float | None,
        total_bytes_hint: int | None,
    ) -> AttachResult | None:
        # a position-less request resumes from the conservative checkpoint
        if not resume_request.has_explicit_position:
            return self._decision_checkpoint(session, http_profile, duration_s)

        target_samples = self._requested_samples(session, resume_request)
        if target_samples is None:
            return None
        ring = session.ring
        if ring.producer_reached_eof() and resume_request.byte_offset is not None:
            if resume_request.byte_offset >= session.produced_offset:
                target_samples = session.produced_samples
        if ring.producer_reached_eof() and target_samples >= session.produced_samples:
            return AttachResult(
                outcome=ResumeOutcome.COMPLETE,
                session=session,
                start_seq=0,
                replay_head=False,
                content_offset=session.produced_offset,
                content_samples=session.produced_samples,
                total_bytes=self._total_bytes(session, duration_s, total_bytes_hint),
            )
        seq = ring.seq_for_samples(target_samples)
        if seq is None:
            return None
        segment = self._segment_by_seq(ring, seq)
        if segment is None:
            return None
        partial = self._partial_allowed(session, http_profile, resume_request)
        total = self._total_bytes(session, duration_s, total_bytes_hint)
        if partial:
            result = AttachResult(
                outcome=ResumeOutcome.CONTINUED,
                session=session,
                start_seq=seq,
                replay_head=False,
                content_offset=segment.start_offset,
                content_samples=segment.start_samples,
                total_bytes=total,
            )
        else:
            result = AttachResult(
                outcome=ResumeOutcome.REPLAYED,
                session=session,
                start_seq=seq,
                replay_head=True,
                content_offset=segment.start_offset,
                content_samples=segment.start_samples,
                total_bytes=total,
            )
        result.headers = self._build_headers(
            result, resume_request=resume_request, duration_s=duration_s
        )
        return result

    def _decision_checkpoint(
        self,
        session: TranscodeSession,
        http_profile: str,
        duration_s: float | None,
    ) -> AttachResult | None:
        target_samples = session.resume_target_samples()
        seq = session.ring.seq_for_samples(target_samples)
        if seq is None:
            # checkpoint left the window (or nothing retained): must re-encode
            return None
        segment = self._segment_by_seq(session.ring, seq)
        if segment is None:
            return None
        result = AttachResult(
            outcome=ResumeOutcome.REPLAYED,
            session=session,
            start_seq=seq,
            replay_head=True,
            content_offset=segment.start_offset,
            content_samples=segment.start_samples,
            total_bytes=self._total_bytes(session, duration_s, None),
        )
        result.headers = self._build_headers(result, resume_request=None, duration_s=duration_s)
        return result

    @staticmethod
    def _segment_by_seq(ring: SegmentRing, seq: int) -> EncodedSegment | None:
        return next((s for s in ring.iter_segments() if s.seq == seq), None)

    @staticmethod
    def _requested_samples(session: TranscodeSession, resume_request: ResumeRequest) -> int | None:
        if resume_request.npt_ms is not None:
            return resume_request.npt_ms * session.sample_rate // 1000
        if resume_request.npt_now:
            return session.resume_target_samples()
        if resume_request.byte_offset is not None:
            samples = session.ring.samples_for_offset(resume_request.byte_offset)
            if samples is not None:
                return samples
            # evicted or not yet produced: fixed-rate codecs still translate
            rate = _LOSSY_BYTES_PER_SECOND.get(session.output_format.content_type)
            if rate is not None:
                return resume_request.byte_offset * session.sample_rate // rate
            return None
        return None

    @staticmethod
    def _target_ms_for_reencode(
        session: TranscodeSession,
        resume_request: ResumeRequest,
    ) -> int:
        base_ms = session.fingerprint.base_media_ms
        if not resume_request.has_explicit_position:
            return base_ms + session.resume_target_samples() * 1000 // session.sample_rate
        if resume_request.npt_ms is not None:
            return base_ms + resume_request.npt_ms
        if resume_request.npt_now:
            return base_ms + session.resume_target_samples() * 1000 // session.sample_rate
        if resume_request.byte_offset is not None:
            samples = TranscodeSessionManager._requested_samples(session, resume_request)
            if samples is not None:
                return base_ms + samples * 1000 // session.sample_rate
        return base_ms

    @staticmethod
    def _target_ms_without_session(
        base_ms: int,
        resume_request: ResumeRequest,
        output_format: AudioFormat,
        duration_s: float | None,
        total_bytes_hint: int | None,
    ) -> int:
        if resume_request.npt_ms is not None:
            return base_ms + resume_request.npt_ms
        if resume_request.byte_offset is None:
            return base_ms
        relative_ms: int | None = None
        rate = _LOSSY_BYTES_PER_SECOND.get(output_format.content_type)
        if rate is not None:
            relative_ms = resume_request.byte_offset * 1000 // rate
            if duration_s:
                relative_ms = min(relative_ms, int(duration_s * 1000))
        elif total_bytes_hint and duration_s:
            fraction = min(resume_request.byte_offset / total_bytes_hint, 1.0)
            relative_ms = int(fraction * duration_s * 1000)
        return base_ms + (relative_ms or 0)

    def _partial_allowed(
        self,
        session: TranscodeSession,
        http_profile: str,
        resume_request: ResumeRequest,
    ) -> bool:
        # a checkpoint resume is always a full replayed entity
        if resume_request.npt_now or not resume_request.has_explicit_position:
            return False
        measured = session.measured_bytes_per_second is not None
        if http_profile == "forced_content_length":
            return measured or session.output_format.content_type in (
                ContentType.MP3,
                ContentType.AAC,
            )
        # chunked / no content length: only commit to 206 on a near-exact rate
        return measured or session.output_format.content_type in (
            ContentType.MP3,
            ContentType.AAC,
        )

    @staticmethod
    def _total_bytes(
        session: TranscodeSession,
        duration_s: float | None,
        total_bytes_hint: int | None,
    ) -> int | None:
        measured = session.measured_bytes_per_second
        if measured is not None and duration_s is not None:
            base_s = session.fingerprint.base_media_ms / 1000
            entity_s = max(duration_s - base_s, 0)
            return int(measured * entity_s)
        rate = _LOSSY_BYTES_PER_SECOND.get(session.output_format.content_type)
        if rate is not None and duration_s is not None:
            base_s = session.fingerprint.base_media_ms / 1000
            return int(rate * max(duration_s - base_s, 0))
        return total_bytes_hint

    def _build_headers(
        self,
        result: AttachResult,
        *,
        resume_request: ResumeRequest | None,
        duration_s: float | None,
    ) -> dict[str, str]:
        session = result.session
        if session is None:
            return {}
        headers = {"Accept-Ranges": "bytes"}
        if result.outcome == ResumeOutcome.FRESH:
            headers["X-MA-Resume"] = "supported"
        elif result.outcome == ResumeOutcome.REPLAYED:
            npt = result.content_samples // session.sample_rate
            headers["X-MA-Resume"] = f"replayed; npt={npt}"
        elif result.outcome == ResumeOutcome.CONTINUED:
            npt = result.content_samples // session.sample_rate
            headers["X-MA-Resume"] = f"continued; npt={npt}"
            if result.total_bytes is not None:
                headers["Content-Range"] = (
                    f"bytes {result.content_offset}-{result.total_bytes - 1}/{result.total_bytes}"
                )
            else:
                headers["Content-Range"] = f"bytes {result.content_offset}-/*"
            if resume_request is not None and resume_request.npt_ms is not None and duration_s:
                start_s = resume_request.npt_ms / 1000
                headers["timeSeekRange.dlna.org"] = (
                    f"npt={start_s:.3f}-{duration_s:.3f}/{duration_s:.3f}"
                )
        elif result.outcome == ResumeOutcome.RE_ENCODED:
            reason = result.reencode_reason or "window"
            npt = result.content_samples // session.sample_rate
            headers["X-MA-Resume"] = f"re-encoded; reason={reason}; npt={npt}"
        elif result.outcome == ResumeOutcome.COMPLETE:
            headers["X-MA-Resume"] = "complete"
        return headers

    def _invalid_reason(
        self,
        session: TranscodeSession,
        fingerprint: TranscodeFingerprint,
        streamdetails: StreamDetails,
        buffer: AudioBuffer,
    ) -> str | None:
        if not fingerprint.matches(session.fingerprint):
            return "fingerprint"
        if streamdetails is not session.ctx.streamdetails:
            # provider capacity reselection replaces the streamdetails object
            return "source"
        if streamdetails.provider != fingerprint.provider_instance:
            return "auth"
        if streamdetails.item_id != fingerprint.item_id:
            return "source"
        # the raw PCM buffer is attached lazily when the chain starts, so on a
        # reconnect it only needs to exist and still be healthy
        if buffer is None or buffer.cancelled or buffer.has_error:
            return "source"
        provider = self.mass.get_provider(streamdetails.provider, return_unavailable=True)
        if not isinstance(provider, MusicProvider) or not provider.available:
            return "auth"
        return None

    @staticmethod
    def _fingerprint(
        streamdetails: StreamDetails,
        base_media_ms: int,
        pcm_format: AudioFormat,
        output_format: AudioFormat,
        filter_params: FilterParams,
    ) -> TranscodeFingerprint:
        filter_signature = sha1(
            json_dumps([str(param) for param in filter_params], separators=(",", ":")).encode()
        ).hexdigest()
        return TranscodeFingerprint(
            provider_instance=streamdetails.provider,
            item_id=streamdetails.item_id,
            base_media_ms=base_media_ms,
            output_key=get_output_format_key(output_format),
            pcm_signature=(
                f"{pcm_format.sample_rate}_{pcm_format.bit_depth}_{pcm_format.channels}"
            ),
            filter_signature=filter_signature,
        )

    async def _create_session(  # noqa: PLR0913
        self,
        *,
        key: SessionKey,
        queue_item: QueueItem,
        streamdetails: StreamDetails,
        buffer: AudioBuffer,
        pcm_format: AudioFormat,
        output_format: AudioFormat,
        filter_params: FilterParams,
        http_profile: str,
        start_media_ms: int,
        hooks: SessionHooks,
        make_chain: Callable[[int], AsyncGenerator[bytes]],
    ) -> TranscodeSession:
        self._enforce_caps()
        fingerprint = self._fingerprint(
            streamdetails, start_media_ms, pcm_format, output_format, filter_params
        )
        context = FrozenSessionContext(
            key=key,
            fingerprint=fingerprint,
            queue_item=queue_item,
            streamdetails=streamdetails,
            buffer=buffer,
            pcm_format=pcm_format,
            output_format=output_format,
            filter_params=list(filter_params),
            base_media_ms=start_media_ms,
            http_profile=http_profile,
            hooks=hooks,
            make_chain=make_chain,
            on_ended=self._on_session_ended,
        )
        session = TranscodeSession(context)
        self._register(session)
        session.start()
        return session

    def _enforce_caps(self) -> None:
        # session count cap: reap the longest-waiting resume sessions first
        waiting = [s for s in self._sessions.values() if s.state == SessionState.WAITING_RESUME]
        waiting.sort(key=lambda s: s.created_at)
        while len(self._sessions) >= RESUME_MAX_SESSIONS and waiting:
            victim = waiting.pop(0)
            self._spawn_eviction(victim)
            self._sessions.pop(victim.key, None)
        # hard byte budget
        total = sum(s.ring.total_bytes for s in self._sessions.values())
        if total <= RESUME_TOTAL_MAX_BYTES:
            return
        waiting = [s for s in self._sessions.values() if s.state == SessionState.WAITING_RESUME]
        waiting.sort(key=lambda s: s.created_at)
        for victim in waiting:
            if total <= RESUME_TOTAL_MAX_BYTES:
                break
            total -= victim.ring.total_bytes
            self._spawn_eviction(victim)
            self._sessions.pop(victim.key, None)

    def _register(self, session: TranscodeSession) -> None:
        key = session.key
        self._sessions[key] = session
        self._by_queue_session.setdefault((key.queue_id, key.queue_session_id), set()).add(key)
        self._by_item.setdefault((key.queue_id, key.queue_item_id), set()).add(key)

    def _on_session_ended(self, session: TranscodeSession) -> None:
        key = session.key
        self._sessions.pop(key, None)
        queue_set = self._by_queue_session.get((key.queue_id, key.queue_session_id))
        if queue_set is not None:
            queue_set.discard(key)
        item_set = self._by_item.get((key.queue_id, key.queue_item_id))
        if item_set is not None:
            item_set.discard(key)
