"""Data models for the persistent event log and the resumable event stream."""

from __future__ import annotations

import contextvars
import importlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mashumaro.mixins.orjson import DataClassORJSONMixin
from music_assistant_models.enums import EventType  # noqa: TC002 - mashumaro resolves the dataclass field annotation at runtime
from music_assistant_models.favorite_update import FavoriteUpdate
from music_assistant_models.media_items import MediaItem, media_from_dict

from music_assistant.helpers.json import get_serializable_value

from .constants import (
    DATA_DICT,
    DATA_FAVORITE_UPDATE,
    DATA_LIST,
    DATA_MEDIA_ITEM,
    DATA_MODEL_PREFIX,
    DATA_RAW,
    LOGGER,
)


class ResumeMode(StrEnum):
    """Outcome of a resumable subscription request."""

    # The cursor was valid: stored events are replayed.
    RESUMED = "resumed"
    # The cursor was missing or unusable: a full snapshot is served instead.
    SNAPSHOT = "snapshot"


class SnapshotReason(StrEnum):
    """Why a subscription request fell back to a snapshot."""

    # No cursor presented at all (first subscription).
    FRESH = "fresh"
    # The window no longer contains (all of) the requested events.
    TRIMMED = "trimmed"
    # Unknown token, foreign server instance or future/non-numeric cursor.
    INVALID_CURSOR = "invalid_cursor"
    # Same cursor, different authenticated principal or permission scope.
    SCOPE_CHANGED = "scope_changed"


@dataclass(frozen=True)
class SequencedEventMessage(DataClassORJSONMixin):
    """
    Event envelope for resumable subscriptions, carrying the global monotonic cursor.

    The event/object_id/data keys are identical to a regular MassEvent; new clients
    detect the envelope by the presence of ``seq``. Legacy clients never receive this
    message type.

    :param event: The type of the event.
    :param seq: Monotonic per-server event sequence (the cursor).
    :param server_id: Identity of the server instance that owns the sequence.
    :param object_id: Optional object id this event relates to.
    :param data: Optional event payload.
    """

    event: EventType
    seq: int
    server_id: str
    object_id: str | None = None
    data: Any = field(
        default=None,
        metadata={"serialize": get_serializable_value},
    )


def _neutral_to_dict(obj: Any) -> Any:
    """
    Serialize an event payload to plain JSON without any per-connection context.

    Runs in a fresh ``Context`` so the image/translation resolvers the signalling task
    happens to carry are ignored: the frozen window stores the same representation for
    every connection, and per-connection enrichment is applied again when the event is
    delivered (live or replayed) through the usual send path.

    :param obj: The model-like object to serialize.
    """
    return contextvars.Context().run(get_serializable_value, obj)


def freeze_payload(data: Any) -> tuple[str, Any]:
    """
    Freeze an event payload to a ``(type tag, json value)`` pair.

    The tag records the exact payload identity up front so the replay path can rebuild
    the same typed object before running the regular per-connection event filter.

    :param data: The raw ``MassEvent.data`` payload.
    """
    if isinstance(data, MediaItem):
        # check MediaItem before generic models (Playlist/Track are subclasses)
        return DATA_MEDIA_ITEM, _neutral_to_dict(data)
    if isinstance(data, FavoriteUpdate):
        return DATA_FAVORITE_UPDATE, _neutral_to_dict(data)
    if isinstance(data, list | tuple | set | frozenset):
        return DATA_LIST, [freeze_payload(item) for item in data]
    if isinstance(data, dict):
        return DATA_DICT, {str(key): freeze_payload(value) for key, value in data.items()}
    if hasattr(data, "to_dict"):
        cls = type(data)
        tag = f"{DATA_MODEL_PREFIX}{cls.__module__}.{cls.__qualname__}"
        return tag, _neutral_to_dict(data)
    return DATA_RAW, data


def _rehydrate_model(tag: str, payload: Any) -> Any:
    """Rebuild a generic mashumaro model from its dotted class path tag."""
    dotted_path = tag.removeprefix(DATA_MODEL_PREFIX)
    module_name, _, class_name = dotted_path.rpartition(".")
    if not module_name:
        return payload
    module = importlib.import_module(module_name)
    cls: Any = module
    for part in class_name.split("."):
        cls = getattr(cls, part)
    return cls.from_dict(payload)


def rehydrate_payload(tag: str, payload: Any) -> Any:
    """
    Rebuild a payload frozen with :func:`freeze_payload` back to its typed value.

    Any failure falls back to the raw json value rather than dropping the event: the
    sequence stays intact and every connection still learns that the cursor advanced.

    :param tag: The payload type tag assigned at admission.
    :param payload: The frozen json value.
    """
    try:
        if tag == DATA_RAW:
            return payload
        if tag == DATA_MEDIA_ITEM:
            return media_from_dict(payload)
        if tag == DATA_FAVORITE_UPDATE:
            return FavoriteUpdate.from_dict(payload)
        if tag == DATA_LIST:
            return [rehydrate_payload(item_tag, item_data) for item_tag, item_data in payload]
        if tag == DATA_DICT:
            return {
                key: rehydrate_payload(item_tag, item_data)
                for key, (item_tag, item_data) in payload.items()
            }
        if tag.startswith(DATA_MODEL_PREFIX):
            return _rehydrate_model(tag, payload)
    except Exception as err:
        LOGGER.debug("Could not rehydrate event payload tagged %s: %s", tag, str(err))
    return payload
