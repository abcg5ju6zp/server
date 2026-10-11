"""Constants for the event log controller."""

from __future__ import annotations

import logging
from typing import Final

from music_assistant.constants import MASS_LOGGER_NAME

LOGGER: Final[logging.Logger] = logging.getLogger(f"{MASS_LOGGER_NAME}.event_log")

DB_SCHEMA_VERSION: Final[int] = 1
DB_TABLE_SETTINGS: Final[str] = "settings"
DB_TABLE_EVENTS: Final[str] = "events"
DB_TABLE_SUBSCRIPTIONS: Final[str] = "subscriptions"

# Bounds of the persistent replay window. Progress events (QUEUE_TIME_UPDATED) are
# already throttled upstream to roughly one every two seconds per playing queue, so
# even with several queues the deque/db comfortably spans the last 30 minutes.
MAX_WINDOW_EVENTS: Final[int] = 20000
MAX_WINDOW_AGE_SEC: Final[int] = 1800
# Cheap count/age pruning rides along with event admission instead of a scheduled task.
PRUNE_EVERY_N_INSERTS: Final[int] = 100

# Resume tokens are single use: every (re)subscription rotates its token. Rows that
# never reconnect are dropped after this age.
SUBSCRIPTION_TTL_SEC: Final[int] = 86400

# The background writer commits at most this many rows in one batch, or flushes once
# this interval elapses with rows pending.
FLUSH_BATCH_SIZE: Final[int] = 50
FLUSH_INTERVAL: Final[float] = 1.0

# Payload type tags stored alongside every frozen event.
DATA_RAW: Final[str] = "raw"
DATA_MEDIA_ITEM: Final[str] = "media_item"
DATA_FAVORITE_UPDATE: Final[str] = "favorite_update"
DATA_LIST: Final[str] = "list"
DATA_DICT: Final[str] = "dict"
DATA_MODEL_PREFIX: Final[str] = "model:"
