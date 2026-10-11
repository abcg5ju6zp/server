"""
Schema migrations for player_commands.db.

The batch player commands database is operational data, fully owned by the
player_commands controller and never shared with the music library. It starts at
schema version 1 (created by database.create_player_commands_tables), so there are
no migration steps yet.

Rules for every future step (same discipline as controllers/music/migrations.py):
- append a new ``if prev_version < N:`` block and bump PC_DB_SCHEMA_VERSION;
- a step runs once against data written by a version that can not be inspected here,
  so it must be idempotent, survive missing and half-written values, and never raise;
- agree the schema change with the developer in the session before implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from music_assistant.helpers.database import DatabaseConnection


async def migrate_database(
    database: DatabaseConnection,
    logger: logging.Logger,
    prev_version: int,
    target_version: int,
) -> None:
    """
    Migrate player_commands.db from ``prev_version`` to ``target_version``.

    :param database: The batch commands database connection.
    :param logger: Logger to report migration progress.
    :param prev_version: Schema version currently stored in the database.
    :param target_version: Schema version the server expects.
    """
    if prev_version == target_version:
        return
    # no migration steps exist at schema v1; the connection is reserved for the
    # first future step appended below
    _ = database
    logger.info(
        "Migrating player_commands database from version %s to %s",
        prev_version,
        target_version,
    )
    # if prev_version < 2:
    #     ... append future migration steps here ...
