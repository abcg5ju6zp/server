"""Audit logging for asynchronous batch player commands.

Every state-changing moment is persisted in the batch_audit table (by the database
layer) and additionally emitted as a structured log line here, so an operator can
correlate the requester, operation, per-target dispatches and the final summary from
the server log without opening the database.
"""

from __future__ import annotations

import logging
from typing import Any

AUDIT_LOGGER_NAME = "music_assistant.player_commands.audit"


def log_audit(
    logger: logging.Logger,
    action: str,
    *,
    operation_id: str,
    actor: str | None,
    server_id: str | None = None,
    player_id: str | None = None,
    attempt: int | None = None,
    **extra: Any,
) -> None:
    """
    Emit one structured audit line for a batch command lifecycle event.

    :param logger: The controller logger.
    :param action: Audit action (operation_created/target_dispatched/...).
    :param operation_id: The batch operation the line belongs to.
    :param actor: Who requested the operation (username or user id).
    :param server_id: Server/boot owner for dispatch related lines.
    :param player_id: Target player id for per-target lines.
    :param attempt: Attempt number for dispatch lines.
    :param extra: Additional key/values appended to the line.
    """
    parts = [f"action={action}", f"operation_id={operation_id}", f"actor={actor or 'system'}"]
    if server_id is not None:
        parts.append(f"server={server_id}")
    if player_id is not None:
        parts.append(f"player={player_id}")
    if attempt is not None:
        parts.append(f"attempt={attempt}")
    for key, value in extra.items():
        parts.append(f"{key}={value}")
    logger.info("[batch-audit] %s", " ".join(parts))
