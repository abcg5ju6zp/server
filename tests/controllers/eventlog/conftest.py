"""Fixtures for event log / resumable event stream tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request
from music_assistant_models.api import CommandMessage
from music_assistant_models.auth import User, UserRole

from music_assistant.controllers.eventlog import EventLogController
from music_assistant.controllers.webserver import WebserverController
from music_assistant.controllers.webserver.websocket_client import (
    EVENT_STREAM_ACK,
    EVENT_STREAM_SUBSCRIBE,
    WebsocketClientHandler,
)

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant


class _StubQueue:
    """Stand-in for a PlayerQueue that the snapshot builder can serialize."""

    def __init__(self, queue_id: str) -> None:
        self.queue_id = queue_id

    def to_dict(self) -> dict[str, str]:
        return {"queue_id": self.queue_id}


class _QueueStub:
    """Minimal stand-in for the player queues controller used by the snapshot builder."""

    def __init__(self, queue_ids: list[str]) -> None:
        self._queues = [_StubQueue(queue_id) for queue_id in queue_ids]

    def __iter__(self) -> Iterator[_StubQueue]:
        return iter(self._queues)

    def items(self, queue_id: str) -> list[dict[str, str]]:
        return [{"queue_id": queue_id, "queue_item_id": f"{queue_id}-item"}]


@pytest.fixture
async def event_log(mass_minimal: MusicAssistant) -> AsyncIterator[EventLogController]:
    """Return a set up event log controller on a minimal mass instance."""
    controller = EventLogController(mass_minimal)
    mass_minimal.event_log = controller
    await controller.setup(None)  # type: ignore[arg-type]
    try:
        yield controller
    finally:
        await controller.close()


def _compute_image_id(provider: Any, path: Any) -> str:
    """Stub for the metadata image id resolver."""
    return f"{provider}--{path}"


def _get_translation(_key: str, **_kwargs: Any) -> None:
    """Stub for the translation resolver."""
    return


def _all_player_states() -> list[dict[str, str]]:
    """Stub for the players/all command."""
    return [{"player_id": "p1"}]


def _get_player(_player_id: str) -> None:
    """Stub returning no resolved player."""
    return


def _list_tasks(_user: User) -> list[Any]:
    """Stub for the per-user task listing."""
    return []


def _get_providers(_user: User) -> list[dict[str, str]]:
    """Stub for the per-user provider listing."""
    return [{"instance_id": "sp"}]


@pytest.fixture
def event_log_ws(
    mass_minimal: MusicAssistant,
    event_log: EventLogController,
) -> dict[str, Any]:
    """Wire the stubs the websocket client and snapshot builder need."""
    mass_minimal.metadata = SimpleNamespace(compute_image_id=_compute_image_id)  # type: ignore[assignment]
    mass_minimal.translations = SimpleNamespace(  # type: ignore[assignment]
        get_translation=_get_translation
    )
    mass_minimal.players = SimpleNamespace(  # type: ignore[assignment]
        all_player_states=_all_player_states,
        get_player=_get_player,
    )
    mass_minimal.player_queues = _QueueStub(["q1"])  # type: ignore[assignment]
    mass_minimal.tasks = SimpleNamespace(list_tasks_for_user=_list_tasks)  # type: ignore[assignment]
    mass_minimal.get_providers_for_user = _get_providers  # type: ignore[method-assign,assignment]
    webserver = WebserverController(mass_minimal)
    mass_minimal.webserver = webserver
    return {"mass": mass_minimal, "webserver": webserver, "event_log": event_log}


def create_client(
    event_log_ws: dict[str, Any],
    user: User | None = None,
    *,
    opt_in: bool = True,
) -> WebsocketClientHandler:
    """
    Create an authenticated websocket client handler (mocked socket).

    :param event_log_ws: The fixture wiring stubs and controllers.
    :param user: The authenticated user (admin by default).
    :param opt_in: Whether the client opted into the resumable event stream at auth.
    """
    request = make_mocked_request("GET", "/ws", app=web.Application())
    client = WebsocketClientHandler(event_log_ws["webserver"], request)
    client._authenticated_user = user or User(
        user_id="admin", username="admin", role=UserRole.ADMIN
    )
    client._event_stream_opt_in = opt_in
    return client


async def send_command(
    client: WebsocketClientHandler, command: str, message_id: str, **args: Any
) -> None:
    """
    Dispatch a connection command and wait until its correlated response is queued.

    :param client: The websocket client handler.
    :param command: The command name.
    :param message_id: Correlation id of the message.
    :param args: Command arguments.
    """
    msg = CommandMessage(message_id=message_id, command=command, args=args or None)
    if command in (EVENT_STREAM_SUBSCRIBE, EVENT_STREAM_ACK):
        await client._handle_event_stream_command(msg)
    else:
        await client._handle_command(msg)
    # the handler runs as a tracked task and serializes large responses in an executor;
    # wait deterministically instead of yielding a fixed number of times
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    needle = f'"message_id":"{message_id}"'
    queued_items: Any = client._to_write._queue  # type: ignore[attr-defined]
    while loop.time() < deadline:
        if any(isinstance(item, str) and needle in item for item in queued_items):
            return
        await asyncio.sleep(0)
    raise AssertionError(f"Timed out waiting for response to {message_id}")


def drain_wire(client: WebsocketClientHandler) -> list[dict[str, Any]]:
    """Pop and decode every message queued for the client's writer."""
    messages: list[dict[str, Any]] = []
    while True:
        try:
            raw = client._to_write.get_nowait()
        except asyncio.QueueEmpty:
            return messages
        if raw is not None:
            messages.append(json.loads(raw))


async def settle() -> None:
    """Let the call_soon event fan-out callbacks run."""
    for _ in range(10):
        await asyncio.sleep(0)
