"""Tests for MusicController.in_progress_items playlog queries."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import patch

import pytest
from music_assistant_models.auth import User, UserRole
from music_assistant_models.config_entries import ProviderAccess
from music_assistant_models.enums import MediaType, ProviderSharing
from music_assistant_models.errors import InsufficientPermissions

from music_assistant.constants import DB_TABLE_PLAYLOG, DB_TABLE_PROVIDER_MAPPINGS
from music_assistant.controllers.music import MusicController
from music_assistant.controllers.webserver.helpers.auth_middleware import set_current_user
from music_assistant.helpers.datetime import utc_timestamp
from music_assistant.mass import MusicAssistant
from tests.common import set_music_source_access

GET_CURRENT_USER = "music_assistant.controllers.music.controller.get_current_user"


async def _add_in_progress_row(
    mass: MusicAssistant,
    item_id: str,
    *,
    provider: str = "library",
    media_type: MediaType = MediaType.AUDIOBOOK,
    userid: str = "user-a",
) -> None:
    """Insert a single in-progress (partially played) playlog row."""
    await mass.music.database.insert(
        DB_TABLE_PLAYLOG,
        {
            "item_id": item_id,
            "provider": provider,
            "media_type": media_type.value,
            "name": f"Item {item_id}",
            "timestamp": int(utc_timestamp()),
            "fully_played": False,
            "seconds_played": 60,
            "userid": userid,
            "user_initiated": True,
        },
    )


async def _add_provider_mapping(
    mass: MusicAssistant,
    item_id: int,
    provider_instance: str,
    media_type: MediaType = MediaType.AUDIOBOOK,
    *,
    available: bool = True,
) -> None:
    """Insert a provider mapping row for a library item."""
    await mass.music.database.insert(
        DB_TABLE_PROVIDER_MAPPINGS,
        {
            "media_type": media_type.value,
            "item_id": item_id,
            "provider_domain": provider_instance.rstrip("0123456789_"),
            "provider_instance": provider_instance,
            "provider_item_id": f"{provider_instance}-{item_id}",
            "available": available,
            "in_library": True,
        },
    )


async def test_in_progress_items_omitted_providers_unchanged(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing callers (no providers filter) keep seeing every in-progress entry."""
    monkeypatch.setattr(mass.music, "get_active_provider_instances", lambda: ["local_1"])
    await _add_provider_mapping(mass, item_id=1, provider_instance="local_1")
    await _add_in_progress_row(mass, "1", provider="library")

    result = await mass.music.in_progress_items(limit=10)

    assert {item.item_id for item in result} == {"1"}


async def test_in_progress_items_filters_direct_provider_entries(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct (non-library) playlog entries are matched against their own provider."""
    monkeypatch.setattr(
        mass.music, "get_active_provider_instances", lambda: ["audible_1", "local_1"]
    )
    await _add_in_progress_row(mass, "audible-book", provider="audible_1")
    await _add_in_progress_row(mass, "local-book", provider="local_1")

    result = await mass.music.in_progress_items(limit=10, providers=["local_1"])

    assert {item.item_id for item in result} == {"local-book"}


async def test_in_progress_items_library_item_or_matches_any_mapping(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A library item mapped to two providers remains when only one is requested."""
    monkeypatch.setattr(
        mass.music, "get_active_provider_instances", lambda: ["audible_1", "local_1"]
    )
    await _add_provider_mapping(mass, item_id=1, provider_instance="audible_1")
    await _add_provider_mapping(mass, item_id=1, provider_instance="local_1")
    await _add_in_progress_row(mass, "1", provider="library")

    result = await mass.music.in_progress_items(limit=10, providers=["local_1"])

    assert {item.item_id for item in result} == {"1"}


async def test_in_progress_items_explicit_empty_providers_returns_no_items(
    mass: MusicAssistant,
) -> None:
    """An explicit empty provider list returns no items, distinct from omitting the filter."""
    await _add_in_progress_row(mass, "book-1")

    result = await mass.music.in_progress_items(limit=10, providers=[])

    assert result == []


async def test_in_progress_items_combines_explicit_and_user_music_sources(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user's music sources narrow an explicit filter that would otherwise pass."""
    monkeypatch.setattr(
        mass.music, "get_active_provider_instances", lambda: ["audible_1", "local_1"]
    )
    # audible_1 is another member's private source, so user-a may not use it
    set_music_source_access(
        mass,
        {
            "local_1": ProviderAccess(owner="user-a", sharing=ProviderSharing.PRIVATE),
            "audible_1": ProviderAccess(owner="user-b", sharing=ProviderSharing.PRIVATE),
        },
    )
    user_a = User(user_id="user-a", username="user-a", role=UserRole.USER)
    await _add_provider_mapping(mass, item_id=1, provider_instance="audible_1")
    await _add_provider_mapping(mass, item_id=1, provider_instance="local_1")
    await _add_in_progress_row(mass, "1", provider="library")

    with patch(GET_CURRENT_USER, return_value=user_a):
        result = await mass.music.in_progress_items(limit=10, providers=["local_1"])
    assert {item.item_id for item in result} == {"1"}

    # requesting a music source the user may not use must not leak the item back
    # in, even though the item does have a (restricted) mapping to it.
    with patch(GET_CURRENT_USER, return_value=user_a):
        result = await mass.music.in_progress_items(limit=10, providers=["audible_1"])
    assert result == []


async def test_in_progress_items_excludes_unavailable_mapping(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mapping that is no longer available on the requested provider cannot match."""
    monkeypatch.setattr(
        mass.music, "get_active_provider_instances", lambda: ["local_1", "audible_1"]
    )
    await _add_provider_mapping(mass, item_id=1, provider_instance="local_1")
    await _add_provider_mapping(mass, item_id=1, provider_instance="audible_1", available=False)
    await _add_in_progress_row(mass, "1", provider="library")

    result = await mass.music.in_progress_items(limit=10, providers=["audible_1"])

    assert result == []


@pytest.mark.parametrize(
    "list_other_users_items",
    [
        pytest.param(lambda music: music.in_progress_items(all_users=True), id="all_users"),
        pytest.param(
            lambda music: music.recently_played(userid="user-b", fully_played_only=False),
            id="userid",
        ),
    ],
)
@pytest.mark.parametrize(
    ("role", "allowed"),
    [(UserRole.USER, False), (UserRole.SERVICE, True)],
    ids=["member", "service"],
)
async def test_other_users_items_need_the_impersonate_scope(
    mass: MusicAssistant,
    monkeypatch: pytest.MonkeyPatch,
    list_other_users_items: Callable[[MusicController], Awaitable[list[object]]],
    role: str,
    allowed: bool,
) -> None:
    """The legacy all_users and userid parameters reach other users' playlog only with the scope."""
    monkeypatch.setattr(mass.music, "get_active_provider_instances", lambda: ["local_1"])
    await _add_provider_mapping(mass, item_id=1, provider_instance="local_1")
    await _add_in_progress_row(mass, "1", userid="user-b")
    set_current_user(User(user_id="user-a", username="user-a", role=role))

    try:
        if allowed:
            assert len(await list_other_users_items(mass.music)) == 1
        else:
            with pytest.raises(InsufficientPermissions):
                await list_other_users_items(mass.music)
    finally:
        set_current_user(None)
