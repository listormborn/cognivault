"""Route tests for Confluence sync/status + chat source-url enrichment (phase 4).

Covers:
* ``POST /api/confluence/sync`` pre-flight 400 when unconfigured;
* ``POST /api/confluence/sync`` 200 ``text/event-stream`` with frames flowing;
* single-flight: a call while the tenant lock is held streams
  ``SYNC_ALREADY_RUNNING``;
* ``GET /api/confluence/status`` reflects ``running`` while the lock is held;
* ``store.manifest_url_index`` reverse index (populated / empty / missing);
* chat ``sources`` SSE event carries a ``url`` only for Confluence-synced paths.

pytest + Starlette ``TestClient`` (no real network). Runs in LOCAL mode so no
bearer header is required and ``resolve_paths`` is monkeypatched to a tmp dir.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import rag, settings  # noqa: E402
from app.config import AppPaths  # noqa: E402
from app.confluence import store, sync  # noqa: E402
from app.main import create_app  # noqa: E402
from app.routes import chat_routes, confluence_routes  # noqa: E402


def _paths(tmp_path) -> AppPaths:
    return AppPaths(root=tmp_path / ".cognivault-ui")


@pytest.fixture(autouse=True)
def _local_mode(monkeypatch):
    """Force LOCAL mode and a clean per-test lock table."""
    monkeypatch.setattr(settings, "is_server", lambda: False)
    sync.SYNC_LOCKS.clear()
    yield
    sync.SYNC_LOCKS.clear()


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into ``[(event, data_dict), ...]``."""
    out: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if event is not None:
            out.append((event, data or {}))
    return out


# --------------------------------------------------------------------------- #
# store.manifest_url_index (unit)
# --------------------------------------------------------------------------- #


def test_manifest_url_index_builds_viewpage_urls(tmp_path):
    paths = _paths(tmp_path)
    store.save_manifest(
        paths,
        {
            "meta": {"base_url": "https://confluence.example.com"},
            "pages": {
                "111": {"path": "Confluence/Space/A.md", "version": 1},
                "222": {"path": "Confluence/Space/B.md", "version": 3},
            },
        },
    )
    index = store.manifest_url_index(paths)
    assert index == {
        "Confluence/Space/A.md": (
            "https://confluence.example.com/pages/viewpage.action?pageId=111"
        ),
        "Confluence/Space/B.md": (
            "https://confluence.example.com/pages/viewpage.action?pageId=222"
        ),
    }


def test_manifest_url_index_strips_trailing_slash(tmp_path):
    paths = _paths(tmp_path)
    store.save_manifest(
        paths,
        {
            "meta": {"base_url": "https://c.example.com/"},
            "pages": {"9": {"path": "P.md"}},
        },
    )
    assert store.manifest_url_index(paths) == {
        "P.md": "https://c.example.com/pages/viewpage.action?pageId=9"
    }


def test_manifest_url_index_missing_manifest_is_empty(tmp_path):
    assert store.manifest_url_index(_paths(tmp_path)) == {}


def test_manifest_url_index_no_base_url_is_empty(tmp_path):
    paths = _paths(tmp_path)
    store.save_manifest(paths, {"meta": {}, "pages": {"1": {"path": "X.md"}}})
    assert store.manifest_url_index(paths) == {}


# --------------------------------------------------------------------------- #
# POST /api/confluence/sync
# --------------------------------------------------------------------------- #


def _save_connection(paths: AppPaths) -> None:
    store.save_config(
        paths,
        {
            "base_url": "https://confluence.example.com",
            "root_url": "https://confluence.example.com/x?pageId=1",
        },
    )
    store.save_secret(paths, {"pat": "tok"})


def test_sync_unconfigured_returns_400(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(confluence_routes, "resolve_paths", lambda request: paths)

    with TestClient(create_app()) as client:
        resp = client.post("/api/confluence/sync")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "CONFLUENCE_NOT_CONFIGURED"


def test_sync_streams_frames(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    _save_connection(paths)
    monkeypatch.setattr(confluence_routes, "resolve_paths", lambda request: paths)

    async def fake_sync_stream(**kwargs):
        yield 'event: step\ndata: {"name": "resolve_root"}\n\n'
        yield 'event: done\ndata: {"synced": 0}\n\n'

    monkeypatch.setattr(confluence_routes, "sync_stream", fake_sync_stream)

    with TestClient(create_app()) as client:
        resp = client.post("/api/confluence/sync", json={"replace": False})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)
    names = [e for e, _ in events]
    assert names == ["step", "done"]


def test_sync_single_flight_reports_already_running(tmp_path, monkeypatch):
    """With the tenant lock held, the real sync_stream guard streams the error."""
    paths = _paths(tmp_path)
    _save_connection(paths)
    monkeypatch.setattr(confluence_routes, "resolve_paths", lambda request: paths)

    # Acquire the "local" lock so the guard trips (no event loop needed later:
    # the guard only inspects ``.locked()``, which is a plain bool).
    async def _hold() -> None:
        await sync._lock_for("local").acquire()

    asyncio.run(_hold())
    assert sync.SYNC_LOCKS["local"].locked() is True

    with TestClient(create_app()) as client:
        resp = client.post("/api/confluence/sync")
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events == [("error", {"code": "SYNC_ALREADY_RUNNING",
                                 "message": "синхронизация уже выполняется"})]


# --------------------------------------------------------------------------- #
# GET /api/confluence/status
# --------------------------------------------------------------------------- #


def test_status_running_reflects_lock(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    _save_connection(paths)
    monkeypatch.setattr(confluence_routes, "resolve_paths", lambda request: paths)

    with TestClient(create_app()) as client:
        assert client.get("/api/confluence/status").json()["running"] is False

        async def _hold() -> None:
            await sync._lock_for("local").acquire()

        asyncio.run(_hold())
        assert client.get("/api/confluence/status").json()["running"] is True


# --------------------------------------------------------------------------- #
# base_url derivation: _is_configured + /validate host guard
# --------------------------------------------------------------------------- #


def test_is_configured_true_with_root_url_only(tmp_path):
    """A parseable root_url alone (no separate base_url) satisfies the base."""
    cfg = {"root_url": "https://confluence.sberbank.ru/x?pageId=1"}
    secret = {"pat": "tok"}
    assert confluence_routes._is_configured(cfg, secret) is True
    # No creds → not configured.
    assert confluence_routes._is_configured(cfg, {}) is False
    # No usable target → not configured.
    assert confluence_routes._is_configured({"base_url": ""}, secret) is False


class _FakeConfluenceClient:
    """Minimal stand-in for ConfluenceClient used by /validate (no network)."""

    last_cfg: dict = {}

    @classmethod
    def from_config(cls, cfg, secret, **kw):
        cls.last_cfg = dict(cfg)
        return cls()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def get_page(self, page_id):
        return {"title": "Корень", "space": "ENG", "id": page_id}

    async def _request(self, method, path, **kw):
        class _R:
            @staticmethod
            def json():
                return {"totalSize": 4}

        return _R()


def _server_mode(monkeypatch, paths):
    monkeypatch.setattr(settings, "is_server", lambda: True)
    monkeypatch.setattr(confluence_routes, "resolve_paths", lambda request: paths)


def test_validate_server_mode_rejects_foreign_host(tmp_path, monkeypatch):
    """A root link on a host other than the admin host → HOST_NOT_ALLOWED."""
    paths = _paths(tmp_path)
    store.save_config(
        paths, {"root_url": "https://evil.example.com/pages/viewpage.action?pageId=9"}
    )
    store.save_secret(paths, {"pat": "tok"})
    _server_mode(monkeypatch, paths)

    with TestClient(create_app()) as client:
        resp = client.post(
            "/api/confluence/validate",
            json={},
            headers={"Authorization": "Bearer tok"},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "HOST_NOT_ALLOWED"


def test_validate_server_mode_same_host_ok_and_stores_base(tmp_path, monkeypatch):
    """Same-host root link validates and persists the derived base_url."""
    paths = _paths(tmp_path)
    store.save_config(
        paths,
        {"root_url": "https://confluence.sberbank.ru/pages/viewpage.action?pageId=1"},
    )
    store.save_secret(paths, {"pat": "tok"})
    _server_mode(monkeypatch, paths)
    monkeypatch.setattr(
        confluence_routes, "ConfluenceClient", _FakeConfluenceClient
    )

    with TestClient(create_app()) as client:
        resp = client.post(
            "/api/confluence/validate",
            json={},
            headers={"Authorization": "Bearer tok"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["root_title"] == "Корень"
    # Derived base persisted to the config file.
    assert store.load_config(paths)["base_url"] == "https://confluence.sberbank.ru"
    # The client was built with the derived base, not a stored/admin one.
    assert _FakeConfluenceClient.last_cfg["base_url"] == "https://confluence.sberbank.ru"


def test_validate_context_path_base_derived(tmp_path, monkeypatch):
    """A /confluence context-path link derives (and stores) the base with it."""
    paths = _paths(tmp_path)
    store.save_config(
        paths,
        {
            "root_url": (
                "https://confluence.sberbank.ru/confluence/pages/"
                "viewpage.action?pageId=7"
            )
        },
    )
    store.save_secret(paths, {"pat": "tok"})
    _server_mode(monkeypatch, paths)
    monkeypatch.setattr(
        confluence_routes, "ConfluenceClient", _FakeConfluenceClient
    )

    with TestClient(create_app()) as client:
        resp = client.post(
            "/api/confluence/validate",
            json={},
            headers={"Authorization": "Bearer tok"},
        )
    assert resp.status_code == 200
    assert (
        store.load_config(paths)["base_url"]
        == "https://confluence.sberbank.ru/confluence"
    )


# --------------------------------------------------------------------------- #
# Chat sources get a Confluence url when the path is in the manifest
# --------------------------------------------------------------------------- #


def test_chat_sources_carry_confluence_url(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    store.save_manifest(
        paths,
        {
            "meta": {"base_url": "https://confluence.example.com"},
            "pages": {"12345": {"path": "Confluence/Space/Page.md", "version": 1}},
        },
    )
    monkeypatch.setattr(chat_routes, "resolve_paths", lambda request: paths)

    sources = [
        {"n": 1, "title": "Conf", "path": "Confluence/Space/Page.md",
         "section_path": "", "score": 0.9, "depth": "chunk"},
        {"n": 2, "title": "Local", "path": "notes/local.md",
         "section_path": "", "score": 0.8, "depth": "chunk"},
    ]
    async def fake_build_rag_context(query, *args, **kwargs):
        return rag.RagContext(
            system_message={"role": "system", "content": "правила"},
            user_message={"role": "user", "content": f"Источники:\n\nВопрос: {query}"},
            sources=sources,
            context_chars=3,
        )

    async def fake_stream_chat(messages, gcfg):
        yield "ответ"

    monkeypatch.setattr(chat_routes.rag, "build_rag_context", fake_build_rag_context)
    monkeypatch.setattr(chat_routes.llm, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(chat_routes.llm, "files_present", lambda gcfg: None)

    with TestClient(create_app()) as client:
        resp = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "вопрос"}], "rag": True},
        )
    assert resp.status_code == 200
    events = dict(_parse_sse(resp.text))
    emitted = events["sources"]["sources"]
    by_path = {s["path"]: s for s in emitted}
    assert by_path["Confluence/Space/Page.md"]["url"] == (
        "https://confluence.example.com/pages/viewpage.action?pageId=12345"
    )
    assert "url" not in by_path["notes/local.md"]


def test_chat_sources_fall_back_to_frontmatter_url(tmp_path, monkeypatch):
    """Links survive when a pod replacement has removed the UI manifest."""
    paths = _paths(tmp_path)
    monkeypatch.setattr(chat_routes, "resolve_paths", lambda request: paths)

    sources = [
        {"n": 1, "title": "Conf", "path": "Confluence/Space/Page.md",
         "section_path": "", "score": 0.9, "depth": "chunk"},
    ]

    async def fake_build_rag_context(query, *args, **kwargs):
        return rag.RagContext(
            system_message={"role": "system", "content": "правила"},
            user_message={"role": "user", "content": f"Источники:\n\nВопрос: {query}"},
            sources=sources,
            context_chars=3,
        )

    async def fake_metadata(path, cv=None):
        assert path == "Confluence/Space/Page.md"
        return {
            "source": "confluence",
            "source_url": "https://confluence.example.com/pages/viewpage.action?pageId=77",
        }

    async def fake_stream_chat(messages, gcfg):
        yield "ответ [Источник 1]"

    monkeypatch.setattr(chat_routes.rag, "build_rag_context", fake_build_rag_context)
    monkeypatch.setattr(chat_routes.cognivault, "metadata", fake_metadata)
    monkeypatch.setattr(chat_routes.llm, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(chat_routes.llm, "files_present", lambda gcfg: None)

    with TestClient(create_app()) as client:
        resp = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "вопрос"}], "rag": True},
        )

    assert resp.status_code == 200
    events = dict(_parse_sse(resp.text))
    assert events["sources"]["sources"][0]["url"] == (
        "https://confluence.example.com/pages/viewpage.action?pageId=77"
    )


def test_source_url_enrichment_rejects_unsafe_frontmatter(monkeypatch):
    source = {"n": 1, "path": "Confluence/Page.md"}

    async def fake_metadata(path, cv=None):
        return {"source_url": "javascript:alert(1)"}

    monkeypatch.setattr(chat_routes.cognivault, "metadata", fake_metadata)
    asyncio.run(chat_routes._enrich_source_urls([source], {}, None))

    assert "url" not in source
