"""Static SPA-shell cache-policy tests.

After a redeploy the browser/proxy must never serve a stale cached ``app.js``
(or ``index.html``/``style.css``/``favicon.svg``). ``NoCacheStaticFiles`` marks
those shell assets ``Cache-Control: no-store`` and strips revalidation
validators so a client always refetches a fresh 200. ``/api/*`` and ``/healthz``
must be left untouched.

pytest + Starlette ``TestClient`` (no real network), LOCAL mode.
"""

from __future__ import annotations

import os
import sys
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import settings  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture(autouse=True)
def _local_mode(monkeypatch):
    monkeypatch.setattr(settings, "is_server", lambda: False)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def test_shell_assets_are_no_store(client: TestClient) -> None:
    """Every SPA shell asset must carry no-store and no long-cache validators."""
    for path in ("/", "/index.html", "/app.js", "/style.css", "/favicon.svg"):
        resp = client.get(path)
        assert resp.status_code == 200, path
        cc = resp.headers.get("cache-control", "")
        assert "no-store" in cc, f"{path} cache-control={cc!r}"
        # Strip revalidation validators so a stale client can't 304 its way back.
        assert "etag" not in resp.headers, path
        assert "last-modified" not in resp.headers, path


def test_app_js_served_fresh_and_is_javascript(client: TestClient) -> None:
    """app.js still serves JS content, and a conditional request re-serves 200."""
    resp = client.get("/app.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers.get("content-type", "")
    assert "no-store" in resp.headers.get("cache-control", "")
    # Even if a client presents stale validators, it gets a fresh body, not 304.
    again = client.get(
        "/app.js", headers={"If-None-Match": '"stale"', "If-Modified-Since": "x"}
    )
    assert again.status_code == 200


def test_new_chat_uses_product_question_suggestions(client: TestClient) -> None:
    script = client.get("/app.js").text
    for question in (
        "Что такое мониторинг?",
        "Как поставить модель на дообучение?",
        "Как работать с Fincert Enricher?",
        "Как настроить алертинг?",
        "Чем отличаются батчевые и стриминговые потоки?",
        "Как получить доступ к Bitbucket Fincert?",
    ):
        assert question in script
    assert "Расскажи о разделе «" not in script


def test_rag_cannot_be_disabled_in_the_ui(client: TestClient) -> None:
    page = client.get("/").text
    script = client.get("/app.js").text

    assert 'id="rag-switch"' not in page
    assert 'id="cfg-rag-default-on"' not in page
    assert "toggleRag" not in script
    assert "rag: true" in script


def test_index_html_served_at_root(client: TestClient) -> None:
    """html=True behaviour preserved: GET / returns the SPA HTML shell."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


def test_chat_composer_preserves_technical_names(client: TestClient) -> None:
    """Mobile/desktop text assistance must not rewrite product names."""

    class ComposerParser(HTMLParser):
        attrs: dict[str, str | None] | None = None

        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            if tag == "textarea" and values.get("id") == "composer-input":
                self.attrs = values

    parser = ComposerParser()
    parser.feed(client.get("/").text)
    assert parser.attrs is not None
    assert parser.attrs["autocomplete"] == "off"
    assert parser.attrs["autocorrect"] == "off"
    assert parser.attrs["autocapitalize"] == "off"
    assert parser.attrs["spellcheck"] == "false"
    assert parser.attrs["data-gramm"] == "false"


def test_ordered_markdown_lists_preserve_explicit_item_numbers(
    client: TestClient,
) -> None:
    script = client.get("/app.js").text
    assert r"(\d+)[.)]" in script
    assert "itemNumber = Number(ol[1])" in script
    assert "<li value=" in script


def test_api_and_healthz_not_no_store(client: TestClient) -> None:
    """API/probe responses must NOT be forced no-store (unchanged behaviour)."""
    for path in ("/api/config", "/healthz"):
        resp = client.get(path)
        assert resp.status_code == 200, path
        cc = resp.headers.get("cache-control", "")
        assert "no-store" not in cc, f"{path} cache-control={cc!r}"
