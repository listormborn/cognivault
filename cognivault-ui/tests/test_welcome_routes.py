"""The empty-chat greeting uses corpus facts and never an LLM."""

from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import catalog, corpus_tree, settings  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture(autouse=True)
def _local_mode(monkeypatch):
    monkeypatch.setattr(settings, "is_server", lambda: False)


def _payload() -> dict:
    prefix = "Confluence/OASISEXT/OASIS External Home"
    documents = [
        {
            "path": f"{prefix}/Архив.md",
            "title": "Архив",
            "summary": None,
            "size": 500,
        },
        {
            "path": f"{prefix}/База знаний.md",
            "title": "База знаний",
            "summary": "Инструкции.",
            "size": 2000,
        },
        {
            "path": f"{prefix}/Продукты.md",
            "title": "Продукты",
            "summary": None,
            "size": 500,
        },
        {
            "path": f"{prefix}/Продукты/Fincert.md",
            "title": "Fincert",
            "summary": "Продукт.",
            "size": 2000,
        },
    ]
    return {
        "status": "ok",
        "documents": documents,
        "total": len(documents),
        "documents_with_summary": 2,
        "document_extensions": ["md"],
    }


def test_top_level_titles_fold_storage_corridor() -> None:
    assert corpus_tree.top_level_titles(_payload()) == [
        "База знаний",
        "Продукты",
    ]


def test_welcome_returns_real_topics_without_summaries(monkeypatch) -> None:
    async def fake_payload(cv=None):
        return _payload()

    monkeypatch.setattr(catalog, "payload", fake_payload)
    response = TestClient(create_app()).get("/api/welcome")

    assert response.status_code == 200
    assert response.json() == {
        "catalog_available": True,
        "topics": ["База знаний", "Продукты"],
        "total": 3,
    }
    assert "summary" not in response.text


def test_welcome_degrades_when_catalogue_is_unavailable(monkeypatch) -> None:
    async def no_payload(cv=None):
        return None

    monkeypatch.setattr(catalog, "payload", no_payload)
    response = TestClient(create_app()).get("/api/welcome")

    assert response.status_code == 200
    assert response.json() == {
        "catalog_available": False,
        "topics": [],
        "total": None,
    }
