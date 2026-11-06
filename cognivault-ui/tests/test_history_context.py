"""Persistence contract for server-derived per-chat active context."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import history  # noqa: E402
from app.config import AppPaths  # noqa: E402


def _paths(tmp_path) -> AppPaths:
    return AppPaths(root=tmp_path / ".cognivault-ui")


def test_active_subject_is_saved_and_loaded(tmp_path):
    paths = _paths(tmp_path)
    history.save_chat(
        "chat-1",
        [{"role": "user", "content": "Что такое мониторинг моделей?"}],
        paths,
        context={"subject": "  Мониторинг   моделей  "},
    )

    assert history.load_chat("chat-1", paths)["context"] == {
        "subject": "Мониторинг моделей"
    }


def test_legacy_save_preserves_existing_active_subject(tmp_path):
    paths = _paths(tmp_path)
    history.save_chat(
        "chat-1",
        [{"role": "user", "content": "первый вопрос"}],
        paths,
        context={"subject": "Fincert Exporter"},
    )
    history.save_chat(
        "chat-1",
        [{"role": "user", "content": "второй вопрос"}],
        paths,
    )

    assert history.load_chat("chat-1", paths)["context"] == {
        "subject": "Fincert Exporter"
    }


def test_invalid_active_subject_is_not_persisted(tmp_path):
    paths = _paths(tmp_path)
    saved = history.save_chat(
        "chat-1",
        [{"role": "user", "content": "вопрос"}],
        paths,
        context={"subject": "x" * 161, "unexpected": "ignored"},
    )

    assert "context" not in saved
