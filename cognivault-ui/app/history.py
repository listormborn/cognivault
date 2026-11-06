"""Chat history persistence under ``~/.cognivault-ui/history/{id}.json``.

Each file: ``{id, title, created_at, updated_at, messages:[...], context:{...}}``.
Writes are atomic; after every save the store is pruned to the newest 10 chats.
RAG system prompts are never persisted.  ``context`` stores only bounded,
server-derived conversation state, currently the canonical active subject.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from .config import PATHS, AppPaths

MAX_CHATS = 10


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _title_from_messages(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            content = str(msg.get("content", "")).strip()
            if content:
                return content[:60]
    return "Без названия"


def _history_files(paths: AppPaths) -> list[Any]:
    if not paths.history_dir.is_dir():
        return []
    return list(paths.history_dir.glob("*.json"))


def _read_chat(path: Any) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _prune(paths: AppPaths) -> None:
    """Keep only the newest ``MAX_CHATS`` chats by ``updated_at``."""
    chats: list[tuple[str, Any]] = []
    for f in _history_files(paths):
        data = _read_chat(f)
        updated = (data or {}).get("updated_at", "") if data else ""
        chats.append((str(updated), f))
    if len(chats) <= MAX_CHATS:
        return
    chats.sort(key=lambda t: t[0], reverse=True)
    for _, f in chats[MAX_CHATS:]:
        try:
            f.unlink()
        except OSError:
            pass


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip system messages (never store RAG prompts); keep known fields."""
    clean: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        entry: dict[str, Any] = {"role": role, "content": msg.get("content", "")}
        for key in ("rag", "sources", "truncated", "context_chars", "invalid_citations"):
            if key in msg:
                entry[key] = msg[key]
        clean.append(entry)
    return clean


def _sanitize_context(value: Any) -> dict[str, str] | None:
    """Return the small, trusted subset of persisted conversation state."""
    if not isinstance(value, dict):
        return None
    subject = " ".join(str(value.get("subject") or "").strip().split())
    if not subject or len(subject) > 160 or len(subject.split()) > 12:
        return None
    return {"subject": subject}


def list_chats(paths: AppPaths | None = None) -> list[dict[str, Any]]:
    """Return newest-10 chat summaries, sorted by ``updated_at`` descending."""
    paths = paths or PATHS
    summaries: list[dict[str, Any]] = []
    for f in _history_files(paths):
        data = _read_chat(f)
        if not data:
            continue
        messages = data.get("messages", []) or []
        has_rag = any(
            isinstance(m, dict) and m.get("role") == "assistant" and m.get("rag")
            for m in messages
        )
        summaries.append(
            {
                "id": data.get("id", f.stem),
                "title": data.get("title", ""),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "message_count": len(messages),
                "rag": has_rag,
            }
        )
    summaries.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
    return summaries[:MAX_CHATS]


def load_chat(chat_id: str, paths: AppPaths | None = None) -> dict[str, Any] | None:
    """Load a full chat by id, or ``None`` if missing/unreadable."""
    paths = paths or PATHS
    path = paths.history_dir / f"{_safe_id(chat_id)}.json"
    if not path.is_file():
        return None
    return _read_chat(path)


def save_chat(
    chat_id: str,
    messages: list[dict[str, Any]],
    paths: AppPaths | None = None,
    created_at: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a chat atomically, then prune to the newest ``MAX_CHATS``.

    System messages are dropped before writing. ``created_at`` is preserved from
    any existing file unless explicitly provided.  Omitting ``context`` keeps
    the existing active subject so legacy callers cannot erase it accidentally.
    """
    paths = paths or PATHS
    paths.ensure_dirs()
    safe = _safe_id(chat_id)
    path = paths.history_dir / f"{safe}.json"

    existing = _read_chat(path) if path.is_file() else None
    now = _now_iso()
    clean = _sanitize_messages(messages)
    record = {
        "id": safe,
        "title": _title_from_messages(clean),
        "created_at": created_at or (existing or {}).get("created_at") or now,
        "updated_at": now,
        "messages": clean,
    }
    clean_context = _sanitize_context(
        context if context is not None else (existing or {}).get("context")
    )
    if clean_context is not None:
        record["context"] = clean_context

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)

    _prune(paths)
    return record


def delete_chat(chat_id: str, paths: AppPaths | None = None) -> bool:
    """Delete a chat by id. Returns ``True`` if a file was removed."""
    paths = paths or PATHS
    path = paths.history_dir / f"{_safe_id(chat_id)}.json"
    if not path.is_file():
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False


def count_chats(paths: AppPaths | None = None) -> int:
    paths = paths or PATHS
    return len(_history_files(paths))


def _safe_id(chat_id: str) -> str:
    """Guard against path traversal in a chat id used as a filename."""
    base = os.path.basename(str(chat_id))
    if not base or base in (".", ".."):
        raise ValueError("invalid chat id")
    return base
