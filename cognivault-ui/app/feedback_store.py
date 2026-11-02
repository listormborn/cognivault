"""Persistent, reviewable answer feedback stored in SQLite.

The old feedback path appended an event to ``rag_log.jsonl``.  That is useful
telemetry, but it cannot be edited, queried by an administrator, or safely used
as a verified answer.  This module keeps the current state of a vote in a small
SQLite database while the JSONL event remains as an audit trail.

The database path is configured with ``UI_FEEDBACK_DB_PATH``.  In server mode
it defaults to ``<UI_DATA_DIR>/feedback.sqlite3``; the deployment must mount a
persistent volume there if feedback has to survive a pod replacement.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import settings
from .config import AppPaths, PATHS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_bucket TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    question TEXT NOT NULL DEFAULT '',
    answer TEXT NOT NULL DEFAULT '',
    sources_json TEXT NOT NULL DEFAULT '[]',
    vote TEXT NOT NULL CHECK (vote IN ('up', 'down')),
    reason TEXT,
    comment TEXT,
    status TEXT NOT NULL DEFAULT 'new'
        CHECK (status IN ('new', 'approved', 'rejected')),
    corrected_answer TEXT,
    applies_globally INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    reviewed_at TEXT,
    UNIQUE (user_bucket, chat_id, message_index)
);
CREATE INDEX IF NOT EXISTS feedback_status_updated
    ON feedback(status, updated_at DESC);
"""

_QUESTION_SPACE_RE = re.compile(r"\s+")
_QUESTION_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def database_path(paths: AppPaths | None = None) -> Path:
    explicit = (os.environ.get("UI_FEEDBACK_DB_PATH") or "").strip()
    if explicit:
        return Path(explicit)
    if settings.is_server():
        return settings.data_root() / "feedback.sqlite3"
    return (paths or PATHS).root / "feedback.sqlite3"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.executescript(_SCHEMA)
    # Forward-compatible with a database created by the first feedback build.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(feedback)")}
    if "applies_globally" not in columns:
        conn.execute(
            "ALTER TABLE feedback ADD COLUMN applies_globally INTEGER NOT NULL DEFAULT 0"
        )
    return conn


def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out = dict(row)
    try:
        sources = json.loads(out.pop("sources_json", "[]"))
    except (TypeError, json.JSONDecodeError):
        sources = []
    out["sources"] = sources if isinstance(sources, list) else []
    return out


def upsert(
    path: Path,
    *,
    user_bucket: str,
    chat_id: str,
    message_index: int,
    vote: str,
    reason: str | None,
    comment: str | None,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a vote or replace its editable fields.

    A changed vote returns the item to the administrator's ``new`` queue and
    clears any earlier approval.  This prevents a formerly approved 👍 from
    staying approved after the user switches it to 👎.
    """
    stamp = now_iso()
    sources_json = json.dumps(sources, ensure_ascii=False)
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO feedback (
                user_bucket, chat_id, message_index, question, answer,
                sources_json, vote, reason, comment, status,
                corrected_answer, created_at, updated_at, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', NULL, ?, ?, NULL)
            ON CONFLICT(user_bucket, chat_id, message_index) DO UPDATE SET
                question = excluded.question,
                answer = excluded.answer,
                sources_json = excluded.sources_json,
                vote = excluded.vote,
                reason = excluded.reason,
                comment = excluded.comment,
                status = 'new',
                corrected_answer = NULL,
                applies_globally = 0,
                updated_at = excluded.updated_at,
                reviewed_at = NULL
            """,
            (
                user_bucket,
                chat_id,
                message_index,
                question,
                answer,
                sources_json,
                vote,
                reason,
                comment,
                stamp,
                stamp,
            ),
        )
        row = conn.execute(
            """SELECT * FROM feedback
               WHERE user_bucket = ? AND chat_id = ? AND message_index = ?""",
            (user_bucket, chat_id, message_index),
        ).fetchone()
    return _decode(row) or {}


def get_for_answer(
    path: Path, *, user_bucket: str, chat_id: str, message_index: int
) -> dict[str, Any] | None:
    with _connect(path) as conn:
        row = conn.execute(
            """SELECT * FROM feedback
               WHERE user_bucket = ? AND chat_id = ? AND message_index = ?""",
            (user_bucket, chat_id, message_index),
        ).fetchone()
    return _decode(row)


def list_items(
    path: Path,
    *,
    status: str | None = None,
    vote: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 500))
    sql = "SELECT * FROM feedback"
    args: list[Any] = []
    clauses: list[str] = []
    if status:
        clauses.append("status = ?")
        args.append(status)
    if vote:
        clauses.append("vote = ?")
        args.append(vote)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    args.append(limit)
    with _connect(path) as conn:
        rows = conn.execute(sql, args).fetchall()
    return [item for row in rows if (item := _decode(row)) is not None]


def review(
    path: Path,
    *,
    feedback_id: int,
    status: str,
    corrected_answer: str | None,
    applies_globally: bool = False,
) -> dict[str, Any] | None:
    stamp = now_iso()
    with _connect(path) as conn:
        cur = conn.execute(
            """UPDATE feedback
               SET status = ?, corrected_answer = ?, applies_globally = ?,
                   reviewed_at = ?, updated_at = ?
               WHERE id = ?""",
            (
                status,
                corrected_answer,
                int(applies_globally),
                stamp,
                stamp,
                feedback_id,
            ),
        )
        if cur.rowcount == 0:
            return None
        row = conn.execute(
            "SELECT * FROM feedback WHERE id = ?", (feedback_id,)
        ).fetchone()
    return _decode(row)


def normalise_question(value: str) -> str:
    value = _QUESTION_PUNCT_RE.sub(" ", (value or "").casefold())
    return _QUESTION_SPACE_RE.sub(" ", value).strip()


def find_approved(
    path: Path, question: str, *, user_bucket: str
) -> dict[str, Any] | None:
    """Return an approved answer for the same normalised question.

    Matching is deliberately exact after case/punctuation normalisation.  A
    semantic near-match could answer the wrong question; that later stage needs
    its own confidence threshold and administrator-visible diagnostics.
    """
    wanted = normalise_question(question)
    if not wanted:
        return None
    with _connect(path) as conn:
        rows = conn.execute(
            """SELECT * FROM feedback
               WHERE status = 'approved'
                 AND (user_bucket = ? OR applies_globally = 1)
               ORDER BY reviewed_at DESC, updated_at DESC""",
            (user_bucket,),
        ).fetchall()
    for row in rows:
        item = _decode(row)
        if item and normalise_question(str(item.get("question") or "")) == wanted:
            answer = str(
                item.get("corrected_answer") or item.get("answer") or ""
            ).strip()
            if answer:
                item["verified_answer"] = answer
                return item
    return None
