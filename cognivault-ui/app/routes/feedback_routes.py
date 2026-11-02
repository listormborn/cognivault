"""Editable answer feedback plus the administrator review queue.

The current vote is kept in SQLite and can be changed.  A compact event is also
appended to ``rag_log.jsonl`` as an immutable diagnostic audit trail.

The body is validated by hand (the project uses no pydantic models) and errors
use the standard ``{"error": {"code", "message"}}`` envelope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from .. import feedback_store, history, rag_log, settings
from ..deps import require_admin, resolve_paths, user_bucket

router = APIRouter(prefix="/api")

# Свободный комментарий пользователя — обрезаем, чтобы одна строка JSONL не
# распухла до мегабайтов.
MAX_COMMENT_CHARS = 1000

_VOTES = ("up", "down")
_REASONS = (
    "incorrect",
    "missing_information",
    "wrong_sources",
    "off_topic",
    "unclear",
    "other",
)
_STATUSES = ("new", "approved", "rejected")


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"code": code, "message": message}}
    )


def _db_path(request: Request) -> Path:
    """Global review database (per-user identity remains a column)."""
    return feedback_store.database_path(resolve_paths(request))


def _identity(request: Request) -> str:
    if not settings.is_server():
        return "local"
    auth = request.headers.get("authorization") or ""
    token = auth.removeprefix("Bearer ").strip()
    return user_bucket(token)


def _request_snapshot(request: Request, chat_id: str, index: int) -> dict[str, Any]:
    paths = resolve_paths(request)
    records = rag_log.read_records(paths)
    snapshot: dict[str, Any] = {}
    for record in reversed(records):
        if (
            record.get("type") == "request"
            and record.get("chat_id") == chat_id
            and record.get("message_index") == index
        ):
            snapshot = dict(record)
            break
    # History is the durable source for the rendered answer and full source
    # objects (including title/url).  It also lets feedback remain useful after
    # rag_log rotation has removed the matching request record.
    chat = history.load_chat(chat_id, paths) or {}
    messages = chat.get("messages") if isinstance(chat, dict) else None
    if isinstance(messages, list) and 0 <= index < len(messages):
        answer = messages[index]
        if isinstance(answer, dict) and answer.get("role") == "assistant":
            snapshot["answer_text"] = str(answer.get("content") or "")
            if isinstance(answer.get("sources"), list):
                snapshot["sources"] = answer["sources"]
        for prior in reversed(messages[:index]):
            if isinstance(prior, dict) and prior.get("role") == "user":
                snapshot["question_raw"] = str(prior.get("content") or "")
                break
    return snapshot


@router.post("/feedback")
async def post_feedback(request: Request) -> Any:
    """Record a 👍/👎 for one assistant answer.

    Body: ``{chat_id: str, message_index: int, vote: "up"|"down", comment?: str}``.
    Returns ``{"ok": true}``; a malformed body is a 400 ``BAD_REQUEST`` and a
    failed log write a 500 ``LOG_WRITE_FAILED``.
    """
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — any parse failure is a client error
        return _error(400, "BAD_REQUEST", "тело запроса должно быть JSON-объектом")
    if not isinstance(body, dict):
        return _error(400, "BAD_REQUEST", "тело запроса должно быть объектом")

    chat_id = str(body.get("chat_id") or "").strip()
    if not chat_id:
        return _error(400, "BAD_REQUEST", "chat_id обязателен")

    index = body.get("message_index")
    # ``bool`` is an ``int`` subclass — reject it explicitly.
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        return _error(400, "BAD_REQUEST", "message_index должен быть целым числом ≥ 0")

    vote = body.get("vote")
    if vote not in _VOTES:
        return _error(400, "BAD_REQUEST", "vote должен быть 'up' или 'down'")

    reason = body.get("reason")
    if reason in (None, ""):
        reason = None
    elif reason not in _REASONS:
        return _error(400, "BAD_REQUEST", "неизвестная причина оценки")

    raw_comment = body.get("comment")
    comment = (
        str(raw_comment).strip()[:MAX_COMMENT_CHARS]
        if isinstance(raw_comment, str)
        else ""
    )

    snapshot = _request_snapshot(request, chat_id, index)
    try:
        item = feedback_store.upsert(
            _db_path(request),
            user_bucket=_identity(request),
            chat_id=chat_id,
            message_index=index,
            vote=vote,
            reason=reason,
            comment=comment or None,
            question=str(snapshot.get("question_raw") or ""),
            answer=str(snapshot.get("answer_text") or ""),
            sources=(
                snapshot.get("sources")
                if isinstance(snapshot.get("sources"), list)
                else []
            ),
        )
    except Exception:  # noqa: BLE001 — storage error must use the API envelope
        return _error(500, "FEEDBACK_WRITE_FAILED", "не удалось сохранить оценку")

    record: dict[str, Any] = {
        "type": "feedback",
        "ts": rag_log.now_iso(),
        "chat_id": chat_id,
        "message_index": index,
        "vote": vote,
        "reason": reason,
        "comment": comment or None,
    }
    # SQLite above is the source of truth.  The append-only event is useful for
    # diagnostics but must not turn a successfully saved vote into a UI error.
    rag_log.append(resolve_paths(request), record)
    return {"ok": True, "feedback": item}


@router.get("/feedback/{chat_id}/{message_index}")
async def get_feedback(request: Request, chat_id: str, message_index: int) -> Any:
    if message_index < 0:
        return _error(400, "BAD_REQUEST", "message_index должен быть ≥ 0")
    item = feedback_store.get_for_answer(
        _db_path(request),
        user_bucket=_identity(request),
        chat_id=chat_id,
        message_index=message_index,
    )
    return {"feedback": item}


@router.get("/admin/feedback", dependencies=[Depends(require_admin)])
async def list_feedback(
    request: Request, status: str | None = "new", limit: int = 100
) -> Any:
    if status in (None, "", "all"):
        status = None
    elif status not in _STATUSES:
        return _error(400, "BAD_REQUEST", "неизвестный статус")
    return {
        "items": feedback_store.list_items(
            _db_path(request), status=status, limit=limit
        )
    }


@router.patch("/admin/feedback/{feedback_id}", dependencies=[Depends(require_admin)])
async def review_feedback(request: Request, feedback_id: int) -> Any:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _error(400, "BAD_REQUEST", "тело запроса должно быть JSON-объектом")
    if not isinstance(body, dict):
        return _error(400, "BAD_REQUEST", "тело запроса должно быть объектом")
    status = body.get("status")
    if status not in ("approved", "rejected"):
        return _error(400, "BAD_REQUEST", "status должен быть approved или rejected")
    raw = body.get("corrected_answer")
    corrected = (
        str(raw).strip()[: rag_log.MAX_TEXT_CHARS]
        if isinstance(raw, str)
        else ""
    )
    applies_globally = body.get("applies_globally", False)
    if not isinstance(applies_globally, bool):
        return _error(400, "BAD_REQUEST", "applies_globally должен быть boolean")
    item = feedback_store.review(
        _db_path(request),
        feedback_id=feedback_id,
        status=status,
        corrected_answer=corrected or None,
        applies_globally=applies_globally if status == "approved" else False,
    )
    if item is None:
        return _error(404, "NOT_FOUND", "оценка не найдена")
    return {"feedback": item}
