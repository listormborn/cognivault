"""Welcome data for an empty chat — real corpus sections, no model call."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from .. import catalog, corpus_tree, deps

router = APIRouter(prefix="/api", tags=["welcome"])

_TOPIC_LIMIT = 8


def _total(data: dict[str, Any]) -> int:
    raw = data.get("total")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    return len(catalog.documents_of(data))


@router.get("/welcome")
async def welcome(request: Request) -> dict[str, Any]:
    """Describe the caller's indexed corpus without invoking KitAI/GigaChat.

    Only section titles and the document count reach the browser; summaries and
    paths remain server-side. Catalogue failures are a normal degraded state,
    represented explicitly so the SPA can retain its static welcome text.
    """
    data = await catalog.payload(deps.cv_context(request))
    if data is None:
        return {"catalog_available": False, "topics": [], "total": None}
    return {
        "catalog_available": True,
        "topics": corpus_tree.top_level_titles(data, limit=_TOPIC_LIMIT),
        "total": _total(data),
    }
