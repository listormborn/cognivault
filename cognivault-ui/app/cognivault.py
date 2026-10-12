"""Async client for the CogniVault REST API.

The base URL and bearer token are read from the freshly-loaded config on every
call so that a config change takes effect without a restart.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .config import load_config


class CogniVaultError(Exception):
    """A typed error from a CogniVault upstream call.

    Carries an HTTP-ish ``status`` and a short ``body`` excerpt so routes can
    surface a meaningful envelope to the browser.
    """

    def __init__(self, message: str, status: int, body: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.body = body


def _cv_config() -> tuple[str, str]:
    """Return ``(base_url, token)`` from current config (base_url trailing '/' stripped)."""
    cfg = load_config()
    cv = cfg.get("cognivault", {})
    base = str(cv.get("base_url", "")).rstrip("/")
    token = str(cv.get("token", "") or "")
    return base, token


def _resolve_cv(cv: dict[str, Any] | None) -> tuple[str, str]:
    """Resolve ``(base_url, token)`` from an explicit ``cv`` dict or the config file.

    When ``cv`` is provided (server mode: per-request context) its ``base_url``/
    ``token`` are used; when ``None`` (local mode) the on-disk config is read via
    :func:`_cv_config`, preserving the exact historical behaviour.
    """
    if cv is not None:
        base = str(cv.get("base_url", "")).rstrip("/")
        token = str(cv.get("token", "") or "")
        return base, token
    return _cv_config()


def _auth_headers(token: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if extra:
        headers.update(extra)
    return headers


def _excerpt(text: str, limit: int = 500) -> str:
    return text[:limit]


async def health(cv: dict[str, Any] | None = None) -> tuple[bool, int, str | None]:
    """Probe ``GET {base}/health`` (no auth, 3s timeout).

    Returns ``(ok, latency_ms, error)``. Never raises.
    """
    base, _ = _resolve_cv(cv)
    if not base:
        return False, 0, "base_url не настроен"
    url = f"{base}/health"
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(url)
        latency = int((time.perf_counter() - start) * 1000)
        if resp.status_code == 200:
            return True, latency, None
        return False, latency, f"HTTP {resp.status_code}"
    except httpx.HTTPError as exc:
        latency = int((time.perf_counter() - start) * 1000)
        return False, latency, str(exc) or exc.__class__.__name__


async def semantic_search(
    query: str,
    limit: int,
    cv: dict[str, Any] | None = None,
    *,
    include_archived: bool | None = None,
) -> dict[str, Any]:
    """POST ``/api/vault/search/semantic``.

    The body is sent as raw UTF-8 (``ensure_ascii=False``) so Cyrillic queries
    are not escaped over the wire.

    ``include_archived`` is written only when set, so an older backend that does
    not know the key still sees the historical ``{query, limit}`` payload.
    """
    base, token = _resolve_cv(cv)
    url = f"{base}/api/vault/search/semantic"
    payload: dict[str, Any] = {"query": query, "limit": limit}
    if include_archived is not None:
        payload["include_archived"] = bool(include_archived)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = _auth_headers(
        token, {"Content-Type": "application/json; charset=utf-8"}
    )
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, content=body, headers=headers)
    if resp.status_code != 200:
        raise CogniVaultError(
            f"semantic search failed ({resp.status_code})",
            resp.status_code,
            _excerpt(resp.text),
        )
    return resp.json()


async def hybrid_search(
    query: str,
    limit: int,
    cv: dict[str, Any] | None = None,
    *,
    group_by_section: bool = False,
    section_max_chars: int | None = None,
    include_archived: bool | None = None,
) -> dict[str, Any]:
    """POST ``/api/vault/search/hybrid``.

    Same request/response contract as :func:`semantic_search` (raw UTF-8 body,
    Bearer auth, ``{results: [...]}`` response). Used by the auto RAG mode; the
    caller falls back to :func:`semantic_search` if this errors/404s.

    ``group_by_section`` asks the backend to deduplicate hits by section and to
    attach ``parent_id`` plus ``section_text`` (the full section body from the
    index, truncated to ``section_max_chars``) to every result. Both keys are
    written into the body **only when set**, so an older backend that does not
    know them still sees exactly the historical ``{query, limit}`` payload.
    """
    base, token = _resolve_cv(cv)
    url = f"{base}/api/vault/search/hybrid"
    payload: dict[str, Any] = {"query": query, "limit": limit}
    if group_by_section:
        payload["group_by_section"] = True
    if section_max_chars is not None:
        payload["section_max_chars"] = section_max_chars
    if include_archived is not None:
        payload["include_archived"] = bool(include_archived)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = _auth_headers(
        token, {"Content-Type": "application/json; charset=utf-8"}
    )
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, content=body, headers=headers)
    if resp.status_code != 200:
        raise CogniVaultError(
            f"hybrid search failed ({resp.status_code})",
            resp.status_code,
            _excerpt(resp.text),
        )
    return resp.json()


async def content(path: str, cv: dict[str, Any] | None = None) -> str:
    """GET ``/api/vault/content?path=<path>`` returning the document body.

    The upstream returns ``{path, content}``; this returns just the ``content``
    string. Raises :class:`CogniVaultError` (with status + body excerpt) on a
    non-200 so the caller can fall back to bare chunks.
    """
    base, token = _resolve_cv(cv)
    url = f"{base}/api/vault/content"
    headers = _auth_headers(token)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, params={"path": path}, headers=headers)
    if resp.status_code != 200:
        raise CogniVaultError(
            f"content failed ({resp.status_code})",
            resp.status_code,
            _excerpt(resp.text),
        )
    data = resp.json()
    return str(data.get("content", "") or "")


async def metadata(path: str, cv: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET ``/api/vault/metadata?path=<path>`` and return its metadata object.

    Confluence sync stores the original page URL in Markdown frontmatter as
    ``source_url``. Unlike the UI-side sync manifest, that metadata travels
    with the vault document and therefore survives a UI pod replacement.
    """
    base, token = _resolve_cv(cv)
    url = f"{base}/api/vault/metadata"
    headers = _auth_headers(token)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, params={"path": path}, headers=headers)
    if resp.status_code != 200:
        raise CogniVaultError(
            f"metadata failed ({resp.status_code})",
            resp.status_code,
            _excerpt(resp.text),
        )
    data = resp.json()
    value = data.get("metadata", {}) if isinstance(data, dict) else {}
    return value if isinstance(value, dict) else {}


async def context(
    query: str,
    token_budget: int,
    min_score: float | None,
    filters: dict[str, Any] | None = None,
    cv: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """POST ``/api/vault/context`` returning the grouped context response."""
    base, token = _resolve_cv(cv)
    url = f"{base}/api/vault/context"
    payload: dict[str, Any] = {
        "query": query,
        "token_budget": token_budget,
        "min_score": min_score,
    }
    if filters:
        payload["filters"] = filters
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = _auth_headers(
        token, {"Content-Type": "application/json; charset=utf-8"}
    )
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, content=body, headers=headers)
    if resp.status_code != 200:
        raise CogniVaultError(
            f"context failed ({resp.status_code})",
            resp.status_code,
            _excerpt(resp.text),
        )
    return resp.json()


async def upload(
    file_bytes: bytes, filename: str, cv: dict[str, Any] | None = None
) -> dict[str, Any]:
    """POST ``/api/vault/upload`` as multipart (field ``file``).

    Raises :class:`CogniVaultError` (with status + body excerpt) on non-200.
    """
    base, token = _resolve_cv(cv)
    url = f"{base}/api/vault/upload"
    headers = _auth_headers(token)
    files = {"file": (filename, file_bytes, "application/zip")}
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, files=files, headers=headers)
    if resp.status_code != 200:
        raise CogniVaultError(
            f"upload failed ({resp.status_code})",
            resp.status_code,
            _excerpt(resp.text),
        )
    return resp.json()


# --------------------------------------------------------------------------- #
# Write wrappers (create / update / delete / list / clear)
# --------------------------------------------------------------------------- #
#
# These power the Confluence-source sync. Each accepts the same optional ``cv``
# context as the read helpers and raises :class:`CogniVaultError` on an
# *unexpected* non-2xx. Two statuses are treated as recoverable signals rather
# than errors so the caller can self-heal POST/PUT drift:
#
# * ``create_note`` → ``EXISTS`` on 409 (fall back to update);
# * ``update_note`` → ``MISSING`` on 404 (fall back to create).

CREATED = "created"
EXISTS = "exists"
UPDATED = "updated"
MISSING = "missing"
DELETED = "deleted"


def _content_body(path: str, content: str) -> bytes:
    """Serialise a ``{path, content}`` write body as raw UTF-8 (Cyrillic-safe)."""
    return json.dumps({"path": path, "content": content}, ensure_ascii=False).encode(
        "utf-8"
    )


async def create_note(
    path: str, content: str, cv: dict[str, Any] | None = None
) -> str:
    """POST ``/api/vault/content`` to create a note.

    Returns :data:`CREATED` on a 2xx (typically 201); returns :data:`EXISTS` on
    a **409** so the caller can fall back to :func:`update_note`. Any other
    non-2xx raises :class:`CogniVaultError`.
    """
    base, token = _resolve_cv(cv)
    url = f"{base}/api/vault/content"
    headers = _auth_headers(token, {"Content-Type": "application/json; charset=utf-8"})
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, content=_content_body(path, content), headers=headers)
    if resp.status_code == 409:
        return EXISTS
    if not (200 <= resp.status_code < 300):
        raise CogniVaultError(
            f"create note failed ({resp.status_code})",
            resp.status_code,
            _excerpt(resp.text),
        )
    return CREATED


async def update_note(
    path: str, content: str, cv: dict[str, Any] | None = None
) -> str:
    """PUT ``/api/vault/content`` to overwrite a note.

    Returns :data:`UPDATED` on a 2xx; returns :data:`MISSING` on a **404** so the
    caller can fall back to :func:`create_note`. Any other non-2xx raises.
    """
    base, token = _resolve_cv(cv)
    url = f"{base}/api/vault/content"
    headers = _auth_headers(token, {"Content-Type": "application/json; charset=utf-8"})
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.put(url, content=_content_body(path, content), headers=headers)
    if resp.status_code == 404:
        return MISSING
    if not (200 <= resp.status_code < 300):
        raise CogniVaultError(
            f"update note failed ({resp.status_code})",
            resp.status_code,
            _excerpt(resp.text),
        )
    return UPDATED


async def delete_note(path: str, cv: dict[str, Any] | None = None) -> str:
    """DELETE ``/api/vault/content`` for ``path``.

    A **404** is tolerated (already gone) and reported as :data:`DELETED`. Any
    other non-2xx raises :class:`CogniVaultError`.
    """
    base, token = _resolve_cv(cv)
    url = f"{base}/api/vault/content"
    headers = _auth_headers(token, {"Content-Type": "application/json; charset=utf-8"})
    body = json.dumps({"path": path}, ensure_ascii=False).encode("utf-8")
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.request("DELETE", url, content=body, headers=headers)
    if resp.status_code == 404:
        return DELETED
    if not (200 <= resp.status_code < 300):
        raise CogniVaultError(
            f"delete note failed ({resp.status_code})",
            resp.status_code,
            _excerpt(resp.text),
        )
    return DELETED


async def list_files(
    cv: dict[str, Any] | None = None,
    recursive: bool = True,
    timeout: float | None = None,
) -> list[str]:
    """GET ``/api/vault/files`` and return the file entry paths (files only).

    Tolerates several response shapes: a bare list, or a dict with ``files`` /
    ``entries`` / ``results``; entries may be plain path strings or objects with
    a ``path``/``name`` (directory-typed entries are filtered out — the backend
    DOES return them, ``{name, path, type: 'file' | 'directory'}``).

    ``timeout`` overrides the default 30s. A caller on the chat hot path
    (:mod:`app.corpus_map`) must not be able to inherit a 30s stall on every
    turn, while the bulk callers (``clear_vault``) keep the patient default.
    """
    base, token = _resolve_cv(cv)
    url = f"{base}/api/vault/files"
    headers = _auth_headers(token)
    params = {"recursive": "true" if recursive else "false"}
    async with httpx.AsyncClient(timeout=timeout if timeout is not None else 30.0) as client:
        resp = await client.get(url, params=params, headers=headers)
    if resp.status_code != 200:
        raise CogniVaultError(
            f"list files failed ({resp.status_code})",
            resp.status_code,
            _excerpt(resp.text),
        )
    data = resp.json()
    if isinstance(data, dict):
        entries = data.get("files") or data.get("entries") or data.get("results") or []
    else:
        entries = data or []
    out: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, dict):
            etype = str(entry.get("type", "") or "").lower()
            if etype in ("dir", "directory", "folder"):
                continue
            if entry.get("isDir") or entry.get("directory"):
                continue
            path = entry.get("path") or entry.get("name")
            if path:
                out.append(str(path))
    return out


async def catalog(
    cv: dict[str, Any] | None = None,
    limit: int | None = None,
    offset: int | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """GET ``/api/vault/catalog`` — the per-document annotations written at index time.

    Returns the response verbatim: ``{status, summaries_enabled, reason,
    documents, total, offset, documents_with_summary, document_extensions}``.
    Every field is load-bearing for the caller and nothing is normalised here —
    in particular ``document_extensions`` IS the service's definition of
    "document" and must not be second-guessed by a client-side allowlist.

    ``limit``/``offset`` are omitted from the query when unset, so an older
    backend sees the bare request it always saw; ``timeout`` exists for the same
    reason as :func:`list_files`' — this is called from the chat hot path and
    must not be able to add 30s to a turn.
    """
    base, token = _resolve_cv(cv)
    url = f"{base}/api/vault/catalog"
    headers = _auth_headers(token)
    params: dict[str, str] = {}
    if limit is not None:
        params["limit"] = str(limit)
    if offset is not None:
        params["offset"] = str(offset)
    async with httpx.AsyncClient(
        timeout=timeout if timeout is not None else 30.0
    ) as client:
        resp = await client.get(url, params=params, headers=headers)
    if resp.status_code != 200:
        raise CogniVaultError(
            f"catalog failed ({resp.status_code})",
            resp.status_code,
            _excerpt(resp.text),
        )
    data = resp.json()
    return data if isinstance(data, dict) else {}


async def clear_vault(cv: dict[str, Any] | None = None) -> dict[str, Any]:
    """Delete every file in the vault (best-effort). Used by replace mode.

    Lists files then deletes each one, collecting per-file failures rather than
    aborting. Returns ``{deleted, failed, total}`` where ``failed`` is a list of
    ``(path, status)`` tuples.
    """
    files = await list_files(cv, recursive=True)
    deleted = 0
    failed: list[tuple[str, int]] = []
    for path in files:
        try:
            await delete_note(path, cv)
            deleted += 1
        except CogniVaultError as exc:
            failed.append((path, exc.status))
    return {"deleted": deleted, "failed": failed, "total": len(files)}


# --------------------------------------------------------------------------- #
# Admin: vault reindex + collection rebuild
# --------------------------------------------------------------------------- #
#
# Both are long-running JOBS, not requests: the POST enqueues and answers
# ``202 {jobId, status, message}`` immediately, and progress is read back from a
# separate status endpoint. Nothing here blocks for the duration of the work.
#
# Every helper raises :class:`CogniVaultError` on a non-2xx, carrying the
# upstream ``status`` and a body excerpt — the three statuses that MEAN something
# are left for the caller to interpret rather than being swallowed here:
#
# * **409** — a job of that kind is already running (attach to it, don't retry);
# * **400** ``CONFIRM_MISMATCH`` — the typed collection name did not match;
# * **404** — an older backend without ``/api/admin/collection*`` at all.

_ADMIN_START_TIMEOUT = 30.0  # POST that only enqueues a job
_ADMIN_READ_TIMEOUT = 15.0  # status / info polls


def _admin_error(name: str, resp: httpx.Response) -> CogniVaultError:
    return CogniVaultError(
        f"{name} failed ({resp.status_code})",
        resp.status_code,
        _excerpt(resp.text),
    )


async def reindex(
    scope: str = "full", cv: dict[str, Any] | None = None
) -> dict[str, Any]:
    """POST ``/api/admin/reindex`` — re-chunk and re-embed the caller's vault.

    Non-destructive to the vault files. Returns the ``202`` body
    ``{jobId, status, message}``; raises with ``status == 409`` when a reindex is
    already running.
    """
    base, token = _resolve_cv(cv)
    url = f"{base}/api/admin/reindex"
    body = json.dumps({"scope": scope}, ensure_ascii=False).encode("utf-8")
    headers = _auth_headers(token, {"Content-Type": "application/json; charset=utf-8"})
    async with httpx.AsyncClient(timeout=_ADMIN_START_TIMEOUT) as client:
        resp = await client.post(url, content=body, headers=headers)
    if not (200 <= resp.status_code < 300):
        raise _admin_error("reindex", resp)
    return resp.json()


async def reindex_status(
    job_id: str, cv: dict[str, Any] | None = None
) -> dict[str, Any]:
    """GET ``/api/admin/reindex/status?jobId=…``.

    Returns ``{jobId, scope, status, filesProcessed, totalFiles, errors,
    errorCount, startedAt, finishedAt}``. A **404** (unknown/expired job) is
    raised like any other non-2xx so the caller can decide it means "no job".
    """
    base, token = _resolve_cv(cv)
    url = f"{base}/api/admin/reindex/status"
    headers = _auth_headers(token)
    async with httpx.AsyncClient(timeout=_ADMIN_READ_TIMEOUT) as client:
        resp = await client.get(url, params={"jobId": job_id}, headers=headers)
    if resp.status_code != 200:
        raise _admin_error("reindex status", resp)
    return resp.json()


async def collection_info(cv: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET ``/api/admin/collection`` — physical collection + scheme version.

    Returns ``{collection, alias, schemeVersion, expectedSchemeVersion,
    pointsCount}``. Raises with ``status == 404`` on a backend that predates the
    collection endpoints; the caller degrades instead of failing.
    """
    base, token = _resolve_cv(cv)
    url = f"{base}/api/admin/collection"
    headers = _auth_headers(token)
    async with httpx.AsyncClient(timeout=_ADMIN_READ_TIMEOUT) as client:
        resp = await client.get(url, headers=headers)
    if resp.status_code != 200:
        raise _admin_error("collection info", resp)
    return resp.json()


async def rebuild_collection(
    confirm: str, cv: dict[str, Any] | None = None
) -> dict[str, Any]:
    """POST ``/api/admin/collection/rebuild`` — drop and rebuild the collection.

    Destructive and cluster-wide. ``confirm`` must be the physical collection
    name; the backend is the authority on the match and answers **400**
    ``CONFIRM_MISMATCH`` when it differs. **409** means a rebuild is already
    running. Returns the ``202`` body ``{jobId, status, message}``.
    """
    base, token = _resolve_cv(cv)
    url = f"{base}/api/admin/collection/rebuild"
    body = json.dumps({"confirm": confirm}, ensure_ascii=False).encode("utf-8")
    headers = _auth_headers(token, {"Content-Type": "application/json; charset=utf-8"})
    async with httpx.AsyncClient(timeout=_ADMIN_START_TIMEOUT) as client:
        resp = await client.post(url, content=body, headers=headers)
    if not (200 <= resp.status_code < 300):
        raise _admin_error("collection rebuild", resp)
    return resp.json()


async def rebuild_status(
    job_id: str, cv: dict[str, Any] | None = None
) -> dict[str, Any]:
    """GET ``/api/admin/collection/rebuild/status?jobId=…``.

    Returns ``{jobId, status, phase, collection, schemeVersion, usersTotal,
    usersDone, filesProcessed, errors, errorCount, startedAt, finishedAt}``.
    """
    base, token = _resolve_cv(cv)
    url = f"{base}/api/admin/collection/rebuild/status"
    headers = _auth_headers(token)
    async with httpx.AsyncClient(timeout=_ADMIN_READ_TIMEOUT) as client:
        resp = await client.get(url, params={"jobId": job_id}, headers=headers)
    if resp.status_code != 200:
        raise _admin_error("rebuild status", resp)
    return resp.json()
