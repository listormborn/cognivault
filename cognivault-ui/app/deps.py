"""Per-request identity, token, and path resolution.

In ``server`` mode the caller's bearer token *is* the tenant identity: history is
bucketed by ``sha256(token)`` and every CogniVault call is made with the caller's
own token. In ``local`` mode these helpers collapse to the historical single-user
behaviour (config-file token, global ``PATHS``) so downstream code has one path.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse

from . import settings
from .config import PATHS, AppPaths

_BEARER_PREFIX = "Bearer "
ROLE_ADMIN = "admin"
ROLE_USER = "user"


class ApiError(Exception):
    """Carries the standard ``{"error": {...}}`` envelope through FastAPI.

    Raised from dependencies/handlers; rendered by :func:`api_error_handler`
    (registered in ``main.create_app``).
    """

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class CVUnavailable(Exception):
    """CogniVault could not be reached / returned an unexpected status.

    Distinct from an *invalid* token (a clean 401 from CogniVault); lets callers
    map connectivity failures to 503 rather than 401.
    """


async def api_error_handler(request: Request, exc: Exception) -> JSONResponse:
    err = exc if isinstance(exc, ApiError) else ApiError(500, "INTERNAL", str(exc))
    return JSONResponse(
        status_code=err.status,
        content={"error": {"code": err.code, "message": err.message}},
    )


def get_token(request: Request) -> str:
    """Return the caller's token.

    * server mode: parse ``Authorization: Bearer <token>``; a missing or
      malformed header raises :class:`ApiError` (401 ``UNAUTHORIZED``).
    * local mode: return the config-file token (no header required) so the same
      downstream code path applies.

    Usable directly *and* as a FastAPI dependency (``Depends(get_token)``): as a
    dependency it enforces the header in server mode and is a no-op locally.
    """
    if not settings.is_server():
        cv = settings.effective_config().get("cognivault", {})
        return str(cv.get("token", "") or "")

    auth = request.headers.get("authorization") or ""
    if not auth.startswith(_BEARER_PREFIX):
        raise ApiError(401, "UNAUTHORIZED", "нет токена доступа")
    token = auth[len(_BEARER_PREFIX):].strip()
    if not token:
        raise ApiError(401, "UNAUTHORIZED", "нет токена доступа")
    return token


def role_for_token(token: str) -> str:
    """Return the caller's role under the optional token-hash RBAC policy."""
    if not settings.is_server() or not settings.rbac_enabled():
        return ROLE_ADMIN
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return (
        ROLE_ADMIN
        if any(
            hmac.compare_digest(digest, expected)
            for expected in settings.admin_token_hashes()
        )
        else ROLE_USER
    )


def get_role(request: Request) -> str:
    """Role for this request; also enforces the normal bearer-token gate."""
    return role_for_token(get_token(request))


def require_admin(request: Request) -> str:
    """FastAPI dependency for administrative routes."""
    token = get_token(request)
    if role_for_token(token) != ROLE_ADMIN:
        raise ApiError(
            403,
            "FORBIDDEN",
            "Недостаточно прав: требуется роль администратора",
        )
    return token


def user_bucket(token: str) -> str:
    """Stable, non-reversible tenant id derived from the token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def resolve_paths(request: Request) -> AppPaths:
    """Per-request :class:`AppPaths`.

    server mode: ``<UI_DATA_DIR>/users/<bucket>`` (its ``history_dir`` lives under
    that root; dirs are created lazily by ``ensure_dirs``). The bucket is ALWAYS
    derived from the token — never from client input. Local mode: the global
    ``PATHS`` singleton.
    """
    if not settings.is_server():
        return PATHS
    token = get_token(request)
    root = settings.data_root() / "users" / user_bucket(token)
    return AppPaths(root=root)


def cv_context(request: Request) -> dict[str, Any] | None:
    """CogniVault call context for this request.

    server mode: ``{"base_url", "token"}`` from the server config + the caller's
    bearer token. Local mode: ``None`` — meaning "read the config file", the
    existing behaviour.
    """
    if not settings.is_server():
        return None
    cfg = settings.server_config()
    base = str(cfg.get("cognivault", {}).get("base_url", "")).rstrip("/")
    return {"base_url": base, "token": get_token(request)}


# --------------------------------------------------------------------------- #
# Token validation against CogniVault (with a small TTL cache)
# --------------------------------------------------------------------------- #

_VALIDATE_TTL_SECONDS = 60.0
_VALIDATE_CACHE_CAP = 1000
# key -> (expires_at, is_valid)
_validate_cache: dict[str, tuple[float, bool]] = {}


def _cache_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> bool | None:
    entry = _validate_cache.get(key)
    if entry is None:
        return None
    expires_at, valid = entry
    if time.monotonic() >= expires_at:
        _validate_cache.pop(key, None)
        return None
    return valid


def _cache_put(key: str, valid: bool) -> None:
    if len(_validate_cache) >= _VALIDATE_CACHE_CAP:
        # Cheap eviction: drop everything rather than track LRU.
        _validate_cache.clear()
    _validate_cache[key] = (time.monotonic() + _VALIDATE_TTL_SECONDS, valid)


async def validate_token(token: str) -> bool:
    """Validate ``token`` against CogniVault (server mode).

    Calls ``GET {base}/api/vault/files?recursive=false`` with the bearer token:
    ``200`` → valid, ``401`` → invalid. Any other status / connection error
    raises :class:`CVUnavailable`. Results are cached for 60s (keyed by token
    hash) to avoid hammering CogniVault.
    """
    key = _cache_key(token)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    cfg = settings.server_config()
    base = str(cfg.get("cognivault", {}).get("base_url", "")).rstrip("/")
    url = f"{base}/api/vault/files"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(
                url, params={"recursive": "false"}, headers=headers
            )
    except httpx.HTTPError as exc:
        raise CVUnavailable(str(exc) or exc.__class__.__name__) from exc

    if resp.status_code == 200:
        _cache_put(key, True)
        return True
    if resp.status_code == 401:
        _cache_put(key, False)
        return False
    raise CVUnavailable(f"HTTP {resp.status_code}")
