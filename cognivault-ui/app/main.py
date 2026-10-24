"""FastAPI application factory for CogniVault UI.

Serves the SPA from ``static/`` (same-origin) and exposes the ``/api`` surface.
The browser talks only to this localhost server; config/certs/token never leave
the machine.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope

from . import deps, settings
from .config import PATHS
from .deps import ApiError, api_error_handler
from .routes import (
    admin_routes,
    chat_routes,
    config_routes,
    confluence_routes,
    env_routes,
    eval_routes,
    feedback_routes,
    history_routes,
    upload_routes,
    welcome_routes,
)

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# SPA shell assets that must never be served stale after a redeploy. The browser
# / proxy must refetch these (not reuse a cached copy or revalidate to a 304) so a
# fresh app.js/index.html always loads. Fingerprinted assets and fonts are absent
# here and keep default (cacheable) behaviour.
_SHELL_ASSETS = frozenset({"index.html", "app.js", "style.css", "favicon.svg"})


class NoCacheStaticFiles(StaticFiles):
    """``StaticFiles`` that serves the SPA shell with a no-store policy.

    Only the shell assets in ``_SHELL_ASSETS`` (index.html — also served at
    ``/`` via ``html=True`` — plus app.js, style.css, favicon.svg) are affected.
    For those we (a) set ``Cache-Control: no-store, must-revalidate`` and strip
    any long-cache validators (ETag/Last-Modified/Expires) and (b) bypass the
    ``is_not_modified`` 304 shortcut so a client that cached an older build can
    never revalidate its way back to stale content — it always gets a fresh 200.

    Scoped to the static mount, so ``/api/*``, SSE, and ``/healthz`` responses
    are never touched. Any other static file keeps default caching behaviour.
    """

    def file_response(
        self,
        full_path: Any,
        stat_result: Any,
        scope: Scope,
        status_code: int = 200,
    ) -> Any:
        if os.path.basename(str(full_path)) in _SHELL_ASSETS:
            # Build the response directly (no is_not_modified check) so shell
            # assets are always returned fresh, never as a 304.
            response = FileResponse(
                full_path, status_code=status_code, stat_result=stat_result
            )
            response.headers["cache-control"] = "no-store, must-revalidate"
            for stale in ("etag", "last-modified", "expires"):
                if stale in response.headers:
                    del response.headers[stale]
            return response
        return super().file_response(full_path, stat_result, scope, status_code)


def _suppress_insecure_warnings() -> None:
    """Silence urllib3/httpx ``verify=False`` noise once at startup.

    Disabling TLS verification is a supported escape hatch for the Sber IFT
    contour; we don't want the warning printed on every request.
    """
    try:
        from urllib3.exceptions import InsecureRequestWarning

        warnings.simplefilter("ignore", InsecureRequestWarning)
    except Exception:  # noqa: BLE001 — urllib3 may be absent; best-effort only
        pass
    warnings.filterwarnings("ignore", message=".*[Uu]nverified HTTPS.*")


def _auto_sync_scheduler_enabled() -> bool:
    """Whether to start the Confluence auto-sync background task.

    Gated behind ``CONFLUENCE_AUTO_SYNC_SCHEDULER``. Default: **on in server
    mode, off in local mode**. Server deployments are the multi-tenant, long-lived
    target where unattended periodic syncs make sense; a local single-user run (and
    every ``TestClient`` startup) leaves it off so importing/serving the app never
    spawns a background loop unless explicitly opted in. Set the env var to
    ``1/true/yes/on`` (or ``0/false/no/off``) to override the per-mode default.
    """
    raw = os.environ.get("CONFLUENCE_AUTO_SYNC_SCHEDULER")
    if raw is None or raw == "":
        return settings.is_server()
    return raw.strip().lower() in ("1", "true", "yes", "on")


_APP_LOGGER = "cognivault-ui"


def _configure_logging() -> None:
    """Make the application's own log lines visible in the pod log.

    Nobody calls ``logging.basicConfig`` here and uvicorn configures only its own
    loggers, so every ``cognivault-ui.*`` record used to fall through to Python's
    last-resort handler: WARNING and above reached stderr, ``log.info`` went
    nowhere at all. That hid exactly the lines an operator needs when the RAG
    pipeline degrades quietly — the grader's fallback pass, the "batch graded but
    fragments omitted" note, the KitAI query id.

    One handler on the ``cognivault-ui`` logger, level from ``UI_LOG_LEVEL``
    (default INFO). Propagation stays ON so a handler attached to the root logger
    for the duration of an eval run (``eval_runner``) still receives the records;
    the last-resort handler is skipped automatically once a handler exists in
    the chain, so nothing is printed twice. Idempotent: ``create_app`` runs once
    per process in production but many times under the test client.
    """
    logger = logging.getLogger(_APP_LOGGER)
    if any(getattr(h, "_cognivault_ui", False) for h in logger.handlers):
        return
    level_name = (os.environ.get("UI_LOG_LEVEL") or "INFO").strip().upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        level = logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler._cognivault_ui = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(level)


def create_app() -> FastAPI:
    _suppress_insecure_warnings()
    _configure_logging()

    app = FastAPI(title="CogniVault UI", version="1.0.0")

    # Render the standard {"error": {...}} envelope for auth/CV failures raised
    # from dependencies or handlers.
    app.add_exception_handler(ApiError, api_error_handler)

    # Bearer-token gate. In LOCAL mode ``get_token`` is a no-op that returns the
    # config-file token (no header required); in SERVER mode it enforces a valid
    # ``Authorization: Bearer`` header (401 otherwise). Attaching it universally
    # keeps local behaviour unchanged while protecting every /api/* in server
    # mode — except public GET /api/config, /api/whoami (self-checks), /healthz,
    # and static assets.
    auth = [Depends(deps.get_token)]

    # Routers first so /api takes precedence over the SPA catch-all mount.
    # config_routes gates PUT/status per-route so GET /api/config stays public.
    app.include_router(config_routes.router)
    app.include_router(chat_routes.router, dependencies=auth)
    app.include_router(history_routes.router, dependencies=auth)
    app.include_router(feedback_routes.router, dependencies=auth)
    app.include_router(upload_routes.router, dependencies=auth)
    # Empty-chat greeting: real top-level corpus sections, no LLM call. It is
    # authenticated because the catalogue is tenant-specific.
    app.include_router(welcome_routes.router, dependencies=auth)
    # Index maintenance (reindex / collection rebuild). Both modes: the operator
    # has no shell or vector-DB access, so the drawer is the only surface.
    app.include_router(admin_routes.router, dependencies=auth)
    # Прогон харнесса оценки. Запускать его надо ИЗНУТРИ пода (снаружи закрытого
    # контура нет ни UI, ни mTLS-эндпоинта судьи), а прав на kubectl exec у того,
    # кто читает отчёты, может не быть вовсе — кнопка снимает это ограничение.
    app.include_router(eval_routes.router, dependencies=auth)
    # Confluence source: per-user data, registered in BOTH modes behind the same
    # bearer-token gate. The CONFLUENCE_ENABLED admin flag (default on) lets a
    # server deployment turn the surface off entirely.
    if settings.confluence_enabled():
        app.include_router(confluence_routes.router, dependencies=auth)
    if not settings.is_server():
        # Env provisioning is a local-only concern; in server mode these routes
        # simply don't exist (the 404 envelope handles them).
        app.include_router(env_routes.router)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness/readiness probe for k8s — auth-free, both modes."""
        return {"status": "ok"}

    @app.get("/api/whoami")
    async def whoami(request: Request) -> Any:
        """Resolve the caller's identity.

        LOCAL: always ``{"userId": "local", "ok": true}``. SERVER: validate the
        bearer token against CogniVault — 200 → ``{"userId": <bucket>, "ok":
        true}``, invalid → 401 UNAUTHORIZED, CogniVault down → 503 CV_UNAVAILABLE.
        """
        if not settings.is_server():
            return {"userId": "local", "ok": True}
        token = deps.get_token(request)  # 401 if header missing/malformed
        try:
            valid = await deps.validate_token(token)
        except deps.CVUnavailable:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "code": "CV_UNAVAILABLE",
                        "message": "CogniVault недоступен",
                    }
                },
            )
        if not valid:
            raise ApiError(401, "UNAUTHORIZED", "нет токена доступа")
        return {"userId": deps.user_bucket(token), "ok": True}

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: Any) -> Any:
        """Return the standard error envelope for unknown ``/api`` paths."""
        if request.url.path.startswith("/api"):
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "code": "NOT_FOUND",
                        "message": f"нет такого маршрута: {request.url.path}",
                    }
                },
            )
        # Non-API 404: let the SPA/default handling apply.
        return JSONResponse(status_code=404, content={"detail": "Not Found"})

    # Catch-all for unknown /api/* paths. Registered after every real router but
    # before the static mount so it only fires when nothing else matched. Without
    # it, a non-GET request to an unregistered /api path (e.g. POST /api/env/setup
    # in server mode) would fall through to the static mount and get a bare
    # ``405 {"detail": ...}`` — here it always yields the standard 404 envelope.
    @app.api_route(
        "/api/{_rest:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def api_not_found(request: Request, _rest: str) -> Any:
        raise ApiError(
            404, "NOT_FOUND", f"нет такого маршрута: {request.url.path}"
        )

    # Confluence auto-sync scheduler (phase 5). Started as a background task on
    # startup ONLY when the Confluence surface is enabled AND the scheduler is
    # opted in (see ``_auto_sync_scheduler_enabled`` — on in server, off in local
    # / under test). The task is fully exception-safe: a scheduler crash never
    # affects request serving, and startup never blocks on it.
    @app.on_event("startup")
    async def _start_confluence_scheduler() -> None:
        app.state.confluence_scheduler_task = None
        app.state.confluence_scheduler_stop = None
        if not (settings.confluence_enabled() and _auto_sync_scheduler_enabled()):
            return
        # Imported lazily so the module (and its asyncio task) only loads when the
        # scheduler is actually enabled — keeps plain imports / tests cheap.
        from .confluence import scheduler

        stop = asyncio.Event()
        app.state.confluence_scheduler_stop = stop
        app.state.confluence_scheduler_task = asyncio.create_task(
            scheduler.run_scheduler(stop_event=stop)
        )

    @app.on_event("shutdown")
    async def _stop_confluence_scheduler() -> None:
        stop = getattr(app.state, "confluence_scheduler_stop", None)
        task = getattr(app.state, "confluence_scheduler_task", None)
        if stop is not None:
            stop.set()
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # SPA + static assets. Mounted last so it only catches non-/api paths.
    _STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/", NoCacheStaticFiles(directory=str(_STATIC_DIR), html=True), name="static"
    )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    if settings.is_server():
        # Multi-tenant: bind the configured address; per-user dirs are created
        # lazily per request. Ensure the data root exists.
        settings.data_root().mkdir(parents=True, exist_ok=True)
        uvicorn.run(app, host=settings.bind_host(), port=settings.bind_port())
    else:
        # Ensure the data dir exists for a bare `python -m app.main` run.
        PATHS.ensure_dirs()
        uvicorn.run(app, host="127.0.0.1", port=8787)
