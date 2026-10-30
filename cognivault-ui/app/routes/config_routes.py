"""Config and status endpoints."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

import logging

from .. import cognivault, history, llm, rag, rag_pipeline, settings
from ..config import (
    DEFAULT_CONFIG,
    PATHS,
    AppPaths,
    ConfigError,
    deep_merge,
    load_config,
    save_config,
)
from ..deps import (
    ROLE_ADMIN,
    ApiError,
    get_role,
    get_token,
    require_admin,
    resolve_paths,
)
from ..llm_errors import GigaChatError, KitaiCatalogForbidden

log = logging.getLogger("cognivault-ui.config")

router = APIRouter(prefix="/api")

# Warning shown (never enforced) when the system prompt drops the citation rule:
# the `[Источник N]` format is what the UI turns into clickable source links and
# what the hallucination check keys off. "Answer without citations" is a
# legitimate choice, so this informs rather than blocks.
_CITATION_MARKER = "Источник"
_CITATION_WARNING = (
    f"В системном промпте нет упоминания «{_CITATION_MARKER}» — ссылки на "
    "источники и проверка галлюцинаций работать не будут."
)


def _module_text(module: Any, *names: str) -> str:
    """First non-empty module attribute among ``names``, else ``""``.

    The prompt constants live in ``app.rag`` / ``app.rag_pipeline`` and are being
    parameterised in parallel; looking them up by name at call time (public name
    first, private fallback) keeps this endpoint working across the rename.
    """
    for name in names:
        value = getattr(module, name, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _default_prompts() -> dict[str, str]:
    """Built-in prompt texts — what ``None`` in the config resolves to."""
    return {
        "system": _module_text(rag, "SYSTEM_PROMPT", "_SYSTEM_PROMPT"),
        "context_reminder": _module_text(
            rag, "CONTEXT_REMINDER", "_CONTEXT_REMINDER"
        ),
    }


def _readonly_prompts() -> dict[str, str]:
    """Pipeline prompts the user cannot edit.

    Two reasons a prompt lands here rather than in the editable section, and both
    are "an edit would break a contract the code depends on":

    * ``condense`` / ``grader`` — their replies are parsed as JSON. Only the
      constant TAILS are exposed; the full prompt is assembled per request from
      the question, the history and the candidate fragments.
    * ``meta`` / ``meta_self`` — the system turns of the branch that answers a
      question about the base itself and about the assistant itself. Their whole
      job is to keep an ungrounded answer from being generated, and an editable
      key would freeze at whatever a user saved on the day they saved it.

    Read-only is not secret: a user who cannot change a prompt must still be able
    to SEE what governs their answers.
    """
    return {
        "condense": _module_text(
            rag_pipeline, "CONDENSE_TASKS", "_CONDENSE_TASKS"
        ),
        "grader": _module_text(rag_pipeline, "GRADE_SCALE", "_GRADE_SCALE"),
        "meta": _module_text(rag, "META_SYSTEM_PROMPT", "_META_SYSTEM_PROMPT"),
        "meta_self": _module_text(
            rag, "META_SELF_SYSTEM_PROMPT", "_META_SELF_SYSTEM_PROMPT"
        ),
    }


def _effective_prompts(cfg: dict[str, Any]) -> dict[str, str]:
    """Prompt texts actually used: the user's override, else the built-in one."""
    defaults = _default_prompts()
    stored = cfg.get("prompts") if isinstance(cfg.get("prompts"), dict) else {}
    out: dict[str, str] = {}
    for key, fallback in defaults.items():
        value = stored.get(key)
        out[key] = value if isinstance(value, str) and value.strip() else fallback
    return out


def _baseline_config() -> dict[str, Any]:
    """What a "reset to default" lands on.

    Server mode: the administrator's env-driven config (the user's overrides are
    what gets discarded). Local mode: the shipped ``DEFAULT_CONFIG``, since the
    file on disk IS the user's own state.
    """
    return settings.server_config() if settings.is_server() else DEFAULT_CONFIG


def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """The user-editable leaves of ``cfg[name]``, in allowlist order."""
    src = cfg.get(name) if isinstance(cfg.get(name), dict) else {}
    return {key: src.get(key) for key in settings.editable_leaves(name)}


def _warnings(prompts: dict[str, str]) -> list[str]:
    """Non-blocking advisories about the effective config."""
    if _CITATION_MARKER not in prompts.get("system", ""):
        return [_CITATION_WARNING]
    return []


def _config_extras(cfg: dict[str, Any]) -> dict[str, Any]:
    """Contract fields shared by both modes: prompts, defaults, locked, warnings."""
    prompts = _effective_prompts(cfg)
    baseline = _baseline_config()
    return {
        "prompts": prompts,
        "defaults": {
            "prompts": _default_prompts(),
            "gigachat": {
                "model": baseline.get("gigachat", {}).get("model"),
                "temperature": baseline.get("gigachat", {}).get("temperature"),
                "max_tokens": baseline.get("gigachat", {}).get("max_tokens"),
                "model_context_tokens": baseline.get("gigachat", {}).get(
                    "model_context_tokens"
                ),
            },
            "rag": _section(baseline, "rag"),
        },
        "readonly": {"prompts": _readonly_prompts()},
        "locked": list(settings.ADMIN_LOCKED_KEYS),
        "warnings": _warnings(prompts),
    }


def _public_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Secret-free view of the effective config for server mode.

    No token, no cert paths, no passphrase, no base_urls — model params, the
    user-editable RAG knobs, the effective prompt texts, plus the metadata the
    UI needs to render the form (``defaults`` for reset, ``readonly`` prompts,
    ``locked`` admin paths, non-blocking ``warnings``).

    Every path in :data:`app.settings.USER_EDITABLE_KEYS` must be readable back
    here — a key the user can PUT but never GET is a setting that silently
    evaporates on reload. ``model_context_tokens`` is the one extra: read-only
    for the user (it is in ``locked``) but needed to bound ``max_tokens``.
    """
    gc = cfg.get("gigachat", {})
    return {
        "mode": "server",
        "gigachat": {
            "model": gc.get("model"),
            # Which transport actually answers, and its model. Without both, the
            # form showed `model` — inert while the provider is `kitai` — and
            # `kitai_model` was PUT-able but not readable, i.e. exactly the
            # "evaporates on reload" case this docstring warns about.
            # `provider` is admin-locked, so the UI renders it read-only.
            "provider": gc.get("provider"),
            "kitai_model": gc.get("kitai_model"),
            "temperature": gc.get("temperature"),
            "max_tokens": gc.get("max_tokens"),
            "model_context_tokens": gc.get("model_context_tokens"),
        },
        "rag": _section(cfg, "rag"),
        "ui": _section(cfg, "ui"),
        **_config_extras(cfg),
    }


def _local_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Full local config (the local user IS the admin) plus the shared extras."""
    return {**cfg, "mode": "local", **_config_extras(cfg)}


def _error(status: int, code: str, message: str, detail: str | None = None) -> JSONResponse:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if detail is not None:
        body["error"]["detail"] = detail
    return JSONResponse(status_code=status, content=body)


def _optional_paths(request: Request) -> AppPaths | None:
    """Per-user paths when the caller presents a token, else ``None``.

    ``GET /api/config`` stays public (the SPA reads it before the user has typed
    a token), so a missing bearer degrades to the admin-only view instead of 401.
    """
    try:
        return resolve_paths(request)
    except ApiError:
        return None


@router.get("/config")
async def get_config(request: Request) -> dict[str, Any]:
    """Return the config for the UI.

    Public in both modes (no auth). SERVER mode returns a SAFE subset (no
    secrets) with the caller's own overrides applied when a bearer token is
    present; LOCAL mode returns the full effective config plus ``mode: local``.
    """
    if settings.is_server():
        return _public_config(settings.effective_config_for(_optional_paths(request)))
    return _local_config(load_config())


@router.put("/config")
async def put_config(
    request: Request, partial: dict[str, Any], _token: str = Depends(get_token)
) -> Any:
    """Persist a partial config for the CALLER and return the new effective view.

    Server mode writes only the allowlisted keys
    (:data:`app.settings.USER_EDITABLE_KEYS`) into the caller's own
    ``config.json``; admin-owned keys are dropped and reported back in
    ``ignored``. Local mode keeps its historical behaviour — the single user is
    the administrator, so the whole body is written to the global file — but the
    editable values go through the same validation.
    """
    if settings.is_server() and get_role(request) != ROLE_ADMIN:
        # Theme is a harmless personal preference exposed outside the admin-only
        # settings drawer. Model, retrieval and prompt changes remain admin-only.
        if not (
            set(partial) == {"ui"}
            and isinstance(partial.get("ui"), dict)
            and set(partial["ui"]) == {"theme"}
        ):
            raise ApiError(
                403,
                "FORBIDDEN",
                "Недостаточно прав: изменять настройки может только администратор",
            )

    paths = resolve_paths(request)
    filtered, ignored = settings.filter_user_overrides(partial)

    try:
        normalized = settings.validate_user_overrides(
            filtered, settings.effective_config_for(paths)
        )
    except settings.ConfigValueError as exc:
        return _error(
            400,
            "CONFIG_INVALID",
            f"недопустимое значение «{exc.key}»: {exc.expected}",
            detail=repr(exc.value),
        )

    to_save = normalized if settings.is_server() else deep_merge(partial, normalized)
    try:
        saved = save_config(to_save, paths)
    except ConfigError as exc:
        return _error(400, "CONFIG_INVALID", str(exc))

    if settings.is_server():
        body = _public_config(settings.effective_config_for(paths))
        body["ignored"] = ignored
        return body
    body = _local_config(saved)
    body["ignored"] = []
    return body


@router.get("/status")
async def get_status(
    request: Request, _token: str = Depends(get_token)
) -> dict[str, Any]:
    """Aggregate cognivault/gigachat/history status for the UI.

    SERVER mode drops the ``env`` block and the config-file path, checks the
    configured ``/certs`` files, probes the server CogniVault base, and buckets
    ``history_count`` by the caller's token.
    """
    cfg = settings.effective_config()
    gc = cfg.get("gigachat", {})
    cert_path = os.path.expanduser(str(gc.get("cert_path", "")))
    key_path = os.path.expanduser(str(gc.get("key_path", "")))
    gigachat_status = {
        "cert_exists": bool(cert_path) and os.path.isfile(cert_path),
        "key_exists": bool(key_path) and os.path.isfile(key_path),
    }

    if settings.is_server():
        cv = {"base_url": str(cfg.get("cognivault", {}).get("base_url", "")), "token": ""}
        ok, latency_ms, error = await cognivault.health(cv=cv)
        return {
            "mode": "server",
            "cognivault": {"ok": ok, "latency_ms": latency_ms, "error": error},
            "gigachat": gigachat_status,
            "history_count": history.count_chats(resolve_paths(request)),
        }

    ok, latency_ms, error = await cognivault.health()
    gigachat_status["base_url"] = str(gc.get("base_url", ""))
    return {
        "config": {
            "exists": PATHS.config_file.is_file(),
            "path": str(PATHS.config_file),
        },
        "cognivault": {"ok": ok, "latency_ms": latency_ms, "error": error},
        "gigachat": gigachat_status,
        "env": {
            "dir_exists": PATHS.root.is_dir(),
            "venv_exists": PATHS.venv_dir.is_dir(),
        },
        "history_count": history.count_chats(),
    }


@router.get("/config/models")
async def get_models(
    request: Request, _token: str = Depends(require_admin)
) -> dict[str, Any]:
    """Models the active provider offers, for the settings dropdown.

    Deliberately NOT part of ``GET /api/config``: that call renders the whole
    form and must stay fast and offline-safe, while this one leaves the pod and
    can time out. A settings page that cannot open because the model platform is
    slow would be a bad trade.

    Both providers publish a catalogue, so ``models`` is a list on success and
    ``null`` only when the call failed — with ``error`` saying why. Filled rather
    than raised: an unreachable platform must degrade the field to free text, not
    break the settings page.
    """
    paths = _optional_paths(request)
    cfg = (
        settings.effective_config_for(paths)
        if paths is not None and hasattr(settings, "effective_config_for")
        else settings.effective_config()
    )
    gc = cfg.get("gigachat", {})
    try:
        models = await llm.list_models(llm.config_for(gc))
    except KitaiCatalogForbidden as exc:
        # Не warning: это штатное состояние прав, а не поломка, и форма его
        # корректно переживает. На уровне warning оно повторялось на каждой
        # загрузке страницы настроек и выглядело как отказ подключения.
        log.info("каталог моделей закрыт для сертификата: %s", exc)
        return {
            "provider": llm.provider_of(gc),
            "models": None,
            "error": "У сертификата нет доступа к списку моделей — впишите имя вручную",
        }
    except GigaChatError as exc:
        log.warning("список моделей недоступен [%s]: %s", exc.code, exc)
        return {"provider": llm.provider_of(gc), "models": None, "error": exc.message}
    except Exception as exc:  # noqa: BLE001 — the form must still render
        log.warning("список моделей недоступен: %s", exc)
        return {
            "provider": llm.provider_of(gc),
            "models": None,
            "error": "Не удалось получить список моделей",
        }
    return {"provider": llm.provider_of(gc), "models": models, "error": None}
