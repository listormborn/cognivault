"""Deployment-mode settings: ``local`` (single-user, config-file) vs ``server``
(centralized, multi-tenant, env-driven).

``local`` is the historical behaviour and the default. ``server`` mode builds the
same config dict shape as :func:`app.config.load_config` — but entirely from
environment variables — so every downstream reader can stay mode-agnostic by
going through :func:`effective_config`.

This module is import-time validated: an invalid ``COGNIVAULT_UI_MODE`` raises a
``RuntimeError`` so a misconfigured deployment fails fast at startup.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any

from .llm import PROVIDERS
from .config import (
    DEFAULT_CONFIG,
    AppPaths,
    _read_raw_config,
    deep_merge,
    load_config,
)

log = logging.getLogger("cognivault-ui.settings")

_VALID_MODES = ("local", "server")


def _resolve_mode() -> str:
    raw = os.environ.get("COGNIVAULT_UI_MODE", "local").strip().lower()
    if raw not in _VALID_MODES:
        raise RuntimeError(
            f"COGNIVAULT_UI_MODE must be one of {_VALID_MODES!r}, got {raw!r}"
        )
    return raw


# Validated once at import — a bad value crashes startup on purpose.
MODE: str = _resolve_mode()


def is_server() -> bool:
    return MODE == "server"


# --------------------------------------------------------------------------- #
# Env parsing helpers (tolerant: fall back to the default on a bad value)
# --------------------------------------------------------------------------- #


def _gc_default(key: str) -> str:
    """Shipped default for a ``gigachat`` key, as a string.

    Reading it from `DEFAULT_CONFIG` instead of repeating the literal keeps the
    env layer and the config file from drifting — the exact failure mode
    CLAUDE.md warns about for this section.
    """
    return str(DEFAULT_CONFIG["gigachat"][key])


def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val is not None and val != "" else default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _env_json_object(name: str, default: dict[str, Any]) -> dict[str, Any]:
    """A JSON object from the environment; anything else → ``default`` + WARNING.

    Never raises: a typo in an operator-supplied JSON must not take the chat
    down at startup — the log says what was ignored and the request goes out
    without the extra fields, which is the behaviour before the key existed.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return dict(default)
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        log.warning("%s: не JSON (%s) — игнорирую", name, exc)
        return dict(default)
    if not isinstance(parsed, dict):
        log.warning("%s: ожидался JSON-объект, получен %s — игнорирую", name, type(parsed).__name__)
        return dict(default)
    return parsed


def _env_opt_float(name: str, default: float | None) -> float | None:
    """Optional float: unset → ``default`` (typically ``None``)."""
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Data root / bind address (server mode)
# --------------------------------------------------------------------------- #


def data_root() -> Path:
    """Root dir for per-user state in server mode (``UI_DATA_DIR``, default ``/data``)."""
    return Path(_env_str("UI_DATA_DIR", "/data"))


def bind_host() -> str:
    return _env_str("UI_HOST", "0.0.0.0")


def bind_port() -> int:
    return _env_int("UI_PORT", 8787)


# --------------------------------------------------------------------------- #
# Confluence source (server-mode admin settings)
# --------------------------------------------------------------------------- #
#
# In SERVER mode these admin-provided values override the per-user Confluence
# config's connection settings (base_url / ca_path / verify_ssl); the rest of
# the per-user config (auth_mode, login, root, sync options) still comes from
# the user's file. In LOCAL mode the per-user config supplies everything and
# these getters are unused.


def confluence_enabled() -> bool:
    return _env_bool("CONFLUENCE_ENABLED", True)


def confluence_base_url() -> str:
    return _env_str("CONFLUENCE_BASE_URL", "https://confluence.sberbank.ru")


def confluence_ca_path() -> str:
    return _env_str("CONFLUENCE_CA_PATH", "")


def confluence_verify_ssl() -> bool:
    return _env_bool("CONFLUENCE_VERIFY_SSL", True)


def confluence_max_concurrency() -> int:
    return _env_int("CONFLUENCE_MAX_CONCURRENCY", 3)


def confluence_min_auto_interval_min() -> int:
    return _env_int("CONFLUENCE_MIN_AUTO_INTERVAL_MIN", 30)


# --------------------------------------------------------------------------- #
# Server-mode config (same shape as load_config, from ENV only)
# --------------------------------------------------------------------------- #


def server_config() -> dict[str, Any]:
    """Build the effective config for server mode purely from the environment.

    The returned dict mirrors :func:`app.config.load_config`'s shape so all
    downstream readers work unchanged. The CogniVault ``token`` is intentionally
    empty here — in server mode the token is per-request (Bearer header), never
    a shared credential.
    """
    rag = copy.deepcopy(DEFAULT_CONFIG["rag"])
    rag.update(
        {
            "mode": _env_str("RAG_MODE", str(rag["mode"])),
            "source": _env_str("RAG_SOURCE", str(rag["source"])),
            "limit": _env_int("RAG_LIMIT", int(rag["limit"])),
            "min_score": _env_opt_float("RAG_MIN_SCORE", rag["min_score"]),
            "max_context_chars": _env_int(
                "RAG_MAX_CONTEXT_CHARS", int(rag["max_context_chars"])
            ),
            "file_full_chars": _env_int(
                "RAG_FILE_FULL_CHARS", int(rag["file_full_chars"])
            ),
            "section_max_chars": _env_int(
                "RAG_SECTION_MAX_CHARS", int(rag["section_max_chars"])
            ),
            "max_expanded_files": _env_int(
                "RAG_MAX_EXPANDED_FILES", int(rag["max_expanded_files"])
            ),
            "condense_enabled": _env_bool(
                "RAG_CONDENSE_ENABLED", bool(rag["condense_enabled"])
            ),
            "condense_first_turn": _env_bool(
                "RAG_CONDENSE_FIRST_TURN", bool(rag["condense_first_turn"])
            ),
            "corpus_tree_enabled": _env_bool(
                "RAG_CORPUS_TREE_ENABLED", bool(rag["corpus_tree_enabled"])
            ),
            "condense_timeout": _env_float(
                "RAG_CONDENSE_TIMEOUT", float(rag["condense_timeout"])
            ),
            "grader_timeout": _env_float(
                "RAG_GRADER_TIMEOUT", float(rag["grader_timeout"])
            ),
            "grader_max_tokens": _env_int(
                "RAG_GRADER_MAX_TOKENS", int(rag["grader_max_tokens"])
            ),
            "condense_max_tokens": _env_int(
                "RAG_CONDENSE_MAX_TOKENS", int(rag["condense_max_tokens"])
            ),
            "grader_enabled": _env_bool(
                "RAG_GRADER_ENABLED", bool(rag["grader_enabled"])
            ),
            "grader_threshold": _env_int(
                "RAG_GRADER_THRESHOLD", int(rag["grader_threshold"])
            ),
            "grader_keep_top": _env_int(
                "RAG_GRADER_KEEP_TOP", int(rag["grader_keep_top"])
            ),
            "rerank_candidates": _env_int(
                "RAG_RERANK_CANDIDATES", int(rag["rerank_candidates"])
            ),
        }
    )

    gigachat = {
        "base_url": _env_str("GIGACHAT_BASE_URL", DEFAULT_CONFIG["gigachat"]["base_url"]),
        "model": _env_str("GIGACHAT_MODEL", "GigaChat-3-Ultra-preview"),
        "cert_path": os.path.expanduser(
            _env_str("GIGACHAT_CERT_PATH", "/certs/client_crt.crt")
        ),
        "key_path": os.path.expanduser(
            _env_str("GIGACHAT_KEY_PATH", "/certs/client_key.key")
        ),
        "key_passphrase": _env_str("GIGACHAT_KEY_PASSPHRASE", ""),
        "verify_ssl": _env_bool("GIGACHAT_VERIFY_SSL", False),
        "temperature": _env_float("GIGACHAT_TEMPERATURE", 0.2),
        "max_tokens": _env_int("GIGACHAT_MAX_TOKENS", 4096),
        "model_context_tokens": _env_int("GIGACHAT_MODEL_CONTEXT_TOKENS", 32768),
        # Транспорт чата. Эмбеддинги живут в TS-бэкенде и этим ключом не
        # затрагиваются: сменить эмбеддер — это другое векторное пространство и
        # полная переиндексация.
        "provider": _env_str("CHAT_PROVIDER", _gc_default("provider")),
        "kitai_host": _env_str("KITAI_HOST", _gc_default("kitai_host")),
        "kitai_model": _env_str("KITAI_MODEL", _gc_default("kitai_model")),
        "kitai_system_name": _env_str(
            "KITAI_SYSTEM_NAME", _gc_default("kitai_system_name")
        ),
        "kitai_module_name": _env_str(
            "KITAI_MODULE_NAME", _gc_default("kitai_module_name")
        ),
        "kitai_profanity_check": _env_bool("KITAI_PROFANITY_CHECK", False),
        # Строка JSON с объектом; кривое значение — WARNING и пустой объект.
        "kitai_extra_body": _env_json_object(
            "KITAI_EXTRA_BODY", DEFAULT_CONFIG["gigachat"]["kitai_extra_body"]
        ),
        # Пусто = взять сертификат GigaChat (см. KitaiConfig.from_dict).
        "kitai_cert_path": os.path.expanduser(_env_str("KITAI_CERT_PATH", "")),
        "kitai_key_path": os.path.expanduser(_env_str("KITAI_KEY_PATH", "")),
        "kitai_key_passphrase": _env_str("KITAI_KEY_PASSPHRASE", ""),
        "kitai_poll_timeout": _env_float("KITAI_POLL_TIMEOUT", 240.0),
        "kitai_poll_initial_delay": _env_float("KITAI_POLL_INITIAL_DELAY", 2.0),
        "kitai_poll_delay": _env_float("KITAI_POLL_DELAY", 2.0),
    }

    return {
        "version": 1,
        "cognivault": {
            "base_url": _env_str("COGNIVAULT_BASE_URL", "http://cognivault:3000"),
            "token": "",
        },
        "gigachat": gigachat,
        "rag": rag,
        # Admin baseline for the prompts: ``None`` = "use the built-in text".
        # Prompts are user-editable only (no env knob), but the section must
        # exist here so the shape matches ``load_config``.
        "prompts": copy.deepcopy(DEFAULT_CONFIG["prompts"]),
        "ui": {"theme": "auto"},
    }


# --------------------------------------------------------------------------- #
# User-editable keys (server mode) — allowlist, filtering, validation
# --------------------------------------------------------------------------- #
#
# In server mode the administrator owns the deployment-wide config (env) and the
# user owns a small set of behaviour knobs, stored per tenant in
# ``<UI_DATA_DIR>/users/<bucket>/config.json``. Anything not listed here stays
# admin-only: base URLs, certificate paths/passphrase, TLS verification, the
# CogniVault token, the ``env`` mirror settings, and
# ``gigachat.model_context_tokens`` (a property of the DEPLOYED model — a wrong
# value silently breaks history trimming).

USER_EDITABLE_KEYS: tuple[str, ...] = (
    "gigachat.temperature",
    "gigachat.max_tokens",
    "gigachat.model",
    # Имя модели у KitAI своё — иначе переключение транспорта затирало бы
    # настройку соседнего. Правится так же, как `model`.
    "gigachat.kitai_model",
    "gigachat.provider",
    "rag.default_on",
    "rag.limit",
    "rag.min_score",
    "rag.max_context_chars",
    "rag.file_full_chars",
    "rag.section_max_chars",
    "rag.max_expanded_files",
    "rag.condense_enabled",
    "rag.condense_first_turn",
    "rag.corpus_tree_enabled",
    "rag.grader_enabled",
    "rag.grader_threshold",
    "rag.condense_timeout",
    "rag.grader_timeout",
    "rag.grader_max_tokens",
    "rag.condense_max_tokens",
    "rag.grader_keep_top",
    "rag.rerank_candidates",
    "prompts.system",
    "prompts.context_reminder",
    "ui.theme",
)

# Admin-owned paths, surfaced to the UI as ``locked`` so it can render them
# read-only instead of guessing. Not an enforcement list (the allowlist above is
# the enforcement) — a documentation contract for the client.
ADMIN_LOCKED_KEYS: tuple[str, ...] = (
    "cognivault.base_url",
    "cognivault.token",
    "gigachat.base_url",
    "gigachat.cert_path",
    "gigachat.key_path",
    "gigachat.key_passphrase",
    "gigachat.verify_ssl",
    "gigachat.model_context_tokens",
    # Адресация KitAI остаётся деплойной заботой, и это не осторожность ради
    # осторожности: `kitai_host` — адрес, КУДА уедет клиентский сертификат, а
    # `kitai_system_name` — то, чьим именем мы представляемся платформе.
    # Пользователь, который может править их, может увести сертификат на свой
    # хост и назваться чужой системой. Сам ВЫБОР транспорта и имена моделей —
    # в USER_EDITABLE_KEYS: они ничего никуда не отправляют.
    "gigachat.kitai_host",
    "gigachat.kitai_cert_path",
    "gigachat.kitai_key_path",
    "gigachat.kitai_key_passphrase",
    "gigachat.kitai_system_name",
    "gigachat.kitai_module_name",
    "gigachat.kitai_profanity_check",
    # Поля запроса к платформе — это то, что уезжает на её сторону от нашего
    # имени; пользователю их не отдаём.
    "gigachat.kitai_extra_body",
    "gigachat.kitai_poll_timeout",
    "gigachat.kitai_poll_initial_delay",
    "gigachat.kitai_poll_delay",
    "rag.mode",
    "rag.source",
    "rag.token_budget",
    "env.pip_index_url",
    "env.pip_token",
)

THEMES: tuple[str, ...] = ("auto", "light", "dark")

_MAX_PROMPT_CHARS = 20000
_MAX_MODEL_CHARS = 200


def editable_leaves(section: str) -> tuple[str, ...]:
    """Leaf keys of ``section`` that a user may edit, in allowlist order."""
    return tuple(
        path.split(".", 1)[1]
        for path in USER_EDITABLE_KEYS
        if path.startswith(f"{section}.")
    )


def filter_user_overrides(
    partial: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Split a partial config into its allowlisted part and the ignored paths.

    Returns ``(filtered, ignored_paths)``. ``filtered`` keeps only the paths in
    :data:`USER_EDITABLE_KEYS`; everything else — unknown sections, scalars at
    the top level, admin-owned leaves — is dropped and reported by dotted path
    so the caller can tell the user what was not saved (rather than silently
    swallowing it).
    """
    allowed: dict[str, set[str]] = {}
    for path in USER_EDITABLE_KEYS:
        section, _, leaf = path.partition(".")
        allowed.setdefault(section, set()).add(leaf)

    filtered: dict[str, Any] = {}
    ignored: list[str] = []
    if not isinstance(partial, dict):
        return filtered, ignored

    for section, value in partial.items():
        if not isinstance(value, dict):
            # A scalar at the top level (``version``, a stray key) — nothing to
            # merge into a section, so it is reported by its bare name.
            ignored.append(str(section))
            continue
        leaves = allowed.get(str(section), frozenset())
        for leaf, leaf_value in value.items():
            if str(leaf) in leaves:
                filtered.setdefault(str(section), {})[str(leaf)] = leaf_value
            else:
                ignored.append(f"{section}.{leaf}")
    return filtered, ignored


class ConfigValueError(ValueError):
    """A user-supplied config VALUE is out of range / of the wrong type.

    Distinct from :class:`app.config.ConfigError`, which only guards the SHAPE
    of a partial config and is relied upon by the local-mode write path.
    """

    def __init__(self, key: str, expected: str, value: Any) -> None:
        super().__init__(f"{key}: ожидается {expected}")
        self.key = key
        self.expected = expected
        self.value = value


def _int_in(key: str, value: Any, low: int, high: int) -> int:
    """Strict integer check (``bool`` is NOT an int here) within ``[low, high]``."""
    expected = f"целое число от {low} до {high}"
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValueError(key, expected, value)
    if not low <= value <= high:
        raise ConfigValueError(key, expected, value)
    return value


def _float_in(key: str, value: Any, low: float, high: float) -> float:
    """Numeric check (``bool`` rejected) within ``[low, high]``, returned as float."""
    expected = f"число от {low} до {high}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValueError(key, expected, value)
    if not low <= float(value) <= high:
        raise ConfigValueError(key, expected, value)
    return float(value)


def _strict_bool(key: str, value: Any) -> bool:
    """Reject ``0``/``1``/``"true"`` — only a real JSON boolean passes."""
    if not isinstance(value, bool):
        raise ConfigValueError(key, "true или false", value)
    return value


def _prompt_text(key: str, value: Any) -> str | None:
    """Normalise a prompt override: blank/``None`` → ``None`` (reset to default)."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigValueError(
            key, f"строка до {_MAX_PROMPT_CHARS} символов или null", value
        )
    if len(value) > _MAX_PROMPT_CHARS:
        raise ConfigValueError(
            key, f"строка до {_MAX_PROMPT_CHARS} символов или null", value
        )
    # An empty textarea means "back to the built-in prompt", not "empty prompt".
    return value if value.strip() else None


def validate_user_overrides(
    partial: dict[str, Any], effective: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate + normalise the VALUES of an allowlisted partial config.

    ``effective`` supplies the context needed for relative bounds — currently
    only ``gigachat.model_context_tokens``, the ceiling for ``max_tokens``.
    Returns a normalised copy (blank prompts collapsed to ``None``, numerics
    coerced to their canonical type). Raises :class:`ConfigValueError` on the
    first offending key.
    """
    effective = effective if effective is not None else effective_config()
    ctx_tokens = effective.get("gigachat", {}).get("model_context_tokens", 32768)
    if isinstance(ctx_tokens, bool) or not isinstance(ctx_tokens, int) or ctx_tokens < 1:
        ctx_tokens = int(DEFAULT_CONFIG["gigachat"]["model_context_tokens"])

    out: dict[str, Any] = copy.deepcopy(partial)

    gigachat = out.get("gigachat")
    if isinstance(gigachat, dict):
        if "temperature" in gigachat:
            gigachat["temperature"] = _float_in(
                "gigachat.temperature", gigachat["temperature"], 0.0, 2.0
            )
        if "max_tokens" in gigachat:
            gigachat["max_tokens"] = _int_in(
                "gigachat.max_tokens", gigachat["max_tokens"], 1, ctx_tokens
            )
        if "provider" in gigachat:
            provider = gigachat["provider"]
            if not isinstance(provider, str) or provider.strip().lower() not in PROVIDERS:
                raise ConfigValueError(
                    "gigachat.provider", f"одно из {list(PROVIDERS)}", provider
                )
            gigachat["provider"] = provider.strip().lower()
        for key in ("model", "kitai_model"):
            if key not in gigachat:
                continue
            model = gigachat[key]
            if (
                not isinstance(model, str)
                or not model.strip()
                or len(model) > _MAX_MODEL_CHARS
            ):
                raise ConfigValueError(
                    f"gigachat.{key}",
                    f"непустая строка до {_MAX_MODEL_CHARS} символов",
                    model,
                )
            gigachat[key] = model.strip()

    rag = out.get("rag")
    if isinstance(rag, dict):
        for key in (
            "default_on",
            "condense_enabled",
            "condense_first_turn",
            "corpus_tree_enabled",
            "grader_enabled",
        ):
            if key in rag:
                rag[key] = _strict_bool(f"rag.{key}", rag[key])
        for key, low, high in (
            ("limit", 1, 100),
            ("rerank_candidates", 1, 100),
            ("grader_threshold", 1, 5),
            ("grader_keep_top", 0, 10),
            ("max_expanded_files", 0, 10),
            ("max_context_chars", 500, 200000),
            ("file_full_chars", 100, 100000),
            ("section_max_chars", 100, 100000),
            ("grader_max_tokens", 64, 65536),
            ("condense_max_tokens", 64, 65536),
        ):
            if key in rag:
                rag[key] = _int_in(f"rag.{key}", rag[key], low, high)
        if "min_score" in rag and rag["min_score"] is not None:
            rag["min_score"] = _float_in("rag.min_score", rag["min_score"], 0.0, 1.0)

    prompts = out.get("prompts")
    if isinstance(prompts, dict):
        for key in ("system", "context_reminder"):
            if key in prompts:
                prompts[key] = _prompt_text(f"prompts.{key}", prompts[key])

    ui = out.get("ui")
    if isinstance(ui, dict) and "theme" in ui:
        if ui["theme"] not in THEMES:
            raise ConfigValueError(
                "ui.theme", f"одно из {', '.join(THEMES)}", ui["theme"]
            )

    return out


def user_overrides(paths: AppPaths) -> dict[str, Any]:
    """Allowlisted per-user overrides, read RAW (no defaults) from ``paths``.

    Reading raw is load-bearing: a defaults-merged read would carry every
    unset key along and clobber the administrator's env values on merge.
    """
    filtered, _ = filter_user_overrides(_read_raw_config(paths))
    return filtered


def effective_config_for(paths: AppPaths | None = None) -> dict[str, Any]:
    """Active config for one caller: admin env + that user's overrides.

    * server mode: :func:`server_config` with the tenant's allowlisted overrides
      deep-merged on top (``paths=None`` → the admin config alone, e.g. for an
      unauthenticated ``GET /api/config``);
    * local mode: the single config file — the local user IS the administrator.
    """
    if not is_server():
        return load_config(paths)
    base = server_config()
    if paths is None:
        return base
    return deep_merge(base, user_overrides(paths))


def effective_config() -> dict[str, Any]:
    """Return the active config: env-driven in server mode, file-driven locally.

    Request-agnostic wrapper around :func:`effective_config_for` — in server
    mode it yields the ADMIN config with no per-user overrides. Callers that
    have a request should prefer ``effective_config_for(resolve_paths(request))``.
    """
    return effective_config_for(None)
