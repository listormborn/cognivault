"""Per-user RAG request/feedback log — plain JSONL under ``<paths.root>``.

One JSON object per line in ``rag_log.jsonl`` (wave 5.1/5.4 of the RAG quality
plan). Two record ``type``s share the file: ``"request"`` (written by the chat
route once a turn finishes, even a truncated/errored one) and ``"feedback"``
(written by ``POST /api/feedback``); a feedback record is matched to its request
by ``(chat_id, message_index)``.

Deliberate choices:

* **Not the ``logging`` module.** The log is a data artefact for eval runs, not
  operator noise — a file write is the honest implementation. (Operator lines do
  reach the pod log: ``app.main._configure_logging`` puts a handler on the
  ``cognivault-ui`` logger at ``UI_LOG_LEVEL``, INFO by default.)
* **No atomic temp-file + ``os.replace``** (unlike ``history.py``/``config.py``).
  Records are appended, never rewritten: a single ``write()`` of one short line
  to a file opened in ``"a"`` mode is effectively atomic on POSIX, and a partial
  line would only ever cost the last record. Rewriting the whole file for every
  turn would be far worse.
* **Best-effort.** Every failure path is swallowed and reported as ``False`` —
  telemetry must never break a chat.
* **No secrets.** Records carry no user id (the file already lives in the
  caller's own directory) and :func:`append` scrubs any key that looks like a
  credential before writing.
* **The rendered context and the answer are written in full** (capped by
  :data:`MAX_TEXT_CHARS`, which MUST stay above the RAG char budget
  ``max_context_chars`` — the cap is what the eval harness feeds the judge, and
  a cap below the budget silently gives the judge fewer blocks than the model
  saw). Without them the
  eval harness has to *guess* what the model saw by re-slicing documents from
  metadata, and that guess is biased in both directions — see
  ``tools/eval/README.md``. The price is record size: a full turn is ~30 KB, so
  the 5 MiB rotation holds roughly 150–300 turns, which comfortably covers a
  golden-set run. Collect the file right after a run.
* **The head block is recorded by SHAPE, not by text** (:func:`head_block_snapshot`).
  ``context_text`` starts at «Источники:» by construction — the structural block
  that opens the same message (``app.rag._head_block``) is outside the slice, so
  before this field nothing in a finished run could say whether the section tree
  had been there at all. Both of its renderings need
  ``GET /api/vault/catalog``, and against a backend that predates the endpoint
  they both silently collapse to ``None``. Its text is not logged: it is
  thousands of characters, identical on every turn of a run, and already
  reconstructible from the vault.

Deployment note: in the production manifest ``/data`` is an ``emptyDir``, so
this log (and the votes in it) does not survive a pod restart. That is fine for
eval runs — collect the file before recycling the pod.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import AppPaths
from .llm import DEFAULT_PROVIDER, PROVIDERS

LOG_NAME = "rag_log.jsonl"

# Rotate at 5 MiB, keep the current file plus a single ``.1`` backup.
MAX_BYTES = 5 * 1024 * 1024
BACKUP_SUFFIX = ".1"

#: Hard cap for the two free-text fields (``answer_text``, ``context_text``).
#: Must stay ABOVE ``rag.max_context_chars`` — the harness judges what this
#: field holds, so a lower cap silently hands the judge a shorter context than
#: the model actually got. Установленные 32 000 писались под бюджет в 24 000 и
#: пережили его подъём до 48 000: в прогоне `baseline` судья недосчитался
#: блоков на 7 парах из 47 (x05, x19, x27, x36, x38, fb21, fb56), и провалы
#: faithfulness там — про отсутствующий у судьи текст, а не про ответ.
#: Truncation is always flagged (``*_truncated_in_log``).
MAX_TEXT_CHARS = 64_000

# Keys whose values are (or may embed) credentials. Never written, at any depth.
_SECRET_KEYS = frozenset(
    {
        "authorization",
        "token",
        "pat",
        "password",
        "secret",
        "passphrase",
        "key_passphrase",
        "api_key",
        "openai_key",
        "cert",
        "cert_path",
        "key_path",
    }
)


def now_iso() -> str:
    """Current UTC timestamp, ISO-8601 — the ``ts`` of every record."""
    return datetime.now(timezone.utc).isoformat()


def log_path(paths: AppPaths) -> Path:
    """Absolute path of the caller's JSONL log."""
    return paths.root / LOG_NAME


def _scrub(value: Any) -> Any:
    """Drop credential-shaped keys anywhere in a record (defence in depth)."""
    if isinstance(value, dict):
        return {
            k: _scrub(v)
            for k, v in value.items()
            if str(k).lower() not in _SECRET_KEYS
        }
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def _rotate(path: Path) -> None:
    """Move the log aside once it exceeds :data:`MAX_BYTES` (keep 2 files)."""
    try:
        if path.stat().st_size <= MAX_BYTES:
            return
    except OSError:
        return
    # ``os.replace`` overwrites the previous backup — exactly the 2-file policy.
    os.replace(path, path.with_name(path.name + BACKUP_SUFFIX))


def append(paths: AppPaths, record: dict[str, Any]) -> bool:
    """Append one JSON line to the caller's log. Returns ``True`` on success.

    Never raises: any I/O or serialisation problem is swallowed and reported as
    ``False`` so a logging failure can never break the chat stream.
    """
    try:
        paths.root.mkdir(parents=True, exist_ok=True)
        path = log_path(paths)
        _rotate(path)
        line = json.dumps(_scrub(record), ensure_ascii=False)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return True
    except Exception:  # noqa: BLE001 — telemetry is strictly best-effort
        return False


def truncate(text: str, limit: int | None = None) -> tuple[str, bool]:
    """``(text capped at limit, was_truncated)`` — for the free-text fields.

    ``limit`` defaults to :data:`MAX_TEXT_CHARS` at call time (not at
    definition time) so the constant stays overridable.
    """
    cap = MAX_TEXT_CHARS if limit is None else limit
    value = text or ""
    if len(value) <= cap:
        return value, False
    return value[:cap], True


# --------------------------------------------------------------------------- #
# Stage timings
# --------------------------------------------------------------------------- #
#
# The chat route wants per-stage latency (condense / search / grader / stream),
# but three of those stages happen deep inside ``rag.build_rag_context``. Rather
# than thread a timer argument through every layer, the collector lives in a
# ``ContextVar``: the route opens one for the duration of a request and any code
# running in that context can contribute. Asyncio tasks inherit a copy of the
# context, so parallel requests never share a collector.
#
# Two ways to contribute: wrap a block in :func:`stage`, or wrap a whole
# coroutine function once with :func:`instrument` (used for the call seams the
# route does not own). Durations of the same stage accumulate — with the grader
# fanning out into parallel batches the sum is the honest "time spent grading",
# while the wall clock is covered by the enclosing ``rag`` stage.

_STAGES: ContextVar[dict[str, float] | None] = ContextVar("rag_log_stages", default=None)

_INSTRUMENTED = "__rag_log_stage__"


@contextmanager
def collect_stages() -> Iterator[dict[str, float]]:
    """Open a stage collector for this context; yields the accumulator dict."""
    acc: dict[str, float] = {}
    token = _STAGES.set(acc)
    try:
        yield acc
    finally:
        _STAGES.reset(token)


def record_stage(name: str, ms: float) -> None:
    """Add ``ms`` to ``name`` in the active collector (no-op without one)."""
    acc = _STAGES.get()
    if acc is None:
        return
    acc[name] = round(acc.get(name, 0.0) + ms, 1)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time a block and accumulate it under ``name``."""
    started = time.perf_counter()
    try:
        yield
    finally:
        record_stage(name, (time.perf_counter() - started) * 1000.0)


def instrument(module: Any, attr: str, name: str) -> bool:
    """Wrap the coroutine ``module.attr`` so its calls land under stage ``name``.

    Idempotent and defensive: a missing attribute or an already-wrapped one is
    a no-op returning ``False``. Monkeypatching the attribute later (as tests
    do) simply removes the wrapper — timings disappear, nothing breaks.
    """
    original = getattr(module, attr, None)
    if original is None or getattr(original, _INSTRUMENTED, None) == name:
        return False

    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        with stage(name):
            return await original(*args, **kwargs)

    setattr(wrapper, _INSTRUMENTED, name)
    setattr(module, attr, wrapper)
    return True


# --------------------------------------------------------------------------- #
# Settings snapshot
# --------------------------------------------------------------------------- #

#: RAG knobs worth pinning to a run: retrieval width, grader bar, feature flags,
#: budgets. Anything not listed here is deliberately left out of the record.
_RAG_SNAPSHOT_KEYS = (
    "mode",
    "source",
    "limit",
    "min_score",
    "rerank_candidates",
    "condense_enabled",
    "condense_first_turn",
    "corpus_tree_enabled",
    "grader_enabled",
    "grader_threshold",
    "grader_keep_top",
    "max_context_chars",
    "file_full_chars",
    "section_max_chars",
    "max_expanded_files",
    "token_budget",
    # Поводки скрытых вызовов. Без них в записи не отличить «грейдер отработал
    # за 20 с» от «грейдер упёрся в свой дедлайн»: и то и другое выглядит как
    # 20 с в `timings_ms`. Наблюдалось на прогоне `baseline`, где грейдер молча
    # не отработал на 41 ходе из 46.
    "condense_timeout",
    "grader_timeout",
    # Бюджеты вывода скрытых вызовов: `finish_reason=length` у грейдера
    # читается только рядом с тем, сколько ему было позволено.
    "grader_max_tokens",
    "condense_max_tokens",
)

#: GigaChat knobs. Whitelisted rather than filtered: the section also holds
#: ``cert_path``/``key_path``/``key_passphrase``, which must never be logged.
#: ``provider`` and ``kitai_model`` are here because the record used to name
#: ``model`` alone — and with ``provider: kitai`` that is NOT the model that
#: answered (``KitaiConfig`` reads ``kitai_model`` first). A report built from
#: the log named the wrong model.
_GIGA_SNAPSHOT_KEYS = (
    "provider",
    "model",
    "kitai_model",
    "temperature",
    "max_tokens",
    "model_context_tokens",
)


def model_effective(gcfg: dict[str, Any] | None) -> dict[str, Any]:
    """``{"provider", "model"}`` — which backend answered, with which model.

    Mirrors the resolution the transports do (:func:`app.llm.provider_of`,
    ``KitaiConfig.from_dict``) without the warning ``provider_of`` logs on an
    unknown value: the snapshot describes, it does not complain. Kept next to
    the whitelist so the harness never has to know that KitAI keeps its model
    under a different key.
    """
    giga_cfg = gcfg if isinstance(gcfg, dict) else {}
    provider = str(giga_cfg.get("provider", "") or DEFAULT_PROVIDER).strip().lower()
    if provider not in PROVIDERS:
        provider = DEFAULT_PROVIDER
    model = giga_cfg.get("model")
    if provider == "kitai":
        model = giga_cfg.get("kitai_model") or model
    return {"provider": provider, "model": model or None}


def _prompt_fingerprint(value: Any) -> str | None:
    """Short stable digest of a prompt override (``None`` = built-in default).

    The text itself is not logged — it can be long and it is the *identity* that
    matters for reproducibility ("was this run made with the same prompts?").
    """
    if not isinstance(value, str) or not value.strip():
        return None
    return hashlib.sha1(value.strip().encode("utf-8")).hexdigest()[:8]


def settings_snapshot(
    rcfg: dict[str, Any] | None,
    gcfg: dict[str, Any] | None,
    prompts: dict[str, Any] | None,
) -> dict[str, Any]:
    """The effective knobs of this turn — what a later run has to match.

    Retrieval width, grader threshold, feature flags, generation parameters and
    a fingerprint of each answer prompt. Credentials are excluded by
    construction (whitelist), not by filtering.
    """
    rag_cfg = rcfg if isinstance(rcfg, dict) else {}
    giga_cfg = gcfg if isinstance(gcfg, dict) else {}
    prompt_cfg = prompts if isinstance(prompts, dict) else {}
    return {
        "rag": {k: rag_cfg.get(k) for k in _RAG_SNAPSHOT_KEYS if k in rag_cfg},
        "gigachat": {k: giga_cfg.get(k) for k in _GIGA_SNAPSHOT_KEYS if k in giga_cfg},
        "model_effective": model_effective(giga_cfg),
        "prompts": {
            key: _prompt_fingerprint(prompt_cfg.get(key))
            for key in ("system", "context_reminder")
        },
    }


# --------------------------------------------------------------------------- #
# Head block
# --------------------------------------------------------------------------- #

#: The two renderings the structural head block can take — the section tree
#: (:mod:`app.corpus_tree`) or the constant-size footprint
#: (:mod:`app.corpus_map`). ``app.rag._head_block`` produces one of them or
#: nothing; these names are its and this record's shared vocabulary.
HEAD_BLOCK_TREE = "tree"
HEAD_BLOCK_FOOTPRINT = "footprint"

_HEAD_BLOCK_KINDS = frozenset({HEAD_BLOCK_TREE, HEAD_BLOCK_FOOTPRINT})


def head_block_snapshot(kind: Any, chars: Any) -> dict[str, Any]:
    """``{"kind": …, "chars": …}`` — what opened the final user turn.

    ``kind`` is :data:`HEAD_BLOCK_TREE`, :data:`HEAD_BLOCK_FOOTPRINT`, or
    ``None`` — and ``None`` is the field's whole reason for existing. It is the
    *degraded* state: both renderings need ``GET /api/vault/catalog``, so a UI
    newer than its backend loses the entire block with no other trace. Whether
    that happened used to be unanswerable after the fact, since ``context_text``
    is sliced from «Источники:» forward.

    Anything unrecognised also becomes ``None`` (and a non-integer ``chars``
    becomes ``0``): the record is telemetry and must never carry a value a later
    reader has to guess about. The block's TEXT is deliberately absent — see the
    module docstring.

    Scope: the RETRIEVAL turn's head block only. A meta turn (``intent ==
    "meta"``) builds its structure through a different seam
    (``app.rag._structure_block``) and leaves this ``None`` — read the field
    together with ``intent``.
    """
    return {
        "kind": kind if kind in _HEAD_BLOCK_KINDS else None,
        "chars": chars if isinstance(chars, int) and not isinstance(chars, bool) else 0,
    }


def read_records(paths: AppPaths) -> list[dict[str, Any]]:
    """Parse the current log into records, skipping unparsable lines.

    Convenience for eval tooling and tests; the write path never reads back.
    """
    path = log_path(paths)
    out: list[dict[str, Any]] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out
