"""Async mTLS client for the KitAI platform — our own, not the vendor SDK.

KitAI is not an OpenAI-compatible endpoint. One completion is three calls:

1. ``POST /api/v1/query/model``      — enqueue the query, we generate its ``query_id``;
2. ``GET  /api/v1/query/{id}/result`` — poll until ``query_status == "finished"``;
3. ``PUT  /api/v1/query/{id}/commit`` — acknowledge, so the platform can release it.

Authentication is the client PEM certificate, exactly as for the direct GigaChat
transport, plus two identification headers that name the calling system/module.

**Why not `sber-kitai-sdk-langchain`.** The adapter is a `langchain_gigachat.GigaChat`
subclass, so taking it pulls langchain-core, langchain-gigachat and the generated
pydantic client into an image whose entire dependency list is nine pure-python
packages — to gain three HTTP calls we already know how to make. It is also
synchronous (`time.sleep` in the polling loop, `asyncio.to_thread` around
`_generate`), which is the wrong shape for this FastAPI service. Two concrete
defects in the shipped 11.2.3 sealed it:

* the polling loop honours only ``polling_timeout_in_sec`` — ``polling_retries``,
  ``polling_delay_in_sec`` and ``polling_start_delay_in_sec`` are accepted and
  silently ignored (`polling_cycle.do_polling`);
* the commit callback is declared as ``commit()`` but invoked as
  ``commitFoo(query_id=..., _headers=...)``, so it raises ``TypeError`` inside a
  ``ThreadPoolExecutor`` whose future is never read — commits never happen and
  nothing says so.

The wire contract below was read off the generated SDK (11.2.0), so it matches
what the platform expects; the field names go over the wire in snake_case,
because the generated DTOs declare no aliases.

**No token streaming.** The platform has no streaming surface here, so
:func:`stream_chat` yields the finished answer as a single chunk. It keeps the
signature of its GigaChat counterpart so the SSE route does not branch.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, NamedTuple

import httpx

from . import llm_trace, mtls
from .gigachat import extract_json
from .llm_errors import (
    GigaChatBadJSON,
    GigaChatError,
    GigaChatHTTP,
    GigaChatTLS,
    KitaiCatalogForbidden,
    KitaiPollingTimeout,
    KitaiQueryFailed,
)

log = logging.getLogger("cognivault-ui.kitai")

# Injectable so tests can drive the polling loop without real time passing.
_sleep = asyncio.sleep


@dataclass
class KitaiConfig:
    """Flattened view of the ``gigachat`` config section for a KitAI request.

    It reads the SAME config section as :class:`app.gigachat.GigaConfig` — the
    provider is one key inside it, not a separate tree — so switching backends
    never means re-entering the certificate paths, temperature or token budget.
    """

    host: str
    model: str
    cert_path: str
    key_path: str
    key_passphrase: str
    verify_ssl: bool
    temperature: float
    max_tokens: int
    system_name: str
    module_name: str
    profanity_check: bool
    poll_timeout: float
    poll_initial_delay: float
    poll_delay: float
    # Дополнительные поля запроса к платформе (см. `_build_body`). Только
    # добавляются; наши поля перекрыть нельзя.
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, gc: dict[str, Any]) -> "KitaiConfig":
        return cls(
            extra_body=_parse_extra_body(gc.get("kitai_extra_body")),
            host=str(gc.get("kitai_host", "")).rstrip("/"),
            model=str(gc.get("kitai_model", "") or gc.get("model", "")),
            **_resolve_cert(gc),
            verify_ssl=bool(gc.get("verify_ssl", False)),
            temperature=float(gc.get("temperature", 0.2)),
            max_tokens=int(gc.get("max_tokens", 4096)),
            system_name=str(gc.get("kitai_system_name", "")),
            module_name=str(gc.get("kitai_module_name", "") or ""),
            profanity_check=bool(gc.get("kitai_profanity_check", False)),
            poll_timeout=float(gc.get("kitai_poll_timeout", 240.0)),
            poll_initial_delay=float(gc.get("kitai_poll_initial_delay", 2.0)),
            poll_delay=float(gc.get("kitai_poll_delay", 2.0)),
        )


def _resolve_cert(gc: dict[str, Any]) -> dict[str, str]:
    """Which client certificate KitAI presents.

    KitAI and GigaChat are DIFFERENT contours and may want different
    certificates, so `kitai_cert_path` exists; empty means "the GigaChat one",
    which is what an install with a single pair wants and what shipped before.

    The subtle part is the CONFIGURED-BUT-ABSENT case. The secret is mounted
    `optional: true`, so setting the path before creating the secret leaves an
    empty directory — and a hard failure there takes the whole chat down with
    "certificate not found" for what is really a deployment ordering mistake.
    We fall back to the shared pair and say so, loudly and every time: the chat
    keeps answering, and the log names both paths so the cause is not a mystery.

    Not silent, deliberately. If the KitAI contour genuinely requires its own
    certificate, the fallback will still be rejected — and the operator needs to
    see WHY that happened rather than conclude the certificate is in use.
    """
    shared_cert = os.path.expanduser(str(gc.get("cert_path", "")))
    shared_key = os.path.expanduser(str(gc.get("key_path", "")))
    shared_pass = str(gc.get("key_passphrase", "") or "")

    own_cert = os.path.expanduser(str(gc.get("kitai_cert_path") or ""))
    own_key = os.path.expanduser(str(gc.get("kitai_key_path") or ""))
    if not own_cert and not own_key:
        return {
            "cert_path": shared_cert,
            "key_path": shared_key,
            "key_passphrase": shared_pass,
        }

    if os.path.isfile(own_cert) and os.path.isfile(own_key):
        return {
            "cert_path": own_cert,
            "key_path": own_key,
            "key_passphrase": str(
                gc.get("kitai_key_passphrase") or gc.get("key_passphrase", "") or ""
            ),
        }

    log.warning(
        "kitai: свой сертификат задан (%s / %s), но файлов нет — иду сертификатом "
        "GigaChat (%s). Секрет cognivault-kitai-certs заведён?",
        own_cert or "—",
        own_key or "—",
        shared_cert or "—",
    )
    return {
        "cert_path": shared_cert,
        "key_path": shared_key,
        "key_passphrase": shared_pass,
    }


def _parse_extra_body(value: Any) -> dict[str, Any]:
    """``kitai_extra_body`` as a dict: a dict as is, a JSON string parsed,
    anything else (or invalid JSON) → ``{}`` with a warning, never an error."""
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError as exc:
            log.warning("kitai_extra_body: не JSON (%s) — игнорирую", exc)
            return {}
        if isinstance(parsed, dict):
            return parsed
    log.warning("kitai_extra_body: ожидался объект, получен %s — игнорирую", type(value).__name__)
    return {}


# Какие наборы дополнительных ключей уже объявлены в логе. Один раз на процесс
# и на набор: значения не пишем никогда (это параметры платформы, а не наши).
_announced_extra_keys: set[tuple[str, ...]] = set()


def _headers(cfg: KitaiConfig) -> dict[str, str]:
    """Identification headers the platform requires alongside the certificate."""
    h = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "cognivault-ui",
        "x-identification-system": cfg.system_name,
    }
    if cfg.module_name:
        h["x-identification-module"] = cfg.module_name
    return h


def _make_client(
    cfg: KitaiConfig, transport: httpx.AsyncBaseTransport | None
) -> httpx.AsyncClient:
    """Client for the query/poll/commit trio.

    ``read`` is a real timeout here (unlike the streaming GigaChat client): every
    call is a short request/response, and the long wait is our own polling loop,
    not a hanging socket.

    With ``transport`` supplied (tests) the certificate check and the whole TLS
    wiring are skipped — the fake transport never reaches the network.
    """
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)
    if transport is not None:
        return httpx.AsyncClient(transport=transport, timeout=timeout)
    mtls.files_present(cfg)
    return mtls.make_client(cfg, timeout)


def _check(resp: httpx.Response, what: str) -> dict[str, Any]:
    """Non-200 → typed error; body → dict."""
    if resp.status_code != 200:
        raise GigaChatHTTP(
            "GIGACHAT_HTTP",
            f"KitAI вернул HTTP {resp.status_code} на {what}",
            resp.text[:500],
        )
    try:
        body = resp.json()
    except ValueError as exc:
        raise GigaChatBadJSON(
            "GIGACHAT_BAD_JSON", f"KitAI вернул не-JSON на {what}", str(exc)
        ) from exc
    return body if isinstance(body, dict) else {"data": body}


def _build_body(messages: list[dict[str, Any]], cfg: KitaiConfig, query_id: str) -> bytes:
    """The ``UniversalModelQueryPDto`` payload.

    ``exclude_none`` semantics are reproduced by simply not emitting keys we do
    not set: the generated DTO serialises with ``exclude_none=True``, and sending
    explicit nulls to a Java service is a good way to find out which fields it
    validates.
    """
    payload: dict[str, Any] = {
        "query_id": query_id,
        "model_name": cfg.model,
        "messages": [
            {"role": str(m.get("role", "")), "content": str(m.get("content", "") or "")}
            for m in messages
        ],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "profanity_check": cfg.profanity_check,
    }
    # `kitai_extra_body` — параметры платформы, которых наш DTO не знает
    # (например, выключение режима рассуждений у модели). Только те ключи,
    # которых у нас нет: `query_id`, `model_name`, `messages` и остальное
    # перекрыть нельзя, иначе оператор одной строкой ENV подменил бы запрос.
    added = tuple(sorted(k for k in cfg.extra_body if k not in payload))
    for key in added:
        payload[key] = cfg.extra_body[key]
    if added and added not in _announced_extra_keys:
        _announced_extra_keys.add(added)
        log.info("kitai: в запрос добавляются поля из kitai_extra_body: %s", ", ".join(added))
    return json.dumps(payload, ensure_ascii=False).encode()


def _failure_detail(data: dict[str, Any]) -> str | None:
    """Everything the platform said about why, in one line.

    `QueryResultPDto` carries the reason in up to three places and fills
    whichever one it feels like: `error` (its own status/message), plus
    `response_code`/`response_body` passed through from upstream. Reading only
    `error.message` — as the first cut did — produced a failure with an empty
    detail and nothing to act on.
    """
    parts: list[str] = []
    err = data.get("error") or {}
    if isinstance(err, dict):
        if err.get("status") is not None:
            parts.append(f"error.status={err['status']}")
        if err.get("message"):
            parts.append(str(err["message"]))
    elif err:
        parts.append(str(err))
    if data.get("response_code") is not None:
        parts.append(f"response_code={data['response_code']}")
    if data.get("response_body"):
        parts.append(f"response_body={str(data['response_body'])[:300]}")
    return "; ".join(parts)[:600] or None


def _retryable_query_failure(exc: KitaiQueryFailed) -> bool:
    """Whether one fresh enqueue can recover from a final KitAI error.

    An empty platform detail is treated as transient: this is the production
    failure seen in q1. Explicit client/configuration failures (4xx, RBAC,
    unknown model) are not retried; a second identical request would only add
    latency and load.
    """
    detail = str(exc.detail or "").casefold()
    if not detail:
        return True
    transient_codes = (
        "response_code=500",
        "response_code=502",
        "response_code=503",
        "response_code=504",
    )
    if any(marker in detail for marker in transient_codes):
        return True
    return any(
        marker in detail
        for marker in (
            "temporar",
            "timeout",
            "timed out",
            "unavailable",
            "connection termination",
        )
    )


class QueryResult(NamedTuple):
    """What a finished query yields: the text plus the metadata around it.

    ``usage`` is the upstream ``response.usage`` block when the platform passed
    one through, else ``None`` — it is telemetry (see :mod:`app.llm_trace`),
    never something the answer path depends on.
    """

    content: str
    finish_reason: str | None
    usage: dict[str, Any] | None


def _extract(result: dict[str, Any]) -> QueryResult:
    """``(content, finish_reason, usage)`` out of a finished ``QueryResultPDto``."""
    data = result.get("data") or {}
    response = data.get("response") or {}
    if not isinstance(response, dict):
        response = {}
    choices = response.get("choices") or []
    if not choices:
        raise KitaiQueryFailed(
            "KITAI_EMPTY_RESULT",
            "KitAI завершил запрос без единого варианта ответа",
            _failure_detail(data),
        )
    first = choices[0] or {}
    message = first.get("message") or {}
    finish = first.get("finish_reason")
    usage = response.get("usage")
    return QueryResult(
        str(message.get("content") or ""),
        str(finish) if finish not in (None, "") else None,
        usage if isinstance(usage, dict) else None,
    )


async def _commit(client: httpx.AsyncClient, cfg: KitaiConfig, query_id: str) -> None:
    """Acknowledge the query. Best-effort: a failed commit must not lose the answer.

    Awaited rather than fired into a background executor — a commit that silently
    never runs is exactly the vendor SDK's bug, and one extra short round-trip is
    cheaper than an un-released query on the platform.
    """
    try:
        resp = await client.put(
            f"{cfg.host}/api/v1/query/{query_id}/commit", headers=_headers(cfg)
        )
        if resp.status_code != 200:
            log.warning(
                "kitai: commit %s вернул HTTP %s: %s",
                query_id,
                resp.status_code,
                resp.text[:200],
            )
    except Exception as exc:  # noqa: BLE001 — commit is bookkeeping, never fatal
        log.warning("kitai: commit %s не удался: %s", query_id, exc)


async def _run_query(
    messages: list[dict[str, Any]],
    cfg: KitaiConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[str, str | None]:
    """Enqueue, poll to completion, commit. Returns ``(content, finish_reason)``.

    The pair is the historical contract; :func:`_query` underneath also carries
    ``usage`` for the callers that stamp telemetry.
    """
    result = await _query(messages, cfg, transport=transport)
    return result.content, result.finish_reason


async def _query(
    messages: list[dict[str, Any]],
    cfg: KitaiConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> QueryResult:
    """Enqueue, poll to completion, commit. Returns a :class:`QueryResult`."""
    if not cfg.host:
        raise GigaChatError(
            "KITAI_NOT_CONFIGURED", "Не задан адрес KitAI (KITAI_HOST)", None
        )
    if not cfg.system_name:
        raise GigaChatError(
            "KITAI_NOT_CONFIGURED",
            "Не задано имя системы для KitAI (KITAI_SYSTEM_NAME)",
            None,
        )

    query_id = str(uuid.uuid4())
    try:
        client = _make_client(cfg, transport)
    except ssl.SSLError as exc:
        raise GigaChatTLS(
            "GIGACHAT_TLS", "Не удалось загрузить сертификат/ключ", str(exc)
        ) from exc

    async with client:
        try:
            resp = await client.post(
                f"{cfg.host}/api/v1/query/model",
                content=_build_body(messages, cfg, query_id),
                headers=_headers(cfg),
            )
        except httpx.HTTPError as exc:
            raise mtls.classify_connect_error(exc, what="KitAI") from exc
        _check(resp, "постановку запроса")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + cfg.poll_timeout
        await _sleep(cfg.poll_initial_delay)

        while True:
            try:
                resp = await client.get(
                    f"{cfg.host}/api/v1/query/{query_id}/result",
                    headers=_headers(cfg),
                )
            except httpx.HTTPError as exc:
                raise mtls.classify_connect_error(exc, what="KitAI") from exc
            body = _check(resp, "получение результата")
            data = body.get("data") or {}
            status = str(data.get("query_status") or "")

            if status == "finished":
                result = _extract(body)
                await _commit(client, cfg, query_id)
                return result

            # `is_final` on a non-finished status means the platform gave up.
            if data.get("is_final"):
                detail = _failure_detail(data)
                # Logged here as well as raised: the query_id is the only handle
                # the platform side has, and it is not part of the user-facing
                # message. Without it a support request is "it said error".
                log.warning(
                    "kitai: запрос %s завершился со статусом %r (модель %s): %s",
                    query_id,
                    status or "неизвестно",
                    cfg.model,
                    detail or "платформа не сообщила причину",
                )
                raise KitaiQueryFailed(
                    "KITAI_QUERY_FAILED",
                    f"KitAI завершил запрос со статусом «{status or 'неизвестно'}»"
                    f" (модель {cfg.model})",
                    detail,
                )

            if loop.time() >= deadline:
                raise KitaiPollingTimeout(
                    "KITAI_TIMEOUT",
                    f"KitAI не вернул ответ за {cfg.poll_timeout:.0f} с",
                    f"query_id={query_id}, последний статус: {status or 'неизвестно'}",
                )
            await _sleep(cfg.poll_delay)


async def list_models(
    cfg: KitaiConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[dict[str, str]]:
    """Models the platform currently offers: ``GET /api/v1/meta/model``.

    Returns ``[{"name", "label"}]``. `display_name` is what the platform wants
    shown to a human; it is optional, so the wire name is the fallback and also
    the value we send back in `model_name`.

    Errors are the caller's to handle — a settings form that cannot reach the
    platform should say so and fall back to a free-text field, not pretend the
    list is empty (an empty list reads as "no models available").
    """
    if not cfg.host:
        raise GigaChatError(
            "KITAI_NOT_CONFIGURED", "Не задан адрес KitAI (KITAI_HOST)", None
        )
    try:
        client = _make_client(cfg, transport)
    except ssl.SSLError as exc:
        raise GigaChatTLS(
            "GIGACHAT_TLS", "Не удалось загрузить сертификат/ключ", str(exc)
        ) from exc

    async with client:
        try:
            resp = await client.get(
                f"{cfg.host}/api/v1/meta/model", headers=_headers(cfg)
            )
        except httpx.HTTPError as exc:
            raise mtls.classify_connect_error(exc, what="KitAI") from exc
        if resp.status_code in (401, 403):
            # Наблюдалось на IFT: `POST /query/model` тем же сертификатом
            # принимается (200), а каталог отвечает 403 «Access denied for the
            # certificate». То есть сертификат ДЕЙСТВИТЕЛЕН, просто у него нет
            # права на мета-эндпоинт — это состояние прав, а не поломка, и
            # рассказывать о нём как об ошибке значит слать оператора искать
            # несуществующую проблему с подключением.
            raise KitaiCatalogForbidden(
                "KITAI_CATALOG_FORBIDDEN",
                "У сертификата нет доступа к списку моделей KitAI",
                resp.text[:300],
            )
        if resp.status_code != 200:
            raise GigaChatHTTP(
                "GIGACHAT_HTTP",
                f"KitAI вернул HTTP {resp.status_code} на список моделей",
                resp.text[:500],
            )
        try:
            body = resp.json()
        except ValueError as exc:
            raise GigaChatBadJSON(
                "GIGACHAT_BAD_JSON", "KitAI вернул не-JSON на список моделей", str(exc)
            ) from exc

    # The endpoint answers with a bare array; tolerate a wrapper too, since the
    # rest of this API wraps everything in `{description, data}`.
    items = body if isinstance(body, list) else (body or {}).get("data") or []
    out: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("model_name") or "").strip()
        if not name:
            continue
        label = str(item.get("display_name") or "").strip() or name
        version = str(item.get("version") or "").strip()
        if version and version not in label:
            label = f"{label} ({version})"
        out.append({"name": name, "label": label})
    return out


async def stream_chat(
    messages: list[dict[str, Any]],
    cfg: KitaiConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[str]:
    """Yield the assistant's answer.

    Exactly one chunk: the platform has no streaming surface, so pretending to
    stream would only move the wait from the spinner into a fake typewriter. The
    caller reads ``cfg.last_finish_reason`` afterwards, same as for GigaChat.
    """
    llm_trace.reset(cfg, cfg.model)
    try:
        result = await _query(messages, cfg, transport=transport)
    except KitaiQueryFailed as exc:
        if not _retryable_query_failure(exc):
            raise
        log.warning("kitai: временный сбой ответа — один повтор запроса: %s", exc)
        await _sleep(min(max(cfg.poll_delay, 0.0), 1.0))
        result = await _query(messages, cfg, transport=transport)
    llm_trace.stamp(
        cfg,
        finish_reason=result.finish_reason,
        usage=result.usage,
        content=result.content,
    )
    if result.content:
        yield result.content


async def complete_json(
    messages: list[dict[str, Any]],
    cfg: KitaiConfig,
    *,
    timeout: float | None = None,
    max_tokens: int | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """One query, one parsed JSON object — for the hidden pipeline steps.

    ``timeout`` caps the polling budget for this call: the condense and grader
    steps sit on the critical path with 10 s / 20 s leashes of their own, and a
    240 s default would let one slow hidden call outlast the user's patience.

    ``finish_reason``, ``usage``, the head of the raw text and the model are
    stamped onto ``cfg`` (the CALLER's object, not the per-call copy below) —
    see :mod:`app.llm_trace`. The stamps go on before ``extract_json`` runs, so
    an answer cut off by ``max_tokens`` is reported as ``length`` rather than
    as a bare «пустой ответ».
    """
    llm_trace.reset(cfg, cfg.model)
    overrides: dict[str, Any] = {}
    if timeout is not None:
        overrides["poll_timeout"] = float(timeout)
        # A first look scheduled after the deadline would time out without ever
        # asking for the result.
        overrides["poll_initial_delay"] = min(cfg.poll_initial_delay, float(timeout))
        overrides["poll_delay"] = min(cfg.poll_delay, float(timeout))
    if max_tokens is not None:
        overrides["max_tokens"] = int(max_tokens)
    # `replace` copies FIELDS only — the trace stamps live on the instance, and
    # a `**__dict__` copy would choke on them.
    call_cfg = replace(cfg, **overrides) if overrides else cfg

    result = await _query(messages, call_cfg, transport=transport)
    llm_trace.stamp(
        cfg,
        finish_reason=result.finish_reason,
        usage=result.usage,
        content=result.content,
    )
    return extract_json(result.content)
