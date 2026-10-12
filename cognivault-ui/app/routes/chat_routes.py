"""Chat endpoint: SSE streaming with optional RAG."""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import time
from dataclasses import replace
from datetime import datetime
from typing import Any, AsyncIterator
from urllib.parse import urlsplit

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import cognivault, history, llm, rag, rag_log, rag_pipeline, settings
from ..config import AppPaths
from ..confluence import store as confluence_store
from ..deps import cv_context, resolve_paths
from ..llm_errors import GigaChatCertMissing, GigaChatError
from ..sse import format_sse, sse_error
from ..tokens import estimate_messages_tokens, trim_history

log = logging.getLogger("cognivault-ui.chat")

router = APIRouter(prefix="/api")

# Запас поверх system + max_tokens: разметка чата, служебные токены модели.
_TRIM_RESERVE_TOKENS = 500

# Цитаты вида «[Источник 3]», «[Источники 1, 2]», «[Источника 4; 5]».
_CITATION_RE = re.compile(
    r"\[\s*Источник(?:и|а)?\s+(\d+(?:\s*[,;]\s*\d+)*)\s*\]"
)

# Любая скобка, начинающаяся со слова «Источник» — включая те, что НЕ являются
# номерной ссылкой: «[Источник базы знаний]», «[Источник: инструкция]». Такой
# маркер выглядит как цитата, но не кликается и ничего не подтверждает —
# проверить утверждение по нему нельзя. `_invalid_citations` их не видит: там
# номера, а тут номера как раз нет.
_CITATION_LIKE_RE = re.compile(r"\[\s*Источник\w*\b[^\]]*\]")

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

# `finish_reason` для ветки готового ответа (`RagContext.answer_override`):
# грейдер не оставил ни одного пригодного фрагмента, GigaChat в этом ходе не
# вызывался — значит подставить его `last_finish_reason` нечем, код наш.
_NO_CONTEXT_FINISH_REASON = "no_context"

# Причина повтора генерации в записи лога (`answer_retry_reason`).
_EMPTY_LENGTH_RETRY_REASON = "empty_length"

# Показывается кадром `notice` (он отрисовывается в пузыре и тостом — см.
# `static/app.js`), когда и повтор не дал ни символа.
_EMPTY_COMPLETION_NOTICE = (
    "Модель не вернула текст: бюджет ответа исчерпан (finish_reason=length). "
    "Попробуйте переформулировать или сократить вопрос."
)


def _retry_config(cfg: Any, giga_dict: dict[str, Any]) -> Any:
    """Copy of the answer config with a doubled output budget, or ``None``.

    ``min(max_tokens * 2, model_context_tokens // 2)`` — the doubling is the
    cheapest thing that can help a reasoning model that spent the whole budget
    before its first visible character; the ceiling keeps the request inside
    the model's window. ``None`` when the ceiling leaves no room to grow: a
    retry with the same budget would only repeat the failure. A copy, never
    the stored settings — this is one request's problem.
    """
    try:
        ctx = int(giga_dict.get("model_context_tokens") or 0)
    except (TypeError, ValueError):
        ctx = 0
    if ctx <= 0:
        ctx = 32768
    budget = min(int(cfg.max_tokens) * 2, ctx // 2)
    if budget <= int(cfg.max_tokens):
        return None
    return replace(cfg, max_tokens=budget)


# Заголовок блока источников в отрендеренном user-сообщении
# (`rag._render_context_message`). Сам блок идёт сразу за ним и имеет ровно
# `ctx.context_chars` символов — по этому инварианту он вырезается обратно в
# лог без парсинга разметки. Ищется поиском, а не по началу строки: перед
# источниками может стоять блок «состав базы» (`app.corpus_map`), и он в лог
# контекста не входит — там должно остаться ровно то, что оценивает eval.
_SOURCES_PREFIX = "Источники:\n\n"

# Инструментирование стадий конвейера. Роут владеет только своими вызовами, а
# condense/поиск/грейдер живут внутри `rag.build_rag_context`; поэтому seams
# оборачиваются один раз при импорте (см. `rag_log.instrument` — обёртка
# идемпотентна, а monkeypatch в тестах её просто снимает). Сами модули при этом
# не правятся: тайминги — забота лога, а не конвейера.
for _module, _attr, _stage in (
    (rag_pipeline, "condense", "condense"),
    (rag_pipeline, "grade", "grade"),
    (cognivault, "hybrid_search", "search"),
    (cognivault, "semantic_search", "search"),
    (cognivault, "content", "content"),
):
    rag_log.instrument(_module, _attr, _stage)


def _effective_config(paths: AppPaths) -> dict[str, Any]:
    """Активный конфиг ЭТОГО пользователя.

    Предпочитаем пер-пользовательский ``settings.effective_config_for(paths)``;
    пока/если его нет — мягко откатываемся на глобальный
    ``settings.effective_config()``. Обращение через :func:`getattr`, чтобы
    маршрут не ломался на сборке без новой функции.
    """
    per_user = getattr(settings, "effective_config_for", None)
    if callable(per_user):
        return per_user(paths)
    return settings.effective_config()


def _new_chat_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def _error(status: int, code: str, message: str, detail: str | None = None) -> JSONResponse:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if detail is not None:
        body["error"]["detail"] = detail
    return JSONResponse(status_code=status, content=body)


def _fit_to_context(
    send: list[dict[str, Any]], giga_dict: dict[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    """Урезать историю в ``send`` под контекстное окно модели.

    Бюджет ``= model_context_tokens - max_tokens - <system> - 500``.
    Системный префикс не режется — его стоимость вычитается из бюджета целиком.

    Остальное уходит в :func:`trim_history`, который защищает хвост, начиная с
    ПОСЛЕДНЕГО ``user``-сообщения, и при этом учитывает его размер в оценке. В
    RAG-режиме последним сообщением идёт как раз user-сообщение с блоком
    «Источники» — значит контекст никогда не режется, а его токены всё равно
    вычитаются из бюджета: обрезается только история между system и ним.

    Возвращает ``(список для модели, число отброшенных сообщений)``.
    """
    head = 0
    while head < len(send) and send[head].get("role") == "system":
        head += 1
    system_prefix = send[:head]
    convo = send[head:]

    model_ctx = int(giga_dict.get("model_context_tokens", 32768) or 32768)
    max_tokens = int(giga_dict.get("max_tokens", 4096) or 4096)
    budget = (
        model_ctx
        - max_tokens
        - estimate_messages_tokens(system_prefix)
        - _TRIM_RESERVE_TOKENS
    )

    convo, dropped = trim_history(convo, max(0, budget))
    return [*system_prefix, *convo], dropped


def _last_user_content(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _without_last_user_turn(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """История без последнего ``user``-хода (и всего, что после него).

    В RAG-режиме сам вопрос переезжает в финальное user-сообщение вместе с
    источниками, поэтому дублировать его в истории не нужно.
    """
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return list(messages[:i])
    return list(messages)


def _invalid_citations(text: str, n_sources: int) -> list[int]:
    """Номера ``[Источник N]`` из ответа, выходящие за ``1..n_sources``.

    Дешёвая серверная проверка галлюцинированных ссылок: модель иногда
    цитирует источник, которого в контексте не было. Возвращает отсортированный
    список уникальных «плохих» номеров (пустой — всё в порядке).
    """
    if not text:
        return []
    found: set[int] = set()
    for match in _CITATION_RE.finditer(text):
        for part in re.split(r"[,;]", match.group(1)):
            part = part.strip()
            if part.isdigit():
                found.add(int(part))
    return sorted(n for n in found if n < 1 or n > n_sources)


def _malformed_citations(text: str) -> list[str]:
    """Маркеры «[Источник …]» без номера — их нельзя ни открыть, ни проверить.

    Наблюдалось в проде: «[Источник базы знаний]». Для пользователя это выглядит
    как ссылка, но никуда не ведёт; для оценки качества — как будто утверждение
    подтверждено источником. Считаем отдельно от `_invalid_citations`: там
    выдуманный НОМЕР, здесь номера нет вовсе.
    """
    if not text:
        return []
    bad: list[str] = []
    for match in _CITATION_LIKE_RE.finditer(text):
        if not _CITATION_RE.fullmatch(match.group(0)):
            bad.append(match.group(0))
    return sorted(set(bad))


def _linked_candidates(
    candidates: list[dict[str, Any]] | None, grades: list[dict[str, Any]] | None
) -> list[dict[str, Any]] | None:
    """Give every candidate the id its grade is filed under.

    The grader numbers fragments 1..N positionally over the candidate list
    (``rag_pipeline._parse_grades``), so ``grades[i].id`` and ``candidates[i]``
    describe the same chunk — but nothing in the record said so, which made
    ``grades[].id`` a number pointing at nothing. Copying the id onto the
    candidate closes the loop: ``candidates[k].id == grades[m].id`` ⇒ same chunk.

    An id already present upstream wins, so a future pipeline that numbers
    candidates itself is not overridden.
    """
    if candidates is None:
        return None
    by_index = {i: g for i, g in enumerate(grades or []) if isinstance(g, dict)}
    out: list[dict[str, Any]] = []
    for i, candidate in enumerate(candidates):
        item = dict(candidate)
        grade = by_index.get(i) or {}
        item["id"] = item.get("id") or grade.get("id") or i + 1
        out.append(item)
    return out


def _safe_page_url(value: Any) -> str | None:
    """Return an absolute HTTP(S) page URL, rejecting unsafe metadata values."""
    text = str(value or "").strip()
    try:
        parsed = urlsplit(text)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    return text


async def _enrich_source_urls(
    sources: list[dict[str, Any]],
    manifest_urls: dict[str, str],
    cv: dict[str, Any] | None,
) -> None:
    """Attach durable Confluence links to RAG sources in place.

    The sync manifest is the cheap first choice, but it can disappear when an
    OpenShift pod backed by ``emptyDir`` is replaced. For sources absent from
    that manifest, read ``source_url`` from the document frontmatter through
    CogniVault. Lookups run in parallel and never make an answer fail.
    """
    unresolved: list[dict[str, Any]] = []
    for source in sources:
        path = str(source.get("path") or "")
        url = _safe_page_url(source.get("url")) or _safe_page_url(
            manifest_urls.get(path)
        )
        if url:
            source["url"] = url
        elif path:
            unresolved.append(source)

    async def from_frontmatter(source: dict[str, Any]) -> None:
        try:
            meta = await cognivault.metadata(str(source["path"]), cv=cv)
            url = _safe_page_url(meta.get("source_url"))
            if url:
                source["url"] = url
        except Exception:  # noqa: BLE001 — source links are best-effort
            log.debug(
                "не удалось получить source_url для %s",
                source.get("path"),
                exc_info=True,
            )

    if unresolved:
        await asyncio.gather(*(from_frontmatter(source) for source in unresolved))


def _chunk_indexes_for(
    source: dict[str, Any], candidates: list[dict[str, Any]] | None
) -> list[int]:
    """Which retrieved chunks of a file this context block actually carries.

    ``sources`` is a UI-facing shape and has no chunk identity, so a golden pair
    could never be checked at chunk level — only "was the file cited". The
    identity is recoverable from ``candidates``, which do carry
    ``chunk_index``:

    * ``depth == "file"`` — the whole document is in the context, so every
      retrieved chunk of that path counts as present;
    * otherwise the block was rendered from one fragment and its ``score`` is
      that fragment's score, which pins the exact candidate;
    * if the score matches nothing (merged chunks, a ``None`` score, an older
      pipeline) fall back to every retrieved chunk of the path — an
      over-approximation, but the honest one: those chunks were merged into the
      block.
    """
    path = source.get("path")
    if not path or not candidates:
        return []
    own = [c for c in candidates if c.get("path") == path]
    indexes = [c.get("chunk_index") for c in own]
    if source.get("depth") != "file":
        score = source.get("score")
        if score is not None:
            exact = [c.get("chunk_index") for c in own if c.get("score") == score]
            if exact:
                indexes = exact
    return [i for i in indexes if isinstance(i, int)]


def _log_sources(
    sources: list[dict[str, Any]], candidates: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Per-source rows of the query log: identity + verdict, no free text."""
    out: list[dict[str, Any]] = []
    for source in sources:
        explicit = source.get("chunk_index")
        indexes = (
            [explicit]
            if isinstance(explicit, int)
            else _chunk_indexes_for(source, candidates)
        )
        out.append(
            {
                "n": source.get("n"),
                "path": source.get("path"),
                "section_path": source.get("section_path"),
                "depth": source.get("depth"),
                "score": source.get("score"),
                "grade": source.get("grade"),
                # Primary chunk plus every chunk the block covers: a whole-file
                # or merged block legitimately answers for several of them.
                "chunk_index": indexes[0] if indexes else None,
                "chunk_indexes": indexes,
            }
        )
    return out


def _rendered_context(user_message: dict[str, Any] | None, context_chars: int) -> str:
    """The «Источники» block exactly as it went to the model.

    Cut by length, not by parsing: ``context_chars`` is by construction the
    length of the rendered block that follows :data:`_SOURCES_PREFIX`. The
    prefix is *located* rather than assumed to be at position 0 — the corpus
    footprint block may precede it — and if it is absent altogether (a custom
    renderer) the whole content is logged rather than a wrong slice of it.
    """
    if not isinstance(user_message, dict):
        return ""
    content = str(user_message.get("content", "") or "")
    start = content.find(_SOURCES_PREFIX)
    if start < 0 or context_chars <= 0:
        return content
    start += len(_SOURCES_PREFIX)
    return content[start : start + context_chars]


@router.post("/chat")
async def chat(request: Request) -> Any:
    """Stream a GigaChat completion (optionally RAG-augmented) as SSE.

    Pre-flight validation may return a plain JSON error (e.g. missing certs)
    before the stream starts; once streaming begins, errors are terminal SSE
    ``error`` frames.

    Frame order is part of the contract: ``meta`` → (``notice`` | ``sources``)
    → ``token``\\* → ``done``. ``sources`` is always emitted *before* the first
    token so the retrieval/grading latency hides behind the first paint.

    Three RAG outcomes are routed here (wave 2):

    * ``answer_override`` — the answer is already known (nothing survived the
      grader): one ``token`` frame, no GigaChat call at all;
    * ``smalltalk``/``clarify`` — ``system_message`` without ``user_message``:
      the no-RAG rules turn in front of the untouched history, no ``sources``
      frame. The system turn is *not* optional here — without it the model
      answers chit-chat from its own parametric memory;
    * ``kb_question`` — the normal path: rules system turn + history without its
      last user turn + the user turn carrying the sources block.

    A fourth outcome (step 2) rides on the ``kb_question`` shape: ``meta`` — a
    question about the base itself. The user turn carries the section tree
    instead of «Источники», so ``sources`` is empty and the frame order is
    unchanged. On the normal path the answer may be followed by one extra
    ``token`` frame carrying ``RagContext.hedge``.
    """
    body = await request.json()
    if not isinstance(body, dict):
        return _error(400, "BAD_REQUEST", "тело запроса должно быть объектом")

    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return _error(400, "BAD_REQUEST", "messages обязательны")

    use_rag = bool(body.get("rag", False))
    chat_id = str(body.get("chat_id") or "") or _new_chat_id()

    # Per-request identity/context: which CogniVault + which history bucket.
    cv = cv_context(request)
    paths = resolve_paths(request)

    # Reverse index {vault_path: confluence_page_url} so RAG source chips can link
    # back to their origin Confluence page. Defensive: a manifest error must never
    # break chat — fall back to no urls.
    try:
        url_index = confluence_store.manifest_url_index(paths)
    except Exception:  # noqa: BLE001 — link enrichment is best-effort
        url_index = {}

    cfg = _effective_config(paths)
    gcfg = llm.config_for(cfg.get("gigachat", {}))

    # Настраиваемые тексты промптов ответа: `None`/пусто в любом поле означает
    # «взять встроенный дефолт» — разбирается в `rag._resolve_prompt`.
    prompts = cfg.get("prompts")
    if not isinstance(prompts, dict):
        prompts = None

    # Per-request overrides.
    if "temperature" in body and body["temperature"] is not None:
        gcfg.temperature = float(body["temperature"])
    if "max_tokens" in body and body["max_tokens"] is not None:
        gcfg.max_tokens = int(float(body["max_tokens"]))

    rcfg = dict(cfg.get("rag", {}))
    if "rag_limit" in body and body["rag_limit"] is not None:
        rcfg["limit"] = int(body["rag_limit"])

    # Gigachat config view for the RAG char budget (respect per-request max_tokens).
    giga_dict = dict(cfg.get("gigachat", {}))
    giga_dict["max_tokens"] = gcfg.max_tokens

    # Pre-flight: cert/key presence (raises a typed error we convert to 400).
    try:
        llm.files_present(gcfg)
    except GigaChatCertMissing as exc:
        return _error(400, exc.code, exc.message, exc.detail)

    # Normalise incoming messages to {role, content} for GigaChat.
    outgoing: list[dict[str, Any]] = [
        {"role": m.get("role"), "content": m.get("content", "")}
        for m in messages
        if isinstance(m, dict)
    ]
    # User messages to persist (exclude any system prompts entirely).
    user_messages = [m for m in outgoing if m.get("role") != "system"]

    # Position of the answer we are about to produce inside the persisted chat
    # (``save_chat`` writes ``[*user_messages, assistant]``). The UI uses the same
    # index when it POSTs a 👍/👎 to ``/api/feedback``, so the two line up.
    message_index = len(user_messages)
    question_raw = _last_user_content(outgoing)

    async def generator() -> AsyncIterator[str]:
        full_text = ""
        # Повтор генерации на пустом ответе с `finish_reason=length` (см.
        # `_retry_config`). Конфиг ответа подменяется на копию с бо́льшим
        # бюджетом; `finish_reason` в конце читается с того объекта, который
        # отвечал последним.
        answer_cfg = gcfg
        answer_retries = 0
        answer_retry_reason: str | None = None
        sources: list[dict[str, Any]] = []
        context_chars = 0
        context_text = ""
        rag_used = False
        finish_reason: str | None = None
        truncated = False
        errored = False
        invalid_citations: list[int] = []
        notice: str | None = None
        # Телеметрия конвейера волны 2 (интент + condense, кандидаты, грейды).
        # Остаётся `None` вне RAG-режима и на старом `RagContext`.
        intent: str | None = None
        question_standalone: str | None = None
        candidates: list[dict[str, Any]] | None = None
        grades: list[dict[str, Any]] | None = None
        # Шаг 2: охват вопроса из condense и оговорка по концентрации
        # доказательств. Оговорка дописывается ПОСЛЕ ответа модели — она его
        # уточняет, а не заменяет.
        scope: str | None = None
        hedge: str | None = None
        # Структурный блок в голове последнего user-турна: чем он был («tree» /
        # «footprint» / ничего) и какого размера. В `context_text` он не
        # попадает — тот режется от «Источники:», — так что без этой пары по
        # логу нельзя сказать, доехало ли дерево разделов до модели вообще.
        head_block_kind: str | None = None
        head_block_chars: int = 0
        # Отчёты двух скрытых вызовов (condense, грейдер по батчам): статус,
        # ошибка, finish_reason, usage, модель. `grades: null` говорит лишь, что
        # реранкер не сработал; это — почему.
        hidden_calls: dict[str, Any] | None = None
        # Стадии конвейера. `condense`/`search`/`grade`/`content` приходят из
        # инструментированных seams (см. верх модуля), остальное меряем здесь.
        turn_started = time.perf_counter()

        def _elapsed_ms(since: float) -> float:
            return round((time.perf_counter() - since) * 1000.0, 1)

        with rag_log.collect_stages() as stages:
            try:
                # `streaming` говорит фронтенду, чего ждать: у KitAI потока
                # токенов нет вовсе (POST → polling → готовый ответ), и бегущий
                # курсор над пустым пузырём в этом случае врёт про прогресс.
                yield format_sse(
                    "meta",
                    {
                        "chat_id": chat_id,
                        "streaming": llm.supports_streaming(gcfg),
                    },
                )

                send = list(outgoing)

                if use_rag:
                    query = _last_user_content(outgoing)
                    rag_started = time.perf_counter()
                    ctx = await rag.build_rag_context(
                        query, rcfg, cv, giga_dict, outgoing, prompts=prompts
                    )
                    stages["rag"] = _elapsed_ms(rag_started)
                    sources = ctx.sources
                    context_chars = ctx.context_chars
                    notice = ctx.notice
                    # Поля волны 2 читаем мягко: они keyword-defaulted и могут
                    # отсутствовать на более старом `RagContext`.
                    intent = getattr(ctx, "intent", None)
                    question_standalone = getattr(ctx, "standalone_question", None)
                    candidates = getattr(ctx, "candidates", None)
                    grades = getattr(ctx, "grades", None)
                    answer_override = getattr(ctx, "answer_override", None)
                    scope = getattr(ctx, "scope", None)
                    hedge = getattr(ctx, "hedge", None)
                    head_block_kind = getattr(ctx, "head_block_kind", None)
                    head_block_chars = getattr(ctx, "head_block_chars", 0)
                    hidden_calls = getattr(ctx, "hidden_calls", None)

                    if answer_override:
                        # Ответ уже готов (шаблонный отказ: ни один кандидат не
                        # прошёл грейдер) — генерацию пропускаем целиком, GigaChat
                        # в этой ветке не вызывается вообще. Кадры те же, что в
                        # обычном ходе: meta → sources (пустой) → token → done.
                        rag_used = True
                        sources = []
                        context_chars = 0
                        full_text = answer_override
                        finish_reason = _NO_CONTEXT_FINISH_REASON
                        log.info(
                            "chat %s: answer_override (intent=%s), генерация пропущена",
                            chat_id,
                            intent,
                        )
                        yield format_sse("sources", {"sources": [], "context_chars": 0})
                        yield format_sse("token", {"text": full_text})
                        yield format_sse(
                            "done", {"chat_id": chat_id, "finish_reason": finish_reason}
                        )
                        return

                    # Сбой поиска (`notice`) — единственная ветка, где системный
                    # турн не подставляется: жаловаться уже есть на что, а правил
                    # для ответа без источников RAG-слой в этом случае не даёт.
                    if ctx.notice:
                        yield format_sse("notice", {"message": ctx.notice})
                    elif ctx.system_message is not None:
                        # Системный турн ставится ВСЕГДА, когда он есть, даже без
                        # `user_message`: у `smalltalk`/`clarify` источников нет,
                        # но правила («не выдавай факты из собственных знаний»)
                        # обязаны доехать до модели — иначе ответ приходит из
                        # параметрической памяти и внешне неотличим от нормального.
                        # Любой пришедший system отбрасывается ради предсказуемости.
                        prior = [m for m in outgoing if m.get("role") != "system"]
                        if ctx.user_message is None:
                            # Заменить последний вопрос нечем — история идёт целиком.
                            send = [ctx.system_message, *prior]
                        else:
                            rag_used = True
                            # Тот самый текст, который увидела модель. Метрики eval
                            # считаются по нему, а не по пересобранным из метаданных
                            # секциям — иначе оценка систематически смещена.
                            context_text = _rendered_context(
                                ctx.user_message, context_chars
                            )
                            # The manifest is fast but ephemeral; frontmatter in
                            # the vault is the durable fallback after a pod
                            # replacement. Mutation is intentional so persisted
                            # chat history keeps the resolved links.
                            await _enrich_source_urls(sources, url_index, cv)
                            yield format_sse(
                                "sources",
                                {"sources": sources, "context_chars": context_chars},
                            )
                            # [rules-only system] + [история без последнего вопроса]
                            # + [user-сообщение с источниками и тем же вопросом].
                            send = [
                                ctx.system_message,
                                *_without_last_user_turn(prior),
                                ctx.user_message,
                            ]

                # Урезаем ТОЛЬКО то, что уходит в модель: `outgoing`/`user_messages`
                # (RAG-эвристика и persistence) остаются нетронутыми.
                send, dropped = _fit_to_context(send, giga_dict)
                if dropped:
                    log.info(
                        "chat %s: history trimmed, dropped %d message(s)",
                        chat_id,
                        dropped,
                    )

                stream_started = time.perf_counter()
                async for delta in llm.stream_chat(send, gcfg):
                    if "first_token" not in stages:
                        stages["first_token"] = _elapsed_ms(stream_started)
                    full_text += delta
                    yield format_sse("token", {"text": delta})
                stages["stream"] = _elapsed_ms(stream_started)

                # Пустой ответ при `finish_reason=length`: модель потратила весь
                # бюджет вывода до первого символа (режим рассуждений считает в
                # тот же max_tokens). Наблюдалось на проде — четыре пустых пузыря.
                # Один повтор с удвоенным бюджетом; настройки не трогаем, это
                # копия конфига на один запрос. Если и он пуст — честное
                # уведомление вместо пустого пузыря.
                if (
                    getattr(answer_cfg, "last_finish_reason", None) == "length"
                    and not full_text.strip()
                ):
                    retry_cfg = _retry_config(gcfg, giga_dict)
                    if retry_cfg is not None:
                        answer_retries = 1
                        answer_retry_reason = _EMPTY_LENGTH_RETRY_REASON
                        log.warning(
                            "chat %s: пустой ответ при finish_reason=length "
                            "(max_tokens=%s) — повтор с max_tokens=%s",
                            chat_id,
                            gcfg.max_tokens,
                            retry_cfg.max_tokens,
                        )
                        retry_started = time.perf_counter()
                        answer_cfg = retry_cfg
                        async for delta in llm.stream_chat(send, retry_cfg):
                            full_text += delta
                            yield format_sse("token", {"text": delta})
                        stages["retry"] = _elapsed_ms(retry_started)
                    if not full_text.strip():
                        notice = _EMPTY_COMPLETION_NOTICE
                        yield format_sse("notice", {"message": notice})

                # Оговорка по концентрации доказательств (шаг 2в): вопрос был
                # про базу целиком, а все выбранные фрагменты пришли из одного
                # документа. Дописывается ПОСЛЕ ответа — она его уточняет, а не
                # подменяет, и никогда не подменяет отказ (у отказа источников
                # нет, значит и оговорки нет). Ноль токенов: чистый Python,
                # текст уезжает тем же кадром `token`, что и обычный дельта.
                if hedge and full_text.strip():
                    full_text += f"\n\n{hedge}"
                    yield format_sse("token", {"text": f"\n\n{hedge}"})

                # Серверная валидация цитат: номера вне 1..len(sources) — признак
                # галлюцинации. Значение уезжает и в assistant-сообщение (history),
                # и в JSONL-лог запросов.
                invalid_citations = _invalid_citations(full_text, len(sources))
                malformed = _malformed_citations(full_text)
                if malformed:
                    log.warning(
                        "chat %s: маркеры цитат без номера %s — не кликаются и "
                        "ничего не подтверждают",
                        chat_id,
                        malformed,
                    )
                if invalid_citations:
                    log.warning(
                        "chat %s: ответ ссылается на несуществующие источники %s "
                        "(в контексте их %d)",
                        chat_id,
                        invalid_citations,
                        len(sources),
                    )

                finish_reason = getattr(answer_cfg, "last_finish_reason", None)
                yield format_sse(
                    "done", {"chat_id": chat_id, "finish_reason": finish_reason}
                )

            except GigaChatError as exc:
                errored = True
                # `exc`, not `exc.message`: the detail is where the cause is.
                log.warning("chat error [%s]: %s", exc.code, exc)
                yield sse_error(exc.code, exc.message, exc.detail)
            except asyncio.CancelledError:
                truncated = True
                raise
            except Exception as exc:  # noqa: BLE001 — last-resort terminal error frame
                errored = True
                log.exception("unexpected chat error")
                yield sse_error("CHAT_FAILED", "Внутренняя ошибка чата", str(exc))
            finally:
                assistant = {
                    "role": "assistant",
                    "content": full_text,
                    "rag": rag_used,
                    "sources": sources if rag_used else [],
                    "context_chars": context_chars if rag_used else 0,
                    "truncated": truncated,
                    "invalid_citations": invalid_citations,
                }
                # Persist even partial/errored turns so the user keeps their history.
                if full_text or not errored:
                    try:
                        history.save_chat(chat_id, [*user_messages, assistant], paths)
                    except Exception:  # noqa: BLE001
                        log.exception("failed to persist chat %s", chat_id)

                stages["total"] = _elapsed_ms(turn_started)
                answer_text, answer_cut = rag_log.truncate(full_text)
                context_logged, context_cut = rag_log.truncate(context_text)

                # Лог качества RAG (волна 5.1). Пишется в `finally`, поэтому даже
                # оборванный или ошибочный ответ оставляет запись. `rag_log.append`
                # никогда не бросает — чат не зависит от телеметрии.
                rag_log.append(
                    paths,
                    {
                        "type": "request",
                        "ts": rag_log.now_iso(),
                        "chat_id": chat_id,
                        "message_index": message_index,
                        # Волна 2: классификатор намерения + переписывание вопроса
                        # в самодостаточный. Вне RAG-режима остаются `None`.
                        "intent": intent,
                        "question_raw": question_raw,
                        "question_standalone": question_standalone,
                        # Шаг 2: охват вопроса ("document"/"corpus") и текст
                        # оговорки, если она была показана. Без этой пары долю
                        # ложных оговорок на живом стенде посчитать нечем.
                        "scope": scope,
                        "hedge": hedge,
                        # Кандидаты поиска ДО отбора ({id, path, chunk_index,
                        # score, rank}) и оценки батч-грейдера ({id, path,
                        # chunk_index, score}) — вход и выход волны-2 реранкинга.
                        # `id` общий: grades[m].id == candidates[k].id ⇒ тот же чанк.
                        "candidates": _linked_candidates(candidates, grades),
                        "grades": grades,
                        "sources": _log_sources(sources, candidates),
                        # Что именно увидела модель и что она ответила. Без этой
                        # пары правило диагностики («чанк был в контексте, но
                        # ответ неверен») проверить нечем.
                        "context_text": context_logged,
                        "context_chars": context_chars,
                        "context_truncated_in_log": context_cut,
                        # Структурный блок в голове того же сообщения — только
                        # форма ({kind, chars}), не текст: он одинаков на всех
                        # ходах прогона и весит тысячи символов. На ходе поиска
                        # (`rag_used: true`, `intent != "meta"`) `kind: null`
                        # означает, что блока не было вовсе — то есть тихую
                        # деградацию (каталог вольта недоступен), которую по
                        # остальному логу не увидеть: `context_text` режется от
                        # «Источники:» и блок в него не попадает. Мета-ход
                        # собирает структуру другим швом (`rag._structure_block`)
                        # и в этом поле не отражается.
                        "head_block": rag_log.head_block_snapshot(
                            head_block_kind, head_block_chars
                        ),
                        "answer_text": answer_text,
                        "answer_chars": len(full_text),
                        "answer_truncated_in_log": answer_cut,
                        # `length` здесь = ответ обрезан лимитом модели; раньше
                        # вычислялось и выбрасывалось, обрыв был невидим.
                        "finish_reason": finish_reason,
                        # То же значение под именем, которое не спутать с
                        # `finish_reason` скрытых вызовов в `hidden_calls`.
                        "answer_finish_reason": finish_reason,
                        # Пустой ответ без ошибки транспорта и без обрыва
                        # клиентом: модель отработала и не выдала ни символа
                        # (типично — `finish_reason: length`, бюджет ушёл на
                        # рассуждения). По `answer_chars: 0` это не отличить от
                        # оборванного соединения.
                        "empty_answer": (
                            not errored and not truncated and not full_text.strip()
                        ),
                        # Повтор генерации: сколько раз и почему (`empty_length`
                        # — пустой ответ при finish_reason=length). Запись одна
                        # на ход: повтор идёт внутри того же генератора.
                        "answer_retries": answer_retries,
                        "answer_retry_reason": answer_retry_reason,
                        # Что сделали скрытые вызовы: condense и грейдер по
                        # батчам — статус, ошибка, finish_reason, usage, модель.
                        # `None` вне RAG-режима; внутри — словарь с двумя
                        # ключами, каждый из которых `None`, если шаг не дошёл.
                        "hidden_calls": hidden_calls,
                        "invalid_citations": invalid_citations,
                        "rag_used": rag_used,
                        "notice": notice,
                        "truncated": truncated,
                        "errored": errored,
                        # Действующие настройки хода — что должен повторить
                        # следующий прогон, чтобы дельта что-то значила.
                        "settings": rag_log.settings_snapshot(rcfg, giga_dict, prompts),
                        "timings_ms": dict(stages),
                    },
                )

    return StreamingResponse(
        generator(), media_type="text/event-stream", headers=_SSE_HEADERS
    )
