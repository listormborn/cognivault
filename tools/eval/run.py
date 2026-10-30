"""Eval-harness (план 5.3): golden-set → живой стек → метрики → отчёт.

Что делает прогон:

1. читает ``golden.jsonl`` (по умолчанию берутся все пары, кроме
   ``accepted: false`` — то есть непровалидированные ``null`` тоже идут в дело);
2. каждый вопрос уходит в UI-API ``POST /api/chat`` с ``rag: true``; SSE-поток
   разбирается в ответ + список ``sources``;
3. контекст для метрик берётся из ``rag_log.jsonl`` UI — там лежит **ровно тот**
   блок «Источники», который видела модель (``context_text``), плюс
   ``chunk_index`` каждого источника и снимок настроек прогона. Если лог
   недоступен, включается фолбэк: текст восстанавливается из метаданных через
   ``GET /api/vault/content``, и весь прогон помечается ПРИБЛИЖЁННЫМ;
4. считаются четыре судейские метрики (``metrics.py``); упавшие сэмплы
   в средние НЕ попадают и выносятся в отчёт отдельной строкой; пары-ловушки
   ``expected_refusal`` тоже вынесены — их средние живут в
   ``aggregate_refusal``, а их ветку меряют ``refusal_ok`` и обратный к нему
   ``false_refusal_rate``; третья корзина — метапары
   (``expected_outcome: "meta"``, вопрос про саму базу или про ассистента):
   средние в ``aggregate_meta``, ветку меряет ``meta_answered_rate``, и отказ
   на такой паре считается ПРОВАЛОМ, а не успехом;
5. результаты режутся по ``category`` golden-пары (``by_category``);
6. пишутся ``report-<label>.json`` и ``report-<label>.md``.

Сравнение прогонов::

    python3 tools/eval/run.py --label baseline
    python3 tools/eval/run.py --label wave-3
    python3 tools/eval/run.py --compare reports/report-baseline.json \\
                                        reports/report-wave-3.json

Абсолютные значения судьи не показательны — смысл только в дельте A/B, и
``--compare`` считает её **парно** (по одним и тем же вопросам) с разбросом и
числом пар, чтобы отличать сигнал от шума судьи.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import httpx

import gen_golden as gen_golden_mod
import metrics as metrics_mod
from gen_golden import BackendClient, BackendError, resolve_backend
from gigachat_client import GigaChatEvalError, GigaChatJudge, JudgeConfig
from metrics import METRIC_NAMES, aggregate, coverage, evaluate_sample

#: Печатается в шапке каждого отчёта — судья врёт в абсолютных числах.
REPORT_DISCLAIMER = (
    "**Абсолютным значениям метрик доверять нельзя.** Их выставляет та же "
    "LLM (GigaChat), что генерирует ответы, по судейским промптам: сдвиг "
    "калибровки, чувствительность к формулировке промпта и разброс между "
    "вызовами делают «0.78 faithfulness» числом без самостоятельного смысла. "
    "Осмысленна только **дельта между прогонами** этого же харнесса на том же "
    "golden-set и той же версии промптов судьи "
    "(`prompt_version`). Сравнивайте `--compare A.json B.json`, а не абсолют."
)

DIAGNOSTIC_RULE = (
    "**Правило диагностики (план, критерий Волны 5):** если нужный чанк был "
    "в контексте, а ответ неверен — чинить генерацию (промпт, порядок блоков, "
    "модель); если нужного чанка в контексте не было — чинить ретрив "
    "(поиск, гибрид, реранкер). За это отвечает метрика `retrieval_hit`: попала "
    "ли пара `(path, chunk_index)` из golden-пары в выданные источники "
    "(гранулярность падает до раздела/файла, если в golden-паре чанк не указан "
    "— см. `retrieval_granularity`)."
)

APPROXIMATE_WARNING = (
    "**ПРИБЛИЖЁННЫЙ ПРОГОН.** Текст контекста для части пар восстановлен из "
    "метаданных (`path`/`section_path`), а не взят из `rag_log.jsonl`. "
    "Восстановление смещает метрики В ОБЕ СТОРОНЫ: для `depth=\"chunk\"` в "
    "судью уходит целая секция (метрики завышаются), для `depth=\"file\"` — "
    "первые N символов файла (занижаются). Сдвиг зависит от состава `depth`, "
    "поэтому A/B-дельта между прогоном по логу и прогоном по метаданным "
    "недостоверна. Дайте харнессу `--rag-log <путь к rag_log.jsonl>`."
)

#: Уровни `retrieval_granularity`, на которых попадание засчитывается ГРУБЕЕ,
#: чем по паре ``(path, chunk_index)``: любой чанк того же раздела (или файла)
#: считается попаданием.
DEGRADED_GRANULARITIES = ("section", "file")

#: Печатается, когда хотя бы одна пара мерилась не на уровне чанка. Раньше это
#: было видно только по словарю `retrieval_granularity` в шапке — то есть
#: молча: число `retrieval_hit` выглядело точным, будучи завышенным.
GRANULARITY_WARNING = (
    "**`retrieval_hit` измерен НЕ на уровне чанка: {degraded} из {measured} "
    "пар.** У этих пар в golden-set нет `source_chunk_index`, поэтому "
    "попаданием считался любой чанк нужного раздела (`section`) или файла "
    "(`file`) — метрика ЗАВЫШЕНА относительно честного сравнения по паре "
    "`(path, chunk_index)`, и её нельзя сравнивать с прогоном, где чанк "
    "известен. Это ограничение генератора, а не сбой: `gen_golden.py` режет "
    "корпус собственным упрощённым сплиттером (H1–H3 + кап по символам), а "
    "бэкенд нумерует чанки по-своему (короткие секции сливаются, длинные "
    "режутся по бюджету токенов, таблицы — построчно, table-summary "
    "дописывается в хвост массива). Проставленный «на глаз» индекс давал бы "
    "ЛОЖНЫЕ ПРОМАХИ, что хуже честного огрубления, поэтому генератор пишет "
    "`null`. Поднять точность можно только вручную: проставить "
    "`source_chunk_index` в `golden.jsonl` по выдаче `/api/search`."
)

BUCKET_NOTE = (
    "**Четыре судейские метрики посчитаны ТОЛЬКО по отвечаемым парам "
    "(n={answerable}); пары-ловушки `expected_refusal` (n={refusal}) вынесены "
    "в отдельную таблицу ниже.** Иначе правильный отказ тянул бы средние вниз: "
    "судья `answer_relevancy_ru` намеренно ставит 0 уклончивому ответу, и общее "
    "число зависело бы от ДОЛИ ловушек в наборе, а не от качества. Следствие: "
    "средние двух прогонов сравнимы, только если совпадают ОБА числа — "
    "сверьте их перед `--compare`. Третья корзина — метапары "
    "`expected_outcome: meta` (n={meta}): вопрос про саму базу или про "
    "ассистента, документа-цели нет, но отказ на нём — ПРОВАЛ, а не успех; их "
    "ветку меряет `meta_answered_rate`."
)

CLIPPED_CONTEXT_WARNING = (
    "**Судья видел меньше контекста, чем модель, на {clipped} парах из "
    "{total}.** `context_text` в `rag_log.jsonl` обрезан своим капом "
    "(`rag_log.MAX_TEXT_CHARS`), и часть блоков до судьи не доехала — а ответ "
    "на них ссылается. Провалы `faithfulness` и `context_precision` на этих "
    "парах говорят про отсутствующий у судьи текст, а не про качество ответа. "
    "Лечится подъёмом капа ВЫШЕ `max_context_chars`; пары помечены "
    "`context_clipped`."
)

GRADER_SILENT_WARNING = (
    "**Грейдер молча не отработал: оценки есть только у {graded} пар(ы) из "
    "{applicable}.** Провалившийся батч деградирует во все `None`, а отбор при "
    "`None` пропускает кандидатов В СЫРОМ ПОРЯДКЕ ПОИСКА — то есть измерен "
    "пайплайн БЕЗ реранкера, хотя `грейдер: включён` в параметрах говорит "
    "обратное. Ошибки при этом нет нигде: ответы получены, метрики посчитаны. "
    "Сравнивать такой прогон с прогоном, где грейдер жив, нельзя — это разные "
    "системы. Смотрите таблицу стадий: время `grade`, совпавшее с поводком, "
    "означает таймаут, а не работу."
)

CONDENSE_NOT_CALLED = (
    "включён, но НЕ ВЫЗЫВАЛСЯ: каждый вопрос задан в новом чате, а "
    "condense_first_turn выключен"
)

JUDGE_FAILURE_WARNING = (
    "**Судья не ответил на {failed} вызов(ов) из {expected}; затронуто пар: "
    "{affected}.** Каждый несостоявшийся вызов — это прочерк в таблице, а не "
    "ноль, так что средние он не занижает: он считает их по НЕПОЛНОМУ и заведомо "
    "НЕСЛУЧАЙНОМУ подмножеству (терялись длинные и медленные пары). Колонка "
    "«Оценено пар» показывает, сколько чисел реально стоит за каждым средним; "
    "пока она заметно меньше числа пар, дельта с другим прогоном ловит только "
    "очень крупные сдвиги. Причины — в разбивке ниже; строка `HTTP 429` "
    "означает, что прогон шёл слишком параллельно для контура."
)

FALSE_REFUSAL_NOTE = (
    "`false_refusal_rate` — **меньше лучше**. Оно ловит обратную ошибку: у "
    "вопроса ответ есть, а ассистент ответил «в источниках ничего нет». Обычно "
    "вырастает от закручивания порога грейдера — того самого движения, которым "
    "«улучшают» `refusal_ok`. Второе такое число — `hedge_rate`: ответ по "
    "существу ЕСТЬ, но открывается оговоркой «прямого ответа не нашлось»; "
    "оговорка не отказ (оценка релевантности ставится содержательной части), "
    "но пользователь читает её как отказ, поэтому её доля меряется отдельно."
)

JUDGE_CLIP_WARNING = (
    "**Кап судьи: на {clipped} парах судья видел контекст, урезанный ЕГО "
    "СОБСТВЕННЫМ капом** (`metrics.format_context`, не `rag_log.MAX_TEXT_CHARS`): "
    "пары {ids}. Провалы `faithfulness`/`context_precision`/`context_recall` "
    "на них говорят про текст, которого судье не дали, а не про ответ. Это "
    "отдельный дефект от `context_clipped` (обрезка в логе) — тот считается и "
    "печатается своей строкой."
)

JUDGE_PROMPT_MISMATCH_WARNING = (
    "**ВНИМАНИЕ: СУДИЛИ РАЗНЫЕ ВЕРСИИ ПРОМПТОВ.** `{label_a}`: `{version_a}`; "
    "`{label_b}`: `{version_b}`. Оценки разных версий судьи несопоставимы по "
    "построению — дельта ниже измеряет смену судьи, а не системы. Перегоните "
    "старый прогон текущей версией; `--compare` продолжает только с "
    "`--allow-model-mismatch`."
)

#: Ключи `raw` метрики, которые не сохраняются в JSON-отчёт (см. `slim_report`).
REPORT_DROPPED_RAW_KEYS = ("replies",)

#: Хвост первоэкранного предупреждения о грейдере, когда причина сбоя ИЗВЕСТНА
#: по записям `hidden_calls`. Без причины предупреждение называло симптом
#: («молча не отработал») и ничего про диагноз.
GRADER_CAUSE_PREFIX = "Причина по батчам: "
#: …и когда записи о скрытых вызовах в логе нет вовсе.
GRADER_CAUSE_NOT_RECORDED = "Причина не записана: UI старее сбора скрытых вызовов"

#: Значение любого поля, которого в записи лога ещё не было. Печатается вместо
#: догадки: старый UI не писал `hidden_calls` / `model_effective` /
#: `empty_answer`, и отчёт обязан это сказать, а не додумать.
NOT_RECORDED = "не записано"

#: Суффикс заголовка отчёта, когда реранкер оценил меньше 90 % применимых пар.
DEGRADED_TITLE_SUFFIX = "(РЕРАНКЕР НЕ РАБОТАЛ)"
#: Ниже этой доли `graded / applicable` прогон измерял систему без реранкера.
GRADER_DEGRADED_THRESHOLD = 0.9

MODEL_MISMATCH_WARNING = (
    "**ВНИМАНИЕ: ОТВЕЧАЛИ РАЗНЫЕ МОДЕЛИ.** `{label_a}`: {model_a}; "
    "`{label_b}`: {model_b}. Дельта ниже — сравнение двух МОДЕЛЕЙ, а не двух "
    "настроек одной системы, и ни одна строка таблицы этого не покажет. "
    "`--compare` продолжает только с `--allow-model-mismatch`."
)

MODEL_UNKNOWN_WARNING = (
    "**Провайдер/модель ответа не записаны в отчёте `{label}`** (сделан до "
    "того, как UI стал писать `model_effective`) — совпадение моделей проверить "
    "нельзя; сверьте вручную."
)

DEGRADED_COMPARE_WARNING = (
    "**ВНИМАНИЕ: в прогоне `{label}` реранкер не работал** (грейдер оценил "
    "{graded} из {applicable} применимых пар, меньше {threshold:.0%}). Это "
    "другая система, а не другая настройка; дельта с ним недостоверна. "
    "`--compare` продолжает только с `--allow-degraded`."
)

GENERATION_FAILED_NOTE = (
    "**Сбои генерации: {count} пар(ы) не получили ответа вовсе** (пустой текст "
    "при `finish_reason`, обычно `length`). Это не отказ и не плохой ответ — это "
    "отсутствие ответа, и судье тут оценивать нечего: такие пары вынесены в "
    "корзину `generation_failed`, не входят ни в одно среднее и перечислены в "
    "секции «Сбои генерации». `retrieval_hit` по ним посчитан — ретрив состоялся."
)

PATH_DRIFT_WARNING = (
    "**Дрейф путей golden-set: {drifted} пар(ы) сопоставлены по имени файла, "
    "{ambiguous} неоднозначны, {missing} не найдены в живом каталоге.** "
    "Golden-пара ссылается на `source_path`, которого в индексе нет под этим "
    "путём (страницу перенесли или переименовали папку). Сопоставленные по "
    "имени файла пути добавлены в `alt_source_paths` пары на время прогона, и "
    "`retrieval_hit` по ним честный; неоднозначные и отсутствующие остались "
    "как есть — их промах может быть промахом РАЗМЕТКИ, а не ретрива. Список — "
    "в секции «Дрейф путей golden»; поправьте `golden.jsonl`."
)

#: Пер-сэмпловый флаг корзины «генерация не дала ответа».
GENERATION_FAILED_KEY = "generation_failed"

RETRIEVAL_KEY = "retrieval_hit"
REFUSAL_KEY = "refusal_ok"
#: Пер-сэмпловый флаг: отвечаемый вопрос, на который ассистент зря отказался
#: отвечать (``None`` у пар, где отказ и ожидался — там мерить нечего).
FALSE_REFUSAL_KEY = "false_refusal"
#: Его доля в агрегатах. МЕНЬШЕ — ЛУЧШЕ, в отличие от всех остальных чисел отчёта.
FALSE_REFUSAL_RATE_KEY = "false_refusal_rate"
#: Пер-сэмпловый флаг третьей корзины: метавопрос получил содержательный ответ,
#: а не отказ (``None`` у всех остальных пар).
META_KEY = "meta_answered"
#: Его доля в агрегатах.
META_RATE_KEY = "meta_answered_rate"
#: Доля отвечаемых пар, где ответ есть, но открыт оговоркой об отсутствии
#: ответа (`metrics.answer_relevancy_ru.hedged`). МЕНЬШЕ — ЛУЧШЕ.
HEDGE_RATE_KEY = "hedge_rate"
#: Пер-сэмпловый флаг: судья видел контекст, урезанный СВОИМ капом.
JUDGE_CLIP_KEY = "judge_context_clipped"

#: Поле golden-пары с ТРЕХЗНАЧНЫМ вердиктом и его значения.
OUTCOME_KEY = "expected_outcome"
OUTCOME_ANSWER = "answer"
OUTCOME_REFUSAL = "refusal"
OUTCOME_META = "meta"
OUTCOMES = (OUTCOME_ANSWER, OUTCOME_REFUSAL, OUTCOME_META)


def expected_outcome(row: dict[str, Any]) -> str:
    """Чего мы ждём от ассистента на этой паре: ответа, отказа или метаответа.

    Три исхода вместо двух заведены 2026-08-01. Двоичное «ответил / отказался»
    не описывало вопрос ПРО САМУ БАЗУ («что ты знаешь?», «о каких продуктах есть
    информация?», «всегда ли ответ в Markdown?»): документа-цели у него нет, но и
    отказ на нём — не правильный ответ, а ровно тот дефект, который чинится.
    Такая пара была вынуждена лежать в ловушках и НАГРАЖДАТЬ отказ.

    ``expected_refusal`` при этом НЕ переопределён: он остаётся флагом ловушки, и
    все ловушечные метрики (`refusal_ok`, `aggregate_refusal`, разрезы стыка 3)
    считают ровно то же, что считали. Строка без поля — старый golden-набор:
    вердикт выводится из ``expected_refusal``, то есть поведение не меняется.
    """
    raw = str(row.get(OUTCOME_KEY, "") or "").strip()
    if raw in OUTCOMES:
        return raw
    return OUTCOME_REFUSAL if row.get("expected_refusal") else OUTCOME_ANSWER

#: Поле golden-пары с ручной категорией вопроса и заглушка для пар без неё.
CATEGORY_KEY = "category"
UNCATEGORIZED = "unclassified"


def category_of(row: dict[str, Any]) -> str:
    """Категория golden-пары (или сэмпла отчёта); пустая → ``unclassified``.

    Поле необязательное: golden-set, сгенерированный `gen_golden.py`, его не
    знает вовсе, и такие строки обязаны продолжать работать — они просто
    сходятся в одну корзину.
    """
    return str(row.get(CATEGORY_KEY, "") or "").strip() or UNCATEGORIZED

#: Формулировки отказа («в источниках ответа нет») — ветка, которую меряют
#: golden-пары с ``expected_refusal``. Держать в согласии с `rag.SYSTEM_PROMPT`.
_REFUSAL_PATTERNS = (
    r"ответа\s+на\s+этот\s+вопрос\s+не\s+нашлось",
    r"в\s+доступных\s+мне\s+документах",
    r"в\s+базе\s+знаний\s+нет\s+данных",
    r"в\s+источниках\s+(?:нет|отсутству)",
    r"не\s+нашлось\s+ответа",
    r"информаци\w+\s+(?:нет|не\s+найдено)",
    r"не\s+наш[её]л\s+информаци\w+",
)
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(answer: str, *, finish_reason: str | None = None) -> bool:
    """Ответил ли ассистент отказом «в источниках этого нет».

    Две улики: служебный ``finish_reason == "no_context"`` (грейдер не оставил
    ни одного фрагмента — генерации не было вовсе) и формулировка отказа из
    системного промпта В ПЕРВОМ ПРЕДЛОЖЕНИИ.

    Привязка к первому предложению обязательна. Те же слова в середине —
    честная оговорка о неполноте («ID потока — 4832. В источниках нет явного
    описания его назначения, однако…»), и поиск по всему тексту засчитывал такой
    содержательный ответ как полный отказ: в прогоне `baseline` так был потерян
    x35, то есть треть всей метрики `false_refusal_rate`. Отказ по системному
    промпту всегда стоит первой фразой, поэтому окно ничего не теряет.
    """
    if finish_reason == "no_context":
        return True
    return bool(_REFUSAL_RE.search(_opening_sentence(answer)))


def _opening_sentence(answer: str) -> str:
    """Первое предложение ответа (весь текст, если сегментация ничего не дала)."""
    text = (answer or "").strip()
    if not text:
        return ""
    sentences = metrics_mod.split_sentences_ru(text)
    return sentences[0] if sentences else text


# --------------------------------------------------------------------------- #
# SSE parsing (pure)
# --------------------------------------------------------------------------- #


class SSEDecoder:
    """Incremental SSE decoder: feed lines, get ``(event, data)`` tuples.

    Tolerant on purpose — an unknown/extra field is ignored and a ``data:``
    payload that is not JSON is surfaced as ``{"raw": "<text>"}`` rather than
    raising, so a malformed frame cannot abort a whole eval run.
    """

    def __init__(self) -> None:
        self._event: str | None = None
        self._data: list[str] = []

    def push(self, line: str) -> list[tuple[str, dict[str, Any]]]:
        """Feed one line (without trailing newline); return completed events."""
        stripped = line.rstrip("\r")
        if stripped == "":
            return self._flush()
        if stripped.startswith(":"):
            return []
        if stripped.startswith("event:"):
            self._event = stripped[len("event:") :].strip()
            return []
        if stripped.startswith("data:"):
            self._data.append(stripped[len("data:") :].lstrip())
            return []
        return []

    def close(self) -> list[tuple[str, dict[str, Any]]]:
        """Flush a trailing frame that was not terminated by a blank line."""
        return self._flush()

    def _flush(self) -> list[tuple[str, dict[str, Any]]]:
        if self._event is None and not self._data:
            return []
        payload = "\n".join(self._data)
        data: dict[str, Any]
        try:
            parsed = json.loads(payload) if payload else {}
            data = parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            data = {"raw": payload}
        event = self._event or "message"
        self._event = None
        self._data = []
        return [(event, data)]


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse a whole SSE body into ``[(event, data), ...]``."""
    decoder = SSEDecoder()
    out: list[tuple[str, dict[str, Any]]] = []
    for line in text.split("\n"):
        out.extend(decoder.push(line))
    out.extend(decoder.close())
    return out


@dataclass
class ChatOutcome:
    """Everything the harness needs from one ``/api/chat`` stream."""

    answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    chat_id: str = ""
    notice: str = ""
    finish_reason: str | None = None
    error: str = ""
    order: list[str] = field(default_factory=list)


def collect_chat(events: Iterable[tuple[str, dict[str, Any]]]) -> ChatOutcome:
    """Fold an SSE event stream into a :class:`ChatOutcome`.

    Reads the ``sources`` payload leniently (``.get`` only) — the contract is
    being extended (``grade``, ``url``, …) by other waves and must not break
    the harness.
    """
    outcome = ChatOutcome()
    for event, data in events:
        outcome.order.append(event)
        if event == "meta":
            outcome.chat_id = str(data.get("chat_id", "") or "")
        elif event == "sources":
            raw = data.get("sources")
            if isinstance(raw, list):
                outcome.sources = [s for s in raw if isinstance(s, dict)]
        elif event == "notice":
            outcome.notice = str(data.get("message", "") or "")
        elif event == "token":
            outcome.answer += str(data.get("text", "") or "")
        elif event == "done":
            outcome.finish_reason = data.get("finish_reason")
        elif event == "error":
            code = str(data.get("code", "") or "")
            message = str(data.get("message", "") or "")
            outcome.error = f"{code}: {message}".strip(": ")
    return outcome


# --------------------------------------------------------------------------- #
# Golden set
# --------------------------------------------------------------------------- #


def load_golden(path: str, *, include_rejected: bool = False) -> list[dict[str, Any]]:
    """Read golden.jsonl, dropping ``accepted: false`` rows unless asked."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: не JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: строка не объект")
            if not include_rejected and row.get("accepted") is False:
                continue
            rows.append(row)
    for ids, question in duplicate_questions(rows):
        _log(f"golden: дубль вопроса {ids} — {question[:70]!r}")
    return rows


def duplicate_questions(rows: Sequence[dict[str, Any]]) -> list[tuple[list[str], str]]:
    """Пары с одним и тем же вопросом — считаются дважды и портят агрегаты.

    Найдено на живом наборе: `x24` ≡ `fb31` и `x29` ≡ `fb18` дали побайтово
    одинаковые ответы, то есть одно наблюдение попало в средние и в разброс
    два раза. Набор от этого не падает — решать, какую из пар оставить, должен
    человек (у `fb*` есть `expected_items`, у `x*` — `source_path`), поэтому
    здесь только предупреждение.
    """
    seen: dict[str, list[str]] = {}
    order: list[str] = []
    for row in rows:
        key = " ".join(str(row.get("question") or "").lower().split()).rstrip("?!. ")
        if not key:
            continue
        if key not in seen:
            seen[key] = []
            order.append(key)
        seen[key].append(str(row.get("id") or "?"))
    return [(seen[k], k) for k in order if len(seen[k]) > 1]


# --------------------------------------------------------------------------- #
# Chat + context retrieval
# --------------------------------------------------------------------------- #


class ChatClient:
    """Thin SSE client for the UI's ``POST /api/chat`` (server mode: Bearer)."""

    def __init__(
        self,
        base_url: str,
        token: str = "",
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 300.0,
    ) -> None:
        self._base = (base_url or "").rstrip("/")
        headers = {"Accept": "text/event-stream"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(
            headers=headers,
            transport=transport,
            timeout=httpx.Timeout(connect=15.0, read=timeout, write=30.0, pool=15.0),
        )

    async def __aenter__(self) -> "ChatClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ask(self, question: str, *, chat_id: str = "") -> ChatOutcome:
        """Ask one RAG question and fold the SSE stream into a outcome."""
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": question}],
            "rag": True,
        }
        if chat_id:
            payload["chat_id"] = chat_id
        # Cyrillic goes over the wire raw, not as \uXXXX escapes.
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}

        decoder = SSEDecoder()
        events: list[tuple[str, dict[str, Any]]] = []
        async with self._client.stream(
            "POST", f"{self._base}/api/chat", content=body, headers=headers
        ) as resp:
            if resp.status_code != 200:
                raw = await resp.aread()
                detail = raw[:500].decode("utf-8", errors="replace")
                outcome = ChatOutcome()
                outcome.error = f"HTTP {resp.status_code}: {detail}"
                return outcome
            async for line in resp.aiter_lines():
                events.extend(decoder.push(line))
        events.extend(decoder.close())
        return collect_chat(events)


def slice_section(content: str, section_path: str, cap: int = 4000) -> str | None:
    """Slice the section named by the tail of ``section_path`` out of ``content``.

    Mirrors the UI's own section slicing closely enough for judging: find the
    heading whose text equals the last ``>``-segment, keep everything up to the
    next heading of the same or higher level, cap the result.
    """
    tail = section_path.split(">")[-1].strip().lower() if section_path else ""
    if not tail or not content:
        return None
    lines = content.splitlines()
    start = -1
    start_level = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        level = len(stripped) - len(stripped.lstrip("#"))
        title = stripped.lstrip("#").strip().lower()
        if title == tail:
            start = index
            start_level = level
            break
    if start < 0:
        return None
    out: list[str] = [lines[start]]
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            if level <= start_level:
                break
        out.append(line)
    text = "\n".join(out).strip()
    return text[:cap] if text else None


# --------------------------------------------------------------------------- #
# Context: the real thing from rag_log.jsonl, or an approximate rebuild
# --------------------------------------------------------------------------- #

#: Заголовок блока контекста (`rag._header`): `### Источник N: title — path…`.
_BLOCK_RE = re.compile(r"^###\s+Источник\s+\d+\s*:", re.MULTILINE)


def split_context_blocks(context_text: str) -> list[str]:
    """Разрезать отрендеренный блок «Источники» на отдельные фрагменты.

    Судейские метрики оценивают фрагменты по отдельности (`context_precision`
    считает долю релевантных), поэтому монолитный блок нужно вернуть в список.
    Разрез идёт по заголовкам `### Источник N:`; если их нет (пустой контекст
    или чужой рендер), возвращается один элемент.
    """
    text = (context_text or "").strip()
    if not text:
        return []
    bounds = [m.start() for m in _BLOCK_RE.finditer(text)]
    if not bounds:
        return [text]
    bounds.append(len(text))
    out: list[str] = []
    if bounds[0] > 0:  # преамбула до первого заголовка — не теряем
        head = text[: bounds[0]].strip()
        if head:
            out.append(head)
    for start, end in zip(bounds, bounds[1:]):
        block = text[start:end].strip()
        if block:
            out.append(block)
    return out


class RagLogIndex:
    """Записи ``rag_log.jsonl`` типа ``request``, разложенные по ``chat_id``.

    Лог — единственное место, где сохранён **фактический** контекст хода. Он
    же несёт ``chunk_index`` источников и снимок настроек, поэтому отчёт умеет
    зафиксировать параметры прогона.
    """

    def __init__(
        self, records: Iterable[dict[str, Any]] = (), path: str | None = None
    ) -> None:
        self._by_chat: dict[str, dict[str, Any]] = {}
        self._path = path
        self._mtime: float | None = None
        self.absorb(records)

    def absorb(self, records: Iterable[dict[str, Any]]) -> None:
        for record in records:
            if not isinstance(record, dict) or record.get("type") != "request":
                continue
            chat_id = str(record.get("chat_id", "") or "")
            if chat_id:
                self._by_chat[chat_id] = record

    def __len__(self) -> int:
        return len(self._by_chat)

    def get(self, chat_id: str) -> dict[str, Any] | None:
        """Запись хода; при промахе перечитывает файл (лог растёт по ходу прогона).

        Запись появляется в логе только когда ход ДОСЕЛЕ закончился, а харнесс
        спрашивает её сразу после стрима — поэтому снимок, снятый при старте,
        всегда пуст. Перечитываем при промахе, но только если файл изменился.
        """
        if not chat_id:
            return None
        record = self._by_chat.get(chat_id)
        if record is None and self.refresh():
            record = self._by_chat.get(chat_id)
        return record

    def refresh(self) -> bool:
        """Перечитать файл, если он изменился. ``True`` — что-то перечитали."""
        if not self._path:
            return False
        try:
            mtime = os.path.getmtime(self._path)
        except OSError:
            return False
        if self._mtime is not None and mtime <= self._mtime:
            return False
        self._mtime = mtime
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                self.absorb(parse_jsonl(fh.read()))
        except OSError:
            return False
        return True

    @classmethod
    def from_text(cls, text: str) -> "RagLogIndex":
        return cls(parse_jsonl(text))

    @classmethod
    def load(cls, path: str) -> "RagLogIndex | None":
        """Открыть лог; ``None`` — файла нет (значит, будет фолбэк)."""
        if not os.path.exists(path):
            return None
        index = cls(path=path)
        index.refresh()
        return index


def parse_jsonl(text: str) -> list[dict[str, Any]]:
    """Разобрать JSONL, молча пропуская битые строки (последняя может рваться)."""
    out: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
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


@dataclass
class ResolvedContext:
    """Контекст одного сэмпла + откуда он взялся."""

    contexts: list[str] = field(default_factory=list)
    origin: str = "none"  # rag_log | metadata | none
    sources: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    @property
    def approximate(self) -> bool:
        """``True`` — контекст восстановлен, а не взят из лога."""
        return self.origin != "rag_log"


def context_from_log(record: dict[str, Any]) -> ResolvedContext | None:
    """Достать фактический контекст хода из записи лога.

    ``None`` — в записи нет ``context_text`` (старый UI): пусть решает фолбэк.
    Пустой контекст при ``rag_used`` — это НЕ ошибка: значит, ретрив честно
    ничего не дал, и метрики должны это увидеть.
    """
    if "context_text" not in record:
        return None
    sources = [s for s in (record.get("sources") or []) if isinstance(s, dict)]
    if record.get("context_truncated_in_log"):
        _log(f"  ! контекст хода {record.get('chat_id')} обрезан в логе")
    return ResolvedContext(
        contexts=split_context_blocks(str(record.get("context_text", "") or "")),
        origin="rag_log",
        sources=sources,
    )


async def rebuild_contexts(
    backend: BackendClient | None,
    sources: Sequence[dict[str, Any]],
    *,
    cache: dict[str, str],
    cap: int = 4000,
) -> ResolvedContext:
    """ФОЛБЭК: восстановить текст источников из метаданных.

    Работает только когда ``rag_log.jsonl`` недоступен. Section-level источники
    вырезаются из документа, file-level режутся по ``cap`` — и то и другое
    ЗАМЕТНО расходится с тем, что видела модель (см. :data:`APPROXIMATE_WARNING`),
    поэтому такой прогон помечается приближённым.

    Ошибка бэкенда больше не «деградирует до заголовка»: раньше это давало
    судье пустой контекст, нули уезжали в среднее и читались как регрессия.
    Теперь сэмпл получает ``error`` и выбывает из агрегатов.
    """
    resolved = ResolvedContext(origin="metadata", sources=list(sources))
    failures: list[str] = []
    for source in sources:
        path = str(source.get("path", "") or "")
        section_path = str(source.get("section_path", "") or "")
        header = " > ".join(x for x in (path, section_path) if x)
        if not path or backend is None:
            failures.append(f"{path or '(без пути)'}: текст недоступен")
            continue
        if path not in cache:
            try:
                cache[path] = await backend.content(path)
            except (BackendError, httpx.HTTPError) as exc:
                cache[path] = ""
                _log(f"  ! контекст {path}: {exc}")
        content = cache.get(path, "")
        if not content:
            failures.append(f"{path}: пустой ответ бэкенда")
            continue
        sliced = slice_section(content, section_path, cap) if section_path else None
        text = sliced or content[:cap]
        resolved.contexts.append(f"{header}\n{text}".strip())
    if failures:
        resolved.error = "context: " + "; ".join(failures[:5])
    return resolved


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


# Куда уходит прогресс вместо stderr. UI-раннер (`app/eval_runner.py`) ставит
# сюда свой приёмник на время прогона и снимает после: перехватывать stderr
# всего процесса он больше не может — туда же пишет логгер приложения, и на
# время прогона предупреждения грейдера пропадали из лога пода. Тип — `Any`,
# а не `Callable`, чтобы не трогать строку импортов: `Callable[[str], None] | None`.
LOG_SINK: Any = None


def _log(message: str) -> None:
    sink = LOG_SINK
    if sink is not None:
        sink(message)
        return
    print(message, file=sys.stderr, flush=True)


def alt_source_paths(row: dict[str, Any]) -> list[str]:
    """Альтернативные источники golden-пары (``alt_source_paths``), очищенные.

    Поле необязательное: строки без него (или со старым форматом набора)
    обязаны работать как раньше. Не-списки и пустые/нестроковые элементы
    молча отбрасываются — битая разметка не должна ронять прогон.
    """
    raw = row.get("alt_source_paths")
    if not isinstance(raw, list):
        return []
    return [p.strip() for p in raw if isinstance(p, str) and p.strip()]


# --------------------------------------------------------------------------- #
# Дрейф путей golden-набора относительно живого каталога
# --------------------------------------------------------------------------- #


def _basename(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1]


def resolve_golden_paths(
    row: dict[str, Any], live_paths: Sequence[str] | set[str] | None
) -> dict[str, Any]:
    """Сверить `source_path` и `alt_source_paths` пары с живым каталогом.

    Страницу переносят, папку переименовывают — и golden-пара начинает
    ссылаться на путь, которого в индексе нет. Раньше это читалось как промах
    ретрива: `retrieval_hit == false` при том, что нужный документ лежал в
    контексте под новым путём. Правило:

    * путь есть в каталоге — ничего не меняется;
    * пути нет, но ровно ОДИН живой путь с тем же именем файла — он добавляется
      в эффективные `alt_source_paths` и записывается как `path_drift`;
    * кандидатов несколько — `path_ambiguous`, ничего не подставляется:
      угадывать между реестрами-близнецами хуже честного промаха;
    * кандидатов нет — `path_missing`.

    ``live_paths is None`` — каталог недоступен, проверка не проводилась и
    пара остаётся ровно такой, какой была.
    """
    alts = alt_source_paths(row)
    out: dict[str, Any] = {
        "checked": live_paths is not None,
        "effective_alt_source_paths": list(alts),
        "path_drift": None,
        "path_ambiguous": [],
        "path_missing": False,
        "alt_path_drift": [],
        "alt_path_ambiguous": [],
        "alt_path_missing": [],
    }
    if live_paths is None:
        return out
    live = set(live_paths)
    by_name: dict[str, list[str]] = {}
    for path in live:
        by_name.setdefault(_basename(path), []).append(path)

    def lookup(path: str) -> tuple[str, list[str]]:
        if path in live:
            return "ok", []
        candidates = sorted(by_name.get(_basename(path), []))
        if len(candidates) == 1:
            return "drift", candidates
        if candidates:
            return "ambiguous", candidates
        return "missing", []

    primary = str(row.get("source_path", "") or "").strip()
    if primary:
        state, candidates = lookup(primary)
        if state == "drift":
            out["path_drift"] = {"golden": primary, "live": candidates[0]}
            if candidates[0] not in out["effective_alt_source_paths"]:
                out["effective_alt_source_paths"].append(candidates[0])
        elif state == "ambiguous":
            out["path_ambiguous"] = candidates
        elif state == "missing":
            out["path_missing"] = True
    for alt in alts:
        state, candidates = lookup(alt)
        if state == "drift":
            out["alt_path_drift"].append({"golden": alt, "live": candidates[0]})
            if candidates[0] not in out["effective_alt_source_paths"]:
                out["effective_alt_source_paths"].append(candidates[0])
        elif state == "ambiguous":
            out["alt_path_ambiguous"].append({"golden": alt, "candidates": candidates})
        elif state == "missing":
            out["alt_path_missing"].append(alt)
    return out


def effective_row(row: dict[str, Any], drift: dict[str, Any]) -> dict[str, Any]:
    """Копия golden-пары с эффективными `alt_source_paths` (после дрейфа)."""
    out = dict(row)
    out["alt_source_paths"] = list(drift.get("effective_alt_source_paths") or [])
    return out


async def fetch_live_paths(backend: BackendClient | None) -> set[str] | None:
    """Множество путей живого каталога; ``None`` — проверка дрейфа пропущена.

    Любая ошибка каталога — одно предупреждение и поведение «как раньше»:
    проверка путей — удобство, а не условие прогона. Пустой каталог тоже
    пропускается: иначе КАЖДАЯ пара стала бы «отсутствующей», и это уже не
    дрейф разметки, а пустой индекс, о котором скажет `retrieval_hit`.
    """
    if backend is None:
        return None
    try:
        paths = await backend.catalog_paths()
    except (BackendError, httpx.HTTPError, ValueError) as exc:
        _log(f"ВНИМАНИЕ: каталог недоступен ({exc}) — дрейф путей golden не проверяется")
        return None
    if not paths:
        _log("ВНИМАНИЕ: каталог пуст — дрейф путей golden не проверяется")
        return None
    _log(f"каталог: {len(paths)} документов — пути golden сверяются с ним")
    return set(paths)


# --------------------------------------------------------------------------- #
# Модель и провайдер ответа — по факту, а не по ключу настроек
# --------------------------------------------------------------------------- #


def effective_model(settings: Any) -> dict[str, Any]:
    """``{provider, model, note}`` — что на самом деле отвечало в этом ходе.

    Снимок настроек носит `gigachat.model` с первого дня, но с появлением
    KitAI этот ключ перестал быть правдой: при `provider: kitai` отвечает
    `kitai_model`, а `gigachat.model` — просто неиспользуемая настройка. UI
    теперь пишет `model_effective`; отчёт берёт его, и только за неимением —
    восстанавливает по `provider` + подходящему ключу, а совсем старый снимок
    показывает со ссылкой на то, что провайдер не записан.
    """
    if not isinstance(settings, dict):
        return {"provider": None, "model": None, "note": NOT_RECORDED}
    eff = settings.get("model_effective")
    if isinstance(eff, dict) and (eff.get("model") or eff.get("provider")):
        return {
            "provider": eff.get("provider") or None,
            "model": eff.get("model") or None,
            "note": None,
        }
    giga = settings.get("gigachat")
    if not isinstance(giga, dict):
        return {"provider": None, "model": None, "note": NOT_RECORDED}
    provider = giga.get("provider")
    if provider:
        key = "kitai_model" if str(provider) == "kitai" else "model"
        return {"provider": str(provider), "model": giga.get(key) or None, "note": None}
    if giga.get("model"):
        return {
            "provider": None,
            "model": giga.get("model"),
            "note": "(ключ провайдера не записан)",
        }
    return {"provider": None, "model": None, "note": NOT_RECORDED}


def _model_label(entry: dict[str, Any]) -> str:
    """«kitai / glm-5.1», «GigaChat-2-Max (ключ провайдера не записан)», «не записано»."""
    provider, model, note = entry.get("provider"), entry.get("model"), entry.get("note")
    if not provider and not model:
        return str(note or NOT_RECORDED)
    head = " / ".join(str(x) for x in (provider, model) if x)
    return f"{head} {note}" if note else head


def retrieval_hit(
    row: dict[str, Any], sources: Sequence[dict[str, Any]]
) -> tuple[bool | None, str]:
    """Попал ли нужный ФРАГМЕНТ в контекст, и с какой точностью это проверено.

    Возвращает ``(hit, granularity)``, где granularity — ``chunk`` / ``section``
    / ``file`` / ``none``.

    Правило по убыванию точности:

    1. ``chunk`` — golden-пара знает свой ``source_chunk_index``: сверяем пару
       ``(path, chunk_index)`` по ``chunk_indexes`` источника (лог их пишет).
       Блок с ``depth == "file"`` несёт весь документ, поэтому считается
       покрывающим любой чанк своего файла;
    2. ``section`` — чанк неизвестен, но известен ``section_path``: сверяем
       ``(path, section_path)``;
    3. ``file`` — совсем без чанка и раздела: старое пофайловое сравнение,
       помеченное как таковое (оно завышает hit — файл мог попасть другим
       фрагментом).

    ``alt_source_paths`` — страницы, которые ТОЖЕ отвечают на вопрос (в корпусе
    есть соседи с пересекающимся содержимым: тематическая страница и её
    «Пользовательская инструкция», реестр, дублирующий строку другого реестра).
    Любой чанк такой страницы засчитывается как попадание НА ЛЮБОМ уровне
    лестницы: раздел и чанк размечены относительно ``source_path``, к
    альтернативе они неприменимы, а сама альтернатива принята в набор целиком —
    иначе корректный ретрив читался бы как промах и смещал тюнинг.
    ``granularity`` при этом остаётся уровнем разметки пары (чем ЕЁ можно было
    мерить), как и раньше, — отчёт о деградации гранулярности не меняется.

    ``None`` — у golden-пары нет ``source_path`` (например, вопрос-отказ),
    проверять нечего.
    """
    expected_path = str(row.get("source_path", "") or "")
    if not expected_path:
        return None, "none"

    alt_paths = alt_source_paths(row)
    alt_hit = any(str(s.get("path", "") or "") in alt_paths for s in sources)

    own = [s for s in sources if str(s.get("path", "") or "") == expected_path]
    if not own:
        granularity = (
            "chunk"
            if row.get("source_chunk_index") is not None
            else ("section" if row.get("section_path") else "file")
        )
        return alt_hit, granularity

    expected_chunk = row.get("source_chunk_index")
    if isinstance(expected_chunk, int):
        for source in own:
            if source.get("depth") == "file":
                return True, "chunk"
            indexes = source.get("chunk_indexes")
            if not isinstance(indexes, list):
                single = source.get("chunk_index")
                indexes = [single] if isinstance(single, int) else []
            if expected_chunk in indexes:
                return True, "chunk"
        return alt_hit, "chunk"

    expected_section = str(row.get("section_path", "") or "").strip()
    if expected_section:
        for source in own:
            if source.get("depth") == "file":
                return True, "section"
            if str(source.get("section_path", "") or "").strip() == expected_section:
                return True, "section"
        return alt_hit, "section"

    return True, "file"


async def run_sample(
    row: dict[str, Any],
    *,
    chat: ChatClient,
    judge: GigaChatJudge,
    backend: BackendClient | None,
    cache: dict[str, str],
    context_cap: int,
    rag_log: RagLogIndex | None = None,
    live_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Ask one golden question, judge the answer, return a report row.

    A sample that could not be produced end-to-end (chat error, unusable
    context) carries ``failed: True`` and empty ``metrics`` — :func:`build_report`
    keeps such rows out of every average instead of letting their zeros read as
    a regression.

    ``live_paths`` — пути живого каталога (см. :func:`resolve_golden_paths`);
    ``None`` — дрейф путей не проверяется, пара берётся как размечена.
    """
    question = str(row.get("question", "") or "")
    ground_truth = str(row.get("ground_truth", "") or "")
    outcome_expected = expected_outcome(row)
    expects_refusal = outcome_expected == OUTCOME_REFUSAL
    drift = resolve_golden_paths(row, live_paths)
    row = effective_row(row, drift)
    started = time.perf_counter()
    sample: dict[str, Any] = {
        "id": row.get("id"),
        "kind": row.get("kind"),
        # Ручная категория из golden-set — по ней строится разрез отчёта.
        CATEGORY_KEY: category_of(row),
        "question": question,
        "ground_truth": ground_truth,
        "source_path": row.get("source_path"),
        "alt_source_paths": alt_source_paths(row),
        "source_chunk_index": row.get("source_chunk_index"),
        "section_path": row.get("section_path"),
        "expected_refusal": expects_refusal,
        OUTCOME_KEY: outcome_expected,
        "accepted": row.get("accepted"),
        # Дрейф путей golden относительно живого каталога (см. resolve_golden_paths).
        "path_checked": drift["checked"],
        "path_drift": drift["path_drift"],
        "path_ambiguous": drift["path_ambiguous"],
        "path_missing": drift["path_missing"],
        "alt_path_drift": drift["alt_path_drift"],
        "alt_path_ambiguous": drift["alt_path_ambiguous"],
        "alt_path_missing": drift["alt_path_missing"],
        "answer": "",
        "sources": [],
        "context_count": 0,
        "context_origin": "none",
        RETRIEVAL_KEY: None,
        "retrieval_granularity": "none",
        REFUSAL_KEY: None,
        FALSE_REFUSAL_KEY: None,
        META_KEY: None,
        # Скрытые вызовы хода и факт пустого ответа — из записи лога; `None`
        # означает «UI их не записал», а не «их не было».
        "hidden_calls": None,
        "empty_answer": None,
        GENERATION_FAILED_KEY: False,
        "metrics": {},
        "metrics_note": "",
        JUDGE_CLIP_KEY: False,
        "error": "",
        "failed": False,
        "latency_ms": 0,
    }

    def fail(message: str) -> dict[str, Any]:
        sample["error"] = message
        sample["failed"] = True
        return sample

    try:
        outcome = await chat.ask(question)
    except httpx.HTTPError as exc:
        sample["latency_ms"] = int((time.perf_counter() - started) * 1000)
        return fail(f"chat transport: {exc}")

    sample["latency_ms"] = int((time.perf_counter() - started) * 1000)
    sample["answer"] = outcome.answer
    sample["chat_id"] = outcome.chat_id
    sample["notice"] = outcome.notice
    sample["finish_reason"] = outcome.finish_reason
    sample["event_order"] = outcome.order

    if outcome.error:
        return fail(outcome.error)

    # Контекст: сперва фактический из лога, иначе — приближённое восстановление.
    record = rag_log.get(outcome.chat_id) if rag_log is not None else None
    resolved = context_from_log(record) if record else None
    if resolved is None:
        resolved = await rebuild_contexts(
            backend, outcome.sources, cache=cache, cap=context_cap
        )
    if record is not None:
        sample["run_settings"] = record.get("settings")
        sample["timings_ms"] = record.get("timings_ms")
        # ВСЕ кандидаты поиска до отбора, с рангом и оценкой грейдера. Без них
        # главный вопрос тюнинга — «нужный документ вообще нашёлся, но не доехал
        # до контекста, или его не было в выдаче?» — по отчёту неотвечаем, а
        # именно на нём стоят правки глубины, стемминга и аббревиатур.
        sample["candidates"] = record.get("candidates")
        # Исход каждого скрытого вызова (condense, батчи грейдера) и признак
        # пустого ответа. Старый UI этих ключей не пишет — остаётся `None`.
        hidden = record.get("hidden_calls")
        sample["hidden_calls"] = hidden if isinstance(hidden, dict) else None
        if "empty_answer" in record:
            sample["empty_answer"] = bool(record.get("empty_answer"))

    # Метаданные источников: из лога они богаче (`chunk_index`), из SSE — беднее.
    sample["sources"] = resolved.sources or outcome.sources
    sample["context_count"] = len(resolved.contexts)
    sample["context_origin"] = resolved.origin
    # Судья видит `context_text` из лога. Если лог обрезан своим капом, блоков
    # там меньше, чем источников у ответа, и судья штрафует ответ за текст,
    # которого ему не дали. Сэмпл из-за этого не выбрасывается (ответ и ретрив
    # измерены честно), но молчать об этом нельзя.
    sample["context_clipped"] = (
        len(sample["sources"] or []) > len(resolved.contexts)
        if resolved.origin == "rag_log"
        else False
    )
    hit, granularity = retrieval_hit(row, sample["sources"])
    sample[RETRIEVAL_KEY] = hit
    sample["retrieval_granularity"] = granularity

    # Пустой ответ — отдельный исход, а не оценка качества. Ретрив состоялся
    # (hit посчитан выше), но ни отказа, ни ответа не было: флаги ветвей
    # остаются `None`, судья не вызывается, метрики — `null` с пометкой.
    if generation_failed(
        answer=outcome.answer,
        finish_reason=outcome.finish_reason,
        empty_answer=sample["empty_answer"],
    ):
        sample[GENERATION_FAILED_KEY] = True
        sample["metrics"] = None
        sample["metrics_note"] = (
            "генерация не дала ответа — судья не вызывался, пара вне средних"
        )
        if resolved.error:
            return fail(resolved.error)
        return sample

    refused = is_refusal(outcome.answer, finish_reason=outcome.finish_reason)
    sample[REFUSAL_KEY] = refused
    # Ложный отказ меряется ТОЛЬКО на отвечаемых парах: на паре-ловушке отказ —
    # это правильный ответ (он живёт в `refusal_ok`), а на метапаре — свой,
    # третий исход (`meta_answered`), у которого свой знаменатель.
    sample[FALSE_REFUSAL_KEY] = refused if outcome_expected == OUTCOME_ANSWER else None
    # Метапара: успех — это СОДЕРЖАТЕЛЬНЫЙ ответ. Отказ здесь ловится тем же
    # `is_refusal` (включая `finish_reason == "no_context"`, то есть «грейдер не
    # оставил ни одного фрагмента»), но засчитывается с обратным знаком.
    sample[META_KEY] = (not refused) if outcome_expected == OUTCOME_META else None

    if resolved.error:
        return fail(resolved.error)

    results = await evaluate_sample(
        judge,
        question=question,
        ground_truth=ground_truth,
        answer=outcome.answer,
        contexts=resolved.contexts,
        # Только у вопросов-перечислений; у остальных метрика даёт None и в
        # среднее не входит.
        expected_items=row.get("expected_items") or (),
    )
    sample["metrics"] = {name: result.to_dict() for name, result in results.items()}
    sample[JUDGE_CLIP_KEY] = judge_context_clipped(sample["metrics"])
    return sample


async def run_all(
    rows: Sequence[dict[str, Any]],
    *,
    chat: ChatClient,
    judge: GigaChatJudge,
    backend: BackendClient | None,
    concurrency: int,
    context_cap: int,
    rag_log: RagLogIndex | None = None,
    live_paths: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Run every golden pair with bounded concurrency (GigaChat is fragile).

    По умолчанию `concurrency=1`. У судьи на контуре фактически один слот: второй
    одновременный запрос получает 429 сразу, а ждать освобождения дольше, чем
    длится чужой вызов, ретраи не могут. Прогон `baseline` шёл в два потока и
    потерял 95 судейских вызовов из 188 — ровно через строку, 0 или все 4 метрики
    сэмпла. Поднимать это число можно, только убедившись, что 429 не вернулись.
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))
    cache: dict[str, str] = {}
    out: list[dict[str, Any]] = [{} for _ in rows]
    done = 0

    async def worker(index: int, row: dict[str, Any]) -> None:
        nonlocal done
        async with semaphore:
            sample = await run_sample(
                row,
                chat=chat,
                judge=judge,
                backend=backend,
                cache=cache,
                context_cap=context_cap,
                rag_log=rag_log,
                live_paths=live_paths,
            )
            out[index] = sample
            done += 1
            mark = "!" if sample.get("failed") else "·"
            if sample.get(GENERATION_FAILED_KEY):
                mark = "∅"
            _log(f"  [{done}/{len(rows)}] {mark} {sample.get('id')}")

    await asyncio.gather(*(worker(i, row) for i, row in enumerate(rows)))
    return out


# --------------------------------------------------------------------------- #
# Report rendering (pure)
# --------------------------------------------------------------------------- #


def is_failed(sample: dict[str, Any]) -> bool:
    """Сэмпл, который не удалось довести до конца — не данные, а сбой прогона.

    Такие строки не участвуют ни в одном среднем: их нули — это «бэкенд лёг»,
    а не «качество упало». Число упавших идёт в отчёт отдельной строкой.
    """
    return bool(sample.get("failed") or sample.get("error"))


def generation_failed(
    *, answer: str, finish_reason: str | None, empty_answer: bool | None
) -> bool:
    """Ответа НЕ БЫЛО: пустой текст от модели — это не отказ и не плохой ответ.

    Источник правды — `empty_answer` из записи лога (UI видит поток целиком).
    Старая запись без этого ключа: пустой текст при ``finish_reason == "length"``
    — модель упёрлась в лимит, не выдав ни токена. ``no_context`` исключён в
    обе стороны: там пустой поток — ШТАТНЫЙ отказ «грейдер ничего не оставил»,
    и его меряет `refusal_ok`, а не эта корзина.
    """
    if finish_reason == "no_context":
        return False
    if empty_answer is not None:
        return bool(empty_answer)
    return not (answer or "").strip() and finish_reason == "length"


def is_generation_failed(sample: dict[str, Any]) -> bool:
    """Сэмпл из корзины «генерация не дала ответа» (см. :func:`generation_failed`).

    Упавший сэмпл (`failed`) — приоритетнее: там не было и ретрива.
    """
    return bool(sample.get(GENERATION_FAILED_KEY)) and not is_failed(sample)


def successful(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Пары, чьи ЧИСЛА идут в средние: не упали и получили ответ."""
    return [s for s in samples if not is_failed(s) and not is_generation_failed(s)]


def retrieved(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Пары, где ретрив состоялся — знаменатель `retrieval_hit`.

    Шире `successful`: сбой генерации случается ПОСЛЕ ретрива, и попадание
    нужного фрагмента в контекст у такой пары измерено честно.
    """
    return [s for s in samples if not is_failed(s)]


def answerable(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Пары, у которых есть ответ в корпусе, — только они идут в судейские средние.

    Пара-ловушка (``expected_refusal``) меряется теми же четырьмя метриками
    против фиксированного эталона отказа, и `answer_relevancy_ru` НАМЕРЕННО
    ставит 0 уклончивому ответу (`metrics.answer_relevancy_ru`). То есть
    ПРАВИЛЬНЫЙ отказ тянул общее среднее вниз, и оно зависело от доли ловушек
    в наборе, а не от качества. Теперь ловушки живут в своей корзине
    (``aggregate_refusal``), а их ветку меряет `refusal_ok`.

    Метапары (``expected_outcome == "meta"``) вынесены по той же причине с
    обратным знаком: документа-цели у них нет, поэтому `context_precision` и
    `context_recall` меряли бы не качество ответа, а отсутствие разметки.
    """
    return [s for s in samples if expected_outcome(s) == OUTCOME_ANSWER]


def refusal_rows(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Пары-ловушки: правильный ответ — отказ."""
    return [s for s in samples if expected_outcome(s) == OUTCOME_REFUSAL]


def meta_rows(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Метапары: вопрос про саму базу или про ассистента.

    Отвечать обязательно, цели в корпусе нет. Отказ на такой паре — дефект, а не
    правильный ответ, поэтому её ветку меряет `meta_answered_rate`, а не
    `refusal_ok`.
    """
    return [s for s in samples if expected_outcome(s) == OUTCOME_META]


def retrieval_hit_rate(samples: Sequence[dict[str, Any]]) -> float | None:
    """Доля пар с состоявшимся ретривом, где нужный фрагмент попал в контекст."""
    hits = [
        s.get(RETRIEVAL_KEY)
        for s in retrieved(samples)
        if isinstance(s.get(RETRIEVAL_KEY), bool)
    ]
    if not hits:
        return None
    return round(sum(1 for v in hits if v) / len(hits), 4)


def refusal_rate(samples: Sequence[dict[str, Any]]) -> float | None:
    """Доля пар «ответа в корпусе нет», где ассистент честно отказался."""
    values = [
        s.get(REFUSAL_KEY)
        for s in successful(samples)
        if s.get("expected_refusal") and isinstance(s.get(REFUSAL_KEY), bool)
    ]
    if not values:
        return None
    return round(sum(1 for v in values if v) / len(values), 4)


def false_refusal_rate(samples: Sequence[dict[str, Any]]) -> float | None:
    """Доля ОТВЕЧАЕМЫХ пар, где ассистент отказался отвечать зря.

    Обратная и более опасная ошибка, чем выдумка: вопрос ответ имеет, нужный
    фрагмент мог быть в контексте, а пользователь получил «в источниках
    ничего нет». Классическая регрессия после закручивания порога грейдера,
    которую `refusal_ok` не видит вовсе (он смотрит только на ловушки).

    **Меньше — лучше.**
    """
    values = [
        s.get(FALSE_REFUSAL_KEY)
        for s in successful(answerable(samples))
        if isinstance(s.get(FALSE_REFUSAL_KEY), bool)
    ]
    if not values:
        return None
    return round(sum(1 for v in values if v) / len(values), 4)


def meta_answered_rate(samples: Sequence[dict[str, Any]]) -> float | None:
    """Доля МЕТАПАР, на которые ассистент ответил, а не отказался.

    Зеркало `refusal_ok` для третьей корзины: там отказ — успех, здесь — провал.
    Отдельное число, а не строка в `false_refusal_rate`, потому что у того свой
    знаменатель (отвечаемые пары), и подмешивание метапар меняло бы существующую
    метрику вместо того, чтобы добавить новую.

    **Больше — лучше.**
    """
    values = [
        s.get(META_KEY)
        for s in successful(samples)
        if expected_outcome(s) == OUTCOME_META and isinstance(s.get(META_KEY), bool)
    ]
    if not values:
        return None
    return round(sum(1 for v in values if v) / len(values), 4)


def hedge_rate(samples: Sequence[dict[str, Any]]) -> float | None:
    """Доля ОТВЕЧАЕМЫХ пар, где ответ открывается оговоркой «ответа не нашлось».

    Судья `answer_relevancy_ru` (промпты v2) не обнуляет такой ответ — оценивает
    содержательную часть и поднимает флаг `hedged`. Флаг считается здесь
    отдельно: пользователь читает оговорку как отказ, и рост её доли — та же
    регрессия, что и `false_refusal_rate`, только тише. Знаменатель — пары, у
    которых у метрики ЕСТЬ поле `hedged` (отчёт прежних промптов его не знает и
    честно даёт `None`, а не 0). **Меньше — лучше.**
    """
    values: list[bool] = []
    for sample in successful(answerable(samples)):
        entry = (sample.get("metrics") or {}).get("answer_relevancy_ru")
        if isinstance(entry, dict) and isinstance(entry.get("hedged"), bool):
            values.append(entry["hedged"])
    if not values:
        return None
    return round(sum(1 for v in values if v) / len(values), 4)


def judge_context_clipped(metrics: Any) -> bool:
    """Судья видел контекст, урезанный СВОИМ капом (`raw.context_clipped_by_judge`).

    Отличается от `context_clipped`: тот — обрезка `context_text` в логе
    (`rag_log.MAX_TEXT_CHARS`), этот — обрезка уже внутри `metrics.format_context`.
    Причины и лечение разные, поэтому и счётчики разные.
    """
    if not isinstance(metrics, dict):
        return False
    for name in metrics_mod.JUDGE_METRIC_NAMES:
        entry = metrics.get(name)
        raw = entry.get("raw") if isinstance(entry, dict) else None
        if isinstance(raw, dict) and raw.get("context_clipped_by_judge"):
            return True
    return False


def is_judge_clipped(sample: dict[str, Any]) -> bool:
    """Флаг сэмпла, восстановленный из метрик, если сам ключ не записан."""
    value = sample.get(JUDGE_CLIP_KEY)
    if isinstance(value, bool):
        return value
    return judge_context_clipped(sample.get("metrics"))


def judge_calls_by_metric(samples: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Сколько вызовов судьи ушло на каждую метрику (`raw.calls`) и всего.

    Стоимость прогона по факту, а не «~4 на пару»: `faithfulness` и `recall`
    режут длинный ответ на несколько вызовов, и без этой строки счётчик
    `judge_calls` клиента не раскладывается по метрикам.
    """
    out: dict[str, int] = {name: 0 for name in metrics_mod.JUDGE_METRIC_NAMES}
    for sample in samples:
        metrics = sample.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for name in metrics_mod.JUDGE_METRIC_NAMES:
            entry = metrics.get(name)
            raw = entry.get("raw") if isinstance(entry, dict) else None
            calls = raw.get("calls") if isinstance(raw, dict) else None
            if isinstance(calls, int) and not isinstance(calls, bool):
                out[name] += calls
    out["total"] = sum(out.values())
    return out


def _judge_calls_line(calls: dict[str, Any] | None) -> str:
    """«судейских вызовов: N (faithfulness a, relevancy b, precision c, recall d)»."""
    if not isinstance(calls, dict):
        return f"судейских вызовов: {NOT_RECORDED}"
    short = (
        ("faithfulness_ru", "faithfulness"),
        ("answer_relevancy_ru", "relevancy"),
        ("context_precision", "precision"),
        ("context_recall", "recall"),
    )
    parts = ", ".join(f"{title} {int(calls.get(key, 0) or 0)}" for key, title in short)
    return f"судейских вызовов: {int(calls.get('total', 0) or 0)} ({parts})"


def slim_report(report: dict[str, Any]) -> dict[str, Any]:
    """Копия отчёта для записи на диск: без `raw.replies` у каждой метрики.

    Сырые ответы судьи по вызовам — самое тяжёлое поле отчёта (десятки КБ на
    пару) и нужны только при отладке промптов; всё остальное (`score`, `raw.calls`,
    вердикты, `context_clipped_by_judge`) сохраняется. Что именно выброшено —
    :data:`REPORT_DROPPED_RAW_KEYS`.
    """
    out = dict(report)
    samples: list[dict[str, Any]] = []
    for sample in report.get("samples", []) or []:
        if not isinstance(sample, dict):
            samples.append(sample)
            continue
        copy = dict(sample)
        metrics = sample.get("metrics")
        if isinstance(metrics, dict):
            slimmed: dict[str, Any] = {}
            for name, entry in metrics.items():
                if isinstance(entry, dict) and isinstance(entry.get("raw"), dict):
                    entry = dict(entry)
                    entry["raw"] = {
                        k: v for k, v in entry["raw"].items() if k not in REPORT_DROPPED_RAW_KEYS
                    }
                slimmed[name] = entry
            copy["metrics"] = slimmed
        samples.append(copy)
    out["samples"] = samples
    return out


def grader_health(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Отработал ли грейдер — по его собственным оценкам, а не по таймингам.

    У каждого источника есть `grade` (1..5 от грейдера, ``None`` — не оценён).
    Провалившийся батч деградирует во все ``None``, и `select` пропускает
    кандидатов В СЫРОМ ПОРЯДКЕ ПОИСКА: пайплайн отвечает, ошибки нигде нет, но
    реранкера в нём не было. Прогон `baseline` так прошёл целиком — оценки
    появились ровно у 2 пар из 47, и по отчёту это было не видно никак.

    Знаменатель — пары, где грейдеру было что оценивать: он включён и источники
    есть. Пара без источников ничего о его здоровье не говорит.
    """
    applicable = 0
    graded = 0
    partial = 0
    enabled_seen = False
    for sample in samples:
        rag_cfg = ((sample.get("run_settings") or {}).get("rag")) or {}
        if not rag_cfg.get("grader_enabled"):
            continue
        enabled_seen = True
        grades = [
            src.get("grade")
            for src in (sample.get("sources") or [])
            if isinstance(src, dict)
        ]
        if not grades:
            continue
        applicable += 1
        scored = [g for g in grades if g is not None]
        if scored:
            graded += 1
            if len(scored) < len(grades):
                partial += 1
    return {
        "enabled": enabled_seen,
        "applicable": applicable,
        "graded": graded,
        "ungraded": applicable - graded,
        "partial": partial,
    }


def grader_degraded(health: dict[str, Any] | None) -> bool:
    """Реранкер оценил меньше :data:`GRADER_DEGRADED_THRESHOLD` применимых пар.

    Порог, а не «хоть одна пара без оценок»: единичный упавший батч — шум
    контура, а не другая система. Ниже 90 % — уже другая система.
    """
    health = health or {}
    if not health.get("enabled"):
        return False
    applicable = int(health.get("applicable", 0) or 0)
    if not applicable:
        return False
    return int(health.get("graded", 0) or 0) / applicable < GRADER_DEGRADED_THRESHOLD


def normalise_error(error: Any) -> str:
    """``«Type: сообщение»`` → тип исключения + первые 120 символов сообщения.

    Ключ для группировки: один и тот же сбой с разными хвостами (id запроса,
    время) должен схлопываться в одну строку, иначе «160 батчей» превращаются
    в 160 строк по одному.
    """
    text = " ".join(str(error or "").split())
    if not text:
        return "неизвестная ошибка"
    kind, sep, message = text.partition(":")
    if not sep or " " in kind.strip():
        return text[:120]
    return f"{kind.strip()}: {message.strip()[:120]}".rstrip(": ")


def _ms_stats(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return round(statistics.median(values), 1), round(max(values), 1)


def _count(bucket: dict[str, int], key: Any) -> None:
    name = "none" if key is None else str(key)
    bucket[name] = bucket.get(name, 0) + 1


def _sorted_counter(bucket: dict[str, int]) -> dict[str, int]:
    return dict(sorted(bucket.items(), key=lambda kv: (-kv[1], kv[0])))


def _collect_call(
    call: dict[str, Any],
    *,
    by_error: dict[str, int],
    by_finish: dict[str, int],
    by_model: dict[str, int],
    error_models: dict[str, set[str]],
    examples: dict[str, list[str]],
    ms: list[float],
    failure_statuses: tuple[str, ...],
) -> None:
    """Учесть один скрытый вызов (condense или батч грейдера) в счётчиках."""
    status = str(call.get("status") or "none")
    _count(by_finish, call.get("finish_reason"))
    _count(by_model, call.get("model"))
    value = call.get("ms")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        ms.append(float(value))
    if status not in failure_statuses and not call.get("error"):
        return
    key = normalise_error(call.get("error")) if call.get("error") else f"status: {status}"
    by_error[key] = by_error.get(key, 0) + 1
    if call.get("model"):
        error_models.setdefault(key, set()).add(str(call["model"]))
    detail = str(call.get("detail") or "").strip()
    bucket = examples.setdefault(key, [])
    if detail and detail not in bucket and len(bucket) < 3:
        bucket.append(detail)


def hidden_call_health(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Почему скрытые вызовы не отработали — по записям `hidden_calls`.

    `grader_health` говорит, СКОЛЬКО пар остались без оценок; эта сводка говорит,
    ПОЧЕМУ: исход каждого батча грейдера и каждого вызова condense, ошибки по
    типам (с примерами `detail`), `finish_reason`, модель, время. Запись без
    `hidden_calls` (старый UI) считается в `not_recorded`, и отчёт печатает
    «не записано», а не додумывает причину.
    """
    recorded = 0
    not_recorded = 0
    grader: dict[str, Any] = {
        "calls": 0,
        "by_status": {},
        "batches_total": 0,
        "batches_ok": 0,
        "batches_failed": 0,
        "batches_truncated": 0,
        "batches_partial": 0,
        "graded": 0,
        "omitted": 0,
        "by_error": {},
        "by_error_model": {},
        "by_finish_reason": {},
        "by_model": {},
        "examples": {},
        "ms_median": None,
        "ms_max": None,
    }
    condense: dict[str, Any] = {
        "calls": 0,
        "by_status": {},
        "by_error": {},
        "by_error_model": {},
        "by_finish_reason": {},
        "by_model": {},
        "examples": {},
        "ms_median": None,
        "ms_max": None,
    }
    grader_ms: list[float] = []
    condense_ms: list[float] = []
    grader_error_models: dict[str, set[str]] = {}
    condense_error_models: dict[str, set[str]] = {}
    for sample in samples:
        hidden = sample.get("hidden_calls")
        if not isinstance(hidden, dict):
            not_recorded += 1
            continue
        recorded += 1
        call = hidden.get("condense")
        if isinstance(call, dict):
            condense["calls"] += 1
            _count(condense["by_status"], call.get("status"))
            _collect_call(
                call,
                by_error=condense["by_error"],
                by_finish=condense["by_finish_reason"],
                by_model=condense["by_model"],
                error_models=condense_error_models,
                examples=condense["examples"],
                ms=condense_ms,
                failure_statuses=("failed", "truncated"),
            )
        entry = hidden.get("grader")
        if isinstance(entry, dict):
            grader["calls"] += 1
            _count(grader["by_status"], entry.get("status"))
            for batch in entry.get("batches") or []:
                if not isinstance(batch, dict):
                    continue
                grader["batches_total"] += 1
                status = str(batch.get("status") or "none")
                if status == "ok":
                    grader["batches_ok"] += 1
                elif status == "failed":
                    grader["batches_failed"] += 1
                elif status == "truncated":
                    grader["batches_truncated"] += 1
                elif status == "partial":
                    grader["batches_partial"] += 1
                for key in ("graded", "omitted"):
                    value = batch.get(key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        grader[key] += value
                _collect_call(
                    batch,
                    by_error=grader["by_error"],
                    by_finish=grader["by_finish_reason"],
                    by_model=grader["by_model"],
                    error_models=grader_error_models,
                    examples=grader["examples"],
                    ms=grader_ms,
                    failure_statuses=("failed", "truncated", "partial"),
                )
    for entry, ms, models in (
        (grader, grader_ms, grader_error_models),
        (condense, condense_ms, condense_error_models),
    ):
        entry["ms_median"], entry["ms_max"] = _ms_stats(ms)
        entry["by_error"] = _sorted_counter(entry["by_error"])
        entry["by_status"] = _sorted_counter(entry["by_status"])
        entry["by_finish_reason"] = _sorted_counter(entry["by_finish_reason"])
        entry["by_model"] = _sorted_counter(entry["by_model"])
        entry["by_error_model"] = {k: sorted(v) for k, v in models.items()}
    return {
        "recorded": recorded,
        "not_recorded": not_recorded,
        "grader": grader,
        "condense": condense,
    }


def grader_cause_clause(report: dict[str, Any]) -> str:
    """Доминирующая причина сбоев грейдера — фраза для первоэкранного предупреждения.

    «KitaiQueryFailed: 404 "No such model" (glm-5.1) — 160 батчей, …» по
    `hidden_call_health.grader.by_error`; без записей — честное «причина не
    записана». Пустая строка — записи есть, но ни одного сбоя в них нет (тогда
    предупреждение по оценкам противоречит записям, и это само по себе факт).
    """
    health = report.get("hidden_call_health") or {}
    if not health.get("recorded"):
        return GRADER_CAUSE_NOT_RECORDED
    grader = health.get("grader") or {}
    by_error = grader.get("by_error") or {}
    if not by_error:
        return ""
    models = grader.get("by_error_model") or {}
    parts: list[str] = []
    for key, count in list(by_error.items())[:3]:
        tail = f" ({', '.join(models[key])})" if models.get(key) else ""
        parts.append(f"{key}{tail} — {count} батч(ей)")
    rest = len(by_error) - 3
    if rest > 0:
        parts.append(f"ещё {rest} тип(а) ошибок — см. «Скрытые вызовы»")
    return GRADER_CAUSE_PREFIX + ", ".join(parts)


def grader_cell(sample: dict[str, Any]) -> str:
    """Колонка «грейдер» в таблице по парам: `ok` / `2/4 ✗` / `trunc` / `—`."""
    hidden = sample.get("hidden_calls")
    entry = hidden.get("grader") if isinstance(hidden, dict) else None
    if not isinstance(entry, dict):
        return "—"
    batches = [b for b in (entry.get("batches") or []) if isinstance(b, dict)]
    total = len(batches)
    if not total:
        return "skip" if entry.get("status") == "skipped" else "—"
    failed = sum(1 for b in batches if b.get("status") == "failed")
    truncated = sum(1 for b in batches if b.get("status") == "truncated")
    partial = sum(1 for b in batches if b.get("status") == "partial")
    if failed:
        return f"{failed}/{total} ✗"
    if truncated:
        return "trunc" if total == 1 else f"{truncated}/{total} trunc"
    if partial:
        return f"{partial}/{total} part"
    return "ok"


def path_drift_summary(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Сводка дрейфа путей golden по всем парам (для JSON и секции отчёта)."""
    checked = 0
    drifted: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for sample in samples:
        if not sample.get("path_checked"):
            continue
        checked += 1
        ident = sample.get("id")
        drift = sample.get("path_drift")
        if isinstance(drift, dict):
            drifted.append({"id": ident, "golden": drift.get("golden"), "live": drift.get("live")})
        for alt in sample.get("alt_path_drift") or []:
            if isinstance(alt, dict):
                drifted.append({"id": ident, "golden": alt.get("golden"), "live": alt.get("live")})
        if sample.get("path_ambiguous"):
            ambiguous.append(
                {
                    "id": ident,
                    "golden": sample.get("source_path"),
                    "candidates": list(sample.get("path_ambiguous") or []),
                }
            )
        for alt in sample.get("alt_path_ambiguous") or []:
            if isinstance(alt, dict):
                ambiguous.append(
                    {"id": ident, "golden": alt.get("golden"), "candidates": alt.get("candidates")}
                )
        if sample.get("path_missing"):
            missing.append({"id": ident, "golden": sample.get("source_path")})
        for alt in sample.get("alt_path_missing") or []:
            missing.append({"id": ident, "golden": alt})
    return {
        "checked": checked,
        "drifted": drifted,
        "ambiguous": ambiguous,
        "missing": missing,
    }


def generation_failures(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Строки секции «Сбои генерации»: id, finish_reason, время стрима, модель."""
    out: list[dict[str, Any]] = []
    for sample in samples:
        if not is_generation_failed(sample):
            continue
        timings = sample.get("timings_ms") or {}
        stream = timings.get("stream") if isinstance(timings, dict) else None
        out.append(
            {
                "id": sample.get("id"),
                "finish_reason": sample.get("finish_reason"),
                "stream_ms": stream,
                "model": _model_label(effective_model(sample.get("run_settings"))),
            }
        )
    return out


def stage_timings(samples: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Медиана/максимум по стадиям хода + сравнение с их поводком.

    `deadline_ms` берётся из снимка настроек хода (`condense_timeout`,
    `grader_timeout`). `at_deadline` — сколько ходов уложились В САМ дедлайн:
    стадия, у которой время совпало с поводком, не «работала ровно столько»,
    а была им обрезана. Живой пример: 41 ход из 46 с `grade` в диапазоне
    20003–20023 мс при `grader_timeout = 20`.
    """
    deadlines = {"condense": "condense_timeout", "grade": "grader_timeout"}
    buckets: dict[str, list[float]] = {}
    limits: dict[str, float] = {}
    for sample in samples:
        rag_cfg = ((sample.get("run_settings") or {}).get("rag")) or {}
        for stage_name, key in deadlines.items():
            value = rag_cfg.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                limits[stage_name] = float(value) * 1000.0
        for name, ms in (sample.get("timings_ms") or {}).items():
            if isinstance(ms, (int, float)) and not isinstance(ms, bool):
                buckets.setdefault(str(name), []).append(float(ms))
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(buckets):
        values = sorted(buckets[name])
        deadline = limits.get(name)
        entry: dict[str, Any] = {
            "n": len(values),
            "median_ms": round(statistics.median(values), 1),
            "max_ms": round(values[-1], 1),
            "deadline_ms": deadline,
            # Дедлайн срабатывает ЧУТЬ позже назначенного (планировщик добавляет
            # миллисекунды), поэтому окно односторонее и с запасом сверху.
            "at_deadline": (
                sum(1 for v in values if deadline * 0.995 <= v <= deadline * 1.05)
                if deadline
                else None
            ),
        }
        out[name] = entry
    return out


def judge_failures(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Сколько судейских вызовов не вернулось и почему.

    Отчёт до сих пор считал только упавшие СЭМПЛЫ (сбой чата), а упавший вызов
    судьи не считал никак: прогон `baseline` показывал «ошибок 0», потеряв при
    этом 95 вызовов из 188 на HTTP 429. Средние от этого не смещаются вниз (у
    метрики стоит ``None``, а `aggregate` пропускает ``None``), но считаются по
    подмножеству, и молчать об этом нельзя — иначе `n=15` читается как свойство
    набора, а не как сбой контура.

    Считаются только метрики с вызовом судьи и только по флагу ``failed``:
    «пустой ground_truth» и «в вопросе нет expected_items» — это структурные
    пропуски, вызова там не было.
    """
    expected = 0
    failed = 0
    affected = 0
    by_error: dict[str, int] = {}
    for sample in samples:
        metrics = sample.get("metrics")
        if not isinstance(metrics, dict):
            continue
        hit = False
        for name in metrics_mod.JUDGE_METRIC_NAMES:
            entry = metrics.get(name)
            if not isinstance(entry, dict):
                continue
            expected += 1
            if not entry.get("failed"):
                continue
            failed += 1
            hit = True
            reason = str(entry.get("error") or "неизвестная ошибка")[:120]
            by_error[reason] = by_error.get(reason, 0) + 1
        if hit:
            affected += 1
    return {
        "expected": expected,
        "failed": failed,
        "samples_affected": affected,
        "by_error": dict(sorted(by_error.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def metric_values(samples: Sequence[dict[str, Any]], name: str) -> list[float]:
    """Оценки метрики по успешным сэмплам (для среднего и разброса)."""
    out: list[float] = []
    for sample in successful(samples):
        entry = (sample.get("metrics") or {}).get(name) or {}
        score = entry.get("score") if isinstance(entry, dict) else None
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            out.append(float(score))
    return out


def dispersion(samples: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Разброс каждой метрики по сэмплам: ``{mean, sd, n, stderr}``.

    Без него из отчёта нельзя понять, сколько «весит» дельта: среднее 0.72 по
    12 парам с sd 0.35 и по 80 парам с sd 0.05 — разные утверждения.
    """
    out: dict[str, dict[str, Any]] = {}
    for name in METRIC_NAMES:
        values = metric_values(samples, name)
        n = len(values)
        mean = round(sum(values) / n, 4) if n else None
        sd = round(statistics.stdev(values), 4) if n > 1 else (0.0 if n == 1 else None)
        stderr = round(sd / math.sqrt(n), 4) if sd is not None and n > 1 else None
        out[name] = {"mean": mean, "sd": sd, "n": n, "stderr": stderr}
    return out


def rate_metrics(samples: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    """Три доли, которые считаются локально, без судьи.

    Одна функция на весь отчёт, чтобы разрез по категориям считался ровно тем
    же кодом, что и общая строка, — иначе они разойдутся при первой же правке.
    """
    return {
        RETRIEVAL_KEY: retrieval_hit_rate(samples),
        REFUSAL_KEY: refusal_rate(samples),
        FALSE_REFUSAL_RATE_KEY: false_refusal_rate(samples),
        META_RATE_KEY: meta_answered_rate(samples),
        HEDGE_RATE_KEY: hedge_rate(samples),
    }


def group_by_category(
    samples: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Разрез отчёта по ``category`` golden-пары.

    Общее среднее по 39 разнородным вопросам скрывает ровно то, ради чего
    заводится ручной golden-set: одна категория может провалиться, вторая
    вырасти, а сумма — не сдвинуться. Категории берутся из данных (какие есть,
    такие и в отчёте), пары без категории собираются в ``unclassified``.

    Судейские средние внутри категории считаются теми же правилами, что и
    общие: без упавших сэмплов, без ``None``-оценок и **без пар-ловушек**.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        groups.setdefault(category_of(sample), []).append(sample)
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(groups):
        rows = groups[name]
        ok = successful(rows)
        entry: dict[str, Any] = {
            "n": len(rows),
            "n_failed": sum(1 for s in rows if is_failed(s)),
            "n_generation_failed": sum(1 for s in rows if is_generation_failed(s)),
        }
        entry.update(aggregate(answerable(ok)))
        entry.update(rate_metrics(rows))
        out[name] = entry
    return out


def granularity_counts(samples: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Чем именно мерился ``retrieval_hit`` — чанком, разделом или файлом."""
    out: dict[str, int] = {}
    for sample in retrieved(samples):
        key = str(sample.get("retrieval_granularity", "none") or "none")
        out[key] = out.get(key, 0) + 1
    return out


def granularity_degradation(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Насколько огрублён ``retrieval_hit`` в этом прогоне.

    ``{degraded, measured, levels}``: сколько пар из измеренных засчитывались по
    разделу/файлу вместо чанка. Ноль ``degraded`` — прогон честно чанковый;
    всё остальное отчёт обязан сказать вслух (см. :data:`GRANULARITY_WARNING`),
    иначе завышенный `retrieval_hit` читается как точный.
    """
    levels = granularity_counts(samples)
    measured = sum(count for key, count in levels.items() if key != "none")
    degraded = sum(levels.get(key, 0) for key in DEGRADED_GRANULARITIES)
    return {"degraded": degraded, "measured": measured, "levels": levels}


def _granularity_label(report: dict[str, Any]) -> str:
    """``chunk: 3, section: 10`` — чем мерилась каждая пара."""
    levels = (report.get("retrieval_degradation") or {}).get("levels") or {}
    pairs = [(key, count) for key, count in sorted(levels.items()) if key != "none"]
    return ", ".join(f"{key}: {count}" for key, count in pairs) or "—"


def run_parameters(
    samples: Sequence[dict[str, Any]],
    *,
    judge_model: str,
    judge_temperature: float | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Всё, без чего прогон не воспроизвести.

    Настройки отвечающей системы (модель, температура, ширина ретрива, порог
    грейдера, отпечатки промптов) берутся из снимка в ``rag_log.jsonl``. Если
    в одном прогоне они разные — это само по себе дефект прогона, поэтому в
    отчёт уезжает ``"(смешанные)"``, а не первое попавшееся значение.
    """
    seen: list[str] = []
    settings: Any = None
    models: list[dict[str, Any]] = []
    for sample in samples:
        snapshot = sample.get("run_settings")
        if not isinstance(snapshot, dict):
            continue
        key = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.append(key)
            settings = snapshot
            models.append(effective_model(snapshot))
    # Модель ответа — отдельными ключами: «смешанные» настройки чаще всего
    # различаются чем-то безобидным (порогом), а модель при этом одна, и её
    # надо видеть; если разная и она — это «(смешанные)» уже по делу.
    distinct = {json.dumps(m, ensure_ascii=False, sort_keys=True) for m in models}
    if len(distinct) == 1:
        answer = models[0]
    elif distinct:
        answer = {"provider": "(смешанные)", "model": "(смешанные)", "note": None}
    else:
        answer = effective_model(None)
    params: dict[str, Any] = {
        "judge_model": judge_model,
        "judge_temperature": judge_temperature,
        "judge_prompt_version": metrics_mod.PROMPT_VERSION,
        "golden_prompt_version": getattr(gen_golden_mod, "PROMPT_VERSION", None),
        "ui_settings": "(смешанные)" if len(seen) > 1 else settings,
        "answer_provider": answer["provider"],
        "answer_model": answer["model"],
        "answer_model_note": answer["note"],
        "judge_calls_by_metric": judge_calls_by_metric(samples),
    }
    params.update(extra or {})
    return params


def answer_model(report: dict[str, Any]) -> dict[str, Any]:
    """``{provider, model, note}`` ответа из отчёта — и из старого тоже.

    Новый отчёт несёт `run_params.answer_*`; старый — только `ui_settings`, по
    которым модель восстанавливается тем же :func:`effective_model`.
    """
    params = report.get("run_params") or {}
    if "answer_model" in params or "answer_provider" in params:
        return {
            "provider": params.get("answer_provider"),
            "model": params.get("answer_model"),
            "note": params.get("answer_model_note"),
        }
    return effective_model(params.get("ui_settings"))


def judge_prompt_version(report: dict[str, Any]) -> str | None:
    """Версия судейских промптов прогона: `run_params.judge_prompt_version`,
    за неимением — верхнеуровневый `prompt_version` (то же число)."""
    params = report.get("run_params") or {}
    value = params.get("judge_prompt_version") or report.get("prompt_version")
    return str(value) if value else None


def judge_prompt_mismatch(report_a: dict[str, Any], report_b: dict[str, Any]) -> bool:
    """Судили разные версии промптов — оценки несопоставимы по построению."""
    a, b = judge_prompt_version(report_a), judge_prompt_version(report_b)
    return bool(a and b and a != b)


def model_mismatch(report_a: dict[str, Any], report_b: dict[str, Any]) -> bool:
    """Отвечали ли в двух прогонах РАЗНЫЕ модели (или провайдеры).

    Считается только по ИЗВЕСТНЫМ значениям: отчёт без записи модели не
    «отличается», он «не проверяем» — про него диф предупреждает отдельно.
    """
    a, b = answer_model(report_a), answer_model(report_b)
    for key in ("provider", "model"):
        if a.get(key) and b.get(key) and str(a[key]) != str(b[key]):
            return True
    return False


def build_report(
    samples: Sequence[dict[str, Any]],
    *,
    label: str,
    golden_path: str,
    ui_url: str,
    judge_model: str,
    judge_temperature: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON report (the same dict is rendered to markdown)."""
    failed = [s for s in samples if is_failed(s)]
    gen_failed = [s for s in samples if is_generation_failed(s)]
    ok = successful(samples)
    # Средние — ТОЛЬКО по успешным: у упавшего сэмпла нули означают «прогон
    # сломался», и в среднем они читались бы как регрессия качества.
    # …и только по ОТВЕЧАЕМЫМ парам: см. :func:`answerable`.
    ok_answerable = answerable(ok)
    ok_refusal = refusal_rows(ok)
    ok_meta = meta_rows(ok)
    aggregates = aggregate(ok_answerable)
    aggregates.update(rate_metrics(samples))
    origins: dict[str, int] = {}
    for sample in samples:
        key = str(sample.get("context_origin", "none") or "none")
        origins[key] = origins.get(key, 0) + 1
    approximate = any(
        sample.get("context_origin") not in (None, "rag_log")
        for sample in ok
        if sample.get("context_count")
    )
    return {
        "label": label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "golden": golden_path,
        "ui_url": ui_url,
        "judge_model": judge_model,
        "prompt_version": metrics_mod.PROMPT_VERSION,
        "approximate": approximate,
        "context_origin": origins,
        "counts": {
            "total": len(samples),
            "failed": len(failed),
            # Ответа не было вовсе (пустой поток): не сбой прогона и не оценка
            # качества — своя корзина, вне всех средних, судья не вызывался.
            GENERATION_FAILED_KEY: len(gen_failed),
            "evaluated": len(ok),
            # Пары, где судье достался урезанный контекст: сэмпл цел, метрики
            # посчитаны, но посчитаны не по тому, что видела модель.
            "context_clipped": sum(1 for s in samples if s.get("context_clipped")),
            # …и отдельно — урезанный уже капом САМОГО судьи (другая причина).
            JUDGE_CLIP_KEY: sum(1 for s in ok if is_judge_clipped(s)),
        },
        # Состав оценённых пар: судейские средние покрывают только `answerable`,
        # и без этих двух чисел прогон на 39 вопросах молча сравнится с
        # прогоном на 30.
        "buckets": {
            "answerable": len(ok_answerable),
            "refusal": len(ok_refusal),
            "meta": len(ok_meta),
        },
        "aggregate": aggregates,
        # Ловушки не выбрасываются — их метрики считаются и лежат отдельно.
        "aggregate_refusal": aggregate(ok_refusal),
        # То же для третьей корзины: судейские числа считаются, но решает
        # `meta_answered_rate` — «ответил ли вообще».
        "aggregate_meta": aggregate(ok_meta),
        "dispersion": dispersion(answerable(samples)),
        "coverage": coverage(ok_answerable),
        "coverage_refusal": coverage(ok_refusal),
        "coverage_meta": coverage(ok_meta),
        "by_category": group_by_category(samples),
        # Здоровье судьи, а не качество ответов: без него «оценено пар: 15» из
        # 36 выглядит как свойство набора, а не как сбой контура.
        "judge_failures": judge_failures(samples),
        # То же для скрытых вызовов самого пайплайна: молча не отработавший
        # грейдер меняет смысл ВСЕХ чисел отчёта (контекст собран сырым
        # порядком поиска), а по метрикам качества это неотличимо.
        "grader_health": grader_health(samples),
        # …и ПОЧЕМУ он не отработал: исходы батчей, ошибки по типам, модель.
        "hidden_call_health": hidden_call_health(samples),
        "generation_failures": generation_failures(samples),
        "path_drift": path_drift_summary(samples),
        "stage_timings": stage_timings(samples),
        "retrieval_granularity": granularity_counts(samples),
        "retrieval_degradation": granularity_degradation(samples),
        "run_params": run_parameters(
            samples,
            judge_model=judge_model,
            judge_temperature=judge_temperature,
            extra=extra,
        ),
        "samples": list(samples),
        "extra": extra or {},
    }


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (int, float)):
        return f"{value:.3f}" if isinstance(value, float) else str(value)
    return "—" if value is None else str(value)


def _fmt_spread(entry: Any) -> str:
    """``mean ±sd (n)`` — одно значение без разброса ничего не говорит."""
    if not isinstance(entry, dict):
        return "—"
    mean, sd, n = entry.get("mean"), entry.get("sd"), entry.get("n", 0)
    if mean is None:
        return "—"
    tail = f" ±{sd:.3f}" if isinstance(sd, (int, float)) else ""
    return f"{mean:.3f}{tail} (n={n})"


def _condense_state(report: dict[str, Any], rag_cfg: dict[str, Any]) -> Any:
    """Значение строки «condense включён» — с поправкой на то, вызывался ли он.

    Флаг в настройках и факт вызова — разные вещи. Харнесс задаёт КАЖДЫЙ вопрос
    в новом чате, то есть всегда первым ходом; при выключенном
    `condense_first_turn` переписывание вопроса не запускается ни разу, и
    прогон меряет пайплайн без одного из двух скрытых вызовов. Отчёт при этом
    честно писал «да» — формально верно и полностью вводит в заблуждение.
    """
    enabled = rag_cfg.get("condense_enabled")
    stage = (report.get("stage_timings") or {}).get("condense")
    if not enabled or not isinstance(stage, dict):
        # Стадия не размечена вовсе (старый отчёт, прогон без rag_log) —
        # «не знаю» лучше уверенной лжи в любую сторону.
        return enabled
    ran = stage.get("max_ms")
    if isinstance(ran, (int, float)) and ran > 0:
        return enabled
    return CONDENSE_NOT_CALLED


def _render_run_params(report: dict[str, Any]) -> list[str]:
    """Секция «Параметры прогона» — всё, без чего результат не повторить."""
    params = report.get("run_params") or {}
    lines = ["## Параметры прогона", ""]
    lines.append(
        "Дельта между прогонами имеет смысл, только если всё ниже совпадает "
        "(кроме того, что вы намеренно меняете)."
    )
    lines.append("")
    lines.append("| Параметр | Значение |")
    lines.append("|---|---|")
    ui = params.get("ui_settings")
    rows: list[tuple[str, Any]] = [
        ("судья: модель", params.get("judge_model")),
        ("судья: температура", params.get("judge_temperature")),
        ("судья: версия промптов", params.get("judge_prompt_version")),
        ("генератор golden: версия промптов", params.get("golden_prompt_version")),
    ]
    if isinstance(ui, dict):
        giga = ui.get("gigachat") or {}
        rag_cfg = ui.get("rag") or {}
        prompts = ui.get("prompts") or {}
        # Провайдер и модель — ПО ФАКТУ (`model_effective`), не по ключу
        # `gigachat.model`, который при провайдере KitAI просто не используется.
        model = answer_model(report)
        note = f" {model['note']}" if model.get("note") and model.get("model") else ""
        rows.extend(
            [
                ("ответ: провайдер", model.get("provider") or NOT_RECORDED),
                (
                    "ответ: модель",
                    f"{model['model']}{note}" if model.get("model") else NOT_RECORDED,
                ),
                ("ответ: температура", giga.get("temperature")),
                ("ответ: max_tokens", giga.get("max_tokens")),
                ("ретрив: ширина (rerank_candidates)", rag_cfg.get("rerank_candidates")),
                ("ретрив: режим", rag_cfg.get("mode")),
                ("грейдер: включён", rag_cfg.get("grader_enabled")),
                ("грейдер: порог", rag_cfg.get("grader_threshold")),
                ("грейдер: keep_top", rag_cfg.get("grader_keep_top")),
                ("condense включён", _condense_state(report, rag_cfg)),
                ("бюджет контекста, симв.", rag_cfg.get("max_context_chars")),
                ("промпт system (отпечаток)", prompts.get("system") or "встроенный"),
                (
                    "промпт reminder (отпечаток)",
                    prompts.get("context_reminder") or "встроенный",
                ),
            ]
        )
    else:
        rows.append(("настройки UI", ui if ui else "не найдены (нет rag_log)"))
    for name, value in rows:
        text = _fmt(value)
        # Значение-фраза в бэктиках — сломанная разметка и нечитаемая строка;
        # моноширинным набирается только то, что действительно идентификатор.
        cell = text if " " in text else f"`{text}`"
        lines.append(f"| {name} | {cell} |")
    lines.append("")
    return lines


def _bucket_numbers(report: dict[str, Any]) -> dict[str, int]:
    """``{answerable, refusal, meta}`` — размеры корзин оценённых пар.

    Отчёт СТАРОГО формата ключа ``buckets`` не знает: числа восстанавливаются
    из сэмплов, чтобы диф со старым отчётом не падал на пустом месте. Он же не
    знает и корзины ``meta`` — там она честно нулевая, а не «не измерена»:
    метапар в наборе тогда не было вовсе.
    """
    buckets = report.get("buckets")
    if isinstance(buckets, dict):
        return {
            "answerable": int(buckets.get("answerable", 0) or 0),
            "refusal": int(buckets.get("refusal", 0) or 0),
            "meta": int(buckets.get("meta", 0) or 0),
        }
    ok = successful([s for s in report.get("samples", []) or [] if isinstance(s, dict)])
    return {
        "answerable": len(answerable(ok)),
        "refusal": len(refusal_rows(ok)),
        "meta": len(meta_rows(ok)),
    }


def _false_refusal_denominator(report: dict[str, Any]) -> int:
    """Сколько отвечаемых пар реально попало в ``false_refusal_rate``."""
    ok = successful([s for s in report.get("samples", []) or [] if isinstance(s, dict)])
    return sum(
        1 for s in answerable(ok) if isinstance(s.get(FALSE_REFUSAL_KEY), bool)
    )


def _hedge_denominator(report: dict[str, Any]) -> int:
    """Сколько отвечаемых пар реально стоит за `hedge_rate` (есть поле `hedged`)."""
    ok = successful([s for s in report.get("samples", []) or [] if isinstance(s, dict)])
    return sum(
        1
        for s in answerable(ok)
        if isinstance(((s.get("metrics") or {}).get("answer_relevancy_ru") or {}).get("hedged"), bool)
    )


def _render_refusal_bucket(report: dict[str, Any]) -> list[str]:
    """Таблица судейских метрик по парам-ловушкам — отдельно от общих средних."""
    aggregates = report.get("aggregate_refusal")
    n = _bucket_numbers(report)["refusal"]
    if not isinstance(aggregates, dict) or not n:
        return []  # в наборе нет ловушек — таблице из прочерков места нет
    cover = report.get("coverage_refusal") or {}
    lines = [f"## Пары-ловушки `expected_refusal` ({n})", ""]
    lines.append(
        "Эти пары НЕ входят в средние выше. Судейские метрики по ним считаются "
        "против эталона отказа и приведены только для полноты: осмысленная "
        f"оценка ветки отказа — `{REFUSAL_KEY}`, а не эти четыре числа "
        "(`answer_relevancy_ru` по построению ставит 0 правильному отказу)."
    )
    lines.append("")
    lines.append("| Метрика | Значение | Оценено пар |")
    lines.append("|---|---:|---:|")
    for name in METRIC_NAMES:
        lines.append(f"| {name} | {_fmt(aggregates.get(name))} | {cover.get(name, 0)} |")
    refused = (report.get("aggregate") or {}).get(REFUSAL_KEY)
    lines.append(f"| {REFUSAL_KEY} | {_fmt(refused)} | {n} |")
    lines.append("")
    return lines


def _render_meta_bucket(report: dict[str, Any]) -> list[str]:
    """Таблица третьей корзины — метапар (вопрос про базу или про ассистента)."""
    aggregates = report.get("aggregate_meta")
    n = _bucket_numbers(report)["meta"]
    if not isinstance(aggregates, dict) or not n:
        return []  # метапар в наборе нет — таблице из прочерков места нет
    cover = report.get("coverage_meta") or {}
    lines = [f"## Метапары `expected_outcome: meta` ({n})", ""]
    lines.append(
        "Вопрос про САМУ базу («что ты знаешь?», «о каких продуктах есть "
        "информация?») или про ассистента. Правильного документа не существует, "
        "поэтому `retrieval_hit` по ним не считается, а судейские метрики ниже "
        "приведены только для полноты: `context_precision`/`context_recall` "
        "меряли бы отсутствие разметки, а не качество ответа. Осмысленная оценка "
        f"ветки — `{META_RATE_KEY}`: доля метапар, на которые ассистент ответил, "
        "а не отказался. **Больше — лучше**; отказ здесь — тот самый дефект, "
        "ради которого корзина и заведена."
    )
    lines.append("")
    lines.append("| Метрика | Значение | Оценено пар |")
    lines.append("|---|---:|---:|")
    for name in METRIC_NAMES:
        lines.append(f"| {name} | {_fmt(aggregates.get(name))} | {cover.get(name, 0)} |")
    answered = (report.get("aggregate") or {}).get(META_RATE_KEY)
    lines.append(f"| {META_RATE_KEY} | {_fmt(answered)} | {n} |")
    lines.append("")
    return lines


#: Колонки разреза по категориям: ключ отчёта → заголовок таблицы.
_CATEGORY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("faithfulness_ru", "faith"),
    ("answer_relevancy_ru", "ans_rel"),
    ("context_precision", "ctx_prec"),
    ("context_recall", "ctx_rec"),
    (RETRIEVAL_KEY, "hit"),
    (REFUSAL_KEY, "refusal_ok"),
    (FALSE_REFUSAL_RATE_KEY, "false_ref ↓"),
    (HEDGE_RATE_KEY, "hedge ↓"),
    (META_RATE_KEY, "meta_ans"),
)


def _render_categories(report: dict[str, Any]) -> list[str]:
    """Таблица по категориям вопросов (пусто, если разреза в отчёте нет)."""
    groups = report.get("by_category")
    if not isinstance(groups, dict) or not groups:
        return []
    if set(groups) == {UNCATEGORIZED}:
        return []  # категорий в golden-set нет — строка дублировала бы средние
    lines = ["## По категориям", ""]
    lines.append(
        "Общее среднее по разнородному набору прячет ровно то, ради чего "
        "категории и заводились: одна группа проседает, вторая растёт, сумма "
        "стоит на месте. Судейские колонки, как и выше, считаются без "
        f"пар-ловушек; `{FALSE_REFUSAL_RATE_KEY}` и `{HEDGE_RATE_KEY}` — "
        "**меньше лучше**."
    )
    lines.append("")
    header = " | ".join(title for _key, title in _CATEGORY_COLUMNS)
    lines.append(f"| категория | пар | упало | {header} |")
    lines.append("|---|---:|---:|" + "---:|" * len(_CATEGORY_COLUMNS))
    for name in sorted(groups):
        entry = groups[name] if isinstance(groups[name], dict) else {}
        cells = " | ".join(_fmt(entry.get(key)) for key, _title in _CATEGORY_COLUMNS)
        lines.append(
            f"| {name} | {entry.get('n', 0)} | {entry.get('n_failed', 0)} | {cells} |"
        )
    lines.append("")
    return lines


def _render_stages(report: dict[str, Any]) -> list[str]:
    """Секция «Скрытые вызовы» — что происходило внутри хода.

    Метрики качества описывают ОТВЕТ; эта таблица описывает пайплайн, который
    его собрал. Без неё «грейдер: включён» в параметрах читается как «грейдер
    работал», и разница между этими двумя утверждениями стоила прогону
    `baseline` всего смысла.
    """
    stages = report.get("stage_timings") or {}
    grader = report.get("grader_health") or {}
    hidden = report.get("hidden_call_health") or {}
    if not stages and not grader.get("applicable") and not hidden.get("recorded"):
        return []
    lines = ["## Скрытые вызовы", ""]
    if grader.get("enabled"):
        line = (
            f"- грейдер вернул оценки на {grader.get('graded', 0)} парах из "
            f"{grader.get('applicable', 0)}"
        )
        if grader.get("partial"):
            line += f" (из них частично, батчами: {grader['partial']})"
        lines.append(line)
        if grader.get("ungraded"):
            lines.append(
                f"  - на остальных {grader['ungraded']} контекст собран сырым "
                "порядком поиска: порог и `keep_top` к ним не применялись"
            )
    lines.extend(_render_hidden_calls(report))
    if stages:
        lines.append("")
        lines.append("| стадия | медиана, мс | максимум, мс | поводок, мс | упёрлось в поводок |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, entry in stages.items():
            deadline = entry.get("deadline_ms")
            hits = entry.get("at_deadline")
            # У стадии без известного поводка нет и знаменателя: «— из 46»
            # выглядело бы как измерение, которого не было.
            hit_cell = "—" if deadline is None else f"{_fmt(hits)} из {entry.get('n')}"
            lines.append(
                f"| {name} | {entry.get('median_ms')} | {entry.get('max_ms')} | "
                f"{_fmt(deadline)} | {hit_cell} |"
            )
        lines.append("")
        lines.append(
            "Время, совпавшее с поводком, — это таймаут, а не работа: стадия не "
            "«успела ровно за столько», её на этом месте оборвали."
        )
    lines.append("")
    return lines


def _md_cell(text: Any, *, cap: int = 160) -> str:
    """Текст в ячейку таблицы: без переводов строк и вертикальных черт, с капом."""
    value = " ".join(str(text if text is not None else "").split()).replace("|", "\\|")
    return value[:cap] + "…" if len(value) > cap else value


def _render_hidden_calls(report: dict[str, Any]) -> list[str]:
    """Исходы скрытых вызовов по записям `hidden_calls` — или честное «не записано».

    Две таблицы про грейдер: «батчи по исходу» (сколько батчей отработало,
    упало, обрезано) и «причины сбоев» (тип ошибки → число батчей → пример
    `detail`). Строка про condense — тем же образом, но короче: у него один
    вызов на ход.
    """
    hidden = report.get("hidden_call_health")
    if not isinstance(hidden, dict):
        return [f"- скрытые вызовы: {NOT_RECORDED} (отчёт прежней версии харнесса)"]
    recorded = int(hidden.get("recorded", 0) or 0)
    not_recorded = int(hidden.get("not_recorded", 0) or 0)
    if not recorded:
        return [
            f"- скрытые вызовы: {NOT_RECORDED} — UI старее сбора `hidden_calls`, "
            "причины сбоев грейдера и condense по этому прогону неизвестны"
        ]
    lines: list[str] = []
    if not_recorded:
        lines.append(
            f"- скрытые вызовы записаны на {recorded} парах из "
            f"{recorded + not_recorded}; у остальных — {NOT_RECORDED}"
        )
    grader = hidden.get("grader") or {}
    total = int(grader.get("batches_total", 0) or 0)
    if grader.get("calls"):
        lines.append("")
        lines.append("**грейдер: батчи по исходу**")
        lines.append("")
        lines.append("| исход | батчей |")
        lines.append("|---|---:|")
        for key, title in (
            ("batches_ok", "ok"),
            ("batches_failed", "failed"),
            ("batches_truncated", "truncated"),
            ("batches_partial", "partial"),
        ):
            lines.append(f"| {title} | {int(grader.get(key, 0) or 0)} |")
        lines.append(f"| всего | {total} |")
        lines.append("")
        tail = []
        if grader.get("ms_median") is not None:
            tail.append(
                f"время батча: медиана {grader['ms_median']} мс, максимум {grader['ms_max']} мс"
            )
        if grader.get("by_model"):
            tail.append(
                "модель: " + ", ".join(f"`{m}` × {n}" for m, n in grader["by_model"].items())
            )
        if grader.get("by_finish_reason"):
            tail.append(
                "finish_reason: "
                + ", ".join(f"`{r}` × {n}" for r, n in grader["by_finish_reason"].items())
            )
        if grader.get("graded") or grader.get("omitted"):
            tail.append(
                f"оценено кандидатов: {grader.get('graded', 0)}, "
                f"пропущено грейдером: {grader.get('omitted', 0)}"
            )
        if tail:
            lines.append("; ".join(tail) + ".")
        by_error = grader.get("by_error") or {}
        if by_error:
            examples = grader.get("examples") or {}
            models = grader.get("by_error_model") or {}
            lines.append("")
            lines.append("**грейдер: причины сбоев**")
            lines.append("")
            lines.append("| тип ошибки | модель | батчей | пример detail |")
            lines.append("|---|---|---:|---|")
            for key, count in by_error.items():
                sample_detail = (examples.get(key) or [""])[0]
                model_cell = ", ".join(models.get(key) or []) or "—"
                lines.append(
                    f"| {_md_cell(key)} | {_md_cell(model_cell)} | {count} "
                    f"| {_md_cell(sample_detail) or '—'} |"
                )
        elif total and not grader.get("batches_ok") == total:
            lines.append("")
            lines.append(
                "Не все батчи `ok`, но ни одной записанной ошибки — смотрите "
                "`finish_reason` и `by_status` в JSON."
            )
    else:
        lines.append("- грейдер: ни одного вызова в записях (`grader: null` на всех парах)")
    if lines and lines[-1] != "" and not lines[-1].startswith("- "):
        lines.append("")  # иначе список ниже прилипает к таблице
    condense = hidden.get("condense") or {}
    if condense.get("calls"):
        statuses = ", ".join(f"`{s}` × {n}" for s, n in (condense.get("by_status") or {}).items())
        line = f"- condense: вызовов {condense['calls']} — {statuses}"
        if condense.get("ms_median") is not None:
            line += f"; медиана {condense['ms_median']} мс, максимум {condense['ms_max']} мс"
        lines.append(line)
        examples = condense.get("examples") or {}
        for key, count in (condense.get("by_error") or {}).items():
            sample_detail = (examples.get(key) or [""])[0]
            suffix = f" — например: {_md_cell(sample_detail)}" if sample_detail else ""
            lines.append(f"  - {count} × `{_md_cell(key)}`{suffix}")
    else:
        lines.append("- condense: ни одного вызова в записях (`condense: null` на всех парах)")
    return lines


def _render_generation_failures(report: dict[str, Any]) -> list[str]:
    """Секция «Сбои генерации» — пары, где ответа не было вовсе."""
    rows = report.get("generation_failures")
    if not isinstance(rows, list):
        rows = generation_failures(
            [s for s in report.get("samples", []) or [] if isinstance(s, dict)]
        )
    if not rows:
        return []
    lines = [f"## Сбои генерации ({len(rows)}) — исключены из средних", ""]
    lines.append(
        "Пустой ответ при живом ретриве: судья не вызывался, `metrics: null`, "
        "флаги отказа не выставлены. `retrieval_hit` у этих пар посчитан."
    )
    lines.append("")
    lines.append("| id | finish_reason | stream, мс | модель |")
    lines.append("|---|---|---:|---|")
    for row in rows:
        lines.append(
            f"| {row.get('id')} | {_fmt(row.get('finish_reason'))} "
            f"| {_fmt(row.get('stream_ms'))} | {_md_cell(row.get('model'))} |"
        )
    lines.append("")
    return lines


def _render_path_drift(report: dict[str, Any]) -> list[str]:
    """Секция «Дрейф путей golden» — расхождения разметки с живым каталогом."""
    drift = report.get("path_drift")
    if not isinstance(drift, dict):
        return []
    drifted = drift.get("drifted") or []
    ambiguous = drift.get("ambiguous") or []
    missing = drift.get("missing") or []
    if not (drifted or ambiguous or missing):
        return []
    lines = ["## Дрейф путей golden", ""]
    lines.append(
        f"Пути golden-набора сверены с живым каталогом (`GET /api/vault/catalog`) "
        f"у {drift.get('checked', 0)} пар. Сопоставленные по имени файла пути "
        "на время прогона добавлены в `alt_source_paths` пары; неоднозначные и "
        "отсутствующие оставлены как есть. Поправьте `golden.jsonl`."
    )
    lines.append("")
    if drifted:
        lines.append("**сопоставлены по имени файла**")
        lines.append("")
        lines.append("| id | в golden | в каталоге |")
        lines.append("|---|---|---|")
        for row in drifted:
            lines.append(
                f"| {row.get('id')} | `{_md_cell(row.get('golden'))}` "
                f"| `{_md_cell(row.get('live'))}` |"
            )
        lines.append("")
    if ambiguous:
        lines.append("**неоднозначны (несколько файлов с таким именем)**")
        lines.append("")
        lines.append("| id | в golden | кандидаты |")
        lines.append("|---|---|---|")
        for row in ambiguous:
            candidates = ", ".join(f"`{_md_cell(c)}`" for c in (row.get("candidates") or []))
            lines.append(f"| {row.get('id')} | `{_md_cell(row.get('golden'))}` | {candidates} |")
        lines.append("")
    if missing:
        lines.append("**не найдены в каталоге**")
        lines.append("")
        for row in missing:
            lines.append(f"- `{row.get('id')}`: `{_md_cell(row.get('golden'))}`")
        lines.append("")
    return lines


def _render_judge_health(report: dict[str, Any]) -> list[str]:
    """Строка о несостоявшихся судейских вызовах — печатается ВСЕГДА.

    В том числе с нулём: «0 из 188» это утверждение о прогоне, а отсутствие
    строки читалось бы как «не мерили».
    """
    health = report.get("judge_failures") or {}
    expected = health.get("expected", 0)
    failed = health.get("failed", 0)
    if not expected:
        return []
    line = f"- судейских вызовов не вернулось: **{failed}** из {expected}"
    if failed:
        line += f" (затронуто пар: {health.get('samples_affected', 0)})"
    lines = [line]
    for reason, count in (health.get("by_error") or {}).items():
        lines.append(f"  - {count} × `{reason}`")
    return lines


def render_report_md(report: dict[str, Any], *, max_rows: int = 200) -> str:
    """Render the markdown report: disclaimer → aggregates → per-sample table."""
    counts = report.get("counts", {})
    buckets = _bucket_numbers(report)
    aggregates = report.get("aggregate", {})
    cover = report.get("coverage", {})
    spread = report.get("dispersion", {}) or {}
    lines: list[str] = []
    title = f"# RAG eval — прогон `{report.get('label')}`"
    if grader_degraded(report.get("grader_health")):
        title += f" {DEGRADED_TITLE_SUFFIX}"
    lines.append(title)
    lines.append("")
    lines.append(f"> {REPORT_DISCLAIMER}")
    lines.append("")
    if report.get("approximate"):
        lines.append(f"> {APPROXIMATE_WARNING}")
        lines.append("")
    drift = report.get("path_drift") or {}
    if any(drift.get(key) for key in ("drifted", "ambiguous", "missing")):
        lines.append(
            "> "
            + PATH_DRIFT_WARNING.format(
                drifted=len(drift.get("drifted") or []),
                ambiguous=len(drift.get("ambiguous") or []),
                missing=len(drift.get("missing") or []),
            )
        )
        lines.append("")
    gen_failed = int(counts.get(GENERATION_FAILED_KEY, 0) or 0)
    if gen_failed:
        lines.append("> " + GENERATION_FAILED_NOTE.format(count=gen_failed))
        lines.append("")
    degradation = report.get("retrieval_degradation") or {}
    if degradation.get("degraded"):
        lines.append(
            "> "
            + GRANULARITY_WARNING.format(
                degraded=degradation.get("degraded", 0),
                measured=degradation.get("measured", 0),
            )
        )
        lines.append("")
    clipped = int((report.get("counts") or {}).get("context_clipped", 0) or 0)
    if clipped:
        lines.append(
            "> "
            + CLIPPED_CONTEXT_WARNING.format(
                clipped=clipped, total=(report.get("counts") or {}).get("total", 0)
            )
        )
        lines.append("")
    judge_clipped_ids = [
        str(s.get("id"))
        for s in report.get("samples", []) or []
        if isinstance(s, dict) and not is_failed(s) and is_judge_clipped(s)
    ]
    if judge_clipped_ids:
        shown = ", ".join(f"`{i}`" for i in judge_clipped_ids[:30])
        if len(judge_clipped_ids) > 30:
            shown += f" и ещё {len(judge_clipped_ids) - 30}"
        lines.append(
            "> " + JUDGE_CLIP_WARNING.format(clipped=len(judge_clipped_ids), ids=shown)
        )
        lines.append("")
    grader = report.get("grader_health") or {}
    if grader.get("enabled") and grader.get("ungraded"):
        warning = GRADER_SILENT_WARNING.format(
            graded=grader.get("graded", 0),
            applicable=grader.get("applicable", 0),
        )
        cause = grader_cause_clause(report)
        if cause:
            warning += f" {cause}."
        lines.append("> " + warning)
        lines.append("")
    judge_health = report.get("judge_failures") or {}
    if judge_health.get("failed"):
        lines.append(
            "> "
            + JUDGE_FAILURE_WARNING.format(
                failed=judge_health.get("failed", 0),
                expected=judge_health.get("expected", 0),
                affected=judge_health.get("samples_affected", 0),
            )
        )
        lines.append("")
    lines.append(f"- дата: `{report.get('generated_at')}`")
    lines.append(f"- golden-set: `{report.get('golden')}`")
    lines.append(f"- UI: `{report.get('ui_url')}`")
    lines.append(
        f"- судья: `{report.get('judge_model')}`, "
        f"промпты: `{report.get('prompt_version')}`"
    )
    model = answer_model(report)
    lines.append(
        f"- ответ: провайдер `{model.get('provider') or NOT_RECORDED}`, "
        f"модель `{model.get('model') or NOT_RECORDED}`"
        + (f" {model['note']}" if model.get("note") and model.get("model") else "")
    )
    lines.append(
        f"- пар: {counts.get('total', 0)} "
        f"(оценено {counts.get('evaluated', 0)}, ошибок {counts.get('failed', 0)}, "
        f"сбоев генерации {counts.get(GENERATION_FAILED_KEY, 0) or 0})"
    )
    lines.append(
        f"- **упало и исключено из средних: {counts.get('failed', 0)}** "
        "(сбой чата или недоступный текст контекста — не качество)"
    )
    lines.extend(_render_judge_health(report))
    lines.append(f"- {_judge_calls_line((report.get('run_params') or {}).get('judge_calls_by_metric'))}")
    lines.append(
        "- кап судьи: нет"
        if not judge_clipped_ids
        else f"- кап судьи: **{len(judge_clipped_ids)}** пар(ы) — см. предупреждение выше"
    )
    lines.append(f"- источник контекста: `{report.get('context_origin', {})}`")
    lines.append(
        f"- гранулярность `retrieval_hit`: `{report.get('retrieval_granularity', {})}`"
    )
    lines.append(
        f"- отвечаемых пар: {buckets.get('answerable', 0)}, "
        f"пар-ловушек `expected_refusal`: {buckets.get('refusal', 0)}, "
        f"метапар `expected_outcome: meta`: {buckets.get('meta', 0)}, "
        f"сбоев генерации: {counts.get(GENERATION_FAILED_KEY, 0) or 0}"
    )
    lines.append("")
    lines.extend(_render_run_params(report))
    lines.extend(_render_stages(report))
    lines.extend(_render_path_drift(report))
    lines.append("## Средние значения")
    lines.append("")
    lines.append(BUCKET_NOTE.format(**_bucket_numbers(report)))
    lines.append("")
    lines.append("| Метрика | Значение | Разброс по сэмплам | Оценено пар |")
    lines.append("|---|---:|---|---:|")
    for name in METRIC_NAMES:
        lines.append(
            f"| {name} | {_fmt(aggregates.get(name))} "
            f"| {_fmt_spread(spread.get(name))} | {cover.get(name, 0)} |"
        )
    lines.append(
        f"| {RETRIEVAL_KEY} (доля успешных пар, где нужный фрагмент попал в контекст; "
        f"гранулярность — {_granularity_label(report)}) "
        f"| {_fmt(aggregates.get(RETRIEVAL_KEY))} | — | {counts.get('evaluated', 0)} |"
    )
    lines.append(
        f"| {REFUSAL_KEY} (доля пар «ответа нет в корпусе», где был отказ) "
        f"| {_fmt(aggregates.get(REFUSAL_KEY))} | — | "
        f"{sum(1 for s in report.get('samples', []) if s.get('expected_refusal'))} |"
    )
    lines.append(
        f"| {FALSE_REFUSAL_RATE_KEY} ↓ (доля ОТВЕЧАЕМЫХ пар, где ассистент "
        "отказался зря; **меньше — лучше**) "
        f"| {_fmt(aggregates.get(FALSE_REFUSAL_RATE_KEY))} | — | "
        f"{_false_refusal_denominator(report)} |"
    )
    lines.append(
        f"| {HEDGE_RATE_KEY} ↓ (доля ОТВЕЧАЕМЫХ пар: ответ есть, но открывается "
        "оговоркой об отсутствии ответа; **меньше — лучше**) "
        f"| {_fmt(aggregates.get(HEDGE_RATE_KEY))} | — | "
        f"{_hedge_denominator(report)} |"
    )
    lines.append(
        f"| {META_RATE_KEY} (доля МЕТАПАР, на которые ассистент ответил, а не "
        "отказался; **больше — лучше**) "
        f"| {_fmt(aggregates.get(META_RATE_KEY))} | — | "
        f"{_bucket_numbers(report)['meta']} |"
    )
    lines.append("")
    lines.append(FALSE_REFUSAL_NOTE)
    lines.append("")
    lines.extend(_render_refusal_bucket(report))
    lines.extend(_render_meta_bucket(report))
    lines.extend(_render_categories(report))
    lines.append(DIAGNOSTIC_RULE)
    lines.append("")
    lines.append("## По парам")
    lines.append("")
    lines.append(
        "| id | тип | категория | чанк найден | зря отказ | грейдер | faith | ans_rel "
        "| ctx_prec | ctx_rec | вопрос |"
    )
    lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---|")
    samples = list(report.get("samples", []))
    for sample in samples[:max_rows]:
        metrics = sample.get("metrics") or {}

        def score(name: str) -> str:
            entry = metrics.get(name) or {}
            return _fmt(entry.get("score") if isinstance(entry, dict) else None)

        question = str(sample.get("question", "")).replace("|", "\\|")
        if len(question) > 90:
            question = question[:90] + "…"
        lines.append(
            f"| {sample.get('id')} | {sample.get('kind')} "
            f"| {category_of(sample)} "
            f"| {_fmt(sample.get(RETRIEVAL_KEY))} "
            f"| {_fmt(sample.get(FALSE_REFUSAL_KEY))} "
            f"| {grader_cell(sample)} "
            f"| {score('faithfulness_ru')} | {score('answer_relevancy_ru')} "
            f"| {score('context_precision')} | {score('context_recall')} "
            f"| {question} |"
        )
    if len(samples) > max_rows:
        lines.append("")
        lines.append(f"_…ещё {len(samples) - max_rows} пар — см. JSON-отчёт._")

    lines.append("")
    lines.extend(_render_generation_failures(report))
    failed = [s for s in samples if is_failed(s)]
    if failed:
        lines.append(f"## Упавшие пары ({len(failed)}) — исключены из средних")
        lines.append("")
        for sample in failed[:50]:
            lines.append(f"- `{sample.get('id')}`: {sample.get('error')}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Paired comparison
# --------------------------------------------------------------------------- #


def sample_scores(report: dict[str, Any], name: str) -> dict[str, float]:
    """``{sample_id: score}`` по успешным сэмплам — основа парного сравнения."""
    out: dict[str, float] = {}
    for sample in report.get("samples", []) or []:
        if not isinstance(sample, dict) or is_failed(sample) or is_generation_failed(sample):
            continue
        ident = sample.get("id")
        entry = (sample.get("metrics") or {}).get(name) or {}
        score = entry.get("score") if isinstance(entry, dict) else None
        if ident is not None and isinstance(score, (int, float)) and not isinstance(
            score, bool
        ):
            out[str(ident)] = float(score)
    return out


def paired_delta(
    report_a: dict[str, Any], report_b: dict[str, Any], name: str
) -> dict[str, Any]:
    """Парная дельта метрики: те же вопросы в обоих прогонах.

    Разность средних смешивает изменение качества с изменением СОСТАВА
    оценённых пар (в одном прогоне судья не ответил на три вопроса, в другом —
    на другие три). Парная дельта считается по пересечению id, поэтому состав
    из уравнения уходит, а остаток — собственно эффект.

    ``{delta, sd, n, stderr, значимость}``; ``n`` — число ПАР, а не сэмплов.
    """
    a = sample_scores(report_a, name)
    b = sample_scores(report_b, name)
    common = sorted(set(a) & set(b))
    deltas = [b[i] - a[i] for i in common]
    n = len(deltas)
    if not n:
        return {"delta": None, "sd": None, "n": 0, "stderr": None}
    mean = sum(deltas) / n
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    stderr = sd / math.sqrt(n) if n > 1 else None
    return {
        "delta": round(mean, 4),
        "sd": round(sd, 4),
        "n": n,
        "stderr": round(stderr, 4) if stderr is not None else None,
    }


def delta_sign(delta: float | None, stderr: float | None, noise: float) -> str:
    """▲/▼ только когда сдвиг больше и шума судьи, и двух стандартных ошибок.

    Раньше знак ставился по одной константе 0.02 — при разбросе 0.3 по 12 парам
    это регулярно объявляло сигналом обычную дрожь судьи.
    """
    if delta is None:
        return "—"
    band = noise if stderr is None else max(noise, 2.0 * stderr)
    if abs(delta) < band:
        return "≈"
    return "▲" if delta > 0 else "▼"


#: Метрики, где РОСТ — это ухудшение. Знак ▲/▼ в дифе показывает качество,
#: поэтому для них он инвертируется относительно знака самой дельты.
LOWER_IS_BETTER = (FALSE_REFUSAL_RATE_KEY, HEDGE_RATE_KEY)


def quality_sign(
    name: str, delta: float | None, stderr: float | None, noise: float
) -> str:
    """``delta_sign``, развёрнутый в сторону КАЧЕСТВА, а не величины числа."""
    oriented = None if delta is None else (-delta if name in LOWER_IS_BETTER else delta)
    return delta_sign(oriented, stderr, noise)


def category_sets(
    report_a: dict[str, Any], report_b: dict[str, Any]
) -> dict[str, list[str]]:
    """``{common, only_a, only_b}`` — категории обоих прогонов.

    Категории берутся из данных, а данные между прогонами меняются: если набор
    разъехался, сравнивать построчно можно только пересечение, а про остальные
    диф обязан сказать вслух — молча выброшенная категория читается как
    «её не было».
    """
    a = report_a.get("by_category") or {}
    b = report_b.get("by_category") or {}
    return {
        "common": sorted(set(a) & set(b)),
        "only_a": sorted(set(a) - set(b)),
        "only_b": sorted(set(b) - set(a)),
    }


def _legacy_keys(report: dict[str, Any]) -> list[str]:
    """Ключи нового формата, которых в отчёте нет (сделан старым харнессом)."""
    missing: list[str] = []
    if not isinstance(report.get("by_category"), dict):
        missing.append("by_category")
    if not isinstance(report.get("aggregate_refusal"), dict):
        missing.append("aggregate_refusal")
    # Именно ОТСУТСТВИЕ ключа, а не `None`: в новом формате `None` — законное
    # значение (в наборе нет ни одной отвечаемой пары).
    if FALSE_REFUSAL_RATE_KEY not in (report.get("aggregate") or {}):
        missing.append(FALSE_REFUSAL_RATE_KEY)
    return missing


def _render_legacy_note(report_a: dict[str, Any], report_b: dict[str, Any]) -> list[str]:
    """Предупредить, что часть разрезов недоступна из-за старого отчёта."""
    lines: list[str] = []
    for report in (report_a, report_b):
        missing = _legacy_keys(report)
        if not missing:
            continue
        lines.append(
            f"> **Отчёт `{report.get('label', '?')}` сделан прежней версией "
            "харнесса:** в нём нет "
            + ", ".join(f"`{key}`" for key in missing)
            + ". Соответствующие строки ниже показаны как `—`, а не додуманы. "
            "Важное следствие: его судейские средние посчитаны ВМЕСТЕ с "
            "парами-ловушками, поэтому они не сопоставимы со средними нового "
            "формата напрямую — перегоните прогон, если нужна честная дельта."
        )
        lines.append("")
    return lines


def _render_category_compare(
    report_a: dict[str, Any], report_b: dict[str, Any], *, noise: float
) -> list[str]:
    """Дельты по категориям + честный список категорий, которых нет в паре."""
    sets = category_sets(report_a, report_b)
    if not any(sets.values()):
        return []
    if set(sum(sets.values(), [])) == {UNCATEGORIZED}:
        return []  # ни в одном прогоне категорий нет — сравнивать нечего
    label_a = str(report_a.get("label", "A"))
    label_b = str(report_b.get("label", "B"))
    groups_a = report_a.get("by_category") or {}
    groups_b = report_b.get("by_category") or {}

    # Отчёт старого формата категорий не знает вовсе — это НЕ «набор категорий
    # изменился», и выдавать его категории за «пропавшие» нельзя.
    stale = [
        str(report.get("label", "?"))
        for report in (report_a, report_b)
        if not isinstance(report.get("by_category"), dict)
    ]
    if stale:
        return [
            "## По категориям",
            "",
            "Разбивка недоступна: в отчёте "
            + ", ".join(f"`{name}`" for name in stale)
            + " нет ключа `by_category` (сделан прежней версией харнесса). "
            "Категории второго прогона показывать в одиночку бессмысленно — "
            "сравнивать их не с чем.",
            "",
        ]

    lines = ["## По категориям", ""]
    lines.append(
        "Дельты здесь — разность СРЕДНИХ по категории, а не парная: внутри "
        "категории пар мало, и такое число слабее общей парной дельты. "
        f"Знак показывает качество (`{FALSE_REFUSAL_RATE_KEY}` — меньше лучше, "
        "знак для него инвертирован)."
    )
    lines.append("")
    if sets["common"]:
        header = " | ".join(title for _key, title in _CATEGORY_COLUMNS)
        lines.append(f"| категория | пар {label_a}→{label_b} | {header} |")
        lines.append("|---|---:|" + "---:|" * len(_CATEGORY_COLUMNS))
        for name in sets["common"]:
            entry_a = groups_a.get(name) or {}
            entry_b = groups_b.get(name) or {}
            cells: list[str] = []
            for key, _title in _CATEGORY_COLUMNS:
                value_a, value_b = entry_a.get(key), entry_b.get(key)
                if isinstance(value_a, (int, float)) and isinstance(
                    value_b, (int, float)
                ):
                    delta = round(float(value_b) - float(value_a), 4)
                    cells.append(
                        f"{delta:+.3f} {quality_sign(key, delta, None, noise)}"
                    )
                else:
                    cells.append("—")
            lines.append(
                f"| {name} | {entry_a.get('n', 0)}→{entry_b.get('n', 0)} "
                f"| {' | '.join(cells)} |"
            )
        lines.append("")
    else:
        lines.append(
            "Общих категорий у прогонов нет — сравнивать построчно нечего."
        )
        lines.append("")
    if sets["only_a"] or sets["only_b"]:
        only_a = ", ".join(f"`{c}`" for c in sets["only_a"]) or "—"
        only_b = ", ".join(f"`{c}`" for c in sets["only_b"]) or "—"
        lines.append(
            "> **ВНИМАНИЕ: набор категорий между прогонами изменился.** Только "
            f"в `{label_a}`: {only_a}. Только в `{label_b}`: {only_b}. Эти "
            "категории в таблицу выше не попали (сравнивать не с чем), а раз "
            "состав набора разъехался — общие средние и парная дельта тоже "
            "считаны по РАЗНЫМ наборам вопросов и недостоверны."
        )
        lines.append("")
    return lines


def render_compare_md(
    report_a: dict[str, Any], report_b: dict[str, Any], *, noise: float = 0.02
) -> str:
    """Markdown diff table between two reports.

    Каждая метрика показывается со своим разбросом в обоих прогонах, парной
    дельтой по общим вопросам, числом пар и знаком, который учитывает
    стандартную ошибку, а не только фиксированный порог ``noise`` (он остаётся
    нижней границей: судья дрожит и на больших выборках).
    """
    label_a = str(report_a.get("label", "A"))
    label_b = str(report_b.get("label", "B"))
    agg_a = report_a.get("aggregate", {}) or {}
    agg_b = report_b.get("aggregate", {}) or {}
    spread_a = report_a.get("dispersion", {}) or {}
    spread_b = report_b.get("dispersion", {}) or {}

    lines: list[str] = []
    lines.append(f"# Сравнение прогонов: `{label_a}` → `{label_b}`")
    lines.append("")
    # Самое громкое — первым: разные модели или мёртвый реранкер обесценивают
    # всю таблицу ниже, и читать её дальше первого экрана уже незачем.
    lines.extend(_render_compare_guards(report_a, report_b))
    lines.append(f"> {REPORT_DISCLAIMER}")
    lines.append("")
    if report_a.get("approximate") or report_b.get("approximate"):
        lines.append(f"> {APPROXIMATE_WARNING}")
        lines.append("")
    if _granularity_label(report_a) != _granularity_label(report_b):
        lines.append(
            "> **ВНИМАНИЕ:** `retrieval_hit` в прогонах измерен с разной "
            f"точностью (`{_granularity_label(report_a)}` vs "
            f"`{_granularity_label(report_b)}`) — доля попаданий сравнима "
            "только при одинаковой гранулярности, её дельта ниже недостоверна."
        )
        lines.append("")
    if report_a.get("prompt_version") != report_b.get("prompt_version"):
        lines.append(
            "> **ВНИМАНИЕ:** прогоны сделаны разными версиями судейских промптов "
            f"(`{report_a.get('prompt_version')}` vs "
            f"`{report_b.get('prompt_version')}`) — дельта недостоверна."
        )
        lines.append("")
    if report_a.get("golden") != report_b.get("golden"):
        lines.append(
            "> **ВНИМАНИЕ:** прогоны сделаны на разных golden-set "
            f"(`{report_a.get('golden')}` vs `{report_b.get('golden')}`) — "
            "дельта недостоверна."
        )
        lines.append("")
    if _params_differ(report_a, report_b):
        lines.append(
            "> **ВНИМАНИЕ:** различаются параметры прогонов (модель/температура/"
            "ширина ретрива/порог грейдера/промпты) — см. «Параметры прогонов» "
            "ниже; дельта отражает их сумму, а не одно изменение."
        )
        lines.append("")
    buckets_a = _bucket_numbers(report_a)
    buckets_b = _bucket_numbers(report_b)
    if buckets_a != buckets_b:
        lines.append(
            "> **ВНИМАНИЕ: состав набора разный.** Отвечаемых пар "
            f"{buckets_a['answerable']} → {buckets_b['answerable']}, "
            f"пар-ловушек {buckets_a['refusal']} → {buckets_b['refusal']}. "
            "Судейские средние считаются только по отвечаемым парам, поэтому "
            "сравнивать их можно лишь при совпадении обоих чисел."
        )
        lines.append("")
    lines.extend(_render_legacy_note(report_a, report_b))

    lines.append(
        f"| Метрика | {label_a} | {label_b} | Δ (парная) | ±sd | пар | Знак |"
    )
    lines.append("|---|---|---|---:|---:|---:|:--:|")
    for name in METRIC_NAMES:
        pair = paired_delta(report_a, report_b, name)
        delta = pair["delta"]
        delta_text = f"{delta:+.3f}" if delta is not None else "—"
        sd_text = f"{pair['sd']:.3f}" if pair["sd"] is not None else "—"
        lines.append(
            f"| {name} | {_fmt_spread(spread_a.get(name))} "
            f"| {_fmt_spread(spread_b.get(name))} | {delta_text} | {sd_text} "
            f"| {pair['n']} | {delta_sign(delta, pair['stderr'], noise)} |"
        )
    # Доли (hit/refusal) парного разложения не имеют — только среднее по прогону.
    for name in (
        RETRIEVAL_KEY,
        REFUSAL_KEY,
        FALSE_REFUSAL_RATE_KEY,
        HEDGE_RATE_KEY,
        META_RATE_KEY,
    ):
        value_a, value_b = agg_a.get(name), agg_b.get(name)
        if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
            delta = round(float(value_b) - float(value_a), 4)
            delta_text = f"{delta:+.3f}"
            sign = quality_sign(name, delta, None, noise)
        else:
            delta_text, sign = "—", "—"
        title = f"{name} ↓ (меньше — лучше)" if name in LOWER_IS_BETTER else name
        lines.append(
            f"| {title} | {_fmt(value_a)} | {_fmt(value_b)} | {delta_text} | — | — "
            f"| {sign} |"
        )
    lines.append("")
    lines.append(
        "«пар» — число вопросов, оценённых В ОБОИХ прогонах; дельта считается "
        "только по ним, поэтому смена состава оценённых пар её не искажает. "
        "Знак ▲/▼ ставится, когда |Δ| больше и порога шума "
        f"({noise:.2f}), и двух стандартных ошибок парной дельты."
    )
    lines.append("")
    lines.append(
        f"Знак означает КАЧЕСТВО, а не направление числа: у `{FALSE_REFUSAL_RATE_KEY}` "
        "(доля отвечаемых вопросов, на которых ассистент зря отказался) и у "
        f"`{HEDGE_RATE_KEY}` (ответ есть, но открыт оговоркой об отсутствии ответа) "
        "меньше — лучше, поэтому ▲ у них стоит при ПАДЕНИИ доли. Четыре судейские "
        "метрики выше посчитаны без пар-ловушек `expected_refusal` — правильный "
        "отказ больше не тянет средние вниз."
    )
    lines.append("")
    counts_a = report_a.get("counts", {}) or {}
    counts_b = report_b.get("counts", {}) or {}
    lines.append(
        f"Пар: {counts_a.get('total', 0)} → {counts_b.get('total', 0)}; "
        f"упало (исключено из средних): {counts_a.get('failed', 0)} → "
        f"{counts_b.get('failed', 0)}; сбоев генерации (тоже вне средних и вне "
        f"парной дельты): {counts_a.get(GENERATION_FAILED_KEY, 0) or 0} → "
        f"{counts_b.get(GENERATION_FAILED_KEY, 0) or 0}."
    )
    lines.append("")
    lines.extend(_render_category_compare(report_a, report_b, noise=noise))
    lines.append("## Параметры прогонов")
    lines.append("")
    lines.append(f"### `{label_a}`")
    lines.append("")
    lines.extend(_render_run_params(report_a)[2:])
    lines.append(f"### `{label_b}`")
    lines.append("")
    lines.extend(_render_run_params(report_b)[2:])
    lines.append(DIAGNOSTIC_RULE)
    lines.append("")
    return "\n".join(lines)


def _render_compare_guards(report_a: dict[str, Any], report_b: dict[str, Any]) -> list[str]:
    """Громкие блоки в самом верху дифа: разные модели, мёртвый реранкер."""
    lines: list[str] = []
    label_a = str(report_a.get("label", "A"))
    label_b = str(report_b.get("label", "B"))
    if model_mismatch(report_a, report_b):
        lines.append(
            "> "
            + MODEL_MISMATCH_WARNING.format(
                label_a=label_a,
                label_b=label_b,
                model_a=_model_label(answer_model(report_a)),
                model_b=_model_label(answer_model(report_b)),
            )
        )
        lines.append("")
    if judge_prompt_mismatch(report_a, report_b):
        lines.append(
            "> "
            + JUDGE_PROMPT_MISMATCH_WARNING.format(
                label_a=label_a,
                label_b=label_b,
                version_a=judge_prompt_version(report_a),
                version_b=judge_prompt_version(report_b),
            )
        )
        lines.append("")
    for report in (report_a, report_b):
        model = answer_model(report)
        if not model.get("model") and not model.get("provider"):
            lines.append(
                "> " + MODEL_UNKNOWN_WARNING.format(label=str(report.get("label", "?")))
            )
            lines.append("")
    for report in (report_a, report_b):
        health = report.get("grader_health") or {}
        if grader_degraded(health):
            lines.append(
                "> "
                + DEGRADED_COMPARE_WARNING.format(
                    label=str(report.get("label", "?")),
                    graded=health.get("graded", 0),
                    applicable=health.get("applicable", 0),
                    threshold=GRADER_DEGRADED_THRESHOLD,
                )
            )
            lines.append("")
    return lines


def compare_blockers(
    report_a: dict[str, Any],
    report_b: dict[str, Any],
    *,
    allow_model_mismatch: bool = False,
    allow_degraded: bool = False,
) -> list[str]:
    """Почему `--compare` отказывается (пусто — можно сравнивать).

    Разные модели ответа и реранкер, не работавший хотя бы в одном прогоне, —
    это сравнение двух СИСТЕМ, а не двух настроек. Диф печатается только с
    явным флагом, и флаг остаётся в имени файла-дифа на совести того, кто его
    поставил.
    """
    out: list[str] = []
    if model_mismatch(report_a, report_b) and not allow_model_mismatch:
        out.append(
            "отвечали разные модели: "
            f"`{report_a.get('label')}` — {_model_label(answer_model(report_a))}, "
            f"`{report_b.get('label')}` — {_model_label(answer_model(report_b))} "
            "(продолжить: --allow-model-mismatch)"
        )
    if judge_prompt_mismatch(report_a, report_b) and not allow_model_mismatch:
        out.append(
            "судили разные версии промптов: "
            f"`{report_a.get('label')}` — {judge_prompt_version(report_a)}, "
            f"`{report_b.get('label')}` — {judge_prompt_version(report_b)} "
            "(перегоните старый прогон; продолжить: --allow-model-mismatch)"
        )
    if not allow_degraded:
        for report in (report_a, report_b):
            health = report.get("grader_health") or {}
            if grader_degraded(health):
                out.append(
                    f"в прогоне `{report.get('label')}` реранкер не работал: грейдер "
                    f"оценил {health.get('graded', 0)} из {health.get('applicable', 0)} "
                    "пар (продолжить: --allow-degraded)"
                )
    return out


def _params_differ(report_a: dict[str, Any], report_b: dict[str, Any]) -> bool:
    """Отличаются ли параметры, влияющие на воспроизводимость."""
    def key(report: dict[str, Any]) -> str:
        params = dict(report.get("run_params") or {})
        # Счётчики вызовов — стоимость прогона, а не его параметр: они разные
        # у любых двух прогонов и предупреждению о параметрах не повод.
        for name in [k for k in params if str(k).startswith("judge_calls")]:
            params.pop(name)
        return json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)

    return bool(report_a.get("run_params")) and key(report_a) != key(report_b)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _default_golden() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.jsonl")


def _default_out_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


#: UI слушает 8787 (см. `bootstrap.print_instructions`, `run.sh`, Dockerfile).
DEFAULT_UI_URL = "http://localhost:8787"


def _default_rag_log() -> str:
    """``rag_log.jsonl`` того же пользователя, чей `config.json` читает харнесс."""
    root = os.environ.get("COGNIVAULT_UI_ROOT") or "~/.cognivault-ui"
    return os.path.join(os.path.expanduser(root), "rag_log.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Прогнать golden-set через живой стек и посчитать метрики."
    )
    parser.add_argument("--golden", default=_default_golden(), help="путь к golden.jsonl")
    parser.add_argument("--label", default="baseline", help="метка прогона (имя отчёта)")
    parser.add_argument(
        "--ui-url",
        default=None,
        help=f"базовый URL UI (default: $COGNIVAULT_UI_URL или {DEFAULT_UI_URL})",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer-токен UI в server-режиме (default: $COGNIVAULT_UI_TOKEN)",
    )
    parser.add_argument("--limit", type=int, default=0, help="ограничить число пар (0 = все)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="параллельных вопросов (1 — судья держит один слот, см. README)",
    )
    parser.add_argument("--out-dir", default=_default_out_dir(), help="куда писать отчёты")
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="брать и пары с accepted: false",
    )
    parser.add_argument(
        "--backend-url", default=None, help="URL бэкенда CogniVault (для текста источников)"
    )
    parser.add_argument("--backend-token", default=None, help="Bearer-токен бэкенда")
    parser.add_argument(
        "--no-context-fetch",
        action="store_true",
        help="не тянуть текст источников (метрики по контексту станут бессмысленны)",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=4000,
        help="кап текста одного источника В ФОЛБЭКЕ (при прогоне по логу не нужен)",
    )
    parser.add_argument(
        "--rag-log",
        default=None,
        help=(
            "путь к rag_log.jsonl UI — оттуда берётся ФАКТИЧЕСКИЙ контекст хода "
            f"(default: $COGNIVAULT_UI_RAG_LOG или {_default_rag_log()})"
        ),
    )
    parser.add_argument(
        "--no-rag-log",
        action="store_true",
        help=(
            "не читать rag_log.jsonl — контекст восстанавливать из метаданных. "
            "Прогон будет помечен ПРИБЛИЖЁННЫМ и не сравним с прогоном по логу."
        ),
    )
    parser.add_argument("--config", default=None, help="путь к config.json UI")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("A.json", "B.json"),
        help="только сравнить два готовых JSON-отчёта и выйти",
    )
    parser.add_argument(
        "--allow-model-mismatch",
        action="store_true",
        help=(
            "--compare: сравнивать, даже если в прогонах отвечали разные "
            "модели/провайдеры или судили разные версии промптов (иначе код "
            "выхода 2)"
        ),
    )
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help=(
            "--compare: сравнивать, даже если в одном из прогонов реранкер не "
            f"работал (грейдер оценил меньше {GRADER_DEGRADED_THRESHOLD} доли "
            "применимых пар; иначе код выхода 2)"
        ),
    )
    return parser


def _resolve_ui(args: argparse.Namespace) -> tuple[str, str]:
    url = (
        args.ui_url or os.environ.get("COGNIVAULT_UI_URL") or DEFAULT_UI_URL
    ).rstrip("/")
    token = args.token or os.environ.get("COGNIVAULT_UI_TOKEN") or ""
    return url, token


def _resolve_rag_log(args: argparse.Namespace) -> RagLogIndex | None:
    """Открыть лог запросов UI; ``None`` — прогон пойдёт по фолбэку."""
    if args.no_rag_log:
        _log("rag-log отключён (--no-rag-log): прогон будет ПРИБЛИЖЁННЫМ")
        return None
    path = (
        args.rag_log
        or os.environ.get("COGNIVAULT_UI_RAG_LOG")
        or _default_rag_log()
    )
    index = RagLogIndex.load(path)
    if index is None:
        _log(
            f"ВНИМАНИЕ: {path} не найден — контекст будет восстанавливаться из "
            "метаданных, прогон помечен ПРИБЛИЖЁННЫМ (--rag-log укажет путь)"
        )
    else:
        _log(f"rag-log: {path} (записей на старте: {len(index)})")
    return index


def do_compare(args: argparse.Namespace) -> int:
    """Render (and persist) the diff table for two existing reports."""
    path_a, path_b = args.compare
    with open(path_a, "r", encoding="utf-8") as fh:
        report_a = json.load(fh)
    with open(path_b, "r", encoding="utf-8") as fh:
        report_b = json.load(fh)
    blockers = compare_blockers(
        report_a,
        report_b,
        allow_model_mismatch=bool(getattr(args, "allow_model_mismatch", False)),
        allow_degraded=bool(getattr(args, "allow_degraded", False)),
    )
    if blockers:
        _log("ОТКАЗ: прогоны нельзя сравнивать как две настройки одной системы:")
        for reason in blockers:
            _log(f"  - {reason}")
        return 2
    text = render_compare_md(report_a, report_b)
    print(text)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f"compare-{report_a.get('label', 'a')}-vs-{report_b.get('label', 'b')}.md",
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    _log(f"diff-таблица: {out_path}")
    return 0


async def main_async(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.compare:
        return do_compare(args)

    try:
        rows = load_golden(args.golden, include_rejected=args.include_rejected)
    except (OSError, ValueError) as exc:
        _log(f"ОШИБКА: не удалось прочитать golden-set: {exc}")
        return 2
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        _log("ОШИБКА: в golden-set нет пар (все accepted: false?)")
        return 2

    ui_url, ui_token = _resolve_ui(args)
    backend_url, backend_token = resolve_backend(
        args.backend_url, args.backend_token, args.config
    )
    cfg = JudgeConfig.from_env(args.config)

    _log(f"golden: {args.golden} — пар: {len(rows)}")
    _log(f"UI: {ui_url} (токен: {'есть' if ui_token else 'нет'})")
    _log(f"судья: {cfg.model} @ {cfg.base_url} (промпты {metrics_mod.PROMPT_VERSION})")

    try:
        judge = GigaChatJudge(cfg)
    except GigaChatEvalError as exc:
        _log(f"ОШИБКА GigaChat: {exc}")
        return 2

    rag_log_index = _resolve_rag_log(args)
    backend = None if args.no_context_fetch else BackendClient(backend_url, backend_token)
    chat = ChatClient(ui_url, ui_token)
    try:
        # Живой каталог — чтобы переехавшая страница читалась как дрейф
        # разметки, а не как промах ретрива. Недоступен — прогон как раньше.
        live_paths = await fetch_live_paths(backend)
        samples = await run_all(
            rows,
            chat=chat,
            judge=judge,
            backend=backend,
            concurrency=args.concurrency,
            context_cap=args.context_chars,
            rag_log=rag_log_index,
            live_paths=live_paths,
        )
    finally:
        await chat.aclose()
        await judge.aclose()
        if backend is not None:
            await backend.aclose()

    report = build_report(
        samples,
        label=args.label,
        golden_path=args.golden,
        ui_url=ui_url,
        judge_model=cfg.model,
        judge_temperature=cfg.temperature,
        extra={
            "concurrency": args.concurrency,
            "context_chars": args.context_chars,
            "context_fetch": not args.no_context_fetch,
            "rag_log": not args.no_rag_log and rag_log_index is not None,
            "judge_calls": judge.calls,
        },
    )

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"report-{args.label}.json")
    md_path = os.path.join(args.out_dir, f"report-{args.label}.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        # Без `raw.replies`: сырые ответы судьи — самое тяжёлое поле отчёта.
        json.dump(slim_report(report), fh, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_report_md(report))

    _log(f"отчёты: {json_path}, {md_path}")
    buckets = report["buckets"]
    _log(
        f"  судейские средние — по {buckets['answerable']} отвечаемым парам; "
        f"пар-ловушек {buckets['refusal']} (их средние: aggregate_refusal)"
    )
    for name, value in report["aggregate"].items():
        _log(f"  {name}: {_fmt(value)}")
    _log("  " + _judge_calls_line(report["run_params"].get("judge_calls_by_metric")))
    clipped_by_judge = report["counts"].get(JUDGE_CLIP_KEY, 0)
    if clipped_by_judge:
        _log(f"ВНИМАНИЕ: кап судьи урезал контекст на {clipped_by_judge} парах")
    failed = report["counts"]["failed"]
    if failed:
        _log(f"  упало и исключено из средних: {failed}")
    gen_failed = report["counts"].get(GENERATION_FAILED_KEY, 0)
    if gen_failed:
        _log(f"  сбоев генерации (пустой ответ, вне средних): {gen_failed}")
    if grader_degraded(report.get("grader_health")):
        health = report["grader_health"]
        _log(
            f"ВНИМАНИЕ: РЕРАНКЕР НЕ РАБОТАЛ — грейдер оценил {health.get('graded', 0)} "
            f"из {health.get('applicable', 0)} пар; {grader_cause_clause(report) or 'причин в записях нет'}"
        )
    drift = report.get("path_drift") or {}
    if any(drift.get(key) for key in ("drifted", "ambiguous", "missing")):
        _log(
            "ВНИМАНИЕ: дрейф путей golden — сопоставлено по имени файла: "
            f"{len(drift.get('drifted') or [])}, неоднозначных: "
            f"{len(drift.get('ambiguous') or [])}, отсутствующих: "
            f"{len(drift.get('missing') or [])} (см. секцию отчёта)"
        )
    if report.get("approximate"):
        _log("ВНИМАНИЕ: прогон ПРИБЛИЖЁННЫЙ — контекст восстановлен из метаданных")
    degradation = report.get("retrieval_degradation") or {}
    if degradation.get("degraded"):
        _log(
            f"ВНИМАНИЕ: retrieval_hit огрублён у {degradation['degraded']} из "
            f"{degradation['measured']} пар (нет source_chunk_index) — "
            f"гранулярность {_granularity_label(report)}, число ЗАВЫШЕНО"
        )
    _log("напоминание: абсолютные числа судьи не показательны — сравнивайте прогоны")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(main_async(argv))


if __name__ == "__main__":
    raise SystemExit(main())
