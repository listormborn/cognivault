"""Golden-set generator (план 5.2): корпус → фрагменты → пары вопрос/эталон.

Pipeline:

1. ``GET /api/vault/files?recursive=true`` → list the corpus (backend REST,
   Bearer token), then ``GET /api/vault/content?path=…`` per file;
2. split each document by markdown headings (H1–H3) with a character cap —
   a dependency-free splitter, see :func:`split_fragments`;
3. ask GigaChat for **one factual + one practical** question per fragment, with
   ``ground_truth`` grounded strictly in that fragment, answer strictly as JSON;
4. on a share of the fragments (``--unanswerable-share``, default 12%) also ask
   for an **unanswerable** question — plausible for this documentation but not
   answered by the corpus. Those rows carry ``expected_refusal: true`` and
   measure the refusal branch, which nothing measured before;
5. write ``golden.jsonl`` — one Q/A pair per line with ``accepted: null``,
   the field a human flips to ``true``/``false`` during manual validation.

Target volume is 50–100 pairs, i.e. ~25–50 fragments (``--limit``).

Example::

    python3 tools/eval/gen_golden.py --dry-run --limit 40
    python3 tools/eval/gen_golden.py --out tools/eval/golden.jsonl --limit 40

Progress goes to stderr, data to the output file.

``source_chunk_index`` строки всегда ``null``: сплиттер этого генератора —
свой, упрощённый, и его нумерация фрагментов не совпадает с ``chunk_index``
бэкенда (см. комментарий в :func:`_row`). ``run.py`` из-за этого меряет
``retrieval_hit`` по разделу, а не по чанку, и печатает это огрубление в
отчёте отдельным предупреждением. Точность до чанка поднимается только ручной
простановкой ``source_chunk_index`` в ``golden.jsonl``.

**Ручная валидация обязательна для ``expected_refusal``-строк.** Модель
придумывает их вслепую, зная лишь один фрагмент; вопрос может случайно
оказаться отвеченным в другом документе корпуса. Проверьте поиском и снимите
``accepted`` там, где ответ в базе всё-таки есть.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import httpx

from gigachat_client import (
    GigaChatEvalError,
    GigaChatJudge,
    JudgeConfig,
    load_ui_config,
)

DEFAULT_EXTENSIONS = (".md", ".markdown", ".txt")
#: v2: вопросы формулируются как пользовательские, без переписывания терминов
#: фрагмента (лексическая утечка), плюс ветка `unanswerable`.
PROMPT_VERSION = "v2"

#: Доля фрагментов, к которым дополнительно генерируется вопрос без ответа
#: в корпусе. 10–15% хватает, чтобы ветка отказа перестала быть неизмеренной,
#: и мало, чтобы не размыть основные метрики.
DEFAULT_UNANSWERABLE_SHARE = 0.12

GEN_SYSTEM = (
    "Ты — методист, который составляет проверочные вопросы по внутренней "
    "документации. Ты работаешь только с предоставленным фрагментом и всегда "
    "отвечаешь строго в формате JSON, без пояснений вне JSON."
)

GEN_PROMPT = """Ниже — фрагмент внутренней документации.

Документ: {title}
Раздел: {section_path}

Фрагмент:
\"\"\"
{fragment}
\"\"\"

Составь по этому фрагменту ДВА вопроса на русском языке:
1. factual — фактологический вопрос («что», «сколько», «какой», «где указано»),
   ответ на который прямо содержится во фрагменте.
2. practical — практический вопрос («как сделать», «что нужно, чтобы…»,
   «в каком порядке»), ответ на который тоже следует из фрагмента.

Как должен звучать вопрос:
- Так, как его задал бы живой сотрудник, который ЕЩЁ НЕ ЧИТАЛ этот текст и
  формулирует своими словами. Он не знает, какими словами написан документ.
- Не переписывай формулировки фрагмента. Где можно — используй естественный
  синоним или описание вместо термина из текста («как ограничить размер
  ответа» вместо «что делает параметр max_tokens»).
- Дословно оставляй только то, без чего вопрос теряет смысл: названия систем и
  сервисов, имена файлов, коды ошибок, идентификаторы. Их пересказывать нельзя.
- Без отсылок к тексту: никаких «в этом фрагменте», «выше», «в данном разделе».
  Вопрос должен быть понятен сам по себе.
- Не подсказывай ответ внутри вопроса и не перечисляй в нём варианты.

Остальные требования:
- ground_truth — краткий (1–3 предложения) ответ СТРОГО по содержимому
  фрагмента. Не добавляй ничего от себя.
- Если фрагмент не позволяет составить осмысленный вопрос одного из типов —
  верни для него null.

Ответ строго в JSON:
{{"factual": {{"question": "...", "ground_truth": "..."}},
  "practical": {{"question": "...", "ground_truth": "..."}}}}"""

UNANSWERABLE_PROMPT = """Ниже — фрагмент внутренней документации.

Документ: {title}
Раздел: {section_path}

Фрагмент:
\"\"\"
{fragment}
\"\"\"

Придумай ОДИН вопрос на русском языке, который сотрудник мог бы задать этой
базе знаний, но ответа на который в документации НЕТ.

Требования:
- Вопрос должен звучать правдоподобно и по теме: та же система, тот же процесс,
  та же предметная область, что и во фрагменте.
- Он должен спрашивать о том, чего в документации нет: точные сроки, цены,
  фамилии ответственных, внутренние решения, статистика, планы, детали
  соседних систем.
- Ответ НЕ должен выводиться из фрагмента ни прямо, ни косвенно.
- Не пиши «в документации не сказано» внутри самого вопроса — вопрос обычный.

Ответ строго в JSON:
{{"unanswerable": {{"question": "...", "why": "почему ответа нет в документации"}}}}"""

#: ``ground_truth`` для строк без ответа в корпусе: правильное поведение —
#: отказ, а не выдуманный факт. Формулировка совпадает с `rag.SYSTEM_PROMPT`.
REFUSAL_GROUND_TRUTH = (
    "К сожалению, в базе знаний я не нашёл информации, которая позволила бы "
    "надёжно ответить на этот вопрос. Попробуйте уточнить формулировку или "
    "указать название продукта, процесса или инструмента — я выполню поиск ещё раз."
)


# --------------------------------------------------------------------------- #
# Backend REST (self-contained; mirrors cognivault-ui/app/cognivault.py idioms)
# --------------------------------------------------------------------------- #


class BackendError(Exception):
    """Non-200 from the CogniVault backend."""

    def __init__(self, message: str, status: int, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class BackendClient:
    """Read-only client for the vault REST surface.

    ``transport=`` is injectable so tests can drive it with
    ``httpx.MockTransport``.
    """

    def __init__(
        self,
        base_url: str,
        token: str = "",
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._base = (base_url or "").rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.AsyncClient(
            headers=headers, transport=transport, timeout=timeout
        )

    async def __aenter__(self) -> "BackendClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_files(self, recursive: bool = True) -> list[str]:
        """Return vault file paths, tolerating the several response shapes."""
        resp = await self._client.get(
            f"{self._base}/api/vault/files",
            params={"recursive": "true" if recursive else "false"},
        )
        if resp.status_code != 200:
            raise BackendError(
                f"list files failed ({resp.status_code})",
                resp.status_code,
                resp.text[:500],
            )
        return extract_paths(resp.json())

    async def content(self, path: str) -> str:
        """Return the document body for ``path`` (``{path, content}`` upstream)."""
        resp = await self._client.get(
            f"{self._base}/api/vault/content", params={"path": path}
        )
        if resp.status_code != 200:
            raise BackendError(
                f"content failed ({resp.status_code})",
                resp.status_code,
                resp.text[:500],
            )
        data = resp.json()
        return str(data.get("content", "") or "")

    async def catalog_paths(self, *, page_size: int = 500) -> list[str]:
        """Пути ВСЕХ проиндексированных документов из ``GET /api/vault/catalog``.

        Каталог отдаёт страницами (``limit``/``offset``) и всегда несёт ``total``,
        поэтому обрезка обнаружима: читаем, пока ``offset + len(documents)`` не
        достигнет ``total``. Пустая страница до достижения ``total`` — конец,
        а не бесконечный цикл. В отличие от ``/api/vault/files`` здесь ровно
        то, что поиск способен вернуть, — по этому списку проверяется дрейф
        ``source_path`` golden-набора.
        """
        out: list[str] = []
        offset = 0
        while True:
            resp = await self._client.get(
                f"{self._base}/api/vault/catalog",
                params={"limit": page_size, "offset": offset},
            )
            if resp.status_code != 200:
                raise BackendError(
                    f"catalog failed ({resp.status_code})",
                    resp.status_code,
                    resp.text[:500],
                )
            data = resp.json()
            documents = data.get("documents") if isinstance(data, dict) else None
            if not isinstance(documents, list):
                raise BackendError("catalog: нет поля documents", resp.status_code, "")
            page = [
                str(d.get("path"))
                for d in documents
                if isinstance(d, dict) and d.get("path")
            ]
            out.extend(page)
            total = data.get("total")
            offset += len(documents)
            if not documents or not isinstance(total, int) or offset >= total:
                break
        return out


def extract_paths(data: Any) -> list[str]:
    """Normalise ``/api/vault/files`` payloads to a flat list of file paths."""
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


# --------------------------------------------------------------------------- #
# Markdown fragment splitter (pure — no external libs)
# --------------------------------------------------------------------------- #


@dataclass
class Fragment:
    """A slice of a document large enough to carry a self-contained answer."""

    path: str
    section_path: str
    text: str

    @property
    def title(self) -> str:
        base = os.path.basename(self.path)
        return os.path.splitext(base)[0]

    def fragment_id(self) -> str:
        """Stable id — re-running the generator keeps ids for unchanged text."""
        key = f"{self.path}::{self.section_path}::{self.text[:200]}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def _strip_front_matter(text: str) -> str:
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end < 0:
        return text
    newline = text.find("\n", end + 1)
    return text[newline + 1 :] if newline >= 0 else ""


def split_fragments(
    path: str,
    text: str,
    *,
    max_chars: int = 2200,
    min_chars: int = 350,
) -> list[Fragment]:
    """Split a markdown document into fragments by H1–H3 headings.

    * headings inside fenced code blocks are ignored;
    * ``section_path`` is the ``H1 > H2 > H3`` breadcrumb of the fragment;
    * sections longer than ``max_chars`` are packed paragraph-by-paragraph
      (a paragraph is never split unless it alone exceeds the cap);
    * neighbouring sections shorter than ``min_chars`` are merged so trailing
      stubs do not become useless one-line fragments.
    """
    body = _strip_front_matter(text or "")
    if not body.strip():
        return []

    sections: list[tuple[str, list[str]]] = []
    stack: list[str] = []
    current: list[str] = []
    current_path = ""
    in_fence = False

    def flush() -> None:
        if current and any(line.strip() for line in current):
            sections.append((current_path, list(current)))

    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            current.append(line)
            continue
        level = _heading_level(line) if not in_fence else 0
        if 1 <= level <= 3:
            flush()
            current = []
            title = line.strip().lstrip("#").strip()
            stack = stack[: level - 1]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(title)
            current_path = " > ".join(part for part in stack if part)
            continue
        current.append(line)
    flush()

    fragments: list[Fragment] = []
    for section_path, lines in sections:
        section_text = "\n".join(lines).strip()
        if not section_text:
            continue
        for piece in _pack_paragraphs(section_text, max_chars):
            fragments.append(
                Fragment(path=path, section_path=section_path, text=piece)
            )

    return _merge_small(fragments, min_chars=min_chars, max_chars=max_chars)


def _heading_level(line: str) -> int:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return 0
    level = len(stripped) - len(stripped.lstrip("#"))
    rest = stripped[level:]
    if level > 6 or (rest and not rest.startswith(" ")):
        return 0
    return level


def _pack_paragraphs(text: str, max_chars: int) -> list[str]:
    """Greedy paragraph packing up to ``max_chars``."""
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for para in text.split("\n\n"):
        chunk = para.strip("\n")
        if not chunk.strip():
            continue
        if len(chunk) > max_chars:
            if buf:
                out.append("\n\n".join(buf))
                buf, size = [], 0
            out.extend(
                chunk[i : i + max_chars] for i in range(0, len(chunk), max_chars)
            )
            continue
        if size + len(chunk) > max_chars and buf:
            out.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(chunk)
        size += len(chunk) + 2
    if buf:
        out.append("\n\n".join(buf))
    return [piece for piece in out if piece.strip()]


def _merge_small(
    fragments: Sequence[Fragment], *, min_chars: int, max_chars: int
) -> list[Fragment]:
    """Merge sub-``min_chars`` fragments of the same document, drop leftovers."""
    out: list[Fragment] = []
    for fragment in fragments:
        if (
            out
            and out[-1].path == fragment.path
            and len(out[-1].text) < min_chars
            and len(out[-1].text) + len(fragment.text) <= max_chars
        ):
            previous = out.pop()
            section_path = previous.section_path or fragment.section_path
            out.append(
                Fragment(
                    path=fragment.path,
                    section_path=section_path,
                    text=f"{previous.text}\n\n{fragment.text}".strip(),
                )
            )
            continue
        out.append(fragment)
    return [f for f in out if len(f.text.strip()) >= min_chars]


# --------------------------------------------------------------------------- #
# Q/A generation
# --------------------------------------------------------------------------- #


def build_prompt(fragment: Fragment) -> str:
    """Render :data:`GEN_PROMPT` for one fragment."""
    return GEN_PROMPT.format(
        title=fragment.title,
        section_path=fragment.section_path or "(без раздела)",
        fragment=fragment.text,
    )


def build_unanswerable_prompt(fragment: Fragment) -> str:
    """Render :data:`UNANSWERABLE_PROMPT` for one fragment."""
    return UNANSWERABLE_PROMPT.format(
        title=fragment.title,
        section_path=fragment.section_path or "(без раздела)",
        fragment=fragment.text,
    )


def _row(
    fragment: Fragment,
    *,
    suffix: str,
    question: str,
    ground_truth: str,
    kind: str,
    expected_refusal: bool,
) -> dict[str, Any]:
    return {
        "id": f"{fragment.fragment_id()}-{suffix}",
        "question": question,
        "ground_truth": ground_truth,
        "kind": kind,
        # Для строк-отказов источника нет по определению: `run.py` пропускает
        # им `retrieval_hit` и меряет вместо него сам факт отказа.
        "source_path": None if expected_refusal else fragment.path,
        "section_path": None if expected_refusal else fragment.section_path,
        # ВСЕГДА None — и это осознанно, а не забытая доработка.
        #
        # `chunk_index` бэкенда — это позиция чанка в массиве чанков ФАЙЛА
        # (`src/plugins/pipeline.ts`: `points = enriched.map((chunk, i) => …
        # chunk_index: i)`). Массив собирает настоящий чанкер: короткие секции
        # сливаются с предыдущей, длинные режутся по бюджету токенов
        # (MAX_CHUNK_TOKENS минус breadcrumb), таблицы уходят построчными
        # группами, а при INDEX_TABLE_SUMMARY в хвост дописываются ещё и
        # table-summary-точки. Здешний сплиттер другой: H1–H3 плюс кап по
        # СИМВОЛАМ (`split_fragments`), с собственным слиянием коротких кусков.
        # Один фрагмент отображается в ноль, один или несколько чанков бэкенда,
        # и после первого же расхождения нумерация уезжает на весь документ.
        #
        # Порядковый номер фрагмента, выданный за `chunk_index`, поэтому давал
        # бы ЛОЖНЫЕ ПРОМАХИ: `run.py` на чанковой ветке сравнивает индексы
        # точно и вернёт False даже там, где ретрив нашёл нужный текст. Честнее
        # оставить None и мерить по разделу — `run.py` печатает такое огрубление
        # явно (см. `GRANULARITY_WARNING`), так что метрика не притворяется
        # точной. Ключ есть всегда: проставленный ВРУЧНУЮ (по выдаче
        # `/api/search` для этого вопроса), он поднимает `retrieval_hit`
        # с уровня раздела до уровня чанка.
        "source_chunk_index": None,
        "expected_refusal": expected_refusal,
        "accepted": None,
    }


def pairs_from_verdict(fragment: Fragment, raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one judge verdict into golden.jsonl rows (skipping nulls)."""
    rows: list[dict[str, Any]] = []
    for kind in ("factual", "practical"):
        item = raw.get(kind)
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "") or "").strip()
        truth = str(item.get("ground_truth", "") or "").strip()
        if not question or not truth:
            continue
        rows.append(
            _row(
                fragment,
                suffix=kind[0],
                question=question,
                ground_truth=truth,
                kind=kind,
                expected_refusal=False,
            )
        )
    return rows


def unanswerable_from_verdict(
    fragment: Fragment, raw: dict[str, Any]
) -> list[dict[str, Any]]:
    """Convert an ``unanswerable`` verdict into a refusal-branch golden row."""
    item = raw.get("unanswerable")
    if not isinstance(item, dict):
        return []
    question = str(item.get("question", "") or "").strip()
    if not question:
        return []
    return [
        _row(
            fragment,
            suffix="u",
            question=question,
            ground_truth=REFUSAL_GROUND_TRUTH,
            kind="unanswerable",
            expected_refusal=True,
        )
    ]


def pick_unanswerable(
    fragments: Sequence[Fragment], share: float, seed: int
) -> set[int]:
    """Индексы фрагментов, для которых спрашиваем вопрос БЕЗ ответа в корпусе.

    Детерминированно по ``seed``: пересборка golden-set с теми же параметрами
    даёт тот же состав, иначе прогоны несравнимы.
    """
    if share <= 0 or not fragments:
        return set()
    count = max(1, round(len(fragments) * min(share, 1.0)))
    rng = random.Random(seed ^ 0x5EED)
    return set(rng.sample(range(len(fragments)), min(count, len(fragments))))


async def generate(
    judge: GigaChatJudge,
    fragments: Sequence[Fragment],
    *,
    concurrency: int = 2,
    unanswerable_share: float = DEFAULT_UNANSWERABLE_SHARE,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Ask the model for Q/A pairs for every fragment (bounded concurrency)."""
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[list[dict[str, Any]]] = [[] for _ in fragments]
    unanswerable = pick_unanswerable(fragments, unanswerable_share, seed)
    done = 0

    async def worker(index: int, fragment: Fragment) -> None:
        nonlocal done
        async with semaphore:
            rows: list[dict[str, Any]] = []
            try:
                raw = await judge.complete_json(
                    build_prompt(fragment), system=GEN_SYSTEM, temperature=0.3
                )
                rows.extend(pairs_from_verdict(fragment, raw))
            except (GigaChatEvalError, httpx.HTTPError) as exc:
                _log(f"  ! фрагмент {index + 1}: {exc}")
            if index in unanswerable:
                try:
                    raw = await judge.complete_json(
                        build_unanswerable_prompt(fragment),
                        system=GEN_SYSTEM,
                        temperature=0.5,
                    )
                    rows.extend(unanswerable_from_verdict(fragment, raw))
                except (GigaChatEvalError, httpx.HTTPError) as exc:
                    _log(f"  ! фрагмент {index + 1} (без ответа): {exc}")
            results[index] = rows
            done += 1
            _log(
                f"  [{done}/{len(fragments)}] {fragment.path} "
                f"> {fragment.section_path or '-'}"
            )

    await asyncio.gather(
        *(worker(i, fragment) for i, fragment in enumerate(fragments))
    )
    return [row for group in results for row in group]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def resolve_backend(
    base_url: str | None, token: str | None, config_path: str | None = None
) -> tuple[str, str]:
    """Resolve ``(base_url, token)``: CLI → ENV → UI config.json → default."""
    cfg = load_ui_config(config_path).get("cognivault", {})
    if not isinstance(cfg, dict):
        cfg = {}
    resolved_url = (
        base_url
        or os.environ.get("COGNIVAULT_BASE_URL")
        or str(cfg.get("base_url", "") or "")
        or "http://localhost:3000"
    )
    resolved_token = (
        token
        or os.environ.get("COGNIVAULT_TOKEN")
        or os.environ.get("COGNIVAULT_API_KEY")
        or str(cfg.get("token", "") or "")
    )
    return resolved_url.rstrip("/"), resolved_token


def select_fragments(
    fragments: Sequence[Fragment], limit: int, seed: int
) -> list[Fragment]:
    """Shuffle deterministically and take ``limit``, spreading across files.

    Round-robin over documents first so a single huge document cannot swallow
    the whole golden set.
    """
    by_path: dict[str, list[Fragment]] = {}
    for fragment in fragments:
        by_path.setdefault(fragment.path, []).append(fragment)

    rng = random.Random(seed)
    paths = sorted(by_path)
    rng.shuffle(paths)
    for path in paths:
        rng.shuffle(by_path[path])

    out: list[Fragment] = []
    round_index = 0
    while len(out) < limit:
        added = False
        for path in paths:
            bucket = by_path[path]
            if round_index < len(bucket):
                out.append(bucket[round_index])
                added = True
                if len(out) >= limit:
                    break
        if not added:
            break
        round_index += 1
    return out


async def collect_fragments(
    client: BackendClient,
    *,
    extensions: Iterable[str],
    max_chars: int,
    min_chars: int,
) -> list[Fragment]:
    """Fetch every eligible document and split it into fragments."""
    paths = await client.list_files(recursive=True)
    wanted = tuple(extensions)
    selected = [p for p in paths if p.lower().endswith(wanted)]
    _log(f"файлов в вольте: {len(paths)}, подходящих по расширению: {len(selected)}")

    fragments: list[Fragment] = []
    for index, path in enumerate(selected, start=1):
        try:
            text = await client.content(path)
        except BackendError as exc:
            _log(f"  ! пропуск {path}: {exc}")
            continue
        pieces = split_fragments(
            path, text, max_chars=max_chars, min_chars=min_chars
        )
        fragments.extend(pieces)
        if index % 20 == 0 or index == len(selected):
            _log(f"  прочитано {index}/{len(selected)}, фрагментов: {len(fragments)}")
    return fragments


def write_jsonl(rows: Sequence[dict[str, Any]], out_path: str) -> None:
    """Write golden rows as UTF-8 JSONL (``ensure_ascii=False``)."""
    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Сгенерировать golden-set вопрос/эталон из корпуса вольта."
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.jsonl"),
        help="куда писать golden.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="сколько фрагментов взять (≈2 пары на фрагмент; 25–50 → 50–100 пар)",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed выборки фрагментов")
    parser.add_argument("--base-url", default=None, help="URL бэкенда CogniVault")
    parser.add_argument("--token", default=None, help="Bearer-токен бэкенда")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="не вызывать GigaChat — только показать, сколько фрагментов найдено",
    )
    parser.add_argument(
        "--concurrency", type=int, default=2, help="параллельных вызовов GigaChat"
    )
    parser.add_argument("--max-chars", type=int, default=2200, help="кап фрагмента")
    parser.add_argument(
        "--min-chars", type=int, default=350, help="минимальный размер фрагмента"
    )
    parser.add_argument(
        "--ext",
        default=",".join(DEFAULT_EXTENSIONS),
        help="расширения файлов через запятую",
    )
    parser.add_argument(
        "--unanswerable-share",
        type=float,
        default=DEFAULT_UNANSWERABLE_SHARE,
        help=(
            "доля фрагментов, к которым генерируется вопрос БЕЗ ответа в корпусе "
            f"(ветка отказа; default {DEFAULT_UNANSWERABLE_SHARE}, 0 — выключить)"
        ),
    )
    parser.add_argument(
        "--config", default=None, help="путь к config.json UI (для base_url/токена)"
    )
    return parser


async def main_async(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_url, token = resolve_backend(args.base_url, args.token, args.config)
    extensions = tuple(
        e.strip() if e.strip().startswith(".") else f".{e.strip()}"
        for e in args.ext.split(",")
        if e.strip()
    )

    _log(f"бэкенд: {base_url} (токен: {'есть' if token else 'нет'})")
    async with BackendClient(base_url, token) as client:
        try:
            fragments = await collect_fragments(
                client,
                extensions=extensions,
                max_chars=args.max_chars,
                min_chars=args.min_chars,
            )
        except (BackendError, httpx.HTTPError) as exc:
            _log(f"ОШИБКА: не удалось прочитать корпус: {exc}")
            return 2

    _log(f"всего фрагментов: {len(fragments)}")
    chosen = select_fragments(fragments, args.limit, args.seed)
    _log(f"отобрано фрагментов: {len(chosen)} (ожидаемо пар: ~{len(chosen) * 2})")

    if args.dry_run:
        for fragment in chosen[:10]:
            _log(
                f"  · {fragment.path} > {fragment.section_path or '-'} "
                f"({len(fragment.text)} симв.)"
            )
        _log("dry-run: вызовов GigaChat не было, файл не записан")
        return 0

    if not chosen:
        _log("ОШИБКА: нечего генерировать — корпус пуст или не проиндексирован")
        return 2

    cfg = JudgeConfig.from_env(args.config)
    _log(f"модель-генератор: {cfg.model} @ {cfg.base_url}")
    try:
        judge = GigaChatJudge(cfg)
    except GigaChatEvalError as exc:
        _log(f"ОШИБКА GigaChat: {exc}")
        return 2

    async with judge:
        rows = await generate(
            judge,
            chosen,
            concurrency=args.concurrency,
            unanswerable_share=args.unanswerable_share,
            seed=args.seed,
        )

    write_jsonl(rows, args.out)
    refusals = sum(1 for row in rows if row.get("expected_refusal"))
    _log(f"записано пар: {len(rows)} → {args.out} (из них без ответа: {refusals})")
    _log("следующий шаг: вручную проставить accepted (true/false) в golden.jsonl")
    if refusals:
        _log(
            "ОСОБО проверьте строки expected_refusal: ответ мог найтись в другом "
            "документе корпуса — тогда accepted: false"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(main_async(argv))


if __name__ == "__main__":
    raise SystemExit(main())
