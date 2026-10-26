"""Footprint of the knowledge base — the «состав базы» block of the RAG context.

A RAG answer is assembled from at most :data:`app.rag._MAX_CONTEXT_BLOCKS`
fragments. Nothing in that context tells the model how much it is *not* holding,
so a single page that happens to be a numbered list reads exactly like the whole
corpus and gets answered from with full confidence.

This module renders a small, constant-size block that states the scale: how many
documents the base has, how many fragments the model was actually given, and how
the corpus is divided at the top. It answers no question — it removes the
omniscience pose.

Design constraints, all load-bearing:

* **Constant size.** The block is folded at depth 1 (or 2 — see :func:`_fold`)
  with counts, capped at :data:`_MAX_SECTIONS` sections, :data:`_MAX_CHILDREN`
  sub-sections each, and hard-capped at :data:`_MAX_MAP_CHARS` characters. A
  corpus ten times bigger renders the same size block with bigger numbers.
* **Zero model calls.** Everything here is one cached ``GET /api/vault/files``,
  one cached ``GET /api/vault/catalog`` (for the definition of "document" —
  see below) plus pure Python.
* **Silent degradation.** :func:`corpus_block` returns ``None`` on *any* failure
  (either endpoint down, malformed payload, empty vault); the caller then renders
  the turn exactly as it did before this module existed.
* **No local idea of what a document is.** The set of counted extensions is
  *fetched* (``document_extensions``, :mod:`app.catalog`), never authored here.
  The list this module used to carry — ``md, markdown, txt, pdf, csv, canvas,
  excalidraw`` — was wrong in both directions the block exists to prevent:
  ``txt`` and ``markdown`` are never scanned, chunked or embedded, so every
  ``.txt`` in a vault was counted as a document the model could be asked about
  and search could never return. When the definition cannot be fetched the block
  is omitted; falling back to a built-in allowlist would be re-authoring the
  same lie.
* **Code-owned text, not a prompt.** The caption that forbids citing the block
  travels *with* the block. Delivering it as prompt wording would never reach a
  user who has already saved a custom ``prompts.system`` — their copy is frozen
  at the day they saved it.
* **Short timeout, cached per user.** The listing is fetched with
  :data:`_LIST_TIMEOUT` (not the client default) and cached per
  ``(base_url, token)``; a hanging vault must not add its timeout to every
  single RAG turn.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any

from . import catalog, cognivault

# Label for documents that sit directly at the folded level (no sub-folder).
_ROOT_LABEL = "(корень)"

# Size guards. The first two shape the fold, the third is the hard ceiling the
# rendered text is squeezed under (see :func:`_compose`).
_MAX_SECTIONS = 12
_MAX_CHILDREN = 4
_MAX_MAP_CHARS = 700

# The same fold, rendered for the OTHER caller: a question about the base itself
# (:func:`render_overview`). There the tree is not navigation furniture next to
# the sources — it IS the material the answer is built from, so it gets a bigger
# budget: every top section, every sub-section it has, ~600 tokens. It is still
# O(1) in the size of the corpus in the sense that matters (no document bodies,
# no annotations, hard char ceiling); a flat listing of 127 paths would be 11k
# tokens and would not fit the chat budget at all.
_MAX_OVERVIEW_SECTIONS = 20
_MAX_OVERVIEW_CHILDREN = 24
_MAX_OVERVIEW_CHARS = 2400

# How many single-folder levels the fold may descend through before it gives up
# (see :func:`_fold`). A guard against a pathological vault, not a design limit:
# a Confluence sync needs 4.
_MAX_DESCENT = 8

# Share of a level's documents that the single sub-folder must hold for the level
# to count as a corridor (see :func:`_fold`). It exists because a corridor level
# is rarely *perfectly* empty: a Confluence sync writes the section's own page
# next to the folder of its children, so ``…/OASIS External Home`` holds one
# stray page and one folder of 126. Requiring the level to be literally empty
# stopped the descent there and rendered the whole base as one line; requiring
# nothing at all would let a vault of 50 loose notes plus one two-file folder
# descend INTO the folder and hide 50 documents the fold is supposed to show.
# 0.9 is a guard, not a tuned number — the strays a real corridor leaves behind
# are the container pages of the corridor itself, one per level.
_CORRIDOR_MIN_SHARE = 0.9

# Listing cache. A vault listing changes on the indexer's timescale, not the
# turn's; the failure TTL is shorter so a brief outage self-heals quickly.
_CACHE_TTL_SECONDS = 300.0
_CACHE_TTL_FAILURE_SECONDS = 60.0
_CACHE_CAP = 256
_LIST_TIMEOUT = 5.0

# Archived branches are intentionally absent from every user-facing map and
# count. The backend may retain them in its index, but the active catalogue must
# not advertise them as available knowledge.
_ARCHIVE_RE = re.compile(r"(?:^|\W)(?:архив|archive[ds]?|deprecated|устаревш)", re.I)

# The two fixed sentences around the numbers.
#
# The caption is the whole "not citable" contract: it sits inside the block, in
# the same message, one line above the numbers — not in the editable system
# prompt, where a user with a saved override would never see it.
_CAPTION = (
    "Состав базы знаний (справочная навигация). Это НЕ источник: ссылаться на "
    "него нельзя, [Источник N] — только на блок «Источники» ниже."
)
_TAIL = "Это объёмы разделов, а не перечень их содержимого."

# The overview's own two sentences. The «не источник» clause is deliberately
# ABSENT here: on a meta turn there is no «Источники» block to defer to, and a
# caption that calls the block un-citable talks the model out of the only
# material it has. What stays is the honest provenance — these are page and
# folder names from the tree, not a summary of what the pages say.
_OVERVIEW_CAPTION = (
    "Структура базы знаний. Построена по дереву разделов (названия страниц и "
    "папок, число документов), не по тексту документов."
)
_OVERVIEW_TAIL = (
    "Это названия и объёмы разделов, а не перечень их содержимого: о том, что "
    "написано внутри, по этому списку судить нельзя."
)

_DOC_FORMS = ("документ", "документа", "документов")
_FRAGMENT_FORMS = ("фрагмент", "фрагмента", "фрагментов")
_SECTION_FORMS = ("раздел", "раздела", "разделов")
_SUBSECTION_FORMS = ("подраздел", "подраздела", "подразделов")


# --------------------------------------------------------------------------- #
# Pure rendering
# --------------------------------------------------------------------------- #


@dataclass
class _Section:
    label: str
    count: int
    children: list[tuple[str, int]] = field(default_factory=list)


def _plural(n: int, forms: tuple[str, str, str]) -> str:
    """Russian plural form of ``forms`` for ``n`` (1 / 2-4 / 5+)."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def _documents(paths: list[Any], extensions: frozenset[str]) -> list[list[str]]:
    """Split a vault listing into path segments, keeping documents only.

    ``extensions`` is the service's own ``document_extensions`` — the extensions
    the indexer actually scans, minus images. It is a REQUIRED argument, with no
    default, so no caller can accidentally reintroduce a local allowlist: a file
    whose extension is not in it is never indexed, and counting it would promise
    a document search can never return. It also does the job the old hard-coded
    list did — a Confluence sync writes attachments
    (``Confluence/attachments/<page-id>/<file>``), and a folder of PNGs must not
    read as a section of the knowledge base.

    Tolerant by design — the listing is upstream data: non-strings, Windows
    separators, leading slashes, ``.`` segments and non-document extensions are
    all dropped rather than raising.
    """
    docs: list[list[str]] = []
    for raw in paths or []:
        if not isinstance(raw, str):
            continue
        segments = [
            s for s in raw.replace("\\", "/").strip().split("/") if s and s != "."
        ]
        if not segments:
            continue
        if any(_ARCHIVE_RE.search(segment) for segment in segments):
            continue
        name = segments[-1]
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext not in extensions:
            continue
        docs.append(segments)
    return docs


def _group(docs: list[list[str]], depth: int) -> list[_Section]:
    """Group documents by their segment at ``depth``, with sub-section counts.

    A document that IS the segment at ``depth`` (i.e. a file sitting directly at
    that level) is counted under :data:`_ROOT_LABEL`. Sections are ordered by
    document count, ties broken by name, so the fold is deterministic.
    """
    counts: dict[str, int] = {}
    children: dict[str, dict[str, int]] = {}
    for segments in docs:
        if len(segments) <= depth:
            continue
        if len(segments) == depth + 1:
            label, child = _ROOT_LABEL, None
        else:
            label = segments[depth]
            child = segments[depth + 1] if len(segments) > depth + 2 else None
        counts[label] = counts.get(label, 0) + 1
        if child:
            bucket = children.setdefault(label, {})
            bucket[child] = bucket.get(child, 0) + 1

    sections = [
        _Section(
            label=label,
            count=count,
            children=sorted(
                children.get(label, {}).items(), key=lambda kv: (-kv[1], kv[0])
            ),
        )
        for label, count in counts.items()
    ]
    sections.sort(key=lambda s: (-s.count, s.label))
    return sections


def _named(sections: list[_Section]) -> list[_Section]:
    """The sections that are real folders.

    :data:`_ROOT_LABEL` is not a branch of the tree — it is the bucket for files
    that happen to lie at this level. Counting it as a section is what made the
    descent in :func:`_fold` stop one level too early on every Confluence vault.
    """
    return [s for s in sections if s.label != _ROOT_LABEL]


def _fold(docs: list[list[str]]) -> tuple[list[_Section], str]:
    """Fold the corpus into top sections, descending while the level is a corridor.

    Returns ``(sections, prefix)``. Depth 1 is the default; a vault whose
    documents ALL live under one folder would render a single useless section,
    so the fold descends and reports the swallowed folders as ``prefix``.

    The descent LOOPS rather than taking a single step, because a real
    Confluence sync nests several levels before it branches:
    ``Confluence/<space>/<ancestor>/…`` (``build_vault_path`` in
    ``app.confluence.convert``). On the corpus this module was built for the
    branch point is at depth 4 — one step short of it left the block reporting
    ``- Confluence — 127 (OASISEXT: 127)`` and nothing else, i.e. paying for a
    block that says nothing about the shape of the base.

    A corridor is a level with exactly one FOLDER (:func:`_named`), not a level
    with exactly one entry. The difference is the whole point: a Confluence sync
    writes a section's own page beside the folder of its children, so
    ``…/OASIS External Home`` holds one stray ``.md`` and one folder of 126
    documents. Read as "two sections" that level was not a corridor, the descent
    stopped, and the entire 127-document base rendered as ``- Разработка … —
    126`` plus ``- (корень) — 1``: three names out of a 2 400-character budget.

    Descending past strays is bounded by :data:`_CORRIDOR_MIN_SHARE`: the single
    folder must hold nearly the whole level, so the documents left behind are the
    container pages a corridor legitimately leaves and never a real branch. It
    still stops as soon as the level branches, and it never descends INTO the
    files themselves: a level whose only entry is :data:`_ROOT_LABEL` is worse
    than the folder above it, so the previous level is kept.
    """
    sections = _group(docs, 0)
    prefix: list[str] = []
    depth = 0
    while depth < _MAX_DESCENT:
        named = _named(sections)
        level = sum(s.count for s in sections)
        if len(named) != 1 or named[0].count < level * _CORRIDOR_MIN_SHARE:
            break
        deeper = _group(docs, depth + 1)
        if not deeper or (len(deeper) == 1 and deeper[0].label == _ROOT_LABEL):
            break
        prefix.append(named[0].label)
        sections = deeper
        depth += 1
    return sections, "/".join(prefix)


def _children_suffix(section: _Section, max_children: int = _MAX_CHILDREN) -> str:
    """``" (Fincert: 12, ППРБ: 9, ещё 6 подразделов)"`` — or empty."""
    if not section.children:
        return ""
    shown = section.children[:max_children]
    parts = [f"{name}: {count}" for name, count in shown]
    hidden = len(section.children) - len(shown)
    if hidden:
        parts.append(f"ещё {hidden} {_plural(hidden, _SUBSECTION_FORMS)}")
    return f" ({', '.join(parts)})"


def _compose(
    total: int,
    n_sources: int | None,
    sections: list[_Section],
    shown: int,
    prefix: str,
    with_children: bool,
    *,
    caption: str = _CAPTION,
    tail: str = _TAIL,
    max_children: int = _MAX_CHILDREN,
) -> str:
    """Render the block with ``shown`` sections, optionally with sub-sections.

    ``caption``/``tail`` are the two fixed sentences around the numbers; they are
    parameters because the same fold serves two callers with opposite contracts
    — the footprint next to the sources («не источник, ссылаться нельзя») and
    the overview that answers a question about the base itself.
    """
    line = f"Всего документов в базе: {total}."
    if isinstance(n_sources, int) and n_sources > 0:
        line += (
            f" Ниже в блоке «Источники» — {n_sources} "
            f"{_plural(n_sources, _FRAGMENT_FORMS)}; остальное содержимое базы "
            "не показано."
        )
    heading = (
        f"Разделы внутри «{prefix}» (число документов):"
        if prefix
        else "Разделы верхнего уровня (число документов):"
    )
    lines = [caption, line, heading]
    for section in sections[:shown]:
        suffix = _children_suffix(section, max_children) if with_children else ""
        lines.append(f"- {section.label} — {section.count}{suffix}")
    rest = sections[shown:]
    if rest:
        docs = sum(s.count for s in rest)
        lines.append(
            f"- ещё {len(rest)} {_plural(len(rest), _SECTION_FORMS)} — "
            f"{docs} {_plural(docs, _DOC_FORMS)}"
        )
    lines.append(tail)
    return "\n".join(lines)


def _squeeze(
    total: int,
    n_sources: int | None,
    sections: list[_Section],
    prefix: str,
    *,
    max_chars: int,
    max_sections: int,
    max_children: int,
    caption: str,
    tail: str,
) -> str:
    """Render under ``max_chars`` by degrading in a fixed order.

    Sub-sections go first, then sections from the smallest up — their documents
    stay counted in the trailing «ещё N разделов» line, so the total always adds
    up. One section still over the cap (an absurdly long folder name) is
    returned anyway: the honest numbers are worth more than the ceiling, and the
    block is still O(1) in the size of the corpus.
    """
    shown = min(len(sections), max_sections)
    for with_children in (True, False):
        text = _compose(
            total,
            n_sources,
            sections,
            shown,
            prefix,
            with_children,
            caption=caption,
            tail=tail,
            max_children=max_children,
        )
        if len(text) <= max_chars:
            return text
    while shown > 1:
        shown -= 1
        text = _compose(
            total, n_sources, sections, shown, prefix, False, caption=caption, tail=tail
        )
        if len(text) <= max_chars:
            return text
    return _compose(
        total, n_sources, sections, 1, prefix, False, caption=caption, tail=tail
    )


def _folded(
    paths: list[Any], extensions: frozenset[str]
) -> tuple[int, list[_Section], str] | None:
    """``(total, sections, prefix)`` for a listing, or ``None`` if it says nothing."""
    docs = _documents(paths, extensions)
    if not docs:
        return None
    sections, prefix = _fold(docs)
    if not sections:
        return None
    return len(docs), sections, prefix


def render(
    paths: list[Any],
    n_sources: int | None = None,
    max_chars: int = _MAX_MAP_CHARS,
    *,
    extensions: frozenset[str],
) -> str | None:
    """Render the corpus footprint block, or ``None`` if there is nothing to say.

    ``n_sources`` is the number of blocks in the «Источники» section of the same
    message; it is what turns the block from trivia into a scale statement
    («127 documents, you were given 5 fragments»). ``None`` omits that sentence.

    ``extensions`` is keyword-only and required — see :func:`_documents`.
    """
    folded = _folded(paths, extensions)
    if folded is None:
        return None
    total, sections, prefix = folded
    return _squeeze(
        total,
        n_sources,
        sections,
        prefix,
        max_chars=max_chars,
        max_sections=_MAX_SECTIONS,
        max_children=_MAX_CHILDREN,
        caption=_CAPTION,
        tail=_TAIL,
    )


def render_overview(
    paths: list[Any],
    max_chars: int = _MAX_OVERVIEW_CHARS,
    *,
    extensions: frozenset[str],
) -> str | None:
    """Render the base's structure as the ANSWER material, or ``None``.

    Same fold, same counts, same degradation ladder as :func:`render` — only the
    budget and the two framing sentences differ. Used by the meta-question
    branch (:mod:`app.corpus_scope`), where the tree is not navigation next to
    the sources but the only grounded thing the model has: retrieval was not run,
    and there is no document in the corpus that lists the corpus.
    """
    folded = _folded(paths, extensions)
    if folded is None:
        return None
    total, sections, prefix = folded
    return _squeeze(
        total,
        None,
        sections,
        prefix,
        max_chars=max_chars,
        max_sections=_MAX_OVERVIEW_SECTIONS,
        max_children=_MAX_OVERVIEW_CHILDREN,
        caption=_OVERVIEW_CAPTION,
        tail=_OVERVIEW_TAIL,
    )


# --------------------------------------------------------------------------- #
# Cached listing
# --------------------------------------------------------------------------- #

# key -> (expires_at, paths | None). ``None`` caches a FAILURE for a shorter TTL.
_cache: dict[str, tuple[float, list[str] | None]] = {}


def _cache_key(cv: dict[str, Any] | None) -> str:
    """Cache identity: the vault, not the caller.

    In server mode the token IS the tenant, so it must be part of the key — one
    user's footprint may never be served to another. It is hashed rather than
    stored, so nothing that gets dumped or logged carries a credential.
    """
    base, token = cognivault._resolve_cv(cv)  # noqa: SLF001 — same package
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16] if token else "-"
    return f"{base}|{digest}"


def reset_cache() -> None:
    """Drop the cached listings (tests; a manual re-sync)."""
    _cache.clear()


async def files(cv: dict[str, Any] | None = None) -> list[str] | None:
    """Cached vault listing for ``cv``; ``None`` when it is unavailable.

    Never raises: a listing failure means "no footprint block this turn", which
    is exactly the behaviour that predates this module.
    """
    key = _cache_key(cv)
    now = time.monotonic()
    entry = _cache.get(key)
    if entry is not None:
        expires_at, cached = entry
        if now < expires_at:
            return cached
        _cache.pop(key, None)

    try:
        paths = await cognivault.list_files(cv, recursive=True, timeout=_LIST_TIMEOUT)
    except Exception:  # noqa: BLE001 — навигация не должна ронять ход
        paths = None

    if len(_cache) >= _CACHE_CAP:
        # Cheap eviction: drop everything rather than track LRU (mirrors
        # `deps._cache_put`).
        _cache.clear()
    ttl = _CACHE_TTL_SECONDS if paths else _CACHE_TTL_FAILURE_SECONDS
    _cache[key] = (now + ttl, paths)
    return paths


async def _listing(
    cv: dict[str, Any] | None,
) -> tuple[list[str], frozenset[str]] | None:
    """``(paths, document_extensions)`` for ``cv``, or ``None`` if either is unknown.

    Both halves are required and neither has a substitute. The listing says which
    files exist; the catalogue says which of them the indexer treats as
    documents. Without the second half this module could only guess — and the
    guess it used to ship counted files that are never indexed. The listing is
    fetched first so a vault that has no listing never pays for a catalogue call.
    """
    paths = await files(cv)
    if not paths:
        return None
    extensions = await catalog.document_extensions(cv)
    if not extensions:
        return None
    return paths, extensions


async def corpus_block(
    cv: dict[str, Any] | None = None, n_sources: int | None = None
) -> str | None:
    """The rendered footprint block for this turn, or ``None``.

    The single entry point used by :mod:`app.rag`. Every failure mode — listing
    unavailable, catalogue unavailable, empty vault, nothing but attachments, an
    unexpected payload — collapses to ``None`` and the turn proceeds exactly as
    before.
    """
    listing = await _listing(cv)
    if listing is None:
        return None
    try:
        return render(listing[0], n_sources, extensions=listing[1])
    except Exception:  # noqa: BLE001 — pure code, but the turn is worth more
        return None


async def overview_block(cv: dict[str, Any] | None = None) -> str | None:
    """The base's structure for a meta turn, or ``None`` when unavailable.

    ``None`` is what makes the meta branch fail closed: no listing means no
    grounded material, and the caller must then route the question exactly as it
    did before this feature existed (retrieval + grader) rather than answer from
    the model's own imagination.
    """
    listing = await _listing(cv)
    if listing is None:
        return None
    try:
        return render_overview(listing[0], extensions=listing[1])
    except Exception:  # noqa: BLE001 — pure code, but the turn is worth more
        return None


def containers(paths: list[Any], extensions: frozenset[str]) -> frozenset[str]:
    """Documents that are also the index page of a section.

    A Confluence sync writes a page with children as ``…/X.md`` beside a folder
    ``…/X/`` holding them (``build_vault_path`` in ``app.confluence.convert``),
    so "this document has descendants" is decidable from the listing alone — no
    extra request, no guess about what a title means.

    The distinction is what keeps the evidence hedge (:func:`app.corpus_scope.
    hedge`) off the one population it could reach: a question the classifier
    called corpus-wide, answered completely by ONE page, is almost always a
    question about a section answered by that section's own index — «какие
    сервисы входят в продукт Fincert?» → ``Продукты/Fincert.md``. All six such
    questions in the acceptance set land on a page with descendants.

    Pure and total: an unusable listing yields an empty set, which is a real
    answer («no page here indexes anything») and not the same as "unknown" — the
    caller distinguishes those by getting ``None`` from
    :func:`container_paths`.
    """
    docs = _documents(paths, extensions)
    known = {"/".join(segments) for segments in docs}
    prefixes = {
        "/".join(segments[:i]) + "/"
        for segments in docs
        for i in range(1, len(segments))
    }
    out = set()
    for path in known:
        stem = path.rsplit(".", 1)[0] if "." in path.rsplit("/", 1)[-1] else path
        if f"{stem}/" in prefixes:
            out.add(path)
    return frozenset(out)


async def container_paths(cv: dict[str, Any] | None = None) -> frozenset[str] | None:
    """Section-index documents of this vault, or ``None`` when unknown.

    Reads the SAME cached listing and catalogue as :func:`corpus_block`, so it
    costs no extra request. ``None`` means the shape of the base could not be
    established this turn — see :func:`containers` for why that is not the same
    as an empty set.
    """
    listing = await _listing(cv)
    if listing is None:
        return None
    try:
        return containers(listing[0], listing[1])
    except Exception:  # noqa: BLE001 — the turn is worth more than the set
        return None


async def document_count(cv: dict[str, Any] | None = None) -> int | None:
    """How many documents the vault holds, or ``None`` when unknown.

    Reads the SAME cached listing and the SAME cached catalogue as
    :func:`corpus_block`, so asking for the number next to the footprint costs no
    extra request. Used by the evidence hedge, which says "one document out of N"
    and must not invent N — and must not inflate N with files that are never
    indexed, which is why the extension set is fetched rather than assumed.
    """
    listing = await _listing(cv)
    if listing is None:
        return None
    try:
        return len(_documents(listing[0], listing[1])) or None
    except Exception:  # noqa: BLE001 — the turn is worth more than the number
        return None
