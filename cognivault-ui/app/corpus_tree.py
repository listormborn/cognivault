"""The section tree of the base, rendered as material an answer can be built from.

The footprint (:mod:`app.corpus_map`) states the *scale* of the corpus: 127
documents, five of them shown. It deliberately says nothing about what is in
them, so it cannot answer «какие витрины ClickHouse описаны в базе?» — and
neither can retrieval, because no document in the corpus is a list of the
corpus. The page that would be that list, ``Продукты``, exists and is **empty**;
the list of products exists only as the *shape* of the tree.

This module renders that shape: every indexed document, nested by folder, with
the annotations the indexer already wrote. It is an ADDITIONAL block, never a
replacement for «Источники» — the retrieved fragments are selected, graded and
rendered exactly as before, and the grader stays the only thing that can decide
"not in my documents". A turn the grader refuses never reaches this code.

Load-bearing decisions:

* **Behind a config flag** (``rag.corpus_tree_enabled``, default off). The tree
  is this informative only because a Confluence hierarchy *is* the product tree.
  A vault laid out as ``2024/Q1/заметки`` would render a calendar and invite the
  model to answer questions about products with the names of months.
* **The catalogue, not the file listing.** ``GET /api/vault/catalog`` lists what
  is INDEXED; ``GET /api/vault/files`` lists what is on disk. The two genuinely
  disagree (the poller runs on its own cycle) and only the first is what search
  can return. A tree used to answer "what does this base cover" must not name a
  document no search will ever cite.
* **Titles for everything, descriptions for the top two levels.** The full
  catalogue is ~18 000 characters for 127 documents — far past what can ride on
  every turn on top of a 24 000-character sources block. The tree of titles is
  ~6 200 characters and it is the part that ANSWERS an enumeration: the four
  target questions («какие витрины…», «что лежит в разделе Архив», «какие
  страницы входят в раздел…», «какие пользовательские инструкции есть») are
  answered by names, not by prose. Annotations are therefore spent where a name
  is genuinely opaque — the top-level sections and their immediate children are
  product names («Marksman», «General API», «PSI»), while everything deeper is
  already a self-describing page title («Пользовательская инструкция. BMPF»).
* **Hide archived branches.** The user-facing catalogue describes the active
  knowledge base only. Pages under folders named ``Архив``/``archive`` (or
  explicitly named as deprecated/obsolete) stay indexed for compatibility but
  are not advertised as available topics. Container and empty-page markers keep
  the remaining inventory honest.
  The human-written list pages are no substitute: «База знаний» lists 2 of its 3
  children, «Fincert» 5 of 6, and «General» links outside its own subtree. The
  index is the only honest source for the tree, which is why the caption says so.
* **Silent degradation.** Any failure — endpoint missing on an older backend,
  timeout, malformed payload, an empty vault — yields ``None`` and the caller
  falls back to the footprint, i.e. to the behaviour that predates this module.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from . import catalog, corpus_map

log = logging.getLogger("cognivault-ui.corpus_tree")

# Budget for the whole rendered block, caption included. The reference corpus
# renders at ~9 400 characters with descriptions and ~6 900 without, so the cap
# is where the degradation ladder starts biting rather than a number the real
# vault sits under by luck. ~3 800 tokens at `tokens.CHARS_PER_TOKEN`; the
# sources block keeps its own budget untouched (see `rag._compute_budget` — the
# head block has never been part of it).
_MAX_TREE_CHARS = 9500

# The meta turn (a question about the base itself, no «Источники» at all) can
# afford the same block: there is nothing else in the message.
_MAX_OVERVIEW_CHARS = 9500

# Depth, 1-based after the corridor prefix is stripped, down to which a
# document's annotation is rendered. 1 = top-level sections, 2 = their children.
_DESC_MAX_DEPTH = 2

# One annotation, trimmed to its first sentence and then hard-capped. Long enough
# to say what a product is, short enough that 25 of them cost ~3 000 characters.
_DESC_MAX_CHARS = 140

# Byte size at or below which a document has no body of its own — only its
# frontmatter and its heading. TWO bounds, because the size test is a tie-break
# on top of "no annotation" and how far it may be trusted depends on whether
# that first signal says anything at all.
#
# Measured on the reference corpus (127 Confluence pages, sizes in BYTES as the
# catalogue reports them — Cyrillic is two bytes a character, so this is roughly
# double the character count): the 17 pages that produce zero chunks span
# 480–651 bytes, and the smallest page with any body at all is 652 bytes with a
# 56-character body. The gap is one byte wide, so size ALONE cannot separate
# them — which is the point.
#
# * With annotations present, an annotated page is by construction a page that
#   produced chunks, so it is excluded before size is even consulted. The bound
#   can then be loose enough to cover every container page (frontmatter grows
#   with the title and the ancestor chain) without touching a real page — as
#   long as that page's annotation actually exists. Annotation is best-effort
#   (one index-time chat call per document), and a page whose call failed looks
#   exactly like a container page to this test: on the reference corpus TEN of
#   the 110 pages with a body are under 1000 bytes, and four of them
#   («Описание витрин», «Потоки наполнения витрин», «Data Lineage REST APIs»,
#   «OASIS UI») are the very pages that answer the acceptance set's section
#   questions. A lost annotation there marks a real page «пусто» while a
#   fragment of it sits in «Источники» below. There is no second signal in the
#   catalogue that separates the two cases (`documents_with_summary < total` is
#   the normal, healthy state — 17 pages legitimately have no annotation), so
#   this is a known, bounded misreport rather than something the bound can fix.
# * With no annotations anywhere — `INDEX_DOC_SUMMARY` off, or a non-gigachat
#   provider — "no annotation" is true of everything and stops discriminating.
#   Size is then the only evidence and gets the strict bound, which under-reports
#   rather than calling a short page empty.
_EMPTY_MAX_BYTES = 1000
_EMPTY_MAX_BYTES_UNANNOTATED = 600

# How many single-child levels the fold may swallow into the heading. A
# Confluence sync needs four (``Confluence/<space>/<ancestor>/<ancestor>``); the
# cap is a guard against a pathological vault, not a design limit.
_MAX_DESCENT = 8

# Segments that hide an archived branch from the active catalogue. Narrow on
# purpose — «old» and «backup» are ordinary words in a product tree.
_ARCHIVE_RE = re.compile(r"(?:^|\W)(?:архив|archive[ds]?|deprecated|устаревш)", re.I)

# Sentence end for trimming an annotation. ``\s`` is required so that
# «т.д.», «v 2.1» and «Fincert.Playground» are not read as sentence ends.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")

_DOC_FORMS = ("документ", "документа", "документов")
_FRAGMENT_FORMS = ("фрагмент", "фрагмента", "фрагментов")

# --------------------------------------------------------------------------- #
# The two framing texts
# --------------------------------------------------------------------------- #
#
# Code-owned, not prompt wording, for the same reason as the footprint's caption:
# `prompts.system` is stored per user, so a user who saved their own copy would
# never receive a sentence added later. The caption travels inside the block.

_CAPTION = (
    "Структура базы знаний — полный перечень разделов и страниц. Это НЕ блок "
    "«Источники»: текста документов здесь нет, ссылаться на него как "
    "[Источник N] нельзя. По нему можно ответить, ЧТО есть в базе и в каком "
    "разделе оно лежит; о содержании страницы по одному её названию не суди."
)

_OVERVIEW_CAPTION = (
    "Структура базы знаний — полный перечень разделов и страниц, построенный по "
    "поисковому индексу. Поиск по документам для этого вопроса не выполнялся: "
    "отвечай по перечню ниже. Это названия страниц, а не их содержимое."
)

_TAIL = (
    "Перечень построен по поисковому индексу, а не по страницам-оглавлениям: на "
    "самих страницах-списках перечислено не всё. Если ответ строится по этому "
    "перечню — так и скажи, что это названия разделов и страниц базы."
)

# Legend entries, emitted only for the markers that actually occur in the render
# — explaining a marker the reader will never see is pure cost.
_LEGEND = {
    "раздел": "[раздел: N] — страница-контейнер, внутри неё N документов",
    "пусто": (
        "[пусто] — у страницы нет собственного текста, только заголовок: "
        "поиск по ней ничего не вернёт"
    ),
}
_LEGEND_ORDER = ("раздел", "пусто")


# --------------------------------------------------------------------------- #
# Tree construction (pure)
# --------------------------------------------------------------------------- #


@dataclass
class _Node:
    """One page of the base: its own document (maybe) plus its children."""

    name: str
    summary: str | None = None
    size: int | None = None
    #: ``True`` when a catalogue row exists for this node itself. A node without
    #: one is a folder with no page — possible in a plain vault, never in a
    #: Confluence sync.
    is_document: bool = False
    children: list["_Node"] = field(default_factory=list)

    @property
    def subtree_documents(self) -> int:
        """Documents inside this node, EXCLUDING the node's own page."""
        return sum(
            c.subtree_documents + (1 if c.is_document else 0) for c in self.children
        )


def _segments(path: str) -> list[str]:
    """Path → node names: separators normalised, extension dropped from the leaf."""
    parts = [s for s in path.replace("\\", "/").strip().split("/") if s and s != "."]
    if not parts:
        return []
    leaf = parts[-1]
    if "." in leaf:
        leaf = leaf.rsplit(".", 1)[0]
    parts[-1] = leaf or parts[-1]
    return parts


def build(documents: list[dict[str, Any]]) -> tuple[list[_Node], str]:
    """``(roots, prefix)`` — the tree of ``documents`` with corridors folded away.

    ``prefix`` is the chain of single-child levels swallowed into the block's
    heading. A Confluence vault nests four of them before it branches
    (``Confluence/OASISEXT/OASIS External Home/<раздел>``); rendering them as
    tree levels would spend four lines and an indent level saying nothing.

    Descending is lossless in the only sense that matters here: a swallowed level
    has exactly one child, so no branch is dropped. The swallowed node's OWN page
    stops being listed — the counts still come from the catalogue's ``total``, so
    the numbers stay honest, and the heading names the level so the reader knows
    where the listing starts.
    """
    root = _Node(name="")
    index: dict[tuple[str, ...], _Node] = {(): root}

    for entry in documents:
        path = str(entry.get("path") or "")
        segments = _segments(path)
        if not segments:
            continue
        node = root
        for i, name in enumerate(segments):
            key = tuple(segments[: i + 1])
            child = index.get(key)
            if child is None:
                child = _Node(name=name)
                index[key] = child
                node.children.append(child)
            node = child
        node.is_document = True
        summary = entry.get("summary")
        node.summary = summary.strip() if isinstance(summary, str) else None
        size = entry.get("size")
        node.size = size if isinstance(size, int) and not isinstance(size, bool) else None

    roots = root.children
    prefix: list[str] = []
    while len(roots) == 1 and roots[0].children and len(prefix) < _MAX_DESCENT:
        prefix.append(roots[0].name)
        roots = roots[0].children
    return roots, "/".join(prefix)


# --------------------------------------------------------------------------- #
# Rendering (pure)
# --------------------------------------------------------------------------- #


def _is_empty(node: _Node, annotated: bool = True) -> bool:
    """Whether the node's own page carries no text of its own.

    Two signals, and both are needed. ``summary is None`` alone over-reports: a
    deployment that cannot run the annotator at all (``EMBEDDING_PROVIDER`` is not
    gigachat, or ``INDEX_DOC_SUMMARY`` is off) has ``summary is None`` on every
    row. ``size`` alone over-reports the other way: in a vault with no
    frontmatter a 500-byte note is a real note. Together they are the claim the
    catalogue schema was built to support — a null summary on a ~500-byte
    container page told apart from a null on a full page whose annotation failed.

    ``annotated`` says whether the catalogue carries annotations at all; it
    picks the size bound (see :data:`_EMPTY_MAX_BYTES`).
    """
    if not node.is_document or node.summary:
        return False
    limit = _EMPTY_MAX_BYTES if annotated else _EMPTY_MAX_BYTES_UNANNOTATED
    return node.size is not None and node.size <= limit


def _marks(node: _Node, annotated: bool) -> tuple[str, set[str]]:
    """``(" [раздел: 6, пусто]", {"раздел", "пусто"})``, or empty.

    One bracket group per line rather than three: three groups on a 40-character
    title is noise, and noise is what a reader skips.
    """
    parts: list[str] = []
    used: set[str] = set()
    inside = node.subtree_documents
    if inside:
        parts.append(f"раздел: {inside}")
        used.add("раздел")
    if _is_empty(node, annotated):
        parts.append("пусто")
        used.add("пусто")
    if not parts:
        return "", used
    return f" [{', '.join(parts)}]", used


def _short(summary: str | None) -> str:
    """First sentence of an annotation, whitespace collapsed, hard-capped."""
    if not summary:
        return ""
    text = " ".join(str(summary).split())
    if not text:
        return ""
    first = _SENTENCE_END.split(text, maxsplit=1)[0].strip()
    if len(first) > _DESC_MAX_CHARS:
        first = first[: _DESC_MAX_CHARS - 1].rstrip(" ,;:—-") + "…"
    return first


def _lines(
    nodes: list[_Node],
    *,
    max_depth: int,
    desc_depth: int,
    annotated: bool,
    depth: int = 1,
) -> tuple[list[str], set[str]]:
    """Render ``nodes`` as indented lines; returns the lines and the markers used.

    ``max_depth`` cuts the tree: a node at the limit is still listed, with its
    ``[раздел: N]`` marker saying how many documents were folded into it, so a
    truncated tree under-promises instead of hiding the remainder.
    """
    out: list[str] = []
    used: set[str] = set()
    for node in sorted(nodes, key=lambda n: n.name.lower()):
        mark, mark_used = _marks(node, annotated)
        used |= mark_used
        line = f"{'  ' * (depth - 1)}- {node.name}{mark}"
        if depth <= desc_depth:
            desc = _short(node.summary)
            if desc:
                line += f" — {desc}"
        out.append(line)
        if node.children and depth < max_depth:
            child_lines, child_used = _lines(
                node.children,
                max_depth=max_depth,
                desc_depth=desc_depth,
                annotated=annotated,
                depth=depth + 1,
            )
            out.extend(child_lines)
            used |= child_used
    return out, used


def _tree_depth(nodes: list[_Node], depth: int = 1) -> int:
    return max(
        (_tree_depth(n.children, depth + 1) for n in nodes if n.children), default=depth
    )


def _compose(
    roots: list[_Node],
    *,
    total: int,
    listed: int,
    n_sources: int | None,
    prefix: str,
    caption: str,
    tail: str,
    max_depth: int,
    desc_depth: int,
    annotated: bool,
) -> str:
    """Assemble caption → scale → legend → tree → tail."""
    scale = f"Всего документов в базе: {total}."
    if listed and listed < total:
        scale += (
            f" В перечне ниже показаны {listed} "
            f"{corpus_map._plural(listed, _DOC_FORMS)} из них."  # noqa: SLF001
        )
    if isinstance(n_sources, int) and n_sources > 0:
        scale += (
            f" Ниже в блоке «Источники» — {n_sources} "
            f"{corpus_map._plural(n_sources, _FRAGMENT_FORMS)} "  # noqa: SLF001
            "с текстом документов; остальное содержимое базы не показано."
        )

    lines, used = _lines(
        roots, max_depth=max_depth, desc_depth=desc_depth, annotated=annotated
    )
    heading = (
        f"Разделы и страницы внутри «{prefix}»:"
        if prefix
        else "Разделы и страницы базы:"
    )
    legend = [_LEGEND[key] for key in _LEGEND_ORDER if key in used]

    parts = [caption, scale]
    if legend:
        parts.append("Пометки: " + "; ".join(legend) + ".")
    parts.append(heading)
    parts.extend(lines)
    parts.append(tail)
    return "\n".join(parts)


def _squeeze(
    roots: list[_Node],
    *,
    total: int,
    listed: int,
    n_sources: int | None,
    prefix: str,
    caption: str,
    tail: str,
    max_chars: int,
    annotated: bool,
) -> str:
    """Render under ``max_chars`` by degrading in a fixed, honest order.

    Descriptions go first — they are the enrichment, the names are the answer —
    and they go one level at a time, so a block 4% over the cap loses the
    annotations of the second level rather than all of them. Then depth, one level at a time from the bottom: a folded node keeps its
    ``[раздел: N]`` count, so the reader is told what was folded instead of
    being handed a shorter list that looks complete. The last rung (top level
    only) is returned even if it still overflows: a corpus whose top-level
    section names alone do not fit has nothing shorter left to say, and honest
    numbers beat a ceiling.
    """
    deepest = _tree_depth(roots)
    ladder: list[tuple[int, int]] = [
        (deepest, d) for d in range(_DESC_MAX_DEPTH, -1, -1)
    ]
    ladder += [(d, 0) for d in range(deepest - 1, 0, -1)]
    text = ""
    for max_depth, desc_depth in ladder:
        text = _compose(
            roots,
            total=total,
            listed=listed,
            n_sources=n_sources,
            prefix=prefix,
            caption=caption,
            tail=tail,
            max_depth=max_depth,
            desc_depth=desc_depth,
            annotated=annotated,
        )
        if len(text) <= max_chars:
            return text
    return text


def _prepare(
    data: dict[str, Any] | None,
) -> tuple[list[_Node], str, int, int, bool] | None:
    """``(roots, prefix, total, listed, annotated)`` for a payload, or ``None``.

    ``None`` whenever the payload says nothing usable — a missing/garbled body,
    ``status == "empty_vault"``, or no document rows. Note that a catalogue with
    documents but NO annotations is perfectly usable: the tree is then titles and
    markers, which is most of its value.
    """
    if not isinstance(data, dict):
        return None
    if str(data.get("status") or "") == "empty_vault":
        return None
    # Archive pages may remain in the backend index for compatibility, but they
    # are not part of the active user-facing knowledge base.
    documents = active_documents(data)
    if not documents:
        return None
    roots, prefix = build(documents)
    if not roots:
        return None
    total = len(documents)
    annotated = any(d.get("summary") for d in documents)
    return roots, prefix, total, len(documents), annotated


def active_documents(data: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Catalogue documents visible to users, excluding every archived branch."""
    if not isinstance(data, dict):
        return []
    return [
        document
        for document in catalog.documents_of(data)
        if not any(
            _ARCHIVE_RE.search(segment)
            for segment in _segments(str(document.get("path") or ""))
        )
    ]


def render(
    data: dict[str, Any] | None,
    n_sources: int | None = None,
    max_chars: int = _MAX_TREE_CHARS,
) -> str | None:
    """The tree as an added block next to «Источники», or ``None``."""
    prepared = _prepare(data)
    if prepared is None:
        return None
    roots, prefix, total, listed, annotated = prepared
    return _squeeze(
        roots,
        total=total,
        listed=listed,
        n_sources=n_sources,
        prefix=prefix,
        caption=_CAPTION,
        tail=_TAIL,
        max_chars=max_chars,
        annotated=annotated,
    )


def render_overview(
    data: dict[str, Any] | None, max_chars: int = _MAX_OVERVIEW_CHARS
) -> str | None:
    """The tree as the ONLY material of a meta turn, or ``None``.

    Same tree, same markers, same ladder — only the caption differs. The «не
    источник» clause is absent here for the reason :mod:`app.corpus_map` states:
    on a meta turn there is no «Источники» block to defer to, and a caption that
    calls the block un-citable talks the model out of the only material it has.
    """
    prepared = _prepare(data)
    if prepared is None:
        return None
    roots, prefix, total, listed, annotated = prepared
    return _squeeze(
        roots,
        total=total,
        listed=listed,
        n_sources=None,
        prefix=prefix,
        caption=_OVERVIEW_CAPTION,
        tail=_TAIL,
        max_chars=max_chars,
        annotated=annotated,
    )


def top_level_titles(
    data: dict[str, Any] | None, *, limit: int = 8
) -> list[str]:
    """Return real top-level section names for lightweight UI navigation.

    The same corridor folding as the model-facing tree is load-bearing here:
    showing ``Confluence`` or a space key as a suggested topic would expose
    storage layout rather than tell the user what the base contains. Invalid,
    unavailable and empty catalogues deliberately produce an empty list so the
    caller can retain its static fallback without inventing sections.
    """
    if limit <= 0:
        return []
    prepared = _prepare(data)
    if prepared is None:
        return []
    roots = prepared[0]
    return [node.name for node in sorted(roots, key=lambda node: node.name.lower())][
        :limit
    ]


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def enabled(rcfg: dict[str, Any] | None) -> bool:
    """Whether ``rag.corpus_tree_enabled`` is on for this caller.

    Strict: only a real boolean ``True`` enables the tree. A stored string or a
    number means a config that was never validated, and the safe reading of an
    unvalidated flag is "off" — the feature is a bet on the vault's shape.
    """
    if not isinstance(rcfg, dict):
        return False
    return rcfg.get("corpus_tree_enabled") is True


async def tree_block(
    cv: dict[str, Any] | None = None, n_sources: int | None = None
) -> str | None:
    """The rendered tree for this turn, or ``None`` — the caller then degrades.

    ``None`` is never an error the user sees: :mod:`app.rag` falls back to the
    footprint, which is the pre-existing behaviour.
    """
    data = await catalog.payload(cv)
    if data is None:
        return None
    try:
        return render(data, n_sources)
    except Exception:  # noqa: BLE001 — pure code, but the turn is worth more
        log.exception("не удалось отрисовать дерево разделов")
        return None


async def overview_block(cv: dict[str, Any] | None = None) -> str | None:
    """The rendered tree for a meta turn, or ``None`` — the caller then degrades."""
    data = await catalog.payload(cv)
    if data is None:
        return None
    try:
        return render_overview(data)
    except Exception:  # noqa: BLE001 — pure code, but the turn is worth more
        log.exception("не удалось отрисовать дерево разделов")
        return None
