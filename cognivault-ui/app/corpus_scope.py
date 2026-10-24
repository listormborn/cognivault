"""Questions about the base itself: recognising them, and qualifying answers.

Three pure pieces, no model calls anywhere in this module:

* :func:`match_meta` — a deterministic recogniser for «что ты знаешь?» and its
  close relatives. It exists because the condense step *cannot* see them: with
  no history there is nothing to resolve against, so ``rag_pipeline.condense``
  short-circuits and the classifier never runs on the first turn — and a
  question about the assistant is almost always the first turn. On a miss it
  returns ``None`` and the caller does exactly what it did before. It takes
  ``has_history`` for the same reason it exists: the patterns whose object is
  ELIDED («какие разделы?») are only unambiguous while there is nothing for the
  ellipsis to point at — see :data:`_PATTERNS`.
* :data:`SCOPES` / :func:`parse_scope` — the ``scope`` field the condense call
  now returns alongside ``intent``. Absent, unknown or malformed ⇒
  :data:`DEFAULT_SCOPE` (``document``), which is today's behaviour. The
  tolerance is for the MODEL, not for a user: the condense prompt is read-only
  (``config_routes._readonly_prompts``), but a model that answers with a JSON
  object missing one key must not break the route.
* :func:`hedge` — the evidence-concentration caveat, shown only when the
  question was corpus-wide AND every selected fragment came from one section,
  and that section is not the base's own index page for it.

None of this touches the refusal path. The grader remains the only thing that
can decide "not in my documents": the matcher answers a question no document was
ever going to answer, and the hedge only qualifies an answer that was already
produced.

Why a matcher at all, rather than one more prompt sentence: prompts are stored
per user, so a user who saved their own ``prompts.system`` holds a frozen copy
and never receives wording added later. Code reaches everybody.
"""

from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------------- #
# Scope (the condense field)
# --------------------------------------------------------------------------- #

#: ``document`` — the answer is expected to live inside one document (the fields
#: of a table, the steps of a procedure). ``corpus`` — the question is about the
#: base as a whole (what products/sections/documents exist at all).
SCOPES: tuple[str, str] = ("document", "corpus")

#: What an absent/unknown ``scope`` means. Deliberately the one that changes
#: nothing: no hedge, no new branch — the behaviour that predates this field.
DEFAULT_SCOPE = "document"
CORPUS_SCOPE = "corpus"


def parse_scope(raw: Any) -> str:
    """Map the model's ``scope`` field onto :data:`SCOPES`.

    Tolerant on purpose — the field arrives from a model, so ``None``, a missing
    key, a number, or a hallucinated value all collapse to :data:`DEFAULT_SCOPE`
    rather than raising or inventing a branch.
    """
    if not isinstance(raw, str):
        return DEFAULT_SCOPE
    value = raw.strip().strip('"').lower()
    return value if value in SCOPES else DEFAULT_SCOPE


# --------------------------------------------------------------------------- #
# Meta-question matcher
# --------------------------------------------------------------------------- #

#: The two families :func:`match_meta` returns. They are NOT decoration: they
#: select different material in :func:`app.rag._build_meta` — the section tree
#: for :data:`META_CORPUS`, the assistant's own operating rules (plus the tree,
#: when available) for :data:`META_ASSISTANT` — and therefore different
#: fail-closed behaviour. A question about the base cannot be answered without
#: the base's structure; a question about the assistant always can.
META_ASSISTANT = "assistant"
META_CORPUS = "corpus"

# Longer than this and the turn is carrying subject matter, not a bare question
# about scope. A cheap guard in front of the patterns (which are anchored anyway)
# so a pathological input never walks the whole list.
_MAX_META_CHARS = 200

# Openers that carry no meaning for the match and are stripped from the head of
# every clause: greetings, politeness, and the verbs that introduce a request.
_FILLERS = (
    "привет",
    "приветствую",
    "здравствуй",
    "здравствуйте",
    "добрый день",
    "добрый вечер",
    "доброе утро",
    "слушай",
    "слушайте",
    "скажи",
    "скажите",
    "подскажи",
    "подскажите",
    "расскажи",
    "расскажите",
    # Imperatives that introduce a request and carry no meaning of their own.
    # «Перечисли разделы базы» is «разделы базы» with a verb in front; without
    # these the clause missed every pattern and fell through to retrieval, where
    # the only possible outcome is the grader's «в документах ответа не нашлось»
    # — the exact refusal this branch exists to remove.
    "перечисли",
    "перечислите",
    "покажи",
    "покажите",
    "назови",
    "назовите",
    "опиши",
    "опишите",
    "пожалуйста",
    "кстати",
    "а",
    "ну",
)
_FILLER_RE = re.compile(rf"^(?:{'|'.join(_FILLERS)})\b[\s,]*")

# Sentence split. A meta turn is often two short questions in a row («Что ты
# вообще знаешь? О чём эта база?»), and EVERY clause has to be a meta clause for
# the turn to match — see `match_meta`.
_CLAUSE_SPLIT = re.compile(r"[?!.;\n]+")

# Characters that only add noise to a match: quotes and the stray punctuation
# people leave around a question.
_STRIP_CHARS = " \t«»\"'`“”„-—–:,()[]"

#: Anchored formulations, each matched against a WHOLE clause. Two families:
#:
#: * ``assistant`` — about the assistant itself («что ты знаешь», «кто ты»,
#:   «всегда ли ответ в Markdown»);
#: * ``corpus`` — about the base as a whole («о чём эта база», «какие разделы»).
#:
#: The families select different material — see :data:`META_ASSISTANT`. The list
#: is narrow BY DESIGN: every pattern is a full-clause match with no room for a
#: topic qualifier, so «Что ты знаешь про PSI?» falls through to retrieval while
#: «Что ты знаешь?» does not. A pattern that would need a wildcard tail is a
#: pattern that belongs in retrieval.
#:
#: The middle field is ``first_turn_only``. This matcher never sees the history,
#: so every pattern has to be safe read as a BARE TURN-2 UTTERANCE, and a handful
#: are not: their object is elided, and on turn 2 the ellipsis has an antecedent.
#: After «Расскажи про продукт Fincert» the turn «Какие разделы?» means *его*
#: разделы, and answering it from the tree of the whole base is a wrong answer
#: delivered with no sources — the matcher is the only branch that bypasses the
#: grader, so a false positive here is not a missed refinement, it is a
#: substituted answer. Those patterns are therefore restricted to the turn where
#: the matcher is the ONLY thing that can act: on turn ≥2 ``condense`` runs, it
#: sees the history, and resolving exactly this anaphora is its job (the rewrite
#: «Какие разделы в продукте Fincert?» then goes to retrieval and the grader).
#: Restricting them costs the first-turn coverage nothing — with no history there
#: is no antecedent and the reading is unambiguous.
#:
#: A pattern is ``first_turn_only`` iff its clause leaves an object slot the
#: history can fill. A clause whose subject is «ты»/«вы» and whose object is
#: present («о чём ты знаешь», «какие темы ты покрываешь», «какая информация у
#: тебя есть») is NOT one: its anaphoric reading needs an explicit complement
#: («…у тебя есть по нему?»), and a complement breaks the full-clause match on
#: its own.
_PATTERNS: tuple[tuple[str, bool, str], ...] = (
    # --- about the assistant ------------------------------------------------ #
    ("assistant", False, r"что (?:ты |вы )?(?:вообще |такого )?зна(?:ешь|ете)(?: вообще)?"),
    # «ещё» is an anaphoric particle: it presupposes something already said. On
    # turn 1 there is nothing, so the clause is the plain «что ты знаешь». On
    # turn 2 «Что ты ещё знаешь?» after a document answer means «ещё про него».
    ("assistant", True, r"что (?:ты |вы )?еще зна(?:ешь|ете)(?: вообще)?"),
    ("assistant", True, r"что (?:ты |вы )?(?:вообще |такого )?зна(?:ешь|ете) еще"),
    ("assistant", False, r"что (?:ты |вы )?(?:вообще |ещё |еще )?(?:умеешь|умеете|можешь|можете)(?: делать)?"),
    ("assistant", False, r"кто (?:ты|вы)(?: такой| такая| такие)?"),
    ("assistant", False, r"что ты за (?:ассистент|бот|модель|помощник)(?: такой| такая)?"),
    ("assistant", False, r"(?:с )?чем (?:ты )?(?:можешь |умеешь )?(?:помочь|быть полезен)"),
    ("assistant", False, r"(?:о чем|о чём|про что) (?:ты )?(?:знаешь|можешь рассказать)"),
    ("assistant", False, r"(?:о себе|про себя)"),
    # --- about how the assistant works -------------------------------------- #
    #
    # A question about the assistant's own behaviour — the format of its
    # answers, where they come from — is answered from its OPERATING RULES
    # (`rag._operating_rules`), not from the corpus. There is no document in any
    # vault that describes this service, so before these patterns existed the
    # only reachable outcome was the grader's refusal (`x23-meta` of the
    # acceptance set). Still full-clause anchored: «в каком формате хранится
    # витрина?» carries a subject and goes to retrieval.
    (
        "assistant",
        False,
        r"(?:в каком (?:виде|формате)|как)(?: ты| вы)? "
        r"(?:отвечаешь|отвечаете|оформляешь ответ|оформляете ответ|"
        r"форматируешь ответ|форматируете ответ)",
    ),
    (
        "assistant",
        False,
        r"(?:всегда ли )?(?:твой |ваш |мой )?ответ(?:ы)? "
        r"(?:всегда |будет |будут )?(?:в |с )?(?:markdown|разметк[еиой]|"
        r"заголовками|разметкой)(?: с заголовками| с разметкой| и заголовками)?",
    ),
    (
        "assistant",
        False,
        r"(?:всегда ли |всегда )?(?:ты |вы )?(?:отвечаешь|отвечаете) "
        r"(?:в |с )(?:markdown|разметк[еиой]|заголовками)"
        r"(?: с заголовками| и заголовками)?",
    ),
    (
        "assistant",
        False,
        r"(?:ты |вы )?(?:используешь|используете)(?: ли)?(?: ты| вы)? "
        r"(?:markdown|разметку)(?: в ответах| в ответе)?",
    ),
    (
        "assistant",
        False,
        r"откуда (?:ты |вы )?бер[её](?:шь|те) (?:ответы|информацию|данные|факты)",
    ),
    ("assistant", False, r"как (?:ты |вы )?(?:работаешь|работаете|устроен|устроена)"),
    # Self-referential by construction: the clause names «ты»/«вы», so there is
    # no subject left for retrieval to look for.
    (
        "assistant",
        False,
        r"(?:какие|каких) (?:темы|тем|разделы|направления|вопросы) "
        r"(?:ты |вы )?(?:покрываешь|покрываете|охватываешь|охватываете|"
        r"знаешь|знаете)",
    ),
    (
        "assistant",
        False,
        r"на какие вопросы (?:ты |вы )?(?:можешь|можете) ответить",
    ),
    (
        "assistant",
        False,
        r"(?:(?:о чем|о чём|про что|что) "
        r"(?:я |мы )?(?:могу|можем) "
        r"(?:(?:у тебя|у вас|тебя|вас) )?спросить(?: у тебя| у вас)?|"
        r"(?:о чем|о чём|про что) можно (?:тебя|вас) спросить)",
    ),
    (
        "assistant",
        False,
        r"(?:какая|какие) (?:информация|документы|материалы|данные) "
        r"у (?:тебя|вас)(?: есть)?",
    ),
    # --- about the base as a whole ------------------------------------------ #
    ("corpus", False, r"(?:о чем|о чём|про что) (?:эта |эта твоя |твоя )?база(?: знаний)?"),
    ("corpus", False, r"что (?:это )?за база(?: знаний)?"),
    ("corpus", False, r"(?:о|об|про) (?:эту |этой )?баз[еу](?: знаний)?"),
    (
        "corpus",
        False,
        r"что (?:есть |лежит |хранится |содержится |находится )?"
        r"(?:у тебя )?в (?:твоей )?базе(?: знаний)?",
    ),
    (
        "corpus",
        False,
        r"(?:какие|каких) (?:есть )?(?:разделы|разделов|темы|тем|направления)"
        r"(?: есть)?(?: у тебя)? в (?:твоей )?базе(?: знаний)?",
    ),
    (
        "corpus",
        False,
        r"из каких разделов (?:состоит|состоит твоя) баз[аы](?: знаний)?",
    ),
    (
        "corpus",
        False,
        r"(?:какая |какова )?структур[ауы] (?:у )?базы(?: знаний)?",
    ),
    # «Какие есть разделы?» / «Разделы базы знаний» — the same question with the
    # «в базе» dropped or the verb moved. Both are full-clause anchored: the
    # noun list is closed and nothing may follow it, so «какие есть разделы у
    # витрины fincert_feeds» is not a match.
    #
    # FIRST TURN ONLY, and this is the pattern that made the rule necessary.
    # «Какие разделы?» with no «базы» in it is the base's own tree only while
    # nothing else has been named; after «Расскажи про продукт Fincert» it is
    # «какие разделы У НЕГО», and the measured behaviour was intent=meta, zero
    # model calls, zero sources, the whole corpus tree — a wrong answer with
    # nothing to check it against. On turn ≥2 the same wording reaches condense
    # («Какие в нём разделы?» already did, and was rewritten and searched
    # correctly), so nothing is lost by handing it over.
    (
        "corpus",
        True,
        r"(?:какие|каких) (?:есть )?"
        r"(?:разделы|разделов|темы|тем|направления)(?: есть)?",
    ),
    (
        "corpus",
        False,
        r"разделы базы(?: знаний)?",
    ),
    # «О каких продуктах ты знаешь?» — «знаешь» is second person: the subject is
    # the assistant's knowledge as a whole, and a follow-up reading would need an
    # explicit complement («…из них?»), which breaks the full-clause match.
    (
        "corpus",
        False,
        r"(?:о каких|про какие) (?:продуктах?|проектах?|темах?|системах?|"
        r"направлениях?) (?:ты )?знаешь",
    ),
    # The NOMINATIVE twin of the pattern above: «Какие продукты ты знаешь?».
    # Measured production miss — the list above covered the prepositional case
    # («о каких продуктах…») only, the clause is `fullmatch`ed so there is no
    # partial credit, and the one `какие X ты знаешь` pattern in the assistant
    # family carries a closed noun list without продукты/проекты. The question
    # therefore took the ordinary retrieval path and was answered from a single
    # archived page («Проекты Ислама») — one engineer's notes served as the list
    # of the company's projects.
    #
    # ANCHORED ON THE SECOND-PERSON VERB, and that is the whole safety argument:
    # this matcher BYPASSES the grader, so a false positive SUBSTITUTES an answer
    # instead of degrading one. «знаешь/знаете» names the assistant as the
    # subject, which leaves retrieval nothing to look for; a wildcard tail
    # («какие проекты.*») would swallow «какие проекты у Ислама» and answer it
    # with the tree of the whole base.
    #
    # The noun list deliberately excludes темы/разделы/направления: the assistant
    # pattern above already matches those with the same verbs, and
    # :func:`_clause_kind` returns the FIRST list match — repeating them here
    # would make the family of «какие темы ты знаешь?» depend on the order of
    # this tuple. Non-overlapping lists need no such rule to stay correct.
    #
    # CONSIDERED AND REJECTED: the bare, object-elided «какие проекты?». It would
    # need ``first_turn_only=True`` (after «Расскажи про Fincert» it means *его*
    # проекты) and it carries real hijack risk even on turn 1 — «какие проекты»
    # is also how one asks about the projects of a team, a person or a section,
    # and those are retrieval's, not the tree's.
    (
        "corpus",
        False,
        r"(?:какие|каких) (?:есть )?(?:продукты|продуктов|проекты|проектов|"
        r"системы) (?:ты |вы )?(?:знаешь|знаете)",
    ),
    # The possessive twin: «Какие продукты у нас есть?». Anchored on the
    # first-person-plural / second-person possessive instead of the verb —
    # «у нас»/«в компании» scopes the question to the base's owner, the same
    # reading «ты знаешь» gives. Nothing may follow it (full-clause match), so
    # «какие проекты у нас в Архиве» stays with retrieval.
    (
        "corpus",
        False,
        r"(?:какие|каких) (?:продукты|проекты|системы|направления) "
        r"(?:есть )?(?:у нас|у тебя|у вас|в компании)(?: есть)?",
    ),
    (
        "corpus",
        False,
        r"(?:о каких|про какие) (?:продуктах?|проектах?|темах?|системах?|"
        r"направлениях?) есть (?:информация|данные|материалы) "
        r"в базе(?: знаний)?",
    ),
    # The same clause with «в базе» dropped: «О каких продуктах есть информация?»
    # names the noun but not WHERE, and on turn 2 the previous answer is the
    # where. Same class as «какие разделы?», weaker only in that the wrong answer
    # is a superset of the right one rather than a different entity.
    (
        "corpus",
        True,
        r"(?:о каких|про какие) (?:продуктах?|проектах?|темах?|системах?|"
        r"направлениях?) есть (?:информация|данные|материалы)",
    ),
    (
        "corpus",
        False,
        r"какие (?:продукты|проекты|темы|разделы|направления) "
        r"(?:есть |описаны |представлены |лежат )?в базе(?: знаний)?",
    ),
    (
        "corpus",
        False,
        r"сколько (?:всего )?(?:страниц|документов|файлов|материалов) "
        r"(?:всего )?в базе(?: знаний)?"
        r"(?: и как (?:они )?(?:распределены|разложены|разбиты)"
        r"(?: по разделам)?)?",
    ),
)

_COMPILED: tuple[tuple[str, bool, re.Pattern[str]], ...] = tuple(
    (kind, first_turn_only, re.compile(pattern))
    for kind, first_turn_only, pattern in _PATTERNS
)


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, drop quotes and edge punctuation."""
    return " ".join(text.lower().replace("ё", "е").split()).strip(_STRIP_CHARS)


def _strip_fillers(clause: str) -> str:
    """Drop leading greetings/politeness («привет», «подскажи, …»), repeatedly."""
    text = clause
    while True:
        stripped = _FILLER_RE.sub("", text, count=1).strip(_STRIP_CHARS)
        if stripped == text:
            return text
        text = stripped


def _clause_kind(clause: str, has_history: bool) -> str | None:
    """The family of a single clause, or ``None`` if it is not a meta clause.

    ``has_history`` disables the ``first_turn_only`` patterns — the ones whose
    object is elided and would therefore be read against the previous turn (see
    :data:`_PATTERNS`).
    """
    for kind, first_turn_only, pattern in _COMPILED:
        if first_turn_only and has_history:
            continue
        if pattern.fullmatch(clause):
            return kind
    return None


def match_meta(question: str, *, has_history: bool = False) -> str | None:
    """``"assistant"`` / ``"corpus"`` for a question about the base, else ``None``.

    Deterministic, zero model calls, works on the very first turn — which is the
    whole point: the condense classifier is skipped when there is no history.

    ``has_history`` says whether this turn has any preceding dialogue — the SAME
    predicate ``rag_pipeline.condense`` calls ``first_turn`` (see
    :func:`app.rag_pipeline.has_history`). It narrows the pattern list to the
    formulations that cannot be read as a follow-up: this matcher never sees the
    history, so a pattern with an elided object («какие разделы?») is only
    unambiguous while there is no antecedent. The default is ``False`` — the
    first-turn reading, i.e. the widest match — because that is the worst case
    the standing invariants («0 hits on the 56-question control», «exactly 5 hits
    on the 251 golden questions») must hold under. A caller that forgets the
    argument therefore gets a measured behaviour, not an unmeasured one; the one
    production call site (:func:`app.rag._build_auto`) passes it.

    A turn matches only when EVERY one of its clauses matches — a bare greeting
    («привет!») is not a clause, it is stripped. That is what keeps the
    recogniser honest on compound input: «Что ты вообще знаешь? О чём эта база?»
    is two meta clauses and matches; «Привет! Какие колонки в таблице
    fincert_feeds?» keeps one subject-matter clause and does not. When the
    clauses disagree in family the corpus one wins — it is the one that decides
    what the answer is built from.

    ``None`` is the fail-closed answer and covers everything not on the narrow
    list, including every topic-qualified variant: «Что ты знаешь про PSI?»
    carries a subject and must go to retrieval, where the grader can refuse.
    """
    if not question or len(question) > _MAX_META_CHARS:
        return None
    normalised = _normalise(question)
    if not normalised:
        return None

    kinds: list[str] = []
    for raw in _CLAUSE_SPLIT.split(normalised):
        clause = _strip_fillers(raw.strip(_STRIP_CHARS))
        if not clause:
            continue
        kind = _clause_kind(clause, has_history)
        if kind is None:
            return None
        kinds.append(kind)
    if not kinds:
        return None
    return META_CORPUS if META_CORPUS in kinds else META_ASSISTANT


# --------------------------------------------------------------------------- #
# Evidence-concentration hedge
# --------------------------------------------------------------------------- #

_HEDGE_HEAD = "Оговорка: вопрос охватывает базу целиком"

_HEDGE_TAIL = (
    "Ответ описывает только его и не является перечнем по всей базе; в других "
    "разделах может быть больше."
)


def _paths(sources: list[dict[str, Any]]) -> set[str]:
    """The distinct document paths behind ``sources``, blanks dropped."""
    paths = {str(s.get("path") or "") for s in sources if isinstance(s, dict)}
    paths.discard("")
    return paths


def _label(sources: list[dict[str, Any]], path: str) -> str:
    """Human-readable name of the document: its title, else the file stem."""
    for source in sources:
        if isinstance(source, dict) and str(source.get("path") or "") == path:
            title = str(source.get("title") or "").strip()
            if title and title != path:
                return title
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0] or path


def _plural_docs(n: int) -> str:
    """Russian plural of «документ» for ``n`` (1 / 2-4 / 5+)."""
    if n % 10 == 1 and n % 100 != 11:
        return "документ"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return "документа"
    return "документов"


def _concentration(
    sources: list[dict[str, Any]], containers: frozenset[str]
) -> tuple[str, int] | None:
    """``(name, n_documents)`` the evidence collapsed onto, or ``None``.

    ``n == 1`` names a document, ``n > 1`` names the single section its documents
    share — the two shapes the caveat can talk about.

    Two shapes count as concentrated, and the second is why the first is not
    enough. Chunks of ``…/Fincert/A.md`` and ``…/Fincert/B.md`` are two paths and
    used to pass as breadth, while 7.3% of the corpus sits in cross-file
    near-duplicate clusters: two sibling pages of one section are no more of a
    survey of the base than one page is.

    The exception is the whole reason this function takes ``containers``. A
    single document that is itself the index page of a section (it has
    descendants in the listing) is the base's OWN enumeration: «какие сервисы
    входят в продукт Fincert?» answered from ``Продукты/Fincert.md`` is
    complete, and a caveat there is a false alarm. All six answerable
    corpus-scope questions of the acceptance set have exactly that shape, and
    they were the only population this hedge could reach.
    """
    paths = _paths(sources)
    if not paths:
        return None
    if len(paths) == 1:
        path = next(iter(paths))
        if path in containers:
            return None
        return _label(sources, path), 1
    directories = {path.rsplit("/", 1)[0] if "/" in path else "" for path in paths}
    if len(directories) != 1:
        return None
    directory = next(iter(directories))
    if not directory:
        # Loose files in the vault root share no section — that is breadth.
        return None
    return directory.rsplit("/", 1)[-1], len(paths)


def hedge(
    scope: str,
    sources: list[dict[str, Any]] | None,
    total_docs: int | None = None,
    containers: frozenset[str] | None = None,
) -> str | None:
    """The caveat to append to a corpus-wide answer built on concentrated evidence.

    Returns ``None`` — no caveat — in every other case, and those cases are the
    point:

    * ``scope != "corpus"``: a question about one document answered from that
      document is exactly right, and the 56 control questions («перечисли все
      поля витрины X») live here. Since :func:`parse_scope` maps a missing field
      to ``document``, a failed, disabled or skipped condense call cannot produce
      a caveat either;
    * evidence from two or more sections: the answer already spans the base,
      which is what a corpus-wide question needs;
    * evidence concentrated on a section's own index page: that page IS the
      enumeration the question asked for (see :func:`_concentration`);
    * no sources at all: there is no answer to qualify (the grader's refusal
      path never reaches here);
    * ``containers is None``: the shape of the base could not be established
      this turn, so the container exception cannot be evaluated. Silence is the
      fail-closed answer — the caveat's only reachable population without it is
      the one where it is wrong.

    The text names what the evidence collapsed onto and the size of the base
    rather than hedging in the abstract, because "this may be incomplete" on
    every answer trains the reader to skip it. ``total_docs`` is omitted from the
    sentence when unknown — an invented denominator would be worse than none.
    """
    if scope != CORPUS_SCOPE or not sources:
        return None
    if isinstance(total_docs, int) and total_docs <= 1:
        return None
    if containers is None:
        return None
    found = _concentration(list(sources), containers)
    if found is None:
        return None
    name, count = found

    known_total = isinstance(total_docs, int) and total_docs > 1
    scale = f" из {total_docs} в базе" if known_total else ""
    subject = (
        f"из одного документа «{name}»{scale}"
        if count == 1
        else f"из одного раздела «{name}» — {count} {_plural_docs(count)}{scale}"
    )
    return f"{_HEDGE_HEAD}, а все найденные фрагменты — {subject}. {_HEDGE_TAIL}"
