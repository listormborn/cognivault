"""RAG-промпт волны 1: правила в system, контекст — в последнем user-сообщении.

Покрывает:
* :func:`app.rag.build_rag_context` — новый контракт :class:`app.rag.RagContext`,
  разделение «правила / источники», порядок «Источники → Напоминание → Вопрос»,
  нумерация блоков, жёсткий кап на число блоков, пустой ``section_path``;
* ``POST /api/chat`` — сборка ``[system, ...история..., user-с-контекстом]``,
  прежний последний вопрос не дублируется, история между ними режется;
* :func:`app.routes.chat_routes._invalid_citations` — серверная валидация цитат.

Мокается ТОЛЬКО транспорт (``rag.cognivault.hybrid_search`` / ``.content``) и
скрытые LLM-вызовы волны 2 (``rag_pipeline.llm.complete_json``), поэтому
тестируется настоящий ``rag.py``, а не заглушка.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import rag, rag_pipeline, settings  # noqa: E402
from app.config import AppPaths  # noqa: E402
from app.main import create_app  # noqa: E402
from app.routes import chat_routes  # noqa: E402


def _paths(tmp_path) -> AppPaths:
    return AppPaths(root=tmp_path / ".cognivault-ui")


@pytest.fixture(autouse=True)
def _local_mode(monkeypatch):
    """Force LOCAL mode (no bearer header required)."""
    monkeypatch.setattr(settings, "is_server", lambda: False)


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into ``[(event, data_dict), ...]``."""
    out: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if event is not None:
            out.append((event, data or {}))
    return out


# --------------------------------------------------------------------------- #
# Фикстуры данных
# --------------------------------------------------------------------------- #


def _hit(
    i: int,
    *,
    section: str = "",
    text: str | None = None,
    section_text: str | None = None,
) -> dict:
    hit = {
        "path": f"doc{i}.md",
        "title": f"Документ {i}",
        "section_path": section,
        "score": 1.0 - i / 100,
        "text": text if text is not None else f"содержимое фрагмента номер {i}",
        "chunk_index": i,
        "rank": i,
    }
    if section_text is not None:
        # Волна 3: бэкенд отдаёт эти поля при group_by_section=True.
        hit["section_text"] = section_text
        hit["parent_id"] = f"doc{i}.md#section"
    return hit


def _install_retrieval(monkeypatch, hits: list[dict], contents: dict | None = None):
    """Замокать транспорт CogniVault: hybrid-поиск и выдачу документов.

    Заодно затыкается GigaChat волны 2: condense возвращает вопрос как есть,
    грейдер — «5» каждому фрагменту (порядок поиска сохраняется, ср. tie-break
    по ранку), поэтому эти тесты продолжают проверять сборку промпта, а не
    отбор — он живёт в ``test_rag_pipeline.py``.

    Дублёр принимает ``**kwargs``: ``rag`` зовёт поиск с волновыми ключами
    (``group_by_section``/``section_max_chars``), и жёсткая сигнатура падала бы
    с ``TypeError``. Перехваченный вызов — ``(query, limit, kwargs)``.
    """
    calls: list[tuple[str, int, dict]] = []

    async def fake_hybrid(query, limit, cv=None, **kwargs):
        calls.append((query, limit, kwargs))
        return {"results": hits}

    async def fake_content(path, cv=None):
        if contents and path in contents:
            return contents[path]
        raise RuntimeError("content unavailable")

    async def fake_complete_json(messages, gcfg, **kwargs):
        prompt = messages[-1]["content"]
        if "Определи тип реплики" in prompt:
            tail = prompt.split("Последняя реплика пользователя: ", 1)[1]
            return {
                "intent": "kb_question",
                "standalone_question": tail.split("\n", 1)[0],
            }
        return {"grades": [{"id": i, "score": 5} for i in range(1, 41)]}

    monkeypatch.setattr(rag.cognivault, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(rag.cognivault, "content", fake_content)
    monkeypatch.setattr(
        rag_pipeline.llm, "complete_json", fake_complete_json, raising=False
    )
    return calls


def _build(
    query: str, hits: list[dict], *, prompts: dict | None = None, **rcfg
) -> rag.RagContext:
    """Синхронная обёртка над ``build_rag_context`` в auto-режиме."""
    cfg = {"mode": "auto", "max_expanded_files": 0}
    cfg.update(rcfg)
    return asyncio.run(
        rag.build_rag_context(query, cfg, None, {}, None, prompts=prompts)
    )


# --------------------------------------------------------------------------- #
# Нормализация
# --------------------------------------------------------------------------- #


def test_norm_semantic_keeps_chunk_index_and_rank():
    out = rag._norm_semantic([_hit(1)])
    assert out[0]["chunk_index"] == 1
    assert out[0]["rank"] == 1


def test_norm_semantic_carries_section_text_and_parent_id():
    """Волна 3: поля group_by_section доезжают до фрагмента."""
    out = rag._norm_semantic([_hit(1, section_text="## Раздел\n\nтело раздела")])
    assert out[0]["section_text"] == "## Раздел\n\nтело раздела"
    assert out[0]["parent_id"] == "doc1.md#section"


def test_norm_semantic_defaults_section_fields_for_old_backend():
    """Ответ без новых полей (semantic-фолбэк) нормализуется в пустые строки."""
    out = rag._norm_semantic([_hit(1)])
    assert out[0]["section_text"] == ""
    assert out[0]["parent_id"] == ""


def test_sources_do_not_leak_internal_fields(monkeypatch):
    _install_retrieval(monkeypatch, [_hit(1)])
    ctx = _build("вопрос про архитектуру сервиса", [_hit(1)])
    assert set(ctx.sources[0]) == {
        "n", "title", "path", "section_path", "score", "depth", "grade"
    }


def test_auto_mode_retrieves_rerank_candidates_wide(monkeypatch):
    """Ширина ретрива в auto — `rerank_candidates` (волна 3: 40), не `limit`."""
    calls = _install_retrieval(monkeypatch, [_hit(1)])
    _build("вопрос про архитектуру сервиса", [_hit(1)], limit=3)
    assert calls[0][1] == 40

    calls.clear()
    _build("вопрос про архитектуру сервиса", [_hit(1)], rerank_candidates=12)
    assert calls[0][1] == 12


def test_auto_mode_asks_backend_to_group_by_section(monkeypatch):
    """Волна 3: hybrid зовётся с group_by_section и капом на текст раздела."""
    calls = _install_retrieval(monkeypatch, [_hit(1)])
    _build("вопрос про архитектуру сервиса", [_hit(1)], section_max_chars=1234)

    assert calls[0][2]["group_by_section"] is True
    assert calls[0][2]["section_max_chars"] == 1234


def test_archived_pages_are_never_searched(monkeypatch):
    """The active knowledge base excludes archived pages for every user."""
    calls = _install_retrieval(monkeypatch, [_hit(1)])
    _build("вопрос про архитектуру сервиса", [_hit(1)])
    assert calls[0][2]["include_archived"] is False

    calls.clear()
    # A stale per-user setting from an older release cannot turn it back on.
    _build("вопрос про архитектуру сервиса", [_hit(1)], include_archived=True)
    assert calls[0][2]["include_archived"] is False


# --------------------------------------------------------------------------- #
# Волна 3: текст раздела приходит с бэкенда, а не режется здесь
# --------------------------------------------------------------------------- #

# Длиннее `file_full_chars` в тестах ниже → whole-file ветка не срабатывает и
# документ раскрывается по разделам.
_LONG_DOC = "# Документ 1\n\n" + "полный текст документа " * 50


def _build_section(monkeypatch, hits: list[dict]) -> rag.RagContext:
    """Собрать контекст так, чтобы документ шёл по ветке section-expansion."""
    _install_retrieval(monkeypatch, hits, {"doc1.md": _LONG_DOC})
    return _build(
        "что написано в разделе документа",
        hits,
        max_expanded_files=1,
        file_full_chars=10,
    )


def test_section_text_from_backend_becomes_section_block(monkeypatch):
    body = "## Раздел\n\nполное тело раздела из индекса"
    ctx = _build_section(
        monkeypatch, [_hit(1, section="Раздел", section_text=body)]
    )

    assert [s["depth"] for s in ctx.sources] == ["section"]
    assert body in ctx.user_message["content"]
    # Именно раздел, а не сырой чанк и не весь документ.
    assert "содержимое фрагмента номер 1" not in ctx.user_message["content"]
    assert "полный текст документа" not in ctx.user_message["content"]


def test_empty_section_text_falls_back_to_chunk(monkeypatch):
    ctx = _build_section(monkeypatch, [_hit(1, section="Раздел", section_text="")])

    assert [s["depth"] for s in ctx.sources] == ["chunk"]
    assert "содержимое фрагмента номер 1" in ctx.user_message["content"]


def test_missing_section_text_falls_back_to_chunk(monkeypatch):
    """Старый бэкенд (поля нет вовсе) — тоже чанк, а не падение."""
    ctx = _build_section(monkeypatch, [_hit(1, section="Раздел")])

    assert [s["depth"] for s in ctx.sources] == ["chunk"]
    assert "содержимое фрагмента номер 1" in ctx.user_message["content"]


def test_semantic_fallback_without_new_fields_still_builds(monkeypatch):
    """Hybrid недоступен → semantic без section_text/parent_id не ломает сборку."""
    hits = [_hit(1, section="Раздел"), _hit(2)]
    _install_retrieval(monkeypatch, hits, {"doc1.md": _LONG_DOC})

    seen: list[tuple[str, int]] = []

    async def boom_hybrid(query, limit, cv=None, **kwargs):
        raise RuntimeError("hybrid не поддерживается")

    async def fake_semantic(query, limit, cv=None, **kwargs):
        seen.append((query, limit))
        return {"results": hits}

    monkeypatch.setattr(rag.cognivault, "hybrid_search", boom_hybrid)
    monkeypatch.setattr(rag.cognivault, "semantic_search", fake_semantic)

    ctx = _build(
        "что написано в разделе документа",
        hits,
        max_expanded_files=1,
        file_full_chars=10,
    )

    assert seen, "semantic-фолбэк не был вызван"
    assert [s["path"] for s in ctx.sources] == ["doc1.md", "doc2.md"]
    assert all(s["depth"] == "chunk" for s in ctx.sources)
    assert "содержимое фрагмента номер 1" in ctx.user_message["content"]


# --------------------------------------------------------------------------- #
# Склейка чанков одного файла: маркер пропуска
# --------------------------------------------------------------------------- #


def _chunk(index: int | None, text: str) -> dict:
    return {"text": text, "chunk_index": index}


def test_adjacent_chunks_are_joined_without_a_marker():
    merged = rag._merge_chunk_text(
        [_chunk(3, "первая часть"), _chunk(4, "вторая часть")]
    )
    assert merged == "первая часть\n\nвторая часть"
    assert rag._CHUNK_GAP_MARKER not in merged


def test_non_adjacent_chunks_get_a_gap_marker():
    merged = rag._merge_chunk_text(
        [_chunk(1, "начало документа"), _chunk(9, "конец документа")]
    )
    assert merged == f"начало документа\n\n{rag._CHUNK_GAP_MARKER}\n\nконец документа"


def test_gap_marker_only_between_non_adjacent_chunks():
    """Смешанный случай: маркер ровно один — на разрыве."""
    merged = rag._merge_chunk_text(
        [_chunk(1, "раз"), _chunk(2, "два"), _chunk(7, "семь")]
    )
    assert merged.count(rag._CHUNK_GAP_MARKER) == 1
    assert merged == f"раз\n\nдва\n\n{rag._CHUNK_GAP_MARKER}\n\nсемь"


def test_missing_chunk_index_counts_as_a_gap():
    """Без `chunk_index` смежность неизвестна — честнее показать разрыв."""
    merged = rag._merge_chunk_text([_chunk(None, "раз"), _chunk(None, "два")])
    assert rag._CHUNK_GAP_MARKER in merged


def test_single_chunk_has_no_marker():
    assert rag._merge_chunk_text([_chunk(1, "один")]) == "один"
    assert rag._merge_chunk_text([]) == ""


def test_gap_marker_reaches_the_context_block(monkeypatch):
    """Маркер доезжает до промпта, а не теряется при рендере блока."""
    hits = [
        {**_hit(1, text="первый кусок файла"), "chunk_index": 1},
        {**_hit(1, text="далёкий кусок того же файла"), "chunk_index": 8},
    ]
    _install_retrieval(monkeypatch, hits)
    ctx = _build("что написано в документе про кластер", hits)

    content = ctx.user_message["content"]
    assert content.count("### Источник ") == 1, "чанки одного файла — один источник"
    assert rag._CHUNK_GAP_MARKER in content


# --------------------------------------------------------------------------- #
# system: только правила
# --------------------------------------------------------------------------- #


def test_system_message_holds_rules_only(monkeypatch):
    hits = [_hit(1, text="секретный текст фрагмента про кластер")]
    _install_retrieval(monkeypatch, hits)
    ctx = _build("что известно про кластер и его настройку", hits)

    assert isinstance(ctx, rag.RagContext)
    system = ctx.system_message["content"]
    # Без переопределения в конфиге system — ровно встроенный промпт…
    assert system == rag.SYSTEM_PROMPT
    # …и правила в нём на месте (проверяем сам дефолт, а не «любой» промпт).
    assert "[Источник N]" in rag.SYSTEM_PROMPT
    assert "Не выдумывай источники" in rag.SYSTEM_PROMPT
    assert "markdown-таблицы" in rag.SYSTEM_PROMPT
    # А текста источников — нет.
    assert "секретный текст фрагмента" not in system
    assert "Источники:" not in system
    assert "### Источник" not in system


def test_context_lives_in_user_message(monkeypatch):
    hits = [_hit(1, text="секретный текст фрагмента про кластер")]
    _install_retrieval(monkeypatch, hits)
    ctx = _build("что известно про кластер и его настройку", hits)

    assert ctx.user_message["role"] == "user"
    assert "секретный текст фрагмента про кластер" in ctx.user_message["content"]
    assert ctx.context_chars > 0


# --------------------------------------------------------------------------- #
# user: порядок «Источники → Напоминание → Вопрос»
# --------------------------------------------------------------------------- #


def test_user_message_section_order(monkeypatch):
    hits = [_hit(1), _hit(2)]
    _install_retrieval(monkeypatch, hits)
    query = "какие сервисы описаны в документации проекта"
    ctx = _build(query, hits)

    content = ctx.user_message["content"]
    i_sources = content.index("Источники:")
    i_reminder = content.index("Напоминание:")
    i_question = content.index("Вопрос:")
    assert i_sources < i_reminder < i_question
    assert content.startswith("Источники:")
    assert content.endswith(f"Вопрос: {query}")


def test_block_numbering_matches_sources(monkeypatch):
    hits = [_hit(1), _hit(2), _hit(3)]
    _install_retrieval(monkeypatch, hits)
    ctx = _build("перечисли все известные документы по теме", hits)

    content = ctx.user_message["content"]
    assert [s["n"] for s in ctx.sources] == [1, 2, 3]
    for src in ctx.sources:
        header = f"### Источник {src['n']}: {src['title']} — {src['path']}"
        assert header in content
    # Нумерация сплошная и не начинается с нуля.
    assert "### Источник 0:" not in content
    assert "### Источник 4:" not in content


def test_header_with_section_path(monkeypatch):
    hits = [_hit(1, section="Раздел > Подраздел")]
    _install_retrieval(monkeypatch, hits)
    ctx = _build("расскажи про подраздел документа подробнее", hits)

    assert (
        "### Источник 1: Документ 1 — doc1.md > Раздел > Подраздел"
        in ctx.user_message["content"]
    )


def test_empty_section_path_has_no_dangling_separator(monkeypatch):
    hits = [_hit(1, section="")]
    _install_retrieval(monkeypatch, hits)
    ctx = _build("что написано в первом документе базы знаний", hits)

    header = [
        line
        for line in ctx.user_message["content"].splitlines()
        if line.startswith("### Источник")
    ][0]
    assert header == "### Источник 1: Документ 1 — doc1.md"
    assert ">" not in header.rstrip()


# --------------------------------------------------------------------------- #
# Настраиваемые промпты (секция конфига ``prompts``)
# --------------------------------------------------------------------------- #

_CUSTOM_SYSTEM = "Ты лаконичный ассистент. Отвечай ровно одним предложением."
_CUSTOM_REMINDER = "Помни: одно предложение, без воды."


def test_custom_system_prompt_reaches_system_message(monkeypatch):
    hits = [_hit(1)]
    _install_retrieval(monkeypatch, hits)
    ctx = _build(
        "что известно про кластер и его настройку",
        hits,
        prompts={"system": _CUSTOM_SYSTEM},
    )

    assert ctx.system_message["content"] == _CUSTOM_SYSTEM
    # Дефолт полностью вытеснен, а не дописан рядом.
    assert rag.SYSTEM_PROMPT not in ctx.system_message["content"]
    # Кастомный промпт без «[Источник N]» — это нормально и не ломает сборку.
    assert "[Источник N]" not in ctx.system_message["content"]
    assert ctx.user_message["content"].startswith("Источники:")


@pytest.mark.parametrize(
    "prompts",
    [
        None,
        {},
        {"system": None},
        {"system": ""},
        {"system": "   \n\t "},
        {"context_reminder": _CUSTOM_REMINDER},  # чужой ключ не трогает system
    ],
    ids=["none", "empty", "explicit-none", "empty-str", "blank-str", "other-key"],
)
def test_system_prompt_falls_back_to_default(monkeypatch, prompts):
    """``None``/отсутствие/пустая строка → встроенный дефолт."""
    hits = [_hit(1)]
    _install_retrieval(monkeypatch, hits)
    ctx = _build("что известно про кластер и его настройку", hits, prompts=prompts)

    assert ctx.system_message["content"] == rag.SYSTEM_PROMPT


def test_custom_reminder_sits_between_sources_and_question(monkeypatch):
    hits = [_hit(1), _hit(2)]
    _install_retrieval(monkeypatch, hits)
    query = "какие сервисы описаны в документации проекта"
    ctx = _build(query, hits, prompts={"context_reminder": _CUSTOM_REMINDER})

    content = ctx.user_message["content"]
    # Порядок секций не изменился: Источники → Напоминание → Вопрос.
    i_sources = content.index("Источники:")
    i_last_block = content.rindex("### Источник 2:")
    i_reminder = content.index(_CUSTOM_REMINDER)
    i_question = content.index("Вопрос:")
    assert i_sources < i_last_block < i_reminder < i_question
    assert content.startswith("Источники:")
    assert content.endswith(f"Вопрос: {query}")
    # Дефолтное напоминание вытеснено.
    assert rag.CONTEXT_REMINDER not in content
    # Системный промпт при этом остался дефолтным.
    assert ctx.system_message["content"] == rag.SYSTEM_PROMPT


@pytest.mark.parametrize(
    "prompts",
    [None, {}, {"context_reminder": None}, {"context_reminder": "  "}],
    ids=["none", "empty", "explicit-none", "blank-str"],
)
def test_reminder_falls_back_to_default(monkeypatch, prompts):
    hits = [_hit(1)]
    _install_retrieval(monkeypatch, hits)
    ctx = _build("какие сервисы описаны в документации", hits, prompts=prompts)

    assert rag.CONTEXT_REMINDER in ctx.user_message["content"]


def test_both_prompts_customised_in_legacy_mode(monkeypatch):
    """Переопределения доезжают и по старому (не ``auto``) пути."""
    hits = [_hit(1)]
    _install_retrieval(monkeypatch, hits)
    ctx = _build(
        "какие сервисы описаны в документации",
        hits,
        mode="legacy",
        source="hybrid",
        prompts={"system": _CUSTOM_SYSTEM, "context_reminder": _CUSTOM_REMINDER},
    )

    assert ctx.system_message["content"] == _CUSTOM_SYSTEM
    assert _CUSTOM_REMINDER in ctx.user_message["content"]
    assert rag.CONTEXT_REMINDER not in ctx.user_message["content"]


def test_private_prompt_aliases_still_point_at_defaults():
    """Старые приватные имена — алиасы публичных (на них завязан config API)."""
    assert rag._SYSTEM_PROMPT is rag.SYSTEM_PROMPT
    assert rag._CONTEXT_REMINDER is rag.CONTEXT_REMINDER


# --------------------------------------------------------------------------- #
# Кап на число блоков (пункт 1.4)
# --------------------------------------------------------------------------- #


def test_context_capped_at_max_blocks(monkeypatch):
    hits = [_hit(i) for i in range(1, 11)]
    _install_retrieval(monkeypatch, hits)
    ctx = _build("подробный вопрос по всей базе знаний сразу", hits)

    assert rag._MAX_CONTEXT_BLOCKS == 5
    assert len(ctx.sources) == 5
    assert ctx.user_message["content"].count("### Источник ") == 5
    assert "### Источник 6:" not in ctx.user_message["content"]
    # Отброшены именно худшие по score.
    assert [s["path"] for s in ctx.sources] == [f"doc{i}.md" for i in range(1, 6)]


def test_cap_applies_to_expanded_files_too(monkeypatch):
    """Экспансия документов не может обойти кап."""
    hits = [_hit(i) for i in range(1, 11)]
    contents = {f"doc{i}.md": f"# Документ {i}\n\nполный текст {i}" for i in range(1, 11)}
    _install_retrieval(monkeypatch, hits, contents)
    ctx = _build(
        "подробный вопрос по всей базе знаний сразу", hits, max_expanded_files=8
    )

    assert len(ctx.sources) <= rag._MAX_CONTEXT_BLOCKS


# --------------------------------------------------------------------------- #
# Пустая/сломанная выдача
# --------------------------------------------------------------------------- #


def test_no_results_yields_empty_context(monkeypatch):
    _install_retrieval(monkeypatch, [])
    ctx = _build("вопрос, на который ничего не находится", [])
    assert ctx.answer_override == rag._NO_ANSWER
    assert (ctx.system_message, ctx.user_message, ctx.sources, ctx.notice) == (
        None, None, [], None
    )


def test_retrieval_failure_yields_notice(monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("нет связи")

    monkeypatch.setattr(rag.cognivault, "hybrid_search", boom)
    monkeypatch.setattr(rag.cognivault, "semantic_search", boom)
    ctx = _build("любой вопрос при недоступном поиске", [])
    assert ctx.notice == rag._RETRIEVAL_UNAVAILABLE
    assert ctx.system_message is None and ctx.user_message is None


# --------------------------------------------------------------------------- #
# Валидация цитат
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,n,expected",
    [
        ("всё хорошо [Источник 1] и [Источник 2]", 3, []),
        ("плохо [Источник 9]", 3, [9]),
        ("[Источники 1, 7] и [Источника 2; 8]", 3, [7, 8]),
        ("[Источник 0]", 3, [0]),
        ("без цитат вообще", 3, []),
        ("", 3, []),
        ("[Источник 1]", 0, [1]),
    ],
)
def test_invalid_citations(text, n, expected):
    assert chat_routes._invalid_citations(text, n) == expected


# --------------------------------------------------------------------------- #
# Интеграция: что реально уходит в GigaChat
# --------------------------------------------------------------------------- #


def _install_chat(monkeypatch, tmp_path, hits, *, answer="ответ", contents=None):
    """Замокать транспорт + GigaChat; вернуть перехваченные ``messages``."""
    captured: list[list[dict]] = []

    async def fake_stream_chat(messages, gcfg):
        captured.append([dict(m) for m in messages])
        yield answer

    _install_retrieval(monkeypatch, hits, contents)
    monkeypatch.setattr(chat_routes, "resolve_paths", lambda request: _paths(tmp_path))
    monkeypatch.setattr(chat_routes.llm, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(chat_routes.llm, "files_present", lambda gcfg: None)
    return captured


def test_chat_sends_system_history_and_context_message(monkeypatch, tmp_path):
    hits = [_hit(1), _hit(2)]
    captured = _install_chat(monkeypatch, tmp_path, hits)
    messages = [
        {"role": "user", "content": "первый вопрос пользователя"},
        {"role": "assistant", "content": "первый ответ ассистента"},
        {"role": "user", "content": "второй вопрос пользователя"},
    ]

    with TestClient(create_app()) as client:
        resp = client.post("/api/chat", json={"messages": messages, "rag": True})

    assert resp.status_code == 200
    assert captured, "stream_chat не был вызван"
    sent = captured[0]

    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    # system — только правила (по умолчанию — встроенный промпт).
    assert sent[0]["content"] == rag.SYSTEM_PROMPT
    assert "### Источник" not in sent[0]["content"]
    # История сохранена как есть.
    assert sent[1]["content"] == "первый вопрос пользователя"
    assert sent[2]["content"] == "первый ответ ассистента"
    # Последнее сообщение — контекст + тот же вопрос, без дубля в истории.
    last = sent[-1]["content"]
    assert last.startswith("Источники:")
    assert last.endswith("Вопрос: второй вопрос пользователя")
    assert [m for m in sent[:-1] if m["content"] == "второй вопрос пользователя"] == []


def _install_config(monkeypatch, prompts: dict) -> None:
    """Подсунуть маршруту конфиг с секцией ``prompts``.

    Патчим ОБА пути чтения конфига: пер-пользовательский
    ``settings.effective_config_for(paths)`` (его предпочитает маршрут) и
    глобальный ``settings.effective_config()`` — тест не должен зависеть от
    того, какой из них доступен в сборке.
    """
    cfg = {**settings.effective_config(), "prompts": prompts}
    monkeypatch.setattr(settings, "effective_config", lambda: cfg)
    monkeypatch.setattr(
        settings, "effective_config_for", lambda paths: cfg, raising=False
    )


def test_chat_uses_configured_prompts(monkeypatch, tmp_path):
    """Кастомные промпты из конфига доезжают до ``gigachat.stream_chat``."""
    hits = [_hit(1), _hit(2)]
    captured = _install_chat(monkeypatch, tmp_path, hits)
    _install_config(
        monkeypatch,
        {"system": _CUSTOM_SYSTEM, "context_reminder": _CUSTOM_REMINDER},
    )

    with TestClient(create_app()) as client:
        resp = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "вопрос про кластер"}],
                "rag": True,
            },
        )

    assert resp.status_code == 200
    sent = captured[0]
    assert sent[0]["role"] == "system"
    assert sent[0]["content"] == _CUSTOM_SYSTEM
    assert rag.SYSTEM_PROMPT not in sent[0]["content"]
    last = sent[-1]["content"]
    assert last.startswith("Источники:")
    assert last.index(_CUSTOM_REMINDER) < last.index("Вопрос:")
    assert rag.CONTEXT_REMINDER not in last


def test_chat_falls_back_to_default_prompts(monkeypatch, tmp_path):
    """Пустые/``None``-поля в конфиге → встроенные дефолты, а не пустой system."""
    hits = [_hit(1)]
    captured = _install_chat(monkeypatch, tmp_path, hits)
    _install_config(monkeypatch, {"system": "  ", "context_reminder": None})

    with TestClient(create_app()) as client:
        resp = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "вопрос про кластер"}],
                "rag": True,
            },
        )

    assert resp.status_code == 200
    sent = captured[0]
    assert sent[0]["content"] == rag.SYSTEM_PROMPT
    assert rag.CONTEXT_REMINDER in sent[-1]["content"]


def test_chat_without_rag_keeps_plain_history(monkeypatch, tmp_path):
    captured = _install_chat(monkeypatch, tmp_path, [_hit(1)])
    messages = [{"role": "user", "content": "просто вопрос"}]

    with TestClient(create_app()) as client:
        resp = client.post("/api/chat", json={"messages": messages})

    assert resp.status_code == 200
    assert captured[0] == [{"role": "user", "content": "просто вопрос"}]


def test_chat_trims_history_but_never_the_context_message(monkeypatch, tmp_path):
    """Бюджет учитывает system и финальное сообщение; режется только середина."""
    hits = [_hit(1, text="ф" * 6000)]
    captured = _install_chat(monkeypatch, tmp_path, hits)

    messages: list[dict] = []
    for i in range(40):
        messages.append({"role": "user", "content": f"в{i} " + "я" * 1500})
        messages.append({"role": "assistant", "content": f"о{i} " + "б" * 1500})
    messages.append({"role": "user", "content": "финальный вопрос"})

    with TestClient(create_app()) as client:
        resp = client.post("/api/chat", json={"messages": messages, "rag": True})

    assert resp.status_code == 200
    sent = captured[0]
    assert len(sent) < len(messages) + 1, "история должна быть урезана"
    assert sent[0]["role"] == "system"
    # Контекст цел: он в защищённом хвосте trim_history.
    assert sent[-1]["content"].startswith("Источники:")
    assert sent[-1]["content"].endswith("Вопрос: финальный вопрос")
    assert "ф" * 6000 in sent[-1]["content"]
    # Между system и контекстом — только целые пары user/assistant.
    roles = [m["role"] for m in sent[1:-1]]
    assert roles == ["user", "assistant"] * (len(roles) // 2)
    # Бюджет соблюдён: system + история + контекст влезают в окно.
    from app.tokens import estimate_messages_tokens

    assert estimate_messages_tokens(sent) <= 32768 - 4096


def test_chat_reports_invalid_citation(monkeypatch, tmp_path, caplog):
    hits = [_hit(1), _hit(2), _hit(3)]
    _install_chat(
        monkeypatch, tmp_path, hits, answer="утверждение [Источник 9] и [Источник 2]"
    )
    saved: list[list[dict]] = []
    monkeypatch.setattr(
        chat_routes.history, "save_chat",
        lambda cid, msgs, paths: saved.append([dict(m) for m in msgs]),
    )

    with caplog.at_level(logging.WARNING, logger="cognivault-ui.chat"):
        with TestClient(create_app()) as client:
            resp = client.post(
                "/api/chat",
                json={
                    "messages": [{"role": "user", "content": "вопрос про источники"}],
                    "rag": True,
                },
            )

    assert resp.status_code == 200
    events = dict(_parse_sse(resp.text))
    assert len(events["sources"]["sources"]) == 3
    # Попало в структуру ассистентского сообщения…
    assert saved and saved[0][-1]["invalid_citations"] == [9]
    # …и в лог.
    assert any("9" in rec.getMessage() for rec in caplog.records)


def test_chat_valid_citations_are_silent(monkeypatch, tmp_path):
    hits = [_hit(1), _hit(2), _hit(3)]
    _install_chat(monkeypatch, tmp_path, hits, answer="факт [Источники 1, 3]")
    saved: list[list[dict]] = []
    monkeypatch.setattr(
        chat_routes.history, "save_chat",
        lambda cid, msgs, paths: saved.append([dict(m) for m in msgs]),
    )

    with TestClient(create_app()) as client:
        client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "вопрос про источники"}],
                "rag": True,
            },
        )

    assert saved and saved[0][-1]["invalid_citations"] == []


# --------------------------------------------------------------------------- #
# n_hits для расширения до целого файла считается ДО отбора
# --------------------------------------------------------------------------- #


def test_whole_file_expansion_counts_retrieval_hits_not_survivors(monkeypatch):
    """Длинная страница разворачивается целиком, если поиск дал по ней 3+ чанка.

    Регрессия: `n_hits` брался из фрагментов, ПЕРЕЖИВШИХ отбор, а грейдер
    оставляет максимум пять на весь контекст. Порог `n_hits >= 3` практически
    никогда не срабатывал, и страница длиннее `file_full_chars` не попадала в
    контекст целиком ни при каких условиях — из-за чего терялись хвосты длинных
    разделов (состав zip-архива, полный список моделей).
    """
    long_body = "полный текст страницы. " * 400  # ~9 000 символов
    hits = [
        {**_hit(1), "path": "big.md", "title": "Большая", "chunk_index": i, "rank": i}
        for i in range(1, 4)
    ]
    # Грейдер оставляет ровно один фрагмент — как в реальном ходе.
    async def one_five(messages, gcfg, **kwargs):
        prompt = messages[-1]["content"]
        if "Определи тип реплики" in prompt:
            return {"intent": "kb_question", "standalone_question": "вопрос"}
        return {"grades": [{"id": 1, "score": 5}]}

    _install_retrieval(monkeypatch, hits, contents={"big.md": long_body})
    monkeypatch.setattr(
        rag_pipeline.llm, "complete_json", one_five, raising=False
    )

    ctx = _build(
        "вопрос",
        hits,
        max_expanded_files=1,
        file_full_chars=6000,
        max_context_chars=48000,
    )

    assert long_body.strip()[:80] in ctx.user_message["content"]
    assert [s["depth"] for s in ctx.sources] == ["file"]


# --------------------------------------------------------------------------- #
# Внутренняя непротиворечивость промпта
# --------------------------------------------------------------------------- #


def test_prompt_does_not_both_demand_and_forbid_the_hedging_opening():
    """Регрессия: две инструкции разрешали ответ «прямого ответа нет, однако…».

    Старый промпт велел писать «в источниках ответа не нашлось», когда ответа
    нет, и отдельно запрещал начинать с такой оговорки, когда ответ есть. Модель
    нашла третий путь — сделала и то, и другое сразу: открыла отказом и следом
    перечислила пять потоков. Теперь правило говорит, что фраза — это ВЕСЬ ответ.
    """
    p = rag.SYSTEM_PROMPT
    assert "Это ответ целиком, а не вступление" in p
    assert "запрещены" in p and "прямого\n  ответа нет, однако" in p


def test_reminder_does_not_undercut_the_answer_first_rule():
    """Напоминание стоит перед вопросом — там, куда модель смотрит лучше всего.

    Пока оно заканчивалось словами «если ответа нет — скажи об этом», последним,
    что читала модель, было приглашение начать с отказа.
    """
    assert "Начинай сразу с ответа" in rag.CONTEXT_REMINDER
    assert "если ответа в источниках нет" not in rag.CONTEXT_REMINDER


def test_incompleteness_caveat_is_conditional():
    """«Список может быть неполным» на полном ответе обесценивает оговорку."""
    p = rag.SYSTEM_PROMPT
    assert "ТОЛЬКО при видимом признаке" in p
    assert "не приписывай оговорку" in p
