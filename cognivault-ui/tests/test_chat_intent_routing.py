"""Маршрутизация интентов в ``POST /api/chat`` (волна 2, пункты 2.1/2.3).

Проверяется поведение роута, а не сам конвейер: `rag.build_rag_context`
замокан и возвращает готовый `RagContext` для каждого из трёх исходов.

* ``smalltalk``/``clarify`` — сообщений нет и ``notice`` нет: обычная генерация
  по истории как есть, кадр ``sources`` не эмитится;
* ``answer_override`` — ответ уже готов: GigaChat не вызывается, ровно один
  кадр ``token``, ``sources`` пустой, ``done`` с ``finish_reason="no_context"``;
* ``kb_question`` — обычный путь, ``sources`` приходит ДО первого ``token``
  (требование 2.3: маскировка латентности двух новых вызовов).

Плюс запись ``"request"`` в ``rag_log.jsonl``: ``intent``/``question_standalone``
всегда, ``candidates``/``grades`` — для ``kb_question``.

pytest + Starlette ``TestClient``, LOCAL-режим, ``resolve_paths`` в tmp-каталог.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import rag, rag_log, settings  # noqa: E402
from app.config import AppPaths  # noqa: E402
from app.history import load_chat  # noqa: E402
from app.main import create_app  # noqa: E402
from app.routes import chat_routes  # noqa: E402

REFUSAL = "В базе знаний нет данных по этому вопросу."


def _paths(tmp_path) -> AppPaths:
    return AppPaths(root=tmp_path / ".cognivault-ui")


@pytest.fixture(autouse=True)
def _local_mode(monkeypatch):
    monkeypatch.setattr(settings, "is_server", lambda: False)


def _parse_sse(text: str) -> list[tuple[str, dict]]:
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


def _events(text: str) -> list[str]:
    return [name for name, _ in _parse_sse(text)]


def _ctx(**fields) -> rag.RagContext:
    """`RagContext` с полями волны 2, даже если датакласс их ещё не объявил.

    Роут читает новые поля через ``getattr``, поэтому тест не должен зависеть
    от того, приземлился ли уже расширенный контракт в ``app/rag.py``.
    """
    known = {"system_message", "user_message", "sources", "notice", "context_chars"}
    ctx = rag.RagContext(**{k: v for k, v in fields.items() if k in known})
    for key, value in fields.items():
        if key not in known:
            setattr(ctx, key, value)
    return ctx


def _install(monkeypatch, paths, ctx, *, answer="ответ модели"):
    """Замокать RAG-слой и GigaChat; вернуть список перехваченных вызовов."""
    calls: list[dict] = []

    async def fake_build_rag_context(query, *args, **kwargs):
        calls.append({"kind": "rag", "query": query})
        return ctx

    async def fake_stream_chat(messages, gcfg):
        calls.append({"kind": "stream", "messages": [dict(m) for m in messages]})
        yield answer

    monkeypatch.setattr(chat_routes, "resolve_paths", lambda request: paths)
    monkeypatch.setattr(chat_routes.rag, "build_rag_context", fake_build_rag_context)
    monkeypatch.setattr(chat_routes.llm, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(chat_routes.llm, "files_present", lambda gcfg: None)
    return calls


def _kb_ctx(question: str = "как настроить ЕФС в 1С?") -> rag.RagContext:
    return _ctx(
        system_message={"role": "system", "content": "правила"},
        user_message={"role": "user", "content": f"Источники:\n\nВопрос: {question}"},
        sources=[
            {
                "n": 1,
                "title": "Док",
                "path": "notes/a.md",
                "section_path": "Раздел",
                "depth": "section",
                "score": 0.91,
                "grade": 5,
            }
        ],
        context_chars=128,
        intent="kb_question",
        standalone_question=question,
        candidates=[
            {"path": "notes/a.md", "chunk_index": 0, "score": 0.91, "rank": 1},
            {"path": "notes/b.md", "chunk_index": 2, "score": 0.42, "rank": 2},
        ],
        grades=[
            {"id": 1, "path": "notes/a.md", "chunk_index": 0, "score": 5},
            {"id": 2, "path": "notes/b.md", "chunk_index": 2, "score": 2},
        ],
    )


def _post(client, messages, **extra):
    return client.post("/api/chat", json={"messages": messages, "rag": True, **extra})


# --------------------------------------------------------------------------- #
# smalltalk / clarify
# --------------------------------------------------------------------------- #


SMALLTALK_HISTORY = [
    {"role": "user", "content": "как настроить ЕФС?"},
    {"role": "assistant", "content": "вот так"},
    {"role": "user", "content": "спасибо!"},
]


@pytest.mark.parametrize("intent", ["smalltalk", "clarify"])
def test_smalltalk_generates_without_sources(tmp_path, monkeypatch, intent):
    """Нет `user_message` и нет notice → генерация по истории, кадра `sources` нет."""
    paths = _paths(tmp_path)
    ctx = _ctx(
        system_message={"role": "system", "content": rag.NO_RAG_SYSTEM_PROMPT},
        intent=intent,
        standalone_question=None,
    )
    calls = _install(monkeypatch, paths, ctx, answer="и тебе привет")

    messages = list(SMALLTALK_HISTORY)
    with TestClient(create_app()) as client:
        resp = _post(client, messages)

    assert resp.status_code == 200
    events = _events(resp.text)
    assert "sources" not in events
    assert "notice" not in events
    assert events == ["meta", "token", "done"]

    stream = [c for c in calls if c["kind"] == "stream"]
    assert stream, "stream_chat не был вызван"
    # Правила «без источников» впереди, история целиком за ними: последний
    # вопрос заменять нечем, поэтому он остаётся на месте.
    sent = stream[0]["messages"]
    assert sent[0] == {"role": "system", "content": rag.NO_RAG_SYSTEM_PROMPT}
    assert sent[1:] == messages
    # Блока «Источники» в отправленном нет вообще.
    assert not any("Источники:" in m["content"] for m in sent)

    rec = rag_log.read_records(paths)[0]
    assert rec["intent"] == intent
    assert rec["rag_used"] is False
    assert rec["sources"] == []
    assert rec["notice"] is None


def test_smalltalk_without_system_message_falls_back_to_bare_history(
    tmp_path, monkeypatch
):
    """Старый `RagContext` (оба поля `None`) — поведение прежнее, ничего не ломается."""
    paths = _paths(tmp_path)
    calls = _install(monkeypatch, paths, _ctx(intent="smalltalk"), answer="привет")

    with TestClient(create_app()) as client:
        _post(client, list(SMALLTALK_HISTORY))

    stream = [c for c in calls if c["kind"] == "stream"]
    assert stream[0]["messages"] == SMALLTALK_HISTORY


def test_kb_question_keeps_message_order(tmp_path, monkeypatch):
    """`user_message` есть → system + история БЕЗ последнего вопроса + источники."""
    paths = _paths(tmp_path)
    calls = _install(monkeypatch, paths, _kb_ctx(), answer="ответ")

    with TestClient(create_app()) as client:
        _post(
            client,
            [
                {"role": "user", "content": "расскажи про ЕФС"},
                {"role": "assistant", "content": "ЕФС — это…"},
                {"role": "user", "content": "а как её настроить?"},
            ],
        )

    sent = [c for c in calls if c["kind"] == "stream"][0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[0]["content"] == "правила"
    # Последний вопрос заменён user-сообщением с источниками, а не продублирован.
    assert sent[-1]["content"].startswith("Источники:")
    assert all(m["content"] != "а как её настроить?" for m in sent)


def test_smalltalk_log_keeps_raw_question(tmp_path, monkeypatch):
    """Для smalltalk `question_standalone` пуст, но `question_raw` — нет."""
    paths = _paths(tmp_path)
    _install(monkeypatch, paths, _ctx(intent="smalltalk"))

    with TestClient(create_app()) as client:
        _post(client, [{"role": "user", "content": "привет"}])

    rec = rag_log.read_records(paths)[0]
    assert rec["question_raw"] == "привет"
    assert rec["question_standalone"] is None
    assert rec["candidates"] is None and rec["grades"] is None


# --------------------------------------------------------------------------- #
# answer_override — готовый ответ без генерации
# --------------------------------------------------------------------------- #


def test_answer_override_skips_gigachat(tmp_path, monkeypatch):
    """Ответ отдаётся одним токеном, GigaChat не вызывается вообще."""
    paths = _paths(tmp_path)
    ctx = _ctx(
        intent="kb_question",
        standalone_question="как настроить ЕФС в 1С?",
        answer_override=REFUSAL,
        candidates=[{"path": "notes/a.md", "chunk_index": 0, "score": 0.3, "rank": 1}],
        grades=[{"id": 1, "path": "notes/a.md", "chunk_index": 0, "score": 2}],
    )
    calls = _install(monkeypatch, paths, ctx)

    with TestClient(create_app()) as client:
        resp = _post(client, [{"role": "user", "content": "как настроить ЕФС?"}])

    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    assert [name for name, _ in frames] == ["meta", "sources", "token", "done"]

    by_event = dict(frames)
    assert by_event["sources"] == {"sources": [], "context_chars": 0}
    assert by_event["token"] == {"text": REFUSAL}
    assert by_event["done"]["finish_reason"] == "no_context"

    assert [c["kind"] for c in calls] == ["rag"], "stream_chat не должен вызываться"


def test_answer_override_persists_like_a_normal_answer(tmp_path, monkeypatch):
    """В истории и в логе отказ выглядит обычным RAG-ответом без источников."""
    paths = _paths(tmp_path)
    _install(
        monkeypatch,
        paths,
        _ctx(
            intent="kb_question",
            standalone_question="как настроить ЕФС в 1С?",
            answer_override=REFUSAL,
            sources=[{"n": 1, "path": "notes/a.md", "score": 0.3, "grade": 2}],
            context_chars=99,
            grades=[{"id": 1, "path": "notes/a.md", "chunk_index": 0, "score": 2}],
        ),
    )

    with TestClient(create_app()) as client:
        resp = _post(client, [{"role": "user", "content": "как настроить ЕФС?"}])
        chat_id = dict(_parse_sse(resp.text))["meta"]["chat_id"]

    chat = load_chat(chat_id, paths)
    assistant = chat["messages"][-1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == REFUSAL
    # Поиск выполнялся, поэтому ход считается RAG-ходом; но в контекст ничего
    # не попало — источников нет и context_chars нулевой.
    assert assistant["rag"] is True
    assert assistant["sources"] == []
    assert assistant["context_chars"] == 0
    assert assistant["truncated"] is False
    assert assistant["invalid_citations"] == []

    rec = rag_log.read_records(paths)[0]
    assert rec["rag_used"] is True
    assert rec["sources"] == []
    assert rec["answer_chars"] == len(REFUSAL)
    assert rec["intent"] == "kb_question"
    assert rec["question_standalone"] == "как настроить ЕФС в 1С?"
    assert rec["grades"]


# --------------------------------------------------------------------------- #
# kb_question — обычный путь и порядок кадров (2.3)
# --------------------------------------------------------------------------- #


def test_kb_question_emits_sources_before_first_token(tmp_path, monkeypatch):
    """`sources` приходит ДО первого `token` — латентность 2 вызовов скрыта."""
    paths = _paths(tmp_path)
    calls = _install(monkeypatch, paths, _kb_ctx(), answer="ответ [Источник 1]")

    with TestClient(create_app()) as client:
        resp = _post(client, [{"role": "user", "content": "как настроить ЕФС?"}])

    assert resp.status_code == 200
    events = _events(resp.text)
    assert events == ["meta", "sources", "token", "done"]
    assert events.index("sources") < events.index("token")
    assert "notice" not in events

    payload = dict(_parse_sse(resp.text))["sources"]
    assert payload["context_chars"] == 128
    assert payload["sources"][0]["grade"] == 5

    stream = [c for c in calls if c["kind"] == "stream"]
    assert stream, "stream_chat не был вызван"
    assert [m["role"] for m in stream[0]["messages"]] == ["system", "user"]


def test_kb_question_logs_pipeline_telemetry(tmp_path, monkeypatch):
    """Запись лога несёт интент, самодостаточный вопрос, кандидатов и грейды."""
    paths = _paths(tmp_path)
    _install(monkeypatch, paths, _kb_ctx(), answer="ответ [Источник 1]")

    with TestClient(create_app()) as client:
        _post(
            client,
            [
                {"role": "user", "content": "расскажи про ЕФС"},
                {"role": "assistant", "content": "ЕФС — это…"},
                {"role": "user", "content": "а как её настроить?"},
            ],
        )

    rec = rag_log.read_records(paths)[0]
    assert rec["type"] == "request"
    assert rec["intent"] == "kb_question"
    assert rec["question_raw"] == "а как её настроить?"
    assert rec["question_standalone"] == "как настроить ЕФС в 1С?"
    assert [c["path"] for c in rec["candidates"]] == ["notes/a.md", "notes/b.md"]
    assert [g["score"] for g in rec["grades"]] == [5, 2]
    assert rec["rag_used"] is True
    assert rec["sources"][0]["path"] == "notes/a.md"


# --------------------------------------------------------------------------- #
# Связывание грейдов с кандидатами и восстановление чанка источника
# --------------------------------------------------------------------------- #


def test_grade_ids_resolve_to_candidates(tmp_path, monkeypatch):
    """`grades[].id` перестаёт быть числом в никуда: тот же `id` есть у кандидата."""
    paths = _paths(tmp_path)
    _install(monkeypatch, paths, _kb_ctx(), answer="ответ [Источник 1]")

    with TestClient(create_app()) as client:
        _post(client, [{"role": "user", "content": "как настроить ЕФС?"}])

    rec = rag_log.read_records(paths)[0]
    by_id = {c["id"]: c for c in rec["candidates"]}
    assert sorted(by_id) == [1, 2]
    for grade in rec["grades"]:
        candidate = by_id[grade["id"]]
        assert (candidate["path"], candidate["chunk_index"]) == (
            grade["path"],
            grade["chunk_index"],
        )


def test_candidates_get_ids_without_grader(tmp_path, monkeypatch):
    """Грейдер выключен (`grades is None`) — нумерация кандидатов всё равно есть."""
    paths = _paths(tmp_path)
    ctx = _ctx(
        system_message={"role": "system", "content": "правила"},
        user_message={"role": "user", "content": "Источники:\n\nВопрос: q"},
        sources=[{"n": 1, "path": "notes/a.md", "depth": "chunk", "score": 0.9}],
        context_chars=0,
        intent="kb_question",
        candidates=[
            {"path": "notes/a.md", "chunk_index": 4, "score": 0.9, "rank": 1},
            {"path": "notes/a.md", "chunk_index": 9, "score": 0.5, "rank": 2},
        ],
        grades=None,
    )
    _install(monkeypatch, paths, ctx)

    with TestClient(create_app()) as client:
        _post(client, [{"role": "user", "content": "вопрос"}])

    rec = rag_log.read_records(paths)[0]
    assert [c["id"] for c in rec["candidates"]] == [1, 2]
    assert rec["grades"] is None
    # Скор блока совпадает с одним кандидатом — чанк восстановлен точно.
    assert rec["sources"][0]["chunk_index"] == 4
    assert rec["sources"][0]["chunk_indexes"] == [4]


def test_whole_file_source_covers_every_retrieved_chunk(tmp_path, monkeypatch):
    """`depth="file"` — в контексте весь документ, значит попали все его чанки."""
    paths = _paths(tmp_path)
    ctx = _ctx(
        system_message={"role": "system", "content": "правила"},
        user_message={"role": "user", "content": "Источники:\n\nВопрос: q"},
        sources=[{"n": 1, "path": "notes/a.md", "depth": "file", "score": 0.9}],
        context_chars=0,
        intent="kb_question",
        candidates=[
            {"path": "notes/a.md", "chunk_index": 0, "score": 0.9, "rank": 1},
            {"path": "notes/a.md", "chunk_index": 7, "score": 0.4, "rank": 2},
            {"path": "notes/b.md", "chunk_index": 1, "score": 0.3, "rank": 3},
        ],
    )
    _install(monkeypatch, paths, ctx)

    with TestClient(create_app()) as client:
        _post(client, [{"role": "user", "content": "вопрос"}])

    source = rag_log.read_records(paths)[0]["sources"][0]
    assert source["chunk_indexes"] == [0, 7]
    assert source["chunk_index"] == 0


def test_log_carries_no_secrets_from_context(tmp_path, monkeypatch):
    """Полный текст контекста в логе — но учётные данные в него не просачиваются."""
    paths = _paths(tmp_path)
    block = "### Источник 1: Док — notes/a.md\nтекст фрагмента\n"
    ctx = _ctx(
        system_message={"role": "system", "content": "правила"},
        user_message={
            "role": "user",
            "content": f"Источники:\n\n{block}\n\nнапоминание\n\nВопрос: q",
        },
        sources=[{"n": 1, "path": "notes/a.md", "depth": "chunk", "score": 0.9}],
        context_chars=len(block),
        intent="kb_question",
    )
    _install(monkeypatch, paths, ctx)

    with TestClient(create_app()) as client:
        _post(client, [{"role": "user", "content": "вопрос"}])

    rec = rag_log.read_records(paths)[0]
    assert rec["context_text"] == block
    assert "token" not in rec["settings"]["gigachat"]
    assert "key_passphrase" not in rec["settings"]["gigachat"]
    raw = rag_log.log_path(paths).read_text(encoding="utf-8")
    for leaked in ("Authorization", "Bearer", "passphrase", "cert_path"):
        assert leaked not in raw


def test_notice_branch_still_wins_over_sources(tmp_path, monkeypatch):
    """Сбой поиска: терминальная ошибка, модель без источников не вызывается."""
    paths = _paths(tmp_path)
    ctx = _ctx(notice="Поиск недоступен", intent="kb_question", standalone_question="ЕФС")
    calls = _install(monkeypatch, paths, ctx)

    with TestClient(create_app()) as client:
        resp = _post(client, [{"role": "user", "content": "как настроить ЕФС?"}])

    events = _events(resp.text)
    assert events == ["meta", "error"]
    error = _parse_sse(resp.text)[-1][1]
    assert error["code"] == "RETRIEVAL_UNAVAILABLE"
    assert "повторить запрос" in error["message"]
    assert [call["kind"] for call in calls] == ["rag"]
    rec = rag_log.read_records(paths)[0]
    assert rec["notice"] == "Поиск недоступен"
    assert rec["rag_used"] is False
    assert rec["intent"] == "kb_question"
    assert rec["errored"] is True


# --------------------------------------------------------------------------- #
# Скрытые вызовы и пустой ответ в записи лога
# --------------------------------------------------------------------------- #


def _hidden_calls() -> dict:
    return {
        "condense": {
            "status": "ok",
            "error": None,
            "detail": None,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
            "model": "glm-5.1",
            "ms": 1200.5,
        },
        "grader": {
            "status": "failed",
            "detail": None,
            "ms": 900.0,
            "batches": [
                {
                    "n": 1,
                    "size": 2,
                    "ids": [1, 2],
                    "status": "failed",
                    "error": "KitaiQueryFailed: KitAI завершил запрос со статусом «failed»",
                    "detail": "error.status=404; No such model",
                    "finish_reason": None,
                    "usage": None,
                    "model": "glm-5.1",
                    "ms": 900.0,
                    "graded": [],
                    "omitted": [],
                }
            ],
        },
    }


def test_log_records_hidden_calls_and_the_effective_model(tmp_path, monkeypatch):
    """Почему реранкер молчал — в записи, а не только в логе пода."""
    paths = _paths(tmp_path)
    ctx = _kb_ctx()
    ctx.hidden_calls = _hidden_calls()
    _install(monkeypatch, paths, ctx, answer="ответ [Источник 1]")

    with TestClient(create_app()) as client:
        _post(client, [{"role": "user", "content": "как настроить ЕФС?"}])

    rec = rag_log.read_records(paths)[0]
    assert rec["hidden_calls"] == _hidden_calls()
    assert rec["hidden_calls"]["grader"]["batches"][0]["error"].startswith("KitaiQueryFailed")
    assert rec["empty_answer"] is False
    assert rec["answer_finish_reason"] == rec["finish_reason"]
    assert rec["settings"]["gigachat"]["provider"] == "gigachat"
    assert "kitai_model" in rec["settings"]["gigachat"]
    assert rec["settings"]["model_effective"] == {
        "provider": "gigachat",
        "model": rec["settings"]["gigachat"]["model"],
    }


def test_empty_answer_is_flagged_only_without_an_error(tmp_path, monkeypatch):
    """Ни символа от модели без ошибки транспорта — это `length` на нуле
    контента, а не обрыв; по `answer_chars: 0` их не различить."""
    paths = _paths(tmp_path)
    _install(monkeypatch, paths, _kb_ctx(), answer="   ")

    with TestClient(create_app()) as client:
        _post(client, [{"role": "user", "content": "как настроить ЕФС?"}])

    rec = rag_log.read_records(paths)[0]
    assert rec["empty_answer"] is True
    assert rec["errored"] is False

    # А вот обрыв транспорта — не «пустой ответ».
    from app.llm_errors import GigaChatHTTP

    async def broken(messages, gcfg):
        raise GigaChatHTTP("GIGACHAT_HTTP_503", "GigaChat вернул HTTP 503", "down")
        yield  # pragma: no cover — делает функцию генератором

    monkeypatch.setattr(chat_routes.llm, "stream_chat", broken)
    with TestClient(create_app()) as client:
        _post(client, [{"role": "user", "content": "как настроить ЕФС?"}])

    rec = rag_log.read_records(paths)[-1]
    assert rec["errored"] is True
    assert rec["empty_answer"] is False


def test_hidden_calls_are_none_outside_rag_mode(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    _install(monkeypatch, paths, _kb_ctx())

    with TestClient(create_app()) as client:
        client.post("/api/chat", json={"messages": [{"role": "user", "content": "привет"}], "rag": False})

    rec = rag_log.read_records(paths)[0]
    assert rec["hidden_calls"] is None
    assert rec["rag_used"] is False


# --------------------------------------------------------------------------- #
# Пустой ответ при finish_reason=length: один повтор с бо́льшим бюджетом
# --------------------------------------------------------------------------- #


def _install_length_streams(monkeypatch, paths, answers: list[str]):
    """`stream_chat`, отдающий по очереди `answers`; пустой = finish_reason=length."""
    calls: list[dict] = []

    async def fake_build_rag_context(query, *args, **kwargs):
        return _kb_ctx()

    async def fake_stream_chat(messages, gcfg):
        text = answers[len(calls)]
        calls.append({"max_tokens": gcfg.max_tokens, "messages": [dict(m) for m in messages]})
        gcfg.last_finish_reason = "length" if not text.strip() else "stop"
        if text:
            yield text

    monkeypatch.setattr(chat_routes, "resolve_paths", lambda request: paths)
    monkeypatch.setattr(chat_routes.rag, "build_rag_context", fake_build_rag_context)
    monkeypatch.setattr(chat_routes.llm, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(chat_routes.llm, "files_present", lambda gcfg: None)
    return calls


def test_empty_length_answer_is_retried_once_with_a_bigger_budget(tmp_path, monkeypatch):
    """Модель рассуждала до исчерпания бюджета и не выдала ни символа.

    Один повтор с удвоенным max_tokens — той же историей, тем же контекстом;
    настройки пользователя не трогаются, запись в логе и в истории одна.
    """
    paths = _paths(tmp_path)
    calls = _install_length_streams(monkeypatch, paths, ["", "ответ со второй попытки [Источник 1]"])

    with TestClient(create_app()) as client:
        resp = _post(client, [{"role": "user", "content": "как настроить ЕФС?"}], max_tokens=1000)

    events = _parse_sse(resp.text)
    assert [n for n, _ in events] == ["meta", "sources", "token", "done"]
    assert dict(events)["done"]["finish_reason"] == "stop"
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 1000 and calls[1]["max_tokens"] == 2000
    assert calls[0]["messages"] == calls[1]["messages"]

    records = rag_log.read_records(paths)
    assert len(records) == 1
    rec = records[0]
    assert rec["answer_retries"] == 1
    assert rec["answer_retry_reason"] == "empty_length"
    assert rec["empty_answer"] is False
    assert rec["answer_text"] == "ответ со второй попытки [Источник 1]"
    assert "retry" in rec["timings_ms"]
    chat_id = dict(events)["meta"]["chat_id"]
    saved = load_chat(chat_id, paths)["messages"]
    assert [m["role"] for m in saved] == ["user", "assistant"]
    assert saved[-1]["content"] == "ответ со второй попытки [Источник 1]"


def _context_window(monkeypatch, tokens: int) -> None:
    """Окно модели — админский ключ; подменяем действующий конфиг целиком."""
    base = chat_routes._effective_config

    def patched(paths):
        cfg = base(paths)
        cfg["gigachat"] = {**cfg.get("gigachat", {}), "model_context_tokens": tokens}
        return cfg

    monkeypatch.setattr(chat_routes, "_effective_config", patched)


def test_retry_budget_is_capped_by_half_the_context_window(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    calls = _install_length_streams(monkeypatch, paths, ["", "ok"])
    _context_window(monkeypatch, 8192)

    with TestClient(create_app()) as client:
        _post(client, [{"role": "user", "content": "как настроить ЕФС?"}], max_tokens=3000)

    assert [c["max_tokens"] for c in calls] == [3000, 4096]


def test_no_retry_when_the_budget_cannot_grow(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    calls = _install_length_streams(monkeypatch, paths, ["", "ok"])
    _context_window(monkeypatch, 8192)

    with TestClient(create_app()) as client:
        resp = _post(client, [{"role": "user", "content": "как настроить ЕФС?"}], max_tokens=4096)

    assert len(calls) == 1
    rec = rag_log.read_records(paths)[0]
    assert rec["answer_retries"] == 0 and rec["answer_retry_reason"] is None
    assert rec["empty_answer"] is True
    assert "notice" in _events(resp.text)


def test_two_empty_answers_end_in_a_visible_notice(tmp_path, monkeypatch):
    """Вместо пустого пузыря — уведомление; `empty_answer` остаётся true."""
    paths = _paths(tmp_path)
    calls = _install_length_streams(monkeypatch, paths, ["", "  "])

    with TestClient(create_app()) as client:
        resp = _post(client, [{"role": "user", "content": "как настроить ЕФС?"}])

    assert len(calls) == 2
    events = _parse_sse(resp.text)
    names = [n for n, _ in events]
    assert names == ["meta", "sources", "token", "notice", "done"]
    notice = dict(events)["notice"]["message"]
    assert notice.startswith("Модель не вернула текст: бюджет ответа исчерпан (finish_reason=length)")
    assert dict(events)["done"]["finish_reason"] == "length"

    records = rag_log.read_records(paths)
    assert len(records) == 1
    rec = records[0]
    assert rec["empty_answer"] is True
    assert rec["answer_retries"] == 1
    assert rec["answer_retry_reason"] == "empty_length"
    assert rec["notice"] == notice
    assert rec["finish_reason"] == "length"


def test_a_normal_answer_is_never_retried(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    calls = _install_length_streams(monkeypatch, paths, ["сразу ответ", "лишний"])

    with TestClient(create_app()) as client:
        _post(client, [{"role": "user", "content": "как настроить ЕФС?"}])

    assert len(calls) == 1
    rec = rag_log.read_records(paths)[0]
    assert rec["answer_retries"] == 0
