"""Блок «состав базы знаний» — итерация 1, шаг 1.

Покрывает:

* :mod:`app.corpus_map` — свёртка листинга в разделы со счётчиками, спуск на
  уровень ниже у вольта с единственным корнем, отсев вложений, постоянный
  размер блока, кэш листинга и молчаливая деградация;
* :func:`app.rag.build_rag_context` — блок стоит ПЕРЕД «Источниками», хвост
  (напоминание → вопрос) не сдвигается, ``context_chars`` по-прежнему меряет
  только блок источников, а при недоступном листинге сообщение совпадает с
  прежним посимвольно;
* :func:`app.routes.chat_routes._rendered_context` — в лог по-прежнему уезжает
  ровно блок источников, без карты;
* :func:`app.routes.chat_routes._invalid_citations` — что валидатор цитат
  ловит, а что нет, если модель всё-таки сошлётся на карту.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import catalog, cognivault, corpus_map, rag, rag_pipeline, settings  # noqa: E402
from app.config import AppPaths  # noqa: E402
from app.main import create_app  # noqa: E402
from app.routes import chat_routes  # noqa: E402
from app.tokens import estimate_tokens  # noqa: E402


# Настоящий загрузчик листинга, снятый ДО того, как общая фикстура
# `conftest._no_vault_listing` подменит его заглушкой: в этом файле проверяется
# как раз он сам (кэш, таймаут, деградация), поэтому здесь он возвращается на
# место.
_REAL_FILES = corpus_map.files


# Что этот сервис считает документом. Не список этого файла и не список
# `corpus_map` — так его отдаёт бэкенд в `GET /api/vault/catalog`
# (`DOCUMENT_EXTENSIONS` в `src/lib/indexer.ts`). Ни `txt`, ни `markdown` в нём
# нет: их индексатор не сканирует вовсе.
_EXTENSIONS = ("md", "pdf", "canvas", "excalidraw", "csv")


def _catalog_payload(extensions=_EXTENSIONS, documents=()) -> dict:
    return {
        "status": "ok" if documents else "summaries_pending",
        "summaries_enabled": True,
        "reason": None,
        "documents": list(documents),
        "total": len(documents),
        "offset": 0,
        "documents_with_summary": 0,
        "document_extensions": (
            list(extensions) if isinstance(extensions, (list, tuple)) else extensions
        ),
    }


@pytest.fixture(autouse=True)
def _local_mode(monkeypatch):
    monkeypatch.setattr(settings, "is_server", lambda: False)
    monkeypatch.setattr(corpus_map, "files", _REAL_FILES)
    # Каталог доступен: без него блок «состав базы» не строится вообще —
    # именно оттуда берётся определение документа (см. `_install_catalog`).
    _install_catalog(monkeypatch, _catalog_payload())


def _install_catalog(monkeypatch, payload=None) -> None:
    """Подменить шов каталога. ``payload=None`` — «каталог недоступен»."""

    async def fake_payload(cv=None):
        return payload

    catalog.reset_cache()
    monkeypatch.setattr(
        catalog, "payload", fake_payload if payload is not None else _unavailable
    )


async def _unavailable(cv=None):
    return None


# Реалистичный корпус: 127 документов, архив, пустые страницы-контейнеры,
# вложения Confluence рядом с заметками.
def _corpus() -> list[str]:
    paths: list[str] = []
    for i in range(54):
        sub = ["Fincert", "ППРБ", "PSI", "Витрины", "Отчётность", "Прочее"][i % 6]
        paths.append(f"Продукты/{sub}/Страница {i}.md")
    for i in range(31):
        paths.append(f"Архив/Проекты {i}.md")
    for i in range(24):
        paths.append(f"Регламенты/Раздел {i % 3}/Док {i}.md")
    for i in range(14):
        paths.append(f"Процессы/Док {i}.md")
    paths += ["Продукты.md", "Индекс.md", "README.md", "Глоссарий.md"]
    # Не документы: вложения и картинки в счёт идти не должны.
    paths += [f"Confluence/attachments/1234/файл{i}.png" for i in range(40)]
    paths += ["Confluence/attachments/1234/смета.xlsx"]
    return paths


# --------------------------------------------------------------------------- #
# Сборка блока из листинга
# --------------------------------------------------------------------------- #


def test_block_reports_totals_and_top_sections():
    block = corpus_map.render(_corpus(), n_sources=5, extensions=frozenset(_EXTENSIONS))

    assert block is not None
    # Архивные 31 и вложения не входят в активную базу: 54 + 24 + 14 + 4 = 96.
    assert "Всего документов в базе: 96." in block
    assert "Ниже в блоке «Источники» — 5 фрагментов" in block
    # Разделы верхнего уровня с числом документов, по убыванию объёма.
    assert "- Продукты — 54" in block
    assert "Архив" not in block
    assert "- Регламенты — 24" in block
    assert "- (корень) — 4" in block
    assert block.index("- Продукты") < block.index("- Процессы")
    # Подразделы — со своими счётчиками.
    assert "Fincert: 9" in block
    # Явная пометка: не источник, ссылаться нельзя.
    assert "НЕ источник" in block and "ссылаться на него нельзя" in block


def test_block_counts_only_documents():
    """Картинки и вложения не документы: 40 png не превращаются в раздел."""
    block = corpus_map.render(
        ["Заметки/а.md", "Confluence/attachments/1/x.png", "logo.svg"],
        extensions=frozenset(_EXTENSIONS),
    )
    assert block is not None
    assert "Всего документов в базе: 1." in block
    assert "attachments" not in block


def test_block_omits_source_count_when_unknown():
    block = corpus_map.render(["Заметки/а.md", "б.md"], extensions=frozenset(_EXTENSIONS))
    assert "Всего документов в базе: 2." in block
    assert "Источники" not in block.split("\n")[1]


def test_single_root_folds_one_level_deeper():
    """Вольт из Confluence-синка весь лежит под ``Confluence/`` — спускаемся."""
    paths = [f"Confluence/ПРОСТРАНСТВО {i % 3}/Раздел {i % 2}/Стр {i}.md" for i in range(30)]
    block = corpus_map.render(paths, n_sources=3, extensions=frozenset(_EXTENSIONS))

    assert block is not None
    assert "Разделы внутри «Confluence»" in block
    assert "- ПРОСТРАНСТВО 0 — 10" in block
    # Ни один документ не потерян при спуске.
    assert "Всего документов в базе: 30." in block


def test_descent_continues_through_every_single_folder_level():
    """Настоящий вольт Confluence-синка ветвится только на ГЛУБИНЕ 5.

    `build_vault_path` (app/confluence/convert.py) кладёт страницу в
    ``Confluence/<пространство>/<предки…>/<Заголовок>.md``, и на боевой выгрузке
    подряд идут ЧЕТЫРЕ папки-коридора: ``Confluence`` → ``OASISEXT`` → ``OASIS
    External Home`` → ``Разработка …``. Спуск ровно на один уровень оставлял блок
    со строкой ``- Confluence — 127 (OASISEXT: 127)`` — то есть платой за блок,
    который про форму базы не говорит НИЧЕГО.

    Четвёртый коридор — тот, на котором спуск спотыкался: рядом с папкой лежит
    её собственная страница-раздел (``…/OASIS External Home/Разработка ….md``),
    и уровень из «одной папки плюс один файл» читался как ветвление. Итог на
    боевом корпусе — ``- Разработка … — 126`` и ``- (корень) — 1``: три названия
    из бюджета в 2 400 символов. Уровень — коридор, когда в нём одна ПАПКА
    (`_named`), а не одна запись.
    """
    root = "Confluence/OASISEXT/OASIS External Home/Разработка"
    paths = (
        [f"{root}/Продукты/Стр {i}.md" for i in range(110)]
        + [f"{root}/Архив/Стр {i}.md" for i in range(10)]
        + [f"{root}/База знаний/Стр {i}.md" for i in range(3)]
        + [f"{root}/{name}.md" for name in ("Продукты", "Архив", "База знаний")]
        + ["Confluence/OASISEXT/OASIS External Home/Разработка.md"]
    )

    block = corpus_map.render(paths, n_sources=5, extensions=frozenset(_EXTENSIONS))

    assert block is not None
    assert "Всего документов в базе: 116." in block
    # Все четыре коридора уехали в заголовок, а разделы — настоящие.
    assert "Разделы внутри «Confluence/OASISEXT/OASIS External Home/Разработка»" in block
    assert "- Продукты — 110" in block
    assert "Архив" not in block
    assert "- База знаний — 3" in block
    # Страницы-разделы, лежащие рядом с папками, остаются видны как «(корень)».
    assert "- (корень) — 2" in block
    assert len(block) <= corpus_map._MAX_MAP_CHARS


def test_descent_does_not_hide_a_level_that_is_mostly_loose_files():
    """Коридор — это «почти всё под одной папкой», а не «одна папка есть».

    Вольт из 50 заметок в корне и одной папки на два документа: спуск в папку
    спрятал бы 50 документов из 52, оставив блок, который про базу врёт. Порог
    :data:`corpus_map._CORRIDOR_MIN_SHARE` держит именно этот случай.
    """
    paths = [f"Заметка {i}.md" for i in range(50)] + ["Проекты/а.md", "Проекты/б.md"]

    block = corpus_map.render(paths, extensions=frozenset(_EXTENSIONS))

    assert block is not None
    assert "Разделы верхнего уровня" in block
    assert "- (корень) — 50" in block
    assert "- Проекты — 2" in block


def test_descent_stops_before_swallowing_the_documents_themselves():
    """Спуск не уходит в уровень, где остались одни файлы: ``A/x.md`` — это «A»."""
    block = corpus_map.render(["A/x.md", "A/y.md"], extensions=frozenset(_EXTENSIONS))
    assert "- A — 2" in block
    assert "внутри" not in block


def test_no_descent_when_top_level_already_branches():
    block = corpus_map.render(["A/x.md", "B/y.md"], extensions=frozenset(_EXTENSIONS))
    assert "Разделы верхнего уровня" in block
    assert "внутри" not in block


def test_empty_or_documentless_listing_yields_no_block():
    assert corpus_map.render([], extensions=frozenset(_EXTENSIONS)) is None
    assert corpus_map.render(["a.png", "b.zip"], extensions=frozenset(_EXTENSIONS)) is None
    assert corpus_map.render([None, 42], extensions=frozenset(_EXTENSIONS)) is None  # мусор в листинге не роняет


# --------------------------------------------------------------------------- #
# Постоянный размер
# --------------------------------------------------------------------------- #


def test_block_size_does_not_grow_with_the_corpus():
    """Корпус в 40 раз больше — блок того же порядка и под потолком."""
    small = corpus_map.render(_corpus(), n_sources=5, extensions=frozenset(_EXTENSIONS))
    huge = corpus_map.render(
        [
            f"Раздел {i % 40}/Подраздел {i % 300}/Глубже/Док {i}.md"
            for i in range(5000)
        ],
        n_sources=5,
        extensions=frozenset(_EXTENSIONS),
    )

    assert small is not None and huge is not None
    for block in (small, huge):
        assert len(block) <= corpus_map._MAX_MAP_CHARS
        # ~150 токенов по проектной (заведомо пессимистичной) оценке 2.5 симв/токен.
        assert estimate_tokens(block) <= 280
    # Урезанные разделы не исчезают из арифметики.
    assert "Всего документов в базе: 5000." in huge
    assert "ещё" in huge


def test_truncation_keeps_every_document_accounted_for():
    paths = [f"Раздел {i}/док.md" for i in range(60)]
    block = corpus_map.render(paths, max_chars=400, extensions=frozenset(_EXTENSIONS))
    assert "Всего документов в базе: 60." in block
    assert len(block) <= 400
    # Урезано до нескольких разделов, но недосчитанных документов нет.
    shown = block.count("\n- ") - 1
    assert f"ещё {60 - shown} " in block


# --------------------------------------------------------------------------- #
# Что считается документом — D8, вторая половина
# --------------------------------------------------------------------------- #
#
# Список расширений больше не живёт в `corpus_map`. Он приходит полем
# `document_extensions` из `GET /api/vault/catalog`, где выводится из той же
# константы, по которой поллер сканирует вольт. Прежний локальный список врал в
# обе стороны сразу: `txt` и `markdown` индексатор не сканирует, не чанкует и не
# эмбеддит — каждый `.txt` считался документом, про который можно спросить и
# который поиск никогда не вернёт.


def test_document_count_follows_the_service_definition_not_a_local_list():
    """Вольт из `a.md` и `b.txt` — это ОДИН документ, а не два."""
    block = corpus_map.render(
        ["a.md", "b.txt"], extensions=frozenset(_EXTENSIONS)
    )
    assert block is not None
    assert "Всего документов в базе: 1." in block


def test_txt_is_not_a_document_end_to_end(monkeypatch):
    """То же самое через оба сетевых шва: считает каталог, не модуль."""
    _install_listing(monkeypatch, ["a.md", "b.txt"])
    assert asyncio.run(corpus_map.document_count(None)) == 1
    block = asyncio.run(corpus_map.corpus_block(None, 1))
    assert "Всего документов в базе: 1." in block


def test_extensions_are_taken_from_the_catalogue_verbatim(monkeypatch):
    """Другой вольт — другое определение документа; модуль своего не имеет."""
    _install_listing(monkeypatch, ["a.md", "b.txt"], extensions=("md", "txt"))
    assert asyncio.run(corpus_map.document_count(None)) == 2


def test_unavailable_catalogue_yields_no_block_rather_than_a_guess(monkeypatch):
    """Определения документа нет — блока нет. Встроенного списка не осталось."""
    _install_listing(monkeypatch, _corpus())
    _install_catalog(monkeypatch, None)

    assert asyncio.run(corpus_map.corpus_block(None, 5)) is None
    assert asyncio.run(corpus_map.overview_block(None)) is None
    assert asyncio.run(corpus_map.document_count(None)) is None


def test_malformed_extension_list_is_not_a_definition(monkeypatch):
    """Пустой/битый `document_extensions` — это «неизвестно», а не «всё подряд»."""
    _install_listing(monkeypatch, _corpus())
    for broken in ([], "md", [""], [1, 2], None):
        _install_catalog(monkeypatch, _catalog_payload(extensions=broken))
        assert asyncio.run(corpus_map.corpus_block(None, 5)) is None


def test_module_no_longer_owns_an_extension_list():
    """Сторож: локальный список удалён и не должен вернуться."""
    assert not hasattr(corpus_map, "_DOC_EXTENSIONS")


# --------------------------------------------------------------------------- #
# Кэш и деградация
# --------------------------------------------------------------------------- #


def test_listing_is_fetched_once_per_ttl(monkeypatch):
    calls: list[dict] = []

    async def fake_list_files(cv=None, recursive=True, timeout=None):
        calls.append({"recursive": recursive, "timeout": timeout})
        return ["Заметки/а.md", "б.md"]

    monkeypatch.setattr(cognivault, "list_files", fake_list_files)
    corpus_map.reset_cache()

    first = asyncio.run(corpus_map.corpus_block(None, 2))
    second = asyncio.run(corpus_map.corpus_block(None, 3))

    assert first is not None and second is not None
    assert len(calls) == 1
    # Короткий таймаут: зависший вольт не должен добавляться к каждому ходу.
    assert calls[0]["timeout"] == corpus_map._LIST_TIMEOUT


def test_listing_is_cached_per_vault(monkeypatch):
    """Токен — это арендатор: чужой листинг не должен подставиться."""
    seen: list[str] = []

    async def fake_list_files(cv=None, recursive=True, timeout=None):
        token = (cv or {}).get("token", "")
        seen.append(token)
        return [f"{token}/док.md", f"{token}/ещё.md"]

    monkeypatch.setattr(cognivault, "list_files", fake_list_files)
    corpus_map.reset_cache()

    a = asyncio.run(corpus_map.corpus_block({"base_url": "http://x", "token": "aaa"}))
    b = asyncio.run(corpus_map.corpus_block({"base_url": "http://x", "token": "bbb"}))

    assert seen == ["aaa", "bbb"]
    assert "aaa" in a and "bbb" in b


def test_failed_listing_degrades_silently(monkeypatch):
    async def boom(cv=None, recursive=True, timeout=None):
        raise cognivault.CogniVaultError("list files failed (503)", 503, "")

    monkeypatch.setattr(cognivault, "list_files", boom)
    corpus_map.reset_cache()

    assert asyncio.run(corpus_map.corpus_block(None, 5)) is None
    # Отказ тоже кэшируется — упавший вольт не зовётся на каждом ходе.
    assert asyncio.run(corpus_map.corpus_block(None, 5)) is None


# --------------------------------------------------------------------------- #
# Место в собранном промпте
# --------------------------------------------------------------------------- #


def _hit(i: int) -> dict:
    return {
        "path": f"doc{i}.md",
        "title": f"Документ {i}",
        "section_path": "",
        "score": 1.0 - i / 100,
        "text": f"содержимое фрагмента номер {i}",
        "chunk_index": i,
        "rank": i,
    }


def _install_retrieval(monkeypatch, hits: list[dict]) -> None:
    async def fake_hybrid(query, limit, cv=None, **kwargs):
        return {"results": hits}

    async def fake_content(path, cv=None):
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


def _install_listing(
    monkeypatch, paths: list[str] | None, extensions=_EXTENSIONS
) -> None:
    async def fake_list_files(cv=None, recursive=True, timeout=None):
        if paths is None:
            raise cognivault.CogniVaultError("list files failed (503)", 503, "")
        return paths

    monkeypatch.setattr(cognivault, "list_files", fake_list_files)
    corpus_map.reset_cache()
    _install_catalog(monkeypatch, _catalog_payload(extensions=extensions))


def _build(query: str) -> rag.RagContext:
    return asyncio.run(
        rag.build_rag_context(
            query, {"mode": "auto", "max_expanded_files": 0}, None, {}, None
        )
    )


def test_block_precedes_sources_and_reminder_keeps_the_tail(monkeypatch):
    _install_retrieval(monkeypatch, [_hit(1), _hit(2)])
    _install_listing(monkeypatch, _corpus())

    ctx = _build("вопрос про архитектуру сервиса")
    content = ctx.user_message["content"]

    assert content.startswith("Состав базы знаний")
    assert content.index("Состав базы знаний") < content.index("Источники:")
    assert content.index("Источники:") < content.index("Напоминание:")
    assert content.index("Напоминание:") < content.index("Вопрос:")
    # Модель видит, сколько фрагментов ей дали из скольких документов.
    assert "Всего документов в базе: 127." in content
    assert f"— {len(ctx.sources)} " in content


def test_context_chars_still_measures_only_the_sources_block(monkeypatch):
    _install_retrieval(monkeypatch, [_hit(1), _hit(2)])
    _install_listing(monkeypatch, _corpus())
    ctx = _build("вопрос про архитектуру сервиса")

    logged = chat_routes._rendered_context(ctx.user_message, ctx.context_chars)
    assert logged.startswith("### Источник 1:")
    assert "Состав базы знаний" not in logged
    assert len(logged) == ctx.context_chars


def test_answer_context_is_unchanged_when_listing_is_unavailable(monkeypatch):
    """Ход без листинга совпадает с прежним посимвольно."""
    _install_retrieval(monkeypatch, [_hit(1), _hit(2)])

    _install_listing(monkeypatch, None)
    without = _build("вопрос про архитектуру сервиса")
    _install_listing(monkeypatch, _corpus())
    with_map = _build("вопрос про архитектуру сервиса")

    assert without.user_message["content"].startswith("Источники:")
    assert without.context_chars == with_map.context_chars
    assert without.sources == with_map.sources
    # Единственная разница — приписанный сверху блок.
    block = with_map.user_message["content"][
        : with_map.user_message["content"].index("Источники:")
    ]
    assert with_map.user_message["content"] == block + without.user_message["content"]


def test_cost_per_turn_stays_inside_the_char_budget(monkeypatch):
    """Цена блока — доли процента от ``max_context_chars`` (24000 по умолчанию)."""
    _install_retrieval(monkeypatch, [_hit(1), _hit(2)])
    _install_listing(monkeypatch, _corpus())
    ctx = _build("вопрос про архитектуру сервиса")

    content = ctx.user_message["content"]
    overhead = len(content) - len(content[content.index("Источники:") :])
    assert overhead <= corpus_map._MAX_MAP_CHARS + 2  # + "\n\n"
    assert overhead < 0.05 * 24000


def test_block_reaches_the_model_and_survives_trimming(monkeypatch, tmp_path):
    """Сквозь маршрут: блок доезжает до GigaChat и не срезается урезанием истории."""
    captured: list[list[dict]] = []

    async def fake_stream_chat(messages, gcfg):
        captured.append([dict(m) for m in messages])
        yield "ответ"

    _install_retrieval(monkeypatch, [_hit(1), _hit(2)])
    _install_listing(monkeypatch, _corpus())
    monkeypatch.setattr(
        chat_routes, "resolve_paths", lambda request: AppPaths(root=tmp_path / "ui")
    )
    monkeypatch.setattr(chat_routes.llm, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(chat_routes.llm, "files_present", lambda gcfg: None)

    history = [
        {"role": "user", "content": f"вопрос {i} " * 200} for i in range(30)
    ]
    # Обычный вопрос по содержанию, НЕ метавопрос: «о каких проектах знаешь»
    # теперь уходит в ветку `corpus_scope.match_meta` и до блока «Источники»
    # не доходит вовсе (см. `tests/test_corpus_scope.py`).
    question = "какие поля у витрины fincert_feeds"
    with TestClient(create_app()) as client:
        resp = client.post(
            "/api/chat",
            json={
                "messages": [*history, {"role": "user", "content": question}],
                "rag": True,
            },
        )

    assert resp.status_code == 200
    last = captured[0][-1]["content"]
    assert last.startswith("Состав базы знаний")
    assert last.index("Всего документов в базе: 127.") < last.index("Источники:")
    assert last.endswith(f"Вопрос: {question}")


# --------------------------------------------------------------------------- #
# Что ловит валидатор цитат, если модель всё-таки сошлётся на карту
# --------------------------------------------------------------------------- #


def test_citation_validator_catches_only_out_of_range_numbers():
    # Модель придумала шестой источник при пяти — поймано.
    assert chat_routes._invalid_citations("факт [Источник 6]", 5) == [6]
    assert chat_routes._invalid_citations("факт [Источник 0]", 5) == [0]
    # Но приписать сведения из карты существующему номеру она может свободно:
    # валидатор проверяет ДИАПАЗОН, а не происхождение утверждения.
    assert chat_routes._invalid_citations("в базе 127 документов [Источник 1]", 5) == []


# --------------------------------------------------------------------------- #
# Страницы-разделы: у какого документа есть потомки
# --------------------------------------------------------------------------- #


def test_containers_finds_pages_that_have_descendants():
    """``X.md`` рядом с папкой ``X/`` — оглавление раздела, а не рядовая страница.

    Это единственный признак, по которому оговорка (`app.corpus_scope.hedge`)
    отличает «ответ собран из одной случайной страницы» от «ответ собран из
    страницы, которая и есть перечень раздела». Считается по листингу, без
    единого дополнительного запроса.
    """
    paths = [
        "Продукты/Fincert.md",
        "Продукты/Fincert/Сервис A.md",
        "Продукты/Fincert/Сервис B.md",
        "Продукты/Одинокая страница.md",
        "Продукты.md",
    ]

    found = corpus_map.containers(paths, frozenset(_EXTENSIONS))

    assert found == {"Продукты/Fincert.md", "Продукты.md"}


def test_containers_ignores_non_documents_and_garbage():
    """Папка вложений не делает страницу разделом, а мусор в листинге не роняет."""
    paths = ["Стр.md", "Стр/фото.png", None, 42, "Другая.md", "Другая/вложенная.md"]

    found = corpus_map.containers(paths, frozenset(_EXTENSIONS))

    assert found == {"Другая.md"}


def test_containers_of_an_empty_listing_is_empty_not_none():
    """Пустое множество — это ответ «страниц-разделов нет», а не «неизвестно».

    Разницу несёт `container_paths`: ``None`` там означает недоступный листинг,
    и оговорка при нём молчит.
    """
    assert corpus_map.containers([], frozenset(_EXTENSIONS)) == frozenset()
