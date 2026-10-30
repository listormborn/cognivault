"""Вопросы про саму базу — итерация 2, шаг 2 (а/б/в).

Покрывает:

* :func:`app.corpus_scope.match_meta` — детерминированный распознаватель
  метавопросов: якорные формулировки совпадают, тематические — нет
  («Что ты знаешь про PSI?» ≠ «Что ты знаешь?»), непопадание проваливается в
  сегодняшнее поведение;
* :func:`app.corpus_scope.parse_scope` — поле охвата из condense, включая
  отсутствие поля (промпт пользовательски редактируем — вырезанная фраза не
  должна ломать маршрут);
* :func:`app.corpus_scope.hedge` — оговорка по концентрации доказательств и,
  главное, ИЗМЕРЕНИЕ: доля ложных оговорок на замороженной контрольной группе
  из 56 вопросов класса B (см. ``tools/eval/golden.control.json``);
* :func:`app.rag.build_rag_context` — мета-ветка отвечает по дереву разделов,
  без поиска и без вызовов модели, и молча уходит в обычный путь, когда листинг
  вольта недоступен;
* :mod:`app.routes.chat_routes` — оговорка дописывается ПОСЛЕ ответа модели,
  тем же кадром ``token``, и попадает в историю.

Чего эти тесты НЕ видят (и не могут увидеть офлайн): вердикта живого
классификатора. Ложная оговорка возможна ровно одним способом — condense назовёт
охват ``corpus`` на вопросе про один документ; здесь зафиксировано, что при
любом ДРУГОМ вердикте (в том числе при отсутствующем, сломанном и выключенном)
оговорки нет.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (  # noqa: E402
    catalog,
    cognivault,
    corpus_map,
    corpus_scope,
    rag,
    rag_pipeline,
    settings,
)
from app.config import AppPaths  # noqa: E402
from app.main import create_app  # noqa: E402
from app.routes import chat_routes  # noqa: E402

# Настоящий загрузчик листинга, снятый ДО подмены из `conftest`: мета-ветка
# строится как раз из него.
_REAL_FILES = corpus_map.files


@pytest.fixture(autouse=True)
def _local_mode(monkeypatch):
    monkeypatch.setattr(settings, "is_server", lambda: False)


# --------------------------------------------------------------------------- #
# 2а. Распознаватель метавопросов
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "question, kind",
    [
        ("Что ты знаешь?", "assistant"),
        ("что ты вообще знаешь", "assistant"),
        ("Что ты умеешь?", "assistant"),
        ("Кто ты?", "assistant"),
        ("Чем ты можешь помочь?", "assistant"),
        ("Привет! Что ты знаешь?", "assistant"),
        ("Подскажи, что ты умеешь?", "assistant"),
        ("О чём эта база?", "corpus"),
        ("О каких проектах знаешь?", "corpus"),
        ("О каких продуктах есть информация в базе знаний?", "corpus"),
        ("Из каких разделов состоит база знаний?", "corpus"),
        ("Какие разделы есть в базе знаний?", "corpus"),
        ("Что лежит в базе?", "corpus"),
        ("Сколько всего страниц в базе и как они распределены по разделам?", "corpus"),
        # Составной метавопрос: обе части — метавопросы, корпусная сильнее.
        ("Что ты вообще знаешь? О чём эта база?", "corpus"),
        # --- D5: проверенные промахи, каждый уходил в отказ грейдера --------- #
        ("Что ты умеешь делать?", "assistant"),
        ("Что вы знаете?", "assistant"),
        ("Что вы умеете?", "assistant"),
        ("Кто вы?", "assistant"),
        ("Что ты знаешь вообще?", "assistant"),
        ("Какие темы ты покрываешь?", "assistant"),
        ("Какая информация у тебя есть?", "assistant"),
        ("Какие документы у тебя есть?", "assistant"),
        ("На какие вопросы ты можешь ответить?", "assistant"),
        ("На какие вопросы ты можешь отвечать?", "assistant"),
        ("О чём я могу у тебя спросить?", "assistant"),
        ("О чем я могу тебя спросить?", "assistant"),
        ("Что я могу у тебя спросить?", "assistant"),
        ("Что я могу вас спросить?", "assistant"),
        ("О чём можно тебя спросить?", "assistant"),
        ("О чем можно спросить?", "assistant"),
        ("О чем можно у тебя спросить?", "assistant"),
        ("Про что я могу у вас спросить?", "assistant"),
        ("Какие вопросы я могу задать?", "assistant"),
        ("Какие вопросы я могу задать тебе?", "assistant"),
        ("Какие вопросы можно задать вам?", "assistant"),
        ("Какие вопросы тебе можно задавать?", "assistant"),
        ("На какую тему можно задать вопрос?", "assistant"),
        ("На какие темы я могу задавать вопросы?", "assistant"),
        ("Какие вопросы мне доступны?", "assistant"),
        ("Какие есть разделы?", "corpus"),
        ("Перечисли разделы базы", "corpus"),
        ("Перечислите разделы базы знаний", "corpus"),
        ("Покажи структуру базы", "corpus"),
        ("Назови разделы базы знаний", "corpus"),
        ("Какие инструменты есть в базе?", "corpus"),
        ("Какие инструменты ты знаешь?", "corpus"),
        ("О каких инструментах есть информация в базе знаний?", "corpus"),
        # --- D4: вопросы о поведении самого ассистента ---------------------- #
        ("Всегда ли ответ в Markdown с заголовками?", "assistant"),
        ("В каком формате ты отвечаешь?", "assistant"),
        ("Как ты работаешь?", "assistant"),
        ("Откуда ты берёшь ответы?", "assistant"),
        ("Используешь ли ты markdown?", "assistant"),
    ],
)
def test_meta_questions_are_recognised(question, kind):
    assert corpus_scope.match_meta(question) == kind


@pytest.mark.parametrize(
    "question",
    [
        # Та самая пара, которую нельзя перепутать.
        "Что ты знаешь про PSI?",
        "Что ты знаешь о витрине fincert_feeds?",
        # Тематические вопросы про охват — у них есть документ-ответ, их место
        # в поиске, а не в дереве.
        "Какие витрины ClickHouse описаны в базе?",
        "Что лежит в разделе «Архив»?",
        "Какие сервисы входят в продукт Fincert?",
        "Перечисли все поля витрины fincert_feeds",
        # Приветствие само по себе — не метавопрос (это smalltalk).
        "привет",
        "Спасибо!",
        # Приветствие + содержательный вопрос: содержательная часть решает.
        "Привет! Какие колонки в таблице fincert_feeds?",
        "",
        # Расширение D5 не должно проглотить предметные формулировки: глагол
        # ушёл в наполнители, но остаток клаузы несёт предмет.
        "Перечисли все поля витрины fincert_feeds",
        "Перечисли этапы расчёта YAFCA",
        "Покажи структуру витрины feeds_all_view",
        "Опиши полный цикл расчёта финэффекта",
        "Назови все потоки наполнения витрин",
        "Какие есть ML-метрики и где про них почитать?",
        "Какие разделы в базе данных ClickHouse?",
        "В каком формате хранится дата в витрине?",
        "Как ты работаешь с витриной fincert_feeds?",
        "Какие вопросы задаются на code review?",
        # Общие формулировки выше остаются fullmatch-шаблонами: явная тема
        # переводит вопрос обратно в поиск по документам.
        "О чем можно спросить про FinCert?",
        "Какие вопросы я могу задать про YAFCA?",
        "На какую тему можно задать вопрос по Feature Store?",
    ],
)
def test_non_meta_questions_fall_through(question):
    """Непопадание — это ``None``: ход идёт ровно как сегодня (поиск + грейдер)."""
    assert corpus_scope.match_meta(question) is None


def test_matcher_ignores_long_input():
    """Длинная реплика несёт предмет, а не вопрос об охвате."""
    assert corpus_scope.match_meta("что ты знаешь " + "и " * 200) is None


@pytest.mark.parametrize(
    "question",
    [
        "На какие вопросы ты модешь отвечать?",
        "На какие вопросы ты можеш отвечать?",
        "На какие вапросы ты можешь отвечать?",
        "На какие вопросы ты можешь отвечат?",
        "О чем я магу у тебя спросить?",
    ],
)
def test_scope_questions_tolerate_one_or_two_typing_errors(question):
    """Опечатка не должна отправлять метавопрос в поиск случайных документов."""
    assert corpus_scope.match_meta(question) == corpus_scope.META_ASSISTANT
    assert corpus_scope.match_meta(question, has_history=True) == (
        corpus_scope.META_ASSISTANT
    )


@pytest.mark.parametrize(
    "question",
    [
        "На какие вопросы отвечает Fincert?",
        "Какие вопросы задаются на code review?",
        "Что ты знаешь про PSI?",
        "О чем можно спросить про Feature Store?",
    ],
)
def test_fuzzy_scope_match_does_not_swallow_subject_questions(question):
    assert corpus_scope.match_meta(question) is None


# --------------------------------------------------------------------------- #
# D6: именительный падеж — «какие продукты ты знаешь?»
# --------------------------------------------------------------------------- #
#
# Промах, снятый с прода: первый ход свежего чата, «какие проекты ты знаешь?» —
# и ответ собран из одной архивной страницы «Проекты Ислама» (личные заметки
# одного инженера в роли перечня проектов компании). Список покрывал только
# предложный падеж («о каких продуктах…»), клауза сверяется `fullmatch`, а
# единственный шаблон вида «какие X ты знаешь» жил в семье `assistant` с
# закрытым списком существительных без продуктов и проектов. `has_history` был
# False — то есть промах не в сужении по истории, а в самом списке.


@pytest.mark.parametrize(
    "question",
    [
        # Якорь — глагол второго лица.
        "Какие проекты ты знаешь?",
        "Какие продукты ты знаешь?",
        "Какие продукты вы знаете?",
        "Какие есть продукты ты знаешь?",
        "Каких проектов ты знаешь",
        "Какие системы ты знаешь?",
        "Подскажи, какие продукты ты знаешь?",
        # Якорь — притяжательное «у нас» / «в компании».
        "Какие проекты у нас есть?",
        "Какие продукты у нас есть?",
        "Какие проекты есть у нас?",
        "Какие системы есть в компании?",
        "Какие направления у вас есть?",
        "Какие продукты у тебя есть?",
    ],
)
def test_nominative_scope_questions_are_recognised(question):
    """Обе формулировки — про базу целиком, значит семья `corpus`."""
    assert corpus_scope.match_meta(question) == corpus_scope.META_CORPUS
    # Дополнение на месте, подставлять из истории нечего: сужение их не трогает.
    assert corpus_scope.match_meta(question, has_history=True) == (
        corpus_scope.META_CORPUS
    )


@pytest.mark.parametrize(
    "question",
    [
        # Голая форма с опущенным дополнением. РАССМОТРЕНА И ОТКЛОНЕНА: ей
        # понадобился бы `first_turn_only=True`, и даже на первом ходу «какие
        # проекты» — это ещё и вопрос про проекты человека, команды или раздела.
        "Какие проекты?",
        "Какие продукты?",
        # Дополнение названо и оно НЕ база: якорь обязан быть на «знаешь» или
        # на «у нас», а не на «какие проекты.*» — иначе матчер, обходящий
        # грейдер, подменит ответ деревом всей базы.
        "Какие проекты у Ислама?",
        "Какие проекты в архиве?",
        "Какие продукты знает команда?",
        "Какие проекты описаны в разделе «Архив»?",
        "Какие витрины ClickHouse описаны в базе?",
        "Какие продукты используют Feature Store?",
    ],
)
def test_the_widening_stays_anchored(question):
    assert corpus_scope.match_meta(question) is None
    assert corpus_scope.match_meta(question, has_history=True) is None


def test_the_two_noun_lists_do_not_overlap():
    """Порядок списка шаблонов не должен решать, к какой семье попал вопрос.

    `_clause_kind` возвращает ПЕРВОЕ совпадение, поэтому «какие темы ты
    знаешь?» достаётся семье `assistant` только потому, что её шаблон стоит
    выше. Повторять темы/разделы/направления в новом корпусном шаблоне значило
    бы сделать классификацию функцией порядка кортежа — вместо этого списки
    разведены, и правило про порядок не нужно вовсе. Поведение прежних
    формулировок при этом не менялось: они как были `assistant`, так и остались.
    """
    assert corpus_scope.match_meta("Какие темы ты знаешь?") == (
        corpus_scope.META_ASSISTANT
    )
    assert corpus_scope.match_meta("Какие разделы ты знаешь?") == (
        corpus_scope.META_ASSISTANT
    )
    assert corpus_scope.match_meta("Какие направления ты знаешь?") == (
        corpus_scope.META_ASSISTANT
    )

    corpus_nouns = {"продукты", "продуктов", "проекты", "проектов", "системы"}
    assistant_nouns = {"темы", "тем", "разделы", "направления", "вопросы"}
    verb_anchored = [
        pattern
        for kind, _first_turn_only, pattern in corpus_scope._PATTERNS
        if kind == corpus_scope.META_CORPUS and "знаешь|знаете" in pattern
    ]
    assert len(verb_anchored) == 1
    for noun in assistant_nouns:
        assert f"{noun}|" not in verb_anchored[0] and f"|{noun}" not in verb_anchored[0]
    for noun in corpus_nouns:
        assert noun in verb_anchored[0]


# --------------------------------------------------------------------------- #
# D-A: матчер и ВТОРОЙ ход. Пробел измерения: все тесты 2а были однооборотными
# --------------------------------------------------------------------------- #
#
# Матчер не видит истории и стоит шагом 0a, ДО condense, на каждом ходу. Значит
# каждая формулировка обязана быть безопасной как голая реплика второго хода.
# Часть формулировок таковой не была: у них опущено дополнение, и на втором ходу
# его подставляет предыдущая реплика.

#: Воспроизведение дефекта: после «Расскажи про продукт Fincert» эти реплики
#: означают разделы/темы ПРОДУКТА, а матчер отвечал деревом всей базы — без
#: единого источника и в обход грейдера (единственная ветка, которая его обходит).
_FOLLOW_UP_UNSAFE = [
    "Какие разделы?",
    "Какие темы?",
    "Какие есть разделы?",
    "Какие направления?",
    "Что ты ещё знаешь?",
    # Существительное названо, а «где» — нет: на втором ходу «где» подставляет
    # предыдущий ответ. Форма с якорем («…в базе знаний») не сужается.
    "О каких продуктах есть информация?",
]


@pytest.mark.parametrize("question", _FOLLOW_UP_UNSAFE)
def test_elided_question_matches_on_the_first_turn(question):
    """На первом ходу подставлять нечего — чтение однозначно, поведение прежнее."""
    assert corpus_scope.match_meta(question, has_history=False) is not None


@pytest.mark.parametrize("question", _FOLLOW_UP_UNSAFE)
def test_elided_question_is_handed_to_condense_on_a_follow_up(question):
    """С историей — не матчер: разрешение анафоры это работа condense."""
    assert corpus_scope.match_meta(question, has_history=True) is None


@pytest.mark.parametrize(
    "question",
    [
        # Якорь «база» — дополнение на месте, подставлять нечего.
        "Какие разделы есть в базе знаний?",
        "О чём эта база?",
        "Из каких разделов состоит база знаний?",
        "Перечисли разделы базы",
        "Сколько всего страниц в базе и как они распределены по разделам?",
        "О каких продуктах есть информация в базе знаний?",
        # Подлежащее — «ты»: у вопроса про сам ассистент нет второго чтения,
        # и типичный поток «привет» → «что ты умеешь?» обязан работать на 2-м ходу.
        "Кто ты?",
        "Что ты умеешь?",
        "Что ты знаешь?",
        "Какие темы ты покрываешь?",
        "Какая информация у тебя есть?",
        "Всегда ли ответ в Markdown с заголовками?",
        "Откуда ты берёшь ответы?",
    ],
)
def test_anchored_questions_survive_a_follow_up(question):
    """Сужение не должно съесть якорные формулировки: они верны на любом ходу."""
    assert corpus_scope.match_meta(question, has_history=True) is not None
    assert corpus_scope.match_meta(question, has_history=True) == corpus_scope.match_meta(
        question
    )


# --------------------------------------------------------------------------- #
# 2б. Поле охвата
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw", ["corpus", " CORPUS ", '"corpus"'])
def test_parse_scope_reads_corpus(raw):
    assert corpus_scope.parse_scope(raw) == "corpus"


@pytest.mark.parametrize("raw", [None, "", "document", "весь корпус", 5, {"a": 1}, True])
def test_parse_scope_defaults_to_document(raw):
    """Нет поля, мусор, вырезанная из промпта фраза — сегодняшнее поведение."""
    assert corpus_scope.parse_scope(raw) == corpus_scope.DEFAULT_SCOPE == "document"


# --------------------------------------------------------------------------- #
# 2в. Оговорка
# --------------------------------------------------------------------------- #


def _sources(*paths: str) -> list[dict]:
    return [
        {"n": i, "path": p, "title": f"Документ {i}"}
        for i, p in enumerate(paths, start=1)
    ]


#: «Структура базы известна, страниц-разделов в ней нет» — не то же самое, что
#: «структура неизвестна» (``None``). Разницу проверяет
#: `test_hedge_is_silent_when_the_shape_of_the_base_is_unknown`.
_NO_CONTAINERS: frozenset[str] = frozenset()


def test_hedge_fires_on_corpus_question_answered_from_one_document():
    text = corpus_scope.hedge("corpus", _sources("раздел/a.md", "раздел/a.md"), 127, _NO_CONTAINERS)

    assert text is not None
    assert "Документ 1" in text and "127" in text
    # Оговорка уточняет ответ, а не заменяет его: это одна фраза, а не отказ.
    assert "не нашлось" not in text


def test_hedge_fires_when_the_evidence_is_one_section():
    """D3: два соседних документа одного раздела — не охват, а та же концентрация.

    Почти-дубликаты по соседним страницам — измеренная популяция (7,3% чанков в
    кросс-файловых кластерах), и по путям они выглядели как ширина.
    """
    text = corpus_scope.hedge(
        "corpus", _sources("Продукты/Fincert/a.md", "Продукты/Fincert/b.md"), 127, _NO_CONTAINERS
    )

    assert text is not None
    assert "«Fincert»" in text and "2 документа" in text


def test_hedge_names_the_section_with_the_right_russian_plural():
    text = corpus_scope.hedge(
        "corpus",
        _sources(*[f"Продукты/Fincert/{i}.md" for i in range(5)]),
        127,
        _NO_CONTAINERS,
    )
    assert text is not None and "5 документов" in text


@pytest.mark.parametrize(
    "scope, sources, total, containers, why",
    [
        ("document", _sources("a/x.md"), 127, _NO_CONTAINERS, "вопрос про один документ"),
        (corpus_scope.DEFAULT_SCOPE, _sources("a/x.md"), 127, _NO_CONTAINERS, "поля scope не было"),
        ("corpus", _sources("a/x.md", "b/y.md"), 127, _NO_CONTAINERS, "источники из двух разделов"),
        ("corpus", [], 127, _NO_CONTAINERS, "источников нет вовсе"),
        ("corpus", _sources("a/x.md"), 1, _NO_CONTAINERS, "в базе один документ"),
        ("corpus", _sources("x.md", "y.md"), 127, _NO_CONTAINERS, "файлы в корне — не раздел"),
        (
            "corpus",
            _sources("Продукты/Fincert.md"),
            127,
            frozenset({"Продукты/Fincert.md"}),
            "единственный документ — сама страница-раздел, она и есть перечень",
        ),
    ],
)
def test_hedge_stays_silent(scope, sources, total, containers, why):
    assert corpus_scope.hedge(scope, sources, total, containers) is None, why


def test_hedge_is_silent_when_the_shape_of_the_base_is_unknown():
    """``containers=None`` — листинг недоступен, исключение проверить нечем.

    Молчание тут не осторожность вообще, а конкретный выбор: без листинга
    страницу-раздел от обычной страницы не отличить, а единственная популяция,
    до которой оговорка при этом доходит, — та, где она ЛОЖНА.
    """
    assert corpus_scope.hedge("corpus", _sources("a/x.md"), 127, None) is None


def test_hedge_omits_the_denominator_when_the_listing_is_unavailable():
    text = corpus_scope.hedge("corpus", _sources("a/x.md"), None, _NO_CONTAINERS)
    assert text is not None and "из None" not in text


# --------------------------------------------------------------------------- #
# ИЗМЕРЕНИЕ: ложные оговорки на 56 вопросах класса B
# --------------------------------------------------------------------------- #

_CONTROL = Path(__file__).resolve().parents[2] / "tools" / "eval" / "golden.control.json"


def _control_questions() -> list[dict]:
    if not _CONTROL.exists():  # пакет собран без набора eval
        pytest.skip(f"контрольная группа недоступна: {_CONTROL}")
    return json.loads(_CONTROL.read_text(encoding="utf-8"))["questions"]


def test_control_group_is_the_frozen_fifty_six():
    assert len(_control_questions()) == 56


def test_false_hedge_rate_on_the_control_group_is_zero():
    """Ни один из 56 отвечаемых сегодня вопросов не получает оговорку.

    Вердикты ФИКСИРОВАНЫ, потому что офлайн классификатора нет. Проверяются все
    вердикты, которые может выдать реальный ход, кроме одного:

    * ``document`` — правильный вердикт для вопроса про один документ;
    * отсутствующее поле ``scope`` — модель не вернула ключ;
    * сломанный/пустой ответ condense и выключенный condense — оба дают
      :data:`app.corpus_scope.DEFAULT_SCOPE`.

    Не проверяется (и офлайн непроверяемо): вердикт ``corpus`` на вопросе класса
    B. Это единственный способ получить здесь ложную оговорку, и он живёт в
    маршрутизации — мерить его можно только на живом стенде по полям
    ``scope``/``hedge`` в ``rag_log.jsonl``. Его цена измерена отдельно —
    см. `test_forced_corpus_verdict_exposure_on_the_control_group`.
    """
    questions = _control_questions()
    verdicts = [
        corpus_scope.parse_scope("document"),
        corpus_scope.parse_scope(None),
        corpus_scope.parse_scope("не знаю"),
    ]

    hedged = [
        (row["id"], scope)
        for row in questions
        for scope in verdicts
        # Класс B отвечается ИЗ ОДНОГО документа — то есть сходится ровно в тот
        # признак, на который смотрит оговорка. Единственное, что её удерживает,
        # это охват.
        if corpus_scope.hedge(scope, _sources(row["source_path"]), 127, _NO_CONTAINERS)
    ]

    assert hedged == [], f"ложная оговорка на {len(hedged)} парах (вопрос, вердикт)"


def test_the_control_measurement_is_not_vacuous():
    """Страховка от «нулевой доли» из-за того, что оговорка не работает вовсе.

    Тот же вопрос класса B с вердиктом ``corpus`` оговорку получает — значит
    ноль выше означает «охват удержал», а не «оговорка мертва».
    """
    row = _control_questions()[0]
    assert (
        corpus_scope.hedge("corpus", _sources(row["source_path"]), 127, _NO_CONTAINERS)
        is not None
    )


def test_forced_corpus_verdict_exposure_on_the_control_group():
    """Сколько из 56 получат ложную оговорку, ЕСЛИ классификатор ошибётся охватом.

    Ноль выше измеряет удержание охвата, а не разделяющую способность оговорки
    (это и был дефект измерения: 168 пар сводились к одному вердикту
    ``document``). Здесь охват ПРИНУДИТЕЛЬНО испорчен на всех 56 — это верхняя
    граница цены ошибки классификатора, и её надо знать числом, а не словами.

    Ответ: ВСЕ 56. Класс B по построению отвечается из одного документа, то есть
    сходится ровно в тот признак, на который смотрит оговорка, и удерживает её
    ровно одно — вердикт охвата. Знать это числом важнее, чем считать ноль выше
    доказательством безопасности: ноль держится на классификаторе, которого
    офлайн-стенд не видит.

    Исключение по страницам-разделам (`containers`) здесь не срабатывает и не
    должно: вопрос «перечисли все поля витрины X» отвечается обычной страницей,
    а не оглавлением раздела. Оно снимает другую популяцию — g500–g505.
    """
    questions = _control_questions()
    exposed = [
        row["id"]
        for row in questions
        if corpus_scope.hedge(
            "corpus", _sources(row["source_path"]), 127, _NO_CONTAINERS
        )
    ]

    assert len(exposed) == len(questions) == 56


# --------------------------------------------------------------------------- #
# ИЗМЕРЕНИЕ: популяция, на которой оговорка БЫЛА БЫ ложной (g500–g505)
# --------------------------------------------------------------------------- #

# Шесть отвечаемых вопросов охвата приёмочного набора. Каждый ПРАВИЛЬНО и
# ПОЛНОСТЬЮ отвечается ровно одной страницей — и каждая из этих страниц является
# страницей-разделом: у неё есть потомки в вольте. Числа сняты с эталонной
# выгрузки (127 страниц, `~/Downloads/confluence-dump.zip`).
#
# Это и есть единственная популяция, до которой оговорка вообще доходила:
# мотивирующий случай плана теперь ловит матчер, ловушки охвата упираются в
# отказ грейдера раньше, а сами эти шесть строк ИСКЛЮЧЕНЫ из контрольной группы
# 56 («строки категории corpus_scope — они и есть измеряемый класс»). То есть до
# этого теста ложные оговорки на них не мерило ничто.
_CONTAINER_ANSWERS = {
    "g500-corpus_scope": 48,  # Продукты/Описание витрин.md
    "g501-corpus_scope": 2,  # Продукты/Потоки наполнения витрин.md
    "g502-corpus_scope": 10,  # Продукты/Fincert.md
    "g503-corpus_scope": 7,  # Продукты/General.md
    "g504-corpus_scope": 3,  # Продукты/Data Lineage/OASIS UI.md
    "g505-corpus_scope": 3,  # Продукты/Data Lineage/Data Lineage REST APIs.md
}

_CORPUS_GOLDEN = (
    Path(__file__).resolve().parents[2] / "tools" / "eval" / "golden.corpus.jsonl"
)


def _container_answer_rows() -> list[dict]:
    if not _CORPUS_GOLDEN.exists():  # пакет собран без набора eval
        pytest.skip(f"набор вопросов охвата недоступен: {_CORPUS_GOLDEN}")
    rows = [
        json.loads(line)
        for line in _CORPUS_GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    found = [row for row in rows if row["id"] in _CONTAINER_ANSWERS]
    assert len(found) == len(_CONTAINER_ANSWERS), "набор переразмечен, а тест — нет"
    return found


def test_no_hedge_on_questions_a_section_page_answers_completely():
    """Принудительный вердикт ``corpus`` — и всё равно ни одной оговорки.

    Живой классификатор назовёт эти шесть корпусными с высокой вероятностью:
    формулировки буквально перечислительные («какие сервисы входят в продукт
    Fincert?»). Оговорка там была бы ЛОЖНОЙ — ответ полон.
    """
    rows = _container_answer_rows()
    containers = frozenset(row["source_path"] for row in rows)

    hedged = [
        row["id"]
        for row in rows
        if corpus_scope.hedge("corpus", _sources(row["source_path"]), 127, containers)
    ]

    assert hedged == [], f"ложная оговорка на {hedged}"


def test_the_container_exemption_is_what_holds_that_zero():
    """Без исключения по страницам-разделам оговорку получили бы все шесть."""
    rows = _container_answer_rows()
    hedged = [
        row["id"]
        for row in rows
        if corpus_scope.hedge("corpus", _sources(row["source_path"]), 127, _NO_CONTAINERS)
    ]
    assert len(hedged) == len(rows)


def test_hedge_still_fires_where_the_plan_says_it_should():
    """Мотивирующий случай плана: «О каких проектах знаешь?» → одна архивная страница.

    Страница-список личных проектов одного инженера, БЕЗ потомков — то есть не
    перечень раздела, а один документ из 127. Именно этот случай матчер ловит
    только в узкой формулировке; всё, что мимо неё, доезжает до ретрива, и вот
    там оговорка и нужна.
    """
    archive = (
        "Confluence/OASISEXT/OASIS External Home/Разработка управления "
        "моделирования и исследования данных/Архив/Проекты Ислама.md"
    )
    text = corpus_scope.hedge(
        "corpus",
        [{"n": 1, "path": archive, "title": "Проекты Ислама"}],
        127,
        frozenset(_CONTAINER_ANSWERS),  # что угодно, лишь бы этой страницы там не было
    )

    assert text is not None
    assert "Проекты Ислама" in text and "127" in text


def test_no_control_question_is_swallowed_by_the_meta_matcher():
    """Ложное срабатывание 2а хуже ложной оговорки: ответ подменяется деревом.

    Проверяется первый ход — там список формулировок шире; с историей он только
    сужается, так что ноль здесь мажорирует оба случая.
    """
    matched = [
        row["id"] for row in _control_questions() if corpus_scope.match_meta(row["question"])
    ]
    assert matched == []


_GOLDEN = [
    Path(__file__).resolve().parents[2] / "tools" / "eval" / name
    for name in ("golden.jsonl", "golden.corpus.jsonl")
]


def test_matcher_hits_exactly_the_meta_rows_of_the_whole_golden_set():
    """Второй стоячий инвариант: на всём наборе ловятся только строки `meta`.

    Матчер подменяет ответ материалом и обходит грейдер, поэтому его точность —
    не метрика удобства, а граница безопасности. Расширение формулировок (D5) и
    новая семья вопросов про сам ассистент (D4) обязаны эту границу удержать:
    любое новое попадание — либо строка с ``expected_outcome == "meta"``, либо
    регрессия.
    """
    if not all(path.exists() for path in _GOLDEN):  # пакет собран без набора eval
        pytest.skip("золотой набор недоступен")
    rows = [
        json.loads(line)
        for path in _GOLDEN
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Размер набора не фиксируем числом: он растёт (в него добавляют пары под
    # новые классы вопросов), и жёсткий счётчик ломался бы на каждом пополнении,
    # ничего при этом не проверяя. Инвариант ниже — про ТОЧНОСТЬ матчера, а не
    # про размер; нижняя граница нужна лишь чтобы заметить пустой/битый файл.
    assert len(rows) > 200

    hits = {
        row["id"]: corpus_scope.match_meta(row["question"])
        for row in rows
        if corpus_scope.match_meta(row["question"])
    }
    outcomes = {row["id"]: row.get("expected_outcome") for row in rows}

    assert set(hits) == {
        "x23-meta",  # D4: вопрос о поведении самого ассистента
        "g506-corpus_scope",
        "g507-corpus_scope",
        "g508-corpus_scope",
        "g513-corpus_scope",
    }
    assert all(outcomes[rid] == "meta" for rid in hits), hits

    # D-A: сужение по истории не должно стоить набору ни одной строки — все пять
    # якорные, дополнение у них на месте. Ноль на контроле тем более сохраняется:
    # с историей список формулировок только сужается.
    with_history = {
        row["id"]: corpus_scope.match_meta(row["question"], has_history=True)
        for row in rows
        if corpus_scope.match_meta(row["question"], has_history=True)
    }
    assert with_history == hits


# --------------------------------------------------------------------------- #
# Мета-ветка в сборке контекста
# --------------------------------------------------------------------------- #


def _corpus() -> list[str]:
    paths = [f"Продукты/{s}/Стр {i}.md" for i, s in enumerate(["Fincert"] * 12)]
    paths += [f"Продукты/АРМ DS/Стр {i}.md" for i in range(9)]
    paths += [f"Архив/Проект {i}.md" for i in range(4)]
    paths += [f"База знаний/Инструкция {i}.md" for i in range(3)]
    return paths


def _install_listing(monkeypatch, paths: list[str] | None) -> None:
    """Оба шва структуры: листинг вольта и каталог.

    Каталог тут нужен ровно за одним полем — `document_extensions`: без него
    `corpus_map` не знает, что считать документом, и честно не строит блок
    вовсе (см. `tests/test_corpus_map.py`). Дерево разделов из шага 3 из этой
    заглушки не соберётся — в ней нет `documents`, — и это специально: мета-ветка
    обязана работать и на голом листинге, флаг `corpus_tree_enabled` выключен.
    """

    async def fake_list_files(cv=None, recursive=True, timeout=None):
        if paths is None:
            raise cognivault.CogniVaultError("list files failed (503)", 503, "")
        return paths

    async def fake_catalog(cv=None):
        return {
            "status": "summaries_pending",
            "summaries_enabled": True,
            "reason": None,
            "documents": [],
            "total": 0,
            "offset": 0,
            "documents_with_summary": 0,
            "document_extensions": ["md", "pdf", "canvas", "excalidraw", "csv"],
        }

    monkeypatch.setattr(corpus_map, "files", _REAL_FILES)
    monkeypatch.setattr(cognivault, "list_files", fake_list_files)
    monkeypatch.setattr(catalog, "payload", fake_catalog)
    corpus_map.reset_cache()
    catalog.reset_cache()


def _install_retrieval(monkeypatch) -> list[str]:
    """Поиск и обе скрытые модели; возвращает журнал вызовов."""
    seen: list[str] = []

    async def fake_hybrid(query, limit, cv=None, **kwargs):
        seen.append("search")
        return {
            "results": [
                {
                    "path": "Архив/Проект 1.md",
                    "title": "Проекты Ислама",
                    "section_path": "",
                    "score": 1.0,
                    "text": "нумерованный список личных проектов",
                    "chunk_index": 0,
                    "rank": 1,
                }
            ]
        }

    async def fake_content(path, cv=None):
        raise RuntimeError("content unavailable")

    async def fake_complete_json(messages, gcfg, **kwargs):
        prompt = messages[-1]["content"]
        seen.append("condense" if "Определи тип реплики" in prompt else "grade")
        if "Определи тип реплики" in prompt:
            return {"intent": "kb_question", "standalone_question": None}
        return {"grades": [{"id": 1, "score": 5}]}

    monkeypatch.setattr(rag.cognivault, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(rag.cognivault, "content", fake_content)
    monkeypatch.setattr(
        rag_pipeline.llm, "complete_json", fake_complete_json, raising=False
    )
    return seen


def _build(query: str, messages: list[dict] | None = None, **rcfg) -> rag.RagContext:
    return asyncio.run(
        rag.build_rag_context(
            query,
            {"mode": "auto", "max_expanded_files": 0, **rcfg},
            None,
            {},
            messages,
        )
    )


#: История из воспроизведения дефекта: предыдущий ход назвал предмет, и «какие
#: разделы?» на следующем ходу означает разделы ЭТОГО предмета.
_FINCERT_HISTORY = [
    {"role": "user", "content": "Расскажи про продукт Fincert"},
    {"role": "assistant", "content": "Fincert — это продукт по работе с фидами."},
]


def test_meta_question_is_answered_from_the_section_tree(monkeypatch):
    seen = _install_retrieval(monkeypatch)
    _install_listing(monkeypatch, _corpus())

    ctx = _build("О чём эта база?")

    assert ctx.intent == "meta"
    assert ctx.sources == []
    # Ни поиска, ни condense, ни грейдера: ветка детерминированная.
    assert seen == []
    # Ответ строится из настоящего дерева, а не из выдуманного текста.
    content = ctx.user_message["content"]
    assert "Структура базы знаний" in content
    assert "Всего документов в базе: 28." in content
    assert "- Продукты — 21" in content and "Fincert: 12" in content
    assert content.endswith("Вопрос: О чём эта база?")
    # Системный турн — свой: `NO_RAG_SYSTEM_PROMPT` запрещает утверждать то,
    # чего нет в истории диалога, то есть запрещает и рассказ об охвате.
    assert ctx.system_message["content"] == rag.META_SYSTEM_PROMPT
    assert ctx.system_message["content"] != rag.NO_RAG_SYSTEM_PROMPT


@pytest.mark.parametrize("question", ["Какие разделы?", "Какие темы?"])
def test_follow_up_about_a_document_is_not_hijacked_by_the_matcher(
    monkeypatch, question
):
    """D-A целиком, на сборке контекста: тот же ход, но с историей.

    Раньше: intent=meta, ноль вызовов, ноль источников, дерево всей базы — на
    вопрос про разделы ПРОДУКТА. Теперь ход идёт обычным путём: condense видит
    историю и переписывает вопрос, поиск и грейдер работают, ответ опирается на
    источники.
    """
    seen = _install_retrieval(monkeypatch)
    _install_listing(monkeypatch, _corpus())

    ctx = _build(question, messages=[*_FINCERT_HISTORY, {"role": "user", "content": question}])

    assert ctx.intent == "kb_question"
    assert seen == ["condense", "search", "grade"]
    assert ctx.sources and "Источники:" in ctx.user_message["content"]


@pytest.mark.parametrize("question", ["Какие разделы?", "Какие темы?"])
def test_the_same_question_first_turn_is_still_answered_from_the_tree(
    monkeypatch, question
):
    """Обратная сторона сужения: на первом ходу поведение не изменилось."""
    seen = _install_retrieval(monkeypatch)
    _install_listing(monkeypatch, _corpus())

    ctx = _build(question)

    assert ctx.intent == "meta" and seen == [] and ctx.sources == []
    assert "Структура базы знаний" in ctx.user_message["content"]


def test_anchored_meta_question_still_works_as_a_follow_up(monkeypatch):
    """«Какие разделы есть в базе знаний?» после разговора о продукте — всё ещё мета.

    Сужение бьёт ровно по формулировкам с опущенным дополнением, а не по
    вопросам про базу вообще.
    """
    seen = _install_retrieval(monkeypatch)
    _install_listing(monkeypatch, _corpus())

    question = "Какие разделы есть в базе знаний?"
    ctx = _build(question, messages=[*_FINCERT_HISTORY, {"role": "user", "content": question}])

    assert ctx.intent == "meta" and seen == []
    assert ctx.system_message["content"] == rag.META_SYSTEM_PROMPT


def test_meta_branch_falls_through_when_the_listing_is_unavailable(monkeypatch):
    """Нет дерева — нет и ответа по дереву: ход идёт обычным путём, до грейдера.

    Только для семьи ``corpus``: у вопроса «о чём эта база?» без структуры нет
    материала вовсе, и выдумать её хуже, чем отказать. У семьи ``assistant``
    материал свой — см. `test_question_about_the_assistant_is_answered_…`.
    """
    seen = _install_retrieval(monkeypatch)
    _install_listing(monkeypatch, None)

    ctx = _build("О чём эта база?")

    assert ctx.intent == "kb_question"
    assert "search" in seen and "grade" in seen
    assert ctx.sources and ctx.system_message["content"] == rag.SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# Семья `assistant`: вопрос про самого ассистента (D4, строка `x23-meta`)
# --------------------------------------------------------------------------- #


def test_question_about_the_assistant_is_answered_from_its_own_rules(monkeypatch):
    """«Всегда ли ответ в Markdown с заголовками?» — вопрос о поведении сервиса.

    Отвечающего документа в корпусе нет и быть не может, поэтому до этой ветки
    строка приёмочного набора `x23-meta` доезжала до грейдера и получала отказ
    «в доступных мне документах ответа не нашлось» — ровно то поведение, которое
    заказчик просил починить, переклассифицировав вопрос в отвечаемый.
    """
    seen = _install_retrieval(monkeypatch)
    _install_listing(monkeypatch, _corpus())

    ctx = _build("Всегда ли ответ в Markdown с заголовками?")

    assert ctx.intent == "meta"
    assert seen == [] and ctx.sources == []
    content = ctx.user_message["content"]
    # Материал — правила, а не дерево папок.
    assert "Как ты работаешь" in content
    assert "Markdown" in content and "таблицы интерфейс НЕ разбирает" in content
    # Дерево тоже на месте: «что ты знаешь» — это и про себя, и про базу.
    assert "Структура базы знаний" in content
    assert ctx.system_message["content"] == rag.META_SELF_SYSTEM_PROMPT


def test_the_two_meta_families_get_different_material(monkeypatch):
    """Различение семей несущее, а не декоративное: разный материал, разный промпт."""
    _install_retrieval(monkeypatch)
    _install_listing(monkeypatch, _corpus())

    about_base = _build("О чём эта база?")
    about_self = _build("Кто ты?")

    assert about_base.system_message["content"] == rag.META_SYSTEM_PROMPT
    assert about_self.system_message["content"] == rag.META_SELF_SYSTEM_PROMPT
    assert "Как ты работаешь" not in about_base.user_message["content"]
    assert "Как ты работаешь" in about_self.user_message["content"]


def test_meta_prompts_request_compact_deduplicated_scope_answers():
    assert "Не повторяй одну сущность" in rag.META_SYSTEM_PROMPT
    assert "компактный список уникальных" in rag.META_SYSTEM_PROMPT
    assert "Не пересказывай\n  всё дерево" in rag.META_SELF_SYSTEM_PROMPT


def test_question_about_the_assistant_survives_an_unavailable_listing(monkeypatch):
    """Правила — код, они не могут пропасть вместе с листингом вольта.

    Семья ``assistant`` поэтому НЕ проваливается в обычный путь: там вопрос про
    сам сервис снова уехал бы в ретрив, где отвечающего документа нет.
    """
    seen = _install_retrieval(monkeypatch)
    _install_listing(monkeypatch, None)

    ctx = _build("Кто ты?")

    assert ctx.intent == "meta"
    assert seen == []
    content = ctx.user_message["content"]
    assert "Как ты работаешь" in content
    assert "Структура базы знаний" not in content


def test_the_effective_answering_rules_are_quoted_not_applied(monkeypatch):
    """Сохранённый `prompts.system` — это ОТВЕТ на «как ты отвечаешь», а не приказ.

    Он попадает в материал (иначе рассказ о себе описывал бы чужую конфигурацию),
    но системным турном остаётся `META_SELF_SYSTEM_PROMPT` — иначе правило
    «отвечай только по блоку Источники» превратило бы этот ход в отказ.
    """
    _install_retrieval(monkeypatch)
    _install_listing(monkeypatch, _corpus())

    ctx = asyncio.run(
        rag.build_rag_context(
            "Кто ты?",
            {"mode": "auto"},
            None,
            {},
            None,
            prompts={"system": "мой собственный системный промпт"},
        )
    )

    assert ctx.system_message["content"] == rag.META_SELF_SYSTEM_PROMPT
    assert "мой собственный системный промпт" in ctx.user_message["content"]


def test_ordinary_question_never_reaches_the_meta_branch(monkeypatch):
    seen = _install_retrieval(monkeypatch)
    _install_listing(monkeypatch, _corpus())

    ctx = _build("Что ты знаешь про PSI?")

    assert ctx.intent == "kb_question"
    assert "search" in seen
    assert "Источники:" in ctx.user_message["content"]


def test_user_system_prompt_does_not_apply_to_the_meta_turn(monkeypatch):
    """Сохранённый `prompts.system` — правила ответа по «Источникам», которых тут нет."""
    _install_retrieval(monkeypatch)
    _install_listing(monkeypatch, _corpus())

    ctx = asyncio.run(
        rag.build_rag_context(
            "О чём эта база?",
            {"mode": "auto"},
            None,
            {},
            None,
            prompts={"system": "мой собственный системный промпт"},
        )
    )

    assert ctx.system_message["content"] == rag.META_SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# Оговорка в маршруте
# --------------------------------------------------------------------------- #


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        event = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if event is not None:
            out.append((event, data or {}))
    return out


def _install_chat(monkeypatch, paths, ctx, answer="ответ модели"):
    async def fake_build_rag_context(query, *args, **kwargs):
        return ctx

    async def fake_stream_chat(messages, gcfg):
        yield answer

    monkeypatch.setattr(chat_routes, "resolve_paths", lambda request: paths)
    monkeypatch.setattr(chat_routes.rag, "build_rag_context", fake_build_rag_context)
    monkeypatch.setattr(chat_routes.llm, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(chat_routes.llm, "files_present", lambda gcfg: None)


def _rag_ctx(hedge: str | None) -> rag.RagContext:
    return rag.RagContext(
        system_message={"role": "system", "content": "правила"},
        user_message={"role": "user", "content": "Источники:\n\nВопрос: вопрос"},
        sources=[{"n": 1, "title": "Док", "path": "a.md", "section_path": "", "depth": "chunk", "score": 1.0, "grade": 5}],
        context_chars=10,
        intent="kb_question",
        standalone_question="вопрос",
        scope="corpus" if hedge else "document",
        hedge=hedge,
    )


HEDGE = "Оговорка: вопрос охватывает базу целиком, а все найденные фрагменты — из одного документа."


def test_hedge_is_appended_after_the_answer(tmp_path, monkeypatch):
    """Оговорка идёт ПОСЛЕ ответа и отдельным кадром — она уточняет, а не заменяет."""
    paths = AppPaths(root=tmp_path / "ui")
    _install_chat(monkeypatch, paths, _rag_ctx(HEDGE), answer="краткий ответ")

    with TestClient(create_app()) as client:
        resp = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "какие продукты"}], "rag": True},
        )

    assert resp.status_code == 200
    tokens = [data["text"] for name, data in _parse_sse(resp.text) if name == "token"]
    assert tokens[0] == "краткий ответ"
    assert tokens[-1].strip() == HEDGE
    # И в сохранённую историю, и в лог качества — иначе долю оговорок не измерить.
    from app.history import load_chat

    saved = load_chat(next(iter(paths.history_dir.glob("*.json"))).stem, paths)
    assert saved["messages"][-1]["content"].endswith(HEDGE)


def test_no_hedge_no_extra_frame(tmp_path, monkeypatch):
    paths = AppPaths(root=tmp_path / "ui")
    _install_chat(monkeypatch, paths, _rag_ctx(None), answer="краткий ответ")

    with TestClient(create_app()) as client:
        resp = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "поля витрины"}], "rag": True},
        )

    tokens = [data["text"] for name, data in _parse_sse(resp.text) if name == "token"]
    assert tokens == ["краткий ответ"]
