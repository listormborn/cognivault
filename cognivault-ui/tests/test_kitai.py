"""Собственный клиент платформы KitAI: POST → polling → commit.

Офлайн: транспорт подменяется ``httpx.MockTransport``, сон — заглушкой, поэтому
цикл опроса прогоняется мгновенно и без сети. Проверяется ровно то, что мы
воспроизводим руками вместо вендорского SDK: форма запроса, разбор результата,
коммит, и все три способа закончить плохо.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import kitai  # noqa: E402
from app.llm_errors import (  # noqa: E402
    GigaChatHTTP,
    KitaiPollingTimeout,
    KitaiQueryFailed,
)

HOST = "https://kitai.test"


def _cfg(**over) -> kitai.KitaiConfig:
    base = {
        "kitai_host": HOST,
        "kitai_model": "glm-5.2",
        "kitai_system_name": "csp_lab",
        "kitai_module_name": "csp_lab_antifraud_edge",
        "temperature": 0.05,
        "max_tokens": 1024,
        "kitai_poll_initial_delay": 0.0,
        "kitai_poll_delay": 0.0,
        "kitai_poll_timeout": 30.0,
    }
    base.update(over)
    return kitai.KitaiConfig.from_dict(base)


def _finished(content: str = "ответ", finish_reason: str = "stop") -> dict:
    return {
        "description": None,
        "data": {
            "query_id": "00000000-0000-0000-0000-000000000000",
            "query_status": "finished",
            "is_final": True,
            "response": {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                },
            },
        },
    }


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def instant(_seconds):
        return None

    monkeypatch.setattr(kitai, "_sleep", instant)


def _recorder(*result_bodies: dict):
    """Транспорт, отдающий переданные тела на последовательные GET /result."""
    seen: list[httpx.Request] = []
    remaining = list(result_bodies)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/query/model"):
            return httpx.Response(200, json={"description": None, "data": None})
        if request.url.path.endswith("/result"):
            body = remaining.pop(0) if len(remaining) > 1 else remaining[0]
            return httpx.Response(200, json=body)
        if request.url.path.endswith("/commit"):
            return httpx.Response(200, json={"description": None, "data": True})
        raise AssertionError(f"неожиданный путь: {request.url.path}")

    return httpx.MockTransport(handler), seen


def _run(cfg, transport, messages=None):
    return asyncio.run(
        kitai._run_query(
            messages or [{"role": "user", "content": "вопрос"}],
            cfg,
            transport=transport,
        )
    )


# --------------------------------------------------------------------------- #
# Счастливый путь
# --------------------------------------------------------------------------- #


def test_posts_polls_and_commits():
    transport, seen = _recorder(_finished("готовый ответ"))

    content, finish_reason = _run(_cfg(), transport)

    assert (content, finish_reason) == ("готовый ответ", "stop")
    methods = [(r.method, r.url.path.split("/")[-1]) for r in seen]
    assert methods == [("POST", "model"), ("GET", "result"), ("PUT", "commit")]


def test_request_body_matches_the_generated_dto():
    """Поля уезжают в snake_case — сгенерированные DTO не объявляют алиасов."""
    transport, seen = _recorder(_finished())

    _run(_cfg(), transport, [{"role": "system", "content": "правила"}])

    body = json.loads(seen[0].content)
    assert body["model_name"] == "glm-5.2"
    assert body["messages"] == [{"role": "system", "content": "правила"}]
    assert body["temperature"] == 0.05
    assert body["max_tokens"] == 1024
    assert body["profanity_check"] is False
    # query_id генерим мы: он же адресует result и commit.
    assert body["query_id"] in seen[1].url.path


def test_identification_headers_travel_with_every_call():
    transport, seen = _recorder(_finished())

    _run(_cfg(), transport)

    for request in seen:
        assert request.headers["x-identification-system"] == "csp_lab"
        assert request.headers["x-identification-module"] == "csp_lab_antifraud_edge"


def test_module_header_is_omitted_when_not_configured():
    transport, seen = _recorder(_finished())

    _run(_cfg(kitai_module_name=""), transport)

    assert "x-identification-module" not in seen[0].headers


def test_polls_until_finished():
    running = {"description": None, "data": {"query_status": "running", "is_final": False}}
    transport, seen = _recorder(running, running, _finished("наконец"))

    content, _ = _run(_cfg(), transport)

    assert content == "наконец"
    assert sum(1 for r in seen if r.url.path.endswith("/result")) == 3


# --------------------------------------------------------------------------- #
# Плохие концовки
# --------------------------------------------------------------------------- #


def test_timeout_is_its_own_error_not_a_transport_failure():
    """Запрос ПРИНЯТ и, возможно, ещё считается — это не обрыв связи.

    Оператор должен прочитать «поднять таймаут / модель медленная», а не
    «сеть сломана», поэтому у случая свой класс и свой код.
    """
    running = {"description": None, "data": {"query_status": "running", "is_final": False}}
    transport, _ = _recorder(running)

    with pytest.raises(KitaiPollingTimeout) as exc:
        _run(_cfg(kitai_poll_timeout=0.0), transport)

    assert exc.value.code == "KITAI_TIMEOUT"


def test_final_non_finished_status_fails_fast():
    """`is_final` при статусе не «finished» — платформа сдалась, ждать нечего."""
    failed = {
        "description": None,
        "data": {
            "query_status": "failed",
            "is_final": True,
            "error": {"status": 500, "message": "модель недоступна"},
        },
    }
    transport, _ = _recorder(failed)

    with pytest.raises(KitaiQueryFailed) as exc:
        _run(_cfg(), transport)

    assert exc.value.code == "KITAI_QUERY_FAILED"
    assert "модель недоступна" in (exc.value.detail or "")


def test_http_error_on_enqueue_is_typed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    with pytest.raises(GigaChatHTTP):
        _run(_cfg(), httpx.MockTransport(handler))


def test_finished_without_choices_is_an_error_not_an_empty_answer():
    """Пустой ответ молча выглядел бы как «модель ничего не нашла»."""
    empty = {
        "description": None,
        "data": {"query_status": "finished", "is_final": True, "response": {"choices": []}},
    }
    transport, _ = _recorder(empty)

    with pytest.raises(KitaiQueryFailed) as exc:
        _run(_cfg(), transport)

    assert exc.value.code == "KITAI_EMPTY_RESULT"


def test_failed_commit_does_not_lose_the_answer():
    """Коммит — бухгалтерия; ответ уже получен и должен дойти до пользователя."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/commit"):
            return httpx.Response(500, text="nope")
        if request.url.path.endswith("/result"):
            return httpx.Response(200, json=_finished("ответ несмотря ни на что"))
        return httpx.Response(200, json={"data": None})

    content, _ = _run(_cfg(), httpx.MockTransport(handler))

    assert content == "ответ несмотря ни на что"


# --------------------------------------------------------------------------- #
# Обёртки, которые видит остальной код
# --------------------------------------------------------------------------- #


def test_stream_chat_yields_one_chunk_and_records_finish_reason():
    """Стриминга у платформы нет — притворяться не пытаемся."""
    transport, _ = _recorder(_finished("целиком", finish_reason="length"))
    cfg = _cfg()

    async def collect():
        chunks = []
        stream = kitai.stream_chat(
            [{"role": "user", "content": "?"}], cfg, transport=transport
        )
        async for chunk in stream:
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())

    assert chunks == ["целиком"]
    assert cfg.last_finish_reason == "length"


def test_stream_chat_retries_one_final_error_without_detail():
    transient = {
        "description": None,
        "data": {"query_status": "error", "is_final": True},
    }
    transport, seen = _recorder(transient, _finished("ответ после повтора"))
    cfg = _cfg()

    async def collect():
        return [
            chunk
            async for chunk in kitai.stream_chat(
                [{"role": "user", "content": "?"}], cfg, transport=transport
            )
        ]

    assert asyncio.run(collect()) == ["ответ после повтора"]
    assert sum(1 for request in seen if request.url.path.endswith("/query/model")) == 2


def test_stream_chat_does_not_retry_permanent_model_error():
    permanent = {
        "description": None,
        "data": {
            "query_status": "error",
            "is_final": True,
            "error": {"status": 400, "message": "unknown model"},
            "response_code": 400,
        },
    }
    transport, seen = _recorder(permanent)

    async def collect():
        return [
            chunk
            async for chunk in kitai.stream_chat(
                [{"role": "user", "content": "?"}], _cfg(), transport=transport
            )
        ]

    with pytest.raises(KitaiQueryFailed):
        asyncio.run(collect())
    assert sum(1 for request in seen if request.url.path.endswith("/query/model")) == 1


def test_complete_json_parses_the_models_object():
    payload = '{"intent": "kb_question", "answer_shape": "list"}'
    transport, _ = _recorder(_finished(payload))

    parsed = asyncio.run(
        kitai.complete_json(
            [{"role": "user", "content": "?"}], _cfg(), transport=transport
        )
    )

    assert parsed == {"intent": "kb_question", "answer_shape": "list"}


def test_complete_json_timeout_shortens_the_polling_budget():
    """У скрытых вызовов свой поводок — 240 с по умолчанию их бы пережил."""
    running = {"description": None, "data": {"query_status": "running", "is_final": False}}
    transport, _ = _recorder(running)

    with pytest.raises(KitaiPollingTimeout):
        asyncio.run(
            kitai.complete_json(
                [{"role": "user", "content": "?"}],
                _cfg(kitai_poll_timeout=999.0),
                timeout=0.0,
                transport=transport,
            )
        )


def test_missing_host_fails_before_any_request():
    with pytest.raises(Exception) as exc:
        _run(_cfg(kitai_host=""), httpx.MockTransport(lambda r: httpx.Response(200)))

    assert getattr(exc.value, "code", "") == "KITAI_NOT_CONFIGURED"


# --------------------------------------------------------------------------- #
# Выбор провайдера (фасад app.llm)
# --------------------------------------------------------------------------- #


def test_list_models_maps_the_catalogue(monkeypatch):
    """`GET /api/v1/meta/model` → пары (имя для запроса, подпись для человека)."""
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["sys"] = request.headers.get("x-identification-system")
        return httpx.Response(200, json=[
            {"id": 1, "model_name": "glm-5.2", "display_name": "GLM 5.2", "version": "1.0"},
            {"id": 2, "model_name": "GigaChat-3-Ultra", "display_name": None},
            {"id": 3, "model_name": ""},  # без имени — отбрасываем
            "мусор",
        ])

    out = asyncio.run(kitai.list_models(_cfg(), transport=httpx.MockTransport(handler)))

    assert seen["path"] == "/api/v1/meta/model"
    assert seen["sys"] == "csp_lab"
    assert out == [
        {"name": "glm-5.2", "label": "GLM 5.2 (1.0)"},
        {"name": "GigaChat-3-Ultra", "label": "GigaChat-3-Ultra"},
    ]


def test_list_models_tolerates_the_wrapped_shape():
    """Остальной API заворачивает всё в {description, data} — примем и так."""
    body = {"description": None, "data": [{"id": 1, "model_name": "glm-5.2"}]}
    out = asyncio.run(kitai.list_models(
        _cfg(), transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))))
    assert out == [{"name": "glm-5.2", "label": "glm-5.2"}]


def test_list_models_raises_so_the_form_can_degrade():
    """Ошибку не глотаем: пустой список читался бы как «моделей нет»."""
    with pytest.raises(GigaChatHTTP):
        asyncio.run(kitai.list_models(
            _cfg(), transport=httpx.MockTransport(lambda r: httpx.Response(503, text="down"))))


def test_gigachat_also_publishes_a_catalogue():
    """У GigaChat есть OpenAI-совместимый `{base_url}/models` — он и спрашивается.

    Сначала я решил, что каталога у него нет, и захардкодил `None`. Проверка по
    исходникам официального SDK (`gigachat/api/models.py`) это опровергла.
    """
    from app import llm
    from app.gigachat import GigaConfig

    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={
            "object": "list",
            "data": [{"id": "GigaChat-3-Ultra-preview", "object": "model",
                      "owned_by": "salutedevices"}],
        })

    from app import gigachat as gc
    out = asyncio.run(gc.list_models(
        GigaConfig.from_dict({"base_url": "https://giga.test/v1"}),
        transport=httpx.MockTransport(handler),
    ))

    assert seen["url"] == "https://giga.test/v1/models"
    assert out == [{"name": "GigaChat-3-Ultra-preview",
                    "label": "GigaChat-3-Ultra-preview"}]


def test_gigachat_listing_failure_propagates():
    """Пустой список означал бы «моделей нет» — это другое утверждение."""
    from app import gigachat as gc
    from app.gigachat import GigaConfig

    with pytest.raises(GigaChatHTTP):
        asyncio.run(gc.list_models(
            GigaConfig.from_dict({"base_url": "https://giga.test/v1"}),
            transport=httpx.MockTransport(lambda r: httpx.Response(404, text="nope")),
        ))


def test_provider_dispatch_picks_the_config_type():
    from app import llm
    from app.gigachat import GigaConfig

    assert isinstance(llm.config_for({"provider": "kitai"}), kitai.KitaiConfig)
    assert isinstance(llm.config_for({"provider": "gigachat"}), GigaConfig)
    # Ключа нет — остаёмся на прежнем транспорте, а не падаем.
    assert isinstance(llm.config_for({}), GigaConfig)


def test_unknown_provider_falls_back_instead_of_breaking_chat(caplog):
    from app import llm

    with caplog.at_level("WARNING"):
        assert llm.provider_of({"provider": "gigacaht"}) == llm.DEFAULT_PROVIDER
    assert "gigacaht" in caplog.text


def test_only_gigachat_advertises_streaming():
    from app import llm
    from app.gigachat import GigaConfig

    assert llm.supports_streaming(GigaConfig.from_dict({})) is True
    assert llm.supports_streaming(_cfg()) is False


def test_kitai_model_falls_back_to_the_shared_model_key():
    """Пустой `kitai_model` не должен обнулять имя модели."""
    cfg = kitai.KitaiConfig.from_dict(
        {"kitai_host": HOST, "kitai_model": "", "model": "GigaChat-3-Ultra"}
    )
    assert cfg.model == "GigaChat-3-Ultra"


def test_failure_detail_gathers_every_field_the_platform_may_fill():
    """Причина приезжает в одном из трёх полей — читаем все."""
    detail = kitai._failure_detail({
        "error": {"status": 500, "message": "model not found: glm-5.2"},
        "response_code": 404,
        "response_body": '{"error":"unknown model"}',
    })
    assert "model not found: glm-5.2" in detail
    assert "error.status=500" in detail
    assert "response_code=404" in detail
    assert "unknown model" in detail


def test_error_str_carries_the_detail_to_the_log():
    """Регрессия: логи писали `%s` от исключения и теряли причину целиком."""
    from app.llm_errors import GigaChatError

    exc = GigaChatError("KITAI_QUERY_FAILED", "статус «error»", "model not found")
    assert str(exc) == "статус «error» — model not found"
    assert str(GigaChatError("X", "без детали")) == "без детали"


def test_failed_query_names_the_model_and_logs_the_query_id(caplog):
    """По логу должно быть видно И модель, И query_id для обращения в поддержку."""
    failed = {
        "description": None,
        "data": {
            "query_status": "error",
            "is_final": True,
            "error": {"status": 400, "message": "unknown model"},
        },
    }
    transport, _ = _recorder(failed)

    with caplog.at_level("WARNING"), pytest.raises(KitaiQueryFailed) as exc:
        _run(_cfg(), transport)

    assert "glm-5.2" in exc.value.message
    assert "unknown model" in (exc.value.detail or "")
    assert "unknown model" in caplog.text
    assert "glm-5.2" in caplog.text


def test_kitai_uses_its_own_certificate_when_the_files_exist(tmp_path):
    """KitAI — другой контур; общий с GigaChat сертификат это частный случай."""
    own_crt = tmp_path / "k.crt"
    own_key = tmp_path / "k.key"
    own_crt.write_text("x")
    own_key.write_text("x")

    cfg = kitai.KitaiConfig.from_dict({
        "cert_path": "/certs/client_crt.crt",
        "key_path": "/certs/client_key.key",
        "key_passphrase": "shared",
        "kitai_host": "https://k",
        "kitai_cert_path": str(own_crt),
        "kitai_key_path": str(own_key),
        "kitai_key_passphrase": "own",
    })

    assert (cfg.cert_path, cfg.key_path, cfg.key_passphrase) == (
        str(own_crt), str(own_key), "own")


def test_no_kitai_certificate_means_the_shared_one():
    """Установка с одной парой не настраивает ничего."""
    cfg = kitai.KitaiConfig.from_dict({
        "cert_path": "/certs/client_crt.crt",
        "key_path": "/certs/client_key.key",
        "key_passphrase": "shared",
        "kitai_host": "https://k",
    })
    assert (cfg.cert_path, cfg.key_path, cfg.key_passphrase) == (
        "/certs/client_crt.crt", "/certs/client_key.key", "shared")


def test_configured_but_missing_certificate_degrades_instead_of_killing_chat(caplog):
    """Путь задан, а секрета ещё нет — это ошибка ПОРЯДКА выкатки, не повод лечь.

    Том смонтирован `optional: true`, поэтому каталог окажется пустым. Жёсткий
    отказ здесь уронил бы весь чат сообщением «сертификат не найден»; вместо
    этого идём общей парой и громко пишем об этом — иначе оператор решит, что
    свой сертификат используется, хотя это не так.
    """
    with caplog.at_level("WARNING"):
        cfg = kitai.KitaiConfig.from_dict({
            "cert_path": "/certs/client_crt.crt",
            "key_path": "/certs/client_key.key",
            "kitai_host": "https://k",
            "kitai_cert_path": "/certs/kitai/client_crt.crt",
            "kitai_key_path": "/certs/kitai/client_key.key",
        })

    assert cfg.cert_path == "/certs/client_crt.crt"
    assert "/certs/kitai/client_crt.crt" in caplog.text
    assert "cognivault-kitai-certs" in caplog.text


def test_kitai_certificate_paths_stay_admin_only():
    """Путь к сертификату пользователю не отдаём — это учётные данные."""
    from app import settings

    for key in ("gigachat.kitai_cert_path", "gigachat.kitai_key_path",
                "gigachat.kitai_key_passphrase"):
        assert key in settings.ADMIN_LOCKED_KEYS
        assert key not in settings.USER_EDITABLE_KEYS


def test_catalog_403_is_a_permission_state_not_a_fault():
    """403 на каталоге при рабочих запросах — это права, а не сломанное подключение.

    Наблюдалось на IFT: `POST /query/model` тем же сертификатом принимается, а
    `/api/v1/meta/model` отвечает «Access denied for the certificate». Отдельный
    класс нужен, чтобы это не логировалось как отказ связи на каждой загрузке
    страницы настроек.
    """
    from app.llm_errors import KitaiCatalogForbidden

    def handler(request):
        return httpx.Response(403, json={"description": "Access denied for the certificate"})

    with pytest.raises(KitaiCatalogForbidden) as exc:
        asyncio.run(kitai.list_models(_cfg(), transport=httpx.MockTransport(handler)))

    assert exc.value.code == "KITAI_CATALOG_FORBIDDEN"
    assert "Access denied" in (exc.value.detail or "")


def test_upstream_503_reaches_the_operator_verbatim():
    """503 от апстрима KitAI: причина — в response_body, и её нельзя терять."""
    failed = {
        "description": None,
        "data": {
            "query_status": "error",
            "is_final": True,
            "response_code": 503,
            "response_body": (
                "upstream connect error or disconnect/reset before headers. "
                "reset reason: connection termination"
            ),
        },
    }
    transport, _ = _recorder(failed)

    with pytest.raises(KitaiQueryFailed) as exc:
        _run(_cfg(), transport)

    assert "response_code=503" in (exc.value.detail or "")
    assert "connection termination" in (exc.value.detail or "")


# --------------------------------------------------------------------------- #
# Штампы вызова на конфиге (finish_reason / usage / голова ответа / модель)
# --------------------------------------------------------------------------- #


def test_complete_json_stamps_the_trace_on_the_callers_config():
    """`timeout` делает копию конфига через `replace` — штампы обязаны лечь на
    объект ВЫЗЫВАЮЩЕГО, иначе их никто не прочтёт."""
    transport, _ = _recorder(_finished('{"grades": []}'))
    cfg = _cfg()

    asyncio.run(
        kitai.complete_json(
            [{"role": "user", "content": "?"}], cfg, timeout=5.0, transport=transport
        )
    )

    assert cfg.last_finish_reason == "stop"
    assert cfg.last_usage == {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
    assert cfg.last_content_head == '{"grades": []}'
    assert cfg.last_model == "glm-5.2"


def test_complete_json_length_with_empty_content_is_traceable():
    """Пустой ответ с `finish_reason: length` — бюджет ушёл на рассуждения."""
    from app.llm_errors import GigaChatBadJSON

    transport, _ = _recorder(_finished("", finish_reason="length"))
    cfg = _cfg()

    with pytest.raises(GigaChatBadJSON):
        asyncio.run(
            kitai.complete_json([{"role": "user", "content": "?"}], cfg, transport=transport)
        )

    assert cfg.last_finish_reason == "length"
    assert cfg.last_content_head == ""
    assert cfg.last_model == "glm-5.2"


def test_failed_query_leaves_the_model_and_no_finish_reason():
    """404 «No such model» (модель снята с платформы): штампы честно пустые,
    но модель названа — это и есть ответ на вопрос «почему»."""
    failed = {
        "description": None,
        "data": {
            "query_status": "failed",
            "is_final": True,
            "error": {"status": 404, "message": "No such model"},
            "response_code": 404,
        },
    }
    transport, _ = _recorder(failed)
    cfg = _cfg(kitai_model="glm-5.1")
    cfg.last_finish_reason = "stop"  # от прошлого вызова

    with pytest.raises(KitaiQueryFailed) as exc:
        asyncio.run(
            kitai.complete_json([{"role": "user", "content": "?"}], cfg, transport=transport)
        )

    assert "No such model" in (exc.value.detail or "")
    assert cfg.last_model == "glm-5.1"
    assert cfg.last_finish_reason is None
    assert cfg.last_usage is None
    assert cfg.last_content_head == ""


def test_extract_tolerates_a_missing_or_malformed_usage_block():
    body = _finished("x")
    del body["data"]["response"]["usage"]
    assert kitai._extract(body).usage is None

    body = _finished("x")
    body["data"]["response"]["usage"] = "garbage"
    assert kitai._extract(body).usage is None


def test_stream_chat_stamps_usage_too():
    transport, _ = _recorder(_finished("целиком"))
    cfg = _cfg()

    async def drain():
        async for _ in kitai.stream_chat([{"role": "user", "content": "?"}], cfg, transport=transport):
            pass

    asyncio.run(drain())
    assert cfg.last_usage == {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
    assert cfg.last_content_head == "целиком"


# --------------------------------------------------------------------------- #
# kitai_extra_body — параметры платформы поверх нашего DTO
# --------------------------------------------------------------------------- #


def test_extra_body_keys_are_added_but_ours_are_not_overridable(caplog):
    """Оператор может выключить thinking; подменить модель или сообщения — нет."""
    import logging

    transport, seen = _recorder(_finished())
    cfg = _cfg(
        kitai_extra_body={
            "enable_thinking": False,
            "model_name": "чужая",
            "messages": [],
            "query_id": "x",
            "max_tokens": 1,
        }
    )
    kitai._announced_extra_keys.clear()

    with caplog.at_level(logging.INFO, logger="cognivault-ui.kitai"):
        _run(cfg, transport)

    body = json.loads(seen[0].content)
    assert body["enable_thinking"] is False
    assert body["model_name"] == "glm-5.2"
    assert body["messages"] == [{"role": "user", "content": "вопрос"}]
    assert body["max_tokens"] == 1024
    assert body["query_id"] != "x"
    # В логе — только имена добавленных ключей, ни значений, ни отвергнутых.
    announced = [r.message for r in caplog.records if "kitai_extra_body" in r.message]
    assert announced == ["kitai: в запрос добавляются поля из kitai_extra_body: enable_thinking"]
    assert "False" not in announced[0]

    # Второй запрос с тем же набором — без повторного объявления.
    with caplog.at_level(logging.INFO, logger="cognivault-ui.kitai"):
        _run(cfg, transport)
    assert sum("kitai_extra_body" in r.message for r in caplog.records) == 1


def test_extra_body_accepts_a_json_string_and_ignores_garbage(caplog):
    import logging

    assert _cfg(kitai_extra_body='{"enable_thinking": false}').extra_body == {"enable_thinking": False}
    assert _cfg().extra_body == {}
    assert _cfg(kitai_extra_body="").extra_body == {}
    with caplog.at_level(logging.WARNING, logger="cognivault-ui.kitai"):
        assert _cfg(kitai_extra_body="{oops").extra_body == {}
        assert _cfg(kitai_extra_body="[1]").extra_body == {}
        assert _cfg(kitai_extra_body=42).extra_body == {}
    assert "kitai_extra_body" in caplog.text

    # Без дополнительных полей тело — ровно прежнее.
    transport, seen = _recorder(_finished())
    _run(_cfg(), transport)
    assert set(json.loads(seen[0].content)) == {
        "query_id", "model_name", "messages", "temperature", "max_tokens", "profanity_check",
    }
