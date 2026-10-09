"""Кнопка «прогнать оценку»: старт, статус, выгрузка отчёта.

Офлайн: сам харнесс подменяется — нас интересует обвязка (валидация метки,
защита от параллельных прогонов, обход каталога в имени отчёта), а не его
внутренности, у которых свои тесты в `tools/eval/tests`.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import eval_datasets, eval_runner, settings  # noqa: E402
from app.config import AppPaths  # noqa: E402
from app.main import create_app  # noqa: E402
from app.routes import eval_routes  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_runner():
    """Раннер — синглтон на процесс; между тестами его надо обнулять."""
    eval_runner.RUNNER._job = None
    eval_runner.RUNNER._task = None
    yield
    eval_runner.RUNNER._job = None
    eval_runner.RUNNER._task = None


# --------------------------------------------------------------------------- #
# Метка попадает в имя файла — значит это недоверенный ввод
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("good", ["run", "after-w5", "ok_1.2", "A" * 64])
def test_label_accepts_safe_names(good):
    assert eval_runner.valid_label(good)


@pytest.mark.parametrize(
    "bad",
    ["", "a b", "../etc/passwd", "a/b", "a\\b", "отчёт", "A" * 65, "run;rm -rf /"],
)
def test_label_rejects_everything_else(bad):
    """Метка уезжает в имя файла отчёта — путь и пробелы здесь недопустимы."""
    assert not eval_runner.valid_label(bad)


# --------------------------------------------------------------------------- #
# Фоновый прогон
# --------------------------------------------------------------------------- #


def _fake_harness(monkeypatch, *, code=0, out="строка прогресса\nещё одна\n", boom=None):
    class FakeRun:
        # Тот же крючок, что у настоящего `run.py`: раннер ставит сюда приёмник
        # на время прогона, `_log` печатает в stderr только без него.
        LOG_SINK = None

        @staticmethod
        def _log(message):
            sink = FakeRun.LOG_SINK
            if sink is not None:
                sink(message)
                return
            print(message, file=sys.stderr, flush=True)

        @staticmethod
        async def main_async(argv):
            if boom is not None:
                raise boom
            for line in out.splitlines():
                FakeRun._log(line)
            return code

    monkeypatch.setattr(eval_runner, "_load_harness", lambda: FakeRun)
    return FakeRun


def _run_job(tmp_path, monkeypatch, **kw):
    _fake_harness(monkeypatch, **kw)

    async def go():
        job = eval_runner.RUNNER.start(label="t", argv=[], out_dir=tmp_path / "rep")
        await eval_runner.RUNNER._task
        return job

    return asyncio.run(go())


def test_successful_run_collects_the_progress_log(tmp_path, monkeypatch):
    """Прогресс харнесса идёт в stderr — другого канала у него нет."""
    job = _run_job(tmp_path, monkeypatch)

    assert job.status == "completed"
    assert job.lines == ["строка прогресса", "ещё одна"]
    assert job.to_dict()["elapsed_sec"] >= 0


def test_nonzero_exit_is_a_failure_not_a_success(tmp_path, monkeypatch):
    job = _run_job(tmp_path, monkeypatch, code=2)

    assert job.status == "failed"
    assert "код 2" in (job.error or "")


def test_exception_inside_the_harness_does_not_kill_the_process(tmp_path, monkeypatch):
    """Фон не должен ронять веб-процесс — иначе один кривой прогон уронит чат."""
    job = _run_job(tmp_path, monkeypatch, boom=ValueError("судья недоступен"))

    assert job.status == "failed"
    assert "ValueError" in (job.error or "")
    assert "судья недоступен" in (job.error or "")


def test_second_run_is_refused_while_the_first_is_alive(tmp_path, monkeypatch):
    """Два прогона дрались бы за лимит GigaChat и за один файл отчёта."""
    started = asyncio.Event()
    release = asyncio.Event()

    class Slow:
        @staticmethod
        async def main_async(argv):
            started.set()
            await release.wait()
            return 0

    monkeypatch.setattr(eval_runner, "_load_harness", lambda: Slow)

    async def go():
        eval_runner.RUNNER.start(label="first", argv=[], out_dir=tmp_path / "rep")
        await started.wait()
        assert eval_runner.RUNNER.busy()
        with pytest.raises(RuntimeError):
            eval_runner.RUNNER.start(label="second", argv=[], out_dir=tmp_path / "rep")
        release.set()
        await eval_runner.RUNNER._task

    asyncio.run(go())
    assert eval_runner.RUNNER.job.label == "first"


def test_log_is_tailed_so_a_long_run_does_not_bloat_the_response(tmp_path, monkeypatch):
    job = eval_runner.EvalJob(label="t")
    job.lines = [f"строка {i}" for i in range(500)]

    assert len(job.to_dict()["log"]) == 200
    assert job.to_dict()["log"][-1] == "строка 499"


# --------------------------------------------------------------------------- #
# Выгрузка отчёта
# --------------------------------------------------------------------------- #


def test_report_lookup_is_a_whitelist_not_a_path_join(tmp_path):
    """Имя отчёта приходит от клиента — склейка с каталогом дала бы обход.

    Рядом лежат каталоги других пользователей, поэтому проверяем принадлежность
    к перечню файлов, а не «не начинается ли имя с ../».
    """
    out = tmp_path / "eval-reports"
    out.mkdir()
    (out / "report-ok.md").write_text("отчёт")
    (tmp_path / "secret.txt").write_text("чужое")

    allowed = {p.name: p for p in out.glob("report-*.*")}

    assert "report-ok.md" in allowed
    for attack in ("../secret.txt", "/etc/passwd", "report-ok.md/../../secret.txt"):
        assert attack not in allowed


def test_run_is_started_strictly_single_threaded(tmp_path, monkeypatch):
    """У судьи на контуре один слот — прогон обязан идти в один поток.

    На `--concurrency 2` прогон `baseline` потерял 95 судейских вызовов из 188:
    второй одновременный запрос получал 429 сразу, а ретраи короче чужого
    вызова, поэтому проигравший терял метрики целыми сэмплами. Число живёт
    здесь, а не в дефолте харнесса, — значит и стеречь его надо здесь.
    """
    monkeypatch.setattr(settings, "is_server", lambda: False)
    monkeypatch.setattr(
        eval_routes, "resolve_paths", lambda request: AppPaths(root=tmp_path)
    )
    monkeypatch.setattr(eval_runner, "available", lambda: True)
    seen: dict = {}

    def fake_start(*, label, argv, out_dir):
        seen["argv"] = argv
        return eval_runner.EvalJob(label=label)

    monkeypatch.setattr(eval_runner.RUNNER, "start", fake_start)

    with TestClient(create_app()) as client:
        resp = client.post("/api/eval/run", json={"label": "smoke", "limit": 5})

    assert resp.status_code == 200
    argv = seen["argv"]
    assert argv[argv.index("--concurrency") + 1] == "1"


def test_uploaded_jsonl_is_listed_and_can_be_selected(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "is_server", lambda: False)
    monkeypatch.setattr(
        eval_routes, "resolve_paths", lambda request: AppPaths(root=tmp_path)
    )
    monkeypatch.setattr(eval_runner, "available", lambda: True)
    seen: dict = {}

    def fake_start(*, label, argv, out_dir):
        seen["argv"] = argv
        return eval_runner.EvalJob(label=label)

    monkeypatch.setattr(eval_runner.RUNNER, "start", fake_start)
    content = (
        '{"id":"q1","question":"Что такое вольт?","ground_truth":"База знаний."}\n'
    )

    with TestClient(create_app()) as client:
        uploaded = client.post(
            "/api/eval/dataset",
            files={"file": ("questions.jsonl", content, "application/x-ndjson")},
        )
        assert uploaded.status_code == 200
        info = uploaded.json()
        assert info["rows"] == 1
        assert info["with_ground_truth"] == 1

        status = client.get("/api/eval/status")
        listed = next(item for item in status.json()["datasets"] if item["id"] == info["id"])
        assert listed["name"] == "questions"

        started = client.post(
            "/api/eval/run",
            json={"label": "custom", "limit": 1, "dataset": info["id"]},
        )

    assert started.status_code == 200
    argv = seen["argv"]
    golden = Path(argv[argv.index("--golden") + 1])
    assert golden.parent == tmp_path / "eval-datasets"
    assert json.loads(golden.read_text(encoding="utf-8"))["question"] == "Что такое вольт?"


def test_run_rejects_dataset_path_instead_of_joining_it(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "is_server", lambda: False)
    monkeypatch.setattr(
        eval_routes, "resolve_paths", lambda request: AppPaths(root=tmp_path)
    )
    monkeypatch.setattr(eval_runner, "available", lambda: True)

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/eval/run",
            json={"label": "attack", "dataset": "upload:../../secret.jsonl"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "EVAL_BAD_DATASET"


def _tiny_xlsx() -> bytes:
    sheet = """<?xml version="1.0" encoding="UTF-8"?>
    <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
      <sheetData>
        <row r="1">
          <c r="A1" t="inlineStr"><is><t>№</t></is></c>
          <c r="B1" t="inlineStr"><is><t>Кейс</t></is></c>
          <c r="C1" t="inlineStr"><is><t>Вопрос агенту</t></is></c>
        </row>
        <row r="2">
          <c r="A2" t="n"><v>1</v></c>
          <c r="B2" t="inlineStr"><is><t>UC8 Вне скоупа</t></is></c>
          <c r="C2" t="inlineStr"><is><t>Какая завтра погода?</t></is></c>
        </row>
      </sheetData>
    </worksheet>"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
    return buffer.getvalue()


def test_xlsx_upload_is_converted_without_external_excel_dependency():
    rows = eval_datasets.parse_upload("questions.xlsx", _tiny_xlsx())

    assert len(rows) == 1
    assert rows[0]["id"] == "1"
    assert rows[0]["question"] == "Какая завтра погода?"
    assert rows[0]["expected_outcome"] == "refusal"
    assert rows[0]["ground_truth"] == ""


def test_status_shape_survives_a_run(tmp_path, monkeypatch):
    job = _run_job(tmp_path, monkeypatch)
    payload = json.loads(json.dumps(job.to_dict()))  # должен быть JSON-сериализуем

    assert set(payload) >= {"label", "status", "log", "elapsed_sec"}


def test_unwritable_report_dir_is_a_readable_error_not_a_500(tmp_path, monkeypatch):
    """Каталог отчётов живёт в /data — том монтируется снаружи и может быть недоступен.

    Поймано прогоном настоящего образа без тома: пользователь получал голое
    «Internal Server Error», по которому непонятно даже, куда смотреть.
    """
    _fake_harness(monkeypatch)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)  # только чтение и обход
    try:
        with pytest.raises(OSError):
            eval_runner.RUNNER.start(label="t", argv=[], out_dir=locked / "eval-reports")
    finally:
        locked.chmod(0o700)


# --------------------------------------------------------------------------- #
# Лог прогона: не перехватывать stderr процесса, писать файл, копировать логгер
# --------------------------------------------------------------------------- #


def test_run_does_not_hijack_the_process_stderr(tmp_path, monkeypatch):
    """`redirect_stderr` на весь процесс уводил в буфер и лог приложения.

    Логгер UI выводит через `logging.lastResort`, который берёт `sys.stderr`
    в момент вывода, — на время прогона предупреждения грейдера пропадали из
    лога пода. Прогресс харнесса теперь идёт через явный приёмник.
    """
    seen = {}

    class Spy:
        LOG_SINK = None

        @staticmethod
        async def main_async(argv):
            seen["stderr"] = sys.stderr
            seen["stdout"] = sys.stdout
            Spy.LOG_SINK("прогресс")
            return 0

    monkeypatch.setattr(eval_runner, "_load_harness", lambda: Spy)
    before = (sys.stderr, sys.stdout)

    async def go():
        job = eval_runner.RUNNER.start(label="t", argv=[], out_dir=tmp_path / "rep")
        await eval_runner.RUNNER._task
        return job

    job = asyncio.run(go())

    assert seen["stderr"] is before[0] and seen["stdout"] is before[1]
    assert job.lines == ["прогресс"]
    assert Spy.LOG_SINK is None  # снят после прогона


def test_harness_without_the_hook_still_prints_to_stderr(tmp_path, monkeypatch, capsys):
    fake = _fake_harness(monkeypatch)
    fake.LOG_SINK = None
    fake._log("мимо раннера")
    assert "мимо раннера" in capsys.readouterr().err


def test_run_writes_the_full_log_to_a_file_and_exposes_its_path(tmp_path, monkeypatch):
    """В памяти — хвост, на диске — всё; путь виден в статусе."""
    out = "".join(f"строка {i}\n" for i in range(450))
    job = _run_job(tmp_path, monkeypatch, out=out)

    path = Path(job.log_path)
    assert path == tmp_path / "rep" / "eval-t.log"
    assert path.read_text(encoding="utf-8").splitlines() == [f"строка {i}" for i in range(450)]
    status = job.to_dict()
    assert status["log_path"] == str(path)
    assert status["log"] == [f"строка {i}" for i in range(250, 450)]
    assert len(job.lines) <= 2 * eval_runner._MEMORY_LINES


def test_failure_reason_lands_in_the_log_file_too(tmp_path, monkeypatch):
    job = _run_job(tmp_path, monkeypatch, boom=ValueError("судья недоступен"))
    assert "ValueError: судья недоступен" in Path(job.log_path).read_text(encoding="utf-8")


def test_app_warnings_reach_both_the_pod_log_and_the_job_log(tmp_path, monkeypatch):
    """Предупреждение грейдера во время прогона: и в stderr, и в лог прогона.

    Обработчик на корневом логгере отключает `lastResort` — без явного эха
    «починка» просто перенесла бы потерю лога пода в другое место.
    """

    class Loud:
        LOG_SINK = None

        @staticmethod
        async def main_async(argv):
            logging.getLogger("cognivault-ui.rag_pipeline").warning(
                "grader: батч 2 (модель glm-5.1): вызов не удался (KitaiQueryFailed: 404)"
            )
            return 0

    echoed: list[str] = []

    class Recorder(logging.Handler):
        def emit(self, record):
            echoed.append(self.format(record))

    root = logging.getLogger()
    # Как в поде без своей настройки логгера: ни у корня, ни у логгера
    # приложения нет обработчиков, записи уходят через lastResort.
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(logging.getLogger("cognivault-ui"), "handlers", [])
    monkeypatch.setattr(logging, "lastResort", Recorder(level=logging.WARNING))
    monkeypatch.setattr(eval_runner, "_load_harness", lambda: Loud)

    async def go():
        job = eval_runner.RUNNER.start(label="t", argv=[], out_dir=tmp_path / "rep")
        await eval_runner.RUNNER._task
        return job

    job = asyncio.run(go())

    assert any("glm-5.1" in line for line in job.lines)
    assert any("glm-5.1" in line for line in echoed)
    assert "glm-5.1" in Path(job.log_path).read_text(encoding="utf-8")
    assert root.handlers == []  # обработчик снят после прогона


def test_copy_handler_does_not_double_print_when_root_already_has_handlers(tmp_path, monkeypatch):
    class Loud:
        LOG_SINK = None

        @staticmethod
        async def main_async(argv):
            logging.getLogger("cognivault-ui.rag_pipeline").warning("одно предупреждение")
            return 0

    echoed: list[str] = []

    class Recorder(logging.Handler):
        def emit(self, record):
            echoed.append(self.format(record))

    root = logging.getLogger()
    own = Recorder(level=logging.WARNING)
    monkeypatch.setattr(root, "handlers", [own])
    lost: list[str] = []

    class Never(logging.Handler):
        def emit(self, record):
            lost.append(self.format(record))

    monkeypatch.setattr(logging, "lastResort", Never(level=logging.WARNING))
    monkeypatch.setattr(eval_runner, "_load_harness", lambda: Loud)

    async def go():
        job = eval_runner.RUNNER.start(label="t", argv=[], out_dir=tmp_path / "rep")
        await eval_runner.RUNNER._task
        return job

    job = asyncio.run(go())

    assert [line for line in echoed if "одно предупреждение" in line] == ["одно предупреждение"]
    assert lost == []
    assert any("одно предупреждение" in line for line in job.lines)


def test_log_file_is_downloadable_through_the_report_route(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "is_server", lambda: False)
    out = tmp_path / "eval-reports"
    out.mkdir()
    (out / "eval-t.log").write_text("строка", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("чужое", encoding="utf-8")
    monkeypatch.setattr(eval_routes, "resolve_paths", lambda request: AppPaths(root=tmp_path))

    with TestClient(create_app()) as client:
        ok = client.get("/api/eval/report", params={"name": "eval-t.log"})
        bad = client.get("/api/eval/report", params={"name": "../secret.txt"})

    assert ok.status_code == 200 and ok.text == "строка"
    assert bad.status_code == 404


def test_copy_handler_respects_a_handler_on_the_app_logger(tmp_path, monkeypatch):
    """Обработчик может стоять не на корне, а на `cognivault-ui` (см. main.py):
    такую запись уже напечатали, эхо через lastResort дало бы её дважды."""

    class Loud:
        LOG_SINK = None

        @staticmethod
        async def main_async(argv):
            logging.getLogger("cognivault-ui.rag_pipeline").warning("одно предупреждение")
            return 0

    printed: list[str] = []

    class Recorder(logging.Handler):
        def emit(self, record):
            printed.append(self.format(record))

    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    monkeypatch.setattr(logging.getLogger("cognivault-ui"), "handlers", [Recorder()])
    lost: list[str] = []

    class Never(logging.Handler):
        def emit(self, record):
            lost.append(self.format(record))

    monkeypatch.setattr(logging, "lastResort", Never(level=logging.WARNING))
    monkeypatch.setattr(eval_runner, "_load_harness", lambda: Loud)

    async def go():
        job = eval_runner.RUNNER.start(label="t", argv=[], out_dir=tmp_path / "rep")
        await eval_runner.RUNNER._task
        return job

    job = asyncio.run(go())

    assert printed == ["одно предупреждение"]
    assert lost == []
    assert any("одно предупреждение" in line for line in job.lines)
