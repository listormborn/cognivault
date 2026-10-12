"""Прогон харнесса оценки прямо из UI — без консоли и без kubectl.

Зачем: харнесс обязан выполняться ВНУТРИ пода (снаружи закрытого контура нет ни
UI, ни mTLS-эндпоинта судьи), а прав на `kubectl exec`/`cp` у того, кто читает
отчёты, может не быть вовсе. Кнопка снимает это ограничение целиком.

Форма повторяет уже существующий фон-процесс переиндексации: старт возвращает
сразу, статус опрашивается, лог копится в памяти и отдаётся в окно. Второй
непохожий механизм для той же задачи только запутал бы.

Прогон дорогой — 47 пар × ~4 судейских вызова ≈ 190 обращений к GigaChat, — и
идёт от имени вызывающего пользователя: его токен, его настройки. Иначе A/B двух
конфигураций был бы бессмысленным, потому что ручки грейдера хранятся
пер-пользовательски.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

log = logging.getLogger("cognivault-ui.eval_runner")

# Сколько строк лога прогона держится в памяти (и отдаётся в статус). Полный
# лог лежит в файле `eval-<label>.log` рядом с отчётами.
_MEMORY_LINES = 200

# Что из логгера приложения копируется в лог прогона. Предупреждения грейдера
# и KitAI — ровно то, ради чего это нужно: без них прогон, у которого умер
# реранкер, выглядит как прогон с плохими метриками.
_COPY_LEVEL = logging.WARNING

# Каталог харнесса внутри образа (см. Dockerfile: `COPY tools/eval/ eval/`).
# В dev-запуске из репозитория он лежит на два уровня выше.
_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "eval",
    Path(__file__).resolve().parents[2] / "tools" / "eval",
)

# Метка + расширение — единственное, что попадает в имя файла отчёта.
# Всё остальное отвергаем, чтобы метка не превратилась в обход каталога.
_LABEL_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def harness_dir() -> Path | None:
    for path in _CANDIDATES:
        if (path / "run.py").is_file():
            return path
    return None


def available() -> bool:
    return harness_dir() is not None


def valid_label(label: str) -> bool:
    return bool(_LABEL_RE.match(label or ""))


@dataclass
class EvalJob:
    """Состояние одного прогона. Живёт в памяти процесса."""

    label: str
    status: str = "running"  # running | completed | failed | cancelled
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    lines: list[str] = field(default_factory=list)
    error: str | None = None
    out_dir: str = ""
    # Полный лог прогона на диске (`eval-<label>.log` в `out_dir`); `None`,
    # пока файл не открыт или если открыть его не удалось.
    log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            # Хвост: лог прогона — сотни строк, а окно показывает последние.
            "log": self.lines[-_MEMORY_LINES:],
            "log_path": self.log_path,
            "error": self.error,
            "elapsed_sec": int((self.finished_at or time.time()) - self.started_at),
        }


def log_file(out_dir: Path, label: str) -> Path:
    """Where the full log of a run goes. ``label`` is already validated
    (:func:`valid_label`), so it cannot escape ``out_dir``."""
    return out_dir / f"eval-{label}.log"


class _JobLog:
    """Приёмник строк прогона: хвост в памяти задачи, всё — в файл.

    Раньше раннер перехватывал `sys.stderr` ВСЕГО процесса на время прогона
    (`contextlib.redirect_stderr`). Но туда же пишет логгер приложения: у UI
    нет своего обработчика, записи уходят через `logging.lastResort`, а он
    берёт `sys.stderr` в момент вывода. Итог — на время прогона каждое
    предупреждение грейдера и KitAI пропадало из лога пода в буфер на 200
    строк. Теперь у харнесса явный приёмник (`run.LOG_SINK`), а записи
    логгера копируются сюда обработчиком (:class:`_CopyHandler`) — и остаются
    в stderr.
    """

    def __init__(self, job: EvalJob, path: Path | None) -> None:
        self._job = job
        self._lock = threading.Lock()
        self._fh: TextIO | None = None
        if path is None:
            return
        try:
            self._fh = path.open("a", encoding="utf-8")
        except OSError as exc:
            # Не фатально: прогон идёт, статус показывает хвост. Но сказать
            # надо — иначе «файла нет» ищут не там.
            log.warning("eval: лог прогона не пишется в %s: %s", path, exc)
            return
        job.log_path = str(path)

    def line(self, text: str) -> None:
        """Одна или несколько строк (текст может содержать переводы строк)."""
        for raw in str(text).splitlines():
            entry = raw.rstrip()
            if not entry.strip():
                continue
            with self._lock:
                lines = self._job.lines
                lines.append(entry)
                # Амортизированная обрезка: в памяти только хвост, в файле всё.
                if len(lines) > 2 * _MEMORY_LINES:
                    del lines[:-_MEMORY_LINES]
                if self._fh is not None:
                    try:
                        self._fh.write(entry + "\n")
                        self._fh.flush()
                    except OSError:
                        pass

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None


class _CopyHandler(logging.Handler):
    """Копирует записи логгера (WARNING и выше) в лог прогона.

    Висит на корневом логгере только на время прогона. Тонкость: пока в цепочке
    логгеров записи нет ни одного обработчика, Python выводит её через
    `logging.lastResort` в stderr — и перестаёт это делать, как только
    обработчик появляется. То есть наивная копия ЗАМЕНИЛА бы вывод в лог пода
    вместо того, чтобы его дополнить. Поэтому запись, у которой в цепочке нет
    других обработчиков, кроме этого, дополнительно отдаётся `lastResort` —
    ровно так, как её вывели бы без нас. Решение принимается ПО ЗАПИСИ, а не
    при подключении: обработчик может стоять не на корне, а на логгере
    приложения (`cognivault-ui`), и такую запись уже напечатали.
    """

    def __init__(self, sink: _JobLog) -> None:
        super().__init__(level=_COPY_LEVEL)
        self._sink = sink
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def _would_reach_last_resort(self, record: logging.LogRecord) -> bool:
        """Повторяет подсчёт `Logger.callHandlers`: любой чужой обработчик в
        цепочке (независимо от его уровня) отключает `lastResort`."""
        logger: logging.Logger | None = logging.getLogger(record.name)
        while logger is not None:
            if any(h is not self for h in logger.handlers):
                return False
            if not logger.propagate:
                break
            logger = logger.parent
        return True

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        try:
            self._sink.line(self.format(record))
        except Exception:  # noqa: BLE001 — лог прогона не должен ронять логгер
            self.handleError(record)
        last_resort = logging.lastResort
        if last_resort is not None and self._would_reach_last_resort(record):
            if record.levelno >= last_resort.level:
                last_resort.handle(record)


class EvalRunner:
    """Один прогон на процесс — больше не нужно и вредно.

    Два параллельных прогона дрались бы за один и тот же лимит GigaChat и за один
    файл отчёта, а сравнивать их всё равно нельзя: настройки между ними не
    менялись.
    """

    def __init__(self) -> None:
        self._job: EvalJob | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def job(self) -> EvalJob | None:
        return self._job

    def busy(self) -> bool:
        return self._job is not None and self._job.status == "running"

    def start(self, *, label: str, argv: list[str], out_dir: Path) -> EvalJob:
        if self.busy():
            raise RuntimeError("прогон уже идёт")
        out_dir.mkdir(parents=True, exist_ok=True)
        job = EvalJob(label=label, out_dir=str(out_dir))
        self._job = job
        self._task = asyncio.create_task(self._run(job, argv))
        return job

    async def stop(self) -> EvalJob:
        """Остановить текущий прогон и дождаться освобождения его ресурсов.

        Простого ``Task.cancel()`` недостаточно: маршрут остановки должен вернуть
        уже окончательный статус, а ``main_async`` обязан успеть закрыть HTTP-
        клиенты чата, судьи и бэкенда в своём ``finally``.
        """
        job = self._job
        task = self._task
        if job is None or job.status != "running" or task is None or task.done():
            raise RuntimeError("активного прогона нет")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            # `_run` намеренно пробрасывает отмену после фиксации статуса и
            # закрытия ресурсов. Для HTTP-маршрута это штатный исход.
            pass
        # Если отмена прилетела до первого такта `_run`, его `except/finally`
        # ещё не успели выполниться. Без этой страховки задача уже мертва, а UI
        # навсегда видит `running` и не даёт запустить следующий прогон.
        if job.status == "running":
            job.status = "cancelled"
            job.error = "остановлен пользователем"
            job.finished_at = time.time()
        return job

    async def _run(self, job: EvalJob, argv: list[str]) -> None:
        sink = _JobLog(job, log_file(Path(job.out_dir), job.label) if job.out_dir else None)
        root = logging.getLogger()
        handler = _CopyHandler(sink)
        run_mod: Any = None
        prev_sink: Any = None
        try:
            run_mod = _load_harness()
            # Явный приёмник прогресса вместо перехвата stderr всего процесса —
            # см. `_JobLog`. Снимается в `finally`, даже если харнесс упал.
            prev_sink = getattr(run_mod, "LOG_SINK", None)
            setattr(run_mod, "LOG_SINK", sink.line)
            root.addHandler(handler)
            code = await run_mod.main_async(argv)
            if code == 0:
                job.status = "completed"
            else:
                job.status = "failed"
                job.error = f"харнесс вернул код {code}"
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.error = "остановлен пользователем"
            raise
        except Exception as exc:  # noqa: BLE001 — фон не должен ронять процесс
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            root.removeHandler(handler)
            if run_mod is not None:
                setattr(run_mod, "LOG_SINK", prev_sink)
            if job.error:
                sink.line(job.error)
            sink.close()
            job.finished_at = time.time()


def _load_harness() -> Any:
    """Импортировать `run.py` харнесса.

    Каталог кладётся в `sys.path`, потому что харнесс импортирует соседей по
    голому имени (`import metrics`, `from gigachat_client import …`) — так он
    устроен и так же запускается из консоли. Модуль регистрируется под
    неконфликтным именем: `run` — слишком общее, чтобы занимать его глобально.
    """
    directory = harness_dir()
    if directory is None:
        raise RuntimeError("харнесс оценки не найден в образе")
    path = str(directory)
    if path not in sys.path:
        sys.path.insert(0, path)
    name = "cognivault_eval_run"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, directory / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("не удалось загрузить run.py харнесса")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = EvalRunner()
