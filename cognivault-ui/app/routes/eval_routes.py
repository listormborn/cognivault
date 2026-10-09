"""Прогон харнесса оценки из UI: старт, статус, выгрузка отчёта.

Три маршрута вместо одного, потому что прогон долгий (47 пар × ~4 судейских
вызова): один HTTP-запрос его не переживёт, а держать соединение открытым
несколько минут — верный способ получить обрыв на прокси. Форма та же, что у
переиндексации: старт → опрос статуса → результат.

Отчёты пишутся в каталог ВЫЗЫВАЮЩЕГО (`AppPaths.root/eval-reports`): прогон
идёт под его токеном и его настройками, значит и результат принадлежит ему.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .. import eval_datasets, eval_runner, settings
from ..deps import get_token, resolve_paths

log = logging.getLogger("cognivault-ui.eval")

router = APIRouter(prefix="/api/eval")


def _error(status: int, code: str, message: str, detail: str | None = None) -> JSONResponse:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if detail is not None:
        body["error"]["detail"] = detail
    return JSONResponse(status_code=status, content=body)


def _reports_dir(request: Request) -> Path:
    return Path(resolve_paths(request).root) / "eval-reports"


def _dataset_list(request: Request) -> list[dict[str, Any]]:
    root = Path(resolve_paths(request).root)
    return eval_datasets.list_datasets(root, eval_runner.harness_dir())


@router.post("/dataset")
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    _token: str = Depends(get_token),
) -> Any:
    """Принять JSONL/XLSX и сохранить нормализованный набор пользователя."""
    content = await file.read(eval_datasets.MAX_UPLOAD_BYTES + 1)
    try:
        info = eval_datasets.save_upload(
            Path(resolve_paths(request).root), file.filename or "", content
        )
    except eval_datasets.DatasetError as exc:
        return _error(400, "EVAL_BAD_DATASET", str(exc))
    except OSError as exc:
        return _error(
            503,
            "EVAL_NO_STORAGE",
            "Некуда сохранить набор вопросов — каталог данных недоступен",
            str(exc),
        )
    return info


@router.post("/run")
async def start_eval(
    request: Request,
    payload: dict[str, Any] = Body(default_factory=dict),
    token: str = Depends(get_token),
) -> Any:
    """Запустить прогон в фоне и сразу вернуть его статус."""
    if not eval_runner.available():
        return _error(
            501,
            "EVAL_NOT_AVAILABLE",
            "Харнесс оценки не входит в этот образ",
            "Нужен образ UI, собранный вместе с tools/eval",
        )

    label = str(payload.get("label") or "run").strip()
    if not eval_runner.valid_label(label):
        return _error(
            400,
            "EVAL_BAD_LABEL",
            "Метка может содержать латиницу, цифры, точку, дефис и подчёркивание",
            label,
        )

    limit = payload.get("limit")
    try:
        limit_int = max(0, int(limit)) if limit is not None else 0
    except (TypeError, ValueError):
        return _error(400, "EVAL_BAD_LIMIT", "limit должен быть целым числом", str(limit))

    if eval_runner.RUNNER.busy():
        return _error(409, "EVAL_BUSY", "Прогон уже идёт — дождитесь его окончания")

    out_dir = _reports_dir(request)
    paths = resolve_paths(request)
    dataset_id = str(payload.get("dataset") or "builtin:golden.jsonl")
    golden = eval_datasets.resolve_dataset(
        Path(paths.root), eval_runner.harness_dir(), dataset_id
    )
    if golden is None:
        return _error(
            400,
            "EVAL_BAD_DATASET",
            "Набор вопросов не найден",
            dataset_id,
        )
    argv = [
        "--label", label,
        "--ui-url", f"http://127.0.0.1:{settings.bind_port()}",
        "--out-dir", str(out_dir),
        "--golden", str(golden),
        "--limit", str(limit_int),
        # Строго один поток. У судьи на контуре фактически один слот: второй
        # одновременный запрос получает 429 сразу, а ретраи короче чужого вызова.
        # На двух потоках прогон `baseline` потерял 95 судейских вызовов из 188.
        "--concurrency", "1",
        # Фактический контекст хода берётся из лога ЭТОГО пользователя — иначе
        # метрики по контексту считались бы по чужим ходам или по фолбэку.
        "--rag-log", str(Path(paths.root) / "rag_log.jsonl"),
    ]
    if token:
        # Прогон обязан идти под токеном вызывающего: настройки грейдера и
        # промпты хранятся пер-пользовательски, и без этого A/B не имеет смысла.
        argv += ["--token", token]

    try:
        job = eval_runner.RUNNER.start(label=label, argv=argv, out_dir=out_dir)
    except RuntimeError as exc:
        return _error(409, "EVAL_BUSY", str(exc))
    except OSError as exc:
        # Каталог отчётов лежит в /data, а он монтируется извне. Без тома или с
        # чужими правами это не «внутренняя ошибка», а понятная проблема
        # развёртывания — и сказать о ней надо так, чтобы её было где искать.
        log.warning("eval: каталог отчётов недоступен (%s): %s", out_dir, exc)
        return _error(
            503,
            "EVAL_NO_STORAGE",
            "Некуда писать отчёты — каталог данных недоступен",
            f"{out_dir}: {exc}",
        )
    log.info("eval: запущен прогон %r (limit=%s)", label, limit_int)
    return job.to_dict()


@router.get("/status")
async def eval_status(request: Request, _token: str = Depends(get_token)) -> Any:
    """Статус текущего/последнего прогона плюс список готовых отчётов."""
    job = eval_runner.RUNNER.job
    out_dir = _reports_dir(request)
    reports = sorted(p.name for p in out_dir.glob("report-*.*")) if out_dir.is_dir() else []
    return {
        "available": eval_runner.available(),
        "job": job.to_dict() if job else None,
        "reports": reports,
        "datasets": _dataset_list(request),
    }


@router.get("/report")
async def eval_report(
    request: Request, name: str = "", _token: str = Depends(get_token)
) -> Any:
    """Отдать готовый отчёт на скачивание.

    Имя проверяется по белому списку из каталога, а не склеивается с ним: любая
    склейка пути с пользовательским вводом — это обход каталога, а тут ещё и
    каталог с чужими данными рядом.
    """
    out_dir = _reports_dir(request)
    if not out_dir.is_dir():
        return _error(404, "EVAL_NO_REPORTS", "Отчётов пока нет")
    # Полный лог прогона (`eval-<label>.log`, см. `eval_runner.log_file`)
    # выгружается тем же маршрутом: статус отдаёт лишь хвост в 200 строк.
    allowed = {p.name: p for p in out_dir.glob("report-*.*")}
    allowed.update({p.name: p for p in out_dir.glob("eval-*.log")})
    target = allowed.get(name)
    if target is None:
        return _error(404, "EVAL_NO_REPORT", "Такого отчёта нет", name)
    media = {".md": "text/markdown", ".log": "text/plain"}.get(
        target.suffix, "application/json"
    )
    return FileResponse(target, media_type=media, filename=target.name)
