"""Загрузка и нормализация пользовательских наборов вопросов для eval.

Файл хранится в каталоге текущего пользователя уже как JSONL.  Исходный XLSX
не сохраняется: это уменьшает поверхность атаки и даёт харнессу один формат.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree as ET


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_EXPANDED_BYTES = 25 * 1024 * 1024
MAX_ROWS = 2_000

_QUESTION_HEADERS = {"question", "query", "вопрос", "вопрос агенту"}
_ID_HEADERS = {"id", "номер", "№", "n"}
_CATEGORY_HEADERS = {"category", "категория", "кейс", "use case", "use_case"}
_GROUND_TRUTH_HEADERS = {
    "ground_truth",
    "ground truth",
    "эталон",
    "эталонный ответ",
    "ожидаемый ответ",
}


class DatasetError(ValueError):
    """Понятная пользователю ошибка входного файла."""


def datasets_dir(root: Path) -> Path:
    return root / "eval-datasets"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normal_header(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value).lower()).strip()


def _safe_id(value: Any, index: int) -> str:
    raw = _text(value)
    if raw.endswith(".0") and raw[:-2].isdigit():
        raw = raw[:-2]
    raw = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._")
    return raw[:64] or f"q{index:03d}"


def _normalise_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    question = _text(row.get("question"))
    if not question:
        raise DatasetError(f"строка {index}: пустой вопрос")

    category = _text(row.get("category")) or "unclassified"
    expected = _text(row.get("expected_outcome")).lower()
    expected_refusal = bool(row.get("expected_refusal"))
    if not expected:
        expected = "refusal" if expected_refusal or "вне скоупа" in category.lower() else "answer"
    if expected not in {"answer", "refusal", "meta"}:
        raise DatasetError(
            f"строка {index}: expected_outcome должен быть answer, refusal или meta"
        )
    expected_refusal = expected == "refusal"

    result = dict(row)
    result.update(
        {
            "id": _safe_id(row.get("id"), index),
            "question": question,
            "ground_truth": _text(row.get("ground_truth")),
            "kind": _text(row.get("kind"))
            or ("unanswerable" if expected_refusal else "factual"),
            "category": category,
            "source_path": row.get("source_path") or None,
            "section_path": row.get("section_path") or None,
            "source_chunk_index": row.get("source_chunk_index"),
            "expected_refusal": expected_refusal,
            "expected_outcome": expected,
            "accepted": row.get("accepted") is not False,
        }
    )
    return result


def _parse_jsonl(content: bytes) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetError("JSONL должен быть в UTF-8") from exc

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"строка {line_no}: невалидный JSON ({exc.msg})") from exc
        if not isinstance(raw, dict):
            raise DatasetError(f"строка {line_no}: ожидается JSON-объект")
        rows.append(_normalise_row(raw, len(rows) + 1))
    return rows


def _column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref.upper() if ch.isalpha())
    result = 0
    for char in letters:
        result = result * 26 + ord(char) - ord("A") + 1
    return max(0, result - 1)


def _xlsx_rows(content: bytes) -> list[list[str]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except (zipfile.BadZipFile, OSError) as exc:
        raise DatasetError("файл не является корректным XLSX") from exc

    with archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_EXPANDED_BYTES:
            raise DatasetError("XLSX слишком большой после распаковки")

        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.itertext()) for node in root]

        sheet_path = "xl/worksheets/sheet1.xml"
        if "xl/workbook.xml" in archive.namelist() and "xl/_rels/workbook.xml.rels" in archive.namelist():
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            rel_map = {node.attrib.get("Id"): node.attrib.get("Target") for node in rels}
            first_sheet = next((node for node in workbook.iter() if node.tag.endswith("}sheet")), None)
            if first_sheet is not None:
                rel_id = next((v for k, v in first_sheet.attrib.items() if k.endswith("}id")), None)
                target = rel_map.get(rel_id)
                if target:
                    clean = str(PurePosixPath("xl") / target).replace("xl/../", "")
                    sheet_path = clean.lstrip("/")

        if sheet_path not in archive.namelist():
            raise DatasetError("в XLSX не найден первый лист")
        root = ET.fromstring(archive.read(sheet_path))

        rows: list[list[str]] = []
        for row_node in (node for node in root.iter() if node.tag.endswith("}row")):
            values: dict[int, str] = {}
            for cell in (node for node in row_node if node.tag.endswith("}c")):
                ref = cell.attrib.get("r", "A1")
                kind = cell.attrib.get("t", "")
                value = ""
                if kind == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.iter() if node.tag.endswith("}t")
                    )
                else:
                    value_node = next(
                        (node for node in cell if node.tag.endswith("}v")), None
                    )
                    value = value_node.text or "" if value_node is not None else ""
                    if kind == "s" and value.isdigit():
                        pos = int(value)
                        value = shared[pos] if pos < len(shared) else ""
                values[_column_index(ref)] = value
            if values:
                width = max(values) + 1
                rows.append([values.get(i, "") for i in range(width)])
        return rows


def _parse_xlsx(content: bytes) -> list[dict[str, Any]]:
    matrix = _xlsx_rows(content)
    header_pos = -1
    headers: list[str] = []
    for pos, row in enumerate(matrix[:20]):
        candidate = [_normal_header(cell) for cell in row]
        if any(cell in _QUESTION_HEADERS for cell in candidate):
            header_pos, headers = pos, candidate
            break
    if header_pos < 0:
        raise DatasetError(
            "не найден столбец с вопросом (поддерживаются «Вопрос агенту», «Вопрос», question)"
        )

    def find(aliases: set[str]) -> int | None:
        return next((i for i, name in enumerate(headers) if name in aliases), None)

    question_col = find(_QUESTION_HEADERS)
    id_col = find(_ID_HEADERS)
    category_col = find(_CATEGORY_HEADERS)
    truth_col = find(_GROUND_TRUTH_HEADERS)
    assert question_col is not None

    records: list[dict[str, Any]] = []
    for row in matrix[header_pos + 1 :]:
        question = _text(row[question_col] if question_col < len(row) else "")
        if not question:
            continue
        raw = {
            "id": row[id_col] if id_col is not None and id_col < len(row) else "",
            "question": question,
            "ground_truth": (
                row[truth_col] if truth_col is not None and truth_col < len(row) else ""
            ),
            "category": (
                row[category_col]
                if category_col is not None and category_col < len(row)
                else "unclassified"
            ),
        }
        records.append(_normalise_row(raw, len(records) + 1))
    return records


def parse_upload(filename: str, content: bytes) -> list[dict[str, Any]]:
    if not content:
        raise DatasetError("файл пуст")
    if len(content) > MAX_UPLOAD_BYTES:
        raise DatasetError("файл больше 5 МБ")
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".jsonl":
        rows = _parse_jsonl(content)
    elif suffix == ".xlsx":
        rows = _parse_xlsx(content)
    else:
        raise DatasetError("поддерживаются только .jsonl и .xlsx")
    if not rows:
        raise DatasetError("в файле нет вопросов")
    if len(rows) > MAX_ROWS:
        raise DatasetError(f"в файле больше {MAX_ROWS} вопросов")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise DatasetError("идентификаторы вопросов должны быть уникальными")
    return rows


def save_upload(root: Path, filename: str, content: bytes) -> dict[str, Any]:
    rows = parse_upload(filename, content)
    digest = hashlib.sha256(content).hexdigest()[:12]
    label = Path(filename).stem.strip()[:80] or "Набор вопросов"
    slug = re.sub(r"[^\w.-]+", "-", label, flags=re.UNICODE).strip("-._")[:80]
    slug = slug or "questions"
    target_dir = datasets_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"dataset-{digest}--{slug}.jsonl"
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return dataset_info(target, source="upload", label=label)


def dataset_info(path: Path, *, source: str, label: str | None = None) -> dict[str, Any]:
    rows = 0
    with_truth = 0
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows += 1
            try:
                if _text(json.loads(line).get("ground_truth")):
                    with_truth += 1
            except (json.JSONDecodeError, AttributeError):
                pass
    except OSError:
        pass
    return {
        "id": f"{source}:{path.name}",
        "name": label or path.stem,
        "filename": path.name,
        "source": source,
        "rows": rows,
        "with_ground_truth": with_truth,
    }


def list_datasets(root: Path, harness: Path | None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if harness is not None:
        for path in sorted(harness.glob("*.jsonl")):
            result.append(dataset_info(path, source="builtin"))
    upload_dir = datasets_dir(root)
    if upload_dir.is_dir():
        for path in sorted(upload_dir.glob("dataset-*.jsonl")):
            stored_label = path.stem.split("--", 1)[1] if "--" in path.stem else path.stem
            result.append(dataset_info(path, source="upload", label=stored_label))
    return result


def resolve_dataset(root: Path, harness: Path | None, dataset_id: str) -> Path | None:
    for info in list_datasets(root, harness):
        if info["id"] != dataset_id:
            continue
        base = harness if info["source"] == "builtin" else datasets_dir(root)
        candidate = base / info["filename"] if base is not None else None
        if candidate is not None and candidate.is_file():
            return candidate
    return None
