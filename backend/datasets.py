import csv
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import DATASET_FILES_DIR, DATASET_META_DIR, ensure_runtime_dirs


SUPPORTED_SUFFIXES = {".csv", ".jsonl", ".json", ".txt"}
GOAL_FIELDS = ("goal", "prompt", "question", "behavior", "instruction", "task")
TARGET_FIELDS = ("target_str", "target", "target_prefix", "target_response", "response_prefix")
CATEGORY_FIELDS = ("category", "label", "type", "class")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(name: str) -> str:
    stem = Path(name).stem or "dataset"
    stem = re.sub(r"[^\w\-.一-龥]+", "_", stem, flags=re.UNICODE).strip("._")
    return stem[:80] or "dataset"


def _normalise_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def _pick(raw: Dict[str, Any], fields: Iterable[str]) -> str:
    normalised = {_normalise_key(str(k)): v for k, v in raw.items()}
    for field in fields:
        value = normalised.get(_normalise_key(field))
        if value is not None:
            return str(value).strip()
    return ""


def _meta_path(dataset_id: str) -> Path:
    return DATASET_META_DIR / f"{dataset_id}.json"


def _write_meta(meta: Dict[str, Any]) -> None:
    path = _meta_path(meta["id"])
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_meta(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_csv(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        for row in reader:
            yield dict(row)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield value
            else:
                yield {"goal": str(value)}


def _iter_json(path: Path) -> Iterable[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    if isinstance(data, dict):
        for key in ("rows", "data", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError("JSON dataset must be an object, an array, or contain rows/data/items.")
    for item in data:
        if isinstance(item, dict):
            yield item
        else:
            yield {"goal": str(item)}


def _iter_txt(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield {"goal": line}


def _iter_raw_rows(path: Path) -> Iterable[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from _iter_csv(path)
    elif suffix == ".jsonl":
        yield from _iter_jsonl(path)
    elif suffix == ".json":
        yield from _iter_json(path)
    elif suffix == ".txt":
        yield from _iter_txt(path)
    else:
        raise ValueError(f"Unsupported dataset suffix: {suffix}")


def _normalise_row(index: int, raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "index": index,
        "goal": _pick(raw, GOAL_FIELDS),
        "target_str": _pick(raw, TARGET_FIELDS),
        "category": _pick(raw, CATEGORY_FIELDS),
        "raw": raw,
    }


def _inspect_dataset(path: Path) -> Dict[str, Any]:
    rows = 0
    columns: List[str] = []
    for row in _iter_raw_rows(path):
        rows += 1
        if not columns:
            columns = [str(key) for key in row.keys()]
    return {"rows": rows, "columns": columns}


def create_dataset(original_name: str, content: bytes) -> Dict[str, Any]:
    ensure_runtime_dirs()
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported dataset type: {suffix}. Use CSV, JSONL, JSON, or TXT.")
    if not content:
        raise ValueError("Dataset file is empty.")

    dataset_id = uuid.uuid4().hex[:12]
    filename = f"{dataset_id}_{_safe_filename(original_name)}{suffix}"
    file_path = DATASET_FILES_DIR / filename
    file_path.write_bytes(content)

    try:
        inspected = _inspect_dataset(file_path)
    except Exception:
        file_path.unlink(missing_ok=True)
        raise

    meta = {
        "id": dataset_id,
        "name": original_name,
        "filename": filename,
        "file_path": str(file_path),
        "suffix": suffix,
        "size": len(content),
        "rows": inspected["rows"],
        "columns": inspected["columns"],
        "created_at": _now(),
    }
    _write_meta(meta)
    return meta


def list_datasets() -> List[Dict[str, Any]]:
    ensure_runtime_dirs()
    datasets = []
    for path in DATASET_META_DIR.glob("*.json"):
        try:
            datasets.append(_read_meta(path))
        except Exception:
            continue
    return sorted(datasets, key=lambda item: item.get("created_at", ""), reverse=True)


def get_dataset(dataset_id: str) -> Dict[str, Any]:
    path = _meta_path(dataset_id)
    if not path.exists():
        raise KeyError(dataset_id)
    return _read_meta(path)


def read_dataset_rows(dataset_id: str, offset: int = 0, limit: int = 50) -> Dict[str, Any]:
    meta = get_dataset(dataset_id)
    path = Path(meta["file_path"])
    if not path.exists():
        raise FileNotFoundError(path)

    rows = []
    for index, raw in enumerate(_iter_raw_rows(path)):
        if index < offset:
            continue
        if len(rows) >= limit:
            break
        rows.append(_normalise_row(index, raw))

    return {"dataset": meta, "offset": offset, "limit": limit, "total": meta.get("rows", 0), "rows": rows}


def get_dataset_row(dataset_id: str, row_index: int) -> Dict[str, Any]:
    if row_index < 0:
        raise IndexError(row_index)
    meta = get_dataset(dataset_id)
    path = Path(meta["file_path"])
    for index, raw in enumerate(_iter_raw_rows(path)):
        if index == row_index:
            return _normalise_row(index, raw)
    raise IndexError(row_index)


def delete_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    meta = get_dataset(dataset_id)
    Path(meta["file_path"]).unlink(missing_ok=True)
    _meta_path(dataset_id).unlink(missing_ok=True)
    return meta
