from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TAP_DIR = ROOT_DIR / "TAP"
FRONTEND_DIR = ROOT_DIR / "frontend"
DATASETS_DIR = ROOT_DIR / "datasets"
DATASET_FILES_DIR = DATASETS_DIR / "files"
DATASET_META_DIR = DATASETS_DIR / "meta"
RUNS_DIR = ROOT_DIR / "platform_runs"


def ensure_runtime_dirs() -> None:
    for path in (DATASETS_DIR, DATASET_FILES_DIR, DATASET_META_DIR, RUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)
