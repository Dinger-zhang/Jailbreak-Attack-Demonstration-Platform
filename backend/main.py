import argparse
import importlib.util
import os
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import datasets
from .config import FRONTEND_DIR, ROOT_DIR, TAP_DIR, ensure_runtime_dirs
from .tap_runner import JobManager


ATTACK_MODELS = [
    "custom-api-model",
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "vicuna",
    "vicuna-api-model",
    "llama-2-api-model",
]
TARGET_MODELS = [
    "custom-api-model",
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "vicuna",
    "vicuna-api-model",
    "llama-2",
    "llama-2-api-model",
    "palm-2",
    "gemini-pro",
]
EVALUATOR_MODELS = [
    "custom-api-model",
    "no-evaluator",
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
]
GPT_MODELS = {"gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4-1106-preview"}


class RunRequest(BaseModel):
    method: str = "tap"
    goal: str = ""
    target_str: str = ""
    dataset_id: Optional[str] = None
    dataset_row_index: Optional[int] = None
    attack_model: str = "custom-api-model"
    target_model: str = "custom-api-model"
    evaluator_model: str = "custom-api-model"
    custom_api_url: str = ""
    custom_model_name: str = ""
    custom_api_token: str = ""
    depth: int = Field(default=1, ge=1, le=20)
    width: int = Field(default=1, ge=1, le=50)
    branching_factor: int = Field(default=1, ge=1, le=20)
    n_streams: int = Field(default=1, ge=1, le=50)
    keep_last_n: int = Field(default=3, ge=1, le=20)
    attack_max_n_tokens: int = Field(default=500, ge=16, le=4096)
    target_max_n_tokens: int = Field(default=150, ge=16, le=4096)
    evaluator_max_n_tokens: int = Field(default=10, ge=1, le=512)
    evaluator_temperature: float = Field(default=0, ge=0, le=2)
    max_n_attack_attempts: int = Field(default=5, ge=1, le=20)
    category: str = "manual"
    index: Optional[int] = None


ensure_runtime_dirs()
app = FastAPI(title="Jailbreak Attack Demonstration Platform", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
manager = JobManager()


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _request_dict(request: RunRequest) -> Dict[str, Any]:
    if hasattr(request, "model_dump"):
        return request.model_dump()
    return request.dict()


def _validate_model_choice(name: str, allowed: list, label: str) -> None:
    if name not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported {label}: {name}")


def _resolve_run_config(request: RunRequest) -> Dict[str, Any]:
    config = _request_dict(request)
    if config["method"] != "tap":
        raise HTTPException(status_code=400, detail="Only TAP is currently registered as an attack method.")

    _validate_model_choice(config["attack_model"], ATTACK_MODELS, "attack model")
    _validate_model_choice(config["target_model"], TARGET_MODELS, "target model")
    _validate_model_choice(config["evaluator_model"], EVALUATOR_MODELS, "evaluator model")

    dataset_context = None
    if config.get("dataset_id") is not None:
        if config.get("dataset_row_index") is None:
            raise HTTPException(status_code=400, detail="dataset_row_index is required when dataset_id is provided.")
        try:
            row = datasets.get_dataset_row(config["dataset_id"], int(config["dataset_row_index"]))
            meta = datasets.get_dataset(config["dataset_id"])
        except KeyError:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        except IndexError:
            raise HTTPException(status_code=404, detail="Dataset row not found.")
        dataset_context = {"dataset_id": meta["id"], "name": meta["name"], "row": row}
        if not config.get("goal"):
            config["goal"] = row.get("goal", "")
        if not config.get("target_str"):
            config["target_str"] = row.get("target_str", "")
        if config.get("category") == "manual" and row.get("category"):
            config["category"] = row["category"]
        if config.get("index") is None:
            config["index"] = row["index"]

    config["goal"] = str(config.get("goal") or "").strip()
    config["target_str"] = str(config.get("target_str") or "").strip()
    config["category"] = str(config.get("category") or "manual").strip() or "manual"
    config["index"] = int(config["index"] if config.get("index") is not None else 0)
    config["dataset"] = dataset_context

    if not config["goal"]:
        raise HTTPException(status_code=400, detail="goal is required.")
    if not config["target_str"]:
        raise HTTPException(status_code=400, detail="target_str is required.")

    selected_models = {config["attack_model"], config["target_model"], config["evaluator_model"]}
    if "custom-api-model" in selected_models:
        api_url = config.get("custom_api_url") or os.getenv("CUSTOM_API_URL", "")
        model_name = config.get("custom_model_name") or os.getenv("CUSTOM_MODEL_NAME", "")
        if not api_url:
            raise HTTPException(status_code=400, detail="CUSTOM_API_URL is required for custom-api-model.")
        if not model_name:
            raise HTTPException(status_code=400, detail="CUSTOM_MODEL_NAME is required for custom-api-model.")

    if selected_models & GPT_MODELS and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is required for OpenAI GPT models.")

    return config


@app.get("/")
def index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Frontend index.html not found.")
    return FileResponse(index_path)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    dependencies = {
        "fastapi": _module_available("fastapi"),
        "uvicorn": _module_available("uvicorn"),
        "pandas": _module_available("pandas"),
        "pyarrow": _module_available("pyarrow"),
        "fastparquet": _module_available("fastparquet"),
        "wandb": _module_available("wandb"),
        "openai": _module_available("openai"),
        "fastchat": _module_available("fastchat"),
        "python_multipart": _module_available("multipart"),
    }
    return {
        "ok": TAP_DIR.exists() and all(dependencies.values()),
        "root_dir": str(ROOT_DIR),
        "tap_dir": str(TAP_DIR),
        "tap_found": TAP_DIR.exists(),
        "python": sys.executable,
        "dependencies": dependencies,
        "environment": {
            "WANDB_MODE": os.getenv("WANDB_MODE", ""),
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
            "CUSTOM_API_URL": os.getenv("CUSTOM_API_URL", ""),
            "CUSTOM_MODEL_NAME": os.getenv("CUSTOM_MODEL_NAME", ""),
            "CUSTOM_API_TOKEN": bool(os.getenv("CUSTOM_API_TOKEN")),
        },
    }


@app.get("/api/methods")
def methods() -> Dict[str, Any]:
    return {
        "methods": [
            {
                "id": "tap",
                "name": "TAP",
                "description": "Tree of Attacks with Pruning; runs TAP/main_TAP.py as an isolated subprocess.",
                "attack_models": ATTACK_MODELS,
                "target_models": TARGET_MODELS,
                "evaluator_models": EVALUATOR_MODELS,
            }
        ]
    }


@app.get("/api/datasets")
def list_datasets() -> Dict[str, Any]:
    return {"datasets": datasets.list_datasets()}


@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Dataset is larger than 50 MB.")
    try:
        meta = datasets.create_dataset(file.filename or "dataset.csv", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse dataset: {exc}")
    return {"dataset": meta}


@app.get("/api/datasets/{dataset_id}/rows")
def dataset_rows(
    dataset_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
) -> Dict[str, Any]:
    try:
        return datasets.read_dataset_rows(dataset_id, offset=offset, limit=limit)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file not found.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str) -> Dict[str, Any]:
    try:
        meta = datasets.delete_dataset(dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {"deleted": meta}


@app.get("/api/jobs")
def list_jobs() -> Dict[str, Any]:
    return {"jobs": manager.list_jobs()}


@app.post("/api/jobs")
def start_job(request: RunRequest) -> Dict[str, Any]:
    config = _resolve_run_config(request)
    try:
        job = manager.start_tap_job(config)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"job": job}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    try:
        return {"job": manager.get_job(job_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found.")


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, Any]:
    try:
        return {"job": manager.cancel_job(job_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found.")


@app.get("/api/jobs/{job_id}/logs")
def job_logs(job_id: str, tail: int = Query(default=300, ge=1, le=5000)) -> Dict[str, Any]:
    try:
        return manager.get_logs(job_id, tail=tail)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found.")


@app.get("/api/jobs/{job_id}/results")
def job_results(job_id: str, limit: int = Query(default=200, ge=1, le=2000)) -> Dict[str, Any]:
    try:
        return manager.load_results(job_id, limit=limit)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load results: {exc}")


@app.get("/api/jobs/{job_id}/results.csv")
def job_results_csv(job_id: str) -> FileResponse:
    try:
        path = manager.export_results_csv(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found.")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Result file not found.")
    return FileResponse(path, media_type="text/csv", filename=f"{job_id}_tap_results.csv")


if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the jailbreak demonstration platform backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
