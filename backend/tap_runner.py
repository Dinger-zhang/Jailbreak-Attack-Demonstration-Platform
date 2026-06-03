import json
import math
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import RUNS_DIR, TAP_DIR, ensure_runtime_dirs


RUNNING_STATUSES = {"queued", "running", "cancelling"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_for_json(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


class JobManager:
    def __init__(self) -> None:
        ensure_runtime_dirs()
        self._lock = threading.RLock()
        self._processes: Dict[str, subprocess.Popen] = {}
        self._mark_stale_jobs()

    def _job_dir(self, job_id: str) -> Path:
        return RUNS_DIR / job_id

    def _metadata_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "metadata.json"

    def _log_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "tap.log"

    def _read_meta(self, job_id: str) -> Dict[str, Any]:
        path = self._metadata_path(job_id)
        if not path.exists():
            raise KeyError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_meta(self, meta: Dict[str, Any]) -> None:
        path = self._metadata_path(meta["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(_clean_for_json(meta), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _append_log(self, job_id: str, text: str) -> None:
        path = self._log_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(text)

    def _mark_stale_jobs(self) -> None:
        for meta_path in RUNS_DIR.glob("*/metadata.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if meta.get("status") in RUNNING_STATUSES:
                meta["status"] = "stale"
                meta["finished_at"] = _now()
                meta["error"] = "Backend restarted before this job finished."
                self._write_meta(meta)

    def _build_command(self, config: Dict[str, Any], iter_index: int, result_dir: Path) -> List[str]:
        command = [
            sys.executable,
            str(TAP_DIR / "main_TAP.py"),
            "--attack-model",
            config["attack_model"],
            "--target-model",
            config["target_model"],
            "--evaluator-model",
            config["evaluator_model"],
            "--goal",
            config["goal"],
            "--target-str",
            config["target_str"],
            "--store-folder",
            str(result_dir),
            "--iter-index",
            str(iter_index),
            "--branching-factor",
            str(config["branching_factor"]),
            "--width",
            str(config["width"]),
            "--depth",
            str(config["depth"]),
            "--n-streams",
            str(config["n_streams"]),
            "--keep-last-n",
            str(config["keep_last_n"]),
            "--attack-max-n-tokens",
            str(config["attack_max_n_tokens"]),
            "--target-max-n-tokens",
            str(config["target_max_n_tokens"]),
            "--evaluator-max-n-tokens",
            str(config["evaluator_max_n_tokens"]),
            "--evaluator-temperature",
            str(config["evaluator_temperature"]),
            "--max-n-attack-attempts",
            str(config["max_n_attack_attempts"]),
            "--index",
            str(config["index"]),
            "--category",
            config["category"],
        ]
        return command

    def _subprocess_env(self, config: Dict[str, Any]) -> Dict[str, str]:
        env = os.environ.copy()
        env.setdefault("WANDB_MODE", "offline")
        env.setdefault("WANDB_SILENT", "true")

        if config.get("custom_api_url"):
            env["CUSTOM_API_URL"] = config["custom_api_url"]
        if config.get("custom_model_name"):
            env["CUSTOM_MODEL_NAME"] = config["custom_model_name"]
        if config.get("custom_api_token"):
            env["CUSTOM_API_TOKEN"] = config["custom_api_token"]

        return env

    def _redacted_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        redacted = dict(config)
        token = redacted.pop("custom_api_token", "")
        redacted["custom_api_token_provided"] = bool(token) or bool(os.getenv("CUSTOM_API_TOKEN"))
        return redacted

    def start_tap_job(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if not TAP_DIR.exists():
            raise FileNotFoundError(f"TAP directory not found: {TAP_DIR}")

        job_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job_dir = self._job_dir(job_id)
        result_dir = job_dir / "tap_results"
        job_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        iter_index = int(time.time() * 1000)
        command = self._build_command(config, iter_index, result_dir)
        meta = {
            "id": job_id,
            "method": "tap",
            "status": "queued",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "exit_code": None,
            "iter_index": iter_index,
            "job_dir": str(job_dir),
            "result_dir": str(result_dir),
            "result_file": str(result_dir / f"iter_{iter_index}_df"),
            "log_file": str(self._log_path(job_id)),
            "command": command,
            "config": self._redacted_config(config),
            "dataset": config.get("dataset"),
            "error": None,
            "cancel_requested": False,
        }
        self._write_meta(meta)
        self._append_log(job_id, f"[{_now()}] queued TAP job {job_id}\n")
        self._append_log(job_id, "[command] " + " ".join(command) + "\n\n")

        try:
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            process = subprocess.Popen(
                command,
                cwd=str(TAP_DIR),
                env=self._subprocess_env(config),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except Exception as exc:
            meta["status"] = "failed"
            meta["finished_at"] = _now()
            meta["error"] = str(exc)
            self._write_meta(meta)
            self._append_log(job_id, f"[{_now()}] failed to start: {exc}\n")
            return meta

        with self._lock:
            self._processes[job_id] = process

        meta["status"] = "running"
        meta["started_at"] = _now()
        meta["pid"] = process.pid
        self._write_meta(meta)
        self._append_log(job_id, f"[{_now()}] started pid={process.pid}\n\n")

        thread = threading.Thread(target=self._watch_process, args=(job_id, process), daemon=True)
        thread.start()
        return meta

    def _watch_process(self, job_id: str, process: subprocess.Popen) -> None:
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    self._append_log(job_id, line)
            exit_code = process.wait()
            with self._lock:
                meta = self._read_meta(job_id)
                meta["exit_code"] = exit_code
                meta["finished_at"] = _now()
                if meta.get("cancel_requested"):
                    meta["status"] = "cancelled"
                else:
                    meta["status"] = "succeeded" if exit_code == 0 else "failed"
                self._write_meta(meta)
                self._processes.pop(job_id, None)
            self._append_log(job_id, f"\n[{_now()}] process exited with code {exit_code}\n")
        except Exception as exc:
            with self._lock:
                try:
                    meta = self._read_meta(job_id)
                    meta["status"] = "failed"
                    meta["finished_at"] = _now()
                    meta["error"] = str(exc)
                    self._write_meta(meta)
                except Exception:
                    pass
                self._processes.pop(job_id, None)
            self._append_log(job_id, f"\n[{_now()}] watcher error: {exc}\n")

    def list_jobs(self) -> List[Dict[str, Any]]:
        ensure_runtime_dirs()
        jobs = []
        for path in RUNS_DIR.glob("*/metadata.json"):
            try:
                jobs.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        return self._read_meta(job_id)

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            meta = self._read_meta(job_id)
            process = self._processes.get(job_id)
            if meta.get("status") not in RUNNING_STATUSES:
                return meta
            meta["cancel_requested"] = True
            meta["status"] = "cancelling"
            self._write_meta(meta)

        self._append_log(job_id, f"\n[{_now()}] cancellation requested\n")
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        return self._read_meta(job_id)

    def get_logs(self, job_id: str, tail: int = 300) -> Dict[str, Any]:
        self._read_meta(job_id)
        path = self._log_path(job_id)
        if not path.exists():
            return {"job_id": job_id, "lines": []}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"job_id": job_id, "lines": lines[-tail:]}

    def load_results(self, job_id: str, limit: int = 200) -> Dict[str, Any]:
        meta = self._read_meta(job_id)
        result_file = Path(meta["result_file"])
        if not result_file.exists():
            return {
                "job": meta,
                "available": False,
                "message": "Result file is not available yet.",
                "rows": [],
                "summary": {},
            }

        import pandas as pd

        df = pd.read_parquet(result_file)
        rows = json.loads(df.head(limit).to_json(orient="records", force_ascii=False))
        summary: Dict[str, Any] = {
            "total_rows": int(len(df)),
            "returned_rows": int(len(rows)),
            "columns": list(df.columns),
        }

        best: Optional[Dict[str, Any]] = None
        if "judge_scores" in df.columns and len(df) > 0:
            scores = pd.to_numeric(df["judge_scores"], errors="coerce")
            if scores.notna().any():
                best_index = scores.idxmax()
                best = json.loads(df.loc[[best_index]].to_json(orient="records", force_ascii=False))[0]
                summary["max_judge_score"] = int(scores.max())
                summary["mean_judge_score"] = float(scores.mean())
                summary["success_count"] = int((scores == 10).sum())
        summary["best"] = best

        return {"job": meta, "available": True, "rows": rows, "summary": _clean_for_json(summary)}

    def export_results_csv(self, job_id: str) -> Path:
        meta = self._read_meta(job_id)
        result_file = Path(meta["result_file"])
        if not result_file.exists():
            raise FileNotFoundError(result_file)

        import pandas as pd

        csv_path = self._job_dir(job_id) / "tap_results.csv"
        pd.read_parquet(result_file).to_csv(csv_path, index=False, encoding="utf-8-sig")
        return csv_path
