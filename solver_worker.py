"""Parent-side worker execution: one active child, nonblocking busy gate.

Blocking subprocess.run executes on a dedicated single-thread executor so
Gradio/MCP event loops never stall. The child environment is stripped of
LLM, app, and encryption secrets.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORKER_SCRIPT = Path(__file__).with_name("game_solver_worker.py")

_STRIPPED_PREFIXES = ("GT_LLM_", "GT_APP_", "GT_CASE_", "GT_SHARE_", "LITELLM_", "OPENAI_")
_DEFAULT_TIMEOUTS = {
    "solve_game": 30.0,
    "estimate": 120.0,
    "sensitivity": 120.0,
    "simulate": 120.0,
}
_BUDGET_SECONDS = 300.0

_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None
_busy = False


def _timeout_for(op: str) -> float:
    import os
    env_map = {
        "solve_game": "GT_EXACT_SOLVER_TIMEOUT_SECONDS",
        "estimate": "GT_ESTIMATION_TIMEOUT_SECONDS",
        "sensitivity": "GT_APPROX_SOLVER_TIMEOUT_SECONDS",
        "simulate": "GT_APPROX_SOLVER_TIMEOUT_SECONDS",
    }
    raw = os.environ.get(env_map.get(op, ""), "")
    try:
        return float(raw) if raw else _DEFAULT_TIMEOUTS.get(op, 30.0)
    except ValueError:
        return _DEFAULT_TIMEOUTS.get(op, 30.0)


def _child_environ() -> dict:
    env = {k: v for k, v in os.environ.items()
           if not any(k.startswith(p) for p in _STRIPPED_PREFIXES)}
    return env


def _run_child(request: dict, timeout: float) -> dict:
    global _busy
    proc = subprocess.run(
        [sys.executable, str(WORKER_SCRIPT)],
        input=json.dumps(request).encode("utf-8"),
        capture_output=True, timeout=timeout, env=_child_environ())
    if proc.returncode != 0:
        return {"status": "solver_unavailable",
                "error": f"worker exited {proc.returncode}: {proc.stderr[-400:]}"}
    try:
        response = json.loads(proc.stdout.decode("utf-8"))
    except Exception as exc:
        return {"status": "solver_unavailable", "error": f"malformed worker output: {exc}"}
    if isinstance(response, dict) and response.get("status") == "error":
        response = {"status": "error", "error": response.get("error", "worker error")}
    return response


def run_worker_request(request: dict) -> dict:
    """Submit one request; concurrent requests get busy without blocking."""
    global _busy, _executor
    op = request.get("op", "solve_game")
    with _lock:
        if _busy:
            return {"status": "solver_unavailable", "error": "busy"}
        _busy = True
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gt-worker")
    try:
        future = _executor.submit(_run_child, request, _timeout_for(op))
        return future.result()
    except subprocess.TimeoutExpired:
        return {"status": "solver_unavailable", "error": f"worker timeout after {_timeout_for(op)}s"}
    except Exception as exc:
        return {"status": "solver_unavailable", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        with _lock:
            _busy = False
