"""Worker fixtures: happy path, timeout, busy gate, error mapping."""
import json
import subprocess
import sys
import threading

import pytest

import solver_worker


BARGAINING = {"template_id": "bargaining",
              "template_options": {"players": ["a", "b"],
                                   "feasible_points": [{"a": 0, "b": 3}, {"a": 3, "b": 0},
                                                       {"a": 1, "b": 1}],
                                   "disagreement": {"a": 0, "b": 0},
                                   "randomization": "not_allowed"}}


def test_worker_child_roundtrip():
    proc = subprocess.run([sys.executable, "game_solver_worker.py"],
                          input=json.dumps({"op": "solve_game", "game_spec": BARGAINING}),
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0
    response = json.loads(proc.stdout)
    assert response["disposition"] == "ok"
    assert response["method_results"][0]["method"] == "nash_bargaining"


def test_worker_child_unknown_op():
    proc = subprocess.run([sys.executable, "game_solver_worker.py"],
                          input=json.dumps({"op": "nonsense"}),
                          capture_output=True, text=True, timeout=30)
    response = json.loads(proc.stdout)
    assert response["status"] == "error"


def test_worker_child_malformed_stdin():
    proc = subprocess.run([sys.executable, "game_solver_worker.py"],
                          input="not json at all{",
                          capture_output=True, text=True, timeout=30)
    response = json.loads(proc.stdout)
    assert response["status"] == "error"
    assert "malformed" in response["error"]


def test_parent_returns_structured_busy():
    gate = threading.Event()

    def slow_child(request, timeout):
        gate.wait(5)
        return {"status": "ok"}

    first = {"op": "solve_game", "game_spec": BARGAINING}
    original = solver_worker._run_child
    solver_worker._run_child = slow_child
    try:
        executor = solver_worker.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(solver_worker.run_worker_request, first)
        second = solver_worker.run_worker_request(first)
        assert second["status"] == "solver_unavailable"
        assert second["error"] == "busy"
        gate.set()
        assert future.result()["status"] == "ok"
        executor.shutdown(wait=True)
    finally:
        solver_worker._run_child = original
        solver_worker._busy = False


def test_timeout_maps_to_solver_unavailable(monkeypatch, tmp_path):
    sleeping = tmp_path / "sleep.py"
    sleeping.write_text("import time; time.sleep(30)")
    monkeypatch.setattr(solver_worker, "WORKER_SCRIPT", sleeping)
    monkeypatch.setenv("GT_EXACT_SOLVER_TIMEOUT_SECONDS", "1")
    out = solver_worker.run_worker_request({"op": "solve_game", "game_spec": {}})
    assert out["status"] == "solver_unavailable"
    assert "timeout" in out["error"]
