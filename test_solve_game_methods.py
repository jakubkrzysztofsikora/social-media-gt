import advisor
import game_solvers
from game_solvers import SolverError

SPEC = {
    "kind": "normal_form",
    "players": ["a", "b"],
    "actions": {"a": ["x", "y"], "b": ["x", "y"]},
    "payoffs": [
        {"action_profile": {"a": "x", "b": "x"}, "utilities": {"a": 2, "b": 2}},
        {"action_profile": {"a": "x", "b": "y"}, "utilities": {"a": 0, "b": 0}},
        {"action_profile": {"a": "y", "b": "x"}, "utilities": {"a": 0, "b": 0}},
        {"action_profile": {"a": "y", "b": "y"}, "utilities": {"a": 1, "b": 1}},
    ],
}


def _explode(*args, **kwargs):
    raise SolverError("unrequested solver ran")


def test_unrequested_method_failure_does_not_discard_requested_results(monkeypatch):
    monkeypatch.setattr(game_solvers, "solve_mixed_nash_two_player", _explode)
    monkeypatch.setattr(game_solvers, "solve_correlated_equilibrium", _explode)

    result = advisor.solve_game(SPEC, True, methods=["pure_nash"])

    assert result["disposition"] == "ok"
    assert [row["method"] for row in result["method_results"]] == ["pure_nash"]
    assert result["method_results"][0]["status"] == "exact_verified"


def test_requested_method_failure_is_still_reported(monkeypatch):
    monkeypatch.setattr(game_solvers, "solve_mixed_nash_two_player", _explode)

    result = advisor.solve_game(SPEC, True, methods=["pure_nash", "mixed_nash"])

    errors = [row for row in result["method_results"] if row.get("status") == "error"]
    assert errors and "unrequested solver ran" in errors[0]["error"]


def test_analysis_result_is_json_serialisable_for_every_exact_method():
    import json

    result = advisor.solve_game(SPEC, True, methods=["pure_nash", "mixed_nash",
                                                     "correlated_equilibrium"])

    json.dumps(result)
    assert all(isinstance(row["status"], str) for row in result["method_results"])


def test_correlated_equilibrium_needs_mediator_confirmation():
    refused = advisor.solve_game(SPEC, True, methods=["correlated_equilibrium"])
    assert refused["method_results"][0]["status"] == "unsupported"

    allowed = advisor.solve_game(SPEC, True, methods=["correlated_equilibrium"],
                                 mediator_confirmed=True)
    assert allowed["method_results"][0]["status"] == "exact_verified"


def test_mcp_solve_game_exposes_mediator_confirmation():
    import json
    import mcp_server

    payload = {"kind": "normal_form", "players": ["a", "b"],
               "actions": {"a": ["x", "y"], "b": ["x", "y"]},
               "payoffs": SPEC["payoffs"]}

    refused = mcp_server.solve_game(payload, True, ["correlated_equilibrium"])
    assert refused["method_results"][0]["status"] == "unsupported"

    allowed = mcp_server.solve_game(payload, True, ["correlated_equilibrium"],
                                    mediator_confirmed=True)
    json.dumps(allowed)
    assert allowed["method_results"][0]["status"] == "exact_verified"


def test_webapp_forwards_mediator_confirmation_to_worker(monkeypatch):
    import json
    import webapp

    seen = {}

    def fake_worker(request):
        seen.update(request)
        return {"disposition": "ok", "method_results": []}

    monkeypatch.setattr(webapp, "run_worker_request", fake_worker)
    monkeypatch.setattr(webapp.plain_view, "describe_result", lambda *a, **k: "")

    webapp.game_solve(json.dumps(SPEC), True, True)

    assert seen["mediator_confirmed"] is True
