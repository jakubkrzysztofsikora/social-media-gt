"""One-request solver worker: JSON on stdin, JSON on stdout.

Isolates PyGambit, SymPy LP, SciPy estimation, and OpenSpiel from the app
process. The parent supplies a stripped environment; timeout, signal,
malformed response, or nonzero exit maps to solver_unavailable upstream.
"""
from __future__ import annotations

import json
import sys


def handle(request: dict) -> dict:
    op = request.get("op")
    if op == "solve_game":
        import advisor
        return advisor.solve_game(request["game_spec"],
                                  confirmed_by_user=True,
                                  methods=request.get("methods"),
                                  epsilon=request.get("epsilon"),
                                  mediator_confirmed=bool(request.get("mediator_confirmed")))
    if op == "estimate":
        import advisor
        envelope = advisor._build_envelope_from_payload(request["game_spec"])
        result = advisor.estimate_game_parameters(envelope, request["observations"],
                                                  request["method"], confirmed_by_user=True)
        return {"status": getattr(result, "status", "ok"),
                "data": _dataclass_to_json(result)}
    if op == "sensitivity":
        import advisor
        envelope = advisor._build_envelope_from_payload(request["game_spec"])
        return advisor.analyze_sensitivity(envelope, request["assignments"],
                                           confirmed_by_user=True,
                                           budget_seconds=request.get("budget_seconds", 300.0))
    if op == "simulate":
        import advisor
        envelope = advisor._build_envelope_from_payload(request["game_spec"])
        sim = request.get("simulation") or {}
        return advisor.simulate_game(envelope, sim.get("algorithm", "simulation"),
                                     sim.get("policies") or {}, sim.get("episodes", 1),
                                     sim.get("seed", 0), confirmed_by_user=True,
                                     iterations=sim.get("iterations"))
    return {"status": "error", "error": f"unknown op {op!r}"}


def _dataclass_to_json(result):
    if hasattr(result, "__dict__"):
        return {k: v for k, v in vars(result).items()}
    return result


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:
        json.dump({"status": "error", "error": f"malformed request: {exc}"}, sys.stdout)
        return 1
    try:
        response = handle(request)
    except Exception as exc:
        response = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    json.dump(response, sys.stdout, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
