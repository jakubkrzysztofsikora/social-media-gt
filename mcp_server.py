"""MCP server for the social-media game-theory advisor.

Exposes the scenario catalogue, the LLM parameter inference, a single-scenario
runner and the full situation analysis.  Every payload is JSON-serializable:
fractions are rendered before they leave the process.
"""
from __future__ import annotations

import argparse

from mcp.server import MCPServer

import advisor

mcp = MCPServer("social-media-gt", log_level="WARNING")


@mcp.tool()
def list_scenarios() -> dict:
    """List the ten scenarios with their parameters and strategy rows."""
    scenarios = []
    for scenario_id, entry in advisor.CATALOG.items():
        scenarios.append({
            "id": scenario_id,
            "title": entry["title"],
            "params": entry["params"],
            "strategies": [view["strategy"]
                           for view in advisor.STRATEGY_VIEWS.get(scenario_id, [])],
        })
    return {"scenarios": scenarios}


@mcp.tool()
def infer_parameters(messages: list[dict]) -> dict:
    """Infer the toolkit's parameter values for all ten scenarios from a chat history."""
    try:
        parsed = advisor.infer_parameters(messages)
    except advisor.InferenceError as exc:
        return {"status": "error", "error": str(exc)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    scenarios = {}
    for scenario_id, entry in (parsed.get("scenarios") or {}).items():
        if isinstance(entry, dict):
            scenarios[scenario_id] = {"params": entry.get("params", {}) or {},
                                      "rationale": entry.get("rationale", "") or ""}
        else:
            scenarios[scenario_id] = {"params": {}, "rationale": ""}
    return {"status": "ok",
            "need_more_information": bool(parsed.get("need_more_information")),
            "questions": [str(question) for question in parsed.get("questions") or []],
            "situation_summary": parsed.get("situation_summary", "") or "",
            "scenarios": scenarios,
            "model": parsed.get("model", "") or "",
            "raw": parsed.get("raw", "") or ""}


@mcp.tool()
def run_scenario(scenario: str, params: dict | None = None) -> dict:
    """Run one scenario with the given parameters and return its full evidence."""
    if scenario not in advisor.CATALOG:
        return {"status": "error",
                "error": f"unknown scenario {scenario!r}; expected one of "
                         f"{', '.join(advisor.CATALOG)}"}
    return advisor.run_one_scenario(scenario, params or {})


@mcp.tool()
def analyze_situation(messages: list[dict]) -> dict:
    """Infer parameters from the chat history, run all ten scenarios and report the evidence."""
    try:
        report = advisor.analyze_situation(messages)
    except advisor.InferenceError as exc:
        return {"status": "error", "error": str(exc)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "ok", **report}


# ---------------------------------------------------------------------------
# General games (exact core + approximate evidence)
# ---------------------------------------------------------------------------

import json as _json  # noqa: E402

import case_store as _case_store_mod  # noqa: E402
from solver_worker import run_worker_request  # noqa: E402


@mcp.tool()
def list_game_templates() -> dict:
    """List the general-game templates with their typed options."""
    return {"templates": advisor.list_game_templates()}


@mcp.tool()
def infer_game_model(messages: list[dict]) -> dict:
    """LLM intake for a general game: returns clarification/unsupported/invalid
    or a draft specification that still needs explicit confirmation."""
    try:
        parsed = advisor.infer_game_model(messages)
    except advisor.InferenceError as exc:
        return {"status": "error", "error": str(exc)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "ok", "disposition": parsed.get("disposition", ""),
            "questions": parsed.get("questions", []),
            "template_id": parsed.get("template_id", ""),
            "template_options": parsed.get("template_options") or {},
            "game": parsed.get("game"),
            "parameters": parsed.get("parameters") or {},
            "raw": parsed.get("raw", ""),
            "model": parsed.get("model", "")}


@mcp.tool()
def solve_game(game_spec: dict, confirmed_by_user: bool,
               methods: list[str] | None = None, epsilon=None,
               mediator_confirmed: bool = False) -> dict:
    """Solve a canonical game spec. Exact methods by default; epsilon_nash is
    approximate and requires a positive rational epsilon. MCP callers must
    pass confirmed_by_user=true explicitly. correlated_equilibrium also needs
    mediator_confirmed=true: the plan must be externally enforced."""
    try:
        return advisor.solve_game(game_spec, confirmed_by_user, methods=methods,
                                  epsilon=epsilon,
                                  mediator_confirmed=mediator_confirmed)
    except Exception as exc:
        return {"disposition": "error", "error": f"{type(exc).__name__}: {exc}"}


@mcp.tool()
def estimate_game_parameters(game_spec: dict, observations: list,
                             method: str, confirmed_by_user: bool) -> dict:
    """Approximate estimation: conditional_logit (joint counts) or
    qre_marginals (marginal counts, known payoffs)."""
    try:
        envelope = advisor._build_envelope_from_payload(game_spec)
        result = advisor.estimate_game_parameters(envelope, observations, method,
                                                  confirmed_by_user)
        if isinstance(result, dict):
            return result
        return {"status": getattr(result, "status", "ok"),
                **{k: v for k, v in vars(result).items()}}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@mcp.tool()
def analyze_sensitivity(game_spec: dict, assignments: list,
                        confirmed_by_user: bool) -> dict:
    """Per-point exact analysis over complete rational in-bound assignments."""
    try:
        envelope = advisor._build_envelope_from_payload(game_spec)
        return advisor.analyze_sensitivity(envelope, assignments, confirmed_by_user)
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@mcp.tool()
def apply_estimates(game_spec: dict, estimate_result: dict, parameter_ids: list,
                    confirmed_by_user: bool) -> dict:
    """Copy accepted displayed decimal estimates into point parameters
    (origin=estimated); never mutates the supplied spec. Solving the updated
    spec needs another explicit confirmation."""
    try:
        envelope = advisor._build_envelope_from_payload(game_spec)
        result = advisor.apply_estimates(envelope, estimate_result, parameter_ids,
                                         confirmed_by_user)
        if isinstance(result, dict) and result.get("status") == "ok":
            envelope = result.pop("envelope")
            return {"status": "ok", "updated_game_spec": {
                "kind": _custom_kind(envelope.game),
                "game": _game_to_payload(envelope.game),
                "parameters": {name: {"value": _render(spec.value),
                                      "provenance": spec.provenance}
                               for name, spec in envelope.parameters.items()},
            }, "provenance": result.get("provenance", {})}
        return result
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _custom_kind(game) -> str:
    from game_models import (BargainingSpec, BayesianGame, ExtensiveFormGame,
                             MechanismSpec, NormalFormGame, OpenSpielGame,
                             RepeatedGame, StackelbergGame)
    return {NormalFormGame: "normal_form", ExtensiveFormGame: "extensive_form",
            BayesianGame: "bayesian", StackelbergGame: "stackelberg",
            RepeatedGame: "repeated", BargainingSpec: "bargaining",
            MechanismSpec: "mechanism", OpenSpielGame: "open_spiel"}[type(game)]


def _render(value):
    from game_models import render_rational
    return render_rational(value) if hasattr(value, "numerator") else str(value)


def _game_to_payload(game) -> dict:
    from game_models import (BayesianGame, ExtensiveFormGame, NormalFormGame)
    if isinstance(game, NormalFormGame):
        return {"players": [p.id for p in game.players], "actions": dict(game.actions),
                "payoffs": game.payoffs}
    if isinstance(game, BayesianGame):
        return {"players": [p.id for p in game.players], "type_spaces": dict(game.type_spaces),
                "prior": game.prior, "actions": dict(game.actions), "payoffs": game.payoffs}
    if isinstance(game, ExtensiveFormGame):
        return {"players": [p.id for p in game.players], "root": game.root,
                "nodes": [{"id": n.id, "actor": n.actor, "edges": n.edges,
                           "information_set_id": n.information_set_id,
                           "terminal_utilities": n.terminal_utilities} for n in game.nodes],
                "information_sets": game.information_sets}
    raise ValueError(f"no JSON payload form for {type(game).__name__}")


@mcp.tool()
def simulate_game(game_spec: dict, algorithm: str, policies: dict,
                  episodes: int, seed: int, confirmed_by_user: bool,
                  iterations: int | None = None) -> dict:
    """Approximate simulation: seeded episode play or CFR/MCCFR on the
    OpenSpiel allowlist. Never an equilibrium or convergence claim."""
    try:
        envelope = advisor._build_envelope_from_payload(game_spec)
        return advisor.simulate_game(envelope, algorithm, policies, episodes, seed,
                                     confirmed_by_user, iterations=iterations)
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _case_store():
    key = _case_store_mod.load_key_from_env()
    import os
    path = os.environ.get("GT_CASE_DB_PATH", "game-cases.sqlite3")
    return _case_store_mod.CaseStore(path, key)


@mcp.tool()
def save_case(title: str, analysis_request: dict, confirmed_by_user: bool) -> dict:
    """Re-run the confirmed analysis server-side and store it encrypted.
    Confirmation is checked before any solver work."""
    if not confirmed_by_user:
        return {"status": "needs_confirmation"}
    try:
        report = run_worker_request({"op": "solve_game",
                                     "game_spec": analysis_request.get("game_spec", {})})
        store = _case_store()
        return store.save_case(title, analysis_request, report, True)
    except _case_store_mod.CaseKeyError as exc:
        return {"status": "error", "error": str(exc)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@mcp.tool()
def list_cases() -> dict:
    """List saved case summaries (encrypted contents stay at rest)."""
    try:
        return {"cases": _case_store().list_cases()}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@mcp.tool()
def load_case(case_id: str) -> dict:
    """Load and decrypt one saved case."""
    try:
        return _case_store().load_case(case_id)
    except _case_store_mod.CaseNotFound:
        return {"status": "not_found"}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@mcp.tool()
def delete_case(case_id: str, confirmed_by_user: bool) -> dict:
    """Delete one saved case; requires explicit confirmation."""
    try:
        return _case_store().delete_case(case_id, confirmed_by_user)
    except _case_store_mod.CaseNotFound:
        return {"status": "not_found"}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
