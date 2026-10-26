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
