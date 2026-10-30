"""Unit tests for the social-media-gt advisor: core, MCP server and web app."""
import asyncio
import inspect
import json
import os
import re
import subprocess
import sys
from fractions import Fraction as F
from types import SimpleNamespace

import pytest

import advisor
import social_media_gt as gt

SCENARIO_IDS = list(gt.SCENARIOS)
INTERNAL = {"tw1_1": {"k"}, "tw8": {"full"}}


class _StubCompletions:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.payloads.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _StubClient:
    def __init__(self, payloads):
        self.chat = SimpleNamespace(completions=_StubCompletions(payloads))


def _default_payload():
    scenarios = {}
    for scenario_id, entry in advisor.CATALOG.items():
        params = {name: spec["default"] for name, spec in entry["params"].items()}
        scenarios[scenario_id] = {"params": params, "rationale": "paper defaults"}
    return {"situation_summary": "the paper's default situation", "scenarios": scenarios}

def _expected_display(value):
    if isinstance(value, F):
        return gt.show(value)
    if isinstance(value, dict):
        return {key: _expected_display(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_expected_display(item) for item in value]
    return value


def _path_value(data, path):
    for part in path.split("."):
        data = data[part]
    return _expected_display(data)


def _assert_tw7_semantics(evidence):
    params = evidence["params"]
    effective = F(params["lam"]) * (1 - F(params["eta"]))
    assert F(evidence["results"]["lambda_effective"]) == effective

    conclusion = " ".join(evidence["conclusion"]).casefold()
    assert "fraction of baseline peer pressure removed" in conclusion
    assert "100 * (1 - lambda) * k" in conclusion
    assert "pressure slope" not in conclusion
    assert "lambda * (1 - eta)" in conclusion
    assert "neither group coverage nor evasion" in conclusion


EXPECTED_STRATEGY_LAYOUTS = {
    "tw1": [
        ("Cooperate together (all-R)", "payoff of staying out; cost of the lone deviator",
         {"payoff of staying out": "payoffs_relative_to_offline.all_R.stay",
          "payoff of the lone deviator": "payoffs_relative_to_offline.all_R.deviate"}),
        ("Use together (all-A)", "payoff of staying in; cost of leaving",
         {"payoff of staying in": "payoffs_relative_to_offline.all_A.stay",
          "payoff of leaving": "payoffs_relative_to_offline.all_A.deviate"}),
        ("Pareto comparison of the two", "all-R better exactly when tau > f1",
         {"all-R Pareto-dominates all-A": "all_r_dominates_pareto",
          "welfare gap tau - f1": "pareto_gap_all_R"}),
    ],
    "tw1_1": [
        ("Unilateral cooperation (first best)", "decision without peer effects",
         {"first-best decision t_FB": "t_first_best"}),
        ("Cooperation with peer effects (Nash)",
         "Nash optimum exceeds the first best (looser cap); welfare loss from peer coupling",
         {"Nash decision t_NE": "t_nash", "welfare loss": "welfare_loss"}),
        ("Private vs peer-aware value", "private value curve vs peer-aware curve",
         {"private value curve V(t)": "curve_V", "peer-aware curve G(t)": "curve_G"}),
    ],
    "tw2": [
        ("Uniform peer share", "mean adoption over a uniform peer share",
         {"mean adoption at a uniform peer share": "uniform_q_mean"}),
        ("Peer-pressure cascade", "limit and rounds to full adoption",
         {"cascade limit": "limit", "rounds to full adoption": "rounds_to_full_adoption",
          "Phi(k) - k on the linear region": "phi_minus_k_on_linear_region"}),
    ],
    "tw3": [
        ("All members use", "equilibrium status and the advantage of staying",
         {"all-A is a Nash equilibrium": "all_a_is_equilibrium",
          "advantage of staying at all-A (a deviation loses this much)": "delta_at_all_a"}),
        ("All Nash profiles", "number of Nash profiles",
         {"number of Nash equilibria": "equilibria_count"}),
    ],
    "tw3_2": [
        ("Cooperate jointly (Pareto better)", "trap class and welfare gap at the midpoint",
         {"Pareto trap class non-empty": "pareto_trap_class_nonempty",
          "welfare gap at the midpoint": "pareto_gap_at_midpoint"}),
        ("All-using (risk-dominant)",
         "cutoff below which all-using is risk-dominant; equality is the tie boundary",
         {"risk-dominant cutoff tau^RD": "tau_risk_dominant"}),
    ],
    "tw5": [
        ("No pact", "adopter share needed before the first outsider leaves",
         {"threshold peer share k*": "k_star",
          "pact size at which the first outsider defects": "m_out"}),
        ("Pact among aware members", "how many members hold before leaving",
         {"members needed for stability": "m_stab", "P(5/9)": "P_peer_5_of_9",
          "P(6/9)": "P_peer_6_of_9"}),
    ],
    "tw6": [
        ("No pact", "adoption returns to full after the pact ends",
         {"rounds to full adoption after release": "release_steps"}),
        ("Pact covering gamma of the total cohort", "adoption level held, contraction, trajectory, exit multiplier",
         {"adoption level K": "K", "contraction a": "a", "trajectory": "trajectory",
          "exit multiplier": "exit_multiplier"}),
    ],
    "tw7": [
        ("Pressure reduction after eta attenuation",
         "effective fraction of baseline peer pressure removed and resulting adoption limit",
         {"effective fraction removed": "lambda_effective", "adoption limit": "limit"}),
        ("Unattenuated pressure reduction (eta = 0)",
         "baseline with the full specified lambda reduction; same toolkit function with eta = 0",
         {"effective fraction removed": "lambda_effective", "adoption limit": "limit"}),
    ],
    "tw7_2": [
        ("Statutory limits", "share of the pressure reduction carried by regulation",
         {"lambda_reg": "lambda_reg"}),
        ("Product defaults", "share of the pressure reduction carried by product defaults",
         {"lambda_plat": "lambda_plat"}),
        ("Optimal hybrid", "weight ratio: regulation against product defaults",
         {"lambda_reg / lambda_plat": "ratio"}),
    ],
    "tw8": [
        ("Exit without a public schedule", "worst-case advantage and number of minimizing states",
         {"discounted advantage": "discounted_min", "minimizing states": "minimizing_states"}),
        ("Public sequential schedule", "undiscounted advantage, worst sequence, orderings checked",
         {"undiscounted advantage": "undiscounted_sum", "worst sequence": "worst_sequence",
          "worst slot": "worst_d_slot", "orderings enumerated": "orderings_enumerated"}),
    ],
}

def test_catalog_matches_the_toolkit():
    assert list(advisor.CATALOG) == SCENARIO_IDS
    for scenario_id, (func, _defaults) in gt.SCENARIOS.items():
        signature = inspect.signature(func)
        expected = {name for name in signature.parameters
                    if name not in INTERNAL.get(scenario_id, set())}
        assert set(advisor.CATALOG[scenario_id]["params"]) == expected
        assert advisor.CATALOG[scenario_id]["title"].startswith("Tw. ")


def test_to_fraction_accepts_numbers_and_rejects_junk():
    assert advisor.to_fraction(0.1) == F(1, 10)
    assert advisor.to_fraction(0.25) == F(1, 4)
    assert advisor.to_fraction("0.25") == F(1, 4)
    assert advisor.to_fraction("1/2") == F(1, 2)
    assert advisor.to_fraction(3) == F(3)
    with pytest.raises(ValueError):
        advisor.to_fraction(True)
    with pytest.raises(ValueError):
        advisor.to_fraction("abc")


def test_normalize_params_repairs_out_of_domain_input():
    values, notes = advisor.normalize_params("tw1_1", {"a": "9/5", "b": 1, "T": 1, "beta": "9/5"})
    assert values["a"] + values["beta"] == 2 * values["b"] * values["T"]
    assert values["a"] / values["beta"] == F(1)
    assert values["a"] > 0 and values["beta"] > 0
    assert any(note.startswith("a: adjusted") for note in notes)
    assert any(note.startswith("beta: adjusted") for note in notes)

    values, notes = advisor.normalize_params("tw3", {"n": 99})
    assert values["n"] == 12
    assert any(note.startswith("n: adjusted") for note in notes)

    values, notes = advisor.normalize_params("tw7", {"eta": 5})
    assert values["eta"] == F(999, 1000)
    assert any(note.startswith("eta: adjusted") for note in notes)

    values, notes = advisor.normalize_params("tw5", {"d": 99})
    assert values["d"] == values["n"] - 1
    assert any(note.startswith("d: adjusted") for note in notes)

    values, notes = advisor.normalize_params("tw5", {"tau": 0})
    assert values["tau"] == F(1, 1000)
    assert any(note.startswith("tau: adjusted") for note in notes)
    values, notes = advisor.normalize_params("tw5", {"tau": 5})
    assert values["tau"] == 3
    assert any(note.startswith("tau: adjusted") for note in notes)

    values, notes = advisor.normalize_params("tw6", {"eps": F(999, 1000), "gamma": F(999, 1000)})
    assert values["eps"] + values["gamma"] < 1
    assert any(note.startswith("gamma: adjusted") for note in notes)
    values, notes = advisor.normalize_params("tw6", {"gamma": 0})
    assert values["gamma"] == F(1, 1000)
    assert any(note.startswith("gamma: adjusted") for note in notes)

    values, notes = advisor.normalize_params("tw1_1", {"a": F(1, 2000)})
    assert values["a"] == F(1, 2000)
    assert notes == []

    values, notes = advisor.normalize_params("tw3", {"n": "22/5"})
    assert values["n"] == 4
    assert any(note.startswith("n: rounded from 22/5 to 4") for note in notes)


def test_normalize_params_drops_unusable_values_with_a_note():
    values, notes = advisor.normalize_params("tw1", {"tau": -1, "f1": "abc", "unknown": 3})
    assert values["tau"] == 0
    assert values["f1"] == F(1, 4)
    assert any(note.startswith("f1: inferred value unusable") for note in notes)
    assert any(note.startswith("unknown: inferred value unusable") for note in notes)


def test_normalize_params_keeps_paper_defaults():
    for scenario_id, (func, _defaults) in gt.SCENARIOS.items():
        values, notes = advisor.normalize_params(scenario_id, {})
        assert notes == []
        signature = inspect.signature(func)
        for name, parameter in signature.parameters.items():
            if name in INTERNAL.get(scenario_id, set()):
                continue
            assert values[name] == F(parameter.default)

def test_infer_parameters_parses_a_fenced_payload():
    payload = json.dumps(_default_payload())
    client = _StubClient([f"Here you go:\n```json\n{payload}\n```\n"])
    parsed = advisor.infer_parameters([{"role": "user", "content": "defaults"}], client=client)
    assert parsed["situation_summary"] == "the paper's default situation"
    assert set(parsed["scenarios"]) == set(SCENARIO_IDS)
    assert parsed["scenarios"]["tw1"]["params"]["tau"] == "1/2"
    assert parsed["raw"].startswith("Here you go:")
    assert parsed["model"] == advisor.load_config().model
    assert len(client.chat.completions.calls) == 1


def test_infer_parameters_retries_once_then_raises():
    client = _StubClient(["not json at all", "still not json"])
    with pytest.raises(advisor.InferenceError):
        advisor.infer_parameters([{"role": "user", "content": "defaults"}], client=client)
    assert len(client.chat.completions.calls) == 2


def test_parse_json_object_keeps_valid_escapes_and_repairs_invalid_ones():
    valid = '{"situation_summary": "line\\nbreak"}'
    assert advisor._parse_json_object(valid)["situation_summary"] == "line\nbreak"
    broken = '{"situation_summary": "stray \\x escape"}'
    assert advisor._parse_json_object(broken)["situation_summary"] == "stray \\x escape"


def test_mcp_tools_report_inference_failures_instead_of_raising(monkeypatch):
    import mcp_server

    def boom(messages, client=None, config=None):
        raise advisor.InferenceError("LLM request failed (401): unauthorized")

    monkeypatch.setattr(advisor, "infer_parameters", boom)
    for call in (mcp_server.analyze_situation([{"role": "user", "content": "x"}]),
                 mcp_server.infer_parameters([{"role": "user", "content": "x"}])):
        assert call["status"] == "error"
        assert "401" in call["error"]


def test_run_scenarios_matches_toolkit_results_and_checks():
    toolkit = gt.run_all()
    evidence = advisor.run_scenarios({})
    assert len(evidence) == toolkit["scenarios_run"] == 10
    assert [item["scenario"] for item in evidence] == [item["scenario"] for item in toolkit["results"]]
    assert all(item["status"] == "ok" for item in evidence)
    failed = sum(1 for item in evidence for check in item["checks"] if not check["ok"])
    assert failed == toolkit["checks_failed"] == 0
    for actual, expected in zip(evidence, toolkit["results"]):
        assert actual["results"] == _expected_display(expected["results"]), actual["scenario"]
        assert actual["checks"] == expected["checks"], actual["scenario"]
        if actual["scenario"] == "tw7":
            _assert_tw7_semantics(actual)
        else:
            assert actual["conclusion"] == [advisor._neutral_conclusion(s)
                                            for s in expected["conclusion"]], actual["scenario"]
        assert actual["strategies"], actual["scenario"]


def test_run_one_scenario_reports_toolkit_exceptions(monkeypatch):
    def fail(tau=F(1, 2), f1=F(1, 4), E1=F(1, 4)):
        raise ValueError("toolkit exploded")

    monkeypatch.setitem(gt.SCENARIOS, "tw1", (fail, {}))
    result = advisor.run_one_scenario("tw1", {})
    assert result["status"] == "error"
    assert "toolkit exploded" in result["error"]
    assert not result["results"]
def test_strategy_table_failure_is_reported(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("strategy rendering broke")

    monkeypatch.setattr(advisor, "_strategy_rows", fail)
    result = advisor.run_one_scenario("tw1", {})
    assert result["status"] == "error"
    assert "strategy rendering broke" in result["error"]
    assert not result["strategies"]
    report = {"situation_summary": "s", "scenarios": [result],
              "totals": {"scenarios": 1, "ok": 0, "errors": 1,
                         "checks_total": len(result["checks"]), "checks_failed": 0}}
    markdown = advisor.render_report(report)
    assert "strategy table unavailable: strategy rendering broke" in markdown
def test_failed_tw6_verification_is_not_reported_as_ok():
    result = advisor.run_one_scenario("tw6", {"eps": F(999, 1000), "gamma": F(999, 1000)})
    assert result["status"] == "error"
    assert "verification checks failed" in result["error"]
    assert result["results"]
    assert any(not check["ok"] for check in result["checks"])
    markdown = advisor.render_report({"situation_summary": "s", "scenarios": [result],
                                      "totals": {"scenarios": 1, "ok": 0, "errors": 1,
                                                 "checks_total": len(result["checks"]),
                                                 "checks_failed": 2}})
    assert "K is a fixed point of the coalition operator" in markdown
    assert "[FAIL]" in markdown
def test_tw6_degenerate_fixed_point_is_reported_as_error():
    params = {"eps": F(1, 1000), "gamma": F(1, 1000), "p_slope": F(45000, 499)}
    result = advisor.run_one_scenario("tw6", params)
    assert result["status"] == "error"
    assert "singular tw6 coalition operator" in result["error"]
    assert result["results"] == {}

def test_analyze_situation_totals_are_green():
    client = _StubClient([json.dumps(_default_payload())])
    report = advisor.analyze_situation([{"role": "user", "content": "defaults"}], client=client)
    assert report["totals"]["scenarios"] == 10
    assert report["totals"]["ok"] == 10
    assert report["totals"]["errors"] == 0
    assert report["totals"]["checks_failed"] == 0
    assert report["totals"]["checks_total"] > 0
    assert report["situation_summary"]
    assert len(report["scenarios"]) == 10
def test_analyze_situation_totals_count_failed_checks(monkeypatch):
    client = _StubClient([json.dumps(_default_payload())])
    sample = [
        {"scenario": "tw1", "title": "Tw. 1", "status": "ok", "params": {},
         "param_notes": [], "results": {}, "checks": [{"ok": True}],
         "conclusion": [], "strategies": [{}]},
        {"scenario": "tw6", "title": "Tw. 6", "status": "error", "error": "failed check",
         "params": {}, "param_notes": [], "results": {}, "checks": [{"ok": False}],
         "conclusion": [], "strategies": [{}]},
    ]
    monkeypatch.setattr(advisor, "run_scenarios", lambda params: sample)
    report = advisor.analyze_situation([{"role": "user", "content": "a public goods game"}], client=client)
    assert report["totals"] == {"scenarios": 2, "ok": 1, "errors": 1,
                                "checks_total": 2, "checks_failed": 1}


def test_render_report_shows_every_scenario_and_exact_values():
    evidence = advisor.run_scenarios({})
    totals = {"scenarios": len(evidence),
              "ok": len(evidence),
              "errors": 0,
              "checks_total": sum(len(item["checks"]) for item in evidence),
              "checks_failed": 0}
    report = {"situation_summary": "a situation", "model": "m", "inference_raw": "",
              "scenarios": evidence, "totals": totals}
    markdown = advisor.render_report(report)
    for scenario_id in SCENARIO_IDS:
        assert f"## {scenario_id} - " in markdown
    for item in evidence:
        for key, value in item["results"].items():
            assert f"| {advisor._result_label(key)} | {advisor._math_value(value)} |" in markdown, (
                item["scenario"], key)
        for row in item["strategies"]:
            assert advisor._math_text(row["strategy"]) in markdown
            assert advisor._math_text(row["outcome"]) in markdown
            for label, value in row["values"].items():
                assert f"{advisor._math_text(label)} = {advisor._math_value(value)}" in markdown
        for sentence in item["conclusion"]:
            assert advisor._math_text(sentence) in markdown
    assert "| $t_{\\mathrm{FB}}$ | $\\frac{1}{2}$ |" in markdown
    assert "| $\\tau^{\\mathrm{RD}}$ | 2 |" in markdown
    assert "| $m_{\\mathrm{out}}$ | 4 |" in markdown
    assert "| $m_{\\mathrm{stab}}$ | 5 |" in markdown
    assert "| $k^*$ | $-1 + \\sqrt{10}/2$ |" in markdown
    assert "| parameter | value |" in markdown
    assert "| strategy | outcome | values |" in markdown
    assert "[PASS]" in markdown
    assert "[FAIL]" not in markdown
    assert "10 scenarios run" in markdown


def test_render_report_shows_dropped_key_notes_and_errors():
    evidence = advisor.run_one_scenario("tw1", {"bogus": 1})
    report = {"situation_summary": "s", "scenarios": [evidence],
              "totals": {"scenarios": 1, "ok": 1, "errors": 0,
                         "checks_total": 0, "checks_failed": 0}}
    assert "bogus: inferred value unusable, paper default used" in advisor.render_report(report)

    evidence = advisor.run_one_scenario("tw5", {"tau": 3})
    evidence["status"] = "error"
    evidence["error"] = "forced failure text"
    report["scenarios"] = [evidence]
    report["totals"] = {"scenarios": 1, "ok": 0, "errors": 1,
                        "checks_total": 0, "checks_failed": 0}
    text = advisor.render_report(report)
    assert "## tw5 - " in text
    assert "> scenario failed: forced failure text" in text

def test_render_report_marks_a_failing_check():
    evidence = advisor.run_scenarios({})
    evidence[0]["checks"] = [{"name": "a broken check", "ok": False,
                               "expected": "True", "got": "False"}]
    totals = {"scenarios": len(evidence), "ok": len(evidence), "errors": 0,
              "checks_total": sum(len(item["checks"]) for item in evidence), "checks_failed": 1}
    markdown = advisor.render_report({"situation_summary": "s", "model": "m",
                                      "inference_raw": "", "scenarios": evidence,
                                      "totals": totals})
    assert "[FAIL] a broken check" in markdown


def test_strategy_views_match_independent_toolkit_expectations():
    assert set(advisor.STRATEGY_VIEWS) == set(SCENARIO_IDS)
    assert set(EXPECTED_STRATEGY_LAYOUTS) == set(SCENARIO_IDS)
    for scenario_id, expected_rows in EXPECTED_STRATEGY_LAYOUTS.items():
        func, defaults = gt.SCENARIOS[scenario_id]
        raw_results = func(**defaults)["results"]
        actual = advisor.run_one_scenario(scenario_id, {})["strategies"]
        expected = []
        for row_index, (strategy, outcome, bindings) in enumerate(expected_rows):
            source = raw_results
            if scenario_id == "tw7" and row_index == 1:
                source = func(**defaults, eta=F(0))["results"]
            expected_values = {label: _path_value(source, path)
                               for label, path in bindings.items()}
            expected.append({"strategy": strategy, "outcome": outcome, "values": expected_values})
        assert actual == expected, scenario_id

    func, defaults = gt.SCENARIOS["tw7"]
    lam, eta = F(1, 5), F(1, 2)
    varied_sources = [
        func(**{**defaults, "lam": lam, "eta": eta})["results"],
        func(**{**defaults, "lam": lam, "eta": F(0)})["results"],
    ]
    varied = advisor.run_one_scenario("tw7", {"lam": lam, "eta": eta})["strategies"]
    for row, source, layout in zip(varied, varied_sources, EXPECTED_STRATEGY_LAYOUTS["tw7"]):
        bindings = layout[2]
        assert row["values"] == {label: _path_value(source, path)
                                 for label, path in bindings.items()}


def test_mcp_server_registers_paper_and_general_tools():
    import mcp_server
    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = {tool.name for tool in tools}
    # the original paper-scenario contracts stay registered
    assert {"analyze_situation", "infer_parameters", "list_scenarios",
            "run_scenario"} <= names
    # the general-game surface adds these
    assert {"list_game_templates", "infer_game_model", "solve_game",
            "estimate_game_parameters", "analyze_sensitivity", "simulate_game",
            "save_case", "list_cases", "load_case", "delete_case"} <= names


def test_mcp_success_payloads_include_status(monkeypatch):
    import mcp_server

    monkeypatch.setattr(advisor, "infer_parameters", lambda messages: {
        "situation_summary": "s", "scenarios": {}, "model": "m", "raw": "{}"})
    parsed = mcp_server.infer_parameters([])
    assert parsed["status"] == "ok"

    monkeypatch.setattr(advisor, "infer_parameters", lambda messages: {
        "need_more_information": True, "questions": ["Who are the players?"],
        "situation_summary": "", "scenarios": {}, "model": "m", "raw": "{}"})
    gated = mcp_server.infer_parameters([])
    assert gated["status"] == "ok"
    assert gated["need_more_information"] is True
    assert gated["questions"] == ["Who are the players?"]

    monkeypatch.setattr(advisor, "analyze_situation", lambda messages: {
        "situation_summary": "s", "model": "m", "inference_raw": "{}",
        "scenarios": [], "totals": {"scenarios": 0}, "markdown": "# s"})
    report = mcp_server.analyze_situation([])
    assert report["status"] == "ok"
    assert report["totals"] == {"scenarios": 0}


def test_webapp_builds_blocks_without_launching():
    import gradio as gr
    import webapp
    demo = webapp.build_demo()
    assert isinstance(demo, gr.Blocks)


def test_query_is_labeled_and_examples_precede_results():
    import webapp

    config = webapp.build_demo().config
    components = config["components"]
    textbox = next(c for c in components if c.get("type") == "textbox")
    examples = next(i for i, c in enumerate(components) if c.get("type") == "dataset")
    chatbot = next(i for i, c in enumerate(components) if c.get("type") == "chatbot")
    evidence = next(i for i, c in enumerate(components)
                   if c.get("type") == "markdown" and c.get("props", {}).get("elem_id") == "evidence")
    assert textbox["props"]["label"] == "Describe the situation"
    assert examples < chatbot < evidence
    textbox_id = textbox["id"]
    submit = [d for d in config["dependencies"] if d["targets"][0] == [textbox_id, "submit"]
              or tuple(d["targets"][0]) == (textbox_id, "submit")]
    assert submit and all(textbox_id in d["outputs"] for d in submit)


def test_respond_keeps_full_report_out_of_chat_and_clears_input(monkeypatch):
    import webapp

    full_report = "# Full report\n\n43 checked claims"
    monkeypatch.setattr(webapp.advisor, "analyze_situation", lambda messages: {
        "situation_summary": "Two factories coordinate output.",
        "markdown": full_report,
        "scenarios": [],
        "totals": {"scenarios": 10},
    })
    chat, state, markdown, raw, cleared, md_state, html_state = webapp.respond([], [], "A public goods game")
    assert chat[-1]["content"] != full_report
    assert "analysis complete" in chat[-1]["content"].lower()
    assert state[-1] == chat[-1]
    assert markdown == full_report
    assert raw["totals"] == {"scenarios": 10}
    assert cleared == ""


def test_respond_clears_input_after_analysis_error(monkeypatch):
    import webapp

    def fail(messages):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(webapp.advisor, "analyze_situation", fail)
    result = webapp.respond([], [], "A coordination game")
    assert result[0][-1]["content"] == "Analysis failed: backend unavailable"
    assert "backend unavailable" in result[2]
    assert result[3] == {}
    assert result[4] == ""


def test_query_is_labeled_and_examples_precede_evidence():
    import webapp

    components = webapp.build_demo().config["components"]
    textbox = next(c for c in components if c.get("type") == "textbox")
    examples = next(i for i, c in enumerate(components) if c.get("type") == "dataset")
    chatbot = next(i for i, c in enumerate(components) if c.get("type") == "chatbot")
    evidence = next(i for i, c in enumerate(components)
                   if c.get("type") == "markdown" and c.get("props", {}).get("elem_id") == "evidence")
    assert textbox["props"]["label"] == "Describe the situation"
    assert examples < chatbot < evidence


def test_respond_clears_submitted_input_and_keeps_report_out_of_chat(monkeypatch):
    import webapp

    full_report = "# Full report\n\n43 checked claims"
    monkeypatch.setattr(webapp.advisor, "analyze_situation", lambda messages: {
        "situation_summary": "Two factories coordinate output.",
        "markdown": full_report,
        "scenarios": [],
        "totals": {"scenarios": 10},
    })
    result = webapp.respond([], [], "A public goods game")
    assert len(result) == 7
    chat, state, markdown, raw, cleared_input, md_state, html_state = result
    assert chat[-1]["content"] != full_report
    assert "analysis complete" in chat[-1]["content"].lower()
    assert state[-1] == chat[-1]
    assert markdown == full_report
    assert raw["totals"] == {"scenarios": 10}
    assert cleared_input == ""


def test_respond_clears_input_after_analysis_error(monkeypatch):
    import webapp

    def fail(messages):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(webapp.advisor, "analyze_situation", fail)
    result = webapp.respond([], [], "A coordination game")
    assert len(result) == 7
    assert result[0][-1]["content"] == "Analysis failed: backend unavailable"
    assert "backend unavailable" in result[2]
    assert result[3] == {}
    assert result[4] == ""

def test_report_uses_latex_symbols_for_parameters_results_and_checks():
    import webapp

    scenarios = advisor.run_scenarios({})
    report = {"situation_summary": "A game-theory situation.",
              "scenarios": scenarios,
              "totals": {"scenarios": 10, "ok": 10, "errors": 0,
                         "checks_total": 43, "checks_failed": 0}}
    markdown = advisor.render_report(report)
    assert r"$\tau$" in markdown
    assert r"$f_1$" in markdown
    assert r"$t_{\mathrm{FB}}$" in markdown
    assert r"$V(t)$" in markdown
    assert r"$\Phi(k)-k$" in markdown
    assert r"$\tau^{\mathrm{RD}}$" in markdown
    assert r"$\frac{1}{2}$" in markdown

    config = webapp.build_demo().config
    evidence = next(c for c in config["components"]
                    if c.get("type") == "markdown" and c.get("props", {}).get("elem_id") == "evidence")
    assert any(d.get("left") == "$" and d.get("right") == "$"
               for d in evidence["props"]["latex_delimiters"])


def _empty_totals():
    return {"scenarios": 0, "ok": 0, "errors": 0, "checks_total": 0, "checks_failed": 0}


def test_render_report_includes_user_vocabulary():
    report = {"situation_summary": "Twelve villages share one irrigation canal.",
              "vocabulary": {"players": "twelve villages sharing an irrigation canal",
                             "decision": "how much water each village draws",
                             "cooperation": "each village cuts its draw by one third",
                             "defection": "a village quietly takes back its cut"},
              "scenarios": [], "totals": _empty_totals()}
    markdown = advisor.render_report(report)
    assert "Players: twelve villages sharing an irrigation canal" in markdown
    assert "Decision: how much water each village draws" in markdown
    assert "Cooperating: each village cuts its draw by one third" in markdown
    assert "Defecting: a village quietly takes back its cut" in markdown


def test_render_report_without_vocabulary_omits_block():
    report = {"situation_summary": "s", "vocabulary": {}, "scenarios": [],
              "totals": _empty_totals()}
    markdown = advisor.render_report(report)
    assert "Players:" not in markdown
    assert "## Mapping" not in markdown

    report = {"situation_summary": "s", "scenarios": [], "totals": _empty_totals()}
    markdown = advisor.render_report(report)
    assert "Players:" not in markdown
    assert "## Mapping" not in markdown

    report = {"situation_summary": "s", "vocabulary": "junk", "scenarios": [],
              "totals": _empty_totals()}
    advisor.render_report(report)


def test_strategy_text_is_domain_neutral():
    banned = ("parent", "social media", "screen", "famil")
    for scenario_id, rows in advisor.STRATEGY_VIEWS.items():
        for row in rows:
            strings = [row["strategy"], row["outcome"], *row.get("values", {})]
            joined = " ".join(strings).lower()
            assert not any(word in joined for word in banned), (scenario_id, strings)


def test_prompt_demands_stated_numbers():
    prompt = advisor.build_system_prompt()
    assert "never return the paper default" in prompt
    assert "user's own situation" in prompt
    contract = json.dumps(advisor._contract_example())
    assert "vocabulary" in contract
    for key in ("players", "decision", "cooperation", "defection"):
        assert f'"{key}"' in contract


def test_conclusions_are_domain_neutral_for_every_scenario():
    banned = ("parent", "social media", "screen", "famil", "class", "cohort", "household")
    for scenario_id in SCENARIO_IDS:
        result = advisor.run_one_scenario(scenario_id, {})
        assert result["status"] == "ok", scenario_id
        assert result["conclusion"], scenario_id
        report = {"situation_summary": "s", "scenarios": [result],
                  "totals": {"scenarios": 1, "ok": 1, "errors": 0,
                             "checks_total": len(result["checks"]), "checks_failed": 0}}
        markdown = advisor.render_report(report)
        rendered = [line[3:] for line in markdown.splitlines() if line.startswith("-> ")]
        assert len(rendered) == len(result["conclusion"]), scenario_id
        for sentence in rendered:
            assert not any(word in sentence.lower() for word in banned), (scenario_id, sentence)


def test_param_notes_are_domain_neutral_on_clamp():
    banned = ("parent", "social media", "screen", "famil")
    result = advisor.run_one_scenario("tw5", {"d": 99})
    assert result["param_notes"], "clamp note expected"
    report = {"situation_summary": "s", "scenarios": [result],
              "totals": {"scenarios": 1, "ok": 0, "errors": 1,
                         "checks_total": len(result["checks"]), "checks_failed": 0}}
    markdown = advisor.render_report(report)
    notes = [line[2:] for line in markdown.splitlines() if line.startswith("- ")]
    assert notes, "parameter notes expected in markdown"
    for note in notes:
        assert not any(word in note.lower() for word in banned), note


def test_calibration_caveat_appended_only_on_deviation():
    caveat_word = "calibration"
    deviated = advisor.run_one_scenario("tw1", {"tau": "1/3"})
    assert any(caveat_word in sentence for sentence in deviated["conclusion"])
    assert any("tau" in sentence for sentence in deviated["conclusion"] if caveat_word in sentence)
    baseline = advisor.run_one_scenario("tw1", {})
    assert not any(caveat_word in sentence for sentence in baseline["conclusion"])


def test_mapping_block_ignores_placeholder_vocabulary():
    report = {"situation_summary": "s",
              "vocabulary": {"players": "Who the players are, in the user's words",
                             "decision": "canal maintenance shifts",
                             "cooperation": None, "defection": ""},
              "scenarios": [], "totals": {}}
    markdown = advisor.render_report(report)
    assert "canal maintenance shifts" in markdown
    assert "Players:" not in markdown
    assert "Cooperating:" not in markdown
    assert "Defecting:" not in markdown


def test_prompt_allows_clarifying_gate():
    prompt = advisor.build_system_prompt()
    assert "need_more_information" in prompt
    assert "never ask questions about unstated numbers" in prompt


def test_prompt_converts_tw6_gamma_to_total_cohort_share():
    prompt = advisor.build_system_prompt().lower()
    assert "gamma is a share of the total cohort" in prompt
    assert "multiply it by (1 - eps)" in prompt
    assert "25% of the remaining 90% is gamma = 0.225" in prompt


def test_gated_response_skips_scenarios(monkeypatch):
    payload = {"need_more_information": True,
               "questions": ["Who are the players?", "What does each player decide?"],
               "situation_summary": "Too vague to map."}
    client = _StubClient([json.dumps(payload)])

    def explode(params):
        raise AssertionError("run_scenarios must not run for a gated response")

    monkeypatch.setattr(advisor, "run_scenarios", explode)
    report = advisor.analyze_situation([{"role": "user", "content": "something vague"}],
                                       client=client)
    assert report["need_more_information"] is True
    assert report["questions"] == payload["questions"]
    assert report["situation_summary"] == "Too vague to map."
    assert "## Clarifying questions" in report["markdown"]
    assert "1. Who are the players?" in report["markdown"]
    assert "2. What does each player decide?" in report["markdown"]


def test_gated_response_webapp_renders_questions(monkeypatch):
    import webapp

    gated = {"need_more_information": True, "questions": ["Who are the players?"],
             "situation_summary": "", "vocabulary": {},
             "markdown": "## Clarifying questions\n\n1. Who are the players?"}
    monkeypatch.setattr(advisor, "analyze_situation", lambda messages: dict(gated))
    chat, state, markdown, raw, cleared, md_state, html_state = webapp.respond([], [], "something vague")
    assistant = [m["content"] for m in chat if m["role"] == "assistant"]
    assert any("Before I run the numbers:" in content for content in assistant)
    assert any("1. Who are the players?" in content for content in assistant)
    assert "Who are the players?" in markdown
    assert raw["need_more_information"] is True
    assert "markdown" not in raw
    assert len(state) == 2
    assert cleared == ""


class _SecondCallRaises:
    def __init__(self, first_payload):
        self.first_payload = first_payload
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            message = SimpleNamespace(content=self.first_payload)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        raise RuntimeError("LLM exploded")


def test_practical_conclusions_rendered():
    bullets = ["The pact holds only 4 of the 9 members.",
               "The limit reaches 1/2 of full adoption."]
    analysis = _default_payload()
    analysis["situation_summary"] = "Two factories share one river."
    client = _StubClient([json.dumps(analysis), json.dumps({"practical_conclusions": bullets})])
    report = advisor.analyze_situation([{"role": "user", "content": "factories"}], client=client)
    assert report["practical_conclusions"] == bullets
    markdown = report["markdown"]
    assert "## Practical conclusions" in markdown
    for bullet in bullets:
        assert f"- {bullet}" in markdown

    entry = advisor.run_one_scenario("tw1_1", {})
    digest = advisor.calculations_digest([entry])[0]
    assert digest["status"] == "ok"
    assert digest["params"] == entry["params"]
    assert digest["checks"] == (f"{sum(1 for check in entry['checks'] if check['ok'])}"
                                f"/{len(entry['checks'])}")
    assert "the best outcome for a single actor" in digest["results"]
    assert "curve t" not in digest["results"]
    assert "curve v" not in digest["results"]

    synthetic = advisor.run_one_scenario("tw1", {})
    synthetic["results"]["long_series"] = list(range(13))
    assert "long series" not in advisor.calculations_digest([synthetic])[0]["results"]


def test_digest_uses_plain_scenario_and_result_names():
    entry = advisor.run_one_scenario("tw5", {})
    digest = advisor.calculations_digest([entry])[0]
    assert not re.search(r"tw\d", digest["scenario"], re.IGNORECASE)
    assert "Tw." not in digest["scenario"]
    for key in digest["results"]:
        assert not re.search(r"\btw\d", key, re.IGNORECASE)
        assert key == key.replace("_", " ") or key in advisor._PLAIN_RESULT_LABELS.values()


def test_digest_scenario_names_avoid_paper_framing():
    entries = [advisor.run_one_scenario(scenario_id, {}) for scenario_id in SCENARIO_IDS]
    names = [item["scenario"] for item in advisor.calculations_digest(entries)]
    banned = ("screen", "social media", "family", "parent", "fixed point")
    assert len(names) == len(SCENARIO_IDS)
    for name in names:
        assert not any(word in name.lower() for word in banned), name


def test_practical_conclusions_drop_jargon_bullets():
    bullets = ["In scenario tw5 the threshold is 4 for m_out.",
               "Restricting together costs 1/2 while deviating pays 0."]
    client = _StubClient([json.dumps(_default_payload()), json.dumps({"practical_conclusions": bullets})])
    report = advisor.analyze_situation([{"role": "user", "content": "defaults"}], client=client)
    assert report["practical_conclusions"] == ["Restricting together costs 1/2 while deviating pays 0."]


def test_practical_conclusions_failure_tolerated():
    client = SimpleNamespace(chat=SimpleNamespace(
        completions=_SecondCallRaises(json.dumps(_default_payload()))))
    report = advisor.analyze_situation([{"role": "user", "content": "defaults"}], client=client)
    assert report["totals"]["errors"] == 0
    assert report["practical_conclusions"] == []
    assert "## Practical conclusions" not in report["markdown"]


def test_calculations_summary_table():
    evidence = advisor.run_one_scenario("tw1", {})
    report = {"situation_summary": "s", "scenarios": [evidence],
              "totals": {"scenarios": 1, "ok": 1, "errors": 0,
                         "checks_total": len(evidence["checks"]), "checks_failed": 0}}
    markdown = advisor.render_report(report)
    assert "## Calculations summary" in markdown
    passed = sum(1 for check in evidence["checks"] if check["ok"])
    assert f"| tw1 | ok | {passed}/{len(evidence['checks'])} |" in markdown


def test_tw6_gamma_report_uses_total_population_denominator():
    item = advisor.run_one_scenario("tw6", {})
    conclusion = " ".join(item["conclusion"])
    meaning = advisor.CATALOG["tw6"]["params"]["gamma"]["meaning"].lower()
    assert "22.5% of the total population" in conclusion.lower()
    assert "25% of conscientious members" in conclusion.lower()
    assert "total cohort" in meaning


def test_infer_parameters_retries_parseable_but_invalid_schema():
    valid = json.dumps(_default_payload())
    client = _StubClient(["{}", valid])
    parsed = advisor.infer_parameters([{"role": "user", "content": "a clear game"}], client)
    assert set(parsed["scenarios"]) == set(SCENARIO_IDS)
    assert len(client.chat.completions.calls) == 2


def test_tw6_singular_coalition_slope_is_rejected_before_calculation():
    item = advisor.run_one_scenario(
        "tw6", {"eps": "1/10", "gamma": "2/5", "p_slope": 2, "f_lo": 0, "f_hi": 1})
    assert item["status"] == "error"
    assert not item["results"]
    assert "singular" in item["error"].lower()


def test_tw3_2_cutoff_direction_matches_risk_dominance():
    item = advisor.run_one_scenario("tw3_2", {"f1": 1, "E1": 3})
    report = {"situation_summary": "A game.", "scenarios": [item],
              "totals": {"scenarios": 1, "ok": 1, "errors": 0,
                         "checks_total": len(item["checks"]), "checks_failed": 0}}
    markdown = advisor.render_report(report)
    assert "cutoff below which all-using is risk-dominant" in markdown


def test_report_labels_default_parameters_as_conditional():
    report = advisor.analyze_situation(
        [{"role": "user", "content": "A clear game with no stated measurements."}],
        client=_StubClient([json.dumps(_default_payload()),
                            json.dumps({"practical_conclusions": ["Under the defaults, value is 1/2."]})]))
    assert "## Parameter assumptions" in report["markdown"]
    assert "conditional on the displayed values" in report["markdown"].lower()
    assert "not estimates for this situation" in report["markdown"].lower()


def test_zero_denominator_parameter_falls_back_to_default():
    item = advisor.run_one_scenario("tw1", {"tau": "1/0", "f1": "1/4"})
    assert item["status"] == "ok"
    assert item["params"]["tau"] == "1/2"
    assert item["params"]["f1"] == "1/4"
    assert any("tau: inferred value unusable" in note for note in item["param_notes"])


def test_practical_prompt_marks_defaults_as_assumptions():
    client = _StubClient([json.dumps(_default_payload()),
                          json.dumps({"practical_conclusions": []})])
    advisor.analyze_situation([{"role": "user", "content": "a qualitative game"}], client)
    prompt = client.chat.completions.calls[1]["messages"][0]["content"].lower()
    assert "paper defaults" in prompt
    assert "conditionally" in prompt


def test_math_text_does_not_replace_tokens_inside_words():
    assert advisor._math_text("steps metadata") == "steps metadata"


def test_math_text_preserves_existing_math_spans():
    text = r"The threshold is $\\tau$ and $\\epsilon$."
    assert advisor._math_text(text) == text


def test_infer_parameters_retries_non_boolean_clarification_flag():
    invalid = _default_payload()
    invalid["need_more_information"] = "false"
    invalid["questions"] = 7
    client = _StubClient([json.dumps(invalid), json.dumps(_default_payload())])
    parsed = advisor.infer_parameters([{"role": "user", "content": "a clear game"}], client)
    assert "need_more_information" not in parsed
    assert len(client.chat.completions.calls) == 2


def test_math_text_formats_tokens_between_escaped_currency_amounts():
    text = r"The price is \$5, tau is 1/2, and the fee is \$10."
    rendered = advisor._math_text(text)
    assert r"\$5" in rendered
    assert r"$\tau$" in rendered
    assert r"\$10" in rendered


def test_unmatched_currency_dollar_does_not_hide_text_substitution():
    assert advisor._math_text("$5 and tau is 1/2") == "$5 and $\\tau$ is 1/2"


def test_math_text_preserves_escaped_currency_inside_inline_and_display_math():
    formulas = [r"$\text{cost: \$5} + \tau$",
                r"$$\text{cost: \$5} + \tau$$"]
    for formula in formulas:
        assert advisor._math_text(formula) == formula


def test_practical_prompt_keeps_user_language_and_field_description_guidance():
    client = _StubClient([json.dumps(_default_payload()),
                          json.dumps({"practical_conclusions": []})])
    advisor.analyze_situation([{"role": "user", "content": "a game"}], client)
    prompt = client.chat.completions.calls[1]["messages"][0]["content"].lower()
    assert "describe what a number measures instead" in prompt
    assert "frame every conclusion in the user's own terms" in prompt


def test_math_text_handles_unmatched_dollar_with_many_backslashes():
    text = "$" + "\\" * 40 + "x"
    code = f"import advisor; text={text!r}; print(advisor._math_text(text) == text)"
    try:
        result = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(__file__),
                                capture_output=True, text=True, timeout=3, check=True)
    except subprocess.TimeoutExpired:
        result = None
    assert result is not None and result.stdout.strip() == "True"


def test_game_solve_rejects_invalid_json():
    import webapp
    out = webapp.game_solve("{not json", True)
    assert out[0]["disposition"] == "invalid"
    assert "bad JSON" in out[0]["error"]


def _intake_outputs(monkeypatch, parsed):
    import webapp
    monkeypatch.setattr(webapp.advisor, "infer_game_model",
                        lambda messages, **kwargs: parsed)
    return webapp.game_intake([], "two cartel members choose output")


def _assert_accepted_by_gradio_outputs(outputs):
    import gradio as gr
    state, chat, note, spec, answer = outputs
    gr.Markdown().postprocess(note)
    gr.Markdown().postprocess(answer)
    assert isinstance(spec, str) and isinstance(state, list)


def test_game_intake_empty_text_skips_llm(monkeypatch):
    import webapp

    def boom(messages, **kwargs):
        raise AssertionError("LLM must not be called for empty intake")

    monkeypatch.setattr(webapp.advisor, "infer_game_model", boom)
    outputs = webapp.game_intake([], "   ")
    assert outputs[0] == [] and outputs[1]["visible"] is False
    assert outputs[3] == "" and "Describe the situation" in outputs[2]
    _assert_accepted_by_gradio_outputs(outputs)


def test_game_intake_every_disposition_feeds_gradio_json_outputs(monkeypatch):
    parsed_by_disposition = {
        "clarification": {"disposition": "clarification", "questions": ["How many firms?"]},
        "unsupported": {"disposition": "unsupported"},
        "invalid": {"disposition": "invalid"},
        "needs_confirmation": {"disposition": "needs_confirmation", "game": {"players": []}},
    }
    for parsed in parsed_by_disposition.values():
        _assert_accepted_by_gradio_outputs(_intake_outputs(monkeypatch, parsed))


def test_game_intake_llm_failure_feeds_gradio_json_outputs(monkeypatch):
    import webapp

    def fail(messages, **kwargs):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(webapp.advisor, "infer_game_model", fail)
    _assert_accepted_by_gradio_outputs(webapp.game_intake([], "any game"))


def test_game_intake_draft_populates_spec_box(monkeypatch):
    outputs = _intake_outputs(monkeypatch, {"disposition": "needs_confirmation",
                                            "template_id": "custom", "game": {"k": 1}})
    assert '"template_id": "custom"' in outputs[3]


def test_case_save_rejects_unparseable_spec_without_touching_store(monkeypatch):
    import webapp

    def boom():
        raise AssertionError("store must not be opened for an unparseable spec")

    monkeypatch.setattr(webapp, "_case_store", boom)
    for bad in ("", "{not json", None):
        out = webapp.case_save("t", bad, True)
        assert out["status"] == "invalid" and "bad JSON" in out["error"]


def test_public_link_empty_markdown_needs_no_network(monkeypatch):
    import webapp

    def boom(*args, **kwargs):
        raise AssertionError("share worker must not be called without content")

    monkeypatch.setattr(webapp.share_report, "public_share_link", boom)
    assert "Run an analysis first" in webapp._public_link("", "t")
    assert "Run an analysis first" in webapp.game_public_link(None)


def test_intake_prompt_caps_and_keeps_questions_plain():
    prompt = advisor.build_game_intake_prompt()
    assert "Ask at most 2 questions, and only once" in prompt
    assert "Never use the words payoff, utility, Nash, equilibrium" in prompt


def test_infer_game_model_caps_clarification_questions():
    payload = json.dumps({"disposition": "clarification",
                          "questions": ["Who decides?", "What can each side do?",
                                        "Which result do they prefer?"]})
    parsed = advisor.infer_game_model([{"role": "user", "content": "a situation"}],
                                      client=_StubClient([payload]),
                                      config=advisor.load_config({}))
    assert parsed["questions"] == ["Who decides?", "What can each side do?"]


def test_infer_game_model_leaves_non_clarification_questions_alone():
    payload = json.dumps({"disposition": "needs_confirmation", "questions": ["a", "b", "c"]})
    parsed = advisor.infer_game_model([{"role": "user", "content": "a situation"}],
                                      client=_StubClient([payload]),
                                      config=advisor.load_config({}))
    assert parsed["questions"] == ["a", "b", "c"]


def test_intake_questions_keeps_only_non_empty_strings():
    assert advisor._intake_questions(["a", None, "  ", "b", "c"]) == ["a", "b"]
    assert advisor._intake_questions("nope") == []
    assert advisor._intake_questions(None) == []


def test_game_intake_clarification_without_questions_asks_for_structure(monkeypatch):
    outputs = _intake_outputs(monkeypatch, {"disposition": "clarification", "questions": []})
    assert "who is involved" in outputs[2].lower()
    _assert_accepted_by_gradio_outputs(outputs)


def test_game_intake_clarification_points_at_the_input_box(monkeypatch):
    outputs = _intake_outputs(monkeypatch,
                              {"disposition": "clarification",
                               "questions": ["How many factories are there?"]})
    assert "box above" in outputs[2]
    assert "How many factories are there?" in outputs[1]["value"][-1]["content"]


def test_game_intake_forwards_asked_already_after_one_round(monkeypatch):
    import webapp
    seen = {}

    def stub(messages, **kwargs):
        seen.update(kwargs)
        return {"disposition": "needs_confirmation", "template_id": "custom", "game": {"k": 1}}

    monkeypatch.setattr(webapp.advisor, "infer_game_model", stub)
    history = [{"role": "user", "content": "two factories"},
               {"role": "assistant", "content": "A couple of questions:\n1. How many?"}]
    webapp.game_intake(history, "two")
    assert seen["asked_already"] is True


def test_game_intake_first_turn_does_not_suppress_questions(monkeypatch):
    import webapp
    seen = {}

    def stub(messages, **kwargs):
        seen.update(kwargs)
        return {"disposition": "clarification", "questions": ["Who decides?"]}

    monkeypatch.setattr(webapp.advisor, "infer_game_model", stub)
    webapp.game_intake([], "two factories")
    assert seen["asked_already"] is False


def test_game_intake_never_asks_a_second_round(monkeypatch):
    import webapp
    history = [{"role": "user", "content": "something vague"},
               {"role": "assistant", "content": "A couple of questions:\n1. Who decides?"}]
    monkeypatch.setattr(webapp.advisor, "infer_game_model",
                        lambda messages, **kwargs: {"disposition": "clarification",
                                                    "questions": ["Which result do they prefer?"]})
    outputs = webapp.game_intake(history, "still vague")
    chat = outputs[1]["value"]
    assert [m["role"] for m in chat] == ["user", "assistant", "user", "assistant"]
    assert "Which result do they prefer?" not in chat[-1]["content"]
    assert "cannot see who is involved" in chat[-1]["content"]
    assert "box above" not in outputs[2]
    assert "concrete example" in outputs[2]


def test_infer_game_model_appends_no_repeat_rule_only_when_asked(monkeypatch):
    payload = json.dumps({"disposition": "needs_confirmation", "game": {"players": []}})
    prompts = []

    class Recording:
        def __init__(self, inner):
            self.chat = self
            self.completions = self
            self.inner = inner

        def create(self, model, messages, temperature):
            prompts.append(messages[0]["content"])
            return self.inner.chat.completions.create(model=model, messages=messages,
                                                      temperature=temperature)

    for asked in (False, True):
        inner = _StubClient([payload])
        advisor.infer_game_model([{"role": "user", "content": "a situation"}],
                                 client=Recording(inner), config=advisor.load_config({}),
                                 asked_already=asked)
    assert advisor.NO_CLARIFICATION_RULE.strip() not in prompts[0]
    assert advisor.NO_CLARIFICATION_RULE.strip() in prompts[1]

