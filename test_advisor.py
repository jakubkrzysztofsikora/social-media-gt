"""Unit tests for the social-media-gt advisor: core, MCP server and web app."""
import asyncio
import inspect
import json
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
        ("All-using (risk-dominant)", "cutoff above which all-using is risk-dominant",
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
        ("Pact of gamma aware members", "adoption level held, contraction, trajectory, exit multiplier",
         {"adoption level K": "K", "contraction a": "a", "trajectory": "trajectory",
          "exit multiplier": "exit_multiplier"}),
    ],
    "tw7": [
        ("Rule covers part of the group", "effective pressure and resulting adoption limit",
         {"effective pressure": "lambda_effective", "adoption limit": "limit"}),
        ("No rule (eta = 0)", "baseline: the same toolkit function called with eta = 0",
         {"effective pressure": "lambda_effective", "adoption limit": "limit"}),
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
        assert actual["conclusion"] == expected["conclusion"], actual["scenario"]
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
    assert "no unique finite fixed point" in result["error"]
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

    varied = advisor.run_one_scenario("tw7", {"eta": F(1, 2)})["strategies"]
    assert varied[0]["values"] == {"effective pressure": "1/10", "adoption limit": "1/2"}
    assert varied[1]["values"] == {"effective pressure": "1/5", "adoption limit": "1/4"}


def test_mcp_server_registers_four_tools():
    import mcp_server
    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert sorted(tool.name for tool in tools) == ["analyze_situation", "infer_parameters",
                                                   "list_scenarios", "run_scenario"]


def test_mcp_success_payloads_include_status(monkeypatch):
    import mcp_server

    monkeypatch.setattr(advisor, "infer_parameters", lambda messages: {
        "situation_summary": "s", "scenarios": {}, "model": "m", "raw": "{}"})
    parsed = mcp_server.infer_parameters([])
    assert parsed["status"] == "ok"

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
    submit = [d for d in config["dependencies"] if d["targets"][0][1] == "submit"]
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
    chat, state, markdown, raw, cleared = webapp.respond([], [], "A public goods game")
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
    assert len(result) == 5
    chat, state, markdown, raw, cleared_input = result
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
    assert len(result) == 5
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
