from fractions import Fraction
import json

import advisor
from game_models import GameSpecError, NormalFormGame, compile_template
import plain_view
import pytest

_P3_CUSTOM_OPTIONS = {
    "type_counts": {"urban": 2, "rural": 1},
    "thresholds": {"urban": "0", "rural": "-2"},
    "payoff_coefficients": {
        "f": {"constant": "1/2", "linear": "2"},
        "E": {"constant": "1/4", "linear": "1/2", "quadratic": "3"},
    },
}

def _payoff(game, profile):
    return next(row["utilities"] for row in game.payoffs
                if row["action_profile"] == profile)


def test_p3_compiles_the_finite_heterogeneous_payoff_table():
    envelope = compile_template("p3_heterogeneous", {})
    game = envelope.game

    assert isinstance(game, NormalFormGame)
    assert len(game.players) == 10
    assert len(game.payoffs) == 1024
    assert sum(player.role == "C" for player in game.players) == 9
    assert sum(player.role == "D" for player in game.players) == 1
    assert all(game.actions[player.id] == ["R", "A"] for player in game.players)

    all_a = {player.id: "A" for player in game.players}
    low_adoption = {player.id: ("A" if player.role == "D" else "R")
                    for player in game.players}
    all_a_utilities = _payoff(game, all_a)
    low_utilities = _payoff(game, low_adoption)

    assert set(all_a_utilities.values()) == {Fraction(1)}
    assert {low_utilities[p.id] for p in game.players if p.role == "C"} == {
        Fraction(223, 162)}
    assert low_utilities[next(p.id for p in game.players if p.role == "D")] == 0

    c_id = next(p.id for p in game.players if p.role == "C")
    c_deviation = dict(low_adoption)
    c_deviation[c_id] = "A"
    assert _payoff(game, c_deviation)[c_id] == Fraction(1, 9)
    d_id = next(p.id for p in game.players if p.role == "D")
    d_deviation = dict(low_adoption)
    d_deviation[d_id] = "R"
    assert _payoff(game, d_deviation)[d_id] == -1


def test_advisor_solves_and_exposes_p3_equilibrium_and_pareto_evidence():
    spec = {"template_id": "p3_heterogeneous", "template_options": {}}
    result = advisor.solve_game(spec, confirmed_by_user=True, methods=["pure_nash"])

    assert result["disposition"] == "ok"
    assert result["normalized_game"]["kind"] == "NormalFormGame"
    method = next(row for row in result["method_results"] if row["method"] == "pure_nash")
    assert method["status"] == "exact_verified"
    assert len(method["witnesses"]) == 2

    evidence = method["benchmark_evidence"]
    assert evidence["pure_nash_count"] == 2
    assert evidence["strict_pure_nash_count"] == 2
    profiles = {row["label"]: row for row in evidence["equilibria"]}
    assert profiles["all-A"]["total_utility"] == "10"
    assert profiles["all-A"]["strict_nash"] is True
    assert profiles["low-adoption"]["total_utility"] == "223/18"
    assert profiles["low-adoption"]["strict_nash"] is True
    assert profiles["low-adoption"]["utility_by_type"] == {"C": "223/162", "D": "0"}

    comparison = evidence["welfare_comparison"]
    assert comparison["utility_change_by_type"] == {"C": "61/162", "D": "-1"}
    assert comparison["pareto_improvement"] is False
    assert all(check["ok"] for check in method["checks"])

    text = plain_view.describe_result(result, json.dumps(spec))
    assert "P3 benchmark" in text
    assert "223/18" in text
    assert "10" in text
    assert "not a Pareto improvement" in text
    assert "D loses 1" in text


def test_p3_template_is_listed_separately_from_homogeneous_tw3():
    template = next(row for row in advisor.list_game_templates()
                    if row["id"] == "p3_heterogeneous")
    assert template["options"] == ["type_counts", "thresholds", "payoff_coefficients"]
    assert template["default_profile"] == "P3 paper benchmark (9 C, 1 D)"
    assert "one-shot" in template["scope"]
    assert "repeated-game" in template["scope"]

    prompt = advisor.build_game_intake_prompt()
    assert "p3_heterogeneous" in prompt
    assert "paper benchmark" in prompt
    assert "tw3" in prompt
    assert "other players" in prompt
    assert "u(A,k)=f(k)" in prompt
    assert "u(R,k)=tau-E(k)" in prompt

    homogeneous = advisor.run_one_scenario("tw3", {"n": 10, "tau": "3/2"})
    assert homogeneous["scenario"] == "tw3"
    assert homogeneous["status"] == "error"
    assert homogeneous["params"]["n"] == "10"
    assert homogeneous["params"]["tau"] == "3/2"


def test_p3_compiles_user_supplied_types_thresholds_and_coefficients():
    envelope = compile_template("p3_heterogeneous", _P3_CUSTOM_OPTIONS)
    game = envelope.game

    assert isinstance(game, NormalFormGame)
    assert len(game.players) == 3
    assert len(game.payoffs) == 8
    assert {player.role for player in game.players} == {"urban", "rural"}
    assert sum(player.role == "urban" for player in game.players) == 2
    assert all(game.actions[player.id] == ["R", "A"] for player in game.players)
    assert envelope.template_options == _P3_CUSTOM_OPTIONS
    assert envelope.assumptions[0] == ("User-specified P3 one-shot model; "
                                       "not the paper benchmark.")

    all_a = {player.id: "A" for player in game.players}
    assert set(_payoff(game, all_a).values()) == {Fraction(5, 2)}

    urban = next(player for player in game.players if player.role == "urban")
    deviation = dict(all_a)
    deviation[urban.id] = "R"
    assert _payoff(game, deviation)[urban.id] == Fraction(-15, 4)


def test_p3_custom_equilibria_use_user_values_without_benchmark_checks():
    spec = {"template_id": "p3_heterogeneous",
            "template_options": _P3_CUSTOM_OPTIONS}
    result = advisor.solve_game(spec, confirmed_by_user=True, methods=["pure_nash"])

    assert result["disposition"] == "ok"
    method = next(row for row in result["method_results"] if row["method"] == "pure_nash")
    assert method["status"] == "exact_verified"
    assert len(method["witnesses"]) == 1
    assert set(method["witnesses"][0].values()) == {"A"}
    assert "benchmark_evidence" not in method

    evidence = method["p3_evidence"]
    assert evidence["pure_nash_count"] == 1
    assert evidence["strict_pure_nash_count"] == 1
    assert evidence["equilibria"][0]["utility_by_type"] == {
        "urban": "5/2", "rural": "5/2"}
    assert evidence["equilibria"][0]["total_utility"] == "15/2"
    assert all(check["ok"] for check in method["checks"])


def test_p3_custom_options_are_complete_and_reject_custom_game_data():
    with pytest.raises(GameSpecError, match="require all of"):
        compile_template("p3_heterogeneous", {"type_counts": {"urban": 2}})
    with pytest.raises(GameSpecError, match="does not accept custom game data"):
        compile_template("p3_heterogeneous", _P3_CUSTOM_OPTIONS,
                         {"kind": "normal_form"})

def test_p3_draft_discloses_one_shot_scope_before_confirmation():
    spec = {"template_id": "p3_heterogeneous", "template_options": {}}

    summary, accepted = plain_view.describe_draft(json.dumps(spec))

    assert accepted is True
    assert "one-shot" in summary
    assert "repeated-game construction" in summary
