import json

import plain_view
from solver_worker import run_worker_request

SPEC = {"template_id": "custom", "game": {
    "kind": "normal_form",
    "players": [{"id": "a", "name": "Cafe A"}, {"id": "b", "name": "Cafe B"}],
    "actions": {"a": ["low", "high"], "b": ["low", "high"]},
    "payoffs": [{"action_profile": {"a": x, "b": y}, "utilities": {"a": u, "b": v}}
                for x, y, u, v in [("low", "low", 2, 2), ("low", "high", 7, 1),
                                   ("high", "low", 1, 7), ("high", "high", 5, 5)]]}}


def test_draft_summary_names_players_options_and_payoffs_without_json():
    text, ok = plain_view.describe_draft(json.dumps(SPEC))
    assert ok
    assert "Cafe A" in text and "Cafe B" in text
    assert "Cafe A 7, Cafe B 1" in text
    assert "{" not in text and "template_id" not in text


def test_draft_summary_flags_unusable_details_instead_of_crashing():
    assert plain_view.describe_draft("") == ("", False)
    assert plain_view.describe_draft("{oops")[1] is False
    text, ok = plain_view.describe_draft(json.dumps({"template_id": "custom", "game": {"kind": "nope"}}))
    assert not ok and "can't be used" in text


def test_result_is_plain_language_for_the_real_solver_output():
    spec_json = json.dumps(SPEC)
    result = run_worker_request({"op": "solve_game", "game_spec": SPEC})
    text = plain_view.describe_result(result, spec_json)
    assert "Cafe A: low, Cafe B: low" in text
    assert "each side gets: Cafe A 2, Cafe B 2" in text
    assert "{" not in text


def test_mixed_strategies_render_as_percentages_and_skip_zero_shares():
    result = {"disposition": "ok",
              "normalized_game": {"players": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
                                  "assumptions": []},
              "method_results": [{"method": "mixed_nash", "status": "exact_verified",
                                  "witnesses": [{"a": {"x": "3/5", "y": "2/5", "z": "0"},
                                                 "b": {"x": "1", "y": "0"}}]}]}
    text = plain_view.describe_result(result)
    assert "A: 60% x, 40% y" in text and "B: 100% x" in text and "z" not in text


def test_failed_results_explain_instead_of_dumping_json():
    text = plain_view.describe_result({"disposition": "invalid", "error": "bad prior"})
    assert "couldn't work this out" in text and "bad prior" in text
    assert "confirm" in plain_view.describe_result({"disposition": "needs_confirmation"})


def test_inapplicable_methods_are_listed_with_their_reason():
    result = {"disposition": "ok",
              "normalized_game": {"players": [{"id": "a", "name": "A"}], "assumptions": []},
              "method_results": [{"method": "correlated_equilibrium", "status": "unsupported",
                                  "assumptions": ["requires a confirmed mediator"]},
                                 {"method": "pure_nash", "status": "exact_verified",
                                  "witnesses": [{"a": "x"}]}]}
    text = plain_view.describe_result(result)
    assert "A: x" in text
    assert "### Not applied" in text
    assert "coordinated play with a shared signal: requires a confirmed mediator" in text.lower()


def test_bayesian_witness_renders_actions_by_type_not_probabilities():
    result = {
        "disposition": "ok",
        "normalized_game": {"players": [{"id": "a", "name": "A"}], "assumptions": []},
        "method_results": [{"method": "bayesian_nash", "status": "exact_verified",
                            "witnesses": [{"a": {"high": "enter", "low": "exit"}}]}],
    }
    text = plain_view.describe_result(result)
    assert "A: type high → enter, type low → exit" in text
    assert "%" not in text


def test_subgame_perfect_nested_witnesses_render_decisions_and_payoffs():
    result = {
        "disposition": "ok",
        "normalized_game": {"players": [{"id": "a", "name": "A"}], "assumptions": []},
        "method_results": [{"method": "subgame_perfect", "status": "exact_verified",
                            "witnesses": [{
                                "continuation": {"root": "offer"},
                                "value": {"a": "3"},
                                "behavior": {"root": ["offer", "walk away"]},
                            }]}],
    }
    text = plain_view.describe_result(result)
    assert "root → offer" in text
    assert "A 3" in text
    assert "offer or walk away" in text
    assert "%" not in text


def test_non_mixed_nested_mapping_is_not_guessed_to_be_a_probability_mix():
    result = {
        "disposition": "ok",
        "normalized_game": {"players": [{"id": "a", "name": "A"}], "assumptions": []},
        "method_results": [{"method": "pure_nash", "status": "exact_verified",
                            "witnesses": [{"a": {"kind": "contingent", "action": "left"}}]}],
    }
    text = plain_view.describe_result(result)
    assert "left" in text
    assert "%" not in text


def test_stackelberg_nested_result_renders_pure_tie_outcomes():
    result = {
        "disposition": "ok",
        "normalized_game": {"players": [{"id": "l", "name": "Leader"},
                                        {"id": "f", "name": "Follower"}],
                            "assumptions": []},
        "method_results": [{"method": "stackelberg", "status": "exact_verified",
                            "witnesses": [{"commitment": "pure",
                                           "leader_favorable": [{"leader": "B", "follower": "Y",
                                                                 "leader_utility": "3"}],
                                           "follower_favorable": [{"leader": "B", "follower": "X",
                                                                   "leader_utility": "1"}]}]}],
    }
    text = plain_view.describe_result(result)
    assert "Leader-favorable: B then Y, utility 3" in text
    assert "Follower-favorable: B then X, utility 1" in text


def test_bargaining_draft_shows_randomization_and_binding_right_before_confirmation():
    spec = {
        "template_id": "bargaining",
        "template_options": {
            "players": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
            "feasible_points": [{"a": 1, "b": 1}],
            "disagreement": {"a": 0, "b": 0},
            "randomization": "allowed",
        },
    }
    text, ok = plain_view.describe_draft(json.dumps(spec))
    assert ok
    assert "**Randomization:** allowed" in text
    assert "**Binding commitment rights:** mediator" in text


def test_unsupported_only_result_explains_that_no_analysis_applies():
    result = {
        "disposition": "ok",
        "normalized_game": {"players": [{"id": "a", "name": "A"}], "assumptions": []},
        "method_results": [{"method": "correlated_equilibrium", "status": "unsupported",
                            "assumptions": ["requires a confirmed mediator"]}],
    }
    text = plain_view.describe_result(result)
    assert "No supported analysis method could be applied" in text
    assert text.strip() != "## What is likely to happen"


def test_bargaining_result_labels_randomization_witness():
    result = {
        "disposition": "ok",
        "normalized_game": {"players": [{"id": "a", "name": "A"},
                                        {"id": "b", "name": "B"}], "assumptions": []},
        "method_results": [{"method": "nash_bargaining", "status": "exact_verified",
                            "witnesses": [{"a": "3/2", "b": "3/2", "nash_product": "9/4",
                                           "mixture": "1/2*(0,3) + 1/2*(3,0)"}]}],
    }
    text = plain_view.describe_result(result)
    assert "Randomized deal" in text
    assert "1/2*(0,3) + 1/2*(3,0)" in text


def test_correlated_witness_renders_distribution_of_joint_recommendations():
    result = {
        "disposition": "ok",
        "normalized_game": {"players": [{"id": "a", "name": "A"},
                                        {"id": "b", "name": "B"}], "assumptions": []},
        "method_results": [{"method": "correlated_equilibrium", "status": "exact_verified",
                            "witnesses": [{
                                "{'a': 'L', 'b': 'R'}": "1/2",
                                "{'a': 'R', 'b': 'L'}": "1/2",
                            }]}],
    }
    text = plain_view.describe_result(result)
    assert "Joint recommendations: A L, B R (50%); A R, B L (50%)" in text


def test_accordion_labels_pin_text_color_for_dark_mode():
    import re
    import webapp
    rules = re.findall(r"([^{}]+)\{([^{}]*)\}", webapp.CSS)
    pinned = [sel for sel, body in rules
              if "body.dark .gradio-container .label-wrap" in sel
              and re.search(r"color:\s*var\(--gt-text\)\s*!important", body)]
    assert pinned, "dark mode accordion label would fall back to near-white on white"


def test_mixed_strategy_preserves_tiny_exact_probability():
    result = {"disposition": "ok",
              "normalized_game": {"players": [{"id": "a", "name": "A"}],
                                  "assumptions": []},
              "method_results": [{"method": "mixed_nash", "status": "exact_verified",
                                  "witnesses": [{"a": {"rare": "1/3000",
                                                       "common": "2999/3000"}}]}]}
    text = plain_view.describe_result(result)
    assert "1/3000" in text and "rare" in text
def test_draft_summary_renders_numeric_action_ids_and_player_names():
    spec = {"template_id": "custom", "game": {
        "kind": "normal_form",
        "players": [{"id": "p1", "name": 7}, {"id": "p2", "name": "B"}],
        "actions": {"p1": [1, 2], "p2": [3]},
        "payoffs": [
            {"action_profile": {"p1": action, "p2": 3},
             "utilities": {"p1": 1, "p2": 2}}
            for action in (1, 2)
        ]}}
    text, ok = plain_view.describe_draft(json.dumps(spec))
    assert ok
    assert "Who is involved:** 7, B" in text
    assert "1 | 7 1, B 2" in text


def test_mixed_strategy_preserves_exact_fraction_near_one():
    result = {"disposition": "ok",
              "normalized_game": {"players": [{"id": "a", "name": "A"}],
                                  "assumptions": []},
              "method_results": [{"method": "mixed_nash", "status": "exact_verified",
                                  "witnesses": [{"a": {"common": "9999/10000",
                                                       "rare": "1/10000"}}]}]}
    text = plain_view.describe_result(result)
    assert "9999/10000" in text
    assert "99.99%" in text
    assert "100%" not in text




def test_draft_with_missing_player_id_returns_invalid_message():
    malformed = {"template_id": "custom", "game": {
        "kind": "normal_form", "players": [{"name": "A"}],
        "actions": {"a": ["x"]}, "payoffs": []}}
    text, ok = plain_view.describe_draft(json.dumps(malformed))
    assert not ok and "can't be used" in text


def test_draft_with_missing_bayesian_type_profile_returns_invalid_message():
    malformed = {"template_id": "custom", "game": {
        "kind": "bayesian", "players": [{"id": "a"}],
        "type_spaces": {"a": ["high"]}, "actions": {"a": ["x"]},
        "prior": [{"probability": "1"}], "payoffs": []}}
    text, ok = plain_view.describe_draft(json.dumps(malformed))
    assert not ok and "can't be used" in text
