"""Game-model validation fixtures: payoff cells, parameter modes, templates."""
from fractions import Fraction as F

import pytest

import game_models as gm
from game_models import (Affine, BayesianGame, ExtensiveFormGame, GameEnvelope, GameSpecError,
                         MechanismSpec, NormalFormGame, ObjectiveSpec, OpenSpielGame,
                         ParameterSpec, PlayerSpec, StackelbergGame, to_rational)


def test_to_rational_accepts_exact_forms_rejects_bool():
    assert to_rational(3, "x") == F(3)
    assert to_rational("1/2", "x") == F(1, 2)
    assert to_rational("0.25", "x") == F(1, 4)
    with pytest.raises(GameSpecError):
        to_rational(True, "x")
    with pytest.raises(GameSpecError):
        to_rational("tau*2", "x")


def test_affine_cell_evaluation_and_unknown_parameter():
    cell = Affine(F(1), {"tau": F(2)})
    assert cell.evaluate({"tau": F(3)}) == F(7)
    with pytest.raises(GameSpecError):
        cell.evaluate({})


def test_parameter_exactly_one_mode():
    with pytest.raises(GameSpecError):
        ParameterSpec(name="x").validate()
    assert ParameterSpec(name="x", value=F(1)).validate() == "point"
    assert ParameterSpec(name="x", lower_bound=F(0), upper_bound=F(1)).validate() == "interval"
    assert ParameterSpec(name="x", prior=[{"value": F(0), "probability": F(1)}]).validate() == "prior"
    with pytest.raises(GameSpecError):
        ParameterSpec(name="x", value=F(1), lower_bound=F(0), upper_bound=F(1)).validate()


def test_prior_must_sum_to_one():
    with pytest.raises(GameSpecError):
        ParameterSpec(name="x", prior=[{"value": F(0), "probability": F(1, 2)}]).validate()


def test_normal_form_rejects_incomplete_payoffs():
    players = [PlayerSpec("a"), PlayerSpec("b")]
    game = NormalFormGame(players=players, actions={"a": ["L"], "b": ["L", "R"]},
                          payoffs=[{"action_profile": {"a": "L", "b": "L"},
                                    "utilities": {"a": 1, "b": 1}}])
    with pytest.raises(GameSpecError):
        game.validate()


def test_objective_rules():
    players = ["a", "b"]
    with pytest.raises(GameSpecError):
        ObjectiveSpec(kind="player_utility").validate(players)
    with pytest.raises(GameSpecError):
        ObjectiveSpec(kind="revenue").validate(players, is_mechanism=False)
    with pytest.raises(GameSpecError):
        ObjectiveSpec(kind="maximin", direction="minimize").validate(players)
    ObjectiveSpec(kind="revenue").validate(players, is_mechanism=True)


def test_mechanism_rejects_incompatible_vcg_overrides():
    players = [PlayerSpec("a")]
    spec = MechanismSpec(designer_id="d", mechanism_kind="vcg", design_mode="verify_candidate",
                         players=players, type_spaces={"a": ["t"]},
                         prior=[{"type_profile": {"a": "t"}, "probability": F(1)}],
                         outcomes=["o"], feasible_outcomes={}, value_table={}, outside_option={},
                         ic_concept="BIC")
    with pytest.raises(GameSpecError):
        spec.validate()


def test_open_spiel_allowlist():
    with pytest.raises(GameSpecError):
        OpenSpielGame(game_id="poker", players=[PlayerSpec("a")]).validate()
    OpenSpielGame(game_id="kuhn_poker", players=[PlayerSpec("a")]).validate()


def test_point_parameters_required_for_solves():
    envelope = GameEnvelope(game=OpenSpielGame(game_id="kuhn_poker", players=[PlayerSpec("a")]),
                            parameters={"x": ParameterSpec(name="x", lower_bound=F(0),
                                                           upper_bound=F(1))})
    with pytest.raises(GameSpecError):
        envelope.point_parameters()


def test_compile_pricing_auction_second_price():
    options = {
        "players": ["a", "b"],
        "auction_rule": "second_price",
        "types": {"a": ["lo", "hi"], "b": ["lo", "hi"]},
        "prior": [{"type_profile": {"a": t_a, "b": t_b}, "probability": F(1, 4)}
                  for t_a in ("lo", "hi") for t_b in ("lo", "hi")],
        "bid_grids": {"a": [1, 2], "b": [1, 2]},
        "values": {"a": {"lo": 1, "hi": 3}, "b": {"lo": 1, "hi": 3}},
        "tie_rule": "player_order",
    }
    envelope = gm.compile_template("pricing_auction", options)
    game = envelope.game
    assert isinstance(game, BayesianGame)
    assert len(game.payoffs) == 16


def test_compile_bargaining_template():
    options = {"players": ["a", "b"],
               "feasible_points": [{"a": 1, "b": 1}],
               "disagreement": {"a": 0, "b": 0},
               "randomization": "not_allowed"}
    envelope = gm.compile_template("bargaining", options)
    assert envelope.commitment_rights == "none"


def test_compile_unknown_template_rejected():
    with pytest.raises(GameSpecError):
        gm.compile_template("weather", {})

def test_p3_benchmark_is_labelled_and_separate_from_user_models():
    envelope = gm.compile_template("p3_heterogeneous", {})
    game = envelope.game
    assert envelope.template_options == {}
    assert [player.role for player in game.players] == ["C"] * 9 + ["D"]
    assert envelope.assumptions[0] == "P3 paper benchmark preset."
    assert any("repeated-game construction is outside" in item
               for item in envelope.assumptions)
    all_adopt = {"C1": "A", "C2": "A", "C3": "A", "C4": "A", "C5": "A",
                 "C6": "A", "C7": "A", "C8": "A", "C9": "A", "D": "A"}
    row = next(row for row in game.payoffs if row["action_profile"] == all_adopt)
    assert set(row["utilities"].values()) == {F(1)}


def test_p3_custom_model_uses_declared_types_thresholds_and_coefficients():
    options = {
        "type_counts": {"urban": 2, "rural": 1},
        "thresholds": {"urban": "1", "rural": "-2"},
        "payoff_coefficients": {
            "f": {"constant": "1/2", "linear": "2"},
            "E": {"constant": "1/4", "linear": "1/2", "quadratic": "3"},
        },
    }
    envelope = gm.compile_template("p3_heterogeneous", options)
    game = envelope.game
    assert envelope.template_options == options
    assert [(player.id, player.role) for player in game.players] == [
        ("p1", "urban"), ("p2", "urban"), ("p3", "rural")]
    assert "User-specified P3 one-shot model; not the paper benchmark." in envelope.assumptions
    assert any("repeated-game construction is outside" in item
               for item in envelope.assumptions)

    def utilities(profile):
        row = next(row for row in game.payoffs if row["action_profile"] == profile)
        return row["utilities"]

    assert utilities({"p1": "R", "p2": "R", "p3": "R"}) == {
        "p1": F(3, 4), "p2": F(3, 4), "p3": F(-9, 4)}
    assert utilities({"p1": "A", "p2": "A", "p3": "A"}) == {
        "p1": F(5, 2), "p2": F(5, 2), "p3": F(5, 2)}
    assert utilities({"p1": "A", "p2": "A", "p3": "R"})["p3"] == F(-23, 4)


def test_p3_custom_model_requires_all_configuration_fields():
    with pytest.raises(GameSpecError, match="require all"):
        gm.compile_template("p3_heterogeneous", {"type_counts": {"C": 2}})


PUBLIC_GOODS_OPTIONS = {
    "players": [{"id": "up", "name": "Upstream"}, {"id": "down", "name": "Downstream"}],
    "actions": {"up": ["clean", "dump"], "down": ["clean", "dump"]},
    "payoffs": [
        {"action_profile": {"up": "clean", "down": "clean"},
         "utilities": {"up": "6", "down": "6"}},
        {"action_profile": {"up": "clean", "down": "dump"},
         "utilities": {"up": "1", "down": "8"}},
        {"action_profile": {"up": "dump", "down": "clean"},
         "utilities": {"up": "8", "down": "1"}},
        {"action_profile": {"up": "dump", "down": "dump"},
         "utilities": {"up": "0", "down": "0"}},
    ],
}

NEGOTIATION_OPTIONS = {
    "players": [{"id": "seller"}, {"id": "buyer"}],
    "offer_tree": {
        "root": "ask",
        "nodes": [
            {"id": "ask", "actor": "seller",
             "edges": [{"action_id": "high", "next_node": "buyer_high"},
                       {"action_id": "low", "next_node": "buyer_low"}]},
            {"id": "buyer_high", "actor": "buyer",
             "edges": [{"action_id": "accept", "next_node": "deal_high"},
                       {"action_id": "walk", "next_node": "no_deal"}]},
            {"id": "buyer_low", "actor": "buyer",
             "edges": [{"action_id": "accept", "next_node": "deal_low"},
                       {"action_id": "walk", "next_node": "no_deal"}]},
            {"id": "deal_high", "terminal_utilities": {"seller": "8", "buyer": "2"}},
            {"id": "deal_low", "terminal_utilities": {"seller": "4", "buyer": "5"}},
            {"id": "no_deal", "terminal_utilities": {"seller": "0", "buyer": "0"}},
        ],
    },
}


def test_public_goods_compiles_to_normal_form():
    envelope = gm.compile_template("public_goods", PUBLIC_GOODS_OPTIONS)
    game = envelope.game
    assert isinstance(game, NormalFormGame)
    assert envelope.template_id == "public_goods"
    assert len(game.payoffs) == 4


def test_public_goods_requires_payoff_table_and_actions():
    broken = dict(PUBLIC_GOODS_OPTIONS, payoffs=[])
    with pytest.raises(GameSpecError, match="payoff table"):
        gm.compile_template("public_goods", broken)
    missing_actions = {k: v for k, v in PUBLIC_GOODS_OPTIONS.items() if k != "actions"}
    with pytest.raises(GameSpecError, match="action set per player"):
        gm.compile_template("public_goods", missing_actions)


def test_negotiation_compiles_perfect_information_offer_tree():
    envelope = gm.compile_template("negotiation", NEGOTIATION_OPTIONS)
    game = envelope.game
    assert isinstance(game, ExtensiveFormGame)
    assert game.has_perfect_information()
    assert {iset["id"] for iset in game.information_sets} == {"ask_info", "buyer_high_info",
                                                              "buyer_low_info"}


def test_negotiation_rejects_terminal_without_utilities_and_repeated_offers():
    nodes = [dict(node) for node in NEGOTIATION_OPTIONS["offer_tree"]["nodes"]]
    nodes[3] = {"id": "deal_high"}
    with pytest.raises(GameSpecError, match="terminal_utilities"):
        gm.compile_template("negotiation", {"players": NEGOTIATION_OPTIONS["players"],
                                            "offer_tree": {"root": "ask", "nodes": nodes}})
    duplicated = [dict(node) for node in NEGOTIATION_OPTIONS["offer_tree"]["nodes"]]
    duplicated[0] = {"id": "ask", "actor": "seller",
                     "edges": [{"action_id": "high", "next_node": "buyer_high"},
                               {"action_id": "high", "next_node": "buyer_low"}]}
    with pytest.raises(GameSpecError, match="repeats an offer action"):
        gm.compile_template("negotiation", {"players": NEGOTIATION_OPTIONS["players"],
                                            "offer_tree": {"root": "ask", "nodes": duplicated}})
