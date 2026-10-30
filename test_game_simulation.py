"""Simulation fixtures (approximate evidence only)."""
from fractions import Fraction as F

import pytest

import game_simulation as gsim
from game_models import (GameEnvelope, GameSpecError, NormalFormGame, OpenSpielGame,
                         PlayerSpec, RepeatedGame)


def _prisoners_dilemma():
    players = [PlayerSpec("p1", "P1"), PlayerSpec("p2", "P2")]
    payoffs = {("C", "C"): (-1, -1), ("C", "D"): (-3, 0), ("D", "C"): (0, -3), ("D", "D"): (-2, -2)}
    rows = [{"action_profile": {"p1": x, "p2": y}, "utilities": {"p1": u1, "p2": u2}}
            for (x, y), (u1, u2) in payoffs.items()]
    return NormalFormGame(players=players, actions={"p1": ["C", "D"], "p2": ["C", "D"]},
                          payoffs=rows)


def _repeated_pd():
    return RepeatedGame(stage_game=_prisoners_dilemma(), horizon=4, discount_factor=F(9, 10))


def render_frac(value: F) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def test_simulation_requires_confirmation():
    out = gsim.simulate_game(GameEnvelope(game=_repeated_pd()), "simulation", {}, 5, 0,
                             confirmed_by_user=False)
    assert out == {"status": "needs_confirmation"}


def test_named_policies_episode_totals_exact():
    game_env = GameEnvelope(game=_repeated_pd())
    policies = {
        "p1": {"name": "always_cooperate", "cooperate_action_id": "C", "defect_action_id": "D"},
        "p2": {"name": "always_defect", "cooperate_action_id": "C", "defect_action_id": "D"},
    }
    out = gsim.simulate_game(game_env, "simulation", policies, 3, 7, confirmed_by_user=True)
    assert out["status"] == "approximate"
    # (C vs D) each round: p1 gets -3, p2 gets 0; cumulative sum Sigma delta^t u_t
    discount = sum(F(9, 10) ** t for t in range(4))
    assert out["per_player"]["p1"]["mean"] == render_frac(-3 * discount)
    assert out["per_player"]["p2"]["mean"] == "0"


def test_grim_trigger_transitions():
    game_env = GameEnvelope(game=_repeated_pd())
    policies = {
        "p1": {"name": "grim_trigger", "cooperate_action_id": "C", "defect_action_id": "D"},
        "p2": {"name": "tit_for_tat", "cooperate_action_id": "C", "defect_action_id": "D"},
    }
    out = gsim.simulate_game(game_env, "simulation", policies, 1, 1, confirmed_by_user=True)
    assert out["status"] == "approximate"
    discount = sum(F(9, 10) ** t for t in range(4))
    # p1 cooperates unconditionally; p2 (tit-for-tat) cooperates back; no defection
    assert out["per_player"]["p1"]["mean"] == render_frac(-1 * discount)
    assert out["per_player"]["p2"]["mean"] == render_frac(-1 * discount)


def test_grim_triggers_on_public_defection():
    game_env = GameEnvelope(game=_repeated_pd())
    policies = {
        "p1": {"name": "grim_trigger", "cooperate_action_id": "C", "defect_action_id": "D"},
        "p2": {"name": "always_defect", "cooperate_action_id": "C", "defect_action_id": "D"},
    }
    out = gsim.simulate_game(game_env, "simulation", policies, 1, 1, confirmed_by_user=True)
    # round 1: p1 C, p2 D -> triggered; rounds 2-4: both D
    d = F(9, 10)
    p1 = -3 + (-2) * (d + d * d + d * d * d)
    p2 = 0 + (-2) * (d + d * d + d * d * d)
    assert out["per_player"]["p1"]["mean"] == render_frac(p1)
    assert out["per_player"]["p2"]["mean"] == render_frac(p2)


def test_cfr_kuhn_poker_approximate():
    game_env = GameEnvelope(game=OpenSpielGame(game_id="kuhn_poker",
                                               players=[PlayerSpec("p1"), PlayerSpec("p2")]))
    out = gsim.simulate_game(game_env, "cfr", {}, episodes=20, seed=3,
                             confirmed_by_user=True, iterations=50)
    assert out["status"] == "approximate"
    assert out["algorithm"] == "cfr"
    assert out["nash_conv"] >= 0
    assert "policy_interval" in out


def test_cfr_rejects_general_sum_allowlist():
    game_env = GameEnvelope(game=OpenSpielGame(game_id="matrix_pd",
                                               players=[PlayerSpec("p1"), PlayerSpec("p2")]))
    with pytest.raises(GameSpecError):
        gsim.simulate_game(game_env, "mccfr", {}, 10, 0, confirmed_by_user=True, iterations=5)


def test_simulation_validates_episodes_and_seed():
    game_env = GameEnvelope(game=_repeated_pd())
    with pytest.raises(GameSpecError):
        gsim.simulate_game(game_env, "simulation", {}, 0, 0, confirmed_by_user=True)
    with pytest.raises(GameSpecError):
        gsim.simulate_game(game_env, "simulation", {}, 5, -1, confirmed_by_user=True)
    with pytest.raises(GameSpecError):
        gsim.simulate_game(game_env, "unknown", {}, 5, 0, confirmed_by_user=True)
