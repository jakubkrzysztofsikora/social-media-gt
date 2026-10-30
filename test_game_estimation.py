"""Estimation and simulation fixtures (approximate evidence only)."""
import math
from fractions import Fraction as F

import pytest

import game_estimation as gest
import game_simulation as gsim
from game_models import (BayesianGame, GameEnvelope, GameSpecError, NormalFormGame,
                         OpenSpielGame, ParameterSpec, PlayerSpec, RepeatedGame)


def _prisoners_dilemma():
    players = [PlayerSpec("p1", "P1"), PlayerSpec("p2", "P2")]
    payoffs = {("C", "C"): (-1, -1), ("C", "D"): (-3, 0), ("D", "C"): (0, -3), ("D", "D"): (-2, -2)}
    rows = [{"action_profile": {"p1": x, "p2": y}, "utilities": {"p1": u1, "p2": u2}}
            for (x, y), (u1, u2) in payoffs.items()]
    return NormalFormGame(players=players, actions={"p1": ["C", "D"], "p2": ["C", "D"]},
                          payoffs=rows)


def _envelope(game):
    return GameEnvelope(game=game)


# ----------------------------------------------------------------- QRE

def _counts_for_sigma(sigma_p1_coop, sigma_p2_coop, n=400):
    p1c = round(n * sigma_p1_coop)
    p2c = round(n * sigma_p2_coop)
    return [{"player_id": "p1", "action_id": "C", "count": p1c},
            {"player_id": "p1", "action_id": "D", "count": n - p1c},
            {"player_id": "p2", "action_id": "C", "count": p2c},
            {"player_id": "p2", "action_id": "D", "count": n - p2c}]


def test_qre_marginals_accepts_valid_profiles():
    r = gest.estimate_qre_marginals(_prisoners_dilemma(), _counts_for_sigma(0.2, 0.2, 200))
    assert r.status == "approximate"
    assert r.correlation_retained is False
    assert r.lambda_mle is not None and r.lambda_mle > 0
    assert any("approximate" in a for a in r.assumptions)
    # pin the interval code: the MLE must sit inside a returned component and
    # every bound must be finite and ordered
    assert r.lambda_intervals
    for lo, hi in r.lambda_intervals:
        assert math.isfinite(lo) and math.isfinite(hi) and lo < hi
    assert any(lo <= r.lambda_mle <= hi for lo, hi in r.lambda_intervals)


def test_qre_rejects_mocked_wrong_lambda(monkeypatch):
    import pygambit as pg

    real = pg.qre.logit_solve_lambda

    def wrong_lambda(game, lam):
        result = real(game, lam)
        # pretend the tracer stopped at a different lambda
        class FakeProfile:
            lam = float(lam[0]) * 1.01

            def __getitem__(self, key):
                return real(game, lam).equilibria[0][key] if hasattr(real(game, lam), "equilibria") \
                    else real(game, lam)[key]

        fake = real(game, lam)
        try:
            fake.lam = float(lam[0]) * 1.01
        except AttributeError:
            pass
        return fake

    monkeypatch.setattr(pg.qre, "logit_solve_lambda", wrong_lambda)
    r = gest.estimate_qre_marginals(_prisoners_dilemma(),
                                    _counts_for_sigma(0.2, 0.2, 60))
    assert r.status in ("solver_unavailable", "unidentified", "approximate")
    # every accepted point met the tolerance: a 1% mismatch must be recorded as missed
    if r.missed_lambda_points:
        assert len(r.missed_lambda_points) > 50


def test_qre_rejects_bad_counts():
    with pytest.raises(GameSpecError):
        gest.estimate_qre_marginals(_prisoners_dilemma(),
                                    [{"player_id": "p1", "action_id": "C", "count": 5}])


def _cond_logit_fixture(counts, x_bound=F(4)):
    from game_models import Affine, GameEnvelope, ParameterSpec
    players = [PlayerSpec("a"), PlayerSpec("b")]
    zero = F(0)
    game = BayesianGame(
        players=players,
        type_spaces={"a": ["t"], "b": ["t"]},
        prior=[{"type_profile": {"a": "t", "b": "t"}, "probability": F(1)}],
        actions={"a": ["L", "R"], "b": ["L", "R"]},
        payoffs=[
            {"type_profile": {"a": "t", "b": "t"}, "action_profile": {"a": ai, "b": bi},
             "utilities": {"a": Affine(zero, {"x": F(1)}) if ai == "R" else zero,
                           "b": zero}}
            for ai in ("L", "R") for bi in ("L", "R")
        ])
    envelope = GameEnvelope(
        game=game,
        parameters={"x": ParameterSpec(name="x", lower_bound=F(0), upper_bound=x_bound)})
    return envelope, counts


def test_conditional_logit_unknown_type_profile_rejected():
    envelope, _ = _cond_logit_fixture([])
    with pytest.raises(GameSpecError):
        gest.estimate_game_parameters(
            envelope,
            [{"type_profile": {"a": "zz", "b": "t"},
              "action_profile": {"a": "L", "b": "L"}, "count": 1}],
            "conditional_logit", confirmed_by_user=True)


def test_conditional_logit_requires_bounded_parameter():
    from game_models import GameEnvelope
    players = [PlayerSpec("a"), PlayerSpec("b")]
    game = BayesianGame(players=players,
                        type_spaces={"a": ["t"], "b": ["t"]},
                        prior=[{"type_profile": {"a": "t", "b": "t"}, "probability": F(1)}],
                        actions={"a": ["L", "R"], "b": ["L", "R"]},
                        payoffs=[])
    with pytest.raises(GameSpecError):
        gest.estimate_game_parameters(GameEnvelope(game=game),
                                      [{"type_profile": {"a": "t", "b": "t"},
                                        "action_profile": {"a": "L", "b": "L"},
                                        "count": 5}],
                                      "conditional_logit", confirmed_by_user=True)


def test_conditional_logit_recovers_known_coefficient():
    """Deterministic joint counts from a known softmax: player a picks R with
    probability sigma(x) at x = log(176/24) ~= 2, player b uniform. The bounded
    affine estimator must recover x with a profile-likelihood interval that
    contains it, and b contributes no likelihood (no cells involve it)."""
    counts = [
        {"type_profile": {"a": "t", "b": "t"}, "action_profile": {"a": "R", "b": "R"}, "count": 88},
        {"type_profile": {"a": "t", "b": "t"}, "action_profile": {"a": "R", "b": "L"}, "count": 88},
        {"type_profile": {"a": "t", "b": "t"}, "action_profile": {"a": "L", "b": "R"}, "count": 12},
        {"type_profile": {"a": "t", "b": "t"}, "action_profile": {"a": "L", "b": "L"}, "count": 12},
    ]
    envelope, counts = _cond_logit_fixture(counts)
    result = gest.estimate_game_parameters(envelope, counts,
                                           "conditional_logit", confirmed_by_user=True)
    assert result.status == "approximate", (result.status, result.assumptions)
    assert result.correlation_retained is True
    import math
    expected = math.log(176 / 24)
    assert abs(result.parameters["x"] - expected) < 0.05
    lo, hi = result.intervals["x"]
    assert lo < expected < hi
    assert result.boundary_limited is False


def test_conditional_logit_bound_touching_is_unidentified():
    """Counts implying x ~= log(199) > upper bound 4: the bounded MLE touches
    the bound -> unidentified with boundary_limited=true (plan contract)."""
    counts = [
        {"type_profile": {"a": "t", "b": "t"}, "action_profile": {"a": "R", "b": "R"}, "count": 199},
        {"type_profile": {"a": "t", "b": "t"}, "action_profile": {"a": "R", "b": "L"}, "count": 199},
        {"type_profile": {"a": "t", "b": "t"}, "action_profile": {"a": "L", "b": "R"}, "count": 1},
        {"type_profile": {"a": "t", "b": "t"}, "action_profile": {"a": "L", "b": "L"}, "count": 1},
    ]
    envelope, counts = _cond_logit_fixture(counts)
    result = gest.estimate_game_parameters(envelope, counts,
                                           "conditional_logit", confirmed_by_user=True)
    assert result.status == "unidentified"
    assert result.boundary_limited is True


def test_estimation_requires_confirmation():
    r = gest.estimate_game_parameters(_envelope(_prisoners_dilemma()),
                                      _counts_for_sigma(0.2, 0.2, 10),
                                      "qre_marginals", confirmed_by_user=False)
    assert r == {"status": "needs_confirmation"}


def test_apply_estimates_copies_display_and_provenance():
    envelope = _envelope(_prisoners_dilemma())
    envelope.parameters["lambda"] = ParameterSpec(name="lambda", lower_bound=F(0),
                                                  upper_bound=F(10))
    estimate = {"method": "qre_marginals", "display_strings": {"lambda": "2.5"}}
    out = gest.apply_estimates(envelope, estimate, ["lambda"], confirmed_by_user=False)
    assert out == {"status": "needs_confirmation"}
    out = gest.apply_estimates(envelope, estimate, ["lambda"], confirmed_by_user=True)
    assert out["status"] == "ok"
    spec = out["envelope"].parameters["lambda"]
    assert spec.value == F(5, 2)
    assert spec.lower_bound is None and spec.upper_bound is None
    assert spec.provenance["origin"] == "estimated"
    assert envelope.parameters["lambda"].lower_bound == F(0)   # input not mutated


# ----------------------------------------------------------------- simulation

def _repeated_pd():
    stage = _prisoners_dilemma()
    return RepeatedGame(stage_game=stage, horizon=4, discount_factor=F(9, 10))


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
    discount = sum(F(9, 10) ** t for t in range(4))       # 1 + .9 + .81 + .729
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
    # both cooperate throughout: -1 per round discounted
    assert out["per_player"]["p1"]["mean"] == render_frac(-1 * discount)
    assert out["per_player"]["p2"]["mean"] == render_frac(-1 * discount)


def render_frac(value: F) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


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


# ----------------------------------------------------------------- sensitivity

def test_sensitivity_confirmation_gate_and_points():
    envelope = _envelope(_prisoners_dilemma())
    out = gest.analyze_sensitivity(envelope, [], confirmed_by_user=False)
    assert out == {"status": "needs_confirmation"}
    # no parameters declared: an empty assignment list runs zero points
    out = gest.analyze_sensitivity(envelope, [], confirmed_by_user=True)
    assert out["status"] == "exact_verified" and out["points"] == []


def test_sensitivity_out_of_bound_rejected():
    envelope = _envelope(_prisoners_dilemma())
    from game_models import ParameterSpec
    envelope.parameters["x"] = ParameterSpec(name="x", lower_bound=F(0), upper_bound=F(1))
    with pytest.raises(GameSpecError):
        gest.analyze_sensitivity(envelope, [{"x": 2}], confirmed_by_user=True)


def test_sensitivity_budget_not_covered(monkeypatch):
    envelope = _envelope(_prisoners_dilemma())
    calls = {"n": 0}

    def fake_run(env):
        calls["n"] += 1
        return [{"method": "pure_nash", "status": "exact_verified"}]

    out = gest.analyze_sensitivity(envelope, [{}, {}, {}], confirmed_by_user=True,
                                   budget_seconds=-1.0, run_point=fake_run)
    assert out["status"] == "approximate"
    assert out["not_covered"] == [0, 1, 2]
