"""Unit tests for the social-media-gt verification toolkit."""
from fractions import Fraction as F

import pytest

from social_media_gt import (
    tw1_static_equilibrium,
    tw1_1_screen_time_cap,
    tw2_cascade,
    tw3_absorption,
    tw3_2_global_games,
    tw5_coalition_thresholds,
    tw6_coalition_continuum,
    tw7_pressure_reduction,
    tw7_2_hybrid_policy,
    tw8_sequential_exit,
    run_all,
)


def all_pass(result):
    failed = [c for c in result["checks"] if not c["ok"]]
    assert not failed, failed
    return result


def test_tw1_defaults():
    res = all_pass(tw1_static_equilibrium(tau=F(1, 2), f1=F(1, 4), E1=F(1, 4)))
    assert res["results"]["delta_at_zero"] == -F(1, 2)
    assert res["results"]["all_a_is_ne"] is True
    assert res["results"]["all_r_dominates_pareto"] is True


def test_tw1_custom_parameters_flip_pareto_verdict():
    assert tw1_static_equilibrium(tau=F(1, 4), f1=F(1, 4), E1=F(1, 4))["results"][
        "all_r_dominates_pareto"] is False


def test_tw1_1_defaults_match_paper():
    res = all_pass(tw1_1_screen_time_cap(a=F(1), b=F(1), T=F(4), beta=F(1)))
    assert res["results"]["t_first_best"] == F(1, 2)
    assert res["results"]["t_nash"] == F(4, 7)
    assert res["results"]["welfare_loss"] == F(1, 196)


def test_tw1_1_rejects_out_of_domain():
    with pytest.raises(ValueError):
        tw1_1_screen_time_cap(a=F(9, 5), b=F(1), T=F(1), beta=F(9, 5))


def test_tw2_defaults_cascade_to_full_adoption():
    res = all_pass(tw2_cascade())
    assert res["results"]["limit"] == 1
    assert res["results"]["rounds_to_full_adoption"] == 19
    assert res["results"]["phi_minus_k_on_linear_region"] == F(1, 20)


def test_tw3_all_a_absorbing_at_defaults():
    res = all_pass(tw3_absorption(tau=F(1, 2), n=4))
    assert res["results"]["all_a_is_equilibrium"] is True
    assert res["results"]["equilibria_count"] == 2


def test_tw3_2_risk_dominant_threshold():
    res = all_pass(tw3_2_global_games(f1=F(1), E1=F(3)))
    assert res["results"]["tau_risk_dominant"] == F(2)
    assert res["results"]["pareto_trap_class_nonempty"] is True
    assert tw3_2_global_games(f1=F(3), E1=F(1))["results"]["pareto_trap_class_nonempty"] is False


def test_tw5_thresholds_at_defaults():
    res = all_pass(tw5_coalition_thresholds(tau=F(3, 2), n=10, d=1))
    assert res["results"]["m_out"] == 4
    assert res["results"]["m_stab"] == 5
    assert res["results"]["P_peer_5_of_9"] == F(115, 81)
    assert res["results"]["P_peer_6_of_9"] == F(16, 9)


def test_tw5_higher_threshold_needs_fewer_pact_members():
    res = tw5_coalition_thresholds(tau=F(2), n=10, d=1)
    assert res["results"]["m_out"] == 3


def test_tw6_defaults_match_paper():
    res = all_pass(tw6_coalition_continuum(gamma=F(9, 40)))
    assert res["results"]["K"] == F(1, 4)
    assert res["results"]["a"] == F(3, 4)
    assert res["results"]["trajectory"][:3] == [F(31, 40), F(103, 160), F(349, 640)]
    assert res["results"]["exit_multiplier"] == F(10, 3)


def test_tw7_branches_at_defaults():
    assert all_pass(tw7_pressure_reduction(lam=F(1, 5), eta=F(0)))["results"]["limit"] == F(1, 4)
    assert tw7_pressure_reduction(lam=F(1, 2), eta=F(0))["results"]["limit"] == F(1, 10)
    assert tw7_pressure_reduction(lam=F(1, 20), eta=F(0))["results"]["limit"] == 1


def test_tw7_evasion_threshold():
    res = tw7_pressure_reduction(lam=F(1, 5), eta=F(3, 4))
    assert res["results"]["limit"] == 1
    assert res["results"]["eta_critical"] == F(3, 4)


def test_tw7_2_hybrid_policy_defaults():
    res = all_pass(tw7_2_hybrid_policy(creg=F(1), cplat=F(1), eta=F(1, 2), lam_target=F(3, 10)))
    assert res["results"]["lambda_reg"] == F(3, 25)
    assert res["results"]["lambda_plat"] == F(6, 25)
    assert res["results"]["ratio"] == F(1, 2)


def test_tw7_2_custom_costs_shift_ratio():
    res = tw7_2_hybrid_policy(creg=F(4), cplat=F(1), eta=F(1, 2), lam_target=F(3, 10))
    assert res["results"]["lambda_reg"] == F(3, 85)
    assert res["results"]["lambda_plat"] == F(24, 85)


def test_tw8_sums_and_worst_sequence():
    res = all_pass(tw8_sequential_exit(tau_C=F(2), delta=F(99, 100), full=False))
    assert res["results"]["undiscounted_sum"] == F(94, 27)
    assert res["results"]["discounted_min"] == F(3064822137458430997, 10 ** 18)
    assert res["results"]["discounted_min"] >= F(697, 270)
    assert res["results"]["worst_sequence"] == [F(1), F(1)] + [F(10 - t, 9) for t in range(2, 10)]


def test_all_scenarios_pass_at_defaults():
    summary = run_all()
    assert summary["scenarios_run"] == 10
    assert summary["checks_failed"] == 0, summary


def test_cli_reports_all_checks_pass(capsys):
    from social_media_gt import main

    exit_code = main(["all"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "FAIL" not in out
    assert "10/10 scenarios" in out


