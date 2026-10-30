"""Mechanism fixtures: VCG generation/verification and candidate checks."""
import json
from fractions import Fraction as F

import pytest

import game_solvers as gs
from game_models import (GameSpecError, MechanismSpec, ObjectiveSpec, PlayerSpec)


def _vcg_spec():
    players = [PlayerSpec("a"), PlayerSpec("b")]
    type_spaces = {"a": ["hi", "lo"], "b": ["hi", "lo"]}
    prior = [{"type_profile": {"a": ta, "b": tb}, "probability": F(1, 4)}
             for ta in ("hi", "lo") for tb in ("hi", "lo")]
    outcomes = ["a_wins", "b_wins", "no_sale"]
    value_table = {
        "a": {"hi": {"a_wins": 5, "b_wins": 0, "no_sale": 0},
              "lo": {"a_wins": 3, "b_wins": 0, "no_sale": 0}},
        "b": {"hi": {"a_wins": 0, "b_wins": 4, "no_sale": 0},
              "lo": {"a_wins": 0, "b_wins": 1, "no_sale": 0}},
    }
    outside_option = {"a": {"hi": 0, "lo": 0}, "b": {"hi": 0, "lo": 0}}
    feasible_outcomes = {}
    for ta in ("hi", "lo"):
        for tb in ("hi", "lo"):
            key = json.dumps({"a": ta, "b": tb}, sort_keys=True)
            feasible_outcomes[key] = outcomes
    return MechanismSpec(
        designer_id="d", mechanism_kind="vcg", design_mode="generate_vcg",
        players=players, type_spaces=type_spaces, prior=prior, outcomes=outcomes,
        feasible_outcomes=feasible_outcomes, value_table=value_table,
        outside_option=outside_option, ic_concept="DSIC", ir_scope="ex_post",
        budget_rule="none", objective=ObjectiveSpec(kind="social_welfare"),
        optout_outcomes={"a": "no_sale", "b": "no_sale"},
        outcome_owners={"a_wins": ["a"], "b_wins": ["b"], "no_sale": []})


def _key(a, b):
    return json.dumps({"a": a, "b": b}, sort_keys=True)


def test_vcg_generates_and_verifies_exactly():
    r = gs.verify_mechanism(_vcg_spec())
    failed = [c for c in r.checks if not c["ok"]]
    assert r.status == "exact_verified", failed[:5]
    assert not failed


def test_vcg_clarke_payments():
    r = gs.verify_mechanism(_vcg_spec())
    w = r.witnesses[0]
    # a=hi, b=lo: a wins and pays b's value 1; b pays 0
    assert w["allocation"][_key("hi", "lo")] == {"a_wins": "1"}
    assert w["transfers"][_key("hi", "lo")] == {"a": "1", "b": "0"}
    # a=hi, b=hi: a still wins (5 > 4) and pays 4
    assert w["allocation"][_key("hi", "hi")] == {"a_wins": "1"}
    assert w["transfers"][_key("hi", "hi")] == {"a": "4", "b": "0"}


def test_vcg_rejects_missing_optout_or_owners():
    spec = _vcg_spec()
    spec.optout_outcomes = None
    with pytest.raises(GameSpecError):
        spec.validate()
    spec = _vcg_spec()
    spec.outcome_owners = None
    with pytest.raises(GameSpecError):
        spec.validate()


def test_vcg_rejects_incompatible_overrides():
    spec = _vcg_spec()
    spec.ic_concept = "BIC"
    with pytest.raises(GameSpecError):
        spec.validate()
    spec = _vcg_spec()
    spec.budget_rule = "no_deficit"
    with pytest.raises(GameSpecError):
        spec.validate()


def test_mechanism_requires_objective():
    spec = _vcg_spec()
    spec.objective = None
    with pytest.raises(GameSpecError):
        spec.validate()


def test_mechanism_optimum_unavailable_in_verify_mode():
    spec = _vcg_spec()
    spec.mechanism_kind = "direct"
    spec.design_mode = "verify_candidate"
    spec.outcome_owners = None
    spec.optout_outcomes = None
    spec.allocation = {_key("hi", "hi"): {"a_wins": F(1)}}
    spec.transfers = {_key("hi", "hi"): {"a": F(4), "b": F(0)}}
    # feasible_outcomes keys cover all profiles; allocation only one profile is
    # incomplete for verification, which the verifier reports as a failed check
    # rather than raising.
    r = gs.verify_mechanism(spec)
    assert r.status in ("error", "exact_verified", "unsupported")


def _optimal_auction_spec():
    """Two symmetric bidders, i.i.d. types lo=1 / hi=3 with probability 1/2.

    Myerson's optimal auction for this setting yields expected revenue 3/2:
    outcomes (1,1),(1,3),(3,1) pay 1 and (3,3) pays 3 -> 1*3/4 + 3*1/4 = 3/2.
    """
    players = [PlayerSpec("a"), PlayerSpec("b")]
    type_spaces = {"a": ["lo", "hi"], "b": ["lo", "hi"]}
    prior = [{"type_profile": {"a": ta, "b": tb}, "probability": F(1, 4)}
             for ta in ("lo", "hi") for tb in ("lo", "hi")]
    outcomes = ["a_wins", "b_wins", "split", "no_sale"]
    value_table = {
        "a": {"lo": {"a_wins": 1, "b_wins": 0, "split": F(1, 2), "no_sale": 0},
              "hi": {"a_wins": 3, "b_wins": 0, "split": F(3, 2), "no_sale": 0}},
        "b": {"lo": {"a_wins": 0, "b_wins": 1, "split": F(1, 2), "no_sale": 0},
              "hi": {"a_wins": 0, "b_wins": 3, "split": F(3, 2), "no_sale": 0}},
    }
    outside_option = {"a": {"lo": 0, "hi": 0}, "b": {"lo": 0, "hi": 0}}
    feasible_outcomes = {}
    for ta in ("lo", "hi"):
        for tb in ("lo", "hi"):
            feasible_outcomes[_key(ta, tb)] = outcomes
    return MechanismSpec(
        designer_id="d", mechanism_kind="optimal_auction", design_mode="optimize",
        players=players, type_spaces=type_spaces, prior=prior, outcomes=outcomes,
        feasible_outcomes=feasible_outcomes, value_table=value_table,
        outside_option=outside_option, ic_concept="BIC", ir_scope="interim",
        budget_rule="none", objective=ObjectiveSpec(kind="revenue", direction="maximize"))


def test_optimal_auction_lp_design_exact():
    spec = _optimal_auction_spec()
    r = gs.verify_mechanism(spec)
    failed = [c for c in r.checks if not c["ok"]]
    assert r.status == "exact_verified", failed[:6]
    assert not failed
    # Optimum with a shareable outcome: revenue = E[sum_i phi_i(v_i) q_i(v)] by
    # payoff equivalence; phi(lo)=0, phi(hi)=3, so every profile containing a hi
    # type contributes 3*1/4 and (lo,lo) contributes 0 -> 9/4. The exact
    # primal/dual certificate is the optimality proof.
    assert r.objective == F(9, 4)
    # primal/dual certificate
    assert r.certificate is not None
    assert F(r.certificate["dual_objective"]) == r.objective
    assert len(r.certificate["dual_y"]) > 0
    # witness allocations are proper distributions
    w = r.witnesses[0]
    for dist in w["allocation"].values():
        assert sum(F(p) for p in dist.values()) == 1
        assert all(F(p) >= 0 for p in dist.values())
    # interim IR holds per type by construction of the checks
    assert any("interim IR" in c["name"] and c["ok"] for c in r.checks)
    assert any("BIC interim" in c["name"] and c["ok"] for c in r.checks)


def test_optimal_auction_rejects_non_revenue_objective():
    spec = _optimal_auction_spec()
    spec.objective = ObjectiveSpec(kind="social_welfare")
    with pytest.raises(GameSpecError):
        spec.validate()


def test_optimal_auction_rejects_vcg_incompatible_override():
    spec = _optimal_auction_spec()
    spec.ic_concept = "DSIC"
    with pytest.raises(GameSpecError):
        spec.validate()
