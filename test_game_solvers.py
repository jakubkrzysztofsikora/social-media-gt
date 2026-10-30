"""Solver fixtures: exact equilibria, Stackelberg ties, bargaining, Bayesian."""
from fractions import Fraction as F

import pytest

import game_solvers as gs
from game_models import (BayesianGame, BargainingSpec, ExtensiveFormGame, ExtensiveNode,
                         NormalFormGame, ObjectiveSpec, PlayerSpec, StackelbergGame)


def bots():
    players = [PlayerSpec("p1", "P1"), PlayerSpec("p2", "P2")]
    payoffs = {("L", "L"): (3, 2), ("L", "R"): (0, 0), ("M", "L"): (0, 0), ("M", "R"): (2, 3)}
    rows = [{"action_profile": {"p1": a, "p2": b}, "utilities": {"p1": u1, "p2": u2}}
            for (a, b), (u1, u2) in payoffs.items()]
    return NormalFormGame(players=players, actions={"p1": ["L", "M"], "p2": ["L", "R"]},
                          payoffs=rows)


def matching_pennies(objective=None):
    players = [PlayerSpec("a", "A"), PlayerSpec("b", "B")]
    payoffs = {("H", "H"): (-1, 1), ("H", "T"): (1, -1), ("T", "H"): (1, -1), ("T", "T"): (-1, 1)}
    rows = [{"action_profile": {"a": x, "b": y}, "utilities": {"a": u1, "b": u2}}
            for (x, y), (u1, u2) in payoffs.items()]
    return NormalFormGame(players=players, actions={"a": ["H", "T"], "b": ["H", "T"]},
                          payoffs=rows, objective=objective)


def test_pure_nash_battle_of_sexes():
    r = gs.solve_pure_nash(bots())
    assert r.status == "exact_verified"
    profiles = [{p["p1"], p["p2"]} for p in r.witnesses]
    assert len(r.witnesses) == 2


def test_mixed_nash_battle_of_sexes_exact():
    r = gs.solve_mixed_nash_two_player(bots())
    assert r.status == "exact_verified"
    assert all(isinstance(v, str)
               for w in r.witnesses for row in w.values() for v in row.values())
    mixed = [w for w in r.witnesses
             if all(sum(1 for v in w[pid].values() if F(v) > 0) > 1 for pid in w)]
    assert mixed
    w = mixed[0]
    assert w["p1"]["L"] == "3/5" and w["p2"]["R"] == "3/5"
    eu = gs._expected_utilities(
        bots(), {pid: {a: F(v) for a, v in row.items()} for pid, row in w.items()})
    assert eu["p1"] == F(6, 5) and eu["p2"] == F(6, 5)


def test_epsilon_nash_exact_regret_check():
    r = gs.solve_epsilon_nash(matching_pennies(), F(1, 10))
    assert r.status == "epsilon_verified"
    assert r.checks[0]["ok"]


def test_epsilon_nash_requires_positive():
    with pytest.raises(Exception):
        gs.solve_epsilon_nash(matching_pennies(), F(0))


def test_ce_uniform_is_exact_with_welfare_objective():
    mp = matching_pennies(objective=ObjectiveSpec(kind="social_welfare"))
    r = gs.solve_correlated_equilibrium(mp, mediator_confirmed=True, objective=mp.objective)
    assert r.status == "exact_verified"
    assert r.objective == 0
    witness = r.witnesses[0]
    assert sum(F(v) for v in witness.values()) == 1
    assert all(c["ok"] for c in r.checks)


def test_ce_requires_confirmed_mediator():
    r = gs.solve_correlated_equilibrium(matching_pennies(), mediator_confirmed=False,
                                        objective=None)
    assert r.status == "unsupported"


def test_ce_feasibility_lp_finds_nonuniform_witness():
    """Prisoners' dilemma: uniform is not a CE; the unique CE is the pure (D,D)
    Nash profile, found via the zero-objective feasibility LP."""
    players = [PlayerSpec("p1"), PlayerSpec("p2")]
    pay = {("C", "C"): (-1, -1), ("C", "D"): (-3, 0), ("D", "C"): (0, -3), ("D", "D"): (-2, -2)}
    rows = [{"action_profile": {"p1": x, "p2": y}, "utilities": {"p1": u1, "p2": u2}}
            for (x, y), (u1, u2) in pay.items()]
    game = NormalFormGame(players=players, actions={"p1": ["C", "D"], "p2": ["C", "D"]},
                          payoffs=rows)
    r = gs.solve_correlated_equilibrium(game, mediator_confirmed=True, objective=None)
    assert r.status == "exact_verified"
    witness = r.witnesses[0]
    assert witness[str({"p1": "D", "p2": "D"})] == "1"


def _stackelberg_stage():
    players = [PlayerSpec("L", "leader"), PlayerSpec("F", "follower")]
    rows = []
    for l_act, lu in (("A", {"X": 2, "Y": 0}), ("B", {"X": 1, "Y": 3})):
        for f_act in ("X", "Y"):
            rows.append({"action_profile": {"L": l_act, "F": f_act},
                         "utilities": {"L": lu[f_act], "F": 0}})
    return NormalFormGame(players=players, actions={"L": ["A", "B"], "F": ["X", "Y"]},
                          payoffs=rows)


def test_stackelberg_pure_both_tie_policies():
    game = StackelbergGame(stage_game=_stackelberg_stage(), leader_id="L",
                           follower_id="F", commitment_mode="pure")
    r = gs.solve_stackelberg(game)
    assert r.status == "exact_verified"
    pure = r.witnesses[0]
    lf = pure["leader_favorable"][0]
    ff = pure["follower_favorable"][0]
    assert (lf["leader"], lf["follower"], lf["leader_utility"]) == ("B", "Y", "3")
    assert (ff["leader"], ff["follower"], ff["leader_utility"]) == ("B", "X", "1")


def test_stackelberg_mixed_both_tie_policies():
    game = StackelbergGame(stage_game=_stackelberg_stage(), leader_id="L",
                           follower_id="F", commitment_mode="mixed")
    r = gs.solve_stackelberg(game)
    mixed = [w for w in r.witnesses if w.get("commitment") == "mixed"]
    lf = next(w for w in mixed if "leader_favorable" in w)
    ff = next(w for w in mixed if "follower_favorable" in w)
    assert lf["leader_favorable"][0]["p"]["A"] == "0"
    assert lf["leader_favorable"][0]["leader_utility_value"] == 3
    assert ff["follower_favorable"][0]["p"]["A"] == "1/2"
    assert ff["follower_favorable"][0]["leader_utility_value"] == F(3, 2)


def test_stackelberg_unattained_supremum_no_fallback():
    players = [PlayerSpec("L", "leader"), PlayerSpec("F", "follower")]
    f_pay = {("L1", "X"): 1, ("L1", "Y"): 0, ("L2", "X"): 0, ("L2", "Y"): 0}
    l_pay = {("L1", "X"): 0, ("L1", "Y"): 0, ("L2", "X"): 1, ("L2", "Y"): 0}
    rows = [{"action_profile": {"L": la, "F": fa},
             "utilities": {"L": l_pay[(la, fa)], "F": f_pay[(la, fa)]}}
            for (la, fa) in f_pay]
    stage = NormalFormGame(players=players, actions={"L": ["L1", "L2"], "F": ["X", "Y"]},
                           payoffs=rows)
    game = StackelbergGame(stage_game=stage, leader_id="L", follower_id="F",
                           commitment_mode="mixed")
    r = gs.solve_stackelberg(game)
    ff = next(w for w in r.witnesses
              if w.get("commitment") == "mixed" and "follower_favorable" in w)
    assert ff["status"] == "solver_unavailable"
    assert ff["supremum"] == "1"
    assert ff["follower_favorable"] == []


def test_stackelberg_pure_follower_favorable_ranks_by_adverse_value():
    """Regression (review M-6): leader actions must be ranked by their
    adverse-tie value, not by their best-case value. Action A: follower is
    indifferent (u_F 5/5), adverse leader value 0 (X=6, Y=0). Action B:
    follower strictly prefers Y (9), leader value 4. Worst-case ranking
    picks B/4; best-case ranking would wrongly pick A."""
    players = [PlayerSpec("L"), PlayerSpec("F")]
    f_pay = {("A", "X"): 5, ("A", "Y"): 5, ("B", "X"): 1, ("B", "Y"): 9}
    l_pay = {("A", "X"): 6, ("A", "Y"): 0, ("B", "X"): 4, ("B", "Y"): 4}
    rows = [{"action_profile": {"L": la, "F": fa},
             "utilities": {"L": l_pay[(la, fa)], "F": f_pay[(la, fa)]}}
            for (la, fa) in f_pay]
    stage = NormalFormGame(players=players, actions={"L": ["A", "B"], "F": ["X", "Y"]},
                           payoffs=rows)
    game = StackelbergGame(stage_game=stage, leader_id="L", follower_id="F",
                           commitment_mode="pure")
    r = gs.solve_stackelberg(game)
    pure = r.witnesses[0]
    lf = pure["leader_favorable"][0]
    ff = pure["follower_favorable"][0]
    assert (lf["leader"], lf["follower"], lf["leader_utility"]) == ("A", "X", "6")
    assert (ff["leader"], ff["follower"], ff["leader_utility"]) == ("B", "Y", "4")


def test_stackelberg_empty_strict_region_reports_attained_best():
    players = [PlayerSpec("L", "leader"), PlayerSpec("F", "follower")]
    f_pay = {("L1", "X"): 2, ("L1", "Y"): 0, ("L2", "X"): 2, ("L2", "Y"): 0}
    l_pay = {("L1", "X"): 1, ("L1", "Y"): 2, ("L2", "X"): 1, ("L2", "Y"): 2}
    rows = [{"action_profile": {"L": la, "F": fa},
             "utilities": {"L": l_pay[(la, fa)], "F": f_pay[(la, fa)]}}
            for (la, fa) in f_pay]
    stage = NormalFormGame(players=players, actions={"L": ["L1", "L2"], "F": ["X", "Y"]},
                           payoffs=rows)
    game = StackelbergGame(stage_game=stage, leader_id="L", follower_id="F",
                           commitment_mode="mixed")
    r = gs.solve_stackelberg(game)
    ff = next(w for w in r.witnesses
              if w.get("commitment") == "mixed" and "follower_favorable" in w)
    assert ff["follower_favorable"][0]["leader_utility_value"] == 1


def test_bargaining_finite_and_convexified():
    players = [PlayerSpec("a", "A"), PlayerSpec("b", "B")]
    spec = BargainingSpec(players=players,
                          feasible_points=[{"a": 0, "b": 3}, {"a": 3, "b": 0}, {"a": 1, "b": 1}],
                          disagreement={"a": 0, "b": 0}, randomization="not_allowed")
    r = gs.solve_nash_bargaining(spec)
    assert r.status == "exact_verified" and r.objective == 1
    assert r.witnesses[0]["a"] == "1" and r.witnesses[0]["b"] == "1"

    spec_r = BargainingSpec(players=players,
                            feasible_points=[{"a": 0, "b": 3}, {"a": 3, "b": 0}, {"a": 1, "b": 1}],
                            disagreement={"a": 0, "b": 0}, randomization="allowed")
    r2 = gs.solve_nash_bargaining(spec_r)
    assert r2.status == "exact_verified" and r2.objective == F(9, 4)
    assert any(w.get("a") == "3/2" and w.get("b") == "3/2" for w in r2.witnesses)


def test_bayesian_nash_contingent_plans():
    players = [PlayerSpec("a", "A"), PlayerSpec("b", "B")]
    prior = [{"type_profile": {"a": ta, "b": tb}, "probability": F(1, 4)}
             for ta in ("a1", "a2") for tb in ("b1", "b2")]
    payoffs = []
    for ta in ("a1", "a2"):
        for tb in ("b1", "b2"):
            good = (ta == "a1" and tb == "b1") or (ta == "a2" and tb == "b2")
            for aa in ("L", "R"):
                for ab in ("L", "R"):
                    match = (aa == "L" and ab == "L") or (aa == "R" and ab == "R")
                    u = F(1) if (good and match) else F(0)
                    payoffs.append({"type_profile": {"a": ta, "b": tb},
                                    "action_profile": {"a": aa, "b": ab},
                                    "utilities": {"a": u, "b": u}})
    game = BayesianGame(players=players, type_spaces={"a": ["a1", "a2"], "b": ["b1", "b2"]},
                        prior=prior, actions={"a": ["L", "R"], "b": ["L", "R"]},
                        payoffs=payoffs)
    r = gs.solve_bayesian_nash(game)
    assert r.status == "exact_verified"
    assert any(w["a"]["a1"] == "L" and w["a"]["a2"] == "R"
               and w["b"]["b1"] == "L" and w["b"]["b2"] == "R" for w in r.witnesses)


def test_spne_two_stage_perfect_information():
    players = [PlayerSpec("p1", "P1"), PlayerSpec("p2", "P2")]
    nodes = [
        ExtensiveNode(id="root", actor="p1",
                      edges=[{"action_id": "L", "next_node": "nL"},
                             {"action_id": "R", "next_node": "nR"}]),
        ExtensiveNode(id="nL", actor="p2",
                      edges=[{"action_id": "l", "next_node": "tLL"},
                             {"action_id": "r", "next_node": "tLR"}]),
        ExtensiveNode(id="nR", actor="p2",
                      edges=[{"action_id": "l", "next_node": "tRL"},
                             {"action_id": "r", "next_node": "tRR"}]),
        ExtensiveNode(id="tLL", actor=None, terminal_utilities={"p1": 2, "p2": 1}),
        ExtensiveNode(id="tLR", actor=None, terminal_utilities={"p1": 0, "p2": 0}),
        ExtensiveNode(id="tRL", actor=None, terminal_utilities={"p1": 0, "p2": 0}),
        ExtensiveNode(id="tRR", actor=None, terminal_utilities={"p1": 1, "p2": 2}),
    ]
    isets = [
        {"id": "is_p1", "player_id": "p1", "node_ids": ["root"], "action_ids": ["L", "R"]},
        {"id": "is_p2L", "player_id": "p2", "node_ids": ["nL"], "action_ids": ["l", "r"]},
        {"id": "is_p2R", "player_id": "p2", "node_ids": ["nR"], "action_ids": ["l", "r"]},
    ]
    game = ExtensiveFormGame(players=players, root="root", nodes=nodes,
                             information_sets=isets)
    r = gs.spne_walk(game)
    assert r.status == "exact_verified"
    w = r.witnesses[0]
    assert w["continuation"] == {"root": "L", "nL": "l"}
    assert w["value"] == {"p1": "2", "p2": "1"}
