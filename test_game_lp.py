"""Exact LP fixtures: known optima, dual equality, bound transforms, statuses."""
from fractions import Fraction as F

import pytest

from game_lp import (LPError, LPResult, solve_rational_lp, verify_lp_optimality)


def test_bounded_lp_exact_optimum():
    r = solve_rational_lp([1, 1], A_ub=[[1, 2], [3, 1]], b_ub=[4, 6],
                          bounds=[(0, 2), (0, 2)])
    assert r.status == "optimal"
    assert r.objective == F(14, 5)
    assert r.x[0] == F(8, 5) and r.x[1] == F(6, 5)
    assert verify_lp_optimality(r)
    assert r.dual_objective == F(14, 5)


def test_free_variable_and_equality():
    r = solve_rational_lp([-1, 0], A_eq=[[1, -1]], b_eq=[3],
                          bounds=[(None, None), (0, None)])
    assert r.status == "optimal" and r.objective == F(-3)
    assert r.x[0] == F(3) and r.x[1] == 0
    assert verify_lp_optimality(r)


def test_infeasible_status():
    r = solve_rational_lp([1], A_ub=[[1], [-1]], b_ub=[1, -2], bounds=[(0, None)])
    assert r.status == "infeasible"


def test_unbounded_status():
    r = solve_rational_lp([1], bounds=[(0, None)])
    assert r.status == "unbounded"


def test_negative_interval_bound_shift():
    r = solve_rational_lp([1], bounds=[(-3, -1)])
    assert r.status == "optimal" and r.objective == F(-1)
    assert r.x[0] == F(-1)


def test_reflected_upper_bound_only():
    r = solve_rational_lp([1], bounds=[(None, 5)])
    assert r.status == "optimal" and r.objective == 5 and r.x[0] == 5
    assert verify_lp_optimality(r)


def test_negative_rhs_rows():
    r = solve_rational_lp([1], A_ub=[[-1], [1]], b_ub=[-2, 5], bounds=[(0, None)])
    assert r.status == "optimal" and r.objective == 5
    assert verify_lp_optimality(r)


def test_degenerate_multiple_optima():
    r = solve_rational_lp([1, 1], A_ub=[[1, 1]], b_ub=[1],
                          bounds=[(0, None), (0, None)])
    assert r.status == "optimal" and r.objective == 1
    assert verify_lp_optimality(r)


def test_dual_witness_matches_hand_computation():
    r = solve_rational_lp([1, 1], A_ub=[[1, 2], [3, 1]], b_ub=[4, 6],
                          bounds=[(0, 2), (0, 2)])
    assert r.y == [F(2, 5), F(1, 5), F(0), F(0)]


def test_over_limit_rejected():
    r = solve_rational_lp([1] * 2001, bounds=[(0, None)] * 2001)
    assert r.status == "over_limit"


def test_bland_simplex_regression_negative_rows_with_artificials():
    """Regression pin: phase 2 must read the rhs from the true rhs column when
    artificials exist (it once read an artificial column as the rhs, corrupting
    the ratio test on any LP with negative-rhs rows)."""
    import game_lp

    # min x s.t. -x <= -1 (surplus row + artificial), x <= 2  ->  x = 1
    status, x, obj = game_lp._bland_simplex([[-1], [1]], [-1, 2], [1])
    assert status == "optimal"
    assert x == [F(1)] and obj == F(1)

    # two artificial rows + free split through solve_rational_lp: max x+y
    # s.t. -x-y <= -3, x <= 2, y <= 2 -> optimum 4 at (2,2)
    r = solve_rational_lp([1, 1], A_ub=[[-1, -1], [1, 0], [0, 1]], b_ub=[-3, 2, 2],
                          bounds=[(0, None), (0, None)])
    assert r.status == "optimal" and r.objective == F(4)
    assert r.x == [F(2), F(2)]
    assert verify_lp_optimality(r)
    # the returned dual witness is feasible and nonnegative
    assert all(y >= 0 for y in r.y)


def test_bounds_length_mismatch_raises():
    with pytest.raises(LPError):
        solve_rational_lp([1, 1], bounds=[(0, None)])
