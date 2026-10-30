"""Exact linear programming over rationals.

Canonical form:  max c^T x  s.t.  A x <= b, x >= 0.  Bounded variables are
shifted/reflected, free variables split, equalities become paired
inequalities. Solving uses a self-contained exact tableau simplex with
Bland's rule (guaranteed termination on degenerate systems — sympy's
_simplex is not: it raised "Oscillating system" on CE/Stackelberg LPs).
The dual min b^T y s.t. A^T y >= c, y >= 0 is solved independently and both
objectives must match exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

MAX_LP_VARIABLES = 2_000
MAX_LP_CONSTRAINTS = 10_000


class LPError(Exception):
    """Structured LP failure."""


@dataclass
class LPResult:
    status: str                      # optimal|infeasible|unbounded|over_limit
    objective: Fraction | None = None
    x: list[Fraction] = field(default_factory=list)
    dual_objective: Fraction | None = None
    y: list[Fraction] = field(default_factory=list)
    slack: list[Fraction] = field(default_factory=list)   # b - A x over original rows


def _rat(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise LPError("booleans are not LP coefficients")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    raise LPError(f"LP input must be rational, got {type(value).__name__}")


def _bland_simplex(A, b, c):
    """Solve min c^T x s.t. A x <= b, x >= 0 via two-phase simplex, Bland's rule.

    Rows with b_i >= 0 get slack +1; rows with b_i < 0 are flipped into
    surplus form (-a)x - s = -b with a +1 artificial. Phase 1 minimizes the
    artificial sum; phase 2 optimizes c over structural + slack columns.
    Returns (status, x, objective), status in optimal|infeasible|unbounded.
    """
    m = len(A)
    n = len(A[0]) if m else 0

    rows = []
    rhs = []
    signs = []            # +1 slack basic; -1 surplus with artificial
    for i in range(m):
        if b[i] >= 0:
            rows.append(A[i][:])
            rhs.append(b[i])
            signs.append(1)
        else:
            rows.append([-v for v in A[i]])
            rhs.append(-b[i])
            signs.append(-1)

    n_slack = m
    total = n + n_slack
    # columns: n structural, then slack/surplus per row, then artificials
    n_art = sum(1 for s in signs if s < 0)
    T = []
    basis = []
    art_index = 0
    art_cols: list[int] = []
    for i in range(m):
        row = rows[i] + [Fraction(0)] * n_slack
        row[n + i] = Fraction(signs[i])
        trow = row + [Fraction(0)] * n_art + [rhs[i]]
        if signs[i] < 0:
            col = n + n_slack + art_index
            art_index += 1
            art_cols.append(col)
            trow[col] = Fraction(1)
            basis.append(col)
        else:
            basis.append(n + i)
        T.append(trow)
    total1 = n + n_slack + n_art
    cost1 = [Fraction(0)] * (n + n_slack) + [Fraction(1)] * n_art + [Fraction(0)]

    def iterate(T, basis, cost, total_cols, rhs_idx):
        mm = len(T)
        while True:
            cb = [cost[basis[i]] for i in range(mm)]
            entering = -1
            for j in range(total_cols):
                if j in basis:
                    continue
                rc = cost[j] - sum(cb[i] * T[i][j] for i in range(mm))
                if rc < 0:
                    entering = j
                    break
            if entering < 0:
                return "optimal"
            leaving, best = -1, None
            for i in range(mm):
                if T[i][entering] > 0:
                    ratio = T[i][rhs_idx] / T[i][entering]
                    if best is None or ratio < best or (ratio == best and basis[i] < basis[leaving]):
                        best, leaving = ratio, i
            if leaving < 0:
                return "unbounded"
            piv = T[leaving][entering]
            T[leaving] = [v / piv for v in T[leaving]]
            for r in range(mm):
                if r != leaving and T[r][entering] != 0:
                    factor = T[r][entering]
                    T[r] = [rv - factor * pv for rv, pv in zip(T[r], T[leaving])]
            basis[leaving] = entering

    iterate(T, basis, cost1, total1, total1)

    art_set = set(art_cols)
    for i in range(m):
        if basis[i] in art_set and T[i][total1] != 0:
            return "infeasible", None, None

    # drive zero-valued artificials out of the basis where possible; pivot only
    # on entries that keep the basic value nonnegative
    for i in range(m):
        if basis[i] in art_set:
            for j in range(n + n_slack):
                if j in basis:
                    continue
                v = T[i][j]
                if v > 0 or (v < 0 and T[i][total1] == 0):
                    piv = v
                    T[i] = [x / piv for x in T[i]]
                    for r in range(m):
                        if r != i and T[r][j] != 0:
                            factor = T[r][j]
                            T[r] = [rv - factor * pv for rv, pv in zip(T[r], T[i])]
                    basis[i] = j
                    break

    cost2 = list(c) + [Fraction(0)] * n_slack
    status2 = iterate(T, basis, cost2, total, total1)
    if status2 == "unbounded":
        return "unbounded", None, None
    x = [Fraction(0)] * n
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = T[i][total1]
    obj = sum((c[j] * x[j] for j in range(n)), Fraction(0))
    return "optimal", x, obj


def _dual_simplex(A, b, c):
    """Solve min b^T y s.t. A^T y >= c, y >= 0 exactly (canonical m rows)."""
    m = len(A)
    n_cols = len(A[0]) if m else 0
    # A^T y >= c  ->  -A^T y <= -c
    neg_AT = [[-A[i][j] for i in range(m)] for j in range(n_cols)]
    neg_c = [-v for v in c]
    return _bland_simplex(neg_AT, neg_c, b)


def solve_rational_lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                      bounds=None) -> LPResult:
    """Solve max c^T x exactly; verify via an independently solved exact dual."""
    c = [_rat(v) for v in (c or [])]
    n = len(c)
    if n == 0:
        raise LPError("LP needs at least one variable")
    bounds = list(bounds) if bounds is not None else [(0, None)] * n
    if len(bounds) != n:
        raise LPError("bounds length mismatch")
    for lo, hi in bounds:
        if lo is not None and hi is not None and lo > hi:
            return LPResult(status="infeasible")

    A_ub_rows = [[_rat(v) for v in row] for row in (A_ub or [])]
    b_ub_vec = [_rat(v) for v in (b_ub or [])]
    A_eq_rows = [[_rat(v) for v in row] for row in (A_eq or [])]
    b_eq_vec = [_rat(v) for v in (b_eq or [])]
    if len(A_ub_rows) != len(b_ub_vec) or len(A_eq_rows) != len(b_eq_vec):
        raise LPError("constraint/vector length mismatch")

    # ---- variable transforms
    columns: list[tuple[str, int, Fraction]] = []
    for i, (lo, hi) in enumerate(bounds):
        if lo is not None and hi is not None:
            columns.append(("shift", i, Fraction(lo)))
        elif lo is not None:
            columns.append(("shift", i, Fraction(lo)))
        elif hi is not None:
            columns.append(("reflect", i, Fraction(hi)))
        else:
            columns.append(("free_pos", i, Fraction(0)))
            columns.append(("free_neg", i, Fraction(0)))
    n_can = len(columns)

    def col_of(kind: str, i: int) -> int:
        for idx, (k, j, _s) in enumerate(columns):
            if k == kind and j == i:
                return idx
        raise LPError("internal column mapping error")

    def transform_row(row: list[Fraction]) -> tuple[list[Fraction], Fraction]:
        coef = [Fraction(0)] * n_can
        c0 = Fraction(0)
        for i, a in enumerate(row):
            if a == 0:
                continue
            for idx, (kind, j, shift) in enumerate(columns):
                if j != i:
                    continue
                if kind == "shift":
                    coef[idx] += a
                    c0 += a * shift
                elif kind == "reflect":
                    coef[idx] -= a
                    c0 += a * shift
                elif kind == "free_pos":
                    coef[idx] += a
                else:
                    coef[idx] -= a
        return coef, c0

    constraints: list[tuple[list[Fraction], Fraction, Fraction]] = []
    for row, rhs in zip(A_ub_rows, b_ub_vec):
        coef, c0 = transform_row(row)
        constraints.append((coef, c0, rhs))
    for row, rhs in zip(A_eq_rows, b_eq_vec):
        coef, c0 = transform_row(row)
        constraints.append((coef, c0, rhs))
        constraints.append(([-v for v in coef], -c0, -rhs))

    canonical_bounds: list[tuple[Fraction | None, Fraction | None]] = [(None, None)] * n_can
    for i, (lo, hi) in enumerate(bounds):
        if lo is not None and hi is not None:
            canonical_bounds[col_of("shift", i)] = (Fraction(0), Fraction(hi) - Fraction(lo))
        elif lo is not None:
            canonical_bounds[col_of("shift", i)] = (Fraction(0), None)
        elif hi is not None:
            canonical_bounds[col_of("reflect", i)] = (Fraction(0), None)

    for idx, (_lo, hi) in enumerate(canonical_bounds):
        if hi is not None:
            row = [Fraction(0)] * n_can
            row[idx] = Fraction(1)
            constraints.append((row, Fraction(0), hi))

    if n_can > MAX_LP_VARIABLES or len(constraints) > MAX_LP_CONSTRAINTS:
        return LPResult(status="over_limit")

    c_can, objective_offset = transform_c_local(c, transform_row)

    if not constraints:
        if any(v > 0 for v in c_can):
            return LPResult(status="unbounded")
        x_can = [Fraction(0)] * n_can
        x_orig = _map_back(x_can, columns, n)
        return LPResult(status="optimal", objective=objective_offset, x=x_orig,
                        dual_objective=objective_offset, y=[])

    A = [row for row, _c0, _rhs in constraints]
    b = [rhs - c0 for _row, c0, rhs in constraints]
    # shift b positive where negative is handled inside simplex prep

    status, x_can, min_val = _bland_simplex([row[:] for row in A], b[:],
                                            [-v for v in c_can])
    if status == "infeasible":
        return LPResult(status="infeasible")
    if status == "unbounded":
        return LPResult(status="unbounded")

    # ---- verification of the primal witness
    if any(v < 0 for v in x_can):
        raise LPError("primal witness violates nonnegativity")
    for row, rhs_val in zip(A, b):
        lhs = sum((av * xv for av, xv in zip(row, x_can)), Fraction(0))
        if lhs > rhs_val:
            raise LPError("primal witness violates a canonical constraint")

    primal_obj = sum((cv * xv for cv, xv in zip(c_can, x_can)), Fraction(0)) + objective_offset

    # ---- dual: min b^T y s.t. A^T y >= c, y >= 0
    d_status, y_can, dual_min = _dual_simplex(A, b, c_can)
    if d_status != "optimal":
        raise LPError(f"dual solve returned {d_status}")
    dual_obj = dual_min
    for i in range(len(A)):
        if y_can[i] < 0:
            raise LPError("dual witness violates nonnegativity")
    for j in range(n_can):
        lhs = sum((y_can[i] * A[i][j] for i in range(len(A))), Fraction(0))
        if lhs < c_can[j]:
            raise LPError("dual witness violates a dual constraint")
    if dual_obj != primal_obj - objective_offset:
        raise LPError("primal and dual objectives disagree")

    x_orig = _map_back(x_can, columns, n)
    slack = [rhs - sum((row[i] * x_orig[i] for i in range(len(row))), Fraction(0))
             for row, rhs in zip(A_ub_rows, b_ub_vec)]
    for row, rhs in zip(A_eq_rows, b_eq_vec):
        slack.append(rhs - sum((row[i] * x_orig[i] for i in range(len(row))), Fraction(0)))

    return LPResult(status="optimal", objective=primal_obj, x=x_orig,
                    dual_objective=dual_obj + objective_offset, y=y_can, slack=slack)


def transform_c_local(c, transform_row):
    coef, c0 = transform_row(c)
    return coef, c0


def _map_back(x_can, columns, n):
    x_orig = [Fraction(0)] * n
    for idx, (kind, i, shift) in enumerate(columns):
        z = x_can[idx]
        if kind == "shift":
            x_orig[i] += z + shift
        elif kind == "reflect":
            x_orig[i] += shift - z
        elif kind == "free_pos":
            x_orig[i] += z
        else:
            x_orig[i] -= z
    return x_orig


def verify_lp_optimality(result: LPResult) -> bool:
    return result.status == "optimal" and result.dual_objective == result.objective
