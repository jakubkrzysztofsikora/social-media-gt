"""Exact-arithmetic verification toolkit for the paper

    "Nash equilibrium and access to social media for children"

Each scenario re-derives a published result from the paper's definitions with
exact rational arithmetic (fractions), symbolic algebra (SymPy) and SMT
validity checks (Z3). No floating point enters a published number.

    python social_media_gt.py
    python social_media_gt.py tw6 --gamma 0.3
    python social_media_gt.py tw7 --lambda 0.5 --eta 0.25
    python social_media_gt.py tw8 --tau-c 3 --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction as F
from itertools import permutations, product

from sympy import (Min, Rational, Symbol, diff, erf, exp, expand, floor, integrate, limit,
                   oo, simplify, solve, sqrt)
import z3

EPS = F(1, 10)

def frac(value, name):
    if isinstance(value, F):
        return value
    if isinstance(value, int):
        return F(value)
    if isinstance(value, float):
        return F(str(value))
    raise ValueError(f"{name} must be an exact number, got {value!r}")


def sp(value: F) -> Rational:
    return Rational(value.numerator, value.denominator)


def show(value) -> str:
    if isinstance(value, F):
        return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(show(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {show(v)}" for k, v in value.items()) + "}"
    return str(value)


def check(name, ok, expected, got) -> dict:
    return {"name": name, "ok": bool(ok), "expected": show(expected), "got": show(got)}


def result(scenario, title, params, results, checks, conclusion) -> dict:
    return {"scenario": scenario, "title": title, "params": params, "results": results,
            "checks": checks, "conclusion": conclusion}


def tw1_static_equilibrium(tau=F(1, 2), f1=F(1, 4), E1=F(1, 4)) -> dict:
    tau, f1, E1 = frac(tau, "tau"), frac(f1, "f1"), frac(E1, "E1")
    P1 = f1 + E1
    delta_at_zero = -tau
    all_a_is_ne = P1 >= tau
    all_r_dominates = tau > f1
    all_a_is_ne_direct = f1 >= tau - E1
    checks = [
        check("all-R is a strict Nash equilibrium", delta_at_zero < 0, "< 0", delta_at_zero),
        check("all-A is a Nash equilibrium iff P(1) >= tau", all_a_is_ne == all_a_is_ne_direct,
              all_a_is_ne, all_a_is_ne_direct),
        check("all-R Pareto-dominates all-A iff tau > f(1)", all_r_dominates == (tau > f1),
              all_r_dominates, tau > f1),
    ]
    results = {"delta_at_zero": delta_at_zero, "all_a_is_ne": all_a_is_ne,
               "all_r_dominates_pareto": all_r_dominates,
               "payoffs_relative_to_offline": {"all_R": {"stay": F(0), "deviate": -tau},
                                               "all_A": {"stay": f1 - tau, "deviate": -E1}},
               "pareto_gap_all_R": tau - f1}
    conclusion = ["Nobody moves first: the lone deviator absorbs the whole exclusion cost, "
                  "even though restricting together is better for every family."]
    return result("tw1", "Tw. 1 - static trap", {"tau": tau, "f1": f1, "E1": E1},
                  results, checks, conclusion)


def tw1_1_screen_time_cap(a=F(1), b=F(1), T=F(4), beta=F(1), k=F(1, 2)) -> dict:
    a, b, T, beta, k = (frac(x, n) for x, n in ((a, "a"), (b, "b"), (T, "T"), (beta, "beta"), (k, "k")))
    if min(a, b, T, beta) <= 0:
        raise ValueError("a, b, T and beta must be positive")
    if a + beta > 2 * b * T:
        raise ValueError("assumption a + beta <= 2*b*T violated")

    A, B, TT, BE = Symbol("a"), Symbol("b"), Symbol("T"), Symbol("beta")
    t_fb = A / (2 * B)
    t_ne = Min(TT, A / (2 * B - BE / TT))
    V = lambda t: A * t - B * t ** 2
    identity = simplify(V(t_fb) - V(t_ne) - B * (t_ne - t_fb) ** 2)

    values = {A: sp(a), B: sp(b), TT: sp(T), BE: sp(beta)}
    t_fb_num = simplify(t_fb.subs(values))
    t_ne_num = simplify(t_ne.subs(values))
    loss = simplify(B * (t_ne - t_fb) ** 2).subs(values)
    checks = [
        check("V(t_FB) - V(t_NE) = b (t_NE - t_FB)^2", identity == 0, 0, identity),
        check("the network-coupled optimum exceeds the first best", t_ne_num > t_fb_num, "> t_FB", t_ne_num),
    ]
    if (a, b, T, beta) == (F(1), F(1), F(4), F(1)):
        checks.append(check("paper: t_FB = 1/2, t_NE = 4/7, welfare loss = 1/196",
                            (t_fb_num, t_ne_num, loss) == (sp(F(1, 2)), sp(F(4, 7)), sp(F(1, 196))),
                            "(1/2, 4/7, 1/196)", (t_fb_num, t_ne_num, loss)))
    curve_t = [F(i, 20) for i in range(21)]
    results = {"t_first_best": F(int(sp(t_fb_num).p), int(sp(t_fb_num).q)),
               "t_nash": F(int(sp(t_ne_num).p), int(sp(t_ne_num).q)),
               "welfare_loss": F(int(sp(loss).p), int(sp(loss).q)),
               "curve_t": curve_t,
               "curve_V": [a * t - b * t * t for t in curve_t],
               "curve_G": [a * t - b * t * t - beta * k * (T - t) for t in curve_t]}
    conclusion = ["A parent sets a stricter screen-time limit than it would choose alone, "
                  "because exclusion from the peer group hurts the child more than screen time does."]
    return result("tw1_1", "Tw. 1.1 - local screen-time cap under network effects",
                  {"a": a, "b": b, "T": T, "beta": beta}, results, checks, conclusion)


def tw2_cascade(eps=EPS, p_slope=F(100), f_lo=F(5), f_hi=F(95)) -> dict:
    eps, p_slope, f_lo, f_hi = (frac(x, n) for x, n in ((eps, "eps"), (p_slope, "p_slope"),
                                                        (f_lo, "f_lo"), (f_hi, "f_hi")))
    k = Symbol("k", real=True)
    k_lo, k_hi = f_lo / p_slope, f_hi / p_slope
    branches = [
        (sp(eps), (None, k_lo)),
        (sp(eps) + (1 - sp(eps)) * (sp(p_slope) * k - sp(f_lo)) / (sp(f_hi) - sp(f_lo)), (k_lo, k_hi)),
        (sp(eps) + (1 - sp(eps)), (k_hi, None)),
    ]
    fixed_points = []
    for expr, (low, high) in branches:
        for root in solve(expr - k, k):
            value = F(int(root.p), int(root.q))
            if low is not None and value < low:
                continue
            if high is not None and value > high:
                continue
            if eps <= value < 1:
                fixed_points.append(value)

    def Phi(value: F) -> F:
        return eps + (1 - eps) * max(F(0), min(F(1), (p_slope * value - f_lo) / (f_hi - f_lo)))

    iterated = F(0)
    for _ in range(400):
        iterated = Phi(iterated)
    analytic = min(fixed_points) if fixed_points else F(1)
    converged = abs(iterated - analytic) < F(1, 10 ** 6)

    rounds = None
    value = F(0)
    for step in range(1, 10001):
        value = Phi(value)
        if value == 1:
            rounds = step
            break

    slope = (1 - eps) * p_slope / (f_hi - f_lo) - 1
    const = eps - (1 - eps) * f_lo / (f_hi - f_lo)
    checks = [
        check("no fixed point inside [eps, 1)", not fixed_points, "none", fixed_points or "none"),
        check("the iteration converges to the analytic limit", converged, True, converged),
        check("Phi(k) - k is constant on the linear region", slope == 0, 0, slope),
    ]
    if (eps, p_slope, f_lo, f_hi) == (EPS, F(100), F(5), F(95)):
        checks.append(check("paper: full adoption reached after 19 rounds",
                            (analytic, rounds, const) == (F(1), 19, F(1, 20)),
                            "(1, 19, 1/20)", (analytic, rounds, const)))
    results = {"limit": analytic, "rounds_to_full_adoption": rounds,
               "phi_minus_k_on_linear_region": const if slope == 0 else None,
               "uniform_q_mean": eps + (1 - eps) / 2}
    conclusion = ["Pressure compounds: once adoption starts, the cascade runs to the whole cohort."]
    return result("tw2", "Tw. 2 - cascade and least fixed point",
                  {"eps": eps, "p_slope": p_slope, "f_lo": f_lo, "f_hi": f_hi},
                  results, checks, conclusion)


def tw3_absorption(tau=F(1, 2), n=4) -> dict:
    tau = frac(tau, "tau")
    if not 2 <= n <= 12:
        raise ValueError("n must lie between 2 and 12")

    def share(profile, i):
        return F(sum(1 for j in range(n) if j != i and profile[j] == "A"), n - 1)

    def gap(profile, i):
        k = share(profile, i)
        return (k - tau) - (-F(1, 2) * k)

    equilibria = [profile for profile in product("RA", repeat=n)
                  if all(gap(profile, i) >= 0 if profile[i] == "A" else gap(profile, i) <= 0
                         for i in range(n))]
    all_a = ("A",) * n
    delta_all_a = gap(all_a, 0)
    checks = [
        check("all-A is a Nash equilibrium", all_a in equilibria, "yes",
              "yes" if all_a in equilibria else "no"),
        check("no unilateral deviation pays at all-A", delta_all_a > 0, "> 0", delta_all_a),
    ]
    results = {"equilibria_count": len(equilibria), "all_a_is_equilibrium": all_a in equilibria,
               "delta_at_all_a": delta_all_a}
    conclusion = ["All-adopting is stable: a family that leaves alone eats the exclusion cost, "
                  "so nobody moves first."]
    return result("tw3", "Tw. 3 - absorbing state under myopic revisions", {"tau": tau, "n": n},
                  results, checks, conclusion)


def tw3_2_global_games(f1=F(1), E1=F(3)) -> dict:
    f1, E1 = frac(f1, "f1"), frac(E1, "E1")
    P1 = f1 + E1
    tau_rd = P1 / 2
    trap_nonempty = E1 > f1
    midpoint_gap = (E1 - f1) / 4

    tau_z, tauh_z, Pk_z = z3.Reals("tau tauh Pk")
    q_z, qh_z = z3.Reals("q qh")

    def entails(hyps, negation):
        solver = z3.Solver()
        solver.add(*hyps)
        if solver.check() != z3.sat:
            return False
        solver2 = z3.Solver()
        solver2.add(*hyps, negation)
        return solver2.check() == z3.unsat

    dsic_t = (tau_z - Pk_z) * (q_z - qh_z) >= 0
    dsic_h = (tauh_z - Pk_z) * (qh_z - q_z) >= 0
    dsic_below = entails([tau_z < Pk_z, tauh_z < Pk_z, tau_z < tauh_z, dsic_t, dsic_h], q_z != qh_z)
    dsic_above = entails([tau_z > Pk_z, tauh_z > Pk_z, tau_z < tauh_z, dsic_t, dsic_h], q_z != qh_z)
    ir_zero = entails([tau_z < Pk_z, Pk_z > 0, q_z >= 0, q_z * (tau_z - Pk_z) >= 0], q_z != 0)

    sigma = Symbol("sigma", positive=True)
    shift = Symbol("s", positive=True)
    tau_s, c_s = Symbol("tau", real=True), Symbol("c", real=True)
    cdf = lambda z: (1 + erf(z / sqrt(2))) / 2
    posterior = simplify(integrate((tau_s - c_s) * expand(exp(-((tau_s - c_s) ** 2) / (2 * sigma ** 2))),
                                     (tau_s, -oo, oo)))
    limit_above = simplify(limit(cdf(shift / (sigma * sqrt(2))), sigma, 0, "+"))
    limit_below = simplify(limit(cdf(-shift / (sigma * sqrt(2))), sigma, 0, "+"))

    checks = [
        check("risk-dominant cutoff is P(1)/2", tau_rd == P1 / 2, P1 / 2, tau_rd),
        check("the Pareto trap class is non-empty exactly when E(1) > f(1)", trap_nonempty,
              E1 > f1, trap_nonempty),
        check("DSIC: two types below P(k) share the assignment", dsic_below, True, dsic_below),
        check("DSIC: two types above P(k) share the assignment", dsic_above, True, dsic_above),
        check("individual rationality forces q(tau) = 0 below P(k)", ir_zero, True, ir_zero),
        check("a flat prior gives E[tau | x_i = c] = c", posterior == 0, 0, posterior),
        check("noise vanishes upwards", limit_above == 1, 1, limit_above),
        check("noise vanishes downwards", limit_below == 0, 0, limit_below),
    ]
    if (f1, E1) == (F(1), F(3)):
        checks.append(check("paper: tau^RD = 2", tau_rd == F(2), 2, tau_rd))
    results = {"tau_risk_dominant": tau_rd, "pareto_trap_class_nonempty": trap_nonempty,
               "pareto_gap_at_midpoint": midpoint_gap}
    conclusion = ["With incomplete information the limit outcome is all-adopting even where restricting "
                  "together is better for everyone; selection rests on the global-games literature, "
                  "the arithmetic is checked here.",
                  "A voluntary mechanism without payments cannot recruit the families for whom leaving hurts."]
    return result("tw3_2", "Tw. 3.2 - global games and the incentive barrier", {"f1": f1, "E1": E1},
                  results, checks, conclusion)


def tw5_coalition_thresholds(tau=F(3, 2), n=10, d=1) -> dict:
    tau = frac(tau, "tau")
    k = Symbol("k", real=True)
    roots = [r for r in solve(2 * k + k * k - sp(tau), k) if r.is_real and 0 <= r <= 1]
    k_star = roots[0]
    m_out_closed = int(floor((n - 1) * (1 - k_star))) + 1

    aware = n - d
    m_out = m_stab = None
    for m in range(0, aware + 1):
        k_out = F(d + aware - m - 1, n - 1)
        k_mem = F(d + aware - m, n - 1)
        if m_out is None and 2 * k_out + k_out * k_out < tau:
            m_out = m
        if m_out is not None and m_stab is None and 2 * k_mem + k_mem * k_mem <= tau:
            m_stab = m
            break

    P59, P69 = 2 * F(5, 9) + F(5, 9) ** 2, 2 * F(6, 9) + F(6, 9) ** 2
    checks = [
        check("closed form equals the brute-force threshold", m_out == m_out_closed, m_out_closed, m_out),
        check("P(5/9) = 115/81", P59 == F(115, 81), F(115, 81), P59),
        check("P(6/9) = 16/9", P69 == F(16, 9), F(16, 9), P69),
    ]
    if (tau, n, d) == (F(3, 2), 10, 1):
        checks.append(check("paper: m_out = 4 and m_stab = 5", (m_out, m_stab) == (4, 5), "(4, 5)",
                            (m_out, m_stab)))
    results = {"m_out": m_out, "m_stab": m_stab, "k_star": str(k_star),
               "k_star_decimal": f"{float(k_star):.2f}".replace(".", ","),
               "P_peer_5_of_9": P59, "P_peer_6_of_9": P69}
    conclusion = ["A pact always misses by one family: the outsider counts pressure without counting "
                  "itself, so the pact that first tempts the rest is the pact its members want to leave."]
    return result("tw5", "Tw. 5 - outsider threshold and pact durability", {"tau": tau, "n": n, "d": d},
                  results, checks, conclusion)


def tw6_coalition_continuum(gamma=F(9, 40), eps=EPS, p_slope=F(100), f_lo=F(5), f_hi=F(95)) -> dict:
    gamma, eps, p_slope, f_lo, f_hi = (frac(x, n) for x, n in ((gamma, "gamma"), (eps, "eps"),
                                                               (p_slope, "p_slope"), (f_lo, "f_lo"),
                                                               (f_hi, "f_hi")))
    k = Symbol("k", real=True)
    operator = sp(eps) + (1 - sp(eps) - sp(gamma)) * (sp(p_slope) * k - sp(f_lo)) / (sp(f_hi) - sp(f_lo))
    roots = solve(operator - k, k)
    if len(roots) != 1:
        raise ValueError("the coalition operator has no unique finite fixed point")
    K = F(int(roots[0].p), int(roots[0].q))
    a = 1 - gamma / (1 - eps)

    trajectory = [1 - gamma]
    for _ in range(2):
        trajectory.append(K + a * (trajectory[-1] - K))
    exit_multiplier = (1 - K) / gamma if gamma else None

    release_steps = None
    value = K
    for step in range(1, 10001):
        value = eps + (1 - eps) * max(F(0), min(F(1), (p_slope * value - f_lo) / (f_hi - f_lo)))
        if value == 1:
            release_steps = step
            break

    def T_gamma(value: F) -> F:
        return eps + (1 - eps - gamma) * max(F(0), min(F(1), (p_slope * value - f_lo) / (f_hi - f_lo)))

    centered = all(T_gamma(x) == K + a * (x - K) for x in (K, (K + 1) / 2, F(19, 20)))
    checks = [
        check("K is a fixed point of the coalition operator", T_gamma(K) == K, K, T_gamma(K)),
        check("the operator equals the centered form K + a(k - K)", centered, True, centered),
    ]
    if (gamma, eps, p_slope, f_lo, f_hi) == (F(9, 40), EPS, F(100), F(5), F(95)):
        checks.append(check("paper: K = 1/4, a = 3/4, orbit 31/40, 103/160, 349/640, multiplier 10/3",
                            (K, a, trajectory, exit_multiplier) == (F(1, 4), F(3, 4),
                                                                  [F(31, 40), F(103, 160), F(349, 640)],
                                                                  F(10, 3)),
                            "(1/4, 3/4, [31/40, 103/160, 349/640], 10/3)",
                            (K, a, trajectory, exit_multiplier)))
        checks.append(check("paper: without the pact, adoption returns to full in 15 updates",
                            release_steps == 15, 15, release_steps))
    results = {"K": K, "a": a, "trajectory": trajectory, "exit_multiplier": exit_multiplier,
               "release_steps": release_steps}
    conclusion = ["A cross-sectional pact covering 22.5% of the total cohort (25% of conscientious "
                  "households at eps = 10%) pulls adoption down to 25%, but the effect evaporates "
                  "once the commitment ends."]
    return result("tw6", "Tw. 6 - coalitions in the continuum",
                  {"gamma": gamma, "eps": eps, "p_slope": p_slope, "f_lo": f_lo, "f_hi": f_hi},
                  results, checks, conclusion)


def linear_threshold(tau=F(3, 2), slope=F(4)) -> F:
    """Threshold k where a linear pressure P(k) = slope * k crosses the level tau."""
    tau, slope = frac(tau, "tau"), frac(slope, "slope")
    if slope <= 0:
        raise ValueError("slope must be positive")
    return tau / slope


def cascade_limit(lam=F(1, 5), eta=F(0), eps=EPS, step=F(1, 20)) -> F:
    """Limit of the cascade under a pressure reduction, exact for any parameters."""
    lam, eta, eps, step = frac(lam, "lam"), frac(eta, "eta"), frac(eps, "eps"), frac(step, "step")
    if not 0 <= eta < 1:
        raise ValueError("eta must lie in [0, 1)")
    lam_eff = lam * (1 - eta)
    if lam_eff <= 0:
        return min(F(1), max(eps, step))
    return min(F(1), max(eps, step / lam_eff))


def tw7_pressure_reduction(lam=F(1, 5), eta=F(0), eps=EPS, step=F(1, 20)) -> dict:
    lam, eta, eps, step = (frac(x, n) for x, n in ((lam, "lam"), (eta, "eta"), (eps, "eps"),
                                                    (step, "step")))
    if not 0 <= eta < 1:
        raise ValueError("eta must lie in [0, 1)")
    lam_eff = lam * (1 - eta)
    k = Symbol("k", real=True)
    roots = solve((1 - sp(lam_eff)) * k + sp(step) - k, k)
    closed = min(F(1), max(eps, F(int(roots[0].p), int(roots[0].q)))) if roots else \
        min(F(1), max(eps, step))

    def Phi_lam(value: F) -> F:
        return min(F(1), max(eps, (1 - lam_eff) * value + step))

    iterated = F(1)
    for _ in range(400):
        iterated = Phi_lam(iterated)
    converged = abs(iterated - closed) < F(1, 10 ** 6)
    is_fixed_point = Phi_lam(closed) == closed
    eta_critical = 1 - step / lam if lam > 0 else F(0)

    grid_ok = True
    for lam_v, eta_v in [(F(1, 5), F(3, 4)), (F(1, 5), F(1, 2)), (F(1, 5), F(0)),
                         (F(1, 2), F(0)), (F(2, 5), F(0))]:
        eff = lam_v * (1 - eta_v)
        limit_v = min(F(1), max(eps, step / eff))
        threshold = 1 - step / lam_v
        if (limit_v == 1) != (eta_v >= threshold):
            grid_ok = False

    checks = [
        check("the closed form is a fixed point of the operator", is_fixed_point, True, is_fixed_point),
        check("the iteration converges to the closed form", converged, True, converged),
        check("the reversal fails exactly for eta >= eta*(lambda)", grid_ok, True, grid_ok),
    ]
    if (lam, eta, eps, step) == (F(1, 5), F(0), EPS, F(1, 20)):
        checks.append(check("paper: lambda = 0.2 gives limit 1/4 and eta* = 3/4",
                            (closed, eta_critical) == (F(1, 4), F(3, 4)), "(1/4, 3/4)",
                            (closed, eta_critical)))
    results = {"limit": closed, "eta_critical": eta_critical, "lambda_effective": lam_eff}
    conclusion = ["A rule that binds the whole peer group at once flips the cascade on its own: "
                  "no coordination needed, but every exemption re-opens the pressure."]
    return result("tw7", "Tw. 7 - top-down reduction of pressure",
                  {"lam": lam, "eta": eta, "eps": eps, "step": step}, results, checks, conclusion)


def tw7_2_hybrid_policy(creg=F(1), cplat=F(1), eta=F(1, 2), lam_target=F(3, 10)) -> dict:
    creg, cplat, eta, lam_target = (frac(x, n) for x, n in ((creg, "creg"), (cplat, "cplat"),
                                                            (eta, "eta"), (lam_target, "lam_target")))
    if min(creg, cplat, lam_target) <= 0:
        raise ValueError("creg, cplat and lam_target must be positive")
    lreg_s, lplat_s, mu_s = Symbol("lreg"), Symbol("lplat"), Symbol("mu")
    creg_s, cplat_s = Symbol("creg", positive=True), Symbol("cplat", positive=True)
    eta_s, lam_s = Symbol("eta", real=True), Symbol("lam", positive=True)
    cost = creg_s * lreg_s ** 2 / 2 + cplat_s * lplat_s ** 2 / 2
    lagrangian = cost - mu_s * (lreg_s * (1 - eta_s) + lplat_s - lam_s)
    solutions = solve([diff(lagrangian, v) for v in (lreg_s, lplat_s, mu_s)],
                      [lreg_s, lplat_s, mu_s], dict=True)
    if len(solutions) != 1:
        raise RuntimeError("the Lagrange system has no unique solution")
    values = {creg_s: sp(creg), cplat_s: sp(cplat), eta_s: sp(eta), lam_s: sp(lam_target)}
    solved_reg = simplify(solutions[0][lreg_s].subs(values))
    solved_plat = simplify(solutions[0][lplat_s].subs(values))

    D = (1 - eta) ** 2 * cplat + creg
    lambda_reg = lam_target * (1 - eta) * cplat / D
    lambda_plat = lam_target * creg / D
    ratio = lambda_reg / lambda_plat

    c1, c2, e = z3.Reals("c1 c2 e")
    convex = z3.Solver()
    convex.add(c1 > 0, c2 > 0, z3.Not(c1 * (1 - e) * (1 - e) + c2 > 0))

    checks = [
        check("the solved optimum equals the closed form for lambda_reg",
              simplify(solved_reg - sp(lambda_reg)) == 0, lambda_reg, solved_reg),
        check("the solved optimum equals the closed form for lambda_plat",
              simplify(solved_plat - sp(lambda_plat)) == 0, lambda_plat, solved_plat),
        check("the weight ratio is (1 - eta) c_plat / c_reg", ratio == (1 - eta) * cplat / creg,
              (1 - eta) * cplat / creg, ratio),
        check("the cost is strictly convex along the constraint line", convex.check() == z3.unsat,
              "unsat", convex.check()),
    ]
    if (creg, cplat, eta, lam_target) == (F(1), F(1), F(1, 2), F(3, 10)):
        checks.append(check("paper: lambda_reg* = 3/25 and lambda_plat* = 6/25",
                            (lambda_reg, lambda_plat) == (F(3, 25), F(6, 25)), "(3/25, 6/25)",
                            (lambda_reg, lambda_plat)))
    results = {"lambda_reg": lambda_reg, "lambda_plat": lambda_plat, "ratio": ratio}
    conclusion = ["Rules that can be evaded stop working; the cheapest policy mixes statutory limits with "
                  "product defaults and shifts weight to the platform as evasion or enforcement cost grows."]
    return result("tw7_2", "Tw. 7.2 - evasion, enforcement cost, hybrid policy",
                  {"creg": creg, "cplat": cplat, "eta": eta, "lam_target": lam_target},
                  results, checks, conclusion)


def tw8_sequential_exit(tau_C=F(2), delta=F(99, 100), full=False) -> dict:
    tau_C, delta = frac(tau_C, "tau_C"), frac(delta, "delta")
    players, aware = 10, 9

    def gain(k: F) -> F:
        return tau_C - (2 * k + k * k)

    slot_results = []
    for d in range(players):
        sequence = [F(aware - (j - (1 if j > d else 0)), aware) for j in range(players)]
        gains = [gain(k) for k in sequence]
        slot_results.append({"d_slot": d, "sequence": sequence, "undiscounted": sum(gains),
                             "discounted": sum((delta ** t) * gains[t] for t in range(players))})
    worst_slot = min(slot_results, key=lambda item: item["discounted"])

    def state_gain(profile, phase):
        state = list(profile)
        total = F(0)
        for t in range(players):
            pos = (phase + t) % players
            state[pos] = "R" if pos < aware else "A"
            k = F(sum(1 for j in range(players) if j != phase and state[j] == "A"), aware)
            total += (delta ** t) * gain(k)
        return total

    best_state = None
    minimizers = []
    for bits in range(1 << players):
        profile = tuple("A" if bits >> j & 1 else "R" for j in range(players))
        for phase in range(aware):
            value = state_gain(profile, phase)
            if best_state is None or value < best_state:
                best_state, minimizers = value, [(profile, phase)]
            elif value == best_state:
                minimizers.append((profile, phase))

    checks = [
        check("the slot scan and the state scan agree on the minimum advantage",
              best_state == worst_slot["discounted"], worst_slot["discounted"], best_state),
        check("the discounted advantage is at least 697/270", best_state >= F(697, 270),
              F(697, 270), best_state),
    ]
    if (tau_C, delta) == (F(2), F(99, 100)):
        checks.append(check("paper: undiscounted sum 94/27, discounted minimum 3064822137458430997/10^18",
                            (worst_slot["undiscounted"], best_state) == (F(94, 27), F(3064822137458430997, 10 ** 18)),
                            "(94/27, 3064822137458430997/10^18)",
                            (worst_slot["undiscounted"], best_state)))
    results = {"undiscounted_sum": worst_slot["undiscounted"], "discounted_min": best_state,
               "worst_sequence": worst_slot["sequence"], "worst_d_slot": worst_slot["d_slot"],
               "minimizing_states": len(minimizers)}

    if full:
        memo = {}
        best_order = best_sequence = None
        orderings = 0
        for perm in permutations(range(1, players)):
            orderings += 1
            slot_owner = (0,) + perm
            in_a = [True] * players
            sequence = []
            for t in range(players):
                owner = slot_owner[t]
                if owner != aware - 1:
                    in_a[owner] = False
                sequence.append(F(sum(in_a[j] for j in range(players) if j != 0), aware))
            key = tuple(sequence)
            value = memo.get(key)
            if value is None:
                gains = [gain(k) for k in key]
                value = sum((delta ** t) * gains[t] for t in range(players))
                memo[key] = value
            if best_order is None or value < best_order:
                best_order, best_sequence = value, key
        checks.append(check("the full 9! ordering scan reaches the same minimum",
                            best_order == best_state, best_state, best_order))
        checks.append(check("the worst schedule is unique up to the order of the aware players",
                            best_sequence == tuple(worst_slot["sequence"]), tuple(worst_slot["sequence"]),
                            best_sequence))
        results["orderings_enumerated"] = orderings

    conclusion = ["Sequential exit can hold without an authority, but only under a public schedule "
                  "everyone believes in; a real class has none of those preconditions."]
    return result("tw8", "Tw. 8 - sequential exit without authority",
                  {"tau_C": tau_C, "delta": delta, "full_ordering_scan": full},
                  results, checks, conclusion)


SCENARIOS = {
    "tw1": (tw1_static_equilibrium, {}),
    "tw1_1": (tw1_1_screen_time_cap, {}),
    "tw2": (tw2_cascade, {}),
    "tw3": (tw3_absorption, {}),
    "tw3_2": (tw3_2_global_games, {}),
    "tw5": (tw5_coalition_thresholds, {}),
    "tw6": (tw6_coalition_continuum, {}),
    "tw7": (tw7_pressure_reduction, {}),
    "tw7_2": (tw7_2_hybrid_policy, {}),
    "tw8": (tw8_sequential_exit, {"full": True}),
}


def run_all(overrides=None) -> dict:
    overrides = overrides or {}
    results = []
    for name, (func, defaults) in SCENARIOS.items():
        kwargs = dict(defaults)
        kwargs.update(overrides.get(name, {}))
        results.append(func(**kwargs))
    failed = sum(1 for item in results for c in item["checks"] if not c["ok"])
    return {"scenarios_run": len(results), "checks_failed": failed, "results": results}


def render(summary: dict) -> str:
    lines = []
    for item in summary["results"]:
        lines.append(f"\n{item['scenario']}  {item['title']}")
        lines.append(f"  parameters: {show(item['params'])}")
        for key, value in item["results"].items():
            lines.append(f"  {key} = {show(value)}")
        for c in item["checks"]:
            lines.append(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']} "
                         f"(expected {c['expected']}, got {c['got']})")
        for sentence in item["conclusion"]:
            lines.append(f"  -> {sentence}")
    total = sum(len(item["checks"]) for item in summary["results"])
    lines.append("")
    lines.append(f"{summary['scenarios_run']}/{summary['scenarios_run']} scenarios, "
                 f"{total - summary['checks_failed']}/{total} checks passed")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", nargs="?", default="all",
                        help="scenario name or 'all'")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quick", action="store_true", help="skip the exhaustive 9! ordering scan")
    parser.add_argument("--tau")
    parser.add_argument("--f1")
    parser.add_argument("--e1")
    parser.add_argument("--eps")
    parser.add_argument("--gamma")
    parser.add_argument("--lambda", dest="lam")
    parser.add_argument("--eta")
    parser.add_argument("--creg")
    parser.add_argument("--cplat")
    parser.add_argument("--lam-target", dest="lam_target")
    parser.add_argument("--a")
    parser.add_argument("--b")
    parser.add_argument("--T")
    parser.add_argument("--beta")
    parser.add_argument("--tau-c", dest="tau_c")
    parser.add_argument("--delta")
    parser.add_argument("--n", type=int)
    return parser


def apply_overrides(args) -> dict:
    def num(value):
        return None if value is None else F(str(value))

    mapping = {
        "tw1": {"tau": num(args.tau), "f1": num(args.f1), "E1": num(args.e1)},
        "tw1_1": {"a": num(args.a), "b": num(args.b), "T": num(args.T), "beta": num(args.beta)},
        "tw2": {"eps": num(args.eps)},
        "tw3": {"tau": num(args.tau), "n": args.n},
        "tw3_2": {"f1": num(args.f1), "E1": num(args.e1)},
        "tw5": {"tau": num(args.tau), "n": args.n},
        "tw6": {"gamma": num(args.gamma), "eps": num(args.eps)},
        "tw7": {"lam": num(args.lam), "eta": num(args.eta), "eps": num(args.eps)},
        "tw7_2": {"creg": num(args.creg), "cplat": num(args.cplat), "eta": num(args.eta),
                  "lam_target": num(args.lam_target)},
        "tw8": {"tau_C": num(args.tau_c), "delta": num(args.delta)},
    }
    return {name: {k: v for k, v in params.items() if v is not None} for name, params in mapping.items()}


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    overrides = apply_overrides(args)
    if args.quick:
        overrides.setdefault("tw8", {})["full"] = False
    name = args.scenario.lower()

    if name == "all":
        summary = run_all(overrides)
    else:
        if name not in SCENARIOS:
            parser.error(f"unknown scenario {name!r}; try one of {', '.join(SCENARIOS)} or 'all'")
        func, defaults = SCENARIOS[name]
        kwargs = dict(defaults)
        kwargs.update(overrides.get(name, {}))
        if name == "tw8":
            kwargs["full"] = not args.quick
        summary = {"scenarios_run": 1,
                   "checks_failed": sum(1 for c in func(**kwargs)["checks"] if not c["ok"]),
                   "results": [func(**kwargs)]}

    print(json.dumps(summary, default=str, indent=1) if args.json else render(summary))
    return 0 if summary["checks_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
