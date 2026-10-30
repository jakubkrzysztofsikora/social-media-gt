"""Approximate parameter estimation: conditional logit and QRE marginals.

Estimates are approximate evidence, never exact proofs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

from game_models import (BayesianGame, GameSpecError, NormalFormGame,
                         evaluate_cell, to_rational)
from game_solvers import _pg_game
from scipy.optimize import brentq

QRE_GRID_POINTS = 145
QRE_LAMBDA_MIN = 1e-6
QRE_LAMBDA_MAX = 1e6
QRE_TARGET_TOLERANCE = 1e-8
CHI2_95_1DF = 3.84
MAX_ESTIMATION_OBSERVATIONS = 100_000

ESTIMATION_METHODS = ("conditional_logit", "qre_marginals")


@dataclass
class EstimateResult:
    method: str
    status: str                      # approximate|unidentified|solver_unavailable
    parameters: dict = field(default_factory=dict)
    display_strings: dict = field(default_factory=dict)
    intervals: dict = field(default_factory=dict)
    lambda_mle: float | None = None
    lambda_intervals: list = field(default_factory=list)
    missed_lambda_points: list = field(default_factory=list)
    boundary_limited: bool = False
    correlation_retained: bool | None = None
    assumptions: list[str] = field(default_factory=list)


def estimate_game_parameters(envelope, observations, method, confirmed_by_user):
    """Entry point used by advisor/worker; dispatches by method."""
    if not confirmed_by_user:
        return {"status": "needs_confirmation"}
    if method not in ESTIMATION_METHODS:
        raise GameSpecError(f"estimation method {method!r} unsupported")
    total = sum(int(row["count"]) for row in observations)
    if total > MAX_ESTIMATION_OBSERVATIONS:
        raise GameSpecError(f"estimation accepts at most {MAX_ESTIMATION_OBSERVATIONS} observations")
    game = envelope.game
    if method == "conditional_logit":
        if not isinstance(game, BayesianGame):
            raise GameSpecError("conditional_logit requires a BayesianGame")
        return _fit_conditional_logit(envelope, game, observations)
    if not isinstance(game, NormalFormGame):
        raise GameSpecError("qre_marginals requires a NormalFormGame")
    return estimate_qre_marginals(game, observations, envelope.point_parameters())


# ------------------------------------------------------------- conditional logit

def _fit_conditional_logit(envelope, game: BayesianGame, counts: list[dict]) -> EstimateResult:
    """Fit interval-bounded affine coefficients over declared parameters.

    Each declared ParameterSpec with interval bounds is one free coefficient
    theta_p; utility cells are affine over the parameters, so
    P(a_i | a_-i, type) = softmax(U_i) is linear in theta. Point parameters
    stay fixed; prior-mode parameters are rejected. MLE via bounded L-BFGS-B;
    a MLE touching a bound or a profile-likelihood interval that cannot close
    reports `unidentified`. Intervals are per-parameter profile likelihoods
    (other coefficients re-optimized within their bounds), chi-square 3.84.
    """
    import numpy as np
    from scipy.optimize import minimize

    if any(spec.prior is not None for spec in envelope.parameters.values()):
        raise GameSpecError("conditional_logit rejects prior-mode parameters")
    free_names = [name for name, spec in envelope.parameters.items()
                  if spec.lower_bound is not None or spec.upper_bound is not None]
    fixed = {}
    for name, spec in envelope.parameters.items():
        if name in free_names:
            continue
        if spec.value is None:
            raise GameSpecError(f"parameter {name}: point parameters must carry a value")
        fixed[name] = spec.value
    if not free_names:
        raise GameSpecError("conditional_logit needs at least one interval-bounded "
                            "parameter to fit")
    lower = np.array([float(envelope.parameters[n].lower_bound)
                      if envelope.parameters[n].lower_bound is not None else -1e12
                      for n in free_names])
    upper = np.array([float(envelope.parameters[n].upper_bound)
                      if envelope.parameters[n].upper_bound is not None else 1e12
                      for n in free_names])
    if np.any(lower >= upper):
        raise GameSpecError("parameter bounds must satisfy lower < upper")

    players = [p.id for p in game.players]
    actions = {pid: list(game.actions[pid]) for pid in players}
    type_profiles = [dict(sorted(row["type_profile"].items())) for row in game.prior]

    count_map = {}
    for row in counts:
        if int(row["count"]) <= 0:
            raise GameSpecError("conditional_logit counts must be positive")
        tp_key = tuple(sorted(row["type_profile"].items()))
        ap_key = tuple(sorted(row["action_profile"].items()))
        count_map[(tp_key, ap_key)] = count_map.get((tp_key, ap_key), 0) + int(row["count"])
    if not count_map:
        raise GameSpecError("conditional_logit needs at least one observation")

    # payoff utility of pid playing own action ai, given the full action
    # profile ap and type profile tp, evaluated at coefficient values theta
    def utility_of(pid: str, ai: str, tp: dict, ap: dict, theta) -> float:
        profile = dict(ap)
        profile[pid] = ai
        key = (tuple(sorted(tp.items())), tuple(sorted(profile.items())))
        row = payoff_rows.get(key)
        if row is None:
            raise GameSpecError(f"payoff row missing for type {tp} actions {profile}")
        values = {n: theta[k] for k, n in enumerate(free_names)}
        values.update(fixed)
        return float(evaluate_cell(row["utilities"][pid], values))

    payoff_rows = {}
    for row in game.payoffs:
        key = (tuple(sorted(row["type_profile"].items())),
               tuple(sorted(row["action_profile"].items())))
        payoff_rows[key] = row

    def neg_log_likelihood(theta) -> float:
        values = {n: theta[k] for k, n in enumerate(free_names)}
        values.update(fixed)
        total = 0.0
        for (tp_key, ap_key), count in count_map.items():
            if tp_key not in {tuple(sorted(tp.items())) for tp in type_profiles}:
                raise GameSpecError(f"counts reference unknown type profile {dict(tp_key)}")
            tp = dict(tp_key)
            ap = dict(ap_key)
            for pid in players:
                if len(actions[pid]) < 2:
                    continue
                scores = [utility_of(pid, a, tp, ap, theta) for a in actions[pid]]
                m = max(scores)
                log_denom = m + math.log(sum(math.exp(s - m) for s in scores))
                chosen = ap[pid]
                if chosen not in actions[pid]:
                    raise GameSpecError(f"counts reference unknown action {chosen!r} for {pid}")
                chosen_score = utility_of(pid, chosen, tp, ap, theta)
                total += count * (chosen_score - log_denom)
        return -total

    theta0 = np.array([min(max(0.0, lo), up) for lo, up in zip(lower, upper)])
    warm = [theta0]

    def bounded_mle(x0) -> tuple:
        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                          bounds=list(zip(lower, upper)))
        return result

    result = bounded_mle(theta0)
    if not result.success:
        return EstimateResult(method="conditional_logit", status="solver_unavailable",
                              assumptions=[f"optimizer failed: {result.message}"])
    theta = np.asarray(result.x, dtype=float)
    ell_max = -float(neg_log_likelihood(theta))

    tol = 1e-6
    touching = [free_names[k] for k in range(len(free_names))
                if abs(theta[k] - lower[k]) <= tol or abs(theta[k] - upper[k]) <= tol]
    if touching:
        return EstimateResult(method="conditional_logit", status="unidentified",
                              boundary_limited=True,
                              assumptions=["MLE touches a parameter bound: "
                                           + ", ".join(touching)])

    parameters, display_strings, intervals = {}, {}, {}
    for k, name in enumerate(free_names):
        parameters[name] = float(theta[k])
        display_strings[name] = _decimal(theta[k])

    def profile_ell(value: float, j: int) -> float:
        """Profile log-likelihood: theta_j fixed at value, the other
        coefficients re-optimized within their bounds."""
        others = [k for k in range(len(free_names)) if k != j]
        bounds = [(lower[k], upper[k]) for k in others]

        def inner_nll(x_others) -> float:
            trial = theta.copy()
            trial[j] = value
            for k, v in zip(others, x_others):
                trial[k] = v
            return neg_log_likelihood(trial)

        if others:
            inner = minimize(inner_nll, theta[others], method="L-BFGS-B", bounds=bounds)
            if not inner.success:
                return -math.inf
            return -float(inner_nll(inner.x))
        trial = theta.copy()
        trial[j] = value
        return -float(neg_log_likelihood(trial))

    target = ell_max - CHI2_95_1DF / 2.0
    boundary_limited = False
    for k, name in enumerate(free_names):
        center = float(theta[k])
        lo, hi = float(lower[k]), float(upper[k])

        def ell_at(v, k=k):
            return profile_ell(v, k)

        # Walk outward until the profile likelihood drops below the target,
        # keeping the last OUTSIDE point as the brentq bracket end.
        def bracket_outward(from_value: float, sign: int) -> tuple[float, float]:
            """Returns (inside_point, outside_point) on one side."""
            step = max(0.25, 0.1 * max(1.0, abs(from_value)))
            inside = from_value
            while True:
                outside = inside + sign * step
                hit_bound = (outside <= lo) if sign < 0 else (outside >= hi)
                if hit_bound:
                    return inside, (lo if sign < 0 else hi)
                if ell_at(outside) <= target:
                    return inside, outside
                inside = outside
                step *= 2.0

        inside_lo, outside_lo = bracket_outward(center, -1)
        inside_hi, outside_hi = bracket_outward(center, +1)

        low_cross = None
        if outside_lo < inside_lo and ell_at(outside_lo) <= target <= ell_at(inside_lo):
            low_cross = brentq(lambda v: ell_at(v) - target, outside_lo, inside_lo,
                               xtol=1e-10, rtol=1e-12)
        elif inside_lo <= lo and ell_at(lo) > target:
            low_cross = lo           # likelihood still above target at the bound
        high_cross = None
        if inside_hi < outside_hi and ell_at(outside_hi) <= target <= ell_at(inside_hi):
            high_cross = brentq(lambda v: ell_at(v) - target, inside_hi, outside_hi,
                                xtol=1e-10, rtol=1e-12)
        elif inside_hi >= hi and ell_at(hi) > target:
            high_cross = hi
        if low_cross is None or high_cross is None:
            return EstimateResult(method="conditional_logit", status="unidentified",
                                  boundary_limited=True,
                                  assumptions=[f"profile-likelihood interval for {name} "
                                               "failed to close"])
        if low_cross <= lo + tol or high_cross >= hi - tol:
            boundary_limited = True
        intervals[name] = [float(low_cross), float(high_cross)]

    return EstimateResult(method="conditional_logit", status="approximate",
                          parameters=parameters, display_strings=display_strings,
                          intervals=intervals, boundary_limited=boundary_limited,
                          correlation_retained=True,
                          assumptions=["nominal pseudo-likelihood intervals; responses within "
                                       "joint profiles are dependent",
                                       "lambda fixed at 1; bounded affine coefficients over "
                                       "declared parameters; profile likelihood re-optimizes "
                                       "other coefficients"])


# ----------------------------------------------------------------- QRE marginals

def _decimal(value: float) -> str:
    return f"{value:.12f}".rstrip("0").rstrip(".") if value == value else "0"


def estimate_qre_marginals(game: NormalFormGame, counts: list[dict],
                           values=None) -> EstimateResult:
    """Profile likelihood over the principal QRE branch, strict lambda checks."""
    import numpy as np
    import pygambit as pg
    from scipy.optimize import brentq, minimize_scalar

    game.validate()
    players = [p.id for p in game.players]
    g = _pg_game(game, values)
    count_map = {}
    for row in counts:
        pid, aid, count = row["player_id"], row["action_id"], int(row["count"])
        if count < 0:
            raise GameSpecError("qre_marginals counts must be nonnegative")
        if pid not in players or aid not in game.actions[pid]:
            raise GameSpecError(f"qre_marginals references unknown player/action {pid}/{aid}")
        count_map[(pid, aid)] = count_map.get((pid, aid), 0) + count
    for pid in players:
        if sum(count_map.get((pid, a), 0) for a in game.actions[pid]) <= 0:
            raise GameSpecError(f"qre_marginals: player {pid} needs a positive total")

    grid = [QRE_LAMBDA_MIN * (QRE_LAMBDA_MAX / QRE_LAMBDA_MIN) ** (i / (QRE_GRID_POINTS - 1))
            for i in range(QRE_GRID_POINTS)]
    lambdas, logliks, missed = [], [], []
    for lam in grid:
        sigma = _qre_probabilities(g, lam, pg, missed)
        if sigma is None:
            continue
        ell = 0.0
        usable = True
        for pi, pid in enumerate(players):
            for ai, action in enumerate(game.actions[pid]):
                n = count_map.get((pid, action), 0)
                if n and sigma[pi][ai] <= 0.0:
                    usable = False
                    break
                if n:
                    ell += n * math.log(sigma[pi][ai])
            if not usable:
                break
        if not usable:
            missed.append(lam)
            continue
        lambdas.append(lam)
        logliks.append(ell)
    if len(lambdas) < 3:
        return EstimateResult(method="qre_marginals", status="solver_unavailable",
                              missed_lambda_points=missed,
                              assumptions=["too few usable QRE points to bound the likelihood"])

    def ell_at(lam: float) -> float:
        sigma = _qre_probabilities(g, lam, pg, None)
        if sigma is None:
            return -math.inf
        ell = 0.0
        for pi, pid in enumerate(players):
            for ai, action in enumerate(game.actions[pid]):
                n = count_map.get((pid, action), 0)
                if n:
                    if sigma[pi][ai] <= 0.0:
                        return -math.inf
                    ell += n * math.log(sigma[pi][ai])
        return ell

    best_i = max(range(len(lambdas)), key=lambda i: logliks[i])
    refined = minimize_scalar(lambda x: -ell_at(x), bounds=(lambdas[max(best_i - 1, 0)],
                                                            lambdas[min(best_i + 1, len(lambdas) - 1)]),
                              method="bounded")
    mle = float(refined.x) if refined.success else lambdas[best_i]
    ell_max = -float(refined.fun) if refined.success else logliks[best_i]
    target = ell_max - CHI2_95_1DF / 2.0
    if not math.isfinite(target):
        return EstimateResult(method="qre_marginals", status="unidentified",
                              lambda_mle=mle, correlation_retained=False,
                              assumptions=["the profile likelihood is flat or non-finite"])

    # find connected components of {ell >= target} on the observed grid
    components, start = [], None
    for i, ell in enumerate(logliks):
        if ell >= target and start is None:
            start = i
        elif ell < target and start is not None:
            components.append((start, i - 1))
            start = None
    if start is not None:
        components.append((start, len(logliks) - 1))
    intervals = []
    touches_boundary = False
    for i0, i1 in components:
        lo, hi = lambdas[i0], lambdas[i1]
        if i0 == 0 or i1 == len(lambdas) - 1:
            touches_boundary = True
        if i0 > 0:
            lo = brentq(lambda x: ell_at(x) - target, lambdas[i0 - 1], lambdas[i0])
        if i1 < len(lambdas) - 1:
            hi = brentq(lambda x: ell_at(x) - target, lambdas[i1], lambdas[i1 + 1])
        intervals.append([float(lo), float(hi)])
    if touches_boundary:
        return EstimateResult(method="qre_marginals", status="unidentified",
                              lambda_mle=mle, lambda_intervals=intervals,
                              missed_lambda_points=missed, correlation_retained=False,
                              assumptions=["a 95% likelihood component touches the grid boundary"])
    if missed:
        near_mle = [lam for lam in missed if abs(lam - mle) / mle < 0.05]
        endpoint_risk = any(any(abs(lam - pt) / pt < 0.05 for lam in missed)
                            for iv in intervals for pt in iv)
        if near_mle or endpoint_risk:
            return EstimateResult(method="qre_marginals", status="solver_unavailable",
                                  lambda_mle=mle, lambda_intervals=intervals,
                                  missed_lambda_points=missed, correlation_retained=False,
                                  assumptions=["a missed lambda point adjoins the MLE or an "
                                               "interval endpoint; the likelihood set may be wrong"])
    display = _decimal(mle)
    return EstimateResult(method="qre_marginals", status="approximate",
                          lambda_mle=mle, lambda_intervals=intervals,
                          missed_lambda_points=missed, correlation_retained=False,
                          display_strings={"lambda": display},
                          assumptions=["principal QRE branch; marginal counts discard correlation",
                                       "every returned profile verified against requested lambda",
                                       "estimate is approximate, never an exact proof"])


def _qre_probabilities(g, lam, pg, missed):
    try:
        result = pg.qre.logit_solve_lambda(g, [lam])
        profiles = result if isinstance(result, list) else getattr(result, "equilibria", [result])
        profile = profiles[0]
        actual = float(profile.lam)
    except Exception:
        if missed is not None:
            missed.append(lam)
        return None
    if abs(actual - lam) / lam > QRE_TARGET_TOLERANCE:
        if missed is not None:
            missed.append(lam)
        return None
    nested = getattr(profile, "profile", None)
    if nested is not None:
        pg_players = list(g.players)
        rows = []
        for pi in range(len(pg_players)):
            player = pg_players[pi]
            rows.append([float(nested[player][st]) for st in player.strategies])
        return rows
    return [[float(profile[pi, ai]) for ai in range(len(list(g.players[pi].strategies)))]
            for pi in range(len(list(g.players)))]


# ----------------------------------------------------------------- apply estimates

def apply_estimates(game_envelope, estimate_result, parameter_ids, confirmed_by_user):
    """Copy displayed decimal strings into point parameters; never mutate input."""
    if not confirmed_by_user:
        return {"status": "needs_confirmation"}
    import copy
    envelope = copy.deepcopy(game_envelope)
    for pid in parameter_ids:
        text = _display_value(estimate_result, pid)
        frac = to_rational(text, f"estimate {pid}")
        if pid not in envelope.parameters:
            raise GameSpecError(f"parameter {pid!r} not declared in game spec")
        spec = envelope.parameters[pid]
        spec.value = frac
        spec.lower_bound = None
        spec.upper_bound = None
        spec.prior = None
        spec.provenance = {"origin": "estimated",
                           "source": estimate_result.get("method", "unknown")}
    envelope.validate()
    return {"status": "ok", "envelope": envelope,
            "provenance": {pid: {"origin": "estimated",
                                 "source": estimate_result.get("method")}
                           for pid in parameter_ids}}


def _display_value(estimate_result, parameter_id) -> str:
    displays = estimate_result.get("display_strings") or {}
    if parameter_id in displays:
        return str(displays[parameter_id])
    raise GameSpecError(f"estimate {parameter_id!r} has no displayed decimal string")


MAX_SENSITIVITY_POINTS = 256


def analyze_sensitivity(envelope, assignments, confirmed_by_user,
                        budget_seconds=300.0, run_point=None):
    """Per-point exact analysis over complete rational in-bound assignments."""
    import copy
    import time

    if not confirmed_by_user:
        return {"status": "needs_confirmation"}
    if len(assignments) > MAX_SENSITIVITY_POINTS:
        raise GameSpecError(f"sensitivity accepts at most {MAX_SENSITIVITY_POINTS} assignments")
    if run_point is None:
        from game_solvers import solve_exact_game
        run_point = solve_exact_game
    results = []
    not_covered = []
    start = time.monotonic()
    for index, assignment in enumerate(assignments):
        if time.monotonic() - start > budget_seconds:
            not_covered.extend(range(index, len(assignments)))
            break
        env = copy.deepcopy(envelope)
        for name, value in assignment.items():
            if name not in env.parameters:
                raise GameSpecError(f"assignment names unknown parameter {name!r}")
            frac = to_rational(value, f"assignment {name}")
            spec = env.parameters[name]
            if spec.lower_bound is not None and frac < spec.lower_bound:
                raise GameSpecError(f"assignment {name} below declared lower bound")
            if spec.upper_bound is not None and frac > spec.upper_bound:
                raise GameSpecError(f"assignment {name} above declared upper bound")
            if spec.prior is not None:
                if frac not in [row["value"] for row in spec.prior]:
                    raise GameSpecError(f"assignment {name} outside declared prior support")
            spec.value = frac
        env.validate()
        results.append({"assignment": assignment, "results": run_point(env)})
    return {"status": "approximate" if not_covered else "exact_verified",
            "points": results, "not_covered": not_covered}
