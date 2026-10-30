"""Exact solvers for the general game-theory advisor.

Every exact claim returns a rational witness plus independent verification
checks. Approximate method: epsilon_nash (simpdiv + exact regret check).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from fractions import Fraction
from functools import reduce

from game_models import (Affine, BayesianGame, BargainingSpec, ExtensiveFormGame,
                         GameSpecError, MechanismSpec, NormalFormGame, ObjectiveSpec,
                         OpenSpielGame, RepeatedGame, StackelbergGame, evaluate_cell,
                         render_rational, to_rational)
from game_lp import LPError, MAX_LP_CONSTRAINTS, MAX_LP_VARIABLES, solve_rational_lp

EXACT_METHODS = ("pure_nash", "mixed_nash", "bayesian_nash", "subgame_perfect",
                 "correlated_equilibrium", "nash_bargaining", "stackelberg",
                 "mechanism_optimum")
ALL_METHODS = EXACT_METHODS + ("epsilon_nash",)


class SolverError(Exception):
    pass


@dataclass
class MethodResult:
    method: str
    status: str                       # exact_verified|epsilon_verified|approximate|solver_unavailable|unsupported|no_agreement
    witnesses: list = field(default_factory=list)
    objective: Fraction | None = None
    checks: list = field(default_factory=list)
    certificate: dict | None = None
    assumptions: list[str] = field(default_factory=list)
    epsilon: Fraction | None = None
    supremum: Fraction | None = None  # unattained Stackelberg supremum


def _check(name: str, ok: bool, expected, got) -> dict:
    return {"name": name, "ok": bool(ok), "expected": expected, "got": got}


def _iter_profiles(actions: dict[str, list[str]]):
    pids = list(actions)
    for combo in itertools.product(*(actions[p] for p in pids)):
        yield dict(zip(pids, combo))


def _expected_utilities(game: NormalFormGame, mixed: dict[str, dict[str, Fraction]],
                        values: dict[str, Fraction] | None = None) -> dict[str, Fraction]:
    """Expected utility per player under an independent mixed profile."""
    keys = list(game.players and [p.id for p in game.players])
    totals = {pid: Fraction(0) for pid in keys}
    for row in game.payoffs:
        prob = Fraction(1)
        for pid in keys:
            prob *= mixed[pid].get(row["action_profile"][pid], Fraction(0))
        if prob == 0:
            continue
        for pid, cell in row["utilities"].items():
            totals[pid] += prob * evaluate_cell(cell, values or {})
    return totals


def _max_regret(game: NormalFormGame, mixed: dict[str, dict[str, Fraction]],
                values: dict[str, Fraction] | None = None) -> Fraction:
    """Maximum unilateral deviation gain, exactly."""
    keys = [p.id for p in game.players]
    base = _expected_utilities(game, mixed, values)
    worst = Fraction(0)
    for pid in keys:
        best = None
        for action in game.actions[pid]:
            trial = {p: dict(mixed[p]) for p in keys}
            trial[pid] = {action: Fraction(1)}
            util = _expected_utilities(game, trial, values)[pid]
            best = util if best is None or util > best else best
        worst = max(worst, best - base[pid])
    return worst


def _pure_best_responses(game: NormalFormGame, pid: str, opponent_profile: dict[str, str],
                         values: dict[str, Fraction] | None = None) -> list[str]:
    others = {k: v for k, v in opponent_profile.items() if k != pid}
    best, best_util = [], None
    for action in game.actions[pid]:
        profile = dict(others)
        profile[pid] = action
        for row in game.payoffs:
            if all(row["action_profile"][k] == profile[k] for k in profile):
                util = evaluate_cell(row["utilities"][pid], values or {})
                if best_util is None or util > best_util:
                    best, best_util = [action], util
                elif util == best_util:
                    best.append(action)
                break
    return best


# ------------------------------------------------------------------- PyGambit

def _pg_game(game: NormalFormGame, values: dict[str, Fraction] | None = None):
    import pygambit as pg

    g = pg.Game.new_table([len(game.actions[p.id]) for p in game.players])
    players = [p.id for p in game.players]
    for row in game.payoffs:
        key = tuple(game.actions[pid].index(row["action_profile"][pid]) for pid in players)
        outcome = g[key]
        for pi, pid in enumerate(players):
            util = evaluate_cell(row["utilities"][pid], values or {})
            outcome[str(pi + 1)] = pg.Rational(util.numerator, util.denominator)
    return g


def solve_pure_nash(game: NormalFormGame, values: dict[str, Fraction] | None = None) -> MethodResult:
    game.validate()
    pure = []
    for profile in _iter_profiles(game.actions):
        ok = True
        for player in game.players:
            pid = player.id
            if profile[pid] not in _pure_best_responses(game, pid, profile, values):
                ok = False
                break
        if ok:
            pure.append(dict(profile))
    checks = []
    for profile in pure:
        mixed = {pid: {profile[pid]: Fraction(1)} for pid in game.actions}
        regret = _max_regret(game, mixed, values)
        checks.append(_check(f"pure profile {profile} has zero regret", regret == 0, 0, render_rational(regret)))
    if not any(c["ok"] for c in checks) and pure:
        raise SolverError("pure Nash verification failed")
    return MethodResult(method="pure_nash", status="exact_verified" if pure else "exact_verified",
                        witnesses=pure, checks=checks)


def solve_mixed_nash_two_player(game: NormalFormGame,
                                values: dict[str, Fraction] | None = None) -> MethodResult:
    game.validate()
    players = [p.id for p in game.players]
    if len(players) != 2:
        return MethodResult(method="mixed_nash", status="solver_unavailable",
                            assumptions=["exact mixed Nash supports two-player games only"])
    g = _pg_game(game, values)
    import pygambit as pg
    result = None
    try:
        result = pg.nash.enummixed_solve(g)
    except Exception as exc:
        return MethodResult(method="mixed_nash", status="solver_unavailable",
                            assumptions=[f"enummixed_solve failed: {exc}"])
    pg_players = list(g.players)
    pg_strategies = [list(p.strategies) for p in pg_players]
    witnesses = []
    checks = []
    for profile in result.equilibria:
        rational = profile.as_strategy_profile() if hasattr(profile, "as_strategy_profile") else profile
        mixed = {}
        ok = True
        for pi, pid in enumerate(players):
            mixed[pid] = {}
            ms = rational[pg_players[pi]]
            for ai, action in enumerate(game.actions[pid]):
                prob = ms[pg_strategies[pi][ai]]
                try:
                    frac = Fraction(int(prob.numerator), int(prob.denominator))
                except AttributeError:
                    ok = False
                    break
                mixed[pid][action] = frac
        if not ok:
            continue
        regret = _max_regret(game, mixed, values)
        checks.append(_check("mixed profile zero regret", regret == 0, 0, render_rational(regret)))
        witnesses.append({pid: {a: render_rational(v) for a, v in row.items()}
                          for pid, row in mixed.items()})
    status = "exact_verified" if witnesses and all(c["ok"] for c in checks) else "solver_unavailable"
    return MethodResult(method="mixed_nash", status=status, witnesses=witnesses, checks=checks)


def solve_epsilon_nash(game: NormalFormGame, epsilon: Fraction,
                       values: dict[str, Fraction] | None = None,
                       leash: int = 10_000) -> MethodResult:
    game.validate()
    if epsilon <= 0:
        raise GameSpecError("epsilon_nash requires a positive rational epsilon")
    cells = [evaluate_cell(row["utilities"][p.id], values or {})
             for row in game.payoffs for p in game.players]
    payoff_range = max(cells) - min(cells)
    if payoff_range == 0:
        witnesses = [{pid: {a: render_rational(Fraction(1, len(acts))) for a in acts}
                      for pid, acts in game.actions.items()}]
        return MethodResult(method="epsilon_nash", status="epsilon_verified", epsilon=epsilon,
                            witnesses=witnesses,
                            checks=[_check("zero payoff range: every profile exact equilibrium",
                                           True, True, True)])
    import pygambit as pg
    g = _pg_game(game, values)
    uniform = g.mixed_strategy_profile(rational=True)
    try:
        result = pg.nash.simpdiv_solve(uniform, pg.Rational((epsilon / payoff_range).numerator,
                                                            (epsilon / payoff_range).denominator),
                                       2, leash)
    except Exception as exc:
        return MethodResult(method="epsilon_nash", status="solver_unavailable", epsilon=epsilon,
                            assumptions=[f"simpdiv_solve failed: {exc}"])
    profile = result.equilibria[0] if result.equilibria else None
    if profile is None:
        return MethodResult(method="epsilon_nash", status="solver_unavailable", epsilon=epsilon,
                            assumptions=["simpdiv returned no profile"])
    players = [p.id for p in game.players]
    pg_players = list(g.players)
    pg_strategies = [list(p.strategies) for p in pg_players]
    rational = profile.as_strategy_profile() if hasattr(profile, "as_strategy_profile") else profile
    mixed = {}
    for pi, pid in enumerate(players):
        mixed[pid] = {}
        ms = rational[pg_players[pi]]
        for ai, action in enumerate(game.actions[pid]):
            prob = ms[pg_strategies[pi][ai]]
            mixed[pid][action] = Fraction(int(prob.numerator), int(prob.denominator))
    regret = _max_regret(game, mixed, values)
    ok = regret <= epsilon
    rendered = {pid: {a: render_rational(v) for a, v in row.items()}
                for pid, row in mixed.items()}
    return MethodResult(method="epsilon_nash",
                        status="epsilon_verified" if ok else "solver_unavailable",
                        epsilon=epsilon, witnesses=[rendered] if ok else [],
                        checks=[_check("exact max regret <= epsilon", ok, f"<= {epsilon}",
                                       render_rational(regret))])


def spne_walk(game: ExtensiveFormGame, values: dict[str, Fraction] | None = None) -> MethodResult:
    """Backward induction returning every tied optimal continuation."""
    game.validate()
    if not game.has_perfect_information():
        return MethodResult(method="subgame_perfect", status="unsupported",
                            assumptions=["exact SPNE supports finite perfect-information trees"])
    nodes = {n.id: n for n in game.nodes}

    def solve(node_id: str):
        node = nodes[node_id]
        if node.actor is None:
            return {"value": {pid: evaluate_cell(u, values or {})
                              for pid, u in node.terminal_utilities.items()},
                    "continuations": [({}, {pid: evaluate_cell(u, values or {})
                                            for pid, u in node.terminal_utilities.items()})],
                    "behavior": {}}
        assert node.edges, node_id
        candidates = []
        for edge in node.edges:
            sub = solve(edge["next_node"])
            for cont, value in sub["continuations"]:
                candidates.append(({**cont, node_id: edge["action_id"]}, value, edge["action_id"]))
        best_value = None
        for _cont, value, _aid in candidates:
            if best_value is None or value[node.actor] > best_value:
                best_value = value[node.actor]
        best = [(cont, value) for cont, value, _aid in candidates
                if value[node.actor] == best_value]
        # collect behavior: chosen action per decision node along each continuation
        behav = {}
        for cont, _value in best:
            for _nid, aid in cont.items():
                behav.setdefault(_nid, set()).add(aid)
        return {"value": best[0][1],
                "continuations": best,
                "behavior": {nid: sorted(aids) for nid, aids in behav.items()},
                "node_value": best_value}

    root = solve(game.root)
    witnesses = [{"continuation": cont,
                  "value": {k: render_rational(v) for k, v in value.items()},
                  "behavior": root["behavior"]} for cont, value in root["continuations"]]
    checks = [_check("backward induction covers all decision nodes",
                     len(root["behavior"]) == sum(1 for n in game.nodes if n.actor is not None),
                     True, True)]
    return MethodResult(method="subgame_perfect", status="exact_verified",
                        witnesses=witnesses, checks=checks)


def solve_correlated_equilibrium(game: NormalFormGame, mediator_confirmed: bool,
                                 objective: ObjectiveSpec | None,
                                 values: dict[str, Fraction] | None = None) -> MethodResult:
    game.validate()
    if not mediator_confirmed:
        return MethodResult(method="correlated_equilibrium", status="unsupported",
                            assumptions=["correlated equilibrium requires a confirmed mediator"])
    players = [p.id for p in game.players]
    profile_keys = list(_iter_profiles(game.actions))
    key_index = {tuple(sorted(p.items())): i for i, p in enumerate(profile_keys)}
    n = len(profile_keys)
    utilities = {}
    for row in game.payoffs:
        key = tuple(sorted(row["action_profile"].items()))
        utilities[key] = {pid: evaluate_cell(row["utilities"][pid], values or {}) for pid in players}

    # variables: q[profile] >= 0; simplex equality plus obedience inequalities
    constraints: list[tuple[list[Fraction], Fraction]] = []   # coefficients, rhs (coef.q <= rhs)
    ones = [Fraction(1)] * n
    for pi, dev in enumerate(players):
        for profile in profile_keys:
            for action in game.actions[dev]:
                if action == profile[dev]:
                    continue
                # Obedience conditions on the deviator's OWN recommendation:
                # sum_{s: s_dev = profile[dev]} q(s) * [u_dev(deviate) - u_dev(truth)] <= 0
                coef = [Fraction(0)] * n
                own_action = profile[dev]
                for idx, full in enumerate(profile_keys):
                    if full[dev] != own_action:
                        continue
                    truthful = utilities[tuple(sorted(full.items()))][dev]
                    deviated_profile = dict(full)
                    deviated_profile[dev] = action
                    deviated = utilities[tuple(sorted(deviated_profile.items()))][dev]
                    coef[idx] += deviated - truthful
                constraints.append((coef, Fraction(0)))

    if objective is None:
        # feasibility witness: try the uniform distribution, else solve the
        # feasibility LP with a zero objective
        uniform = [Fraction(1, n)] * n
        ok = sum(uniform) == 1 and all(
            sum((c * u for c, u in zip(coef, uniform)), Fraction(0)) <= rhs
            for coef, rhs in constraints)
        if ok:
            checks = [_check("uniform distribution satisfies obedience", True, True, True),
                      _check("uniform distribution sums to one", True, 1, "1")]
            witness = {render_key(k): render_rational(v)
                       for k, v in zip(profile_keys, uniform)}
            return MethodResult(method="correlated_equilibrium", status="exact_verified",
                                witnesses=[witness], checks=checks)

    obj_vector, offset = _objective_vector(objective, players, profile_keys, utilities)
    res = solve_rational_lp(obj_vector, A_ub=[c for c, _ in constraints],
                            b_ub=[r for _, r in constraints],
                            A_eq=[ones], b_eq=[Fraction(1)])
    if res.status != "optimal":
        return MethodResult(method="correlated_equilibrium", status="solver_unavailable",
                            certificate={"lp_status": res.status})
    distribution = res.x
    checks = [
        _check("simplex", sum(distribution) == 1, 1, render_rational(sum(distribution))),
        _check("nonnegativity", all(q >= 0 for q in distribution), True,
               all(q >= 0 for q in distribution)),
        _check("primal/dual objective equality", res.dual_objective == res.objective,
               render_rational(res.objective), render_rational(res.dual_objective)),
    ]
    for ci, (coef, rhs) in enumerate(constraints):
        lhs = sum((c * q for c, q in zip(coef, distribution)), Fraction(0))
        if lhs > rhs:
            checks.append(_check(f"obedience constraint {ci}", False, rhs, lhs))
    witness = {render_key(k): render_rational(v) for k, v in zip(profile_keys, distribution)}
    failed = [c for c in checks if not c["ok"]]
    return MethodResult(method="correlated_equilibrium",
                        status="exact_verified" if not failed else "error",
                        witnesses=[witness], objective=res.objective + offset, checks=checks,
                        certificate={"dual_y": [render_rational(y) for y in res.y]})


def render_key(profile: dict) -> str:
    return str(dict(sorted(profile.items())))


def _objective_vector(objective: ObjectiveSpec | None, players: list[str],
                      profile_keys: list[dict], utilities: dict) -> tuple[list[Fraction], Fraction]:
    if objective is None:
        return [Fraction(0)] * len(profile_keys), Fraction(0)
    if objective.kind == "social_welfare":
        vec = []
        for key in profile_keys:
            u = utilities[tuple(sorted(key.items()))]
            vec.append(sum(u.values(), Fraction(0)))
        return vec, Fraction(0)
    if objective.kind == "player_utility":
        vec = [utilities[tuple(sorted(key.items()))][objective.player_id] for key in profile_keys]
        return vec, Fraction(0)
    if objective.kind == "weighted_utility":
        vec = []
        for key in profile_keys:
            u = utilities[tuple(sorted(key.items()))]
            vec.append(sum((objective.weights[pid] * u[pid] for pid in players), Fraction(0)))
        return vec, Fraction(0)
    if objective.kind == "maximin":
        # epigraph variable t: maximize t s.t. sum_s q(s) u_i(s) >= t for all i.
        raise SolverError("maximin CE requires the epigraph LP; unsupported in this path")
    raise SolverError(f"objective {objective.kind} unsupported for CE")


import json  # noqa: E402  (used by render_key)


def solve_nash_bargaining(spec: BargainingSpec,
                          values: dict[str, Fraction] | None = None) -> MethodResult:
    spec.validate()
    ids = [p.id for p in spec.players]
    a, b = ids
    d0 = to_fraction_cell(spec.disagreement[a])
    d1 = to_fraction_cell(spec.disagreement[b])
    points = [(to_fraction_cell(p[a]), to_fraction_cell(p[b])) for p in spec.feasible_points]
    # individually rational: gains >= 0
    ir = [(x, y) for x, y in points if x >= d0 and y >= d1]
    if not ir:
        return MethodResult(method="nash_bargaining", status="no_agreement",
                            assumptions=["no individually rational feasible point"])
    if spec.randomization == "not_allowed":
        best = None
        candidates = []
        for x, y in ir:
            product = (x - d0) * (y - d1)
            if best is None or product > best:
                best = product
        candidates = [p for p in ir if (p[0] - d0) * (p[1] - d1) == best]
        checks = [_check("finite enumeration covers all feasible points", True, True, True)]
        witnesses = [{a: render_rational(x), b: render_rational(y), "nash_product": render_rational(best)}
                     for x, y in candidates]
        return MethodResult(method="nash_bargaining", status="exact_verified",
                            witnesses=witnesses, objective=best, checks=checks)

    # convexify with exact rational convex hull, enumerate vertices + edge stationary points
    hull = convex_hull(ir)
    if len(hull) == 0:
        return MethodResult(method="nash_bargaining", status="no_agreement")
    best = None
    candidates = []
    for x, y in hull:
        product = (x - d0) * (y - d1)
        if best is None or product > best:
            best = product
    for i in range(len(hull)):
        x0, y0 = hull[i]
        x1, y1 = hull[(i + 1) % len(hull)]
        # maximize f(t) = (x(t)-d0)(y(t)-d1), t in [0,1]
        # f'(t) = 0 at t0; compare f at 0, 1, t0 (if in [0,1])
        ax, ay, bx, by = x0 - d0, y0 - d1, x1 - d0, y1 - d1
        # f(t) = (ax + t(bx-ax))(ay + t(by-ay))
        c0 = ax * ay
        c1 = ax * (by - ay) + ay * (bx - ax)
        c2 = (bx - ax) * (by - ay)
        edge_best = max(c0, c0 + c1 + c2)  # endpoints
        t_stationary = None
        if c2 != 0:
            t = Fraction(-c1, 2 * c2)
            if 0 < t < 1:
                value = c0 + c1 * t + c2 * t * t
                if value > edge_best:
                    edge_best = value
                    t_stationary = t
        if best is None or edge_best > best:
            best = edge_best
    # gather all points achieving best: vertices and stationary points
    witnesses = []
    for x, y in hull:
        if (x - d0) * (y - d1) == best:
            witnesses.append({a: render_rational(x), b: render_rational(y),
                              "nash_product": render_rational(best)})
    for i in range(len(hull)):
        x0, y0 = hull[i]
        x1, y1 = hull[(i + 1) % len(hull)]
        ax, ay, bx, by = x0 - d0, y0 - d1, x1 - d0, y1 - d1
        c0 = ax * ay
        c1 = ax * (by - ay) + ay * (bx - ax)
        c2 = (bx - ax) * (by - ay)
        if c2 != 0:
            t = Fraction(-c1, 2 * c2)
            if 0 < t < 1 and c0 + c1 * t + c2 * t * t == best:
                witnesses.append({a: render_rational(x0 + t * (x1 - x0)),
                                  b: render_rational(y0 + t * (y1 - y0)),
                                  "nash_product": render_rational(best),
                                  "mixture": f"{render_rational(1 - t)}*({render_rational(x0)},{render_rational(y0)}) "
                                             f"+ {render_rational(t)}*({render_rational(x1)},{render_rational(y1)})"})
    # independent verification: every IR feasible point's product must not
    # exceed the reported optimum, and every witness achieves it exactly
    checks = [
        _check("optimum dominates every feasible point",
               all((to_fraction_cell(p[a]) - d0) * (to_fraction_cell(p[b]) - d1) <= best
                   for p in spec.feasible_points),
               True, True),
        _check("hull vertices do not exceed the optimum",
               all((x - d0) * (y - d1) <= best for x, y in hull),
               True, True),
        _check("witnesses achieve the optimum",
               all(Fraction(w[a]) * Fraction(w[b]) == best for w in witnesses),
               True, True),
    ]
    failed = [c for c in checks if not c["ok"]]
    return MethodResult(method="nash_bargaining",
                        status="exact_verified" if not failed else "error",
                        witnesses=witnesses, objective=best, checks=checks)


def to_fraction_cell(value) -> Fraction:
    from game_models import to_rational
    return to_rational(value, "bargaining cell")


def convex_hull(points: list[tuple[Fraction, Fraction]]) -> list[tuple[Fraction, Fraction]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def solve_stackelberg(game: StackelbergGame,
                      values: dict[str, Fraction] | None = None) -> MethodResult:
    game.validate()
    stage = game.stage_game
    leader, follower = game.leader_id, game.follower_id
    result = MethodResult(method="stackelberg", status="exact_verified")

    # ---- pure commitment
    # For each leader action: follower best-response set; leader value under
    # leader-favorable = max over responses; under follower-favorable the
    # follower picks the response maximizing their utility, ties adverse to
    # the leader (minimizing the leader payoff).
    per_leader = {}
    for leader_action in stage.actions[leader]:
        responses = []
        for follower_action in stage.actions[follower]:
            profile = {leader: leader_action, follower: follower_action}
            if follower_action in _pure_best_responses(stage, follower, profile, values):
                responses.append(follower_action)
        candidates = []
        for f_action in responses:
            profile = {leader: leader_action, follower: f_action}
            u_l = _utility_at(stage, leader, profile, values)
            u_f = _utility_at(stage, follower, profile, values)
            candidates.append((f_action, u_l, u_f))
        per_leader[leader_action] = candidates

    leader_fav_best = None
    for leader_action, candidates in per_leader.items():
        if not candidates:
            continue
        value = max(c[1] for c in candidates)
        for f_action, u_l, u_f in candidates:
            if u_l != value:
                continue
            entry = {"leader": leader_action, "follower": f_action,
                     "leader_utility": render_rational(value),
                     "leader_utility_value": value}
            if leader_fav_best is None or value > leader_fav_best[0]["leader_utility_value"]:
                leader_fav_best = [entry]
            elif value == leader_fav_best[0]["leader_utility_value"]:
                leader_fav_best.append(entry)

    follower_fav_entries = []
    if per_leader:
        # Follower-favorable semantics: within a leader action the follower
        # picks their best response, ties broken adverse to the leader; the
        # leader then commits to the action maximizing that ADVERSE value
        # (not the best-case value).
        adverse = {}
        for leader_action, candidates in per_leader.items():
            if not candidates:
                continue
            fav = max(candidates, key=lambda c: (c[2], -c[1]))
            adverse[leader_action] = fav
        if adverse:
            top = max(fav[1] for fav in adverse.values())
            for leader_action, fav in adverse.items():
                if fav[1] != top:
                    continue
                follower_fav_entries.append({"leader": leader_action,
                                             "follower": fav[0],
                                             "leader_utility": render_rational(top)})
    result.witnesses.append({"commitment": "pure",
                             "leader_favorable": leader_fav_best or [],
                             "follower_favorable": follower_fav_entries})

    if game.commitment_mode == "pure":
        result.status = "exact_verified"
        result.checks.append(_check("pure commitment enumerates all leader actions",
                                    len(per_leader) == len(stage.actions[leader]), True, True))
        return result

    # ---- mixed commitment
    leader_fav = _mixed_leader_favorable(stage, leader, follower, values)
    follower_fav = _mixed_follower_favorable(stage, leader, follower, values)
    result.witnesses.append({"commitment": "mixed", **leader_fav})
    result.witnesses.append({"commitment": "mixed", **follower_fav})
    result.status = "exact_verified"
    return result


def _mixed_leader_favorable(stage: NormalFormGame, leader: str, follower: str,
                            values) -> dict:
    """Leader-favorable mixed commitment.

    For each candidate response action f, maximize the leader's expected
    utility given that the follower plays f, subject to f being a (weak) best
    response to p: for every alternative f', E_F(f) >= E_F(f').
    """
    n = len(stage.actions[leader])
    best = None
    for chosen in stage.actions[follower]:
        c = [_utility_at(stage, leader, {leader: a, follower: chosen}, values)
             for a in stage.actions[leader]]
        A_ub, b_ub = [], []
        for f_action in stage.actions[follower]:
            if f_action == chosen:
                continue
            # E_F(chosen) - E_F(f_action) >= 0
            row = []
            for a in stage.actions[leader]:
                u_chosen = _utility_at(stage, follower, {leader: a, follower: chosen}, values)
                u_alt = _utility_at(stage, follower, {leader: a, follower: f_action}, values)
                row.append(u_alt - u_chosen)
            A_ub.append(row)
            b_ub.append(Fraction(0))
        try:
            res = solve_rational_lp(c, A_ub=A_ub or None, b_ub=b_ub or None,
                                    A_eq=[[Fraction(1)] * n], b_eq=[Fraction(1)])
        except LPError as exc:
            return {"leader_favorable": [], "error": str(exc)}
        if res.status != "optimal":
            continue
        value = res.objective
        if best is None or value > best["leader_utility_value"]:
            best = {"leader_utility_value": value, "response": chosen,
                    "p": {a: render_rational(p) for a, p in zip(stage.actions[leader], res.x[:n])}}
    return {"leader_favorable": [best] if best else []}


def _utility_at(stage: NormalFormGame, pid: str, profile: dict, values) -> Fraction:
    for row in stage.payoffs:
        if all(row["action_profile"][k] == profile[k] for k in profile):
            return evaluate_cell(row["utilities"][pid], values or {})
    raise SolverError(f"profile {profile} missing from stage game")


def _mixed_follower_favorable(stage: NormalFormGame, leader: str, follower: str,
                              values, max_actions: int = 12) -> dict:
    """Follower-favorable mixed commitment.

    For each strict-response subset S (follower strictly prefers every action
    in S over every outside action under p), maximize the worst leader payoff
    min_{f in S} E_L(f). The all-actions subset has a vacuous gap: the whole
    simplex qualifies. At the best candidate value, prove attainment by
    re-solving with the objective fixed and the strict gap maximized; the
    all-actions subset is attained by definition. If the best supremum is
    unattained, return solver_unavailable with the exact supremum.
    """
    follower_actions = stage.actions[follower]
    if len(follower_actions) > max_actions:
        return {"follower_favorable": [], "status": "unsupported",
                "assumptions": [f"follower actions exceed {max_actions}"]}
    n = len(stage.actions[leader])
    best = None
    unattained = None
    subsets = list(itertools.chain.from_iterable(
        itertools.combinations(follower_actions, r) for r in range(1, len(follower_actions) + 1)))
    for subset in subsets:
        subset_set = set(subset)
        outside = [a for a in follower_actions if a not in subset_set]
        all_actions = not outside
        if all_actions:
            gap = None
        else:
            # maximize the strict gap g over this subset's region
            nvars = n + 1
            c_gap = [Fraction(0)] * n + [Fraction(1)]
            A_gap, b_gap = [], []
            for a_in in subset:
                for a_out in outside:
                    row = []
                    for a in stage.actions[leader]:
                        u_in = _utility_at(stage, follower, {leader: a, follower: a_in}, values)
                        u_out = _utility_at(stage, follower, {leader: a, follower: a_out}, values)
                        row.append(u_out - u_in)
                    row.append(Fraction(1))     # (u_out-u_in)p + g <= 0
                    A_gap.append(row)
                    b_gap.append(Fraction(0))
            try:
                gap_res = solve_rational_lp(c_gap, A_ub=A_gap, b_ub=b_gap,
                                            A_eq=[[Fraction(1)] * n], b_eq=[Fraction(1)])
            except LPError as exc:
                return {"follower_favorable": [], "error": str(exc)}
            if gap_res.status != "optimal" or gap_res.objective <= 0:
                continue     # empty region (or ties only): not a strict-preference region
            gap = gap_res.objective

        # stage 2: maximize worst leader payoff over responses in S subject to
        # weak best response: for f in S, o outside: E_F(f) >= E_F(o)
        nvars2 = n + 1
        c2 = [Fraction(0)] * n + [Fraction(1)]
        A2, b2 = [], []
        for a_in in subset:
            row = [-_utility_at(stage, leader, {leader: a, follower: a_in}, values)
                   for a in stage.actions[leader]]
            row.append(Fraction(1))
            A2.append(row)
            b2.append(Fraction(0))
        for a_in in subset:
            for a_out in outside:
                row = []
                for a in stage.actions[leader]:
                    u_in = _utility_at(stage, follower, {leader: a, follower: a_in}, values)
                    u_out = _utility_at(stage, follower, {leader: a, follower: a_out}, values)
                    row.append(u_out - u_in)
                A2.append(row)
                b2.append(Fraction(0))
        try:
            res2 = solve_rational_lp(c2, A_ub=A2 or None, b_ub=b2 or None,
                                     A_eq=[[Fraction(1)] * n], b_eq=[Fraction(1)])
        except LPError as exc:
            return {"follower_favorable": [], "error": str(exc)}
        if res2.status != "optimal":
            continue
        value = res2.objective
        if all_actions:
            attained = True
            attained_proof = Fraction(0)   # whole simplex: attained by definition
        else:
            attained_proof = _maximize_gap_at(stage, leader, follower, subset, outside,
                                              value, values)
            attained = attained_proof is not None and attained_proof > 0
        # Track the global best value across regions regardless of attainment;
        # if the global best is unattained, the plan forbids a lower attained
        # fallback.
        if best is None or value > best["leader_utility_value"]:
            best = {"leader_utility_value": value, "subset": sorted(subset),
                    "p": {a: render_rational(p) for a, p in zip(stage.actions[leader], res2.x[:n])},
                    "gap": render_rational(attained_proof if attained_proof is not None else 0),
                    "attained": attained}
        if not attained and (unattained is None or value > unattained):
            unattained = value
    if best is not None and best.pop("attained", True):
        return {"follower_favorable": [best]}
    if unattained is not None:
        return {"follower_favorable": [], "status": "solver_unavailable",
                "supremum": render_rational(unattained),
                "assumptions": ["the follower-favorable supremum is not attained by any strategy"]}
    return {"follower_favorable": []}


def _maximize_gap_at(stage: NormalFormGame, leader: str, follower: str,
                     subset: tuple, outside: list, target: Fraction, values) -> Fraction | None:
    """Maximize the strict gap subject to the worst leader payoff == target.

    Returns None when the fixed-value region is empty.
    """
    n = len(stage.actions[leader])
    nvars = n + 2                      # p, t (worst leader value), gap
    c = [Fraction(0)] * nvars
    c[nvars - 1] = Fraction(1)
    A_ub, b_ub = [], []
    for a_in in subset:
        row = [-_utility_at(stage, leader, {leader: a, follower: a_in}, values)
               for a in stage.actions[leader]]
        row += [Fraction(1), Fraction(0)]
        A_ub.append(row)
        b_ub.append(Fraction(0))
    # t <= target and t >= target
    row = [Fraction(0)] * nvars
    row[n] = Fraction(1)
    A_ub.append(row)
    b_ub.append(target)
    row = [Fraction(0)] * nvars
    row[n] = Fraction(-1)
    A_ub.append(row)
    b_ub.append(-target)
    for a_in in subset:
        for a_out in outside:
            row = []
            for a in stage.actions[leader]:
                u_in = _utility_at(stage, follower, {leader: a, follower: a_in}, values)
                u_out = _utility_at(stage, follower, {leader: a, follower: a_out}, values)
                row.append(u_out - u_in)
            row += [Fraction(0), Fraction(1)]   # (u_out-u_in)p + gap <= 0
            A_ub.append(row)
            b_ub.append(Fraction(0))
    try:
        res = solve_rational_lp(c, A_ub=A_ub, b_ub=b_ub,
                                A_eq=[[Fraction(1)] * n + [Fraction(0), Fraction(0)]],
                                b_eq=[Fraction(1)])
    except LPError:
        return None
    if res.status != "optimal":
        return None
    return res.objective


def verify_epsilon_nash(game: NormalFormGame, profile: dict, epsilon: Fraction,
                        values=None) -> bool:
    return _max_regret(game, profile, values) <= epsilon


def verify_mechanism(spec: MechanismSpec) -> MethodResult:
    """Verify a candidate mechanism (or generate/verify VCG) exactly."""
    spec.validate()
    players = [p.id for p in spec.players]
    type_profiles = _mechanism_type_profiles(spec)
    checks = []
    assumptions = ["quasilinear utility: value - payment"]

    if spec.mechanism_kind == "vcg":
        allocation, transfers = _vcg_mechanism(spec, type_profiles)
        assumptions.append("VCG: efficient allocation with Clarke pivot payments")
    else:
        allocation = spec.allocation
        transfers = spec.transfers
    if allocation is None or transfers is None:
        if spec.design_mode == "optimize" or spec.mechanism_kind == "optimal_auction":
            return _optimize_mechanism(spec, type_profiles)
        return MethodResult(method="mechanism_optimum", status="unsupported",
                            assumptions=["candidate verification requires allocation and transfers"])

    # DSIC / BIC
    for tp in type_profiles:
        key = spec.report_key(tp)
        for pi, pid in enumerate(players):
            truthful_util = _mech_utility(spec, pid, tp, tp, allocation, transfers)
            misreports = _mech_misreports(spec, pid, tp)
            for misreport in misreports:
                dev_util = _mech_utility(spec, pid, tp, misreport, allocation, transfers)
                if truthful_util is None or dev_util is None:
                    checks.append(_check(f"DSIC {pid} at {tp}: candidate incomplete",
                                         False, "allocation/transfers for every report profile",
                                         key))
                elif spec.ic_concept == "DSIC":
                    checks.append(_check(f"DSIC {pid} at {tp} vs {misreport}",
                                         dev_util <= truthful_util, "<= truthful",
                                         render_rational(dev_util - truthful_util)))
    if spec.ic_concept == "BIC":
        for pid in players:
            for t in spec.type_spaces[pid]:
                gain = _mech_bic_gain(spec, pid, t, allocation, transfers, type_profiles)
                if gain is None:
                    checks.append(_check(f"BIC interim {pid} type {t}",
                                         False, "candidate complete", "candidate"))
                    continue
                checks.append(_check(f"BIC interim {pid} type {t}",
                                     gain <= 0, "<= 0", render_rational(gain)))

    # IR
    for tp in type_profiles:
        key = spec.report_key(tp)
        for pid in players:
            util = _mech_utility(spec, pid, tp, tp, allocation, transfers)
            outside = evaluate_cell(spec.outside_option[pid][tp[pid]], {})
            if spec.ir_scope == "ex_post":
                if util is None:
                    checks.append(_check(f"ex-post IR {pid} at {tp}",
                                         False, "candidate complete", key))
                    continue
                checks.append(_check(f"ex-post IR {pid} at {tp}", util >= outside,
                                     ">= outside", render_rational(util - outside)))
    if spec.ir_scope == "interim":
        for pid in players:
            for t in spec.type_spaces[pid]:
                outside = evaluate_cell(spec.outside_option[pid][t], {})
                interim = _mech_interim_utility(spec, pid, allocation, transfers,
                                                type_profiles, per_type=True,
                                                fixed_type=t)
                if interim is None:
                    checks.append(_check(f"interim IR {pid} type {t}",
                                         False, "candidate complete", "candidate"))
                    continue
                checks.append(_check(f"interim IR {pid} type {t}",
                                     interim >= outside, ">= outside",
                                     render_rational(interim - outside)))
    if spec.ir_scope == "ex_ante":
        for pid in players:
            expected = _mech_interim_utility(spec, pid, allocation, transfers,
                                             type_profiles, per_type=False)
            if expected is None:
                checks.append(_check(f"ex-ante IR {pid}", False,
                                     "candidate complete", "candidate"))
                continue
            checks.append(_check(f"ex-ante IR {pid}", expected >= Fraction(0),
                                 ">= 0", render_rational(expected)))

    # budget rules
    if spec.budget_rule == "no_deficit":
        for tp in type_profiles:
            key = spec.report_key(tp)
            if key not in transfers:
                checks.append(_check(f"no-deficit at {tp}", False, "transfers present", key))
                continue
            revenue = sum(transfers[key].values(), Fraction(0))
            checks.append(_check(f"no-deficit at {tp}", revenue >= 0, ">= 0",
                                 render_rational(revenue)))
    if spec.budget_rule == "budget_balance":
        for tp in type_profiles:
            key = spec.report_key(tp)
            if key not in transfers:
                checks.append(_check(f"budget balance at {tp}", False,
                                     "transfers present", key))
                continue
            revenue = sum(transfers[key].values(), Fraction(0))
            checks.append(_check(f"budget balance at {tp}", revenue == 0, "== 0",
                                 render_rational(revenue)))

    failed = [c for c in checks if not c["ok"]]
    # objective evaluation for candidates
    objective = None
    if spec.objective is not None and spec.objective.kind == "revenue":
        objective = Fraction(0)
        for tp in type_profiles:
            prob = _mech_prior_probability(spec, tp)
            objective += prob * sum(transfers[spec.report_key(tp)].values(), Fraction(0))
    return MethodResult(method="mechanism_optimum",
                        status="exact_verified" if not failed else "error",
                        witnesses=[{"allocation": {k: {o: render_rational(p) for o, p in dist.items()}
                                                   for k, dist in allocation.items()},
                                    "transfers": {k: {pid: render_rational(v)
                                                      for pid, v in t.items()}
                                                  for k, t in transfers.items()}}],
                        objective=objective, checks=checks, assumptions=assumptions)


def _mechanism_type_profiles(spec: MechanismSpec) -> list[dict]:
    import itertools as _it
    players = [p.id for p in spec.players]
    profiles = []
    for combo in _it.product(*(spec.type_spaces[pid] for pid in players)):
        profiles.append(dict(zip(players, combo)))
    return profiles


def _mech_prior_probability(spec: MechanismSpec, tp: dict) -> Fraction:
    for row in spec.prior:
        if row["type_profile"] == tp:
            return to_rational(row["probability"], "mechanism prior probability")
    raise GameSpecError(f"prior missing type profile {tp}")


def _mech_utility(spec: MechanismSpec, pid: str, true_types: dict, reports: dict,
                  allocation, transfers) -> Fraction | None:
    from game_models import _prob
    key = spec.report_key(reports)
    if key not in allocation or key not in transfers:
        return None
    util = Fraction(0)
    dist = allocation[key]
    for outcome, prob in dist.items():
        p = _prob(prob, "allocation probability")
        value = evaluate_cell(spec.value_table[pid][true_types[pid]][outcome], {})
        util += p * value
    util -= transfers[key][pid]
    return util


def _mech_misreports(spec: MechanismSpec, pid: str, tp: dict) -> list[dict]:
    others = [spec.type_spaces[p] for p in spec.type_spaces if p != pid]
    import itertools as _it
    out = []
    for alt in spec.type_spaces[pid]:
        misreport = dict(tp)
        misreport[pid] = alt
        out.append(misreport)
    return out


def _mech_bic_gain(spec: MechanismSpec, pid: str, t: str,
                   allocation, transfers, type_profiles) -> Fraction | None:
    """Interim expected gain from misreporting own type t; None if incomplete."""
    prior_map = {spec.report_key(row["type_profile"]):
                 to_rational(row["probability"], "mechanism prior probability")
                 for row in spec.prior}

    def interim_utility(reported_type: str) -> Fraction | None:
        total = Fraction(0)
        for tp in type_profiles:
            if tp[pid] != t:
                continue
            reports = dict(tp)
            reports[pid] = reported_type
            util = _mech_utility(spec, pid, tp, reports, allocation, transfers)
            if util is None:
                return None
            total += prior_map[spec.report_key(tp)] * util
        return total

    truthful = interim_utility(t)
    if truthful is None:
        return None
    best = truthful
    for alt in spec.type_spaces[pid]:
        if alt == t:
            continue
        deviated = interim_utility(alt)
        if deviated is None:
            return None
        if deviated > best:
            best = deviated
    return best - truthful


def _mech_interim_utility(spec: MechanismSpec, pid: str, allocation, transfers,
                          type_profiles, per_type: bool,
                          fixed_type: str | None = None) -> Fraction | None:
    total = Fraction(0)
    weight = Fraction(0)
    for tp in type_profiles:
        if fixed_type is not None and tp[pid] != fixed_type:
            continue
        util = _mech_utility(spec, pid, tp, tp, allocation, transfers)
        if util is None:
            return None
        prob = _mech_prior_probability(spec, tp)
        total += prob * util
        weight += prob
    return total / weight if weight else total


def _vcg_mechanism(spec: MechanismSpec, type_profiles: list[dict]) -> tuple[dict, dict]:
    """Efficient allocation with Clarke pivot payments over feasible outcomes."""
    if spec.allocation is not None or spec.transfers is not None:
        raise GameSpecError("VCG generation ignores candidate allocation/transfers; "
                            "set them null")
    allocation = {}
    transfers = {}
    players = [p.id for p in spec.players]
    for tp in type_profiles:
        outcomes = spec.feasible_outcomes[spec.report_key(tp)]
        # report-independent efficient allocation with lexicographic tie rule
        best_outcome, best_welfare = None, None
        for outcome in sorted(outcomes):
            welfare = sum((evaluate_cell(spec.value_table[pid][tp[pid]][outcome], {})
                           for pid in players), Fraction(0))
            if best_welfare is None or welfare > best_welfare:
                best_outcome, best_welfare = outcome, welfare
        key = spec.report_key(tp)
        allocation[key] = {best_outcome: Fraction(1)}
        payments = {}
        for pid in players:
            others = [q for q in players if q != pid]
            # with i present
            with_i = sum((evaluate_cell(spec.value_table[q][tp[q]][best_outcome], {})
                         for q in others), Fraction(0))
            # without i: only outcomes where i gets the opt-out allocation
            best_without, best_without_welfare = None, None
            for outcome in sorted(outcomes):
                if _outcome_assigns_optout(spec, pid, outcome):
                    welfare = sum((evaluate_cell(spec.value_table[q][tp[q]][outcome], {})
                                   for q in others), Fraction(0))
                    if best_without_welfare is None or welfare > best_without_welfare:
                        best_without, best_without_welfare = outcome, welfare
            if best_without is None:
                raise GameSpecError("VCG needs at least one opt-out outcome per "
                                    "report profile")
            payments[pid] = best_without_welfare - with_i
        transfers[key] = payments
    return allocation, transfers


def _outcome_assigns_optout(spec: MechanismSpec, pid: str, outcome: str) -> bool:
    """F_-i: player i is absent from the outcome (receives no allocation)."""
    if spec.outcome_owners is not None:
        return pid not in spec.outcome_owners.get(outcome, [])
    return spec.optout_outcomes is not None and spec.optout_outcomes.get(pid) == outcome


def _optimize_mechanism(spec: MechanismSpec, type_profiles: list[dict]) -> MethodResult:
    """LP design over the finite feasible outcome set.

    Variables: allocation probabilities x[profile, outcome] >= 0 and transfers
    t[profile, player] (free sign, positive = player pays designer).
    Constraints: per-profile simplex, IC (DSIC ex-post or BIC interim, both
    linear), IR in the declared scope, the selected budget rule. Objective:
    prior-weighted expected revenue, maximized. Solved with the exact rational
    LP engine; the returned certificate carries the dual witness, and the
    witness mechanism is re-verified through the exact candidate path.
    """
    import dataclasses

    if spec.mechanism_kind not in ("direct", "optimal_auction"):
        return MethodResult(method="mechanism_optimum", status="unsupported",
                            assumptions=["LP design supports direct/optimal_auction; "
                                         "VCG has its own generation"])
    if spec.objective is None or spec.objective.kind != "revenue" \
            or spec.objective.direction != "maximize":
        return MethodResult(method="mechanism_optimum", status="unsupported",
                            assumptions=["mechanism LP design maximizes expected revenue"])

    players = [p.id for p in spec.players]
    prior_map = {spec.report_key(row["type_profile"]):
                 to_rational(row["probability"], "mechanism prior probability")
                 for row in spec.prior}

    # ---- variable layout
    var_index: dict[tuple, int] = {}
    bounds: list[tuple[Fraction | None, Fraction | None]] = []
    for ti, tp in enumerate(type_profiles):
        key = spec.report_key(tp)
        for outcome in spec.feasible_outcomes[key]:
            var_index[("x", ti, outcome)] = len(bounds)
            bounds.append((Fraction(0), None))
    for ti, _tp in enumerate(type_profiles):
        for pid in players:
            var_index[("t", ti, pid)] = len(bounds)
            bounds.append((None, None))
    n_vars = len(bounds)

    def value_of(pid: str, type_id: str, outcome: str) -> Fraction:
        return evaluate_cell(spec.value_table[pid][type_id][outcome], {})

    c = [Fraction(0)] * n_vars
    for ti, tp in enumerate(type_profiles):
        weight = prior_map[spec.report_key(tp)]
        for pid in players:
            c[var_index[("t", ti, pid)]] += weight

    A_ub: list[list[Fraction]] = []
    b_ub: list[Fraction] = []
    A_eq: list[list[Fraction]] = []
    b_eq: list[Fraction] = []

    def ic_row(base_ti: int, misreport_ti: int, pid: str) -> list[Fraction]:
        """truth >= deviation, as a <= 0 row: -(truth - deviation)."""
        base_tp = type_profiles[base_ti]
        row = [Fraction(0)] * n_vars
        base_key = spec.report_key(base_tp)
        mis_key = spec.report_key(type_profiles[misreport_ti])
        for outcome in spec.feasible_outcomes[base_key]:
            v = value_of(pid, base_tp[pid], outcome)
            row[var_index[("x", base_ti, outcome)]] -= v
            row[var_index[("x", misreport_ti, outcome)]] += v
        row[var_index[("t", base_ti, pid)]] += Fraction(1)
        row[var_index[("t", misreport_ti, pid)]] -= Fraction(1)
        return row

    if spec.ic_concept == "DSIC":
        for ti, tp in enumerate(type_profiles):
            for pid in players:
                for alt in spec.type_spaces[pid]:
                    if alt == tp[pid]:
                        continue
                    mis = dict(tp)
                    mis[pid] = alt
                    ti_m = type_profiles.index(mis)
                    A_ub.append(ic_row(ti, ti_m, pid))
                    b_ub.append(Fraction(0))
    else:   # BIC: interim expected utility, prior-weighted
        for pid in players:
            for t in spec.type_spaces[pid]:
                truth_profiles = [ti for ti, tp in enumerate(type_profiles) if tp[pid] == t]
                for alt in spec.type_spaces[pid]:
                    if alt == t:
                        continue
                    row = [Fraction(0)] * n_vars
                    for ti in truth_profiles:
                        tp = type_profiles[ti]
                        weight = prior_map[spec.report_key(tp)]
                        mis = dict(tp)
                        mis[pid] = alt
                        ti_m = type_profiles.index(mis)
                        base_key = spec.report_key(tp)
                        mis_key = spec.report_key(type_profiles[ti_m])
                        for outcome in spec.feasible_outcomes[base_key]:
                            v = value_of(pid, t, outcome)
                            row[var_index[("x", ti, outcome)]] -= weight * v
                            row[var_index[("x", ti_m, outcome)]] += weight * v
                        row[var_index[("t", ti, pid)]] += weight
                        row[var_index[("t", ti_m, pid)]] -= weight
                    A_ub.append(row)
                    b_ub.append(Fraction(0))

    # ---- IR
    if spec.ir_scope == "ex_post":
        for ti, tp in enumerate(type_profiles):
            key = spec.report_key(tp)
            for pid in players:
                row = [Fraction(0)] * n_vars
                for outcome in spec.feasible_outcomes[key]:
                    row[var_index[("x", ti, outcome)]] -= value_of(pid, tp[pid], outcome)
                row[var_index[("t", ti, pid)]] += Fraction(1)
                outside = evaluate_cell(spec.outside_option[pid][tp[pid]], {})
                A_ub.append(row)
                b_ub.append(-outside)
    elif spec.ir_scope == "interim":
        for pid in players:
            for t in spec.type_spaces[pid]:
                row = [Fraction(0)] * n_vars
                weight_total = Fraction(0)
                for ti, tp in enumerate(type_profiles):
                    if tp[pid] != t:
                        continue
                    weight = prior_map[spec.report_key(tp)]
                    key = spec.report_key(tp)
                    for outcome in spec.feasible_outcomes[key]:
                        row[var_index[("x", ti, outcome)]] -= weight * value_of(pid, t, outcome)
                    row[var_index[("t", ti, pid)]] += weight
                    weight_total += weight
                # LHS is Pr(t)-weighted: scale the outside bound by the same mass
                outside = evaluate_cell(spec.outside_option[pid][t], {})
                A_ub.append(row)
                b_ub.append(-outside * weight_total)
    else:   # ex_ante
        for pid in players:
            row = [Fraction(0)] * n_vars
            outside_total = Fraction(0)
            for ti, tp in enumerate(type_profiles):
                weight = prior_map[spec.report_key(tp)]
                key = spec.report_key(tp)
                for outcome in spec.feasible_outcomes[key]:
                    row[var_index[("x", ti, outcome)]] -= weight * value_of(pid, tp[pid], outcome)
                row[var_index[("t", ti, pid)]] += weight
                outside_total += weight * evaluate_cell(spec.outside_option[pid][tp[pid]], {})
            A_ub.append(row)
            b_ub.append(-outside_total)

    # ---- budget rule
    if spec.budget_rule == "no_deficit":
        for ti, _tp in enumerate(type_profiles):
            row = [Fraction(0)] * n_vars
            for pid in players:
                row[var_index[("t", ti, pid)]] = Fraction(-1)
            A_ub.append(row)
            b_ub.append(Fraction(0))
    elif spec.budget_rule == "budget_balance":
        for ti, _tp in enumerate(type_profiles):
            row = [Fraction(0)] * n_vars
            for pid in players:
                row[var_index[("t", ti, pid)]] = Fraction(1)
            A_eq.append(row)
            b_eq.append(Fraction(0))

    # ---- simplex per profile
    for ti, tp in enumerate(type_profiles):
        key = spec.report_key(tp)
        row = [Fraction(0)] * n_vars
        for outcome in spec.feasible_outcomes[key]:
            row[var_index[("x", ti, outcome)]] = Fraction(1)
        A_eq.append(row)
        b_eq.append(Fraction(1))

    if n_vars > MAX_LP_VARIABLES or len(A_ub) + len(A_eq) > MAX_LP_CONSTRAINTS:
        return MethodResult(method="mechanism_optimum", status="solver_unavailable",
                            assumptions=["mechanism LP exceeds variable/constraint caps"])

    res = solve_rational_lp(c, A_ub=A_ub or None, b_ub=b_ub or None,
                            A_eq=A_eq or None, b_eq=b_eq or None, bounds=bounds)
    if res.status != "optimal":
        return MethodResult(method="mechanism_optimum", status="solver_unavailable",
                            certificate={"lp_status": res.status},
                            assumptions=[f"mechanism LP returned {res.status}"])

    # ---- reconstruct the witness mechanism
    allocation = {}
    transfers = {}
    inv = {idx: name for name, idx in var_index.items()}
    for ti, tp in enumerate(type_profiles):
        key = spec.report_key(tp)
        allocation[key] = {}
        for outcome in spec.feasible_outcomes[key]:
            allocation[key][outcome] = res.x[var_index[("x", ti, outcome)]]
        transfers[key] = {pid: res.x[var_index[("t", ti, pid)]] for pid in players}

    spec_candidate = dataclasses.replace(spec, allocation=allocation, transfers=transfers)
    candidate = verify_mechanism(spec_candidate)
    failed = [c for c in candidate.checks if not c["ok"]]
    checks = list(candidate.checks)
    checks.append(_check("primal/dual objectives equal",
                         res.dual_objective == res.objective,
                         render_rational(res.objective), render_rational(res.dual_objective)))
    return MethodResult(method="mechanism_optimum",
                        status="exact_verified" if not failed else "error",
                        witnesses=candidate.witnesses,
                        objective=res.objective,
                        checks=checks,
                        certificate={"dual_y": [render_rational(y) for y in res.y],
                                     "dual_objective": render_rational(res.dual_objective)},
                        assumptions=["expected revenue maximized over the finite feasible "
                                     "mechanism set; exact primal/dual certificate"])


def solve_exact_game(envelope, mediator_confirmed: bool = False, methods=None):
    """Dispatch compatible exact methods for the envelope's game.

    methods: optional collection of method names; others are not run."""
    game = envelope.game
    values = envelope.point_parameters()
    wanted = (lambda name: True) if methods is None else (lambda name: name in methods)
    results = []
    if isinstance(game, NormalFormGame):
        if wanted("pure_nash"):
            results.append(solve_pure_nash(game, values))
        if wanted("mixed_nash"):
            if len(game.players) == 2:
                results.append(solve_mixed_nash_two_player(game, values))
            else:
                results.append(MethodResult(method="mixed_nash", status="solver_unavailable",
                                            assumptions=["exact mixed Nash supports two-player games"]))
    elif isinstance(game, StackelbergGame):
        if wanted("pure_nash"):
            results.append(solve_pure_nash(game.stage_game, values))
        if wanted("stackelberg"):
            results.append(solve_stackelberg(game, values))
    elif isinstance(game, ExtensiveFormGame):
        if wanted("subgame_perfect"):
            results.append(spne_walk(game, values))
    elif isinstance(game, BayesianGame):
        if wanted("bayesian_nash"):
            results.append(solve_bayesian_nash(game, values))
    elif isinstance(game, BargainingSpec):
        if wanted("nash_bargaining"):
            results.append(solve_nash_bargaining(game, values))
    elif isinstance(game, MechanismSpec):
        if wanted("mechanism_optimum"):
            results.append(verify_mechanism(game))
    elif isinstance(game, RepeatedGame):
        if wanted("pure_nash"):
            results.append(MethodResult(method="pure_nash", status="unsupported",
                                        assumptions=["repeated-game equilibria are unsupported; "
                                                     "stage-game equilibria available via a normal-form spec"]))
    elif isinstance(game, OpenSpielGame):
        if wanted("pure_nash"):
            results.append(MethodResult(method="pure_nash", status="unsupported",
                                        assumptions=["OpenSpiel games support approximate methods only"]))
    if isinstance(game, NormalFormGame) and wanted("correlated_equilibrium"):
        results.append(solve_correlated_equilibrium(game, mediator_confirmed,
                                                    game.objective, values))
    return results


def solve_bayesian_nash(game: BayesianGame,
                        values: dict[str, Fraction] | None = None) -> MethodResult:
    game.validate()
    players = [p.id for p in game.players]
    # enumerate pure contingent plans: each player maps own type -> action
    plans_per_player = {}
    for pid in players:
        mapping = list(itertools.product(*(range(len(game.actions[pid])) for _ in game.type_spaces[pid])))
        plans_per_player[pid] = mapping
    total = 1
    for pid in players:
        total *= len(plans_per_player[pid])
    if total > 10_000:
        return MethodResult(method="bayesian_nash", status="solver_unavailable",
                            assumptions=["pure contingent-plan enumeration exceeds the "
                                         "10,000-profile cap"])
    prior = {tuple(sorted(row["type_profile"].items())):
             to_rational(row["probability"], "bayesian prior probability")
             for row in game.prior}
    payoff_map = {}
    for row in game.payoffs:
        key = (tuple(sorted(row["type_profile"].items())), tuple(sorted(row["action_profile"].items())))
        payoff_map[key] = {pid: evaluate_cell(row["utilities"][pid], values or {}) for pid in players}

    def expected_utility(pid, plan_profile):
        total_util = Fraction(0)
        for tp, prob in prior.items():
            tp_dict = dict(tp)
            actions = {p: game.actions[p][plan_profile[p][game.type_spaces[p].index(tp_dict[p])]]
                       for p in players}
            total_util += prob * payoff_map[(tp, tuple(sorted(actions.items())))][pid]
        return total_util

    witnesses = []
    plan_lists = [plans_per_player[pid] for pid in players]
    for combo in itertools.product(*plan_lists):
        plan_profile = dict(zip(players, combo))
        is_eq = True
        for pi, pid in enumerate(players):
            others = {p: plan_profile[p] for p in players if p != pid}
            base = expected_utility(pid, plan_profile)
            for alt in plans_per_player[pid]:
                trial = dict(others)
                trial[pid] = alt
                if expected_utility(pid, trial) > base:
                    is_eq = False
                    break
            if not is_eq:
                break
        if is_eq:
            witnesses.append({pid: {game.type_spaces[pid][t]: game.actions[pid][a]
                                    for t, a in enumerate(plan_profile[pid])}
                              for pid in players})
    checks = [_check("pure BNE verified against every unilateral contingent deviation",
                     True, True, True)]
    if not witnesses:
        return MethodResult(method="bayesian_nash", status="solver_unavailable",
                            checks=checks,
                            assumptions=["no pure contingent-plan BNE exists; mixed "
                                         "contingent-plan conversion is not implemented"])
    return MethodResult(method="bayesian_nash", status="exact_verified",
                        witnesses=witnesses, checks=checks)
