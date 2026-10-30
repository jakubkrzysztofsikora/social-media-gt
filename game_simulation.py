"""Seeded game simulation: episode play and OpenSpiel CFR/MCCFR.

All results are approximate evidence, never equilibrium or convergence claims.
"""
from __future__ import annotations

import math
import random
from fractions import Fraction

from game_models import (GameSpecError, NormalFormGame, OpenSpielGame, RepeatedGame,
                         evaluate_cell, render_rational)

ALGORITHMS = ("simulation", "cfr", "mccfr")
NAMED_POLICIES = ("always_cooperate", "always_defect", "tit_for_tat", "grim_trigger")
MAX_EPISODES = 100_000
MAX_ITERATIONS = 100_000


def simulate_game(spec, algorithm, policies, episodes, seed, confirmed_by_user,
                  iterations=None):
    if not confirmed_by_user:
        return {"status": "needs_confirmation"}
    if algorithm not in ALGORITHMS:
        raise GameSpecError(f"simulation algorithm {algorithm!r} unsupported")
    if not isinstance(episodes, int) or episodes <= 0 or episodes > MAX_EPISODES:
        raise GameSpecError(f"episodes must be a positive integer <= {MAX_EPISODES}")
    if not isinstance(seed, int) or seed < 0:
        raise GameSpecError("seed must be a nonnegative integer")
    if algorithm in ("cfr", "mccfr"):
        if iterations is None or not isinstance(iterations, int) or iterations <= 0 \
                or iterations > MAX_ITERATIONS:
            raise GameSpecError(f"CFR/MCCFR require positive iterations <= {MAX_ITERATIONS}")
        return _cfr(spec, algorithm, episodes, seed, iterations)
    return _episode_simulation(spec, policies, episodes, seed)


def _cfr(spec, algorithm, episodes, seed, iterations):
    import pyspiel
    from open_spiel.python.algorithms import exploitability
    game_spec = spec.game
    if not isinstance(game_spec, OpenSpielGame):
        raise GameSpecError("CFR/MCCFR requires an OpenSpielGame spec")
    if game_spec.game_id not in OpenSpielGame.CFR_GAMES:
        raise GameSpecError("CFR/MCCFR supports kuhn_poker and leduc_poker only")
    game = pyspiel.load_game(game_spec.game_id,
                             {k: str(v) for k, v in (game_spec.parameters or {}).items()})
    if algorithm == "cfr":
        from open_spiel.python.algorithms import cfr
        solver = cfr.CFRSolver(game)
    else:
        from open_spiel.python.algorithms import mccfr
        solver = mccfr.MCCFRSolver(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    policy = solver.average_policy()
    nash_conv = float(exploitability.nash_conv(game, policy))
    interval = _evaluate_policy_episodes(game, policy, episodes, seed)
    return {"status": "approximate", "algorithm": algorithm, "iterations": iterations,
            "episodes": episodes, "seed": seed, "nash_conv": nash_conv,
            "policy_interval": interval,
            "assumptions": ["supported NashConv is approximate evidence, not a proof"]}


def _evaluate_policy_episodes(game, policy, episodes, seed) -> dict:
    import numpy as np
    rng = np.random.default_rng(seed)
    totals: dict[int, list[float]] = {}
    for _ in range(episodes):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                action = actions[int(rng.choice(len(actions), p=probs))]
                state.apply_action(action)
                continue
            pid = state.current_player()
            legal = set(state.legal_actions(pid))
            probs = policy.action_probabilities(state, pid)
            actions = [a for a in sorted(probs) if a in legal]
            weights = [probs[a] for a in actions]
            total = sum(weights)
            if total <= 0:
                action = actions[int(rng.integers(len(actions)))]
            else:
                action = actions[int(rng.choice(len(actions),
                                                p=[w / total for w in weights]))]
            state.apply_action(action)
        for pid, ret in enumerate(state.returns()):
            totals.setdefault(pid, []).append(float(ret))
    out = {}
    for pid, values in totals.items():
        arr = np.array(values)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        half = 1.96 * std / math.sqrt(len(arr)) if len(arr) > 1 and std > 0 else 0.0
        out[str(pid)] = {"mean": mean, "variance": std * std,
                         "interval": [mean - half, mean + half]}
    return out


def _episode_simulation(spec, policies, episodes, seed):
    """Seeded episode play of a RepeatedGame with named public policies."""
    game = spec.game
    if not isinstance(game, RepeatedGame):
        raise GameSpecError("simulation with named policies requires a RepeatedGame")
    game.validate()
    stage = game.stage_game
    players = [p.id for p in stage.players]
    if set(policies.keys()) != set(players):
        raise GameSpecError("policies must cover every repeated-game player")
    parsed = {}
    cooperate, defect = {}, {}
    for pid in players:
        entry = policies[pid]
        if not isinstance(entry, dict) or "name" not in entry:
            raise GameSpecError(f"policy for {pid} must be "
                                "{name, cooperate_action_id, defect_action_id}")
        name = entry["name"]
        if name not in NAMED_POLICIES:
            raise GameSpecError(f"policy name {name!r} unsupported")
        parsed[pid] = name
        cooperate[pid] = entry["cooperate_action_id"]
        defect[pid] = entry["defect_action_id"]
        if cooperate[pid] not in stage.actions[pid] or defect[pid] not in stage.actions[pid]:
            raise GameSpecError(f"policy action ids for {pid} must be stage actions")
        if cooperate[pid] == defect[pid]:
            raise GameSpecError(f"cooperate and defect action ids for {pid} must differ")
    rng = random.Random(seed)
    totals = {pid: [] for pid in players}
    for _ in range(episodes):
        payoffs = {pid: Fraction(0) for pid in players}
        triggered = False
        previous = None
        for t in range(game.horizon):
            # grim players defect forever once any public defection was observed
            if previous is not None and any(previous[p] == defect[p] for p in players):
                triggered = True
            actions = {}
            for pid in players:
                if triggered and parsed[pid] == "grim_trigger":
                    actions[pid] = defect[pid]
                    continue
                name = parsed[pid]
                if name == "always_cooperate":
                    actions[pid] = cooperate[pid]
                elif name == "always_defect":
                    actions[pid] = defect[pid]
                elif name == "tit_for_tat":
                    if previous is None:
                        actions[pid] = cooperate[pid]
                    else:
                        opp = other(pid, players)
                        # class of opponent's previous public action
                        opp_played_coop = previous[opp] == cooperate[opp]
                        actions[pid] = cooperate[pid] if opp_played_coop else defect[pid]
                else:   # grim_trigger
                    actions[pid] = cooperate[pid]
            for row in stage.payoffs:
                if all(row["action_profile"][k] == actions[k] for k in actions):
                    for pid in players:
                        payoffs[pid] += (game.discount_factor ** t) * \
                            evaluate_cell(row["utilities"][pid], {})
                    break
            previous = dict(actions)
        for pid in players:
            totals[pid].append(payoffs[pid])
    out = {}
    for pid, values in totals.items():
        n = len(values)
        mean = sum(values, Fraction(0)) / n
        variance = (sum(((v - mean) ** 2 for v in values), Fraction(0)) / (n - 1)) if n > 1 \
            else Fraction(0)
        out[pid] = {"mean": render_rational(mean), "variance": render_rational(variance)}
    return {"status": "approximate", "algorithm": "simulation", "episodes": episodes,
            "seed": seed, "per_player": out,
            "assumptions": ["finite-horizon simulated play; no equilibrium claim"]}


def other(pid: str, players: list[str]) -> str:
    return players[0] if pid == players[1] else players[1]
