"""Plain-language views of a game draft and its solver result.

Both renderers are deterministic over the same objects the solver consumes:
the draft summary compiles the spec JSON with the solver's own builder, so
what the user confirms is exactly what gets solved.
"""
import ast
import json
from collections.abc import Mapping
from fractions import Fraction

KIND_TITLES = {
    "NormalFormGame": "Everyone chooses at the same time",
    "ExtensiveFormGame": "Players move in turns",
    "BayesianGame": "Players have private information",
    "StackelbergGame": "One player commits first, the other responds",
    "BargainingSpec": "Two sides negotiate a deal",
    "MechanismSpec": "Designing the rules of the game",
    "RepeatedGame": "The same game is repeated",
    "OpenSpielGame": "A standard benchmark game",
}

METHOD_TITLES = {
    "pure_nash": "Outcomes where nobody gains by switching alone",
    "mixed_nash": "Stable mixes where each side randomises",
    "bayesian_nash": "Stable choices given private information",
    "subgame_perfect": "Best plan when moves come in turns",
    "correlated_equilibrium": "Coordinated play with a shared signal",
    "nash_bargaining": "The fair deal (Nash bargaining)",
    "stackelberg": "What happens when one side commits first",
    "mechanism_optimum": "The best rule design",
    "epsilon_nash": "Nearly stable outcomes (approximate)",
}

STATUS_TITLES = {
    "exact_verified": "Checked exactly",
    "approximate": "Approximate result",
    "solver_unavailable": "Could not be solved here",
    "unsupported": "Not available for this kind of game",
    "no_agreement": "No deal is better for both sides",
    "error": "Something went wrong",
}

MAX_SHOWN = 5


def _cell(value) -> str:
    return "a formula" if isinstance(value, Mapping) else str(value)


def _names(game) -> dict:
    return {p.id: str(p.id if p.name is None or p.name == "" else p.name)
            for p in getattr(game, "players", [])}


def _payoff_table(game, names: dict) -> str:
    ids = [p.id for p in game.players]
    if len(ids) == 2:
        first, second = ids
        lookup = {(r["action_profile"][first], r["action_profile"][second]): r["utilities"]
                  for r in game.payoffs}
        cols = game.actions[second]
        lines = [f"| {names[first]} \\ {names[second]} | " +
                 " | ".join(map(str, cols)) + " |",
                 "|---" * (len(cols) + 1) + "|"]
        for row in game.actions[first]:
            cells = []
            for col in cols:
                u = lookup[(row, col)]
                cells.append(f"{names[first]} {_cell(u[first])}, {names[second]} {_cell(u[second])}")
            lines.append(f"| {row} | " + " | ".join(cells) + " |")
        return "\n".join(lines)
    lines = []
    for r in game.payoffs[:20]:
        who = ", ".join(f"{names[p]}: {r['action_profile'][p]}" for p in ids)
        got = ", ".join(f"{names[p]} {_cell(r['utilities'][p])}" for p in ids)
        lines.append(f"- {who} → {got}")
    if len(game.payoffs) > 20:
        lines.append(f"- … and {len(game.payoffs) - 20} more combinations")
    return "\n".join(lines)


def describe_draft(spec_json: str) -> tuple[str, bool]:
    """Return (markdown, ok). ok is False when the spec cannot be solved as written."""
    import advisor
    from game_models import GameSpecError

    if not spec_json or not str(spec_json).strip():
        return "", False
    try:
        spec = json.loads(spec_json)
    except ValueError as exc:
        return f"**The details are not valid yet:** {exc}", False
    try:
        envelope = advisor._build_envelope_from_payload(spec)
        game = envelope.game
    except GameSpecError as exc:
        return f"**These details can't be used yet:** {exc}", False
    except (KeyError, TypeError, AttributeError, IndexError):
        return "**These details can't be used yet:** required game fields are missing or malformed", False
    names = _names(game)
    lines = ["### Here is how I understood it", "",
             f"**Type of situation:** {KIND_TITLES.get(type(game).__name__, type(game).__name__)}",
             ""]
    if names:
        lines += ["**Who is involved:** " + ", ".join(names.values()), ""]
    if type(game).__name__ == "NormalFormGame":
        lines += ["**Their options:**", ""]
        lines += [f"- {names[p]}: " + ", ".join(map(str, opts))
                  for p, opts in game.actions.items()]
        lines += ["", "**What each side gets** (higher is better for that side):", "",
                  _payoff_table(game, names)]
    elif type(game).__name__ == "BargainingSpec":
        lines += ["**Possible deals** (what each side gets):", ""]
        lines += ["- " + ", ".join(f"{names.get(p, p)} {_cell(v)}" for p, v in pt.items())
                  for pt in game.feasible_points]
        lines += ["", "**If there is no deal:** " + ", ".join(
            f"{names.get(p, p)} {_cell(v)}" for p, v in game.disagreement.items())]
        lines += ["", f"**Randomization:** {game.randomization}",
                  f"**Binding commitment rights:** {envelope.commitment_rights}"]
        if game.randomization == "allowed":
            lines.append("Randomized deals need a binding contract.")
            lines.append("")
    else:
        lines.append("The full details are in *Edit details* below.")
    if envelope.template_id == "p3_heterogeneous":
        lines += ["", f"**Scope:** {advisor.P3_TEMPLATE_SCOPE}", ""]
    lines.append("\nIf this is right, press the button below. If not, change your description "
                 "above or edit the details.")
    return "\n".join(lines), True


def _percent(text) -> str:
    probability = Fraction(str(text))
    pct = probability * 100
    if pct.denominator == 1:
        return f"{pct.numerator}%"
    rendered_pct = f"{float(pct):.12g}"
    if float(pct) >= 100 or rendered_pct == "100":
        return str(probability)
    return f"{probability} ({rendered_pct}%)"


def _as_fraction(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Fraction)):
        return None
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def _distribution(value, actions=None):
    if not isinstance(value, Mapping) or not value:
        return None
    probabilities = {}
    for action, raw in value.items():
        probability = _as_fraction(raw)
        if probability is None or not 0 <= probability <= 1:
            return None
        probabilities[action] = probability
    if sum(probabilities.values(), Fraction(0)) != 1:
        return None
    if actions is not None and not set(probabilities).issubset(actions):
        return None
    return probabilities


def _distribution_text(value, actions=None):
    probabilities = _distribution(value, actions)
    if probabilities is None:
        return None
    return ", ".join(f"{_percent(probability)} {action}"
                     for action, probability in probabilities.items() if probability)


def _label(key, names):
    return names.get(key, str(key).replace("_", " ").capitalize())


def _plain_value(value):
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, default=str, ensure_ascii=False)
    return str(value)


def _render_plain_witness(witness, names):
    return ", ".join(f"{_label(key, names)}: {_plain_value(value)}"
                     for key, value in witness.items())


def _render_mixed_witness(witness, names, game):
    parts = []
    for player, mix in witness.items():
        if player not in names or not isinstance(mix, Mapping):
            return None
        actions = None
        if type(game).__name__ == "NormalFormGame":
            actions = game.actions.get(player, [])
        rendered = _distribution_text(mix, actions)
        if rendered is None:
            return None
        parts.append(f"{names[player]}: {rendered}")
    return ", ".join(parts) if parts else None


def _render_bayesian_witness(witness, names, game):
    parts = []
    for player, plan in witness.items():
        if player not in names or not isinstance(plan, Mapping):
            return None
        if type(game).__name__ == "BayesianGame":
            if not set(plan).issubset(game.type_spaces.get(player, [])):
                return None
        if any(isinstance(action, (Mapping, list, tuple)) for action in plan.values()):
            return None
        choices = ", ".join(f"type {type_id} → {action}"
                            for type_id, action in plan.items())
        parts.append(f"{names[player]}: {choices}")
    return "; ".join(parts) if parts else None


def _render_subgame_witness(witness, names):
    parts = []
    continuation = witness.get("continuation")
    if isinstance(continuation, Mapping):
        choices = ", ".join(f"{node} → {action}" for node, action in continuation.items())
        parts.append(f"Continuation: {choices}")
    values = witness.get("value")
    if isinstance(values, Mapping):
        payoffs = ", ".join(f"{_label(player, names)} {value}"
                            for player, value in values.items())
        parts.append(f"Payoffs: {payoffs}")
    behavior = witness.get("behavior")
    if isinstance(behavior, Mapping):
        choices = ", ".join(
            f"{node}: {' or '.join(map(str, actions)) if isinstance(actions, (list, tuple)) else actions}"
            for node, actions in behavior.items())
        parts.append(f"Best choices: {choices}")
    return "; ".join(parts) if parts else None


def _render_correlated_witness(witness, names, game):
    outcomes = []
    total = Fraction(0)
    player_ids = set(names)
    for raw_profile, raw_probability in witness.items():
        if not isinstance(raw_profile, str):
            return None
        try:
            profile = ast.literal_eval(raw_profile)
        except (ValueError, SyntaxError):
            return None
        if not isinstance(profile, Mapping) or set(profile) != player_ids:
            return None
        if type(game).__name__ == "NormalFormGame":
            if any(action not in game.actions.get(player, []) for player, action in profile.items()):
                return None
        probability = _as_fraction(raw_probability)
        if probability is None or not 0 <= probability <= 1:
            return None
        total += probability
        recommendation = ", ".join(f"{names[player]} {action}"
                                   for player, action in profile.items())
        outcomes.append(f"{recommendation} ({_percent(probability)})")
    if total != 1 or not outcomes:
        return None
    return "Joint recommendations: " + "; ".join(outcomes)


def _render_bargaining_witness(witness, names):
    parts = [f"{names[player]} {value}" for player, value in witness.items()
             if player in names and not isinstance(value, (Mapping, list, tuple))]
    if witness.get("nash_product") is not None:
        parts.append(f"Nash product {witness['nash_product']}")
    if witness.get("mixture"):
        parts.append(f"Randomized deal: {witness['mixture']}")
    return "; ".join(parts) if parts else None


def _render_stackelberg_witness(witness, names, game):
    parts = []
    for key, label in (("leader_favorable", "Leader-favorable"),
                       ("follower_favorable", "Follower-favorable")):
        entries = witness.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            outcome = []
            leader_actions = None
            if type(game).__name__ == "StackelbergGame":
                leader_actions = game.stage_game.actions.get(game.leader_id, [])
            if isinstance(entry.get("p"), Mapping):
                mix = _distribution_text(entry["p"], leader_actions)
                if mix is None:
                    return None
                outcome.append(f"leader mix {mix}")
            elif entry.get("leader") is not None:
                outcome.append(str(entry["leader"]))
            follower = entry.get("follower", entry.get("response"))
            if follower is not None:
                if outcome:
                    outcome[-1] = f"{outcome[-1]} then {follower}"
                else:
                    outcome.append(str(follower))
            if entry.get("subset"):
                outcome.append("best responses " + ", ".join(map(str, entry["subset"])))
            utility = entry.get("leader_utility", entry.get("leader_utility_value"))
            if utility is not None:
                outcome.append(f"utility {utility}")
            if entry.get("gap") is not None:
                outcome.append(f"strict gap {entry['gap']}")
            if outcome:
                parts.append(f"{label}: {', '.join(outcome)}")
    if witness.get("supremum") is not None:
        parts.append(f"Follower-favorable supremum: {witness['supremum']}")
    return "; ".join(parts) if parts else None


def _render_witness(witness, names: dict, game, method: str | None) -> str:
    if not isinstance(witness, Mapping):
        return str(witness)
    if method in ("mixed_nash", "epsilon_nash"):
        text = _render_mixed_witness(witness, names, game)
        if text is not None:
            return text
    elif method == "bayesian_nash":
        text = _render_bayesian_witness(witness, names, game)
        if text is not None:
            return text
    elif method == "subgame_perfect":
        text = _render_subgame_witness(witness, names)
        if text is not None:
            return text
    elif method == "correlated_equilibrium":
        text = _render_correlated_witness(witness, names, game)
        if text is not None:
            return text
    elif method == "nash_bargaining":
        text = _render_bargaining_witness(witness, names)
        if text is not None:
            return text
    elif method == "stackelberg":
        text = _render_stackelberg_witness(witness, names, game)
        if text is not None:
            return text
    text = _render_plain_witness(witness, names)
    if method == "pure_nash" and type(game).__name__ == "NormalFormGame" and all(
            isinstance(value, str) for value in witness.values()):
        for row in game.payoffs:
            if row["action_profile"] == dict(witness):
                got = ", ".join(f"{names[player]} {_cell(utility)}"
                                for player, utility in row["utilities"].items())
                return f"{text} → each side gets: {got}"
    return text


def describe_result(result: Mapping, spec_json: str | None = None) -> str:
    if not result:
        return ""
    disposition = result.get("disposition")
    if disposition == "needs_confirmation":
        return "Please press the button to confirm the details first."
    if disposition != "ok":
        reason = result.get("error") or disposition or "unknown problem"
        return f"### I couldn't work this out\n{reason}"
    game = None
    if spec_json:
        try:
            import advisor
            game = advisor._build_envelope_from_payload(json.loads(spec_json)).game
        except Exception:
            game = None
    names = _names(game) if game is not None else {
        p["id"]: p.get("name") or p["id"] for p in result["normalized_game"]["players"]}
    lines = ["## What is likely to happen"]
    rendered_methods = 0
    skipped: list[str] = []
    for method in result.get("method_results", []):
        if method.get("status") == "unsupported":
            method_id = method.get("method") or "unknown"
            title = METHOD_TITLES.get(method_id, method_id.replace("_", " ").capitalize())
            reason = (method.get("error") or "; ".join(method.get("assumptions") or [])
                      or "not applicable to this game")
            skipped.append(f"- {title}: {reason}")
            continue
        rendered_methods += 1
        method_id = method.get("method") or "unknown"
        title = METHOD_TITLES.get(method_id, method_id.replace("_", " ").capitalize())
        status = STATUS_TITLES.get(method.get("status"), method.get("status", "unknown"))
        lines += ["", f"### {title}", f"*{status}*"]
        witnesses = method.get("witnesses") or []
        for witness in witnesses[:MAX_SHOWN]:
            lines.append(f"- {_render_witness(witness, names, game, method_id)}")
        if len(witnesses) > MAX_SHOWN:
            lines.append(f"- … and {len(witnesses) - MAX_SHOWN} more")
        if method.get("error"):
            lines.append(f"- {method['error']}")
        lines += [f"- Note: {a}" for a in method.get("assumptions", [])]
        benchmark = method.get("benchmark_evidence")
        if isinstance(benchmark, Mapping):
            lines += ["", "**P3 benchmark evidence:**",
                      f"- {benchmark.get('pure_nash_count', 0)} pure Nash profiles; "
                      f"{benchmark.get('strict_pure_nash_count', 0)} strict."]
            for equilibrium in benchmark.get("equilibria", []):
                lines.append(f"- {equilibrium['label']}: total utility "
                             f"{equilibrium['total_utility']}.")
            comparison = benchmark.get("welfare_comparison") or {}
            changes = comparison.get("utility_change_by_type", {})
            if changes.get("D") == "-1" and not comparison.get("pareto_improvement"):
                lines.append("- D loses 1 relative to all-A; low-adoption is not a Pareto improvement.")
            elif comparison and not comparison.get("pareto_improvement"):
                lines.append("- Low-adoption is not a Pareto improvement over all-A.")
        p3 = method.get("p3_evidence")
        if isinstance(p3, Mapping):
            lines += ["", "**P3 exact equilibrium evidence:**",
                      f"- {p3.get('pure_nash_count', 0)} pure Nash profiles; "
                      f"{p3.get('strict_pure_nash_count', 0)} strict."]
            for equilibrium in p3.get("equilibria", []):
                lines.append(
                    f"- {equilibrium['label']}: {equilibrium['action_profile']}; "
                    f"utility by type {equilibrium['utility_by_type']}; total utility "
                    f"{equilibrium['total_utility']}; strict: {equilibrium['strict_nash']}.")
    if skipped:
        lines += ["", "### Not applied", *skipped]
    if not rendered_methods:
        lines += ["", "No supported analysis method could be applied to this game."]
    notes = result.get("normalized_game", {}).get("assumptions", [])
    if notes:
        lines += ["", "**Assumptions:** " + "; ".join(notes)]
    return "\n".join(lines)
