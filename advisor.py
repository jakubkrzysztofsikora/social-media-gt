"""Advisor core for the social-media game-theory toolkit.

Infers the toolkit's parameter values from a conversation, runs every scenario
of ``social_media_gt`` with those parameters and renders the full mathematical
evidence.  Imported by both the MCP server and the Gradio app; it depends on
neither of them.
"""
from __future__ import annotations

import inspect
import json
import os
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

import social_media_gt as gt

load_dotenv(Path(__file__).with_name(".env"))


class InferenceError(RuntimeError):
    """The LLM call failed or returned something that is not usable."""


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str
    model: str


def load_config(env: Mapping[str, str] | None = None) -> Config:
    source = os.environ if env is None else env

    base_url = (source.get("GT_LLM_BASE_URL") or "").strip()
    if not base_url:
        host = (source.get("LITELLM_HOST") or "").strip()
        if host:
            scheme = (source.get("LITELLM_SCHEME") or "https").strip()
            port = (source.get("LITELLM_PORT") or "").strip()
            default_port = "443" if scheme == "https" else "80"
            suffix = f":{port}" if port and port != default_port else ""
            base_url = f"{scheme}://{host}{suffix}/v1"
        else:
            base_url = "http://127.0.0.1:4000/v1"

    api_key = ((source.get("GT_LLM_API_KEY") or "").strip()
               or (source.get("LITELLM_MASTER_KEY") or "").strip())
    model = (source.get("GT_LLM_MODEL") or "").strip() or "nemotron-bolt"
    return Config(base_url=base_url, api_key=api_key, model=model)


INTERNAL_PARAMS = {"tw1_1": {"k"}, "tw8": {"full"}}
INTEGER_PARAMS = {"tw3": {"n"}, "tw5": {"n", "d"}}

PARAM_TEXT: dict[str, dict[str, tuple[str, str]]] = {
    "tw1": {
        "tau": ("cost of using social media for one family. Using while others do not gives payoff "
                "-tau; when everyone uses, using gives f1 - tau", "tau >= 0"),
        "f1": ("private benefit of using social media for one family, excluding peer effects",
               "f1 >= 0"),
        "E1": ("harm of losing peer connection when every other family uses: deviating out of an "
               "all-using profile gives -E1", "E1 >= 0"),
    },
    "tw1_1": {
        "a": ("marginal benefit of screen time; the private value is V(t) = a t - b t^2", "a > 0"),
        "b": ("curvature of the screen-time harm (the b t^2 term)", "b > 0"),
        "T": ("maximum screen time a cap permits (outside option)", "T > 0"),
        "beta": ("strength of the peer externality; the peer penalty is beta * k * (T - t) with k "
                 "the share of peers online (k is fixed at 1/2 inside this scenario and is not "
                 "inferred)", "beta > 0"),
    },
    "tw2": {
        "eps": ("baseline share of families using social media before peer pressure acts (floor of "
                "the adoption map)", "0 < eps < 1"),
        "p_slope": ("slope of peer pressure: adoption probability rises p_slope percentage points "
                    "per 1.0 of peer adoption share", "p_slope > 0"),
        "f_lo": ("lower peer-pressure threshold (percent) where adoption starts responding",
                 "f_lo < f_hi"),
        "f_hi": ("upper peer-pressure threshold (percent) where adoption saturates", "f_hi > f_lo"),
    },
    "tw3": {
        "tau": ("cost of using social media, in units where one adopting peer is worth 1: deviating "
                "pays when peer benefit k minus cost tau falls below the exclusion loss k/2",
                "tau >= 0"),
        "n": ("number of families in the cohort", "integer 2 <= n <= 12"),
    },
    "tw3_2": {
        "f1": ("private benefit of using social media for one family", "f1 >= 0"),
        "E1": ("harm of losing peer connection when every other family uses", "E1 >= 0"),
    },
    "tw5": {
        "tau": ("resistance threshold in this scenario's normalization: a family with peer share k "
                "stays while 2k + k^2 >= tau", "1/1000 <= tau <= 3"),
        "n": ("number of families in the cohort", "integer n >= 2"),
        "d": ("families unaware of the pact", "integer 0 <= d <= n - 1"),
    },
    "tw6": {
        "gamma": ("share of the total cohort signing the pact; with eps=0.10, gamma=0.225 is 25% of "
                  "the remaining conscientious households", "1/1000 <= gamma < 1 - eps; "
                  "(1 - eps - gamma) * p_slope != f_hi - f_lo"),
        "eps": ("baseline share of families using social media before peer pressure acts (floor of "
                "the adoption map)", "1/1000 <= eps <= 998/1000"),
        "p_slope": ("slope of peer pressure: adoption probability rises p_slope percentage points "
                    "per 1.0 of peer adoption share", "p_slope > 0"),
        "f_lo": ("lower peer-pressure threshold (percent) where adoption starts responding",
                 "f_lo < f_hi"),
        "f_hi": ("upper peer-pressure threshold (percent) where adoption saturates", "f_hi > f_lo"),
    },
    "tw7": {
        "lam": ("fraction of baseline peer pressure removed; when eta = 0, pressure is "
                "100 * (1 - lam) * k; effective reduction is lam * (1 - eta)",
                "1/1000 <= lam <= 1"),
        "eta": ("attenuation factor in the tw7 recurrence; it scales the specified reduction "
                "by (1 - eta)", "0 <= eta < 1"),
        "eps": ("floor of the adoption map", "0 < eps < 1"),
        "step": ("exogenous adoption pressure added each round", "step > 0"),
    },
    "tw7_2": {
        "creg": ("cost weight of statutory limits; the regulatory cost is creg * lambda_reg^2 / 2",
                 "creg > 0"),
        "cplat": ("cost weight of platform defaults; the platform cost is cplat * lambda_plat^2 / 2",
                  "cplat > 0"),
        "eta": ("evasion rate: fraction of regulatory pressure families route around", "0 <= eta < 1"),
        "lam_target": ("required total reduction in peer pressure", "lam_target > 0"),
    },
    "tw8": {
        "tau_C": ("coordination threshold: a family with peer share k gains tau_C - 2k - k^2 from "
                  "staying", "tau_C >= 0"),
        "delta": ("discount factor per round applied to gains", "0 < delta <= 1"),
    },
}

SCENARIO_NOTES: dict[str, list[str]] = {
    "tw1": [
        "Scale: benefit/cost of one family in units where full private benefit of using social "
        "media equals 1.",
        "Model fact: all-using is a Nash equilibrium iff f1 + E1 >= tau.",
        "Model fact: restricting together Pareto-dominates iff tau > f1.",
    ],
    "tw1_1": [
        "Constraint: a + beta <= 2 b T.",
    ],
    "tw3_2": [
        "Model fact: the risk-dominant cutoff is tau^RD = (f1 + E1)/2.",
        "Model fact: the Pareto trap class is non-empty iff E1 > f1.",
    ],
    "tw7": [
        "Paper P6: lambda in [0, 1] is the fraction of baseline peer pressure removed; when "
        "eta = 0, pressure is 100 * (1 - lambda) * k.",
        "Toolkit extension: eta scales the reduction by (1 - eta), so the effective fraction "
        "removed is lambda * (1 - eta); eta has no defined coverage or evasion mechanism.",
    ],
    "tw8": [
        "Fixed inside the scenario (not inferred): 10 players, 9 aware.",
    ],
}


def _scenario_title(scenario_id: str) -> str:
    func, defaults = gt.SCENARIOS[scenario_id]
    kwargs = dict(defaults)
    if scenario_id == "tw8":
        kwargs["full"] = False
    return func(**kwargs)["title"]


def _build_catalog() -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    for scenario_id, (func, _defaults) in gt.SCENARIOS.items():
        signature = inspect.signature(func)
        internal = INTERNAL_PARAMS.get(scenario_id, set())
        params: dict[str, dict] = {}
        for name, parameter in signature.parameters.items():
            if name in internal:
                continue
            meaning, domain = PARAM_TEXT[scenario_id][name]
            params[name] = {"meaning": meaning, "domain": domain,
                            "default": gt.show(parameter.default)}
        catalog[scenario_id] = {"id": scenario_id, "title": _scenario_title(scenario_id),
                                "params": params, "notes": SCENARIO_NOTES.get(scenario_id, [])}
    return catalog


CATALOG: dict[str, dict] = _build_catalog()


STRATEGY_VIEWS: dict[str, list[dict]] = {
    "tw1": [
        {"strategy": "Cooperate together (all-R)",
         "outcome": "payoff of staying out; cost of the lone deviator",
         "values": {"payoff of staying out": "payoffs_relative_to_offline.all_R.stay",
                    "payoff of the lone deviator": "payoffs_relative_to_offline.all_R.deviate"}},
        {"strategy": "Use together (all-A)",
         "outcome": "payoff of staying in; cost of leaving",
         "values": {"payoff of staying in": "payoffs_relative_to_offline.all_A.stay",
                    "payoff of leaving": "payoffs_relative_to_offline.all_A.deviate"}},
        {"strategy": "Pareto comparison of the two",
         "outcome": "all-R better exactly when tau > f1",
         "values": {"all-R Pareto-dominates all-A": "all_r_dominates_pareto",
                    "welfare gap tau - f1": "pareto_gap_all_R"}},
    ],
    "tw1_1": [
        {"strategy": "Unilateral cooperation (first best)",
         "outcome": "decision without peer effects",
         "values": {"first-best decision t_FB": "t_first_best"}},
        {"strategy": "Cooperation with peer effects (Nash)",
         "outcome": "Nash optimum exceeds the first best (looser cap); welfare loss from peer coupling",
         "values": {"Nash decision t_NE": "t_nash", "welfare loss": "welfare_loss"}},
        {"strategy": "Private vs peer-aware value",
         "outcome": "private value curve vs peer-aware curve",
         "values": {"private value curve V(t)": "curve_V",
                    "peer-aware curve G(t)": "curve_G"}},
    ],
    "tw2": [
        {"strategy": "Uniform peer share",
         "outcome": "mean adoption over a uniform peer share",
         "values": {"mean adoption at a uniform peer share": "uniform_q_mean"}},
        {"strategy": "Peer-pressure cascade",
         "outcome": "limit and rounds to full adoption",
         "values": {"cascade limit": "limit",
                    "rounds to full adoption": "rounds_to_full_adoption",
                    "Phi(k) - k on the linear region": "phi_minus_k_on_linear_region"}},
    ],
    "tw3": [
        {"strategy": "All members use",
         "outcome": "equilibrium status and the advantage of staying",
         "values": {"all-A is a Nash equilibrium": "all_a_is_equilibrium",
                    "advantage of staying at all-A (a deviation loses this much)": "delta_at_all_a"}},
        {"strategy": "All Nash profiles",
         "outcome": "number of Nash profiles",
         "values": {"number of Nash equilibria": "equilibria_count"}},
    ],
    "tw3_2": [
        {"strategy": "Cooperate jointly (Pareto better)",
         "outcome": "trap class and welfare gap at the midpoint",
         "values": {"Pareto trap class non-empty": "pareto_trap_class_nonempty",
                    "welfare gap at the midpoint": "pareto_gap_at_midpoint"}},
        {"strategy": "All-using (risk-dominant)",
         "outcome": "cutoff below which all-using is risk-dominant; equality is the tie boundary",
         "values": {"risk-dominant cutoff tau^RD": "tau_risk_dominant"}},
    ],
    "tw5": [
        {"strategy": "No pact",
         "outcome": "adopter share needed before the first outsider leaves",
         "values": {"threshold peer share k*": "k_star",
                    "pact size at which the first outsider defects": "m_out"}},
        {"strategy": "Pact among aware members",
         "outcome": "how many members hold before leaving",
         "values": {"members needed for stability": "m_stab",
                    "P(5/9)": "P_peer_5_of_9", "P(6/9)": "P_peer_6_of_9"}},
    ],
    "tw6": [
        {"strategy": "No pact",
         "outcome": "adoption returns to full after the pact ends",
         "values": {"rounds to full adoption after release": "release_steps"}},
        {"strategy": "Pact covering gamma of the total cohort",
         "outcome": "adoption level held, contraction, trajectory, exit multiplier",
         "values": {"adoption level K": "K", "contraction a": "a",
                    "trajectory": "trajectory", "exit multiplier": "exit_multiplier"}},
    ],
    "tw7": [
        {"strategy": "Pressure reduction after eta attenuation",
         "outcome": "effective fraction of baseline peer pressure removed and resulting adoption limit",
         "values": {"effective fraction removed": "lambda_effective", "adoption limit": "limit"}},
        {"strategy": "Unattenuated pressure reduction (eta = 0)",
         "outcome": "baseline with the full specified lambda reduction; same toolkit function with eta = 0",
         "values": {"effective fraction removed": "lambda_effective", "adoption limit": "limit"},
         "baseline": {"eta": 0}},
    ],
    "tw7_2": [
        {"strategy": "Statutory limits",
         "outcome": "share of the pressure reduction carried by regulation",
         "values": {"lambda_reg": "lambda_reg"}},
        {"strategy": "Product defaults",
         "outcome": "share of the pressure reduction carried by product defaults",
         "values": {"lambda_plat": "lambda_plat"}},
        {"strategy": "Optimal hybrid",
         "outcome": "weight ratio: regulation against product defaults",
         "values": {"lambda_reg / lambda_plat": "ratio"}},
    ],
    "tw8": [
        {"strategy": "Exit without a public schedule",
         "outcome": "worst-case advantage and number of minimizing states",
         "values": {"discounted advantage": "discounted_min",
                    "minimizing states": "minimizing_states"}},
        {"strategy": "Public sequential schedule",
         "outcome": "undiscounted advantage, worst sequence, orderings checked",
         "values": {"undiscounted advantage": "undiscounted_sum",
                    "worst sequence": "worst_sequence",
                    "worst slot": "worst_d_slot",
                    "orderings enumerated": "orderings_enumerated"}},
    ],
}


def to_fraction(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"boolean is not a numeric value: {value!r}")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(str(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("empty string is not a numeric value")
        return Fraction(text)
    raise ValueError(f"unsupported numeric value: {value!r}")


def _round_int(value: Fraction) -> int:
    if value >= 0:
        return int(value + Fraction(1, 2))
    return -int(-value + Fraction(1, 2))


def _round_to_int(values: dict[str, Fraction], notes: list[str], key: str) -> None:
    """Round a whole-number parameter and say so."""
    value = values[key]
    rounded = Fraction(_round_int(value))
    if rounded != value:
        values[key] = rounded
        notes.append(f"{key}: rounded from {gt.show(value)} to {gt.show(rounded)} "
                     "(cohort sizes are whole numbers)")


def _clamp_nonpositive(values: dict[str, Fraction], notes: list[str], key: str,
                        floor: Fraction, reason: str) -> None:
    """Repair only zero or negative values; legal positives are left untouched."""
    value = values[key]
    if value <= 0:
        values[key] = floor
        notes.append(f"{key}: adjusted from {gt.show(value)} to {gt.show(floor)} ({reason})")


def _clamp(values: dict[str, Fraction], notes: list[str], key: str,
           low: Fraction | None, high: Fraction | None, reason: str) -> None:
    value = values[key]
    new = value
    if low is not None and new < low:
        new = low
    elif high is not None and new > high:
        new = high
    if new != value:
        values[key] = new
        notes.append(f"{key}: adjusted from {gt.show(value)} to {gt.show(new)} ({reason})")


def normalize_params(scenario_id: str, raw: Mapping[str, object]) -> tuple[dict[str, Fraction], list[str]]:
    """Map inferred values onto a scenario's parameters, repairing out-of-domain input."""
    func, _defaults = gt.SCENARIOS[scenario_id]
    signature = inspect.signature(func)
    internal = INTERNAL_PARAMS.get(scenario_id, set())

    values: dict[str, Fraction] = {}
    for name, parameter in signature.parameters.items():
        if name in internal:
            continue
        values[name] = to_fraction(parameter.default)

    notes: list[str] = []
    for key, raw_value in (raw or {}).items():
        if key not in values:
            notes.append(f"{key}: inferred value unusable, paper default used")
            continue
        try:
            values[key] = to_fraction(raw_value)
        except (ValueError, TypeError, ZeroDivisionError):
            notes.append(f"{key}: inferred value unusable, paper default used")

    if scenario_id in ("tw1", "tw3_2"):
        for key, reason in (("tau", "cost cannot be negative"),
                            ("f1", "private benefit cannot be negative"),
                            ("E1", "exclusion harm cannot be negative")):
            if key in values:
                _clamp(values, notes, key, Fraction(0), None, reason)

    if scenario_id == "tw1_1":
        for key in ("a", "b", "T", "beta"):
            _clamp_nonpositive(values, notes, key, Fraction(1, 1000), "must stay positive")
        a, b, t, beta = values["a"], values["b"], values["T"], values["beta"]
        if a + beta > 2 * b * t:
            scale = 2 * b * t / (a + beta)
            for key, before in (("a", a), ("beta", beta)):
                values[key] = before * scale
                notes.append(f"{key}: adjusted from {gt.show(before)} to {gt.show(values[key])} "
                             "(scaled to satisfy a + beta <= 2 b T)")

    if scenario_id in ("tw2", "tw6"):
        upper = Fraction(998, 1000) if scenario_id == "tw6" else Fraction(999, 1000)
        _clamp(values, notes, "eps", Fraction(1, 1000), upper,
               "adoption floor must stay inside (0, 1)")
        _clamp(values, notes, "p_slope", Fraction(1, 1000), None, "peer-pressure slope must stay positive")
        if values["f_hi"] <= values["f_lo"]:
            old = values["f_hi"]
            values["f_hi"] = values["f_lo"] + 1
            notes.append(f"f_hi: adjusted from {gt.show(old)} to {gt.show(values['f_hi'])} "
                         "(must exceed f_lo)")

    if scenario_id == "tw3":
        _clamp(values, notes, "tau", Fraction(0), None, "cost cannot be negative")
        _round_to_int(values, notes, "n")
        _clamp(values, notes, "n", Fraction(2), Fraction(12), "cohort size must stay in [2, 12]")

    if scenario_id == "tw5":
        _clamp(values, notes, "tau", Fraction(1, 1000), Fraction(3),
               "the threshold must leave a root inside [0, 1]")
        _round_to_int(values, notes, "n")
        _clamp(values, notes, "n", Fraction(2), None, "cohort size must stay at least 2")
        _round_to_int(values, notes, "d")
        _clamp(values, notes, "d", Fraction(0), values["n"] - 1,
               "unaware members must stay between 0 and n - 1")

    if scenario_id == "tw6":
        _clamp(values, notes, "gamma", Fraction(1, 1000), Fraction(999, 1000),
               "pact share must stay positive")
        room = Fraction(1) - values["eps"] - Fraction(1, 1000)
        if values["gamma"] > room:
            old = values["gamma"]
            values["gamma"] = room
            notes.append(f"gamma: adjusted from {gt.show(old)} to {gt.show(room)} "
                         "(the pact cannot cover the whole cohort)")
        slope_term = (1 - values["eps"] - values["gamma"]) * values["p_slope"]
        span = values["f_hi"] - values["f_lo"]
        if slope_term == span:
            raise ValueError("singular tw6 coalition operator parameter combination")

    if scenario_id == "tw7":
        _clamp(values, notes, "lam", Fraction(1, 1000), Fraction(1),
               "pressure-reduction fraction must stay in [1/1000, 1]")
        _clamp(values, notes, "eta", Fraction(0), Fraction(999, 1000),
               "reduction attenuation must stay in [0, 1)")
        _clamp(values, notes, "eps", Fraction(1, 1000), Fraction(999, 1000),
               "adoption floor must stay inside (0, 1)")
        _clamp(values, notes, "step", Fraction(1, 1000), None, "exogenous pressure must stay positive")

    if scenario_id == "tw7_2":
        for key in ("creg", "cplat", "lam_target"):
            _clamp(values, notes, key, Fraction(1, 1000), None, "must stay positive")
        _clamp(values, notes, "eta", Fraction(0), Fraction(999, 1000),
               "evasion rate must stay in [0, 1)")

    if scenario_id == "tw8":
        _clamp(values, notes, "tau_C", Fraction(0), None, "threshold cannot be negative")
        _clamp(values, notes, "delta", Fraction(1, 1000), Fraction(1),
               "discount factor must stay inside (0, 1]")

    return values, notes


def _render(value: Any) -> Any:
    if isinstance(value, Fraction):
        return gt.show(value)
    if isinstance(value, dict):
        return {key: _render(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_render(item) for item in value]
    return value


_MISSING = object()


def _lookup(results: Mapping[str, Any], path: str) -> Any:
    current: Any = results
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


def _contract_example() -> dict:
    example: dict[str, Any] = {"situation_summary": "one paragraph restating the situation in "
                                                     "game-theory terms",
                               "vocabulary": {"players": "who the players are, in the user's words",
                                              "decision": "the decision each player makes, in the "
                                                          "user's words",
                                              "cooperation": "what cooperating looks like, in the "
                                                             "user's words",
                                              "defection": "what defecting looks like, in the "
                                                           "user's words"},
                               "scenarios": {}}
    for scenario_id, entry in CATALOG.items():
        params = {}
        for name, spec in entry["params"].items():
            default = spec["default"]
            params[name] = int(default) if default.lstrip("-").isdigit() else default
        example["scenarios"][scenario_id] = {"params": params,
                                             "rationale": "why these values follow from the "
                                                          "conversation"}
    return example


def build_system_prompt() -> str:
    lines = [
        "You are a game-theory analyst using an exact-arithmetic proof toolkit.",
        "Map a real situation from the conversation onto the parameters of ten game-theoretic "
        "scenarios taken from a published paper.",
        "Unit scale: benefit and cost of one family, in units where the full private benefit of "
        "using social media equals 1.",
        "",
        "Scenarios and their parameters:",
    ]
    for scenario_id, entry in CATALOG.items():
        lines.append("")
        lines.append(f"{scenario_id} - {entry['title']}")
        for note in entry["notes"]:
            lines.append(f"  {note}")
        for name, spec in entry["params"].items():
            lines.append(f"  {name}: {spec['meaning']} Domain: {spec['domain']}. "
                         f"Default: {spec['default']}.")
    lines += [
        "",
        "How to map the conversation onto parameters:",
        "- A cohort size stated in the conversation becomes n in tw3 and n in tw5.",
        "- A stated number of families unaware of a pact becomes d in tw5.",
        "- gamma is a share of the total cohort. If the conversation states a share of the "
        "remaining conscientious households, multiply it by (1 - eps); with eps = 0.10, "
        "25% of the remaining 90% is gamma = 0.225.",
        "- Do not copy a percentage across different denominators; convert it to the "
        "parameter's stated denominator.",
        "- A stated fraction of baseline peer pressure removed becomes lambda (lam) in tw7; "
        "P(k) = 100 * (1 - lambda) * k when eta = 0. Do not treat lambda as a pressure slope.",
        "- In tw7, eta attenuates the specified reduction by the factor (1 - eta). Do not map "
        "group coverage or evasion to eta; map it only when the conversation states reduction "
        "attenuation.",
        "- A stated cost, benefit or harm relative to the full private benefit becomes tau, f1 or "
        "E1 in tw1 and f1 or E1 in tw3_2.",
        "- A stated screen-time limit becomes T in tw1_1.",
        "- Prefer a number stated in the conversation over the paper default.",
        "- Every number stated in the conversation that matches a parameter's meaning MUST be "
        "mapped using its units and denominator; never return the paper default for a parameter "
        "the conversation quantifies.",
        "- If a stated number matches no parameter's meaning, leave it unmapped instead of "
        "forcing it into an unrelated parameter.",
        "- situation_summary and vocabulary MUST describe the user's own situation: their "
        "players, actions and setting. Describe the paper's social-media example only when the "
        "user's situation is about social media.",
        "",
        "Reply with ONLY a JSON object of this shape:",
        "",
        json.dumps(_contract_example(), indent=2),
        "",
        "Always include all 10 scenario ids.",
        "Use the paper default for any parameter the conversation gives no signal about.",
        "Values may be JSON numbers or fraction strings like '1/2'.",
        "Do not invent scenarios or extra keys.",
        "If the conversation does not yet pin down who is involved, what each side can do, "
        'and what each side wants, reply with ONLY {"need_more_information": true, '
        '"questions": [up to 2 short questions]} instead of the analysis. '
        "Write each question the way you would ask a friend who knows no game theory, and "
        "never use the words payoff, utility, Nash, equilibrium, strategy, player, type, "
        "threshold or coefficient in a question. Ask about the real situation instead: who "
        "decides, what each side can do, and which result each side prefers. Ask at most once "
        "in a conversation. If the situation is clear, never ask questions about unstated "
        "numbers - analyze with the paper defaults and the existing caveat covers them.",
    ]
    return "\n".join(lines)


def _repair_escapes(text: str) -> str:
    """Double the backslashes that start an invalid JSON escape inside string literals."""
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char != '"':
            out.append(char)
            index += 1
            continue
        out.append(char)
        index += 1
        while index < length:
            current = text[index]
            if current == '"':
                out.append(current)
                index += 1
                break
            if current != "\\":
                out.append(current)
                index += 1
                continue
            if index + 1 < length and text[index + 1] in '"\\/bfnrtu':
                out.append(current)
                out.append(text[index + 1])
                index += 2
                continue
            out.append("\\\\")
            index += 1
    return "".join(out)


def _parse_json_object(raw: str) -> dict:
    text = raw.strip()
    if "```" in text:
        for block in text.split("```")[1:]:
            body = block.strip()
            if body.lower().startswith("json"):
                body = body[4:].strip()
            if body.startswith("{"):
                text = body
                break
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in the reply")
    candidate = text[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = json.loads(_repair_escapes(candidate))
    if not isinstance(parsed, dict):
        raise ValueError("the reply is not a JSON object")
    return parsed


def _validate_inference_payload(payload: Mapping[str, Any]) -> None:
    needs_more = payload.get("need_more_information", False)
    if not isinstance(needs_more, bool):
        raise ValueError("need_more_information must be a boolean")
    if needs_more:
        questions = payload.get("questions")
        if (not isinstance(questions, list) or not 1 <= len(questions) <= 2
                or any(not isinstance(question, str) or not question.strip()
                       for question in questions)):
            raise ValueError("clarification response needs 1 to 2 non-empty string questions")
        return

    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, Mapping):
        raise ValueError("analysis response needs a scenarios object")
    missing = set(CATALOG) - set(scenarios)
    if missing:
        raise ValueError(f"analysis response is missing scenarios: {', '.join(sorted(missing))}")
    if not isinstance(payload.get("situation_summary"), str):
        raise ValueError("analysis response needs a string situation_summary")
    vocabulary = payload.get("vocabulary", {})
    if (not isinstance(vocabulary, Mapping)
            or any(not isinstance(value, str) for value in vocabulary.values())):
        raise ValueError("analysis response vocabulary must map labels to strings")
    for scenario_id, entry in scenarios.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"scenario {scenario_id} must be an object")
        params = entry.get("params", entry)
        if not isinstance(params, Mapping):
            raise ValueError(f"scenario {scenario_id} parameters must be an object")


def infer_parameters(messages: list[dict], client=None, config: Config | None = None) -> dict:
    """Ask the LLM for a parameter set per scenario."""
    config = config or load_config()
    if client is None:
        import openai
        client = openai.OpenAI(base_url=config.base_url, api_key=config.api_key or "none")

    history: list[dict] = [{"role": "system", "content": build_system_prompt()}]
    history.extend(messages or [])

    last_error = ""
    for attempt in (1, 2):
        try:
            completion = client.chat.completions.create(
                model=config.model, messages=history, temperature=0)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status is None:
                status = getattr(getattr(exc, "response", None), "status_code", None)
            raise InferenceError(f"LLM request failed ({status or type(exc).__name__}): {exc}") from exc
        raw = completion.choices[0].message.content or ""
        try:
            parsed = _parse_json_object(raw)
            _validate_inference_payload(parsed)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt == 1:
                history = history + [{"role": "assistant", "content": raw},
                                     {"role": "user", "content": f"Invalid response: {last_error}. "
                                      "Reply with ONLY the JSON object described above."}]
                continue
            raise InferenceError(f"LLM returned invalid response: {last_error}; {raw[:400]}") from exc
        parsed["raw"] = raw
        parsed["model"] = config.model
        return parsed

    raise InferenceError(f"LLM did not return parseable JSON: {last_error}")


def _scenario_kwargs(scenario_id: str, normalized: Mapping[str, Fraction],
                     defaults: Mapping[str, Any]) -> dict:
    """Call arguments for a scenario; integer parameters must reach the toolkit as ints."""
    kwargs = dict(defaults)
    integers = INTEGER_PARAMS.get(scenario_id, set())
    for name, value in normalized.items():
        kwargs[name] = int(value) if name in integers else value
    if scenario_id == "tw8":
        kwargs["full"] = True
    return kwargs


def _strategy_rows(scenario_id: str, func, defaults: Mapping[str, Any],
                   normalized: Mapping[str, Fraction], results: Mapping[str, Any]) -> list[dict]:
    rows: list[dict] = []
    for view in STRATEGY_VIEWS.get(scenario_id, []):
        source = results
        baseline = view.get("baseline")
        if baseline:
            merged = {key: value for key, value in normalized.items()}
            merged.update(baseline)
            baseline_values, _notes = normalize_params(scenario_id, merged)
            baseline_kwargs = _scenario_kwargs(scenario_id, baseline_values, defaults)
            if scenario_id == "tw8":
                baseline_kwargs["full"] = False
            source = func(**baseline_kwargs)["results"]
        values: dict[str, Any] = {}
        for label, path in view.get("values", {}).items():
            value = _lookup(source, path)
            if value is _MISSING:
                continue
            values[label] = _render(value)
        rows.append({"strategy": view["strategy"], "outcome": view["outcome"], "values": values})
    return rows


def run_one_scenario(scenario_id: str, raw: Mapping[str, object] | None = None) -> dict:
    """Run a single scenario with the given (possibly partial) parameter overrides."""
    func, defaults = gt.SCENARIOS[scenario_id]
    raw = dict(raw or {})
    entry = {"scenario": scenario_id, "title": CATALOG[scenario_id]["title"], "status": "ok",
             "params": {}, "param_notes": [], "results": {}, "checks": [], "conclusion": [],
             "strategies": []}
    full_flag = raw.pop("full", None) if scenario_id == "tw8" else None
    try:
        normalized, notes = normalize_params(scenario_id, raw)
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = f"could not normalize parameters: {exc}"
        return entry

    entry["params"] = {key: gt.show(value) for key, value in normalized.items()}
    entry["param_notes"] = notes

    kwargs = _scenario_kwargs(scenario_id, normalized, defaults)
    if scenario_id == "tw8":
        kwargs["full"] = True if full_flag is None else bool(full_flag)

    try:
        result = func(**kwargs)
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        return entry

    entry["results"] = _render(result["results"])
    entry["checks"] = result["checks"]
    if scenario_id == "tw7":
        entry["conclusion"] = [
            "P6 lambda is the fraction of baseline peer pressure removed; when eta = 0, pressure "
            "is 100 * (1 - lambda) * k.",
            "In tw7, eta scales the specified reduction by (1 - eta), so the effective fraction "
            "removed is lambda * (1 - eta); eta identifies neither group coverage nor evasion.",
        ]
        if all(check["ok"] for check in entry["checks"]):
            entry["conclusion"].append(
                f"The recurrence converges to adoption limit {entry['results']['limit']} for these parameters.")
    else:
        entry["conclusion"] = [_neutral_conclusion(str(sentence))
                               for sentence in result["conclusion"]]
    deviating = sorted(name for name, value in normalized.items()
                       if name in CATALOG[scenario_id]["params"]
                       and value != to_fraction(CATALOG[scenario_id]["params"][name]["default"]))
    if deviating:
        entry["conclusion"].append(
            "Parameters were mapped from the conversation; narrative numbers quoted from "
            "the paper's default calibration may not match the values above (differing: "
            f"{', '.join(deviating)}).")
    if any(not check["ok"] for check in entry["checks"]):
        entry["status"] = "error"
        entry["error"] = "one or more verification checks failed"
    try:
        entry["strategies"] = _strategy_rows(scenario_id, func, defaults, normalized,
                                             result["results"])
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = f"strategy table failed: {type(exc).__name__}: {exc}"
        entry["strategies"] = []
        entry["param_notes"] = list(notes) + [f"strategy table unavailable: {exc}"]
    return entry


def run_scenarios(params_by_scenario: Mapping[str, Mapping[str, object]]) -> list[dict]:
    """Run every scenario with the given (possibly partial) parameter overrides."""
    overrides = params_by_scenario or {}
    return [run_one_scenario(scenario_id, overrides.get(scenario_id))
            for scenario_id in gt.SCENARIOS]


def _contains_long_list(value: Any, limit: int = 8) -> bool:
    if isinstance(value, list):
        return len(value) > limit or any(_contains_long_list(item, limit) for item in value)
    if isinstance(value, dict):
        return any(_contains_long_list(item, limit) for item in value.values())
    return False


_PLAIN_RESULT_LABELS = {
    "delta_at_zero": "the payoff difference at the start",
    "all_a_is_ne": "whether everyone adopting is stable",
    "all_r_dominates_pareto": "whether restricting together beats adopting together",
    "payoffs_relative_to_offline": "the payoffs compared with staying out",
    "pareto_gap_all_R": "the welfare gap between the two profiles",
    "t_first_best": "the best outcome for a single actor",
    "t_nash": "the outcome each actor picks in equilibrium",
    "welfare_loss": "the welfare lost to the interaction",
    "limit": "the level the process converges to",
    "rounds_to_full_adoption": "the number of rounds until everyone joins",
    "uniform_q_mean": "the average outcome over uniform starting shares",
    "equilibria_count": "the number of stable profiles",
    "all_a_is_equilibrium": "whether everyone adopting is stable",
    "delta_at_all_a": "the advantage of staying when everyone joins",
    "tau_risk_dominant": "the cost threshold where the safer choice wins",
    "pareto_trap_class_nonempty": "whether the trap profile exists",
    "pareto_gap_at_midpoint": "the welfare gap at the midpoint",
    "m_out": "how many members the pact needs before an outsider defects",
    "m_stab": "how many members keep the pact stable",
    "k_star": "the critical share of participants",
    "P_peer_5_of_9": "the chance five of nine participants follow the rule",
    "P_peer_6_of_9": "the chance six of nine participants follow the rule",
    "K": "the level held after the pact starts",
    "trajectory": "the path the level takes over time",
    "exit_multiplier": "how strongly the pact decays when it ends",
    "release_steps": "how many rounds until the level recovers",
    "eta_critical": "the attenuation threshold where the adoption limit reaches full adoption",
    "lambda_effective": "the effective fraction of baseline peer pressure removed after eta attenuation",
    "lambda_reg": "the pressure a regulatory cost creates",
    "lambda_plat": "the pressure a platform creates",
    "ratio": "the ratio of the two pressures",
    "undiscounted_sum": "the undiscounted total",
    "discounted_min": "the smallest discounted total",
    "worst_sequence": "the worst order of states",
    "worst_d_slot": "the least favourable slot",
    "minimizing_states": "the states that make the total smallest",
    "orderings_enumerated": "the number of orders checked",
}

_JARGON_PATTERN = re.compile(
    r"\btw\d+(?:_\d+)?\b|\b(?:m_out|m_stab|worst_d_slot|worst_sequence|minimizing_states"
    r"|orderings_enumerated|pareto_gap_all_R|all_a_is_ne|all_a_is_equilibrium|tau_risk_dominant"
    r"|k_star|eta_critical|lambda_effective|lambda_reg|lambda_plat)\b", re.IGNORECASE)


def _plain_result_label(key: str) -> str:
    label = _PLAIN_RESULT_LABELS.get(key)
    if label:
        return label
    return key.replace("_", " ").strip()


_PLAIN_SCENARIO_NAMES = {
    "tw1": "shared-cost dilemma",
    "tw1_1": "individual choice with peer effects",
    "tw2": "self-reinforcing participation",
    "tw3": "stability under individual changes",
    "tw3_2": "uncertain incentives and participation",
    "tw5": "outsider incentives and agreement durability",
    "tw6": "coalition formation and persistence",
    "tw7": "fractional peer-pressure reduction",
    "tw7_2": "enforcement and evasion",
    "tw8": "sequential withdrawal without authority",
}


def _plain_scenario_name(scenario_id: str, title: str) -> str:
    return _PLAIN_SCENARIO_NAMES.get(scenario_id, title)


def calculations_digest(scenarios: list[Mapping[str, Any]]) -> list[dict]:
    digest = []
    for item in scenarios:
        checks = item.get("checks", [])
        results = {_plain_result_label(key): value
                   for key, value in item.get("results", {}).items()
                   if not _contains_long_list(value)}
        digest.append({"scenario": _plain_scenario_name(item.get("scenario", ""),
                                                        item.get("title", "")),
                       "status": item.get("status", ""),
                       "params": dict(item.get("params", {})),
                       "checks": f"{sum(1 for check in checks if check.get('ok'))}"
                                 f"/{len(checks)}",
                       "results": results})
    return digest


def _practical_conclusions(summary: str, vocabulary: Mapping[str, Any],
                           digest: list[dict], client=None,
                           config: Config | None = None) -> list[str]:
    config = config or load_config()
    if client is None:
        import openai
        client = openai.OpenAI(base_url=config.base_url, api_key=config.api_key or "none")
    system = ("You are a game-theory analyst advising a decision-maker. Using ONLY the numbers "
              "in this evidence, write 4 to 7 short practical conclusions, each one sentence, "
              "each citing at least one exact number from the evidence. Refer to each scenario "
              "by its plain name from the evidence, never by an id. Never use internal field "
              "names: describe what a number measures instead. Frame every conclusion in the "
              "user's own terms, using their players, decision and forms of cooperation and "
              "defection. Treat paper defaults as assumptions, not measured facts; frame "
              "conclusions conditionally on the displayed parameter values. For the fractional "
              "peer-pressure scenario, lambda is the fraction removed from baseline peer pressure, "
              "not a pressure slope; eta only attenuates the reduction in this recurrence and "
              "does not identify coverage or evasion. Reply with ONLY a JSON object "
              '{"practical_conclusions": ["..."]}')
    user = json.dumps({"situation_summary": summary, "vocabulary": dict(vocabulary),
                       "calculations": digest})
    try:
        completion = client.chat.completions.create(
            model=config.model, temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        parsed = _parse_json_object(completion.choices[0].message.content or "")
        bullets = parsed.get("practical_conclusions")
        if not isinstance(bullets, list):
            return []
        return [str(bullet).strip() for bullet in bullets
                if str(bullet).strip() and not _JARGON_PATTERN.search(str(bullet))]
    except Exception:
        return []


def analyze_situation(messages: list[dict], client=None, config: Config | None = None) -> dict:
    """Infer parameters from the conversation, run all scenarios and assemble the report."""
    config = config or load_config()
    inference = infer_parameters(messages, client=client, config=config)

    questions = inference.get("questions")
    if inference.get("need_more_information") is True and questions:
        questions = [str(question) for question in questions]
        lines = ["## Clarifying questions", ""]
        lines.extend(f"{index}. {_math_text(question)}"
                     for index, question in enumerate(questions, 1))
        return {"need_more_information": True,
                "questions": questions,
                "situation_summary": inference.get("situation_summary", "") or "",
                "vocabulary": inference.get("vocabulary") or {},
                "model": inference.get("model") or config.model,
                "markdown": "\n".join(lines)}

    scenario_params: dict[str, dict] = {}
    for scenario_id, entry in (inference.get("scenarios") or {}).items():
        if scenario_id not in CATALOG:
            continue
        if isinstance(entry, dict) and isinstance(entry.get("params"), dict):
            scenario_params[scenario_id] = dict(entry["params"])
        elif isinstance(entry, dict):
            scenario_params[scenario_id] = dict(entry)
        else:
            scenario_params[scenario_id] = {}

    scenarios = run_scenarios(scenario_params)
    checks_total = sum(len(item["checks"]) for item in scenarios)
    checks_failed = sum(1 for item in scenarios for check in item["checks"] if not check["ok"])
    totals = {"scenarios": len(scenarios),
              "ok": sum(1 for item in scenarios if item["status"] == "ok"),
              "errors": sum(1 for item in scenarios if item["status"] == "error"),
              "checks_total": checks_total,
              "checks_failed": checks_failed}

    report = {"situation_summary": inference.get("situation_summary", "") or "",
              "vocabulary": inference.get("vocabulary") or {},
              "model": inference.get("model") or config.model,
              "inference_raw": inference.get("raw", "") or "",
              "scenarios": scenarios,
              "totals": totals}
    report["practical_conclusions"] = _practical_conclusions(
        report["situation_summary"], report["vocabulary"],
        calculations_digest(scenarios), client=client, config=config)
    report["markdown"] = render_report(report)
    return report


PARAMETER_MATH_LABELS = {
    "tau": r"$\tau$", "f1": r"$f_1$", "E1": r"$E_1$", "beta": r"$\beta$",
    "eps": r"$\epsilon$", "p_slope": r"$p_{\mathrm{slope}}$",
    "f_lo": r"$f_{\mathrm{lo}}$", "f_hi": r"$f_{\mathrm{hi}}$", "n": "$n$",
    "d": "$d$", "gamma": r"$\gamma$", "lam": r"$\lambda$", "eta": r"$\eta$",
    "step": r"$\mathrm{step}$", "creg": r"$c_{\mathrm{reg}}$", "cplat": r"$c_{\mathrm{plat}}$",
    "lam_target": r"$\lambda_{\mathrm{target}}$", "tau_C": r"$\tau_C$",
}

RESULT_MATH_LABELS = {
    "delta_at_zero": r"$\Delta(0)$", "all_a_is_ne": "All-A is Nash",
    "all_r_dominates_pareto": "All-R Pareto dominance",
    "payoffs_relative_to_offline": "Payoffs relative to offline",
    "pareto_gap_all_R": "Pareto gap, all-R",
    "t_first_best": r"$t_{\mathrm{FB}}$", "t_nash": r"$t_{\mathrm{NE}}$",
    "welfare_loss": "Welfare loss", "curve_t": "$t$ values",
    "curve_V": "$V(t)$", "curve_G": "$G(t)$", "limit": "Fixed-point limit",
    "rounds_to_full_adoption": "Rounds to full adoption",
    "phi_minus_k_on_linear_region": r"$\Phi(k)-k$ on linear region",
    "uniform_q_mean": r"$\mathbb{E}[q]$ for uniform $q$",
    "equilibria_count": "Nash equilibrium count", "all_a_is_equilibrium": "All-A is Nash",
    "delta_at_all_a": r"$\Delta_A$ (advantage of staying)",
    "tau_risk_dominant": r"$\tau^{\mathrm{RD}}$",
    "pareto_trap_class_nonempty": "Pareto trap class non-empty",
    "pareto_gap_at_midpoint": "Midpoint Pareto gap", "m_out": r"$m_{\mathrm{out}}$",
    "m_stab": r"$m_{\mathrm{stab}}$", "k_star": "$k^*$",
    "k_star_decimal": "$k^*$ (decimal)", "P_peer_5_of_9": "$P(5/9)$",
    "P_peer_6_of_9": "$P(6/9)$", "K": "$K$", "a": "$a$",
    "trajectory": "Adoption trajectory", "exit_multiplier": "Exit multiplier",
    "release_steps": "Release rounds",
    "eta_critical": "Critical attenuation for full adoption ($\\eta^*$)",
    "lambda_effective": r"$\lambda_{\mathrm{eff}}$ (fraction of baseline pressure removed)",
    "lambda_reg": r"$\lambda_{\mathrm{reg}}$",
    "lambda_plat": r"$\lambda_{\mathrm{plat}}$", "ratio": r"$\lambda_{\mathrm{reg}}/\lambda_{\mathrm{plat}}$",
    "undiscounted_sum": "Undiscounted sum", "discounted_min": "Discounted minimum",
    "worst_sequence": "Worst sequence", "worst_d_slot": "Worst slot",
    "minimizing_states": "Minimizing states", "orderings_enumerated": "Orderings enumerated",
}

_MATH_TEXT = [
    (r"V\(t_FB\) - V\(t_NE\) = b \(t_NE - t_FB\)\^2",
     r"$V(t_{\mathrm{FB}})-V(t_{\mathrm{NE}})=b(t_{\mathrm{NE}}-t_{\mathrm{FB}})^2$"),
    (r"tau\^RD = \(f1 \+ E1\)/2", r"$\tau^{\mathrm{RD}}=(f_1+E_1)/2$"),
    (r"tau\^RD = 2", r"$\tau^{\mathrm{RD}}=2$"),
    (r"risk-dominant cutoff is P\(1\)/2", r"risk-dominant cutoff is $P(1)/2$"),
    (r"E\[tau \| x_i = c\]", r"$E[\tau\mid x_i=c]$"),
    (r"q\(tau\)", r"$q(\tau)$"),
    (r"P\(k\)", r"$P(k)$"),
    (r"f\(1\)", r"$f(1)$"),
    (r"P\(1\) >= tau", r"$P(1)\ge\tau$"),
    (r"tau > f1", r"$\tau>f_1$"),
    (r"f1 \+ E1 >= tau", r"$f_1+E_1\ge\tau$"),
    (r"a \+ beta <= 2 b T", r"$a+\beta\le2bT$"),
    (r"2k \+ k\^2 >= tau", r"$2k+k^2\ge\tau$"),
    (r"tau_C - 2k - k\^2", r"$\tau_C-2k-k^2$"),
    (r"Phi\(k\) - k", r"$\Phi(k)-k$"),
    (r"V\(t_FB\)", r"$V(t_{\mathrm{FB}})$"),
    (r"V\(t_NE\)", r"$V(t_{\mathrm{NE}})$"),
    (r"P\(1\)", r"$P(1)$"),
    (r"P\(5/9\)", r"$P(5/9)$"),
    (r"P\(6/9\)", r"$P(6/9)$"),
    (r"tau\^RD", r"$\tau^{\mathrm{RD}}$"),
    (r"t_FB", r"$t_{\mathrm{FB}}$"),
    (r"t_NE", r"$t_{\mathrm{NE}}$"),
    (r"lambda_effective", r"$\lambda_{\mathrm{eff}}$"),
    (r"lambda_reg", r"$\lambda_{\mathrm{reg}}$"),
    (r"lambda_plat", r"$\lambda_{\mathrm{plat}}$"),
    (r"lam_target", r"$\lambda_{\mathrm{target}}$"),
    (r"p_slope", r"$p_{\mathrm{slope}}$"),
    (r"f_lo", r"$f_{\mathrm{lo}}$"), (r"f_hi", r"$f_{\mathrm{hi}}$"),
    (r"tau_C", r"$\tau_C$"), (r"f1", r"$f_1$"), (r"E1", r"$E_1$"),
    (r"beta", r"$\beta$"), (r"gamma", r"$\gamma$"), (r"eps", r"$\epsilon$"),
    (r"eta", r"$\eta$"), (r"delta", r"$\delta$"), (r"lambda", r"$\lambda$"),
    (r"lam", r"$\lambda$"), (r"tau", r"$\tau$"), (r"Phi", r"$\Phi$"),
]
_MATH_TEXT_PATTERN = re.compile("|".join(
    f"(?<![\\w])(?P<m{i}>{pattern})(?![\\w])"
    for i, (pattern, _) in enumerate(_MATH_TEXT)))
_MATH_SPAN_PATTERN = re.compile(r"(?<!\\)([$]{1,2})(?:\\.|[^\\$])*?(?<!\\)\1")
_MATH_TEXT_REPLACEMENTS = {f"m{i}": replacement for i, (_, replacement) in enumerate(_MATH_TEXT)}


def _math_text(value: Any) -> str:
    text = str(value)
    output = []
    cursor = 0
    for span in _MATH_SPAN_PATTERN.finditer(text):
        outside = text[cursor:span.start()]
        output.append(_MATH_TEXT_PATTERN.sub(
            lambda match: _MATH_TEXT_REPLACEMENTS[match.lastgroup], outside))
        output.append(span.group())
        cursor = span.end()
    output.append(_MATH_TEXT_PATTERN.sub(
        lambda match: _MATH_TEXT_REPLACEMENTS[match.lastgroup], text[cursor:]))
    return "".join(output)


def _math_value(value: Any) -> str:
    if value is None:
        return "null"
    text = gt.show(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d+/\d+", text):
        numerator, denominator = text.split("/", 1)
        return rf"$\frac{{{numerator}}}{{{denominator}}}$"
    if isinstance(value, str) and "sqrt(" in text:
        text = re.sub(r"sqrt\(([^()]*)\)", lambda match: rf"\sqrt{{{match.group(1)}}}", text)
        return f"${text}$"
    if isinstance(value, list):
        return text
    if isinstance(value, dict):
        return re.sub(r"(?<![\w])(-?\d+)/(\d+)(?![\w])",
                      lambda match: rf"$\frac{{{match.group(1)}}}{{{match.group(2)}}}$", text)
    return text


def _result_label(key: str) -> str:
    return RESULT_MATH_LABELS.get(key, _math_text(key.replace("_", " ")))


def _parameter_label(key: str) -> str:
    return PARAMETER_MATH_LABELS.get(key, _math_text(key))


_NEUTRAL_CONCLUSIONS = [
    ("A parent sets a stricter screen-time limit", "A member sets a stricter decision cap"),
    ("hurts the child more than screen time does", "hurts the member more than the decision does"),
    ("restricting together", "cooperating together"),
    ("to the platform", "to product defaults"),
    ("families", "members"),
    ("family", "member"),
    ("parents", "members"),
    ("parent", "member"),
    ("screen-time", "decision"),
    ("screen time", "decision"),
    ("a real class", "the modelled situation"),
    ("conscientious households", "conscientious members"),
    ("households", "members"),
    ("household", "member"),
    ("total cohort", "total population"),
    ("cohort", "population"),
]


def _neutral_conclusion(sentence: str) -> str:
    for old, new in _NEUTRAL_CONCLUSIONS:
        sentence = sentence.replace(old, new)
    return sentence


_VOCABULARY_PLACEHOLDERS = {
    "who the players are, in the user's words",
    "the decision each player makes, in the user's words",
    "what cooperating looks like, in the user's words",
    "what defecting looks like, in the user's words",
}


def render_report(report: Mapping[str, Any]) -> str:
    """Render the report as Markdown."""
    lines = ["# Situation analysis", ""]
    summary = (report.get("situation_summary") or "").strip()
    lines.append(_math_text(summary) if summary else "_no situation summary_")
    lines.append("")

    lines.extend(["## Parameter assumptions", "",
                  "Parameters not established by the conversation may use paper defaults. "
                  "Results are conditional on the displayed values; paper defaults are "
                  "illustrative, not estimates for this situation.", ""])

    vocabulary = report.get("vocabulary")
    if not isinstance(vocabulary, Mapping):
        vocabulary = {}
    mapping = [(label, vocabulary.get(key))
               for label, key in (("Players", "players"), ("Decision", "decision"),
                                  ("Cooperating", "cooperation"), ("Defecting", "defection"))]
    mapping = [(label, value) for label, value in mapping
               if value and str(value).strip().lower() not in _VOCABULARY_PLACEHOLDERS]
    if mapping:
        lines.append("## Mapping")
        lines.append("")
        lines.extend(f"{label}: {_math_text(value)}" for label, value in mapping)
        lines.append("")

    lines.append("## Calculations summary")
    lines.append("")
    lines.append("| scenario | status | checks |")
    lines.append("| --- | --- | --- |")
    for item in report.get("scenarios", []):
        checks = item.get("checks", [])
        passed = sum(1 for check in checks if check.get("ok"))
        lines.append(f"| {_math_text(item.get('scenario', ''))} | {item.get('status', '')} | "
                     f"{passed}/{len(checks)} |")
    lines.append("")

    bullets = [str(bullet) for bullet in (report.get("practical_conclusions") or [])
               if str(bullet).strip()]
    if bullets:
        lines.append("## Practical conclusions")
        lines.append("")
        lines.extend(f"- {_math_text(bullet)}" for bullet in bullets)
        lines.append("")

    for item in report.get("scenarios", []):
        lines.append(f"## {item['scenario']} - {item['title']}")
        lines.append("")
        notes = [str(note) for note in item.get("param_notes", [])]
        if notes:
            lines.append("Parameter notes:")
            lines.append("")
            for note in notes:
                lines.append(f"- {_math_text(note)}")
            lines.append("")
        if item.get("status") == "error":
            lines.append(f"> scenario failed: {_math_text(item.get('error', 'unknown error'))}")
            lines.append("")
            if not item.get("results"):
                continue

        lines.append("| parameter | value |")
        lines.append("| --- | --- |")
        for name, value in item.get("params", {}).items():
            lines.append(f"| {_parameter_label(name)} | {_math_value(value)} |")
        lines.append("")


        lines.append("| strategy | outcome | values |")
        lines.append("| --- | --- | --- |")
        for row in item.get("strategies", []):
            values = "; ".join(f"{_math_text(label)} = {_math_value(value)}"
                               for label, value in row.get("values", {}).items())
            lines.append(f"| {_math_text(row.get('strategy', ''))} | "
                         f"{_math_text(row.get('outcome', ''))} | {values} |")
        lines.append("")

        lines.append("| result | value |")
        lines.append("| --- | --- |")
        for key, value in item.get("results", {}).items():
            lines.append(f"| {_result_label(key)} | {_math_value(value)} |")
        lines.append("")

        for check in item.get("checks", []):
            marker = "PASS" if check.get("ok") else "FAIL"
            lines.append(f"- [{marker}] {_math_text(check.get('name', ''))} "
                         f"(expected {_math_text(_math_value(check.get('expected')))}, "
                         f"got {_math_text(_math_value(check.get('got')))})")
        lines.append("")

        for sentence in item.get("conclusion", []):
            lines.append(f"-> {_math_text(sentence)}")
        lines.append("")

    totals = report.get("totals", {})
    total = totals.get("checks_total", 0)
    failed = totals.get("checks_failed", 0)
    lines.append("## Totals")
    lines.append("")
    lines.append(f"{totals.get('scenarios', 0)} scenarios run, {totals.get('ok', 0)} ok, "
                 f"{totals.get('errors', 0)} errors, {total} checks, {total - failed} passed, "
                 f"{failed} failed.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# General-game orchestration (never touches the paper path above).
# ---------------------------------------------------------------------------

from game_estimation import analyze_sensitivity, apply_estimates, estimate_game_parameters  # noqa: E402
from game_models import (GameEnvelope, GameSpecError, SCHEMA_VERSION, TEMPLATE_IDS,  # noqa: E402
                         compile_template)
from game_simulation import simulate_game  # noqa: E402
from game_solvers import (ALL_METHODS, EXACT_METHODS, MethodResult,  # noqa: E402
                          SolverError, solve_exact_game)

GAME_TEMPLATE_IDS = TEMPLATE_IDS

P3_TEMPLATE_SCOPE = ("Finite heterogeneous one-shot peer-share model. Empty options select the "
                     "labelled P3 paper benchmark. The paper's repeated-game construction is "
                     "outside this template.")


def list_game_templates() -> list[dict]:
    templates = []
    for template_id in GAME_TEMPLATE_IDS:
        template = {"id": template_id,
                    "options": gm_TEMPLATE_OPTIONS.get(template_id, [])}
        if template_id == "p3_heterogeneous":
            template["default_profile"] = "P3 paper benchmark (9 C, 1 D)"
            template["scope"] = P3_TEMPLATE_SCOPE
        templates.append(template)
    return templates


gm_TEMPLATE_OPTIONS = {
    "pricing_auction": ["auction_rule", "types", "prior", "bid_grids", "values", "tie_rule"],
    "security_patrol": ["players", "schedules", "targets", "coverage",
                        "covered_payoff", "uncovered_payoff", "commitment_mode"],
    "negotiation": ["offer_tree"],
    "bargaining": ["players", "feasible_points", "disagreement", "randomization"],
    "public_goods": ["players", "actions", "payoffs"],
    "mechanism_design": ["mechanism_kind", "design_mode", "type_spaces", "prior", "outcomes",
                         "feasible_outcomes", "value_table", "outside_option",
                         "allocation", "transfers", "ic_concept", "ir_scope",
                         "budget_rule", "objective"],
    "multi_agent_learning": ["game_id", "parameters"],
    "p3_heterogeneous": ["type_counts", "thresholds", "payoff_coefficients"],
    "custom": ["kind"],
}


def _method_result_to_json(result: MethodResult) -> dict:
    data = {"method": result.method, "status": result.status,
            "witnesses": result.witnesses, "checks": result.checks,
            "assumptions": result.assumptions}
    if result.objective is not None:
        from game_models import render_rational
        data["objective"] = render_rational(result.objective)
    if result.certificate is not None:
        data["certificate"] = result.certificate
    if result.epsilon is not None:
        from game_models import render_rational
        data["epsilon"] = render_rational(result.epsilon)
    if result.supremum is not None:
        from game_models import render_rational
        data["supremum"] = render_rational(result.supremum)
    return data


def _p3_utility_at(game, profile: Mapping[str, str]) -> dict[str, Fraction]:
    from game_models import evaluate_cell
    row = next(row for row in game.payoffs if row["action_profile"] == profile)
    return {player_id: evaluate_cell(value, {}) for player_id, value in row["utilities"].items()}


def _p3_benchmark_evidence(game, witnesses: list[Mapping[str, str]]) -> tuple[dict, list[dict]]:
    from game_models import render_rational

    player_ids = [player.id for player in game.players]
    all_a = {player.id: "A" for player in game.players}
    low_adoption = {player.id: ("A" if player.role == "D" else "R")
                    for player in game.players}
    expected_profiles = {
        tuple(all_a[player_id] for player_id in player_ids),
        tuple(low_adoption[player_id] for player_id in player_ids),
    }
    equilibria = []
    utilities_by_label = {}
    checks = []
    for profile in witnesses:
        utilities = _p3_utility_at(game, profile)
        if all(action == "A" for action in profile.values()):
            label = "all-A"
        elif dict(profile) == low_adoption:
            label = "low-adoption"
        else:
            label = "other"
        strict_checks = []
        for player in game.players:
            for action in game.actions[player.id]:
                if action == profile[player.id]:
                    continue
                deviation = dict(profile)
                deviation[player.id] = action
                loss = utilities[player.id] - _p3_utility_at(game, deviation)[player.id]
                strict_checks.append({
                    "name": f"{label}: {player.id} switching to {action} loses utility",
                    "ok": loss > 0,
                    "expected": "positive utility loss",
                    "got": render_rational(loss),
                })
        strict = all(check["ok"] for check in strict_checks)
        utilities_by_label[label] = utilities
        type_utilities = {}
        for role in {player.role for player in game.players}:
            role_utilities = {player.id: utilities[player.id]
                              for player in game.players if player.role == role}
            distinct = set(role_utilities.values())
            type_utilities[role] = (
                render_rational(next(iter(distinct))) if len(distinct) == 1
                else {player_id: render_rational(value)
                      for player_id, value in role_utilities.items()})
        equilibria.append({
            "label": label,
            "action_profile": dict(profile),
            "utility_by_type": type_utilities,
            "utility_by_player": {player_id: render_rational(value)
                                  for player_id, value in utilities.items()},
            "total_utility": render_rational(sum(utilities.values(), Fraction(0))),
            "strict_nash": strict,
        })
        checks.extend(strict_checks)

    profile_keys = {tuple(profile.get(player_id) for player_id in player_ids)
                    for profile in witnesses}
    strict_count = sum(row["strict_nash"] for row in equilibria)
    checks.extend([
        {"name": "P3 pure Nash set contains the two benchmark profiles",
         "ok": profile_keys == expected_profiles,
         "expected": "all-A and low-adoption",
         "got": sorted(row["label"] for row in equilibria)},
        {"name": "both P3 pure Nash profiles are strict",
         "ok": len(equilibria) == 2 and strict_count == 2,
         "expected": "2 strict profiles",
         "got": f"{len(equilibria)} profiles, {strict_count} strict"},
    ])

    welfare_comparison = None
    baseline = utilities_by_label.get("all-A")
    candidate = utilities_by_label.get("low-adoption")
    if baseline is not None and candidate is not None:
        changes_by_type = {}
        changes = {}
        for role in {player.role for player in game.players}:
            role_changes = {player.id: candidate[player.id] - baseline[player.id]
                            for player in game.players if player.role == role}
            changes.update(role_changes)
            distinct = set(role_changes.values())
            changes_by_type[role] = (
                render_rational(next(iter(distinct))) if len(distinct) == 1
                else {player_id: render_rational(value)
                      for player_id, value in role_changes.items()})
        pareto_improvement = all(change >= 0 for change in changes.values()) and any(
            change > 0 for change in changes.values())
        welfare_comparison = {
            "baseline": "all-A",
            "candidate": "low-adoption",
            "utility_change_by_type": changes_by_type,
            "pareto_improvement": pareto_improvement,
        }
        all_a_total = sum(baseline.values(), Fraction(0))
        low_total = sum(candidate.values(), Fraction(0))
        checks.extend([
            {"name": "P3 all-A total utility is 10",
             "ok": all_a_total == 10, "expected": "10",
             "got": render_rational(all_a_total)},
            {"name": "P3 low-adoption total utility is 223/18",
             "ok": low_total == Fraction(223, 18), "expected": "223/18",
             "got": render_rational(low_total)},
            {"name": "P3 low-adoption is not a Pareto improvement",
             "ok": not pareto_improvement and changes_by_type.get("D") == "-1",
             "expected": "D loses 1 utility unit",
             "got": changes_by_type},
        ])

    evidence = {
        "equilibrium_count": len(equilibria),
        "pure_nash_count": len(equilibria),
        "strict_pure_nash_count": strict_count,
        "equilibria": equilibria,
        "welfare_comparison": welfare_comparison,
    }
    return evidence, checks


def _p3_game_evidence(game, witnesses: list[Mapping[str, str]]) -> tuple[dict, list[dict]]:
    from game_models import render_rational

    equilibria = []
    checks = []
    strict_count = 0
    for index, profile in enumerate(witnesses, start=1):
        utilities = _p3_utility_at(game, profile)
        deviation_checks = []
        strict_deviations = []
        for player in game.players:
            for action in game.actions[player.id]:
                if action == profile[player.id]:
                    continue
                deviation = dict(profile)
                deviation[player.id] = action
                gain = _p3_utility_at(game, deviation)[player.id] - utilities[player.id]
                strict_deviations.append(gain < 0)
                deviation_checks.append({
                    "name": f"{player.id} switching to {action} has no profitable deviation",
                    "ok": gain <= 0,
                    "expected": "utility gain <= 0",
                    "got": render_rational(gain),
                })
        strict = all(strict_deviations)
        strict_count += strict
        checks.extend(deviation_checks)
        type_utilities = {}
        for role in dict.fromkeys(player.role for player in game.players):
            role_utilities = {
                player.id: utilities[player.id]
                for player in game.players if player.role == role
            }
            distinct = set(role_utilities.values())
            type_utilities[role] = (
                render_rational(next(iter(distinct))) if len(distinct) == 1
                else {player_id: render_rational(value)
                      for player_id, value in role_utilities.items()})
        equilibria.append({
            "label": f"equilibrium-{index}",
            "action_profile": dict(profile),
            "utility_by_type": type_utilities,
            "utility_by_player": {player_id: render_rational(value)
                                  for player_id, value in utilities.items()},
            "total_utility": render_rational(sum(utilities.values(), Fraction(0))),
            "strict_nash": strict,
        })
    return {
        "equilibrium_count": len(equilibria),
        "pure_nash_count": len(equilibria),
        "strict_pure_nash_count": strict_count,
        "equilibria": equilibria,
    }, checks


def _attach_p3_evidence(envelope: GameEnvelope,
                                  method_results: list[dict]) -> None:
    if envelope.template_id != "p3_heterogeneous":
        return
    benchmark = not envelope.template_options
    for result in method_results:
        if result.get("method") != "pure_nash":
            continue
        if benchmark:
            evidence, checks = _p3_benchmark_evidence(envelope.game, result["witnesses"])
            result["benchmark_evidence"] = evidence
            error = "P3 paper benchmark verification failed"
        else:
            evidence, checks = _p3_game_evidence(envelope.game, result["witnesses"])
            result["p3_evidence"] = evidence
            error = "P3 exact payoff verification failed"
        result["checks"].extend(checks)
        if not all(check["ok"] for check in checks):
            result["status"] = "error"
            result["error"] = error


def solve_game(game_spec: Mapping[str, Any], confirmed_by_user: bool,
               methods: list[str] | None = None, epsilon: Any = None,
               mediator_confirmed: bool = False) -> dict:
    """Run exact (and optionally epsilon) methods over a canonical game spec."""
    if not confirmed_by_user:
        return {"schema_version": SCHEMA_VERSION, "disposition": "needs_confirmation"}
    try:
        envelope = _build_envelope_from_payload(game_spec)
    except GameSpecError as exc:
        return {"schema_version": SCHEMA_VERSION, "disposition": "invalid", "error": str(exc)}
    requested = list(methods) if methods is not None else None
    if requested:
        known = set(ALL_METHODS)
        for name in requested:
            if name not in known:
                return {"schema_version": SCHEMA_VERSION, "disposition": "invalid",
                        "error": f"unknown method {name!r}"}
    epsilon_frac = None
    if epsilon is not None:
        from game_models import to_rational
        try:
            epsilon_frac = to_rational(epsilon, "epsilon")
        except (GameSpecError, TypeError, ValueError) as exc:
            return {"schema_version": SCHEMA_VERSION, "disposition": "invalid",
                    "error": f"epsilon must be an exact rational: {exc}"}
        if epsilon_frac <= 0:
            return {"schema_version": SCHEMA_VERSION, "disposition": "invalid",
                    "error": "epsilon must be a positive rational"}
    game = envelope.game
    incompatible = [name for name in (requested or []) if name != "epsilon_nash"
                    and not _method_compatible(name, game)]
    if requested and len(incompatible) == len([m for m in requested if m != "epsilon_nash"]) \
            and epsilon_frac is None:
        return {"schema_version": SCHEMA_VERSION, "disposition": "unsupported",
                "error": f"no requested method is compatible: {incompatible}"}
    method_results: list[dict] = []
    exact_enabled = requested is None or any(m != "epsilon_nash" for m in requested)
    if exact_enabled:
        try:
            for result in solve_exact_game(envelope, mediator_confirmed=mediator_confirmed,
                                           methods=requested):
                method_results.append(_method_result_to_json(result))
        except (GameSpecError, SolverError) as exc:
            method_results.append({"method": "solve_exact_game", "status": "error",
                                   "error": str(exc)})
    if epsilon_frac is not None and (requested is None or "epsilon_nash" in requested):
        from game_solvers import solve_epsilon_nash
        if type(game).__name__ == "NormalFormGame":
            method_results.append(_method_result_to_json(
                solve_epsilon_nash(game, epsilon_frac)))
        else:
            method_results.append({"method": "epsilon_nash", "status": "unsupported",
                                   "assumptions": ["epsilon_nash supports finite "
                                                   "normal-form games only"]})
    _attach_p3_evidence(envelope, method_results)
    return {"schema_version": SCHEMA_VERSION, "disposition": "ok",
            "normalized_game": _normalized_game_json(envelope),
            "method_results": method_results}


def _method_compatible(name: str, game) -> bool:
    from game_models import (BargainingSpec, BayesianGame, ExtensiveFormGame,
                             MechanismSpec, NormalFormGame, RepeatedGame, StackelbergGame)
    table = {
        "pure_nash": (NormalFormGame, StackelbergGame),
        "mixed_nash": (NormalFormGame, ExtensiveFormGame),
        "bayesian_nash": (BayesianGame,),
        "subgame_perfect": (ExtensiveFormGame,),
        "correlated_equilibrium": (NormalFormGame,),
        "nash_bargaining": (BargainingSpec,),
        "stackelberg": (StackelbergGame,),
        "mechanism_optimum": (MechanismSpec,),
    }
    return isinstance(game, table.get(name, ()))


def _build_envelope_from_payload(game_spec: Mapping[str, Any]) -> GameEnvelope:
    if not isinstance(game_spec, Mapping):
        raise GameSpecError("game_spec must be an object")
    template_id = game_spec.get("template_id", "custom")
    template_options = game_spec.get("template_options") or {}
    if template_id == "p3_heterogeneous":
        if game_spec.get("game") not in (None, {}):
            raise GameSpecError("p3_heterogeneous does not accept custom game data")
        if game_spec.get("parameters"):
            raise GameSpecError("p3_heterogeneous does not accept external parameters")
        if game_spec.get("commitment_rights", "none") != "none":
            raise GameSpecError("p3_heterogeneous does not use commitment rights")
        if game_spec.get("analysis_role", "observer") != "observer":
            raise GameSpecError("p3_heterogeneous uses the observer analysis role")
        envelope = compile_template(template_id, template_options)
        return _apply_envelope_extras(envelope, game_spec)
    if template_id != "custom" or game_spec.get("game"):
        payload = game_spec.get("game")
        if payload:
            envelope = compile_template(template_id, template_options, payload)
            return _apply_envelope_extras(envelope, game_spec)
    envelope = compile_template(template_id, template_options, game_spec.get("game") or game_spec)
    return _apply_envelope_extras(envelope, game_spec)


def _apply_envelope_extras(envelope: GameEnvelope, payload: Mapping[str, Any]) -> GameEnvelope:
    from game_models import ParameterSpec, to_rational
    for name, raw in (payload.get("parameters") or {}).items():
        if isinstance(raw, ParameterSpec):
            envelope.parameters[name] = raw
        elif isinstance(raw, Mapping):
            value = raw.get("value")
            envelope.parameters[name] = ParameterSpec(
                name=name,
                value=None if value is None else to_rational(value, f"parameter {name}"),
                lower_bound=None if raw.get("lower_bound") is None
                else to_rational(raw["lower_bound"], f"parameter {name} lower"),
                upper_bound=None if raw.get("upper_bound") is None
                else to_rational(raw["upper_bound"], f"parameter {name} upper"),
                prior=raw.get("prior"),
                unit=raw.get("unit", "utility"),
                provenance=raw.get("provenance") or {"origin": "user_stated",
                                                     "source": "user"})
    envelope.commitment_rights = payload.get("commitment_rights", envelope.commitment_rights)
    envelope.analysis_role = payload.get("analysis_role", envelope.analysis_role)
    if payload.get("assumptions"):
        envelope.assumptions.extend(payload["assumptions"])
    envelope.validate()
    return envelope


def _normalized_game_json(envelope: GameEnvelope) -> dict:
    game = envelope.game
    return {"kind": type(game).__name__,
            "template_id": envelope.template_id,
            "players": [p.to_json() for p in getattr(game, "players", [])],
            "assumptions": envelope.assumptions}


NO_CLARIFICATION_RULE = (
    "\n- The user has already been asked once. Do not ask again: choose the closest template "
    "and fill any detail you are missing with a sensible default."
)


def infer_game_model(messages: list[dict], client=None, config=None, asked_already=False) -> dict:
    """LLM intake for general games: never runs a worker."""
    config = config or load_config()
    if client is None:
        import openai
        client = openai.OpenAI(base_url=config.base_url, api_key=config.api_key or "none")
    system = build_game_intake_prompt() + (NO_CLARIFICATION_RULE if asked_already else "")
    history = [{"role": "system", "content": system}, *(messages or [])]
    for attempt in (1, 2):
        try:
            completion = client.chat.completions.create(model=config.model,
                                                        messages=history, temperature=0)
        except Exception as exc:
            raise InferenceError(f"LLM request failed: {exc}") from exc
        raw = completion.choices[0].message.content or ""
        try:
            parsed = _parse_json_object(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            if attempt == 1:
                history = history + [{"role": "assistant", "content": raw},
                                     {"role": "user",
                                      "content": f"Invalid JSON: {exc}. Reply with ONLY the "
                                                 "JSON object described."}]
                continue
            raise InferenceError(f"LLM returned invalid JSON: {raw[:400]}") from exc
        parsed["raw"] = raw
        parsed["model"] = config.model
        if parsed.get("disposition") == "clarification":
            parsed["questions"] = _intake_questions(parsed.get("questions"))
        return parsed
    raise InferenceError("LLM did not return parseable JSON")


MAX_INTAKE_QUESTIONS = 2


def _intake_questions(questions) -> list[str]:
    if not isinstance(questions, list):
        return []
    clean = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
    return clean[:MAX_INTAKE_QUESTIONS]


def build_game_intake_prompt() -> str:
    return "\n".join([
        "You are a game-theory intake analyst. Classify the user's situation.",
        "Reply with ONLY one JSON object of this shape:",
        json.dumps({
            "disposition": "clarification|unsupported|invalid|needs_confirmation",
            "questions": [],
            "template_id": "pricing_auction|security_patrol|negotiation|bargaining|"
                           "public_goods|mechanism_design|multi_agent_learning|"
                           "p3_heterogeneous|custom",
            "template_options": {},
            "game": None,
            "notes": "",
        }, indent=2),
        "Rules:",
        "- clarification: only when a whole side or a whole set of outcomes is still missing. "
        "Ask at most 2 questions, and only once in a conversation.",
        "- Write questions the way you would ask a friend who knows no game theory. Never use "
        "the words payoff, utility, Nash, equilibrium, strategy, player, type, threshold, or "
        "coefficient in a question. Ask about the real situation instead: who decides, what "
        "each side can do, and which result each side prefers.",
        "- Infer the game-theory structure yourself from the answers. Never ask the user to "
        "name a concept, and never ask for numbers the user has not mentioned unless a whole "
        "outcome is unscored.",
        "- unsupported: outside game theory (weather, chat, math homework).",
        "- invalid: game-theoretic but impossible to formalize with exact rationals.",
        "- needs_confirmation: a complete typed draft; fill template_id, template_options "
        "and, for custom, the canonical game object. Use exact rationals "
        "(integers, 'a/b', decimal strings). Never execute formulas; propose typed fields only.",
        "- Never ask about anything the user already stated. If players, their options and "
        "a score for every outcome are given, reply needs_confirmation, not clarification. "
        "Assume everyone chooses at the same time unless told otherwise.",
        "- p3_heterogeneous models a finite heterogeneous one-shot peer-share game, separate "
        "from homogeneous paper scenario tw3. Empty template_options select the clearly "
        "labelled P3 paper benchmark: 9 type-C players with tau=3/2, 1 type-D player "
        "with tau=-1, f(k)=k, and E(k)=k+k^2.",
        "- For other P3 games, never substitute benchmark defaults. Require complete "
        "type_counts, thresholds, and payoff_coefficients: f has constant and linear "
        "rational coefficients; E has constant, linear, and quadratic rational "
        "coefficients. Use u(A,k)=f(k), u(R,k)=tau-E(k), with tau the type's "
        "threshold and k the A share among the other players.",
        "- The template does not implement the paper's repeated-game construction. "
        "Current repeated-game support only simulates named public policies in "
        "two-player finite-horizon games; repeated-game equilibria are unsupported.",
        "Simultaneous game with stated scores: template_id custom and game shaped like this "
        "(one payoffs row for EVERY combination of options):",
        json.dumps({"kind": "normal_form",
                    "players": [{"id": "a", "name": "Cafe A"}, {"id": "b", "name": "Cafe B"}],
                    "actions": {"a": ["low", "high"], "b": ["low", "high"]},
                    "payoffs": [{"action_profile": {"a": "low", "b": "high"},
                                 "utilities": {"a": 7, "b": 1}}]}),
    ])
