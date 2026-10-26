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
        "gamma": ("share of aware parents signing the pact", "1/1000 <= gamma < 1 - eps; "
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
        "lam": ("slope of peer pressure per unit adoption; the effective pressure is lam * (1 - eta)",
                "lam > 0"),
        "eta": ("share of the group covered by the rule; uncovered peers keep the pressure",
                "0 <= eta < 1"),
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
         "outcome": "cutoff above which all-using is risk-dominant",
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
        {"strategy": "Pact of gamma aware members",
         "outcome": "adoption level held, contraction, trajectory, exit multiplier",
         "values": {"adoption level K": "K", "contraction a": "a",
                    "trajectory": "trajectory", "exit multiplier": "exit_multiplier"}},
    ],
    "tw7": [
        {"strategy": "Rule covers part of the group",
         "outcome": "effective pressure and resulting adoption limit",
         "values": {"effective pressure": "lambda_effective", "adoption limit": "limit"}},
        {"strategy": "No rule (eta = 0)",
         "outcome": "baseline: the same toolkit function called with eta = 0",
         "values": {"effective pressure": "lambda_effective", "adoption limit": "limit"},
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
        except (ValueError, TypeError):
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

    if scenario_id == "tw7":
        _clamp(values, notes, "lam", Fraction(1, 1000), None, "pressure slope must stay positive")
        _clamp(values, notes, "eta", Fraction(0), Fraction(999, 1000),
               "rule coverage must stay in [0, 1)")
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
        "- A stated share of aware parents signing a pact becomes gamma in tw6.",
        "- A stated share of the group covered by a rule becomes eta in tw7.",
        "- A stated evasion rate becomes eta in tw7_2.",
        "- A stated cost, benefit or harm relative to the full private benefit becomes tau, f1 or "
        "E1 in tw1 and f1 or E1 in tw3_2.",
        "- A stated screen-time limit becomes T in tw1_1.",
        "- Prefer a number stated in the conversation over the paper default.",
        "- Every number stated in the conversation that matches a parameter's meaning MUST be "
        "copied into that parameter; never return the paper default for a parameter the "
        "conversation quantifies.",
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
        "If the conversation does not yet pin down who the players are, what each player "
        'decides, and what cooperating vs defecting means, reply with ONLY '
        '{"need_more_information": true, "questions": [up to 3 short questions]} instead of '
        "the analysis. If the situation is clear, never ask questions about unstated numbers - "
        "analyze with the paper defaults and the existing caveat covers them.",
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
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt == 1:
                history = history + [{"role": "assistant", "content": raw},
                                     {"role": "user", "content": f"Invalid JSON: {last_error}. "
                                      "Reply with ONLY the JSON object described above."}]
                continue
            raise InferenceError(f"LLM did not return parseable JSON: {raw[:400]}") from exc
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
    "eta_critical": "the share of the group that makes the rule bite",
    "lambda_effective": "the pressure the rule creates",
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
    "tw7": "group-wide pressure reduction",
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
              "defection. Reply with ONLY a JSON object "
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
    if inference.get("need_more_information") and questions:
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
    "release_steps": "Release rounds", "eta_critical": r"$\eta^*$",
    "lambda_effective": r"$\lambda_{\mathrm{eff}}$", "lambda_reg": r"$\lambda_{\mathrm{reg}}$",
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
_MATH_TEXT_PATTERN = re.compile("|".join(f"(?P<m{i}>{pattern})" for i, (pattern, _) in enumerate(_MATH_TEXT)))
_MATH_TEXT_REPLACEMENTS = {f"m{i}": replacement for i, (_, replacement) in enumerate(_MATH_TEXT)}


def _math_text(value: Any) -> str:
    text = str(value)
    return _MATH_TEXT_PATTERN.sub(lambda match: _MATH_TEXT_REPLACEMENTS[match.lastgroup], text)


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
