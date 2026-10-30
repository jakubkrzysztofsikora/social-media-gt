"""Canonical game-model contracts for general game-theory analysis.

Every numeric payoff is an exact rational. A cell is either a rational literal
(int, "a/b", decimal string) or an affine function of declared parameters
({"constant": r, "coefficients": {param: r}}). No expression text is ever
evaluated. See plan decisions: exact claims need exact inputs.
"""
from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Mapping

SCHEMA_VERSION = 1

MAX_TOTAL_PROFILES = 10_000
MAX_TREE_NODES = 10_000
MAX_STACKELBERG_ACTIONS = 12
DECIMAL_RE = re.compile(r"^-?\d+(\.\d+)?$")


class GameSpecError(ValueError):
    """Structured rejection of a game specification."""


# ---------------------------------------------------------------- rationals

def to_rational(value: Any, where: str) -> Fraction:
    if isinstance(value, bool):
        raise GameSpecError(f"{where}: booleans are not exact numbers")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Fraction):
        return value
    if isinstance(value, str):
        text = value.strip()
        if "/" in text:
            num, _, den = text.partition("/")
            if not (num.strip().lstrip("-").isdigit() and den.strip().lstrip("-").isdigit()):
                raise GameSpecError(f"{where}: invalid fraction {value!r}")
            frac = Fraction(int(num), int(den))
            return frac
        if DECIMAL_RE.match(text):
            return Fraction(text)
        raise GameSpecError(f"{where}: {value!r} is not a rational literal")
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise GameSpecError(f"{where}: non-finite number")
        return Fraction(str(value))
    raise GameSpecError(f"{where}: expected exact number, got {type(value).__name__}")


def render_rational(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _prob(value: Any, where: str) -> Fraction:
    frac = to_rational(value, where)
    if frac < 0 or frac > 1:
        raise GameSpecError(f"{where}: probability outside [0,1]: {render_rational(frac)}")
    return frac


# ------------------------------------------------------------- payoff cells

@dataclass(frozen=True)
class Affine:
    """constant + sum(coefficients[p] * parameters[p]) over declared parameters."""
    constant: Fraction
    coefficients: Mapping[str, Fraction]

    def evaluate(self, values: Mapping[str, Fraction]) -> Fraction:
        total = self.constant
        for name, coef in self.coefficients.items():
            if name not in values:
                raise GameSpecError(f"parameter {name} has no assigned value")
            total += coef * values[name]
        return total

    def to_json(self) -> dict:
        return {"constant": render_rational(self.constant),
                "coefficients": {k: render_rational(v) for k, v in self.coefficients.items()}}

    @staticmethod
    def from_json(data: Mapping[str, Any], parameters: Mapping[str, Any], where: str) -> "Affine":
        if not isinstance(data, Mapping):
            raise GameSpecError(f"{where}: affine cell must be an object")
        constant = to_rational(data.get("constant", 0), f"{where}.constant")
        raw = data.get("coefficients", {})
        if not isinstance(raw, Mapping):
            raise GameSpecError(f"{where}.coefficients: must be an object")
        coefficients = {}
        for key, val in raw.items():
            if key not in parameters:
                raise GameSpecError(f"{where}.coefficients: unknown parameter {key!r}")
            coefficients[key] = to_rational(val, f"{where}.coefficients.{key}")
        return Affine(constant, coefficients)


def payoff_from_json(data: Any, parameters: Mapping[str, Any], where: str) -> Fraction | Affine:
    if isinstance(data, Mapping):
        return Affine.from_json(data, parameters, where)
    return to_rational(data, where)


def evaluate_cell(cell: Fraction | Affine, values: Mapping[str, Fraction]) -> Fraction:
    if isinstance(cell, Affine):
        return cell.evaluate(values)
    if isinstance(cell, bool):
        raise GameSpecError("booleans are not valid payoff values")
    return Fraction(cell)


# ------------------------------------------------------------ core entities

@dataclass
class PlayerSpec:
    id: str
    name: str = ""
    role: str = "player"
    utility_unit: str = "utility"
    normalization_basis: str = "stated"

    def __post_init__(self):
        if not self.name:
            self.name = self.id

    def to_json(self) -> dict:
        return {"id": self.id, "name": self.name, "role": self.role,
                "utility_unit": self.utility_unit, "normalization_basis": self.normalization_basis}


@dataclass
class ParameterSpec:
    name: str
    value: Fraction | None = None
    lower_bound: Fraction | None = None
    upper_bound: Fraction | None = None
    prior: list[dict] | None = None   # [{value, probability}] summing to one
    unit: str = "utility"
    normalization_basis: str = "stated"
    provenance: dict = field(default_factory=lambda: {"origin": "user_stated", "source": "user"})

    def mode(self) -> str:
        modes = sum(1 for m in (self.value is not None, self.lower_bound is not None
                                or self.upper_bound is not None, self.prior is not None) if m)
        if modes != 1:
            raise GameSpecError(f"parameter {self.name}: exactly one of point/interval/prior required")
        if self.value is not None:
            return "point"
        if self.prior is not None:
            total = sum(_prob(row["probability"], f"parameter {self.name} prior") for row in self.prior)
            if total != 1:
                raise GameSpecError(f"parameter {self.name}: prior must sum to one")
            return "prior"
        return "interval"

    def validate(self) -> str:
        mode = self.mode()
        if mode == "interval":
            if self.lower_bound is None or self.upper_bound is None:
                raise GameSpecError(f"parameter {self.name}: interval needs both bounds")
            if self.lower_bound > self.upper_bound:
                raise GameSpecError(f"parameter {self.name}: lower bound exceeds upper bound")
        return mode


@dataclass
class ObjectiveSpec:
    kind: str   # player_utility|social_welfare|revenue|maximin|weighted_utility
    direction: str = "maximize"
    player_id: str | None = None
    weights: Mapping[str, Fraction] | None = None

    def validate(self, players: list[str], is_mechanism: bool = False) -> None:
        if self.kind not in ("player_utility", "social_welfare", "revenue", "maximin", "weighted_utility"):
            raise GameSpecError(f"objective.kind {self.kind!r} unsupported")
        if self.direction not in ("maximize", "minimize"):
            raise GameSpecError(f"objective.direction {self.direction!r} unsupported")
        if self.kind == "player_utility" and self.player_id not in players:
            raise GameSpecError("objective.player_utility requires a valid player_id")
        if self.kind == "revenue" and not is_mechanism:
            raise GameSpecError("objective.revenue is valid only for mechanism design")
        if self.kind == "maximin":
            if self.direction != "maximize":
                raise GameSpecError("objective.maximin supports only maximize")
        if self.kind == "weighted_utility":
            if not self.weights or set(self.weights) != set(players):
                raise GameSpecError("objective.weighted_utility needs weights for every player")
            total = sum(self.weights.values())
            if any(w < 0 for w in self.weights.values()) or total != 1:
                raise GameSpecError("objective weights must be nonnegative and sum to one")


# ------------------------------------------------------------- game variants

@dataclass
class NormalFormGame:
    players: list[PlayerSpec]
    actions: Mapping[str, list[str]]                 # player_id -> ordered action ids
    payoffs: list[dict]                              # rows over the full Cartesian product
    objective: ObjectiveSpec | None = None

    def profile_key(self, profile: Mapping[str, str]) -> tuple:
        return tuple(profile[p.id] for p in self.players)

    def validate(self) -> None:
        players = [p.id for p in self.players]
        if not players or len(set(players)) != len(players):
            raise GameSpecError("normal-form game needs unique non-empty players")
        for pid in players:
            if pid not in self.actions or not self.actions[pid]:
                raise GameSpecError(f"player {pid}: missing action set")
        total = 1
        for pid in players:
            total *= len(self.actions[pid])
            if total > MAX_TOTAL_PROFILES:
                raise GameSpecError(f"normal-form game exceeds {MAX_TOTAL_PROFILES} pure profiles")
        if len(self.payoffs) != total:
            raise GameSpecError(f"normal-form payoffs must cover all {total} pure profiles; "
                                f"got {len(self.payoffs)}")
        seen = set()
        player_ids = set(players)
        for index, row in enumerate(self.payoffs):
            profile = row.get("action_profile", {})
            if set(profile) != player_ids:
                raise GameSpecError(f"payoff row {index}: action_profile must cover all players")
            for pid, aid in profile.items():
                if aid not in self.actions[pid]:
                    raise GameSpecError(f"payoff row {index}: unknown action {aid!r} for {pid}")
            key = self.profile_key(profile)
            if key in seen:
                raise GameSpecError(f"payoff row {index}: duplicate action profile")
            seen.add(key)
            utilities = row.get("utilities", {})
            if set(utilities) != player_ids:
                raise GameSpecError(f"payoff row {index}: utilities must cover all players")


@dataclass
class ExtensiveNode:
    id: str
    actor: str | None                      # None => terminal
    edges: list[dict] | None = None        # {action_id, next_node, probability?}
    information_set_id: str | None = None
    terminal_utilities: Mapping[str, Any] | None = None


@dataclass
class ExtensiveFormGame:
    players: list[PlayerSpec]
    root: str
    nodes: list[ExtensiveNode]
    information_sets: list[dict] = field(default_factory=list)  # {id,player_id,node_ids,action_ids}
    objective: ObjectiveSpec | None = None

    def validate(self) -> None:
        player_ids = {p.id for p in self.players}
        by_id: dict[str, ExtensiveNode] = {}
        for node in self.nodes:
            if node.id in by_id:
                raise GameSpecError(f"extensive node {node.id!r} duplicated")
            by_id[node.id] = node
        if self.root not in by_id:
            raise GameSpecError(f"extensive root {self.root!r} missing")
        if len(self.nodes) > MAX_TREE_NODES:
            raise GameSpecError(f"extensive game exceeds {MAX_TREE_NODES} nodes")
        # acyclicity + reachability + edge validity
        color: dict[str, int] = {}
        def visit(node_id: str) -> None:
            state = color.get(node_id, 0)
            if state == 1:
                raise GameSpecError(f"extensive game has a cycle at {node_id!r}")
            if state == 2:
                return
            color[node_id] = 1
            node = by_id[node_id]
            if node.actor is None:
                if not node.terminal_utilities:
                    raise GameSpecError(f"terminal node {node.id!r} needs terminal_utilities")
                for pid in node.terminal_utilities:
                    if pid not in player_ids:
                        raise GameSpecError(f"terminal node {node.id!r}: unknown player {pid!r}")
            else:
                if node.actor not in player_ids and node.actor != "chance":
                    raise GameSpecError(f"node {node.id!r}: unknown actor {node.actor!r}")
                if not node.edges:
                    raise GameSpecError(f"node {node.id!r}: non-terminal node needs edges")
                if node.actor == "chance":
                    total = sum(_prob(edge.get("probability"), f"chance edge in {node.id}")
                                for edge in node.edges)
                    if total != 1:
                        raise GameSpecError(f"chance node {node.id!r}: probabilities must sum to one")
                else:
                    if any("probability" in edge for edge in node.edges):
                        raise GameSpecError(f"node {node.id!r}: player edges carry no probabilities")
                for edge in node.edges:
                    if edge["next_node"] not in by_id:
                        raise GameSpecError(f"node {node.id!r}: edge target missing")
                    if node.actor != "chance":
                        visit(edge["next_node"])
            color[node_id] = 2
        visit(self.root)
        unreachable = [nid for nid in by_id if color.get(nid, 0) == 0]
        if unreachable:
            raise GameSpecError(f"extensive nodes unreachable from root: {unreachable[:5]}")
        # information sets
        for iset in self.information_sets:
            if iset["player_id"] not in player_ids:
                raise GameSpecError(f"information set {iset['id']}: unknown player")
            for nid in iset["node_ids"]:
                node = by_id.get(nid)
                if node is None or node.actor != iset["player_id"]:
                    raise GameSpecError(f"information set {iset['id']}: node {nid!r} not owned")
                node.information_set_id = iset["id"]
            edge_actions = [e["action_id"] for nid in iset["node_ids"]
                            for e in by_id[nid].edges or []]
            if set(edge_actions) != set(iset["action_ids"]):
                raise GameSpecError(f"information set {iset['id']}: action sets differ across nodes")

    def has_perfect_information(self) -> bool:
        by_id = {n.id: n for n in self.nodes}
        return all(len(iset["node_ids"]) == 1 for iset in self.information_sets) and \
            not any(n.actor == "chance" and len(n.edges or []) and False for n in by_id.values())


@dataclass
class BayesianGame:
    players: list[PlayerSpec]
    type_spaces: Mapping[str, list[str]]             # player_id -> type ids
    prior: list[dict]                                # [{type_profile:{pid:tid}, probability}]
    actions: Mapping[str, list[str]]
    payoffs: list[dict]                              # {type_profile, action_profile, utilities}
    objective: ObjectiveSpec | None = None

    def validate(self) -> None:
        players = [p.id for p in self.players]
        for pid in players:
            if pid not in self.type_spaces or not self.type_spaces[pid]:
                raise GameSpecError(f"player {pid}: missing type space")
            if pid not in self.actions or not self.actions[pid]:
                raise GameSpecError(f"player {pid}: missing action set")
        prior_total = sum(_prob(row["probability"], "bayesian prior") for row in self.prior)
        if prior_total != 1:
            raise GameSpecError("bayesian prior must sum to one")
        total = 1
        for pid in players:
            total *= len(self.type_spaces[pid]) * len(self.actions[pid])
        if total > MAX_TOTAL_PROFILES:
            raise GameSpecError(f"bayesian game exceeds {MAX_TOTAL_PROFILES} contingent profiles")
        seen_types = {tuple(sorted(row["type_profile"].items())) for row in self.prior}
        if len(seen_types) != len(self.prior):
            raise GameSpecError("bayesian prior has duplicate type profiles")
        seen = set()
        for row in self.payoffs:
            tp, ap = row["type_profile"], row["action_profile"]
            for pid in players:
                if tp.get(pid) not in self.type_spaces[pid] or ap.get(pid) not in self.actions[pid]:
                    raise GameSpecError(f"bayesian payoff row references unknown type/action for {pid}")
            key = (tuple(sorted(tp.items())), tuple(sorted(ap.items())))
            if key in seen:
                raise GameSpecError("bayesian payoffs have duplicate type/action profile")
            seen.add(key)
            if set(row["utilities"]) != set(players):
                raise GameSpecError("bayesian payoff utilities must cover all players")


@dataclass
class StackelbergGame:
    stage_game: NormalFormGame
    leader_id: str
    follower_id: str
    commitment_mode: str = "pure"                    # pure|mixed
    tie_policy: str = "report_both"

    def validate(self) -> None:
        if self.commitment_mode not in ("pure", "mixed"):
            raise GameSpecError("stackelberg.commitment_mode must be pure|mixed")
        if self.tie_policy != "report_both":
            raise GameSpecError("stackelberg.tie_policy is fixed to report_both")
        ids = [p.id for p in self.stage_game.players]
        if self.leader_id == self.follower_id or self.leader_id not in ids or self.follower_id not in ids:
            raise GameSpecError("stackelberg needs distinct leader and follower from the stage game")
        if len(ids) != 2:
            raise GameSpecError("stackelberg games support exactly two players")
        if len(self.stage_game.actions[self.leader_id]) > MAX_STACKELBERG_ACTIONS or \
                len(self.stage_game.actions[self.follower_id]) > MAX_STACKELBERG_ACTIONS:
            raise GameSpecError(f"stackelberg actions capped at {MAX_STACKELBERG_ACTIONS} per player")
        self.stage_game.validate()


@dataclass
class RepeatedGame:
    stage_game: NormalFormGame
    horizon: int
    discount_factor: Fraction
    monitoring: str = "perfect_public"

    def validate(self) -> None:
        if not isinstance(self.horizon, int) or self.horizon <= 0:
            raise GameSpecError("repeated.horizon must be a positive integer")
        if self.discount_factor < 0 or self.discount_factor > 1:
            raise GameSpecError("repeated.discount_factor must lie in [0,1]")
        if self.monitoring != "perfect_public":
            raise GameSpecError("repeated monitoring supports only perfect_public")
        if len(self.stage_game.players) != 2:
            raise GameSpecError("repeated games support exactly two players")
        self.stage_game.validate()
        for player in self.stage_game.players:
            if len(self.stage_game.actions[player.id]) != 2:
                raise GameSpecError("repeated named policies need two actions per player")


@dataclass
class BargainingSpec:
    players: list[PlayerSpec]                        # exactly two
    feasible_points: list[Mapping[str, Any]]         # {player_id: rational}
    disagreement: Mapping[str, Any]                  # {player_id: rational}
    randomization: str = "not_allowed"               # allowed requires binding contract right

    def validate(self) -> None:
        ids = [p.id for p in self.players]
        if len(ids) != 2 or len(set(ids)) != 2:
            raise GameSpecError("bargaining needs exactly two players")
        if self.randomization not in ("allowed", "not_allowed"):
            raise GameSpecError("bargaining.randomization must be allowed|not_allowed")
        if not self.feasible_points:
            raise GameSpecError("bargaining needs at least one feasible point")
        for index, point in enumerate(self.feasible_points):
            if set(point) != set(ids):
                raise GameSpecError(f"bargaining feasible point {index} must cover both players")
        if set(self.disagreement) != set(ids):
            raise GameSpecError("bargaining disagreement must cover both players")


@dataclass
class MechanismSpec:
    designer_id: str
    mechanism_kind: str                              # direct|vcg|optimal_auction
    design_mode: str                                 # verify_candidate|optimize|generate_vcg
    players: list[PlayerSpec]
    type_spaces: Mapping[str, list[str]]
    prior: list[dict]                                # prior over type profiles
    outcomes: list[str]
    feasible_outcomes: Mapping[str, list[str]]       # report-profile key -> outcome ids
    value_table: Mapping[str, Mapping[str, Mapping[str, Any]]]   # pid -> type -> outcome -> value
    outside_option: Mapping[str, Mapping[str, Any]]  # pid -> type -> value
    allocation: Mapping[str, Mapping[str, float]] | None = None  # report-profile -> outcome -> prob
    transfers: Mapping[str, Mapping[str, Any]] | None = None     # report-profile -> pid -> payment
    utility_model: str = "quasilinear"
    ic_concept: str = "DSIC"
    ir_scope: str = "ex_post"
    budget_rule: str = "none"
    objective: ObjectiveSpec | None = None
    vcg_tie_rule: str = "lexicographic_outcome_id"
    optout_outcomes: Mapping[str, str] | None = None   # player_id -> outcome id
    outcome_owners: Mapping[str, list[str]] | None = None   # outcome_id -> allocated players

    def report_key(self, reports: Mapping[str, str]) -> str:
        return json.dumps({pid: reports[pid] for pid in sorted(reports)}, sort_keys=True)

    def validate(self) -> None:
        if self.mechanism_kind not in ("direct", "vcg", "optimal_auction"):
            raise GameSpecError(f"mechanism.kind {self.mechanism_kind!r} unsupported")
        if self.design_mode not in ("verify_candidate", "optimize", "generate_vcg"):
            raise GameSpecError(f"mechanism.design_mode {self.design_mode!r} unsupported")
        if self.utility_model != "quasilinear":
            raise GameSpecError("mechanism.utility_model supports only quasilinear")
        if self.ic_concept not in ("DSIC", "BIC"):
            raise GameSpecError("mechanism.ic_concept must be DSIC|BIC")
        if self.ir_scope not in ("ex_post", "interim", "ex_ante"):
            raise GameSpecError("mechanism.ir_scope must be ex_post|interim|ex_ante")
        if self.budget_rule not in ("none", "no_deficit", "budget_balance"):
            raise GameSpecError("mechanism.budget_rule must be none|no_deficit|budget_balance")
        if self.mechanism_kind == "vcg":
            if self.ic_concept != "DSIC" or self.ir_scope != "ex_post":
                raise GameSpecError("vcg fixes DSIC and ex-post IR")
            if self.budget_rule != "none":
                raise GameSpecError("vcg fixes no budget constraint")
        if self.mechanism_kind == "optimal_auction":
            if self.ic_concept != "BIC" or self.ir_scope != "interim":
                raise GameSpecError("optimal_auction fixes BIC and interim IR")
            if self.objective is None or self.objective.kind != "revenue":
                raise GameSpecError("optimal_auction requires the revenue objective")
        if self.mechanism_kind == "vcg":
            players = [p.id for p in self.players]
            if not self.optout_outcomes or set(self.optout_outcomes) != set(players):
                raise GameSpecError("vcg needs an explicit opt-out outcome per player")
            for pid, outcome in self.optout_outcomes.items():
                if outcome not in self.outcomes:
                    raise GameSpecError(f"vcg opt-out outcome {outcome!r} unknown")
            if not self.outcome_owners or set(self.outcome_owners) != set(self.outcomes):
                raise GameSpecError("vcg needs outcome_owners for every outcome")
            for outcome, owners in self.outcome_owners.items():
                unknown = [o for o in owners if o not in players]
                if unknown:
                    raise GameSpecError(f"outcome {outcome!r} names unknown players {unknown}")
            # VCG efficiency and pivot payments require the feasible set to be
            # identical under every report profile
            sets = {tuple(sorted(outcomes)) for outcomes in self.feasible_outcomes.values()}
            if len(sets) != 1:
                raise GameSpecError("vcg requires a report-invariant feasible outcome set")
        if self.objective is None:
            raise GameSpecError("mechanism design requires an objective")
        players = [p.id for p in self.players]
        self.objective.validate(players, is_mechanism=True)
        prior_total = sum(_prob(row["probability"], "mechanism prior") for row in self.prior)
        if prior_total != 1:
            raise GameSpecError("mechanism prior must sum to one")
        if len(set(self.outcomes)) != len(self.outcomes) or not self.outcomes:
            raise GameSpecError("mechanism outcomes must be unique and non-empty")
        for pid in players:
            if pid not in self.type_spaces or not self.type_spaces[pid]:
                raise GameSpecError(f"mechanism player {pid}: missing type space")
        # value table complete
        for pid in players:
            for tid in self.type_spaces[pid]:
                table = self.value_table.get(pid, {}).get(tid)
                if table is None or set(table) != set(self.outcomes):
                    raise GameSpecError(f"value_table[{pid}][{tid}] must cover every outcome")
        if self.allocation is not None:
            for key, dist in self.allocation.items():
                total = sum(_prob(p, f"allocation[{key}]") for p in dist.values())
                if total != 1:
                    raise GameSpecError(f"allocation[{key}] probabilities must sum to one")
                for outcome in dist:
                    if outcome not in self.outcomes:
                        raise GameSpecError(f"allocation[{key}]: unknown outcome {outcome!r}")


@dataclass
class OpenSpielGame:
    game_id: str
    players: list[PlayerSpec]
    parameters: Mapping[str, Any] = field(default_factory=dict)

    ALLOWLIST = ("kuhn_poker", "leduc_poker", "matrix_pd", "tic_tac_toe")
    CFR_GAMES = ("kuhn_poker", "leduc_poker")

    def validate(self) -> None:
        if self.game_id not in self.ALLOWLIST:
            raise GameSpecError(f"open_spiel game {self.game_id!r} is not allowlisted")


GameSpec = (NormalFormGame | ExtensiveFormGame | BayesianGame | StackelbergGame
            | RepeatedGame | BargainingSpec | MechanismSpec | OpenSpielGame)


@dataclass
class GameEnvelope:
    """Top-level spec wrapper with provenance and assumptions."""
    schema_version: int = SCHEMA_VERSION
    template_id: str = "custom"
    template_options: dict = field(default_factory=dict)
    game: GameSpec | None = None
    parameters: dict[str, ParameterSpec] = field(default_factory=dict)
    commitment_rights: str = "none"                  # none|leader|mediator|designer
    analysis_role: str = "observer"                  # observer|leader|follower|designer|mediator
    assumptions: list[str] = field(default_factory=list)
    provenance: dict[str, dict] = field(default_factory=dict)   # json_pointer -> {origin,source}

    def point_parameters(self) -> dict[str, Fraction]:
        values = {}
        for name, spec in self.parameters.items():
            if spec.validate() != "point":
                raise GameSpecError(
                    f"parameter {name}: solves require point values; "
                    "intervals feed sensitivity and priors only explicit Bayesian models")
            values[name] = spec.value
        return values

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise GameSpecError(f"schema_version must be {SCHEMA_VERSION}")
        for spec in self.parameters.values():
            spec.validate()
        if self.game is None:
            raise GameSpecError("envelope needs a game")
        self.game.validate()


def build_envelope(game: GameSpec, *, parameters=None, template_id="custom",
                   template_options=None, commitment_rights="none",
                   analysis_role="observer", assumptions=None, provenance=None) -> GameEnvelope:
    envelope = GameEnvelope(template_id=template_id, template_options=dict(template_options or {}),
                            game=game, parameters=dict(parameters or {}),
                            commitment_rights=commitment_rights, analysis_role=analysis_role,
                            assumptions=list(assumptions or []), provenance=dict(provenance or {}))
    envelope.validate()
    return envelope


# ---------------------------------------------------------------- templates

TEMPLATE_IDS = ("pricing_auction", "security_patrol", "negotiation", "bargaining",
                "public_goods", "mechanism_design", "multi_agent_learning",
                "p3_heterogeneous", "custom")


def _players_from(options: Mapping[str, Any]) -> list[PlayerSpec]:
    raw = options.get("players")
    if not isinstance(raw, list) or not raw:
        raise GameSpecError("template options need a players list")
    players = []
    for entry in raw:
        if isinstance(entry, str):
            players.append(PlayerSpec(id=entry, name=entry))
        elif isinstance(entry, Mapping):
            players.append(PlayerSpec(id=entry["id"], name=entry.get("name", entry["id"]),
                                      role=entry.get("role", "player"),
                                      utility_unit=entry.get("utility_unit", "utility")))
        else:
            raise GameSpecError("template players entries must be ids or objects")
    return players


def _compile_pricing_auction(options: Mapping[str, Any]) -> GameEnvelope:
    rule = options.get("auction_rule")
    if rule not in ("first_price", "second_price"):
        raise GameSpecError("pricing_auction.auction_rule must be first_price|second_price")
    tie_rule = options.get("tie_rule", "player_order")
    if tie_rule not in ("player_order", "uniform_random"):
        raise GameSpecError("pricing_auction.tie_rule must be player_order|uniform_random")
    players = _players_from(options)
    ids = [p.id for p in players]
    types = options.get("types")
    if not isinstance(types, Mapping) or set(types) != set(ids):
        raise GameSpecError("pricing_auction needs finite types per player")
    prior = options.get("prior")
    if not isinstance(prior, list) or not prior:
        raise GameSpecError("pricing_auction needs a finite type prior")
    bids = options.get("bid_grids")
    if not isinstance(bids, Mapping) or set(bids) != set(ids):
        raise GameSpecError("pricing_auction needs a bid grid per player")
    for pid in ids:
        if not bids[pid] or len(set(bids[pid])) != len(bids[pid]):
            raise GameSpecError(f"pricing_auction bid grid for {pid} must be non-empty and unique")
    # Type payoffs: value minus payment for each type-profile/bid-profile pair.
    payoffs: list[dict] = []
    type_profiles: list[dict] = []

    def type_profiles_rec(index: int, current: dict) -> None:
        if index == len(ids):
            type_profiles.append(dict(current))
            return
        for tid in types[ids[index]]:
            current[ids[index]] = tid
            type_profiles_rec(index + 1, current)
    type_profiles_rec(0, {})

    prior_total = sum(_prob(row["probability"], "auction prior") for row in prior)
    if prior_total != 1:
        raise GameSpecError("pricing_auction prior must sum to one")
    prior_keys = {tuple(sorted(row["type_profile"].items())) for row in prior}
    if prior_keys != {tuple(sorted(tp.items())) for tp in type_profiles}:
        raise GameSpecError("pricing_auction prior must cover every type profile exactly once")

    def bid_profiles_rec(index: int, current: dict, bid_lists: list[list]) -> None:
        if index == len(bid_lists):
            payoffs.append(dict(current))
            return
        for bid in bid_lists[index]:
            current[ids[index]] = bid
            bid_profiles_rec(index + 1, current, bid_lists)
    bid_profiles_rec(0, {}, [bids[pid] for pid in ids])

    typed_payoffs: list[dict] = []
    for tp in type_profiles:
        for profile in payoffs:
            top = max(profile[pid] for pid in ids)
            winners = [pid for pid in ids if profile[pid] == top]
            if tie_rule == "uniform_random":
                share = Fraction(1, len(winners))
                winner_share = {pid: (share if pid in winners else Fraction(0)) for pid in ids}
            else:
                winner = min(winners, key=ids.index)
                winner_share = {pid: (Fraction(1) if pid == winner else Fraction(0))
                                for pid in ids}
            losing = [profile[pid] for pid in ids if pid not in winners]
            second = max(losing) if losing else Fraction(0)
            payment = top if rule == "first_price" else second
            utilities = {}
            for pid in ids:
                value = to_rational(options["values"][pid][tp[pid]], f"values[{pid}][{tp[pid]}]")
                utilities[pid] = value * winner_share[pid] - payment * winner_share[pid]
            typed_payoffs.append({"type_profile": dict(tp), "action_profile": dict(profile),
                                  "utilities": utilities})
    game = BayesianGame(players=players, type_spaces={pid: list(types[pid]) for pid in ids},
                        prior=prior, actions={pid: list(bids[pid]) for pid in ids},
                        payoffs=typed_payoffs)
    return build_envelope(game, template_id="pricing_auction", template_options=dict(options),
                          assumptions=[f"auction_rule={rule}", f"tie_rule={tie_rule}",
                                       "values and bids are exact rationals"])


def _compile_security_patrol(options: Mapping[str, Any]) -> GameEnvelope:
    players = _players_from(options)
    if len(players) != 2:
        raise GameSpecError("security_patrol needs defender (leader) and attacker (follower)")
    leader, follower = players
    schedules = options.get("schedules")
    targets = options.get("targets")
    coverage = options.get("coverage")
    if not schedules or not targets:
        raise GameSpecError("security_patrol needs schedules and targets")
    if not isinstance(coverage, Mapping):
        raise GameSpecError("security_patrol needs coverage probabilities per schedule")
    covered = options.get("covered_payoff")
    uncovered = options.get("uncovered_payoff")
    if covered is None or uncovered is None:
        raise GameSpecError("security_patrol needs covered/uncovered payoff values")
    payoffs = []
    for schedule in schedules:
        probs = coverage.get(schedule)
        if not isinstance(probs, Mapping) or set(probs) != set(targets):
            raise GameSpecError(f"coverage for schedule {schedule!r} must cover every target")
        for target in targets:
            p = _prob(probs[target], f"coverage[{schedule}][{target}]")
            c_val = to_rational(covered, "covered_payoff")
            u_val = to_rational(uncovered, "uncovered_payoff")
            payoffs.append({
                "action_profile": {leader.id: schedule, follower.id: target},
                "utilities": {leader.id: p * c_val + (1 - p) * u_val,
                              follower.id: -(p * c_val + (1 - p) * u_val)}})
    stage = NormalFormGame(players=players, actions={leader.id: list(schedules),
                                                    follower.id: list(targets)}, payoffs=payoffs)
    game = StackelbergGame(stage_game=stage, leader_id=leader.id, follower_id=follower.id,
                           commitment_mode=options.get("commitment_mode", "pure"))
    return build_envelope(game, template_id="security_patrol", template_options=dict(options),
                          commitment_rights="leader", analysis_role="leader",
                          assumptions=["coverage probabilities are exact and known to both sides"])


def _compile_bargaining(options: Mapping[str, Any]) -> GameEnvelope:
    players = _players_from(options)
    ids = [p.id for p in players]
    points = options.get("feasible_points")
    disagreement = options.get("disagreement")
    randomization = options.get("randomization", "not_allowed")
    if not isinstance(points, list) or not points or not isinstance(disagreement, Mapping):
        raise GameSpecError("bargaining template needs feasible_points and disagreement")
    game = BargainingSpec(players=players,
                          feasible_points=points, disagreement=disagreement,
                          randomization=randomization)
    envelope = build_envelope(game, template_id="bargaining", template_options=dict(options),
                              commitment_rights="mediator" if randomization == "allowed" else "none",
                              assumptions=["Nash product solution; randomization needs a binding contract"])
    return envelope


def _compile_public_goods(options: Mapping[str, Any]) -> GameEnvelope:
    players = _players_from(options)
    ids = [p.id for p in players]
    actions = options.get("actions")
    payoffs = options.get("payoffs")
    if not isinstance(actions, Mapping) or set(actions) != set(ids):
        raise GameSpecError("public_goods needs an action set per player")
    if not isinstance(payoffs, list) or not payoffs:
        raise GameSpecError("public_goods needs a complete payoff table")
    game = NormalFormGame(players=players,
                          actions={pid: list(actions[pid]) for pid in ids},
                          payoffs=payoffs)
    return build_envelope(game, template_id="public_goods", template_options=dict(options),
                          assumptions=["contributions and payoffs are exact and known "
                                       "to every contributor"])


def _compile_negotiation(options: Mapping[str, Any]) -> GameEnvelope:
    players = _players_from(options)
    tree = options.get("offer_tree")
    if not isinstance(tree, Mapping):
        raise GameSpecError("negotiation needs an offer_tree object with root and nodes")
    root = tree.get("root")
    raw_nodes = tree.get("nodes")
    if not root or not isinstance(raw_nodes, list) or not raw_nodes:
        raise GameSpecError("negotiation offer_tree needs a root id and a nodes list")
    nodes, information_sets = [], []
    for raw in raw_nodes:
        node_id = raw.get("id")
        if not node_id:
            raise GameSpecError("negotiation offer_tree node needs an id")
        actor = raw.get("actor")
        if actor is None:
            terminal = raw.get("terminal_utilities")
            if not isinstance(terminal, Mapping):
                raise GameSpecError(f"negotiation terminal {node_id!r} needs terminal_utilities")
            nodes.append(ExtensiveNode(id=node_id, actor=None,
                                       terminal_utilities=dict(terminal)))
            continue
        edges = raw.get("edges")
        if not isinstance(edges, list) or not edges:
            raise GameSpecError(f"negotiation node {node_id!r} needs offer edges")
        action_ids = []
        for edge in edges:
            if not edge.get("action_id") or not edge.get("next_node"):
                raise GameSpecError(f"negotiation edge in {node_id!r} needs action_id and "
                                    "next_node")
            action_ids.append(edge["action_id"])
        if len(set(action_ids)) != len(action_ids):
            raise GameSpecError(f"negotiation node {node_id!r} repeats an offer action")
        nodes.append(ExtensiveNode(id=node_id, actor=actor, edges=list(edges)))
        if actor != "chance":
            information_sets.append({"id": f"{node_id}_info", "player_id": actor,
                                     "node_ids": [node_id], "action_ids": action_ids})
    game = ExtensiveFormGame(players=players, root=root, nodes=nodes,
                             information_sets=information_sets)
    return build_envelope(game, template_id="negotiation", template_options=dict(options),
                          assumptions=["finite offer tree with perfect information",
                                       "an accepted offer is binding for both sides"])


def _compile_mechanism_design(options: Mapping[str, Any]) -> GameEnvelope:
    players = _players_from(options)
    kind = options.get("mechanism_kind", "direct")
    mode = options.get("design_mode", "verify_candidate")
    spec = MechanismSpec(
        designer_id=options.get("designer_id", "designer"),
        mechanism_kind=kind, design_mode=mode, players=players,
        type_spaces=options.get("type_spaces", {}),
        prior=options.get("prior", []),
        outcomes=options.get("outcomes", []),
        feasible_outcomes=options.get("feasible_outcomes", {}),
        value_table=options.get("value_table", {}),
        outside_option=options.get("outside_option", {}),
        allocation=options.get("allocation"),
        transfers=options.get("transfers"),
        ic_concept=options.get("ic_concept", "DSIC"),
        ir_scope=options.get("ir_scope", "ex_post"),
        budget_rule=options.get("budget_rule", "none"),
        objective=ObjectiveSpec(**options["objective"]) if isinstance(options.get("objective"), Mapping)
        else options.get("objective"),
    )
    return build_envelope(spec, template_id="mechanism_design", template_options=dict(options),
                          commitment_rights="designer", analysis_role="designer",
                          assumptions=["quasilinear utility; transfers positive from player to designer"])


def _compile_multi_agent_learning(options: Mapping[str, Any]) -> GameEnvelope:
    game = OpenSpielGame(game_id=options.get("game_id", ""), players=_players_from(options),
                         parameters=dict(options.get("parameters", {})))
    return build_envelope(game, template_id="multi_agent_learning", template_options=dict(options),
                          assumptions=["CFR/MCCFR approximate results only"])


P3_PAPER_BENCHMARK_OPTIONS = {
    "type_counts": {"C": 9, "D": 1},
    "thresholds": {"C": "3/2", "D": "-1"},
    "payoff_coefficients": {
        "f": {"constant": "0", "linear": "1"},
        "E": {"constant": "0", "linear": "1", "quadratic": "1"},
    },
}


def _compile_p3_heterogeneous(options: Mapping[str, Any]) -> GameEnvelope:
    benchmark = not options
    expected = {"type_counts", "thresholds", "payoff_coefficients"}
    if benchmark:
        raw = P3_PAPER_BENCHMARK_OPTIONS
    else:
        if set(options) != expected:
            raise GameSpecError(
                "p3_heterogeneous user options require all of type_counts, thresholds, "
                "and payoff_coefficients; empty options select the labelled paper benchmark")
        raw = options

    type_counts = raw["type_counts"]
    thresholds_raw = raw["thresholds"]
    if not isinstance(type_counts, Mapping) or not type_counts:
        raise GameSpecError("p3_heterogeneous.type_counts must be a non-empty mapping")
    if not all(isinstance(role, str) and role.strip() for role in type_counts):
        raise GameSpecError("p3_heterogeneous type names must be non-empty strings")
    counts = {}
    for role, count in type_counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise GameSpecError(f"p3_heterogeneous type count for {role!r} must be a positive integer")
        counts[role] = count
    if not isinstance(thresholds_raw, Mapping) or set(thresholds_raw) != set(counts):
        raise GameSpecError("p3_heterogeneous.thresholds must cover every player type exactly")
    thresholds = {
        role: to_rational(value, f"p3_heterogeneous.thresholds.{role}")
        for role, value in thresholds_raw.items()
    }

    raw_coefficients = raw["payoff_coefficients"]
    if not isinstance(raw_coefficients, Mapping) or set(raw_coefficients) != {"f", "E"}:
        raise GameSpecError("p3_heterogeneous.payoff_coefficients must define exactly f and E")
    coefficient_terms = {
        "f": {"constant", "linear"},
        "E": {"constant", "linear", "quadratic"},
    }
    coefficients = {}
    for function, terms in coefficient_terms.items():
        raw_terms = raw_coefficients[function]
        if not isinstance(raw_terms, Mapping) or set(raw_terms) != terms:
            expected_terms = ", ".join(sorted(terms))
            raise GameSpecError(
                f"p3_heterogeneous payoff coefficients for {function} must define "
                f"exactly {expected_terms}")
        coefficients[function] = {
            term: to_rational(value, f"p3_heterogeneous.payoff_coefficients.{function}.{term}")
            for term, value in raw_terms.items()
        }

    total_players = sum(counts.values())
    if total_players < 2:
        raise GameSpecError("p3_heterogeneous needs at least two players")
    if total_players > MAX_TOTAL_PROFILES.bit_length() - 1:
        raise GameSpecError(
            f"p3_heterogeneous exceeds {MAX_TOTAL_PROFILES} pure profiles")

    players = []
    for role, count in counts.items():
        for type_index in range(1, count + 1):
            player_id = (f"C{type_index}" if role == "C" else "D") if benchmark else \
                f"p{len(players) + 1}"
            players.append(PlayerSpec(
                id=player_id,
                name=player_id if benchmark else f"{role} {type_index}",
                role=role,
                utility_unit="per-period utility",
                normalization_basis="P3 paper benchmark" if benchmark else
                "user-specified P3 payoffs"))
    ids = [player.id for player in players]
    actions = {player_id: ["R", "A"] for player_id in ids}
    payoffs = []
    for action_profile_tuple in itertools.product(("R", "A"), repeat=total_players):
        action_profile = dict(zip(ids, action_profile_tuple))
        adopting = sum(action == "A" for action in action_profile_tuple)
        utilities = {}
        for player in players:
            peer_share = Fraction(
                adopting - (action_profile[player.id] == "A"), total_players - 1)
            f_value = (coefficients["f"]["constant"]
                       + coefficients["f"]["linear"] * peer_share)
            e_value = (coefficients["E"]["constant"]
                       + coefficients["E"]["linear"] * peer_share
                       + coefficients["E"]["quadratic"] * peer_share ** 2)
            utilities[player.id] = (
                f_value if action_profile[player.id] == "A"
                else thresholds[player.role] - e_value)
        payoffs.append({"action_profile": action_profile, "utilities": utilities})
    game = NormalFormGame(players=players, actions=actions, payoffs=payoffs)

    show = lambda value: render_rational(value)
    type_summary = ", ".join(f"{role}={count}" for role, count in counts.items())
    threshold_summary = ", ".join(
        f"{role}={show(thresholds[role])}" for role in counts)
    coefficient_summary = ", ".join(
        f"{function}({', '.join(f'{term}={show(value)}' for term, value in values.items())})"
        for function, values in coefficients.items())
    assumptions = [
        "P3 paper benchmark preset." if benchmark
        else "User-specified P3 one-shot model; not the paper benchmark.",
        f"Type counts: {type_summary}; thresholds: {threshold_summary}.",
        "Actions are R (resist) and A (adopt); k is the A share among the other "
        f"{total_players - 1} players.",
        "Exact payoffs are u(A,k)=f(k) and u(R,k)=tau_type-E(k), with "
        f"f(k)=f_constant+f_linear*k; E(k)=E_constant+E_linear*k+E_quadratic*k^2. "
        f"Coefficients: {coefficient_summary}.",
        "The paper's repeated-game construction is outside this finite one-shot template.",
    ]
    return build_envelope(game, template_id="p3_heterogeneous",
                          template_options={} if benchmark else dict(options),
                          assumptions=assumptions)


_COMPILERS = {
    "pricing_auction": _compile_pricing_auction,
    "security_patrol": _compile_security_patrol,
    "bargaining": _compile_bargaining,
    "negotiation": _compile_negotiation,
    "public_goods": _compile_public_goods,
    "mechanism_design": _compile_mechanism_design,
    "multi_agent_learning": _compile_multi_agent_learning,
    "p3_heterogeneous": _compile_p3_heterogeneous,
}


def compile_template(template_id: str, template_options: Mapping[str, Any],
                     game_data: Mapping[str, Any] | None = None) -> GameEnvelope:
    """Compile a typed template into a validated GameSpec envelope.

    Templates compile typed inputs and fixed benchmarks. Calculations use exact rational
    arithmetic only; no supplied expression text is evaluated.
    """
    if template_id not in TEMPLATE_IDS:
        raise GameSpecError(f"unknown template {template_id!r}")
    options = dict(template_options or {})
    if template_id == "p3_heterogeneous" and game_data not in (None, {}):
        raise GameSpecError("p3_heterogeneous does not accept custom game data")
    if template_id == "custom":
        return _compile_custom(game_data or {})
    compiler = _COMPILERS.get(template_id)
    if compiler is None:
        raise GameSpecError(f"template {template_id!r} has no compiler")
    return compiler(options)


def _decode_players(raw):
    if not raw:
        return []
    out = []
    for entry in raw:
        if isinstance(entry, str):
            out.append(PlayerSpec(id=entry))
        elif isinstance(entry, Mapping):
            out.append(PlayerSpec(id=entry["id"], name=entry.get("name", ""),
                                  role=entry.get("role", "player"),
                                  utility_unit=entry.get("utility_unit", "utility")))
        else:
            raise GameSpecError("players entries must be ids or objects")
    return out


def _decode_custom(kind: str, payload: dict):
    """Coerce JSON payload fields into the dataclass constructor's types."""
    needs_players = kind in ("normal_form", "extensive_form", "bayesian", "bargaining",
                             "mechanism", "open_spiel")
    base = {"players": _decode_players(payload.get("players"))} if needs_players else {}
    if kind == "normal_form":
        base["actions"] = payload.get("actions") or {}
        base["payoffs"] = payload.get("payoffs") or []
    if kind == "extensive_form":
        base["root"] = payload.get("root")
        nodes = []
        for node in payload.get("nodes") or []:
            nodes.append(ExtensiveNode(
                id=node["id"], actor=node.get("actor"),
                edges=node.get("edges"), information_set_id=node.get("information_set_id"),
                terminal_utilities=node.get("terminal_utilities")))
        base["root"] = payload.get("root")
        base["nodes"] = nodes
        base["information_sets"] = payload.get("information_sets") or []
    if kind == "bayesian":
        base["type_spaces"] = payload.get("type_spaces") or {}
        base["prior"] = payload.get("prior") or []
        base["actions"] = payload.get("actions") or {}
        base["payoffs"] = payload.get("payoffs") or []
    if kind == "stackelberg":
        stage = payload.get("stage_game") or {}
        base["stage_game"] = NormalFormGame(
            players=_decode_players(stage.get("players")),
            actions=stage.get("actions") or {}, payoffs=stage.get("payoffs") or [])
        base["leader_id"] = payload.get("leader_id")
        base["follower_id"] = payload.get("follower_id")
        base["commitment_mode"] = payload.get("commitment_mode", "pure")
        base["tie_policy"] = payload.get("tie_policy", "report_both")
    if kind == "repeated":
        stage = payload.get("stage_game") or {}
        base["stage_game"] = NormalFormGame(
            players=_decode_players(stage.get("players")),
            actions=stage.get("actions") or {}, payoffs=stage.get("payoffs") or [])
        base["horizon"] = payload.get("horizon")
        base["discount_factor"] = to_rational(payload.get("discount_factor"),
                                              "repeated.discount_factor")
        base["monitoring"] = payload.get("monitoring", "perfect_public")
    if kind == "bargaining":
        base["feasible_points"] = payload.get("feasible_points") or []
        base["disagreement"] = payload.get("disagreement") or {}
        base["randomization"] = payload.get("randomization", "not_allowed")
    if kind == "mechanism":
        base.update({k: v for k, v in payload.items() if k != "players"})
        raw_objective = payload.get("objective")
        if isinstance(raw_objective, Mapping):
            base["objective"] = ObjectiveSpec(
                kind=raw_objective.get("kind", ""),
                direction=raw_objective.get("direction", "maximize"),
                player_id=raw_objective.get("player_id"),
                weights={k: to_rational(v, "objective weight")
                         for k, v in (raw_objective.get("weights") or {}).items()}
                or None)
    if kind == "open_spiel":
        base["game_id"] = payload.get("game_id", "")
        base["parameters"] = payload.get("parameters") or {}
    return base


def _compile_custom(data: Mapping[str, Any]) -> GameEnvelope:
    kind = data.get("kind")
    if kind is None:
        raise GameSpecError("custom game data needs a kind")
    registry = {
        "normal_form": NormalFormGame,
        "extensive_form": ExtensiveFormGame,
        "bayesian": BayesianGame,
        "stackelberg": StackelbergGame,
        "repeated": RepeatedGame,
        "bargaining": BargainingSpec,
        "mechanism": MechanismSpec,
        "open_spiel": OpenSpielGame,
    }
    cls = registry.get(kind)
    if cls is None:
        raise GameSpecError(f"custom kind {kind!r} unsupported")
    payload = {k: v for k, v in data.items() if k != "kind"}
    decoded = _decode_custom(kind, payload)
    try:
        game = cls(**decoded)
    except TypeError as exc:
        raise GameSpecError(f"custom {kind}: {exc}") from exc
    return build_envelope(game, template_id="custom", template_options={},
                          assumptions=data.get("assumptions", []),
                          provenance=data.get("provenance", {}))
