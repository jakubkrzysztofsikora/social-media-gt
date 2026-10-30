# Social-media-gt advisor: MCP server + Gradio chat app over exact game-theory proofs

## Context

`social_media_gt.py` is an exact-arithmetic verification toolkit for the paper *"Nash equilibrium and access to social media for children"*. It exposes 10 scenario functions (`tw1`, `tw1_1`, `tw2`, `tw3`, `tw3_2`, `tw5`, `tw6`, `tw7`, `tw7_2`, `tw8`), each a pure function `params -> {scenario, title, params, results, checks, conclusion}` using `fractions.Fraction`, SymPy, and Z3 (43 machine checks pass at paper defaults). CLI runs one or all scenarios, `--json` emits machine-readable results. The script itself stays untouched; the new code consumes it by import.

Goal: a chat-style web app (Gradio) plus an MCP server, sharing one core, where a multi-turn conversation about a real situation lets the LLM infer the toolkit's parameter values, then the app runs **all 10 scenarios** with those parameters and renders full mathematical evidence per scenario — results, PASS/FAIL checks, the game-theoretic conclusion, and a strategy table (each strategy's payoff, cost, benefit). Same report available through MCP tools.

Inference endpoint (verified live this session): NexGate LiteLLM gateway, `nemotron-bolt` model (MLX-LM server behind the gateway). Loopback `http://127.0.0.1:4000/v1` and tailnet `https://litellm.tail5d39b4.ts.net/v1` both returned HTTP 200 with 211 models including `nemotron-bolt`; unauthenticated calls return 401, so the master key is required. Credentials live in a **gitignored** `.env`.

## Approach

Work order: 1 environment, 2 core, 3 MCP server, 4 web app, 5 tests, 6 README. Steps 3–4 both depend on step 2; step 5 depends on 2–4; step 6 last.

### Step 1 — Environment, deps, gitignored env

1. Create venv with Homebrew Python 3.12 (Gradio 6 declares support up to 3.13; 3.14 is unsupported and is the system default):
   `cd /Users/jakubsikora/Repos/personal/math/social-media-gt && /opt/homebrew/bin/python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`
2. Append to `requirements.txt` (existing content `sympy>=1.13`, `z3-solver>=4.12` stays):
   ```
   gradio==6.30.0
   mcp==2.3.0
   openai>=1.40
   python-dotenv>=1.0
   pytest>=8
   ```
   Rationale (verified): `mcp` 2.3.0 is current PyPI, exposes `from mcp.server import MCPServer`, `@mcp.tool()`, `mcp.run(transport=...)`, and `from mcp import Client`; `gradio` 6.30.0 is current. Do **not** add `gradio[mcp]` — its extra pins `mcp<2.0.0` and would conflict.
3. Create `.gitignore` (repo has none):
   ```
   .env
   .venv/
   __pycache__/
   .pytest_cache/
   *.pyc
   gradio_cached_examples/
   ```
4. Create committed `.env.example` (placeholders only):
   ```
   GT_LLM_BASE_URL=https://litellm.tail5d39b4.ts.net/v1
   GT_LLM_API_KEY=replace-with-litellm-master-key
   GT_LLM_MODEL=nemotron-bolt
   ```
5. Create gitignored `.env` with the real values: `GT_LLM_BASE_URL=https://litellm.tail5d39b4.ts.net/v1`, `GT_LLM_API_KEY=<value of LITELLM_MASTER_KEY from /Users/jakubsikora/Repos/personal/hybrid-llm-stack-public-candidate/.env.nexgate>`, `GT_LLM_MODEL=nemotron-bolt`. Read the key from that file programmatically; never print it, never commit it, never paste it into source, docs, or plan artifacts.

### Step 2 — `advisor.py`: shared core (new file)

Single module imported by both the MCP server and the web app. No dependency on Gradio or MCP SDK.

**Config** — `load_config(env: Mapping[str, str] | None = None) -> Config`, dataclass `Config(base_url: str, api_key: str, model: str)`:
- Call `dotenv.load_dotenv(Path(__file__).with_name(".env"))` once at import of `advisor` (dotenv does not override already-set environment variables).
- `base_url`: `GT_LLM_BASE_URL` if non-empty; else if `LITELLM_HOST` set → `f"{LITELLM_SCHEME or 'https'}://{LITELLM_HOST}"` + (`f":{LITELLM_PORT}"` only when `LITELLM_PORT` is set and not the scheme default 443/80) + `"/v1"`; else `"http://127.0.0.1:4000/v1"`.
- `api_key`: `GT_LLM_API_KEY` else `LITELLM_MASTER_KEY` else `""`.
- `model`: `GT_LLM_MODEL` else `"nemotron-bolt"`.

**Catalog** — module-level `CATALOG: dict[str, dict]`, one entry per scenario id, built so it cannot drift from the toolkit: scenario ids, titles, and defaults come from `social_media_gt.SCENARIOS` (`inspect.signature` for default values, rendered with `social_media_gt.show()`); only the human text is hand-written. Exact per-parameter text to use (this is the LLM-facing semantic contract; keep the wording as-is):

- `tw1` — "Tw. 1 - static trap". Scale: benefit/cost of one family in units where full private benefit of using social media equals 1.
  - `tau` — cost of using social media for one family. Using while others do not gives payoff `-tau`; when everyone uses, using gives `f1 - tau`. Domain `tau >= 0`, default 1/2.
  - `f1` — private benefit of using social media for one family, excluding peer effects. Domain `f1 >= 0`, default 1/4.
  - `E1` — harm of losing peer connection when every other family uses: deviating out of an all-using profile gives `-E1`. Domain `E1 >= 0`, default 1/4.
  - Model facts the LLM must respect: all-using is a Nash equilibrium iff `f1 + E1 >= tau`; restricting together Pareto-dominates iff `tau > f1`.
- `tw1_1` — "Tw. 1.1 - local screen-time cap under network effects".
  - `a` — marginal benefit of screen time; private value `V(t) = a t - b t^2`. Domain `a > 0`, default 1.
  - `b` — curvature of screen-time harm (the `b t^2` term). Domain `b > 0`, default 1.
  - `T` — maximum screen time a cap permits (outside option). Domain `T > 0`, default 4.
  - `beta` — strength of the peer externality; peer penalty is `beta * k * (T - t)` with `k` the share of peers online (`k` is fixed at 1/2 inside this scenario and is not inferred). Domain `beta > 0`, default 1. Constraint: `a + beta <= 2 b T`.
- `tw2` — "Tw. 2 - cascade and least fixed point".
  - `eps` — baseline share of families using social media before peer pressure acts (floor of the adoption map). Domain `0 < eps < 1`, default 1/10.
  - `p_slope` — slope of peer pressure: adoption probability rises `p_slope` percentage points per 1.0 of peer adoption share. Domain `p_slope > 0`, default 100.
  - `f_lo` — lower peer-pressure threshold (percent) where adoption starts responding. Domain `f_lo < f_hi`, default 5.
  - `f_hi` — upper peer-pressure threshold (percent) where adoption saturates. Domain `f_hi > f_lo`, default 95.
- `tw3` — "Tw. 3 - absorbing state under myopic revisions".
  - `tau` — cost of using social media, in units where one adopting peer is worth 1: deviating pays when peer benefit `k` minus cost `tau` falls below the exclusion loss `k/2`. Domain `tau >= 0`, default 1/2.
  - `n` — number of families in the cohort. Domain integer `2 <= n <= 12`, default 4.
- `tw3_2` — "Tw. 3.2 - global games and the incentive barrier".
  - `f1` — private benefit of using social media for one family. Domain `f1 >= 0`, default 1.
  - `E1` — harm of losing peer connection when every other family uses. Domain `E1 >= 0`, default 3.
  - Model facts: risk-dominant cutoff `tau^RD = (f1 + E1)/2`; the Pareto trap class is non-empty iff `E1 > f1`.
- `tw5` — "Tw. 5 - outsider threshold and pact durability".
  - `tau` — resistance threshold in this scenario's normalization: a family with peer share `k` stays while `2k + k^2 >= tau`. Domain `tau >= 0`, default 3/2.
  - `n` — number of families in the cohort. Domain integer `n >= 2`, default 10.
  - `d` — families unaware of the pact. Domain integer `0 <= d <= n - 1`, default 1.
- `tw6` — "Tw. 6 - coalitions in the continuum".
  - `gamma` — share of aware parents signing the pact. Domain `0 <= gamma < 1`, default 9/40.
  - `eps`, `p_slope`, `f_lo`, `f_hi` — exactly as in `tw2` (same meanings, defaults 1/10, 100, 5, 95).
- `tw7` — "Tw. 7 - top-down reduction of pressure".
  - `lam` — slope of peer pressure per unit adoption; effective pressure is `lam * (1 - eta)`. Domain `lam > 0`, default 1/5.
  - `eta` — share of the group covered by the rule; uncovered peers keep the pressure. Domain `0 <= eta < 1`, default 0.
  - `eps` — floor of the adoption map. Domain `0 < eps < 1`, default 1/10.
  - `step` — exogenous adoption pressure added each round. Domain `step > 0`, default 1/20.
- `tw7_2` — "Tw. 7.2 - evasion, enforcement cost, hybrid policy".
  - `creg` — cost weight of statutory limits; regulatory cost `creg * lambda_reg^2 / 2`. Domain `creg > 0`, default 1.
  - `cplat` — cost weight of platform defaults; platform cost `cplat * lambda_plat^2 / 2`. Domain `cplat > 0`, default 1.
  - `eta` — evasion rate: fraction of regulatory pressure families route around. Domain `0 <= eta < 1`, default 1/2.
  - `lam_target` — required total reduction in peer pressure. Domain `lam_target > 0`, default 3/10.
- `tw8` — "Tw. 8 - sequential exit without authority".
  - `tau_C` — coordination threshold: a family with peer share `k` gains `tau_C - 2k - k^2` from staying. Domain `tau_C >= 0`, default 2.
  - `delta` — discount factor per round applied to gains. Domain `0 < delta <= 1`, default 99/100.
  - Fixed inside the scenario (not inferred): 10 players, 9 aware.

**Parameter coercion** — `to_fraction(value) -> Fraction`: accepts `int`, `float` (via `Fraction(str(value))`), and strings of form `"a/b"` or decimal (`Fraction(text)`); rejects `bool` and anything else with `ValueError`.

**Normalization + domain adjustments** — `normalize_params(scenario_id: str, raw: Mapping[str, object]) -> tuple[dict[str, Fraction], list[str]]`:
- Start from the scenario function's own defaults (`inspect.signature`), override with any supplied key.
- Unknown keys and unusable values (non-numeric, bool) are dropped, and each drop appends a note `"{key}: inferred value unusable, paper default used"`.
- Then apply these exact repairs, each appending a note `"{key}: adjusted from X to Y (reason)"`:
  - `tw1`, `tw3_2`: `tau`, `f1`, `E1` clamped to `>= 0` (negative → 0).
  - `tw1_1`: `a`, `b`, `T`, `beta` clamped to `>= 1/1000` when `<= 0`; if `a + beta > 2 b T`, scale both `a` and `beta` by `s = 2 b T / (a + beta)` (preserves their ratio, forces `a + beta = 2 b T`).
  - `tw2`, `tw6`: `eps` clamped into `[1/1000, 999/1000]`; if `f_hi <= f_lo`, set `f_hi = f_lo + 1`; `p_slope` clamped to `>= 1/1000`.
  - `tw3`: `tau` clamped `>= 0`; `n` clamped into `[2, 12]`.
  - `tw5`: `tau` clamped `>= 0`; `n` clamped to `>= 2`; `d` clamped into `[0, n - 1]`.
  - `tw6`: `gamma` clamped into `[0, 999/1000]`.
  - `tw7`: `lam` clamped `>= 1/1000`; `eta` clamped into `[0, 999/1000]`; `eps` clamped into `[1/1000, 999/1000]`; `step` clamped `>= 1/1000`.
  - `tw7_2`: `creg`, `cplat`, `lam_target` clamped `>= 1/1000`; `eta` clamped into `[0, 999/1000]`.
  - `tw8`: `tau_C` clamped `>= 0`; `delta` clamped into `[1/1000, 1]`.
- Integer parameters (`n`, `d`) are rounded to the nearest integer before clamping.
- After repairs, every scenario function must accept its parameters; if a function still raises `ValueError`/`RuntimeError`, that scenario is reported as `status="error"` with the exception message and the run continues.

**LLM inference** — `infer_parameters(messages: list[dict], client=None, config=None) -> dict`:
- Client: `openai.OpenAI(base_url=config.base_url, api_key=config.api_key or "none")`; call `client.chat.completions.create(model=config.model, messages=[system, *history], temperature=0)`. `messages` are OpenAI-style dicts (`role`, `content`); the app and MCP layer pass the full chat history.
- System prompt (single string, built programmatically from `CATALOG`): role = "game-theory analyst using an exact-arithmetic proof toolkit"; state the unit scale (benefit/cost of one family, private benefit of using = 1); list every scenario id with its title, each parameter's meaning text, domain, and default; state that values must respect domains and the model facts above; state the JSON contract:

  ```json
  {
    "situation_summary": "one paragraph restating the situation in game-theory terms",
    "scenarios": {
      "tw1": {"params": {"tau": 0.5, "f1": 0.25, "E1": 0.25}, "rationale": "why these values follow from the conversation"},
      "tw1_1": {"params": {...}, "rationale": "..."},
      "tw2": {...}, "tw3": {...}, "tw3_2": {...}, "tw5": {...},
      "tw6": {...}, "tw7": {...}, "tw7_2": {...}, "tw8": {...}
    }
  }
  ```
  plus: "Reply with ONLY that JSON object. Always include all 10 scenario ids. Use the paper default for any parameter the conversation gives no signal about. Values may be JSON numbers or fraction strings like '1/2'. Do not invent scenarios or extra keys."
- `response_format` is **not** sent: the MLX backend behind the gateway ignores it (verified upstream in `mlx_lm/server.py`). Parse tolerantly: strip ``` fences, take the substring from the first `{` to the last `}`, `json.loads`.
- On parse failure, retry **once** with the assistant's raw text appended to the history plus a user message `"Invalid JSON: {error}. Reply with ONLY the JSON object described above."`; on a second failure raise `InferenceError(f"LLM did not return parseable JSON: {raw[:400]}")`.
- On transport/auth failure (gateway down, 401), let the exception propagate wrapped in `InferenceError` with the status code — never fabricate parameters.
- Return the parsed object plus `"raw"` (the raw completion text) and `"model"`.

**Scenario execution + strategy table** — `run_scenarios(params_by_scenario: dict[str, Mapping[str, object]]) -> list[dict]`:
- Iterate `social_media_gt.SCENARIOS` in its defined order (`tw1, tw1_1, tw2, tw3, tw3_2, tw5, tw6, tw7, tw7_2, tw8`).
- For each: `normalize_params`, call the scenario function with the normalized values plus the scenario's own defaults (`full=True` for `tw8`), catch exceptions into `status="error"`.
- Build the evidence dict per scenario with exactly these keys: `scenario`, `title`, `status` (`"ok"`/`"error"`), `params` (normalized, rendered via `social_media_gt.show`), `param_notes` (list of strings), `results` (scenario `results` dict, rendered via `social_media_gt.show` for fractions), `checks` (scenario `checks` list), `conclusion` (scenario `conclusion` list), `strategies` (list of rows).
- Strategy rows come from a module-level `STRATEGY_VIEWS: dict[str, list[dict]]`; each row is `{"strategy": str, "outcome": str, "values": {label: rendered_value}}`, where values are read from that scenario's `results` dict. Exact table (labels and key bindings):

  | scenario | strategy | outcome text | result keys |
  | --- | --- | --- | --- |
  | tw1 | Restrict together (all-R) | payoff of staying out; cost of the lone deviator | `payoffs_relative_to_offline.all_R.stay`, `payoffs_relative_to_offline.all_R.deviate` |
  | tw1 | Use together (all-A) | payoff of staying in; cost of leaving | `payoffs_relative_to_offline.all_A.stay`, `payoffs_relative_to_offline.all_A.deviate` |
  | tw1 | Pareto comparison of the two | all-R better exactly when tau > f1 | `all_r_dominates_pareto`, `pareto_gap_all_R` |
  | tw1_1 | Parent alone (first best) | screen-time choice without peer effects | `t_first_best` |
  | tw1_1 | Parent with peer effects (Nash) | stricter cap; welfare loss from peer coupling | `t_nash`, `welfare_loss` |
  | tw1_1 | Private vs peer-aware value | private value curve vs peer-aware curve | `curve_V`, `curve_G` |
  | tw2 | No peer pressure | adoption level without cascade | `uniform_q_mean` |
  | tw2 | Peer-pressure cascade | limit and rounds to full adoption | `limit`, `rounds_to_full_adoption`, `phi_minus_k_on_linear_region` |
  | tw3 | All families use | equilibrium status and gain from deviating | `all_a_is_equilibrium`, `delta_at_all_a` |
  | tw3 | Other stable profiles | number of Nash profiles | `equilibria_count` |
  | tw3_2 | Restrict jointly (Pareto better) | trap class and welfare gap at the midpoint | `pareto_trap_class_nonempty`, `pareto_gap_at_midpoint` |
  | tw3_2 | All-using (risk-dominant) | cutoff above which all-using is risk-dominant | `tau_risk_dominant` |
  | tw5 | No pact | adopter share needed before the first outsider leaves | `k_star`, `m_out` |
  | tw5 | Pact among aware parents | how many members hold before leaving | `m_stab`, `P_peer_5_of_9`, `P_peer_6_of_9` |
  | tw6 | No pact | adoption returns to full after the pact ends | `release_steps` |
  | tw6 | Pact of gamma aware parents | adoption level held, contraction, trajectory, exit multiplier | `K`, `a`, `trajectory`, `exit_multiplier` |
  | tw7 | Rule covers part of the group | effective pressure and resulting adoption limit | `lambda_effective`, `limit` |
  | tw7 | Rule covers everyone (eta = 1) | baseline row computed by calling the same function with `eta = 0` and `eta = 1` | baseline calls: `{"eta": 0}` and `{"eta": 1}` — note `eta = 1` is rejected by the toolkit, so the baseline row uses `eta = 0` only, labelled "No rule (eta = 0)" |
  | tw7_2 | Statutory limits | share of pressure carried by regulation and its cost weight | `lambda_reg` |
  | tw7_2 | Platform defaults | share of pressure carried by product defaults and its cost weight | `lambda_plat` |
  | tw7_2 | Optimal hybrid | cost-efficiency ratio of the two instruments | `ratio` |
  | tw8 | Exit without a public schedule | worst-case advantage and number of minimizing states | `discounted_min`, `minimizing_states` |
  | tw8 | Public sequential schedule | undiscounted advantage, worst sequence, orderings checked | `undiscounted_sum`, `worst_sequence`, `worst_d_slot`, `orderings_enumerated` |

  Baseline rows are permitted only where stated (`tw7` with `eta = 0`), and are produced by calling the **same** toolkit function with the stated baseline parameters — no new mathematics anywhere in `advisor.py`.

**Report assembly + rendering** — `analyze_situation(messages: list[dict], client=None, config=None) -> dict` returns exactly:
```
{
  "situation_summary": str,
  "model": str,
  "inference_raw": str,
  "scenarios": [ {scenario evidence dict}, ... 10 entries ],
  "totals": {"scenarios": 10, "ok": int, "errors": int, "checks_total": int, "checks_failed": int},
  "markdown": str
}
```
`render_report(report) -> str` emits Markdown: situation summary; per scenario a heading with title, a parameter table (`parameter | value | note`), the strategy table (strategy | outcome | values), a results table (every key of `results` rendered with `social_media_gt.show`), the check list (`[PASS]`/`[FAIL]` + expected/got), the conclusion sentences, and a totals line. Errors appear as `> scenario failed: <message>`; no scenario is silently dropped.

### Step 3 — `mcp_server.py`: MCP server (new file)

- `from mcp.server import MCPServer`; `mcp = MCPServer("social-media-gt")`.
- Tools (exact names, signatures, return types):
  - `list_scenarios() -> dict` — `{"scenarios": [{"id", "title", "params": {name: {"meaning", "domain", "default"}}, "strategies": [labels]}]}` built from `CATALOG` and `STRATEGY_VIEWS`.
  - `infer_parameters(messages: list[dict]) -> dict` — `{"situation_summary", "scenarios": {id: {"params", "rationale"}}, "model", "raw"}`.
  - `run_scenario(scenario: str, params: dict | None = None) -> dict` — evidence dict for one scenario (same shape as one entry of `analyze_situation` scenarios list); unknown `scenario` id → `{"status": "error", "error": "unknown scenario ..."}` rather than an exception.
  - `analyze_situation(messages: list[dict]) -> dict` — the full report from Step 2.
- Every returned dict must be JSON-serializable: all fractions rendered through `social_media_gt.show()` before serialization (no `Fraction` objects leak).
- Transport: `if __name__ == "__main__":` parses `--transport {stdio,streamable-http}` (default `stdio`) and `--port` (default 8765, used only for streamable-http) and calls `mcp.run(transport=...)` — `mcp.run()` for stdio, `mcp.run(transport="streamable-http", port=port)` otherwise. No HTTP server on stdio mode, no banner noise on stdout (stdio transport uses stdout for protocol).

### Step 4 — `webapp.py`: Gradio chat app (new file)

- `gr.Blocks()` layout, in order: `gr.Markdown` header; `gr.Chatbot(type="messages", label="Situation analysis", height=520)`; `gr.Textbox(lines=3, label="Describe the situation (multi-turn)")`; `gr.Button("Analyze", variant="primary")`; `gr.Markdown(label="Latest full evidence")`; `gr.JSON(label="Raw evidence JSON")`.
- State: `gr.State` holding the chat history list.
- Handler `respond(history, state, ...)`: appends the user turn, calls `advisor.analyze_situation(history)`, appends an assistant turn containing `report["markdown"]`, and returns the updated history, the markdown, and the raw report dict (minus the duplicated `markdown` key).
- Both the Button click and the Textbox submit trigger the same handler.
- `InferenceError` and any other exception are caught and shown as a chat message `Analysis failed: <message>` — the app never shows results it did not compute.
- Launch: argparse `--host` (default 127.0.0.1), `--port` (default 7860), `--share` (default off), then `demo.launch(server_name=host, server_port=port, share=share)`.
- Gradio 6.30.0 API note: `gr.Chatbot(type="messages")` is the 5/6 API. If 6.30.0 rejects the `type` argument, call `gr.Chatbot(label=..., height=...)` without it and keep the message-list format (`{"role", "content"}`).

### Step 5 — `test_advisor.py`: tests (new file)

pytest, matching the existing style of `test_social_media_gt.py` (plain functions, `F` fractions, no network). Required cases:
1. `CATALOG` covers exactly the 10 ids in `social_media_gt.SCENARIOS`, and each catalog parameter set equals the scenario function's signature parameters (minus internal-only ones: `tw8.full` is internal; `tw1_1.k` is internal and must not appear).
2. `to_fraction` accepts `0.25`, `"0.25"`, `"1/2"`, `3`; rejects `True` and `"abc"`.
3. `normalize_params` repairs out-of-domain input exactly as specified: `tw1_1` with `a=9/5, b=1, T=1, beta=9/5` yields `a + beta == 2 b T` with both positive; `tw3` with `n=99` → `n == 12`; `tw7` with `eta=5` → `eta == 999/1000`; `tw5` with `d=99` → `d == n - 1`; each repair produces a note.
4. `normalize_params` leaves paper defaults untouched when given `{}` for every scenario.
5. `infer_parameters` with a stub client returning a fenced JSON payload parses correctly; a stub returning invalid JSON twice raises `InferenceError`.
6. `run_scenarios` with paper-default params for all 10 scenarios returns 10 entries, all `status == "ok"`, and `totals["checks_failed"] == 0` (mirrors `run_all()` in the toolkit).
7. `render_report` output contains each scenario id, each `[PASS]`/`[FAIL]` marker, and the totals line.
8. `STRATEGY_VIEWS` covers all 10 scenario ids, and every `result key` it references exists in that scenario's `results` for default params (guards against typos and toolkit drift).
9. Import smoke: `import mcp_server` registers 4 tools; `import webapp` builds the Blocks without launching.

### Step 6 — README

Append a section "Advisor apps (web + MCP)" describing: `cp .env.example .env` and filling the key; `.venv/bin/python webapp.py` (http://127.0.0.1:7860); `.venv/bin/python mcp_server.py` (stdio) and `--transport streamable-http --port 8765`; the four MCP tools; the env variables and their defaults; and the fact that all 10 scenarios always run with the inferred parameters.

## Critical files & anchors

- `social_media_gt.py` — `SCENARIOS` (line 557), `show` (42), `run_all` (571), `apply_overrides` (627). Imported, never modified; source of scenario ids, defaults, result keys.
- `advisor.py` (new) — `CATALOG`, `STRATEGY_VIEWS`, `load_config`, `to_fraction`, `normalize_params`, `infer_parameters`, `run_scenarios`, `analyze_situation`, `render_report`. The single source of truth for both front ends.
- `mcp_server.py` (new) — `MCPServer` instance and the four `@mcp.tool()` functions.
- `webapp.py` (new) — Gradio `Blocks`, chat handler, launch arguments.
- `requirements.txt` — pins for `gradio==6.30.0`, `mcp==2.3.0`, `openai`, `python-dotenv`, `pytest`.

## Verification

Run from `/Users/jakubsikora/Repos/personal/math/social-media-gt` with `.venv/bin/python`.

1. **Toolkit still green**: `.venv/bin/python -m pytest -q` → all existing tests plus the new ones pass; the new default-params case reports `checks_failed == 0` for all 10 scenarios.
2. **Live LLM inference (new behavior)**: `.venv/bin/python -c` script calling `advisor.analyze_situation([{"role":"user","content":"30 families in one school; a family's harm from screen time is about half of what it gains when all peers are online; one family in five is unaware of any parent pact; a rule would cover two thirds of the cohort and be evaded half the time"}])` → expected: report contains 10 scenario entries, every `status == "ok"`, `totals["checks_failed"] == 0`, `situation_summary` non-empty, and `tw1` params show a `tau` around 0.5 with `f1`/`E1` in their domains. Also assert the raw completion parses (no `InferenceError`).
3. **Gradio UI (real surface)**: launch `.venv/bin/python webapp.py --port 7860`, open `http://127.0.0.1:7860` in a browser, type the same situation, press Analyze → expected: chat shows an assistant turn, the "Latest full evidence" panel shows 10 scenario sections with parameter tables, strategy tables, `[PASS]` check lines, and a totals line; the Raw evidence JSON panel shows a populated object. Confirm no console errors on the page.
4. **MCP stdio**: start `.venv/bin/python mcp_server.py` as a subprocess and drive it with an MCP client script using `from mcp import Client` + `StdioServerParameters` → expected: `list_tools()` returns exactly `list_scenarios`, `infer_parameters`, `run_scenario`, `analyze_situation`; `call_tool("run_scenario", {"scenario": "tw1", "params": {"tau": 0.75}})` returns `status == "ok"` with passing checks; `call_tool("analyze_situation", {"messages": [{"role": "user", "content": "..."}]})` returns 10 scenario entries.
5. **MCP streamable HTTP**: start `.venv/bin/python mcp_server.py --transport streamable-http --port 8765` and connect a `Client("http://127.0.0.1:8765/mcp")` → expected: `list_tools()` returns the same 4 tools.
6. **Secrets stay out of git**: `git status --porcelain` shows no `.env` entry and `git check-ignore -v .env` resolves to the new `.gitignore` line; `.env.example` contains only placeholders.

## Assumptions & contingencies

- **Model output quality**: `nemotron-bolt` is a 4B MLX model and may not return clean JSON. Contingency, already wired: one strict-retry, then `InferenceError` surfaced in the UI/MCP result. Do not add a paid-model fallback silently; the operator switches with `GT_LLM_MODEL`.
- **Gateway auth**: the master key is required (401 otherwise). If `.env` is missing, the app still starts with the loopback default and an empty key and reports the 401 as an analysis error.
- **Parameter drift**: if a repair rule cannot make a scenario's arguments valid, that scenario reports `status="error"` and the other nine still report. The toolkit's own exceptions are never swallowed silently.
- **`tw8` cost**: the exhaustive 9! ordering scan runs on every analysis. Measure it in step 2 of Verification; if it exceeds 10 s on this machine, run `tw8` with `full=False` in `run_scenarios` (keep `full=True` available through `run_scenario(scenario="tw8", params={"full": True})`) and note the change in the README.
- **Gradio API drift**: pinned to 6.30.0; the only known variance is the `gr.Chatbot(type="messages")` argument (drop it if rejected).
- **MCP SDK import**: pinned to `mcp==2.3.0` where the class is `MCPServer`; if the resolved version exposes `FastMCP` instead (only possible if the pin is changed), use `from mcp.server.fastmcp import FastMCP as MCPServer` — the decorator and `run()` API are identical.
- **Scope**: the toolkit file, its tests, and its math stay untouched; all new behavior lives in `advisor.py`, `mcp_server.py`, `webapp.py`, `test_advisor.py`.
