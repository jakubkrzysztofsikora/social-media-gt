# social-media-gt

Reproducible verification of the mathematical results in

> *Nash equilibrium and access to social media for children*

The paper argues that parents restricting social media for their children fail
one by one, and that collective rules only work when they bind everyone at
once. Every number in the paper is re-derived here from the paper's own
definitions, with exact rational arithmetic.

Nothing in this repository is an approximation: fractions stay exact, symbolic
algebra goes through SymPy, and the "no profitable deviation" claims are
discharged by the Z3 solver rather than by hand.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer.

## Run everything

```bash
python social_media_gt.py
```

```
tw6  Tw. 6 - coalitions in the continuum
  parameters: {gamma: 9/40, eps: 1/10, p_slope: 100, f_lo: 5, f_hi: 95}
  K = 1/4
  a = 3/4
  trajectory = [31/40, 103/160, 349/640]
  exit_multiplier = 10/3
  release_steps = 15
  [PASS] K is a fixed point of the coalition operator (expected 1/4, got 1/4)
  ...

10/10 scenarios, 43/43 checks passed
```

Exit code is 0 when every check passes, 1 otherwise. `--json` prints the same
results as JSON, `--quick` skips the exhaustive 9! ordering scan in `tw8`.

## Run one scenario, with your own parameters

Every parameter defaults to the value printed in the paper, so a default run
reproduces the published numbers. Pass different values to explore:

```bash
python social_media_gt.py tw6 --gamma 0.3
python social_media_gt.py tw7 --lambda 0.5 --eta 0.25
python social_media_gt.py tw8 --tau-c 3 --delta 0.95
python social_media_gt.py tw5 --tau 2 --n 10
```

| scenario | paper result | parameters |
| --- | --- | --- |
| `tw1` | restricting alone never pays; all-R is Pareto better | `--tau --f1 --e1` |
| `tw1_1` | a screen-time cap ends above the first best once peers are online | `--a --b --T --beta` |
| `tw2` | cascade to full adoption, reached after 19 rounds | `--eps` |
| `tw3` | all-adopting is absorbing | `--tau --n` |
| `tw3_2` | risk-dominant cutoff P(1)/2; voluntary mechanisms cannot recruit | `--f1 --e1` |
| `tw5` | first outsider leaves at m = 4, members hold to m = 5 | `--tau --n` |
| `tw6` | a 22.5% pact pulls the cohort to 25% adoption | `--gamma --eps` |
| `tw7` | P6: lambda removes a fraction of baseline peer pressure; eta only attenuates it in the toolkit recurrence | `--lambda --eta --eps` |
| `tw7_2` | cheapest policy mixes statute and product standards | `--creg --cplat --eta --lam-target` |
| `tw8` | sequential exit holds only under a public schedule | `--tau-c --delta` |

## Tests

```bash
python -m pytest -q
```

The tests pin the paper's printed constants and check that the two independent
computation routes inside each scenario agree, so a parameter change is
validated rather than trusted.

## What is checked, and what is assumed

Machine-checked: equilibrium conditions and their boundaries, the cascade
criterion and its limit, coalition thresholds, the hybrid policy optimum
(solved, then proved strictly convex), the appendix sums and the worst-case
schedule (exhaustive over 2^10 profiles and 9 phases, optionally over all 9!
slot orderings), and the DSIC and individual-rationality implications.

Assumed, and named as such in the paper: the global-games selection theorem
(Tw. 3.2) and the one-deviation principle (Tw. 4, Tw. 8). Both are cited
results, not claims this code can discharge.

## Advisor apps (web + MCP)

`advisor.py` turns a plain-language description of a real situation into the
toolkit's parameters: an LLM reads the conversation and maps it onto the ten
scenarios, then the app re-runs **all ten scenarios** with those parameters and
renders the full evidence for each one - parameters, strategy payoffs, results,
PASS/FAIL checks and the game-theoretic conclusion. The same core backs a
Gradio chat app and an MCP server.

### Setup

```bash
cp .env.example .env      # then put the LiteLLM master key in .env
```

`.env` is gitignored. It holds three variables:

| variable | default | meaning |
| --- | --- | --- |
| `GT_LLM_BASE_URL` | `http://127.0.0.1:4000/v1` | OpenAI-compatible gateway |
| `GT_LLM_API_KEY` | `LITELLM_MASTER_KEY`, else empty | gateway key |
| `GT_LLM_MODEL` | `nemotron-bolt` | model used for the inference |

### Web app

```bash
.venv/bin/python webapp.py                 # http://127.0.0.1:7860
.venv/bin/python webapp.py --host 0.0.0.0 --port 8080 --share
```

Choose an example to fill the query, then press **Analyze**. The input clears after submission.
The chat keeps user turns and concise confirmations; the evidence panel holds the full
Markdown report, and the JSON panel exposes the same raw evidence.

### MCP server

```bash
.venv/bin/python mcp_server.py                                   # stdio
.venv/bin/python mcp_server.py --transport streamable-http --port 8765
```

Four tools:

| tool | arguments | returns |
| --- | --- | --- |
| `list_scenarios` | - | every scenario with its parameters and strategy rows |
| `infer_parameters` | `messages` | the inferred parameters and rationales per scenario |
| `run_scenario` | `scenario`, `params` | the full evidence for one scenario |
| `analyze_situation` | `messages` | the whole report for all ten scenarios |

All ten scenarios always run, with the inferred parameters; a parameter that the
conversation gives no signal about falls back to the paper's default, and a
value outside the scenario's domain is repaired and reported in the parameter
notes. The toolkit's own numbers are never re-derived by the advisor - it only
chooses the arguments and renders what the toolkit returns.

## General games

Beyond the paper scenarios the advisor analyses arbitrary finite games with the
same discipline: exact rational arithmetic everywhere, a solver witness plus
independent verification for every exact claim, and clearly labelled
approximate evidence otherwise. The paper engine (`social_media_gt.py`) and its
tests stay untouched; all general-game behaviour lives in separate modules.

### Modes and templates

| template | compiles to | notes |
| --- | --- | --- |
| `pricing_auction` | BayesianGame | first/second price, finite types, bid grids |
| `security_patrol` | StackelbergGame | defender commits schedules over targets |
| `negotiation` | ExtensiveFormGame | finite offer tree, exact terminal utilities |
| `bargaining` | BargainingSpec | Nash product; randomization needs a binding contract |
| `public_goods` | NormalFormGame | finite contribution actions, complete payoff table |
| `mechanism_design` | MechanismSpec | VCG or finite optimal-auction, DSIC/BIC, IR scopes |
| `multi_agent_learning` | OpenSpielGame | kuhn_poker, leduc_poker, matrix_pd, tic_tac_toe |
| `p3_heterogeneous` | NormalFormGame | User-defined finite one-shot model; empty options select the labelled P3 paper benchmark |
| `custom` | any canonical form | the typed schema directly |

P3 custom inputs require `type_counts`, `thresholds`, and `payoff_coefficients`
together. Each type count is a positive integer. Thresholds cover each type
exactly. All coefficients are rational. For peer share `k` among other players,
the template uses `f(k)=f_constant+f_linear*k`, `E(k)=E_constant+E_linear*k+
E_quadratic*k^2`, `u(A,k)=f(k)`, and `u(R,k)=tau_type-E(k)`. Each player has
actions `R` and `A`; the 10,000-profile cap limits this binary model to 13
players. Empty options select only the clearly labelled paper benchmark.

This template does not model the paper's repeated-game construction. Current
`RepeatedGame` support requires two players, two actions each, and perfect
public monitoring. It supports named-policy finite-horizon simulation, not
repeated-game equilibrium solving. A faithful P3 repeated-game analysis needs
an explicit multi-player history and monitoring model plus a compatible
equilibrium solver. The one-shot template makes no repeated-game equivalence
claim.

### Methods

Exact (rational witnesses, independently verified): `pure_nash`, `mixed_nash`
(two-player), `bayesian_nash` (pure contingent plans), `subgame_perfect`
(finite perfect-information trees), `correlated_equilibrium` (confirmed
mediator; exact primal/dual LP certificate), `nash_bargaining` (two players),
`stackelberg` (both leader- and follower-favorable tie policies), and
`mechanism_optimum` (finite feasible mechanism set).

Approximate (always labelled `approximate`, never a proof): `epsilon_nash`
(simpdiv plus an exact maximum-regret check), `conditional_logit` and
`qre_marginals` estimation with profile-likelihood intervals, bounded
sensitivity analysis, seeded episode simulation, and CFR/MCCFR on the two-player
zero-sum OpenSpiel allowlist.

Resource limits (timeouts, caps, worker seriality) come from `.env`; a missed
limit returns a structured `solver_unavailable`, never a silent fallback.

### Additional MCP tools

| tool | notes |
| --- | --- |
| `list_game_templates` | template ids and their typed options |
| `infer_game_model` | LLM intake; never runs a solver |
| `solve_game` | `confirmed_by_user` must be `true`; `methods`, optional `epsilon`, `mediator_confirmed` for `correlated_equilibrium` |
| `estimate_game_parameters` | approximate; requires confirmation |
| `analyze_sensitivity` | at most 256 rational assignments |
| `simulate_game` | `simulation`, `cfr`, or `mccfr` |
| `save_case` / `list_cases` / `load_case` / `delete_case` | encrypted single-owner case store |

### Configuration

The web app and the case store need extra variables (see `.env.example`):
`GT_APP_USERNAME` / `GT_APP_PASSWORD` (the whole app, including paper mode,
fails to start without them), `GT_CASE_ENCRYPTION_KEY` (32-byte Base64 AES-GCM
key - losing it makes saved cases unreadable) and `GT_CASE_DB_PATH`.
`game-cases.sqlite3*` is gitignored. Saved cases re-run the confirmed analysis
server-side, encrypt title/request/report with AES-GCM, and bind the metadata
via associated data.

In the **Plan a decision** tab, any edit to the details, or a new draft or opened
plan, clears the previous answer, raw result, share link, what-if result and
confirmation. Confirmation is bound to the exact details shown: pressing
**Looks right** or ticking the confirmation box confirms that spec only, and
what-if tools and **Save this plan** refuse to run for any other spec. Results
that finish after the details changed are discarded.

`correlated_equilibrium` runs only when **Someone can enforce a joint plan
(mediator)** is ticked; otherwise the method reports `unsupported` with that
reason instead of proposing an unenforceable plan.

### Solver stack

`pygambit==16.7.0` (no Linux/macOS wheels - built from sdist with Cython and a
C++ toolchain), `open_spiel==2.0.2`, exact rational LP via SymPy's simplex
primal/dual pair, `scipy` for bounded estimation. The repository stays MIT;
Gambit is GPL-2.0-or-later, so any image push needs notices and a source
offer, both recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). A
GPL review of that notice is still an open human sign-off.

### Sharing a finished analysis

Both tabs expose **Download PDF** and **Public link** once an analysis is
rendered:

- *Download PDF* opens the evidence as a self-contained HTML page (KaTeX from
  CDN) in a new tab and shows the browser print dialog - save as PDF locally.
- *Public link* uploads the rendered HTML to a Cloudflare Worker at
  `share.jakub.team` and returns a random-id URL (e.g.
  `https://share.jakub.team/gb2bg6g22w`). Links are `noindex`, expire after
  **90 days**, and serve read-only HTML.

The worker lives in `share_worker/` (`wrangler deploy`; writes require the
`SHARE_TOKEN` worker secret, reads are public by design). Configure the app
with `GT_SHARE_TOKEN` (same value as the worker secret) and
`GT_SHARE_ENDPOINT`; uploads fail closed with an error message when the token
is missing or wrong. In k8s both values ride in `game-case-secrets`.

## License

MIT. See [LICENSE](LICENSE).
