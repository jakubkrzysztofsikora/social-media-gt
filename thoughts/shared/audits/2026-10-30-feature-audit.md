# social-media-gt — feature audit and live verification

Date: 2026-10-30. Auditor: omp agent session (skills `feature-audit`, `social-gt-deploy-verify`).
Repo: `~/Repos/personal/math/social-media-gt`. Live: <https://social-gt.tail5d39b4.ts.net> (namespace `social-gt`, homelab k3s).

## Level definitions

| Level | Meaning |
| --- | --- |
| L0 | Specified — described in README / plan document |
| L1 | Implemented — code exists and is importable |
| L2 | Tested — covered by an automated test |
| L3 | Exposed — reachable from the HTTP UI or the MCP server |
| L4 | Deployed — running in the cluster from a pinned image |
| L5 | Verified live — exercised end to end against the deployed instance |

## Verdict per surface

| Surface | L0 | L1 | L2 | L3 | L4 | L5 |
| --- | --- | --- | --- | --- | --- | --- |
| Paper scenarios tab (10 exact-arithmetic scenarios) | yes | yes | yes | yes | yes | yes |
| Plan-a-decision tab (template compiler + 3 solver methods) | yes | yes | yes | yes | yes | yes |
| MCP server (`solve_game`, `game_solve_tagged`) | yes | yes | yes | yes | yes | partial — no live MCP client run in this audit |
| Case store / share worker | yes | yes | yes | yes | yes | no — untested live in this audit |
| GPL / third-party compliance | yes | yes | n/a | n/a | yes | no — legal sign-off open |

## Defects found and fixed in this audit

| # | Defect | Impact | Fix | Evidence |
| --- | --- | --- | --- | --- |
| 1 | `mixed_nash` / `epsilon_nash` witnesses carried raw `Fraction` | `json.dumps` raised; MCP and UI both failed on those methods | render through `render_rational` in `game_solvers.py` | unit tests + in-pod `webapp.game_solve` run |
| 2 | `mediator_confirmed` implemented in advisor/worker but absent from `mcp_server.solve_game` and the UI | `correlated_equilibrium` unreachable from every entry point | parameter added to MCP tool, checkbox `#game-mediator` in UI, threaded through `game_solve` / `game_solve_tagged` / worker | live UI: `#game-mediator` visible; in-pod solve returns `correlated_equilibrium` `exact_verified` |
| 3 | `negotiation` and `public_goods` templates advertised in the catalog but had no compiler | selecting them produced no model | `_compile_negotiation` / `_compile_public_goods` in `game_models.py`, registered in `_COMPILERS` | 4 new tests in `test_game_models.py` |
| 4 | `plain_view.describe_result` silently dropped unsupported methods | user unticking the mediator got no explanation | added a `### Not applied` section carrying the reason | `test_inapplicable_methods_are_listed_with_their_reason` |
| 5 | Paper-domain vocabulary leaked into the domain-neutral report (`a real class`, `cohort`, `households`) | foreign-domain report read as if it were about a school class | substitutions added to `_NEUTRAL_CONCLUSIONS`; banned-word test extended to `class`, `cohort`, `household` | live in-pod render of all 10 scenarios: 0 banned terms |
| 6 | GPL-2.0 obligations of the `pygambit` sdist were undocumented | distribution without licence notices | `THIRD_PARTY_NOTICES.md`, `LICENSE` + notices copied into the image, `.dockerignore` corrected | image contents; README § Licence |
| 7 | Probes used the 1 s default timeout against a 1-CPU limit | readiness flapped (`context deadline exceeded`) while an analysis held the CPU | `timeoutSeconds: 5`, `failureThreshold: 6` on both probes | manifest applied; no probe failures since |
| 8 | 10 stale ReplicaSets accumulated | cluster noise | `revisionHistoryLimit: 3` | `kubectl get rs` → 4 |

## Open items

| # | Item | Severity | Note |
| --- | --- | --- | --- |
| O1 | GPL-2.0-or-later legal review of distributing the `pygambit` sdist | medium | notices are in place; sign-off is a human decision, not a code change |
| O2 | ~~25 Python modules are untracked in git~~ | resolved | all modules, tests, manifests and docs are committed; `main` = `d318a69`, tag `v-audit-neutral`; pod files are byte-identical to `HEAD` |
| O3 | `requirements.txt` mixes pins with open ranges (`z3-solver>=4.12`, `openai>=1.40`, `python-dotenv>=1.0`, `numpy>=2.0`, `pytest>=8`) | medium | image builds are not reproducible; the Dockerfile wheelhouse also omits `gradio`, `openai`, `python-dotenv`, `z3-solver`, so the build needs PyPI access |
| O4 | MCP server and share worker were not exercised live in this audit | low | verified by unit tests and in-pod imports only |
| O5 | `#game-input` lives only in the second tab | low | not a defect, but browser probes must switch tabs before waiting on it; the login page is a Gradio form, not HTTP basic auth |
| O6 | `.DS_Store` untracked at repo root | trivial | hygiene |

## Live evidence

| Check | Result |
| --- | --- |
| `GET /` over the funnel | 200 |
| Login form (`GT_APP_USERNAME` / `GT_APP_PASSWORD` from secret `game-case-secrets`) | succeeds; tabs `Paper scenarios`, `Plan a decision` render |
| Dark-mode probe (emulated `prefers-color-scheme: dark`) | body `rgb(255,255,255)`, text `rgb(43,43,43)`, 0 opaque near-black elements |
| `#game-mediator` checkbox | visible in the Plan-a-decision tab |
| Foreign-domain analysis (12 villages, one irrigation canal) | report 15 478 chars: `Mapping` in the user's vocabulary, `Calculations summary` table, `Practical conclusions`; **10 scenarios run, 10 ok, 43 checks, 43 passed, 0 failed**; 0 banned terms |
| In-pod scenario render after the vocabulary fix | `tw1`, `tw6`, `tw8` conclusions domain-neutral; all 10 scenarios free of `parent` / `social media` / `screen` / `famil` / `class` / `cohort` / `household` |
| Deployed image | `forgejo.tail5d39b4.ts.net/jakub/social-gt@sha256:a12a78361570205301d92699b386ea84c405b37b74cd9b292833063e814ccce7` |
| Test suite | 246 passed |

## Notes for the next session

- Warm-cache rebuild of the amd64 image takes about 20 s; the cold `pygambit` sdist build under QEMU takes over an hour and will exceed a 60 min deadline.
- Pin the new image digest in `k8s/deployment.yaml` after every push; `:latest` is not used by the manifest.
- A live analysis costs 3–5 minutes (two LLM calls). Wait on text only a finished run produces (`Practical conclusions`, `scenarios run`), never on pre-existing chat text.