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
| `tw7` | a rule covering the whole group flips the cascade | `--lambda --eta --eps` |
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

## License

MIT. See [LICENSE](LICENSE).
