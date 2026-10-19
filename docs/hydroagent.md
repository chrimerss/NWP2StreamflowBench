# hydroagent — a Claude hydrologic-modelling agent refined with Harness²

An LLM agent calibrates the dCREST rainfall–runoff model on gauged CAMELS-US donor
basins and predicts daily discharge in held-out, anonymised "virtual ungauged" basins.
Google's [Harness²](https://github.com/google-research/harness2) (Xu et al. 2026,
arXiv:2610.14224) refines the agent's harness — system prompt, guardrails, memory,
tool guidance, skills, scripts, subagents, plugins — between attempts, with the model
weights fixed. Results are compared with classical regionalization of the released
dCREST-CAMELS v1.0 parameters on the community benchmark test period (reference
forcing, not forecast mode). Dashboard tab: `docs/agent.html`.

## Can Claude be the backbone? Yes, with a 10-line patch

Harness² has two model roles. Both run through OpenCode, which accepts
`anthropic/<model>` ids with `ANTHROPIC_API_KEY`.

| Role | Upstream behaviour (commit 1cfe254) | Here |
|---|---|---|
| Executor (does the hydrology) | adapter-owned launcher; the key passes through | `anthropic/claude-opus-5-5` by default |
| Improver (proposes, composes, reflects, selects) | `core/proposer.py` strips `ANTHROPIC_API_KEY` and forces Claude through Vertex AI | `h2adapter.patch_claude()` keeps the key; default `anthropic/claude-opus-5-5` |
| Judge | benchmark LLM judges | not used: deterministic NSE/KGE against USGS |

The patch is applied at runtime; the Harness² checkout itself is unmodified.

## Protocol

* **Data.** CAMELS v1.2 Daymet basin-mean forcing and USGS discharge (Zenodo 15529996,
  CC BY 4.0), Oudin PET. `hydroagent prepare` rebuilds the 665-basin record and checks it
  by re-running the released parameters: median test NSE 0.6616 vs stored 0.6616.
* **Periods.** Donor observations 1999-10-01..2008-09-30 (the calibration period of the
  released parameters); prediction scored 1989-10-01..1999-09-30. Every simulation starts
  from dcrest's default states three years before its window, as the parameters were
  calibrated; a 1980 spin-up degrades two slow-aquifer basins badly (NSE 0.23 → −95).
* **Folds.** 5 region-stratified random folds (`data/hydroagent/folds.csv`); each fold is
  held out once with the other 532 basins as donors. Pilot task `fold1-pilot30`: 30
  region-stratified fold-1 targets.
* **Leakage controls.** Targets have forcing and attributes only; ids are anonymous
  (T###, D####); coordinates rounded to 0.1°; no gauge names or HUCs; streamflow
  signatures (`camels_hydro`) withheld. OpenCode web tools are denied, shell tools get a
  dead HTTP proxy while OpenCode itself reaches the model API, and every trajectory is
  audited for data-download patterns. The guardrail forbidding internet data is a hard
  harness invariant (`validate_tree`), so Harness² cannot edit it away.
* **Agent action space.** Any regionalization scheme, recalibration (gradient descent via
  `scripts/hydrokit.py`), ensembles, and edits to the model structure in the workspace's
  own copy of `dcrest/`.

## Calibration framing (primary experiment)

The question: given only raw data for the training gauges of a split, does a Claude agent
refined by Harness² calibrate dCREST and predict out-of-distribution basins better than
dCREST's own differentiable optimizer?

* **Agent** (`--variant calib`, base harness `harness_calib/`): no pre-calibrated
  parameters. It calibrates the training gauges itself (per basin, jointly, or by learning
  an attribute-to-parameter mapping) and decides how to predict the held-out basins.
* **Baseline** (`hydroagent dpl`, `dpl.py`): differentiable parameter learning. An MLP
  (two hidden layers of 256) maps 23 standardised CAMELS attributes to the 16 dCREST
  parameters; it is trained by Adam (lr 3e-3, up to 500 steps) through dCREST on all
  training gauges at once with loss mean(1 − NSE) over 1999-10..2008-09 (spin-up from
  1996-10), keeps the step with the best validation median NSE on 10% held-out training
  gauges, and is applied unchanged to the held-out basins.
* **Splits.** Five region-stratified random folds and nine leave-one-climate-region-out
  splits. The agent ran on one split fixed before any result was seen: the largest region,
  Southeast (126 held-out basins, 539 training gauges).

### Result: Southeast held out (median test NSE / KGE, 1989–1999)

Reference baseline: the 3-seed ensemble of the parameter-learning network (discharge
averaged over seeds; single seeds score 0.529, 0.546 and 0.535).

| Method | NSE | KGE | beats ensemble | p (NSE vs ensemble) |
|---|---|---|---|---|
| Gauged upper bound (each basin's own released calibration) | 0.682 | 0.717 | 94/126 | < 0.001 |
| Agent + Harness², selected harness | 0.585 | 0.627 | 81/126 | 0.025 |
| Harness² candidate, general edits | 0.607 | 0.630 | 90/126 | < 0.001 |
| Harness² candidate, domain edits | 0.595 | 0.630 | 77/126 | 0.059 |
| Agent, base harness | 0.546 | 0.626 | 66/126 | 0.34 |
| Released per-basin calibrations + attribute transfer (reference) | 0.565 | 0.650 | 72/126 | 0.20 |
| Differentiable parameter learning, 3-seed ensemble (baseline) | 0.559 | 0.627 | | |
| Differentiable parameter learning, single seed | 0.529 | 0.618 | 31/126 | < 0.001 |

p: paired Wilcoxon signed-rank test across the 126 basins. Against every *single* baseline
seed all three Harness² harnesses win (largest p = 0.014), and all three beat the base
harness (p < 0.001). Cost: $10.6 for the four agent rollouts and $8.4 for the Harness²
sessions (Claude Opus 5.5).

* **The agent alone ties the optimizer; Harness² makes it better.** The base harness
  already calibrated all 539 gauges, built its own regional network and compared about 40
  transfer schemes, but finished level with the baseline. Harness² turned that first
  attempt's discoveries into harness content: a training basin with all-missing attributes
  that froze joint training, and a hold-out design measurably easier than the real task
  (hold-out basins 88 km from the nearest training gauge against 176 km for the targets),
  fixed with a 150 km exclusion buffer. The later attempts started from those facts.
* **Against the strongest baseline the margin is real but modest** for the selected harness
  (+0.026 median NSE, p = 0.025) and large for the general-edit candidate (+0.048,
  p < 0.001). Harness² selects blindly, so it did not pick its best candidate.
* **Caveats.** One split; one base-harness replicate in this framing; Harness² used four
  attempts where the base harness used one. A k=10 sequential run with Claude Sonnet 5.5
  (in progress) adds ten base-harness replicates and ten optimization rounds.

### Differentiable baseline on every split (median test NSE of held-out basins)

| Split | Kind | Held out | Baseline (single seed) | Released calibrations + attribute transfer | Gauged upper bound |
|---|---|---|---|---|---|
| fold1 | random-fold | 133 | 0.587 | 0.604 | 0.651 |
| fold2 | random-fold | 133 | 0.630 | 0.644 | 0.664 |
| fold3 | random-fold | 133 | 0.606 | 0.635 | 0.671 |
| fold4 | random-fold | 133 | 0.611 | 0.619 | 0.656 |
| fold5 | random-fold | 133 | 0.607 | 0.602 | 0.659 |
| Northeast | leave-region-out | 91 | 0.591 | 0.585 | 0.668 |
| Northern Rockies and Plains | leave-region-out | 63 | 0.319 | 0.318 | 0.464 |
| Northwest | leave-region-out | 78 | 0.619 | 0.685 | 0.776 |
| Ohio Valley | leave-region-out | 69 | 0.630 | 0.606 | 0.717 |
| South | leave-region-out | 97 | 0.508 | 0.513 | 0.602 |
| Southeast | leave-region-out | 126 | 0.529 | 0.565 | 0.682 |
| Southwest | leave-region-out | 54 | 0.236 | 0.360 | 0.548 |
| Upper Midwest | leave-region-out | 36 | 0.352 | 0.377 | 0.631 |
| West | leave-region-out | 51 | 0.540 | 0.467 | 0.689 |

Out of distribution the baseline loses most in the dry and snowy regions (Southwest 0.24,
Northern Rockies and Plains 0.32, Upper Midwest 0.35), where training gauges from other
climates teach the network little.

## Transfer-only pilot: classical baselines (median test NSE / KGE)

| Method | 665 basins, 5-fold CV | Pilot, 30 targets |
|---|---|---|
| Gauged upper bound (own calibration) | 0.662 / 0.723 | 0.685 / 0.747 |
| Attribute similarity, 5 donors, output averaging | 0.623 / 0.616 | 0.662 / 0.645 |
| Spatial proximity, 5 donors, output averaging | 0.613 / 0.614 | 0.618 / 0.604 |
| Nearest donor | 0.555 / 0.593 | 0.625 / 0.652 |
| Median donor parameters | 0.451 / 0.473 | 0.490 / 0.537 |

Success criterion: the agent beats attribute similarity on median NSE.

## Transfer-only pilot results (2026-10-18, `fold1-pilot30`, k=1 parallel, Claude Opus 5.5 for both roles)

Median test NSE / KGE over the 30 held-out targets:

| Run | NSE | KGE | targets beating attribute similarity |
|---|---|---|---|
| Gauged upper bound | 0.685 | 0.747 | |
| Agent + Harness², selected harness (`agg1`) | 0.664 | 0.664 | 22/30 |
| Harness² probe, domain edits (`dom1`) | 0.668 | 0.710 | 16/30 |
| Harness² probe, general edits (`inv1`) | 0.668 | 0.659 | 20/30 |
| Agent, base harness (`base1`) | 0.664 | 0.659 | 21/30 |
| Attribute similarity, 5 donors (best classical) | 0.662 | 0.645 | |

* **The agent matches the best classical method.** Every rollout lands at or above
  attribute similarity on median NSE and, in all but one, on median KGE, and the agent's
  mean NSE (clipped at −1) is higher in all eight rollouts (0.50–0.54 vs 0.48). No paired
  difference is significant after accounting for multiple comparisons.
* **Harness² did not yet produce a measurable gain.** Five base-harness replicates (four
  from pilot 0, where the editor failed, plus pilot 1's base) give median NSE 0.645–0.664
  (sd 0.008). The refined harnesses (0.664–0.668) sit at the top of that band, so one k=1
  step on 30 basins cannot separate them from run-to-run noise.
* **The edits were sensible.** The general candidate added a deliverable checker and a
  donor hold-out skill; the domain candidate rewrote the regionalization skill with facts
  the base run established in its own cross-validation (a donor with missing coordinates
  that made hold-outs select themselves, 1/d² weighting beating equal weighting, regime
  classes). The reflector stored the lesson "gate every ranking on a K=1-below-ceiling
  check". No rollout changed the model structure, and no rollout attempted a download.
* **Cost.** Executor rollouts $1.2–1.9 each; improver sessions $1.8 each. Pilot 0 $7.8,
  pilot 1 $11.4.

Next: run the five full folds (133 targets each) in sequence so the Harness² library
carries lessons across tasks, which also gives the statistical power the 30-basin pilot
lacks (roughly $12 per fold at k=1).

## Gotchas found while wiring Claude in

* Harness² strips `ANTHROPIC_API_KEY` from improver sessions (patched at runtime).
* A parent Claude Code session exports `ANTHROPIC_BASE_URL` without `/v1`; OpenCode then
  calls `/messages` and gets 404. Both roles drop inherited base URLs.
* The pinned OpenCode maps effort `high` to the legacy thinking budget for any Claude model
  newer than Opus 4.8, which Claude 5.x rejects; Harness² then silently falls back to the
  base harness. `hydroagent setup` patches OpenCode, and `hydroagent run` now fails loudly
  when any stage falls back.

## Running it

```bash
pip install -e ".[agent,basins]"                 # harness2 pinned at 1cfe254
hydroagent setup                                 # pinned OpenCode + bun, as harness2-setup does
hydroagent prepare                               # needs the CAMELS archive in ~/.cache/nwp2sf/camels
echo 'ANTHROPIC_API_KEY=...' > .env              # git-ignored
hydroagent check
hydroagent run --task fold1-pilot30 --k 1 --mode parallel
hydroagent evaluate && hydroagent report
```

Harness² resolves `dependencies/` relative to its own checkout when installed editable
(here `~/.cache/hydroagent/harness2`), or to the working directory otherwise.
`hydroagent run --offline` replaces both model roles
with scripted stand-ins and exercises the whole loop without API calls.

## Cost

A k=1 Harness² step is 4 executor rollouts and 3 improver sessions. At Claude Opus 5.5
prices ($4 / $20 per million input / output tokens, cache reads $0.20) a long agentic
rollout is roughly $5–15 and an improver session $3–8, so the pilot is roughly $40–80.
`--executor anthropic/claude-sonnet-5-5` ($2 / $10) halves the executor share. The run
ledger in each task summary records the actual cost.

## Known limitations

* The network fence is soft: a determined agent could unset the proxy variables. The
  audit flags such runs; it does not prevent them.
* The released parameters are themselves the donor set's calibration, so baselines and
  agent start from the same donor information; the comparison measures transfer skill.
* Harness² refines a harness for *one* task at a time and carries lessons forward through
  its library; running all five folds in sequence is what tests whether lessons transfer.
