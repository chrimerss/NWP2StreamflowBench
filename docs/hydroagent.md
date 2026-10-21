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
  attempts where the base harness used one. The two experiments below add base-harness
  replicates, more rounds, and three runs per harness.

### More rounds: k=10 sequential with Claude Sonnet 5.5 (stopped after round 3)

Same Southeast task, both roles on Claude Sonnet 5.5 (half the Opus price per token).
Sequential mode refines the previous round's composed harness each round. An early-stop rule
ended the run after three successive rounds in which the composed harness failed to beat the
best so far by more than the base-harness spread (δ = 0.017 median NSE).

| | median NSE |
|---|---|
| Ten base-harness runs | 0.531–0.587 (mean 0.559, sd 0.017) |
| Round 1: general / domain / composed | 0.552 / 0.539 / 0.553 |
| Round 2: general / domain / composed | 0.555 / 0.510 / 0.555 |
| Round 3: general / domain / composed | 0.555 / 0.580 / 0.540 |
| Differentiable baseline, 3-seed ensemble | 0.559 |

* **No gain from more rounds with Sonnet.** The Sonnet base harness already matches the
  baseline ensemble on average; no refined harness beat it significantly, and one domain
  candidate was significantly worse (round 2, p = 0.002).
* **Why: the harness converged and stopped exploring.** Each round Harness² wrote more of the
  procedure into scripts (`attrs.py`, `deliverable.py`, then `jobs.py` and `schemes.py`) and
  the always-on text grew from 3.3k to 6.3k characters. From round 2 the agent mostly
  re-ran the scripted pipeline in 13–15 turns, and three runs produced byte-identical
  predictions. Harness² locked in the first workable procedure rather than a good one.
* **Run-to-run noise matters.** The ten base runs span 0.056 in median NSE, about as wide as
  every harness effect measured here, so single-rollout comparisons (as at k=1) are fragile.
* **Opus vs Sonnet.** The Opus k=1 candidates (0.585–0.607) sit at or above the best of the
  ten Sonnet base runs, but the comparison mixes model and harness, so it does not show
  which role made the difference.
* **Cost and time.** $43.0 in total ($25.0 for 19 agent runs, $14.1 for 9 Harness²
  sessions, $4.0 lost to runs cut short by OpenCode's doom-loop guard before it was
  disabled), 12.7 hours.

### Which role matters: Opus vs Sonnet as the Harness² editor (3 runs per harness)

Same Southeast task. The agent is fixed at Claude Sonnet 5.5 and only the editor (the
Harness² improver) changes. A parallel k=3 Harness² run with Claude Opus 5.5 as editor
produced nine candidate harnesses: three with general edits, three with domain edits and
three composed. Every candidate was then run twice more, as were the three round-1
harnesses that Sonnet edited in the k=10 run. The base harness has 13 runs (ten from the
k=10 run, three from this one).

| Harness | Editor | Median test NSE of each run | Mean | p vs base runs |
|---|---|---|---|---|
| Base harness | – | 13 runs, 0.531–0.615 | 0.564 (sd 0.022) | |
| General edits | Sonnet 5.5 | 0.552 · 0.592 · 0.582 | 0.575 | 0.52 |
| Domain edits | Sonnet 5.5 | 0.539 · 0.538 · 0.550 | 0.542 | 0.08 |
| Composed | Sonnet 5.5 | 0.553 · 0.562 · 0.510 | 0.542 | 0.24 |
| General edits 1 | Opus 5.5 | 0.544 · 0.586 · 0.542 | 0.557 | 0.61 |
| General edits 2 | Opus 5.5 | 0.575 · 0.594 · 0.594 | 0.588 | 0.06 |
| General edits 3 | Opus 5.5 | 0.570 · 0.577 · 0.562 | 0.570 | 0.36 |
| Domain edits 1 | Opus 5.5 | 0.582 · 0.577 · 0.586 | 0.582 | 0.15 |
| Domain edits 2 | Opus 5.5 | 0.587 · 0.600 · 0.611 | 0.600 | 0.025 |
| Domain edits 3 | Opus 5.5 | 0.545 · 0.601 · 0.595 | 0.580 | 0.44 |
| Composed 1 (selected by Harness²) | Opus 5.5 | 0.555 · 0.579 · 0.590 | 0.575 | 0.44 |
| Composed 2 | Opus 5.5 | 0.588 · 0.580 · 0.565 | 0.578 | 0.19 |
| Composed 3 | Opus 5.5 | 0.566 · 0.574 · 0.594 | 0.578 | 0.19 |
| Differentiable baseline, 3-seed ensemble | | 0.559 | | |

The first value in each row is the run Harness² itself saw. p: Mann–Whitney test of the
harness's run medians against the 13 base-harness run medians.

| Group | Harnesses | Mean of harness means | Runs above the baseline ensemble | p, runs vs base runs |
|---|---|---|---|---|
| Base harness | 1 | 0.564 | 7 of 13 | |
| Sonnet 5.5 editor | 3 | 0.553 | 3 of 9 | 0.26 |
| Opus 5.5 editor | 9 | 0.579 | 23 of 27 | 0.023 |

Opus against Sonnet editor, harness means: Mann–Whitney p = 0.064.

* **The editor is the role that matters.** With the same Sonnet agent, eight of nine
  Opus-edited harnesses beat the base mean, and 23 of 27 of their runs beat the baseline
  ensemble. Only one of three Sonnet-edited harnesses beat the base mean, and the other two
  were worse. With three Sonnet harnesses the editor difference is suggestive, not
  conclusive (p = 0.064).
* **The gain is modest.** On average an Opus-edited harness adds 0.015 median NSE over the
  base harness and 0.020 over the baseline ensemble. The best one (domain edits 2) adds
  0.036 and 0.041 and is the only harness significant on its own at the run level. Picking
  it after seeing test scores is selection on the test set, so the fair estimate for one
  Harness² run is the selected harness (0.575) or the group mean (0.579).
* **Harness² still selects poorly.** It chose composed 1, seventh of nine, from one run per
  candidate. One run is a weak signal here: the spread across runs of one harness reaches
  0.056, and domain edits 3 scored 0.545 on the run Harness² saw and about 0.60 on both
  replicates. Composing did not help either: the three composed harnesses sit mid-pack,
  below the best general and domain edits.
* **Sharing edits across agents works.** An Opus-written harness lifts a Sonnet agent, so the
  expensive model is needed only for the few editor sessions, not for every rollout.
* **What this design cannot separate.** The Sonnet harnesses come from round 1 of the
  sequential run, which also starts from the base harness, so the comparison is editor
  against editor at the first edit. It does not show whether parallel mode beats
  sequential mode over more rounds.
* **Cost and time.** $89.5 in total over 13.6 hours: $44.4 for 11 Opus editor sessions,
  $16.6 for 12 agent runs inside Harness², and $28.6 for 24 replicate runs. A Sonnet agent
  run costs about $0.9–1.4.

### Experiment 4: Gemini 4 Argon as the agent (editing stage pending)

Same Southeast task and design as experiment 3, with the agent run by Google's Antigravity
CLI on Gemini 4 Argon (High) and Claude Opus 5.5 as the Harness² editor. The agent's web
search, URL tools and downloads are blocked (see "Antigravity CLI as the agent" below).

| Base harness | Runs | Mean median NSE (sd) | Runs above the baseline ensemble |
|---|---|---|---|
| Gemini 4 Argon (High) agent | 17 | 0.571 (0.022) | 13 of 17 |
| Sonnet 5.5 agent (experiment 3) | 13 | 0.564 (0.022) | 7 of 13 |
| Sonnet 5.5 agent, Opus-edited harnesses (experiment 3) | 27 | 0.579 | 23 of 27 |
| Differentiable baseline, 3-seed ensemble | | 0.559 | |

Gemini against Sonnet on the base harness, run medians: Mann–Whitney p = 0.24. Gemini run
medians range from 0.519 to 0.607; rollouts take 32–53 minutes.

* **On the unedited harness the Gemini agent edges out the Sonnet agent** (0.571 against
  0.564, not significant) and beats the baseline ensemble in 13 of 17 runs against 7 of 13.
  It lands between the Sonnet base harness and the Opus-edited Sonnet harnesses.
* **The editing stage has not run.** The Anthropic API key ran out of credit on 2026-10-21,
  so all nine Opus editing sessions failed and Harness² fell back to the base harness for
  each. Those nine rollouts ran the byte-identical base harness and are counted as
  base-harness runs above. Editor failures on billing or authentication now stop the run
  instead of falling back, and the experiment chain only replicates candidates after a
  clean Harness² stage.
* **Quota.** The Antigravity subscription quota ran out once (after about 3.5 rollouts at
  roughly 300k input, 110k output and 8–13 million cached input tokens each) and was
  refreshed by the user; the refreshed credits covered 14 further rollouts without a stop.
* **Two base runs were invalid and are archived.** A permission bug (agy refused the agent's
  read of agy's own saved output once the private home moved to a `/var/...` path) ended
  them after 2 minutes. Homes are now resolved paths with an explicit read grant, the fix
  held in all later rollouts, and a rollout that ends on such a refusal is left out.
* **Resume the editing stage** with `~/.cache/hydroagent/exp4_resume.sh` once the editor's
  model account has credit; the base runs are cached and reused.

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

### Antigravity CLI as the agent

An executor ID of the form `agy/<model slug>` runs each rollout with Google's Antigravity
CLI (`agy` 1.4.3) in headless mode instead of OpenCode. It draws on the Google account's
subscription credits. The improver is unchanged, so Claude still edits the harness. The
subscription lists Gemini 4 Argon, Gemini 3.8 Flash, Claude Opus 5.5 and Sonnet 5.5, and
GPT-OSS 120B (`agy models`).

```bash
curl -fsSL https://antigravity.google/cli/install.sh | bash   # installs ~/.local/bin/agy
mkdir -p "$TMPDIR/hydro_agy_ws" && cd "$TMPDIR/hydro_agy_ws"
agy                                              # once, interactively: sign in and trust this folder
hydroagent agy-smoke --model gemini-4-argon-high # 12 PASS/FAIL checks
hydroagent run --task region-Southeast --variant calib --k 3 --mode parallel \
    --executor agy/gemini-4-argon-high --improver anthropic/claude-opus-5-5
hydroagent replicate --task region-Southeast --variant calib --executor agy/gemini-4-argon-high \
    --harness base=base --n 3 --start 1
```

* **Harness delivery.** Harness² edits the same tree as for OpenCode. Each workspace gets
  the always-on text as `AGENTS.md`, skills under `.agents/skills/`, and `scripts/`.
* **Workspaces** live under `$TMPDIR/hydro_agy_ws` (`HYDRO_AGY_WS_ROOT`), outside the
  hydroagent cache that holds the hidden target mapping and the full CAMELS record.
* **Isolated home.** Each rollout runs with `HOME` pointed at a per-rollout directory that
  holds copies of agy's sign-in and onboarding files and one link, to
  `~/Library/Keychains`, where agy keeps its token. The user's global `GEMINI.md`,
  conversations, memories, MCP servers and other credentials (`~/.config/gcloud`) never
  reach a rollout, and the user's own files are never written.
* **Permissions.** That home's settings allow shell commands and writes inside the
  workspace, and deny URL reading, browser actions, MCP tools, nested `agy`, and download or install
  commands. agy's default mode asks before every write, which a headless run refuses.
* **No web search.** `search_web` runs on Google's side, so neither a permission rule nor
  the fence can stop it; in the first smoke test it identified a gauge from its USGS
  number. A `PreToolUse` hook in the isolated home's `~/.gemini/config/hooks.json` denies
  it and every browser or URL tool. The agent's file tools cannot write there.
* **Network fence.** Each rollout gets a local proxy that tunnels only to the hosts `agy`
  needs, and only when the connecting process is the `agy` binary itself, checked with
  `lsof` and the process's executable path. The agent's Python therefore cannot reach
  Google's model APIs or `www.googleapis.com`, which also serves Cloud Storage downloads.
  Every refusal is recorded in `proxy.json` and flagged by the audit.
* **Failures stop the run.** A rollout that ends on sign-in, quota or empty output raises
  an error that Harness² does not swallow, so a spent quota cannot silently decide
  harness selection; the run resumes later. SIGTERM from the budget or early-stop guards,
  and Ctrl-C, end agy and everything the agent started and stop the run at once.
* **Separate caches.** The substrate token carries the model, so namespaces
  (`camels-calib-ag-gemini-4-argon-high-agg3`), replicates and the reflection library
  never mix executors. A cached rollout is reused only by the executor that made it.
* **Cost tracking.** Rollouts carry no dollar cost; only the improver's API spend is
  metered. The `/usage` command in an interactive `agy` session shows the credits used.

Smoke test, 2026-10-20, Gemini 4 Argon (High): all 14 checks pass. The CLI runs headless in
a workspace below the trusted folder, reads a 15,000-character `AGENTS.md` in full (the
rules cap in the web docs does not bind), finds workspace skills, uses the venv's
`python3` and may write in the workspace; `curl` is denied by rule, Python downloads are
refused by the fence, the web search is refused by the hook, and the parser and audit read
the event stream.

First real rollout, same model, base harness, Southeast: delivered in 15 minutes with a
clean audit, median test NSE 0.454 (Sonnet 5.5 base-harness runs: 0.531–0.615). It ran
before workspace writes were granted, its file tool was refused, and it delivered the same
quick early prediction as the two cut-short runs of experiment 4 (identical to 1e-5); it is
archived, not counted.
agy also saves long tool outputs under its home and has the agent read them back; that
read needs an explicit grant (added after experiment 4's pause, not yet smoke-tested).

## Cost

A k=1 Harness² step is 4 executor rollouts and 3 improver sessions. At Claude Opus 5.5
prices ($4 / $20 per million input / output tokens, cache reads $0.20) a long agentic
rollout is roughly $5–15 and an improver session $3–8, so the pilot is roughly $40–80.
`--executor anthropic/claude-sonnet-5-5` ($2 / $10) halves the executor share. The run
ledger in each task summary records the actual cost.

Actual spend on the Southeast calibration task:

| Run | Cost | Wall time |
|---|---|---|
| Opus 5.5 both roles, k=1 parallel | $19 | |
| Sonnet 5.5 both roles, k=10 sequential, stopped after round 3 | $43.0 | 12.7 h |
| Opus 5.5 editor, Sonnet 5.5 agent, k=3 parallel, plus replicates | $89.5 | 13.6 h |
| Opus 5.5 editor, Gemini 4 Argon agent via Antigravity (editing pending) | $2.26 + subscription quota, 19 rollouts | about 9 h |

## Known limitations

* The network fence is soft: a determined agent could unset the proxy variables. The
  audit flags such runs; it does not prevent them.
* The released parameters are themselves the donor set's calibration, so baselines and
  agent start from the same donor information; the comparison measures transfer skill.
* Harness² refines a harness for *one* task at a time and carries lessons forward through
  its library; running all five folds in sequence is what tests whether lessons transfer.
