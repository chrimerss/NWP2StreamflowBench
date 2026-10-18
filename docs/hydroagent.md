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

## Baselines (median test NSE / KGE)

| Method | 665 basins, 5-fold CV | Pilot, 30 targets |
|---|---|---|
| Gauged upper bound (own calibration) | 0.662 / 0.723 | 0.685 / 0.747 |
| Attribute similarity, 5 donors, output averaging | 0.623 / 0.616 | 0.662 / 0.645 |
| Spatial proximity, 5 donors, output averaging | 0.613 / 0.614 | 0.618 / 0.604 |
| Nearest donor | 0.555 / 0.593 | 0.625 / 0.652 |
| Median donor parameters | 0.451 / 0.473 | 0.490 / 0.537 |

Success criterion: the agent beats attribute similarity on median NSE.

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
