# Pilot 0 (2026-10-18): four replicates of the base harness

The first Claude-backed run of `fold1-pilot30` (k=1, parallel). Every Harness² improver
session failed: the pinned OpenCode sent Claude Opus 5.5 the legacy
`thinking: {type: "enabled"}` request for effort variant `high`, which the model rejects
(HTTP 400). Harness² therefore fell back to the base harness for `inv1`, `dom1` and `agg1`,
so all four rollouts (`base1`, `inv1`, `dom1`, `agg1`) ran the *same* base harness with
Claude Opus 5.5 as executor. The `role` labels in the CSVs are Harness²'s slot names, not
distinct harnesses.

They are kept as a measure of run-to-run variability of one harness: median test NSE
0.645 / 0.659 / 0.658 / 0.647 and median KGE 0.634 / 0.658 / 0.690 / 0.696 across the four
replicates, against attribute-similarity transfer at 0.662 / 0.645. Executor cost:
$1.20–1.78 per rollout. Fixed by `hydroagent setup`, which teaches OpenCode the adaptive
effort API of Claude 5.x models.
