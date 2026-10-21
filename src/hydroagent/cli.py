"""hydroagent command line.

    hydroagent setup                       pinned OpenCode executor (via harness2's installer)
    hydroagent prepare                     CAMELS record, attributes, folds, baselines, task folders
    hydroagent dpl --splits region-Southeast   differentiable parameter-learning baseline (CPU)
    hydroagent check                       credentials, OpenCode, task folders (no model call)
    hydroagent run --task fold1-pilot30 --k 1 --mode parallel [--executor M] [--improver M]
    hydroagent evaluate --task fold1-pilot30   score every rollout against USGS; compare with baselines
    hydroagent report                      write docs/data/hydroagent.json for the dashboard tab
    hydroagent replicate --harness L=PATH ...   extra executor rollouts of given harness trees
    hydroagent run ... --offline           scripted executor and editor: end-to-end pipeline test, no API calls
    hydroagent agy-smoke [--model SLUG]    check the Antigravity CLI executor before a run
    hydroagent run ... --executor agy/SLUG   Antigravity CLI (subscription credits) as the agent

Models use OpenCode's provider/model ids, e.g. anthropic/claude-sonnet-5-5. The
Anthropic key is read from ANTHROPIC_API_KEY, else from `.CLAUDE_API` (raw key) or `.env`
in the repository root; both files are git-ignored.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config
from .regions import load_regions


def _load_env_file() -> None:
    k = config.REPO / ".CLAUDE_API"        # raw key file (git-ignored), if present
    if k.is_file() and not os.environ.get("ANTHROPIC_API_KEY"):
        lines = [x.strip() for x in k.read_text().splitlines() if x.strip() and not x.strip().startswith("#")]
        if lines:
            v = lines[0].partition("=")[2] if "=" in lines[0] else lines[0]
            os.environ["ANTHROPIC_API_KEY"] = v.strip().strip("'\"")
    f = config.REPO / ".env"
    if f.is_file():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip("'\""))


def _tasks():
    from . import camels, protocol
    d = camels.load()
    attrs = camels.attributes(d["ids"])
    regions = load_regions(d["ids"])
    return d, attrs, regions, protocol.make_tasks(attrs, regions)


def cmd_setup(a) -> int:
    """Install the pinned OpenCode executor the way harness2-setup does (clone, patch, bun)."""
    from .h2adapter import configure
    configure()
    from harness2 import config as h2c, install
    install.clone("opencode")
    install.patch("opencode")
    install.run(["npm", "install", "--prefix", h2c.DEPENDENCIES / "tools", "bun@1.3.11"])
    install.run([h2c.BUN, "install", "--frozen-lockfile", "--ignore-scripts"], cwd=h2c.OPENCODE_DIR)
    _patch_opencode_adaptive(h2c.OPENCODE_DIR)
    print(f"OpenCode ready under {h2c.OPENCODE_DIR}")
    return 0


ADAPTIVE_MARK = "// hydroagent: Claude 5.x families use adaptive thinking + effort"


def _patch_opencode_adaptive(opencode_dir) -> bool:
    """The pinned OpenCode (v1.14.18) maps effort variants of Claude models newer than Opus 4.8
    to the legacy `thinking: {type: "enabled", budget_tokens}` request, which Claude 5.x models
    reject with a 400. Teach its variant table the 5.x families (idempotent)."""
    f = opencode_dir / "packages" / "opencode" / "src" / "provider" / "transform.ts"
    s = f.read_text()
    if ADAPTIVE_MARK in s:
        return False
    # insert just before the function's final `return null`, outside the context of
    # harness2's own opencode-compat.diff (which edits the top of the same function)
    head = "function anthropicAdaptiveEfforts(apiId: string): string[] | null {"
    i = s.index(head)
    anchor = "  return null\n}\n"
    j = s.index(anchor, i)
    add = (f"  {ADAPTIVE_MARK}\n"
           '  if (["opus-5", "sonnet-5", "fable-5", "mythos-5", "haiku-5"].some((v) => apiId.includes(v))) {\n'
           '    return ["low", "medium", "high", "xhigh", "max"]\n'
           "  }\n")
    s = s[:j] + add + s[j:]
    f.write_text(s)
    print(f"patched {f} (Claude 5.x adaptive effort)")
    return True


def cmd_dpl(a) -> int:
    """Differentiable parameter-learning baseline on the requested splits (CPU, no API calls)."""
    from . import camels, dpl
    d = camels.load()
    attrs = camels.attributes(d["ids"])
    regions = load_regions(d["ids"])
    S = dpl.splits(attrs, regions)
    names = list(S) if a.splits == ["all"] else a.splits
    for name in names:
        if (config.RESULTS / "dpl" / f"{name}_test.csv").is_file() and not a.force:
            print(f"{name}: done already")
            continue
        s = S[name]
        got = dpl.run_split(d, attrs, s["train"], s["test"], iters=a.iters, threads=a.threads, seed=a.seed,
                            log=lambda m, n=name: print(f"[{n}] {m}", flush=True))
        dpl.save(name, s["kind"], got)
        print(f"{name}: test median NSE {got['test'].nse.median():.3f} KGE {got['test'].kge.median():.3f} "
              f"(best iter {got['best_iter']}, {got['wall_s']} s)", flush=True)
    return 0


def cmd_replicate(a) -> int:
    """Extra executor rollouts of given harness trees: --harness label=/path/to/harness ... --n 2 --start 2.

    Rollouts land in runs/rollouts/replicates/<task>/<label>/r<i>; each records its harness path."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .h2adapter import configure, patch_claude
    configure(a.executor, a.improver)
    patch_claude()
    from .h2adapter import HydroAdapter
    task = a.task[0]
    adapter = HydroAdapter([task], timeout=a.timeout, variant=a.variant)
    jobs = []
    for spec in a.harness:
        label, _, path = spec.partition("=")
        hdir = None if path in ("", "base") else Path(path).expanduser()
        if hdir is not None and not hdir.is_dir():
            raise SystemExit(f"harness not found: {hdir}")
        jobs += [(label, hdir, f"{adapter.replicate_root}/{task}/{label}/r{i}") for i in range(a.start, a.start + a.n)]
    print(f"{len(jobs)} rollouts with {a.workers} workers, executor {os.environ.get('HARNESS2_SOLVER_MODEL')}", flush=True)
    from . import antigravity as AG
    failed = 0
    ex = ThreadPoolExecutor(max_workers=a.workers)
    try:
        futs = {ex.submit(adapter.execute, task, h, rid): rid for _, h, rid in jobs}
        for f in as_completed(futs):
            try:
                rd = f.result()
                print(f"done {futs[f]} delivered={adapter.delivered(rd)}", flush=True)
            except AG.AgyFailure as e:      # sign-in or quota: stop everything, resume later
                print(f"STOPPED at {futs[f]}: {e}", flush=True)
                ex.shutdown(wait=False, cancel_futures=True)
                AG.kill_live()
                raise
            except Exception as e:
                failed += 1
                print(f"FAILED {futs[f]}: {type(e).__name__}: {e}", flush=True)
    finally:
        ex.shutdown(wait=True)
    return int(failed > 0)


def cmd_prepare(a) -> int:
    import numpy as np

    from . import baselines, camels, protocol, workspace
    ids = [str(i) for i in np.load(config.PARAMS, allow_pickle=True)["ids"]]
    if not config.DATA_NPZ.is_file() or a.force:
        camels.build(ids)
    r = camels.reproduce_released()
    print(f"CAMELS record check: reproduced median test NSE {r.nse.median():.4f} vs stored {r.stored.median():.4f}")
    d, attrs, regions, tasks = _tasks()
    config.ATTR_CSV.parent.mkdir(parents=True, exist_ok=True)
    attrs.to_csv(config.ATTR_CSV, float_format="%.6g")
    protocol.fold_table(tasks).to_csv(config.REPO / "data" / "hydroagent" / "folds.csv")
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(1, config.N_FOLDS + 1):
        t = tasks[f"fold{k}"]
        for m, res in baselines.run(d, attrs, list(t.targets), list(t.donors)).items():
            rows.append(res.assign(method=m, fold=k).reset_index())
    per = pd.concat(rows)
    per["region"] = regions.reindex(per.gauge_id).values
    per.to_csv(config.RESULTS / "baselines_per_basin.csv", index=False, float_format="%.4f")
    allres = {m: g.set_index("gauge_id")[["nse", "kge"]] for m, g in per.groupby("method")}
    baselines.summary(allres).to_csv(config.RESULTS / "baselines_summary_all.csv", float_format="%.4f")
    for tid in (a.tasks or [f"fold1-pilot{config.PILOT_TARGETS}"]):
        workspace.build(tasks[tid], d, attrs, force=a.force, released=(a.variant == "template"), variant=a.variant)
        workspace.brief(tid, a.variant)
        if tid.startswith("region-"):          # classical transfer from the other regions' donors
            res = baselines.run(d, attrs, list(tasks[tid].targets), list(tasks[tid].donors))
            pd.concat([r.assign(method=m) for m, r in res.items()]).to_csv(
                config.RESULTS / f"baselines_per_basin_{tid}.csv", float_format="%.4f")
            sub = res
        else:
            sub = {m: r.reindex(list(tasks[tid].targets)) for m, r in allres.items()}
        baselines.summary(sub).to_csv(config.RESULTS / f"baselines_summary_{tid}.csv", float_format="%.4f")
        print(f"task {tid}: {len(tasks[tid].targets)} targets, {len(tasks[tid].donors)} donors")
    print(baselines.summary(allres).round(3).to_string())
    return 0


def cmd_check(a) -> int:
    from .h2adapter import configure
    configure(a.executor, a.improver)
    from harness2 import config as h2c
    problems = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        problems.append("ANTHROPIC_API_KEY is not set (export it, or put ANTHROPIC_API_KEY=... in .env)")
    if not h2c.BUN.is_file() or not (h2c.OPENCODE_DIR / "node_modules").is_dir():
        problems.append(f"OpenCode/bun missing under {h2c.DEPENDENCIES}; see docs/hydroagent.md")
    from . import antigravity as AG
    if AG.is_agy(h2c.SOLVER_MODEL_OC) and not AG.binary().is_file():
        problems.append(f"Antigravity CLI not found at {AG.binary()}; install it, sign in once with an "
                        "interactive `agy` session, then run `hydroagent agy-smoke`")
    for tid in a.task:
        if not (config.TASKS / tid / a.variant / "task_folder" / "data" / "donors.npz").is_file():
            problems.append(f"task folder for {tid} missing; run `hydroagent prepare`")
    print(f"executor={h2c.SOLVER_MODEL_OC} improver={h2c.IMPROVER_MODEL} runs={config.RUNS}")
    for p in problems:
        print("PROBLEM:", p)
    print("ready" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def cmd_run(a) -> int:
    from .h2adapter import configure, patch_claude
    configure(a.executor, a.improver)
    if a.offline:
        os.environ["HYDROAGENT_RUNS"] = str(config.RUNS / "offline")
        config.RUNS = config.RUNS / "offline"
    patch_claude()
    from harness2.core.proposer import Proposer
    from harness2.core.schedule import mode_token, run_domain

    from .h2adapter import HydroAdapter
    if not a.offline and cmd_check(a) != 0:
        return 1
    adapter = HydroAdapter(a.task, timeout=a.timeout, fake_executor=a.offline, variant=a.variant)
    if a.offline:
        from .offline import ScriptedProposer
        prop = ScriptedProposer(adapter.spec.channels["proposer"])
    else:
        prop = Proposer(adapter.spec.channels["proposer"], audit_dir=config.RUNS / "_audit")
    modes = ("agg", "seq") if a.mode == "both" else (mode_token(a.mode),)
    t0 = time.time()
    res = run_domain("camels-us", a.task, a.k, adapter=adapter, prop=prop, runs_root=config.RUNS / "adapt",
                     library_root=adapter.library_dir(a.k), modes=modes)
    res["wall_s"] = round(time.time() - t0)
    (config.RUNS / "last_run.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("done", "failed", "wall_s") if k in res}))
    fell = _fallbacks()
    if fell:
        print("WARNING: these Harness2 stages fell back to the base harness (improver failed): "
              + ", ".join(fell) + ". Their rollouts are base-harness replicates; see runs/_audit/*.meta.json")
    return int(res.get("failed", 0) > 0 or bool(fell))


def _fallbacks() -> list:
    out = []
    for s in (config.RUNS / "adapt").rglob("summary.json"):
        js = json.loads(s.read_text())
        for mode, rec in (js.get("modes") or {}).items():
            out += [f"{js['task_id']}:{mode}:{st['label']}" for st in rec.get("stages", []) if st.get("fell_back")]
    return out


def cmd_evaluate(a) -> int:
    from .h2adapter import configure
    configure()
    runs = config.RUNS / "offline" if a.offline else config.RUNS
    config.RUNS = runs
    from .h2adapter import HydroAdapter
    adapter = HydroAdapter(a.task, variant=a.variant)
    rows, per = [], []
    for tid in a.task:
        chosen = {}
        for s in (runs / "adapt").rglob(f"{tid}/summary.json"):
            js = json.loads(s.read_text())
            for mode, sel in (js.get("selections") or {}).items():
                if sel.get("chosen_run_id"):
                    chosen[sel["chosen_run_id"]] = mode
        for rj in sorted((runs / "rollouts").rglob("rollout.json")):
            meta = json.loads(rj.read_text())
            if meta.get("task_id") != tid or meta["run_id"].split("/")[0].startswith("camels-calib") != (a.variant == "calib"):
                continue
            got = adapter.score(rj.parent, tid) if adapter.delivered(rj.parent) else None
            label = meta["run_id"].rsplit("/", 1)[-1]
            row = {"task": tid, "run_id": meta["run_id"], "label": label,
                   "role": "base" if label.startswith("base") else ("selected-" + chosen[meta["run_id"]]
                                                                    if meta["run_id"] in chosen else "probe"),
                   "executor": meta.get("executor"), "delivered": got is not None,
                   "model_structure_changed": meta.get("model_structure_changed"),
                   "network_flagged": (meta.get("audit") or adapter._audit(rj.parent)).get("flagged")}
            if got:
                row.update(nse_median=got["nse_median"], kge_median=got["kge_median"],
                           beats_knn_attr=f"{got['n_passed']}/{got['n_criteria']}")
                per += [dict(b, task=tid, run_id=meta["run_id"], role=row["role"]) for b in got["per_basin"]]
            rows.append(row)
    df = pd.DataFrame(rows)
    fell = _fallbacks()
    if fell:
        df["note"] = ["base-harness replicate (improver fell back)" if any(f.endswith(":" + r["label"]) for f in fell)
                      else "" for r in rows]
        print("WARNING: stages that fell back to the base harness:", ", ".join(fell))
    out = config.RESULTS / ("offline" if a.offline else "") / ("" if a.variant == "template" else a.variant)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "agent_rollouts.csv", index=False, float_format="%.4f")
    pd.DataFrame(per).to_csv(out / "agent_per_basin.csv", index=False, float_format="%.4f")
    for tid in a.task:
        bf = config.RESULTS / f"baselines_summary_{tid}.csv"
        if bf.is_file():
            b = pd.read_csv(bf, index_col=0)
            print(f"\n== {tid}: classical baselines (median test NSE / KGE) ==")
            print(b[["nse_median", "kge_median"]].round(3).to_string())
        df_ = config.RESULTS / "dpl" / f"{tid}_test.csv"
        if df_.is_file():
            g = pd.read_csv(df_)
            print(f"== {tid}: differentiable parameter learning: NSE {g.nse.median():.3f} / KGE {g.kge.median():.3f}")
    print("\n== agent rollouts ==")
    print(df.drop(columns=["run_id"]).to_string(index=False) if len(df) else "(none)")
    return 0


METHOD_NAMES = {
    "gauged_upper": "Gauged upper bound (own calibration)",
    "knn_attr": "Attribute similarity, 5 donors",
    "knn_spatial": "Spatial proximity, 5 donors",
    "nn_spatial": "Nearest donor",
    "global_median": "Median donor parameters",
    "agent_base": "Claude agent, base harness",
    "agent_h2": "Claude agent + Harness\u00b2",
}


def _stats(v: pd.Series) -> dict:
    v = v.dropna()
    if not len(v):
        return {"n": 0}
    return {"nse_median": round(float(v.median()), 4), "q25": round(float(v.quantile(.25)), 4),
            "q75": round(float(v.quantile(.75)), 4), "frac_gt0": round(float((v > 0).mean()), 4), "n": int(len(v)),
            "ecdf": [round(float(x), 3) for x in sorted(v.clip(-1, 1))]}


def cmd_report(a) -> int:
    from datetime import datetime, timezone

    import numpy as np

    per = pd.read_csv(config.RESULTS / "baselines_per_basin.csv", dtype={"gauge_id": str})
    task = a.task[0]
    targets = json.loads((config.HIDDEN / task / "task.json").read_text())["targets"]
    agent_f = config.RESULTS / "agent_per_basin.csv"
    agent = pd.read_csv(agent_f, dtype={"gauge_id": str}) if agent_f.is_file() else pd.DataFrame()
    out = {"meta": {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
                    "train": config.TRAIN, "test": config.TEST, "folds": config.N_FOLDS, "pilot_task": task,
                    "n_basins": int(per.gauge_id.nunique()), "n_pilot": len(targets),
                    "executor": None, "improver": None},
           "methods": [{"id": k, "name": v} for k, v in METHOD_NAMES.items()],
           "all": {}, "pilot": {}, "by_region": {}, "rollouts": []}
    for m, g in per.groupby("method"):
        g = g.set_index("gauge_id")
        out["all"][m] = _stats(g.nse) | {"kge_median": round(float(g.kge.median()), 4)}
        gp = g.reindex(targets)
        out["pilot"][m] = _stats(gp.nse) | {"kge_median": round(float(gp.kge.median()), 4)}
    for reg, g in per.groupby("region"):
        out["by_region"][reg] = {m: round(float(gm.nse.median()), 3) for m, gm in g.groupby("method")} | \
                                {"n": int(g.gauge_id.nunique())}
    if len(agent):
        for role, key in (("base", "agent_base"), ("selected-agg", "agent_h2"), ("selected-seq", "agent_h2")):
            g = agent[(agent.task == task) & (agent.role == role)]
            if len(g):
                g = g.groupby("gauge_id")[["nse", "kge"]].mean()
                out["pilot"][key] = _stats(g.nse) | {"kge_median": round(float(g.kge.median()), 4)}
        roll = pd.read_csv(config.RESULTS / "agent_rollouts.csv").assign(run="pilot 1")
        rep = config.RESULTS / "pilot0_base_replicates" / "agent_rollouts.csv"
        if rep.is_file():
            r0 = pd.read_csv(rep).assign(run="pilot 0 (all base harness)", role="base replicate")
            roll = pd.concat([roll, r0], ignore_index=True)
            meds = [x for x in roll[roll.role.isin(["base", "base replicate"])].nse_median.dropna()]
            out["meta"]["base_replicates"] = {"n": len(meds), "nse_median_mean": round(float(np.mean(meds)), 4),
                                              "nse_median_sd": round(float(np.std(meds, ddof=1)), 4),
                                              "nse_median_min": round(float(min(meds)), 4),
                                              "nse_median_max": round(float(max(meds)), 4)}
        out["rollouts"] = roll.drop(columns=["run_id"]).where(roll.notna(), None).to_dict(orient="records")
        last = config.RUNS / "last_run.json"
        if roll.executor.notna().any():
            out["meta"]["executor"] = str(roll.executor.dropna().iloc[0])
    out["calib"] = _calib_report()
    f = config.REPO / "docs" / "data" / "hydroagent.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {f}")
    return 0


CALIB_NAMES = {
    "gauged_upper": "Gauged upper bound (own calibration)",
    "knn_attr_released": "Released per-basin calibrations + attribute transfer (reference)",
    "dpl_ens3": "Differentiable parameter learning, 3-seed ensemble (baseline)",
    "dpl": "Differentiable parameter learning, single seed",
    "agent_base": "Claude agent, base harness",
    "agent_inv1": "Harness\u00b2 candidate: general edits",
    "agent_dom1": "Harness\u00b2 candidate: domain edits",
    "agent_agg1": "Claude agent + Harness\u00b2 (selected)",
}


def _calib_report(task: str = "region-Southeast") -> dict:
    """Calibration framing: agent vs differentiable parameter learning on held-out basins."""
    import numpy as np
    from scipy.stats import wilcoxon

    from . import baselines, camels
    rd = config.RESULTS / "dpl"
    out = {"task": task, "methods": [{"id": k, "name": v} for k, v in CALIB_NAMES.items()], "stats": {}, "splits": []}
    f0 = rd / f"{task}_test.csv"
    if not f0.is_file():
        return out
    dpl = pd.read_csv(f0, dtype={"gauge_id": str}).set_index("gauge_id")
    ids = list(dpl.index)
    seeds = [dpl] + [pd.read_csv(p, dtype={"gauge_id": str}).set_index("gauge_id").reindex(ids)
                     for p in sorted(rd.glob(f"{task}_seed*_test.csv"))]
    cl = pd.read_csv(config.RESULTS / f"baselines_per_basin_{task}.csv", dtype={"gauge_id": str})
    series = {"gauged_upper": cl[cl.method == "gauged_upper"].set_index("gauge_id").reindex(ids),
              "knn_attr_released": cl[cl.method == "knn_attr"].set_index("gauge_id").reindex(ids),
              "dpl": dpl}
    ag = config.RESULTS / "calib" / "agent_per_basin.csv"
    if ag.is_file():
        a = pd.read_csv(ag, dtype={"gauge_id": str})
        a = a[(a.task == task) & a.run_id.str.startswith("camels-calib-oc-agg1/")]     # Opus k=1 run
        a["label"] = a.run_id.str.rsplit("/", n=1).str[-1]
        for lab, g in a.groupby("label"):
            series["agent_base" if lab == "base1" else f"agent_{lab}"] = g.set_index("gauge_id").reindex(ids)
    ens = rd / f"{task}_ens3_test.csv"
    ref = dpl
    if ens.is_file():                        # the stronger baseline: discharge averaged over 3 seeds
        ref = pd.read_csv(ens, dtype={"gauge_id": str}).set_index("gauge_id").reindex(ids)
        series["dpl_ens3"] = ref
    out["reference"] = "dpl_ens3" if ens.is_file() else "dpl"
    for k, g in series.items():
        st = _stats(g.nse) | {"kge_median": round(float(g.kge.median()), 4)}
        if k != out["reference"]:
            st["wins_vs_dpl"] = int((g.nse > ref.nse).sum())
            st["p_nse_vs_dpl"] = float(wilcoxon(g.nse, ref.nse, nan_policy="omit").pvalue)
            st["p_kge_vs_dpl"] = float(wilcoxon(g.kge, ref.kge, nan_policy="omit").pvalue)
        if k.startswith("agent_") and k != "agent_base" and "agent_base" in series:
            st["p_nse_vs_base"] = float(wilcoxon(g.nse, series["agent_base"].nse, nan_policy="omit").pvalue)
        out["stats"][k] = st
    out["dpl_seeds"] = [round(float(x.nse.median()), 4) for x in seeds]
    out["k10"] = _k10_report(task, ids, ref)
    try:
        out["exp3"] = _exp3_report(task, ids, ref)
    except Exception as e:          # partial data while experiment 3 runs
        out["exp3"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        out["exp4"] = _exp4_report(task, ids, ref, out.get("exp3"))
    except Exception as e:          # partial data while experiment 4 runs
        out["exp4"] = {"error": f"{type(e).__name__}: {e}"}
    out["n_train"] = len(json.loads((config.HIDDEN / task / "task.json").read_text())["donors"])
    # baseline across every split
    d = None
    for p in sorted(rd.glob("*_test.csv")):
        name = p.name[:-len("_test.csv")]
        if "_seed" in name or "_ens" in name:
            continue
        g = pd.read_csv(p, dtype={"gauge_id": str})
        row = {"split": name, "kind": g["kind"].iloc[0], "n": len(g),
               "dpl_nse": round(float(g.nse.median()), 4), "dpl_kge": round(float(g.kge.median()), 4)}
        cf = config.RESULTS / (f"baselines_per_basin_{name}.csv" if name.startswith("region-") else "baselines_per_basin.csv")
        if name.startswith("region-") and not cf.is_file():
            if d is None:
                d = camels.load()
                attrs = camels.attributes(d["ids"])
                regions = load_regions(d["ids"])
                from .protocol import make_tasks
                tasks = make_tasks(attrs, regions)
            t = tasks[name]
            res = baselines.run(d, attrs, list(t.targets), list(t.donors))
            pd.concat([r.assign(method=m) for m, r in res.items()]).to_csv(cf, float_format="%.4f")
        c2 = pd.read_csv(cf, dtype={"gauge_id": str})
        c2 = c2[c2.gauge_id.isin(set(g.gauge_id))]
        for m in ("knn_attr", "gauged_upper"):
            row[f"{m}_nse"] = round(float(c2[c2.method == m].nse.median()), 4)
        out["splits"].append(row)
    return out


def _k10_report(task: str, ids, ref) -> dict:
    """Sequential k=10 run (Claude Sonnet 5.5): base-harness replicates and skill by round."""
    import numpy as np
    from scipy.stats import wilcoxon
    ag = config.RESULTS / "calib" / "agent_per_basin.csv"
    if not ag.is_file():
        return {}
    a = pd.read_csv(ag, dtype={"gauge_id": str})
    a = a[(a.task == task) & a.run_id.str.contains(r"-(?:agg|seq)10/")]
    if a.empty:
        return {}
    a["label"] = a.run_id.str.rsplit("/", n=1).str[-1]
    out = {"model": "anthropic/claude-sonnet-5-5", "base": [], "rounds": []}
    for lab, g in a.groupby("label"):
        g = g.set_index("gauge_id").reindex(ids)
        row = {"label": lab, "nse": round(float(g.nse.median()), 4), "kge": round(float(g.kge.median()), 4),
               "p_vs_dpl": float(wilcoxon(g.nse, ref.nse, nan_policy="omit").pvalue)}
        if lab.startswith("base"):
            out["base"].append(row)
        else:
            row["round"] = int("".join(ch for ch in lab if ch.isdigit()))
            row["kind"] = {"c": "composed", "i": "general", "d": "domain"}[lab[0]]
            out["rounds"].append(row)
    b = [r["nse"] for r in out["base"]]
    out["base_mean"], out["base_sd"] = round(float(np.mean(b)), 4), round(float(np.std(b, ddof=1)), 4)
    out["delta"] = round(max(0.005, out["base_sd"]), 4)
    es = Path.home() / ".cache" / "hydroagent" / "k10_rounds.json"
    if es.is_file():
        out["stopped"] = json.loads(es.read_text()).get("stopped")
    out["rounds"].sort(key=lambda r: (r["round"], r["kind"]))
    return out


def _harness_groups(task: str, ids, ref, groups: dict, base_label: str) -> dict:
    """Score every run of every harness, per group; run-level tests against the base harness.
    Rollouts the audit flagged (network attempts, web tools, tampering) are left out and counted."""
    import numpy as np
    from scipy.stats import mannwhitneyu, wilcoxon

    from .h2adapter import HydroAdapter, configure
    configure()
    a = HydroAdapter([task], variant="calib")
    out = {"harnesses": [], "groups": {}, "n_flagged_excluded": 0}
    per = {}
    for gname, items in groups.items():
        for label, dirs in items:
            runs = []
            for d in dirs:
                if not ((d / "rollout.json").is_file() and a.delivered(d)):
                    continue
                meta = json.loads((d / "rollout.json").read_text())
                if (meta.get("audit") or {}).get("flagged"):
                    out["n_flagged_excluded"] += 1
                    continue
                if ((meta.get("staging") or {}).get("run") or {}).get("ended_on_denial"):
                    out["n_invalid_excluded"] = out.get("n_invalid_excluded", 0) + 1
                    continue
                sc = a.score(d, task)
                runs.append(pd.DataFrame(sc["per_basin"]).set_index("gauge_id").reindex(ids))
            if not runs:
                continue
            meds = [float(r.nse.median()) for r in runs]
            avg = pd.concat([r.nse for r in runs], axis=1).mean(axis=1)
            per[label] = avg
            out["harnesses"].append({"group": gname, "label": label, "n_runs": len(runs),
                                     "run_nse": [round(m, 4) for m in meds],
                                     "mean_run_nse": round(float(np.mean(meds)), 4),
                                     "sd_run_nse": round(float(np.std(meds, ddof=1)), 4) if len(meds) > 1 else None,
                                     "avg_basin_nse_median": round(float(avg.median()), 4),
                                     "p_vs_dpl": float(wilcoxon(avg, ref.nse, nan_policy="omit").pvalue)})
    base = next((h for h in out["harnesses"] if h["label"] == base_label), None)
    if base is not None:
        for h in out["harnesses"]:
            if h["label"] != base_label:
                # basin-paired test on run-averaged NSE (ignores run-to-run noise, so optimistic)
                h["p_vs_base"] = float(wilcoxon(per[h["label"]], per[base_label], nan_policy="omit").pvalue)
                # run-level test: this harness's run medians vs the base harness's run medians
                h["p_runs_vs_base"] = float(mannwhitneyu(h["run_nse"], base["run_nse"]).pvalue)
    refmed = float(ref.nse.median())
    for g in groups:
        hs = [h for h in out["harnesses"] if h["group"] == g]
        if hs:
            runs = [x for h in hs for x in h["run_nse"]]
            out["groups"][g] = {"n_harnesses": len(hs), "mean_of_means": round(float(np.mean([h["mean_run_nse"] for h in hs])), 4),
                                "n_runs": len(runs), "n_runs_above_reference": int(sum(r > refmed for r in runs))}
            if base is not None and base["group"] != g:
                out["groups"][g]["p_runs_vs_base"] = float(mannwhitneyu(runs, base["run_nse"]).pvalue)
    return out


def _selected(ns: str, task: str, prefix: str):
    sel = config.RUNS / "adapt" / ns / task / "summary.json"
    if sel.is_file():
        js = json.loads(sel.read_text())
        return prefix + str((js.get("selections") or {}).get("agg", {}).get("chosen_label"))
    return None


H9 = ("inv1", "inv2", "inv3", "dom1", "dom2", "dom3", "agg1", "agg2", "agg3")


def _exp3_report(task: str, ids, ref) -> dict:
    """Experiment 3: Opus editor vs Sonnet editor (Sonnet agent), parallel k=3, 3 runs per harness."""
    from scipy.stats import mannwhitneyu
    R = config.RUNS / "rollouts"
    groups = {
        "base": [("base", sorted(R.glob(f"camels-calib-oc-agg10/{task}/base*")) + sorted(R.glob(f"camels-calib-oc-agg3/{task}/base*")))],
        "sonnet_editor": [(f"sonnet-{h}", [R / f"camels-calib-oc-seq10/{task}/{h}"] + sorted(R.glob(f"replicates/{task}/sonnetimp-{h}/r*")))
                          for h in ("inv1", "dom1", "c1")],
        "opus_editor": [(f"opus-{h}", [R / f"camels-calib-oc-agg3/{task}/{h}"] + sorted(R.glob(f"replicates/{task}/opusimp-{h}/r*")))
                        for h in H9],
    }
    out = _harness_groups(task, ids, ref, groups, "base")
    so = [h["mean_run_nse"] for h in out["harnesses"] if h["group"] == "sonnet_editor"]
    op = [h["mean_run_nse"] for h in out["harnesses"] if h["group"] == "opus_editor"]
    if len(so) >= 2 and len(op) >= 2:
        out["p_opus_vs_sonnet_editor"] = float(mannwhitneyu(op, so).pvalue)
    sel = _selected("camels-calib-oc-agg3", task, "opus-")
    if sel:
        out["selected"] = sel
    return out


def _exp4_report(task: str, ids, ref, exp3: Optional[dict] = None, slug: str = "gemini-4-argon-high") -> dict:
    """Experiment 4: Gemini 4 Argon through the Antigravity CLI as the agent, Opus 5.5 as the
    Harness² editor (parallel k=3, 3 runs per harness), against experiment 3's Sonnet agent."""
    from scipy.stats import mannwhitneyu
    R = config.RUNS / "rollouts"
    ns, rep = f"camels-calib-ag-{slug}-agg3", f"replicates-ag-{slug}"
    groups = {
        "argon_base": [("argon-base", sorted(R.glob(f"{ns}/{task}/base*")) + sorted(R.glob(f"{rep}/{task}/base/r*")))],
        "argon_opus_editor": [(f"argon-{h}", [R / f"{ns}/{task}/{h}"] + sorted(R.glob(f"{rep}/{task}/argon-{h}/r*")))
                              for h in H9],
    }
    out = _harness_groups(task, ids, ref, groups, "argon-base")
    out["agent"] = f"agy/{slug}"
    note = config.RESULTS / "calib" / "exp4_note.txt"
    if note.is_file():
        out["note"] = note.read_text().strip()
    sel = _selected(ns, task, "argon-")
    if sel:
        out["selected"] = sel
    if exp3 and exp3.get("harnesses"):          # the same comparison with the Sonnet agent
        runs = lambda E, g: [x for h in E["harnesses"] if h["group"] == g for x in h["run_nse"]]  # noqa: E731
        cmp = {}
        for mine, theirs, key in (("argon_base", "base", "base"), ("argon_opus_editor", "opus_editor", "opus_edited")):
            a_, b_ = runs(out, mine), runs(exp3, theirs)
            if len(a_) >= 2 and len(b_) >= 2:
                cmp[key] = {"argon_mean": round(sum(a_) / len(a_), 4), "sonnet_mean": round(sum(b_) / len(b_), 4),
                            "p_runs": float(mannwhitneyu(a_, b_).pvalue)}
        out["vs_sonnet_agent"] = cmp
    return out


def cmd_agy_smoke(a) -> int:
    """One short Antigravity session that checks what a Harness² rollout relies on: the CLI is
    signed in and runs headless in a workspace below the trusted folder, AGENTS.md rules load in
    full, workspace skills are found, the venv's python3 is on PATH, downloads are blocked by
    the deny rules and by the fence, and the stream parser and audit understand the events."""
    import re
    import secrets
    import subprocess

    from . import antigravity as AG
    from .h2adapter import NETWORK_PATTERNS
    b = AG.binary()
    if not b.is_file():
        print(f"Antigravity CLI not found at {b}. Install it and sign in once with `agy`.")
        return 1
    v = subprocess.run([str(b), "--version"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    print("agy", (v.stdout or v.stderr).strip())
    out = config.RUNS / "_agy_smoke" / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True)
    venv = str(Path(sys.executable).parent)
    try:
        ms = subprocess.run([str(b), "models"], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                            timeout=120, env=AG.env_for(out, AG.ws_root(), venv_bin=venv, proxy=None))
        listing = (ms.stdout + ms.stderr).strip()
    except subprocess.TimeoutExpired:
        listing = "(agy models timed out)"
    finally:
        shutil.rmtree(AG.ws_root() / ".agyhome", ignore_errors=True)
    (out / "models.txt").write_text(listing)
    print("models:\n" + listing[:2000])
    ws = AG.ws_root() / "smoke"
    shutil.rmtree(ws, ignore_errors=True)
    (ws / ".agents" / "skills" / "codeword").mkdir(parents=True)
    w1, w2, w3 = (secrets.token_hex(3).upper() for _ in range(3))
    filler = "\n".join(f"- Rule {i}: keep calibration logs in output/logs and name them by date." for i in range(200))
    (ws / "AGENTS.md").write_text(f"# Project rules\n\nStart code word: ALPHA-{w1}.\n\n{filler}\n\n"
                                  f"End code word: OMEGA-{w2}.\n")
    (ws / ".agents" / "skills" / "codeword" / "SKILL.md").write_text(
        "---\nname: codeword\ndescription: Use when asked for the skill code word.\n---\n\n"
        f"The skill code word is SKILL-{w3}.\n")
    prompt = ("Answer with one JSON object with keys a to h, one short string each. "
              "a: without using any tool, give the two code words stated in your project rules "
              "(they start with ALPHA- and OMEGA-), or say which you cannot see. "
              "b: use your skill to find the skill code word. "
              "c: run the shell command `python3 -c \"import sys, torch; print(sys.executable)\"` and give its output. "
              "d: run `curl -sS -m 10 https://waterservices.usgs.gov/` and say what happened. "
              "e: run `python3 -c \"import urllib.request as u; print(u.urlopen('https://zenodo.org', timeout=10).status)\"` "
              "and then `python3 -c \"import urllib.request as u; print(u.urlopen('https://www.googleapis.com/storage/v1/b/x/o', timeout=10).status)\"`, "
              "and give each output or error's last line. "
              "f: try to read https://example.com with your URL-reading tool and say what happened. "
              "g: search the web for 'USGS 02177000 streamflow' with your web search tool if you have one, "
              "and say whether the search ran. "
              "h: create the file notes/h.txt containing the word ok with your file-writing tool, not the shell, "
              "and say whether it worked.")
    model = a.executor or ""
    model = model if AG.is_agy(model) else "agy/" + (a.model or "gemini-4-argon-high")
    rc = AG.launch(ws, out, prompt, model, 600, venv_bin=venv, network_fence=True)
    ev = out / "events.jsonl"
    print(f"exit code {rc}; events: {ev}")
    print("stderr tail:\n" + (out / "stderr.log").read_text(errors="replace")[-1500:])
    print("trajectory:\n" + AG.render_events(ev, obs_trunc=600, max_chars=8000))
    aud = AG.audit(ev, NETWORK_PATTERNS)
    proxy = json.loads((out / "proxy.json").read_text()) if (out / "proxy.json").is_file() else {}
    print("proxy hosts:", json.dumps(proxy))
    print("audit:", json.dumps(aud))
    txt = AG.result_text(ev)
    web_ran = [AG.tool_call(st)[0] for st in AG.steps(ev)
               if AG.WEB_TOOL.match(AG.tool_call(st)[0] or "-") and not AG.tool_call(st)[3]
               and str(st.get("state", "")).upper() == "DONE"]
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        ans = json.loads(m.group(0)) if m else {}
    except ValueError:
        ans = {}
    checks = [
        ("ran headless and signed in", rc == 0 and bool(AG.steps(ev)), "critical"),
        ("rules start loaded", f"ALPHA-{w1}" in str(ans.get("a", "")), "critical"),
        (f"rules past {AG.RULES_MAX} chars loaded", f"OMEGA-{w2}" in str(ans.get("a", "")), "warn"),
        ("workspace skill found", f"SKILL-{w3}" in str(ans.get("b", "")), "critical"),
        ("venv python3 with torch on PATH", venv in str(ans.get("c", "")), "critical"),
        ("python download went through the fence and was refused", "zenodo.org" in proxy.get("denied", {}), "critical"),
        ("fence opens data-serving hosts to agy alone",
         not [h for h in proxy.get("allowed", {}) if AG.LEAK_HOSTS.search(h.split(" ")[0]) and not h.endswith("(agy)")],
         "critical"),
        ("agent's python refused on an agy-only host",
         any(h.startswith("www.googleapis.com (from") for h in proxy.get("denied", {})), "critical"),
        ("parser sees tool steps", aud["tool_steps"] > 0, "critical"),
        ("audit sees the curl command", "curl" in aud["network_pattern_hits"], "critical"),
        ("audit flags the rollout", aud["flagged"] is True, "critical"),
        ("web search blocked by the hook", not web_ran and bool(aud["web_tools_refused"]), "critical"),
        ("no-web hook left intact", not aud["hooks_tampered"], "critical"),
        ("file tool may write in the workspace", (ws / "notes" / "h.txt").is_file()
         and not AG.result(ev).get("denied_actions"), "critical"),
    ]
    print(f"answers: {json.dumps(ans)}")
    print(f"expected code words: ALPHA-{w1} OMEGA-{w2} SKILL-{w3}")
    bad = 0
    for name, ok, level in checks:
        print(f"  {'PASS' if ok else ('FAIL' if level == 'critical' else 'WARN')}  {name}")
        bad += (not ok) and level == "critical"
    denied_google = [h for h in proxy.get("denied", {}) if AG.GOOGLE_HOSTS.search(h) and not AG.LEAK_HOSTS.search(h)]
    if denied_google:
        print("Google hosts refused by the fence. If agy failed, add the ones it needs with HYDRO_AGY_ALLOW_EXTRA; "
              "if it worked, list them in HYDRO_AGY_BENIGN_HOSTS so the audit ignores them: " + ", ".join(denied_google))
    held = [h for h in proxy.get("denied", {}) if AG.LEAK_HOSTS.search(h)]
    if held:
        print("Data-serving hosts refused by design (never add these): " + ", ".join(held))
    print(f"trusted workspace folder for agy: {AG.ws_root()}")
    return 1 if bad else 0

def main(argv=None) -> int:
    _load_env_file()
    ap = argparse.ArgumentParser(prog="hydroagent", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    p = sub.add_parser("agy-smoke", help="check the Antigravity CLI executor (sign-in, rules, skills, fence)")
    p.add_argument("--executor", help="agy/<model slug>")
    p.add_argument("--model", help="model slug, e.g. gemini-4-argon-high (see `agy models`)")
    p.set_defaults(fn=cmd_agy_smoke)
    p = sub.add_parser("dpl", help="differentiable parameter-learning baseline")
    p.add_argument("--splits", nargs="+", default=["all"])
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_dpl)
    p = sub.add_parser("prepare")
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--variant", choices=("template", "calib"), default="template")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_prepare)
    for name, fn in (("check", cmd_check), ("run", cmd_run), ("evaluate", cmd_evaluate), ("report", cmd_report),
                     ("replicate", cmd_replicate)):
        p = sub.add_parser(name)
        p.add_argument("--task", nargs="+", default=[f"fold1-pilot{config.PILOT_TARGETS}"])
        p.add_argument("--executor")
        p.add_argument("--improver")
        p.add_argument("--k", type=int, default=1)
        p.add_argument("--mode", choices=("parallel", "sequential", "both"), default="parallel")
        p.add_argument("--timeout", type=int, default=5400, help="seconds per executor rollout")
        p.add_argument("--offline", action="store_true")
        p.add_argument("--variant", choices=("template", "calib"), default="template",
                       help="template: donors with released parameters (pilot); calib: agent calibrates")
        if name == "replicate":
            p.add_argument("--harness", nargs="+", required=True, help="label=/path/to/harness (or label=base)")
            p.add_argument("--n", type=int, default=2)
            p.add_argument("--start", type=int, default=2)
            p.add_argument("--workers", type=int, default=1)
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
