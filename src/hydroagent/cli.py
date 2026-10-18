"""hydroagent command line.

    hydroagent setup                       pinned OpenCode executor (via harness2's installer)
    hydroagent prepare                     CAMELS record, attributes, folds, baselines, task folders
    hydroagent check                       credentials, OpenCode, task folders (no model call)
    hydroagent run --task fold1-pilot30 --k 1 --mode parallel [--executor M] [--improver M]
    hydroagent evaluate --task fold1-pilot30   score every rollout against USGS; compare with baselines
    hydroagent report                      write docs/data/hydroagent.json for the dashboard tab
    hydroagent run ... --offline           scripted executor and editor: end-to-end pipeline test, no API calls

Models use OpenCode's provider/model ids, e.g. anthropic/claude-sonnet-5-5. The
Anthropic key is read from ANTHROPIC_API_KEY, else from `.CLAUDE_API` (raw key) or `.env`
in the repository root; both files are git-ignored.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from . import config


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
    from nwp2sf.basins import load_basins

    from . import camels, protocol
    d = camels.load()
    attrs = camels.attributes(d["ids"])
    regions = load_basins(config.REPO / "data", include_excluded=True).region.reindex(d["ids"]).fillna("Other")
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
    print(f"OpenCode ready under {h2c.OPENCODE_DIR}")
    return 0


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
        workspace.build(tasks[tid], d, attrs, force=a.force)
        workspace.brief(tid)
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
    for tid in a.task:
        if not (config.TASKS / tid / "template" / "task_folder" / "data" / "donors.npz").is_file():
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
    adapter = HydroAdapter(a.task, timeout=a.timeout, fake_executor=a.offline)
    if a.offline:
        from .offline import ScriptedProposer
        prop = ScriptedProposer(adapter.spec.channels["proposer"])
    else:
        prop = Proposer(adapter.spec.channels["proposer"], audit_dir=config.RUNS / "_audit")
    modes = ("agg", "seq") if a.mode == "both" else (mode_token(a.mode),)
    t0 = time.time()
    res = run_domain("camels-us", a.task, a.k, adapter=adapter, prop=prop, runs_root=config.RUNS / "adapt",
                     library_root=config.RUNS / "library" / f"k{a.k}", modes=modes)
    res["wall_s"] = round(time.time() - t0)
    (config.RUNS / "last_run.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("done", "failed", "wall_s") if k in res}))
    return int(res.get("failed", 0) > 0)


def cmd_evaluate(a) -> int:
    from .h2adapter import configure
    configure()
    runs = config.RUNS / "offline" if a.offline else config.RUNS
    config.RUNS = runs
    from .h2adapter import HydroAdapter
    adapter = HydroAdapter(a.task)
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
            if meta.get("task_id") != tid:
                continue
            got = adapter.score(rj.parent, tid) if adapter.delivered(rj.parent) else None
            label = meta["run_id"].rsplit("/", 1)[-1]
            row = {"task": tid, "run_id": meta["run_id"], "label": label,
                   "role": "base" if label.startswith("base") else ("selected-" + chosen[meta["run_id"]]
                                                                    if meta["run_id"] in chosen else "probe"),
                   "executor": meta.get("executor"), "delivered": got is not None,
                   "model_structure_changed": meta.get("model_structure_changed"),
                   "network_flagged": (meta.get("audit") or {}).get("flagged")}
            if got:
                row.update(nse_median=got["nse_median"], kge_median=got["kge_median"],
                           beats_knn_attr=f"{got['n_passed']}/{got['n_criteria']}")
                per += [dict(b, task=tid, run_id=meta["run_id"], role=row["role"]) for b in got["per_basin"]]
            rows.append(row)
    df = pd.DataFrame(rows)
    out = config.RESULTS / ("offline" if a.offline else "")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "agent_rollouts.csv", index=False, float_format="%.4f")
    pd.DataFrame(per).to_csv(out / "agent_per_basin.csv", index=False, float_format="%.4f")
    for tid in a.task:
        b = pd.read_csv(config.RESULTS / f"baselines_summary_{tid}.csv", index_col=0)
        print(f"\n== {tid}: baselines (median test NSE / KGE) ==")
        print(b[["nse_median", "kge_median"]].round(3).to_string())
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
                g = g.groupby("gauge_id")[["nse", "kge"]].mean()    # average over repeated base rollouts
                out["pilot"][key] = _stats(g.nse) | {"kge_median": round(float(g.kge.median()), 4)}
        roll = pd.read_csv(config.RESULTS / "agent_rollouts.csv")
        out["rollouts"] = roll.drop(columns=["run_id"]).to_dict(orient="records")
        last = config.RUNS / "last_run.json"
        if roll.executor.notna().any():
            out["meta"]["executor"] = str(roll.executor.dropna().iloc[0])
    f = config.REPO / "docs" / "data" / "hydroagent.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {f}")
    return 0


def main(argv=None) -> int:
    _load_env_file()
    ap = argparse.ArgumentParser(prog="hydroagent", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    p = sub.add_parser("prepare")
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_prepare)
    for name, fn in (("check", cmd_check), ("run", cmd_run), ("evaluate", cmd_evaluate), ("report", cmd_report)):
        p = sub.add_parser(name)
        p.add_argument("--task", nargs="+", default=[f"fold1-pilot{config.PILOT_TARGETS}"])
        p.add_argument("--executor")
        p.add_argument("--improver")
        p.add_argument("--k", type=int, default=1)
        p.add_argument("--mode", choices=("parallel", "sequential", "both"), default="parallel")
        p.add_argument("--timeout", type=int, default=5400, help="seconds per executor rollout")
        p.add_argument("--offline", action="store_true")
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
