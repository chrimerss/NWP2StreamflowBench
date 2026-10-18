"""dCREST toolkit for the workspace: inspect data, simulate, evaluate on donors, calibrate.

Run from the workspace root, e.g. `python3 scripts/hydrokit.py info`. Every command
reads task_folder/data/ and imports the editable model package in ./dcrest.

    info                                   data inventory, periods, parameter bounds
    simulate  --set donors|targets --params P.csv --out Q.csv [--start --end --spinup-from]
              (targets default to the prediction window with a 3-year spin-up)
    evaluate  --params P.csv [--ids D0001,D0002] [--out per_basin.csv]
    calibrate --ids D0001,... --out P.csv [--iters 400 --loss nse|kge --init released|default]

Parameter CSVs have an `id` column plus the 16 dCREST parameters; `area_km2` is taken
from the data when absent. Discharge is mm/day.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("HYDRO_WORKSPACE", Path.cwd()))
sys.path.insert(0, str(ROOT))
import dcrest  # noqa: E402  (the workspace copy)
import torch  # noqa: E402

DATA = ROOT / "task_folder" / "data"
META = json.loads((DATA / "meta.json").read_text())
PKEYS = META["param_names"]


def load(which: str) -> dict:
    z = np.load(DATA / f"{which}.npz", allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["ids"] = [str(i) for i in d["ids"]]
    d["dates"] = pd.DatetimeIndex(d["dates"])
    return d


def window(dates, start, end):
    return slice(int(dates.searchsorted(pd.Timestamp(start))), int(dates.searchsorted(pd.Timestamp(end), side="right")))


def config():
    return dcrest.CrestConfig(dt_hours=24.0, uh_nstep=META["uh_nstep"], uh_kmin_h=META["uh_kmin_h"],
                              bounds={k: tuple(v) for k, v in META["bounds"].items()})


def params_from_csv(path: str, d: dict, ids: list) -> dict:
    p = pd.read_csv(path, dtype={"id": str}).set_index("id")
    missing = [i for i in ids if i not in p.index]
    if missing:
        raise SystemExit(f"{len(missing)} ids missing from {path}: {missing[:5]}")
    p = p.loc[ids]
    pos = {g: j for j, g in enumerate(d["ids"])}
    out = {k: torch.as_tensor(p[k].to_numpy(copy=True), dtype=torch.float32) for k in PKEYS}
    area = p["area_km2"].values if "area_km2" in p else np.asarray(d["area_km2"])[[pos[i] for i in ids]]
    out["area_km2"] = torch.as_tensor(area, dtype=torch.float32)
    return out


def simulate(d: dict, params: dict, ids: list, start=None, end=None, spinup_from=None):
    """Run from `spinup_from` (default: 3 years before `start`) to `end` from dcrest's default
    initial states, and return the days from `start` to `end`. The released parameters were
    calibrated with exactly this 3-year spin-up; much longer spin-ups change the state of
    slow aquifers and can degrade some basins badly."""
    pos = [d["ids"].index(i) for i in ids]
    start = pd.Timestamp(start or d["dates"][0])
    end = pd.Timestamp(end or d["dates"][-1])
    spin = pd.Timestamp(spinup_from) if spinup_from else max(d["dates"][0], start - pd.DateOffset(years=3))
    run = window(d["dates"], spin, end)
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a[run][:, pos]), dtype=torch.float32)  # noqa: E731
    with torch.no_grad():
        Q = dcrest.simulate(t(d["P"]), t(d["T"]), t(d["PET"]), params, config()).numpy()
    dates = d["dates"][run]
    keep = (dates >= start) & (dates <= end)
    return dates[keep], Q[keep]


def nse(sim, obs):
    out = []
    for j in range(sim.shape[1]):
        m = np.isfinite(obs[:, j]) & np.isfinite(sim[:, j])
        o, s = obs[m, j], sim[m, j]
        out.append(np.nan if len(o) < 365 or o.var() == 0 else 1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())
    return np.asarray(out)


def kge(sim, obs):
    out = []
    for j in range(sim.shape[1]):
        m = np.isfinite(obs[:, j]) & np.isfinite(sim[:, j])
        o, s = obs[m, j], sim[m, j]
        if len(o) < 365 or o.std() == 0 or s.std() == 0:
            out.append(np.nan)
            continue
        r = np.corrcoef(s, o)[0, 1]
        out.append(1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2))
    return np.asarray(out)


def cmd_info(a):
    for which in ("donors", "targets"):
        d = load(which)
        q = "Q_obs" in d
        print(f"{which}: {len(d['ids'])} basins, {d['dates'][0].date()}..{d['dates'][-1].date()}, arrays "
              f"{[k for k in d if k not in ('ids', 'dates')]}" + (" (Q_obs only inside the observation window)" if q else ""))
    print("observation window for donors:", META["obs_window"], " model spin-up from:", META["spinup_from"])
    print("predict window for targets:", META["predict_window"])
    print("parameters:", ", ".join(PKEYS))
    print("bounds:", json.dumps(META["bounds"]))


def cmd_simulate(a):
    d = load(a.set)
    ids = a.ids.split(",") if a.ids else d["ids"]
    if a.set == "targets" and not a.start:
        a.start, a.end = META["predict_window"]
    dates, Q = simulate(d, params_from_csv(a.params, d, ids), ids, a.start, a.end, a.spinup_from)
    pd.DataFrame(Q, index=pd.Index(dates, name="date"), columns=ids).to_csv(a.out, float_format="%.5f")
    print(f"wrote {a.out}: {Q.shape[0]} days x {Q.shape[1]} basins")


def cmd_evaluate(a):
    d = load("donors")
    ids = a.ids.split(",") if a.ids else d["ids"]
    s, e = META["obs_window"]
    dates, Q = simulate(d, params_from_csv(a.params, d, ids), ids, s, e, META["spinup_from"])
    pos = [d["ids"].index(i) for i in ids]
    obs = d["Q_obs"][window(d["dates"], s, e)][:, pos]
    df = pd.DataFrame({"nse": nse(Q, obs), "kge": kge(Q, obs)}, index=pd.Index(ids, name="id"))
    print(f"donor fit {s}..{e} over {len(ids)} basins: median NSE {df.nse.median():.3f}, median KGE {df.kge.median():.3f}")
    if a.out:
        df.to_csv(a.out, float_format="%.4f")


def cmd_calibrate(a):
    d = load("donors")
    ids = a.ids.split(",")
    pos = [d["ids"].index(i) for i in ids]
    s, e = META["obs_window"]
    run = slice(window(d["dates"], META["spinup_from"], e).start, window(d["dates"], s, e).stop)
    sc = window(d["dates"], s, e)
    k0 = sc.start - run.start
    t = lambda x: torch.as_tensor(np.ascontiguousarray(x[run][:, pos]), dtype=torch.float32)  # noqa: E731
    P, T, E = t(d["P"]), t(d["T"]), t(d["PET"])
    obs = torch.as_tensor(np.ascontiguousarray(d["Q_obs"][sc][:, pos]), dtype=torch.float32)
    mask = torch.isfinite(obs)
    bounds = {k: tuple(v) for k, v in META["bounds"].items()}
    init = None
    if a.init == "released":
        rel = pd.read_csv(DATA / "donor_params_released.csv", dtype={"id": str}).set_index("id").loc[ids]
        init = {k: torch.as_tensor(rel[k].to_numpy(copy=True), dtype=torch.float32) for k in PKEYS}
    model = dcrest.CrestLumped(n_units=len(ids), area_km2=np.asarray(d["area_km2"])[pos], config=config(), bounds=bounds)
    if init is not None:
        from dcrest import params as dp
        raw = dp.to_raw(init, bounds) if hasattr(dp, "to_raw") else None
        if raw is not None:
            with torch.no_grad():
                for k in PKEYS:
                    model.raw[k].copy_(raw[k])
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    metric = dcrest.metrics.kge if a.loss == "kge" else dcrest.metrics.nse
    best, best_p = None, None
    for it in range(a.iters):
        opt.zero_grad()
        q = model(P, T, E)[k0:]
        m = metric(q, torch.nan_to_num(obs), mask=mask)
        loss = (1 - m).mean()
        loss.backward()
        opt.step()
        if best is None or loss.item() < best:
            best, best_p = loss.item(), {k: v.detach().clone() for k, v in model.physical.items()}
        if it % max(1, a.iters // 10) == 0:
            print(f"iter {it}: median {a.loss.upper()} {m.median().item():.3f}", flush=True)
    out = pd.DataFrame({k: best_p[k].numpy() for k in PKEYS}, index=pd.Index(ids, name="id"))
    out["area_km2"] = best_p["area_km2"].numpy()
    out.to_csv(a.out, float_format="%.6g")
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info").set_defaults(fn=cmd_info)
    p = sub.add_parser("simulate")
    p.add_argument("--set", choices=("donors", "targets"), required=True)
    p.add_argument("--params", required=True); p.add_argument("--out", required=True)
    p.add_argument("--ids"); p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--spinup-from", dest="spinup_from", help="default: 3 years before --start")
    p.set_defaults(fn=cmd_simulate)
    p = sub.add_parser("evaluate")
    p.add_argument("--params", required=True); p.add_argument("--ids"); p.add_argument("--out")
    p.set_defaults(fn=cmd_evaluate)
    p = sub.add_parser("calibrate")
    p.add_argument("--ids", required=True); p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=400); p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--loss", choices=("nse", "kge"), default="nse")
    p.add_argument("--init", choices=("released", "default"), default="default")
    p.set_defaults(fn=cmd_calibrate)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
