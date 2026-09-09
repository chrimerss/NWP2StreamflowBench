"""End-to-end benchmark: analysis run, forecast runs, scoring, metric tables."""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import analysis, metrics, usgs
from .basins import load_basins
from .config import Config
from .hydro import extend_forcing, load_model, run_forecasts, simulate
from .nwp.base import load_stored
from .regions import REGION_ORDER

log = logging.getLogger(__name__)

SCORES = {"nse": metrics.nse, "kge": metrics.kge, "corr": metrics.pearson, "bias": metrics.bias_ratio}


def _score_table(sim: np.ndarray, obs: np.ndarray, ids: List[str], min_n: int) -> pd.DataFrame:
    out = {k: f(sim, obs, min_n) for k, f in SCORES.items()}
    out["n_days"] = metrics.valid_count(sim, obs)
    return pd.DataFrame(out, index=pd.Index(ids, name="gauge_id"))


def run_benchmark(cfg: Config) -> Dict[str, pd.DataFrame]:
    basins = load_basins(cfg.data_dir)
    ids = list(basins.gauge_id)
    n = len(ids)
    params, mcfg, meta = load_model(cfg.parameter_set, ids)

    # ---- analysis forcing (spin-up + reference) -------------------------------------------
    an = analysis.load(cfg.data_dir)
    P = analysis.to_wide(an, "P", ids)
    dates = pd.DatetimeIndex(P.index)
    P = P.values.astype(np.float32)
    T = analysis.to_wide(an, "T", ids).values.astype(np.float32)
    E = analysis.to_wide(an, "PET", ids).values.astype(np.float32)
    lag = cfg.forcing_lag_days
    if lag:
        # model day D is driven by forcing of calendar day D - lag (see configs/benchmark.yaml)
        dates = dates + pd.Timedelta(days=lag)
    an_end = dates[-1]
    for name, X in (("P", P), ("T", T), ("PET", E)):
        bad = ~np.isfinite(X)
        if bad.any():
            log.warning("%s: %d NaN cells filled with column means", name, int(bad.sum()))
            col = np.nanmean(X, axis=0)
            X[bad] = np.take(col, np.where(bad)[1])

    # ---- forecasts ---------------------------------------------------------------------------
    store: Dict[str, Dict[date, pd.DataFrame]] = {m.id: load_stored(cfg.data_dir, m.id, cfg.init_hour) for m in cfg.models}
    start = pd.Timestamp(cfg.start_date)
    all_inits = sorted({d for s in store.values() for d in s if pd.Timestamp(d) >= start})
    if not all_inits:
        raise RuntimeError("no stored NWP initialisations; run fetch-nwp first")
    last_init = pd.Timestamp(all_inits[-1])
    horizon = last_init + pd.Timedelta(days=cfg.lead_days + lag)
    dates, T = extend_forcing(dates, T, horizon)
    _, E = extend_forcing(pd.DatetimeIndex(dates[:len(P)]), E, horizon)
    _, P = extend_forcing(pd.DatetimeIndex(dates[:len(P)]), P, horizon, how="zeros")

    # ---- reference (analysis-forced) run ---------------------------------------------------
    log.info("reference run: %d days x %d basins", len(dates), n)
    Q_ref = simulate(P, T, E, params, mcfg)

    # ---- observations ------------------------------------------------------------------------
    obs = usgs.load(cfg.data_dir, ids).reindex(dates).values.astype(np.float32)
    obs_end = pd.Timestamp(usgs.load(cfg.data_dir, ids).dropna(how="all").index.max())

    # common scoring window across leads: valid dates every lead can reach, within analysis and obs
    win0 = start + pd.Timedelta(days=cfg.lead_days - 1 + lag)
    win1 = min(last_init + pd.Timedelta(days=lag), an_end, obs_end)
    win = (dates >= win0) & (dates <= win1)
    log.info("scoring window %s .. %s (%d days)", win0.date(), win1.date(), int(win.sum()))

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "discharge").mkdir(exist_ok=True)

    ref = _score_table(Q_ref[win], obs[win], ids, cfg.min_score_days)
    ref.to_csv(results_dir / "metrics_reference.csv", float_format="%.4f")
    pd.DataFrame(Q_ref[win], index=dates[win], columns=ids).to_parquet(results_dir / "discharge" / "reference.parquet")

    rows_q, rows_p = [], []
    leads = cfg.leads
    for m in cfg.models:
        fc = {d: df.reindex(ids)[[f"lead_{l}" for l in leads]].values.T.astype(np.float32)
              for d, df in store[m.id].items() if pd.Timestamp(d) >= start}
        if not fc:
            log.warning("%s: no forecasts", m.id)
            continue
        inits, Q = run_forecasts(dates, P, T, E, params, mcfg, fc, cfg.lead_days, lag=lag)
        Pf = np.stack([fc[d] for d in inits])  # (n_init, L, n)
        np.savez_compressed(results_dir / "discharge" / f"forecast_{m.id}.npz", inits=np.asarray([str(d) for d in inits]),
                            leads=np.asarray(leads), Q=Q, P=Pf, gauge_id=np.asarray(ids))
        init_pos = np.asarray([dates.get_loc(pd.Timestamp(d) + pd.Timedelta(days=lag)) for d in inits])
        for li, lead in enumerate(leads):
            vpos = init_pos + (lead - 1)
            ok = vpos < len(dates)
            sel = ok & win[np.minimum(vpos, len(dates) - 1)]
            q_l = Q[sel, li]                     # (n_valid, n) forecast discharge
            p_l = Pf[sel, li]
            o_l = obs[vpos[sel]]
            pa_l = P[vpos[sel]]
            tq = _score_table(q_l, o_l, ids, cfg.min_score_days)
            tq.insert(0, "lead", lead); tq.insert(0, "model", m.id)
            rows_q.append(tq.reset_index())
            tp = _score_table(p_l, pa_l, ids, cfg.min_score_days)
            tp["rmse"] = metrics.rmse(p_l, pa_l, cfg.min_score_days)
            tp.insert(0, "lead", lead); tp.insert(0, "model", m.id)
            rows_p.append(tp.reset_index())
        log.info("%s: %d inits scored", m.id, len(inits))

    mq = pd.concat(rows_q, ignore_index=True)
    mp = pd.concat(rows_p, ignore_index=True)
    mq.to_csv(results_dir / "metrics_streamflow.csv", index=False, float_format="%.4f")
    mp.to_csv(results_dir / "metrics_precip.csv", index=False, float_format="%.4f")

    summary = summarise(mq, mp, ref, basins)
    summary["meta"] = {
        "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
        "window_start": str(win0.date()), "window_end": str(win1.date()), "n_days": int(win.sum()),
        "first_init": str(all_inits[0]), "last_init": str(all_inits[-1]), "init_hour": cfg.init_hour,
        "n_inits": {m.id: int(len([d for d in store[m.id] if pd.Timestamp(d) >= start])) for m in cfg.models},
        "analysis_end": str((an_end - pd.Timedelta(days=lag)).date()), "obs_end": str(obs_end.date()),
        "forcing_lag_days": lag,
        "n_basins": n, "n_scored": int(ref.nse.notna().sum()), "min_score_days": cfg.min_score_days,
        "parameter_set": Path(cfg.parameter_set).name, "leads": leads,
        "models": [{"id": m.id, "name": m.name} for m in cfg.models],
    }
    with open(results_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    return {"streamflow": mq, "precip": mp, "reference": ref}


def _q(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) == 0:
        return {"median": None, "q25": None, "q75": None, "n": 0}
    return {"median": round(float(x.median()), 4), "q25": round(float(x.quantile(0.25)), 4),
            "q75": round(float(x.quantile(0.75)), 4), "n": int(len(x))}


def summarise(mq: pd.DataFrame, mp: pd.DataFrame, ref: pd.DataFrame, basins: pd.DataFrame) -> dict:
    reg = basins.region
    regions = [r for r in REGION_ORDER if (reg == r).any()]
    out = {"regions": regions, "reference": {"overall": {k: _q(ref[k]) for k in ("nse", "kge")}, "by_region": {}},
           "streamflow": {}, "precip": {}}
    for r in regions:
        sub = ref[reg.reindex(ref.index).values == r]
        out["reference"]["by_region"][r] = {k: _q(sub[k]) for k in ("nse", "kge")}
    for name, tbl, keys in (("streamflow", mq, ("nse", "kge", "corr", "bias")), ("precip", mp, ("nse", "corr", "bias", "rmse"))):
        for mid, g in tbl.groupby("model"):
            entry = {"overall": [], "by_region": {r: [] for r in regions}}
            for lead, gl in g.groupby("lead"):
                gl = gl.set_index("gauge_id")
                entry["overall"].append({"lead": int(lead), **{k: _q(gl[k]) for k in keys}})
                rr = reg.reindex(gl.index).values
                for r in regions:
                    entry["by_region"][r].append({"lead": int(lead), **{k: _q(gl[k][rr == r]) for k in keys}})
            out[name][mid] = entry
    return out
