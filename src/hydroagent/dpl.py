"""Differentiable parameter learning (dPL) baseline for out-of-distribution basins.

A small network g(static attributes) -> 16 dCREST parameters is trained by gradient
descent *through* dCREST on all training gauges of a split at once (Tsai et al. 2021
style, static-attribute variant). It never sees an out-of-distribution basin; at test
time the same network is applied to their attributes.

Protocol, identical to the released dCREST-CAMELS calibration:
  training loss   mean over training gauges of (1 - NSE), 1999-10-01..2008-09-30,
                  spin-up from 1996-10-01 (default initial states)
  validation      10% of the training gauges, held out of the loss, same window;
                  the iterate with the best validation median NSE is kept
  test            out-of-distribution basins, 1989-10-01..1999-09-30, spin-up 1986-10-01

Parameter bounds and the unit-hydrograph clamp are those the released set was fitted
under; routing uses each basin's own area.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import dcrest

from . import config
from .baselines import SIMILARITY_ATTRS, attr_matrix
from .metrics import kge, nse
from .simulate import window

HIDDEN = 256
ITERS = 500
LR = 3e-3
VAL_FRAC = 0.10
EVAL_EVERY = 20


class ParamNet(torch.nn.Module):
    def __init__(self, n_in: int, keys: List[str], init_raw: Dict[str, float], dropout: float = 0.1):
        super().__init__()
        self.keys = keys
        self.body = torch.nn.Sequential(
            torch.nn.Linear(n_in, HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(HIDDEN, len(keys)))
        last = self.body[-1]
        torch.nn.init.normal_(last.weight, std=1e-3)
        with torch.no_grad():                      # start from the package default parameters
            last.bias.copy_(torch.tensor([init_raw[k] for k in keys]))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.body(x)
        return {k: z[:, i] for i, k in enumerate(self.keys)}


def _features(attrs: pd.DataFrame, train_ids: List[str]) -> pd.DataFrame:
    X = attr_matrix(attrs)
    mu, sd = X.loc[train_ids].mean(), X.loc[train_ids].std().replace(0, 1)
    return ((X - mu) / sd).fillna(0.0)


def run_split(data: dict, attrs: pd.DataFrame, train_ids: List[str], test_ids: List[str], *,
              seed: int = 0, iters: int = ITERS, threads: int = 4, log=None) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed)
    _, meta = dcrest.load_parameters(str(config.PARAMS), subset=train_ids[:1])
    cfg = dcrest.config_from_meta(meta)
    bounds = cfg.bounds
    keys = list(bounds)
    defaults = dcrest.params.default_raw(1, bounds=bounds)
    init_raw = {k: float(v.detach()[0]) for k, v in defaults.items()}

    tr = list(train_ids)
    rng.shuffle(tr)
    n_val = max(1, int(round(VAL_FRAC * len(tr))))
    val, fit = tr[:n_val], tr[n_val:]
    ids = fit + val
    pos = {g: i for i, g in enumerate(data["ids"])}
    cols = np.asarray([pos[g] for g in ids])
    F = _features(attrs, train_ids)
    X = torch.as_tensor(F.loc[ids].values, dtype=torch.float32)
    area = torch.as_tensor(np.asarray(data["area_km2"])[cols], dtype=torch.float32)

    dates = data["dates"]
    run = slice(window(dates, config.TRAIN_SPINUP_FROM, config.TRAIN[1]).start, window(dates, *config.TRAIN).stop)
    sc = window(dates, *config.TRAIN)
    k0 = sc.start - run.start
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a[run][:, cols]), dtype=torch.float32)  # noqa: E731
    P, T, E = t(data["P"]), t(data["T"]), t(data["PET"])
    obs = torch.as_tensor(np.ascontiguousarray(data["obs"][sc][:, cols]), dtype=torch.float32)
    mask = torch.isfinite(obs)
    obs0 = torch.nan_to_num(obs)
    nfit = len(fit)

    net = ParamNet(X.shape[1], keys, init_raw)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    best = (-np.inf, None, -1)
    hist = []
    t0 = time.time()
    for it in range(iters):
        net.train()
        opt.zero_grad()
        p = dcrest.params.to_physical(net(X), bounds)
        p["area_km2"] = area
        q = dcrest.simulate(P, T, E, p, cfg, validate=False)[k0:]
        m = dcrest.metrics.nse(q, obs0, mask=mask)
        loss = (1 - m[:nfit]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if it % EVAL_EVERY == 0 or it == iters - 1:
            net.eval()
            with torch.no_grad():
                p = dcrest.params.to_physical(net(X), bounds)
                p["area_km2"] = area
                q = dcrest.simulate(P, T, E, p, cfg, validate=False)[k0:]
                m = dcrest.metrics.nse(q, obs0, mask=mask)
            fit_med, val_med = float(m[:nfit].median()), float(m[nfit:].median())
            hist.append({"iter": it, "fit_nse": fit_med, "val_nse": val_med, "s": round(time.time() - t0)})
            if log:
                log(f"iter {it}: fit median NSE {fit_med:.3f}, validation {val_med:.3f}")
            if val_med > best[0]:
                best = (val_med, {k: v.detach().clone() for k, v in net.state_dict().items()}, it)
    net.load_state_dict(best[1])
    net.eval()

    # out-of-distribution test
    tcols = np.asarray([pos[g] for g in test_ids])
    Ft = torch.as_tensor(F.loc[test_ids].values, dtype=torch.float32)
    with torch.no_grad():
        pt = dcrest.params.to_physical(net(Ft), bounds)
    pt["area_km2"] = torch.as_tensor(np.asarray(data["area_km2"])[tcols], dtype=torch.float32)
    from .simulate import run_period
    Q, o = run_period(data, pt, cfg, *config.TEST, spinup_from=config.TEST_SPINUP_FROM, cols=tcols)
    res = pd.DataFrame({"nse": nse(Q, o), "kge": kge(Q, o)}, index=pd.Index(test_ids, name="gauge_id"))
    params = pd.DataFrame({k: v.numpy() for k, v in pt.items()}, index=pd.Index(test_ids, name="gauge_id"))
    return {"test": res, "params": params, "history": hist, "best_iter": best[2], "best_val_nse": best[0],
            "n_fit": nfit, "n_val": len(val), "wall_s": round(time.time() - t0)}


def splits(attrs: pd.DataFrame, regions: pd.Series) -> Dict[str, dict]:
    """Random folds (protocol.folds) and leave-one-region-out splits."""
    from .protocol import make_tasks
    out = {}
    for name, t in make_tasks(attrs, regions).items():
        if name.startswith("fold") and "-" not in name:
            out[name] = {"train": list(t.donors), "test": list(t.targets), "kind": "random-fold"}
    ids = sorted(attrs.index)
    for r in sorted(set(regions.reindex(ids))):
        test = [g for g in ids if regions.get(g) == r]
        out["region-" + r.replace(" ", "-")] = {"train": [g for g in ids if regions.get(g) != r], "test": test,
                                                "kind": "leave-region-out"}
    return out


def save(name: str, kind: str, got: dict) -> None:
    out = config.RESULTS / "dpl"
    out.mkdir(parents=True, exist_ok=True)
    got["test"].assign(split=name, kind=kind).to_csv(out / f"{name}_test.csv", float_format="%.4f")
    got["params"].to_csv(out / f"{name}_params.csv", float_format="%.6g")
    (out / f"{name}_meta.json").write_text(json.dumps(
        {k: got[k] for k in ("history", "best_iter", "best_val_nse", "n_fit", "n_val", "wall_s")}, indent=1))
