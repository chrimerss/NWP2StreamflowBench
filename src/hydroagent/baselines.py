"""Classical regionalization baselines for the virtual-ungauged experiment.

Every method transfers the released dCREST-CAMELS v1.0 parameters of *donor* basins
(the other folds) to each held-out *target*. Each donor's parameters were calibrated
on its own 1999-2008 record only, so no target information enters a baseline.
The routing scale depends on catchment area, so a transferred set always takes the
target's own area.

    gauged_upper   target's own released parameters (the gauged upper bound)
    nn_spatial     nearest donor by centroid distance
    knn_spatial    5 nearest donors, output averaging (Oudin et al. 2008)
    knn_attr       5 most similar donors in standardized attribute space, output averaging
    global_median  median donor parameter set
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import dcrest

from . import config
from .metrics import kge, nse
from .simulate import run_period

SIMILARITY_ATTRS = (
    "p_mean", "pet_mean", "aridity", "p_seasonality", "frac_snow", "high_prec_freq", "high_prec_dur",
    "low_prec_freq", "low_prec_dur", "elev_mean", "slope_mean", "log_area", "frac_forest", "lai_max",
    "gvf_max", "soil_depth_pelletier", "soil_porosity", "soil_conductivity", "max_water_content",
    "sand_frac", "clay_frac", "carbonate_rocks_frac", "geol_permeability",
)
K = 5


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1)[:, None], np.deg2rad(lat2)[None, :]
    dl = np.deg2rad(lon2)[None, :] - np.deg2rad(lon1)[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def attr_matrix(attrs: pd.DataFrame) -> pd.DataFrame:
    a = attrs.copy()
    a["log_area"] = np.log10(a["area_gages2"])
    return a[list(SIMILARITY_ATTRS)].astype(float)


def donor_choice(attrs: pd.DataFrame, targets: List[str], donors: List[str]) -> Dict[str, np.ndarray]:
    """Index arrays into ``donors`` for each method: shape (n_targets, k)."""
    d = haversine_km(attrs.loc[targets, "gauge_lat"].values, attrs.loc[targets, "gauge_lon"].values,
                     attrs.loc[donors, "gauge_lat"].values, attrs.loc[donors, "gauge_lon"].values)
    X = attr_matrix(attrs)
    mu, sd = X.loc[donors].mean(), X.loc[donors].std().replace(0, 1)
    Z = ((X - mu) / sd).fillna(0.0)
    da = np.sqrt(((Z.loc[targets].values[:, None, :] - Z.loc[donors].values[None, :, :]) ** 2).sum(-1))
    return {"nn_spatial": np.argsort(d, axis=1)[:, :1], "knn_spatial": np.argsort(d, axis=1)[:, :K],
            "knn_attr": np.argsort(da, axis=1)[:, :K]}


def _params_for(params_all: dict, pos: Dict[str, int], donor_ids: List[str], target_area: np.ndarray) -> dict:
    idx = torch.as_tensor([pos[g] for g in donor_ids])
    p = {k: v[idx].clone() for k, v in params_all.items()}
    p["area_km2"] = torch.as_tensor(target_area, dtype=p["area_km2"].dtype)
    return p


def run(data: dict, attrs: pd.DataFrame, targets: List[str], donors: List[str]) -> Dict[str, pd.DataFrame]:
    """Per-target NSE/KGE on the test period for every baseline."""
    params_all, meta = dcrest.load_parameters(str(config.PARAMS), subset=data["ids"])
    cfg = dcrest.config_from_meta(meta)
    pos = {g: i for i, g in enumerate(data["ids"])}
    tcols = np.asarray([pos[g] for g in targets])
    area_t = np.asarray(data["area_km2"])[tcols]
    choice = donor_choice(attrs, targets, donors)
    out: Dict[str, pd.DataFrame] = {}

    def score(Q, obs):
        return pd.DataFrame({"nse": nse(Q, obs), "kge": kge(Q, obs)}, index=pd.Index(targets, name="gauge_id"))

    p = _params_for(params_all, pos, targets, area_t)
    Q, obs = run_period(data, p, cfg, *config.TEST, spinup_from=config.TEST_SPINUP_FROM, cols=tcols)
    out["gauged_upper"] = score(Q, obs)

    for name, ix in choice.items():
        k = ix.shape[1]
        donor_ids = [donors[j] for j in ix.reshape(-1)]          # target-major, k per target
        cols = np.repeat(tcols, k)
        p = _params_for(params_all, pos, donor_ids, np.repeat(area_t, k))
        Qk, _ = run_period(data, p, cfg, *config.TEST, spinup_from=config.TEST_SPINUP_FROM, cols=cols)
        Qm = Qk.reshape(Qk.shape[0], len(targets), k).mean(axis=2)
        out[name] = score(Qm, obs)

    med = {k: torch.as_tensor(np.median(params_all[k][[pos[g] for g in donors]].numpy()), dtype=torch.float32)
           .repeat(len(targets)) for k in params_all if k != "area_km2"}
    med["area_km2"] = torch.as_tensor(area_t, dtype=torch.float32)
    Q, _ = run_period(data, med, cfg, *config.TEST, spinup_from=config.TEST_SPINUP_FROM, cols=tcols)
    out["global_median"] = score(Q, obs)
    return out


def summary(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {m: {"nse_median": r.nse.median(), "kge_median": r.kge.median(),
                "nse_q25": r.nse.quantile(.25), "nse_q75": r.nse.quantile(.75),
                "frac_nse_gt0": float((r.nse > 0).mean()), "n": int(r.nse.notna().sum())}
            for m, r in results.items()}
    return pd.DataFrame(rows).T
