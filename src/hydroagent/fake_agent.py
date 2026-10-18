"""Scripted stand-in for the LLM agent, used only to test the pipeline offline.

kNN physical-similarity transfer with output averaging (k=5), using only workspace data.
Runs inside a task workspace: `python3 output/code/knn_transfer.py`.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("HYDRO_WORKSPACE", Path.cwd()))
sys.path.insert(0, str(ROOT / "scripts"))
import hydrokit as hk  # noqa: E402

ATTRS = ["p_mean", "pet_mean", "aridity", "frac_snow", "high_prec_freq", "elev_mean", "slope_mean",
         "frac_forest", "soil_depth_pelletier", "soil_conductivity", "lat", "lon"]
dd = ROOT / "task_folder" / "data"
da = pd.read_csv(dd / "donor_attributes.csv").set_index("id")
ta = pd.read_csv(dd / "target_attributes.csv").set_index("id")
mu, sd = da[ATTRS].mean(), da[ATTRS].std()
Z = ((da[ATTRS] - mu) / sd).fillna(0).values
ZT = ((ta[ATTRS] - mu) / sd).fillna(0).values
rel = pd.read_csv(dd / "donor_params_released.csv").set_index("id")
targets = hk.load("targets")
s, e = hk.META["predict_window"]
cols = {}
for i, t in enumerate(ta.index):
    nn = np.argsort(((Z - ZT[i]) ** 2).sum(1))[:5]
    p = rel.iloc[nn].copy()
    p["area_km2"] = targets["area_km2"][targets["ids"].index(t)]
    p.index = [t] * len(p)
    p = p.reset_index().rename(columns={"index": "id"})
    params = {k: hk.torch.as_tensor(p[k].values, dtype=hk.torch.float32) for k in hk.PKEYS + ["area_km2"]}
    d = dict(targets)
    j = targets["ids"].index(t)
    for key in ("P", "T", "PET"):
        d[key] = np.repeat(targets[key][:, j:j + 1], len(nn), axis=1)
    d["ids"] = [f"{t}_{k}" for k in range(len(nn))]
    dates, Q = hk.simulate(d, params, d["ids"], s, e, hk.META["predict_spinup_from"])
    cols[t] = Q.mean(axis=1)
out = pd.DataFrame(cols, index=pd.Index(dates, name="date"))
out.to_csv(ROOT / "output" / "target_discharge.csv", float_format="%.5f")
(ROOT / "output" / "method.md").write_text(
    "# Method\n\nPhysical-similarity regionalization: for each target the 5 donors closest in a "
    "standardized space of 12 attributes; their released parameters (with the target's area) are run "
    "and the discharges averaged.\n")
print("wrote", out.shape)
