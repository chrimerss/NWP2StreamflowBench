"""Write the data files behind the static GitHub Pages dashboard (``docs/``).

``docs/index.html`` is a hand-written page (Plotly.js) that reads
``docs/data/summary.json``; this module only regenerates that JSON from the
metric tables so that the site updates with every pipeline run.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .basins import load_basins
from .config import Config


def _clean(x):
    return None if (x is None or (isinstance(x, float) and not np.isfinite(x))) else x


def build_site(cfg: Config) -> Path:
    results = Path(cfg.results_dir)
    docs = Path(cfg.docs_dir)
    (docs / "data").mkdir(parents=True, exist_ok=True)
    with open(results / "summary.json") as fh:
        summary = json.load(fh)
    basins = load_basins(cfg.data_dir)
    ref = pd.read_csv(results / "metrics_reference.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    mq = pd.read_csv(results / "metrics_streamflow.csv", dtype={"gauge_id": str})
    mp = pd.read_csv(results / "metrics_precip.csv", dtype={"gauge_id": str})

    ids = list(basins.gauge_id)
    leads = summary["meta"]["leads"]
    per_basin = {"ids": ids,
                 "name": basins.name.tolist(), "lat": basins.lat.round(4).tolist(), "lon": basins.lon.round(4).tolist(),
                 "region": basins.region.tolist(), "state": basins.state.tolist(),
                 "area_km2": basins.area_km2.round(1).tolist(),
                 "ref_nse": [_clean(v) for v in ref.nse.reindex(ids).round(3).tolist()],
                 "ref_kge": [_clean(v) for v in ref.kge.reindex(ids).round(3).tolist()],
                 "nse": {}, "kge": {}, "p_corr": {}}
    for mid, g in mq.groupby("model"):
        for key in ("nse", "kge"):
            tab = g.pivot(index="gauge_id", columns="lead", values=key).reindex(index=ids, columns=leads)
            per_basin[key][mid] = [[_clean(v) for v in row] for row in tab.round(3).values.tolist()]
    for mid, g in mp.groupby("model"):
        tab = g.pivot(index="gauge_id", columns="lead", values="corr").reindex(index=ids, columns=leads)
        per_basin["p_corr"][mid] = [[_clean(v) for v in row] for row in tab.round(3).values.tolist()]
    summary["basins"] = per_basin
    with open(docs / "data" / "summary.json", "w") as fh:
        json.dump(summary, fh, separators=(",", ":"))
    # keep the metric tables next to the site so the "download" links work on Pages
    for f in ("metrics_streamflow.csv", "metrics_precip.csv", "metrics_reference.csv"):
        shutil.copy(results / f, docs / "data" / f)
    (docs / ".nojekyll").touch()
    return docs / "data" / "summary.json"
