"""NOAA/NCEI climate region of each CAMELS gauge (by gauge state), from data/hydroagent/regions.csv."""
from __future__ import annotations

from typing import List

import pandas as pd

from . import config


def load_regions(ids: List[str] | None = None) -> pd.Series:
    r = pd.read_csv(config.REPO / "data" / "hydroagent" / "regions.csv", dtype={"gauge_id": str}).set_index("gauge_id").region
    return (r.reindex(ids) if ids is not None else r).fillna("Other")
