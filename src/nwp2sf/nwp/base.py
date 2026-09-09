"""Interface every NWP precipitation source implements.

A source turns one forecast initialisation into basin-mean daily precipitation
for lead days 1..L (``mm/day`` on the benchmark's basin set). Everything after
that — the hydrologic run, scoring, the dashboard — is model-agnostic, so adding
an in-house model means writing one subclass and one entry in the config.

Daily lead ``d`` covers ``[init + h0 + 24 (d-1) h, init + h0 + 24 d h)`` where ``h0`` is
``day_start_hour`` (6 UTC by default, matching the gridMET day used for the analysis).
"""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


class PrecipForecastSource(ABC):
    #: name of the basin weight grid this source needs (``data/basins/weights_<grid>.npz``)
    grid: str = ""

    def __init__(self, data_dir: Path, cache_dir: Path, basin_ids: List[str], lead_days: int, init_hour: int = 0,
                 day_start_hour: int = 0):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.basin_ids = list(basin_ids)
        self.lead_days = int(lead_days)
        self.init_hour = int(init_hour)
        self.day_start_hour = int(day_start_hour)

    @abstractmethod
    def available(self, start: date, end: date) -> List[date]:
        """Initialisation dates in ``[start, end]`` that the source can serve."""

    @abstractmethod
    def fetch(self, init: date) -> pd.DataFrame:
        """Basin-mean precipitation, ``DataFrame`` indexed by gauge id with columns ``lead_1..lead_L`` (mm/day)."""

    # ---- storage of the extracted basin-scale product (shared by all sources) ----
    def store_dir(self, model_id: str) -> Path:
        return self.data_dir / "nwp" / model_id

    def path(self, model_id: str, init: date) -> Path:
        return self.store_dir(model_id) / f"{init:%Y%m%d}{self.init_hour:02d}.parquet"

    def fetch_and_store(self, model_id: str, init: date, overwrite: bool = False) -> Optional[Path]:
        p = self.path(model_id, init)
        if p.exists() and not overwrite:
            return p
        df = self.fetch(init)
        if df is None:
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        df = df.reindex(self.basin_ids).astype(np.float32)
        df.index.name = "gauge_id"
        df.to_parquet(p)
        return p


def load_fetcher(spec: str):
    """``'package.module:ClassName'`` -> class."""
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls)


def load_stored(data_dir: Path, model_id: str, init_hour: int = 0) -> dict:
    """All stored inits for a model: ``{date: DataFrame}``."""
    out = {}
    d = Path(data_dir) / "nwp" / model_id
    if not d.exists():
        return out
    for p in sorted(d.glob("*.parquet")):
        stem = p.stem
        if len(stem) != 10 or int(stem[8:10]) != init_hour:
            continue
        out[pd.Timestamp(stem[:8]).date()] = pd.read_parquet(p)
    return out
