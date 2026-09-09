"""ECMWF open data (IFS HRES and AIFS-single) via the official ``ecmwf-opendata`` client.

Total precipitation ``tp`` is accumulated from the start of the forecast, so
the daily amount for lead day ``d`` is ``tp(h0 + 24 d) - tp(h0 + 24 (d-1))`` with
``h0 = day_start_hour``. Units differ
between models (IFS: m, AIFS: kg m-2) and are read from the GRIB header.

The AWS mirror (``s3://ecmwf-forecasts``) keeps the full archive from 2023, which
is what makes a retrospective benchmark from 2026-01-01 possible; ECMWF's own
server only holds the last few days. Only the ``tp`` messages are transferred
(the client resolves byte ranges from the ``.index`` files).
"""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ..gridweights import check_grid, load_weights, weighted_mean
from .base import PrecipForecastSource

log = logging.getLogger(__name__)

UNIT_TO_MM = {"m": 1000.0, "kg m**-2": 1.0, "mm": 1.0}


class ECMWFOpenData(PrecipForecastSource):
    grid = "ecmwf0p25"

    def __init__(self, *args, model: str = "ifs", source: str = "aws", resol: str = "0p25", **kw):
        super().__init__(*args, **kw)
        self.model = model
        self.source = source
        self.resol = resol
        self.W, self.lat, self.lon, wids = load_weights(self.data_dir / "basins" / f"weights_{self.grid}.npz")
        if wids != self.basin_ids:
            raise ValueError("weight file basin order does not match the requested basin set")
        from ecmwf.opendata import Client
        self.client = Client(source=source, model=model, resol=resol)
        # 360 h is the open-data horizon for both IFS (oper) and AIFS-single
        self.steps = [self.day_start_hour + 24 * d for d in range(0, self.lead_days + 1)]
        if self.steps[-1] > 360:
            raise ValueError(f"lead_days={self.lead_days} with day_start_hour={self.day_start_hour} exceeds the 360 h horizon")

    # ------------------------------------------------------------------ availability
    def available(self, start: date, end: date) -> List[date]:
        try:
            latest = self.client.latest(type="fc", stream="oper", time=self.init_hour, step=self.steps[-1], param="tp")
            last = latest.date()
        except Exception as e:  # pragma: no cover - network
            log.warning("latest() failed (%s); assuming yesterday", e)
            last = date.today() - timedelta(days=1)
        end = min(end, last)
        return [start + timedelta(days=i) for i in range((end - start).days + 1)]

    # ------------------------------------------------------------------ fetch
    def fetch(self, init: date) -> Optional[pd.DataFrame]:
        import eccodes
        with tempfile.TemporaryDirectory(dir=self.cache_dir if self.cache_dir.exists() else None) as td:
            target = os.path.join(td, f"{self.model}_{init:%Y%m%d}_tp.grib2")
            try:
                self.client.retrieve(date=init.strftime("%Y-%m-%d"), time=self.init_hour, type="fc",
                                     stream="oper", step=self.steps, param="tp", target=target)
            except Exception as e:
                log.warning("%s %s: retrieve failed: %s", self.model, init, e)
                return None
            accum = {}
            with open(target, "rb") as fh:
                while True:
                    h = eccodes.codes_grib_new_from_file(fh)
                    if h is None:
                        break
                    try:
                        step = int(eccodes.codes_get(h, "endStep"))
                        units = eccodes.codes_get(h, "units")
                        nj, ni = eccodes.codes_get(h, "Nj"), eccodes.codes_get(h, "Ni")
                        lat0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
                        lon0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
                        dlat = eccodes.codes_get(h, "jDirectionIncrementInDegrees")
                        dlon = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
                        jpos = eccodes.codes_get(h, "jScansPositively")
                        vals = eccodes.codes_get_values(h)
                    finally:
                        eccodes.codes_release(h)
                    lat = lat0 + (dlat if jpos else -dlat) * np.arange(nj)
                    lon = ((lon0 + dlon * np.arange(ni)) + 180.0) % 360.0 - 180.0
                    check_grid(self.lat, self.lon, lat, lon)
                    factor = UNIT_TO_MM.get(units)
                    if factor is None:
                        raise ValueError(f"unknown tp units {units!r}")
                    accum[step] = weighted_mean(self.W, vals.reshape(1, -1) * factor)[0]
        missing = [s for s in self.steps if s not in accum]
        if missing:
            log.warning("%s %s: missing steps %s", self.model, init, missing)
            return None
        cols = {}
        for d in range(1, len(self.steps)):
            daily = accum[self.steps[d]] - accum[self.steps[d - 1]]
            cols[f"lead_{d}"] = np.clip(daily, 0.0, None)
        return pd.DataFrame(cols, index=pd.Index(self.basin_ids, name="gauge_id"))
