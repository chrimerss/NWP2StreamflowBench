"""Google DeepMind WeatherNext 3 — IMERG-calibrated precipitation at 0.1°.

WeatherNext 3 publishes hourly forecasts as Zarr v3 stores on Google Cloud
Storage (docs: https://developers.google.com/weathernext/guides/gcs):

* ``gs://weathernext3_statistics_spatial/weathernext_3_0_0_statistics/zarr/``
  — precomputed ensemble statistics (``<var>_mean``, ``_p10`` … ``_p90``),
  no requester-pays;
* ``gs://weathernext3_spatial/weathernext_3_0_0/zarr/`` — the raw 64-member
  ensemble (``sample`` dimension), requester-pays.

Both hold one store per initialisation under
``<year_dir>/<YYYYMMDD>_<HH>hr_<NN>_preds/predictions.zarr`` with the 0.1°
surface grid on ``lat_0p1`` / ``lon_0p1`` (longitudes 0–360) and hourly
accumulations in metres. ``imerg_tp_1hr`` is the precipitation head trained
against NASA IMERG; ``total_precipitation_1hr`` is the ERA5-style head.

Access is allowlisted per Google account (request form on the WeatherNext
developer site). Locally, authenticate once with
``gcloud auth application-default login``; in CI provide a service-account key
through ``GOOGLE_APPLICATION_CREDENTIALS``. Set ``GOOGLE_CLOUD_PROJECT`` for the
requester-pays ensemble bucket. Without credentials the source reports no
available initialisations and the rest of the benchmark proceeds.

The basin/grid weights for the 0.1° grid are built on first use from the
store's own coordinates (needs the ``basins`` extra and the GAGES-II cache) and
then committed like the other weight files.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..gridweights import check_grid, load_weights, weighted_mean
from .base import PrecipForecastSource

log = logging.getLogger(__name__)

STATS_ROOT = "gs://weathernext3_statistics_spatial/weathernext_3_0_0_statistics/zarr"
ENSEMBLE_ROOT = "gs://weathernext3_spatial/weathernext_3_0_0/zarr"
LEAD_DIMS = ("lead_time", "prediction_timedelta", "forecast_hour", "time")
INIT_DIMS = ("init_time", "time_init", "forecast_reference_time")


class WeatherNext3(PrecipForecastSource):
    grid = "wn3_0p1"

    def __init__(self, *args, variable: str = "imerg_tp_1hr", store: str = "statistics", stat: str = "mean",
                 member: Optional[int] = None, root: Optional[str] = None, year_dir: str = "2026_to_present",
                 project: Optional[str] = None, **kw):
        super().__init__(*args, **kw)
        self.variable = variable
        self.store_kind = store
        self.stat = stat
        self.member = member
        self.year_dir = year_dir
        self.root = root or (STATS_ROOT if store == "statistics" else ENSEMBLE_ROOT)
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.max_hours = self.day_start_hour + 24 * self.lead_days
        if self.max_hours > 360:
            raise ValueError("WeatherNext 3 6-hourly cycles reach 360 h; reduce lead_days or day_start_hour")
        self._fs = None
        self._weights = None

    # ------------------------------------------------------------------ storage access
    @property
    def fs(self):
        if self._fs is None:
            import fsspec
            if self.root.startswith("gs://"):
                import gcsfs
                self._fs = gcsfs.GCSFileSystem(token="google_default", project=self.project,
                                               requester_pays=bool(self.project) and self.store_kind == "ensemble")
            else:
                self._fs = fsspec.filesystem(fsspec.core.split_protocol(self.root)[0] or "file")
        return self._fs

    def _init_prefix(self) -> str:
        return f"{self.root.rstrip('/')}/{self.year_dir}/"

    def _store_path(self, init: date) -> Optional[str]:
        pattern = f"{init:%Y%m%d}_{self.init_hour:02d}hr_"
        try:
            entries = self.fs.ls(self._init_prefix(), detail=False)
        except Exception as e:
            log.warning("WeatherNext 3: cannot list %s (%s)", self._init_prefix(), e)
            return None
        cands = sorted(e for e in entries if os.path.basename(e.rstrip("/")).startswith(pattern))
        if not cands:
            return None
        return cands[-1].rstrip("/") + "/predictions.zarr"

    def available(self, start: date, end: date) -> List[date]:
        try:
            entries = self.fs.ls(self._init_prefix(), detail=False)
        except Exception as e:
            log.warning("WeatherNext 3 not accessible (%s); skipping. See nwp2sf/nwp/weathernext.py for access setup.", e)
            return []
        rx = re.compile(rf"(\d{{8}})_{self.init_hour:02d}hr_\d+_preds$")
        found = set()
        for e in entries:
            m = rx.search(os.path.basename(e.rstrip("/")))
            if m:
                found.add(pd.Timestamp(m.group(1)).date())
        # only initialisations whose 360 h horizon is fully published
        cutoff = date.today() - timedelta(days=1)
        return sorted(d for d in found if start <= d <= min(end, cutoff))

    # ------------------------------------------------------------------ grid weights
    def _get_weights(self, lat: np.ndarray, lon: np.ndarray):
        if self._weights is not None:
            return self._weights
        path = self.data_dir / "basins" / f"weights_{self.grid}.npz"
        if path.exists():
            W, wlat, wlon, wids = load_weights(path)
            try:
                check_grid(wlat, wlon, lat, lon)
            except ValueError:
                log.warning("%s does not match the store grid; rebuilding", path.name)
                W = None
        else:
            W = None
        if W is None:
            from ..basins import build_grid_weights
            W, wids = build_grid_weights(self.data_dir, self.cache_dir, self.grid, lat, lon)
        pos = {b: i for i, b in enumerate(wids)}
        W = W[[pos[b] for b in self.basin_ids]]
        self._weights = W
        return W

    # ------------------------------------------------------------------ fetch
    def fetch(self, init: date) -> Optional[pd.DataFrame]:
        import xarray as xr
        path = self._store_path(init)
        if path is None:
            log.warning("WeatherNext 3 %s: no store found", init)
            return None
        try:
            mapper = self.fs.get_mapper(path)
            ds = xr.open_zarr(mapper, consolidated=None, chunks=None)
        except Exception as e:
            log.warning("WeatherNext 3 %s: cannot open %s (%s)", init, path, e)
            return None
        with ds:
            da = self._select_variable(ds)
            da = self._squeeze_init(da)
            lead_dim = next((d for d in LEAD_DIMS if d in da.dims), None)
            if lead_dim is None:
                raise ValueError(f"no lead dimension in {list(da.dims)}")
            lat_name = next(d for d in da.dims if d.startswith("lat"))
            lon_name = next(d for d in da.dims if d.startswith("lon"))
            hours = self._lead_hours(da[lead_dim].values)
            lat = da[lat_name].values.astype(float)
            lon_raw = da[lon_name].values.astype(float)
            lon = (lon_raw + 180.0) % 360.0 - 180.0
            # CONUS window of the basin set, in the store's own longitude convention
            from ..basins import load_basins
            b = load_basins(self.data_dir)
            lat_sel = (lat >= b.lat.min() - 1.5) & (lat <= b.lat.max() + 1.5)
            lon_sel = (lon >= b.lon.min() - 1.5) & (lon <= b.lon.max() + 1.5)
            need = (hours > self.day_start_hour) & (hours <= self.max_hours)
            if need.sum() < 24 * self.lead_days:
                log.warning("WeatherNext 3 %s: only %d of %d hourly steps present", init, int(need.sum()), 24 * self.lead_days)
                return None
            sub = da.isel({lead_dim: np.where(need)[0], lat_name: np.where(lat_sel)[0], lon_name: np.where(lon_sel)[0]})
            sub = sub.transpose(lead_dim, lat_name, lon_name)
            vals = np.asarray(sub.values, dtype=np.float64) * 1000.0     # m -> mm
            hrs = hours[need]
        # daily totals on the full grid footprint (zeros outside the window) so the weights apply unchanged
        W = self._get_weights(lat, lon)
        order = np.argsort(lon)                      # weights are built on ascending -180..180 longitudes
        nlat, nlon = len(lat), len(lon)
        li, lo = np.where(lat_sel)[0], np.where(lon_sel)[0]
        cols = {}
        for d in range(1, self.lead_days + 1):
            h0, h1 = self.day_start_hour + 24 * (d - 1), self.day_start_hour + 24 * d
            daily = vals[(hrs > h0) & (hrs <= h1)].sum(axis=0)
            full = np.full((nlat, nlon), np.nan)
            full[np.ix_(li, lo)] = daily
            full = full[:, order]
            cols[f"lead_{d}"] = np.clip(weighted_mean(W, full.reshape(1, -1))[0], 0.0, None)
        return pd.DataFrame(cols, index=pd.Index(self.basin_ids, name="gauge_id"))

    # ------------------------------------------------------------------ helpers
    def _select_variable(self, ds):
        if self.store_kind == "statistics":
            name = f"{self.variable}_{self.stat}"
            if name not in ds:
                raise KeyError(f"{name} not in store; variables: {list(ds.data_vars)[:20]}")
            return ds[name]
        da = ds[self.variable]
        if "sample" in da.dims:
            da = da.mean("sample") if self.member is None else da.isel(sample=self.member)
        return da

    @staticmethod
    def _squeeze_init(da):
        for d in INIT_DIMS:
            if d in da.dims:
                if da.sizes[d] != 1:
                    raise ValueError(f"expected a single initialisation, found {da.sizes[d]} along {d}")
                da = da.isel({d: 0})
        return da

    @staticmethod
    def _lead_hours(v: np.ndarray) -> np.ndarray:
        if np.issubdtype(v.dtype, np.timedelta64):
            return (v / np.timedelta64(1, "h")).astype(float)
        if np.issubdtype(v.dtype, np.datetime64):
            raise ValueError("lead coordinate is absolute time; expected a timedelta or hours")
        return np.asarray(v, dtype=float)
