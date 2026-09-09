"""Observed-forcing proxy ("analysis") for spin-up and as the precipitation reference: gridMET.

gridMET (Abatzoglou 2013) is a 4 km daily CONUS product blending PRISM and
NLDAS-2, updated with ~1 day latency and openly served as annual NetCDF files.
It supplies precipitation, Tmin/Tmax and reference ET. dCREST-CAMELS v1.0 was
calibrated on Daymet with Oudin PET, so by default PET is recomputed here with
the Oudin formula from gridMET mean temperature and gauge latitude.

Derived basin-mean series are stored as monthly parquet files under
``data/forcing/gridmet/YYYY-MM.parquet`` with columns ``P``, ``T``, ``PET``
(and ``PET_gridmet``), indexed by ``(date, gauge_id)``.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from .gridweights import check_grid, weighted_mean

log = logging.getLogger(__name__)

GRIDMET_URL = "https://www.northwestknowledge.net/metdata/data/{var}_{year}.nc"
VARS = {"pr": "precipitation_amount", "tmmn": "air_temperature", "tmmx": "air_temperature",
        "pet": "potential_evapotranspiration"}
GRID = "gridmet"


def gridmet_file(cache_dir: Path, var: str, year: int, refresh: bool = False) -> Path:
    """Local copy of one gridMET annual file (downloaded if missing or ``refresh``)."""
    import requests
    p = Path(cache_dir) / "gridmet" / f"{var}_{year}.nc"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not refresh:
        return p
    url = GRIDMET_URL.format(var=var, year=year)
    log.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=1800) as r:
        r.raise_for_status()
        tmp = p.with_suffix(".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
        tmp.replace(p)
    return p


def gridmet_axes(cache_dir: Path, year: Optional[int] = None):
    year = year or date.today().year
    with xr.open_dataset(gridmet_file(cache_dir, "pr", year)) as ds:
        return ds["lat"].values.astype(float), ds["lon"].values.astype(float)


def oudin_pet(tmean_c: np.ndarray, doy: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Oudin et al. (2005) PET [mm/day]. ``tmean_c`` (t, n), ``doy`` (t,), ``lat_deg`` (n,)."""
    lat = np.deg2rad(lat_deg)[None, :]
    d = doy[:, None].astype(float)
    dr = 1 + 0.033 * np.cos(2 * np.pi * d / 365)
    dec = 0.409 * np.sin(2 * np.pi * d / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(dec), -1, 1))
    Ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.sin(ws))
    pet = Ra / 2.45 * (tmean_c + 5.0) / 100.0   # Ra in MJ m-2 d-1, lambda = 2.45 MJ/kg -> mm/day
    return np.where(tmean_c + 5.0 > 0, pet, 0.0)


def store_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "forcing" / GRID


def _extract_year(cache_dir: Path, W, wlat, wlon, year: int, months: List[int], refresh_file: bool) -> Dict[int, pd.DataFrame]:
    files = {v: gridmet_file(cache_dir, v, year, refresh=refresh_file) for v in VARS}
    dss = {v: xr.open_dataset(files[v]) for v in VARS}
    lat, lon = dss["pr"]["lat"].values, dss["pr"]["lon"].values
    check_grid(wlat, wlon, lat, lon)
    days = pd.DatetimeIndex(dss["pr"]["day"].values)
    out = {}
    for m in months:
        sel = np.where(days.month == m)[0]
        if len(sel) == 0:
            continue
        sl = slice(int(sel[0]), int(sel[-1]) + 1)
        cols = {}
        for v, name in VARS.items():
            arr = dss[v][name].isel(day=sl).values.astype(np.float32)
            cols[v] = weighted_mean(W, arr.reshape(arr.shape[0], -1)).astype(np.float32)
        out[m] = (days[sl], cols)
    for ds in dss.values():
        ds.close()
    return out


def update(data_dir: Path, cache_dir: Path, basin_ids: List[str], basin_lat: np.ndarray, start: date,
           refresh_months: int = 3, pet: str = "oudin") -> pd.DataFrame:
    """Bring the stored basin-mean forcing up to date and return the whole record."""
    from .basins import load_grid_weights
    W, wlat, wlon, _ = load_grid_weights(data_dir, GRID, list(basin_ids))
    sd = store_dir(data_dir)
    sd.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().normalize()
    months = pd.period_range(pd.Timestamp(start).to_period("M"), today.to_period("M"), freq="M")
    refresh_from = (today - pd.DateOffset(months=refresh_months)).to_period("M")
    todo: Dict[int, List[int]] = {}
    for per in months:
        p = sd / f"{per}.parquet"
        if not p.exists() or per >= refresh_from:
            todo.setdefault(per.year, []).append(per.month)
    for year, ms in sorted(todo.items()):
        log.info("gridMET %d months %s", year, ms)
        res = _extract_year(cache_dir, W, wlat, wlon, year, ms, refresh_file=(year >= refresh_from.year))
        for m, (days, cols) in res.items():
            tmean = 0.5 * (cols["tmmn"] + cols["tmmx"]) - 273.15
            pet_oudin = oudin_pet(tmean, days.dayofyear.values, np.asarray(basin_lat, dtype=float))
            idx = pd.MultiIndex.from_product([days, basin_ids], names=["date", "gauge_id"])
            df = pd.DataFrame({
                "P": cols["pr"].ravel(), "T": tmean.ravel().astype(np.float32),
                "PET": (pet_oudin if pet == "oudin" else cols["pet"]).ravel().astype(np.float32),
                "PET_gridmet": cols["pet"].ravel(),
            }, index=idx)
            df.to_parquet(sd / f"{year}-{m:02d}.parquet")
    return load(data_dir)


def load(data_dir: Path) -> pd.DataFrame:
    """Full stored record, long format indexed by ``(date, gauge_id)``."""
    files = sorted(store_dir(data_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError("no gridMET basin forcing stored; run the analysis update first")
    return pd.concat([pd.read_parquet(f) for f in files]).sort_index()


def to_wide(df: pd.DataFrame, var: str, basin_ids: List[str]) -> pd.DataFrame:
    return df[var].unstack("gauge_id").reindex(columns=basin_ids)
