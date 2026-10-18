"""Build the CAMELS-US Daymet basin-mean daily record used by dCREST-CAMELS.

Source: CAMELS v1.2 (Newman et al. 2015; Addor et al. 2017), Zenodo record 15529996,
``basin_timeseries_v1p2_metForcing_obsFlow.zip`` (Daymet lumped forcing + USGS
streamflow) and the ``camels_*.txt`` attribute tables.

Output ``config.DATA_NPZ`` mirrors the layout of dCREST's ``forcing_daily.npz``:

    P        precipitation, mm/day                    (T, N) float32
    T        mean air temperature (tmax+tmin)/2, degC  (T, N)
    PET      Oudin potential ET, mm/day                (T, N)
    obs      USGS discharge, mm/day, NaN where missing (T, N)
    ids, lat, elev_m, area_km2, dates

Basin area is the forcing-file header area, the same value the released parameter
set carries. Check the build with ``reproduce_released()``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from . import config

CFS_TO_M3S = 0.028316846592


def _raw_root() -> Path:
    root = config.CAMELS_RAW / "basin_dataset_public_v1p2"
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} missing: download basin_timeseries_v1p2_metForcing_obsFlow.zip from "
            "https://zenodo.org/records/15529996 into CAMELS_RAW and unzip it")
    return root


def oudin_pet(tmean: np.ndarray, doy: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Oudin et al. (2005): PET = Re / (lambda rho) * (T + 5) / 100 when T > -5 degC [mm/day]."""
    lat = np.deg2rad(lat_deg)[None, :]
    d = doy[:, None].astype(float)
    dr = 1 + 0.033 * np.cos(2 * np.pi * d / 365)
    dec = 0.409 * np.sin(2 * np.pi * d / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(dec), -1, 1))
    re = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.sin(ws))
    return np.where(tmean + 5 > 0, re / 2.45 * (tmean + 5) / 100, 0.0)


def _read_forcing(path: Path):
    with open(path) as fh:
        lat = float(fh.readline())
        elev = float(fh.readline())
        area_m2 = float(fh.readline())
    df = pd.read_csv(path, sep=r"\s+", skiprows=3)
    dates = pd.to_datetime(dict(year=df.Year, month=df.Mnth, day=df.Day))
    df.index = dates
    return lat, elev, area_m2 / 1e6, df


def _read_flow(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["gauge", "y", "m", "d", "q", "flag"],
                     dtype={"gauge": str})
    s = pd.Series(df.q.values, index=pd.to_datetime(dict(year=df.y, month=df.m, day=df.d)))
    return s.where(s >= 0)


def build(ids: List[str], out: Path = config.DATA_NPZ) -> Path:
    root = _raw_root()
    forcing = {p.name[:8]: p for p in (root / "basin_mean_forcing" / "daymet").glob("*/*_lump_cida_forcing_leap.txt")}
    flows = {p.name[:8]: p for p in (root / "usgs_streamflow").glob("*/*_streamflow_qc.txt")}
    missing = [i for i in ids if i not in forcing or i not in flows]
    if missing:
        raise KeyError(f"{len(missing)} basins missing from CAMELS raw data: {missing[:5]}")
    dates = pd.date_range("1980-01-01", "2014-12-31", freq="D")
    T, N = len(dates), len(ids)
    P, TA, PET, OBS = (np.full((T, N), np.nan, np.float32) for _ in range(4))
    lat, elev, area = np.zeros(N), np.zeros(N), np.zeros(N)
    for j, g in enumerate(ids):
        lat[j], elev[j], area[j], df = _read_forcing(forcing[g])
        df = df[~df.index.duplicated()].reindex(dates)
        P[:, j] = df["prcp(mm/day)"].values
        TA[:, j] = 0.5 * (df["tmax(C)"].values + df["tmin(C)"].values)
        q = _read_flow(flows[g]).reindex(dates)
        OBS[:, j] = (q.values * CFS_TO_M3S * 86400.0 / (area[j] * 1e6) * 1000.0).astype(np.float32)
    PET[:] = oudin_pet(TA, dates.dayofyear.values, lat).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, P=P, T=TA, PET=PET, obs=OBS, ids=np.asarray(ids), lat=lat, elev_m=elev,
                        area_km2=area, dates=dates.values.astype("datetime64[D]"))
    return out


def load(path: Path = config.DATA_NPZ) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["ids"] = [str(i) for i in d["ids"]]
    d["dates"] = pd.DatetimeIndex(d["dates"])
    return d


def attributes(ids: List[str]) -> pd.DataFrame:
    """Static CAMELS attributes (topo, climate, soil, vegetation, geology) for ``ids``."""
    frames = []
    for f in config.ATTR_FILES:
        df = pd.read_csv(config.CAMELS_RAW / f, sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
        frames.append(df)
    a = pd.concat(frames, axis=1)
    a = a.loc[:, ~a.columns.duplicated()]
    return a.reindex(ids)


def reproduce_released(data: dict | None = None) -> pd.DataFrame:
    """Re-run the released parameters on the test period; compare with their stored test NSE."""
    import dcrest
    import torch

    from .simulate import run_period
    from .metrics import nse

    d = data or load()
    params, meta = dcrest.load_parameters(str(config.PARAMS), subset=d["ids"])
    cfg = dcrest.config_from_meta(meta)
    Q, obs = run_period(d, params, cfg, *config.TEST, spinup_from=config.TEST_SPINUP_FROM)
    got = nse(Q, obs)
    return pd.DataFrame({"nse": got, "stored": np.asarray(meta["metric_test_nse"], float)},
                        index=pd.Index(d["ids"], name="gauge_id"))
