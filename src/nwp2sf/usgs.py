"""USGS daily mean discharge from the Water Data OGC API (``api.waterdata.usgs.gov``).

Values are converted from ft3/s to mm/day over the CAMELS basin area so that
they compare directly with dCREST output. Stored monthly under
``data/obs/usgs/YYYY-MM.parquet`` (wide: date x gauge_id, mm/day); the most
recent months are re-fetched each run because provisional data are revised.
An optional API key (``API_USGS_PAT`` environment variable) raises rate limits.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

API = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"
CFS_TO_M3S = 0.028316846592


def store_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "obs" / "usgs"


def _get(params: dict, retries: int = 5) -> dict:
    headers = {"Accept": "application/geo+json"}
    key = os.environ.get("API_USGS_PAT")
    if key:
        headers["X-Api-Key"] = key
    for k in range(retries):
        r = requests.get(API, params=params, headers=headers, timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            wait = 5 * (k + 1)
            log.warning("USGS API %s; retrying in %ds", r.status_code, wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
    raise RuntimeError("USGS API: too many retries")


def fetch_daily_cfs(site_ids: List[str], start: date, end: date, chunk: int = 20) -> pd.DataFrame:
    """Daily mean discharge (statistic 00003, parameter 00060) in ft3/s, wide date x site."""
    frames = []
    for i in range(0, len(site_ids), chunk):
        ids = ",".join(f"USGS-{s}" for s in site_ids[i:i + chunk])
        params = {"monitoring_location_id": ids, "parameter_code": "00060", "statistic_id": "00003",
                  "time": f"{start:%Y-%m-%d}/{end:%Y-%m-%d}", "f": "json", "limit": 10000}
        while True:
            js = _get(params)
            rows = [(f["properties"]["time"], f["properties"]["monitoring_location_id"][5:],
                     f["properties"]["value"]) for f in js.get("features", [])]
            if rows:
                frames.append(pd.DataFrame(rows, columns=["date", "gauge_id", "cfs"]))
            nxt = [l for l in js.get("links", []) if l.get("rel") == "next"]
            if not nxt:
                break
            params = None  # the next link carries the query
            js_url = nxt[0]["href"]
            r = requests.get(js_url, timeout=120)
            r.raise_for_status()
            js = r.json()
            rows = [(f["properties"]["time"], f["properties"]["monitoring_location_id"][5:],
                     f["properties"]["value"]) for f in js.get("features", [])]
            if rows:
                frames.append(pd.DataFrame(rows, columns=["date", "gauge_id", "cfs"]))
            nxt = [l for l in js.get("links", []) if l.get("rel") == "next"]
            while nxt:
                r = requests.get(nxt[0]["href"], timeout=120)
                r.raise_for_status()
                js = r.json()
                rows = [(f["properties"]["time"], f["properties"]["monitoring_location_id"][5:],
                         f["properties"]["value"]) for f in js.get("features", [])]
                if rows:
                    frames.append(pd.DataFrame(rows, columns=["date", "gauge_id", "cfs"]))
                nxt = [l for l in js.get("links", []) if l.get("rel") == "next"]
            break
        log.info("USGS: %d/%d sites", min(i + chunk, len(site_ids)), len(site_ids))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df["date"] = pd.to_datetime(df["date"])
    df["cfs"] = pd.to_numeric(df["cfs"], errors="coerce")
    df = df.dropna(subset=["cfs"])
    df = df[df.cfs >= 0]
    return df.groupby(["date", "gauge_id"])["cfs"].mean().unstack("gauge_id")


def cfs_to_mm_day(q_cfs: pd.DataFrame, area_km2: pd.Series) -> pd.DataFrame:
    a = area_km2.reindex(q_cfs.columns).values
    return q_cfs * CFS_TO_M3S * 86.4 / a


def update(data_dir: Path, basin_ids: List[str], area_km2: pd.Series, start: date, refresh_months: int = 3) -> pd.DataFrame:
    sd = store_dir(data_dir)
    sd.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().normalize()
    months = pd.period_range(pd.Timestamp(start).to_period("M"), today.to_period("M"), freq="M")
    refresh_from = (today - pd.DateOffset(months=refresh_months)).to_period("M")
    todo = [per for per in months if not (sd / f"{per}.parquet").exists() or per >= refresh_from]
    if todo:
        s, e = todo[0].start_time.date(), min(todo[-1].end_time.date(), today.date())
        log.info("USGS daily discharge %s..%s for %d sites", s, e, len(basin_ids))
        cfs = fetch_daily_cfs(basin_ids, s, e)
        mm = cfs_to_mm_day(cfs, area_km2).reindex(columns=basin_ids).astype(np.float32)
        for per in todo:
            sel = mm[(mm.index >= per.start_time) & (mm.index <= per.end_time)]
            if len(sel) or (sd / f"{per}.parquet").exists():
                sel.to_parquet(sd / f"{per}.parquet")
    return load(data_dir, basin_ids)


def load(data_dir: Path, basin_ids: List[str]) -> pd.DataFrame:
    files = sorted(store_dir(data_dir).glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=basin_ids)
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    df.index.name = "date"
    return df.reindex(columns=basin_ids)
