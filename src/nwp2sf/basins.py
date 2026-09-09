"""Basin set: CAMELS-US catchments carried by dCREST-CAMELS, with GAGES-II polygons.

Two artefacts are committed under ``data/basins`` so that routine runs (and CI)
never need the 200 MB GAGES-II shapefile archive:

* ``basins.csv`` — one row per gauge: id, name, lat, lon, state, climate region,
  HUC02, drainage area (km2, CAMELS value used by the parameter set).
* ``weights_<grid>.npz`` — sparse basin/grid coverage weights per source grid.
* ``excluded.csv`` — gauges dropped from the benchmark (e.g. inactive gauges with
  no observations in the benchmark period); ``load_basins`` removes them and
  weight rows are subset accordingly.

``build`` regenerates both from the GAGES-II shapefiles; ``load_basins`` and
``load_grid_weights`` read them back.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .gridweights import coverage_weights, load_weights, save_weights
from .regions import region_of_state

GAGESII_BOUNDARIES = ("https://www.sciencebase.gov/catalog/file/get/631405bbd34e36012efa304a"
                      "?f=__disk__03%2Fe7%2F12%2F03e7123a54d9a7e87587595e09da7d093cbbf98d")
GAGESII_POINTS = ("https://www.sciencebase.gov/catalog/file/get/631405bbd34e36012efa304a"
                  "?f=__disk__53%2Fba%2F17%2F53ba17cfb64d2d50d3876864edf519eb7295945a")

# Source grids on which basin means are needed. Each entry gives the 1-D cell
# centre axes exactly as the data files store them.
GRIDS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    # ECMWF open data, 0.25 deg regular lat/lon: lat 90..-90, lon -180..179.75
    "ecmwf0p25": (np.round(90.0 - 0.25 * np.arange(721), 6), np.round(-180.0 + 0.25 * np.arange(1440), 6)),
}


def basins_csv(data_dir: Path) -> Path:
    return Path(data_dir) / "basins" / "basins.csv"


def weights_path(data_dir: Path, grid: str) -> Path:
    return Path(data_dir) / "basins" / f"weights_{grid}.npz"


def excluded_csv(data_dir: Path) -> Path:
    return Path(data_dir) / "basins" / "excluded.csv"


def load_basins(data_dir: Path, include_excluded: bool = False) -> pd.DataFrame:
    """Benchmark basin set: ``basins.csv`` minus the gauges listed in ``excluded.csv``."""
    df = pd.read_csv(basins_csv(data_dir), dtype={"gauge_id": str, "huc02": str})
    ex = excluded_csv(data_dir)
    if ex.exists() and not include_excluded:
        drop = set(pd.read_csv(ex, dtype={"gauge_id": str}).gauge_id)
        df = df[~df.gauge_id.isin(drop)]
    return df.set_index("gauge_id", drop=False)


def load_grid_weights(data_dir: Path, grid: str, ids: Optional[List[str]] = None):
    """``(W, lat, lon, ids)`` for ``grid``; rows subset and ordered to ``ids`` when given."""
    W, lat, lon, wids = load_weights(weights_path(data_dir, grid))
    if ids is not None:
        pos = {b: i for i, b in enumerate(wids)}
        miss = [b for b in ids if b not in pos]
        if miss:
            raise KeyError(f"{len(miss)} basins missing from weights_{grid}.npz: {miss[:5]}")
        W = W[[pos[b] for b in ids]]
        wids = list(ids)
    return W, lat, lon, wids


def parameter_ids(param_npz: Path) -> List[str]:
    z = np.load(param_npz, allow_pickle=True)
    return [str(i) for i in z["ids"]]


def _download(url: str, dest: Path) -> Path:
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with requests.get(url, stream=True, timeout=900) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    return dest


def fetch_gagesii(cache_dir: Path) -> Path:
    """Download and unpack the GAGES-II shapefiles into ``cache_dir/gagesii``."""
    import zipfile
    root = Path(cache_dir) / "gagesii"
    root.mkdir(parents=True, exist_ok=True)
    for url, name in ((GAGESII_BOUNDARIES, "boundaries_shapefiles_by_aggeco.zip"),
                      (GAGESII_POINTS, "gagesII_9322_point_shapefile.zip")):
        z = _download(url, root / name)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)
    return root


def build_grid_weights(data_dir: Path, cache_dir: Path, grid: str, lat: np.ndarray, lon: np.ndarray):
    """Build (and save) the weight file for one grid from the GAGES-II polygons. Needs geopandas.

    Used by sources whose grid is only known once their data are opened (e.g.
    WeatherNext 3). Returns ``(W, ids)`` in ``basins.csv`` order (all gauges,
    excluded ones included, so the file stays valid if the exclusion list changes).
    """
    import geopandas as gpd

    root = fetch_gagesii(cache_dir)
    meta = pd.read_csv(basins_csv(data_dir), dtype={"gauge_id": str})
    ids = list(meta.gauge_id)
    shp = root / "boundaries-shapefiles-by-aggeco"
    polys = pd.concat([gpd.read_file(f) for f in sorted(shp.glob("bas_*.shp"))])
    polys = polys.drop_duplicates("GAGE_ID").set_index("GAGE_ID").loc[ids].to_crs(4326)
    lon = np.sort(np.asarray(lon, dtype=float))
    W = coverage_weights(polys.geometry.values, lat, lon)
    save_weights(weights_path(data_dir, grid), W, lat, lon, ids)
    return W, ids


def build(data_dir: Path, cache_dir: Path, param_npz: Path, grids: Dict[str, Tuple[np.ndarray, np.ndarray]] | None = None,
          extra_grids: Dict[str, Tuple[np.ndarray, np.ndarray]] | None = None) -> pd.DataFrame:
    """Create ``basins.csv`` and one weight file per grid. Needs geopandas."""
    import geopandas as gpd

    grids = dict(GRIDS if grids is None else grids)
    grids.update(extra_grids or {})
    root = fetch_gagesii(cache_dir)
    ids = parameter_ids(param_npz)
    z = np.load(param_npz, allow_pickle=True)
    area = pd.Series(np.asarray(z["area_km2"], dtype=float), index=ids)

    pts = gpd.read_file(root / "gagesII_9322_sept30_2011.shp").set_index("STAID")
    shp = root / "boundaries-shapefiles-by-aggeco"
    polys = pd.concat([gpd.read_file(f) for f in sorted(shp.glob("bas_*.shp"))])
    polys = polys.drop_duplicates("GAGE_ID").set_index("GAGE_ID")
    missing = [i for i in ids if i not in polys.index]
    if missing:
        raise KeyError(f"{len(missing)} basins have no GAGES-II polygon: {missing[:10]}")
    polys = polys.loc[ids].to_crs(4326)

    meta = pd.DataFrame({
        "gauge_id": ids,
        "name": pts.loc[ids, "STANAME"].values,
        "lat": pts.loc[ids, "LAT_GAGE"].astype(float).values,
        "lon": pts.loc[ids, "LNG_GAGE"].astype(float).values,
        "state": pts.loc[ids, "STATE"].values,
        "huc02": pts.loc[ids, "HUC02"].astype(str).str.zfill(2).values,
        "area_km2": area.loc[ids].values,
        "area_gagesii_km2": pts.loc[ids, "DRAIN_SQKM"].astype(float).values,
    })
    meta["region"] = [region_of_state(s) for s in meta.state]
    out = basins_csv(data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out, index=False, float_format="%.6f")

    for grid, (lat, lon) in grids.items():
        W = coverage_weights(polys.geometry.values, lat, lon)
        save_weights(weights_path(data_dir, grid), W, lat, lon, ids)
    return meta
