"""Fractional-coverage weights between basin polygons and a regular lat/lon grid.

The weight matrix ``W`` has shape ``(n_basins, nlat * nlon)``; row ``b`` holds,
for every grid cell, the fraction of basin ``b`` that lies in that cell (rows
sum to 1). A basin-mean field is then ``W @ field.ravel()``. Cells are indexed
``j * nlon + i`` with ``j`` the latitude index of the *stored* grid, whatever
its direction, so the same code serves ECMWF (lat descending, lon from -180)
and gridMET (lat descending, lon ascending).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse


def _step(x: np.ndarray) -> float:
    d = np.diff(x)
    if not np.allclose(d, d[0], rtol=1e-4, atol=1e-6):
        raise ValueError("grid axis is not regular")
    return float(d[0])


def coverage_weights(geoms: Iterable, lat: np.ndarray, lon: np.ndarray) -> sparse.csr_matrix:
    """Exact polygon/cell intersection weights (area-weighted with cos(lat))."""
    import shapely

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    dlat, dlon = _step(lat), _step(lon)
    nlat, nlon = len(lat), len(lon)
    rows, cols, vals = [], [], []
    for b, geom in enumerate(geoms):
        minx, miny, maxx, maxy = geom.bounds
        # index ranges of candidate cells (works for ascending or descending axes)
        ia, ib = sorted(((minx - lon[0]) / dlon, (maxx - lon[0]) / dlon))
        ja, jb = sorted(((miny - lat[0]) / dlat, (maxy - lat[0]) / dlat))
        i_idx = np.arange(max(0, int(np.floor(ia + 0.5))), min(nlon - 1, int(np.floor(ib + 0.5))) + 1)
        j_idx = np.arange(max(0, int(np.floor(ja + 0.5))), min(nlat - 1, int(np.floor(jb + 0.5))) + 1)
        if len(i_idx) == 0 or len(j_idx) == 0:
            continue
        II, JJ = np.meshgrid(i_idx, j_idx)
        II, JJ = II.ravel(), JJ.ravel()
        cx, cy = lon[II], lat[JJ]
        boxes = shapely.box(cx - abs(dlon) / 2, cy - abs(dlat) / 2, cx + abs(dlon) / 2, cy + abs(dlat) / 2)
        inter = shapely.area(shapely.intersection(boxes, geom)) * np.cos(np.deg2rad(cy))
        keep = inter > 0
        if not keep.any():  # degenerate: fall back to the cell containing the centroid
            c = geom.centroid
            i = int(np.clip(np.floor((c.x - lon[0]) / dlon + 0.5), 0, nlon - 1))
            j = int(np.clip(np.floor((c.y - lat[0]) / dlat + 0.5), 0, nlat - 1))
            rows.append(b); cols.append(j * nlon + i); vals.append(1.0)
            continue
        w = inter[keep] / inter[keep].sum()
        rows.extend([b] * int(keep.sum()))
        cols.extend((JJ[keep] * nlon + II[keep]).tolist())
        vals.extend(w.tolist())
    n = b + 1
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, nlat * nlon))


def save_weights(path: Path, W: sparse.csr_matrix, lat: np.ndarray, lon: np.ndarray, ids: Iterable[str]) -> None:
    W = W.tocoo()
    np.savez_compressed(path, row=W.row.astype(np.int32), col=W.col.astype(np.int32),
                        val=W.data.astype(np.float32), shape=np.asarray(W.shape),
                        lat=np.asarray(lat, dtype=np.float64), lon=np.asarray(lon, dtype=np.float64),
                        ids=np.asarray(list(ids)))


def load_weights(path: Path):
    """Return ``(W, lat, lon, ids)``."""
    z = np.load(path, allow_pickle=False)
    W = sparse.csr_matrix((z["val"].astype(np.float64), (z["row"], z["col"])), shape=tuple(z["shape"]))
    return W, z["lat"], z["lon"], [str(s) for s in z["ids"]]


def check_grid(lat_file: np.ndarray, lon_file: np.ndarray, lat: np.ndarray, lon: np.ndarray, atol: float = 1e-3) -> None:
    if len(lat_file) != len(lat) or len(lon_file) != len(lon) or \
            not np.allclose(lat_file, lat, atol=atol) or not np.allclose(lon_file, lon, atol=atol):
        raise ValueError("grid of the field does not match the grid the weights were built on")


def weighted_mean(W: sparse.csr_matrix, X: np.ndarray) -> np.ndarray:
    """NaN-aware basin means. ``X`` is ``(t, ncell)``; returns ``(t, n_basins)``."""
    X = np.asarray(X, dtype=np.float64)
    M = np.isfinite(X)
    Xz = np.where(M, X, 0.0)
    num = (W @ Xz.T).T
    den = (W @ M.astype(np.float64).T).T
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 0.5] = np.nan  # less than half of the basin has valid data
    return out
