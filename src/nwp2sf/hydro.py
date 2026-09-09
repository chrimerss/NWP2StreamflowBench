"""Drive dCREST with analysis forcing and with NWP precipitation substituted over the forecast window.

Design ("perfect initial state, imperfect forcing"): every forecast starts from
the state the analysis-forced run has at the initialisation date. Only
precipitation is replaced during the lead window; temperature and PET stay at
their analysis values so that the score isolates precipitation error. Because
the model is cheap on CPU (a 665-basin, 3.5-year daily run takes ~0.25 s), each
forecast is realised by re-running the whole record with the forecast days
spliced in, which avoids serialising model states. Several initialisations are
batched along the basin axis in one call.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import dcrest

log = logging.getLogger(__name__)


def load_model(param_npz, basin_ids: Sequence[str]):
    params, meta = dcrest.load_parameters(str(param_npz), subset=list(basin_ids))
    cfg = dcrest.config_from_meta(meta)
    return params, cfg, meta


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32))


def simulate(P: np.ndarray, T: np.ndarray, E: np.ndarray, params, cfg) -> np.ndarray:
    with torch.no_grad():
        Q = dcrest.simulate(_t(P), _t(T), _t(E), params, cfg)
    return Q.numpy()


def extend_forcing(dates: pd.DatetimeIndex, X: np.ndarray, until: pd.Timestamp, how: str = "doy") -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """Pad ``X`` (t, n) past the analysis end with day-of-year climatology (or zeros)."""
    if until <= dates[-1]:
        return dates, X
    new = pd.date_range(dates[-1] + pd.Timedelta(days=1), until, freq="D")
    if how == "zeros":
        pad = np.zeros((len(new), X.shape[1]), dtype=X.dtype)
    else:
        doy = dates.dayofyear.values
        clim = np.stack([np.nanmean(X[doy == d], axis=0) if (doy == d).any() else np.full(X.shape[1], np.nan)
                         for d in range(1, 367)])
        # smooth the climatology with a +-7 day window to tame sampling noise
        k = 7
        padc = np.concatenate([clim[-k:], clim, clim[:k]])
        clim = np.stack([np.nanmean(padc[i:i + 2 * k + 1], axis=0) for i in range(366)])
        pad = clim[new.dayofyear.values - 1]
        pad = np.where(np.isfinite(pad), pad, np.nanmean(X, axis=0))
    return dates.append(new), np.concatenate([X, pad.astype(X.dtype)])


def run_forecasts(dates: pd.DatetimeIndex, P: np.ndarray, T: np.ndarray, E: np.ndarray, params, cfg,
                  forecasts: Dict[date, np.ndarray], leads: int, lag: int = 0, batch: int = 16) -> Tuple[List[date], np.ndarray]:
    """Return ``(inits, Q)`` with ``Q`` of shape ``(n_init, leads, n_basins)`` in mm/day.

    ``forecasts[init]`` is ``(leads, n_basins)`` daily precipitation for lead days 1..L.
    ``dates`` is the model timeline; precipitation of lead day ``l`` (which falls on
    ``init + l - 1``) enters the model on ``init + l - 1 + lag`` and ``Q[i, l-1]`` is the
    discharge of that model day.
    """
    n = P.shape[1]
    inits = sorted(d for d in forecasts if pd.Timestamp(d) + pd.Timedelta(days=lag) in dates)
    pos = {d: int(dates.get_loc(pd.Timestamp(d) + pd.Timedelta(days=lag))) for d in inits}
    Q = np.full((len(inits), leads, n), np.nan, dtype=np.float32)
    keys = list(params.keys())
    for b0 in range(0, len(inits), batch):
        chunk = inits[b0:b0 + batch]
        k = len(chunk)
        t_end = max(pos[d] for d in chunk) + leads
        t_end = min(t_end, len(dates))
        Pb = np.tile(P[:t_end], (1, k)).astype(np.float32)
        Tb = np.tile(T[:t_end], (1, k))
        Eb = np.tile(E[:t_end], (1, k))
        for j, d in enumerate(chunk):
            t0 = pos[d]
            L = min(leads, t_end - t0)
            fc = forecasts[d][:L]
            fc = np.where(np.isfinite(fc), fc, P[t0:t0 + L])  # missing lead -> analysis (rare)
            Pb[t0:t0 + L, j * n:(j + 1) * n] = fc
        pk = {key: torch.tile(params[key], (k,)) for key in keys}
        Qb = simulate(Pb, Tb, Eb, pk, cfg)
        for j, d in enumerate(chunk):
            t0 = pos[d]
            L = min(leads, t_end - t0)
            Q[b0 + j, :L] = Qb[t0:t0 + L, j * n:(j + 1) * n]
        log.info("forecast runs %d/%d", min(b0 + batch, len(inits)), len(inits))
    return inits, Q
