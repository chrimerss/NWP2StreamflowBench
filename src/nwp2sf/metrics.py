"""Column-wise skill scores with NaN handling. All inputs are ``(t, n)`` arrays."""
from __future__ import annotations

import numpy as np


def _prep(sim, obs, min_n):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    m = np.isfinite(sim) & np.isfinite(obs)
    n = m.sum(axis=0)
    s = np.where(m, sim, 0.0)
    o = np.where(m, obs, 0.0)
    ok = n >= min_n
    return s, o, m, n, ok


def nse(sim, obs, min_n: int = 1) -> np.ndarray:
    s, o, m, n, ok = _prep(sim, obs, min_n)
    with np.errstate(invalid="ignore", divide="ignore"):
        obar = o.sum(0) / n
        num = ((s - o) ** 2 * m).sum(0)
        den = ((o - obar) ** 2 * m).sum(0)
        out = 1.0 - num / den
    out[~ok | (den <= 0)] = np.nan
    return out


def kge(sim, obs, min_n: int = 1) -> np.ndarray:
    s, o, m, n, ok = _prep(sim, obs, min_n)
    with np.errstate(invalid="ignore", divide="ignore"):
        sbar, obar = s.sum(0) / n, o.sum(0) / n
        sd = np.sqrt((((s - sbar) ** 2) * m).sum(0) / n)
        od = np.sqrt((((o - obar) ** 2) * m).sum(0) / n)
        r = (((s - sbar) * (o - obar)) * m).sum(0) / n / (sd * od)
        alpha, beta = sd / od, sbar / obar
        out = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    out[~ok | (od <= 0) | (obar <= 0)] = np.nan
    return out


def pearson(sim, obs, min_n: int = 1) -> np.ndarray:
    s, o, m, n, ok = _prep(sim, obs, min_n)
    with np.errstate(invalid="ignore", divide="ignore"):
        sbar, obar = s.sum(0) / n, o.sum(0) / n
        sd = np.sqrt((((s - sbar) ** 2) * m).sum(0))
        od = np.sqrt((((o - obar) ** 2) * m).sum(0))
        out = (((s - sbar) * (o - obar)) * m).sum(0) / (sd * od)
    out[~ok | (sd <= 0) | (od <= 0)] = np.nan
    return out


def bias_ratio(sim, obs, min_n: int = 1) -> np.ndarray:
    """mean(sim)/mean(obs) over jointly valid days."""
    s, o, m, n, ok = _prep(sim, obs, min_n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s.sum(0) / o.sum(0)
    out[~ok | (o.sum(0) <= 0)] = np.nan
    return out


def rmse(sim, obs, min_n: int = 1) -> np.ndarray:
    s, o, m, n, ok = _prep(sim, obs, min_n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sqrt(((s - o) ** 2 * m).sum(0) / n)
    out[~ok] = np.nan
    return out


def valid_count(sim, obs) -> np.ndarray:
    return (np.isfinite(np.asarray(sim, float)) & np.isfinite(np.asarray(obs, float))).sum(axis=0)
