"""NSE and KGE over the valid days of each column of ``(T, N)`` arrays."""
from __future__ import annotations

import numpy as np


def _mask(sim, obs):
    sim = np.asarray(sim, float)
    obs = np.asarray(obs, float)
    return sim, obs, np.isfinite(sim) & np.isfinite(obs)


def nse(sim, obs, min_n: int = 365) -> np.ndarray:
    sim, obs, m = _mask(sim, obs)
    out = np.full(sim.shape[1], np.nan)
    for j in range(sim.shape[1]):
        o, s = obs[m[:, j], j], sim[m[:, j], j]
        if len(o) >= min_n and np.var(o) > 0:
            out[j] = 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)
    return out


def kge(sim, obs, min_n: int = 365) -> np.ndarray:
    sim, obs, m = _mask(sim, obs)
    out = np.full(sim.shape[1], np.nan)
    for j in range(sim.shape[1]):
        o, s = obs[m[:, j], j], sim[m[:, j], j]
        if len(o) >= min_n and o.std() > 0 and o.mean() > 0 and s.std() > 0:
            r = np.corrcoef(s, o)[0, 1]
            out[j] = 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)
    return out
