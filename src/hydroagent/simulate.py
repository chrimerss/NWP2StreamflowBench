"""Run dCREST over a scoring window with spin-up, the way the released sets were scored."""
from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np
import pandas as pd
import torch

import dcrest


def window(dates: pd.DatetimeIndex, start: str, end: str) -> slice:
    i0 = int(dates.searchsorted(pd.Timestamp(start)))
    i1 = int(dates.searchsorted(pd.Timestamp(end), side="right"))
    return slice(i0, i1)


def run_period(data: Mapping, params, cfg, start: str, end: str, *, spinup_from: str,
               cols=None) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate from ``spinup_from`` to ``end`` and return (Q, obs) over ``[start, end]``."""
    dates = data["dates"]
    run = slice(window(dates, spinup_from, end).start, window(dates, start, end).stop)
    score = window(dates, start, end)
    sl = slice(None) if cols is None else cols
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a[run][:, sl]), dtype=torch.float32)  # noqa: E731
    with torch.no_grad():
        Q = dcrest.simulate(t(data["P"]), t(data["T"]), t(data["PET"]), params, cfg).numpy()
    k0 = score.start - run.start
    return Q[k0:], np.asarray(data["obs"][score][:, sl], dtype=np.float32)


def full_record(data: Mapping, params, cfg, cols=None) -> np.ndarray:
    """Discharge for the whole 1980-2014 record (first years act as spin-up)."""
    sl = slice(None) if cols is None else cols
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a[:, sl]), dtype=torch.float32)  # noqa: E731
    with torch.no_grad():
        return dcrest.simulate(t(data["P"]), t(data["T"]), t(data["PET"]), params, cfg).numpy()


def subset_params(params: Mapping[str, torch.Tensor], idx) -> dict:
    idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
    return {k: v[idx] for k, v in params.items()}
