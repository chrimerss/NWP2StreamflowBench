"""Reading and writing parameter sets."""
from __future__ import annotations

import os
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from . import params as _params

Tensor = torch.Tensor
FORMAT_VERSION = 1


def save_parameters(path: str, params: Mapping[str, Tensor], ids: Sequence[str],
                    *, config=None, bounds=None, metrics: Optional[Mapping] = None,
                    provenance: Optional[Mapping[str, str]] = None) -> str:
    """Write a parameter set to ``.npz`` with enough metadata to reload it safely.

    The stored ``uh_kmin_h`` matters: a set calibrated under one UH clamp gives
    different discharge under another, so :func:`load_parameters` refuses to
    guess and returns the stored config alongside the parameters.
    """
    b = _params.BOUNDS if bounds is None else dict(bounds)
    arrays = {k: np.asarray(_to_np(params[k]), dtype=np.float32) for k in b}
    arrays["area_km2"] = np.asarray(_to_np(params["area_km2"]), dtype=np.float32)
    meta = {
        "ids": np.asarray([str(i) for i in ids]),
        "pkeys": np.asarray(list(b)),
        "bounds": np.asarray([[b[k][0], b[k][1]] for k in b], dtype=np.float64),
        "format_version": np.asarray(FORMAT_VERSION),
    }
    if config is not None:
        meta["dt_hours"] = np.asarray(float(config.dt_hours))
        meta["uh_nstep"] = np.asarray(int(config.uh_nstep))
        meta["uh_kmin_h"] = np.asarray(
            np.nan if config.uh_kmin_h is None else float(config.uh_kmin_h))
    for k, v in (metrics or {}).items():
        meta[f"metric_{k}"] = np.asarray(v)
    for k, v in (provenance or {}).items():
        meta[f"prov_{k}"] = np.asarray(str(v))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **arrays, **meta)
    return path


def load_parameters(path: str, *, device="cpu", dtype=torch.float32,
                    subset: Optional[Sequence[str]] = None):
    """Load a parameter set. Returns ``(params, meta)``.

    ``meta`` carries ``ids``, the bounds the set was calibrated under, and the
    UH clamp. Build a :class:`~dcrest.model.CrestConfig` from
    ``meta['uh_kmin_h']`` rather than assuming the default, or the reloaded set
    will not reproduce its own calibration.

    ``subset`` selects and *orders* units by id.
    """
    z = np.load(path, allow_pickle=True)
    keys = [str(k) for k in z["pkeys"]] if "pkeys" in z.files \
        else list(_params.BOUNDS)
    ids = np.asarray([str(i) for i in z["ids"]])
    idx = slice(None)
    if subset is not None:
        pos = {b: i for i, b in enumerate(ids)}
        miss = [b for b in subset if b not in pos]
        if miss:
            raise KeyError(f"{len(miss)} requested ids not in {path}: {miss[:5]}")
        idx = np.asarray([pos[b] for b in subset])
        ids = ids[idx]
    out: Dict[str, Tensor] = {
        k: torch.as_tensor(np.asarray(z[k])[idx], dtype=dtype, device=device)
        for k in keys}
    out["area_km2"] = torch.as_tensor(np.asarray(z["area_km2"])[idx],
                                      dtype=dtype, device=device)
    meta = {"ids": ids, "path": path}
    if "bounds" in z.files:
        meta["bounds"] = {k: tuple(map(float, r))
                          for k, r in zip(keys, np.asarray(z["bounds"]))}
    for k in ("dt_hours", "uh_nstep", "uh_kmin_h", "format_version"):
        if k in z.files:
            v = float(np.asarray(z[k]))
            meta[k] = None if (k == "uh_kmin_h" and np.isnan(v)) else v
    n_all = len(np.asarray(z["ids"]))
    for k in z.files:
        if k.startswith(("metric_", "prov_")):
            v = np.asarray(z[k])
            # per-unit diagnostics follow the subset; provenance scalars do not
            meta[k] = v[idx] if (v.ndim == 1 and len(v) == n_all) else v
    return out, meta


def config_from_meta(meta: Mapping, **overrides):
    """Build the :class:`~dcrest.model.CrestConfig` a stored set was made with."""
    from .model import CrestConfig
    kw = {}
    if "dt_hours" in meta:
        kw["dt_hours"] = float(meta["dt_hours"])
    if "uh_nstep" in meta:
        kw["uh_nstep"] = int(meta["uh_nstep"])
    if "uh_kmin_h" in meta:
        kw["uh_kmin_h"] = meta["uh_kmin_h"]
    if meta.get("bounds"):
        kw["bounds"] = meta["bounds"]
    kw.update(overrides)
    return CrestConfig(**kw)


def to_dataframe(params: Mapping[str, Tensor], ids: Sequence[str]):
    """Parameter set as a tidy ``pandas.DataFrame`` (one row per unit)."""
    import pandas as pd
    cols = {"id": [str(i) for i in ids]}
    cols.update({k: _to_np(params[k]) for k in _params.BOUNDS})
    cols["area_km2"] = _to_np(params["area_km2"])
    return pd.DataFrame(cols)


def _to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


__all__ = ["save_parameters", "load_parameters", "config_from_meta",
           "to_dataframe", "FORMAT_VERSION"]
