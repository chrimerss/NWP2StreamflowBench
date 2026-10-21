"""Build the agent-facing task folder for one ungauged-prediction task.

What the agent gets (``config.TASKS/<task_id>/template``):

    TASK.md                      the assignment
    task_folder/data/
        donors.npz               P, T, PET 1980-2014; Q_obs only inside 1999-10-01..2008-09-30
        targets.npz              P, T, PET 1980-2014; no discharge at all
        donor_attributes.csv     CAMELS static attributes (no names, HUC or exact coordinates)
        target_attributes.csv
        donor_params_released.csv  dCREST-CAMELS v1.0 parameters of the donors
        meta.json                windows, parameter names and bounds, UH settings
    dcrest/                      editable copy of the model package
    output/                      deliverables go here

What stays hidden (``config.HIDDEN/<task_id>``): the gauge ids behind the anonymous
T###/D#### labels. Target discharge is never written anywhere the agent can read;
the scorer reads it from the CAMELS record.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import dcrest

from . import config
from .protocol import Task

TASK_MD = """# Predict daily streamflow in {n_targets} ungauged catchments

You are a hydrologist. {n_donors} gauged *donor* catchments and {n_targets} *target*
catchments across the conterminous United States are described in `task_folder/data/`.
Targets have meteorological forcing and static attributes but **no streamflow record**.

Produce daily discharge for every target using the dCREST rainfall-runoff model
(`./dcrest`, a differentiable PyTorch implementation of CREST). The released
dCREST-CAMELS v1.0 parameters of every donor are provided; they were calibrated on
the donor's own observations over {obs0}..{obs1}. You may recalibrate donors, design
any parameter-transfer or regionalization scheme, ensemble several, and change the
model structure in `./dcrest` if your donor-side evidence supports it.

Your predictions will be compared with observed discharge over {pred0}..{pred1}, a
period in which no discharge is given for any catchment. Use donor-side
cross-validation (treat some donors as if ungauged, within {obs0}..{obs1}) to choose
between approaches.

## Data (`task_folder/data/`)
- `donors.npz`: `ids`, `dates` (daily 1980-01-01..2014-12-31), `P` precipitation
  [mm/day], `T` mean air temperature [degC], `PET` Oudin potential ET [mm/day],
  `Q_obs` discharge [mm/day] (NaN outside {obs0}..{obs1} and where missing),
  `area_km2`, `lat`, `lon` (rounded to 0.1 deg). Arrays are (days, basins).
- `targets.npz`: the same without `Q_obs`.
- `donor_attributes.csv`, `target_attributes.csv`: CAMELS topography, climate, soil,
  vegetation and geology attributes.
- `donor_params_released.csv`: the 16 dCREST parameters per donor (+ `area_km2`).
- `meta.json`: windows, parameter names and bounds, routing settings.

## Deliverables (`output/`)
1. `output/target_discharge.csv` - column `date` plus one column per target id
   ({first_target}..{last_target}); daily discharge in mm/day, non-negative, covering at
   least {pred0}..{pred1}.
2. `output/method.md` - the approach, the donor-side validation design and its
   numbers, and what you would try next.
3. `output/code/` - the scripts that regenerate `target_discharge.csv` from the data.

## Model spin-up
Start every simulation from dcrest's default initial states three years before the
window you use: {spin_pred} for the prediction window, {spin_obs} for the observation
window. The released parameters were calibrated that way; much longer spin-ups move
slow aquifer states and can badly degrade some catchments. `scripts/hydrokit.py`
follows this rule by default.

Write a first valid `target_discharge.csv` early, then improve it.
"""


def _anon_attrs(attrs: pd.DataFrame, ids, labels) -> pd.DataFrame:
    a = attrs.loc[list(ids)].copy()
    a["lat"] = a.pop("gauge_lat").round(1)
    a["lon"] = a.pop("gauge_lon").round(1)
    a = a.drop(columns=[c for c in config.DROP_ATTRS if c in a.columns])
    a.index = pd.Index([labels[g] for g in ids], name="id")
    return a


TASK_MD_CALIB = """# Calibrate dCREST on gauged catchments and predict {n_targets} out-of-distribution catchments

You are a hydrologist. `task_folder/data/` describes {n_donors} *training* catchments with
streamflow observations and {n_targets} *target* catchments with meteorological forcing and
static attributes but **no streamflow record**. {split_note}

Calibrate the dCREST rainfall-runoff model (`./dcrest`, a differentiable PyTorch
implementation of CREST) on the training catchments over {obs0}..{obs1}, and use what you
learn to produce daily discharge for every target. No pre-calibrated parameters are given:
the calibration is yours. You may calibrate catchment by catchment, jointly across
catchments, learn a mapping from attributes to parameters, ensemble, and change the model
structure in `./dcrest`, as long as your choices are supported by evidence you generate
from the training catchments (for example by holding some of them out).

Your predictions will be compared with observed discharge over {pred0}..{pred1}, a period
in which no discharge is given for any catchment.

## Data (`task_folder/data/`)
- `donors.npz` (the training catchments): `ids`, `dates` (daily 1980-01-01..2014-12-31),
  `P` precipitation [mm/day], `T` mean air temperature [degC], `PET` Oudin potential ET
  [mm/day], `Q_obs` discharge [mm/day] (NaN outside {obs0}..{obs1} and where missing),
  `area_km2`, `lat`, `lon` (rounded to 0.1 deg). Arrays are (days, catchments).
- `targets.npz`: the same without `Q_obs`.
- `donor_attributes.csv`, `target_attributes.csv`: CAMELS topography, climate, soil,
  vegetation and geology attributes.
- `meta.json`: windows, parameter names and bounds, routing settings.

## Deliverables (`output/`)
1. `output/target_discharge.csv` - column `date` plus one column per target id
   ({first_target}..{last_target}); daily discharge in mm/day, non-negative, covering at
   least {pred0}..{pred1}.
2. `output/method.md` - calibration and transfer approach, the hold-out design and its
   numbers, and what you would try next.
3. `output/code/` - the scripts that regenerate `target_discharge.csv` from the data.

## Model spin-up
Start every simulation from dcrest's default initial states three years before the
window you use: {spin_pred} for the prediction window, {spin_obs} for the observation
window. Much longer spin-ups move slow aquifer states and can badly degrade some
catchments. `scripts/hydrokit.py` follows this rule by default.

## Compute
CPU only. One gradient step of dCREST over the 12-year calibration window costs about
4 s whatever the number of catchments, so calibrate many catchments in one batch and run
long calibrations in the background with a log. Write a first valid
`target_discharge.csv` early, then improve it.
"""


def build(task: Task, data: dict, attrs: pd.DataFrame, force: bool = False, released: bool = True,
          variant: str = "template") -> Path:
    """Write the task folder and the hidden mapping; return its path.

    ``released=False`` builds the calibration variant: no pre-calibrated parameters and
    the TASK_MD_CALIB assignment.
    """
    root = config.TASKS / task.task_id / variant
    if root.exists() and not force:
        return root
    shutil.rmtree(root, ignore_errors=True)
    dd = root / "task_folder" / "data"
    dd.mkdir(parents=True)
    (root / "output").mkdir()
    labels = task.anon()
    pos = {g: i for i, g in enumerate(data["ids"])}
    dates = data["dates"]
    obs_mask = (dates >= pd.Timestamp(config.TRAIN[0])) & (dates <= pd.Timestamp(config.TRAIN[1]))

    def pack(ids, with_obs):
        cols = [pos[g] for g in ids]
        out = {"ids": np.asarray([labels[g] for g in ids]), "dates": dates.values.astype("datetime64[D]"),
               "P": data["P"][:, cols], "T": data["T"][:, cols], "PET": data["PET"][:, cols],
               "area_km2": np.asarray(data["area_km2"])[cols].astype(np.float32),
               "lat": attrs.loc[list(ids), "gauge_lat"].round(1).values,
               "lon": attrs.loc[list(ids), "gauge_lon"].round(1).values}
        if with_obs:
            q = data["obs"][:, cols].copy()
            q[~obs_mask] = np.nan
            out["Q_obs"] = q
        return out

    np.savez(dd / "donors.npz", **pack(task.donors, True))
    np.savez(dd / "targets.npz", **pack(task.targets, False))
    _anon_attrs(attrs, task.donors, labels).to_csv(dd / "donor_attributes.csv", float_format="%.6g")
    _anon_attrs(attrs, task.targets, labels).to_csv(dd / "target_attributes.csv", float_format="%.6g")

    params, meta = dcrest.load_parameters(str(config.PARAMS), subset=list(task.donors))
    keys = [k for k in params if k != "area_km2"]
    if released:
        rel = pd.DataFrame({k: params[k].numpy() for k in keys + ["area_km2"]},
                           index=pd.Index([labels[g] for g in task.donors], name="id"))
        rel.to_csv(dd / "donor_params_released.csv", float_format="%.6g")
    (dd / "meta.json").write_text(json.dumps({
        "obs_window": list(config.TRAIN), "spinup_from": config.TRAIN_SPINUP_FROM,
        "predict_window": list(config.TEST), "predict_spinup_from": config.TEST_SPINUP_FROM,
        "param_names": keys, "bounds": {k: list(v) for k, v in meta["bounds"].items()},
        "uh_kmin_h": meta.get("uh_kmin_h"), "uh_nstep": int(meta.get("uh_nstep", 30)),
        "units": {"P": "mm/day", "T": "degC", "PET": "mm/day", "Q_obs": "mm/day"},
    }, indent=1))

    shutil.copytree(Path(dcrest.__file__).parent, root / "dcrest",
                    ignore=shutil.ignore_patterns("__pycache__", "VENDORED.md"))
    tg = sorted(labels[g] for g in task.targets)
    note = ("The targets lie in a climate region that contains no training catchment."
            if task.task_id.startswith("region-") else
            "The targets are interspersed with the training catchments across the country.")
    tmpl = TASK_MD if released else TASK_MD_CALIB
    (root / "TASK.md").write_text(tmpl.format(
        n_targets=len(task.targets), n_donors=len(task.donors), obs0=config.TRAIN[0], obs1=config.TRAIN[1],
        pred0=config.TEST[0], pred1=config.TEST[1], first_target=tg[0], last_target=tg[-1],
        spin_pred=config.TEST_SPINUP_FROM, spin_obs=config.TRAIN_SPINUP_FROM, split_note=note))
    shutil.copy(root / "TASK.md", root / "task_folder" / "TASK.md")

    hidden = config.HIDDEN / task.task_id
    hidden.mkdir(parents=True, exist_ok=True)
    (hidden / "mapping.json").write_text(json.dumps({v: k for k, v in labels.items()}, indent=0))
    (hidden / "task.json").write_text(task.to_json())
    return root


def brief(task_id: str, variant: str = "template") -> Path:
    """Small, text-only copy of the task for the harness improver (no arrays)."""
    root = config.TASKS / task_id / variant
    b = config.TASKS / task_id / f"brief-{variant}"
    if not b.exists():
        # Harness² builds evidence for parallel stages concurrently: build aside, then rename,
        # so no caller sees a half-written brief and a lost race is harmless
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix=f".brief-{variant}-", dir=b.parent))
        (tmp / "data").mkdir()
        shutil.copy(root / "TASK.md", tmp / "TASK.md")
        for f in ("meta.json", "target_attributes.csv"):
            shutil.copy(root / "task_folder" / "data" / f, tmp / "data" / f)
        da = pd.read_csv(root / "task_folder" / "data" / "donor_attributes.csv")
        da.head(20).to_csv(tmp / "data" / "donor_attributes_head.csv", index=False)
        try:
            tmp.rename(b)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            if not (b / "TASK.md").is_file():    # lost the race only if someone else won it
                raise
    return b
