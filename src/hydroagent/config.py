"""Paths, periods and fixed choices of the ungauged-basin experiment."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("HYDROAGENT_CACHE", "~/.cache/hydroagent")).expanduser()
CAMELS_RAW = Path(os.environ.get("CAMELS_RAW", "~/.cache/nwp2sf/camels")).expanduser()

DATA_NPZ = CACHE / "camels_daymet_daily.npz"      # 665 basins x 1980-2014, built by camels.build
ATTR_CSV = REPO / "data" / "hydroagent" / "camels_attributes.csv"   # static attributes (committed)
PARAMS = REPO / "data" / "params" / "dCREST_CAMELS_v1.0_daymet665.npz"
RESULTS = REPO / "results" / "hydroagent"
TASKS = CACHE / "tasks"          # per-task agent workspaces (big; never committed)
HIDDEN = CACHE / "hidden"        # target ids and observations the agent never sees
RUNS = CACHE / "runs"            # harness2 rollouts and scheduler state

# Community benchmark protocol (Newman et al. 2017; Kratzert et al. 2019), the one the
# released dCREST-CAMELS parameters were calibrated and tested under.
TRAIN = ("1999-10-01", "2008-09-30")
TRAIN_SPINUP_FROM = "1996-10-01"
TEST = ("1989-10-01", "1999-09-30")
TEST_SPINUP_FROM = "1986-10-01"

N_FOLDS = 5
FOLD_SEED = 20261018
PILOT_TARGETS = 30

# CAMELS static attributes the agent may use (camels_hydro is excluded: its signatures
# are computed from observed discharge, including the test period).
ATTR_FILES = ("camels_topo.txt", "camels_clim.txt", "camels_soil.txt", "camels_vege.txt",
              "camels_geol.txt")
DROP_ATTRS = ("gauge_name", "gauge_lat", "gauge_lon", "huc_02")
