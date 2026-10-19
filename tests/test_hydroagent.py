"""hydroagent: leakage guards on the agent workspace and scorer/baseline consistency.

Needs the locally built CAMELS record and pilot task folder (`hydroagent prepare`);
skipped elsewhere (e.g. CI without the 3.4 GB CAMELS archive).
"""
import json

import numpy as np
import pandas as pd
import pytest

from hydroagent import config

TASK = f"fold1-pilot{config.PILOT_TARGETS}"
TEMPLATE = config.TASKS / TASK / "template"
pytestmark = pytest.mark.skipif(not (TEMPLATE / "task_folder" / "data" / "donors.npz").is_file()
                                or not config.DATA_NPZ.is_file(), reason="run `hydroagent prepare` first")


def test_workspace_hides_target_and_test_period_discharge():
    dd = TEMPLATE / "task_folder" / "data"
    t = np.load(dd / "targets.npz")
    assert "Q_obs" not in t.files
    d = np.load(dd / "donors.npz")
    dates = pd.DatetimeIndex(d["dates"])
    outside = (dates < pd.Timestamp(config.TRAIN[0])) | (dates > pd.Timestamp(config.TRAIN[1]))
    assert np.isnan(d["Q_obs"][outside]).all()
    assert np.isfinite(d["Q_obs"][~outside]).mean() > 0.9
    for f in ("donor_attributes.csv", "target_attributes.csv"):
        cols = set(pd.read_csv(dd / f).columns)
        assert not cols & {"gauge_id", "gauge_name", "gauge_lat", "gauge_lon", "huc_02"}
    ids = set(pd.read_csv(dd / "target_attributes.csv").id)
    assert all(i.startswith("T") for i in ids)
    hidden = json.loads((config.HIDDEN / TASK / "mapping.json").read_text())
    for gauge in hidden.values():
        assert gauge not in (TEMPLATE / "TASK.md").read_text()


def test_scorer_matches_baseline_code(tmp_path):
    """Score the targets' own released parameters through the agent-facing path; the result
    must equal the gauged upper bound computed by baselines.run (same NSE per basin)."""
    import dcrest

    from hydroagent import camels
    from hydroagent.h2adapter import HydroAdapter, configure
    from hydroagent.simulate import run_period

    configure()
    d = camels.load()
    mapping = json.loads((config.HIDDEN / TASK / "mapping.json").read_text())
    targets = sorted(pd.read_csv(TEMPLATE / "task_folder" / "data" / "target_attributes.csv").id)
    gauges = [mapping[t] for t in targets]
    pos = [d["ids"].index(g) for g in gauges]
    params, meta = dcrest.load_parameters(str(config.PARAMS), subset=gauges)
    Q, _ = run_period(d, params, dcrest.config_from_meta(meta), *config.TEST,
                      spinup_from=config.TEST_SPINUP_FROM, cols=np.asarray(pos))
    w = (d["dates"] >= config.TEST[0]) & (d["dates"] <= config.TEST[1])
    run = tmp_path / "run"
    (run / "output").mkdir(parents=True)
    pd.DataFrame(Q, index=pd.Index(d["dates"][w], name="date"), columns=targets).to_csv(
        run / "output" / "target_discharge.csv")
    (run / "rollout.json").write_text(json.dumps({"task_id": TASK, "audit": {}}))
    a = HydroAdapter([TASK])
    assert a.delivered(run)
    got = pd.DataFrame(a.score(run, TASK)["per_basin"]).set_index("gauge_id").nse
    ref = pd.read_csv(config.RESULTS / "baselines_per_basin.csv", dtype={"gauge_id": str})
    ref = ref[ref.method == "gauged_upper"].set_index("gauge_id").nse.reindex(got.index)
    assert np.nanmax(np.abs(got.values - ref.values)) < 1e-3   # CSV rounding only


def test_offline_validation_rejects_dropping_the_internet_rule():
    from hydroagent.h2adapter import HydroAdapter, configure

    configure()
    a = HydroAdapter([TASK])
    with pytest.raises(ValueError):
        a.validate_tree({"guardrails.md": "- Be careful."})
    a.validate_tree({"guardrails.md": "- Do not download data from the internet."})


CALIB = config.TASKS / "region-Southeast" / "calib"


@pytest.mark.skipif(not (CALIB / "task_folder" / "data" / "donors.npz").is_file(), reason="calib task not built")
def test_calib_variant_has_no_precalibrated_parameters():
    from hydroagent.h2adapter import HydroAdapter, configure

    configure()
    files = {p.name for p in (CALIB / "task_folder" / "data").iterdir()}
    assert "donor_params_released.csv" not in files
    assert "No pre-calibrated parameters" in (CALIB / "TASK.md").read_text()
    a = HydroAdapter(["region-Southeast"], variant="calib")
    assert a.base_harness().name == "harness_calib" and a.spec.name == "camels-calib"
    assert "released" not in (a.base_harness() / "systemprompt.md").read_text()
    # targets all in the held-out region, donors in none of it
    from nwp2sf.basins import load_basins
    reg = load_basins(config.REPO / "data", include_excluded=True).region
    mapping = json.loads((config.HIDDEN / "region-Southeast" / "mapping.json").read_text())
    assert {reg[g] for t, g in mapping.items() if t.startswith("T")} == {"Southeast"}
    assert "Southeast" not in {reg[g] for t, g in mapping.items() if t.startswith("D")}
