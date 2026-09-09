"""WeatherNext 3 fetcher against a synthetic local store with the documented layout."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

zarr = pytest.importorskip("zarr")
from nwp2sf.config import REPO_ROOT
from nwp2sf.gridweights import save_weights, coverage_weights
from nwp2sf.nwp.weathernext import WeatherNext3


@pytest.fixture
def store(tmp_path):
    """Two basins on a coarse 0.1-degree-like grid; hourly imerg_tp_1hr_mean in metres."""
    lat = np.round(np.arange(30.05, 45.0, 0.1), 3)          # ascending
    lon360 = np.round(np.arange(230.05, 300.0, 0.1), 3)     # 0..360 convention (-130 .. -60)
    hours = np.arange(1, 361)
    rng = np.random.default_rng(1)
    rain = rng.random((len(hours), len(lat), len(lon360))).astype("f4") * 0.002   # up to 2 mm/h
    rain[:, :, :] *= (np.arange(len(lon360)) % 2)[None, None, :]                    # odd columns rain, even are dry
    ds = xr.Dataset({"imerg_tp_1hr_mean": (("lead_time", "lat_0p1", "lon_0p1"), rain),
                     "total_precipitation_1hr_mean": (("lead_time", "lat_0p1", "lon_0p1"), rain * 2)},
                    coords={"lead_time": hours.astype("timedelta64[h]"), "lat_0p1": lat, "lon_0p1": lon360})
    root = tmp_path / "stats"
    p = root / "2026_to_present" / "20260301_00hr_01_preds" / "predictions.zarr"
    p.parent.mkdir(parents=True)
    ds.to_zarr(p, zarr_format=3)
    (root / "2026_to_present" / "20260302_00hr_01_preds").mkdir()
    (root / "2026_to_present" / "20260302_06hr_01_preds").mkdir()
    return root, lat, lon360, rain


def _weights(tmp_path, lat, lon360):
    import shapely
    lon = np.sort((lon360 + 180) % 360 - 180)
    # basin A sits on one odd (rainy) column, basin B on an even (dry) column
    ia, ib = 101, 100
    ga = shapely.box(lon[ia] - 0.04, 40.0, lon[ia] + 0.04, 40.5)
    gb = shapely.box(lon[ib] - 0.04, 40.0, lon[ib] + 0.04, 40.5)
    W = coverage_weights([ga, gb], lat, lon)
    d = tmp_path / "data" / "basins"; d.mkdir(parents=True)
    save_weights(d / "weights_wn3_0p1.npz", W, lat, lon, ["A", "B"])
    pd.DataFrame({"gauge_id": ["A", "B"], "lat": [40.25, 40.25], "lon": [lon[ia], lon[ib]], "state": ["CO", "CO"],
                  "region": ["Southwest"] * 2, "area_km2": [10.0, 10.0], "name": ["a", "b"], "huc02": ["14", "14"]}
                 ).to_csv(d / "basins.csv", index=False)
    return tmp_path / "data", ia, ib


def test_available_and_daily_totals(tmp_path, store):
    root, lat, lon360, rain = store
    data_dir, ia, ib = _weights(tmp_path, lat, lon360)
    src = WeatherNext3(data_dir, tmp_path / "cache", ["A", "B"], lead_days=14, init_hour=0, day_start_hour=6,
                       root=str(root), store="statistics", stat="mean")
    avail = src.available(pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-12-31").date())
    assert avail == [pd.Timestamp("2026-03-01").date(), pd.Timestamp("2026-03-02").date()]   # 06Z run ignored
    df = src.fetch(pd.Timestamp("2026-03-01").date())
    assert list(df.columns) == [f"lead_{d}" for d in range(1, 15)]
    # basin B is on a dry column
    assert np.allclose(df.loc["B"].values, 0.0)
    # basin A: day d = sum of hours (6+24(d-1), 6+24d] on its column, in mm
    lon = np.sort((lon360 + 180) % 360 - 180)
    j = int(np.argmin(np.abs(lat - 40.25)))
    col = int(np.argmin(np.abs(lon360 - ((lon[ia] + 360) % 360))))
    jj = np.where((lat >= 40.0) & (lat <= 40.5))[0]
    for d in (1, 7, 14):
        h0, h1 = 6 + 24 * (d - 1), 6 + 24 * d
        expect = rain[h0:h1, jj, col].sum(axis=0).mean() * 1000
        assert abs(df.loc["A", f"lead_{d}"] - expect) / expect < 0.02
    assert src.fetch(pd.Timestamp("2026-03-05").date()) is None      # no store for that day


def test_no_credentials_is_not_fatal(tmp_path):
    src = WeatherNext3(tmp_path, tmp_path, ["A"], lead_days=14, init_hour=0, day_start_hour=6,
                       root="gs://weathernext3_statistics_spatial/does_not_matter")
    assert src.available(pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-01-02").date()) == []
