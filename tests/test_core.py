"""Offline unit tests: weights, metrics, Oudin PET, forecast splice identity."""
import numpy as np
import pandas as pd
import pytest
import shapely

from nwp2sf import metrics
from nwp2sf.analysis import oudin_pet
from nwp2sf.gridweights import coverage_weights, weighted_mean
from nwp2sf.regions import region_of_state


def test_coverage_weights_sum_to_one_and_locate_cells():
    lat = 50.0 - 0.25 * np.arange(20)         # descending, like ECMWF
    lon = -110.0 + 0.25 * np.arange(40)
    poly = shapely.box(-105.1, 46.1, -104.6, 46.4)   # spans several cells
    W = coverage_weights([poly, shapely.Point(-104.0, 47.0).buffer(0.01)], lat, lon)
    rs = np.asarray(W.sum(axis=1)).ravel()
    assert np.allclose(rs, 1.0)
    # field = longitude of the cell centre -> basin mean must be near the polygon's centre
    field = np.tile(lon, (20, 1)).ravel()   # (nlat, nlon)
    m = weighted_mean(W, field[None, :])[0]
    assert abs(m[0] - (-104.85)) < 0.1
    assert abs(m[1] - (-104.0)) < 0.13


def test_weighted_mean_nan_aware():
    lat = np.array([1.0, 0.0]); lon = np.array([0.0, 1.0])
    W = coverage_weights([shapely.box(-0.5, -0.5, 1.5, 1.5)], lat, lon)
    X = np.array([[1.0, np.nan, 3.0, np.nan]])
    assert np.isclose(weighted_mean(W, X)[0, 0], 2.0, atol=1e-3)   # cos(lat) weighting
    assert np.isnan(weighted_mean(W, np.full((1, 4), np.nan))[0, 0])


def test_nse_kge_basic():
    obs = np.array([[1, 2, 3, 4, 5.0]]).T
    assert np.isclose(metrics.nse(obs, obs)[0], 1.0)
    assert np.isclose(metrics.kge(obs, obs)[0], 1.0)
    assert np.isclose(metrics.nse(np.full_like(obs, 3.0), obs)[0], 0.0)
    o = obs.copy(); o[2] = np.nan
    assert np.isclose(metrics.nse(obs, o)[0], 1.0)
    assert np.isnan(metrics.nse(obs, o, min_n=5)[0])


def test_oudin_pet_is_zero_when_cold_and_positive_in_summer():
    pet = oudin_pet(np.array([[-10.0, 25.0]]), np.array([180]), np.array([40.0, 40.0]))
    assert pet[0, 0] == 0.0 and 3.0 < pet[0, 1] < 8.0


def test_regions():
    assert region_of_state("CO") == "Southwest"
    assert region_of_state("xx") == "Other"


def test_forecast_splice_identity():
    dcrest = pytest.importorskip("dcrest")
    from nwp2sf.hydro import run_forecasts, simulate
    n, T = 4, 400
    rng = np.random.default_rng(0)
    P = (rng.random((T, n)) * 8).astype(np.float32)
    Ta = (10 + 8 * rng.standard_normal((T, n))).astype(np.float32)
    E = np.full((T, n), 3.0, dtype=np.float32)
    params = dcrest.default_parameters(n_units=n, area_km2=[50.0, 300.0, 1200.0, 5000.0])
    cfg = dcrest.CrestConfig()
    dates = pd.date_range("2020-01-01", periods=T, freq="D")
    Qref = simulate(P, Ta, E, params, cfg)
    inits = [dates[300].date(), dates[310].date(), dates[350].date()]
    L, lag = 7, 1
    fc = {d: P[dates.get_loc(pd.Timestamp(d)) + lag:][:L] for d in inits}
    ii, Q = run_forecasts(dates, P, Ta, E, params, cfg, fc, L, lag=lag, batch=2)
    for k, d in enumerate(ii):
        t0 = dates.get_loc(pd.Timestamp(d)) + lag
        assert np.allclose(Q[k], Qref[t0:t0 + L], atol=1e-4)
