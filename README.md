# NWP2StreamflowBench

**How much of an NWP model's precipitation skill survives the journey to the river?**

NWP2StreamflowBench propagates precipitation forecasts from numerical weather
prediction (NWP) models through a calibrated hydrologic model on hundreds of
U.S. catchments and scores the resulting streamflow against USGS observations,
as a function of forecast lead time and climate region. The first two entrants
are ECMWF's physics-based **IFS** and its AI model **AIFS-single**; the pipeline
is model-agnostic so in-house AI NWP models can be added with one class.

**Dashboard:** https://chrimerss.github.io/NWP2StreamflowBench/

## Design

```
ECMWF open data (AWS)  ─┐  basin-mean daily precipitation (mm/day)
in-house NWP (future)  ─┘        │  lead days 1..14, 00 UTC cycles
                                 ▼
gridMET analysis ──► dCREST (dCREST-CAMELS v1.0) ──► simulated discharge ──► NSE / KGE vs USGS
(spin-up + reference)  same state at every init;                 per (model, lead, basin)
                       only P is replaced during the lead window  → medians by lead & region
```

* **Hydrologic model.** [dCREST](https://github.com/Skyan1002/dCREST), a
  differentiable, lumped daily implementation of CREST (the EF5 core), with the
  released **dCREST-CAMELS v1.0** parameters (665 catchments, calibrated on
  Daymet under the community benchmark protocol). Parameters are used as
  published; nothing is recalibrated.
* **Basins.** The 665 catchments in the parameter set minus 32 gauges that no
  longer report (`data/basins/excluded.csv`), i.e. 633 basins, with GAGES-II
  polygons (the CAMELS source). Basin means are exact area-weighted grid-cell
  coverage fractions on each source grid (`data/basins/weights_*.npz`).
* **Analysis forcing.** [gridMET](https://www.climatologylab.org/gridmet.html)
  (4 km, ~1-day latency): precipitation, Tmin/Tmax, and Oudin PET (the PET
  formulation dCREST-CAMELS was calibrated with). The analysis-forced run from
  2023-01-01 provides the warm state at every initialisation and the
  **reference** (ceiling) skill.
* **Forecast experiment.** For every 00 UTC initialisation, precipitation on
  lead days 1…14 is replaced by the forecast; temperature and PET keep their
  analysis values so that differences isolate precipitation error ("perfect
  state, imperfect forcing"). Forecast days are 06–06 UTC accumulations to match
  the gridMET day. Because Daymet (and hence the calibration) labels
  precipitation by the station-observer day, all forcing enters the model with a
  one-day lag; without it the simulated hydrograph leads observations by a day
  (reference median NSE 0.15 instead of 0.55).
* **Scoring.** For each lead, the discharge valid on each day is gathered across
  initialisations into one series per basin and scored against USGS daily mean
  discharge (mm/day over the CAMELS area) with NSE and KGE over a window common
  to all leads (≥ 60 valid days per basin). Reported numbers are medians and
  quartiles across basins (NSE and KGE side by side on the dashboard); regions are the nine NOAA/NCEI climate regions,
  assigned by gauge state. Precipitation skill (correlation/NSE of basin-mean
  forecast vs analysis precipitation) is scored alongside so that the loss from
  rain to river is visible.

## Quick start

```bash
git clone https://github.com/chrimerss/NWP2StreamflowBench.git && cd NWP2StreamflowBench
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .

nwp2sf fetch-analysis          # gridMET basin forcing (downloads ~1.5 GB of annual NetCDF into ~/.cache/nwp2sf)
nwp2sf fetch-obs               # USGS daily discharge
nwp2sf fetch-nwp --workers 6   # basin-mean IFS/AIFS precipitation for every missing init since 2026-01-01
nwp2sf run                     # dCREST runs + metrics -> results/
nwp2sf dashboard               # -> docs/data/summary.json (the site reads it)
```

`nwp2sf all` chains the five steps; every step is incremental, so a daily run
only touches new initialisations and the provisional tail of gridMET/USGS.
`configs/benchmark.yaml` holds all settings (start date, leads, models, paths).

The committed `data/basins/` files were built with
`python scripts/build_basins.py` (needs the `basins` extra and the 200 MB
GAGES-II archive); routine runs and CI never need the shapefiles.

## Adding a model

Implement `nwp2sf.nwp.base.PrecipForecastSource` — two methods,
`available(start, end)` and `fetch(init) -> DataFrame[gauge_id × lead_1..lead_L]`
in mm/day — and register it in `configs/benchmark.yaml`:

```yaml
models:
  - id: mymodel
    name: "In-house AI NWP v1"
    fetcher: mypkg.mymodule:MyModelSource
    kwargs: {member: control}
```

If the model lives on a new grid, add its axes to `nwp2sf.basins.GRIDS` and
rebuild the weights. Extracted basin-mean forecasts are stored per init under
`data/nwp/<id>/` and are never re-downloaded.

## Layout

```
configs/benchmark.yaml      settings
src/nwp2sf/
  basins.py, gridweights.py   basin set, GAGES-II polygons, coverage weights
  nwp/                        forecast sources (base interface, ECMWF open data)
  analysis.py                 gridMET forcing + Oudin PET
  usgs.py                     USGS Water Data API
  hydro.py                    dCREST driver, forecast splicing
  metrics.py, pipeline.py     scores, benchmark orchestration, summary
  dashboard.py, cli.py        site data, command line
data/                       basin metadata & weights, parameter set, basin-scale forcing/obs/forecasts (committed)
results/                    metric tables (committed), simulated discharge (not committed)
docs/                       GitHub Pages dashboard
.github/workflows/          daily update + tests
```

## Data sources and licences

* ECMWF open data (IFS HRES, AIFS-single; 0.25°) via `ecmwf-opendata` from the
  AWS Open Data mirror — © ECMWF, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
* gridMET — Abatzoglou (2013), Climatology Lab, UC Merced.
* USGS Water Data for the Nation daily values (provisional data are revised).
* GAGES-II (Falcone 2011); CAMELS-US (Newman et al. 2015; Addor et al. 2017).
* dCREST and dCREST-CAMELS parameters — HyDROS Lab, University of Oklahoma, MIT licence.

Code: MIT.
