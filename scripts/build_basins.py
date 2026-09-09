#!/usr/bin/env python
"""Build data/basins/basins.csv and the basin/grid weight files from GAGES-II.

Run once (or whenever the basin set or a source grid changes). Needs the
``basins`` extra (geopandas) and downloads ~200 MB of GAGES-II shapefiles into
the cache directory.
"""
import argparse
import logging

from nwp2sf import load_config
from nwp2sf.analysis import GRID as GRIDMET, gridmet_axes
from nwp2sf.basins import build

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)
lat, lon = gridmet_axes(cfg.cache_dir)
meta = build(cfg.data_dir, cfg.cache_dir, cfg.parameter_set, extra_grids={GRIDMET: (lat, lon)})
print(meta.region.value_counts())
print(f"{len(meta)} basins written to {cfg.data_dir / 'basins'}")
