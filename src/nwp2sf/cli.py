"""Command-line entry point: ``nwp2sf <command> [options]``.

Commands
    fetch-analysis   update gridMET basin forcing (spin-up + precipitation reference)
    fetch-obs        update USGS daily discharge
    fetch-nwp        extract basin-mean NWP precipitation for every missing initialisation
    run              drive dCREST with analysis + forecast precipitation, score, write metrics
    dashboard        build the static GitHub Pages site from the metrics
    all              fetch-analysis, fetch-obs, fetch-nwp, run, dashboard
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd

from . import analysis, usgs
from .basins import load_basins
from .config import Config, load_config
from .nwp import load_fetcher

log = logging.getLogger("nwp2sf")


def _dates(cfg: Config):
    start = pd.Timestamp(cfg.start_date).date()
    end = pd.Timestamp(cfg.end_date).date() if cfg.end_date else date.today()
    return start, end


def cmd_fetch_analysis(cfg: Config, args):
    b = load_basins(cfg.data_dir)
    df = analysis.update(cfg.data_dir, cfg.cache_dir, list(b.gauge_id), b.lat.values,
                         pd.Timestamp(cfg.spinup_start).date(), cfg.refresh_months, cfg.pet)
    d0, d1 = df.index.get_level_values("date").min(), df.index.get_level_values("date").max()
    print(f"analysis forcing: {d0:%Y-%m-%d} .. {d1:%Y-%m-%d}, {df.index.get_level_values('gauge_id').nunique()} basins")


def cmd_fetch_obs(cfg: Config, args):
    b = load_basins(cfg.data_dir)
    start = (pd.Timestamp(cfg.start_date) - pd.DateOffset(months=1)).date()
    q = usgs.update(cfg.data_dir, list(b.gauge_id), b.area_km2, start, cfg.refresh_months)
    print(f"observations: {q.index.min():%Y-%m-%d} .. {q.index.max():%Y-%m-%d}, "
          f"{int(q.notna().any().sum())} basins with data")


def cmd_fetch_nwp(cfg: Config, args):
    b = load_basins(cfg.data_dir)
    ids = list(b.gauge_id)
    start, end = _dates(cfg)
    if args.start:
        start = pd.Timestamp(args.start).date()
    if args.end:
        end = pd.Timestamp(args.end).date()
    for spec in cfg.models:
        if args.model and spec.id not in args.model:
            continue
        cls = load_fetcher(spec.fetcher)
        src = cls(cfg.data_dir, cfg.cache_dir, ids, cfg.lead_days, cfg.init_hour, cfg.day_start_hour, **spec.kwargs)
        inits = src.available(start, end)
        todo = [d for d in inits if args.overwrite or not src.path(spec.id, d).exists()]
        log.info("%s: %d inits available, %d to fetch", spec.id, len(inits), len(todo))
        ok = fail = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(src.fetch_and_store, spec.id, d, args.overwrite): d for d in todo}
            for f in as_completed(futs):
                d = futs[f]
                try:
                    p = f.result()
                    ok += p is not None
                    fail += p is None
                except Exception as e:  # keep going; the init is retried next run
                    fail += 1
                    log.error("%s %s failed: %s", spec.id, d, e)
                if (ok + fail) % 20 == 0:
                    log.info("%s: %d/%d done (%d failed)", spec.id, ok + fail, len(todo), fail)
        print(f"{spec.id}: fetched {ok}, failed {fail}, stored {len(list(src.store_dir(spec.id).glob('*.parquet')))} inits")


def cmd_run(cfg: Config, args):
    from .pipeline import run_benchmark
    run_benchmark(cfg)


def cmd_dashboard(cfg: Config, args):
    from .dashboard import build_site
    build_site(cfg)


def cmd_all(cfg: Config, args):
    cmd_fetch_analysis(cfg, args)
    cmd_fetch_obs(cfg, args)
    cmd_fetch_nwp(cfg, args)
    cmd_run(cfg, args)
    cmd_dashboard(cfg, args)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nwp2sf", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to benchmark.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-analysis")
    sub.add_parser("fetch-obs")
    for name in ("fetch-nwp", "all"):
        p = sub.add_parser(name)
        p.add_argument("--model", nargs="*", help="restrict to these model ids")
        p.add_argument("--start"); p.add_argument("--end")
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--overwrite", action="store_true")
    sub.add_parser("run")
    sub.add_parser("dashboard")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    cfg = load_config(args.config)
    {"fetch-analysis": cmd_fetch_analysis, "fetch-obs": cmd_fetch_obs, "fetch-nwp": cmd_fetch_nwp,
     "run": cmd_run, "dashboard": cmd_dashboard, "all": cmd_all}[args.cmd](cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
