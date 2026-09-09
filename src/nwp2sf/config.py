"""Configuration loading."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelSpec:
    id: str
    name: str
    fetcher: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    start_date: str
    end_date: Optional[str]
    init_hour: int
    day_start_hour: int
    lead_days: int
    min_score_days: int
    spinup_start: str
    forcing_lag_days: int
    models: List[ModelSpec]
    analysis_source: str
    pet: str
    refresh_months: int
    parameter_set: Path
    cache_dir: Path
    data_dir: Path
    results_dir: Path
    docs_dir: Path
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def leads(self) -> List[int]:
        return list(range(1, self.lead_days + 1))

    def model(self, mid: str) -> ModelSpec:
        for m in self.models:
            if m.id == mid:
                return m
        raise KeyError(mid)


def _path(p: str, root: Path) -> Path:
    q = Path(os.path.expanduser(p))
    return q if q.is_absolute() else (root / q)


def load_config(path: Optional[str | Path] = None, root: Optional[Path] = None) -> Config:
    root = Path(root) if root else REPO_ROOT
    path = Path(path) if path else root / "configs" / "benchmark.yaml"
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    b, a, h, p = raw["benchmark"], raw["analysis"], raw["hydro"], raw["paths"]
    models = [ModelSpec(id=m["id"], name=m["name"], fetcher=m["fetcher"],
                        kwargs=dict(m.get("kwargs") or {})) for m in raw["models"]]
    return Config(
        start_date=str(b["start_date"]), end_date=(str(b["end_date"]) if b.get("end_date") else None),
        init_hour=int(b.get("init_hour", 0)), day_start_hour=int(b.get("day_start_hour", 0)), lead_days=int(b["lead_days"]),
        min_score_days=int(b.get("min_score_days", 60)), spinup_start=str(b["spinup_start"]), forcing_lag_days=int(b.get("forcing_lag_days", 0)),
        models=models, analysis_source=a["source"], pet=a.get("pet", "oudin"),
        refresh_months=int(a.get("refresh_months", 3)),
        parameter_set=_path(h["parameter_set"], root),
        cache_dir=_path(p["cache_dir"], root), data_dir=_path(p["data_dir"], root),
        results_dir=_path(p["results_dir"], root), docs_dir=_path(p["docs_dir"], root),
        raw=raw,
    )
