"""Spatial cross-validation folds, the pilot target set, and anonymisation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config


def folds(ids: List[str], strata: List[str], n: int = config.N_FOLDS, seed: int = config.FOLD_SEED) -> Dict[str, int]:
    """Stratified random k-fold over basins: each stratum is dealt round-robin after a shuffle,
    so every fold samples every region. Returns ``{gauge_id: fold}`` with folds 1..n."""
    rng = np.random.default_rng(seed)
    out: Dict[str, int] = {}
    offset = 0
    for s in sorted(set(strata)):
        members = [i for i, t in zip(ids, strata) if t == s]
        rng.shuffle(members)
        for j, g in enumerate(members):
            out[g] = (j + offset) % n + 1
        offset += len(members)
    return out


@dataclass(frozen=True)
class Task:
    """One ungauged-prediction task: calibrate on donors, predict targets."""
    task_id: str
    targets: tuple      # gauge ids
    donors: tuple       # gauge ids

    def anon(self) -> Dict[str, str]:
        """gauge id -> anonymous id seen by the agent (T### targets, D#### donors)."""
        m = {g: f"T{i + 1:03d}" for i, g in enumerate(self.targets)}
        m.update({g: f"D{i + 1:04d}" for i, g in enumerate(self.donors)})
        return m

    def to_json(self) -> str:
        return json.dumps({"task_id": self.task_id, "targets": list(self.targets), "donors": list(self.donors)})


def make_tasks(attrs: pd.DataFrame, regions: pd.Series, n_pilot: int = config.PILOT_TARGETS) -> Dict[str, Task]:
    """``fold1``..``fold5`` (all held-out basins of a fold) and ``fold1-pilot30``."""
    ids = sorted(attrs.index)
    f = folds(ids, [regions.get(i, "Other") for i in ids])
    tasks = {}
    for k in range(1, config.N_FOLDS + 1):
        tg = tuple(sorted(i for i in ids if f[i] == k))
        dn = tuple(sorted(i for i in ids if f[i] != k))
        tasks[f"fold{k}"] = Task(f"fold{k}", tg, dn)
    # pilot: n_pilot fold-1 targets, stratified by region, donors unchanged
    rng = np.random.default_rng(config.FOLD_SEED + 1)
    f1 = list(tasks["fold1"].targets)
    by_reg: Dict[str, List[str]] = {}
    for g in f1:
        by_reg.setdefault(regions.get(g, "Other"), []).append(g)
    picked: List[str] = []
    quota = {r: max(1, round(n_pilot * len(v) / len(f1))) for r, v in by_reg.items()}
    for r, members in sorted(by_reg.items()):
        rng.shuffle(members)
        picked += members[:quota[r]]
    picked = sorted(picked)[:n_pilot] if len(picked) > n_pilot else sorted(picked)
    tasks[f"fold1-pilot{n_pilot}"] = Task(f"fold1-pilot{n_pilot}", tuple(picked), tasks["fold1"].donors)
    for r in sorted(set(regions.reindex(ids).fillna("Other"))):
        tg = tuple(sorted(i for i in ids if regions.get(i, "Other") == r))
        dn = tuple(sorted(i for i in ids if regions.get(i, "Other") != r))
        tasks["region-" + r.replace(" ", "-")] = Task("region-" + r.replace(" ", "-"), tg, dn)
    return tasks


def fold_table(tasks: Dict[str, Task]) -> pd.DataFrame:
    rows = [(g, int(t.task_id[4:])) for t in tasks.values() if t.task_id.startswith("fold") and "-" not in t.task_id
            for g in t.targets]
    return pd.DataFrame(rows, columns=["gauge_id", "fold"]).set_index("gauge_id").sort_index()
