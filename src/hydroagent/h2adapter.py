"""Harness² adapter: the ungauged-basin task as a harness2 benchmark, Claude as backbone.

Import order matters: harness2 reads its model settings from the environment when
``harness2.config`` is first imported, so call :func:`configure` before importing
anything else from harness2.

Claude through a direct Anthropic API key
-----------------------------------------
Upstream harness2 (commit 1cfe254) strips ``ANTHROPIC_API_KEY`` from improver sessions
(``core.proposer._DROP_ENV``) and forces Claude through Vertex AI. :func:`patch_claude`
keeps the key for both channels and, for the ``claude_cli`` channel, drops the Vertex
switches when a key is present. The executor (OpenCode ``anthropic/<model>``) reads the
key from the environment unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config

DEFAULT_EXECUTOR = "anthropic/claude-opus-5-5"
DEFAULT_IMPROVER = "anthropic/claude-opus-5-5"
PROMPT = (
    "Read TASK.md in the workspace root ({ws}) and complete the assignment. "
    "Data are in {ws}/task_folder/data; write every deliverable to {ws}/output. "
    "Stay inside {ws}."
)
NETWORK_PATTERNS = re.compile(
    r"waterservices\.usgs|waterdata\.usgs|api\.waterdata|nwis\.|dataretrieval|ucar\.edu|gdex|zenodo\.org|"
    r"hydroshare|ornl\.gov|sciencebase|\bcurl\b|\bwget\b|requests\.(get|post)|urllib\.request|urlopen|"
    r"http\.client|socket\.create_connection|pip install|uv pip",
    re.IGNORECASE)


def configure(executor: str | None = None, improver: str | None = None) -> None:
    os.environ.setdefault("HARNESS2_SOLVER_MODEL", executor or DEFAULT_EXECUTOR)
    os.environ.setdefault("HARNESS2_IMPROVER_MODEL", improver or DEFAULT_IMPROVER)
    os.environ.setdefault("HARNESS2_JUDGE_MODEL", "dcrest-nse-vs-usgs")   # deterministic scorer, no LLM
    os.environ.setdefault("HARNESS2_RUNS_DIR", str(config.RUNS))


def patch_claude() -> None:
    from harness2.core import proposer as P

    # keep the key; still drop ANTHROPIC_BASE_URL/AUTH_TOKEN, which a parent Claude Code session
    # may export (OpenCode treats the base URL as including /v1 and would call /messages)
    P._DROP_ENV = tuple(k for k in P._DROP_ENV if k != "ANTHROPIC_API_KEY")
    if getattr(P.Proposer.env, "_hydroagent", False):
        return
    upstream = P.Proposer.env

    def env(self):
        e = upstream(self)
        if os.environ.get("ANTHROPIC_API_KEY"):
            e["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]
            if self.spec.kind == "claude_cli":
                for k in ("CLAUDE_CODE_USE_VERTEX", "ANTHROPIC_VERTEX_PROJECT_ID", "CLOUD_ML_REGION"):
                    e.pop(k, None)
        return e

    env._hydroagent = True
    P.Proposer.env = env

    # keep each improver session's OpenCode event stream (harness2 only keeps the reply), so
    # model turns and tool calls of the editor/composer/reflector can be counted afterwards
    import subprocess as _sp
    import time as _time
    real_run = _sp.run

    def run(cmd, *a, **kw):
        env = kw.get("env")
        if isinstance(env, dict) and env.get("OPENCODE_PERMISSION"):
            try:      # same doom-loop fix for improver sessions
                perm = json.loads(env["OPENCODE_PERMISSION"])
                perm.setdefault("doom_loop", "allow")
                kw["env"] = {**env, "OPENCODE_PERMISSION": json.dumps(perm)}
            except ValueError:
                pass
        res = real_run(cmd, *a, **kw)
        try:
            if isinstance(cmd, (list, tuple)) and "src/index.ts" in [str(c) for c in cmd] and \
                    isinstance(getattr(res, "stdout", None), str):
                d = config.RUNS / "_audit" / "events"
                d.mkdir(parents=True, exist_ok=True)
                cwd = Path(str(kw.get("cwd", "."))).name
                (d / f"{_time.strftime('%Y%m%dT%H%M%S')}_{cwd}.jsonl").write_text(res.stdout)
        except Exception:
            pass
        return res

    P.subprocess = type("SubprocessProxy", (), {"run": staticmethod(run), "PIPE": _sp.PIPE,
                                                "TimeoutExpired": _sp.TimeoutExpired,
                                                "DEVNULL": _sp.DEVNULL})()


def _clone_tree(src: Path, dst: Path) -> None:
    """Copy-on-write clone on APFS (instant, no extra space); plain copy elsewhere."""
    if sys.platform == "darwin" and subprocess.run(["cp", "-cR", str(src), str(dst)]).returncode == 0:
        return
    shutil.copytree(src, dst)


def _tree_hash(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in d.rglob("*.py") if "__pycache__" not in p.parts):
        h.update(f.relative_to(d).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


class HydroAdapter:
    """BenchmarkAdapter for ``fold<k>`` / ``fold1-pilot<n>`` tasks over CAMELS-US."""

    def __init__(self, task_ids: list[str], *, timeout: int = 5400, steps: int = 250,
                 network_fence: bool = True, fake_executor: bool = False, variant: str = "template"):
        """``variant="template"``: donors come with released parameters (pilot design).
        ``variant="calib"``: no pre-calibrated parameters; the agent calibrates the training gauges."""
        self.variant = variant
        from harness2 import config as h2c
        from harness2.core.spec import BenchPrompts, BenchSpec, resolve_channel_model
        from harness2.substrate.opencode import OpencodeSubstrate

        self.h2c = h2c
        self.task_ids = list(task_ids)
        self.timeout = int(timeout)
        self.steps = int(steps)
        self.network_fence = network_fence
        self.fake_executor = fake_executor
        self.substrate = OpencodeSubstrate()
        self.rollouts = config.RUNS / "rollouts"
        ch = resolve_channel_model("opencode", h2c.IMPROVER_MODEL, effort="high")
        self.spec = BenchSpec(
            name="camels" if variant == "template" else "camels-calib",
            runs_root=config.RUNS,
            install_mode="workspace_agents_md",
            prompts=BenchPrompts(
                work="predicting daily streamflow in ungauged catchments by calibrating the dCREST "
                     "rainfall-runoff model on gauged donor catchments and transferring it",
                domain_word="region",
                inv_levers="workflow discipline that holds for any regional hydrology study: donor-side "
                           "cross-validation design, time budgeting of slow calibrations, writing a valid "
                           "deliverable early, reproducible code",
                dom_levers="hydrologic knowledge: regionalization schemes (spatial proximity, physical "
                           "similarity, output averaging, regression), attribute choice, donor weighting, "
                           "snow and aridity regimes, model-structure changes justified by donor evidence",
                yardstick="A strong run compares several transfer schemes with a fixed donor hold-out, "
                          "reports median NSE and KGE on held-out donors, picks the best-supported scheme, "
                          "and delivers complete, physically plausible daily discharge for every target.",
                deliverable_word="streamflow prediction",
            ),
            channels={"proposer": ch, "selector": ch},
            domain_word="region",
            leak_globs=("*.npz", "hidden/**"),
            allow_subagents=True,
            proposer_deliverables="contents",
        )

    # ------------------------------------------------------------------ tasks
    def tasks(self, split: str = "all") -> list[str]:
        return list(self.task_ids)

    def domain_of(self, task_id: str) -> str:
        return "camels-us"

    def task_prompt(self, task_id: str) -> str:
        return (config.TASKS / task_id / self.variant / "TASK.md").read_text()

    def task_files(self, task_id: str) -> Path:
        from .workspace import brief
        return brief(task_id, self.variant)

    def _template(self, task_id: str) -> Path:
        t = config.TASKS / task_id / self.variant
        if not (t / "task_folder" / "data" / "donors.npz").is_file():
            raise FileNotFoundError(f"task template missing: {t}; run `hydroagent prepare`")
        return t

    def base_harness(self) -> Path:
        return Path(__file__).parent / ("harness_base" if self.variant == "template" else "harness_calib")

    def validate_tree(self, tree: dict) -> None:
        g = tree.get("guardrails.md", "")
        if "internet" not in g.lower():
            raise ValueError("guardrails.md must keep the rule that no data may be searched for or "
                             "downloaded from the internet (the targets are deliberately anonymous).")

    # ------------------------------------------------------------------ rollouts
    def _run_dir(self, run_id: str) -> Path:
        return self.rollouts / run_id

    def execute(self, task_id: str, harness_dir: Optional[Path], run_id: str, *, timeout: int | None = None) -> Path:
        from harness2.benchmarks.jobbench import render_opencode_tree
        from harness2.core import runmark as RM

        out = self._run_dir(run_id)
        if RM.is_complete(out, harness_dir=harness_dir) and self.delivered(out):
            return out
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        src = Path(harness_dir) if harness_dir is not None else self.base_harness()
        install = self.substrate.materialise(src, self.spec, {})
        tmp = Path(tempfile.mkdtemp(prefix="hydro_ws_"))
        ws = tmp / "ws"
        try:
            _clone_tree(self._template(task_id), ws)
            render = tmp / "render"
            render_opencode_tree(install.staging_dir, render, {"domain_focus": "regional hydrology"})
            for name in (".opencode", "scripts"):
                if (render / name).exists():
                    shutil.copytree(render / name, ws / name, dirs_exist_ok=True)
            if (render / "AGENTS.md").is_file():
                shutil.copy(render / "AGENTS.md", ws / "AGENTS.md")
            dcrest0 = _tree_hash(ws / "dcrest")
            (ws / "opencode.json").write_text(json.dumps({"agent": {"build": {"steps": self.steps}}}))
            if self.fake_executor:
                rc = self._fake_rollout(ws, out)
            else:
                rc = self._opencode(ws, out, int(timeout or self.timeout), run_id)
            shutil.copytree(ws / "output", out / "output", dirs_exist_ok=True)
            changed = _tree_hash(ws / "dcrest") != dcrest0
            if changed:
                shutil.copytree(ws / "dcrest", out / "dcrest_modified",
                                ignore=shutil.ignore_patterns("__pycache__"))
            (out / "rollout.json").write_text(json.dumps(
                {"task_id": task_id, "run_id": run_id, "returncode": rc, "model_structure_changed": changed,
                 "harness": "base" if harness_dir is None else str(harness_dir),
                 "executor": "fake" if self.fake_executor else self.h2c.SOLVER_MODEL_OC,
                 "audit": self._audit(out)}, indent=1))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(install.staging_dir, ignore_errors=True)
        RM.mark_complete(out, run_id=run_id, harness_dir=harness_dir)
        return out

    def _opencode(self, ws: Path, out: Path, timeout: int, run_id: str) -> int:
        h2c = self.h2c
        # doom_loop: OpenCode asks permission when a tool call repeats (e.g. polling a background
        # calibration log); a non-interactive run auto-rejects it and the session ends early.
        perm = {"external_directory": {"*": "deny", str(ws): "allow", f"{ws}/**": "allow"},
                "webfetch": "deny", "websearch": "deny", "codesearch": "deny", "doom_loop": "allow"}
        venv_bin = str(Path(sys.executable).parent)
        inherited = {k: v for k, v in os.environ.items()
                     if k not in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN")}
        env = {**inherited,
               "PATH": f"{venv_bin}:{os.environ.get('PATH', '')}",
               "PYTHONPATH": str(ws), "HYDRO_WORKSPACE": str(ws),
               "OPENCODE_PERMISSION": json.dumps(perm),
               "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "64000",
               "XDG_DATA_HOME": str(out / ".ocdata"), "XDG_STATE_HOME": str(out / ".ocstate")}
        if self.network_fence:
            # Soft fence: tools started from the agent's shell inherit a dead proxy, while the
            # OpenCode process reaches the model API (and its model registry) directly.
            env.update({"HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
                        "https_proxy": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9",
                        "NO_PROXY": "api.anthropic.com,models.dev,localhost,127.0.0.1",
                        "no_proxy": "api.anthropic.com,models.dev,localhost,127.0.0.1"})
        cmd = [str(h2c.BUN), "run", "--cwd", str(h2c.OPENCODE_DIR / "packages" / "opencode"),
               "--conditions=browser", "src/index.ts", "run", PROMPT.format(ws=ws), "--dir", str(ws),
               "--model", h2c.SOLVER_MODEL_OC, "--title", run_id.replace("/", " "), "--format", "json"]
        with open(out / "events.jsonl", "w") as so, open(out / "stderr.log", "w") as se:
            try:
                return subprocess.run(cmd, cwd=h2c.OPENCODE_DIR, env=env, stdout=so, stderr=se,
                                      timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                se.write(f"\n[hydroagent] rollout wall-timeout after {timeout}s\n")
                return -9

    def _fake_rollout(self, ws: Path, out: Path) -> int:
        """Scripted stand-in for the LLM executor (offline pipeline tests): kNN-attribute
        output averaging, written with the workspace toolkit, plus a synthetic event stream."""
        code = (Path(__file__).parent / "fake_agent.py").read_text()
        (ws / "output" / "code").mkdir(parents=True, exist_ok=True)
        (ws / "output" / "code" / "knn_transfer.py").write_text(code)
        p = subprocess.run([sys.executable, str(ws / "output" / "code" / "knn_transfer.py")], cwd=ws,
                           env={**os.environ, "PYTHONPATH": str(ws), "HYDRO_WORKSPACE": str(ws)},
                           capture_output=True, text=True, timeout=1800)
        ev = [{"type": "tool_use", "part": {"tool": "bash", "state": {"status": "completed",
               "input": {"command": "python3 output/code/knn_transfer.py"},
               "output": (p.stdout + p.stderr)[-4000:]}}},
              {"type": "text", "part": {"text": "Wrote output/target_discharge.csv and output/method.md."}}]
        (out / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\n")
        return p.returncode

    def _audit(self, out: Path) -> dict:
        """Look for network access in what the agent *ran* (tool inputs), not in prose it wrote."""
        ev = out / "events.jsonl"
        hits = set()
        if ev.is_file():
            for line in ev.read_text(errors="replace").splitlines():
                if not line.startswith("{"):
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") != "tool_use":
                    continue
                inp = (e.get("part") or {}).get("state", {}).get("input", {})
                text = inp.get("command", "") if isinstance(inp, dict) else ""
                text += " " + (inp.get("url", "") if isinstance(inp, dict) else "")
                for m in NETWORK_PATTERNS.finditer(text):
                    hits.add(m.group(0).lower())
        return {"network_pattern_hits": sorted(hits), "flagged": bool(hits)}

    # ------------------------------------------------------------------ artifacts
    def _targets(self, task_id: str) -> list[str]:
        return list(pd.read_csv(config.TASKS / task_id / self.variant / "task_folder" / "data" /
                                "target_attributes.csv").id)

    def _read_q(self, run_dir: Path) -> Optional[pd.DataFrame]:
        f = run_dir / "output" / "target_discharge.csv"
        if not f.is_file():
            return None
        try:
            df = pd.read_csv(f)
            col = "date" if "date" in df.columns else df.columns[0]
            df.index = pd.to_datetime(df.pop(col))
            return df
        except Exception:
            return None

    def delivered(self, run_dir: Path) -> bool:
        q = self._read_q(Path(run_dir))
        if q is None:
            return False
        meta = json.loads((Path(run_dir) / "rollout.json").read_text()) if (Path(run_dir) / "rollout.json").is_file() else {}
        task_id = meta.get("task_id") or self.task_ids[0]
        need = self._targets(task_id)
        if any(t not in q.columns for t in need):
            return False
        w = q.loc[config.TEST[0]:config.TEST[1], need]
        n_days = (pd.Timestamp(config.TEST[1]) - pd.Timestamp(config.TEST[0])).days + 1
        return len(w) >= n_days and bool(np.isfinite(w.values).all()) and bool((w.values >= 0).all())

    def trajectory(self, run_dir: Path, *, obs_trunc: int) -> str:
        ev = Path(run_dir) / "events.jsonl"
        if not ev.is_file():
            return "(no trajectory)"
        try:
            return self.substrate.read_trajectory(ev, obs_trunc=obs_trunc, max_chars=4_000_000)
        except Exception:
            return ev.read_text(errors="replace")[-200_000:]

    def deliverable_files(self, run_dir: Path) -> dict:
        run_dir = Path(run_dir)
        files = {}
        m = run_dir / "output" / "method.md"
        if m.is_file():
            files["method.md"] = m.read_text(errors="replace")[:60_000]
        q = self._read_q(run_dir)
        if q is not None:
            w = q.loc[config.TEST[0]:config.TEST[1]]
            stats = pd.DataFrame({"mean_mm_d": w.mean(), "p99_mm_d": w.quantile(.99),
                                  "frac_below_0.01": (w < 0.01).mean(), "n_days": w.notna().sum()})
            files["target_discharge_summary.txt"] = (
                "Per-target statistics of output/target_discharge.csv over the prediction window\n"
                + stats.round(3).to_string())
        code = run_dir / "output" / "code"
        if code.is_dir():
            for f in sorted(code.rglob("*.py"))[:8]:
                files[f"code/{f.relative_to(code)}"] = f.read_text(errors="replace")[:12_000]
        if (run_dir / "dcrest_modified").is_dir():
            files["MODEL_STRUCTURE_CHANGED.txt"] = "The rollout modified the dCREST package (see dcrest_modified/)."
        return files

    def deliverable_text(self, run_dir: Path) -> str:
        return "\n\n".join(f"=== {k} ===\n{v}" for k, v in self.deliverable_files(run_dir).items())

    # ------------------------------------------------------------------ scoring (never shown to the agent)
    def score(self, run_dir: Path, task_id: str) -> Optional[dict]:
        from . import camels
        from .metrics import kge, nse

        q = self._read_q(Path(run_dir))
        if q is None:
            return None
        mapping = json.loads((config.HIDDEN / task_id / "mapping.json").read_text())
        targets = self._targets(task_id)
        d = camels.load()
        pos = {g: i for i, g in enumerate(d["ids"])}
        dates = d["dates"]
        w = (dates >= pd.Timestamp(config.TEST[0])) & (dates <= pd.Timestamp(config.TEST[1]))
        obs = d["obs"][w][:, [pos[mapping[t]] for t in targets]]
        sim = q.reindex(dates[w])[targets].values
        r = pd.DataFrame({"nse": nse(sim, obs), "kge": kge(sim, obs)}, index=pd.Index(targets, name="id"))
        r["gauge_id"] = [mapping[t] for t in targets]
        dplf = config.RESULTS / "dpl" / f"{task_id}_test.csv"
        if self.variant != "template" and dplf.is_file():     # compare with the differentiable baseline
            knn = pd.read_csv(dplf, dtype={"gauge_id": str}).set_index("gauge_id").nse.reindex(r.gauge_id).values
        else:
            ref = pd.read_csv(config.RESULTS / "baselines_per_basin.csv", dtype={"gauge_id": str})
            knn = ref[ref.method == "knn_attr"].set_index("gauge_id").nse.reindex(r.gauge_id).values
        audit = json.loads((Path(run_dir) / "rollout.json").read_text()).get("audit", {}) \
            if (Path(run_dir) / "rollout.json").is_file() else {}
        return {"nse_median": float(np.nanmedian(r.nse)), "kge_median": float(np.nanmedian(r.kge)),
                "n_passed": int(np.sum(r.nse.values > knn)), "n_criteria": len(targets),
                "all_pass": False, "audit": audit,
                "per_basin": r.reset_index().to_dict(orient="records")}
