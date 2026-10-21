"""Google Antigravity CLI (``agy``) as the Harness² executor substrate.

Select it with an executor ID of the form ``agy/<model slug>`` (``agy models`` lists the
slugs), e.g. ``hydroagent run --executor agy/gemini-4-argon-high``. The improver keeps its
own channel (OpenCode or Claude CLI), so Harness² still edits the harness with Claude.

What differs from the OpenCode executor
---------------------------------------
* Harness layout. The same c1..c8 tree is rendered as for OpenCode, then delivered the way
  Antigravity reads it: ``AGENTS.md`` (always-on rules), ``.agents/skills/<name>/SKILL.md``
  (on-demand skills) and ``scripts/``. Anything else in the render is listed in the
  rollout record as dropped.
* Permissions. Headless ``agy -p`` cannot ask, so every rollout gets its own settings file
  with explicit rules (``deny`` wins over ``allow``): shell commands are allowed except
  download and install commands, and URL reading, browser actions and MCP tools are
  denied. Web search is not a permission action and cannot be denied; the audit flags any
  use of it.
* Isolated home. Settings live in ``~/.gemini/antigravity-cli/settings.json``. Each rollout
  runs with ``HOME`` pointed at a per-rollout directory: the home's other entries are
  linked, but under ``~/.gemini`` only sign-in files are linked, so the user's global
  ``GEMINI.md`` rules, conversations and memories never reach a rollout and rollouts never
  share state. The user's own files are never written.
* Network fence. Each rollout gets its own local CONNECT proxy that only tunnels to the
  Google hosts ``agy`` needs (``ALLOW_HOSTS``) and logs every host it refuses. The agent's
  shell inherits the proxy, so ``urllib``/``requests``/``curl`` calls to any other host
  fail and show up in the audit. Raw sockets bypass it; it is a soft fence.
* Cost. ``agy`` draws on the Google account's subscription credits, not per-token billing,
  so rollouts carry no dollar cost; only the improver's API spend is metered.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import select
import shutil
import signal
import socket
import socketserver
import subprocess
import tempfile
import threading
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

PREFIX = "agy/"
RULES_MAX = 12_000          # Antigravity's documented cap on always-on rules text

# Hosts the proxy tunnels to. Exact names only: a suffix such as googleapis.com would also
# open public Cloud Storage (storage.googleapis.com), where CAMELS-US discharge is mirrored.
# Extend with HYDRO_AGY_ALLOW_EXTRA once `hydroagent agy-smoke` shows what agy needs.
ALLOW_HOSTS = (
    "cloudcode-pa.googleapis.com", "daily-cloudcode-pa.googleapis.com",
    "daily-cloudcode-pa.sandbox.googleapis.com",
    "oauth2.googleapis.com", "accounts.google.com", "play.googleapis.com",
    "antigravity-unleash.goog",                       # feature flags
)
# Hosts agy itself must reach but that also serve data (www.googleapis.com: agy's account
# check at /oauth2/v2/userinfo, and Cloud Storage objects at /storage/v1/...). Tunnelled only
# when the connecting process is the agy binary itself, never the agent's shell commands.
AGY_ONLY_HOSTS = ("www.googleapis.com", "lh3.googleusercontent.com")   # account check, profile picture
# Denied hosts that would carry data: always flagged by the audit, even though they are Google's
LEAK_HOSTS = re.compile(r"(^|\.)(storage\.googleapis\.com|www\.googleapis\.com|googleusercontent\.com|"
                        r"drive\.google\.com|docs\.google\.com|dl\.google\.com)$")
GOOGLE_HOSTS = re.compile(r"(^|\.)(google\.com|googleapis\.com|gstatic\.com|googleusercontent\.com|google)$")

# Files copied into each rollout's home (agy 1.4.3 needs no others; tokens are in the keychain)
GEMINI_COPY = ("google_accounts.json", "installation_id", "oauth_creds.json", "user_id")
CLI_COPY = ("installation_id", "jetski_state.pbtxt", "last_check.timestamp")
HOME_LINKS = ("Library/Keychains",)

ALLOW = ["command(*)"]
DENY = [
    "read_url(*)",
    "execute_url(*)",
    "mcp(*)",
    r"command(regex:(?s).*\b(curl|wget|ssh|scp|sftp|rsync|ncat|telnet|ftp|gsutil|gcloud|agy)\b.*)",
    r"command(regex:(?s).*\bgit\s+(clone|fetch|pull|submodule)\b.*)",
    r"command(regex:(?s).*\b(pip3?|uv|conda|mamba|micromamba|brew|npm|pnpm|yarn|apt|apt-get)\s+(install|add)\b.*)",
]
WEB_TOOL = None   # set below from DENY_TOOLS: the tools the hook refuses are the ones the audit counts
AGY_OWN = re.compile(r"[^\s\"']*/\.agyhome/[^\s\"']*")
# Tools a PreToolUse hook hard-denies. search_web runs on Google's side, so neither the
# permission rules nor the fence can stop it; agy 1.4.3 answered "USGS 02177000" with the
# gauge's river and town. One matcher group per name, since the matcher syntax is loosely
# documented; the regex forms catch browser and URL tools under other names.
DENY_TOOLS = ("search_web", "web_search", "read_url_content", "read_url", "view_content_chunk",
              "browser_subagent", ".*browser.*", ".*_url.*", ".*url_.*", ".*web.*", ".*content_chunk.*",
              ".*mcp.*")
HOOK_REASON = ("Web access is disabled for this task: the catchments are deliberately anonymous. "
               "Work only with the data in the workspace.")
WEB_TOOL = re.compile(r"^(?:" + "|".join(DENY_TOOLS) + r")$", re.IGNORECASE)
FAILURE = re.compile(r"quota|rate.?limit|\b429\b|resource.?exhausted|unauthenticated|authentication required|"
                     r"not (signed|logged) in|login required|credits? (exhausted|exceeded|remaining: 0)",
                     re.IGNORECASE)
# Refused hosts agy itself tries but does not need (telemetry etc.); filled in from the smoke test.
BENIGN_HOSTS = ("antigravity-cli-auto-updater-974169037036.us-central1.run.app",) + tuple(
    h.strip().lower() for h in os.environ.get("HYDRO_AGY_BENIGN_HOSTS", "").split(",") if h.strip())
# Credentials that would let the agent's shell call Google models (search grounding) directly
STRIP_ENV = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_GENAI_USE_VERTEXAI",
             "GOOGLE_CLOUD_PROJECT", "VERTEX_API_KEY")


class AgyFailure(BaseException):
    """agy did not run the task (sign-in, quota, empty output). A BaseException on purpose:
    Harness² records ordinary exceptions as dead rollouts and carries on, which would let a
    spent quota silently decide harness selection. This stops the run; it resumes later."""


_LIVE: set = set()             # process groups of running agy rollouts
_LIVE_LOCK = threading.RLock()


def kill_live(sig=signal.SIGTERM) -> None:
    with _LIVE_LOCK:
        groups = list(_LIVE)
    for g in groups:
        _killpg(g, sig)


def install_cleanup() -> None:
    """agy and the agent's children run in their own process groups, so a signal to the run
    does not reach them. On SIGTERM (budget guard, early stop) or Ctrl-C end them and exit at
    once, without waiting for queued rollouts or improver sessions."""
    if getattr(install_cleanup, "_done", False):
        return
    install_cleanup._done = True
    atexit.register(kill_live)

    def on_signal(signum, frame):
        # exit now: a KeyboardInterrupt would wait for busy pool threads to finish their work.
        # Unfinished rollouts have no rollout.json and simply run again when the run resumes.
        kill_live()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    try:
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
    except ValueError:         # not the main thread
        pass


def is_agy(model: Optional[str]) -> bool:
    return bool(model) and str(model).startswith(PREFIX)


def slug(model: str) -> str:
    return str(model)[len(PREFIX):]


def substrate_name(model: str, effort: Optional[str] = None) -> str:
    """Namespace token: one per model and effort, so cached rollouts never cross models."""
    tag = re.sub(r"[^A-Za-z0-9.]+", "-", slug(model)).strip("-")
    return f"ag-{tag}" + (f"-{effort}" if effort else "")


def binary() -> Path:
    env = os.environ.get("HYDRO_AGY_BIN")
    if env:
        return Path(env).expanduser()
    found = shutil.which("agy")
    return Path(found) if found else Path.home() / ".local" / "bin" / "agy"


def ws_root() -> Path:
    """Parent of all agy workspaces: outside the hydroagent cache, so the hidden target
    mapping and the full CAMELS record are not next door. Trust it once in `agy`."""
    return Path(os.environ.get("HYDRO_AGY_WS_ROOT") or Path(tempfile.gettempdir()) / "hydro_agy_ws")


def allow_hosts() -> tuple:
    base = os.environ.get("HYDRO_AGY_ALLOW_HOSTS")
    hosts = [h.strip().lower() for h in (base.split(",") if base else ALLOW_HOSTS) if h.strip()]
    hosts += [h.strip().lower() for h in os.environ.get("HYDRO_AGY_ALLOW_EXTRA", "").split(",") if h.strip()]
    refused = [h for h in hosts if LEAK_HOSTS.search(h)]
    if refused:
        import warnings
        warnings.warn(f"not opening data-serving hosts in the agy fence: {', '.join(refused)}")
    return tuple(dict.fromkeys(h for h in hosts if not LEAK_HOSTS.search(h)))


def settings(ws: Optional[Path] = None, home: Optional[Path] = None) -> dict:
    """Permission rules. agy 1.4.3 runs in "request-review" mode, where even writes inside the
    workspace wait for approval and a headless run refuses them, so the workspace is granted
    explicitly (under both its given and resolved path: /var is /private/var on macOS)."""
    allow = list(ALLOW)
    if ws is not None:
        allow += [f"write_file({p})" for p in dict.fromkeys((str(ws), os.path.realpath(ws)))]
    if home is not None:
        # agy saves long tool outputs under ~/.gemini/antigravity-cli/brain and tells the agent
        # to read them there; outside the workspace that read is refused unless granted, and the
        # refusal ends the session (2026-10-21: two rollouts stopped after 2 minutes)
        brain = Path(home) / ".gemini" / "antigravity-cli" / "brain"
        allow += [f"read_file({p})" for p in dict.fromkeys((str(brain), os.path.realpath(brain)))]
    return {"permissions": {"allow": allow, "deny": list(DENY)}, "allowNonWorkspaceAccess": False}


# ---------------------------------------------------------------- isolated home
def isolated_home(run_dir: Path, real_home: Optional[Path] = None, ws: Optional[Path] = None) -> Path:
    """A per-rollout HOME holding only what agy needs: copies of the sign-in and onboarding
    files under ~/.gemini, this module's settings, the no-web hook, and one link to
    ~/Library/Keychains, where agy keeps its sign-in token (agy 1.4.3 finds the keychain
    through $HOME). Nothing else points back into the real home, so a rollout cannot write
    the user's files, share state with another rollout, or find other credentials such as
    ~/.config/gcloud by a relative path."""
    real = Path(real_home or Path.home()).resolve()
    # resolved, so the paths agy reports (/private/var/...) match HOME (macOS: /var -> /private/var)
    home = Path(os.path.realpath(run_dir)) / ".agyhome"
    if home.resolve() == real:
        raise ValueError(f"refusing to build an isolated home at {home}")
    if home.is_symlink():
        raise ValueError(f"{home} is a symlink")
    if home.exists():
        shutil.rmtree(home)
    home.mkdir(parents=True)
    gem_real, gem = real / ".gemini", home / ".gemini"
    cli_real, cli = gem_real / "antigravity-cli", gem / "antigravity-cli"
    cli.mkdir(parents=True)
    for rel in HOME_LINKS:
        if (real / rel).exists():
            (home / rel).parent.mkdir(parents=True, exist_ok=True)
            (home / rel).symlink_to(real / rel)
    for src, dst, names in ((gem_real, gem, GEMINI_COPY + _extra_links()), (cli_real, cli, CLI_COPY)):
        for name in names:
            f = src / name
            if f.is_file() and not f.is_symlink():
                shutil.copy2(f, dst / name)
    base = {}
    if (cli_real / "settings.json").is_file():
        try:     # keep the user's auth, trust and display choices; replace permission rules only
            base = json.loads((cli_real / "settings.json").read_text())
        except ValueError:
            base = {}
    _write_hooks(gem / "config")
    base.pop("permissions", None)
    base.update(settings(ws, home))
    (cli / "settings.json").write_text(json.dumps(base, indent=1))
    return home


def _write_hooks(cfg: Path) -> None:
    """Global hooks (~/.gemini/config/hooks.json of the isolated home, outside the workspace)."""
    cfg.mkdir(parents=True, exist_ok=True)
    script = cfg / "deny_web.sh"
    import shlex
    reply = json.dumps({"decision": "deny", "reason": HOOK_REASON})
    script.write_text(f"#!/bin/sh\ncat > /dev/null\nprintf '%s\\n' {shlex.quote(reply)}\n")
    script.chmod(0o755)
    (cfg / "hooks.json").write_text(json.dumps({"hydroagent-no-web": {"PreToolUse": [
        {"matcher": m, "hooks": [{"type": "command", "command": f"sh {shlex.quote(str(script))}", "timeout": 10}]}
        for m in DENY_TOOLS]}}, indent=1))
    for f in (script, cfg / "hooks.json"):
        f.chmod(0o444 if f.suffix == ".json" else 0o555)


def hook_fingerprint(home: Path) -> str:
    """Digest of the two hook files (agy itself adds other files to the same folder)."""
    import hashlib
    h = hashlib.sha256()
    for name in ("hooks.json", "deny_web.sh"):
        f = Path(home) / ".gemini" / "config" / name
        h.update(name.encode() + (f.read_bytes() if f.is_file() else b"<missing>"))
    return h.hexdigest()[:16]


def _extra_links() -> tuple:
    """HYDRO_AGY_LINK_EXTRA: more top-level ~/.gemini files to copy into each rollout's home.
    Plain names only, so no entry can reach into antigravity-cli."""
    out = []
    for x in os.environ.get("HYDRO_AGY_LINK_EXTRA", "").split(","):
        x = x.strip()
        if x and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*", x) and x != "antigravity-cli" and ".." not in x:
            out.append(x)
    return tuple(out)


# ---------------------------------------------------------------- allowlisting proxy
class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            line = self.rfile.readline(65537).decode("latin-1")
            while True:
                h = self.rfile.readline(65537)
                if h in (b"\r\n", b"\n", b""):
                    break
        except OSError:
            return
        parts = line.split()
        if len(parts) < 2:
            return
        method, target = parts[0].upper(), parts[1]
        if method == "CONNECT":
            host, _, port = target.rpartition(":")
        else:
            u = urllib.parse.urlsplit(target)
            host, port = u.hostname or "", str(u.port or 80)
        host = host.strip("[]").lower()
        ok = method == "CONNECT" and port.isdigit() and host in (self.server.allowed | self.server.agy_only)
        label = host
        # with a trusted executable set, every tunnel must come from it (the agy binary): the
        # agent's shell could otherwise reach Google's model APIs with the user's credentials
        if ok and (self.server.trusted or host in self.server.agy_only):
            exes = {_exe(pid) for pid in _client_pids(self.client_address[1])}
            ok = bool(exes) and exes <= self.server.trusted
            if not ok:
                label = f"{host} (from {', '.join(sorted(Path(e).name for e in exes if e)) or '?'})"
            elif host in self.server.agy_only:
                label = f"{host} (agy)"
        self.server.record(label, ok)
        if not ok:
            self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            return
        try:
            up = socket.create_connection((host, int(port)), timeout=30)
        except OSError:
            self.wfile.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            return
        self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        self.wfile.flush()
        a, b = self.connection, up
        a.settimeout(None)
        b.settimeout(None)
        try:
            while True:
                r, _, x = select.select([a, b], [], [a, b], 3600)
                if x or not r:
                    break
                for s in r:
                    data = s.recv(65536)
                    if not data:
                        return
                    (b if s is a else a).sendall(data)
        except OSError:
            pass
        finally:
            b.close()


def _client_pids(peer_port: int) -> set:
    """Processes holding the client end of a loopback connection (macOS/Linux, via lsof)."""
    try:
        out = subprocess.run(["lsof", "-nP", "-a", f"-iTCP@127.0.0.1:{int(peer_port)}", "-Fp"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {int(x[1:]) for x in out.splitlines() if x[1:].isdigit() and x.startswith("p")} - {os.getpid()}


def _exe(pid: int) -> str:
    """Executable path of a process (macOS libproc; /proc elsewhere)."""
    try:
        if os.path.exists(f"/proc/{pid}/exe"):
            return os.path.realpath(f"/proc/{pid}/exe")
        import ctypes
        import ctypes.util
        lib = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
        buf = ctypes.create_string_buffer(4096)
        n = lib.proc_pidpath(int(pid), buf, ctypes.sizeof(buf))
        return os.path.realpath(buf.value[:n].decode()) if n > 0 else ""
    except Exception:
        return ""


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class Fence:
    """A local CONNECT proxy for one rollout: tunnels only to ``allowed`` hosts, counts the rest."""

    def __init__(self, allowed: Iterable[str], agy_only: Iterable[str] = (), trusted: Iterable[str] = ()):
        self.srv = _Server(("127.0.0.1", 0), _Handler)
        self.srv.allowed = frozenset(allowed)
        self.srv.agy_only = frozenset(agy_only)
        self.srv.trusted = frozenset(os.path.realpath(t) for t in trusted)
        self.lock = threading.Lock()
        self.hits = {"allowed": Counter(), "denied": Counter()}

        def record(host, ok):
            with self.lock:
                self.hits["allowed" if ok else "denied"][host] += 1
        self.srv.record = record
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.srv.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.srv.shutdown()
        self.srv.server_close()

    def summary(self) -> dict:
        with self.lock:
            return {k: dict(v) for k, v in self.hits.items()}


# ---------------------------------------------------------------- staging and launch
def stage(render: Path, ws: Path) -> dict:
    """Deliver a rendered OpenCode-style harness in the layout Antigravity reads."""
    render, ws = Path(render), Path(ws)
    info = {"agents_md_chars": 0, "skills": [], "subagents": [], "dropped": []}
    if (render / "AGENTS.md").is_file():
        shutil.copy(render / "AGENTS.md", ws / "AGENTS.md")
        info["agents_md_chars"] = len((ws / "AGENTS.md").read_text())
    if (render / "scripts").is_dir():
        shutil.copytree(render / "scripts", ws / "scripts", dirs_exist_ok=True)
    sk = render / ".opencode" / "skills"
    if sk.is_dir():
        for d in sorted(p for p in sk.iterdir() if (p / "SKILL.md").is_file()):
            shutil.copytree(d, ws / ".agents" / "skills" / d.name, dirs_exist_ok=True)
            info["skills"].append(d.name)
    ag = render / ".opencode" / "agent"
    if ag.is_dir():
        for f in sorted(ag.glob("*.md")):
            (ws / ".agents" / "agents").mkdir(parents=True, exist_ok=True)
            shutil.copy(f, ws / ".agents" / "agents" / f.name)
            info["subagents"].append(f.stem)
    kept = {"AGENTS.md", "scripts", ".opencode/skills", ".opencode/agent"}
    for p in sorted(render.rglob("*")):
        rel = p.relative_to(render).as_posix()
        if p.is_file() and not any(rel == k or rel.startswith(k + "/") for k in kept):
            info["dropped"].append(rel)
    info["rules_over_cap"] = info["agents_md_chars"] > RULES_MAX
    return info


def env_for(ws: Path, home_parent: Path, *, venv_bin: str, proxy: Optional[str],
            base: Optional[dict] = None) -> dict:
    """Environment for one agy process. ``home_parent`` receives the isolated home: rollouts
    put it next to the workspace (outside the hydroagent cache, whose path would otherwise
    show the agent where the hidden target mapping lives)."""
    inherited = {k: v for k, v in (base if base is not None else os.environ).items()
                 if k not in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY",
                              "HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY",
                              "all_proxy", "NO_PROXY", "no_proxy") + STRIP_ENV}
    env = {**inherited,
           "HOME": str(isolated_home(home_parent, ws=ws)),
           "PATH": f"{venv_bin}:{inherited.get('PATH', '')}",
           "PYTHONPATH": str(ws), "HYDRO_WORKSPACE": str(ws)}
    if proxy:
        env.update({"HTTPS_PROXY": proxy, "HTTP_PROXY": proxy, "https_proxy": proxy, "http_proxy": proxy,
                    "ALL_PROXY": proxy, "all_proxy": proxy,
                    "NO_PROXY": "localhost,127.0.0.1", "no_proxy": "localhost,127.0.0.1"})
    return env


def argv(prompt: str, model: str, timeout: int, effort: Optional[str] = None,
         log_file: Optional[Path] = None) -> list[str]:
    cmd = [str(binary()), "-p", prompt, "--output-format", "stream-json",
           "--model", slug(model) if is_agy(model) else model,
           "--print-timeout", f"{max(1, int(timeout) // 60)}m"]
    if effort:
        cmd += ["--effort", effort]          # low | medium | high | xhigh | max (agy 1.4)
    if log_file:
        cmd += ["--log-file", str(log_file)]
    return cmd


def _killpg(pgid: int, sig=signal.SIGTERM) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def launch(ws: Path, out: Path, prompt: str, model: str, timeout: int, *, venv_bin: str,
           network_fence: bool = True, effort: Optional[str] = None) -> int:
    """Run one headless rollout. Writes events.jsonl, stderr.log and proxy.json into ``out``.

    agy and everything the agent starts (background calibrations included) run in their own
    process group, which is terminated when agy exits or times out."""
    out = Path(out)
    fence = Fence(allow_hosts(), AGY_ONLY_HOSTS, [str(binary())]) if network_fence else None
    rc = None
    try:
        if fence:
            fence.__enter__()
        env = env_for(ws, Path(ws).parent, venv_bin=venv_bin, proxy=fence.url if fence else None)
        home = Path(env["HOME"])
        hook0 = hook_fingerprint(home)
        with open(out / "events.jsonl", "w") as so, open(out / "stderr.log", "w") as se:
            # stdin closed: print mode otherwise waits on an open stdin when not on a TTY
            p = subprocess.Popen(argv(prompt, model, timeout, effort, out / "agy.log"), cwd=ws, env=env,
                                 stdin=subprocess.DEVNULL,
                                 stdout=so, stderr=se, start_new_session=True)
            with _LIVE_LOCK:
                _LIVE.add(p.pid)
            try:
                rc = p.wait(timeout=int(timeout) + 180)
            except subprocess.TimeoutExpired:
                _killpg(p.pid)
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    _killpg(p.pid, signal.SIGKILL)
                    p.wait()
                se.write(f"\n[hydroagent] agy rollout wall-timeout after {timeout}s\n")
                rc = -9
            finally:
                _killpg(p.pid)          # leftover children of the agent
                with _LIVE_LOCK:
                    _LIVE.discard(p.pid)
        return rc
    finally:
        if fence:
            (out / "proxy.json").write_text(json.dumps(fence.summary(), indent=1))
            fence.__exit__(None, None, None)
        try:
            same = hook_fingerprint(home) == hook0
        except (OSError, NameError):
            same = False
        (out / "hook_check.json").write_text(json.dumps({"hooks_intact": same}))
        try:
            shutil.rmtree(home, ignore_errors=True)
        except NameError:
            pass


QUOTA = re.compile(r"quota|rate.?limit|\b429\b|resource.?exhausted|credits? (exhausted|exceeded|remaining: 0)",
                   re.IGNORECASE)


def check_ran(out: Path, rc: int) -> dict:
    """Raise AgyFailure when agy did not actually work on the task.

    agy 1.4.3 reports errors in its log file and the result, not on stderr. Its log also
    says "not logged into Antigravity" at every start, before the keychain token is read,
    so sign-in errors count only when nothing was done; after work began, only a quota or
    rate-limit stop aborts the run (a stalled model stream is an ordinary bad rollout)."""
    out = Path(out)
    ev = out / "events.jsonl"
    worked = [b for b in steps(ev) if b.get("step_type") in ("tool", "agent_response") or tool_call(b)[0]]
    res = result(ev)
    status = str(res.get("status", "")).upper()
    err = (out / "stderr.log").read_text(errors="replace")[-4000:] if (out / "stderr.log").is_file() else ""
    tail = ""
    if (out / "agy.log").is_file():
        tail = "\n".join(ln for ln in (out / "agy.log").read_text(errors="replace").splitlines()[-400:]
                         if ln.startswith(("E", "F")))
    final = "\n".join([str(_first(res, _TEXT) or ""), str(res.get("error") or "")])
    if not worked and status != "SUCCESS":
        raise AgyFailure(f"agy exit {rc}, status {status or '-'}, nothing done; {(err + final).strip()[-600:]}")
    why = QUOTA.search("\n".join([err, final, tail])) if status != "SUCCESS" else None
    if why:
        raise AgyFailure(f"agy stopped on quota or rate limit (exit {rc}, {len(worked)} steps): {why.group(0)}; "
                         f"{(err + final).strip()[-600:]}")
    # agy refuses tool calls it would ask about ("user denied permission"); a session that ends
    # on such a refusal stopped because of our settings, not the agent, and is not a valid rollout
    tools = [tool_call(b) for b in steps(ev) if tool_call(b)[0]]
    soft = [f"{t}({json.dumps(a, default=str)[:160]})" for t, a, _, e in tools if e and "user denied permission" in str(e)]
    ended = bool(tools) and bool(tools[-1][3]) and "user denied permission" in str(tools[-1][3])
    return {"steps": len(worked), "status": status, "error": str(res.get("error") or "")[:300],
            "soft_denials": soft[:10], "ended_on_denial": ended}


# ---------------------------------------------------------------- event stream
def _events(path: Path) -> Iterable[dict]:
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for ln in raw.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                ev = json.loads(ln)
            except ValueError:
                continue
            if isinstance(ev, dict):
                yield ev


def _first(d: dict, keys: tuple) -> object:
    for k in keys:
        v = d.get(k)
        if v not in (None, "", [], {}):
            return v
    return None


# agy 1.4.3 stream-json: {"event": "init"|"step_update"|"result", "<event>": {...}}. A step_update
# carries step_index, step_type (user_input | agent_response | tool), state (ACTIVE | DONE |
# ERROR), and either text_delta (streamed text, to be joined) or tool_name plus tool_info
# {name, parameters, output | error}. The fallback keys cover the documented older shape.
_ID = ("step_index", "stepIndex", "id", "step_id", "stepId", "index")
_TEXT = ("text", "response", "content", "message", "output_text")
_THINK = ("thinking", "thought", "reasoning")
_TOOL = ("tool_name", "toolName", "tool", "name", "action")
_ARGS = ("parameters", "args", "input", "arguments", "params", "command", "target")
_OUT = ("output", "result", "observation", "stdout", "tool_output", "toolOutput")


def _kind(ev: dict) -> str:
    return str(ev.get("event") or ev.get("type") or "")


def _body(ev: dict) -> dict:
    k = _kind(ev)
    if k and isinstance(ev.get(k), dict):
        return ev[k]
    for key in ("step", "data", "update", "payload"):
        if isinstance(ev.get(key), dict):
            return ev[key]
    return ev


def steps(path: Path) -> list[dict]:
    """step_update events merged per step (later fields win, text deltas are joined), in order."""
    order, merged = [], {}
    for n, ev in enumerate(_events(path)):
        if _kind(ev) != "step_update":
            continue
        b = _body(ev)
        sid = _first(b, _ID)
        sid = ("anon", n) if sid is None else ("id", str(b.get("conversation_id", "")), str(sid))
        cur = merged.get(sid)
        if cur is None:
            order.append(sid)
            cur = merged[sid] = {}
        for k, v in b.items():
            if k != "text_delta":
                cur[k] = v
        if isinstance(b.get("text_delta"), str):
            cur["text"] = cur.get("text", "") + b["text_delta"]
    return [merged[x] for x in order]


def tool_call(b: dict) -> tuple:
    """(tool name, arguments, output, error) of one merged step; name is '' for non-tool steps."""
    info = b.get("tool_info") if isinstance(b.get("tool_info"), dict) else {}
    tool = b.get("tool_name") or info.get("name") or ("" if b.get("step_type") else _first(b, _TOOL)) or ""
    args = info.get("parameters") if info else _first(b, _ARGS)
    out = info.get("output") if info else _first(b, _OUT)
    err = info.get("error") if info else b.get("error")
    if isinstance(err, dict):
        err = err.get("message") or json.dumps(err)
    return str(tool), args, out, err


def _clip(s, n: int) -> str:
    s = "" if s is None else (s if isinstance(s, str) else json.dumps(s, default=str, ensure_ascii=False))
    s = s.replace("\x00", "")
    return s if n <= 0 or len(s) <= n else s[:n] + f" …[{len(s) - n} chars truncated]"


def render_events(path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 400) -> str:
    """Render ``agy --output-format stream-json`` into Harness²'s trajectory grammar."""
    lines: list[str] = []
    evs = list(_events(path))
    for ev in evs:
        if _kind(ev) == "init":
            b = _body(ev)
            lines.append(f"[INIT] model={b.get('model', '?')} cwd={b.get('cwd', '?')}")
            break
    for b in steps(path):
        if b.get("step_type") == "user_input":
            continue
        tool, args, out, err = tool_call(b)
        think = _first(b, _THINK)
        if think:
            lines.append(f"[THINKING] {_clip(think, obs_trunc)}")
        if tool:
            lines.append(f"[ACTION] {tool}({_clip(args, args_trunc)})")
            if err or str(b.get("state", b.get("status", ""))).upper() in ("ERROR", "FAILED", "DENIED"):
                lines.append(f"[OBSERVATION ERROR] {_clip(err or out, obs_trunc)}")
            elif out is not None:
                lines.append(f"[OBSERVATION] {_clip(out, obs_trunc)}")
        else:
            text = _first(b, _TEXT)
            if text:
                lines.append(f"[ASSISTANT] {_clip(text, obs_trunc)}")
    for ev in evs:
        if _kind(ev) == "result":
            b = _body(ev)
            lines.append(f"[RESULT] status={b.get('status', '?')} {_clip(_first(b, _TEXT), obs_trunc)}")
    rendered = "\n".join(lines).strip()
    if not rendered:
        return "(no parseable events in trajectory)"
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars] + f"\n…[{len(rendered) - max_chars} chars truncated]"
    return rendered


def audit(path: Path, patterns: re.Pattern) -> dict:
    """Network patterns in executed commands, any web-facing tool, and hosts the fence refused.

    ``flagged`` is None (unknown, not clean) when the stream has events but no recognisable
    tool steps, i.e. when the parser does not understand this agy version's schema."""
    path = Path(path)
    hits, web, refused, n_tools = set(), set(), set(), 0
    st = steps(path)
    for b in st:
        tool, args, _, err = tool_call(b)
        if not tool or args is None:
            continue
        n_tools += 1
        if WEB_TOOL.match(tool) and err and ("denied by pre-tool hook" in str(err) or HOOK_REASON[:30] in str(err)):
            refused.add(tool)       # attempted, refused by the hook: no data reached the agent
            continue
        text = args if isinstance(args, str) else json.dumps(args, default=str) if args is not None else ""
        text = AGY_OWN.sub(" ", text)      # agy saves long outputs under its home and reads them back
        if WEB_TOOL.match(tool):
            web.add(tool)
        for m in patterns.finditer(text):
            hits.add(m.group(0).lower())
    denied, leak = [], []
    pj = path.parent / "proxy.json"
    if pj.is_file():
        denied = sorted(json.loads(pj.read_text()).get("denied", {}))
        leak = [h for h in denied if h not in BENIGN_HOSTS]
    hc = path.parent / "hook_check.json"
    tampered = hc.is_file() and not json.loads(hc.read_text()).get("hooks_intact", True)
    flagged = bool(hits or web or leak or tampered)
    if not flagged and n_tools == 0 and any(_kind(e) == "step_update" for e in _events(path)):
        flagged = None
    return {"network_pattern_hits": sorted(hits), "web_tools_used": sorted(web), "web_tools_refused": sorted(refused),
            "proxy_denied": denied, "proxy_denied_flagged": leak, "hooks_tampered": tampered,
            "tool_steps": n_tools, "flagged": flagged}


def result(path: Path) -> dict:
    for ev in _events(path):
        if _kind(ev) == "result":
            return _body(ev)
    return {}


def result_text(path: Path) -> str:
    return str(_first(result(path), _TEXT) or "")


# ---------------------------------------------------------------- substrate
def substrate(model: str, effort: Optional[str] = None):
    """An OpenCode-surface substrate with its own namespace token and trajectory reader.

    Harness² edits the same c1..c8 tree for both executors; only delivery (``stage``),
    launch and the event grammar differ. The name carries the model (and effort), so
    namespaces (``camels-calib-ag-gemini-4-argon-high-agg3``), replicate folders and the
    reflection library never mix executors."""
    from harness2.substrate.opencode import OpencodeSubstrate

    class AntigravitySubstrate(OpencodeSubstrate):
        name = substrate_name(model, effort)

        def read_trajectory(self, stream_path, *, obs_trunc, max_chars, args_trunc=400):
            return render_events(stream_path, obs_trunc=obs_trunc, max_chars=max_chars,
                                 args_trunc=args_trunc)

        def preflight(self) -> None:
            b = binary()
            if not b.is_file():
                raise RuntimeError(f"Antigravity CLI not found at {b}; install with "
                                   "`curl -fsSL https://antigravity.google/cli/install.sh | bash`")

    return AntigravitySubstrate()
