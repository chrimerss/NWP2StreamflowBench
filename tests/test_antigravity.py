"""Antigravity CLI executor: harness delivery, isolated settings, fence, event parsing, audit.

The unit tests need no data, no network and no `agy`. The rollout test drives the real
adapter path with a stand-in `agy` binary and needs the pilot task folder
(`hydroagent prepare`).
"""
import json
import os
import re
import shutil
import socket
import stat
import textwrap
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from hydroagent import antigravity as AG
from hydroagent import config


def _events(path, evs):
    path.write_text("\n".join(json.dumps(e) for e in evs) + "\n")
    return path


def test_stage_delivers_rules_skills_scripts(tmp_path):
    r, ws = tmp_path / "render", tmp_path / "ws"
    (r / ".opencode" / "skills" / "regionalization").mkdir(parents=True)
    (r / ".opencode" / "skills" / "regionalization" / "SKILL.md").write_text("---\nname: regionalization\n---\nx")
    (r / ".opencode" / "agent").mkdir(parents=True)
    (r / ".opencode" / "agent" / "calibrator.md").write_text("calibrate")
    (r / ".opencode" / "plugin").mkdir(parents=True)
    (r / ".opencode" / "plugin" / "hook.ts").write_text("x")
    (r / "scripts").mkdir()
    (r / "scripts" / "hydrokit.py").write_text("print(1)")
    (r / "AGENTS.md").write_text("rules " * 100)
    ws.mkdir()
    info = AG.stage(r, ws)
    assert (ws / "AGENTS.md").read_text() == "rules " * 100
    assert (ws / ".agents" / "skills" / "regionalization" / "SKILL.md").is_file()
    assert (ws / ".agents" / "agents" / "calibrator.md").is_file()
    assert (ws / "scripts" / "hydrokit.py").is_file()
    assert not (ws / ".opencode").exists()
    assert info == {"agents_md_chars": 600, "skills": ["regionalization"], "subagents": ["calibrator"],
                    "dropped": [".opencode/plugin/hook.ts"], "rules_over_cap": False}


def test_isolated_home_copies_signin_files_only(tmp_path, monkeypatch):
    monkeypatch.delenv("HYDRO_AGY_LINK_EXTRA", raising=False)
    real = tmp_path / "home"
    cli = real / ".gemini" / "antigravity-cli"
    (cli / "conversations").mkdir(parents=True)
    (cli / "settings.json").write_text(json.dumps({"trustedWorkspaces": ["/ws"],
                                                   "permissions": {"allow": ["read_url(*)"]}}))
    for name in ("installation_id", "jetski_state.pbtxt", "conversation_summaries.db",
                 "conversation_summaries.db-wal", "conversation_summaries.db-shm"):
        (cli / name).write_text(name)
    (real / ".gemini" / "google_accounts.json").write_text("{}")
    (real / ".gemini" / "GEMINI.md").write_text("user's global rules")
    (real / "Library" / "Keychains").mkdir(parents=True)
    (real / ".config" / "gcloud").mkdir(parents=True)
    (real / ".config" / "gcloud" / "application_default_credentials.json").write_text("secret")
    home = AG.isolated_home(tmp_path / "run", real_home=real, ws=tmp_path / "ws")
    g = home / ".gemini"
    s = json.loads((g / "antigravity-cli" / "settings.json").read_text())
    assert s["trustedWorkspaces"] == ["/ws"]
    assert f"write_file({tmp_path / 'ws'})" in s["permissions"]["allow"]
    assert home == Path(os.path.realpath(home))                          # resolved: /var is /private/var
    assert f"read_file({home / '.gemini' / 'antigravity-cli' / 'brain'})" in s["permissions"]["allow"]
    assert "read_url(*)" in s["permissions"]["deny"] and "read_url(*)" not in s["permissions"]["allow"]
    assert (g / "antigravity-cli" / "jetski_state.pbtxt").read_text() == "jetski_state.pbtxt"
    assert (g / "google_accounts.json").is_file()
    links = [p.relative_to(home).as_posix() for p in home.rglob("*") if p.is_symlink()]
    assert links == ["Library/Keychains"]                                # everything else is a copy
    for absent in ("GEMINI.md", "antigravity-cli/conversations", "antigravity-cli/conversation_summaries.db",
                   "antigravity-cli/conversation_summaries.db-wal", "antigravity-cli/conversation_summaries.db-shm"):
        assert not (g / absent).exists(), absent
    assert not (home / ".config").exists()
    # writes in the isolated home never reach the user's files
    (g / "antigravity-cli" / "jetski_state.pbtxt").write_text("changed")
    assert (cli / "jetski_state.pbtxt").read_text() == "jetski_state.pbtxt"
    monkeypatch.setenv("HYDRO_AGY_LINK_EXTRA", "antigravity-cli,../x,./antigravity-cli/settings.json,"
                                               "x/../antigravity-cli/settings.json,GEMINI.md")
    assert AG._extra_links() == ("GEMINI.md",)
    assert json.loads((cli / "settings.json").read_text())["permissions"] == {"allow": ["read_url(*)"]}
    shutil.rmtree(home)
    assert (cli / "installation_id").read_text() == "installation_id"


def test_deny_rules_cover_downloads_without_false_positives():
    regexes = [re.compile(r[len("command(regex:"):-1]) for r in AG.DENY if r.startswith("command(regex:")]
    blocked = ["curl -sS https://waterservices.usgs.gov", "cd x && wget http://a", "pip install dataretrieval",
               "uv pip install x", "git clone https://github.com/a/b", "gsutil cp gs://b/x .",
               "python3 -m pip install x", "uv add requests", "cd output\ncurl https://zenodo.org/x"]
    allowed = ["python3 scripts/hydrokit.py calibrate --ids all", "ls output", "tail -n 20 output/logs/cal.log",
               "ncdump -h forcing.nc", "uv run python x.py", "python3 -c 'import numpy as np'"]
    assert all(any(r.fullmatch(c) for r in regexes) for c in blocked)
    assert not [c for c in allowed if any(r.fullmatch(c) for r in regexes)]


def test_fence_never_opens_data_hosts(monkeypatch):
    monkeypatch.setenv("HYDRO_AGY_ALLOW_EXTRA", "www.googleapis.com,storage.googleapis.com,foo.googleapis.com")
    with pytest.warns(UserWarning):
        hosts = AG.allow_hosts()
    assert "foo.googleapis.com" in hosts and not [h for h in hosts if AG.LEAK_HOSTS.search(h)]


def test_fence_tunnels_allowed_hosts_and_refuses_the_rest():
    up = socket.socket()
    up.bind(("127.0.0.1", 0))
    up.listen(1)

    def echo():
        c, _ = up.accept()
        c.sendall(c.recv(100).upper())
        c.close()
    threading.Thread(target=echo, daemon=True).start()
    port = up.getsockname()[1]
    with AG.Fence({"127.0.0.1"}) as f:
        s = socket.create_connection(("127.0.0.1", int(f.url.rsplit(":", 1)[1])))
        s.sendall(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        assert s.recv(100).startswith(b"HTTP/1.1 200")
        s.sendall(b"hello")
        assert s.recv(100) == b"HELLO"
        s.close()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": f.url, "http": f.url}))
        for url in ("https://zenodo.org/records/1", "http://waterservices.usgs.gov/nwis/iv"):
            with pytest.raises((urllib.error.URLError, OSError)):
                opener.open(url, timeout=5)
        got = f.summary()
    assert got["allowed"] == {"127.0.0.1": 1}
    assert set(got["denied"]) == {"zenodo.org", "waterservices.usgs.gov"}
    # with a trusted executable, even an allow-listed host is refused to anything else
    with AG.Fence({"127.0.0.1"}, trusted=["/usr/bin/true"]) as f:
        s = socket.create_connection(("127.0.0.1", int(f.url.rsplit(":", 1)[1])))
        s.sendall(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n".encode())
        assert s.recv(100).startswith(b"HTTP/1.1 403")
        s.close()
        assert list(f.summary()["denied"]) == ["127.0.0.1 (from ?)"]
    up.close()


def _step(i, state="DONE", **kw):
    return {"event": "step_update", "step_update": {"conversation_id": "c", "step_index": i, "state": state, **kw}}


def _tool(i, name, params, state="DONE", output=None, error=None):
    info = {"name": name, "parameters": params}
    if output is not None:
        info["output"] = output
    if error is not None:
        info["error"] = {"type": "TOOL_ERROR", "message": error}
    return _step(i, state, step_type="tool", tool_name=name, tool_info=info)


def test_render_audit_and_failure_detection(tmp_path, monkeypatch):
    """Event shapes as agy 1.4.3 emits them (stream-json)."""
    from hydroagent.h2adapter import NETWORK_PATTERNS
    ev = _events(tmp_path / "events.jsonl", [
        {"event": "init", "conversation_id": "c", "init": {"model": "gemini-4-argon-high", "cwd": "/ws"}},
        _step(0, step_type="user_input"),
        _step(1, "ACTIVE", step_type="agent_response", text_delta="Plan: read "),
        _step(1, "DONE", step_type="agent_response", text_delta="TASK.md first."),
        _tool(2, "run_command", {"CommandLine": "python3 scripts/hydrokit.py info"}, "ACTIVE"),
        _tool(2, "run_command", {"CommandLine": "python3 scripts/hydrokit.py info"}, output="539 training catchments"),
        _tool(3, "run_command", {"CommandLine": "curl https://zenodo.org/x"}, "ERROR",
              error="Permission denied for unsandboxed(curl https://zenodo.org/x). Matches user-configured deny rule."),
        _tool(4, "search_web", {"query": "USGS 02177000"}, "ERROR", error=AG.HOOK_REASON),
        {"event": "result", "result": {"conversation_id": "c", "status": "SUCCESS", "response": "done"}},
    ])
    txt = AG.render_events(ev, obs_trunc=200, max_chars=10_000)
    assert "[ASSISTANT] Plan: read TASK.md first." in txt and txt.count("Plan:") == 1
    assert '[ACTION] run_command({"CommandLine": "python3 scripts/hydrokit.py info"})' in txt
    assert "[OBSERVATION] 539 training catchments" in txt and "[OBSERVATION ERROR] Permission denied" in txt
    assert "[RESULT] status=SUCCESS done" in txt and AG.result_text(ev) == "done"
    a = AG.audit(ev, NETWORK_PATTERNS)
    # the refused search is not a leak; the curl attempt is flagged
    assert a["flagged"] is True and "curl" in a["network_pattern_hits"] and a["web_tools_used"] == []
    assert AG.check_ran(tmp_path, 0) == {"steps": 4, "status": "SUCCESS", "error": "", "soft_denials": [],
                                         "ended_on_denial": False}   # the prompt is not a step
    (tmp_path / "d").mkdir()
    _events(tmp_path / "d" / "events.jsonl", [
        _tool(1, "run_command", {"CommandLine": "python3 a.py"}, output="long"),
        _tool(2, "read_file", {"Path": "/x/.agyhome/.gemini/antigravity-cli/brain/s/output.txt"}, "ERROR",
              error="permission check failed for read_file: user denied permission for read_file(/x)"),
        {"event": "result", "result": {"status": "SUCCESS", "response": ""}}])
    got = AG.check_ran(tmp_path / "d", 0)
    assert got["ended_on_denial"] and len(got["soft_denials"]) == 1
    # startup sign-in noise and a stalled stream after real work are not failures
    (tmp_path / "agy.log").write_text("E1021 errorreport.go:224] error getting token source: You are not logged into Antigravity.\n")
    stalled = _events(tmp_path / "s" / "events.jsonl" if (tmp_path / "s").mkdir() is None else None, [
        _tool(1, "run_command", {"CommandLine": "ls"}, output="x"),
        {"event": "result", "result": {"status": "ERROR", "error": "API error: stream idle timeout (5m0s)"}}])
    (tmp_path / "s" / "agy.log").write_text("E1021 You are not logged into Antigravity.\n")
    assert AG.check_ran(stalled.parent, 1)["status"] == "ERROR"
    # agy reading back its own saved output is not network evidence
    own = _events(tmp_path / "own.jsonl", [_tool(1, "read_file", {"Path": "/x/.cache/hydroagent/runs/r1/.agyhome/.gemini/antigravity-cli/brain/s/output.txt"}, output="x")])
    assert AG.audit(own, NETWORK_PATTERNS)["flagged"] is False
    leaked = _events(tmp_path / "leak.jsonl", [_tool(1, "search_web", {"query": "USGS 02177000"}, output="Chattooga")])
    assert AG.audit(leaked, NETWORK_PATTERNS)["web_tools_used"] == ["search_web"]
    assert AG.audit(ev, NETWORK_PATTERNS)["web_tools_refused"] == ["search_web"]
    greps = _events(tmp_path / "grep.jsonl", [_tool(1, "grep_search", {"Query": "NSE"}, output="x")])
    assert AG.audit(greps, NETWORK_PATTERNS)["flagged"] is False
    # steps of a subagent conversation do not overwrite the main conversation's
    two = [_tool(1, "run_command", {"CommandLine": "curl x"}, output="x")]
    two.append({"event": "step_update", "step_update": {**two[0]["step_update"], "conversation_id": "sub",
                                                        "tool_name": "read_file", "tool_info": {"name": "read_file",
                                                        "parameters": {"Path": "/ws/a"}, "output": "y"}}})
    assert len(AG.steps(_events(tmp_path / "two.jsonl", two))) == 2

    # every refused host is flagged unless it is a known agy background host
    (tmp_path / "c").mkdir()
    clean = _events(tmp_path / "c" / "events.jsonl", [
        _tool(1, "run_command", {"CommandLine": "ls"}, output="x"),
        {"event": "result", "result": {"status": "SUCCESS", "response": "ok"}}])
    (clean.parent / "proxy.json").write_text(json.dumps({"allowed": {}, "denied": {"play.googleapis.com": 2}}))
    assert AG.audit(clean, NETWORK_PATTERNS)["flagged"] is True
    monkeypatch.setattr(AG, "BENIGN_HOSTS", ("play.googleapis.com",))
    assert AG.audit(clean, NETWORK_PATTERNS)["flagged"] is False

    # a stream the parser does not understand is "unknown", not clean
    odd = _events(tmp_path / "odd.jsonl", [{"event": "step_update", "step_update": {"step_index": 1, "x": 1}},
                                           {"event": "result", "result": {"status": "SUCCESS"}}])
    assert AG.audit(odd, NETWORK_PATTERNS)["flagged"] is None

    # sign-in, quota and empty output never pass as a rollout
    mid = [_tool(i, "run_command", {"CommandLine": "ls"}, output="x") for i in range(40)] + [
        {"event": "result", "result": {"status": "ERROR", "response": ""}}]
    quiet = [_step(0, step_type="user_input"),
             {"event": "result", "result": {"status": "ERROR", "response": "RESOURCE_EXHAUSTED: credits exhausted"}}]
    for d, rc, err, evs in (("r", 0, "", quiet), ("q", 1, "Error: RESOURCE_EXHAUSTED quota exceeded", []),
                            ("a", 1, "error: authentication required", []),
                            ("e", 0, "", []),
                            ("m", 1, "429 rate limit reached", mid)):
        (tmp_path / d).mkdir()
        _events(tmp_path / d / "events.jsonl", evs) if evs else (tmp_path / d / "events.jsonl").write_text("")
        (tmp_path / d / "stderr.log").write_text(err)
        with pytest.raises(AG.AgyFailure):
            AG.check_ran(tmp_path / d, rc)


def test_isolated_home_installs_the_no_web_hook(tmp_path):
    import subprocess
    real = tmp_path / "home"
    (real / ".gemini").mkdir(parents=True)
    home = AG.isolated_home(tmp_path / "run", real_home=real)
    hooks = json.loads((home / ".gemini" / "config" / "hooks.json").read_text())["hydroagent-no-web"]["PreToolUse"]
    assert "search_web" in [g["matcher"] for g in hooks]
    out = subprocess.run(["sh", "-c", hooks[0]["hooks"][0]["command"]], input='{"toolCall": {"name": "search_web"}}',
                         capture_output=True, text=True)
    assert json.loads(out.stdout) == {"decision": "deny", "reason": AG.HOOK_REASON}


def test_executor_id_selects_substrate(monkeypatch, tmp_path):
    pytest.importorskip("harness2")
    from hydroagent.h2adapter import HydroAdapter, configure
    configure()
    from harness2 import config as h2c
    fake = tmp_path / "agy"
    fake.write_text("")
    monkeypatch.setenv("HYDRO_AGY_BIN", str(fake))
    monkeypatch.delenv("HYDRO_AGY_EFFORT", raising=False)
    monkeypatch.setattr(h2c, "SOLVER_MODEL_OC", "agy/gemini-4-argon-high")
    a = HydroAdapter(["region-Southeast"], variant="calib")
    assert a.agy and a.substrate.name == "ag-gemini-4-argon-high"
    assert a.replicate_root == "replicates-ag-gemini-4-argon-high"
    assert a.library_dir(3).name == "ag-gemini-4-argon-high-k3"
    assert AG.substrate_name("agy/Gemini 3.5 Flash", "high") == "ag-Gemini-3.5-Flash-high"
    assert AG.argv("p", "agy/gemini-4-argon-high", 5400)[3:] == [
        "--output-format", "stream-json", "--model", "gemini-4-argon-high", "--print-timeout", "90m"]
    big = {"guardrails.md": "no internet " + "x" * 12_000, "systemprompt.md": "y"}
    a.validate_tree(big)                                   # no cap unless asked for
    monkeypatch.setenv("HYDRO_AGY_RULES_CAP", "11000")
    with pytest.raises(ValueError, match="always-on"):
        a.validate_tree(big)
    monkeypatch.delenv("HYDRO_AGY_RULES_CAP")
    a.validate_tree({"guardrails.md": "no internet", "systemprompt.md": "y" * 3000})
    monkeypatch.setattr(h2c, "SOLVER_MODEL_OC", "anthropic/claude-sonnet-5-5")
    b = HydroAdapter(["region-Southeast"], variant="calib")
    assert not b.agy and b.substrate.name == "oc" and b.replicate_root == "replicates"
    assert b.library_dir(3).name == "k3"
    b.validate_tree(big)          # no cap for OpenCode
    monkeypatch.setenv("HYDRO_AGY_BIN", str(tmp_path / "missing"))
    monkeypatch.setattr(h2c, "SOLVER_MODEL_OC", "agy/gemini-4-argon-high")
    with pytest.raises(RuntimeError, match="not found"):
        HydroAdapter(["region-Southeast"], variant="calib")


FAKE_AGY = textwrap.dedent('''\
    #!/usr/bin/env python3
    """Stand-in for `agy -p`: checks what the adapter set up, probes the fence, runs the kNN agent."""
    import json, os, subprocess, sys
    from pathlib import Path
    ws = Path.cwd()
    gem = Path(os.environ["HOME"]) / ".gemini"
    s = json.loads((gem / "antigravity-cli/settings.json").read_text())
    probe = subprocess.run([sys.executable, "-c", "import urllib.request as u; u.urlopen('https://zenodo.org', timeout=5)"],
                           capture_output=True, text=True)
    checks = {"agents_md": (ws / "AGENTS.md").is_file(), "skills": (ws / ".agents/skills").is_dir(),
              "deny": "read_url(*)" in s["permissions"]["deny"], "no_global_rules": not (gem / "GEMINI.md").exists(),
              "no_web_hook": (gem / "config/hooks.json").is_file(),
              "proxy": os.environ.get("HTTPS_PROXY", "").startswith("http://127.0.0.1:"),
              "probe_refused": probe.returncode != 0,
              "outside_cache": ".cache/hydroagent" not in str(ws) + os.environ["HOME"],
              "stream": sys.argv[sys.argv.index("--output-format") + 1], "stdin_closed": sys.stdin.read() == ""}
    print(json.dumps({"event": "init", "init": {"model": sys.argv[sys.argv.index("--model") + 1], "cwd": str(ws)}}))
    (ws / "output" / "code").mkdir(parents=True, exist_ok=True)
    (ws / "output" / "code" / "knn_transfer.py").write_text(Path(os.environ["FAKE_AGENT_SRC"]).read_text())
    p = subprocess.run([sys.executable, "output/code/knn_transfer.py"], capture_output=True, text=True)
    print(json.dumps({"event": "step_update", "step_update": {"step_index": 1, "state": "DONE", "step_type": "tool",
          "tool_name": "run_command", "tool_info": {"name": "run_command",
          "parameters": {"CommandLine": "python3 output/code/knn_transfer.py"}, "output": p.stdout[-500:]}}}))
    print(json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": json.dumps(checks)}}))
    sys.exit(p.returncode)
''')


@pytest.mark.skipif(not (config.TASKS / f"fold1-pilot{config.PILOT_TARGETS}" / "template" / "task_folder" /
                         "data" / "donors.npz").is_file(), reason="run `hydroagent prepare` first")
def test_rollout_through_agy_path(tmp_path, monkeypatch):
    pytest.importorskip("harness2")
    from hydroagent.h2adapter import HydroAdapter, configure
    configure()
    from harness2 import config as h2c
    agy = tmp_path / "agy"
    agy.write_text(FAKE_AGY)
    agy.chmod(agy.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("HYDRO_AGY_BIN", str(agy))
    monkeypatch.setenv("HYDRO_AGY_WS_ROOT", str(tmp_path / "agy_ws"))
    monkeypatch.setenv("FAKE_AGENT_SRC", str(Path(AG.__file__).parent / "fake_agent.py"))
    monkeypatch.setattr(h2c, "SOLVER_MODEL_OC", "agy/gemini-test")
    monkeypatch.setattr(config, "RUNS", tmp_path / "runs")
    task = f"fold1-pilot{config.PILOT_TARGETS}"
    a = HydroAdapter([task], variant="template")
    rid = f"{a.replicate_root}/{task}/base/r1"
    out = a.execute(task, None, rid, timeout=900)
    meta = json.loads((out / "rollout.json").read_text())
    assert meta["returncode"] == 0 and meta["substrate"] == "ag-gemini-test" and meta["executor"] == "agy/gemini-test"
    assert json.loads(meta["agy_result"]["response"]) == {
        "agents_md": True, "skills": True, "deny": True, "no_global_rules": True, "no_web_hook": True, "proxy": True,
        "probe_refused": True, "outside_cache": True, "stream": "stream-json", "stdin_closed": True}
    assert meta["staging"]["skills"] == ["regionalization"] and meta["staging"]["run"]["steps"] == 1
    # the agent's refused download is on record and flags the rollout
    assert meta["audit"]["proxy_denied"] == ["zenodo.org"] and meta["audit"]["flagged"] is True
    assert a.delivered(out) and not (out / ".agyhome").exists()
    assert not list((tmp_path / "agy_ws").glob("*/.agyhome"))            # removed with the workspace
    assert "[ACTION] run_command(" in a.trajectory(out, obs_trunc=500)
    assert a.execute(task, None, rid) == out                 # cached rollout by the same executor is reused
    monkeypatch.setattr(h2c, "SOLVER_MODEL_OC", "agy/other-model")
    with pytest.raises(RuntimeError, match="refusing"):     # never reused or overwritten by another one
        HydroAdapter([task], variant="template").execute(task, None, rid)


@pytest.mark.skipif(shutil.which("lsof") is None, reason="the fence identifies client processes with lsof")
def test_fence_opens_agy_only_hosts_to_the_agy_binary_alone():
    import subprocess
    import sys
    up = socket.socket()
    up.bind(("127.0.0.1", 0))
    up.listen(4)

    def serve():
        while True:
            try:
                c, _ = up.accept()
            except OSError:
                return
            c.sendall(b"hi")
            c.close()
    threading.Thread(target=serve, daemon=True).start()
    port = up.getsockname()[1]
    client = ("import socket,sys; s=socket.create_connection(('127.0.0.1', int(sys.argv[1]))); "
              f"s.sendall(b'CONNECT 127.0.0.1:{port} HTTP/1.1\\r\\n\\r\\n'); print(s.recv(40)[:12].decode())")
    for trusted, expect in (([sys.executable], "HTTP/1.1 200"), (["/usr/bin/true"], "HTTP/1.1 403")):
        with AG.Fence(set(), agy_only={"127.0.0.1"}, trusted=trusted) as f:
            got = subprocess.run([sys.executable, "-c", client, f.url.rsplit(":", 1)[1]],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            assert got == expect, (trusted, got)
            hosts = f.summary()
        assert (expect.endswith("200")) == ("127.0.0.1 (agy)" in hosts["allowed"])
    up.close()
