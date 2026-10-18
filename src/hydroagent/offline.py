"""Scripted harness editor for offline pipeline tests (no model calls).

Mirrors harness2's own DemoProposer: proposal sessions edit the harness tree in the
session workspace, selection picks the first candidate, reflection records nothing.
"""
from __future__ import annotations

from pathlib import Path

from harness2.core.proposer import Proposer, Reply


class ScriptedProposer(Proposer):
    def call(self, mission, context_dir, label, **kwargs) -> Reply:
        context = Path(context_dir)
        workspace = None
        if label.startswith("select."):
            cands = sorted(p.stem for p in (context / "candidates").glob("*.md"))
            text = f"<why>first candidate (offline test)</why>\n```choice\n{cands[0] if cands else 'A'}\n```"
        elif label.startswith("experience."):
            text = "NO_ENTRY"
        else:
            workspace = context / "harness"
            sp = workspace / "systemprompt.md"
            sp.write_text(sp.read_text() + "\nPrefer output averaging over several similar donors.\n")
            text = "<notes>Added an output-averaging preference to the system prompt (offline test edit).</notes>"
        reply = Reply(label=label, text=text, ok=True, attempts=1, workspace=workspace)
        self._calls.append(reply)
        return reply
