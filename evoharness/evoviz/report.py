# EvoHarness original (viz_design.md): visualization is a pure function of
# the run directory — no instrumentation, no server, one self-contained
# HTML file. This is the minimal cut: metric cards, fitness curve, history
# timeline, lineage tree, candidate table. Panels auto-extend later.
"""Render a self-contained report.html from a run directory."""

from __future__ import annotations

import html
import json
from pathlib import Path

from evoharness.evocore import MetricLog, PopulationConfig, PopulationStore

OPERATOR_COLORS = {
    "seed": "#888780",
    "revise": "#1D9E75",
    "rewrite": "#534AB7",
    "recombine": "#D4537E",
    "repair": "#D85A30",
    "human": "#BA7517",
}

_CSS = """
body{font-family:-apple-system,'Segoe UI',sans-serif;max-width:960px;margin:2rem auto;
padding:0 1rem;color:#222;background:#fafaf8}
h1{font-size:20px}h2{font-size:15px;margin-top:1.6rem;color:#444}
.cards{display:flex;gap:10px;flex-wrap:wrap}
.card{background:#fff;border:1px solid #e2e0da;border-radius:8px;padding:10px 16px;min-width:120px}
.card .k{font-size:11px;color:#777}.card .v{font-size:20px;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff}
td,th{border-bottom:1px solid #eee;padding:5px 8px;text-align:left}
th{color:#666;font-weight:600;font-size:11px}
.op{display:inline-block;padding:1px 8px;border-radius:8px;color:#fff;font-size:11px}
.dup{color:#993C1D;font-size:11px}
.mono{font-family:ui-monospace,monospace;font-size:11px;color:#666}
.skip{color:#999}.fail{color:#A32D2D}
ul.tree{list-style:none;padding-left:18px;border-left:1px dotted #ccc;font-size:12.5px}
svg{background:#fff;border:1px solid #e2e0da;border-radius:8px}
"""


def load_run(run_dir: Path | str) -> dict:
    run_dir = Path(run_dir)
    store = PopulationStore(PopulationConfig(), run_dir / "run.db")
    candidates = store.all_candidates()
    store.close()
    metrics = MetricLog(run_dir / "metrics.jsonl")
    manifest = {}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    checkpoint = {}
    ckpt_path = run_dir / "checkpoint.json"
    if ckpt_path.exists():
        checkpoint = json.loads(ckpt_path.read_text())
    return {
        "run_dir": str(run_dir),
        "candidates": candidates,
        "metrics": metrics,
        "manifest": manifest,
        "history": checkpoint.get("run_report", {}).get("history", []),
    }


def _curve_svg(metrics: MetricLog, width=880, height=200) -> str:
    best = metrics.series("sys/best_fitness")
    fit = metrics.series("sys/fitness")
    if not best:
        return "<p class='skip'>no metrics recorded</p>"
    xs = [s for s, _ in best]
    ys = [v for _, v in best]
    x_max = max(max(xs), 1)
    y_lo = min(min(v for _, v in fit or best), min(ys))
    y_hi = max(max(v for _, v in fit or best), max(ys))
    span = (y_hi - y_lo) or 1.0
    pad, w, h = 34, width - 50, height - 40

    def pt(step, val):
        return (
            pad + step / x_max * w,
            height - 24 - (val - y_lo) / span * h,
        )

    line = " ".join(f"{pt(s, v)[0]:.1f},{pt(s, v)[1]:.1f}" for s, v in best)
    dots = "".join(
        f'<circle cx="{pt(s, v)[0]:.1f}" cy="{pt(s, v)[1]:.1f}" r="3" '
        f'fill="#1D9E75" opacity="0.7"/>'
        for s, v in fit
    )
    return f"""<svg viewBox="0 0 {width} {height}" width="100%">
<text x="{pad}" y="14" font-size="11" fill="#666">fitness — line: best, dots: per-candidate</text>
<line x1="{pad}" y1="{height - 24}" x2="{width - 12}" y2="{height - 24}" stroke="#ccc"/>
<line x1="{pad}" y1="12" x2="{pad}" y2="{height - 24}" stroke="#ccc"/>
<text x="{pad - 4}" y="{height - 20}" font-size="10" fill="#888" text-anchor="end">{y_lo:.2f}</text>
<text x="{pad - 4}" y="20" font-size="10" fill="#888" text-anchor="end">{y_hi:.2f}</text>
<polyline points="{line}" fill="none" stroke="#534AB7" stroke-width="2"/>
{dots}</svg>"""


def _lineage_html(candidates) -> str:
    by_parent: dict = {}
    for c in candidates:
        by_parent.setdefault(c.parent_id, []).append(c)

    def node(c):
        color = OPERATOR_COLORS.get(c.operator, "#888")
        dup = " <span class='dup'>duplicate</span>" if c.behavior_duplicate else ""
        title = html.escape(c.change_title or "")
        kids = "".join(node(k) for k in by_parent.get(c.id, []))
        return (
            f"<li><span class='op' style='background:{color}'>{c.operator}</span> "
            f"<span class='mono'>{c.id}</span> gen{c.generation} "
            f"fit <b>{c.fitness:.3g}</b> {title}{dup}"
            f"{f'<ul class=tree>{kids}</ul>' if kids else ''}</li>"
        )

    roots = by_parent.get(None, [])
    return "<ul class='tree'>" + "".join(node(r) for r in roots) + "</ul>"


def render_html(data: dict) -> str:
    cands = data["candidates"]
    graded = [c for c in cands if c.report is not None]
    best = max((c for c in graded if c.passed), key=lambda c: c.fitness, default=None)
    manifest = data["manifest"]
    cost = 0.0
    if manifest:
        rep = manifest.get("report", {})
        cost = rep.get("total_llm_cost", 0) + rep.get("total_eval_cost", 0)

    cards = "".join(
        f"<div class='card'><div class='k'>{k}</div><div class='v'>{v}</div></div>"
        for k, v in [
            ("best fitness", f"{best.fitness:.3g}" if best else "—"),
            ("candidates", len(cands)),
            ("distinct signatures",
             len({c.behavior_signature for c in cands if c.behavior_signature})),
            ("duplicates", sum(c.behavior_duplicate for c in cands)),
            ("recorded cost $", f"{cost:.4f}"),
            ("recipe", manifest.get("recipe", "?")),
        ]
    )

    hist_rows = "".join(
        f"<tr class='{h.get('status', '')}'><td>{h.get('generation')}</td>"
        f"<td>{h.get('status')}</td><td>{h.get('operator', '')}</td>"
        f"<td>{h.get('fitness', '')}</td>"
        f"<td class='mono'>{h.get('candidate_id', '')}</td></tr>"
        for h in data["history"]
    )

    cand_rows = "".join(
        f"<tr><td>{c.generation}</td>"
        f"<td><span class='op' style='background:"
        f"{OPERATOR_COLORS.get(c.operator, '#888')}'>{c.operator}</span></td>"
        f"<td>{c.fitness:.3g}</td><td>{html.escape(c.change_title or '')}</td>"
        f"<td class='mono'>{c.behavior_signature or ''}</td>"
        f"<td>{'yes' if c.in_archive else ''}</td></tr>"
        for c in sorted(cands, key=lambda c: (c.generation, c.timestamp))
    )

    best_block = ""
    if best is not None:
        best_block = (
            f"<h2>Best candidate ({best.id})</h2>"
            f"<pre style='background:#fff;border:1px solid #e2e0da;"
            f"border-radius:8px;padding:12px;font-size:12px'>"
            f"{html.escape(best.workspace.main_text())}</pre>"
        )

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>EvoHarness report — {html.escape(data["run_dir"])}</title>
<style>{_CSS}</style></head><body>
<h1>EvoHarness run report</h1>
<p class='mono'>{html.escape(data["run_dir"])}</p>
<div class='cards'>{cards}</div>
<h2>Fitness curve</h2>{_curve_svg(data["metrics"])}
<h2>Generation history</h2>
<table><tr><th>gen</th><th>status</th><th>operator</th><th>fitness</th><th>candidate</th></tr>{hist_rows}</table>
<h2>Lineage</h2>{_lineage_html(cands)}
<h2>Candidates</h2>
<table><tr><th>gen</th><th>operator</th><th>fitness</th><th>change</th><th>behavior signature</th><th>archive</th></tr>{cand_rows}</table>
{best_block}
</body></html>"""


def generate(run_dir: Path | str, out: Path | str | None = None) -> Path:
    data = load_run(run_dir)
    out = Path(out) if out else Path(run_dir) / "report.html"
    out.write_text(render_html(data))
    return out
