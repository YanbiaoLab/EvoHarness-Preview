import type { CandidateRow, RunDetail, RunSummary } from "./api";
import { OP_COLORS } from "./api";

export function OpBadge({ op }: { op: string }) {
  return (
    <span className="op" style={{ background: OP_COLORS[op] ?? "#888" }}>
      {op}
    </span>
  );
}

export function RunList({
  runs,
  current,
  onOpen,
  onDelete,
}: {
  runs: RunSummary[];
  current: string | null;
  onOpen: (name: string) => void;
  onDelete: (name: string) => void;
}) {
  if (runs.length === 0)
    return <p className="r-sub">No runs found in this directory.</p>;
  return (
    <>
      {runs.map((r) => {
        // Reconcile the "live" claim against the heartbeat: a checkpoint
        // that stopped moving means the process likely died, whatever the
        // report says (the loop only writes stopped_reason on clean exit).
        const running = r.stopped_reason === "running";
        const stalled =
          running && r.heartbeat_age_s != null && r.heartbeat_age_s > 120;
        return (
        <div className="runrow" key={r.name}>
          <button
            className="runitem"
            aria-current={r.name === current}
            onClick={() => onOpen(r.name)}
          >
            <div className="r-name">
              <span
                className={`dot ${running ? (stalled ? "stale" : "live") : ""}`}
                title={stalled
                  ? "No checkpoint update for over 2 minutes — the process may have died."
                  : undefined}
              />
              {r.name}
            </div>
            <div className="r-sub mono">
              {r.recipe} · gen {r.generation}
              {r.target_generations ? `/${r.target_generations}` : ""} ·{" "}
              {stalled ? "stalled?" : r.stopped_reason}
            </div>
          </button>
          <button
            className="del"
            aria-label={`Delete run ${r.name}`}
            onClick={() => onDelete(r.name)}
          >
            Delete
          </button>
        </div>
        );
      })}
    </>
  );
}

export function Hero({ detail, live }: { detail: RunDetail; live: boolean }) {
  const target = detail.target_generations ?? "?";
  const best = detail.report.best_fitness;
  return (
    <>
      <h1 className="run-title">
        {detail.name}
        <span className="chip" title={detail.recipe_description}>
          {detail.recipe}
        </span>
        {live && (
          <span className="live-badge">
            <span className="pulse" />
            following live
          </span>
        )}
      </h1>
      <div className="hero">
        <div>
          <div className="gen-label">GENERATION</div>
          <div className="gen-counter mono">
            {String(detail.generation).padStart(3, "0")}
            <span className="of">/{String(target).padStart(3, "0")}</span>
          </div>
        </div>
        <div className="hero-stats">
          <Stat k="best fitness" v={best == null ? "–" : best.toFixed(3)} />
          <Stat k="candidates" v={String(detail.candidates.length)} />
          <Stat k="behaviors" v={String(detail.distinct_signatures)} />
        </div>
      </div>
    </>
  );
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="stat">
      <div className="k">{k}</div>
      <div className="v mono">{v}</div>
    </div>
  );
}

export function FitnessChart({
  detail,
  lastGenSeen,
}: {
  detail: RunDetail;
  lastGenSeen: number;
}) {
  const W = 900, H = 220, padL = 46, padB = 28, padT = 18;
  const best = detail.series.best_fitness;
  const fit = detail.series.fitness;
  if (best.length === 0)
    return (
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
           aria-label="Fitness over generations">
        <text x={20} y={30} style={{ fill: "var(--muted)" }} fontSize={13}>
          No metrics yet — the chart appears after generation 0.
        </text>
      </svg>
    );
  const xMax = Math.max(detail.target_generations ?? 0, ...best.map((p) => p[0]), 1);
  const ys = [...best.map((p) => p[1]), ...fit.map((p) => p[1])];
  const yLo = Math.min(...ys), yHi = Math.max(...ys);
  const span = yHi - yLo || 1;
  const X = (s: number) => padL + (s / xMax) * (W - padL - 16);
  const Y = (v: number) => H - padB - ((v - yLo) / span) * (H - padB - padT);
  const line = best.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ");
  const bestLast = best[best.length - 1];
  const area =
    `${X(best[0][0]).toFixed(1)},${(H - padB).toFixed(1)} ${line} ` +
    `${X(bestLast[0]).toFixed(1)},${(H - padB).toFixed(1)}`;
  const ticks = [yLo, yLo + span / 2, yHi];
  const newest = Math.max(...fit.map((p) => p[0]));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
         aria-label="Fitness over generations">
      <defs>
        <linearGradient id="fitgrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" style={{ stopColor: "var(--violet)" }} stopOpacity={0.16} />
          <stop offset="1" style={{ stopColor: "var(--violet)" }} stopOpacity={0} />
        </linearGradient>
      </defs>
      {ticks.map((v) => (
        <g key={v}>
          <line x1={padL} y1={Y(v)} x2={W - 10} y2={Y(v)}
                style={{ stroke: "var(--rule)" }}
                strokeDasharray={v === yLo ? undefined : "3 5"} />
          <text x={padL - 8} y={Y(v) + 4} textAnchor="end" fontSize={11}
                style={{ fill: "var(--muted)" }} className="mono">
            {v.toFixed(2)}
          </text>
        </g>
      ))}
      <text x={W - 10} y={H - 8} textAnchor="end" fontSize={11}
            style={{ fill: "var(--muted)" }} className="mono">gen {xMax}</text>
      <polygon points={area} fill="url(#fitgrad)" />
      <polyline points={line} fill="none" strokeWidth={2.5}
                strokeLinejoin="round" strokeLinecap="round"
                style={{ stroke: "var(--violet)" }} />
      {(detail.report.history ?? [])
        .filter((h) => h.status === "infra_error" || h.status === "skipped")
        .map((h, i) => (
          <line key={`ev${i}`} x1={X(h.generation)} x2={X(h.generation)}
                y1={padT} y2={H - padB} strokeDasharray="2 4" strokeWidth={1.5}
                style={{ stroke: h.status === "infra_error" ? "var(--coral)" : "var(--muted)" }}
                opacity={0.6}>
            <title>{`gen ${h.generation}: ${h.status}${h.error ? ` — ${h.error}` : ""}`}</title>
          </line>
        ))}
      {detail.candidates
        .filter((c) => c.fitness != null)
        .map((c) => (
          <circle
            key={c.id}
            className={c.generation === newest && c.generation > lastGenSeen ? "pt-new" : ""}
            cx={X(c.generation)}
            cy={Y(c.fitness as number)}
            r={3.2}
            style={
              c.passed
                ? { fill: OP_COLORS[c.operator] ?? "var(--teal)" }
                : { fill: "none", stroke: OP_COLORS[c.operator] ?? "var(--muted)", strokeWidth: 1.4 }
            }
            opacity={0.85}
          >
            <title>{`${c.id.slice(0, 8)} · ${c.operator} · ${c.fitness?.toFixed(3)}${c.passed ? "" : " · failed"}`}</title>
          </circle>
        ))}
      <circle cx={X(bestLast[0])} cy={Y(bestLast[1])} r={4}
              style={{ fill: "var(--violet)" }} />
      <text x={X(bestLast[0]) - 8} y={Y(bestLast[1]) - 10} textAnchor="end"
            fontSize={11.5} fontWeight={600}
            style={{ fill: "var(--violet)" }} className="mono">
        {bestLast[1].toFixed(3)}
      </text>
    </svg>
  );
}

export function Tags({ c }: { c: CandidateRow }) {
  return (
    <>
      {c.duplicate && <span className="tag-dup"> duplicate</span>}
      {c.in_archive && <span className="tag-arch"> archive</span>}
    </>
  );
}


/** Task-emitted eval/* series (e.g. per-tier accuracy): one chart, ordinal
 * single-hue ramp (--ord1..3, validated light+dark), direct label at each
 * line end so identity is never color-alone. */
export function EvalSeriesChart({ detail }: { detail: RunDetail }) {
  const entries = Object.entries(detail.eval_series ?? {});
  if (entries.length === 0) return null;
  const W = 900, H = 190, padL = 46, padB = 26, padT = 14, padR = 110;
  const ramp = ["var(--ord1)", "var(--ord2)", "var(--ord3)", "var(--coral)", "var(--violet)"];
  const xMax = Math.max(1, ...entries.flatMap(([, pts]) => pts.map((p) => p[0])));
  const X = (s: number) => padL + (s / xMax) * (W - padL - padR);
  const Y = (v: number) => H - padB - v * (H - padB - padT);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
         aria-label="Task eval metrics over generations">
      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <line x1={padL} y1={Y(v)} x2={W - padR + 20} y2={Y(v)}
                style={{ stroke: "var(--rule)" }}
                strokeDasharray={v === 0 ? undefined : "3 5"} />
          <text x={padL - 8} y={Y(v) + 4} textAnchor="end" fontSize={11}
                style={{ fill: "var(--muted)" }} className="mono">{v.toFixed(1)}</text>
        </g>
      ))}
      {entries.map(([name, pts], i) => {
        if (pts.length === 0) return null;
        const line = pts.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ");
        const last = pts[pts.length - 1];
        return (
          <g key={name}>
            <polyline points={line} fill="none" strokeWidth={2}
                      strokeLinejoin="round" strokeLinecap="round"
                      style={{ stroke: ramp[i % ramp.length] }} opacity={0.9} />
            <text x={X(last[0]) + 8} y={Y(last[1]) + 4} fontSize={11}
                  fontWeight={600} className="mono"
                  style={{ fill: ramp[i % ramp.length] }}>
              {name} {last[1].toFixed(2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
