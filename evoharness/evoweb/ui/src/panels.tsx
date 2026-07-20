import { useState } from "react";
import type { CandidateRow, HistoryEntry, RunDetail } from "./api";
import { OpBadge, Tags } from "./components";
import { groupLedger } from "./ledger";

function EntryRow({
  h,
  newBest,
  onPick,
}: {
  h: HistoryEntry;
  newBest: boolean;
  onPick: (cid: string) => void;
}) {
  return (
    <tr
      className={`${h.candidate_id ? "rowbtn" : ""} status-${h.status}`}
      onClick={() => h.candidate_id && onPick(h.candidate_id)}
    >
      <td className="mono">{String(h.generation).padStart(3, "0")}</td>
      <td>
        {h.status}
        {newBest && <span className="star" title="New best fitness"> ★ new best</span>}
      </td>
      <td>{h.operator ? <OpBadge op={h.operator} /> : null}</td>
      <td className="mono">{h.fitness != null ? h.fitness.toFixed(3) : ""}</td>
      <td className="mono">{h.candidate_id ?? ""}</td>
    </tr>
  );
}

export function Ledger({
  detail,
  onPick,
}: {
  detail: RunDetail;
  onPick: (cid: string) => void;
}) {
  const [open, setOpen] = useState<Set<string>>(new Set());
  const rows = groupLedger(detail.report.history ?? []);
  const toggle = (key: string) =>
    setOpen((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  return (
    <table>
      <thead>
        <tr>
          <th>GEN</th><th>STATUS</th><th>OPERATOR</th><th>FITNESS</th><th>CANDIDATE</th>
        </tr>
      </thead>
      <tbody>
        {rows.length === 0 && (
          <tr><td colSpan={5} className="status-skipped">No generations yet.</td></tr>
        )}
        {rows.map((row, i) => {
          if (row.kind === "entry")
            return <EntryRow key={i} h={row.entry} newBest={row.newBest} onPick={onPick} />;
          const key = `${row.genFrom}-${row.genTo}`;
          const fitness =
            row.fitnessFrom != null && row.fitnessTo != null
              ? ` · fitness ${row.fitnessFrom.toFixed(3)} → ${row.fitnessTo.toFixed(3)}`
              : "";
          return [
            <tr key={key} className="quiet rowbtn" onClick={() => toggle(key)}>
              <td className="mono">
                {String(row.genTo).padStart(3, "0")}–{String(row.genFrom).padStart(3, "0")}
              </td>
              <td colSpan={4}>
                {open.has(key) ? "▾" : "▸"} {row.entries.length} quiet generations
                {fitness}
              </td>
            </tr>,
            ...(open.has(key)
              ? [...row.entries].reverse().map((h) => (
                  <EntryRow key={`${key}-${h.generation}`} h={h} newBest={false} onPick={onPick} />
                ))
              : []),
          ];
        })}
      </tbody>
    </table>
  );
}

export function Lineage({
  detail,
  onPick,
}: {
  detail: RunDetail;
  onPick: (cid: string) => void;
}) {
  const byParent = new Map<string | null, CandidateRow[]>();
  for (const c of detail.candidates) {
    const key = c.parent_id ?? null;
    byParent.set(key, [...(byParent.get(key) ?? []), c]);
  }
  const Node = ({ c }: { c: CandidateRow }) => (
    <li>
      <button className="node" onClick={() => onPick(c.id)}>
        <OpBadge op={c.operator} /> <span className="mono">{c.id}</span> · gen{" "}
        {c.generation} · fit{" "}
        <b className="mono">{c.fitness != null ? c.fitness.toFixed(3) : "–"}</b>
        {c.title ? ` · ${c.title}` : ""}
        <Tags c={c} />
      </button>
      {(byParent.get(c.id) ?? []).length > 0 && (
        <ul className="tree">
          {byParent.get(c.id)!.map((k) => <Node key={k.id} c={k} />)}
        </ul>
      )}
    </li>
  );
  return (
    <ul className="tree">
      {(byParent.get(null) ?? []).map((c) => <Node key={c.id} c={c} />)}
    </ul>
  );
}

export function Candidates({
  detail,
  onPick,
}: {
  detail: RunDetail;
  onPick: (cid: string) => void;
}) {
  return (
    <table>
      <thead>
        <tr>
          <th>GEN</th><th>OPERATOR</th><th>FITNESS</th><th>CHANGE</th><th>BEHAVIOR</th><th></th>
        </tr>
      </thead>
      <tbody>
        {detail.candidates.map((c) => (
          <tr key={c.id} className="rowbtn" onClick={() => onPick(c.id)}>
            <td className="mono">{c.generation}</td>
            <td><OpBadge op={c.operator} /></td>
            <td className="mono">{c.fitness != null ? c.fitness.toFixed(3) : "–"}</td>
            <td>{c.title}</td>
            <td>
              <span className="mono sig" title={c.signature ?? ""}>
                {c.signature ?? ""}
              </span>
            </td>
            <td><Tags c={c} /></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
