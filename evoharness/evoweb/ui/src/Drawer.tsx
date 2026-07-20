import { useEffect, useState } from "react";
import type { CandidateDetail, DirectiveDraft } from "./api";
import { fetchCandidate } from "./api";
import { OpBadge } from "./components";

// The drawer never sends directives itself. Steering buttons build a
// prefilled draft and hand it to the Control tab, where the human reviews
// and sends it — same discipline as "reproduce" flows in research
// workbenches: buttons draft, people decide.
export function Drawer({
  run,
  cid,
  onNavigate,
  onClose,
  onDraft,
}: {
  run: string;
  cid: string;
  onNavigate: (cid: string) => void;
  onClose: () => void;
  onDraft: (d: DirectiveDraft) => void;
}) {
  const [cand, setCand] = useState<CandidateDetail | null>(null);

  useEffect(() => {
    let alive = true;
    fetchCandidate(run, cid).then((c) => alive && setCand(c));
    return () => { alive = false; };
  }, [run, cid]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!cand) return null;
  const fb = cand.report?.structured_feedback;
  const failed = fb?.items.filter((i) => !i.passed) ?? [];
  const hist = new Map<string, number>();
  for (const i of failed) {
    const k = i.error_category || "uncategorized";
    hist.set(k, (hist.get(k) ?? 0) + 1);
  }
  const maxN = Math.max(1, ...hist.values());
  const topCats = [...hist.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([k]) => k);

  const steerDraft = () => {
    const text = topCats.length
      ? `Candidate ${cand.id} (gen ${cand.generation}) keeps failing on: ` +
        `${topCats.join(", ")}. To fix this, try `
      : `Build on the approach of candidate ${cand.id}` +
        `${cand.title ? ` ("${cand.title}")` : ""}: `;
    onDraft({ kind: "guidance", text, source: `candidate ${cand.id}` });
  };

  const vetoDraft = () =>
    onDraft({
      kind: "lineage_veto",
      candidate_ids: [cand.id],
      source: `candidate ${cand.id}`,
    });

  return (
    <aside className="drawer" aria-label="Candidate detail">
      <div className="drawer-actions">
        <button onClick={steerDraft} title="Prefill a guidance draft from this candidate's failures — review it on the Control tab before sending.">
          Steer from here…
        </button>
        <button onClick={vetoDraft} title="Prefill a lineage veto for review on the Control tab. Nothing is sent yet.">
          Veto lineage…
        </button>
        <span className="spacer" />
        <button onClick={onClose}>Close</button>
      </div>
      <p className="meta mono">
        {cand.id} · gen {cand.generation}
        {cand.parent_id && (
          <>
            {" · parent "}
            <a href="#" onClick={(e) => { e.preventDefault(); onNavigate(cand.parent_id!); }}>
              {cand.parent_id}
            </a>
          </>
        )}
      </p>
      <h3 className="cand-title">
        <OpBadge op={cand.operator} /> {cand.title || "(untitled change)"}
      </h3>
      <p>{cand.summary}</p>
      <p>
        <span className="chip mono">
          fitness {cand.fitness != null ? cand.fitness.toFixed(4) : "–"}
        </span>
        <span className="chip mono">island {cand.island}</span>
        {cand.model && <span className="chip mono">{cand.model}</span>}
        {cand.in_archive && <span className="chip chip-teal">archive</span>}
        {!cand.passed && <span className="chip chip-coral">failed evaluation</span>}
        {cand.duplicate && <span className="chip chip-coral">behavioral duplicate</span>}
      </p>
      {cand.report?.fault && (
        <p className="mono status-failed">fault: {cand.report.fault}</p>
      )}
      {fb && (
        <>
          <h3>Failure categories</h3>
          {failed.length === 0 ? (
            <p className="status-skipped">All items passed.</p>
          ) : (
            [...hist.entries()].map(([k, n]) => (
              <div className="fb-row" key={k}>
                <span className="mono" style={{ minWidth: 130 }}>{k}</span>
                <span className="fb-bar" style={{ width: (n / maxN) * 140 }} /> {n}
              </div>
            ))
          )}
        </>
      )}
      <h3>Code</h3>
      <pre className="mono">{cand.code}</pre>
      {cand.report?.stderr_log && (
        <>
          <h3>stderr</h3>
          <pre className="mono">{cand.report.stderr_log.slice(-1500)}</pre>
        </>
      )}
    </aside>
  );
}
