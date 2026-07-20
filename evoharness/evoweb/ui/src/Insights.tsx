// Research-copilot tab (docs/research_copilot_design.md §4): findings feed
// with clickable evidence chips, and proposal cards in PR form. Interaction
// contract: nothing is applied silently — accept/reject are explicit, a
// rejection requires a written reason (it feeds the agent's memory), and an
// L2 proposal without version bumps has its accept button disabled (the
// server enforces the same rule; the UI just explains it earlier).
import { useEffect, useState } from "react";
import type { EvidenceChip, Finding, InsightsFile, Proposal } from "./api";
import { decideProposal, fetchInsights, fetchProposalPatch } from "./api";

const SEV: Record<string, { icon: string; cls: string }> = {
  observation: { icon: "◇", cls: "sev-obs" },
  actionable: { icon: "◆", cls: "sev-act" },
  warning: { icon: "⚠", cls: "sev-warn" },
  speculation: { icon: "?", cls: "sev-spec" },
};

function Chip({
  e,
  onPickCandidate,
}: {
  e: EvidenceChip;
  onPickCandidate: (cid: string) => void;
}) {
  const label =
    e.type === "candidate"
      ? `cand:${(e.ref ?? "").slice(0, 8)}`
      : e.type === "series"
        ? `${e.ref}${e.gen_range ? ` g${e.gen_range[0]}–${e.gen_range[1]}` : ""}`
        : e.type === "cohort"
          ? "cohort"
          : "query";
  const clickable = e.type === "candidate" && !!e.ref;
  return (
    <button
      className={`ev-chip ${clickable ? "" : "ev-static"}`}
      title={e.note ?? (e.query || e.ref)}
      onClick={clickable ? () => onPickCandidate(e.ref as string) : undefined}
    >
      {label}
      {e.note && <span className="ev-note"> · {e.note}</span>}
    </button>
  );
}

function FindingCard({
  f,
  onPickCandidate,
  onJumpProposal,
}: {
  f: Finding;
  onPickCandidate: (cid: string) => void;
  onJumpProposal: (pid: string) => void;
}) {
  const sev = SEV[f.severity] ?? SEV.observation;
  return (
    <article className={`finding ${sev.cls}`} id={`finding-${f.id}`}>
      <div className="f-claim">
        <span className="f-icon">{sev.icon}</span>
        {f.claim}
      </div>
      <div className="f-evidence">
        {f.evidence.map((e, i) => (
          <Chip key={i} e={e} onPickCandidate={onPickCandidate} />
        ))}
      </div>
      <div className="f-meta">
        置信 {f.confidence}
        {f.noise_note ? ` · ${f.noise_note}` : ""}
        {f.playbook && <span className="pb-tag">{f.playbook}</span>}
        {f.proposals.map((pid) => (
          <button key={pid} className="p-link" onClick={() => onJumpProposal(pid)}>
            → 提案 {pid}
          </button>
        ))}
      </div>
    </article>
  );
}

function ProposalCard({
  run,
  p,
  findings,
  onDecided,
}: {
  run: string;
  p: Proposal;
  findings: Finding[];
  onDecided: (updated: Proposal) => void;
}) {
  const [patch, setPatch] = useState<string | null>(null);
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState("");
  const [err, setErr] = useState("");
  const l2NoBump = p.level === "L2" && !p.version_bumps;
  const motivClaims = p.motivation
    .map((fid) => findings.find((f) => f.id === fid))
    .filter(Boolean) as Finding[];

  const decide = async (action: "accept" | "reject") => {
    setErr("");
    try {
      onDecided(await decideProposal(run, p.id, action, reason));
    } catch (e) {
      setErr(String((e as Error).message));
    }
  };

  return (
    <article className={`proposal p-${p.status}`} id={`proposal-${p.id}`}>
      <div className="p-head">
        <span className={`lvl lvl-${p.level}`}>{p.level}</span>
        <b>
          {p.id} {p.title}
        </b>
        <span className={`p-status s-${p.status}`}>{p.status}</span>
      </div>
      {motivClaims.length > 0 && (
        <div className="p-motiv">
          动机:{" "}
          {motivClaims.map((f) => (
            <a key={f.id} href={`#finding-${f.id}`}>
              {f.claim.slice(0, 40)}…
            </a>
          ))}
        </div>
      )}
      {p.version_bumps && (
        <div className="p-verbump">
          ⚡ 版本升级(强制):{" "}
          {Object.entries(p.version_bumps).map(([k, v]) => (
            <code key={k} className="mono">
              {k}: {v}
            </code>
          ))}
        </div>
      )}
      {l2NoBump && (
        <div className="p-verbump p-verbump-missing">
          ⛔ L2 提案缺少版本升级 — 按评估纪律不可接受(护栏,服务端同样强制)
        </div>
      )}
      {p.predicted_effect && (
        <div className="p-effect">预期影响: {p.predicted_effect}</div>
      )}
      {p.validation_plan && (
        <div className="p-effect">
          验证计划: {p.validation_plan.paired_control ? "配对对照 " : ""}
          {(p.validation_plan.recipes ?? []).join(" + ")} ·{" "}
          {p.validation_plan.budget_est}
        </div>
      )}
      {p.patch_file && (
        <details
          onToggle={(ev) => {
            if ((ev.target as HTMLDetailsElement).open && patch == null)
              fetchProposalPatch(run, p.id).then(setPatch);
          }}
        >
          <summary>diff({p.patch_file})</summary>
          <pre className="codeblock patch">{patch ?? "加载中…"}</pre>
        </details>
      )}
      {p.status === "draft" && (
        <div className="p-actions">
          <button
            className="btn-primary"
            disabled={l2NoBump}
            title={
              l2NoBump
                ? "L2 无版本升级不可接受"
                : "标记接受;v1 中补丁经 git 手动应用(R2 将接管应用步)"
            }
            onClick={() => decide("accept")}
          >
            接受
          </button>
          {!rejecting ? (
            <button className="btn-danger" onClick={() => setRejecting(true)}>
              拒绝…
            </button>
          ) : (
            <>
              <input
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="拒绝原因(必填,回流 agent 记忆)"
                autoFocus
              />
              <button className="btn-danger" onClick={() => decide("reject")}>
                确认拒绝
              </button>
            </>
          )}
          {err && <span className="p-err">{err}</span>}
        </div>
      )}
      {p.status === "accepted" && (
        <div className="p-actions r-sub">
          已接受 · v1 请手动应用: <code>git apply insights/proposals/{p.id}.patch</code>
        </div>
      )}
      {p.status === "rejected" && p.rejection_reason && (
        <div className="p-actions r-sub">拒绝原因: {p.rejection_reason}</div>
      )}
    </article>
  );
}

export function Insights({
  run,
  onPickCandidate,
}: {
  run: string;
  onPickCandidate: (cid: string) => void;
}) {
  const [data, setData] = useState<InsightsFile | null>(null);
  useEffect(() => {
    setData(null);
    fetchInsights(run).then(setData).catch(() => setData({ findings: [], proposals: [] }));
  }, [run]);
  if (!data) return <p className="r-sub">加载中…</p>;
  if (data.findings.length === 0 && data.proposals.length === 0)
    return (
      <div className="empty">
        <p>此 run 暂无研究员产出。</p>
        <p>
          Findings 由 copilot agent(或人工黄金样本)写入
          <code>&lt;run&gt;/insights/findings.jsonl</code> —— 见
          docs/research_copilot_design.md。
        </p>
      </div>
    );
  const onDecided = (updated: Proposal) =>
    setData({
      ...data,
      proposals: data.proposals.map((p) => (p.id === updated.id ? updated : p)),
    });
  const jump = (pid: string) =>
    document
      .getElementById(`proposal-${pid}`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  const drafts = data.proposals.filter((p) => p.status === "draft");
  const decided = data.proposals.filter((p) => p.status !== "draft");
  return (
    <div className="insights">
      {data.proposals.length > 0 && (
        <section>
          <h3 className="ins-h">
            提案 {drafts.length > 0 && <span className="badge">{drafts.length} 待审</span>}
          </h3>
          {[...drafts, ...decided].map((p) => (
            <ProposalCard key={p.id} run={run} p={p} findings={data.findings}
                          onDecided={onDecided} />
          ))}
        </section>
      )}
      <section>
        <h3 className="ins-h">发现流</h3>
        {data.findings.map((f) => (
          <FindingCard key={f.id} f={f} onPickCandidate={onPickCandidate}
                       onJumpProposal={jump} />
        ))}
      </section>
    </div>
  );
}
