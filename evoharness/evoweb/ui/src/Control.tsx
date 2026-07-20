import { useCallback, useEffect, useState } from "react";
import type { DirectiveDraft, DirectiveFile } from "./api";
import { fetchDirectives, postDirective } from "./api";

// HITL control tab: active directives + guidance form. Directives take
// effect at the NEXT generation (the loop re-reads the control file).
// Drafts arrive prefilled from elsewhere in the UI (candidate drawer);
// nothing is sent until the human presses Send here.
export function Control({
  run,
  draft,
  onDraftConsumed,
}: {
  run: string;
  draft: DirectiveDraft | null;
  onDraftConsumed: () => void;
}) {
  const [file, setFile] = useState<DirectiveFile>({ version: 0, directives: [] });
  const [text, setText] = useState("");
  const [draftSource, setDraftSource] = useState("");
  const [ttl, setTtl] = useState(10);
  const [error, setError] = useState("");

  const reload = useCallback(() => {
    fetchDirectives(run).then(setFile).catch(() => {});
  }, [run]);
  useEffect(reload, [reload]);

  // Consume a guidance draft into the form (veto drafts render their own
  // review card below and are consumed on send/discard).
  useEffect(() => {
    if (draft?.kind === "guidance") {
      setText(draft.text ?? "");
      setDraftSource(draft.source);
      onDraftConsumed();
    }
  }, [draft, onDraftConsumed]);

  const send = async () => {
    if (!text.trim()) return;
    setError("");
    try {
      await postDirective(run, {
        kind: "guidance",
        text: text.trim(),
        ttl_generations: ttl > 0 ? ttl : null,
      });
      setText("");
      setDraftSource("");
      reload();
    } catch (e) {
      setError(String(e));
    }
  };

  const sendVeto = async () => {
    if (draft?.kind !== "lineage_veto") return;
    setError("");
    try {
      await postDirective(run, {
        kind: "lineage_veto",
        candidate_ids: draft.candidate_ids ?? [],
      });
      onDraftConsumed();
      reload();
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <div>
      {draft?.kind === "lineage_veto" && (
        <div className="veto-card" role="alertdialog" aria-label="Review lineage veto">
          <b>Veto lineage — review before sending</b>
          <p>
            Candidate <code className="mono">{(draft.candidate_ids ?? []).join(", ")}</code>{" "}
            and all of its descendants will be excluded from parent selection
            from the next generation on. They stay in the population and the
            record — only future selection is affected. Vetoes do not expire.
          </p>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn-danger" onClick={sendVeto}>Send veto</button>
            <button onClick={onDraftConsumed}>Discard</button>
          </div>
        </div>
      )}

      <h3 style={{ marginTop: 0 }}>Send guidance to the mutation prompts</h3>
      <p className="status-skipped" style={{ marginTop: "-.4rem" }}>
        Takes effect from the next generation. Expires after the TTL so stale
        advice does not linger.
      </p>
      {draftSource && (
        <p className="draft-banner">
          Prefilled from {draftSource} — edit freely; nothing is sent until you
          press Send.{" "}
          <a href="#" onClick={(e) => { e.preventDefault(); setText(""); setDraftSource(""); }}>
            Discard draft
          </a>
        </p>
      )}
      <textarea
        rows={3}
        style={{ width: "100%", font: "inherit", padding: 10,
                 border: "1px solid var(--rule)", borderRadius: 8,
                 background: "var(--card)", color: "var(--ink)" }}
        placeholder="e.g. Stop tuning the prompt wording; try an inverted index for retrieval."
        value={text}
        onChange={(e) => { setText(e.target.value); }}
      />
      <div style={{ display: "flex", gap: 12, alignItems: "center", margin: "10px 0 28px" }}>
        <label className="status-skipped">
          expires after{" "}
          <input type="number" min={0} max={200} value={ttl}
                 style={{ width: 58, font: "inherit", padding: "3px 6px",
                          border: "1px solid var(--rule)", borderRadius: 6,
                          background: "var(--card)", color: "var(--ink)" }}
                 onChange={(e) => setTtl(Number(e.target.value))} />{" "}
          generations (0 = never)
        </label>
        <button className="btn-primary" onClick={send}>Send guidance</button>
        {error && <span className="tag-dup">{error}</span>}
      </div>

      <h3>Active directives (v{file.version})</h3>
      {file.directives.length === 0 ? (
        <p className="status-skipped">
          None yet. Guidance appears in every mutation prompt; vetoes remove a
          candidate and its descendants from parent selection.
        </p>
      ) : (
        <table>
          <thead>
            <tr><th>ID</th><th>KIND</th><th>CONTENT</th><th>TTL</th></tr>
          </thead>
          <tbody>
            {file.directives.map((d) => (
              <tr key={d.id}>
                <td className="mono">{d.id}</td>
                <td>{d.kind}</td>
                <td>{d.kind === "guidance" ? d.text : (d.candidate_ids ?? []).join(", ")}</td>
                <td className="mono">{d.ttl_generations ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
