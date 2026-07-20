// Task-details tab: what EXACTLY this run is optimizing, and under which
// frozen contract — recipe, eval versions (protocol §6 provenance carried on
// candidates), the full task_sys_msg injected into every mutation, the brief
// fingerprint, and the three config dataclasses from the run manifest.
import type { RunDetail } from "./api";

function KvTable({
  obj,
  skip = [],
}: {
  obj: Record<string, unknown> | undefined;
  skip?: string[];
}) {
  const rows = Object.entries(obj ?? {}).filter(([k]) => !skip.includes(k));
  if (rows.length === 0) return <p className="r-sub">(empty)</p>;
  return (
    <table className="kv">
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <td>{k}</td>
            <td className="mono">
              {typeof v === "object" ? JSON.stringify(v) : String(v)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function TaskPanel({ detail }: { detail: RunDetail }) {
  const m = detail.manifest ?? {};
  const search = m.search ?? {};
  const sysMsg = (search.task_sys_msg as string) ?? "";
  // Eval-side provenance rides on graded candidates (RemoteGrader §6 /
  // GradeFnGrader); take it from the first candidate that carries it.
  const carrier = detail.candidates.find((c) => c.metadata?.task_version);
  const versions = carrier?.metadata ?? {};
  return (
    <div className="taskpanel">
      <section className="card-block">
        <h3>Task &amp; recipe</h3>
        <table className="kv">
          <tbody>
            <tr>
              <td>task</td>
              <td>
                <b>{(m as Record<string, unknown>).task as string ?? "?"}</b>
              </td>
            </tr>
            <tr>
              <td>recipe</td>
              <td>
                <b>{detail.recipe}</b> — {detail.recipe_description}
              </td>
            </tr>
            <tr>
              <td>task_version / eval_set_version</td>
              <td className="mono">
                {versions.task_version ?? "(not carried by candidates)"} /{" "}
                {versions.eval_set_version ?? "—"}
              </td>
            </tr>
            <tr>
              <td>research brief sha256</td>
              <td className="mono">{m.research_brief_sha256 ?? "(none)"}</td>
            </tr>
            <tr>
              <td>platform / python</td>
              <td className="mono">
                {m.platform ?? "—"} / {m.python ?? "—"}
              </td>
            </tr>
          </tbody>
        </table>
      </section>

      <section className="card-block">
        <h3>
          Task system message{" "}
          <span className="r-sub">
            injected into every mutation prompt · {sysMsg.length} chars
          </span>
        </h3>
        {sysMsg ? (
          <details open={sysMsg.length < 4000}>
            <summary>show / hide</summary>
            <pre className="codeblock">{sysMsg}</pre>
          </details>
        ) : (
          <p className="r-sub">(empty — the task ships no system message)</p>
        )}
      </section>

      <section className="card-block">
        <h3>SearchConfig</h3>
        <KvTable obj={search} skip={["task_sys_msg"]} />
      </section>
      <section className="card-block">
        <h3>PopulationConfig</h3>
        <KvTable obj={m.population} />
      </section>
      <section className="card-block">
        <h3>PlusConfig (C1 / C2 / C3)</h3>
        <KvTable obj={m.plus} />
      </section>
    </div>
  );
}
