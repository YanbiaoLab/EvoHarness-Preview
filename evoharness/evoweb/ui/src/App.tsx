import { useCallback, useEffect, useRef, useState } from "react";
import type { DirectiveDraft, RunDetail, RunSummary } from "./api";
import { deleteRun, fetchRun, fetchRuns } from "./api";
import { EvalSeriesChart, FitnessChart, Hero, RunList } from "./components";
import { TaskPanel } from "./Task";
import { Insights } from "./Insights";
import { fetchInsights } from "./api";
import { Control } from "./Control";
import { Drawer } from "./Drawer";
import { NewRun } from "./NewRun";
import { Candidates, Ledger, Lineage } from "./panels";

type Tab = "ledger" | "insights" | "lineage" | "candidates" | "task" | "control";
const TABS: Tab[] = ["ledger", "insights", "lineage", "candidates", "task", "control"];

type Theme = "light" | "dark";
const initTheme = (): Theme => {
  const saved = localStorage.getItem("evoweb-theme");
  if (saved === "light" || saved === "dark") return saved;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
};

// Restore view state from the URL so a specific run/candidate is shareable.
const initFromUrl = () => {
  const q = new URLSearchParams(location.search);
  const tab = q.get("tab");
  return {
    run: q.get("run"),
    cid: q.get("cid"),
    tab: (TABS as string[]).includes(tab ?? "") ? (tab as Tab) : "ledger",
  };
};

export default function App() {
  const [initial] = useState(initFromUrl);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [current, setCurrent] = useState<string | null>(initial.run);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [tab, setTab] = useState<Tab>(initial.tab);
  const [drawerCid, setDrawerCid] = useState<string | null>(initial.cid);
  const [draft, setDraft] = useState<DirectiveDraft | null>(null);
  const [theme, setTheme] = useState<Theme>(initTheme);
  const [pendingProposals, setPendingProposals] = useState(0);
  const lastGenSeen = useRef(-1);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("evoweb-theme", theme);
  }, [theme]);

  // Deep links: keep ?run=&tab=&cid= in sync (replace, not push — the
  // console has no back-navigation semantics).
  useEffect(() => {
    const q = new URLSearchParams();
    if (current) q.set("run", current);
    if (tab !== "ledger") q.set("tab", tab);
    if (drawerCid) q.set("cid", drawerCid);
    const s = q.toString();
    history.replaceState(null, "", s ? `?${s}` : location.pathname);
  }, [current, tab, drawerCid]);

  const loadRuns = useCallback(() => {
    fetchRuns().then(setRuns).catch(() => {});
  }, []);

  useEffect(() => {
    loadRuns();
    const t = setInterval(loadRuns, 10_000);
    return () => clearInterval(t);
  }, [loadRuns]);

  const refresh = useCallback(async (name: string) => {
    const d = await fetchRun(name);
    setDetail((prev) => {
      lastGenSeen.current = prev?.name === name ? prev.generation : -1;
      return d;
    });
    fetchInsights(name)
      .then((i) =>
        setPendingProposals(
          i.proposals.filter((p) => p.status === "draft").length,
        ),
      )
      .catch(() => setPendingProposals(0));
  }, []);

  // SSE: follow the selected run live.
  useEffect(() => {
    if (!current) return;
    refresh(current).catch(() => setCurrent(null)); // stale deep link
    const es = new EventSource(`/api/runs/${current}/events`);
    es.addEventListener("update", () => refresh(current));
    return () => es.close();
  }, [current, refresh]);

  const onDelete = async (name: string) => {
    const ok = confirm(
      `Delete run "${name}" permanently?\nThis removes its entire directory ` +
      `(population, metrics, checkpoint). This cannot be undone.`,
    );
    if (!ok) return;
    await deleteRun(name);
    if (current === name) {
      setCurrent(null);
      setDetail(null);
      setDrawerCid(null);
    }
    loadRuns();
  };

  // Steering buttons anywhere in the UI land here: stash the prefilled
  // draft and surface it on the Control tab for human review.
  const openDraft = (d: DirectiveDraft) => {
    setDraft(d);
    setDrawerCid(null);
    setTab("control");
  };

  return (
    <div className="layout">
      <nav className="side" aria-label="Runs">
        <div className="side-head">
          <div className="wordmark"><span className="mark">◆</span> EVOHARNESS</div>
          <button
            className="theme-toggle"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label="Toggle color theme"
          >
            {theme === "dark" ? "Light" : "Dark"}
          </button>
        </div>
        <NewRun onStarted={() => setTimeout(loadRuns, 1500)} />
        <RunList runs={runs} current={current}
                 onOpen={(n) => { setCurrent(n); setDrawerCid(null); }}
                 onDelete={onDelete} />
      </nav>
      <main>
        {!detail ? (
          <div className="empty">
            <p>No run selected.</p>
            <p>
              Start one with<br />
              <code>python -m experiments.run_evolution --recipe e3r --task demo_counter --run-dir results/my_run</code>
              <br />then pick it on the left.
            </p>
          </div>
        ) : (
          <>
            <Hero detail={detail} live={current != null} />
            <div className="chartwrap">
              <FitnessChart detail={detail} lastGenSeen={lastGenSeen.current} />
              <EvalSeriesChart detail={detail} />
            </div>
            <div className="tabs" role="tablist">
              {TABS.map((t) => (
                <button key={t} className="tab" role="tab"
                        aria-selected={tab === t} onClick={() => setTab(t)}>
                  {t[0].toUpperCase() + t.slice(1)}
                  {t === "control" && draft && <span className="draft-dot" />}
                  {t === "insights" && pendingProposals > 0 && (
                    <span className="draft-dot" title={`${pendingProposals} 个提案待审`} />
                  )}
                </button>
              ))}
            </div>
            {tab === "ledger" && <Ledger detail={detail} onPick={setDrawerCid} />}
            {tab === "lineage" && <Lineage detail={detail} onPick={setDrawerCid} />}
            {tab === "candidates" && <Candidates detail={detail} onPick={setDrawerCid} />}
            {tab === "task" && <TaskPanel detail={detail} />}
            {tab === "insights" && current && (
              <Insights run={current} onPickCandidate={setDrawerCid} />
            )}
            {tab === "control" && current && (
              <Control run={current} draft={draft}
                       onDraftConsumed={() => setDraft(null)} />
            )}
          </>
        )}
      </main>
      {drawerCid && current && (
        <Drawer run={current} cid={drawerCid}
                onNavigate={setDrawerCid} onClose={() => setDrawerCid(null)}
                onDraft={openDraft} />
      )}
    </div>
  );
}
