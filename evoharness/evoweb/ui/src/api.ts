// Typed mirror of evoweb/data.py — the single data contract.

export interface RunSummary {
  name: string;
  recipe: string;
  generation: number;
  target_generations: number | null;
  stopped_reason: string;
  best_fitness: number | null;
  /** Seconds since the checkpoint last moved; null before generation 0. */
  heartbeat_age_s: number | null;
}

export interface CandidateRow {
  id: string;
  generation: number;
  parent_id: string | null;
  island: number;
  operator: string;
  fitness: number | null;
  passed: boolean;
  title: string;
  summary: string;
  model: string;
  signature: string | null;
  duplicate: boolean;
  in_archive: boolean;
  fault: string | null;
  metadata: Record<string, string>;
}

export interface HistoryEntry {
  generation: number;
  status: "ok" | "failed" | "skipped" | "infra_error";
  candidate_id?: string;
  operator?: string;
  fitness?: number;
  parent_id?: string;
  error?: string;
}

export interface RunDetail {
  name: string;
  recipe: string;
  recipe_description: string;
  generation: number;
  target_generations: number | null;
  report: {
    history?: HistoryEntry[];
    best_fitness?: number | null;
    stopped_reason?: string;
  };
  candidates: CandidateRow[];
  series: {
    best_fitness: [number, number][];
    fitness: [number, number][];
  };
  /** Task-emitted eval/* metric series, task-agnostic (per candidate). */
  eval_series: Record<string, [number, number][]>;
  /** Run manifest: configs, brief sha, platform — the task-details panel. */
  manifest: {
    recipe?: string;
    task?: string;
    research_brief_sha256?: string | null;
    platform?: string;
    python?: string;
    search?: Record<string, unknown> & { task_sys_msg?: string };
    population?: Record<string, unknown>;
    plus?: Record<string, unknown>;
  };
  distinct_signatures: number;
}

export interface FeedbackItem {
  item_id: string;
  passed: boolean;
  predicted: string;
  expected: string;
  error_category: string;
}

export interface CandidateDetail extends CandidateRow {
  code: string;
  report: {
    fitness: number;
    passed: boolean;
    fault?: string | null;
    visible_metrics?: Record<string, unknown>;
    stderr_log?: string;
    structured_feedback?: { items: FeedbackItem[]; summary?: string } | null;
  } | null;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json() as Promise<T>;
}

export const fetchRuns = () => get<RunSummary[]>("/api/runs");
export const fetchRun = (name: string) => get<RunDetail>(`/api/runs/${name}`);
export const fetchCandidate = (run: string, cid: string) =>
  get<CandidateDetail>(`/api/runs/${run}/candidates/${cid}`);
export const deleteRun = (name: string) =>
  fetch(`/api/runs/${name}`, { method: "DELETE" });

export interface DirectiveEntry {
  id: string;
  kind: "guidance" | "lineage_veto";
  text?: string;
  candidate_ids?: string[];
  ttl_generations?: number | null;
}

export interface DirectiveFile {
  version: number;
  directives: DirectiveEntry[];
}

/** A prefilled, not-yet-sent directive. Buttons anywhere in the UI produce
 * drafts; only the Control tab sends them, after human review. Nothing is
 * written to the control file until the human presses Send. */
export interface DirectiveDraft {
  kind: "guidance" | "lineage_veto";
  text?: string;
  candidate_ids?: string[];
  /** Where the prefill came from, e.g. "candidate cand-a1b2c3". */
  source: string;
}

export const fetchDirectives = (run: string) =>
  get<DirectiveFile>(`/api/runs/${run}/directives`);

export async function postDirective(
  run: string,
  d: Omit<DirectiveEntry, "id">,
): Promise<void> {
  const res = await fetch(`/api/runs/${run}/directives`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(d),
  });
  if (!res.ok) throw new Error((await res.json()).error ?? `${res.status}`);
}

export interface NewRunRequest {
  name: string;
  recipe: string;
  task: string;
  generations?: number;
  model?: string;
  live?: boolean;
}

export async function createRun(req: NewRunRequest): Promise<void> {
  const res = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error((await res.json()).error ?? `${res.status}`);
}

export const RECIPES = ["e3r", "e3g", "e2", "e1", "e0", "b0"];
export const TASKS = ["demo_counter", "modmul", "equational"];

export const OP_COLORS: Record<string, string> = {
  seed: "#79806F",
  revise: "#177E62",
  rewrite: "#5B50C8",
  recombine: "#B0447E",
  repair: "#C4501F",
  human: "#8A6A14",
};

/* ---- research copilot (docs/research_copilot_design.md §3) ---- */

export interface EvidenceChip {
  type: "candidate" | "cohort" | "series" | "query";
  ref?: string;
  query?: string;
  gen_range?: [number, number];
  note?: string;
}

export interface Finding {
  id: string;
  run: string;
  claim: string;
  severity: "observation" | "actionable" | "warning" | "speculation";
  confidence: string;
  noise_note?: string | null;
  evidence: EvidenceChip[];
  playbook?: string;
  proposals: string[];
}

export interface Proposal {
  id: string;
  level: "L1" | "L2" | "L3";
  title: string;
  motivation: string[];
  patch_file?: string | null;
  version_bumps?: Record<string, string> | null;
  predicted_effect?: string;
  validation_plan?: {
    paired_control?: boolean;
    recipes?: string[];
    budget_est?: string;
  };
  status: "draft" | "accepted" | "rejected";
  rejection_reason?: string;
}

export interface InsightsFile {
  findings: Finding[];
  proposals: Proposal[];
}

export const fetchInsights = (run: string) =>
  get<InsightsFile>(`/api/runs/${run}/insights`);

export const fetchProposalPatch = (run: string, pid: string) =>
  fetch(`/api/runs/${run}/proposals/${pid}/patch`).then((r) =>
    r.ok ? r.text() : "(no patch)",
  );

export async function decideProposal(
  run: string,
  pid: string,
  action: "accept" | "reject",
  reason?: string,
): Promise<Proposal> {
  const res = await fetch(`/api/runs/${run}/proposals/${pid}/decision`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, reason }),
  });
  const body = await res.json();
  if (!res.ok) throw new Error(body.error ?? `${res.status}`);
  return body as Proposal;
}
