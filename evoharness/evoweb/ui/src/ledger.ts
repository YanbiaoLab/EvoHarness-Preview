// Quiet/prominent split for the generation ledger: routine generations fold
// into one collapsible line; only events a reviewer acts on stay as rows
// (failures, repair/human operators, new-best jumps). Pure function so the
// grouping rule is testable without React.
import type { HistoryEntry } from "./api";

export type LedgerRow =
  | { kind: "entry"; entry: HistoryEntry; newBest: boolean }
  | {
      kind: "quiet";
      entries: HistoryEntry[];
      genFrom: number;
      genTo: number;
      fitnessFrom: number | null;
      fitnessTo: number | null;
    };

const PROMINENT_OPS = new Set(["repair", "human"]);
const MIN_FOLD = 3; // folding 1-2 rows hides more than it saves

/** history in chronological order -> rows newest-first. */
export function groupLedger(history: HistoryEntry[]): LedgerRow[] {
  const rows: LedgerRow[] = [];
  let quiet: HistoryEntry[] = [];
  let best = -Infinity;

  const flush = () => {
    if (quiet.length === 0) return;
    if (quiet.length < MIN_FOLD) {
      for (const e of quiet) rows.push({ kind: "entry", entry: e, newBest: false });
    } else {
      const fits = quiet.filter((e) => e.fitness != null).map((e) => e.fitness!);
      rows.push({
        kind: "quiet",
        entries: quiet,
        genFrom: quiet[0].generation,
        genTo: quiet[quiet.length - 1].generation,
        fitnessFrom: fits.length ? fits[0] : null,
        fitnessTo: fits.length ? fits[fits.length - 1] : null,
      });
    }
    quiet = [];
  };

  for (const e of history) {
    const newBest = e.status === "ok" && e.fitness != null && e.fitness > best;
    if (newBest) best = e.fitness!;
    const prominent =
      e.status === "failed" ||
      newBest ||
      (e.operator != null && PROMINENT_OPS.has(e.operator));
    if (prominent) {
      flush();
      rows.push({ kind: "entry", entry: e, newBest });
    } else {
      quiet.push(e); // includes "skipped" — routine noise, not a decision point
    }
  }
  flush();
  return rows.reverse();
}
