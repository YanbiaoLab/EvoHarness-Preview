# Portions derived from SakanaAI/ShinkaEvolve (Apache-2.0)
# Upstream: shinka/edit/apply_diff.py (patch parsing, indent-tolerant match,
#           marker enforcement, diagnostic errors), shinka/edit/apply_full.py,
#           shinka/edit/marker_validation.py, shinka/core/sampler.py and
#           shinka/prompts/* (prompt assembly order, operator selection)
# Upstream revision: 7939f6b44046a2b92e4baa6687b52b23e6236898
# Behavior-aligned port with independent surface: our editable-region marker
# is EDIT-REGION-BEGIN/END and the patch block format is ORIGINAL/UPDATED
# (semantics equal to upstream's search/replace patches). Prompt wording is
# original; assembly order and injected context match upstream. The
# PromptContributor injection slot is an EvoHarness extension placed where
# upstream injects meta recommendations.
"""Mutation operators: patch engine, editable-region markers, prompt builder."""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SearchConfig
from .interfaces import MutationContext, PromptContributor
from .population import Candidate

logger = logging.getLogger(__name__)

# -- editable region markers --------------------------------------------------

EDIT_BEGIN = re.compile(
    r"(?:#|//|!|<!--|\(\*)?[^\S\r\n]*EDIT-REGION-BEGIN[^\S\r\n]*(?:-->|\*\))?"
)
EDIT_END = re.compile(
    r"(?:#|//|!|<!--|\(\*)?[^\S\r\n]*EDIT-REGION-END[^\S\r\n]*(?:-->|\*\))?"
)


def editable_ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges between paired BEGIN/END markers (stack pairing to
    tolerate multiple/nested regions, upstream behavior)."""
    events: list[tuple[int, str]] = []
    for m in EDIT_BEGIN.finditer(text):
        events.append((m.end(), "begin"))
    for m in EDIT_END.finditer(text):
        events.append((m.start(), "end"))
    events.sort()
    ranges: list[tuple[int, int]] = []
    stack: list[int] = []
    for pos, kind in events:
        if kind == "begin":
            stack.append(pos)
        elif stack:
            ranges.append((stack.pop(), pos))
    return ranges


def validate_edit_markers(text: str) -> str | None:
    """Return an error string if markers are missing or unbalanced."""
    n_begin = len(EDIT_BEGIN.findall(text))
    n_end = len(EDIT_END.findall(text))
    if n_begin == 0 or n_end == 0:
        return (
            "missing EDIT-REGION markers: the code must contain matching "
            "EDIT-REGION-BEGIN and EDIT-REGION-END comment lines"
        )
    if n_begin != n_end:
        return (
            f"unbalanced EDIT-REGION markers: {n_begin} BEGIN vs {n_end} END"
        )
    return None


# -- patch engine --------------------------------------------------------------

PATCH_PATTERN = re.compile(
    r"<{7}\s*ORIGINAL\s*\n(.*?)\n\s*={7}\s*\n(.*?)\n\s*>{7}\s*UPDATED\s*",
    re.DOTALL,
)

_TITLE_RE = re.compile(r"^\s*TITLE:\s*(.+)$", re.MULTILINE)
_SUMMARY_RE = re.compile(r"^\s*SUMMARY:\s*(.+)$", re.MULTILINE)


def parse_change_header(text: str) -> tuple[str, str]:
    title = _TITLE_RE.search(text)
    summary = _SUMMARY_RE.search(text)
    return (
        title.group(1).strip() if title else "",
        summary.group(1).strip() if summary else "",
    )


def extract_code_block(text: str, language: str = "python") -> str | None:
    """Last fenced code block in the LLM output (rewrite/recombine/repair)."""
    fence = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", re.DOTALL)
    blocks = fence.findall(text)
    return blocks[-1].strip("\n") if blocks else None


_FILE_BLOCK_RE = re.compile(
    r"^###\s*FILE:\s*(\S+)\s*\n```[a-zA-Z0-9_+-]*\s*\n(.*?)```",
    re.DOTALL | re.MULTILINE,
)


# A path the model echoed from the instructions rather than chose. Reasoning
# models quote the response format back to themselves while thinking, and the
# block regex cannot tell that apart from a real answer: run modmul_r1
# admitted a genome file literally named "<relative/path.py>" whose body was
# the model's own reasoning transcript. It scored top marks, because the three
# real files were untouched and the cached weights made it tie its parent.
_PLACEHOLDER_PATH = re.compile(r"[<>{}\[\]|*?\"']|^\s*$|\.\.")


def _is_usable_path(path: str) -> bool:
    if _PLACEHOLDER_PATH.search(path) or path.startswith("/"):
        return False
    # A genome file is source, not prose. Requiring a suffix rejects headings
    # the model wrote as if they were paths.
    return "." in Path(path).name and not path.endswith(".")


def parse_file_blocks(text: str) -> dict[str, str]:
    """Parse `### FILE: path` + fenced-block sections into {path: content}
    (M2.5 multi-file answers). Empty dict = not a multi-file answer; the
    caller falls back to the single-block lane. Bodies are normalized to end
    with a newline so derived git patches stay free of no-newline noise.

    Paths that look like the instructions rather than an answer are dropped,
    not passed on to the workspace."""
    out: dict[str, str] = {}
    for path, body in _FILE_BLOCK_RE.findall(text):
        if not _is_usable_path(path):
            logger.warning("ignoring implausible file path in answer: %r", path)
            continue
        out[path] = body if body.endswith("\n") else body + "\n"
    return out


@dataclass
class PatchOutcome:
    new_code: str | None
    error: str | None
    n_applied: int = 0

    @property
    def ok(self) -> bool:
        return self.new_code is not None


def _strip_marker_lines(block: str) -> str:
    lines = [
        ln
        for ln in block.split("\n")
        if "EDIT-REGION-BEGIN" not in ln and "EDIT-REGION-END" not in ln
    ]
    return "\n".join(lines)


def _reindent(block: str, delta: int) -> str:
    out = []
    for ln in block.split("\n"):
        if not ln.strip():
            out.append(ln)
        elif delta >= 0:
            out.append(" " * delta + ln)
        else:
            strip = min(-delta, len(ln) - len(ln.lstrip(" ")))
            out.append(ln[strip:])
    return "\n".join(out)


def _find_indented_match(
    text: str, search: str, replace: str
) -> tuple[int, str, str] | None:
    """Exact match first; otherwise infer a uniform indentation shift from the
    first non-empty search line and retry (upstream indent tolerance)."""
    idx = text.find(search)
    if idx != -1:
        return idx, search, replace
    first = next((ln for ln in search.split("\n") if ln.strip()), None)
    if first is None:
        return None
    want = first.strip()
    search_indent = len(first) - len(first.lstrip(" "))
    for line in text.split("\n"):
        if line.strip() == want:
            delta = (len(line) - len(line.lstrip(" "))) - search_indent
            if delta == 0:
                continue
            shifted_search = _reindent(search, delta)
            idx = text.find(shifted_search)
            if idx != -1:
                return idx, shifted_search, _reindent(replace, delta)
    return None


def _closest_block(text: str, search: str) -> str | None:
    """Best-matching window of the same line count, for diagnostics."""
    search_lines = search.split("\n")
    text_lines = text.split("\n")
    n = len(search_lines)
    if n == 0 or len(text_lines) < 1:
        return None
    best_ratio, best_window = 0.0, None
    for i in range(max(1, len(text_lines) - n + 1)):
        window = "\n".join(text_lines[i : i + n])
        ratio = difflib.SequenceMatcher(None, search, window).ratio()
        if ratio > best_ratio:
            best_ratio, best_window = ratio, window
    if best_window is not None and best_ratio > 0.6:
        diff = "\n".join(
            difflib.unified_diff(
                search.split("\n"),
                best_window.split("\n"),
                "your ORIGINAL block",
                "closest code in file",
                lineterm="",
            )
        )
        return diff
    return None


class PatchEngine:
    """Parses ORIGINAL/UPDATED patch blocks and applies them sequentially,
    restricted to editable regions."""

    def apply(self, original: str, llm_output: str) -> PatchOutcome:
        blocks = PATCH_PATTERN.findall(llm_output)
        if not blocks:
            return PatchOutcome(
                None,
                "no patch blocks found: expected one or more "
                "<<<<<<< ORIGINAL / ======= / >>>>>>> UPDATED blocks",
            )
        text = original
        applied = 0
        for raw_search, raw_replace in blocks:
            search = _strip_marker_lines(raw_search)
            replace = _strip_marker_lines(raw_replace)
            if not search.strip():
                return PatchOutcome(
                    None, "empty ORIGINAL block: it must quote existing code"
                )
            found = _find_indented_match(text, search, replace)
            if found is None:
                hint = _closest_block(text, search)
                msg = (
                    "ORIGINAL block not found in the current code. It must be "
                    "an exact copy (including indentation) of existing lines."
                )
                if hint:
                    msg += (
                        "\nClosest match differs as follows:\n" + hint +
                        "\nFix the ORIGINAL block to match the file exactly, "
                        "or use a smaller block."
                    )
                return PatchOutcome(None, msg, applied)
            idx, search, replace = found
            ranges = editable_ranges(text)
            end = idx + len(search)
            inside = any(start <= idx and end <= stop for start, stop in ranges)
            if not inside:
                return PatchOutcome(
                    None,
                    "the ORIGINAL block touches code outside the editable "
                    "region: only code between EDIT-REGION-BEGIN and "
                    "EDIT-REGION-END may be modified",
                    applied,
                )
            text = text[:idx] + replace + text[end:]
            applied += 1
        err = validate_edit_markers(text)
        if err:
            return PatchOutcome(None, f"patch corrupted markers: {err}", applied)
        return PatchOutcome(text, None, applied)


def apply_rewrite(original: str, llm_output: str, language: str) -> PatchOutcome:
    """Full rewrite: the fenced block replaces the whole program; markers must
    survive (upstream full-rewrite semantics)."""
    code = extract_code_block(llm_output, language)
    if code is None:
        return PatchOutcome(None, "no fenced code block found in the response")
    err = validate_edit_markers(code)
    if err:
        return PatchOutcome(None, err)
    return PatchOutcome(code, None, 1)


# -- operator sampling ([parity]: fixed probabilities, recombine needs peers) --

def sample_operator(
    cfg: SearchConfig, has_inspirations: bool, rng: np.random.Generator
) -> str:
    ops = list(cfg.operators)
    probs = list(cfg.operator_probs)
    if not has_inspirations and "recombine" in ops:
        probs = [p for o, p in zip(ops, probs) if o != "recombine"]
        ops = [o for o in ops if o != "recombine"]
        total = sum(probs)
        probs = [p / total for p in probs]
    return str(ops[int(rng.choice(len(ops), p=np.array(probs)))])


# -- prompt construction --------------------------------------------------------

_BASE_SYS = (
    "You are an expert developer evolving a program through guided mutation. "
    "You will see the current program, its evaluation results, and possibly "
    "other high-performing programs for reference. Propose one focused "
    "improvement that maximizes the fitness score.\n"
)
_WORKSPACE_AGENT_SPEC = """
# Workspace-agent execution
Use the available tools to inspect and modify the materialized candidate
workspace. Run focused diagnostics when useful. Do not return source code or
patch blocks as a substitute for editing files. Before finishing, ensure the
workspace contains the complete proposed implementation. Your final response
must contain only:
TITLE: <short name>
SUMMARY: <one or two sentences describing the completed change>
"""
_MULTIFILE_FORMAT = """## Response format (multi-file workspace)
This candidate is a multi-file workspace. Reply with one block per file you
change, giving the FULL new content of that file rather than a diff. Use a
line naming the file, then a fenced Python block:

### FILE: arch.py
(a fenced ```python block containing the entire new arch.py)

### FILE: train.py
(a fenced ```python block containing the entire new train.py)

Name real files from the list above. Include only files you actually change.
Never use `..` or absolute paths. The entry file must keep existing.
"""

# A revise on a MULTI-FILE genome. Patch blocks carry no file path, so on a
# multi-file workspace they can only ever reach the entry file — and the
# constant worth changing usually lives elsewhere. Worse, the multi-file
# format instruction is appended to the same prompt, so the model received
# two contradictory formats and every revise was rejected for complying with
# the wrong one. Here the operator's INTENT survives and the format comes
# from _MULTIFILE_FORMAT alone.
_REVISE_MULTIFILE_SPEC = """
# What a revise is
Make ONE small, targeted change — the smallest edit that could plausibly
move the score. Do not restructure, do not rewrite a file wholesale, and do
not bundle unrelated improvements. Keep everything you are not deliberately
changing byte-identical, including comments and docstrings.
"""

_REVISE_SPEC = """
# Response format
Reply with exactly:
TITLE: <short name for the change>
SUMMARY: <one or two sentences describing the change and why it should help>
Then one or more patch blocks, each of the form:
<<<<<<< ORIGINAL
(exact copy of existing lines, including indentation)
=======
(replacement lines)
>>>>>>> UPDATED

Rules:
- Only modify code between EDIT-REGION-BEGIN and EDIT-REGION-END comments;
  never modify or remove the marker lines themselves.
- Each ORIGINAL block must be a verbatim copy of current code.
- Blocks are applied in order; later blocks see earlier edits applied.
- Keep the program's inputs/outputs and public interface unchanged.
"""

_REWRITE_RULES = """
# Response format
Reply with exactly:
TITLE: <short name>
SUMMARY: <one or two sentences>
Then the complete new program in a single fenced code block.

Rules:
- Rewrite only the code between EDIT-REGION-BEGIN and EDIT-REGION-END;
  reproduce everything outside the markers unchanged, and keep the marker
  comment lines in place.
- Keep the program's inputs/outputs and public interface unchanged.
"""

# The same rules minus the format half, for a multi-file genome. rewrite,
# recombine and repair all end in _REWRITE_RULES, which asks for ONE fenced
# block and for EDIT-REGION markers to be preserved. A multi-file genome has
# no markers — the modmul genome has none in any of its three files — and
# _MULTIFILE_FORMAT is appended to the same prompt asking for one block PER
# FILE. So the model was handed two contradictory formats and told to keep
# markers that do not exist. Every repair in runs modmul_r7 and modmul_r8's
# smoke was rejected for "missing EDIT-REGION markers".
#
# Same defect revise had, and _REVISE_MULTIFILE_SPEC only fixed revise.
_MULTIFILE_RULES = """
Rules:
- Keep the program's inputs/outputs and public interface unchanged.
"""


def _for_multifile(spec: str) -> str:
    """Strip the single-file format rules; _MULTIFILE_FORMAT replaces them."""
    return (
        spec.replace(_REWRITE_RULES, _MULTIFILE_RULES)
        .replace("the editable region", "the program")
    )


# Five guidance variants, sampled uniformly ([parity] with upstream's five
# full-rewrite system prompt variants; wording original).
_REWRITE_VARIANTS = [
    "Rewrite the editable region to maximize performance on the stated "
    "metrics.\n" + _REWRITE_RULES,
    "Rewrite the editable region using a fundamentally different algorithm "
    "or strategy than the current one.\n" + _REWRITE_RULES,
    "Study the reference programs shown earlier, then rewrite the editable "
    "region borrowing their strongest ideas without copying them verbatim.\n"
    + _REWRITE_RULES,
    "Restructure the editable region: reorganize data flow, decompose or "
    "merge functions, simplify control flow — then optimize.\n"
    + _REWRITE_RULES,
    "Keep the current algorithm but systematically retune its parameters, "
    "thresholds and configuration values.\n" + _REWRITE_RULES,
]

_RECOMBINE_SPEC = (
    "Merge the current program with the partner program shown below into one "
    "stronger program, combining their best ideas.\n" + _REWRITE_RULES
)

_REPAIR_SPEC = (
    "The program below FAILED evaluation. Analyze the error output, find the "
    "root cause, and produce a corrected program. Correctness first, "
    "performance second.\n" + _REWRITE_RULES
)


_OPERATOR_INTENT = {
    "revise": (
        "# This mutation: REVISE\n"
        "Make one focused, targeted change to the current program. Keep its "
        "overall structure and strategy; improve a specific weakness you can "
        "point to in the evaluation feedback."
    ),
    "rewrite": (
        "# This mutation: REWRITE\n"
        "Restructure the program's approach rather than tuning it. Prefer a "
        "strategy meaningfully different from the current one, even at the "
        "risk of scoring worse — small safe edits are what the REVISE "
        "operator is for."
    ),
    # Filled in per session: whether the reference program's source is in the
    # prompt or behind a tool depends on the runtime, and naming the wrong one
    # sends the agent looking for a tool it does not have.
    "recombine": (
        "# This mutation: RECOMBINE\n"
        "Merge the strongest ideas of a reference program into the current "
        "one. {read} then combine rather than replace."
    ),
}

#: How to reach a reference program's source when no tool can fetch it — it is
#: already in the prompt, because a listing the reader cannot open is worse
#: than no listing at all.
_PEER_SOURCE_INLINE = "Study the reference programs above,"


def _operator_intent(operator: str, peer_fetch_tool: str | None) -> str:
    """What the operator asks the agent to DO.

    Distinct from the operator's response-format spec, which a tool-using
    agent never follows. Without this the operator is invisible to it.
    """
    text = _OPERATOR_INTENT.get(
        operator, f"# This mutation: {operator.upper()}"
    )
    if "{read}" not in text:
        return text
    return text.format(
        read=(
            f"Read the reference with {peer_fetch_tool} first,"
            if peer_fetch_tool
            else _PEER_SOURCE_INLINE
        )
    )


def _render_candidate(
    c: Candidate,
    language: str,
    heading: str,
    *,
    include_code: bool = True,
    max_chars: int = 24_000,
    note: str | None = None,
) -> str:
    """Render one candidate for the prompt.

    `include_code=False` emits an inventory instead of file bodies, for
    readers that can fetch the text themselves. Full rendering is the only
    part of the prompt whose size tracks the size of the program being
    evolved rather than a budget, so it is also the first thing to breach
    a context window as a project grows: bound it.
    """
    parts = [f"### {heading}"]

    if note:
        # Why THIS program was chosen. Without it a policy's pick reads as
        # just another ranked reference and the signal is wasted.
        parts.append(f"Note: {note}")

    if c.change_title:
        parts.append(f"Change: {c.change_title} — {c.change_summary}")
    if c.report:
        parts.append(c.report.render_for_prompt())
    texts = c.workspace.texts()
    if not include_code:
        inventory = ", ".join(
            f"{path} ({len(texts[path].splitlines())} lines)"
            for path in sorted(texts)
        )
        parts.append(f"Files: {inventory}")
        return "\n".join(parts)
    if len(texts) > 1:
        # Multi-file genome: render every file in the SAME format the LLM
        # must answer in (### FILE: blocks) — the example IS the spec.
        for path in sorted(texts):
            parts.append(f"### FILE: {path}\n```{language}\n{texts[path]}\n```")
    else:
        parts.append(f"```{language}\n{c.workspace.main_text()}\n```")
    rendered = "\n".join(parts)
    if len(rendered) > max_chars:
        rendered = (
            rendered[:max_chars]
            + f"\n... [truncated at {max_chars} characters]"
        )
    return rendered


class PromptBuilder:
    """Assembles (system, user) messages for each operator.

    Assembly order matches upstream: base + task message, then contributor
    sections (the slot where upstream injects meta recommendations), then the
    operator format spec. The user message carries reference programs, the
    parent, and its evaluation results.
    """

    def __init__(
        self,
        task_sys_msg: str,
        language: str = "python",
        contributors: list[PromptContributor] | None = None,
        rng: np.random.Generator | None = None,
        workspace_agent: bool = False,
        peer_fetch_tool: str | None = None,
        workspace_read_tool: str | None = None,
    ):
        self.task_sys_msg = task_sys_msg
        self.language = language
        self.contributors = contributors or []
        self.rng = rng or np.random.default_rng()
        if not isinstance(workspace_agent, bool):
            raise TypeError("workspace_agent must be bool")
        for name, value in (
            ("peer_fetch_tool", peer_fetch_tool),
            ("workspace_read_tool", workspace_read_tool),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be non-empty when set")
        self.workspace_agent = workspace_agent
        #: The tool this reader can call to fetch another candidate's source,
        #: or None when it has none. None does not mean a shorter prompt — it
        #: means the source must be rendered inline, because there is no
        #: second chance to fetch it.
        self.peer_fetch_tool = peer_fetch_tool
        #: The tool this reader calls to read its own materialized files, or
        #: None to describe the workspace without naming a tool. Losing this
        #: one costs a name, not data: the files are already on disk and any
        #: file tool reaches them.
        self.workspace_read_tool = workspace_read_tool

    def _system(self, ctx: MutationContext, spec: str) -> str:
        parts = [_BASE_SYS]
        if self.task_sys_msg:
            parts.append("# Task\n" + self.task_sys_msg)
        for contrib in self.contributors:
            section = contrib.contribute(ctx)
            if section:
                parts.append(section)
        if self.workspace_agent:
            # The operator spec is a RESPONSE-FORMAT spec: patch blocks for
            # revise, a full code block for rewrite. A workspace agent edits
            # files with tools and returns only TITLE/SUMMARY, so including
            # it contradicts _WORKSPACE_AGENT_SPEC — and it was the only
            # thing that differed between operators here, which is why three
            # operators produced byte-identical edits from one parent.
            parts.append(_operator_intent(ctx.operator, self.peer_fetch_tool))
            parts.append(_WORKSPACE_AGENT_SPEC)
        else:
            parts.append(spec)
        return "\n\n".join(parts)

    def _history(self, ctx: MutationContext) -> str:
        # Reference programs exist only in the population store. A reader that
        # can fetch them gets an inventory to expand, which keeps them off
        # every turn of the session; a reader that cannot gets the source,
        # because an inventory it has no way to open is worse than no
        # inventory — it names programs and then withholds them.
        #
        # This asks about the tool rather than about agentic mode. The two
        # used to be the same question and stopped being one when an external
        # runtime became able to drive proposals with its own tools.
        fetch = self.peer_fetch_tool
        code = fetch is None
        sections = []
        for c in ctx.archive_inspirations:
            sections.append(
                _render_candidate(
                    c, self.language, f"Reference program (archive) id={c.id}",
                    include_code=code,
                    note=ctx.inspiration_notes.get(c.id),
                )
            )
        for c in ctx.top_k_inspirations:
            sections.append(
                _render_candidate(
                    c, self.language, f"Reference program (top) id={c.id}",
                    include_code=code,
                    note=ctx.inspiration_notes.get(c.id),
                )
            )
        ledger = self._failed_attempts(ctx)
        if not sections:
            return ledger
        header = "## Previously evaluated programs"
        if fetch:
            header += f"\nRead any of them with {fetch}(candidate_id, path)."
        out = header + "\n\n" + "\n\n".join(sections)
        return out + ledger

    @staticmethod
    def _failed_attempts(ctx: MutationContext) -> str:
        """已试过而没涨分的改动:只列标题和分差,不列代码。

        参考程序按分数选,分数持平的 run 里失败尝试因此不可见,同一类改动会
        被反复提出。
        """
        if not ctx.failed_attempts:
            return ""
        lines = "\n".join(
            f"- {title} — {delta:+.4f}" for title, delta in ctx.failed_attempts
        )
        return (
            "\n\n## Already tried, no gain\n"
            "Changes proposed earlier in this run that did not improve the "
            "score. Proposing the same idea again costs a full evaluation and "
            "returns the same number.\n\n" + lines
        )

    def build(self, ctx: MutationContext) -> tuple[str, str]:
        multifile = (
            len(ctx.parent.workspace.texts()) > 1 and not self.workspace_agent
        )
        if ctx.operator == "revise":
            spec = _REVISE_MULTIFILE_SPEC if multifile else _REVISE_SPEC
        elif ctx.operator == "rewrite":
            spec = _REWRITE_VARIANTS[int(self.rng.integers(len(_REWRITE_VARIANTS)))]
            if multifile:
                spec = _for_multifile(spec)
        elif ctx.operator == "recombine":
            spec = _for_multifile(_RECOMBINE_SPEC) if multifile \
                else _RECOMBINE_SPEC
        else:
            raise ValueError(f"build() cannot handle operator {ctx.operator!r}")
        system = self._system(ctx, spec)

        user_parts = []
        history = self._history(ctx)
        if history:
            user_parts.append(history)
        # The workspace agent already has the parent's files materialised on
        # disk; repeating them in the prompt pays for what it can read for
        # free, on every turn of the session.
        user_parts.append(
            _render_candidate(
                ctx.parent,
                self.language,
                "Current program",
                include_code=not self.workspace_agent,
            )
        )
        if self.workspace_agent:
            user_parts.append(
                "The current program's files are already in your workspace. "
                + (
                    f"Read them with {self.workspace_read_tool} before editing."
                    if self.workspace_read_tool
                    # An external runtime brings its own file tools under its
                    # own names. Naming ours sends the candidate after a tool
                    # it does not have; its own tool list already documents
                    # what it does have.
                    else "Read them with your file tools before editing."
                )
            )
        if multifile:
            user_parts.append(_MULTIFILE_FORMAT)
            
        if ctx.operator == "recombine":
            peers = ctx.archive_inspirations + ctx.top_k_inspirations
            partner = peers[int(self.rng.integers(len(peers)))]
            user_parts.append(
                _render_candidate(partner, self.language, "Partner program")
            )
        user_parts.append(
            "Propose your improvement now, following the response format."
        )
        return system, "\n\n".join(user_parts)

    def build_repair(self, cand: Candidate) -> tuple[str, str]:
        ctx = MutationContext(
            parent=cand,
            archive_inspirations=[],
            top_k_inspirations=[],
            operator="repair",
            generation=cand.generation,
        )
        multifile = (
            len(cand.workspace.texts()) > 1 and not self.workspace_agent
        )
        system = self._system(
            ctx, _for_multifile(_REPAIR_SPEC) if multifile else _REPAIR_SPEC
        )
        parts = [
            _render_candidate(cand, self.language, "Failing program"),
        ]
        if multifile:
            parts.append(_MULTIFILE_FORMAT)
        if cand.report:
            if cand.report.fault:
                parts.append(f"## Failure reason\n{cand.report.fault}")
            if cand.report.stderr_log:
                parts.append("## stderr (tail)\n" + cand.report.stderr_log[-2000:])
            if cand.report.stdout_log:
                parts.append("## stdout (tail)\n" + cand.report.stdout_log[-2000:])
        parts.append("Produce the corrected program now.")
        return system, "\n\n".join(parts)
