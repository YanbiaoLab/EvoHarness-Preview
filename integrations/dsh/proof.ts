/**
 * Five tools that let a dsh session drive EvoHarness's proof graph.
 *
 * Copy or symlink this next to the other evo-harness example plugins and add a
 * row for it to the host cordis config; see `host.proof.cordis.yml` beside it.
 *
 * Every tool shells out to `python -m evoharness.proof.cli`, for the reason
 * `peer.ts` already gives: the schema lives in Python, and a second reader
 * written here would drift from it silently, because both would keep
 * answering. Nothing about the graph is modelled on this side.
 *
 * **What the model gets and what it does not.** It can research, open a goal,
 * propose a decomposition, dispatch a lemma and read the result. It cannot
 * adjudicate any of them: `proof_sketch` returns Lean's verdict on the
 * decomposition, `proof_attack` returns an outcome derived from the run's own
 * report, and `proof_assemble` returns the compiler's answer about the
 * finished proof. That division is the point -- the model says which way is
 * worth trying, never whether the result is correct.
 *
 * @module evoharness/integrations/dsh/proof
 */

import { join } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import type { ToolRunContext } from '@deepseek-ai/dsh-tools'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { HARNESS_ENV, callHarness, requireEnv } from './cli.ts'

export const name = 'evo-proof'
export const inject = ['tools']

/** Set only when goals import Mathlib; absent means a bare `lean`. */
const LEAN_PROJECT = 'EVO_LEAN_PROJECT'

/** The session workspace subdirectory holding `graph.db` and every run. */
const WORK_DIR = '.evo'

const MODULE = 'evoharness.proof.cli'

/** Render whatever the CLI printed, unchanged. It is already one json object. */
const passthrough = {
  schema: { type: 'string' as const },
  render: (_args: unknown, value: string) => [{ type: 'text' as const, text: value }],
}

/**
 * Resolve the environment at CALL time, as `peer.ts` does.
 *
 * Doing it in `apply` would fail plugin load when a variable is missing, which
 * takes the whole runtime down; doing it here fails one tool with a message
 * naming what to set, and the session carries on.
 */
function harness(): { python: string, root: string } {
  const env = requireEnv(HARNESS_ENV)
  return {
    python: env['EVO_PYTHON'] as string,
    root: env['EVO_HARNESS_ROOT'] as string,
  }
}

/**
 * Where THIS session's graph lives: `<session workspace>/.evo`.
 *
 * The workspace is the session's own, read per call from its header, the way
 * `tool-bash`, `tool-lsp` and the filesystem tools read it. It is deliberately
 * not an environment variable: one web server process serves many sessions
 * with different workspaces, so a process-level value cannot express this and
 * `DSH_CWD` is simply absent on that path.
 *
 * **A session without a workspace is refused, not defaulted.** The sandbox
 * falls back to a configured root because writing a file in the wrong place is
 * visible. This is not that: a fallback would silently open a SECOND graph,
 * then spend real money filling it while `proof_status` in the intended
 * directory keeps answering `no goals`. Refusing costs one turn; the fallback
 * costs the run and says nothing.
 */
function workspace(exec: ToolRunContext): string {
  const cwd = exec.agent?.session.header.cwd
  if (cwd === undefined || cwd === '') {
    throw new Error(
      'this session has no workspace directory, so there is nowhere to keep '
      + `the proof graph (it lives in <workspace>/${WORK_DIR}). Start the `
      + 'session in the directory the proof belongs to and call again.',
    )
  }
  return join(cwd, WORK_DIR)
}

/** Flags every subcommand accepts, from the session and the deployment. */
function commonFlags(exec: ToolRunContext): string[] {
  const project = process.env[LEAN_PROJECT]
  return [
    '--work', workspace(exec),
    ...project === undefined || project === '' ? [] : ['--lean-project', project],
  ]
}

export function apply(ctx: Context) {

  ctx.tools.register(defineTool({
    name: 'proof_open',
    description:
      'Put a Lean goal on the board and get its goal_id. `statement` is a '
      + 'declaration SIGNATURE: everything up to but NOT including `:=`, for '
      + 'example "theorem foo (a b : Nat) : a + b = b + a". Calling it twice '
      + 'with the same proposition returns the same goal, so it is safe to '
      + 'reopen one you are already working on.',
    parameters: {
      statement: { type: 'string', description: 'Declaration signature.', required: true },
      preamble: { type: 'string', description: 'Imports, e.g. "import Mathlib".' },
      label: { type: 'string', description: 'A name for this line of enquiry.' },
    },
    output: passthrough,
    execute: (args, exec) => callHarness(
      harness(),
      ['-m', MODULE, ...commonFlags(exec), 'open',
        '--statement', String(args.statement),
        ...(args.preamble ? ['--preamble', String(args.preamble)] : []),
        ...(args.label ? ['--label', String(args.label)] : [])],
      exec.signal, 'opening a goal',
    ),
  }))

  ctx.tools.register(defineTool({
    name: 'proof_status',
    description:
      'Read the board: which goals are open, proved or exhausted, what has '
      + 'been attempted on each, and what has been spent. Call it before '
      + 'deciding what to do next; it is cheap and it is the only place the '
      + 'real state lives.',
    parameters: {
      goal_id: { type: 'string', description: 'One goal; omit for everything.' },
    },
    output: passthrough,
    execute: (args, exec) => callHarness(
      harness(),
      ['-m', MODULE, ...commonFlags(exec), 'status',
        ...(args.goal_id ? ['--goal', String(args.goal_id)] : [])],
      exec.signal, 'reading the board',
    ),
  }))

  ctx.tools.register(defineTool({
    name: 'proof_sketch',
    description:
      'Propose a decomposition and let LEAN judge it. `proposal` is a json '
      + 'object: {"lemmas": [{"name": ..., "signature": ...}], "parent_body": '
      + '"<a Lean term or `by ...` closing the goal from those lemmas>"}. '
      + 'It is accepted only if the file compiles AND `sorry` appears solely '
      + 'in the proposed lemmas -- a parent body containing `sorry` moves the '
      + 'goal instead of reducing it, and a lemma restating an ancestor is '
      + 'refused for making no progress. Rejection comes with a reason; read '
      + 'it and propose differently rather than repeating. Costs one Lean '
      + 'compile and no model budget, so check a route before spending on it.',
    parameters: {
      goal_id: { type: 'string', description: 'The goal to decompose.', required: true },
      proposal: { type: 'string', description: 'The json object described above.', required: true },
    },
    output: passthrough,
    execute: (args, exec) => callHarness(
      harness(),
      ['-m', MODULE, ...commonFlags(exec), 'sketch',
        '--goal', String(args.goal_id),
        '--proposal', String(args.proposal)],
      exec.signal, 'validating a decomposition',
    ),
  }))

  ctx.tools.register(defineTool({
    name: 'proof_attack',
    description:
      'Hand one goal to EvoHarness to solve. This is the expensive call: it '
      + 'runs an agent that edits Lean and compiles it, repeatedly. Attack '
      + 'LEAVES -- goals with no accepted decomposition -- because a goal '
      + 'whose route is already being worked is carried by its subgoals. The '
      + 'outcome comes from the run itself: `proved`, `task-failed` (tried and '
      + 'could not), or one of `infra-failed` / `budget-exhausted` / '
      + '`interrupted`, which say nothing about the goal and are not evidence '
      + 'that it is hard. Read `warnings` in the result BEFORE spending again: '
      + 'attacking a subgoal whose route Lean has not accepted is allowed on '
      + 'purpose -- a proved lemma is an asset whatever asked for it -- but it '
      + 'closes nothing upstream until that route is accepted.',
    parameters: {
      goal_id: { type: 'string', description: 'The goal to solve.', required: true },
      budget: { type: 'number', description: 'Ceiling for this call. Default 10.' },
      level: {
        type: 'string',
        description: 'L1 one-shot, or L2 (default) an agent session that can call Lean.',
      },
      allow_unaccepted_route: {
        type: 'boolean',
        description:
          'Spend even though no accepted route uses this goal. Say why in the '
          + 'conversation before setting it.',
      },
    },
    output: passthrough,
    execute: (args, exec) => callHarness(
      harness(),
      ['-m', MODULE, ...commonFlags(exec), 'attack',
        '--goal', String(args.goal_id),
        ...(args.budget === undefined ? [] : ['--budget', String(args.budget)]),
        ...(args.level ? ['--level', String(args.level)] : []),
        ...(args.allow_unaccepted_route ? ['--allow-unaccepted-route'] : [])],
      exec.signal, 'attacking a goal',
    ),
  }))

  ctx.tools.register(defineTool({
    name: 'proof_assemble',
    description:
      'Put the proved lemmas back together and compile the whole thing. This '
      + 'is the ONLY thing that certifies a root goal: per-node greens were '
      + 'earned minutes apart and only compiling the finished article catches '
      + 'a lemma that typechecks alone and not in company. Returns whether it '
      + 'compiled and which axioms Lean says it depends on.',
    parameters: {
      goal_id: { type: 'string', description: 'The root goal.', required: true },
      out: { type: 'string', description: 'Where to write the assembled proof.' },
      show_text: { type: 'boolean', description: 'Include the proof in the reply.' },
    },
    output: passthrough,
    execute: (args, exec) => callHarness(
      harness(),
      ['-m', MODULE, ...commonFlags(exec), 'assemble',
        '--goal', String(args.goal_id),
        ...(args.out ? ['--out', String(args.out)] : []),
        ...(args.show_text ? ['--show-text'] : [])],
      exec.signal, 'assembling the proof',
    ),
  }))
}
