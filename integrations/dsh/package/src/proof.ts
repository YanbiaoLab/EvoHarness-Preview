/**
 * Five tools that let a dsh session drive EvoHarness's proof graph.
 *
 * Copy or symlink this next to the other evo-harness example plugins, then
 * either add a row for it to the host cordis config (`host.proof.cordis.yml`
 * beside this file) or install the agent preset that mounts it
 * (`presets/proof/`, installed by `scripts/install_dsh_presets.sh`).
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

import { isAbsolute, join, relative } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import type { ToolRunContext } from '@deepseek-ai/dsh-tools'
import { defineTool } from '@deepseek-ai/dsh-tools'
import {
  HARNESS_SETTINGS,
  type Harness,
  callHarness,
  modelCredential,
  optionalSetting,
  resolveSettings,
} from './cli.ts'

export const name = 'evo-proof'
export const inject = ['tools']

/** Set only when goals import Mathlib; absent means a bare `lean`. */
const LEAN_PROJECT = 'EVO_LEAN_PROJECT'

/** The session workspace subdirectory holding `graph.db` and every run. */
const WORK_DIR = '.evo'

const MODULE = 'evoharness.proof.cli'

/**
 * Ceiling on one `proof_attack`, which runs a solver rather than a query.
 *
 * Bounded from both sides. Below it, the work itself: a solver edits Lean and
 * compiles it repeatedly, so minutes are its normal shape. Above it, the
 * session's own request timeout — a tool call that outlives the turn takes the
 * conversation with it — and that number belongs to the deployment, which is
 * what `attackTimeoutMs` is for.
 */
const ATTACK_TIMEOUT_MS = 900_000

/**
 * Ceiling on a call that compiles: `proof_sketch` and `proof_assemble`.
 *
 * Must stay above the Python side's own `--lean-timeout`. Whichever ceiling
 * fires first decides what the caller is told, and only Lean's timeout
 * produces a verdict about the proof; this one firing first severs a call that
 * was about to answer and leaves an unjudged decomposition on the board.
 *
 * Sized for a Mathlib import, which alone costs tens of seconds before the
 * goal is even read. What a call costs in money says nothing about what it
 * costs in time: these two spend no model budget and are the slowest in the
 * set after an attack.
 */
export const LEAN_TIMEOUT_MS = 360_000

/**
 * Where this deployment keeps EvoHarness, and how long it may spend.
 *
 * Every field is optional and every one has an environment variable behind
 * it: a runtime launched by `scripts/dsh_proof.sh` sets those variables, and
 * an agent preset — mounted by a dsh the launcher never touched — sets these
 * instead. Config wins where both are present.
 *
 * No runtime schema, unlike a first-party plugin row: this package carries no
 * schema dependency, and a wrong path fails the first call with the path in
 * the message rather than silently doing something else.
 */
export interface Config {
  /** Interpreter EvoHarness is installed in. Env: `EVO_PYTHON`. */
  readonly python?: string
  /** Checkout `evoharness` imports from. Env: `EVO_HARNESS_ROOT`. */
  readonly harnessRoot?: string
  /** Lake project providing Mathlib; omit for core-Lean goals. Env: `EVO_LEAN_PROJECT`. */
  readonly leanProject?: string
  /** Ceiling on one `proof_attack`, in milliseconds. */
  readonly attackTimeoutMs?: number
  /**
   * Ceiling on a call that compiles (`proof_sketch`, `proof_assemble`), in
   * milliseconds. Keep it above the Python side's `--lean-timeout` so a slow
   * compile comes back as Lean's verdict rather than as a severed call.
   */
  readonly leanTimeoutMs?: number
}

/** Render whatever the CLI printed, unchanged. It is already one json object. */
const passthrough = {
  schema: { type: 'string' as const },
  render: (_args: unknown, value: string) => [{ type: 'text' as const, text: value }],
}

/** The interpreter and checkout, from the row or the environment. */
function harness(config: Config): Harness {
  const resolved = resolveSettings(config, HARNESS_SETTINGS)
  return { python: resolved.python as string, root: resolved.harnessRoot as string }
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
function sessionDir(exec: ToolRunContext): string {
  const cwd = exec.agent?.session.header.cwd
  if (cwd === undefined || cwd === '') {
    throw new Error(
      'this session has no workspace directory, so there is nowhere to keep '
      + `the proof graph (it lives in <workspace>/${WORK_DIR}). Start the `
      + 'session in the directory the proof belongs to and call again.',
    )
  }
  return cwd
}

/** Where THIS session's graph lives: `<session workspace>/.evo`. */
function workspace(exec: ToolRunContext): string {
  return join(sessionDir(exec), WORK_DIR)
}


function resolveOut(exec: ToolRunContext, name: string): string {
  if (name === '' || name === '.' || name === '..') {
    throw new Error('out must be the name of a file')

  }
  if (isAbsolute(name) || /[/\\]/.test(name)) {
    throw new Error(
      `out must be a plain file name, not a path: ${name}. The assembled `
      + "proof is written into this session's own directory.",
    )
  }
  const dir = sessionDir(exec)
  const resolved = join(dir, name)
  const inside = relative(dir, resolved)

  if (inside.startsWith('..') || isAbsolute(inside)) {
    throw new Error(`out resolves outside the session workspace: ${name}`)
  }
  return resolved

}


/** Flags every subcommand accepts, from the session and the deployment. */
function commonFlags(exec: ToolRunContext, config: Config): string[] {
  const project = optionalSetting(config.leanProject, LEAN_PROJECT)
  return [
    '--work', workspace(exec),
    ...project === undefined ? [] : ['--lean-project', project],
  ]
}

/**
 * Register the five tools.
 * @param ctx - the mounting context; under a preset, one agent's scope.
 * @param config - this deployment's binding, empty when it uses the variables.
 */
export function apply(ctx: Context, config: Config = {}) {

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
      harness(config),
      ['-m', MODULE, ...commonFlags(exec, config), 'open',
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
      + 'been attempted on each, and what has been spent. `proved` means the '
      + 'route closed; `certified` says whether the assembled proof compiled '
      + 'as one file, and is null until proof_assemble has run. Call it '
      + 'before deciding what to do next; it is cheap and it is the only '
      + 'place the real state lives.',
    parameters: {
      goal_id: { type: 'string', description: 'One goal; omit for everything.' },
    },
    output: passthrough,
    execute: (args, exec) => callHarness(
      harness(config),
      ['-m', MODULE, ...commonFlags(exec, config), 'status',
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
      harness(config),
      ['-m', MODULE, ...commonFlags(exec, config), 'sketch',
        '--goal', String(args.goal_id),
        '--proposal', String(args.proposal)],
      exec.signal, 'validating a decomposition',
      { timeoutMs: config.leanTimeoutMs ?? LEAN_TIMEOUT_MS },
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
    execute: async (args, exec) => callHarness(
      harness(config),
      ['-m', MODULE, ...commonFlags(exec, config), 'attack',
        '--goal', String(args.goal_id),
        ...(args.budget === undefined ? [] : ['--budget', String(args.budget)]),
        ...(args.level ? ['--level', String(args.level)] : []),
        ...(args.allow_unaccepted_route ? ['--allow-unaccepted-route'] : [])],
      exec.signal, 'attacking a goal',
      {
        timeoutMs: config.attackTimeoutMs ?? ATTACK_TIMEOUT_MS,
        env: await modelCredential(ctx),
      },
    ),
  }))

  ctx.tools.register(defineTool({
    name: 'proof_assemble',
    description:
      'Put the proved lemmas back together and compile the whole thing. This '
      + 'is the ONLY thing that certifies a root goal: per-node greens were '
      + 'earned minutes apart and only compiling the finished article catches '
      + 'a lemma that typechecks alone and not in company. Returns whether it '
      + 'compiled and which axioms Lean says it depends on, and records that '
      + 'verdict on the goal, passed or failed.',
    parameters: {
      goal_id: { type: 'string', description: 'The root goal.', required: true },
      out: {
        type: 'string',
        description:
          "File NAME to write the assembled proof to, in this session's own "
          + 'directory. A name, not a path.',
      },
      show_text: { type: 'boolean', description: 'Include the proof in the reply.' },
    },
    output: passthrough,
    execute: (args, exec) => callHarness(
      harness(config),
      ['-m', MODULE, ...commonFlags(exec, config), 'assemble',
        '--goal', String(args.goal_id),
        ...(args.out ? ['--out', resolveOut(exec, String(args.out))] : []),
        ...(args.show_text ? ['--show-text'] : [])],
      exec.signal, 'assembling the proof',
      { timeoutMs: config.leanTimeoutMs ?? LEAN_TIMEOUT_MS },
    ),
  }))
}
