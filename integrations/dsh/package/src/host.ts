/**
 * Read EvoHarness runs from the session a person is sitting in.
 *
 * The counterpart to `peer.ts`: that one runs inside a candidate's own
 * runtime and answers about one program, this one runs in the host session
 * and answers about runs. Both shell out to the same Python for the same
 * reason — the run directory's format lives there.
 *
 * Read-only on purpose. Starting a run is not here and should not be added
 * without deciding one thing first: a tool call is issued by the model, so an
 * `evo_start` tool means the model decides when to spend hours of compute.
 * "Go ahead" typed in a chat is the model's reading of a sentence, not a
 * person's authorisation. Reading a run carries none of that, which is why
 * these two go first.
 *
 * Mount this in the HOST's cordis configuration. It must never appear in a
 * candidate's: a candidate that can read the trajectory sees every rival's
 * score and bypasses the inspiration selection the search is tuning.
 *
 * @module @deepseek-ai/dsh-evo-spike/host
 */

import { isAbsolute, join, relative } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { HARNESS_SETTINGS, type Setting, callHarness, resolveSettings } from './cli.ts'

export const name = 'evo-host'
export const inject = ['tools']

/** What these tools need beyond the shared interpreter settings. */
const RUNS_SETTINGS: readonly Setting[] = [
  {
    key: 'runsRoot',
    variable: 'EVO_RUNS_ROOT',
    purpose: 'the directory run directories live under',
  },
  ...HARNESS_SETTINGS,
]

/** What the governance tools need. Separate: a deployment may want runs
 *  without governance, and demanding both would make the common case fail. */
const GOVERNANCE_SETTINGS: readonly Setting[] = [
  {
    key: 'researchRoot',
    variable: 'EVO_RESEARCH_ROOT',
    purpose: 'the research store holding the decision ledger',
  },
  ...HARNESS_SETTINGS,
]

/** What checking a task needs. Shared with `start.ts`, which resolves task
 *  names against the same root — a task the session can start is one it
 *  could have checked first. */
const TASKS_SETTINGS: readonly Setting[] = [
  {
    key: 'tasksRoot',
    variable: 'EVO_TASKS_ROOT',
    purpose: 'where authored task directories live',
  },
  ...HARNESS_SETTINGS,
]

/**
 * Where this deployment keeps EvoHarness and its directories.
 *
 * Every field is optional and every one has an environment variable behind it:
 * a runtime launched by an EvoHarness script has those exported, and an agent
 * preset — mounted into a dsh no script touched — declares these instead.
 */
export interface Config {
  /** Interpreter EvoHarness is installed in. Env: `EVO_PYTHON`. */
  readonly python?: string
  /** Checkout `evoharness` imports from. Env: `EVO_HARNESS_ROOT`. */
  readonly harnessRoot?: string
  /** Where run directories live. Env: `EVO_RUNS_ROOT`. */
  readonly runsRoot?: string
  /** Where authored task directories live. Env: `EVO_TASKS_ROOT`. */
  readonly tasksRoot?: string
  /** The research store holding the decision ledger. Env: `EVO_RESEARCH_ROOT`. */
  readonly researchRoot?: string
}

/**
 * Resolve a run name against the configured root, refusing anything that is
 * not a plain name.
 *
 * The tool takes a NAME and never a path. A model-supplied path is not a
 * parameter but a capability — it chooses which directory on this machine
 * gets read. Confining resolution to one configured root keeps the choice
 * where the deployment made it.
 *
 * @param root - EVO_RUNS_ROOT.
 * @param runName - model-supplied name.
 * @returns the absolute run directory.
 */
export function resolveRun(root: string, runName: string): string {
  if (runName === '' || runName === '.' || runName === '..') {
    throw new Error('run must be the name of a run directory')
  }
  if (isAbsolute(runName) || /[/\\]/.test(runName)) {
    throw new Error(
      `run must be a plain name, not a path: ${runName}. Runs are resolved `
      + 'under the directory this deployment configured.',
    )
  }
  const resolved = join(root, runName)
  // Belt and braces. The checks above already exclude every separator, so
  // this cannot currently fire — which is the point of keeping it: the day
  // the rules above are relaxed, escaping the root stays refused.
  const inside = relative(root, resolved)
  if (inside.startsWith('..') || isAbsolute(inside)) {
    throw new Error(`run resolves outside the configured root: ${runName}`)
  }
  return resolved
}

/**
 * Register the read-only tools.
 * @param ctx - the mounting context; under a preset, one agent's scope.
 * @param config - this deployment's binding, empty when it uses the variables.
 */
export function apply(ctx: Context, config: Config = {}) {
  const environment = () => {
    const resolved = resolveSettings(config, RUNS_SETTINGS)
    return {
      root: resolved.runsRoot as string,
      harness: {
        python: resolved.python as string,
        root: resolved.harnessRoot as string,
      },
    }
  }

  const governance = () => {
    const resolved = resolveSettings(config, GOVERNANCE_SETTINGS)
    return {
      research: resolved.researchRoot as string,
      harness: {
        python: resolved.python as string,
        root: resolved.harnessRoot as string,
      },
    }
  }

  ctx.tools.register(defineTool({
    name: 'evo_status',
    description:
      'How an evolution run is doing: which generation it reached, its best '
      + 'fitness so far, whether it is still moving, and why it stopped if it '
      + 'did. Omit run to list every run instead. Read-only.',
    parameters: {
      run: {
        type: 'string',
        description: 'Name of one run directory; omit to list all of them.',
      },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    execute(args, exec) {
      const { root, harness } = environment()
      const runName = args.run ?? ''
      const argv = runName === ''
        ? ['-m', 'evoharness.readout', 'list', '--root', root]
        : [
          '-m', 'evoharness.readout', 'status',
          '--run-dir', resolveRun(root, runName),
        ]
      return callHarness(
        harness,
        argv,
        exec.signal,
        runName === ''
          ? `could not list runs under ${root}`
          : `could not read run ${runName}`,
      )
    },
  }))

  ctx.tools.register(defineTool({
    name: 'evo_trajectory',
    description:
      'What each generation of a run tried and whether anything improved: '
      + 'per generation, how many candidates, how many passed, the best '
      + 'fitness, the best so far, and a count of each failure kind. Pass '
      + 'detail="full" to list every candidate instead of counting them. '
      + 'Read-only.',
    parameters: {
      run: {
        type: 'string',
        description: 'Name of the run directory.',
        required: true,
      },
      detail: {
        type: 'string',
        enum: ['summary', 'full'],
        description:
          'summary counts each generation; full lists every candidate and is '
          + 'much larger.',
      },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    execute(args, exec) {
      const { root, harness } = environment()
      return callHarness(
        harness,
        [
          '-m', 'evoharness.readout', 'trajectory',
          '--run-dir', resolveRun(root, args.run),
          '--detail', args.detail ?? 'summary',
        ],
        exec.signal,
        `could not read the trajectory of ${args.run}`,
      )
    },
  }))

  ctx.tools.register(defineTool({
    name: 'evo_decisions',
    description:
      'Decision requests waiting for a person: what is being asked, what '
      + 'happens if nobody acts, and which experiment it belongs to. Omit '
      + 'request_id for the queue, pass one to read that card in full with '
      + 'its evidence references. Read-only — you can show a card and draft a '
      + 'reply for the person to consider, but you cannot answer one, and '
      + 'saying you have is false.',
    parameters: {
      request_id: {
        type: 'string',
        description: 'One card to read in full; omit for the whole queue.',
      },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    execute(args, exec) {
      const { research, harness } = governance()
      const id = args.request_id ?? ''
      return callHarness(
        harness,
        id === ''
          ? ['-m', 'evoharness.readout', 'cards', '--research-root', research]
          : [
            '-m', 'evoharness.readout', 'card',
            '--research-root', research, '--id', id,
          ],
        exec.signal,
        id === ''
          ? 'could not read the decision queue'
          : `could not read decision request ${id}`,
      )
    },
  }))

  ctx.tools.register(defineTool({
    name: 'evo_decided',
    description:
      'Cards that have already been answered, newest first: the action, who '
      + 'signed it and why. This is the audit record, not a queue. Read-only.',
    parameters: {
      limit: {
        type: 'integer',
        description: 'How many to return; defaults to 20.',
      },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    execute(args, exec) {
      const { research, harness } = governance()
      return callHarness(
        harness,
        [
          '-m', 'evoharness.readout', 'decided',
          '--research-root', research,
          '--limit', String(args.limit ?? 20),
        ],
        exec.signal,
        'could not read the decision record',
      )
    },
  }))

  ctx.tools.register(defineTool({
    name: 'evo_task_check',
    description:
      'Load an authored task directory the way a run would, and report what '
      + 'it found: its identity hashes, how many knowledge chunks and '
      + 'preflight checks it has, and which seed files. Use it after writing '
      + 'or editing a task. Loading it is the only way to know the grade '
      + 'function imports and the declaration parses — otherwise the first '
      + 'thing that says so is a run that already spent twenty minutes. '
      + 'Read-only: it never starts anything.',
    parameters: {
      task: {
        type: 'string',
        description:
          'Name of the task directory. A name, not a path: the deployment '
          + 'decides where tasks live.',
        required: true,
      },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    execute(args, exec) {
      const resolved = resolveSettings(config, TASKS_SETTINGS)
      return callHarness(
        {
          python: resolved.python as string,
          root: resolved.harnessRoot as string,
        },
        [
          '-m', 'evoharness.authoring', 'check',
          '--task-dir', resolveRun(resolved.tasksRoot as string, args.task),
        ],
        exec.signal,
        `could not load task ${args.task}`,
      )
    },
  }))
}
