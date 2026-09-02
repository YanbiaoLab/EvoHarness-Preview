/**
 * Start an evolution run from a session — with a person's approval, not a
 * model's decision.
 *
 * Deliberately its own plugin rather than another tool in `host.ts`. That
 * module is read-only and asserts its exact tool list, and a deployment that
 * wants people to watch runs without being able to launch them should be able
 * to mount reading alone. Everything that can act lives here.
 *
 * A run costs hours of compute and real money, and a tool call is issued by
 * the model. "Go ahead" typed in a chat is the model's reading of a sentence,
 * so this asks `ctx.approval.request` and starts only on `allowed-once`. Three
 * properties of that seam are what make it usable for this:
 *
 *   - a missing or failing answerer fails closed, so an unconfigured
 *     deployment cannot launch rather than launching unattended;
 *   - a grant applies only to the requested action, so approving one run is
 *     not approving the next;
 *   - the audit pair is written by the service and the model sees only the
 *     outcome, so an approval cannot be fabricated in the transcript.
 *
 * `approval` is injected, so this plugin will not mount at all without it.
 * The check belongs at mount, not at the call: a deployment discovering at
 * launch time that it had no answerer has already been asked to launch.
 *
 * @module @deepseek-ai/dsh-evo-spike/start
 */

import type { Context } from '@deepseek-ai/cordis'
// Imported for its Context augmentation: `ctx.approval` only exists on
// the type once this module is in the compilation.
import type {} from '@deepseek-ai/dsh-user-approval'
import { defineTool } from '@deepseek-ai/dsh-tools'
import {
  HARNESS_SETTINGS,
  type Setting,
  callHarness,
  modelCredential,
  resolveSettings,
} from './cli.ts'
import { resolveRun } from './host.ts'

export const name = 'evo-start'
export const inject = ['tools', 'approval']

/**
 * What starting a run needs. The deployment supplies every path; the model
 * supplies names. A cordis config chosen by the model would be the model
 * choosing the candidate's entire capability set, and a task directory chosen
 * by the model would be the model choosing the function that scores it.
 */
const REQUIRED_SETTINGS: readonly Setting[] = [
  {
    key: 'runsRoot',
    variable: 'EVO_RUNS_ROOT',
    purpose: 'where run directories are created',
  },
  {
    key: 'tasksRoot',
    variable: 'EVO_TASKS_ROOT',
    purpose: 'where authored task directories live',
  },
  {
    key: 'dshConfig',
    variable: 'EVO_DSH_CONFIG',
    purpose: "the candidate runtime's cordis config",
  },
  {
    key: 'dshRuntime',
    variable: 'EVO_DSH_RUNTIME',
    purpose: 'the dsh runtime entry the candidates run',
  },
  {
    key: 'dshProvider',
    variable: 'EVO_DSH_PROVIDER',
    purpose: 'the provider route that config declares',
  },
  // The same variable `candidate.cordis.yml` builds its model catalog from.
  // Read here rather than given a name of its own so the two cannot disagree:
  // when they did, EvoHarness asked for its own default and every request
  // 404'd against a catalog holding one other name — five generations, five
  // seconds, no offspring, and the run stopped as `proposer_dead`, which
  // sends someone to debug a proposer that was working.
  {
    key: 'model',
    variable: 'DSH_MODEL',
    purpose: 'the model name the candidate config offers',
  },
  ...HARNESS_SETTINGS,
]

/**
 * Where this deployment keeps EvoHarness, and what a candidate runs as.
 *
 * Every field is optional and every one has an environment variable behind it:
 * a runtime launched by an EvoHarness script has those exported, and an agent
 * preset — mounted into a dsh no script touched — declares these instead.
 *
 * None of them is a model-supplied value, and that is the point of them being
 * here: a cordis config named by the model would be the model choosing the
 * candidate's whole capability set.
 */
export interface Config {
  /** Interpreter EvoHarness is installed in. Env: `EVO_PYTHON`. */
  readonly python?: string
  /** Checkout `evoharness` imports from. Env: `EVO_HARNESS_ROOT`. */
  readonly harnessRoot?: string
  /** Where run directories are created. Env: `EVO_RUNS_ROOT`. */
  readonly runsRoot?: string
  /** Where authored task directories live. Env: `EVO_TASKS_ROOT`. */
  readonly tasksRoot?: string
  /** The candidate runtime's cordis config. Env: `EVO_DSH_CONFIG`. */
  readonly dshConfig?: string
  /** The dsh runtime entry candidates run. Env: `EVO_DSH_RUNTIME`. */
  readonly dshRuntime?: string
  /** The provider route that config declares. Env: `EVO_DSH_PROVIDER`. */
  readonly dshProvider?: string
  /** The model name the candidate config offers. Env: `DSH_MODEL`. */
  readonly model?: string
}

/**
 * Ceiling on generations a session may ask for. Not a safety boundary — the
 * approval is — but a bound on the size of the thing being approved, so a
 * person is never asked to approve "and then four hundred more" in the same
 * breath as "start this".
 */
const MAX_GENERATIONS = 50

/**
 * Register the one tool that can act.
 * @param ctx - the mounting context; under a preset, one agent's scope.
 * @param config - this deployment's binding, empty when it uses the variables.
 */
export function apply(ctx: Context, config: Config = {}) {
  ctx.tools.register(defineTool({
    name: 'evo_start',
    description:
      'Start an evolution run over an authored task. Requires a person to '
      + 'approve it: this spends hours of compute and real money, so calling '
      + 'this tool asks, it does not launch. Returns once the run is confirmed '
      + 'alive — poll evo_status afterwards, the run outlives this session. '
      + 'Check the task with evo_task_check first; a task that fails to load '
      + 'wastes the approval.',
    parameters: {
      task: {
        type: 'string',
        description:
          'Name of an authored task directory. A name, not a path: the '
          + 'deployment decides where tasks live.',
        required: true,
      },
      run: {
        type: 'string',
        description:
          'Name for this run\'s directory. Must not already hold a live run.',
        required: true,
      },
      generations: {
        type: 'integer',
        description: `How many generations to plan. At most ${MAX_GENERATIONS}.`,
        required: true,
      },
      purpose: {
        type: 'string',
        description:
          'One sentence on what this run is meant to find out. Shown to the '
          + 'person being asked to approve it, and recorded in the audit.',
        required: true,
      },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args, exec) {
      const resolved = resolveSettings(config, REQUIRED_SETTINGS)
      const agent = exec.agent
      if (agent === undefined) {
        // Approval routes by agent and writes the audit onto its session. With
        // no agent there is nobody to ask and nowhere to record the answer, so
        // starting here would be starting unapproved.
        throw new Error('evo_start needs a live agent session to ask through')
      }
      if (!Number.isInteger(args.generations)
        || args.generations < 1
        || args.generations > MAX_GENERATIONS) {
        throw new Error(
          `generations must be a whole number from 1 to ${MAX_GENERATIONS}`,
        )
      }

      const runsRoot = resolved.runsRoot as string
      const runDir = resolveRun(runsRoot, args.run)
      const taskDir = resolveRun(resolved.tasksRoot as string, args.task)

      const outcome = await ctx.approval.request({
        agent,
        toolName: 'evo_start',
        ...exec.callId === undefined ? {} : { callId: exec.callId },
        // What the person is actually deciding. The cost is the point of the
        // question, so it is in the question rather than in a log they would
        // have to go and read.
        reason:
          `Start an evolution run: task "${args.task}", `
          + `${args.generations} generations, run directory "${args.run}". `
          + `Purpose: ${args.purpose}. This spends model calls and evaluation `
          + 'time for as long as it runs, and it keeps running after this '
          + 'session ends.',
        ...exec.signal === undefined ? {} : { signal: exec.signal },
      })

      if (outcome !== 'allowed-once') {
        // Every other outcome is reported as itself. Collapsing them into one
        // refusal would tell a model that a missing answerer and a person
        // saying no are the same thing, and it would retry the wrong one.
        throw new Error(
          `not started: approval ${outcome}. `
          + (outcome === 'unavailable'
            ? 'This deployment has no approval answerer, so runs cannot be '
              + 'started from a session at all — use the command line.'
            : 'Do not ask again unless the person asks you to.'),
        )
      }

      return callHarness(
        {
          python: resolved.python as string,
          root: resolved.harnessRoot as string,
        },
        [
          '-m', 'evoharness.launch.start',
          '--recipe', 'e0',
          '--task-dir', taskDir,
          '--run-dir', runDir,
          '--live',
          '--dsh-config', resolved.dshConfig as string,
          '--dsh-runtime', resolved.dshRuntime as string,
          '--dsh-provider', resolved.dshProvider as string,
          '--model', resolved.model as string,
          '--set', 'proposal.mode=agentic',
          `search.num_generations=${args.generations}`,
        ],
        exec.signal,
        `approved, but the run did not start (${args.run})`,
        {
          env: {
            ...await modelCredential(ctx),
            // The run is `--live` and outlives this call, so what it
            // inherits here is what it spends for hours. dsh's managed
            // credential document is never materialized into the process
            // environment, so a stored key reaches the run only by being
            // handed over.
            //
            // `DSH_MODEL` travels with it because the candidate runtime's
            // cordis file builds its model catalog from that variable while
            // the run asks for `--model`. Both have to name the same model:
            // a catalog holding one name answers 404 to every request for
            // another, which surfaces as a dead proposer rather than as a
            // configuration mismatch.
            DSH_MODEL: resolved.model as string,
          },
        },
      )
    },
  }))
}
