/**
 * Read another evaluated candidate's source, from inside a candidate's own
 * runtime.
 *
 * EvoHarness offers a mutation an inventory of reference programs instead of
 * their source, on the promise that a tool can expand any of them on demand.
 * In-process that tool is `inspect_candidate` over the population store.
 * Substituting an external runtime clears every in-process tool, so under
 * this deployment the promise had nothing behind it: the candidate got a list
 * of program ids and no way to open one.
 *
 * This is that tool for this side. It does not read `run.db` itself. The
 * schema lives in Python and a second reader written here would drift from
 * it — silently, because both would keep answering. It shells out to the one
 * narrow view Python exposes for this audience, which is built from a fixed
 * set of keys rather than filtered down from the full evaluation report: a
 * task author's withheld metrics cannot reach a candidate by being forgotten.
 *
 * @module @deepseek-ai/dsh-evo-spike/peer
 */

import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { HARNESS_ENV, callHarness, requireEnv } from './cli.ts'

export const name = 'evo-peer'
export const inject = ['tools']

/** What this plugin needs on top of the shared interpreter variables. */
const REQUIRED_ENV: ReadonlyArray<readonly [string, string]> = [
  ['EVO_RUN_DIR', 'the run directory holding run.db'],
  ...HARNESS_ENV,
]

export function apply(ctx: Context) {
  ctx.tools.register(defineTool({
    name: 'evo_inspect_candidate',
    description:
      'Read the source of a previously evaluated candidate program. '
      + 'candidate_id must be copied verbatim from an "id=..." shown in the '
      + 'reference-program list of your prompt — it is an opaque identifier, '
      + 'not a command. Omit path to see that candidate\'s files, then pass a '
      + 'path to read one. If your prompt listed no reference programs, there '
      + 'is nothing to inspect.',
    parameters: {
      candidate_id: {
        type: 'string',
        description: 'Opaque id copied from your prompt\'s reference list.',
        required: true,
      },
      path: {
        type: 'string',
        description: 'A file inside that candidate; omit for its file list.',
      },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    execute(args, exec) {
      const env = requireEnv(REQUIRED_ENV)
      const path = args.path ?? ''
      return callHarness(
        { python: env['EVO_PYTHON'] as string, root: env['EVO_HARNESS_ROOT'] as string },
        [
          '-m', 'evoharness.readout', 'peer',
          '--run-dir', env['EVO_RUN_DIR'] as string,
          '--id', args.candidate_id,
          ...(path === '' ? [] : ['--path', path]),
        ],
        exec.signal,
        `could not read candidate ${args.candidate_id}`,
      )
    },
  }))
}
