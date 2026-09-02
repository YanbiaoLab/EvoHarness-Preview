/**
 * Calling the EvoHarness command line from a dsh tool.
 *
 * Two plugins here shell out to the same interpreter for the same reasons:
 * the run directory's format lives in Python, and a second reader written in
 * TypeScript would drift from it silently, since both would keep answering.
 * This is the part they share — locate the interpreter, run it without a
 * shell, and turn its exit code back into something a model can act on.
 *
 * @module @deepseek-ai/dsh-evo-spike/cli
 */

import { execFile } from 'node:child_process'
import type { Context } from '@deepseek-ai/cordis'
import { credentialRef } from '@deepseek-ai/dsh-credentials'

/**
 * The two variables EvoHarness reads to reach a model — in
 * `evoharness.proof.cli`'s `_transport` and in `evoharness.launch.build`,
 * which is where the authority for these names lives. Restated here because
 * the pair crosses a language boundary and no one declaration spans it.
 */
export const MODEL_ENV = ['EVOHARNESS_API_BASE', 'EVOHARNESS_API_KEY'] as const

/**
 * Exit code the EvoHarness CLIs use for "your request was wrong" as opposed
 * to "something broke". Both print JSON on stdout; only this one carries a
 * message worth handing to a model.
 */
export const EXIT_REFUSED = 2

/**
 * Default ceiling on one call, sized for a query: reading a run, a board or a
 * candidate returns nothing worth minutes, and a hang there burns the turn
 * budget for nothing.
 *
 * A caller whose work is not a query passes its own — compiling and solving
 * both take minutes by nature rather than as a symptom.
 */
export const TIMEOUT_MS = 30_000

/** Where the interpreter is and where it has to start. */
export interface Harness {
  readonly python: string
  readonly root: string
}

/** What one call may vary, beyond the interpreter and the argv. */
export interface CallOptions {
  /** This call's ceiling, defaulting to {@link TIMEOUT_MS}. */
  readonly timeoutMs?: number
  /**
   * Variables for the child, layered over the inherited environment.
   *
   * For the one thing an inherited environment cannot carry: a credential
   * dsh resolved from its own store. The managed document is never
   * materialized into the process environment, so a key configured there
   * reaches a subprocess only by being handed to it.
   */
  readonly env?: Readonly<Record<string, string>>
}

/** What a finished child left behind, whichever way it exited. */
interface CommandResult {
  readonly code: number
  readonly stdout: string
  readonly stderr: string
}

/**
 * Run a command and report how it exited rather than throwing on non-zero.
 *
 * `@deepseek-ai/dsh-native-command` is the shared runner for host-native OS
 * integrations, and it fits everything here but two things: it takes no
 * working directory, and a non-zero exit reaches the caller as a rejection.
 * Both matter — the interpreter has to start where `evoharness` imports, and
 * exit code 2 is a normal answer here, not a failure.
 */
const run = (
  command: string,
  args: readonly string[],
  options: {
    readonly cwd: string
    readonly signal: AbortSignal
    readonly env?: NodeJS.ProcessEnv
  },
): Promise<CommandResult> =>
  new Promise((resolve, reject) => {
    execFile(
      command,
      [...args],
      { encoding: 'utf8', windowsHide: true, ...options },
      (error, stdout, stderr) => {
        // An aborted or unspawnable child has no exit code to report; only
        // those are failures of the call itself.
        const code = (error as { code?: unknown } | null)?.code
        if (error !== null && typeof code !== 'number') {
          reject(new Error(error.message, { cause: error }))
          return
        }
        resolve({ code: typeof code === 'number' ? code : 0, stdout, stderr })
      },
    )
  })

/** A refusal the Python side produced, or null when the output was not one. */
const refusalIn = (stdout: string): string | null => {
  try {
    const parsed: unknown = JSON.parse(stdout)
    if (typeof parsed === 'object' && parsed !== null && 'error' in parsed) {
      const { error } = parsed as { error: unknown }
      return typeof error === 'string' ? error : null
    }
  } catch {
    // Not JSON at all — an interpreter that failed before reaching our code.
    // Reported below as an infrastructure fault, not as a refusal.
  }
  return null
}

/**
 * Read the variables a plugin needs, or throw naming every one that is missing.
 *
 * Read per call rather than at mount. A runtime launched by EvoHarness has
 * these set by its launcher; a deployment mounted by hand has none of them,
 * and failing at the call is what puts the reason in front of the model
 * instead of in a boot log nobody reads.
 *
 * @param required - variable name paired with what it is for.
 * @returns the resolved values, keyed by variable name.
 */
export function requireEnv(
  required: ReadonlyArray<readonly [string, string]>,
): Record<string, string> {
  const missing = required.filter(([key]) => {
    const value = process.env[key]
    return value === undefined || value === ''
  })
  if (missing.length > 0) {
    throw new Error(
      `this runtime is not configured for EvoHarness: ${missing
        .map(([key, purpose]) => `${key} (${purpose})`)
        .join(', ')} not set`,
    )
  }
  return Object.fromEntries(
    required.map(([key]) => [key, process.env[key] as string]),
  )
}

/** The two variables every EvoHarness-calling tool needs. */
export const HARNESS_ENV: ReadonlyArray<readonly [string, string]> = [
  ['EVO_PYTHON', 'the interpreter EvoHarness is installed in'],
  ['EVO_HARNESS_ROOT', 'the checkout `evoharness` imports from'],
]

/** One deployment setting: what a row calls it, and the variable behind it. */
export interface Setting {
  /** The key on the plugin row's `config`. */
  readonly key: string
  /** The environment variable a launcher exports instead. */
  readonly variable: string
  /** What it is for, written once, for the message a caller has to act on. */
  readonly purpose: string
}

/** The interpreter and checkout every EvoHarness-calling plugin needs. */
export const HARNESS_SETTINGS: readonly Setting[] = [
  {
    key: 'python',
    variable: 'EVO_PYTHON',
    purpose: 'the interpreter EvoHarness is installed in',
  },
  {
    key: 'harnessRoot',
    variable: 'EVO_HARNESS_ROOT',
    purpose: 'the checkout `evoharness` imports from',
  },
]

/** A configured value, else the variable's, else undefined. Empty is unset. */
export function optionalSetting(
  configured: unknown, variable: string,
): string | undefined {
  const value = typeof configured === 'string' && configured !== ''
    ? configured
    : process.env[variable]
  return value === undefined || value === '' ? undefined : value
}

/**
 * Resolve settings from a plugin row, falling back to the environment.
 *
 * Two sources because there are two ways these plugins are deployed and both
 * are legitimate: a runtime launched by an EvoHarness script has the variables
 * exported, and an agent preset mounted into a dsh that no script touched has
 * only what its row declares. Config wins where both speak.
 *
 * Called per tool call rather than at mount, as `peer.ts` does: failing at
 * mount takes the whole runtime down, and under a preset it would take down
 * every session that named the preset rather than the one call that needed
 * the setting.
 *
 * @param config - the plugin row's config object.
 * @param settings - what to resolve, keyed as the caller wants them back.
 * @returns the resolved values, keyed by `Setting.key`.
 * @throws naming every unresolved setting and BOTH ways to supply it — a
 * reader told only the variable name goes looking for a launcher that, under
 * a preset, is not in the picture at all.
 */
export function resolveSettings(
  config: object,
  settings: readonly Setting[],
): Record<string, string> {
  // One cast here rather than an index signature on each plugin's `Config`:
  // those interfaces are what a reader consults for what a row may declare,
  // and an index signature would let any key onto them silently.
  const row = config as Record<string, unknown>
  const resolved: Record<string, string> = {}
  const missing: Setting[] = []
  for (const setting of settings) {
    const value = optionalSetting(row[setting.key], setting.variable)
    if (value === undefined) missing.push(setting)
    else resolved[setting.key] = value
  }
  if (missing.length > 0) {
    throw new Error(
      'this runtime is not configured for EvoHarness: '
      + missing
        .map(({ key, variable, purpose }) =>
          `${variable} (${purpose}) — set it as \`${key}\` on this plugin `
          + 'row, or export the variable')
        .join('; '),
    )
  }
  return resolved
}

/**
 * The credential a launched run or an attack spends, from dsh's own store.
 *
 * A subprocess inherits the process environment and nothing else, while dsh's
 * managed credential document is deliberately never materialized there. So a
 * key configured the way dsh configures keys — the Models page, or
 * `$DSH_HOME/.credentials.yaml` — is invisible to the interpreter unless it is
 * resolved and handed over. Under a preset that is the only way it arrives:
 * no launcher exported anything.
 *
 * Resolved per call, which is what the credential service is for: a rotated
 * key reaches the next run without restarting anything.
 *
 * `ctx.get` rather than `inject`, for the same reason resolution is late: a
 * runtime with no credential store still mounts, and falls back to the
 * inherited environment, which is exactly a launcher's arrangement.
 *
 * @param ctx - the mounting context.
 * @returns the resolved variables, empty when this runtime stores none.
 */
export async function modelCredential(ctx: Context): Promise<Record<string, string>> {
  const credentials = ctx.get('credentials')
  if (credentials === undefined) return {}
  const resolved: Record<string, string> = {}
  for (const variable of MODEL_ENV) {
    const hit = await credentials.resolve(credentialRef(variable))
    if (hit !== undefined) resolved[variable] = hit.value
  }
  return resolved
}

/**
 * Run one EvoHarness module and return its stdout.
 *
 * @param harness - interpreter and working directory.
 * @param argv - module and arguments, as argv. Never a shell string: these
 *   carry model-authored values, and a shell would make them executable
 *   rather than merely wrong.
 * @param signal - the caller's lifetime; a timeout is applied on top.
 * @param context - what the caller was trying to do, for the failure message.
 * @param options - this call's ceiling and any variables it adds for the child.
 * @returns trimmed stdout on success.
 */
export async function callHarness(
  harness: Harness,
  argv: readonly string[],
  signal: AbortSignal | undefined,
  context: string,
  options: CallOptions = {},
): Promise<string> {
  const timeoutMs = options.timeoutMs ?? TIMEOUT_MS
  const timeout = AbortSignal.timeout(timeoutMs)
  const combined = signal === undefined
    ? timeout
    : AbortSignal.any([signal, timeout])

  let finished: CommandResult
  try {
    finished = await run(harness.python, argv, {
      cwd: harness.root,
      signal: combined,
      // Layered over the inherited environment rather than replacing it: the
      // child needs PATH, HOME and whatever else the interpreter reads, and
      // an `env` handed to execFile is the WHOLE environment.
      ...options.env === undefined ? {} : { env: { ...process.env, ...options.env } },
    })
  } catch (error) {
    // The ceiling and the caller's own cancellation both arrive here as an
    // abort with no exit code, and only the ceiling is worth a sentence: a
    // caller told nothing but "aborted" cannot tell a limit it could raise
    // from a child that died on its own.
    if (!timeout.aborted) throw error
    // Seconds for the ceilings a person sets, milliseconds below that. A
    // ceiling reported as `0s` reads as a reporting bug rather than as the
    // number the deployment chose.
    const elapsed = timeoutMs >= 1000
      ? `${String(Math.round(timeoutMs / 1000))}s`
      : `${String(timeoutMs)}ms`
    throw new Error(
      `${context}: nothing came back within ${elapsed}, so the call was cut off. `
      + 'Work it started elsewhere may still be running.',
      { cause: error },
    )
  }
  const { code, stdout, stderr } = finished
  if (code === 0) return stdout.trim()

  if (code === EXIT_REFUSED) {
    const refusal = refusalIn(stdout)
    // The model can act on this one: a name that does not exist, a file the
    // run does not have. Surfaced as the tool error so it reaches the turn.
    if (refusal !== null) throw new Error(refusal)
  }
  // Anything else is the deployment being wrong, not the model. Say so
  // plainly — a caller told "no such run" for a broken interpreter spends its
  // remaining turns guessing names.
  throw new Error(
    `${context} (exit ${code}): `
    + (stderr.trim() || stdout.trim() || 'no output'),
  )
}
