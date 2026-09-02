/**
 * The proof tools' deployment binding.
 *
 * `src/proof.ts` is a copy; `tests/mirror.spec.ts` says where the original is
 * and why the copy has to exist.
 *
 * What is under test is the deployment seam: the paths reach the interpreter
 * from the plugin row as readily as from the environment, and each call's
 * ceiling is sized for the work it does.
 */

import { chmodSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import { CallId } from '@deepseek-ai/dsh-llm'

import * as EvoProof from '../src/proof.ts'
import type { Config } from '../src/proof.ts'
import { LEAN_TIMEOUT_MS } from '../src/proof.ts'
import { TIMEOUT_MS } from '../src/cli.ts'

/** `evoharness.proof.cli`'s own `--lean-timeout` default, in milliseconds. */
const PYTHON_LEAN_TIMEOUT_MS = 300_000

const testToolSignal = new AbortController().signal

/** Echo argv back the way the proof CLI prints JSON, and exit 0. */
const ECHO_ARGV = `
process.stdout.write(JSON.stringify({ argv: process.argv.slice(2) }))
`

/**
 * Report what the child was actually given to reach a model with.
 *
 * Empty reads as absent, which is what `_transport` in the Python CLI decides
 * on (`if not base or not key`); a test that told them apart would be pinning
 * a distinction nothing downstream makes.
 */
const ECHO_MODEL_ENV = `
process.stdout.write(JSON.stringify({
  base: process.env.EVOHARNESS_API_BASE || null,
  key: process.env.EVOHARNESS_API_KEY || null,
  inherited: process.env.PATH !== undefined,
}))
`

/** Answer, but not soon. Long enough to outlast a ceiling set in the test. */
const SLOW = `
setTimeout(() => process.stdout.write('{"ok":true}'), 30_000)
`

/**
 * A stand-in for the interpreter. What these tools owe Python is argv shape
 * and exit-code handling, so a script echoing its argv tests both without an
 * interpreter, a graph, or Lean.
 */
function fakePython(body: string): string {
  const dir = mkdtempSync(join(tmpdir(), 'evo-proof-'))
  const path = join(dir, 'fake-python')
  writeFileSync(path, `#!/usr/bin/env node\n${body}\n`, 'utf8')
  chmodSync(path, 0o755)
  return path
}

/** A caller carrying just the one field the tools read off a session. */
function callerIn(cwd: string | undefined): Agent {
  return { session: { header: { cwd } } } as unknown as Agent
}

async function mount(config: Config = {}): Promise<Context> {
  const ctx = new Context()
  await ctx.plugin(SystemPrompt, {})
  await ctx.plugin(ToolRuntime)
  await ctx.plugin(EvoProof, config)
  return ctx
}

/** The workspace a session normally has, and where its graph would go. */
const WORKSPACE = '/work/session'

/**
 * Run one tool and return whatever it put in front of the model.
 *
 * A failing tool is not a rejection here: `tools.execute` renders the error
 * as the call's text content, because that is what the turn has to read.
 * Every assertion below is therefore on the returned string.
 */
async function callIn(
  ctx: Context,
  cwd: string | undefined,
  name: string,
  args: Record<string, unknown>,
): Promise<string> {
  const result = await ctx.tools.execute({
    signal: testToolSignal,
    callId: CallId('c1'),
    name,
    arguments: args,
    agent: callerIn(cwd),
  })
  const first = result.content[0]
  return first?.type === 'text' ? first.text : JSON.stringify(result.content)
}

const call = (
  ctx: Context,
  name: string,
  args: Record<string, unknown>,
): Promise<string> => callIn(ctx, WORKSPACE, name, args)

/** The argv the fake interpreter reported, for a call expected to succeed. */
async function argvOf(
  ctx: Context,
  name: string,
  args: Record<string, unknown>,
): Promise<string[]> {
  const printed = await call(ctx, name, args)
  const parsed = JSON.parse(printed) as { argv?: string[] }
  if (parsed.argv === undefined) throw new Error(`not an argv echo: ${printed}`)
  return parsed.argv
}

afterEach(() => {
  vi.unstubAllEnvs()
})

describe('where the interpreter comes from', () => {
  it('takes the plugin row over the environment, which is the preset case', async () => {
    // Nothing exported: an agent preset mounts into a dsh the EvoHarness
    // launcher never touched, so the variables are simply not there.
    vi.stubEnv('EVO_PYTHON', '')
    vi.stubEnv('EVO_HARNESS_ROOT', '')
    const ctx = await mount({ python: fakePython(ECHO_ARGV), harnessRoot: process.cwd() })

    expect(await argvOf(ctx, 'proof_status', {})).toContain('status')
  })

  it('still takes the environment when the row configures nothing', async () => {
    // The launcher path. It has to keep working: `dsh_proof.sh` passes a
    // patch file with no config on the row.
    vi.stubEnv('EVO_PYTHON', fakePython(ECHO_ARGV))
    vi.stubEnv('EVO_HARNESS_ROOT', process.cwd())
    const ctx = await mount()

    expect(await argvOf(ctx, 'proof_status', {})).toContain('status')
  })

  it('prefers the row when both say something', async () => {
    vi.stubEnv('EVO_PYTHON', fakePython('process.exit(3)'))
    vi.stubEnv('EVO_HARNESS_ROOT', process.cwd())
    const ctx = await mount({ python: fakePython(ECHO_ARGV), harnessRoot: process.cwd() })

    expect(await argvOf(ctx, 'proof_status', {})).toContain('status')
  })

  it('names both places an unset path can come from', async () => {
    vi.stubEnv('EVO_PYTHON', '')
    vi.stubEnv('EVO_HARNESS_ROOT', '')
    const ctx = await mount()

    // A reader told only "EVO_PYTHON not set" under a preset goes looking for
    // a launcher that is not in the picture.
    const reported = await call(ctx, 'proof_status', {})
    expect(reported).toMatch(/EVO_PYTHON[\s\S]*`python`/)
    expect(reported).toMatch(/EVO_HARNESS_ROOT[\s\S]*`harnessRoot`/)
  })
})

describe('the lake project', () => {
  beforeEach(() => {
    vi.stubEnv('EVO_PYTHON', fakePython(ECHO_ARGV))
    vi.stubEnv('EVO_HARNESS_ROOT', process.cwd())
  })

  it('reaches the CLI from the row', async () => {
    const ctx = await mount({ leanProject: '/lake/mathlib' })

    expect(await argvOf(ctx, 'proof_status', {})).toEqual(
      expect.arrayContaining(['--lean-project', '/lake/mathlib']),
    )
  })

  it('treats an empty row value as unset rather than as a path', async () => {
    // The installer always writes the key, because a template that has to
    // add or drop a whole line for one machine is a template that renders
    // wrong on some other one. Empty therefore has to mean "no Mathlib".
    vi.stubEnv('EVO_LEAN_PROJECT', '')
    const ctx = await mount({ leanProject: '' })

    expect(await argvOf(ctx, 'proof_status', {})).not.toContain('--lean-project')
  })

  it('falls back to the environment, which is what the launcher sets', async () => {
    vi.stubEnv('EVO_LEAN_PROJECT', '/env/mathlib')
    const ctx = await mount()

    expect(await argvOf(ctx, 'proof_status', {})).toEqual(
      expect.arrayContaining(['--lean-project', '/env/mathlib']),
    )
  })
})

describe('how long a call may take', () => {
  beforeEach(() => {
    vi.stubEnv('EVO_PYTHON', fakePython(SLOW))
    vi.stubEnv('EVO_HARNESS_ROOT', process.cwd())
  })

  it('gives attacking its own ceiling, apart from every other call', async () => {
    // `proof_attack` runs a solver, not a query, so it carries its own
    // ceiling. Both tools below face the same slow interpreter; only the one
    // carrying that ceiling is cut off.
    const ctx = await mount({ attackTimeoutMs: 50 })

    expect(await call(ctx, 'proof_attack', { goal_id: 'g1' }))
      .toMatch(/nothing came back within 50ms/)

    const pending = call(ctx, 'proof_status', {})
    const stillRunning = new Promise((resolve) => {
      setTimeout(() => { resolve('still running') }, 300)
    })
    await expect(Promise.race([pending.then(() => 'answered'), stillRunning]))
      .resolves.toBe('still running')
  })

  it('gives the compiling calls a ceiling above Lean\'s own', async () => {
    // Ordering is what is asserted, not a number. A compiling call must
    // outlast the Python side's `--lean-timeout` so that a slow compile comes
    // back as Lean's verdict rather than as a severed call, and it must
    // outlast the query ceiling because a Mathlib import alone costs tens of
    // seconds. Costing no model budget does not make a call fast.
    const ctx = await mount()
    void ctx

    expect(LEAN_TIMEOUT_MS).toBeGreaterThan(PYTHON_LEAN_TIMEOUT_MS)
    expect(LEAN_TIMEOUT_MS).toBeGreaterThan(TIMEOUT_MS)
  })

  it('cuts off a compile on its own ceiling, not the default one', async () => {
    const ctx = await mount({ leanTimeoutMs: 50 })

    expect(await call(ctx, 'proof_sketch', { goal_id: 'g1', proposal: '{}' }))
      .toMatch(/nothing came back within 50ms/)
    expect(await call(ctx, 'proof_assemble', { goal_id: 'g1' }))
      .toMatch(/nothing came back within 50ms/)

    // Same slow interpreter, no ceiling of its own: still running.
    const pending = call(ctx, 'proof_status', {})
    const stillRunning = new Promise((resolve) => {
      setTimeout(() => { resolve('still running') }, 300)
    })
    await expect(Promise.race([pending.then(() => 'answered'), stillRunning]))
      .resolves.toBe('still running')
  })

  it('says the work may outlive the call it cut off', async () => {
    // Killing the child does not kill what the child started; a reader who
    // assumes otherwise reruns an attack that is already spending money.
    const ctx = await mount({ attackTimeoutMs: 50 })

    expect(await call(ctx, 'proof_attack', { goal_id: 'g1' }))
      .toMatch(/may still be running/)
  })
})

describe('the model credential an attack spends', () => {
  /** A credential store holding whatever the test says it holds. */
  async function withStore(held: Record<string, string>): Promise<Context> {
    const ctx = await mount()
    ctx.provide('credentials', {
      resolve: (ref: string) => Promise.resolve(
        held[ref] === undefined ? undefined : { value: held[ref], source: 'file' },
      ),
    } as never, true)
    return ctx
  }

  beforeEach(() => {
    vi.stubEnv('EVO_PYTHON', fakePython(ECHO_MODEL_ENV))
    vi.stubEnv('EVO_HARNESS_ROOT', process.cwd())
    // The launching shell has nothing: this is the preset case, where no
    // launcher exported anything and the store is the only source.
    vi.stubEnv('EVOHARNESS_API_BASE', '')
    vi.stubEnv('EVOHARNESS_API_KEY', '')
  })

  it('hands the store\'s key to the interpreter, which cannot read the store', async () => {
    // dsh's managed document is deliberately never materialized into the
    // process environment, so a subprocess sees a key configured there only
    // by being given it.
    const ctx = await withStore({
      EVOHARNESS_API_BASE: 'https://gateway.example/codex/v1',
      EVOHARNESS_API_KEY: 'sk-from-the-store',
    })

    const seen = JSON.parse(await call(ctx, 'proof_attack', { goal_id: 'g1' })) as {
      base: string | null
      key: string | null
      inherited: boolean
    }
    expect(seen.base).toBe('https://gateway.example/codex/v1')
    expect(seen.key).toBe('sk-from-the-store')
    // Layered, not substituted: an environment replaced wholesale takes PATH
    // with it, and the interpreter needs what it was started with.
    expect(seen.inherited).toBe(true)
  })

  it('leaves the child with nothing when the store holds nothing', async () => {
    // Not an error here. The refusal belongs to the Python side, which says
    // which two variables an attack needs; inventing a second refusal in this
    // file would put two different wordings in front of the same reader.
    const ctx = await withStore({})

    const seen = JSON.parse(await call(ctx, 'proof_attack', { goal_id: 'g1' })) as { key: string | null }
    expect(seen.key).toBeNull()
  })

  it('falls back to the inherited environment, which is the launcher\'s way', async () => {
    vi.stubEnv('EVOHARNESS_API_KEY', 'sk-from-the-shell')
    // No store at all: `dsh_proof.sh` exports the variables and mounts a
    // runtime that may carry no credential service.
    const ctx = await mount()

    const seen = JSON.parse(await call(ctx, 'proof_attack', { goal_id: 'g1' })) as { key: string | null }
    expect(seen.key).toBe('sk-from-the-shell')
  })

  it('does not spend a credential lookup on the calls that need no model', async () => {
    // Only attacking reaches a model. A status read that resolved a key would
    // put the secret in a child that has no use for it.
    const ctx = await withStore({ EVOHARNESS_API_KEY: 'sk-from-the-store' })

    const seen = JSON.parse(await call(ctx, 'proof_status', {})) as { key: string | null }
    expect(seen.key).toBeNull()
  })
})

describe('the session workspace', () => {
  it('refuses a session with nowhere to keep the graph', async () => {
    vi.stubEnv('EVO_PYTHON', fakePython(ECHO_ARGV))
    vi.stubEnv('EVO_HARNESS_ROOT', process.cwd())
    const ctx = await mount()

    expect(await callIn(ctx, undefined, 'proof_status', {}))
      .toMatch(/no workspace directory/)
  })
})

describe('where an assembled proof is written', () => {
  beforeEach(() => {
    vi.stubEnv('EVO_PYTHON', fakePython(ECHO_ARGV))
    vi.stubEnv('EVO_HARNESS_ROOT', process.cwd())
  })

  it('resolves a name against the session, not against the interpreter', async () => {
    // The interpreter starts in the EvoHarness checkout so `evoharness`
    // imports, and a relative path would resolve against that — putting the
    // proof in the source tree rather than beside the work it belongs to.
    const ctx = await mount()

    expect(await argvOf(ctx, 'proof_assemble', { goal_id: 'g1', out: 'P.lean' }))
      .toEqual(expect.arrayContaining(['--out', `${WORKSPACE}/P.lean`]))
  })

  it('refuses a path, because the model chose it', async () => {
    // The interpreter runs outside the session's file sandbox, so a path that
    // climbs out of the workspace is written wherever it points. A name is a
    // parameter; a path is a capability.
    const ctx = await mount()

    for (const out of ['../escape.lean', '/tmp/escape.lean', 'sub/dir.lean']) {
      expect(await call(ctx, 'proof_assemble', { goal_id: 'g1', out }))
        .toMatch(/plain file name, not a path/)
    }
  })

  it('refuses a name that addresses a directory rather than a file', async () => {
    const ctx = await mount()

    for (const out of ['.', '..']) {
      expect(await call(ctx, 'proof_assemble', { goal_id: 'g1', out }))
        .toMatch(/out must be the name of a file/)
    }
  })

  it('writes nothing anywhere when the caller names no file', async () => {
    // `out` is optional: omitting it asks for the verdict without a copy.
    const ctx = await mount()

    expect(await argvOf(ctx, 'proof_assemble', { goal_id: 'g1' })).not.toContain('--out')
  })
})
