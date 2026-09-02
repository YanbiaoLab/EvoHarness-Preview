import { mkdtempSync, writeFileSync, chmodSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import { CallId } from '@deepseek-ai/dsh-llm'

import * as EvoPeer from '../src/peer.ts'

const testToolSignal = new AbortController().signal

/**
 * A stand-in for the EvoHarness readout CLI. The plugin's contract with
 * Python is three things and nothing else: argv shape, exit code, and JSON on
 * stdout. A script asserting the argv it received tests all three without a
 * Python interpreter, a run directory, or a populated store.
 */
function fakeReadout(body: string): string {
  const dir = mkdtempSync(join(tmpdir(), 'evo-peer-'))
  const path = join(dir, 'fake-python')
  writeFileSync(path, `#!/usr/bin/env node\n${body}\n`, 'utf8')
  chmodSync(path, 0o755)
  return path
}

/** Echo the argv back as the peer view would, and exit 0. */
const ECHO_ARGV = `
const argv = process.argv.slice(2)
process.stdout.write(JSON.stringify({ ok: true, argv }))
`

async function mount(): Promise<Context> {
  const ctx = new Context()
  await ctx.plugin(SystemPrompt, {})
  await ctx.plugin(ToolRuntime)
  await ctx.plugin(EvoPeer)
  return ctx
}

async function inspect(
  ctx: Context,
  args: Record<string, unknown>,
): Promise<string> {
  const result = await ctx.tools.execute({
    signal: testToolSignal,
    callId: CallId('c1'),
    name: 'evo_inspect_candidate',
    arguments: args,
  })
  const first = result.content[0]
  return first?.type === 'text' ? first.text : JSON.stringify(result.content)
}

beforeEach(() => {
  vi.stubEnv('EVO_RUN_DIR', '/runs/r1')
  vi.stubEnv('EVO_HARNESS_ROOT', process.cwd())
})

afterEach(() => {
  vi.unstubAllEnvs()
})

describe('evo_inspect_candidate', () => {
  it('asks python for the narrow peer view, not the full candidate view', async () => {
    vi.stubEnv('EVO_PYTHON', fakeReadout(ECHO_ARGV))
    const ctx = await mount()

    const { argv } = JSON.parse(await inspect(ctx, { candidate_id: 'abc123' }))

    // `peer`, never `candidate`: the full view carries the metrics a task
    // author withheld from the optimizer on purpose.
    expect(argv).toEqual([
      '-m', 'evoharness.readout', 'peer',
      '--run-dir', '/runs/r1',
      '--id', 'abc123',
    ])
  })

  it('passes a path through only when one was given', async () => {
    vi.stubEnv('EVO_PYTHON', fakeReadout(ECHO_ARGV))
    const ctx = await mount()

    const { argv } = JSON.parse(
      await inspect(ctx, { candidate_id: 'abc123', path: 'main.py' }),
    )
    expect(argv.slice(-2)).toEqual(['--path', 'main.py'])
  })

  it('never lets an argument become a shell string', async () => {
    vi.stubEnv('EVO_PYTHON', fakeReadout(ECHO_ARGV))
    const ctx = await mount()

    // Model-authored input reaches argv verbatim. Were this going through a
    // shell it would run `touch`; arriving intact is the evidence it is not.
    const hostile = 'abc123"; touch /tmp/evo-peer-pwned; echo "'
    const { argv } = JSON.parse(await inspect(ctx, { candidate_id: hostile }))
    expect(argv).toContain(hostile)
  })

  // The three cases below all fail, and what is under test is what the MODEL
  // reads. A tool that throws does not reject here — the runtime turns the
  // error into the call's result text, which is the only channel back into
  // the turn. An unhelpful message is therefore a wasted turn, not a stack
  // trace someone will find in a log.

  it("hands a python refusal to the model in the tool's own words", async () => {
    vi.stubEnv('EVO_PYTHON', fakeReadout(`
      process.stdout.write(JSON.stringify({
        error: "no candidate with id 'list'; ids are opaque",
        kind: 'readout',
      }))
      process.exit(2)
    `))
    const ctx = await mount()

    expect(await inspect(ctx, { candidate_id: 'list' })).toMatch(
      /ids are opaque/,
    )
  })

  it('does not dress a broken deployment up as a bad candidate id', async () => {
    // Exit 1 is the readout CLI's "something unforeseen". A candidate told
    // "unknown candidate" here would spend its remaining turns guessing ids.
    vi.stubEnv('EVO_PYTHON', fakeReadout(`
      process.stderr.write('ModuleNotFoundError: evoharness')
      process.exit(1)
    `))
    const ctx = await mount()

    const outcome = await inspect(ctx, { candidate_id: 'abc123' })
    expect(outcome).toMatch(/exit 1/)
    expect(outcome).toMatch(/ModuleNotFoundError/)
    expect(outcome).not.toMatch(/opaque|no candidate with id/)
  })

  it('names the missing variable when the runtime was not launched by EvoHarness', async () => {
    vi.stubEnv('EVO_RUN_DIR', undefined)
    vi.stubEnv('EVO_PYTHON', fakeReadout(ECHO_ARGV))
    const ctx = await mount()

    expect(await inspect(ctx, { candidate_id: 'abc123' })).toMatch(
      /EVO_RUN_DIR/,
    )
  })
})
