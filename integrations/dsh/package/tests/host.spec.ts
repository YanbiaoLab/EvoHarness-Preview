import { mkdtempSync, writeFileSync, chmodSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import { CallId } from '@deepseek-ai/dsh-llm'

import * as EvoHost from '../src/host.ts'
import { resolveRun } from '../src/host.ts'

const testToolSignal = new AbortController().signal

/** Echo the argv back the way the readout CLI prints JSON, and exit 0. */
const ECHO_ARGV = `
process.stdout.write(JSON.stringify({ argv: process.argv.slice(2) }))
`

/**
 * A stand-in for the EvoHarness readout CLI. What these tools owe Python is
 * argv shape and exit-code handling; a script echoing its argv tests both
 * without an interpreter, a run directory, or a populated store.
 */
function fakeReadout(body: string): string {
  const dir = mkdtempSync(join(tmpdir(), 'evo-host-'))
  const path = join(dir, 'fake-python')
  writeFileSync(path, `#!/usr/bin/env node\n${body}\n`, 'utf8')
  chmodSync(path, 0o755)
  return path
}

async function mount(): Promise<Context> {
  const ctx = new Context()
  await ctx.plugin(SystemPrompt, {})
  await ctx.plugin(ToolRuntime)
  await ctx.plugin(EvoHost)
  return ctx
}

async function call(
  ctx: Context,
  name: string,
  args: Record<string, unknown>,
): Promise<string> {
  const result = await ctx.tools.execute({
    signal: testToolSignal,
    callId: CallId('c1'),
    name,
    arguments: args,
  })
  const first = result.content[0]
  return first?.type === 'text' ? first.text : JSON.stringify(result.content)
}

beforeEach(() => {
  vi.stubEnv('EVO_RUNS_ROOT', '/runs')
  vi.stubEnv('EVO_RESEARCH_ROOT', '/research')
  vi.stubEnv('EVO_TASKS_ROOT', '/tasks')
  vi.stubEnv('EVO_HARNESS_ROOT', process.cwd())
  vi.stubEnv('EVO_PYTHON', fakeReadout(ECHO_ARGV))
})

afterEach(() => {
  vi.unstubAllEnvs()
})

describe('resolveRun', () => {
  it('joins a plain name onto the configured root', () => {
    expect(resolveRun('/runs', 'exp1')).toBe('/runs/exp1')
  })

  // A model-supplied path is not a parameter, it is a choice of which
  // directory on the machine gets read. Every spelling of "leave the root"
  // has to be refused, not sanitised into something plausible.
  it.each([
    ['a parent traversal', '../etc'],
    ['a nested traversal', 'a/../../etc'],
    ['an absolute path', '/etc/passwd'],
    ['a bare separator', 'a/b'],
    ['a windows separator', 'a\\b'],
    ['dot-dot alone', '..'],
    ['empty', ''],
  ])('refuses %s', (_label, name) => {
    expect(() => resolveRun('/runs', name)).toThrow()
  })
})

describe('evo_status', () => {
  it('asks for one run when given a name', async () => {
    const ctx = await mount()
    const { argv } = JSON.parse(await call(ctx, 'evo_status', { run: 'exp1' }))
    expect(argv).toEqual([
      '-m', 'evoharness.readout', 'status', '--run-dir', '/runs/exp1',
    ])
  })

  it('lists every run when given none', async () => {
    const ctx = await mount()
    const { argv } = JSON.parse(await call(ctx, 'evo_status', {}))
    expect(argv).toEqual([
      '-m', 'evoharness.readout', 'list', '--root', '/runs',
    ])
  })

  it('refuses a run name that leaves the configured root', async () => {
    const ctx = await mount()
    // The error reaches the model as the call's result text, which is the
    // only channel back into the turn.
    expect(await call(ctx, 'evo_status', { run: '../../etc' }))
      .toMatch(/plain name, not a path/)
  })
})

describe('evo_task_check', () => {
  it('loads the task the way a run would', async () => {
    const ctx = await mount()
    const { argv } = JSON.parse(
      await call(ctx, 'evo_task_check', { task: 'digit_power' }),
    )
    // The authoring loader, not a lint: the only way to know the grade
    // function imports and the declaration parses is to load it.
    expect(argv).toEqual([
      '-m', 'evoharness.authoring', 'check',
      '--task-dir', '/tasks/digit_power',
    ])
  })

  it('refuses a task name that leaves the configured root', async () => {
    const ctx = await mount()
    expect(await call(ctx, 'evo_task_check', { task: '../../etc' }))
      .toMatch(/plain name, not a path/)
  })

  it('names the missing variable when tasks have no configured home', async () => {
    vi.stubEnv('EVO_TASKS_ROOT', undefined)
    const ctx = await mount()
    expect(await call(ctx, 'evo_task_check', { task: 'x' }))
      .toMatch(/EVO_TASKS_ROOT/)
  })
})

describe('evo_trajectory', () => {
  it('defaults to the summary view', async () => {
    const ctx = await mount()
    const { argv } = JSON.parse(
      await call(ctx, 'evo_trajectory', { run: 'exp1' }),
    )
    // Summary by default because full lists every candidate of every
    // generation, and a run makes hundreds.
    expect(argv).toEqual([
      '-m', 'evoharness.readout', 'trajectory',
      '--run-dir', '/runs/exp1',
      '--detail', 'summary',
    ])
  })

  it('passes the full view through when asked', async () => {
    const ctx = await mount()
    const { argv } = JSON.parse(
      await call(ctx, 'evo_trajectory', { run: 'exp1', detail: 'full' }),
    )
    expect(argv.slice(-2)).toEqual(['--detail', 'full'])
  })
})

describe('failure reporting', () => {
  it('hands a python refusal to the model in its own words', async () => {
    vi.stubEnv('EVO_PYTHON', fakeReadout(`
      process.stdout.write(JSON.stringify({
        error: 'not a run directory: /runs/nope', kind: 'readout',
      }))
      process.exit(2)
    `))
    const ctx = await mount()

    expect(await call(ctx, 'evo_status', { run: 'nope' }))
      .toMatch(/not a run directory/)
  })

  it('does not dress a broken deployment up as a missing run', async () => {
    vi.stubEnv('EVO_PYTHON', fakeReadout(`
      process.stderr.write('ModuleNotFoundError: evoharness')
      process.exit(1)
    `))
    const ctx = await mount()

    const outcome = await call(ctx, 'evo_status', { run: 'exp1' })
    expect(outcome).toMatch(/exit 1/)
    expect(outcome).toMatch(/ModuleNotFoundError/)
    expect(outcome).not.toMatch(/not a run directory/)
  })

  it('names the missing variable when the host is not configured', async () => {
    vi.stubEnv('EVO_RUNS_ROOT', undefined)
    const ctx = await mount()

    expect(await call(ctx, 'evo_status', {})).toMatch(/EVO_RUNS_ROOT/)
  })
})

describe('governance is readable and not answerable', () => {
  it('reads the queue when given no request id', async () => {
    const ctx = await mount()
    const { argv } = JSON.parse(await call(ctx, 'evo_decisions', {}))
    expect(argv).toEqual([
      '-m', 'evoharness.readout', 'cards', '--research-root', '/research',
    ])
  })

  it('reads one card in full when given one', async () => {
    const ctx = await mount()
    const { argv } = JSON.parse(
      await call(ctx, 'evo_decisions', { request_id: 'req-1' }),
    )
    expect(argv).toEqual([
      '-m', 'evoharness.readout', 'card',
      '--research-root', '/research', '--id', 'req-1',
    ])
  })

  it('reads the audit record of what was already signed', async () => {
    const ctx = await mount()
    const { argv } = JSON.parse(await call(ctx, 'evo_decided', { limit: 5 }))
    expect(argv).toEqual([
      '-m', 'evoharness.readout', 'decided',
      '--research-root', '/research', '--limit', '5',
    ])
  })

  it('registers exactly these tools and nothing that writes', async () => {
    // DH-5's whole boundary. A tool call is issued by the model, so an
    // answering tool would mean the model decides when a governance decision
    // is signed — "go ahead" in a chat is the model's reading of a sentence,
    // not a person's signature.
    //
    // An exact list rather than a pattern over names. The first version of
    // this matched /decide/ and failed on `evo_decided`, which reads the
    // audit record: what makes a tool dangerous is what it does, and a name
    // test both accuses the innocent and would clear an `evo_confirm`.
    // Spelling the set out means adding any tool has to come here first.
    // `evo_task_check` joined the list by coming through here: it loads a
    // task directory and reports what it found, and starts nothing. Anything
    // that can act belongs in `start.ts`, behind the approval seam.
    const ctx = await mount()
    expect(ctx.tools.schemas().map(schema => schema.name).sort()).toEqual([
      'evo_decided',
      'evo_decisions',
      'evo_status',
      'evo_task_check',
      'evo_trajectory',
    ])
  })

  it('needs the research root named before it will read anything', async () => {
    vi.stubEnv('EVO_RESEARCH_ROOT', undefined)
    const ctx = await mount()
    expect(await call(ctx, 'evo_decisions', {})).toMatch(/EVO_RESEARCH_ROOT/)
  })

  it('does not demand a research root from the run tools', async () => {
    // The two sets are separate on purpose: a deployment that only watches
    // runs should not fail because it configured no governance ledger.
    vi.stubEnv('EVO_RESEARCH_ROOT', undefined)
    const ctx = await mount()
    const { argv } = JSON.parse(await call(ctx, 'evo_status', { run: 'exp1' }))
    expect(argv).toContain('/runs/exp1')
  })
})

describe('where the read-only tools take their roots from', () => {
  const settings = {
    python: '',
    harnessRoot: process.cwd(),
    runsRoot: '/configured-runs',
    tasksRoot: '/configured-tasks',
    researchRoot: '/configured-research',
  }

  async function mountConfigured(config: Record<string, unknown>): Promise<Context> {
    const ctx = new Context()
    await ctx.plugin(SystemPrompt, {})
    await ctx.plugin(ToolRuntime)
    await ctx.plugin(EvoHost, config)
    return ctx
  }

  beforeEach(() => {
    // Nothing exported: this is the preset case, where no launcher ran.
    for (const name of [
      'EVO_RUNS_ROOT', 'EVO_TASKS_ROOT', 'EVO_RESEARCH_ROOT', 'EVO_HARNESS_ROOT',
    ]) vi.stubEnv(name, '')
  })

  it('resolves names under the roots the plugin row declares', async () => {
    const ctx = await mountConfigured({ ...settings, python: fakeReadout(ECHO_ARGV) })

    expect(await call(ctx, 'evo_status', { run: 'exp1' }))
      .toContain('/configured-runs/exp1')
    expect(await call(ctx, 'evo_task_check', { task: 't1' }))
      .toContain('/configured-tasks/t1')
    expect(await call(ctx, 'evo_decisions', {}))
      .toContain('/configured-research')
  })

  it('lets a deployment configure runs without a ledger', async () => {
    // Two separate setting lists on purpose: demanding the research root for
    // a run readout would fail the common case, where there is no ledger yet.
    const ctx = await mountConfigured({
      python: fakeReadout(ECHO_ARGV),
      harnessRoot: process.cwd(),
      runsRoot: '/configured-runs',
    })

    expect(await call(ctx, 'evo_status', {})).toContain('/configured-runs')
    expect(await call(ctx, 'evo_decisions', {})).toMatch(/EVO_RESEARCH_ROOT/)
  })

  it('names both places a missing root can come from', async () => {
    const ctx = await mountConfigured({ python: fakeReadout(ECHO_ARGV) })

    expect(await call(ctx, 'evo_status', {})).toMatch(/EVO_RUNS_ROOT[\s\S]*`runsRoot`/)
  })
})
