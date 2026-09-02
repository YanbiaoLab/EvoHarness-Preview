import { mkdtempSync, writeFileSync, chmodSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import { CallId } from '@deepseek-ai/dsh-llm'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { SessionId } from '@deepseek-ai/dsh-session'
import type { ApprovalOutcome, ApprovalRequest } from '@deepseek-ai/dsh-user-approval'

import * as EvoStart from '../src/start.ts'

/**
 * What `evo_start` owes: it asks, and it launches only on `allowed-once`.
 *
 * A run costs hours of compute and real money, and a tool call is issued by
 * the model. The approval seam is the whole mechanism, so these cases drive it
 * directly with a stub answerer rather than through a model — the question is
 * not whether a model would ask nicely, it is whether the launch is reachable
 * without a person.
 */

const testToolSignal = new AbortController().signal

/** Echoes its argv the way the launch CLI prints JSON, and exits 0. */
const ECHO_ARGV = `
process.stdout.write(JSON.stringify({ argv: process.argv.slice(2) }))
`

function fakePython(body: string): string {
  const dir = mkdtempSync(join(tmpdir(), 'evo-start-'))
  const path = join(dir, 'fake-python')
  writeFileSync(path, `#!/usr/bin/env node\n${body}\n`, 'utf8')
  chmodSync(path, 0o755)
  return path
}

function agent(): Agent {
  return { id: 'session-1' as SessionId } as unknown as Agent
}

interface Asked {
  readonly requests: ApprovalRequest[]
}

/** Mount with a stub approval service that answers with `outcome`. */
async function mount(outcome: ApprovalOutcome): Promise<[Context, Asked]> {
  const asked: Asked = { requests: [] }
  const ctx = new Context()
  await ctx.plugin(SystemPrompt, {})
  await ctx.plugin(ToolRuntime)
  ctx.provide('approval', {
    request: (req: ApprovalRequest) => {
      asked.requests.push(req)
      return Promise.resolve(outcome)
    },
  } as never)
  await ctx.plugin(EvoStart)
  return [ctx, asked]
}

async function start(
  ctx: Context,
  args: Record<string, unknown> = {},
): Promise<string> {
  const result = await ctx.tools.execute({
    signal: testToolSignal,
    callId: CallId('c1'),
    name: 'evo_start',
    agent: agent(),
    arguments: {
      task: 'digit_power', run: 'exp1', generations: 6,
      purpose: 'see whether the seed can be beaten',
      ...args,
    },
  })
  const first = result.content[0]
  return first?.type === 'text' ? first.text : JSON.stringify(result.content)
}

beforeEach(() => {
  vi.stubEnv('EVO_RUNS_ROOT', '/runs')
  vi.stubEnv('EVO_TASKS_ROOT', '/tasks')
  vi.stubEnv('EVO_DSH_CONFIG', '/deploy/candidate.cordis.yml')
  vi.stubEnv('EVO_DSH_RUNTIME', '/deploy/bin.ts')
  vi.stubEnv('EVO_DSH_PROVIDER', 'evo-gateway')
  vi.stubEnv('DSH_MODEL', 'deepseek-v4-pro')
  vi.stubEnv('EVO_HARNESS_ROOT', process.cwd())
  vi.stubEnv('EVO_PYTHON', fakePython(ECHO_ARGV))
})

afterEach(() => {
  vi.unstubAllEnvs()
})

describe('nothing starts without a person', () => {
  it.each<ApprovalOutcome>(['rejected', 'cancelled', 'unavailable'])(
    'does not launch when approval is %s',
    async (outcome) => {
      const [ctx] = await mount(outcome)
      const result = await start(ctx)

      expect(result).toContain(`approval ${outcome}`)
      // The argv echo is what a launch would have produced. Its absence is
      // the assertion: no process was started.
      expect(result).not.toContain('evoharness.launch.start')
    },
  )

  it('says so plainly when the deployment has no answerer at all', async () => {
    // `unavailable` is not a person declining. Telling a model they are the
    // same invites it to retry a deployment problem forever.
    const [ctx] = await mount('unavailable')
    const result = await start(ctx)

    expect(result).toContain('no approval answerer')
    expect(result).toContain('command line')
  })

  it('refuses when there is no agent to ask through', async () => {
    const [ctx] = await mount('allowed-once')
    const result = await ctx.tools.execute({
      signal: testToolSignal,
      callId: CallId('c1'),
      name: 'evo_start',
      arguments: {
        task: 'digit_power', run: 'exp1', generations: 6, purpose: 'x',
      },
    })
    const text = result.content[0]?.type === 'text' ? result.content[0].text : ''
    expect(text).toContain('needs a live agent session')
  })
})

describe('what the person is asked', () => {
  it('states the cost and the purpose in the question itself', async () => {
    // The cost is the point of the question, so it belongs in the question
    // rather than in a log the approver would have to go and read.
    const [ctx, asked] = await mount('allowed-once')
    await start(ctx, { purpose: 'test whether islands help' })

    expect(asked.requests).toHaveLength(1)
    const reason = asked.requests[0]?.reason ?? ''
    expect(reason).toContain('test whether islands help')
    expect(reason).toContain('6 generations')
    expect(reason).toContain('keeps running after this session ends')
  })

  it('attaches the question to the tool call it is about', async () => {
    const [ctx, asked] = await mount('allowed-once')
    await start(ctx)
    expect(asked.requests[0]?.toolName).toBe('evo_start')
    expect(asked.requests[0]?.callId).toBe(CallId('c1'))
  })
})

describe('an approved run', () => {
  it('launches with paths the deployment chose, not the model', async () => {
    const [ctx] = await mount('allowed-once')
    const { argv } = JSON.parse(await start(ctx))

    expect(argv).toEqual([
      '-m', 'evoharness.launch.start',
      '--recipe', 'e0',
      '--task-dir', '/tasks/digit_power',
      '--run-dir', '/runs/exp1',
      '--live',
      '--dsh-config', '/deploy/candidate.cordis.yml',
      '--dsh-runtime', '/deploy/bin.ts',
      '--dsh-provider', 'evo-gateway',
      '--model', 'deepseek-v4-pro',
      '--set', 'proposal.mode=agentic',
      'search.num_generations=6',
    ])
  })

  it('asks for the model the candidate config actually offers', async () => {
    // One variable, two readers. `candidate.cordis.yml` builds its catalog
    // from DSH_MODEL and the runtime resolves the request against that
    // catalog, so a second source for the same name is a 404 waiting to
    // happen — and it did: run etp_one_run1 spent five generations, five
    // seconds and zero offspring asking for EvoHarness's own default before
    // stopping as `proposer_dead`, a verdict about a proposer that worked.
    vi.stubEnv('DSH_MODEL', 'some-other-model')
    const [ctx] = await mount('allowed-once')
    const { argv } = JSON.parse(await start(ctx))

    expect(argv[argv.indexOf('--model') + 1]).toBe('some-other-model')
  })

  it('will not launch at all when the model is not named', async () => {
    // Fails before asking. A person approving a run that cannot make a single
    // request has spent their approval on nothing.
    vi.stubEnv('DSH_MODEL', undefined)
    const [ctx, asked] = await mount('allowed-once')
    const result = await start(ctx)

    expect(result).toContain('DSH_MODEL')
    expect(result).not.toContain('evoharness.launch.start')
    expect(asked.requests).toHaveLength(0)
  })

  it('asks before it launches, not after', async () => {
    // Ordering is the property, and a stub that records both is the only way
    // to see it. A launch that happened first would still "have an approval".
    const order: string[] = []
    const ctx = new Context()
    await ctx.plugin(SystemPrompt, {})
    await ctx.plugin(ToolRuntime)
    ctx.provide('approval', {
      request: () => {
        order.push('asked')
        return Promise.resolve('allowed-once' as ApprovalOutcome)
      },
    } as never)
    await ctx.plugin(EvoStart)

    vi.stubEnv('EVO_PYTHON', fakePython(`
      process.stdout.write(JSON.stringify({ argv: process.argv.slice(2) }))
    `))
    await start(ctx)
    order.push('launched')

    expect(order).toEqual(['asked', 'launched'])
  })

  it('reports a launch that failed after approval as exactly that', async () => {
    // Distinct from a refusal: the person said yes, and the deployment is
    // what broke. A caller told "not approved" would ask again for nothing.
    vi.stubEnv('EVO_PYTHON', fakePython(`
      process.stderr.write('run directory is inside /private/tmp')
      process.exit(1)
    `))
    const [ctx] = await mount('allowed-once')
    const result = await start(ctx)

    expect(result).toContain('approved, but the run did not start')
    expect(result).toContain('inside /private/tmp')
  })
})

describe('what the model may choose', () => {
  it.each([
    ['a path instead of a run name', { run: '../../etc' }],
    ['a path instead of a task name', { task: '/etc/passwd' }],
  ])('refuses %s', async (_label, args) => {
    const [ctx, asked] = await mount('allowed-once')
    const result = await start(ctx, args)

    expect(result).toMatch(/plain name, not a path/)
    // Refused before asking: a person should not be shown a question about a
    // request that was never going to be honoured.
    expect(asked.requests).toHaveLength(0)
  })

  // What matters is that it is refused and nobody is asked, not which layer
  // said so: the parameter schema rejects a non-integer before `execute`
  // runs, and the range check catches the rest. Asserting one message would
  // have made a stricter schema look like a regression.
  it.each([0, -1, 51, 2.5])('refuses %s generations', async (generations) => {
    const [ctx, asked] = await mount('allowed-once')
    const result = await start(ctx, { generations })

    expect(result).toMatch(/must be (a whole number|an integer)/)
    expect(result).not.toContain('evoharness.launch.start')
    expect(asked.requests).toHaveLength(0)
  })

  it('accepts a count inside the ceiling', async () => {
    // The liveness control for the case above: a check that refused every
    // number would satisfy it while making the tool unusable.
    const [ctx, asked] = await mount('allowed-once')
    const { argv } = JSON.parse(await start(ctx, { generations: 50 }))

    expect(argv).toContain('search.num_generations=50')
    expect(asked.requests).toHaveLength(1)
  })

  it('registers exactly one tool, and it is the one that asks', async () => {
    // Anything else added here would act without going through the seam
    // above, so adding a tool has to come past this case first.
    const [ctx] = await mount('allowed-once')
    expect(ctx.tools.schemas().map(schema => schema.name)).toEqual(['evo_start'])
  })
})

describe('where a launch takes its settings from', () => {
  /** Report what the launched run was actually given, beyond its argv. */
  const ECHO_LAUNCH_ENV = `
process.stdout.write(JSON.stringify({
  argv: process.argv.slice(2),
  key: process.env.EVOHARNESS_API_KEY || null,
  model: process.env.DSH_MODEL || null,
  inherited: process.env.PATH !== undefined,
}))
`

  /** Mount with an approving answerer, a config, and optionally a store. */
  async function mountConfigured(
    config: Record<string, unknown>,
    held?: Record<string, string>,
  ): Promise<Context> {
    const ctx = new Context()
    await ctx.plugin(SystemPrompt, {})
    await ctx.plugin(ToolRuntime)
    ctx.provide('approval', { request: () => Promise.resolve('allowed-once') } as never)
    if (held !== undefined) {
      ctx.provide('credentials', {
        resolve: (ref: string) => Promise.resolve(
          held[ref] === undefined ? undefined : { value: held[ref], source: 'file' },
        ),
      } as never, true)
    }
    await ctx.plugin(EvoStart, config)
    return ctx
  }

  const settings = (python: string) => ({
    python,
    harnessRoot: process.cwd(),
    runsRoot: '/runs',
    tasksRoot: '/tasks',
    dshConfig: '/candidate.cordis.yml',
    dshRuntime: '/runtime/bin.ts',
    dshProvider: 'evo-gateway',
    model: 'a-model-the-catalog-offers',
  })

  beforeEach(() => {
    // Nothing exported: this is the preset case, where no launcher ran.
    for (const name of [
      'EVO_RUNS_ROOT', 'EVO_TASKS_ROOT', 'EVO_DSH_CONFIG', 'EVO_DSH_RUNTIME',
      'EVO_DSH_PROVIDER', 'DSH_MODEL', 'EVO_PYTHON', 'EVO_HARNESS_ROOT',
      'EVOHARNESS_API_KEY',
    ]) vi.stubEnv(name, '')
  })

  it('takes every path from the plugin row when nothing is exported', async () => {
    const ctx = await mountConfigured(settings(fakePython(ECHO_ARGV)))

    const { argv } = JSON.parse(await start(ctx)) as { argv: string[] }
    expect(argv).toEqual(expect.arrayContaining(['--dsh-config', '/candidate.cordis.yml']))
    expect(argv).toEqual(expect.arrayContaining(['--dsh-runtime', '/runtime/bin.ts']))
    expect(argv).toEqual(expect.arrayContaining(['--model', 'a-model-the-catalog-offers']))
  })

  it('names both places a missing setting can come from', async () => {
    // A reader told only "EVO_DSH_CONFIG not set" under a preset goes looking
    // for a launcher that is not in the picture.
    const ctx = await mountConfigured({})

    const reported = await start(ctx)
    expect(reported).toMatch(/EVO_DSH_CONFIG[\s\S]*`dshConfig`/)
  })

  it('hands the run a credential the process environment does not carry', async () => {
    // The run outlives this call and is `--live`, so what it inherits now is
    // what it spends for hours. dsh's managed credential document is never
    // materialized into the process environment.
    const ctx = await mountConfigured(
      settings(fakePython(ECHO_LAUNCH_ENV)),
      { EVOHARNESS_API_KEY: 'sk-from-the-store' },
    )

    const seen = JSON.parse(await start(ctx)) as {
      key: string | null
      inherited: boolean
    }
    expect(seen.key).toBe('sk-from-the-store')
    // Layered, not substituted: an environment replaced wholesale loses PATH.
    expect(seen.inherited).toBe(true)
  })

  it('carries the model name the run was told to ask for', async () => {
    // The candidate runtime's cordis file builds its catalog from DSH_MODEL
    // while the run asks for `--model`, and nothing in a preset exports it.
    // A catalog falling back to its own default answers 404 to every request,
    // which surfaces as a dead proposer rather than as a wrong model name.
    const ctx = await mountConfigured(settings(fakePython(ECHO_LAUNCH_ENV)))

    const seen = JSON.parse(await start(ctx)) as { model: string | null }
    expect(seen.model).toBe('a-model-the-catalog-offers')
  })
})
