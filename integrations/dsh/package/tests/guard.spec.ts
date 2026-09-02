import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import type { PreToolDecision, ToolDefinition } from '@deepseek-ai/dsh-tools'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { CallId } from '@deepseek-ai/dsh-llm'
import type { SessionId } from '@deepseek-ai/dsh-session'

import * as EvoSpike from '../src/index.ts'

const testToolSignal = new AbortController().signal

/**
 * Mount the tool registry, a stub `subagents` service the spike injects but
 * these cases never reach, and the spike itself.
 */
async function mount(): Promise<Context> {
  const ctx = new Context()
  await ctx.plugin(SystemPrompt, {})
  await ctx.plugin(ToolRuntime)
  ctx.provide('subagents', { start: () => Promise.reject(new Error('unused')) } as never)
  await ctx.plugin(EvoSpike)
  return ctx
}

/**
 * A stand-in for an agent at a known delegation depth. `delegationDepthOf`
 * reads both the runtime option and the persisted header, so both carry it.
 */
function agentAtDepth(id: string, depth: number): Agent {
  return {
    id: id as SessionId,
    options: { subagentDepth: depth },
    session: { header: { delegationDepth: depth } },
  } as unknown as Agent
}

function stubTool(name: string): ToolDefinition {
  return {
    name,
    description: `stub ${name}`,
    parameters: { type: 'object', properties: {} },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value as string }],
    },
    execute: (): Promise<string> => Promise.resolve(`ran:${name}`),
  }
}

async function run(ctx: Context, name: string, agent?: Agent): Promise<string> {
  const result = await ctx.tools.execute({
    signal: testToolSignal,
    callId: CallId('c1'),
    name,
    arguments: {},
    ...agent ? { agent } : {},
  })
  const first = result.content[0]
  return first?.type === 'text' ? first.text : JSON.stringify(result.content)
}

/** Read the spike's own report tool and parse its canonical JSON payload. */
async function report(ctx: Context): Promise<{
  denied: string[]
  allowed: string[]
  denylistNotVisible: string[]
  ledger: { tool: string; agent: string; depth: number | null; at: string }[]
}> {
  return JSON.parse(await run(ctx, 'evo_spike_tools')) as never
}

describe('evo-spike guard', () => {
  it('denies a denylisted capability to a child agent and returns the reason', async () => {
    const ctx = await mount()
    ctx.tools.register(stubTool('cordis_define'))

    const outcome = await run(ctx, 'cordis_define', agentAtDepth('child', 1))

    expect(outcome).toContain('not available in a candidate runtime')
  })

  it('denies the ROOT agent too — that is the candidate', async () => {
    // This case asserted the opposite, and the opposite is the bug. One
    // runtime runs one candidate, and the SDK makes that candidate the root
    // agent at depth 0, so "allow below depth 1" let the very agent the guard
    // exists for through every time. A live evolution ran that way.
    const ctx = await mount()
    ctx.tools.register(stubTool('cordis_define'))

    expect(await run(ctx, 'cordis_define', agentAtDepth('root', 0)))
      .toContain('not available in a candidate runtime')
    expect((await report(ctx)).ledger).toMatchObject([
      { tool: 'cordis_define', agent: 'root', depth: 0 },
    ])
  })

  it('denies a call no agent can be attributed to', async () => {
    // Nothing in this runtime is more privileged than the candidate, so an
    // unattributed call is not a safer one — it is one the ledger cannot name.
    const ctx = await mount()
    ctx.tools.register(stubTool('cordis_define'))

    expect(await run(ctx, 'cordis_define'))
      .toContain('not available in a candidate runtime')
    expect((await report(ctx)).ledger).toMatchObject([
      { tool: 'cordis_define', agent: '(no agent)', depth: null },
    ])
  })

  it('leaves capabilities outside the denylist available to a child', async () => {
    const ctx = await mount()
    ctx.tools.register(stubTool('bash'))

    expect(await run(ctx, 'bash', agentAtDepth('child', 1))).toBe('ran:bash')
  })

  it('survives a prepended force-allow pre-execute listener', async () => {
    const ctx = await mount()
    ctx.tools.register(stubTool('cordis_define'))
    // The strongest force-allow the seam admits: it never calls `next()`, so it
    // short-circuits the whole extensible waterfall, and it is prepended ahead
    // of every other policy. The guard runs after the waterfall regardless.
    ctx.on('tools/pre-execute', (): Promise<PreToolDecision> => Promise.resolve({ kind: 'allow' }), { prepend: true })

    expect(await run(ctx, 'cordis_define', agentAtDepth('child', 1)))
      .toContain('not available in a candidate runtime')
  })

  it('fires for a grandchild, not just the direct child', async () => {
    const ctx = await mount()
    ctx.tools.register(stubTool('cordis_run'))

    await run(ctx, 'cordis_run', agentAtDepth('child', 1))
    await run(ctx, 'cordis_run', agentAtDepth('grandchild', 2))

    expect((await report(ctx)).ledger.map(entry => ({ agent: entry.agent, depth: entry.depth })))
      .toEqual([{ agent: 'child', depth: 1 }, { agent: 'grandchild', depth: 2 }])
  })

  it('reports the denylist split against the live registry', async () => {
    const ctx = await mount()
    ctx.tools.register(stubTool('cordis_define'))
    ctx.tools.register(stubTool('bash'))

    const view = await report(ctx)

    expect(view.denied).toEqual(['cordis_define'])
    expect(view.allowed).toEqual(['bash', 'evo_spike_probe', 'evo_spike_tools'])
    expect(view.denylistNotVisible).toEqual([
      'ask_user_question',
      'cordis_run',
      'cordis_stop',
      'cordis_undefine',
      'web_fetch',
      'web_search',
    ])
  })
})
