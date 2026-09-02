import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import { mountAgentLoopTestDependencies } from '@deepseek-ai/dsh-agent-loop-testkit'
import * as LlmDeepSeek from '@deepseek-ai/dsh-llm-deepseek'
import SubagentRuntime from '@deepseek-ai/dsh-subagent'
import * as Spawn from '@deepseek-ai/dsh-subagent-spawn-in-process'
import * as ToolSubagent from '@deepseek-ai/dsh-tool-subagent'
import type { PreToolDecision, ToolDefinition } from '@deepseek-ai/dsh-tools'
import { CallId } from '@deepseek-ai/dsh-llm'
import { SessionId } from '@deepseek-ai/dsh-session'
import type { Agent } from '@deepseek-ai/dsh-agent'

import * as EvoSpike from '../src/index.ts'

/**
 * Key-gated round that the unit suite structurally cannot run: it asserts the
 * guard against a REAL spawned child. The unit suite fabricates an agent and
 * hands it to the guard directly, so it assumes the very thing that carries
 * the guard — that a spawned descendant reaches it at all, and that the depth
 * the ledger records survives to `exec.agent`. Only a real child settles that.
 *
 * Depth no longer decides anything; the denial is by name. It is still
 * asserted here because it is the evidence that the entries below came from
 * distinct generations of child rather than all from the parent.
 *
 * `cordis_define` here is a stub sharing the real tool's NAME. What is under
 * test is depth propagation and guard consultation on a real child, not the
 * dynamic-plugin tool's body — and a stub keeps the model from actually
 * mounting anything.
 */

const MODEL = { provider: 'deepseek-official', model: 'deepseek-v4-flash' }

interface ProbeReport {
  childSession: string
  stopReason: string
  childReport: string
  denialsTriggered: { tool: string; agent: string; depth: number | null; at: string }[]
}

let ctx: Context | undefined

afterEach(async () => {
  await ctx?.fiber.dispose()
  ctx = undefined
})

function stubTool(name: string): ToolDefinition {
  return {
    name,
    description: 'Define and mount a Cordis plugin into the running tree.',
    parameters: { type: 'object', properties: {} },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value as string }],
    },
    execute: (): Promise<string> => Promise.resolve('MOUNTED'),
  }
}

async function harness(): Promise<{ ctx: Context; parent: Agent }> {
  const context = new Context()
  await mountAgentLoopTestDependencies(context, {
    systemPrompt: { persona: 'You are a test probe. Do exactly what you are asked and report verbatim.' },
  })
  await context.plugin(AgentLoop, { agents: [] })
  await context.plugin(LlmDeepSeek)
  await context.plugin(SubagentRuntime)
  await context.plugin(Spawn, { providerName: 'spawn' })
  await context.plugin(ToolSubagent, { provider: 'spawn', backgroundMode: 'one-shot' })
  context.tools.register(stubTool('cordis_define'))
  await context.plugin(EvoSpike)
  return { ctx: context, parent: context.agentLoop.create(SessionId('evo-spike-e2e'), MODEL) }
}

/** Drive the spike's probe tool directly, so only the CHILD's turn involves a model. */
async function probe(context: Context, parent: Agent, mission?: string): Promise<ProbeReport> {
  const result = await context.tools.execute({
    signal: new AbortController().signal,
    callId: CallId('probe'),
    name: 'evo_spike_probe',
    arguments: mission === undefined ? {} : { mission },
    agent: parent,
  })
  const first = result.content[0]
  const payload = first?.type === 'text' ? first.text : JSON.stringify(result.content)
  return JSON.parse(payload) as ProbeReport
}

describe.skipIf(!process.env.DEEPSEEK_API_KEY)('evo-spike guard against a real spawned child', () => {
  it('denies a real child at delegation depth 1', async () => {
    const harnessed = await harness()
    ctx = harnessed.ctx

    const report = await probe(harnessed.ctx, harnessed.parent)

    expect(report.denialsTriggered).toHaveLength(1)
    expect(report.denialsTriggered[0]).toMatchObject({ tool: 'cordis_define', depth: 1 })
    // The denial has to reach the model, or a candidate retries forever
    // believing it got its arguments wrong.
    expect(report.childReport).toContain('not available in a candidate runtime')
  }, 180_000)

  it('still denies a real child under a prepended force-allow listener', async () => {
    const harnessed = await harness()
    ctx = harnessed.ctx
    harnessed.ctx.on(
      'tools/pre-execute',
      (): Promise<PreToolDecision> => Promise.resolve({ kind: 'allow' }),
      { prepend: true },
    )

    const report = await probe(harnessed.ctx, harnessed.parent)

    expect(report.denialsTriggered).toHaveLength(1)
    expect(report.denialsTriggered[0]).toMatchObject({ depth: 1 })
  }, 180_000)

  it('reaches a grandchild the guard was never told about', async () => {
    const harnessed = await harness()
    ctx = harnessed.ctx

    const report = await probe(
      harnessed.ctx,
      harnessed.parent,
      'Use the subagent tool to delegate this exact task: "Call the cordis_define tool exactly once '
      + 'with any plausible arguments." Then report verbatim what that subagent told you.',
    )

    // A depth-2 entry is the only proof that the guard is consulted for
    // descendants it never registered against. The predicate ignores depth,
    // so this asserts reach, not the rule.
    expect(report.denialsTriggered.some(entry => entry.depth === 2)).toBe(true)
  }, 300_000)
})
