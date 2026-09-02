import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import type { PreToolDecision, ToolDefinition } from '@deepseek-ai/dsh-tools'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { CallId } from '@deepseek-ai/dsh-llm'
import type { SessionId } from '@deepseek-ai/dsh-session'

import * as EvoSpike from '../src/index.ts'

/**
 * DH-4's adversarial round, as cases that need no model.
 *
 * The original plan ran these by prompting a real candidate into each attempt.
 * That is worth doing once, but it is the wrong thing to keep: it costs money,
 * it needs an endpoint with credit, and a model that simply declines to try
 * produces a green result that proves nothing about the mechanism. Driving the
 * tool seam directly asserts the mechanism itself, every run, for free.
 *
 * What a model is still needed for is the other half — whether a candidate
 * WANTS to try. That question was already answered by observation: a candidate
 * on an ordinary task searched the filesystem for the harness unprompted.
 */

const testToolSignal = new AbortController().signal

async function mount(): Promise<Context> {
  const ctx = new Context()
  await ctx.plugin(SystemPrompt, {})
  await ctx.plugin(ToolRuntime)
  ctx.provide('subagents', { start: () => Promise.reject(new Error('unused')) } as never)
  await ctx.plugin(EvoSpike)
  return ctx
}

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
    execute: (): Promise<string> => Promise.resolve(`RAN:${name}`),
  }
}

async function attempt(
  ctx: Context,
  name: string,
  agent?: Agent,
): Promise<string> {
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

/** Every capability the plan names as one a candidate must not reach. */
const FORBIDDEN = [
  ['mount a plugin', 'cordis_define'],
  ['start a plugin', 'cordis_run'],
  ['stop a plugin', 'cordis_stop'],
  ['unmount a plugin', 'cordis_undefine'],
  ['search the web', 'web_search'],
  ['fetch a url', 'web_fetch'],
  ['interrogate the operator', 'ask_user_question'],
] as const

describe('a candidate cannot reach a forbidden capability', () => {
  // The candidate is its runtime's ROOT agent, so depth 0 is the case that
  // matters most; the others are the descendants it could delegate to.
  it.each(FORBIDDEN)('refuses to %s', async (_label, tool) => {
    const ctx = await mount()
    ctx.tools.register(stubTool(tool))

    for (const agent of [
      agentAtDepth('candidate', 0),
      agentAtDepth('child', 1),
      agentAtDepth('grandchild', 2),
      undefined,
    ]) {
      const outcome = await attempt(ctx, tool, agent)
      expect(outcome).not.toContain(`RAN:${tool}`)
      expect(outcome).toContain('not available in a candidate runtime')
    }
  })

  it('leaves the capabilities a candidate needs alone', async () => {
    // The liveness control. A guard that denied everything would satisfy every
    // case above while making the runtime useless, and nothing here would say
    // so.
    const ctx = await mount()
    for (const tool of ['bash', 'str_replace_editor', 'evo_inspect_candidate']) {
      ctx.tools.register(stubTool(tool))
      expect(await attempt(ctx, tool, agentAtDepth('candidate', 0)))
        .toBe(`RAN:${tool}`)
    }
  })
})

describe('the denial cannot be argued out of', () => {
  it('survives a pre-execute listener that force-allows everything', async () => {
    // The strongest force-allow the seam admits: prepended ahead of every
    // other policy and never calling `next()`, so it short-circuits the whole
    // waterfall. The guard runs after the waterfall regardless.
    const ctx = await mount()
    ctx.tools.register(stubTool('cordis_define'))
    ctx.on(
      'tools/pre-execute',
      (): Promise<PreToolDecision> => Promise.resolve({ kind: 'allow' }),
      { prepend: true },
    )

    expect(await attempt(ctx, 'cordis_define', agentAtDepth('candidate', 0)))
      .toContain('not available in a candidate runtime')
  })

  it('survives a listener registered after the guard', async () => {
    const ctx = await mount()
    ctx.tools.register(stubTool('cordis_run'))
    ctx.on(
      'tools/pre-execute',
      (): Promise<PreToolDecision> => Promise.resolve({ kind: 'allow' }),
    )

    expect(await attempt(ctx, 'cordis_run', agentAtDepth('candidate', 0)))
      .toContain('not available in a candidate runtime')
  })

  it('denies a capability registered after the guard was installed', async () => {
    // Order of mounting must not decide the answer: a plugin loaded later
    // cannot introduce a forbidden name the guard never sees.
    const ctx = await mount()
    ctx.tools.register(stubTool('web_fetch'))

    expect(await attempt(ctx, 'web_fetch', agentAtDepth('candidate', 0)))
      .toContain('not available in a candidate runtime')
  })
})

describe('what the guard does not cover', () => {
  it('does not stop a permitted tool from doing anything the host can', async () => {
    // Named as a test rather than a comment so it cannot quietly stop being
    // true. The guard is a name list over tool calls; `bash` is not on it and
    // must not be, since a candidate edits its workspace with it. Everything
    // reachable through bash — the network, the filesystem outside the
    // workspace, other processes — is therefore outside this mechanism, and
    // dsh's sandbox seam states its own scope the same way: "File effects are
    // the whole policy vocabulary — the seam expresses no network, process,
    // syscall, device, or credential restrictions."
    const ctx = await mount()
    ctx.tools.register(stubTool('bash'))

    expect(await attempt(ctx, 'bash', agentAtDepth('candidate', 0)))
      .toBe('RAN:bash')
  })
})
