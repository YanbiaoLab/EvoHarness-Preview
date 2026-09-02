import { execFileSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

const packageRoot = fileURLToPath(new URL('../', import.meta.url))
const fixtures = join(packageRoot, 'fixtures')
const renderer = join(packageRoot, 'scripts', 'render-config.mjs')

function withoutPluginNames(source: string): string {
  return source.replace(/^(\s*name:).*$/gm, '$1')
}

describe('portable evo-harness Cordis configs', () => {
  it('routes editor mutations through the workspace-write filesystem', () => {
    const candidate = readFileSync(join(fixtures, 'candidate.cordis.yml'), 'utf8')

    expect(candidate).toContain("name: '@deepseek-ai/dsh-fs-sandbox'")
    expect(candidate).not.toContain("name: '@deepseek-ai/dsh-fs-local'")
    expect(candidate).toContain('mode: workspace-write')
  })

  it('keeps candidate identity independent of the checkout location', () => {
    const candidate = readFileSync(join(fixtures, 'candidate.cordis.yml'), 'utf8')

    expect(candidate).toContain('name: \'../src/index.ts\'')
    expect(candidate).toContain('name: \'../src/peer.ts\'')
    expect(candidate).not.toMatch(/\/Users\/|\/home\/[^/]+/)
  })

  it('names one credential, after a vendor, and stores none', () => {
    const candidate = readFileSync(join(fixtures, 'candidate.cordis.yml'), 'utf8')

    // `EVOHARNESS_API_KEY` is the single name every config here, the proof
    // CLI and EvoHarness's launch builder read. A credential named after a
    // vendor became wrong the moment the endpoint moved: the variable still
    // resolved, so the config went on describing a provider it no longer
    // reached, and the mismatch surfaced as an authorization error.
    expect(candidate).toContain('apiKeyEnv: EVOHARNESS_API_KEY')
    expect(candidate).not.toContain('ALIYUN_MAAS_API_KEY')
    // The default endpoint is unchanged and is still Aliyun's; what moved is
    // which name the key is kept under, not where an unconfigured run goes.
    expect(candidate).toContain(
      'https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',
    )
    expect(candidate).not.toMatch(/sk-[A-Za-z0-9_.-]+/)
  })

  it('materializes only plugin name rows for a source-checkout patch', () => {
    // `host.cordis.yml` rather than the deleted demo config: the renderer is
    // still what `dsh_proof.sh` runs on a patch file, and this fixture is the
    // tracked one carrying a relative plugin row. Row COUNT is not what is
    // under test — that every other byte survives is.
    const dir = mkdtempSync(join(tmpdir(), 'evo-cordis-portable-'))
    const sourcePath = join(fixtures, 'host.cordis.yml')
    const outputPath = join(dir, 'rendered config.yml')
    try {
      execFileSync(process.execPath, [renderer, sourcePath, outputPath])
      const source = readFileSync(sourcePath, 'utf8')
      const rendered = readFileSync(outputPath, 'utf8')

      expect(rendered).toContain(join(packageRoot, 'src', 'host.ts'))
      expect(rendered).not.toContain("name: '../src/host.ts'")
      expect(withoutPluginNames(rendered)).toBe(withoutPluginNames(source))
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('stores no personal absolute path in any trust-boundary config', () => {
    const configs = [
      join(packageRoot, 'cordis.yml'),
      join(fixtures, 'candidate.cordis.yml'),
      join(fixtures, 'host.cordis.yml'),
    ]

    for (const config of configs) {
      expect(readFileSync(config, 'utf8')).not.toMatch(/\/Users\/|\/home\/[^/]+/)
    }
  })
})
