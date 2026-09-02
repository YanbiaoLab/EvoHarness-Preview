/**
 * This package is a copy. EvoHarness holds the original at
 * `integrations/dsh/package/`, and `scripts/install_dsh_presets.sh` copies it
 * here.
 *
 * A copy rather than a link, because of Node's resolution: every plugin here
 * imports `@deepseek-ai/dsh-tools`, resolved by walking up from the file's
 * REAL path, and that walk reaches dsh's dependencies only from inside this
 * package. A symlink resolves to its target first, so it does not help.
 *
 * This spec keeps the copy honest, and runs here rather than beside the
 * original on purpose: this is where the suite runs, so this is where a person
 * already has a terminal open and edits the wrong file.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

const packageRoot = join(dirname(fileURLToPath(import.meta.url)), '..')

/** Produced by the dsh toolchain, not authored; the mirror does not carry them. */
const GENERATED = new Set(['node_modules', 'lib'])

/**
 * The authored copy, or undefined when this machine has no EvoHarness beside
 * the dsh checkout.
 *
 * `EVO_HARNESS_ROOT` first — the same variable every plugin here reads, so a
 * configured machine needs nothing extra — then the conventional layout of the
 * two checkouts side by side.
 */
function mirrorRoot(): string | undefined {
  const candidates = [
    process.env.EVO_HARNESS_ROOT,
    join(packageRoot, '..', '..', '..', '..', 'EvoHarness'),
  ]
  for (const candidate of candidates) {
    if (candidate === undefined || candidate === '') continue
    const mirror = join(candidate, 'integrations', 'dsh', 'package')
    try {
      if (statSync(mirror).isDirectory()) return mirror
    } catch {
      // Not this one. A missing checkout is the common case for the first
      // candidate and is not itself the failure — running out of them is.
    }
  }
  return undefined
}

/** Every authored file under `root`, as paths relative to it. */
function authoredFiles(root: string, at = ''): string[] {
  const found: string[] = []
  for (const entry of readdirSync(join(root, at), { withFileTypes: true })) {
    if (GENERATED.has(entry.name)) continue
    const rel = at === '' ? entry.name : join(at, entry.name)
    if (entry.isDirectory()) found.push(...authoredFiles(root, rel))
    else found.push(rel)
  }
  return found.sort()
}

describe('this package is a copy of the one EvoHarness authors', () => {
  const mirror = mirrorRoot()

  it('can find the original', () => {
    // Deliberately a failure and not a skip. A mirror check that passes
    // quietly when it cannot see the original reports green for exactly the
    // situation it exists to catch.
    expect(
      mirror,
      'no EvoHarness checkout found: set EVO_HARNESS_ROOT, or place it beside '
      + 'the dsh checkout. This suite cannot tell whether these sources are '
      + 'the authored ones without it.',
    ).toBeDefined()
  })

  it('holds the same files as the original', () => {
    if (mirror === undefined) return
    // Names before contents: a file added on one side only is a different
    // mistake from one edited on both, and the message should say which.
    expect(authoredFiles(join(packageRoot)))
      .toEqual(authoredFiles(mirror))
  })

  it('holds them byte for byte', () => {
    if (mirror === undefined) return
    const differing = authoredFiles(mirror).filter((rel) => {
      try {
        return readFileSync(join(packageRoot, rel), 'utf8')
          !== readFileSync(join(mirror, rel), 'utf8')
      } catch {
        // Absent here; the file-list case above already reports it.
        return false
      }
    })

    expect(
      differing,
      `edited in the copy rather than in the original. Move the change to `
      + `${relative(process.cwd(), mirror)} and re-run `
      + 'EvoHarness\'s scripts/install_dsh_presets.sh.',
    ).toEqual([])
  })
})
