#!/usr/bin/env node
/**
 * Materialize an evo-harness Cordis patch for a source checkout.
 *
 * Patch files are composed into a selected dsh profile and therefore resolve
 * modules from that profile, not from the patch file's directory. Source
 * configs keep portable relative specifiers; this script replaces only their
 * `name:` values with checkout-local absolute entrypoints.
 */

import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const [sourceArg, outputArg] = process.argv.slice(2)
if (sourceArg === undefined || outputArg === undefined || process.argv.length !== 4) {
  process.stderr.write('usage: render-config.mjs <source.cordis.yml> <output.cordis.yml>\n')
  process.exit(2)
}

const source = resolve(sourceArg)
const output = resolve(outputArg)
if (source === output) {
  process.stderr.write('render-config.mjs: source and output must be different files\n')
  process.exit(2)
}

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const entrypoints = ['index', 'peer', 'host', 'start', 'proof']
let rendered = readFileSync(source, 'utf8')
let replacements = 0

for (const entrypoint of entrypoints) {
  const modulePath = join(packageRoot, 'src', `${entrypoint}.ts`)
  let specifier = relative(dirname(source), modulePath).split(sep).join('/')
  if (!specifier.startsWith('.')) specifier = `./${specifier}`
  const escaped = specifier.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const nameRow = new RegExp(
    `^(\\s*name:\\s*)["']${escaped}["'][ \\t]*$`,
    'gm',
  )
  rendered = rendered.replace(nameRow, (_row, prefix) => {
    if (!existsSync(modulePath)) {
      throw new Error(`plugin entrypoint does not exist: ${modulePath}`)
    }
    replacements += 1
    return `${prefix}${JSON.stringify(modulePath)}`
  })
}

if (replacements === 0) {
  process.stderr.write(
    `render-config.mjs: ${source} contains no relative evo-harness plugin name rows\n`,
  )
  process.exit(1)
}

writeFileSync(output, rendered, 'utf8')
