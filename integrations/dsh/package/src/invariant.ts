/**
 * Package-owned invariant companion for `@deepseek-ai/dsh-evo-spike`.
 * @module @deepseek-ai/dsh-evo-spike/invariant
 */

import type { Context } from '@deepseek-ai/cordis'
import type { InvariantInstaller } from '@deepseek-ai/dsh-invariants'

const PACKAGE_NAME = '@deepseek-ai/dsh-evo-spike'

/** Cordis companion plugin name. */
export const name = 'evo-spike-invariant'
/** Service required before the companion can reserve package ownership. */
export const inject = ['invariants']

/**
 * No runtime invariant: the spike owns no event stream or durable data of its
 * own. Its denial ledger is an in-memory measurement artifact, asserted
 * directly by the package's guard suite.
 */
const install: InvariantInstaller = () => {}

/**
 * Register this package's invariant companion.
 * @param ctx - Cordis context carrying the invariant service.
 * @returns the installed registration's disposer after setup succeeds.
 */
export const apply = (ctx: Context): Promise<() => void> =>
  Promise.resolve(ctx.invariants.register(PACKAGE_NAME, install))
