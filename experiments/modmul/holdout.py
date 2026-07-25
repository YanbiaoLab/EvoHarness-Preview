# modmul/holdout.py — 一次性生成私有种子测试集（可选，但强烈建议跑）
#
# 为什么要它：fitness 用的是公开基准的固定前 n 题，进化会朝它过拟合。
# IMO 那轮的教训（memory imo-evolution-results：val 0.595 -> test 0.389）说明
# 只看一个可见集会把"运气"当"能力"。这个脚本用**另一个 master seed**、
# 官方同一套生成器造一份结构相同但样本不同的集合，grade 在 R2 把它的分数
# 回填进 hidden_metrics —— 只做过拟合预警，不进 fitness（否则它就不再是隐藏集）。
#
# 用法（在仓库根，评测机上执行一次）：
#     python -m modmul.holdout --cases 20
#     python -m modmul.holdout --cases 20 --seed-hex <另一个种子>   # 冠军复测用
#
# 依赖 sympy（仅 harness 侧造素数；候选侧仍然禁止 import sympy）。
# 高层素数（1024-2048 bit）用 sympy.nextprime 找，耗时以分钟计，属正常。

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[1] / "third_party" / "modular-arithmetic-challenge" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_SEED_PHRASE = "modmul-round1-holdout-v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=20,
                        help="cases per tier (grade reads the first 20)")
    parser.add_argument("--seed-hex", default=None,
                        help="32-byte master seed in hex; default is derived "
                             f"from {DEFAULT_SEED_PHRASE!r}")
    parser.add_argument("--out", type=Path, default=_HERE / ".holdout")
    parser.add_argument("--tiers", type=int, nargs="*", default=None,
                        help="restrict to these tiers (default: 1..10)")
    args = parser.parse_args(argv)

    # `_generate_tier_cases` is the official per-tier generator that
    # generate_private_test_set() itself calls. Using it directly (instead of
    # the whole-set entry point) keeps tier 0 — whose 2048-bit Mersenne primes
    # are expensive — out of the way and lets --tiers restrict the work.
    # Pinned to the vendored third_party commit; re-check after a clone update.
    from modchallenge.config import TIERS, EvalConfig
    from modchallenge.testgen.generator import _generate_tier_cases

    seed = (
        bytes.fromhex(args.seed_hex)
        if args.seed_hex
        else hashlib.sha256(DEFAULT_SEED_PHRASE.encode()).digest()
    )
    wanted = set(args.tiers) if args.tiers else set(range(1, 11))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SEED.txt").write_text(seed.hex() + "\n")

    defaults = EvalConfig()
    for tier in TIERS:
        if tier.tier_id not in wanted or tier.is_multiplication_only:
            continue
        cases = _generate_tier_cases(
            tier=tier,
            num_cases=args.cases,
            num_primes=defaults.primes_per_tier,
            edge_cases=defaults.edge_cases_per_tier,
            seed=seed,
        )
        path = args.out / f"tier_{tier.tier_id}.jsonl"
        path.write_text(
            "\n".join(json.dumps(c.to_full_dict()) for c in cases) + "\n"
        )
        print(f"tier {tier.tier_id}: {len(cases)} cases -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
