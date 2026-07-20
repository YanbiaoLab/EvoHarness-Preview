"""Layering rules: generic layers must never import specific ones.

evoharness/ is the framework main package; modmul/tasks/recipes/experiments
are consumer packages that import DOWNWARD only. Within the framework:
evoguard = generic sandbox infra; evoserve = protocol-standalone (§8).
"""

from pathlib import Path

# scan-dir (repo-relative) -> import roots it must never mention
FORBIDDEN = {
    "evoharness/evoguard": (
        "evoharness.evocore", "evoharness.evoplus", "evoharness.evoserve",
        "tasks", "recipes", "modmul",
    ),
    "evoharness/evoserve": (
        "evoharness.evocore", "evoharness.evoplus", "evoharness.evoguard",
        "tasks", "recipes", "modmul",
    ),
    # 引擎永不 import 任务/装配层(依赖方向只准从消费者指向框架)
    "evoharness/evocore": ("modmul", "tasks", "recipes", "evoharness.evoplus"),
}


def test_generic_layers_import_no_specific_layers():
    root = Path(__file__).resolve().parent.parent
    for pkg, banned in FORBIDDEN.items():
        for py in (root / pkg).rglob("*.py"):
            src = py.read_text()
            for target in banned:
                assert (
                    f"import {target}" not in src and f"from {target}" not in src
                ), f"{py.relative_to(root)} imports {target}: layering violation"
