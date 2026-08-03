"""Layering rules for the framework and experiment packages.

evoharness/ is the framework main package; modmul/tasks/recipes/experiments
are consumer packages that import DOWNWARD only. Within the framework:
evoguard = generic sandbox infra; evoserve = protocol-standalone (§8).
"""

import ast
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

# Comparison adapters may build directly on the domain they compare against.
# Keep the exception explicit so ordinary experiment packages remain isolated.
ALLOWED_EXPERIMENT_DEPS = {
    "hyperagents_imo": {"imo_proof"},
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


def test_experiment_packages_do_not_import_each_other():
    root = Path(__file__).resolve().parent.parent
    experiments = root / "experiments"
    packages = tuple(
        path.name
        for path in experiments.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )

    for package in packages:
        allowed = ALLOWED_EXPERIMENT_DEPS.get(package, set())
        forbidden = {
            f"experiments.{name}"
            for name in packages
            if name != package and name not in allowed
        }
        for path in (experiments / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
            violations = {
                module
                for module in imported
                if any(
                    module == target or module.startswith(target + ".")
                    for target in forbidden
                )
            }
            assert not violations, (
                f"{path.relative_to(root)} imports another experiment: "
                f"{sorted(violations)}"
            )


def test_imo_grade_depends_only_on_evaluation_contract():
    root = Path(__file__).resolve().parent.parent
    grade_source = (root / "experiments/imo_proof/grade.py").read_text()
    contract_source = (
        root / "experiments/imo_proof/evaluation/contract.py"
    ).read_text()
    service_source = (
        root / "experiments/imo_proof/evaluation/service.py"
    ).read_text()

    assert "evaluation.engine" not in grade_source
    assert "evaluator" not in grade_source
    assert "import evoharness" not in contract_source
    assert "from evoharness" not in contract_source
    assert "import evoharness" not in service_source
    assert "from evoharness" not in service_source
