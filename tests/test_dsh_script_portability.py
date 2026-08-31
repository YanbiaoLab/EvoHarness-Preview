"""Regression checks for the two-checkout DSH launch seam."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    ["scripts/dsh_demo.sh", "scripts/try_dsh_backend.py"],
)
def test_launch_helpers_do_not_pin_a_personal_checkout(relative_path):
    source = (ROOT / relative_path).read_text(encoding="utf-8")

    assert "/Users/" not in source
    assert "/home/" not in source
    assert "Documents/Projects/deepseek-harness" not in source


def test_demo_launcher_is_valid_bash():
    subprocess.run(
        ["bash", "-n", str(ROOT / "scripts/dsh_demo.sh")],
        check=True,
        capture_output=True,
        text=True,
    )


def test_demo_launcher_has_one_explicit_checkout_override():
    source = (ROOT / "scripts/dsh_demo.sh").read_text(encoding="utf-8")

    assert '${DSH_ROOT:-$HARNESS_ROOT/../deepseek-harness}' in source
    assert 'export EVO_DSH_CONFIG="${EVO_DSH_CONFIG:-' in source
    assert 'export EVO_DSH_RUNTIME="${EVO_DSH_RUNTIME:-' in source


def test_demo_launcher_never_reinterprets_a_legacy_key_as_an_aliyun_key():
    source = (ROOT / "scripts/dsh_demo.sh").read_text(encoding="utf-8")

    assert 'export ALIYUN_MAAS_API_KEY="$EVOHARNESS_API_KEY"' not in source
    assert 'export EVOHARNESS_API_KEY="$ALIYUN_MAAS_API_KEY"' in source
