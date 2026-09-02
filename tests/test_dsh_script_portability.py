"""Regression checks for the two-checkout DSH launch seam."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    ["scripts/dsh_proof.sh", "scripts/install_dsh_presets.sh", "scripts/try_dsh_backend.py"],
)
def test_launch_helpers_do_not_pin_a_personal_checkout(relative_path):
    source = (ROOT / relative_path).read_text(encoding="utf-8")

    assert "/Users/" not in source
    assert "/home/" not in source
    assert "Documents/Projects/deepseek-harness" not in source


SHELL_HELPERS = ["scripts/dsh_proof.sh", "scripts/install_dsh_presets.sh"]


@pytest.mark.parametrize("relative_path", SHELL_HELPERS)
def test_shell_helpers_are_valid_bash(relative_path):
    subprocess.run(
        ["bash", "-n", str(ROOT / relative_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_the_installer_has_one_explicit_checkout_override():
    """One place says where the other checkout is, and every path derives.

    Two overrides drift: a person who moved dsh sets the one they read about
    and leaves the other pointing into a directory that no longer exists,
    which surfaces as a plugin failing to import rather than as a wrong path.
    """
    source = (ROOT / "scripts/install_dsh_presets.sh").read_text(encoding="utf-8")

    assert '${DSH_ROOT:-$HARNESS_ROOT/../deepseek-harness}' in source
    assert 'EVO_DSH_CONFIG="${EVO_DSH_CONFIG:-' in source
    assert 'EVO_DSH_RUNTIME="${EVO_DSH_RUNTIME:-' in source


LAUNCHERS = ["scripts/dsh_proof.sh"]


@pytest.mark.parametrize("relative_path", LAUNCHERS)
def test_a_launcher_never_relabels_a_key_as_a_vendor_credential(relative_path):
    """A key may be adopted from the old name, never published back into it.

    The direction is the whole rule. `ALIYUN_MAAS_API_KEY` names a vendor, and
    a credential copied into it is announced as belonging to a gateway it may
    have nothing to do with — after which any config reading that name spends
    against one endpoint while describing another.
    """
    source = (ROOT / relative_path).read_text(encoding="utf-8")

    assert 'export ALIYUN_MAAS_API_KEY="$EVOHARNESS_API_KEY"' not in source
    assert 'export EVOHARNESS_API_KEY="$ALIYUN_MAAS_API_KEY"' in source


@pytest.mark.parametrize("relative_path", LAUNCHERS)
def test_a_launcher_requires_the_project_credential_name(relative_path):
    """What a launcher refuses to start without is the name everything reads.

    Demanding the legacy name instead would reject a machine that is correctly
    configured, and the adoption above makes the check equivalent for one that
    is not — so this is which name a person is told to set.
    """
    source = (ROOT / relative_path).read_text(encoding="utf-8")

    assert "EVOHARNESS_API_KEY is not set" in source or (
        'if [ -z "${EVOHARNESS_API_KEY:-}" ]; then' in source
    )


def _credential_names(source: str) -> list[str]:
    """Every variable an `apiKeyEnv:` row declares, ignoring prose.

    Matching the declaration rather than the whole text is deliberate: a
    comment explaining which name was retired has to be allowed to write it
    down, and a test that forbids the string forbids the explanation.
    """
    return re.findall(r"^\s*apiKeyEnv:\s*(\S+)\s*$", source, re.MULTILINE)


def test_the_proof_session_config_names_one_credential():
    """The config EvoHarness owns must name the same credential as its CLI.

    `evoharness.proof.cli` reads `EVOHARNESS_API_KEY` and nothing else, so a
    session config declaring a second variable produces the failure this whole
    unification is about: the tools spend against one gateway and the
    conversation against another, and nothing reports the split.
    """
    source = (ROOT / "integrations/dsh/package/fixtures/host.proof.cordis.yml").read_text(
        encoding="utf-8"
    )

    assert _credential_names(source) == ["EVOHARNESS_API_KEY"]


def test_the_proof_preset_declares_no_credential_at_all():
    """A preset composes an agent; the model route belongs to the deployment.

    Declaring one here would put a route — and the credential reference that
    goes with it — in a file copied into a person's harness home, where it
    would shadow whatever the deployment configured for every other mode.
    """
    source = (ROOT / "integrations/dsh/presets/proof/agent.cordis.yml").read_text(
        encoding="utf-8"
    )

    assert _credential_names(source) == []


@pytest.mark.parametrize("relative_path", LAUNCHERS + ["scripts/install_dsh_presets.sh"])
def test_a_launcher_reads_no_relocated_env_file(relative_path):
    """No launcher may depend on another checkout's `.env` having been renamed.

    The file it used to source existed because dsh refuses to boot when a
    discovered `.env` declares a bootstrap-only name, so someone moved dsh's
    own `.env` aside and taught these scripts the new name. That leaves a
    checkout in a state only a script explains, under a filename ending in
    `.bak` that reads as deletable — and after the credential unification the
    values it carried had no reader here at all.
    """
    source = (ROOT / relative_path).read_text(encoding="utf-8")

    assert ".env.moved-by-demo" not in source
    assert ".bak" not in source
