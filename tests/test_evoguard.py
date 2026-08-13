import sys

import pytest

from evoharness.guard import (
    AntiHackScanner,
    BudgetExhausted,
    BudgetMeter,
    Sandbox,
    sha256_file,
    write_manifest,
)


def test_budget_cap_and_persistence(tmp_path):
    state = tmp_path / "budget.json"
    meter = BudgetMeter(hard_cap_usd=1.0, state_path=state)
    meter.charge(0.4)
    assert not meter.should_stop()
    assert meter.remaining() == pytest.approx(0.6)
    meter.charge(0.7)
    assert meter.should_stop()
    # resumes from persisted state
    resumed = BudgetMeter(hard_cap_usd=1.0, state_path=state)
    assert resumed.spent_usd == pytest.approx(1.1)
    assert resumed.should_stop()


def test_budget_strict_raises():
    meter = BudgetMeter(hard_cap_usd=0.5, strict=True)
    with pytest.raises(BudgetExhausted):
        meter.charge(0.6)
    with pytest.raises(ValueError):
        meter.charge(-1)


def test_sandbox_runs_and_captures_output(tmp_path):
    sandbox = Sandbox()
    result = sandbox.run(
        [sys.executable, "-c", "print('hi'); import sys; sys.stderr.write('err')"],
        workdir=tmp_path,
        timeout_s=30,
    )
    assert result.ok
    assert result.stdout.strip() == "hi"
    assert "err" in result.stderr


def test_sandbox_kills_on_timeout(tmp_path):
    sandbox = Sandbox()
    result = sandbox.run(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        workdir=tmp_path,
        timeout_s=1.0,
    )
    assert result.timed_out and not result.ok
    assert result.elapsed_s < 10


def test_sandbox_blocks_proxy_env(tmp_path):
    sandbox = Sandbox(allow_network=False)
    result = sandbox.run(
        [sys.executable, "-c", "import os; print(os.environ['http_proxy'])"],
        workdir=tmp_path,
        timeout_s=30,
    )
    assert "127.0.0.1:9" in result.stdout


def test_antihack_detects_malicious_patterns():
    scanner = AntiHackScanner(holdout_globs=("*holdout*",))
    code = (
        "import socket\n"
        "from urllib import request\n"
        "eval('1+1')\n"
        "import os\n"
        "os.system('rm -rf /')\n"
        "data = open('data/holdout.jsonl').read()\n"
    )
    rules = {f.rule for f in scanner.scan(code)}
    assert rules == {
        "banned-import", "dynamic-exec", "process-escape", "holdout-access",
    }


def test_antihack_passes_clean_code_and_flags_syntax_error():
    scanner = AntiHackScanner()
    assert scanner.scan("import math\nprint(math.pi)\n") == []
    findings = scanner.scan("def broken(:\n")
    assert findings[0].rule == "syntax-error"


def test_manifest_roundtrip(tmp_path):
    from evoharness.core import SearchConfig

    data = tmp_path / "train.jsonl"
    data.write_text('{"x": 1}\n')
    path = tmp_path / "manifest.json"
    manifest = write_manifest(
        path,
        config=SearchConfig(num_generations=5),
        holdout_sha256=sha256_file(data),
    )
    assert path.exists()
    assert manifest["config"]["num_generations"] == 5
    assert len(manifest["holdout_sha256"]) == 64


def test_scan_files_flags_filename_and_tags_path():
    """M2.5: the FILENAME itself is attack surface — a file named after the
    holdout matches even with clean content; findings carry the culprit path."""
    from evoharness.guard import AntiHackScanner

    scanner = AntiHackScanner()
    findings = scanner.scan_files({
        "main.py": "import math\n",              # clean
        "data/holdout_cache.py": "x = 1\n",      # clean content, dirty NAME
        "evil.py": "import socket\n",            # dirty content
    })
    by_rule = {(f.rule, f.path) for f in findings}
    assert ("holdout-access", "data/holdout_cache.py") in by_rule
    assert ("banned-import", "evil.py") in by_rule
    assert not any(f.path == "main.py" for f in findings)  # clean file untouched
