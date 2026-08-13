# EvoHarness original guardrail (plan section 7: holdout isolation and
# static scanning of every candidate program).
"""Static scanner for candidate programs: network, process escape, holdout
access, dynamic code execution."""

from __future__ import annotations

import ast
import fnmatch
from dataclasses import dataclass


@dataclass
class Finding:
    rule: str
    lineno: int
    detail: str
    path: str | None = None


DEFAULT_BANNED_IMPORTS = frozenset(
    {
        "socket", "requests", "urllib", "urllib3", "http", "httpx", "aiohttp",
        "ftplib", "smtplib", "telnetlib", "paramiko", "websocket", "websockets",
        "subprocess", "ctypes", "importlib", "pty",
    }
)

_BANNED_CALLS = frozenset({"eval", "exec", "compile", "__import__"})
_BANNED_OS_ATTRS = frozenset(
    {"system", "popen", "execv", "execve", "execvp", "spawnl", "fork", "kill"}
)


class AntiHackScanner:
    def __init__(
        self,
        holdout_globs: tuple[str, ...] = ("*holdout*",),
        banned_imports: frozenset[str] = DEFAULT_BANNED_IMPORTS,
    ):
        self.holdout_globs = holdout_globs
        self.banned_imports = banned_imports

    def scan(self, code: str) -> list[Finding]:
        findings: list[Finding] = []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [Finding("syntax-error", e.lineno or 0, str(e))]

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in self.banned_imports:
                        findings.append(
                            Finding("banned-import", node.lineno, alias.name)
                        )
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in self.banned_imports:
                    findings.append(
                        Finding("banned-import", node.lineno, node.module or "")
                    )
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in _BANNED_CALLS:
                    findings.append(
                        Finding("dynamic-exec", node.lineno, func.id)
                    )
                elif (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "os"
                    and func.attr in _BANNED_OS_ATTRS
                ):
                    findings.append(
                        Finding("process-escape", node.lineno, f"os.{func.attr}")
                    )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                for pattern in self.holdout_globs:
                    if fnmatch.fnmatch(node.value.lower(), pattern.lower()):
                        findings.append(
                            Finding("holdout-access", node.lineno, node.value)
                        )
                        break
        return findings
    
    def scan_files(self, texts: dict[str, str]) -> list[Finding]:
        findings: list[Finding] = []
        for path in sorted(texts):
            for pattern in self.holdout_globs:
                if fnmatch.fnmatch(path.lower(), pattern.lower()):
                    findings.append(
                        Finding("holdout-access", 0, path, path)
                    )
                    break

            if path.endswith(".py"):
                findings.extend(
                    Finding(f.rule, f.lineno, f.detail, path)
                    for f in self.scan(texts[path])
                )

        return findings
        
