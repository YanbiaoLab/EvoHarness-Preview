# EvoHarness original (webui_design.md): stdlib-only console server —
# read-only JSON API + SSE over the run directories; the engine never knows
# the web exists. No FastAPI dependency at this scale.
"""python -m evoharness.evoweb <results_root> [--port 7861]"""

from __future__ import annotations

import json
import mimetypes
import re
import shutil
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from evoharness.evoplus.directives import append_directive

from .data import (candidate_detail, checkpoint_mtime, decide_proposal,
                   insights, list_runs, proposal_patch, run_detail)

STATIC = Path(__file__).parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
KNOWN_RECIPES = ("b0", "e0", "e1", "e2", "e3g", "e3r")


def make_handler(root: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _json(self, payload, status=200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _run_dir(self, name: str) -> Path | None:
            run_dir = (root / name).resolve()
            if root.resolve() not in run_dir.parents or not (
                run_dir / "run.db"
            ).exists():
                return None
            return run_dir

        def _static(self, rel: str):
            target = (STATIC / rel).resolve()
            if STATIC.resolve() not in target.parents or not target.is_file():
                return self._json({"error": "not found"}, 404)
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 (http.server API)
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            if not parts:
                return self._static("index.html")
            if parts[0] != "api":
                return self._static("/".join(parts))
            if parts[1:] == ["runs"]:
                return self._json(list_runs(root))
            if len(parts) >= 3 and parts[1] == "runs":
                run_dir = self._run_dir(parts[2])
                if run_dir is None:
                    return self._json({"error": "unknown run"}, 404)
                if len(parts) == 3:
                    return self._json(run_detail(run_dir))
                if len(parts) == 5 and parts[3] == "candidates":
                    detail = candidate_detail(run_dir, parts[4])
                    if detail is None:
                        return self._json({"error": "unknown candidate"}, 404)
                    return self._json(detail)
                if len(parts) == 4 and parts[3] == "events":
                    return self._sse(run_dir)
                if len(parts) == 4 and parts[3] == "insights":
                    return self._json(insights(run_dir))
                if len(parts) == 6 and parts[3] == "proposals" and parts[5] == "patch":
                    if not RUN_NAME.match(parts[4]):
                        return self._json({"error": "bad proposal id"}, 400)
                    patch = proposal_patch(run_dir, parts[4])
                    if patch is None:
                        return self._json({"error": "no patch"}, 404)
                    body = patch.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    return self.wfile.write(body)
                if len(parts) == 4 and parts[3] == "directives":
                    ctl = run_dir / "control" / "directives.json"
                    if ctl.exists():
                        return self._json(json.loads(ctl.read_text()))
                    return self._json({"version": 0, "directives": []})
            return self._json({"error": "not found"}, 404)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length) or b"{}")

        def do_POST(self):  # noqa: N802 (http.server API)
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            if parts == ["api", "runs"]:
                return self._create_run(self._body())
            if len(parts) == 6 and parts[:2] == ["api", "runs"] and \
                    parts[3] == "proposals" and parts[5] == "decision":
                run_dir = self._run_dir(parts[2])
                if run_dir is None:
                    return self._json({"error": "unknown run"}, 404)
                if not RUN_NAME.match(parts[4]):
                    return self._json({"error": "bad proposal id"}, 400)
                body = self._body()
                try:
                    updated = decide_proposal(
                        run_dir, parts[4], body.get("action", ""),
                        body.get("reason"),
                    )
                except ValueError as e:
                    return self._json({"error": str(e)}, 400)
                if updated is None:
                    return self._json({"error": "unknown proposal"}, 404)
                return self._json(updated)
            if len(parts) == 4 and parts[:2] == ["api", "runs"] and \
                    parts[3] == "directives":
                run_dir = self._run_dir(parts[2])
                if run_dir is None:
                    return self._json({"error": "unknown run"}, 404)
                try:
                    stored = append_directive(
                        run_dir / "control" / "directives.json", self._body()
                    )
                except (ValueError, KeyError) as e:
                    return self._json({"error": str(e)}, 400)
                return self._json({"stored": stored})
            return self._json({"error": "not found"}, 404)

        def _create_run(self, body: dict):
            name = body.get("name", "")
            recipe = body.get("recipe", "e3r")
            task = body.get("task", "demo_counter")
            if not RUN_NAME.match(name):
                return self._json(
                    {"error": "run name must match [A-Za-z0-9_-]{1,64}"}, 400
                )
            if recipe not in KNOWN_RECIPES:
                return self._json({"error": f"unknown recipe {recipe}"}, 400)
            if (root / name).exists():
                return self._json({"error": f"run {name!r} already exists"}, 409)
            cmd = [
                sys.executable, "-m", "experiments.run_evolution",
                "--recipe", recipe, "--task", task,
                "--run-dir", str(root / name),
            ]
            overrides = []
            if body.get("generations"):
                overrides.append(f"search.num_generations={int(body['generations'])}")
            if body.get("model"):
                overrides.append(f'search.llm_models=["{body["model"]}"]')
            if overrides:
                cmd += ["--set", *overrides]
            if body.get("live"):
                cmd.append("--live")
            log = open(root / f"{name}.launch.log", "w")  # noqa: SIM115
            subprocess.Popen(
                cmd, cwd=str(REPO_ROOT), stdout=log, stderr=log,
                start_new_session=True,
            )
            return self._json({"started": name})

        def do_DELETE(self):  # noqa: N802 (http.server API)
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            if len(parts) == 3 and parts[:2] == ["api", "runs"]:
                run_dir = self._run_dir(parts[2])
                if run_dir is None:
                    return self._json({"error": "unknown run"}, 404)
                shutil.rmtree(run_dir)
                return self._json({"deleted": parts[2]})
            return self._json({"error": "not found"}, 404)

        def _sse(self, run_dir: Path):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last = checkpoint_mtime(run_dir)
            try:
                while True:
                    time.sleep(1.0)
                    now = checkpoint_mtime(run_dir)
                    if now != last:
                        last = now
                        self.wfile.write(b"event: update\ndata: {}\n\n")
                    else:
                        self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def serve(root: Path, port: int = 7861) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(root))
    print(f"evoweb console: http://127.0.0.1:{port}  (root: {root})")
    server.serve_forever()
