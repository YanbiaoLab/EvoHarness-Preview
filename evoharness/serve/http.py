# serve/http.py
# Transport shell for EvalService (protocol §3). Zero domain logic lives here:
# routing, JSON (de)serialization, and exception -> status translation only.
"""Stdlib HTTP shell: three endpoints over an EvalService."""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .service import EvalService

_JOB_PATH = re.compile(r"^/v1/evaluations/([\w-]+)$")


class EvalHTTPServer(ThreadingHTTPServer):
    """Carries the service instance; handlers are per-request and stateless."""

    daemon_threads = True

    def __init__(self, address, service: EvalService, token: str | None = None):
        super().__init__(address, _Handler)
        self.service = service
        self.token = token


class _Handler(BaseHTTPRequestHandler):
    server: EvalHTTPServer  # narrow the type for readability

    def log_message(self, fmt, *args):  # keep pytest output clean
        pass

    def _send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        if self.server.token is None:
            return True
        return self.headers.get("Authorization") == f"Bearer {self.server.token}"

    def do_GET(self):
        if not self._authorized():
            return self._send_json(401, {"error": "missing or bad bearer token"})
        if self.path == "/v1/meta":
            return self._send_json(200, self.server.service.meta())
        m = _JOB_PATH.match(self.path)
        if m:
            try:
                return self._send_json(200, self.server.service.poll(m.group(1)))
            except KeyError:
                return self._send_json(404, {"error": f"unknown job: {m.group(1)}"})
        self._send_json(404, {"error": f"no route: {self.path}"})

    def do_POST(self):
        if not self._authorized():
            return self._send_json(401, {"error": "missing or bad bearer token"})
        if self.path != "/v1/evaluations":
            return self._send_json(404, {"error": f"no route: {self.path}"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
        except ValueError:  # includes json.JSONDecodeError
            return self._send_json(400, {"error": "body is not valid JSON"})
        try:
            job_id = self.server.service.submit(payload)
        except ValueError as exc:
            return self._send_json(400, {"error": str(exc)})
        self._send_json(202, {"job_id": job_id})  # fresh AND replay: same 202


def serve(
    service: EvalService,
    host: str = "127.0.0.1",
    port: int = 8321,
    token: str | None = None,
) -> EvalHTTPServer:
    """Bind and return the server; caller drives serve_forever()/shutdown()."""
    return EvalHTTPServer((host, port), service, token)
