"""Tiny health service: proves Postgres schema readiness without a GPU/model."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from postgres_repository import StateRepository


repository = StateRepository(os.environ["DATABASE_URL"])


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path != "/healthz":
            self.send_error(404)
            return
        try:
            repository.initialize()
        except Exception as error:  # noqa: BLE001 - convert readiness failure to HTTP 503
            body = json.dumps({"ready": False, "error": str(error)}).encode()
            self.send_response(503)
        else:
            body = b'{"ready":true}'
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
