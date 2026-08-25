#!/usr/bin/env python3
"""Serve the pilot SPA and proxy its API to the active Slurm pilot job.

The public Tailscale Funnel mounts this server at ``/pilot``.  Tailscale
versions differ in whether a reverse-proxy mount prefix reaches the backend,
so the handler accepts both stripped paths (``/api/...``) and full paths
(``/pilot/api/...``).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
import subprocess
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class PilotTargetResolver:
    def __init__(self, explicit_url: str | None, job_name: str, cache_seconds: float = 5.0):
        self.explicit_url = explicit_url.rstrip("/") if explicit_url else None
        self.job_name = job_name
        self.cache_seconds = cache_seconds
        self.cached_url: str | None = None
        self.cached_at = 0.0

    def resolve(self) -> str | None:
        if self.explicit_url:
            return self.explicit_url
        now = time.monotonic()
        if now - self.cached_at < self.cache_seconds:
            return self.cached_url
        self.cached_at = now
        self.cached_url = self._resolve_from_slurm()
        return self.cached_url

    def _resolve_from_slurm(self) -> str | None:
        try:
            result = subprocess.run(
                [
                    "squeue",
                    "--noheader",
                    "--name",
                    self.job_name,
                    "--states=RUNNING",
                    "--format=%N",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        for line in result.stdout.splitlines():
            node = line.strip()
            if node and node not in {"(null)", "N/A"}:
                return f"http://{node}:8010"
        return None


class PilotGatewayHandler(BaseHTTPRequestHandler):
    build_root: Path
    resolver: PilotTargetResolver
    server_version = "TEARSPilotGateway/1.0"

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._dispatch()

    def _dispatch(self) -> None:
        path = urlsplit(self.path).path
        # Funnel normally strips the /pilot mount before proxying.  Accept the
        # unstripped form too, but do not mistake /pilot-posters for the mount.
        if path == "/pilot":
            normalized = "/"
        elif path.startswith("/pilot/"):
            normalized = path[len("/pilot") :]
        else:
            normalized = path
        if normalized == "/_gateway/health":
            self._send_gateway_health()
        elif normalized == "/api" or normalized.startswith("/api/"):
            self._proxy_api(normalized)
        elif self.command in {"GET", "HEAD"}:
            self._serve_spa(normalized)
        else:
            self.send_error(HTTPStatus.METHOD_NOT_ALLOWED)

    def _send_gateway_health(self) -> None:
        target = self.resolver.resolve()
        payload = json.dumps(
            {"status": "running", "pilot_api": target or "waiting-for-slurm-job"}
        ).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _proxy_api(self, normalized_path: str) -> None:
        target = self.resolver.resolve()
        if not target:
            self._send_json_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "The pilot model job is waiting for a GPU allocation. Please try again shortly.",
            )
            return

        query = urlsplit(self.path).query
        upstream_url = f"{target}{normalized_path}"
        if query:
            upstream_url = f"{upstream_url}?{query}"
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS | {"host", "content-length"}
        }
        request = Request(upstream_url, data=body, headers=headers, method=self.command)
        try:
            response = urlopen(request, timeout=120)
        except HTTPError as error:
            self._copy_upstream_response(error)
        except (URLError, TimeoutError, OSError) as error:
            self._send_json_error(
                HTTPStatus.BAD_GATEWAY,
                f"The pilot model API is temporarily unavailable: {error}",
            )
        else:
            with response:
                self._copy_upstream_response(response)

    def _copy_upstream_response(self, response) -> None:
        payload = response.read()
        self.send_response(response.status)
        for key, value in response.headers.items():
            if key.lower() not in HOP_BY_HOP_HEADERS | {"content-length"}:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _serve_spa(self, normalized_path: str) -> None:
        relative = normalized_path.lstrip("/")
        candidate = (self.build_root / relative).resolve()
        try:
            candidate.relative_to(self.build_root)
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            candidate = self.build_root / "index.html"
        try:
            payload = candidate.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache" if candidate.name == "index.html" else "public, max-age=3600")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_json_error(self, status: HTTPStatus, message: str) -> None:
        payload = json.dumps({"detail": message}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3010)
    parser.add_argument("--pilot-api-url", default=os.environ.get("PILOT_API_URL"))
    parser.add_argument("--pilot-job-name", default="tears_pilot_api")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_root = args.build_root.resolve()
    if not (build_root / "index.html").is_file():
        raise SystemExit(f"pilot build is missing index.html: {build_root}")
    PilotGatewayHandler.build_root = build_root
    PilotGatewayHandler.resolver = PilotTargetResolver(
        args.pilot_api_url, args.pilot_job_name
    )
    server = ThreadingHTTPServer((args.host, args.port), PilotGatewayHandler)
    print(f"pilot gateway listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
