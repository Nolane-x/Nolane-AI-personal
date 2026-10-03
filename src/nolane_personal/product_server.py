from __future__ import annotations

import argparse
import hmac
import json
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .product_runtime import ProductRuntime


MAX_BODY_BYTES = 1024 * 1024


class ProductHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address,
        runtime: ProductRuntime,
        *,
        auth_token: str = "",
    ):
        super().__init__(server_address, ProductHandler)
        self.runtime = runtime
        self.auth_token = str(auth_token)


class ProductHandler(BaseHTTPRequestHandler):
    server: ProductHTTPServer

    def log_message(self, format: str, *args) -> None:
        return

    def _json(
        self,
        status: int,
        payload: Any,
    ) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-site")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = self.server.auth_token
        if not expected:
            return True
        supplied = self.headers.get("X-Nolane-Token", "")
        return hmac.compare_digest(supplied, expected)

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        self._json(
            HTTPStatus.UNAUTHORIZED,
            {"error": "unauthorized"},
        )
        return False

    def _body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        if length == 0:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _dispatch_error(self, exc: Exception) -> None:
        if isinstance(exc, ValueError):
            status = HTTPStatus.BAD_REQUEST
        elif isinstance(exc, RuntimeError):
            status = HTTPStatus.CONFLICT
        elif isinstance(exc, FileNotFoundError):
            status = HTTPStatus.SERVICE_UNAVAILABLE
        else:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
        self._json(
            status,
            {
                "error": type(exc).__name__,
                "message": str(exc),
            },
        )

    def do_GET(self) -> None:
        try:
            if self.path == "/v1/health":
                self._json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "service": "nolane-product-runtime",
                    },
                )
                return
            if not self._require_auth():
                return
            if self.path == "/v1/status":
                self._json(
                    HTTPStatus.OK,
                    self.server.runtime.status(),
                )
                return
            if self.path.startswith("/v1/history"):
                self._json(
                    HTTPStatus.OK,
                    {
                        "messages": self.server.runtime.history(),
                    },
                )
                return
            if self.path == "/v1/profile":
                self._json(
                    HTTPStatus.OK,
                    self.server.runtime.get_profile(),
                )
                return
            self._json(
                HTTPStatus.NOT_FOUND,
                {"error": "not_found"},
            )
        except Exception as exc:
            self._dispatch_error(exc)

    def do_POST(self) -> None:
        if not self._require_auth():
            return
        try:
            payload = self._body()
            if self.path == "/v1/power":
                enabled = payload.get("enabled")
                if not isinstance(enabled, bool):
                    raise ValueError("enabled must be boolean")
                self._json(
                    HTTPStatus.OK,
                    self.server.runtime.set_power(enabled),
                )
                return
            if self.path == "/v1/chat":
                self._json(
                    HTTPStatus.OK,
                    self.server.runtime.send_message(
                        str(payload.get("text", ""))
                    ),
                )
                return
            if self.path == "/v1/tick":
                self._json(
                    HTTPStatus.OK,
                    self.server.runtime.tick(),
                )
                return
            self._json(
                HTTPStatus.NOT_FOUND,
                {"error": "not_found"},
            )
        except Exception as exc:
            self._dispatch_error(exc)

    def do_PUT(self) -> None:
        if not self._require_auth():
            return
        try:
            payload = self._body()
            if self.path == "/v1/profile":
                self._json(
                    HTTPStatus.OK,
                    self.server.runtime.update_profile(payload),
                )
                return
            self._json(
                HTTPStatus.NOT_FOUND,
                {"error": "not_found"},
            )
        except Exception as exc:
            self._dispatch_error(exc)


def _ticker(runtime: ProductRuntime, stop: threading.Event) -> None:
    while not stop.is_set():
        profile = runtime.current_profile()
        delay = {
            "off": 5.0,
            "gentle": 30.0,
            "active": 12.0,
        }.get(profile.initiative, 30.0)
        if stop.wait(delay):
            return
        try:
            runtime.tick()
        except Exception:
            # The request surface owns visible error reporting. Background
            # initiative must never crash the product process.
            continue


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nolane-product-runtime")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=46831)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-bundle", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--ceremony", required=True)
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    p.add_argument("--auth-token", default="")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"} and not args.auth_token:
        raise SystemExit(
            "non-loopback product runtime requires --auth-token"
        )

    runtime = ProductRuntime(
        Path(args.data_dir),
        checkpoint=Path(args.model_bundle),
        tokenizer_path=Path(args.tokenizer),
        device=args.device,
        release_ceremony=Path(args.ceremony),
    )
    server = ProductHTTPServer(
        (args.host, args.port),
        runtime,
        auth_token=args.auth_token,
    )
    stop = threading.Event()
    ticker = threading.Thread(
        target=_ticker,
        args=(runtime, stop),
        daemon=True,
        name="nolane-product-ticker",
    )
    ticker.start()

    def shutdown(*_args) -> None:
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, shutdown)

    host, port = server.server_address[:2]
    print(
        json.dumps(
            {
                "status": "ready",
                "host": host,
                "port": port,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        stop.set()
        runtime.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
