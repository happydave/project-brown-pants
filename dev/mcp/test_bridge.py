#!/usr/bin/env python3
"""Self-test for the WI 639 MCP bridge — proves the path end to end without the windowed game.

Spins a stub HTTP server standing in for the runtime bus, drives `sounding_mcp.py` as a subprocess
over stdio, and asserts the MCP handshake (`initialize` / `tools/list` / `tools/call`) plus that the
two tools actually proxy to `GET /telemetry` and `POST /command`.

Run: ``python3 dev/mcp/test_bridge.py`` (stdlib only; exits non-zero on failure).
"""

import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

BRIDGE = Path(__file__).with_name("sounding_mcp.py")
SHOT_PATH = str(Path(tempfile.gettempdir()) / "sounding-mcp-selftest-shot.png")

received = {"posts": []}


class StubBus(BaseHTTPRequestHandler):
    def log_message(self, *_):  # silence
        pass

    def do_GET(self):
        if self.path == "/telemetry":
            body = b'{"clock":{"time":1.0,"warp":1.0,"paused":false}}'
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/telemetry/history":
            body = b'[{"t":1.0},{"t":2.0}]'
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/screenshot":
            # Write a fake PNG so the bridge's poll-and-read path can be exercised without a window.
            with open(SHOT_PATH, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\nfake")
            body = json.dumps({"ok": True, "path": SHOT_PATH}).encode("utf-8")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        received["posts"].append((self.path, body))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok":true}')


def rpc(proc, message):
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def main():
    server = HTTPServer(("127.0.0.1", 0), StubBus)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    proc = subprocess.Popen(
        [sys.executable, str(BRIDGE)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env={"SOUNDING_BUS_URL": f"http://127.0.0.1:{port}", "PATH": "/usr/bin:/bin"},
    )
    try:
        init = rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert init["result"]["serverInfo"]["name"] == "sounding-mcp", init

        tools = rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in tools["result"]["tools"]}
        assert names == {
            "get_telemetry",
            "get_telemetry_history",
            "get_screenshot",
            "replay",
            "send_command",
        }, names

        rep = rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": "replay", "arguments": {"action": {"scrub": -1}}},
            },
        )
        assert rep["result"]["content"][0]["text"] == '{"ok":true}', rep
        assert ("/replay", '{"scrub": -1}') in received["posts"], received["posts"]

        shot = rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": "get_screenshot", "arguments": {}},
            },
        )
        block = shot["result"]["content"][0]
        assert block["type"] == "image" and block["mimeType"] == "image/png", block

        hist = rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "get_telemetry_history", "arguments": {}},
            },
        )
        htext = hist["result"]["content"][0]["text"]
        assert isinstance(json.loads(htext), list), htext

        tele = rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "get_telemetry", "arguments": {}},
            },
        )
        text = tele["result"]["content"][0]["text"]
        assert json.loads(text)["clock"]["paused"] is False, text

        cmd = rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "send_command", "arguments": {"command": {"SetPaused": True}}},
            },
        )
        assert json.loads(cmd["result"]["content"][0]["text"])["ok"] is True, cmd
        assert ("/command", '{"SetPaused": true}') in received["posts"], received["posts"]

        # A notification draws no reply (id-less) — the bridge must not wedge.
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        proc.stdin.flush()
        ping = rpc(proc, {"jsonrpc": "2.0", "id": 5, "method": "tools/list"})
        assert ping["id"] == 5, ping

        print("OK: MCP handshake + telemetry/command proxying verified")
    finally:
        proc.stdin.close()
        proc.terminate()
        server.shutdown()


if __name__ == "__main__":
    main()
