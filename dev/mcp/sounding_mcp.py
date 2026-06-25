#!/usr/bin/env python3
"""Dev-only MCP bridge to the running Sounding game (WI 639 spike).

A thin, **stdlib-only** Model Context Protocol server (stdio transport, newline-delimited JSON-RPC
2.0) that proxies to the game's always-on runtime bus (`crates/app/src/bus.rs`, default
`http://127.0.0.1:8787`). It exposes two domain tools so an assistant can observe and steer the exact
scenario under test instead of reproducing it headlessly:

  * ``get_telemetry`` → ``GET /telemetry`` — the versioned `Telemetry` snapshot (clock, orbit,
    active-flight autonomy, and — in a rover scene — a ``rover`` block with pose/contact/per-wheel
    state, WI 640).
  * ``send_command``  → ``POST /command``  — inject one JSON `Command` (e.g. ``{"SetPaused": true}``,
    ``{"SetWarp": 4.0}``).

This is the **command-bus** path the spike recommends over BRP-generic ECS introspection: always-on
(no `dev` rebuild), already JSON, already a versioned public surface. It is dev-only tooling and is
never part of the game build.

Registration is the user's step (see README.md) — this script is not wired into any Claude config by
itself.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BUS_URL = os.environ.get("SOUNDING_BUS_URL", "http://127.0.0.1:8787")
PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "sounding-mcp", "version": "0.1.0"}

TOOLS = [
    {
        "name": "get_telemetry",
        "description": "Read the running game's current telemetry snapshot (GET /telemetry).",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "send_command",
        "description": (
            "Inject one Command into the running game (POST /command). `command` is the JSON body, "
            'e.g. {"SetPaused": true} or {"SetWarp": 4.0}.'
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "object",
                    "description": "A serialized sounding_sim::command::Command.",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
]


def _http_get(path: str) -> str:
    with urllib.request.urlopen(f"{BUS_URL}{path}", timeout=5) as resp:
        return resp.read().decode("utf-8")


def _http_post(path: str, body: str) -> str:
    req = urllib.request.Request(
        f"{BUS_URL}{path}",
        data=body.encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.read().decode("utf-8")


def _call_tool(name: str, arguments: dict) -> dict:
    """Dispatch a tool call, returning an MCP `tools/call` result. Bus/IO errors are surfaced as
    `isError` tool results (not transport errors) so the assistant sees what failed."""
    try:
        if name == "get_telemetry":
            text = _http_get("/telemetry")
        elif name == "send_command":
            command = arguments.get("command")
            if command is None:
                raise ValueError("send_command requires a `command` object")
            text = _http_post("/command", json.dumps(command))
        else:
            raise ValueError(f"unknown tool: {name}")
        return {"content": [{"type": "text", "text": text}]}
    except (urllib.error.URLError, ValueError, OSError) as exc:
        return {"content": [{"type": "text", "text": f"bus error: {exc}"}], "isError": True}


def _handle(message: dict):
    """Route one JSON-RPC message. Returns a response dict, or None for notifications."""
    method = message.get("method")
    msg_id = message.get("id")

    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = message.get("params") or {}
        result = _call_tool(params.get("name", ""), params.get("arguments") or {})
    elif method is not None and method.startswith("notifications/"):
        return None  # notifications get no reply
    else:
        if msg_id is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }

    if msg_id is None:
        return None  # a notification we happened to recognise
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = _handle(message)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
