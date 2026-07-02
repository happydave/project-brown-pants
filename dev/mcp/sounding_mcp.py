#!/usr/bin/env python3
"""Dev-only MCP bridge to the running Sounding game (WI 639 spike).

A thin, **stdlib-only** Model Context Protocol server (stdio transport, newline-delimited JSON-RPC
2.0) that proxies to the game's always-on runtime bus (`crates/app/src/bus.rs`, default
`http://127.0.0.1:8787`). It exposes two domain tools so an assistant can observe and steer the exact
scenario under test instead of reproducing it headlessly:

  * ``get_telemetry`` → ``GET /telemetry`` — the versioned `Telemetry` snapshot (clock, orbit,
    active-flight autonomy, and — in a rover scene — a ``rover`` block with pose/contact/per-wheel
    state, WI 640).
  * ``get_telemetry_history`` → ``GET /telemetry/history`` — a JSON array of the last few seconds of
    snapshots, oldest-first (WI 644); a time series for spiky signals (pair with a paused ``Step``).
  * ``get_screenshot`` → ``GET /screenshot`` — capture the window and return it as an image (WI 647);
    a gestalt read of the scene / cockpit overlay (the image arrives as a file the bridge reads back).
  * ``replay`` → ``POST /replay`` — drive the Tier-B replay cam (WI 648): enter/exit/toggle, scrub, or
    seek the last few seconds (workshop Test); pair with ``get_screenshot`` to capture a scrubbed moment.
  * ``send_command``  → ``POST /command``  — inject one JSON `Command` (e.g. ``{"SetPaused": true}``,
    ``{"SetWarp": 4.0}``, or — while paused — ``{"Step": {"seconds": 0.1}}`` to advance a frozen scene
    a known amount for inspection, WI 643).

This is the **command-bus** path the spike recommends over BRP-generic ECS introspection: always-on
(no `dev` rebuild), already JSON, already a versioned public surface. It is dev-only tooling and is
never part of the game build.

Registration is the user's step (see README.md) — this script is not wired into any Claude config by
itself.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
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
        "name": "get_telemetry_history",
        "description": (
            "Read the recent telemetry history (GET /telemetry/history): a JSON array of the last few "
            "seconds of snapshots, oldest-first — a time series for inspecting how a signal evolves "
            "(pair with a paused Step). Prefer this over polling get_telemetry for spiky signals."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_screenshot",
        "description": (
            "Capture the running game's window and return it as an image (GET /screenshot, WI 647). "
            "Prefer this over raw telemetry for a gestalt read of the scene / cockpit overlay."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "replay",
        "description": (
            "Drive the Tier-B replay cam (POST /replay, WI 648), in the workshop Test. `action` is the "
            'JSON body: "enter" / "exit" / "toggle", {"scrub": -2} (frames), or {"seek": 0.5} '
            "(fraction). Replay freezes the sim and re-poses the last few seconds; pair with "
            "get_screenshot to capture a scrubbed moment."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"description": "A serialized ReplayCommand (string or object)."}
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    {
        "name": "send_command",
        "description": (
            "Inject one Command into the running game (POST /command). `command` is the JSON body, "
            'e.g. {"SetPaused": true}, {"SetWarp": 4.0}, or while paused {"Step": {"seconds": 0.1}}.'
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
    {
        "name": "set_camera",
        "description": (
            "Position/aim the scene's debug camera (POST /camera, WI 784) — for scenes with a "
            "controllable camera (start: `-- surface`). `command` is a serialized DebugCommand, e.g. "
            '{"set_orbit": {"altitude_m": 8000, "lat_deg": 12, "lon_deg": -30, "look": "nadir"}} '
            '(look = "nadir" | "horizon" | {"direction": [x,y,z]}); '
            '{"named_pose": "grazing_horizon_6km"}; '
            '{"set_pose": {"position": [x,y,z], "look_at": [x,y,z]}} (metres from body centre); or '
            '{"nudge": {"forward_m": 100, "yaw_deg": 5}}. Then get_screenshot; read get_camera for the '
            "resulting pose. No-op in scenes without a controllable camera."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "object", "description": "A serialized DebugCommand."}
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    {
        "name": "set_overlay",
        "description": (
            "Toggle debug overlays (POST /debug, WI 784). `command` is a DebugCommand, e.g. "
            '{"set_overlay": {"lod": true}} to show the LOD/chunk wireframe overlay (F3). Pair with '
            "get_screenshot to see which LOD levels meet at a seam."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "object", "description": "A serialized DebugCommand (set_overlay)."}
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_camera",
        "description": (
            "Read the debug camera's current pose (GET /camera, WI 784): body-relative position, "
            "altitude, and look direction, or {\"available\": false} in scenes without a controllable "
            "camera. Use to confirm where set_camera framed before screenshotting."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
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


def _capture_screenshot() -> dict:
    """Trigger a capture (GET /screenshot deletes the stale file + asks Bevy to recapture), then poll
    for the PNG to reappear and return it as an MCP image content block. The bus and this bridge share
    a machine, so the file path in the response is directly readable."""
    info = json.loads(_http_get("/screenshot"))
    path = info.get("path")
    if not path:
        raise ValueError("screenshot response had no path")
    # Poll for the freshly-written file (capture lands a frame or two later, on the render thread).
    deadline = time.time() + 5.0
    last_size = -1
    while time.time() < deadline:
        time.sleep(0.1)
        if not os.path.exists(path):
            continue
        size = os.path.getsize(path)
        if size > 0 and size == last_size:  # stable across two polls → write finished
            data = base64.b64encode(open(path, "rb").read()).decode("ascii")
            return {"content": [{"type": "image", "data": data, "mimeType": "image/png"}]}
        last_size = size
    raise TimeoutError("screenshot did not appear (is the windowed game running?)")


def _call_tool(name: str, arguments: dict) -> dict:
    """Dispatch a tool call, returning an MCP `tools/call` result. Bus/IO errors are surfaced as
    `isError` tool results (not transport errors) so the assistant sees what failed."""
    try:
        if name == "get_telemetry":
            text = _http_get("/telemetry")
        elif name == "get_telemetry_history":
            text = _http_get("/telemetry/history")
        elif name == "get_screenshot":
            return _capture_screenshot()
        elif name == "replay":
            action = arguments.get("action")
            if action is None:
                raise ValueError("replay requires an `action`")
            text = _http_post("/replay", json.dumps(action))
        elif name == "send_command":
            command = arguments.get("command")
            if command is None:
                raise ValueError("send_command requires a `command` object")
            text = _http_post("/command", json.dumps(command))
        elif name == "set_camera":
            command = arguments.get("command")
            if command is None:
                raise ValueError("set_camera requires a `command` (DebugCommand)")
            text = _http_post("/camera", json.dumps(command))
        elif name == "set_overlay":
            command = arguments.get("command")
            if command is None:
                raise ValueError("set_overlay requires a `command` (DebugCommand)")
            text = _http_post("/debug", json.dumps(command))
        elif name == "get_camera":
            text = _http_get("/camera")
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
