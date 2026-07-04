# Dev MCP bridge (WI 639 spike)

A dev-only [MCP](https://modelcontextprotocol.io) server that lets an AI assistant **observe and steer
the running game** through its always-on runtime bus — so it can inspect the exact scenario under test
instead of reproducing it headlessly and guessing the build/inputs (the slow loop hit during the
631/634 rover work).

## What it is

`sounding_mcp.py` is a thin, **stdlib-only** stdio MCP server that proxies to the runtime bus
(`crates/app/src/bus.rs`, default `http://127.0.0.1:8787`):

| MCP tool | Bus call | Purpose |
| --- | --- | --- |
| `get_telemetry` | `GET /telemetry` | Read the versioned `Telemetry` snapshot (clock, orbit, active-flight autonomy, additive rover/thermal/scenario blocks). |
| `get_telemetry_history` | `GET /telemetry/history` | The last few seconds of snapshots, oldest-first (WI 644) — a time series for spiky signals. |
| `get_screenshot` | `GET /screenshot` | Capture the window and return it as an image (WI 647). |
| `replay` | `POST /replay` | Drive the Tier-B replay cam (WI 648): enter/exit/toggle, scrub, seek. |
| `send_command` | `POST /command` | Inject one JSON `Command`, e.g. `{"SetPaused": true}`, `{"SetWarp": 4.0}`. |
| `set_camera` / `set_overlay` / `get_camera` | `POST /camera` / `POST /debug` / `GET /camera` | Debug camera placement + overlays + pose readback (WI 784). |
| `send_input` | `POST /input` | Inject a keyboard/mouse action (WI 830; **requires a `--features dev` game build** — 404 otherwise): tap/hold keys, move the cursor, click, scroll — indistinguishable from real input, so mode toggles and palette picks become scripted screenshot anchors. |

It is **not** part of the game build and is never load-bearing.

## Run the game with the bus

The bus is always on; just run any scene, e.g.:

```sh
cargo run -p sounding -- workshop
```

Quick manual check (no MCP needed):

```sh
curl -s http://127.0.0.1:8787/telemetry
curl -s -X POST http://127.0.0.1:8787/command -d '{"SetPaused": true}'
```

## Self-test (no game required)

```sh
python3 dev/mcp/test_bridge.py   # spins a stub bus, drives the bridge over stdio
```

## Register it with Claude — **user step (not done automatically)**

Per project convention, registering tooling into your Claude config is left to you. Easiest is the CLI:

```sh
claude mcp add sounding --scope user \
  --env SOUNDING_BUS_URL=http://127.0.0.1:8787 \
  -- python3 /home/dave/Documents/projects/project_brown_pants/dev/mcp/sounding_mcp.py
```

(`--` separates Claude's own flags from the command to run.) Verify with `claude mcp list`.

`claude mcp add-json` works too, but its payload is the **single server's config object** — *not* the
`{"mcpServers": {...}}` wrapper, and the name is the positional arg:

```sh
claude mcp add-json sounding --scope user '{
  "command": "python3",
  "args": ["/home/dave/Documents/projects/project_brown_pants/dev/mcp/sounding_mcp.py"],
  "env": { "SOUNDING_BUS_URL": "http://127.0.0.1:8787" }
}'
```

Or edit `~/.claude.json` (or a project `.mcp.json`) by hand — *this* is where the full wrapper goes:

```json
{
  "mcpServers": {
    "sounding": {
      "command": "python3",
      "args": ["/home/dave/Documents/projects/project_brown_pants/dev/mcp/sounding_mcp.py"],
      "env": { "SOUNDING_BUS_URL": "http://127.0.0.1:8787" }
    }
  }
}
```

After registering (and restarting the session), the assistant gets the bridge's tools (the table
above) acting on whatever scene is running.

## Resolved follow-ups

The spike's recommended follow-up — rover pose / contact signals in `Telemetry` — landed as WI 640
(the `rover` block: pose, `hull_penetration`, per-wheel state); the bridge forwarded it unchanged, as
predicted. Later additions rode the same pattern: history (644), screenshots (647), replay (648),
debug camera (784), and input injection (830 — the one tool that needs a `--features dev` game build).

For the full go/no-go and the BRP-vs-command-bus decision, see the work item's `spike.md`.
