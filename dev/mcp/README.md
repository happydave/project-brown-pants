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
| `get_telemetry` | `GET /telemetry` | Read the versioned `Telemetry` snapshot (clock, orbit, active-flight autonomy). |
| `send_command` | `POST /command` | Inject one JSON `Command`, e.g. `{"SetPaused": true}`, `{"SetWarp": 4.0}`. |

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

Per project convention, registering tooling into your Claude config is left to you. Add to
`~/.claude.json` (or the project `.mcp.json`) under `mcpServers`:

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

Then the assistant gets `get_telemetry` / `send_command` tools that act on whatever scene is running.

## Known gap (recommended follow-up)

`Telemetry` does not yet carry **rover pose / contact signals** (e.g. `hull_penetration`, per-wheel
state). Reading those live is exactly what would have lifted the 631/634 blindfold, so the recommended
next step is a separate WI to enrich the `Telemetry` snapshot (on the versioned surface) with a
grounded/wheeled section. The bridge needs no change when that lands — it already forwards whatever
`/telemetry` returns.

For the full go/no-go and the BRP-vs-command-bus decision, see the work item's `spike.md`.
