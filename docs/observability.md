# Observability & dev tooling

How to **see what a running scenario is doing** — for faster iteration and so an AI assistant can
inspect the exact situation being tested (instead of re-deriving it from playtest prose). This is the
standing reference; the actionable items are tracked as work items (linked below).

Motivation: several recent rover playtest loops were slow because the assistant couldn't see the
running build (e.g. the ramp-lip kraken was only found by reproducing headlessly and guessing the
build/speed). The items below remove that blindfold.

## The four threads (status)

| Thread | Why | Status / work item |
| --- | --- | --- |
| **Vehicle (craft) save/load** | Save a built rover/craft to a file; reload it. Saves the user re-building; lets the assistant load the *exact* craft under test. | **Done (WI 637)** — `editor.rs` `K` save / `O` open `craft.json` via the versioned `persist` envelope. |
| **World pause** | Freeze the running scene to inspect a state without it evolving. | **Done (WI 638)** — `P` emits `Command::SetPaused` on the bus; `workshop`/`rover`/`play` gate their step on `SimClock.paused`, with a `⏸ PAUSED` HUD banner (`crate::pause`). |
| **World save/load + save-file inspector** | Persist a whole scenario (craft + world + clock); a CLI/tool to **dump/inspect** a save file so the assistant can read what's there. | Partly designed: **WI 537** (fp-world-save, deferred) and **WI 553** (sc-content-aware-persistence, pending — subsumes 537). The *inspector tool* is the new, assistant-facing piece → tracked in 553's scope / a small CLI. |
| **Dev MCP (live introspection)** | An MCP bridge to the running game so the assistant can query ECS state / telemetry / the command bus live. | **Spike done (WI 639)** — bridge at `dev/mcp/` (command-bus over BRP-generic); self-test green. **`Telemetry` now carries a rover block (WI 640)** — `-- rover` and the workshop Test publish pose/contact/per-wheel state. Remaining follow-up: user registers the MCP. See findings below. |

(Foundational observability already exists: **WI 499** observability harness + physics-invariant
metrics; `crates/sim/src/{telemetry,diagnostics}.rs`; the `Telemetry` snapshot is a versioned API.)

## Dev-MCP findings (WI 639 spike — resolved)

Full write-up: `tickets/docs/pending/639-snd-dev-mcp-spike/spike.md`. Summary:

- **Verdict: GO**, via the **command bus** (`bus.rs`, always-on `127.0.0.1:8787`), not BRP-generic.
  The bus is always compiled (no `dev` rebuild), already JSON, and a **versioned public surface**
  (`Telemetry`/`Command`); BRP stays the dev-only god-mode escape hatch for raw ECS poking.
- **Artifact:** `dev/mcp/sounding_mcp.py` — a stdlib-only stdio MCP bridge exposing `get_telemetry` /
  `send_command`; `dev/mcp/test_bridge.py` proves the `initialize`/`tools/list`/`tools/call` handshake
  + HTTP proxying against a stub bus (green) without needing the windowed game.
- **Two follow-ups:** (1) **user** registers the MCP in `~/.claude.json` (`dev/mcp/README.md`); the
  assistant can't reach it until then. (2) ~~Enrich `Telemetry` with rover pose / `hull_penetration` /
  per-wheel state~~ — **Done (WI 640).** `Telemetry` now carries a serde-defaulted `rover` block
  (`RoverTelemetry`: pose, velocity, angular velocity, `contact_jitter`, `hull_penetration`, per-wheel
  `axle_drop`/load/slip/grip/`inert`+failure flags); the `-- rover` scene and the workshop Test
  publish it each frame via a `GroundedRover` bus bridge (mirrors the WI 569 `ActiveFlight` path). The
  bridge forwards it unchanged, lifting the 631/634 blindfold.

## Notes

- Keep all of this **dev-only** (cargo `dev` feature) and out of release/headless builds.
- The `Telemetry` snapshot and `Command` envelope are the **versioned public surface** — prefer
  exposing observability through them (sweep consumers on change) rather than ad-hoc taps.
- Save-file formats should reuse the versioned `persist` path; the inspector reads the same format.
</content>
