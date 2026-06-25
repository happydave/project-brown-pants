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
| **Vehicle (craft) save/load** | Save a built rover/craft to a file; reload it. Saves the user re-building; lets the assistant load the *exact* craft under test. | **New** → `tickets/docs/pending/637-snd-craft-save-load/`. Builds on `crates/sim/src/persist.rs` (the versioned lattice persist format already exists). |
| **World pause** | Freeze the running scene to inspect a state without it evolving. | **New** → `tickets/docs/pending/638-snd-world-pause/`. The sim already has a `pause`/warp command path (`apply_command`); this is wiring scene-level pause + a clear indicator. |
| **World save/load + save-file inspector** | Persist a whole scenario (craft + world + clock); a CLI/tool to **dump/inspect** a save file so the assistant can read what's there. | Partly designed: **WI 537** (fp-world-save, deferred) and **WI 553** (sc-content-aware-persistence, pending — subsumes 537). The *inspector tool* is the new, assistant-facing piece → tracked in 553's scope / a small CLI. |
| **Dev MCP (live introspection)** | An MCP bridge to the running game so the assistant can query ECS state / telemetry / the command bus live. | **Spike** → `tickets/docs/pending/639-snd-dev-mcp-spike/`. See findings below. |

(Foundational observability already exists: **WI 499** observability harness + physics-invariant
metrics; `crates/sim/src/{telemetry,diagnostics}.rs`; the `Telemetry` snapshot is a versioned API.)

## Dev-MCP findings (what's there, what's missing)

- The design intent is recorded in `tickets/docs/projects/sounding/project.md` (§dev tooling): **Bevy
  Remote Protocol (BRP) + MCP** for ECS introspection and scenario/test harnesses, plus a custom
  **command-bus MCP** for domain-level verbs — all **dev-only, behind the `dev` cargo feature**, never
  load-bearing.
- BRP is gated behind the app's `dev` feature (see CLAUDE.md "Dev tooling"). So the *engine side* of
  live introspection is partly in place.
- **Missing:** the MCP **bridge** that connects the running game (BRP and/or the always-on command bus)
  to the assistant's tool surface, and the wiring to register it (e.g. in the Claude config). In the
  current environment **no dev MCP is reachable from the assistant** — confirmed during the 631/634
  work; that's why reproduction was headless-only.
- Spike goal: stand up the BRP-or-command-bus → MCP path end to end on one scenario (e.g. `workshop`),
  confirm the assistant can read live state (rover pose, contact signals like `hull_penetration`,
  telemetry), and decide BRP-generic vs a domain command-bus MCP.

## Notes

- Keep all of this **dev-only** (cargo `dev` feature) and out of release/headless builds.
- The `Telemetry` snapshot and `Command` envelope are the **versioned public surface** — prefer
  exposing observability through them (sweep consumers on change) rather than ad-hoc taps.
- Save-file formats should reuse the versioned `persist` path; the inspector reads the same format.
</content>
