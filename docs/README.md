# Sounding — repo docs

Map of where things live. This repo (`project_brown_pants`) is the **code**; the **planning/backlog**
lives in the workflow ticket tree (a separate location), and this folder points to both.

## In this repo (code-adjacent docs)

- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — the LLM-oriented **code map**: crate layout, the
  `Forbidden:`/`Required:` boundaries, the component index, and the primary data flow. Read this first
  to find code.
- [`../CLAUDE.md`](../CLAUDE.md) — load-bearing repo conventions (headless-crate rule, command-routed
  state, **scale-relative physics** + its relationships/dissipative corollaries, dev-feature gating,
  one-patch-per-ticket).
- [`../README.md`](../README.md) — build/run, and the catalog of `cargo run -p sounding -- <scene>`
  scenarios.
- [`observability.md`](observability.md) — how to **observe a running scenario** (dev tooling: BRP,
  save-file inspection, world pause/save, the dev-MCP plan). The standing reference for the
  observability work.

## Planning / backlog (workflow tickets, outside this repo)

Source of truth for *what to build and why* (do not duplicate it here — link to it):

- **Sounding (parent):** `tickets/docs/projects/sounding/project.md` — the game vision, the
  Toys/foundation, the dev-tooling plan (BRP + MCP), and the run catalog.
- **Sounding — Rover:** `tickets/docs/projects/sounding-rover/project.md` — the buildable-rover
  initiative backlog (the active line), with `design.md`, `design-drivetrain.md`, `design-contact.md`
  and their reviews.
- **Other Sounding lines:** `sounding-first-playable`, `sounding-content` (scenario/mission/economy +
  content-aware persistence, which subsumes the old world-save WI 537 → 553).
- **Work items:** `tickets/docs/pending/<id>-<slug>/` — each has `workitem.md` (+ `plan.md`,
  `code.md`, `reflect.md`, … as it progresses).

> Path note: the ticket tree is at `/home/dave/Documents/tickets/docs/…` in the current environment.

## How to run (quick)

`cargo run -p sounding -- <scene>` — e.g. `workshop` (build & drive a rover), `rover`, `dive`, `planet`,
`play`. See [`../README.md`](../README.md) for the full list and controls.

## Conventions recap (see CLAUDE.md for the authority)

- Headless core: `crates/sim` (`sounding_sim`) never pulls render deps (`cargo tree -p sounding_sim`
  clean).
- Scale-relative physics: per-body coefficients are mass/inertia-relative **rates**; coupled rates keep
  their ordering scale-relative too; prefer a dissipative force over deleting one.
- **Test the shipping configuration:** rover contact/ride tests use a light editor-scale build, not the
  heavy 3-ton fixture (heavy fixtures mask instabilities).
- One patch version bump per work item.
</content>
