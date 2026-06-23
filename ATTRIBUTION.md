# Attribution

This project bundles third-party code (as dependencies) and a small set of baked assets.

## Code dependencies

The crates are pulled at build time from crates.io (not vendored in this repo); their license
texts ship with each crate. The principal dependencies and their licenses:

| Crate | License |
| ----- | ------- |
| `bevy` (engine, 0.18) | MIT OR Apache-2.0 |
| `glam` (f64 math) | MIT OR Apache-2.0 |
| `serde` / `serde_json` | MIT OR Apache-2.0 |
| `parry3d-f64` (collision detection) + `glamx` | Apache-2.0 |

All are compatible with this project's `MIT OR Apache-2.0` license. A complete, generated
third-party license report (including transitive dependencies) can be produced with
[`cargo about`](https://github.com/EmbarkStudios/cargo-about) or
[`cargo deny`](https://github.com/EmbarkStudios/cargo-deny) when preparing a binary distribution.

## Bundled assets

Baked outputs from the **asset-harness** pipeline (the workflow/seed/prompt in the asset-harness
repo is the source of truth; the files here are the cached, versioned output):

| Asset | Location | Notes |
| ----- | -------- | ----- |
| PBR material sets (`hull_panel`, `rocky_ground`) | `crates/app/assets/materials/` | see [materials/ATTRIBUTION.md](crates/app/assets/materials/ATTRIBUTION.md) |
| Terrain mesh (`terrain.glb`) | `crates/app/assets/models/` | Z-Image concept → MoGe-2 depth (`Ruicheng/moge-2-vitl-normal`, MIT; DINOv2 Apache-2.0) → Blender displacement; see [models/ATTRIBUTION.md](crates/app/assets/models/ATTRIBUTION.md) |

> Note: the generated assets' licensing follows the terms of the generating models/tools, which
> are recorded in the asset-harness track docs. Confirm those terms before redistributing the
> baked assets outside this project.
