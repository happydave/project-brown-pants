# Model attribution

`terrain.glb` is a baked output from the **asset-harness** `terrainmesh` track: a relief mesh
generated from a height/geometry estimation step (MoGe) and exported to glTF. The
workflow/seed/prompt in the asset-harness repo is the source of truth; this `.glb` is the cached,
versioned output, previewed by `cargo run -p sounding -- terrainmesh`.

License: follows the terms of the generating model/tool (MoGe and the export pipeline), as
recorded in the asset-harness `terrainmesh` track docs. **Confirm those terms before
redistributing this asset** outside the project.
