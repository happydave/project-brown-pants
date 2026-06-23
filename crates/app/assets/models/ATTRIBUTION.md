# Model attribution

`terrain.glb` is a baked terrain chunk from the **asset-harness** `3d-static-props` track
(Blender **heightmap-displacement** pipeline; previewed by `cargo run -p sounding -- terrainmesh`).
The asset-harness repo holds the source-of-truth seeds/prompts/models; this `.glb` is the cached,
game-light (decimated + 512 px JPEG texture) output.

## Provenance

1. **Concept image** — a canyon concept generated with **Z-Image** (Apache-2.0, per the
   asset-harness track docs).
2. **Heightmap** — **MoGe-2** monocular geometry: model `moge_2_vitl_normal_fp16`
   (HuggingFace [`Ruicheng/moge-2-vitl-normal`](https://huggingface.co/Ruicheng/moge-2-vitl-normal),
   Microsoft) — used to derive a depth/height map from the concept image.
3. **Mesh** — Blender displaces a grid by that heightmap, then decimates and bakes a 512 px JPEG
   texture. It is **not** the raw MoGe relief mesh (that stretched on the oblique input).

## Licenses

- **MoGe-2** — **MIT** for both the **weights and codebase**
  ([`Ruicheng/moge-2-vitl-normal`](https://huggingface.co/Ruicheng/moge-2-vitl-normal) /
  [github.com/microsoft/MoGe](https://github.com/microsoft/MoGe)), **except** the bundled
  **DINOv2** backbone (`moge/model/dinov2`) which is **Apache-2.0** (Meta AI). Both are compatible
  with this project's `MIT OR Apache-2.0`. (Confirmed against the model card / repo.)
- **DINOv2** — Apache-2.0 (Meta AI).
- **Z-Image** (Alibaba Tongyi-MAI) — generated images are released under **Apache-2.0**: free
  commercial use, modification allowed, **no attribution required** (credited here for provenance
  only).

> **Cleared for redistribution.** Every link in the chain — the Z-Image concept image
> (Apache-2.0), the MoGe-2 tool/weights (MIT; DINOv2 Apache-2.0), and the Blender step — is
> permissive and compatible with this project's `MIT OR Apache-2.0`. The asset-harness repo
> remains authoritative for the exact model/seed/prompt.
