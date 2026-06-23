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

- **MoGe** ([github.com/microsoft/MoGe](https://github.com/microsoft/MoGe)) — code MIT, except the
  bundled **DINOv2** (`moge/model/dinov2`) which is **Apache-2.0** (Meta AI).
- **Z-Image** — Apache-2.0 (per the asset-harness track docs).
- **DINOv2** — Apache-2.0 (Meta AI).

> **Verify before redistributing** this asset: confirm the **model-card / weights** license for
> `Ruicheng/moge-2-vitl-normal` — a code license (MIT) does not automatically cover the published
> weights or their outputs — and the Z-Image output terms. The asset-harness repo is authoritative
> for the exact models/seeds used.
