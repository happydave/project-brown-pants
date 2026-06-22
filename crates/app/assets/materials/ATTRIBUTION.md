# Material attribution

PBR material sets in this directory are baked outputs from the **asset-harness**
`pbr-materials` track (generated locally; albedo via Z-Image, maps derived locally). The
workflow/seed/prompt is the source of truth in the asset-harness repo; these PNGs are the
cached, versioned output.

License: Apache-2.0 / MIT (per the generating models' terms recorded in the asset-harness
track docs).

| Set | Maps | Used by |
| --- | ---- | ------- |
| `hull_panel` | albedo, normal, metallic_roughness, occlusion | craft skins (`-- skins`), WI 582/583 |
| `rocky_ground` | albedo, normal, metallic_roughness, occlusion | textured ground (`-- skins`/`-- play`/`-- autopilot`/`-- launch`), WI 588 |
