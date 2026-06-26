# Part mesh attribution

The `*.glb` part meshes in this directory are baked outputs from the **asset-harness**
`mechanical-kit` track: parametric geometry authored in Blender (`blender_parts.py`) with AI PBR
surfaces from the `pbr-materials` track (`gen_part_materials.py` + the WI 624 delight pass), skinned
and exported via `blender_texture_parts.py`. The workflow/scripts are the source of truth in the
asset-harness repo; these glb are the cached, versioned output (WI 653).

License: Apache-2.0 / MIT (Blender outputs are the user's; Z-Image surfaces are Apache).

Frame contract (per the catalog manifest): authored +Y-up / +Z-forward / +X-axle, metres, reference
cell 0.5 m. Note: the glTF export lands the authored "up" along -Z, so the Sounding loader
(`parts::spawn_part_mesh`) applies a +90°-about-X correction to bring parts to Bevy's +Y-up.
