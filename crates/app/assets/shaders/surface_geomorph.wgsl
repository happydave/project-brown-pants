// CDLOD geomorph vertex shader for the streamed procedural surface (WI 780 + 795).
//
// Replaces the StandardMaterial vertex stage (via ExtendedMaterial::vertex_shader).
// Each vertex carries a morph target (its position on the parent / coarser grid,
// @location(8)); a per-vertex morph factor is derived from the vertex's distance to
// the camera and the per-chunk ramp `(start, end)`, and the rendered local position
// is `mix(base, morph_target, factor)`. Because the factor is a continuous function of
// world-space distance, two chunks that share an edge morph that edge identically (no
// crack). Where a chunk borders a *different-level* neighbour, the distance factor
// alone cannot make both sides agree (per-level ramps cannot cover the realized
// boundary distances — WI 795), so the WI 795 **edge weld** forces the factor on
// boundary rows from the chunk's realized neighbour relation (the `weld` uniform):
// factor → 1 on edges bordering a coarser neighbour (odd vertices land exactly on the
// coarser chunk's un-morphed surface chord), factor → 0 on edges bordering a finer
// one (the finer side's welded vertices equal this chunk's surface grid), each fading
// inward over `band` rows. This must stay in lockstep with `weld_factor` in
// `sounding_sim::surface_mesh` — the headless oracle scans that formula.
// The fragment stage is the standard PBR path (ExtendedMaterial keeps the base
// fragment shader).

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
    forward_io::VertexOutput,
}

// Extension uniforms. `morph_range` = (start, end, _, _) morph-ramp distances in
// metres; `weld` = (coarser-edge bitmask, finer-edge bitmask, grid res, band rows),
// edge bit order v0, u1, v1, u0. The material bind group index is substituted by
// Bevy (`MATERIAL_BIND_GROUP`), not hardcoded — a literal @group(2) is the mesh
// group in this Bevy version.
struct GeomorphExt {
    morph_range: vec4<f32>,
}
struct GeomorphWeld {
    weld: vec4<f32>,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> geomorph: GeomorphExt;
@group(#{MATERIAL_BIND_GROUP}) @binding(101)
var<uniform> geomorph_weld: GeomorphWeld;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
#ifdef VERTEX_COLORS
    // Per-vertex biome tint (WI 869) — computed headless in sounding_sim and
    // passed through untouched (no color math on the shader side: the WI 795
    // one-side-only lockstep rule). The standard PBR fragment multiplies it.
    @location(5) color: vec4<f32>,
#endif
    @location(8) morph_target: vec3<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;

    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);

    // Per-vertex CDLOD morph factor from this vertex's camera distance.
    let base_world = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(vertex.position, 1.0),
    );
    let dist = distance(view.world_position, base_world.xyz);
    var factor = smoothstep(geomorph.morph_range.x, geomorph.morph_range.y, dist);

    // WI 795 edge weld (mirror of sounding_sim::surface_mesh::weld_factor —
    // keep in lockstep): distances to the four edges (v0, u1, v1, u0, bits 0-3)
    // and the Chebyshev distances to the four corners (c00, c10, c11, c01, bits
    // 4-7); per-source falloffs combine by max; finer forcings pull toward 0,
    // then coarser forcings toward 1 (coarser wins). Skirt vertices inherit
    // border UVs but their morph target is their own position, so the forcing
    // is a no-op on them.
    let res = geomorph_weld.weld.z;
    let band = geomorph_weld.weld.w;
    let coarser = u32(geomorph_weld.weld.x);
    let finer = u32(geomorph_weld.weld.y);
    let a = round(vertex.uv.x * res);
    let b = round(vertex.uv.y * res);
    var d: array<f32, 8> = array<f32, 8>(
        b,
        res - a,
        res - b,
        a,
        max(a, b),
        max(res - a, b),
        max(res - a, res - b),
        max(a, res - b),
    );
    var w_finer = 0.0;
    var w_coarser = 0.0;
    for (var k = 0u; k < 8u; k += 1u) {
        let w = clamp(1.0 - d[k] / band, 0.0, 1.0);
        if ((finer >> k) & 1u) == 1u {
            w_finer = max(w_finer, w);
        }
        if ((coarser >> k) & 1u) == 1u {
            w_coarser = max(w_coarser, w);
        }
    }
    factor = mix(factor, 0.0, w_finer);
    factor = mix(factor, 1.0, w_coarser);

    let local = mix(vertex.position, vertex.morph_target, factor);

    out.world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(local, 1.0),
    );
    out.position = position_world_to_clip(out.world_position.xyz);

#ifdef VERTEX_NORMALS
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex.normal,
        vertex.instance_index,
    );
#endif

#ifdef VERTEX_UVS_A
    out.uv = vertex.uv;
#endif

#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif

    return out;
}
