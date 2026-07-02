// CDLOD geomorph vertex shader for the streamed procedural surface (WI 780).
//
// Replaces the StandardMaterial vertex stage (via ExtendedMaterial::vertex_shader).
// Each vertex carries a morph target (its position on the parent / coarser grid,
// @location(8)); a per-vertex morph factor is derived from the vertex's distance to
// the camera and the per-chunk ramp `(start, end)`, and the rendered local position
// is `mix(base, morph_target, factor)`. Because the factor is a continuous function of
// world-space distance, two chunks that share an edge morph that edge identically (no
// crack), and a fine chunk is fully morphed to the coarse shape exactly where it meets
// a coarser neighbour or merges (no step, no pop). The fragment stage is the standard
// PBR path (ExtendedMaterial keeps the base fragment shader).

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
    forward_io::VertexOutput,
}

// Extension uniform: (start, end, _, _) morph-ramp distances in metres. The material
// bind group index is substituted by Bevy (`MATERIAL_BIND_GROUP`), not hardcoded — a
// literal @group(2) is the mesh group in this Bevy version.
struct GeomorphExt {
    morph_range: vec4<f32>,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> geomorph: GeomorphExt;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
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
    let factor = smoothstep(geomorph.morph_range.x, geomorph.morph_range.y, dist);
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

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif

    return out;
}
