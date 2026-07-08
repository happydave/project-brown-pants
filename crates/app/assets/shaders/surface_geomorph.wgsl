// CDLOD geomorph vertex shader + biome splat fragment shader for the streamed
// procedural surface (WI 780 + 795 + 872).
//
// VERTEX (WI 780/795): each vertex carries a morph target (its position on the
// parent / coarser grid, @location(8)); a per-vertex morph factor is derived from
// the vertex's distance to the camera and the per-chunk ramp `(start, end)`, and
// the rendered local position is `mix(base, morph_target, factor)`. Because the
// factor is a continuous function of world-space distance, two chunks that share
// an edge morph that edge identically (no crack). Where a chunk borders a
// *different-level* neighbour, the WI 795 **edge weld** forces the factor on
// boundary rows from the chunk's realized neighbour relation (the `weld`
// uniform). This must stay in lockstep with `weld_factor` in
// `sounding_sim::surface_mesh` — the headless oracle scans that formula.
//
// FRAGMENT (WI 872): terrain texture splatting. Per-vertex slot weights
// (@location(10)/(11), body-wide fixed slot semantics — see
// `sounding_sim::biome::texture_slot_names`) blend up to 8 layers of the three
// global KTX2 texture arrays; a period-exact stochastic detile breaks albedo
// repetition; a wide camera-distance fade converges albedo, normal, roughness
// and AO to the WI 869 tint look (the far regime samples nothing). The blend
// math lives ONLY here (fragment-side; the sim ships inputs, not a mirror —
// the WI 795 one-side rule). Two constants are deliberately shared with the
// sim and must match: the terrain-UV period (`TERRAIN_UV_PERIOD` = 2048 tiles)
// and the slot count (8 = two vec4 attributes).

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
    forward_io::{VertexOutput, FragmentOutput},
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
// WI 872 splat config: per-body slot→array-layer indices, per-slot anchor tints
// (linear RGB — the mean tint of the biome rows consuming the slot's texture,
// computed from the biome table at startup), and params =
// (fade_start_m, fade_end_m, enabled, _).
struct SplatUniform {
    layers_a: vec4<u32>,
    layers_b: vec4<u32>,
    anchors: array<vec4<f32>, 8>,
    params: vec4<f32>,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> geomorph: GeomorphExt;
@group(#{MATERIAL_BIND_GROUP}) @binding(101)
var<uniform> geomorph_weld: GeomorphWeld;
@group(#{MATERIAL_BIND_GROUP}) @binding(102)
var<uniform> splat: SplatUniform;
@group(#{MATERIAL_BIND_GROUP}) @binding(103)
var terrain_albedo: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(104)
var terrain_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(105)
var terrain_normal: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(106)
var terrain_surface: texture_2d_array<f32>;

// Must equal sounding_sim::surface_mesh::TERRAIN_UV_PERIOD (tiles): chunk UV
// anchors snap to multiples of this, so the detile hash below must be exactly
// periodic in it or the pattern re-rolls along anchor lines (a straight seam).
const TERRAIN_UV_PERIOD: f32 = 2048.0;
// Fragment-side weight floor: slots below this (after normalization) are not
// sampled. Applied as a continuous shrink — `max(0, w - floor)` — so a slot
// fades out exactly at the threshold instead of stepping (the project's
// no-iso-line invariant, kept even at sub-visible scale).
const SPLAT_WEIGHT_FLOOR: f32 = 0.03;

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
    // WI 872: terrain UV (tile units, chunk-anchored, f32-safe) + slot weights.
    @location(9) terrain_uv: vec2<f32>,
    @location(10) splat_a: vec4<f32>,
    @location(11) splat_b: vec4<f32>,
};

// forward_io::VertexOutput plus the WI 872 splat varyings. The fragment stage
// copies the standard fields into a `VertexOutput` value to feed the stock
// `pbr_input_from_standard_material`, so the untouched path IS the standard
// pipeline.
struct SplatVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
#ifdef VERTEX_UVS_A
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(5) color: vec4<f32>,
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    @location(6) @interpolate(flat) instance_index: u32,
#endif
    @location(9) terrain_uv: vec2<f32>,
    @location(10) splat_a: vec4<f32>,
    @location(11) splat_b: vec4<f32>,
}

@vertex
fn vertex(vertex: Vertex) -> SplatVertexOutput {
    var out: SplatVertexOutput;

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

    out.terrain_uv = vertex.terrain_uv;
    out.splat_a = vertex.splat_a;
    out.splat_b = vertex.splat_b;

    return out;
}

// Integer-lattice hash, exactly periodic in TERRAIN_UV_PERIOD (cells are hashed
// modulo the period, so the stochastic pattern is world-consistent across chunk
// anchors). Plain 2D integer scramble; quality needs are low (one offset/cell).
fn cell_hash(cell: vec2<f32>) -> vec2<f32> {
    let c = cell - TERRAIN_UV_PERIOD * floor(cell / TERRAIN_UV_PERIOD);
    let p = vec2<f32>(
        dot(c, vec2<f32>(127.1, 311.7)),
        dot(c, vec2<f32>(269.5, 183.3)),
    );
    return fract(sin(p) * 43758.5453);
}

// Period-exact stochastic detile (WI 872, replaces the plan's hex-tiling — see
// code.md): 1-tile cells, hash-translated sample offsets (a translated seamless
// texture stays seamless), 4-tap bilinear cell blend sharpened (pow 3,
// renormalized) to restore contrast, gradients from the un-offset UV so mip
// selection matches plain tiling.
fn detiled_albedo(layer: u32, uv: vec2<f32>, ddx_uv: vec2<f32>, ddy_uv: vec2<f32>) -> vec3<f32> {
    let base = uv - 0.5;
    let ic = floor(base);
    let fr = fract(base);
    let bl = vec4<f32>(
        (1.0 - fr.x) * (1.0 - fr.y),
        fr.x * (1.0 - fr.y),
        (1.0 - fr.x) * fr.y,
        fr.x * fr.y,
    );
    // Strong contrast restoration: near-equal cell weights (the 4-way blend
    // zone) otherwise average four offset crops into mush. pow-5 keeps a thin
    // continuous blend seam and single-cell content elsewhere.
    let b2 = bl * bl;
    let sharp = b2 * b2 * bl;
    let wsum = sharp.x + sharp.y + sharp.z + sharp.w;
    var color = vec3<f32>(0.0);
    let offs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    for (var k = 0u; k < 4u; k += 1u) {
        let cell = ic + offs[k];
        let jitter = cell_hash(cell);
        let s = textureSampleGrad(
            terrain_albedo, terrain_sampler, uv + jitter, layer, ddx_uv, ddy_uv);
        color += (sharp[k] / wsum) * s.rgb;
    }
    return color;
}

@fragment
fn fragment(
    in: SplatVertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // Feed the stock PBR input builder the standard fields, so the tint-only
    // path (far regime / splat disabled / untextured surface) is EXACTLY the
    // pre-872 pipeline: base white × vertex tint under standard lighting.
    var std_in: VertexOutput;
    std_in.position = in.position;
    std_in.world_position = in.world_position;
    std_in.world_normal = in.world_normal;
#ifdef VERTEX_UVS_A
    std_in.uv = in.uv;
#endif
#ifdef VERTEX_COLORS
    std_in.color = in.color;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    std_in.instance_index = in.instance_index;
#endif
    var pbr_input = pbr_input_from_standard_material(std_in, is_front);

    // WI 872 splat, before lighting. Gradients are taken up front (uniform
    // control flow); all sampling below uses textureSampleGrad, which is legal
    // in the divergent branches.
    let fade_start = splat.params.x;
    let fade_end = splat.params.y;
    let enabled = splat.params.z > 0.5;
    // Camera distance in view space (camera at origin by construction) — robust
    // against any floating-origin/world_position semantics.
    let dist = length((view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).xyz);
    let fade = smoothstep(fade_start, fade_end, dist);
    let ddx_uv = dpdx(in.terrain_uv);
    let ddy_uv = dpdy(in.terrain_uv);

    // Interpolated slot weights: renormalize (interpolation drifts the sum),
    // floor tiny slots, renormalize again. `tex_total` (the textured fraction:
    // 1 − marine/untextured weight) scales the whole effect, so seabeds stay
    // pure tint and beach→ocean frontiers feather.
    var w: array<f32, 8> = array<f32, 8>(
        in.splat_a.x, in.splat_a.y, in.splat_a.z, in.splat_a.w,
        in.splat_b.x, in.splat_b.y, in.splat_b.z, in.splat_b.w,
    );
    var tex_total = 0.0;
    for (var i = 0u; i < 8u; i += 1u) {
        tex_total += w[i];
    }
    let tint = pbr_input.material.base_color.rgb;

    if enabled && fade < 1.0 && tex_total > 0.02 {
        var norm = 0.0;
        for (var i = 0u; i < 8u; i += 1u) {
            w[i] = max(0.0, w[i] / tex_total - SPLAT_WEIGHT_FLOOR);
            norm += w[i];
        }
        // Degenerate spread (every slot under the floor) ⇒ stay on the tint
        // path rather than dividing by ~0.
        if norm > 1e-4 {

        // Pass 1: heights sharpen the kept weights (bounded — the cubic keeps
        // frontiers feathered, never stepped), and roughness/AO accumulate.
        var wh: array<f32, 8>;
        var height: array<f32, 8>;
        var rough_ao = vec2<f32>(0.0);
        var sharp_sum = 0.0;
        for (var i = 0u; i < 8u; i += 1u) {
            if w[i] > 0.0 {
                let layer = layer_of(i);
                let s = textureSampleGrad(
                    terrain_surface, terrain_sampler, in.terrain_uv, layer, ddx_uv, ddy_uv);
                let b = (w[i] / norm) * (0.4 + 0.6 * s.b);
                let bs = b * b * b;
                wh[i] = bs;
                sharp_sum += bs;
                height[i] = s.b;
                rough_ao += (w[i] / norm) * s.rg;
            } else {
                wh[i] = 0.0;
            }
        }

        // Pass 2: albedo (detiled) + tangent-space normal with the sharpened
        // weights; the anchor sum backs the macro-tint modulation.
        var albedo = vec3<f32>(0.0);
        var n_ts = vec3<f32>(0.0, 0.0, 0.0);
        var anchor = vec3<f32>(0.0);
        for (var i = 0u; i < 8u; i += 1u) {
            if wh[i] > 0.0 {
                let layer = layer_of(i);
                let ws = wh[i] / sharp_sum;
                albedo += ws * detiled_albedo(layer, in.terrain_uv, ddx_uv, ddy_uv);
                let ns = textureSampleGrad(
                    terrain_normal, terrain_sampler, in.terrain_uv, layer, ddx_uv, ddy_uv);
                n_ts += ws * (ns.rgb * 2.0 - 1.0);
                anchor += ws * splat.anchors[i].rgb;
            }
        }

        // Macro tint modulation: the WI 871 albedos are tone-anchored to their
        // consuming rows' tint means, so a plain tint multiply would double-
        // darken; dividing the tint by the blended anchor yields ≈1 where the
        // texture already matches the biome look and re-expresses the tint
        // where rows share a texture (highland vs alpine rock) — the macro
        // variation that breaks large-scale repetition.
        let macro_mod = clamp(tint / max(anchor, vec3<f32>(1e-3)), vec3<f32>(0.4), vec3<f32>(2.5));
        let near_rgb = albedo * macro_mod;

        // The textured fraction fades with distance and with the untextured
        // (marine) deficit; everything it drives converges to the tint-regime
        // constants together (albedo → tint, normal → geometric, roughness →
        // base 1.0, AO → 1.0) so the band cannot show as a lighting front.
        let mix_amount = clamp(tex_total, 0.0, 1.0) * (1.0 - fade);
        pbr_input.material.base_color =
            vec4<f32>(mix(tint, near_rgb, mix_amount), pbr_input.material.base_color.a);
        pbr_input.material.perceptual_roughness =
            mix(pbr_input.material.perceptual_roughness, clamp(rough_ao.x, 0.045, 1.0), mix_amount);
        pbr_input.diffuse_occlusion *= mix(1.0, clamp(rough_ao.y, 0.0, 1.0), mix_amount);

        // Screen-space cotangent frame (fragment-only, so no tangent attribute
        // and no sim mirror); the blended tangent-space normal is flattened
        // toward +Z by the same mix, then rotated into world space.
        let n_flat = normalize(mix(vec3<f32>(0.0, 0.0, 1.0), normalize(n_ts), mix_amount));
        let n_geo = normalize(pbr_input.N);
        let dp1 = dpdx(in.world_position.xyz);
        let dp2 = dpdy(in.world_position.xyz);
        let dp2perp = cross(dp2, n_geo);
        let dp1perp = cross(n_geo, dp1);
        let t = dp2perp * ddx_uv.x + dp1perp * ddy_uv.x;
        let b = dp2perp * ddx_uv.y + dp1perp * ddy_uv.y;
        let det = max(dot(t, t), dot(b, b));
        if det > 1e-12 {
            let invmax = inverseSqrt(det);
            let tbn = mat3x3<f32>(t * invmax, b * invmax, n_geo);
            pbr_input.N = normalize(tbn * n_flat);
        }
        }
    }

    var out: FragmentOutput;
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}

// Slot → global array layer (the per-body mapping in the splat uniform).
fn layer_of(slot: u32) -> u32 {
    if slot < 4u {
        return splat.layers_a[slot];
    }
    return splat.layers_b[slot - 4u];
}
