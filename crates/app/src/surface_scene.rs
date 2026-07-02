//! Procedural surface streaming scene (`-- surface`, WI 764).
//!
//! Makes WI 763's analytic [`SurfaceField`](sounding_sim::surface_field::SurfaceField)
//! **visible**: a generated body renders from orbit down to a coarse-to-fine,
//! crack-free procedural surface. A spherified-cube quadtree (headless
//! [`sounding_sim::surface_mesh`]) is traversed against the camera each frame;
//! near nodes split, far nodes merge; chunk geometry is meshed **off the main
//! thread** on the async compute pool and uploaded under a per-frame budget.
//! Skirts hide LOD seams. Floating origin + Bevy's HDR atmosphere give continuous
//! orbit-to-surface (the same proven camera as `planet.rs`). No physics yet — the
//! triangle geometry is a render artifact only (contact is WI 765).
//!
//! Run: `cargo run -p sounding -- surface [seed] [archetype]`.
//! Controls: `W/A/S/D` move, `R/F` up/down, arrows look, `F3` debug overlay,
//! `F4` telemetry box (FPS graph + streaming stats).
//!
//! Stale chunks are held until their replacement coverage is resident (a split
//! parent survives until its children upload; merged children until their parent
//! does), so LOD transitions never flash an uncovered hole.
//!
//! App-side only: all geometry/atmosphere math is the headless, unit-tested
//! `sounding_sim`; this scene is the Bevy/task-pool/entity/gizmo adapter.

use bevy::asset::RenderAssetUsages;
use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::DVec3;
use bevy::mesh::{
    Indices, MeshVertexAttribute, MeshVertexBufferLayoutRef, PrimitiveTopology, VertexFormat,
};
use bevy::pbr::{
    Atmosphere, AtmosphereMode, AtmosphereSettings, ExtendedMaterial, MaterialExtension,
    MaterialExtensionKey, MaterialExtensionPipeline, MaterialPlugin, ScatteringMedium,
};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;
use bevy::tasks::{block_on, futures_lite::future, AsyncComputeTaskPool, Task};
use std::collections::HashMap;

use sounding_sim::body_asset::BodyAsset;
use sounding_sim::bodygen::{generate, Archetype};
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::surface_field::SurfaceField;
use sounding_sim::surface_mesh::{
    build_chunk, morph_range, AtmosphereParams, ChunkMesh, QuadNode, DEFAULT_MAX_LEVEL,
    DEFAULT_RESOLUTION,
};
use sounding_sim::surface_scan::resident_leaves;

use crate::floating_origin::{
    render_translation, AnchorCamera, FloatingOrigin, FloatingOriginPlugin, WorldPlacement,
};
use crate::sparkline::{apply_panel, spawn_panel, SparkBar, Sparkline, SparklineLabel, SPARK_BARS};

/// Meshes uploaded to the world per frame (the bounded GPU-upload budget).
const UPLOAD_BUDGET: usize = 6;
/// New mesh-build tasks spawned per frame (bounds the async backlog).
const SPAWN_BUDGET: usize = 12;

/// Custom vertex attribute: each surface vertex's CDLOD morph target (its position on
/// the parent/coarser grid), consumed by the geomorph vertex shader at `@location(8)`.
const ATTRIBUTE_MORPH_TARGET: MeshVertexAttribute =
    MeshVertexAttribute::new("MorphTarget", 0x4d4f_5250_5447, VertexFormat::Float32x3);

/// StandardMaterial extended with a CDLOD geomorph vertex shader. The extension uniform
/// carries the per-level morph ramp `(start, end, _, _)` (camera-distance metres).
type SurfaceGeomorph = ExtendedMaterial<StandardMaterial, GeomorphExt>;

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone, Default)]
struct GeomorphExt {
    // Vertex-visible: the geomorph ramp is read in the vertex shader.
    #[uniform(100, visibility(vertex))]
    morph_range: Vec4,
}

impl MaterialExtension for GeomorphExt {
    fn vertex_shader() -> ShaderRef {
        "shaders/surface_geomorph.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Add the morph-target attribute (@location 8) alongside the standard ones so
        // the custom vertex shader's input matches the uploaded mesh layout.
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(2),
            ATTRIBUTE_MORPH_TARGET.at_shader_location(8),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}

/// The procedural-surface streaming scene.
pub struct SurfaceScenePlugin;

impl Plugin for SurfaceScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .add_plugins(MaterialPlugin::<SurfaceGeomorph>::default())
            .init_resource::<DebugOverlay>()
            .init_resource::<StreamStats>()
            .init_resource::<SurfaceTelemetry>()
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (
                    fly_camera,
                    stream_surface,
                    toggle_overlay,
                    draw_overlay,
                    update_telemetry,
                    hud,
                )
                    .chain(),
            );
    }
}

/// The body being rendered and its analytic field.
#[derive(Resource)]
struct SurfaceBody {
    asset: BodyAsset,
    field: SurfaceField,
    /// World placement of the body centre. Following `planet.rs`, the body sits so
    /// the approached surface is near the world origin with +Y up (Bevy's
    /// atmosphere convention): centre at `(0, -radius, 0)`.
    center_world: DVec3,
}

/// The surface material: a base `StandardMaterial` config plus one geomorph material
/// per LOD level (cached), so every chunk of a level shares an identical morph ramp
/// (same-level shared edges stay matched) and streaming does not leak material assets.
#[derive(Resource)]
struct SurfaceMat {
    base: StandardMaterial,
    by_level: HashMap<u32, Handle<SurfaceGeomorph>>,
}

/// Live (uploaded), in-flight (meshing), and built-but-not-yet-uploaded (ready)
/// chunk state, keyed by quadtree node. The `ready` queue holds completed builds
/// awaiting the per-frame upload budget, so finished work is never discarded.
#[derive(Resource, Default)]
struct ChunkStreamer {
    live: HashMap<QuadNode, Entity>,
    meshing: HashMap<QuadNode, Task<ChunkMesh>>,
    ready: Vec<(QuadNode, ChunkMesh)>,
}

/// Marks a streamed chunk entity (root-only despawn marker). The CDLOD morph factor is
/// computed per-vertex in the geomorph shader, so no per-chunk node data is needed here.
#[derive(Component)]
struct SurfaceChunk;

/// Whether the debug overlay (LOD wireframe + contact patch) is shown.
#[derive(Resource, Default)]
struct DebugOverlay(bool);

#[derive(Component)]
struct HudText;

/// Per-frame streaming statistics, published by `stream_surface` for the telemetry
/// overlay (and the HUD).
#[derive(Resource, Default)]
struct StreamStats {
    live: usize,
    meshing: usize,
    ready: usize,
    uploaded: usize,
}

/// Triangles per chunk (grid + skirt) at the default resolution — for the triangle
/// telemetry (all chunks use `DEFAULT_RESOLUTION`, so this is exact).
const TRIS_PER_CHUNK: usize =
    (DEFAULT_RESOLUTION * DEFAULT_RESOLUTION * 2 + DEFAULT_RESOLUTION * 4 * 2) as usize;

/// The scalar signals the surface telemetry box plots, in panel order.
#[derive(Clone, Copy)]
enum SurfStat {
    Fps,
    Chunks,
    Pending,
    Uploads,
    TrianglesK,
}

impl SurfStat {
    const ALL: [SurfStat; 5] = [
        SurfStat::Fps,
        SurfStat::Chunks,
        SurfStat::Pending,
        SurfStat::Uploads,
        SurfStat::TrianglesK,
    ];

    fn label(self) -> &'static str {
        match self {
            SurfStat::Fps => "fps",
            SurfStat::Chunks => "chunks",
            SurfStat::Pending => "pending",
            SurfStat::Uploads => "uploads/frame",
            SurfStat::TrianglesK => "tris x1000",
        }
    }

    /// A sane default visual max (the sparkline scale floor).
    fn default_max(self) -> f32 {
        match self {
            SurfStat::Fps => 60.0,
            SurfStat::Chunks => 300.0,
            SurfStat::Pending => SPAWN_BUDGET as f32,
            SurfStat::Uploads => UPLOAD_BUDGET as f32,
            SurfStat::TrianglesK => 400.0,
        }
    }

    fn sample(self, stats: &StreamStats, fps: f32) -> f32 {
        match self {
            SurfStat::Fps => fps,
            SurfStat::Chunks => stats.live as f32,
            SurfStat::Pending => (stats.meshing + stats.ready) as f32,
            SurfStat::Uploads => stats.uploaded as f32,
            SurfStat::TrianglesK => (stats.live * TRIS_PER_CHUNK) as f32 / 1000.0,
        }
    }
}

/// Sample the telemetry sparklines every Nth frame so the window spans a few
/// seconds (matches the cockpit overlay's cadence).
const TELEMETRY_SAMPLE_EVERY: u32 = 6;

/// The surface telemetry overlay state: one sample ring per [`SurfStat`].
#[derive(Resource)]
struct SurfaceTelemetry {
    sparks: Vec<Sparkline>,
}

impl Default for SurfaceTelemetry {
    fn default() -> Self {
        Self {
            sparks: SurfStat::ALL
                .iter()
                .map(|_| Sparkline::new(SPARK_BARS))
                .collect(),
        }
    }
}

/// The toggleable telemetry box container (visibility flipped by `F4`).
#[derive(Component)]
struct TelemetryRoot;

/// Parses `-- surface [seed] [archetype]` (archetype by name or index).
fn parse_args() -> (u64, Archetype) {
    let args: Vec<String> = std::env::args().collect();
    // args: [bin, "surface", maybe seed, maybe archetype]
    let after: Vec<&String> = args
        .iter()
        .skip_while(|a| *a != "surface")
        .skip(1)
        .collect();
    let seed = after.first().and_then(|s| s.parse().ok()).unwrap_or(7);
    let archetype = after
        .get(1)
        .and_then(|name| {
            Archetype::ALL
                .iter()
                .find(|a| {
                    a.slug().eq_ignore_ascii_case(name) || a.label().eq_ignore_ascii_case(name)
                })
                .copied()
        })
        .unwrap_or(Archetype::Moon);
    (seed, archetype)
}

fn body_tint(asset: &BodyAsset) -> Color {
    let m = &asset.fluid_medium;
    if m.ocean_surface_density > 0.0 {
        Color::srgb(0.35, 0.40, 0.45)
    } else if m.atmosphere_surface_density > 0.0 {
        Color::srgb(0.60, 0.50, 0.38)
    } else {
        Color::srgb(0.52, 0.52, 0.55)
    }
}

fn setup(mut commands: Commands, mut scattering: ResMut<Assets<ScatteringMedium>>) {
    let (seed, archetype) = parse_args();
    let asset = generate(seed, archetype);
    let field = SurfaceField::from_asset(&asset);
    let radius = asset.radius;
    let center_world = DVec3::new(0.0, -radius, 0.0);

    // Base surface material config; per-level geomorph materials are built on demand.
    let base_material = StandardMaterial {
        base_color: body_tint(&asset),
        perceptual_roughness: 1.0,
        ..default()
    };

    // Sun (raw pre-scattering sunlight, the atmosphere's input).
    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, 0.6, 0.0)),
    ));

    // HDR free-fly camera, floating-origin anchor. Starts at orbital altitude
    // looking down at the approached surface (world origin).
    let start_alt = (radius * 0.6).max(30_000.0);
    let mut cam = commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, start_alt as f32, (start_alt * 0.4) as f32)
            .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
        Exposure { ev100: 13.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        WorldPlacement(WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(0.0, start_alt, start_alt * 0.4),
        )),
        AnchorCamera,
    ));
    // Per-body atmosphere (R5, data-driven): airless bodies get no atmosphere.
    if let Some(a) = AtmosphereParams::from_asset(&asset) {
        cam.insert((
            Atmosphere {
                bottom_radius: a.bottom_radius,
                top_radius: a.top_radius,
                ground_albedo: Vec3::from_array(a.ground_albedo),
                medium: scattering.add(ScatteringMedium::default()),
            },
            // Raymarch the sky instead of the default lookup-texture path: the
            // sky-view LUT is singular at the zenith, so looking straight up
            // renders its texel grid as a "waffle". Raymarching integrates the
            // scattering numerically (Bevy's recommended mode for planets seen from
            // orbit), removing the artifact at a higher per-frame cost (watch the
            // F4 FPS graph).
            AtmosphereSettings {
                rendering_method: AtmosphereMode::Raymarched,
                ..default()
            },
            AtmosphereEnvironmentMapLight::default(),
        ));
    }

    // HUD.
    commands.spawn((
        Text::new(String::new()),
        HudText,
        TextFont {
            font_size: 15.0,
            ..default()
        },
        TextColor(Color::srgb(0.9, 0.93, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));

    // Telemetry box (top-right): FPS graph + streaming stats. `F4` toggles it.
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                right: Val::Px(12.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(2.0),
                padding: UiRect::all(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.45)),
            Visibility::Visible,
            TelemetryRoot,
        ))
        .with_children(|root| {
            root.spawn((
                Text::new("telemetry (F4)"),
                TextFont {
                    font_size: 12.0,
                    ..default()
                },
                TextColor(Color::srgb(0.6, 0.7, 0.8)),
            ));
            for (panel, sig) in SurfStat::ALL.iter().enumerate() {
                spawn_panel(root, panel, sig.label());
            }
        });

    commands.insert_resource(SurfaceBody {
        asset,
        field,
        center_world,
    });
    commands.insert_resource(SurfaceMat {
        base: base_material,
        by_level: HashMap::new(),
    });
    commands.insert_resource(ChunkStreamer::default());
}

/// Toggle the telemetry box (`F4`), then — decimated to a readable rate — sample
/// FPS (from the frame delta) and the streaming stats into their sparklines.
#[allow(clippy::too_many_arguments)]
fn update_telemetry(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    stats: Res<StreamStats>,
    mut telemetry: ResMut<SurfaceTelemetry>,
    mut root: Query<&mut Visibility, With<TelemetryRoot>>,
    mut bars: Query<(&SparkBar, &mut Node, &mut BackgroundColor)>,
    mut labels: Query<(&SparklineLabel, &mut Text), Without<HudText>>,
    mut frame: Local<u32>,
) {
    if keys.just_pressed(KeyCode::F4) {
        for mut vis in &mut root {
            *vis = match *vis {
                Visibility::Hidden => Visibility::Visible,
                _ => Visibility::Hidden,
            };
        }
    }

    *frame = frame.wrapping_add(1);
    if !frame.is_multiple_of(TELEMETRY_SAMPLE_EVERY) {
        return;
    }
    let dt = time.delta_secs();
    let fps = if dt > 1e-6 { 1.0 / dt } else { 0.0 };

    for (panel, sig) in SurfStat::ALL.iter().enumerate() {
        let v = sig.sample(&stats, fps);
        let spark = &mut telemetry.sparks[panel];
        spark.push(v);
        apply_panel(panel, &spark.bars(sig.default_max()), &mut bars);
        for (label, mut text) in &mut labels {
            if label.panel == panel {
                text.0 = format!(
                    "{}: {:.0} (max {:.0})",
                    sig.label(),
                    spark.latest(),
                    spark.window_max().max(sig.default_max())
                );
            }
        }
    }
}

/// The desired resident **leaf** set: traverse the six face roots, splitting where
/// the camera is close, and collect the leaves.
fn desired_leaves(camera_body: DVec3, radius: f64) -> Vec<QuadNode> {
    // Single source of truth with the headless seam scan (WI 785).
    resident_leaves(camera_body, radius, DEFAULT_MAX_LEVEL)
}

/// The CDLOD streaming step: compute desired leaves, enqueue new chunk builds
/// off-thread, upload completed ones under budget, and despawn merged/stale nodes.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn stream_surface(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mat: ResMut<SurfaceMat>,
    mut geomorph_materials: ResMut<Assets<SurfaceGeomorph>>,
    body: Res<SurfaceBody>,
    mut streamer: ResMut<ChunkStreamer>,
    mut stats: ResMut<StreamStats>,
    camera: Query<&WorldPlacement, With<AnchorCamera>>,
) {
    let Ok(cam) = camera.single() else {
        return;
    };
    let camera_body = cam.0.pos - body.center_world;
    let radius = body.field.radius();

    let desired: std::collections::HashSet<QuadNode> =
        desired_leaves(camera_body, radius).into_iter().collect();

    // 1. Poll in-flight builds once; move completed ones to the ready queue. Each
    //    task is polled to completion exactly once (its output is captured here).
    let in_flight: Vec<QuadNode> = streamer.meshing.keys().copied().collect();
    for node in in_flight {
        let done = streamer
            .meshing
            .get_mut(&node)
            .and_then(|task| block_on(future::poll_once(task)));
        if let Some(chunk) = done {
            streamer.meshing.remove(&node);
            streamer.ready.push((node, chunk));
        }
    }

    // 2. Upload ready chunks up to the per-frame budget; drop ones no longer
    //    desired; keep the overflow for next frame (built work is never wasted).
    let ready_items = std::mem::take(&mut streamer.ready);
    let mut keep = Vec::new();
    let mut uploaded = 0;
    for (node, chunk) in ready_items {
        if !desired.contains(&node) {
            continue;
        }
        if uploaded >= UPLOAD_BUDGET {
            keep.push((node, chunk));
            continue;
        }
        let mesh = to_bevy_mesh(&mut meshes, &chunk);
        let world = body.center_world + chunk.center;
        // Per-level geomorph material (built once per level, shared by all its chunks).
        let level = node.level;
        if !mat.by_level.contains_key(&level) {
            let (start, end) = morph_range(level, radius);
            let handle = geomorph_materials.add(SurfaceGeomorph {
                base: mat.base.clone(),
                extension: GeomorphExt {
                    morph_range: Vec4::new(start, end, 0.0, 0.0),
                },
            });
            mat.by_level.insert(level, handle);
        }
        let material = mat.by_level[&level].clone();
        let entity = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                Transform::default(),
                WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, world)),
                SurfaceChunk,
            ))
            .id();
        streamer.live.insert(node, entity);
        uploaded += 1;
    }
    streamer.ready = keep;

    // 3. Enqueue builds for desired nodes not already live/meshing/ready (bounded).
    let pool = AsyncComputeTaskPool::get();
    let mut spawned = 0;
    for &node in &desired {
        if spawned >= SPAWN_BUDGET {
            break;
        }
        if streamer.live.contains_key(&node)
            || streamer.meshing.contains_key(&node)
            || streamer.ready.iter().any(|(n, _)| *n == node)
        {
            continue;
        }
        let field = body.field; // Copy
        let task = pool.spawn(async move { build_chunk(&field, node, DEFAULT_RESOLUTION) });
        streamer.meshing.insert(node, task);
        spawned += 1;
    }

    // 4. Despawn stale live chunks — but only once their replacement coverage is
    //    resident. A stale node's area is exactly covered by the desired leaves that
    //    overlap it (desired leaves partition the sphere); holding it until all of
    //    those are live means a split parent survives until its children upload and
    //    merged children survive until their parent uploads, so no uncovered gap
    //    (the "window" pop) ever shows. Transient coarse+fine overlap is harmless.
    let live_nodes: std::collections::HashSet<QuadNode> = streamer.live.keys().copied().collect();
    let stale: Vec<QuadNode> = streamer
        .live
        .keys()
        .copied()
        .filter(|&n| !desired.contains(&n))
        .filter(|&n| {
            desired
                .iter()
                .filter(|&&d| n.overlaps(d))
                .all(|&d| live_nodes.contains(&d))
        })
        .collect();
    for node in stale {
        if let Some(entity) = streamer.live.remove(&node) {
            commands.entity(entity).despawn();
        }
    }

    // Publish streaming stats for the telemetry overlay.
    *stats = StreamStats {
        live: streamer.live.len(),
        meshing: streamer.meshing.len(),
        ready: streamer.ready.len(),
        uploaded,
    };
}

/// Builds a Bevy `Mesh` from the headless chunk buffers, including the CDLOD morph
/// target as a custom vertex attribute (`ATTRIBUTE_MORPH_TARGET`, `@location(8)`) that
/// the geomorph vertex shader blends toward per vertex.
fn to_bevy_mesh(meshes: &mut Assets<Mesh>, chunk: &ChunkMesh) -> Handle<Mesh> {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, chunk.positions.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, chunk.normals.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, chunk.uvs.clone());
    mesh.insert_attribute(ATTRIBUTE_MORPH_TARGET, chunk.morph_targets.clone());
    mesh.insert_indices(Indices::U32(chunk.indices.clone()));
    meshes.add(mesh)
}

/// Free-fly camera editing the f64 world placement (movement) and f32 rotation
/// (aim); speed scales with altitude above the surface. Mirrors `planet.rs`.
fn fly_camera(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    body: Res<SurfaceBody>,
    mut camera: Query<(&mut Transform, &mut WorldPlacement), With<AnchorCamera>>,
) {
    let Ok((mut tf, mut placement)) = camera.single_mut() else {
        return;
    };
    let dt = time.delta_secs();
    let rot_speed = 1.2;
    let (mut yaw, mut pitch) = (0.0, 0.0);
    if keys.pressed(KeyCode::ArrowLeft) {
        yaw += 1.0;
    }
    if keys.pressed(KeyCode::ArrowRight) {
        yaw -= 1.0;
    }
    if keys.pressed(KeyCode::ArrowUp) {
        pitch += 1.0;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        pitch -= 1.0;
    }
    tf.rotate_y(yaw * rot_speed * dt);
    tf.rotate_local_x(pitch * rot_speed * dt);

    let (forward, right) = (*tf.forward(), *tf.right());
    let mut dir = Vec3::ZERO;
    for (key, v) in [
        (KeyCode::KeyW, forward),
        (KeyCode::KeyS, -forward),
        (KeyCode::KeyD, right),
        (KeyCode::KeyA, -right),
        (KeyCode::KeyR, Vec3::Y),
        (KeyCode::KeyF, -Vec3::Y),
    ] {
        if keys.pressed(key) {
            dir += v;
        }
    }
    if dir != Vec3::ZERO {
        // Altitude above the surface along the camera's nadir direction.
        let cam_body = placement.0.pos - body.center_world;
        let alt = (cam_body.length() - body.field.radius()).max(1.0);
        let speed = (alt * 0.6).clamp(5.0, 2.0e6);
        let step = dir.normalize().as_dvec3() * speed * dt as f64;
        placement.0.pos += step;
    }
}

/// Toggles the debug overlay with `F3`.
fn toggle_overlay(keys: Res<ButtonInput<KeyCode>>, mut overlay: ResMut<DebugOverlay>) {
    if keys.just_pressed(KeyCode::F3) {
        overlay.0 = !overlay.0;
    }
}

/// Debug overlay (R7): resident chunk outlines coloured by LOD level, plus the
/// contact-patch sample (surface point + outward normal) under the camera nadir.
fn draw_overlay(
    overlay: Res<DebugOverlay>,
    origin: Res<FloatingOrigin>,
    body: Res<SurfaceBody>,
    streamer: Res<ChunkStreamer>,
    camera: Query<&WorldPlacement, With<AnchorCamera>>,
    mut gizmos: Gizmos,
) {
    if !overlay.0 {
        return;
    }
    let anchor = origin.0;
    let to_render = |world: DVec3| render_translation(body.center_world + world, anchor);

    // Chunk/LOD wireframe: draw each live node's four edges, coloured by level.
    for node in streamer.live.keys() {
        let color = lod_color(node.level);
        let [c00, c10, c11, c01] = node.corner_dirs();
        let r = body.field.radius();
        let pts = [c00 * r, c10 * r, c11 * r, c01 * r];
        for k in 0..4 {
            gizmos.line(to_render(pts[k]), to_render(pts[(k + 1) % 4]), color);
        }
    }

    // Contact-patch sample under the camera nadir (the seam WI 765 will bind to).
    if let Ok(cam) = camera.single() {
        let cam_body = cam.0.pos - body.center_world;
        let nadir = cam_body.normalize_or_zero();
        if nadir != DVec3::ZERO {
            let elev = body.field.elevation(nadir);
            let surf = nadir * (body.field.radius() + elev);
            let normal = body.field.normal(nadir);
            let p = to_render(surf);
            let tangent = normal.cross(DVec3::Y).normalize_or_zero();
            let t = tangent.as_vec3() * 200.0;
            gizmos.line(p - t, p + t, Color::srgb(1.0, 0.2, 0.2));
            gizmos.line(p, p + normal.as_vec3() * 400.0, Color::srgb(0.2, 1.0, 0.4));
        }
    }
}

/// A per-LOD-level wireframe colour (cycles through a small palette).
fn lod_color(level: u32) -> Color {
    const PALETTE: [Color; 6] = [
        Color::srgb(0.2, 0.6, 1.0),
        Color::srgb(0.2, 1.0, 0.6),
        Color::srgb(1.0, 0.9, 0.2),
        Color::srgb(1.0, 0.5, 0.2),
        Color::srgb(1.0, 0.3, 0.5),
        Color::srgb(0.8, 0.4, 1.0),
    ];
    PALETTE[(level as usize) % PALETTE.len()]
}

/// Updates the HUD each frame.
fn hud(
    body: Res<SurfaceBody>,
    overlay: Res<DebugOverlay>,
    streamer: Res<ChunkStreamer>,
    camera: Query<&WorldPlacement, With<AnchorCamera>>,
    mut text: Query<&mut Text, With<HudText>>,
) {
    let Ok(mut t) = text.single_mut() else {
        return;
    };
    let alt = camera
        .single()
        .map(|c| (c.0.pos - body.center_world).length() - body.field.radius())
        .unwrap_or(0.0);
    let atmo = if AtmosphereParams::from_asset(&body.asset).is_some() {
        "on"
    } else {
        "off (airless)"
    };
    **t = format!(
        "SURFACE (-- surface)\n\
         body: {name}\nradius: {rkm:.0} km\naltitude: {akm:.1} km\n\
         chunks: {live} live, {pend} meshing\natmosphere: {atmo}\n\n\
         W/A/S/D move \u{b7} R/F up/down \u{b7} arrows look \u{b7} F3 overlay [{ov}]",
        name = body.asset.name,
        rkm = body.field.radius() / 1000.0,
        akm = alt / 1000.0,
        live = streamer.live.len(),
        pend = streamer.meshing.len(),
        atmo = atmo,
        ov = if overlay.0 { "on" } else { "off" },
    );
}
