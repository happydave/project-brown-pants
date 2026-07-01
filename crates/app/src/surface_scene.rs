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
//! Controls: `W/A/S/D` move, `R/F` up/down, arrows look, `F3` toggle debug overlay.
//!
//! App-side only: all geometry/atmosphere math is the headless, unit-tested
//! `sounding_sim`; this scene is the Bevy/task-pool/entity/gizmo adapter.

use bevy::asset::RenderAssetUsages;
use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::DVec3;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::tasks::{block_on, futures_lite::future, AsyncComputeTaskPool, Task};
use std::collections::HashMap;

use sounding_sim::body_asset::BodyAsset;
use sounding_sim::bodygen::{generate, Archetype};
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::surface_field::SurfaceField;
use sounding_sim::surface_mesh::{
    build_chunk, should_split, AtmosphereParams, ChunkMesh, QuadNode, DEFAULT_MAX_LEVEL,
    DEFAULT_RESOLUTION,
};

use crate::floating_origin::{
    render_translation, AnchorCamera, FloatingOrigin, FloatingOriginPlugin, WorldPlacement,
};

/// Meshes uploaded to the world per frame (the bounded GPU-upload budget).
const UPLOAD_BUDGET: usize = 6;
/// New mesh-build tasks spawned per frame (bounds the async backlog).
const SPAWN_BUDGET: usize = 12;

/// The procedural-surface streaming scene.
pub struct SurfaceScenePlugin;

impl Plugin for SurfaceScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .init_resource::<DebugOverlay>()
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (
                    fly_camera,
                    stream_surface,
                    toggle_overlay,
                    draw_overlay,
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

/// The shared surface material (relief comes from per-vertex normals).
#[derive(Resource)]
struct SurfaceMat(Handle<StandardMaterial>);

/// Live (uploaded), in-flight (meshing), and built-but-not-yet-uploaded (ready)
/// chunk state, keyed by quadtree node. The `ready` queue holds completed builds
/// awaiting the per-frame upload budget, so finished work is never discarded.
#[derive(Resource, Default)]
struct ChunkStreamer {
    live: HashMap<QuadNode, Entity>,
    meshing: HashMap<QuadNode, Task<ChunkMesh>>,
    ready: Vec<(QuadNode, ChunkMesh)>,
}

/// Marks a streamed chunk entity (root-only despawn marker).
#[derive(Component)]
struct SurfaceChunk;

/// Whether the debug overlay (LOD wireframe + contact patch) is shown.
#[derive(Resource, Default)]
struct DebugOverlay(bool);

#[derive(Component)]
struct HudText;

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

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering: ResMut<Assets<ScatteringMedium>>,
) {
    let (seed, archetype) = parse_args();
    let asset = generate(seed, archetype);
    let field = SurfaceField::from_asset(&asset);
    let radius = asset.radius;
    let center_world = DVec3::new(0.0, -radius, 0.0);

    let material = materials.add(StandardMaterial {
        base_color: body_tint(&asset),
        perceptual_roughness: 1.0,
        ..default()
    });

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
            AtmosphereSettings::default(),
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

    commands.insert_resource(SurfaceBody {
        asset,
        field,
        center_world,
    });
    commands.insert_resource(SurfaceMat(material));
    commands.insert_resource(ChunkStreamer::default());
}

/// The desired resident **leaf** set: traverse the six face roots, splitting where
/// the camera is close, and collect the leaves.
fn desired_leaves(camera_body: DVec3, radius: f64) -> Vec<QuadNode> {
    let mut leaves = Vec::new();
    let mut stack: Vec<QuadNode> = QuadNode::roots().to_vec();
    while let Some(node) = stack.pop() {
        if should_split(node, camera_body, radius, DEFAULT_MAX_LEVEL) {
            stack.extend_from_slice(&node.children());
        } else {
            leaves.push(node);
        }
    }
    leaves
}

/// The CDLOD streaming step: compute desired leaves, enqueue new chunk builds
/// off-thread, upload completed ones under budget, and despawn merged/stale nodes.
#[allow(clippy::type_complexity)]
fn stream_surface(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    body: Res<SurfaceBody>,
    mat: Res<SurfaceMat>,
    mut streamer: ResMut<ChunkStreamer>,
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
        let entity = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(mat.0.clone()),
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

    // 4. Despawn live chunks that are no longer desired (merged or out of range).
    let stale: Vec<QuadNode> = streamer
        .live
        .keys()
        .copied()
        .filter(|n| !desired.contains(n))
        .collect();
    for node in stale {
        if let Some(entity) = streamer.live.remove(&node) {
            commands.entity(entity).despawn();
        }
    }
}

/// Builds a Bevy `Mesh` from the headless chunk buffers.
fn to_bevy_mesh(meshes: &mut Assets<Mesh>, chunk: &ChunkMesh) -> Handle<Mesh> {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, chunk.positions.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, chunk.normals.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, chunk.uvs.clone());
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
