//! Land-and-drive on a generated cratered moon (`-- moon`, WI 765 — the aspect's
//! acceptance milestone made visible).
//!
//! A rover drives on the **analytic** procedural surface field (WI 763) via the
//! [`SurfacePatch`](sounding_sim::contact_surface::SurfacePatch) contact rebind
//! (WI 765), while the WI 764 spherified-cube streamer renders that same surface
//! textured around it. Physics queries the field, never the mesh — so the rover
//! can't fall through ungenerated ground and contact is independent of LOD/render
//! state; the mesh is purely what you see.
//!
//! The landing site is field-direction **+Y**, so the patch's local tangent frame
//! coincides with the render frame (body centre at `(0,-radius,0)`, landing point
//! at the world origin, +Y up — the `planet.rs`/`surface_scene.rs` convention), and
//! the rover lives in that frame near the origin (floating-origin-friendly).
//!
//! Controls: `W`/`S` throttle/reverse · `A`/`D` steer · `Space` brake · `P` pause.

use bevy::asset::RenderAssetUsages;
use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::DVec3;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::pbr::{Atmosphere, AtmosphereMode, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::tasks::{block_on, futures_lite::future, AsyncComputeTaskPool, Task};
use std::collections::HashMap;

use sounding_sim::bodygen::{generate, Archetype};
use sounding_sim::contact_surface::SurfacePatch;
use sounding_sim::powertrain::RoverPowertrain;
use sounding_sim::rover::{assemble_rover, Rover, SUBSTEP_DT};
use sounding_sim::sim::SimClock;
use sounding_sim::surface_field::SurfaceField;
use sounding_sim::surface_mesh::{
    build_chunk, should_split, AtmosphereParams, ChunkMesh, QuadNode, DEFAULT_MAX_LEVEL,
    DEFAULT_RESOLUTION,
};
use sounding_sim::voxel::{
    Material, Part, PartKind, RimSpec, SuspensionSpec, TireSpec, Voxel, VoxelCraft,
};

use crate::floating_origin::{
    render_translation, AnchorCamera, FloatingOrigin, FloatingOriginPlugin, WorldPlacement,
};

const UPLOAD_BUDGET: usize = 6;
const SPAWN_BUDGET: usize = 12;
const MAX_SUBSTEPS: u32 = 64;
/// Voxel cell size (m) — editor-scale light buggy (matches `-- rover`).
const CELL: f64 = 0.3;
const BRAKE_PER_KG: f64 = 35.0;
const STEER_LOCK: f64 = 0.35;
const STEER_RATE: f64 = 3.0;
const STEER_SPEED_REF: f64 = 7.0;

pub struct MoonScenePlugin;

impl Plugin for MoonScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (
                    crate::pause::toggle_pause,
                    crate::pause::step_scene,
                    drive_input,
                    step_rover,
                    stream_surface,
                    chase_camera,
                    draw_rover,
                    hud,
                )
                    .chain(),
            );
    }
}

/// The moon, its analytic field + landing patch, and the rover driving on it.
#[derive(Resource)]
struct MoonWorld {
    field: SurfaceField,
    patch: SurfacePatch,
    /// Body centre in world coordinates (`(0,-radius,0)`), so the landing site is at the world origin.
    body_center: DVec3,
    rover: Rover,
    drive: Vec<usize>,
    steer: Vec<usize>,
    powertrain: RoverPowertrain,
    throttle: f64,
    steer_input: f64,
    brake: f64,
    accumulator: f64,
}

#[derive(Resource)]
struct SurfaceMat(Handle<StandardMaterial>);

#[derive(Resource, Default)]
struct ChunkStreamer {
    live: HashMap<QuadNode, Entity>,
    meshing: HashMap<QuadNode, Task<ChunkMesh>>,
    ready: Vec<(QuadNode, ChunkMesh)>,
}

#[derive(Component)]
struct SurfaceChunk;

#[derive(Component)]
struct HudText;

/// Builds the editor-scale four-station buggy (mirrors `-- rover`).
fn build_buggy() -> VoxelCraft {
    let mut craft = VoxelCraft::new(CELL);
    for x in 0..3 {
        for z in 0..5 {
            craft.voxels.push(Voxel {
                cell: IVec3::new(x, 0, z),
                material: Material::COMPOSITE,
            });
        }
    }
    let (fx, fz, drop) = (3.0 * CELL, 5.0 * CELL, 0.2 * CELL);
    let mounts = [
        (DVec3::new(0.0, -drop, 0.0), false),
        (DVec3::new(fx, -drop, 0.0), false),
        (DVec3::new(0.0, -drop, fz), true),
        (DVec3::new(fx, -drop, fz), true),
    ];
    for (station, (mount, steer)) in mounts.into_iter().enumerate() {
        let id = station as u32;
        let wheel_mass = (8.0 * CELL).max(0.5);
        craft.parts.push(Part {
            mount,
            mass: wheel_mass,
            kind: PartKind::Rim(RimSpec {
                radius: 0.5 * CELL,
                drive: true,
                steer,
            }),
            station: Some(id),
        });
        craft.parts.push(Part {
            mount,
            mass: wheel_mass,
            kind: PartKind::Tire(TireSpec::new(CELL)),
            station: Some(id),
        });
        craft.parts.push(Part {
            mount,
            mass: (4.0 * CELL).max(0.3),
            kind: PartKind::Suspension(SuspensionSpec::for_cell_size(CELL)),
            station: Some(id),
        });
    }
    craft
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering: ResMut<Assets<ScatteringMedium>>,
) {
    let asset = generate(918_273_645, Archetype::Moon);
    let field = SurfaceField::from_asset(&asset);
    let radius = asset.radius;
    let g = (asset.mu / (radius * radius)).max(0.5);
    // Land at field-direction +Y so the patch frame == the render frame.
    let patch = SurfacePatch::new(field, DVec3::Y);
    let body_center = DVec3::new(0.0, -radius, 0.0);

    // Assemble the buggy just above the surface, in the patch's local frame.
    let craft = build_buggy();
    let ground = {
        use sounding_sim::contact_surface::ContactSurface;
        patch.height(0.0, 0.0)
    };
    let asm = assemble_rover(&craft, DVec3::new(0.0, ground + 3.0 * CELL, 0.0), g)
        .expect("buggy has four complete wheel stations");

    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.52, 0.52, 0.55),
        perceptual_roughness: 1.0,
        ..default()
    });

    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.6, 0.5, 0.0)),
    ));

    // Chase camera: floating-origin anchor + HDR + per-body atmosphere.
    let mut cam = commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 8.0 * CELL as f32, -16.0 * CELL as f32)
            .looking_at(Vec3::new(0.0, CELL as f32, 4.0 * CELL as f32), Vec3::Y),
        Exposure { ev100: 13.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        WorldPlacement(sounding_sim::frame::WorldPos::new(
            sounding_sim::frame::FrameId::CENTRAL_BODY,
            DVec3::new(0.0, 8.0 * CELL, -16.0 * CELL),
        )),
        AnchorCamera,
    ));
    if let Some(a) = AtmosphereParams::from_asset(&asset) {
        cam.insert((
            Atmosphere {
                bottom_radius: a.bottom_radius,
                top_radius: a.top_radius,
                ground_albedo: Vec3::from_array(a.ground_albedo),
                medium: scattering.add(ScatteringMedium::default()),
            },
            AtmosphereSettings {
                rendering_method: AtmosphereMode::Raymarched,
                ..default()
            },
            AtmosphereEnvironmentMapLight::default(),
        ));
    }

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

    commands.insert_resource(MoonWorld {
        field,
        patch,
        body_center,
        rover: asm.rover,
        drive: asm.drive,
        steer: asm.steer,
        powertrain: asm.powertrain,
        throttle: 0.0,
        steer_input: 0.0,
        brake: 0.0,
        accumulator: 0.0,
    });
    commands.insert_resource(SurfaceMat(material));
    commands.insert_resource(ChunkStreamer::default());
}

/// World position of a rover local-frame point (patch frame → body-centred → world).
fn rover_world(world: &MoonWorld, local: DVec3) -> DVec3 {
    world.body_center + world.patch.local_to_world(local)
}

fn drive_input(time: Res<Time>, keys: Res<ButtonInput<KeyCode>>, mut world: ResMut<MoonWorld>) {
    world.throttle = if keys.pressed(KeyCode::KeyW) {
        1.0
    } else if keys.pressed(KeyCode::KeyS) {
        -1.0
    } else {
        0.0
    };
    world.brake = if keys.pressed(KeyCode::Space) {
        world.rover.body.mass * BRAKE_PER_KG
    } else {
        0.0
    };
    let target = if keys.pressed(KeyCode::KeyA) {
        1.0
    } else if keys.pressed(KeyCode::KeyD) {
        -1.0
    } else {
        0.0
    };
    let step = STEER_RATE * time.delta_secs_f64();
    world.steer_input += (target - world.steer_input).clamp(-step, step);
    let speed = world.rover.body.velocity.length();
    let max_angle = STEER_LOCK / (1.0 + speed / STEER_SPEED_REF);
    let steer = world.steer.clone();
    let steer_input = world.steer_input;
    world.rover.set_steer(steer_input, max_angle, &steer);
}

fn step_rover(time: Res<Time>, mut clock: ResMut<SimClock>, mut world: ResMut<MoonWorld>) {
    let Some(dt) = crate::pause::frame_step_dt(&mut clock, &time) else {
        return;
    };
    world.accumulator += dt;
    let mut substeps = 0;
    let drive = world.drive.clone();
    while world.accumulator >= SUBSTEP_DT && substeps < MAX_SUBSTEPS {
        let throttle = world.throttle;
        let torque = world.powertrain.drive_torque(throttle, SUBSTEP_DT);
        let brake = world.brake;
        for (i, w) in world.rover.wheels.iter_mut().enumerate() {
            w.drive_torque = if drive.contains(&i) { torque } else { 0.0 };
            w.brake = brake;
        }
        let patch = world.patch;
        world.rover.step(&patch, SUBSTEP_DT);
        world.accumulator -= SUBSTEP_DT;
        substeps += 1;
    }
}

/// Streams the textured surface around the rover (CDLOD, off-thread meshing, budgeted upload) — the
/// same machine as `-- surface`, anchored to the rover's world position.
#[allow(clippy::type_complexity)]
fn stream_surface(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    world: Res<MoonWorld>,
    mat: Res<SurfaceMat>,
    mut streamer: ResMut<ChunkStreamer>,
) {
    let radius = world.field.radius();
    // Camera-of-interest for LOD = the rover, in body-centred coordinates.
    let rover_body = world.patch.local_to_world(world.rover.body.position);

    let desired: std::collections::HashSet<QuadNode> = {
        let mut leaves = Vec::new();
        let mut stack: Vec<QuadNode> = QuadNode::roots().to_vec();
        while let Some(node) = stack.pop() {
            if should_split(node, rover_body, radius, DEFAULT_MAX_LEVEL) {
                stack.extend_from_slice(&node.children());
            } else {
                leaves.push(node);
            }
        }
        leaves.into_iter().collect()
    };

    // Poll builds → ready queue.
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

    // Upload under budget.
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
        let entity = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(mat.0.clone()),
                Transform::default(),
                WorldPlacement(sounding_sim::frame::WorldPos::new(
                    sounding_sim::frame::FrameId::CENTRAL_BODY,
                    world.body_center + chunk.center,
                )),
                SurfaceChunk,
            ))
            .id();
        streamer.live.insert(node, entity);
        uploaded += 1;
    }
    streamer.ready = keep;

    // Enqueue new builds.
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
        let field = world.field;
        let task = pool.spawn(async move { build_chunk(&field, node, DEFAULT_RESOLUTION) });
        streamer.meshing.insert(node, task);
        spawned += 1;
    }

    // Coverage-gated despawn (no LOD-transition holes; WI 771).
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
        if let Some(e) = streamer.live.remove(&node) {
            commands.entity(e).despawn();
        }
    }
}

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

/// Chase camera: follows the rover from behind-and-above along its heading and looks at it. Position is
/// written to the `WorldPlacement` (the floating origin sets the render translation); rotation is set
/// here (floating origin leaves it untouched).
fn chase_camera(
    world: Res<MoonWorld>,
    mut cam: Query<(&mut WorldPlacement, &mut Transform), With<AnchorCamera>>,
) {
    let rw = rover_world(&world, world.rover.body.position);
    // The patch frame == the render/world frame here (landing at +Y), so the rover's local forward is
    // the world forward. Sit behind-and-above it and look at it.
    let fwd = (world.rover.body.orientation * DVec3::Z).normalize_or_zero();
    let fwd = if fwd == DVec3::ZERO { DVec3::Z } else { fwd };
    let cam_world = rw - fwd * (16.0 * CELL) + DVec3::Y * (8.0 * CELL);
    if let Ok((mut wp, mut tf)) = cam.single_mut() {
        wp.0.pos = cam_world;
        tf.look_to((rw - cam_world).as_vec3(), Vec3::Y);
    }
}

/// Draws the rover (chassis + wheels) with gizmos, in floating-origin render space.
fn draw_rover(mut gizmos: Gizmos, world: Res<MoonWorld>, origin: Res<FloatingOrigin>) {
    let anchor = origin.0;
    let to_render = |body_local: DVec3| render_translation(rover_world(&world, body_local), anchor);
    let body = &world.rover.body;
    // Chassis (oriented box). The rover orientation is in the patch/local frame == render frame.
    let q = body.orientation.as_quat();
    gizmos.primitive_3d(
        &Cuboid::new(3.0 * CELL as f32, 1.0 * CELL as f32, 5.0 * CELL as f32),
        bevy::math::Isometry3d::new(to_render(body.position), q),
        Color::srgb(0.80, 0.81, 0.86),
    );
    // Wheels.
    for w in &world.rover.wheels {
        if w.inert {
            continue;
        }
        let hub = body.position + body.orientation * w.mount;
        let axle = DVec3::new(hub.x, hub.y - w.axle_drop, hub.z);
        gizmos.line(to_render(hub), to_render(axle), Color::srgb(0.5, 0.5, 0.55));
        gizmos.sphere(
            bevy::math::Isometry3d::from_translation(to_render(axle)),
            w.radius as f32,
            Color::srgb(0.12, 0.12, 0.15),
        );
    }
}

fn hud(world: Res<MoonWorld>, mut text: Query<&mut Text, With<HudText>>) {
    let Ok(mut t) = text.single_mut() else {
        return;
    };
    use sounding_sim::contact_surface::ContactSurface;
    let speed = world.rover.body.velocity.length();
    let p = world.rover.body.position;
    let clearance = p.y - world.patch.height(p.x, p.z);
    **t = format!(
        "MOON (-- moon) — land & drive on the analytic surface\n\
         speed: {speed:5.1} m/s\nclearance: {clearance:5.2} m\n\n\
         W/S throttle \u{b7} A/D steer \u{b7} Space brake \u{b7} P pause",
    );
}
