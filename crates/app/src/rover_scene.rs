//! Toy 6 rover scene (WI 506; modernised WI 641).
//!
//! Drives the headless rover (`sounding_sim::rover`) over the analytic terrain and
//! visualizes it. Rendering is **rover-anchored floating origin**: the rover sits
//! at a large f64 world offset (so contact stability is exercised away from the
//! origin) while everything is drawn relative to the rover, keeping f32 render
//! coordinates near zero. The wheels write a track trail.
//!
//! The rover is built through the **same `assemble_rover` component path as the
//! workshop Test** (WI 641): four `OffRoad`-preset wheel stations (rim + tire +
//! suspension) on a voxel chassis, so it exercises the current drivetrain —
//! quarter-car unsprung mass (WI 631a), tire grip/slip presets (WI 630), the
//! failure ladder (WI 631b), drive/steer groups, and the powertrain (WI 609) — and
//! its WI 640 telemetry reports live `axle_drop`/`static_load`/`grip_scale`. This
//! stays a minimal fixed-rover sandbox: no build/edit UI, gizmo render.
//!
//! Controls: `W`/`S` throttle/reverse · `A`/`D` steer · `Space` brake · `P` pause.

use crate::bus::GroundedRover;
use crate::editor::WheelPreset;
use bevy::math::{DVec3, Isometry3d};
use bevy::prelude::*;
use sounding_sim::powertrain::RoverPowertrain;
use sounding_sim::rover::{assemble_rover, Rover, SUBSTEP_DT};
use sounding_sim::sim::SimClock;
use sounding_sim::telemetry::RoverTelemetry;
use sounding_sim::terrain::Terrain;
use sounding_sim::voxel::{Material, Part, PartKind, SuspensionSpec, Voxel, VoxelCraft};

/// Maximum physics sub-steps per frame (keeps up at 60 fps with headroom).
const MAX_SUBSTEPS: u32 = 64;
const MAX_TRACK: usize = 600;

/// Rover acceleration gravity (m/s²).
const ROVER_GRAVITY: f64 = 9.81;
/// Voxel cell size (m) of the fixed sandbox chassis (WI 641). Editor-scale so the rover is a light
/// buggy, not a 3-tonne solid-composite brick that the hills throw around. Wheel size, mounts, spawn,
/// and the chase camera all scale from this.
const ROVER_CELL: f64 = 0.3;
// Drive feel — mirrors the workshop Test (`workshop_scene.rs`) so the standalone scene drives the same.
/// Brake torque per kg of rover mass (so braking scales with the build).
const ROVER_BRAKE_PER_KG: f64 = 35.0;
/// Steering lock (rad) at a standstill; tapers with speed (see [`STEER_SPEED_REF`]).
const ROVER_STEER: f64 = 0.35;
/// Steer-input slew rate (per second) — a tap is a small correction, not instant full lock.
const STEER_RATE: f64 = 3.0;
/// Speed (m/s) at which steering authority halves: `ROVER_STEER / (1 + v/ref)`.
const STEER_SPEED_REF: f64 = 7.0;

/// The rover, its drivetrain groups + powertrain, drive intent, terrain, track trail, and accumulator.
#[derive(Resource)]
struct RoverWorld {
    rover: Rover,
    /// Wheel indices that receive drive torque (from the assembly).
    drive: Vec<usize>,
    /// Wheel indices that turn with steering (from the assembly).
    steer: Vec<usize>,
    /// Drive power source derived from the build (a self-sustaining default here).
    powertrain: RoverPowertrain,
    /// Throttle intent in [-1, 1]; the powertrain turns it into torque each frame.
    throttle: f64,
    /// Smoothed steering input in [-1, 1].
    steer_input: f64,
    /// Brake torque magnitude (N·m) applied to all wheels.
    brake: f64,
    terrain: Terrain,
    track: Vec<DVec3>,
    accumulator: f64,
    record: u32,
}

impl RoverWorld {
    fn new() -> Self {
        // Long rolling hills: gentle enough to build speed, but at speed the rover
        // still catches air over the crests (v²·curvature ≫ g).
        let terrain = Terrain {
            amplitude: 0.7,
            wavelength: 55.0,
            ..Default::default()
        };
        // Place the rover at a large world offset so rendering rebases and contact
        // stability is exercised away from the origin (the kraken condition).
        let (ox, oz) = (6_378_000.0, -1_200_000.0);
        // Editor-scale cell (WI 641): a solid COMPOSITE 3×5 slab at 0.5 m is a ~3-tonne brick that
        // catches air on the hills and slams down (the heavy-fixture instability our convention warns
        // about). A 0.3 m cell makes a ~0.65-tonne buggy — a stable demonstrator of the current
        // drivetrain — while everything below stays cell-relative.
        let cell = ROVER_CELL;
        let mut craft = VoxelCraft::new(cell);
        for x in 0..3 {
            for z in 0..5 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        // Four wheel stations (rim + tire + suspension), the OffRoad preset — the same component path
        // the workshop authors (WI 630/641). Mount at the four bottom corners of the slab footprint
        // (lattice metres: x∈[0, 3·cell], z∈[0, 5·cell]), just below it. The +z (front) pair steers;
        // all drive.
        let preset = WheelPreset::OffRoad;
        let (fx, fz, drop) = (3.0 * cell, 5.0 * cell, 0.2 * cell);
        let mounts = [
            (DVec3::new(0.0, -drop, 0.0), false),
            (DVec3::new(fx, -drop, 0.0), false),
            (DVec3::new(0.0, -drop, fz), true),
            (DVec3::new(fx, -drop, fz), true),
        ];
        for (station, (mount, steer)) in mounts.into_iter().enumerate() {
            let id = station as u32;
            let wheel_mass = (8.0 * cell).max(0.5);
            craft.parts.push(Part {
                mount,
                mass: wheel_mass,
                kind: PartKind::Rim(preset.rim(cell, steer)),
                station: Some(id),
            });
            craft.parts.push(Part {
                mount,
                mass: wheel_mass,
                kind: PartKind::Tire(preset.tire(cell)),
                station: Some(id),
            });
            craft.parts.push(Part {
                mount,
                mass: (4.0 * cell).max(0.3),
                kind: PartKind::Suspension(SuspensionSpec::for_cell_size(cell)),
                station: Some(id),
            });
        }
        // Assemble at the CoM-relative spawn: drop a little so the rover settles on its suspension.
        // Clearance is cell-relative so it tracks the build scale.
        let ground = terrain.height(ox, oz);
        let spawn = DVec3::new(ox, ground + 3.0 * cell, oz);
        let asm = assemble_rover(&craft, spawn, ROVER_GRAVITY)
            .expect("the fixed rover craft has voxels + four complete wheel stations");
        Self {
            rover: asm.rover,
            drive: asm.drive,
            steer: asm.steer,
            powertrain: asm.powertrain,
            throttle: 0.0,
            steer_input: 0.0,
            brake: 0.0,
            terrain,
            track: Vec::new(),
            accumulator: 0.0,
            record: 0,
        }
    }
}

/// Marks the heads-up readout text.
#[derive(Component)]
struct Hud;

pub struct RoverScenePlugin;

impl Plugin for RoverScenePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(RoverWorld::new())
            .add_systems(Startup, setup_view)
            .add_systems(
                Update,
                (
                    crate::pause::toggle_pause,
                    crate::pause::step_scene,
                    drive_input,
                    step_rover,
                    publish_rover,
                    draw_rover,
                    update_hud,
                )
                    .chain(),
            );
    }
}

fn setup_view(mut commands: Commands) {
    // The rover is rendered at the origin (rover-anchored), so a fixed chase
    // camera behind and above it works without tracking. The framing was tuned at a 0.5 m cell, so
    // scale it to the current cell to keep the (smaller, editor-scale) rover framed (WI 641).
    let s = (ROVER_CELL / 0.5) as f32;
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 7.0 * s, -16.0 * s)
            .looking_at(Vec3::new(0.0, 1.0 * s, 4.0 * s), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            ..default()
        },
        Transform::from_xyz(6.0, 14.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    // Heads-up speed / height readout.
    commands.spawn((
        Text::new("speed:   0.0 m/s\nheight:  0.00 m"),
        TextFont {
            font_size: 20.0,
            ..default()
        },
        TextColor(Color::srgb(0.9, 0.95, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(12.0),
            ..default()
        },
        Hud,
    ));
}

fn update_hud(world: Res<RoverWorld>, clock: Res<SimClock>, mut hud: Query<&mut Text, With<Hud>>) {
    if let Ok(mut text) = hud.single_mut() {
        let speed = world.rover.body.velocity.length();
        let height = world.rover.height_above_terrain(&world.terrain);
        let paused = crate::pause::paused_banner(&clock);
        text.0 = format!("speed: {speed:5.1} m/s\nheight: {height:5.2} m{paused}");
    }
}

/// Set the drive **intent** by group (WI 641, mirroring the workshop Test): throttle the drive wheels,
/// steer the steer wheels (speed-sensitive, smoothed), brake all. The powertrain turns throttle into
/// torque in `step_rover`; `set_steer` applies coordinated counter-steer (WI 616).
fn drive_input(time: Res<Time>, keys: Res<ButtonInput<KeyCode>>, mut world: ResMut<RoverWorld>) {
    world.throttle = if keys.pressed(KeyCode::KeyW) {
        1.0
    } else if keys.pressed(KeyCode::KeyS) {
        -1.0
    } else {
        0.0
    };
    world.brake = if keys.pressed(KeyCode::Space) {
        world.rover.body.mass * ROVER_BRAKE_PER_KG
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
    let max_angle = ROVER_STEER / (1.0 + speed / STEER_SPEED_REF);
    let steer = world.steer.clone();
    let steer_input = world.steer_input;
    world.rover.set_steer(steer_input, max_angle, &steer);
}

fn step_rover(time: Res<Time>, mut clock: ResMut<SimClock>, mut world: ResMut<RoverWorld>) {
    // Paused (WI 638): freeze the rover physics; the accumulator does not grow, so resume is jump-free.
    // While paused, a step (WI 643) advances a bounded chunk; `frame_step_dt` returns `None` to freeze.
    let Some(dt) = crate::pause::frame_step_dt(&mut clock, &time) else {
        return;
    };
    world.accumulator += dt;
    let mut substeps = 0;
    // The drive group is fixed while stepping (no obstacles shear a wheel in this scene), so clone once.
    let drive = world.drive.clone();
    while world.accumulator >= SUBSTEP_DT && substeps < MAX_SUBSTEPS {
        let terrain = world.terrain;
        // The powertrain turns throttle into (default self-sustaining) drive torque each step (WI 609).
        let throttle = world.throttle;
        let torque = world.powertrain.drive_torque(throttle, SUBSTEP_DT);
        let brake = world.brake;
        for (i, w) in world.rover.wheels.iter_mut().enumerate() {
            w.drive_torque = if drive.contains(&i) { torque } else { 0.0 };
            w.brake = brake;
        }
        world.rover.step(&terrain, SUBSTEP_DT);
        world.accumulator -= SUBSTEP_DT;
        substeps += 1;
        // Record a track point under the rear axle every so often.
        world.record += 1;
        if world.record.is_multiple_of(16) {
            let p = world.rover.body.position;
            let g = world.terrain.height(p.x, p.z);
            world.track.push(DVec3::new(p.x, g, p.z));
            if world.track.len() > MAX_TRACK {
                world.track.remove(0);
            }
        }
    }
}

/// Publishes the rover's live state onto the bus bridge each frame (WI 640), so
/// `GET /telemetry` and the dev-MCP bridge can introspect the running rover.
fn publish_rover(world: Res<RoverWorld>, mut grounded: ResMut<GroundedRover>) {
    grounded.0 = Some(RoverTelemetry::from_rover(&world.rover));
}

fn draw_rover(mut gizmos: Gizmos, world: Res<RoverWorld>) {
    let anchor = world.rover.body.position;
    let to_render = |w: DVec3| (w - anchor).as_vec3();
    let terrain = &world.terrain;

    // Terrain wireframe grid around the rover (origin-local tiles of the analytic
    // surface — the same function the wheels query).
    let step = 3.0;
    let n = 14;
    let grid_color = Color::srgb(0.30, 0.26, 0.22);
    for i in -n..=n {
        let mut row = Vec::new();
        let mut col = Vec::new();
        for j in -n..=n {
            let (xi, zj) = (anchor.x + i as f64 * step, anchor.z + j as f64 * step);
            let (xj, zi) = (anchor.x + j as f64 * step, anchor.z + i as f64 * step);
            row.push(to_render(DVec3::new(xi, terrain.height(xi, zj), zj)));
            col.push(to_render(DVec3::new(xj, terrain.height(xj, zi), zi)));
        }
        gizmos.linestrip(row, grid_color);
        gizmos.linestrip(col, grid_color);
    }

    // Track trail the wheels wrote.
    if world.track.len() > 1 {
        gizmos.linestrip(
            world.track.iter().map(|p| to_render(*p)),
            Color::srgb(0.9, 0.7, 0.2),
        );
    }

    // Chassis.
    let body = &world.rover.body;
    let chassis_q = body.orientation.as_quat();
    gizmos.primitive_3d(
        &Cuboid::new(1.5, 0.5, 2.5),
        Isometry3d::new(to_render(body.position), chassis_q),
        Color::srgb(0.80, 0.81, 0.86),
    );

    // Wheels and suspension legs. Each wheel renders at its **true axle height** (`hub.y − axle_drop`
    // on the quarter-car path, WI 631a/641) so suspension travel and hop are visible; a sheared wheel
    // (inert) is skipped. The leg runs from the hub down to the axle.
    for w in &world.rover.wheels {
        if w.inert {
            continue;
        }
        let hub = body.position + body.orientation * w.mount;
        let axle = DVec3::new(hub.x, hub.y - w.axle_drop, hub.z);
        let wheel_color = if w.tire_blown || w.rim_bent {
            Color::srgb(0.45, 0.20, 0.15) // damaged corner reads warmer
        } else {
            Color::srgb(0.15, 0.15, 0.18)
        };
        gizmos.line(to_render(hub), to_render(axle), Color::srgb(0.5, 0.5, 0.55));
        gizmos.sphere(
            Isometry3d::from_translation(to_render(axle)),
            w.radius as f32,
            wheel_color,
        );
    }
}
