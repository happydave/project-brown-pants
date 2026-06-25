//! Toy 6 rover scene (WI 506).
//!
//! Drives the headless rover (`sounding_sim::rover`) over the analytic terrain and
//! visualizes it. Rendering is **rover-anchored floating origin**: the rover sits
//! at a large f64 world offset (so contact stability is exercised away from the
//! origin) while everything is drawn relative to the rover, keeping f32 render
//! coordinates near zero. The wheels write a track trail.
//!
//! Controls: `W`/`S` throttle/reverse · `A`/`D` steer · `Space` brake.

use crate::bus::GroundedRover;
use bevy::math::{DVec3, Isometry3d};
use bevy::prelude::*;
use sounding_sim::active::ActiveBody;
use sounding_sim::rover::{Rover, Wheel, SUBSTEP_DT};
use sounding_sim::sim::SimClock;
use sounding_sim::telemetry::RoverTelemetry;
use sounding_sim::terrain::Terrain;
use sounding_sim::voxel::{Material, Voxel, VoxelCraft};

/// Maximum physics sub-steps per frame (keeps up at 60 fps with headroom).
const MAX_SUBSTEPS: u32 = 64;
const MAX_TRACK: usize = 600;

/// The rover, its terrain, the track trail, and the sub-step accumulator.
#[derive(Resource)]
struct RoverWorld {
    rover: Rover,
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
        let mut craft = VoxelCraft::new(0.5);
        for x in 0..3 {
            for z in 0..5 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        let mp = craft.mass_properties().unwrap();
        let ground = terrain.height(ox, oz);
        // Spawn essentially on the ground (a 1 m drop lands violently on the short
        // suspension and can kick the rover into a spin).
        let body =
            ActiveBody::from_mass_properties(DVec3::new(ox, ground + 0.9, oz), DVec3::ZERO, &mp);
        let wheels = vec![
            Wheel::new(DVec3::new(-1.0, -0.2, -2.0)),
            Wheel::new(DVec3::new(1.0, -0.2, -2.0)),
            Wheel::new(DVec3::new(-1.0, -0.2, 2.0)),
            Wheel::new(DVec3::new(1.0, -0.2, 2.0)),
        ];
        Self {
            rover: Rover::new(body, wheels, 9.81),
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
    // camera behind and above it works without tracking.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 7.0, -16.0).looking_at(Vec3::new(0.0, 1.0, 4.0), Vec3::Y),
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

fn drive_input(keys: Res<ButtonInput<KeyCode>>, mut world: ResMut<RoverWorld>) {
    let throttle = if keys.pressed(KeyCode::KeyW) {
        2_500.0
    } else if keys.pressed(KeyCode::KeyS) {
        -2_500.0
    } else {
        0.0
    };
    let brake = if keys.pressed(KeyCode::Space) {
        3_000.0
    } else {
        0.0
    };
    let steer = if keys.pressed(KeyCode::KeyA) {
        0.4
    } else if keys.pressed(KeyCode::KeyD) {
        -0.4
    } else {
        0.0
    };
    for (i, w) in world.rover.wheels.iter_mut().enumerate() {
        w.drive_torque = throttle;
        w.brake = brake;
        // Steer the front wheels (the +z pair).
        w.steer = if i >= 2 { steer } else { 0.0 };
    }
}

fn step_rover(time: Res<Time>, clock: Res<SimClock>, mut world: ResMut<RoverWorld>) {
    // Paused (WI 638): freeze the rover physics; the accumulator does not grow, so resume is jump-free.
    if clock.paused {
        return;
    }
    world.accumulator += time.delta_secs_f64();
    let mut substeps = 0;
    while world.accumulator >= SUBSTEP_DT && substeps < MAX_SUBSTEPS {
        let terrain = world.terrain;
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

    // Wheels and suspension legs.
    for w in &world.rover.wheels {
        let hub = body.position + body.orientation * w.mount;
        let ground = terrain.height(hub.x, hub.z);
        let contact = DVec3::new(hub.x, ground, hub.z);
        gizmos.line(
            to_render(hub),
            to_render(contact),
            Color::srgb(0.5, 0.5, 0.55),
        );
        gizmos.sphere(
            Isometry3d::from_translation(to_render(contact)),
            w.radius as f32,
            Color::srgb(0.15, 0.15, 0.18),
        );
    }
}
