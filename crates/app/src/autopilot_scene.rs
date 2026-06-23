//! Autopilot — a continuous one-craft session, flown automatically (WI 534; renamed
//! from `play` in SideQuest 539 since there is no player interaction yet — that
//! arrives with the controls, WI 535).
//!
//! One craft runs Build → Launch → Flight → Recovery on the **unified flight
//! pipeline** (`sounding_sim::flight`), its phase driven by simulation state
//! (`sounding_sim::session`). The demo is a **sounding**: the craft rests on the pad
//! (`launch`), auto-throttles up with SAS holding it vertical (`propulsion` +
//! `attitude`), ascends, burns out, coasts to apoapsis, falls back through the
//! atmosphere, and touches down — Recovery. The HUD shows the phase, a throttle bar,
//! G-force, altitude/speed, and tilt; an attitude gizmo draws the nose and velocity.

use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::DVec3;
use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use sounding_sim::active::ActiveBody;
use sounding_sim::attitude::{AttitudeControl, AttitudePilot, ReactionWheels, Sas};
use sounding_sim::command::{Command, SasMode};
use sounding_sim::control::ControlSystem;
use sounding_sim::flight::{flight_step, FlightCraft, FlightParams};
use sounding_sim::fluid::FluidMedium;
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::launch::LaunchPad;
use sounding_sim::medium::max_cross_section;
use sounding_sim::propulsion::{Engine, EngineCommand, Propulsion};
use sounding_sim::resource::{Reservoir, ReservoirId, ResourceGraph, ResourceType};
use sounding_sim::session::{GameSession, Outcome, Phase};
use sounding_sim::sim::CentralBody;
use sounding_sim::voxel::{Material, Voxel, VoxelCraft};

use crate::bus::ActiveFlight;
use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};
use sounding_sim::telemetry::ActiveFlightTelemetry;

const BODY: CentralBody = CentralBody::EARTHLIKE;
const SUBSTEP_DT: f64 = 0.004;
const MAX_SUBSTEPS: u32 = 250;
const PROPELLANT: ResourceType = ResourceType(0);
const HOLD_TIME: f64 = 2.0;
const RAMP_TIME: f64 = 1.5;
/// Standard gravity for the G-force readout, m/s².
const G0: f64 = 9.80665;

/// The auto-flown craft + session.
#[derive(Resource)]
struct AutopilotWorld {
    body: ActiveBody,
    craft: FlightCraft,
    params: FlightParams,
    pad: LaunchPad,
    session: GameSession,
    elapsed: f64,
    accumulator: f64,
    /// Felt (proper) acceleration in g — what an onboard accelerometer reads.
    g_force: f64,
}

impl AutopilotWorld {
    fn new() -> Self {
        // A slim rocket along +Y, with an engine, propellant, and reaction wheels.
        let mut voxels = VoxelCraft::new(1.0);
        for y in 0..5 {
            voxels.voxels.push(Voxel {
                cell: IVec3::new(0, y, 0),
                material: Material::COMPOSITE,
            });
        }
        let mp = voxels.mass_properties().expect("non-empty craft");
        let drag_area = max_cross_section(&voxels);
        let propellant = 5_000.0;

        let propulsion = Propulsion {
            graph: ResourceGraph {
                reservoirs: vec![Reservoir::new(PROPELLANT, propellant, propellant)],
                ..Default::default()
            },
            tank_mounts: vec![DVec3::new(mp.center_of_mass.x, 0.5, mp.center_of_mass.z)],
            engines: vec![Engine {
                tank: ReservoirId(0),
                exhaust_velocity: 3_000.0,
                max_mass_flow: 70.0, // ~210 kN; wet weight ≈ 128 kN → TWR ≈ 1.6
                mount: DVec3::new(mp.center_of_mass.x, 0.0, mp.center_of_mass.z),
                axis: DVec3::Y,
                max_gimbal: 0.0,
            }],
            commands: vec![EngineCommand::default()],
        };
        let attitude = AttitudePilot {
            sas: Sas::default(),
            manual: DVec3::ZERO,
            authority: 5_000.0,
            recapture_on_release: true,
            actuators: AttitudeControl {
                wheels: Some(ReactionWheels::new(8_000.0, 1e9)),
                rcs: None,
            },
        };

        let pad_radius = BODY.radius + mp.center_of_mass.y; // base rests on the pad
        let body = ActiveBody::new(
            DVec3::new(0.0, pad_radius, 0.0),
            DVec3::ZERO,
            mp.mass + propellant,
            mp.inertia,
        );
        let mut session = GameSession::new();
        session.begin_launch();

        Self {
            body,
            params: FlightParams {
                mu: BODY.mu,
                surface_radius: BODY.radius,
                medium: FluidMedium::EARTHLIKE,
                drag_area,
                drag_coefficient: 1.0,
                lift: None,
                ground: None,
            },
            craft: FlightCraft {
                dry_mass: mp.mass,
                dry_com: mp.center_of_mass,
                voxels,
                propulsion,
                attitude,
                control: ControlSystem::crewed_stabilized(),
                autopilot: None,
            },
            pad: LaunchPad::resting(pad_radius),
            session,
            elapsed: 0.0,
            accumulator: 0.0,
            g_force: 1.0,
        }
    }

    fn render_world(&self) -> DVec3 {
        self.body.position - DVec3::new(0.0, BODY.radius, 0.0)
    }

    fn altitude(&self) -> f64 {
        self.body.position.length() - BODY.radius
    }

    fn throttle(&self) -> f64 {
        ((self.elapsed - HOLD_TIME) / RAMP_TIME).clamp(0.0, 1.0)
    }

    /// Tilt of the nose (+Y body axis) from the local vertical, degrees.
    fn tilt_degrees(&self) -> f64 {
        let r = self.body.position.length();
        if r <= 0.0 {
            return 0.0;
        }
        let up = self.body.position / r;
        let nose = (self.body.orientation * DVec3::Y).normalize_or_zero();
        nose.dot(up).clamp(-1.0, 1.0).acos().to_degrees()
    }
}

#[derive(Component)]
struct CraftMarker;
#[derive(Component)]
struct Hud;

/// The autopilot scene.
pub struct AutopilotScenePlugin;

impl Plugin for AutopilotScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .insert_resource(AutopilotWorld::new())
            .add_systems(Startup, setup_scene)
            .add_systems(
                Update,
                (
                    step_autopilot,
                    publish_active_flight,
                    track_craft,
                    follow_camera,
                    update_hud,
                    draw_attitude_gizmo,
                )
                    .chain(),
            );
    }
}

fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering: ResMut<Assets<ScatteringMedium>>,
    world: Res<AutopilotWorld>,
) {
    crate::ground::spawn_ground(&mut commands, &mut meshes, &mut materials, &asset_server);
    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(Sphere {
            radius: BODY.radius as f32,
        }))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.30, 0.27, 0.22),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::default(),
        WorldPlacement(WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(0.0, -BODY.radius, 0.0),
        )),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(Cuboid::new(1.0, 5.0, 1.0)))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.88, 0.88, 0.90),
            metallic: 0.5,
            perceptual_roughness: 0.4,
            ..default()
        })),
        Transform::default(),
        WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, world.render_world())),
        CraftMarker,
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_x(-0.5) * Quat::from_rotation_y(0.6)),
    ));

    commands.spawn((
        Text::new("phase:    LAUNCH"),
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

    let cam = world.render_world() + DVec3::new(16.0, 7.0, 16.0);
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(cam.as_vec3()).looking_at(Vec3::new(0.0, 4.0, 0.0), Vec3::Y),
        Atmosphere::earthlike(scattering.add(ScatteringMedium::default())),
        AtmosphereSettings::default(),
        Exposure { ev100: 13.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        AtmosphereEnvironmentMapLight::default(),
        WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, cam)),
        AnchorCamera,
    ));
}

/// Sub-steps the session on the unified flight pipeline, driving the auto-throttle
/// + SAS hold, tracking felt G-force, and advancing the phase from sim state.
fn step_autopilot(time: Res<Time>, mut world: ResMut<AutopilotWorld>) {
    if world.session.is_terminal() {
        return; // rest after recovery
    }
    world.accumulator += time.delta_secs_f64();
    let mut n = 0;
    while world.accumulator >= SUBSTEP_DT && n < MAX_SUBSTEPS && !world.session.is_terminal() {
        world.elapsed += SUBSTEP_DT;
        let throttle = world.throttle();
        let orientation = world.body.orientation;
        // Route through the tier-gated applicator (WI 562); this craft is Stabilized.
        world
            .craft
            .apply_command(&Command::SetThrottle(throttle), orientation);
        if throttle > 0.0 && world.craft.attitude.sas.mode == SasMode::Off {
            world
                .craft
                .apply_command(&Command::SetSas(SasMode::Hold), orientation);
        }

        // Capture state for the felt-acceleration (G-force) readout.
        let v0 = world.body.velocity;
        let r0 = world.body.position.length();
        let up0 = if r0 > 0.0 {
            world.body.position / r0
        } else {
            DVec3::Y
        };

        let AutopilotWorld {
            body,
            craft,
            params,
            pad,
            ..
        } = &mut *world;
        flight_step(body, craft, params, pad, SUBSTEP_DT);

        // Felt (proper) acceleration = actual − gravitational (an accelerometer
        // reads 1 g on the pad, >1 g under thrust, 0 g in free fall).
        let gravity_accel = -BODY.mu / (r0 * r0) * up0;
        let felt = (world.body.velocity - v0) / SUBSTEP_DT - gravity_accel;
        world.g_force = felt.length() / G0;

        // Advance the session phase from sim state.
        let altitude = world.body.position.length() - BODY.radius;
        let vertical_speed = world.body.velocity.dot(up0);
        let speed = world.body.velocity.length();
        let released = world.pad.released;
        world
            .session
            .update(released, altitude, vertical_speed, speed);

        world.accumulator -= SUBSTEP_DT;
        n += 1;
    }
}

/// Publishes the auto-flown craft's autonomy state onto the bus bridge each frame (WI 569).
fn publish_active_flight(world: Res<AutopilotWorld>, mut active: ResMut<ActiveFlight>) {
    active.0 = Some(ActiveFlightTelemetry::from_flight(&world.craft));
}

fn track_craft(
    world: Res<AutopilotWorld>,
    mut craft: Query<(&mut WorldPlacement, &mut Transform), With<CraftMarker>>,
) {
    if let Ok((mut wp, mut tf)) = craft.single_mut() {
        wp.0 = WorldPos::new(FrameId::CENTRAL_BODY, world.render_world());
        tf.rotation = world.body.orientation.as_quat(); // show the craft's attitude
    }
}

fn follow_camera(
    world: Res<AutopilotWorld>,
    mut camera: Query<(&mut Transform, &mut WorldPlacement), With<AnchorCamera>>,
) {
    if let Ok((mut tf, mut placement)) = camera.single_mut() {
        let target = world.render_world();
        let eye = target + DVec3::new(16.0, 7.0, 16.0);
        placement.0 = WorldPos::new(FrameId::CENTRAL_BODY, eye);
        let look_dir = (target - eye).as_vec3().normalize_or_zero();
        if look_dir != Vec3::ZERO {
            tf.rotation = Transform::default().looking_to(look_dir, Vec3::Y).rotation;
        }
    }
}

/// Draws an attitude gizmo at the craft: its nose axis (green) and velocity (orange).
#[allow(clippy::type_complexity)]
fn draw_attitude_gizmo(
    world: Res<AutopilotWorld>,
    mut gizmos: Gizmos,
    craft: Query<&Transform, With<CraftMarker>>,
) {
    if let Ok(tf) = craft.single() {
        let pos = tf.translation;
        let nose = (world.body.orientation.as_quat() * Vec3::Y).normalize_or_zero();
        gizmos.line(pos, pos + nose * 6.0, Color::srgb(0.3, 1.0, 0.3));
        let vel = world.body.velocity.as_vec3();
        if vel.length() > 1.0 {
            gizmos.line(pos, pos + vel.normalize() * 5.0, Color::srgb(1.0, 0.6, 0.2));
        }
    }
}

/// A 10-cell text gauge for a `[0, 1]` fraction, e.g. `[######----]`.
fn gauge(fraction: f64) -> String {
    let filled = (fraction * 10.0).round().clamp(0.0, 10.0) as usize;
    let mut s = String::from("[");
    s.push_str(&"#".repeat(filled));
    s.push_str(&"-".repeat(10 - filled));
    s.push(']');
    s
}

fn update_hud(world: Res<AutopilotWorld>, mut hud: Query<&mut Text, With<Hud>>) {
    if let Ok(mut text) = hud.single_mut() {
        let phase = match world.session.phase {
            Phase::Build => "BUILD",
            Phase::Launch => "LAUNCH",
            Phase::Flight => "FLIGHT",
            Phase::Recovery => match world.session.outcome {
                Outcome::Landed => "RECOVERY (landed)",
                Outcome::Crashed => "RECOVERY (crashed)",
                Outcome::None => "RECOVERY",
            },
        };
        let throttle = world.throttle();
        let altitude = world.altitude();
        let speed = world.body.velocity.length();
        let (fuel_amt, fuel_cap) = world
            .craft
            .propulsion
            .graph
            .reservoirs
            .first()
            .map(|r| (r.amount, r.capacity))
            .unwrap_or((0.0, 0.0));
        let fuel_frac = if fuel_cap > 0.0 {
            fuel_amt / fuel_cap
        } else {
            0.0
        };
        // Throttle commanded but the tank is empty → flame-out (no thrust).
        let flameout = throttle > 0.0 && fuel_amt <= 1.0;
        let throttle_note = if flameout { "  FLAMEOUT" } else { "" };
        text.0 = format!(
            "phase:    {phase}\nthrottle: {tbar} {pct:3.0}%{note}\nfuel:     {fbar} {fuel:6.0} kg\nG-force:  {g:5.1} g\naltitude: {altitude:8.0} m\nspeed:    {speed:7.1} m/s\ntilt:     {tilt:5.0}°",
            tbar = gauge(throttle),
            pct = throttle * 100.0,
            note = throttle_note,
            fbar = gauge(fuel_frac),
            fuel = fuel_amt,
            g = world.g_force,
            tilt = world.tilt_degrees(),
        );
    }
}
