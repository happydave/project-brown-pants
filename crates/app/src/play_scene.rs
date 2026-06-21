//! Play — the interactive first-playable scene (WI 535): fly a craft by hand.
//!
//! The player flies one craft through Launch → Flight → Recovery on the unified
//! flight pipeline (`sounding_sim::flight`), driving it **entirely through the
//! command envelope** — throttle, manual attitude, SAS, and time-warp all emit the
//! same `Command`s an autopilot (or the AI) would, applied via the craft's command
//! applicators. The HUD adds the rocket-equation **Δv** alongside the orbit
//! apsides/energy, altitude/speed/medium, fuel, G-force, and SAS mode. The
//! auto-flown `-- autopilot` scene is the hands-off counterpart.
//!
//! Controls: Shift/Ctrl throttle up/down, Z/X full/cutoff · W/S/A/D/Q/E attitude
//! (pitch/yaw/roll) · T toggle SAS hold, R kill-rotation, F SAS off · `.`/`,` warp.

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
use sounding_sim::control::{ControlSystem, ControlTier};
use sounding_sim::flight::{flight_step, FlightCraft, FlightParams};
use sounding_sim::fluid::{FluidMedium, MediumKind};
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::launch::LaunchPad;
use sounding_sim::medium::max_cross_section;
use sounding_sim::propulsion::{Engine, EngineCommand, Propulsion};
use sounding_sim::resource::{Reservoir, ReservoirId, ResourceGraph, ResourceType};
use sounding_sim::session::{GameSession, Outcome, Phase};
use sounding_sim::sim::CentralBody;
use sounding_sim::voxel::{Material, Voxel, VoxelCraft};

use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};

const BODY: CentralBody = CentralBody::EARTHLIKE;
const SUBSTEP_DT: f64 = 0.004;
const MAX_SUBSTEPS: u32 = 250;
const PROPELLANT: ResourceType = ResourceType(0);
const G0: f64 = 9.80665;
const THROTTLE_RATE: f64 = 1.0; // per second, for Shift/Ctrl ramp
const MIN_WARP: f64 = 1.0;
const MAX_WARP: f64 = 16.0;

/// The player-flown craft + session.
#[derive(Resource)]
struct PlayWorld {
    body: ActiveBody,
    craft: FlightCraft,
    params: FlightParams,
    pad: LaunchPad,
    session: GameSession,
    accumulator: f64,
    throttle: f64,
    warp: f64,
    g_force: f64,
}

impl PlayWorld {
    fn new() -> Self {
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

        let pad_radius = BODY.radius + mp.center_of_mass.y;
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
            },
            craft: FlightCraft {
                dry_mass: mp.mass,
                dry_com: mp.center_of_mass,
                voxels,
                propulsion,
                attitude,
                control: ControlSystem::crewed_stabilized(),
            },
            pad: LaunchPad::resting(pad_radius),
            session,
            accumulator: 0.0,
            throttle: 0.0,
            warp: 1.0,
            g_force: 1.0,
        }
    }

    fn render_world(&self) -> DVec3 {
        self.body.position - DVec3::new(0.0, BODY.radius, 0.0)
    }

    fn altitude(&self) -> f64 {
        self.body.position.length() - BODY.radius
    }

    /// Specific orbital energy, J/kg.
    fn specific_energy(&self) -> f64 {
        let r = self.body.position.length();
        if r <= 0.0 {
            return 0.0;
        }
        0.5 * self.body.velocity.length_squared() - BODY.mu / r
    }

    /// (apoapsis_alt, periapsis_alt) above the surface in metres if the orbit is
    /// bound, else `None`. Derived from the current 3D state.
    fn apsides(&self) -> Option<(f64, f64)> {
        let r = self.body.position.length();
        let energy = self.specific_energy();
        if energy >= 0.0 || r <= 0.0 {
            return None; // unbound (escape)
        }
        let mu = BODY.mu;
        let a = -mu / (2.0 * energy);
        let h = self.body.position.cross(self.body.velocity).length();
        let e = (1.0 + 2.0 * energy * h * h / (mu * mu)).max(0.0).sqrt();
        let apo = a * (1.0 + e) - BODY.radius;
        let peri = a * (1.0 - e) - BODY.radius;
        Some((apo, peri))
    }

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

/// The interactive play scene.
pub struct PlayScenePlugin;

impl Plugin for PlayScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .insert_resource(PlayWorld::new())
            .add_systems(Startup, setup_scene)
            .add_systems(
                Update,
                (
                    player_input,
                    step_play,
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
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering: ResMut<Assets<ScatteringMedium>>,
    world: Res<PlayWorld>,
) {
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
            font_size: 18.0,
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

    commands.spawn((
        Text::new(
            "Shift/Ctrl throttle · Z/X full/cut · WSAD QE attitude · T hold  R kill  F off · ,/. warp",
        ),
        TextFont {
            font_size: 14.0,
            ..default()
        },
        TextColor(Color::srgb(0.7, 0.75, 0.8)),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(10.0),
            left: Val::Px(12.0),
            ..default()
        },
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

/// Translates keyboard input into `Command`s applied to the craft (no direct state
/// mutation): throttle, manual attitude, SAS mode, and time-warp.
fn player_input(time: Res<Time>, keys: Res<ButtonInput<KeyCode>>, mut world: ResMut<PlayWorld>) {
    let dt = time.delta_secs_f64();

    // Throttle.
    if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
        world.throttle = (world.throttle + THROTTLE_RATE * dt).min(1.0);
    }
    if keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight) {
        world.throttle = (world.throttle - THROTTLE_RATE * dt).max(0.0);
    }
    if keys.just_pressed(KeyCode::KeyZ) {
        world.throttle = 1.0;
    }
    if keys.just_pressed(KeyCode::KeyX) {
        world.throttle = 0.0;
    }
    let orientation = world.body.orientation;
    let throttle = world.throttle;
    // Route all manual input through the tier-gated FlightCraft applicator (WI 562):
    // an uncontrolled craft would ignore these; here the craft is Stabilized.
    world
        .craft
        .apply_command(&Command::SetThrottle(throttle), orientation);

    // Manual attitude intent (pitch/yaw/roll about the body axes).
    let mut manual = DVec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        manual.x += 1.0;
    }
    if keys.pressed(KeyCode::KeyS) {
        manual.x -= 1.0;
    }
    if keys.pressed(KeyCode::KeyA) {
        manual.z += 1.0;
    }
    if keys.pressed(KeyCode::KeyD) {
        manual.z -= 1.0;
    }
    if keys.pressed(KeyCode::KeyQ) {
        manual.y += 1.0;
    }
    if keys.pressed(KeyCode::KeyE) {
        manual.y -= 1.0;
    }
    world
        .craft
        .apply_command(&Command::SetAttitude(manual), orientation);

    // SAS mode.
    if keys.just_pressed(KeyCode::KeyT) {
        let mode = if world.craft.attitude.sas.mode == SasMode::Hold {
            SasMode::Off
        } else {
            SasMode::Hold
        };
        world
            .craft
            .apply_command(&Command::SetSas(mode), orientation);
    }
    if keys.just_pressed(KeyCode::KeyR) {
        world
            .craft
            .apply_command(&Command::SetSas(SasMode::KillRotation), orientation);
    }
    if keys.just_pressed(KeyCode::KeyF) {
        world
            .craft
            .apply_command(&Command::SetSas(SasMode::Off), orientation);
    }
    // Toggle the SAS hold-target re-capture policy (WI 564): nudge-sticks vs return-to-target.
    if keys.just_pressed(KeyCode::KeyG) {
        let next = !world.craft.attitude.recapture_on_release;
        world
            .craft
            .apply_command(&Command::SetSasRecapture(next), orientation);
    }

    // Time-warp.
    if keys.just_pressed(KeyCode::Period) {
        world.warp = (world.warp * 2.0).min(MAX_WARP);
    }
    if keys.just_pressed(KeyCode::Comma) {
        world.warp = (world.warp / 2.0).max(MIN_WARP);
    }
}

/// Sub-steps the flight (player time-warp), tracks G-force, advances the phase.
fn step_play(time: Res<Time>, mut world: ResMut<PlayWorld>) {
    if world.session.is_terminal() {
        return;
    }
    world.accumulator += time.delta_secs_f64() * world.warp;
    let mut n = 0;
    while world.accumulator >= SUBSTEP_DT && n < MAX_SUBSTEPS && !world.session.is_terminal() {
        let v0 = world.body.velocity;
        let r0 = world.body.position.length();
        let up0 = if r0 > 0.0 {
            world.body.position / r0
        } else {
            DVec3::Y
        };

        let PlayWorld {
            body,
            craft,
            params,
            pad,
            ..
        } = &mut *world;
        flight_step(body, craft, params, pad, SUBSTEP_DT);

        let gravity_accel = -BODY.mu / (r0 * r0) * up0;
        let felt = (world.body.velocity - v0) / SUBSTEP_DT - gravity_accel;
        world.g_force = felt.length() / G0;

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

fn track_craft(
    world: Res<PlayWorld>,
    mut craft: Query<(&mut WorldPlacement, &mut Transform), With<CraftMarker>>,
) {
    if let Ok((mut wp, mut tf)) = craft.single_mut() {
        wp.0 = WorldPos::new(FrameId::CENTRAL_BODY, world.render_world());
        tf.rotation = world.body.orientation.as_quat();
    }
}

fn follow_camera(
    world: Res<PlayWorld>,
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

#[allow(clippy::type_complexity)]
fn draw_attitude_gizmo(
    world: Res<PlayWorld>,
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

fn gauge(fraction: f64) -> String {
    let filled = (fraction * 10.0).round().clamp(0.0, 10.0) as usize;
    format!("[{}{}]", "#".repeat(filled), "-".repeat(10 - filled))
}

/// Altitude/distance in km if large, else m.
fn fmt_alt(m: f64) -> String {
    if m.abs() >= 10_000.0 {
        format!("{:.0} km", m / 1_000.0)
    } else {
        format!("{m:.0} m")
    }
}

fn update_hud(world: Res<PlayWorld>, mut hud: Query<&mut Text, With<Hud>>) {
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
        let throttle = world.throttle;
        let fuel = world.craft.propulsion.propellant();
        let fuel_cap = world
            .craft
            .propulsion
            .graph
            .reservoirs
            .first()
            .map(|r| r.capacity)
            .unwrap_or(1.0);
        let flameout = throttle > 0.0 && fuel <= 1.0;
        let dv = world.craft.propulsion.delta_v(world.craft.dry_mass);
        let medium = match world.params.medium.sample_altitude(world.altitude()).medium {
            MediumKind::Vacuum => "vacuum",
            MediumKind::Atmosphere => "atmosphere",
            MediumKind::Liquid => "ocean",
        };
        let r = world.body.position.length();
        let up = if r > 0.0 {
            world.body.position / r
        } else {
            DVec3::Y
        };
        let v_speed = world.body.velocity.dot(up);
        let orbit_line = match world.apsides() {
            Some((apo, peri)) if peri >= 0.0 => {
                format!(
                    "orbit:    ORBIT  apo {} / peri {}",
                    fmt_alt(apo),
                    fmt_alt(peri)
                )
            }
            Some((apo, _)) => format!("orbit:    suborbital  apoapsis {}", fmt_alt(apo)),
            None => "orbit:    escape".to_string(),
        };
        let tier = world.craft.resolve_control();
        // SAS shows "unavail" when the tier can't stabilize (no powered command core).
        let sas = if !tier.allows_stabilization() {
            "unavail"
        } else {
            match world.craft.attitude.sas.mode {
                SasMode::Off => "off",
                SasMode::KillRotation => "kill-rot",
                SasMode::Hold => "hold",
                SasMode::Point(_) => "point",
            }
        };
        let recap = if world.craft.attitude.recapture_on_release {
            "recap"
        } else {
            "return"
        };
        let ctrl = match tier {
            ControlTier::Uncontrolled => "UNCONTROLLED",
            ControlTier::Direct => "direct",
            ControlTier::Stabilized => "stabilized",
        };
        text.0 = format!(
            "phase:    {phase}\nthrottle: {tbar} {pct:3.0}%{note}\nfuel:     {fbar} {fuel:6.0} kg\n\u{0394}v:       {dv:6.0} m/s\nG-force:  {g:5.1} g\naltitude: {alt}\nv-speed:  {v_speed:+7.0} m/s\nspeed:    {speed:7.1} m/s\n{orbit_line}\nenergy:   {energy:8.2} MJ/kg\nmedium:   {medium}   tilt {tilt:.0}\u{00b0}   SAS {sas} ({recap})\ncontrol:  {ctrl}",
            tbar = gauge(throttle),
            pct = throttle * 100.0,
            note = if flameout { "  FLAMEOUT" } else { "" },
            fbar = gauge(fuel / fuel_cap),
            g = world.g_force,
            alt = fmt_alt(world.altitude()),
            speed = world.body.velocity.length(),
            energy = world.specific_energy() / 1.0e6,
            tilt = world.tilt_degrees(),
        );
    }
}
