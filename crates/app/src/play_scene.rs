//! Play — a continuous one-craft session (WI 534): the first-playable shell.
//!
//! One craft runs Build → Launch → Flight → Recovery on the **unified flight
//! pipeline** (`sounding_sim::flight`), its phase driven by simulation state
//! (`sounding_sim::session`). The demo is a **sounding**: the craft rests on the pad
//! (`launch`), auto-throttles up with SAS holding it vertical (`propulsion` +
//! `attitude`), ascends, burns out, coasts to apoapsis, falls back through the
//! atmosphere, and touches down — Recovery. The auto-throttle/SAS stand in for
//! player controls (WI 535). Rendering reuses the dive/launch flat-ground convention.

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

use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};

const BODY: CentralBody = CentralBody::EARTHLIKE;
const SUBSTEP_DT: f64 = 0.004;
const MAX_SUBSTEPS: u32 = 250;
const PROPELLANT: ResourceType = ResourceType(0);
const HOLD_TIME: f64 = 2.0;
const RAMP_TIME: f64 = 1.5;

/// The played craft + session.
#[derive(Resource)]
struct PlayWorld {
    body: ActiveBody,
    craft: FlightCraft,
    params: FlightParams,
    pad: LaunchPad,
    session: GameSession,
    elapsed: f64,
    accumulator: f64,
}

impl PlayWorld {
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
            },
            craft: FlightCraft {
                dry_mass: mp.mass,
                dry_com: mp.center_of_mass,
                voxels,
                propulsion,
                attitude,
            },
            pad: LaunchPad::resting(pad_radius),
            session,
            elapsed: 0.0,
            accumulator: 0.0,
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
}

#[derive(Component)]
struct CraftMarker;
#[derive(Component)]
struct Hud;

/// The play scene.
pub struct PlayScenePlugin;

impl Plugin for PlayScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .insert_resource(PlayWorld::new())
            .add_systems(Startup, setup_scene)
            .add_systems(
                Update,
                (step_play, track_craft, follow_camera, update_hud).chain(),
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
        Text::new("phase:    LAUNCH\nthrottle:   0%\naltitude:        0 m\nspeed:       0.0 m/s"),
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
/// + SAS hold and advancing the phase from sim state.
fn step_play(time: Res<Time>, mut world: ResMut<PlayWorld>) {
    if world.session.is_terminal() {
        return; // rest after recovery
    }
    world.accumulator += time.delta_secs_f64();
    let mut n = 0;
    while world.accumulator >= SUBSTEP_DT && n < MAX_SUBSTEPS && !world.session.is_terminal() {
        world.elapsed += SUBSTEP_DT;
        let throttle = world.throttle();
        world
            .craft
            .propulsion
            .apply_command(&Command::SetThrottle(throttle));
        // Engage SAS hold (vertical) once we start throttling up.
        if throttle > 0.0 && world.craft.attitude.sas.mode == SasMode::Off {
            let orientation = world.body.orientation;
            world
                .craft
                .attitude
                .apply_command(&Command::SetSas(SasMode::Hold), orientation);
        }

        let PlayWorld {
            body,
            craft,
            params,
            pad,
            ..
        } = &mut *world;
        flight_step(body, craft, params, pad, SUBSTEP_DT);

        // Advance the session phase from sim state.
        let r = world.body.position.length();
        let up = if r > 0.0 {
            world.body.position / r
        } else {
            DVec3::Y
        };
        let altitude = r - BODY.radius;
        let vertical_speed = world.body.velocity.dot(up);
        let speed = world.body.velocity.length();
        let released = world.pad.released;
        world
            .session
            .update(released, altitude, vertical_speed, speed);

        world.accumulator -= SUBSTEP_DT;
        n += 1;
    }
}

fn track_craft(world: Res<PlayWorld>, mut craft: Query<&mut WorldPlacement, With<CraftMarker>>) {
    if let Ok(mut wp) = craft.single_mut() {
        wp.0 = WorldPos::new(FrameId::CENTRAL_BODY, world.render_world());
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
        let throttle = world.throttle() * 100.0;
        let altitude = world.altitude();
        let speed = world.body.velocity.length();
        text.0 = format!(
            "phase:    {phase}\nthrottle: {throttle:3.0}%\naltitude: {altitude:8.0} m\nspeed:    {speed:7.1} m/s"
        );
    }
}
