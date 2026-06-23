//! Skins — a side-by-side voxel-skin comparison (`-- skins`, WI 582/583).
//!
//! One craft is stepped on the unified flight pipeline (a short auto-flown ascent) and
//! rendered **twice from the same sim state** at offset transforms, so two skinnings of
//! the identical lattice fly in formation under identical lighting and the same
//! `hull_panel` PBR material: the **blocky** skin (left, per-cell cubes, WI 582) and the
//! **greedy-meshed hull** (right, the primary look, WI 583), side by side.

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
use sounding_sim::session::GameSession;
use sounding_sim::sim::CentralBody;
use sounding_sim::voxel::{Material, Voxel, VoxelCraft};

use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};
use crate::voxel_skin::{build_skin_mesh, material_set_for, pbr_material, VoxelSkin};

const BODY: CentralBody = CentralBody::EARTHLIKE;
const SUBSTEP_DT: f64 = 0.004;
const MAX_SUBSTEPS: u32 = 250;
const PROPELLANT: ResourceType = ResourceType(0);
const HOLD_TIME: f64 = 2.0;
const RAMP_TIME: f64 = 1.5;
/// Lateral separation between the two rendered skins, metres.
const SLOT_OFFSET: f64 = 4.0;

/// The single auto-flown craft whose state both skins render.
#[derive(Resource)]
struct SkinsWorld {
    body: ActiveBody,
    craft: FlightCraft,
    params: FlightParams,
    pad: LaunchPad,
    session: GameSession,
    elapsed: f64,
    accumulator: f64,
}

impl SkinsWorld {
    fn new() -> Self {
        // A chunky 2×2×5 hull (20 cells): blocky shows every cube; greedy will merge each
        // broad side into a few panels — a legible blocky-vs-hull contrast.
        let mut voxels = VoxelCraft::new(1.0);
        for x in 0..2 {
            for z in 0..2 {
                for y in 0..5 {
                    voxels.voxels.push(Voxel {
                        cell: IVec3::new(x, y, z),
                        material: Material::COMPOSITE,
                    });
                }
            }
        }
        let mp = voxels.mass_properties().expect("non-empty craft");
        let drag_area = max_cross_section(&voxels);
        let propellant = 8_000.0;

        let propulsion = Propulsion {
            graph: ResourceGraph {
                reservoirs: vec![Reservoir::new(PROPELLANT, propellant, propellant)],
                ..Default::default()
            },
            tank_mounts: vec![DVec3::new(mp.center_of_mass.x, 0.5, mp.center_of_mass.z)],
            engines: vec![Engine {
                tank: ReservoirId(0),
                exhaust_velocity: 3_000.0,
                max_mass_flow: 220.0, // sized for TWR ≈ 1.9 on the 20-cell hull
                mount: DVec3::new(mp.center_of_mass.x, 0.0, mp.center_of_mass.z),
                axis: DVec3::Y,
                max_gimbal: 0.0,
            }],
            commands: vec![EngineCommand::default()],
        };
        let attitude = AttitudePilot {
            sas: Sas::default(),
            manual: DVec3::ZERO,
            authority: 8_000.0,
            recapture_on_release: true,
            actuators: AttitudeControl {
                wheels: Some(ReactionWheels::new(20_000.0, 1e9)),
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
        }
    }

    fn render_world(&self) -> DVec3 {
        self.body.position - DVec3::new(0.0, BODY.radius, 0.0)
    }

    fn throttle(&self) -> f64 {
        ((self.elapsed - HOLD_TIME) / RAMP_TIME).clamp(0.0, 1.0)
    }
}

/// One rendered skin of the craft: a lateral offset from the shared sim state.
#[derive(Component)]
struct SkinSlot {
    offset: DVec3,
}

#[derive(Component)]
struct Hud;

/// The skins comparison scene.
pub struct SkinsScenePlugin;

impl Plugin for SkinsScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .insert_resource(SkinsWorld::new())
            .add_systems(Startup, setup_scene)
            .add_systems(Update, (step_skins, track_skins, follow_camera).chain());
    }
}

fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering: ResMut<Assets<ScatteringMedium>>,
    world: Res<SkinsWorld>,
) {
    // The same lattice the sim flies (the world owns the authoritative copy).
    let craft_voxels = world.craft.voxels.clone();
    let material = pbr_material(
        material_set_for(Material::COMPOSITE),
        &asset_server,
        &mut materials,
    );

    // Two slots from one sim state, separated laterally so they read side by side: the
    // blocky skin (left) and the greedy-meshed hull (right, WI 583) of the same lattice.
    for (offset_z, skin) in [
        (-SLOT_OFFSET, VoxelSkin::Blocky),
        (SLOT_OFFSET, VoxelSkin::Hull),
    ] {
        let mesh = meshes.add(build_skin_mesh(&craft_voxels, skin));
        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material.clone()),
            Transform::default(),
            WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, world.render_world())),
            SkinSlot {
                offset: DVec3::new(0.0, 0.0, offset_z),
            },
        ));
    }

    // Ground sphere (planet body / horizon).
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
    // Textured rocky ground patch under the craft (WI 588).
    crate::ground::spawn_ground(&mut commands, &mut meshes, &mut materials, &asset_server);

    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_x(-0.5) * Quat::from_rotation_y(0.6)),
    ));

    commands.spawn((
        Text::new("skins: left blocky (per-cell cubes)  |  right hull (greedy-meshed)"),
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

    let cam = world.render_world() + DVec3::new(18.0, 6.0, 0.0);
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

/// Steps the one shared craft up the unified flight pipeline (throttle ramp + SAS hold).
fn step_skins(time: Res<Time>, mut world: ResMut<SkinsWorld>) {
    if world.session.is_terminal() {
        return;
    }
    world.accumulator += time.delta_secs_f64();
    let mut n = 0;
    while world.accumulator >= SUBSTEP_DT && n < MAX_SUBSTEPS && !world.session.is_terminal() {
        world.elapsed += SUBSTEP_DT;
        let throttle = world.throttle();
        let orientation = world.body.orientation;
        world
            .craft
            .apply_command(&Command::SetThrottle(throttle), orientation);
        if throttle > 0.0 && world.craft.attitude.sas.mode == SasMode::Off {
            world
                .craft
                .apply_command(&Command::SetSas(SasMode::Hold), orientation);
        }

        let r0 = world.body.position.length();
        let up0 = if r0 > 0.0 {
            world.body.position / r0
        } else {
            DVec3::Y
        };
        let SkinsWorld {
            body,
            craft,
            params,
            pad,
            ..
        } = &mut *world;
        flight_step(body, craft, params, pad, SUBSTEP_DT);

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

/// Places each skin slot at the shared craft state plus its lateral offset, with the
/// craft's attitude.
fn track_skins(
    world: Res<SkinsWorld>,
    mut slots: Query<(&SkinSlot, &mut WorldPlacement, &mut Transform)>,
) {
    let base = world.render_world();
    let rot = world.body.orientation.as_quat();
    for (slot, mut wp, mut tf) in &mut slots {
        wp.0 = WorldPos::new(FrameId::CENTRAL_BODY, base + slot.offset);
        tf.rotation = rot;
    }
}

fn follow_camera(
    world: Res<SkinsWorld>,
    mut camera: Query<(&mut Transform, &mut WorldPlacement), With<AnchorCamera>>,
) {
    if let Ok((mut tf, mut placement)) = camera.single_mut() {
        let target = world.render_world();
        let eye = target + DVec3::new(18.0, 6.0, 0.0);
        placement.0 = WorldPos::new(FrameId::CENTRAL_BODY, eye);
        let look_dir = (target - eye).as_vec3().normalize_or_zero();
        if look_dir != Vec3::ZERO {
            tf.rotation = Transform::default().looking_to(look_dir, Vec3::Y).rotation;
        }
    }
}
