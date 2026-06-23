//! Land — craft↔terrain collision demo (`-- land`, WI 592).
//!
//! A craft is dropped above the textured ground with no thrust; gravity pulls it down and the
//! penalty contact response (`sounding_sim::contact`, via the WI 591 detection adapter) brings
//! it to rest on the surface — the first end-to-end collision slice. No player input; the HUD
//! shows altitude, speed, and a RESTING flag.

use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::DVec3;
use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use sounding_sim::active::ActiveBody;
use sounding_sim::attitude::{AttitudeControl, AttitudePilot, ReactionWheels, Sas};
use sounding_sim::contact::ContactParams;
use sounding_sim::control::ControlSystem;
use sounding_sim::flight::{flight_step, FlightCraft, FlightParams, GroundContact};
use sounding_sim::fluid::FluidMedium;
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::launch::LaunchPad;
use sounding_sim::medium::max_cross_section;
use sounding_sim::propulsion::Propulsion;
use sounding_sim::resource::ResourceGraph;
use sounding_sim::sim::CentralBody;
use sounding_sim::voxel::{Material, Voxel, VoxelCraft};

use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};
use crate::voxel_skin::{build_skin_mesh, material_set_for, pbr_material, VoxelSkin};

const BODY: CentralBody = CentralBody::EARTHLIKE;
const SUBSTEP_DT: f64 = 0.004;
const MAX_SUBSTEPS: u32 = 250;
const DROP_HEIGHT: f64 = 12.0;

/// The dropped craft.
#[derive(Resource)]
struct LandWorld {
    body: ActiveBody,
    craft: FlightCraft,
    params: FlightParams,
    pad: LaunchPad,
    accumulator: f64,
}

impl LandWorld {
    fn new() -> Self {
        // A compact 2×2×2 cube — a stable flat base for resting.
        let mut voxels = VoxelCraft::new(1.0);
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    voxels.voxels.push(Voxel {
                        cell: IVec3::new(x, y, z),
                        material: Material::COMPOSITE,
                    });
                }
            }
        }
        let mp = voxels.mass_properties().expect("non-empty craft");
        let drag_area = max_cross_section(&voxels);

        let propulsion = Propulsion {
            graph: ResourceGraph::default(),
            tank_mounts: vec![],
            engines: vec![],
            commands: vec![],
        };
        let attitude = AttitudePilot {
            sas: Sas::default(),
            manual: DVec3::ZERO,
            authority: 0.0,
            recapture_on_release: true,
            actuators: AttitudeControl {
                wheels: Some(ReactionWheels::new(0.0, 1.0)),
                rcs: None,
            },
        };

        // Start above the surface; a released pad so the craft free-falls (collision, not the
        // pad, brings it to rest).
        let surface = BODY.radius;
        let mut pad = LaunchPad::resting(surface);
        pad.released = true;
        let body = ActiveBody::new(
            DVec3::new(0.0, surface + DROP_HEIGHT, 0.0),
            DVec3::ZERO,
            mp.mass,
            mp.inertia,
        );

        Self {
            body,
            params: FlightParams {
                mu: BODY.mu,
                surface_radius: BODY.radius,
                medium: FluidMedium::EARTHLIKE,
                drag_area,
                drag_coefficient: 1.0,
                lift: None,
                ground: Some(GroundContact {
                    normal: DVec3::Y,
                    offset: surface,
                    contact: ContactParams::default(),
                }),
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
            pad,
            accumulator: 0.0,
        }
    }

    fn render_world(&self) -> DVec3 {
        self.body.position - DVec3::new(0.0, BODY.radius, 0.0)
    }

    fn altitude(&self) -> f64 {
        self.body.position.length() - BODY.radius
    }
}

#[derive(Component)]
struct CraftMarker;
#[derive(Component)]
struct Hud;

/// The landing demo scene.
pub struct LandScenePlugin;

impl Plugin for LandScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .insert_resource(LandWorld::new())
            .add_systems(Startup, setup_scene)
            .add_systems(
                Update,
                (step_land, track_craft, follow_camera, update_hud).chain(),
            );
    }
}

fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering: ResMut<Assets<ScatteringMedium>>,
    world: Res<LandWorld>,
) {
    crate::ground::spawn_ground(&mut commands, &mut meshes, &mut materials, &asset_server);

    let mesh = meshes.add(build_skin_mesh(&world.craft.voxels, VoxelSkin::Hull));
    let material = pbr_material(
        material_set_for(Material::COMPOSITE),
        &asset_server,
        &mut materials,
    );
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
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
        Text::new("land: dropping…"),
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

    let cam = world.render_world() + DVec3::new(14.0, 6.0, 14.0);
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(cam.as_vec3()).looking_at(Vec3::new(0.0, 2.0, 0.0), Vec3::Y),
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

fn step_land(time: Res<Time>, mut world: ResMut<LandWorld>) {
    world.accumulator += time.delta_secs_f64();
    let mut n = 0;
    while world.accumulator >= SUBSTEP_DT && n < MAX_SUBSTEPS {
        let LandWorld {
            body,
            craft,
            params,
            pad,
            ..
        } = &mut *world;
        flight_step(body, craft, params, pad, SUBSTEP_DT);
        world.accumulator -= SUBSTEP_DT;
        n += 1;
    }
}

fn track_craft(
    world: Res<LandWorld>,
    mut craft: Query<(&mut WorldPlacement, &mut Transform), With<CraftMarker>>,
) {
    if let Ok((mut wp, mut tf)) = craft.single_mut() {
        wp.0 = WorldPos::new(FrameId::CENTRAL_BODY, world.render_world());
        tf.rotation = world.body.orientation.as_quat();
    }
}

fn follow_camera(
    world: Res<LandWorld>,
    mut camera: Query<(&mut Transform, &mut WorldPlacement), With<AnchorCamera>>,
) {
    if let Ok((mut tf, mut placement)) = camera.single_mut() {
        let target = world.render_world();
        let eye = target + DVec3::new(14.0, 6.0, 14.0);
        placement.0 = WorldPos::new(FrameId::CENTRAL_BODY, eye);
        let look_dir = (target - eye).as_vec3().normalize_or_zero();
        if look_dir != Vec3::ZERO {
            tf.rotation = Transform::default().looking_to(look_dir, Vec3::Y).rotation;
        }
    }
}

fn update_hud(world: Res<LandWorld>, mut hud: Query<&mut Text, With<Hud>>) {
    if let Ok(mut text) = hud.single_mut() {
        let speed = world.body.velocity.length();
        let state = if speed < 0.1 { "RESTING" } else { "falling" };
        text.0 = format!(
            "land: {state}\naltitude: {alt:6.2} m\nspeed:    {speed:6.2} m/s",
            alt = world.altitude(),
        );
    }
}
