//! Launch — surface lift-off (WI 532), the first first-playable visible milestone.
//!
//! A craft rests on the launch pad (the unilateral support, `sounding_sim::launch`)
//! and, when its engine auto-throttles up, lifts off and climbs under thrust
//! (`sounding_sim::propulsion`, WI 531) against gravity + atmospheric drag — the
//! reverse of the dive. Rendering reuses the dive scene's floating-origin
//! flat-ground convention (sea level at world Y = 0, planet centre at `(0, −R, 0)`,
//! `CentralBody::EARTHLIKE`). The auto-throttle stands in for player controls
//! (WI 535) so the lift-off is visible on its own.

use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::DVec3;
use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use sounding_sim::active::ActiveBody;
use sounding_sim::command::Command;
use sounding_sim::fluid::FluidMedium;
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::launch::LaunchPad;
use sounding_sim::medium::{drag_force, max_cross_section};
use sounding_sim::propulsion::{Engine, EngineCommand, Propulsion};
use sounding_sim::resource::{Reservoir, ReservoirId, ResourceGraph, ResourceType};
use sounding_sim::voxel::{Material, Voxel, VoxelCraft};

use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};
use sounding_sim::sim::CentralBody;

/// The canonical Earth-like body, SI (shared, WI 527).
const BODY: CentralBody = CentralBody::EARTHLIKE;
/// Active sub-step, seconds.
const SUBSTEP_DT: f64 = 0.004;
/// Cap on sub-steps per frame.
const MAX_SUBSTEPS: u32 = 250;
/// Propellant tag.
const PROPELLANT: ResourceType = ResourceType(0);
/// Hold on the pad this long (s) before the auto-throttle ramps up.
const HOLD_TIME: f64 = 2.0;
/// Auto-throttle ramp duration (s).
const RAMP_TIME: f64 = 1.5;

/// The launching craft and its state. Self-contained, like the dive scene's world.
#[derive(Resource)]
struct LaunchWorld {
    body: ActiveBody,
    dry_com: DVec3,
    dry_mass: f64,
    propulsion: Propulsion,
    pad: LaunchPad,
    drag_area: f64,
    elapsed: f64,
    accumulator: f64,
}

impl LaunchWorld {
    fn new() -> Self {
        // A slim rocket: a 1×5×1 composite stack along +Y (up = thrust axis).
        let mut craft = VoxelCraft::new(1.0);
        for y in 0..5 {
            craft.voxels.push(Voxel {
                cell: IVec3::new(0, y, 0),
                material: Material::COMPOSITE,
            });
        }
        let mp = craft.mass_properties().expect("non-empty craft");
        let dry_mass = mp.mass;
        let dry_com = mp.center_of_mass;
        let drag_area = max_cross_section(&craft);

        // Engine at the base, thrust +Y, on the CoM's vertical line (no tip torque).
        // Sized for a thrust-to-weight a little above 1 when fuelled.
        let propellant = 4_000.0; // kg
        let propulsion = Propulsion {
            graph: ResourceGraph {
                reservoirs: vec![Reservoir::new(PROPELLANT, propellant, propellant)],
                ..Default::default()
            },
            tank_mounts: vec![DVec3::new(dry_com.x, 0.5, dry_com.z)],
            engines: vec![Engine {
                tank: ReservoirId(0),
                exhaust_velocity: 3_000.0,
                max_mass_flow: 60.0, // max thrust 180 kN (wet weight ≈ 118 kN → TWR ≈ 1.5)
                mount: DVec3::new(dry_com.x, 0.0, dry_com.z),
                axis: DVec3::Y,
                max_gimbal: 0.0,
            }],
            commands: vec![EngineCommand::default()],
        };

        // Rest the craft's **base** on the pad, not its CoM. The mesh is centred on
        // the CoM, which sits `base→CoM` (= `dry_com.y`, the lowest voxel base is at
        // local y = 0) above the base — so hold the CoM that far above the ground,
        // putting the base at altitude 0 (the pad). The pad's `surface_radius` is the
        // CoM rest radius; `altitude()` then reads 0 at rest (base on the pad).
        let base_to_com = dry_com.y;
        let pad_radius = BODY.radius + base_to_com;
        let body = ActiveBody::new(
            DVec3::new(0.0, pad_radius, 0.0),
            DVec3::ZERO,
            dry_mass + propellant,
            mp.inertia,
        );

        Self {
            body,
            dry_com,
            dry_mass,
            propulsion,
            pad: LaunchPad::resting(pad_radius),
            drag_area,
            elapsed: 0.0,
            accumulator: 0.0,
        }
    }

    /// The craft's flat-ground render position (sea level at Y = 0).
    fn render_world(&self) -> DVec3 {
        self.body.position - DVec3::new(0.0, BODY.radius, 0.0)
    }

    /// Auto-throttle: hold, then ramp 0 → 1 (stands in for player input, WI 535).
    fn target_throttle(&self) -> f64 {
        ((self.elapsed - HOLD_TIME) / RAMP_TIME).clamp(0.0, 1.0)
    }
}

/// Marks the rendered craft.
#[derive(Component)]
struct CraftMarker;

/// Marks the HUD.
#[derive(Component)]
struct Hud;

/// The launch scene.
pub struct LaunchScenePlugin;

impl Plugin for LaunchScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .insert_resource(LaunchWorld::new())
            .add_systems(Startup, setup_scene)
            .add_systems(
                Update,
                (step_launch, track_craft, follow_camera, update_hud).chain(),
            );
    }
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering: ResMut<Assets<ScatteringMedium>>,
    world: Res<LaunchWorld>,
) {
    // Ground: an opaque sphere whose surface is sea level (Y = 0).
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

    // The rocket (slim, tall along +Y).
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

    // The sun.
    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_x(-0.5) * Quat::from_rotation_y(0.6)),
    ));

    // HUD.
    commands.spawn((
        Text::new("throttle:   0%\naltitude:        0 m\nspeed:       0.0 m/s\nstate:    on pad"),
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

    // HDR camera with the physically-based atmosphere, beside the pad.
    let cam = world.render_world() + DVec3::new(14.0, 6.0, 14.0);
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

/// Sub-steps the launch: auto-throttle, then gravity + drag + thrust through the pad.
fn step_launch(time: Res<Time>, mut world: ResMut<LaunchWorld>) {
    world.accumulator += time.delta_secs_f64();
    let mut n = 0;
    while world.accumulator >= SUBSTEP_DT && n < MAX_SUBSTEPS {
        world.elapsed += SUBSTEP_DT;
        let throttle = world.target_throttle();
        world
            .propulsion
            .apply_command(&Command::SetThrottle(throttle));

        // Wet mass (propellant folds into mass/CoM, WI 531).
        let wet = world.propulsion.wet_mass(world.dry_mass, world.dry_com);
        world.body.mass = wet.mass;

        // Net wrench: gravity + atmospheric drag + thrust.
        let r = world.body.position.length();
        let up = if r > 0.0 {
            world.body.position / r
        } else {
            DVec3::Y
        };
        let g = -BODY.mu * world.body.mass / (r * r);
        let gravity = g * up;
        let altitude = r - BODY.radius;
        let sample = FluidMedium::EARTHLIKE.sample_altitude(altitude);
        let drag = drag_force(&sample, world.body.velocity, world.drag_area, 1.0);

        let (thrust, torque) = {
            let LaunchWorld {
                body, propulsion, ..
            } = &mut *world;
            propulsion.thrust_step(body.orientation, wet.center_of_mass, SUBSTEP_DT)
        };

        let force = gravity + drag + thrust;
        let LaunchWorld { body, pad, .. } = &mut *world;
        pad.step(body, force, torque, SUBSTEP_DT);

        world.accumulator -= SUBSTEP_DT;
        n += 1;
    }
}

fn track_craft(world: Res<LaunchWorld>, mut craft: Query<&mut WorldPlacement, With<CraftMarker>>) {
    if let Ok(mut wp) = craft.single_mut() {
        wp.0 = WorldPos::new(FrameId::CENTRAL_BODY, world.render_world());
    }
}

fn follow_camera(
    world: Res<LaunchWorld>,
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

fn update_hud(world: Res<LaunchWorld>, mut hud: Query<&mut Text, With<Hud>>) {
    if let Ok(mut text) = hud.single_mut() {
        let throttle = world.target_throttle() * 100.0;
        let altitude = world.pad.altitude(&world.body);
        let speed = world.body.velocity.length();
        let state = if world.pad.released {
            "ASCENT"
        } else {
            "on pad"
        };
        text.0 = format!(
            "throttle: {throttle:3.0}%\naltitude: {altitude:8.0} m\nspeed:    {speed:7.1} m/s\nstate:    {state}"
        );
    }
}
