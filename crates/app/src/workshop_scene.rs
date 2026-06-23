//! Workshop — grounded build-and-test sandbox (`-- workshop`).
//!
//! The build-and-test loop in one scene with a **Build ↔ Test** toggle (`Enter`):
//!
//! - **Build** (WI 603): the voxel editor runs on the workshop's editable craft (reusing the
//!   `editor` module's systems under a state run-condition) — place/remove cells, materials,
//!   devices, the live mass/inertia gizmos. The edited lattice persists across toggles.
//! - **Test** (WI 599 / 602): a controllable craft on the textured ground, hand-flown through
//!   `flight::flight_step` with **live collision** — it lands, rests, drives, and shatters on a
//!   hard crash (`breakage::fracture_on_impact`), substep-capped near the surface
//!   (`warp::safe_substep_dt`). `Backspace` rebuilds the test craft.
//!
//! Build and Test are different coordinate worlds (the editor works near the origin; Test runs in
//! planetary coordinates with floating origin), so each mode spawns and despawns its own entities
//! on transition — they never coexist. (Until WI 604, Test flies a fixed preset craft, not the
//! Build lattice.)
//!
//! Test controls: Shift/Ctrl throttle · Z/X full/cut · W/S/A/D/Q/E attitude · T SAS · F off ·
//! `,`/`.` warp · Backspace reset. Build controls: arrows/PageUp-Dn cursor · Space add ·
//! Backspace remove · Tab material · Q/E/R/F/Z/C camera.

use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::DVec3;
use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;

use sounding_sim::active::ActiveBody;
use sounding_sim::attitude::{AttitudeControl, AttitudePilot, ReactionWheels, Sas};
use sounding_sim::breakage::fracture_on_impact;
use sounding_sim::collision::{
    craft_bounding_radius, craft_bounds, craft_collision_shape, ground_half_space, Bounds,
    CollisionShape,
};
use sounding_sim::command::{Command, SasMode};
use sounding_sim::contact::{body_contact_wrench, ground_contact_wrench, ContactParams};
use sounding_sim::control::{assemble_control, BatterySpec, ControlComputer};
use sounding_sim::flight::{flight_step, FlightCraft, FlightParams, GroundContact};
use sounding_sim::fluid::FluidMedium;
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::launch::LaunchPad;
use sounding_sim::medium::max_cross_section;
use sounding_sim::propulsion::{Engine, EngineCommand, Propulsion};
use sounding_sim::resource::{Reservoir, ReservoirId, ResourceGraph, ResourceType};
use sounding_sim::sim::CentralBody;
use sounding_sim::voxel::{Device, Material, Voxel, VoxelCraft};
use sounding_sim::warp::safe_substep_dt;

use crate::editor::{draw_editor, editor_input, orbit_camera, EditorState, OrbitCam};
use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};
use crate::voxel_skin::{build_skin_mesh, material_set_for, pbr_material, VoxelSkin};

const BODY: CentralBody = CentralBody::EARTHLIKE;
const SUBSTEP_DT: f64 = 0.004;
const MAX_SUBSTEPS: u32 = 250;
const PROPELLANT: ResourceType = ResourceType(0);
const THROTTLE_RATE: f64 = 1.0;
const MIN_WARP: f64 = 1.0;
const MAX_WARP: f64 = 8.0;
/// Contact tolerance for the anti-tunnel substep cap, m.
const CONTACT_TOL: f64 = 0.1;
/// A lightweight test frame: flies and lands fine, but a hard crash overruns its bonds.
const FRAME: Material = Material {
    density: 1_600.0,
    strength: 3.0e6,
};

/// Which half of the build-and-test loop is active.
#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
enum WorkshopMode {
    #[default]
    Build,
    Test,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum CraftState {
    Intact,
    Fractured,
}

/// The grounded workshop Test state: one controllable craft, or its debris after a crash.
#[derive(Resource)]
struct WorkshopWorld {
    body: ActiveBody,
    craft: FlightCraft,
    params: FlightParams,
    pad: LaunchPad,
    accumulator: f64,
    throttle: f64,
    warp: f64,
    state: CraftState,
    fragments: Vec<(VoxelCraft, ActiveBody)>,
    dirty: bool,
}

/// A 2×2×2 test craft: a stable base that rests without toppling, with a control point, a
/// computer, a battery, and a downward engine — assembled into a `FlightCraft`.
fn build_craft() -> (FlightCraft, ActiveBody, LaunchPad) {
    let mut voxels = VoxelCraft::new(1.0);
    for x in 0..2 {
        for y in 0..2 {
            for z in 0..2 {
                voxels.voxels.push(Voxel {
                    cell: IVec3::new(x, y, z),
                    material: FRAME,
                });
            }
        }
    }
    voxels
        .devices
        .push(Device::control_point(IVec3::new(0, 0, 0), 120.0, true));
    voxels.devices.push(Device::computer(
        IVec3::new(1, 1, 1),
        40.0,
        ControlComputer::tuning_computer(0.4),
    ));
    voxels.devices.push(Device::battery(
        IVec3::new(1, 0, 1),
        60.0,
        BatterySpec::full(120.0),
    ));

    let mp = voxels.mass_properties().expect("non-empty craft");
    let propellant = 3_000.0;

    let mut propulsion = Propulsion {
        graph: ResourceGraph {
            reservoirs: vec![Reservoir::new(PROPELLANT, propellant, propellant)],
            ..Default::default()
        },
        tank_mounts: vec![DVec3::new(mp.center_of_mass.x, 1.0, mp.center_of_mass.z)],
        engines: vec![Engine {
            tank: ReservoirId(0),
            exhaust_velocity: 3_000.0,
            max_mass_flow: 90.0, // ~270 kN; wet weight ≈ 156 kN → TWR ≈ 1.7
            mount: DVec3::new(mp.center_of_mass.x, 0.0, mp.center_of_mass.z),
            axis: DVec3::Y,
            max_gimbal: 0.0,
        }],
        commands: vec![EngineCommand::default()],
    };
    let mut control = assemble_control(&voxels, &mut propulsion.graph);
    control.low_power_reserve = 6.0;
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

    let rest_radius = BODY.radius + mp.center_of_mass.y;
    let body = ActiveBody::new(
        DVec3::new(0.0, rest_radius, 0.0),
        DVec3::ZERO,
        mp.mass + propellant,
        mp.inertia,
    );
    let mut pad = LaunchPad::resting(rest_radius);
    pad.released = true;

    let craft = FlightCraft {
        dry_mass: mp.mass,
        dry_com: mp.center_of_mass,
        voxels,
        propulsion,
        attitude,
        control,
        autopilot: None,
    };
    (craft, body, pad)
}

impl WorkshopWorld {
    fn new() -> Self {
        let (craft, body, pad) = build_craft();
        Self {
            body,
            params: FlightParams {
                mu: BODY.mu,
                surface_radius: BODY.radius,
                medium: FluidMedium::EARTHLIKE,
                drag_area: max_cross_section(&craft.voxels),
                drag_coefficient: 1.0,
                lift: None,
                ground: Some(GroundContact {
                    normal: DVec3::Y,
                    offset: BODY.radius,
                    contact: ContactParams::default(),
                }),
            },
            craft,
            pad,
            accumulator: 0.0,
            throttle: 0.0,
            warp: 1.0,
            state: CraftState::Intact,
            fragments: Vec::new(),
            dirty: true,
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    fn render_of(&self, pos: DVec3) -> DVec3 {
        pos - DVec3::new(0.0, BODY.radius, 0.0)
    }

    fn focus(&self) -> DVec3 {
        match self.state {
            CraftState::Intact => self.render_of(self.body.position),
            CraftState::Fractured => {
                if self.fragments.is_empty() {
                    DVec3::ZERO
                } else {
                    let sum: DVec3 = self.fragments.iter().map(|(_, b)| b.position).sum();
                    self.render_of(sum / self.fragments.len() as f64)
                }
            }
        }
    }

    fn altitude(&self) -> f64 {
        self.body.position.length() - BODY.radius
    }

    fn gravity_force(body: &ActiveBody) -> DVec3 {
        let r = body.position;
        let r2 = r.length_squared();
        if r2 <= 0.0 {
            return DVec3::ZERO;
        }
        -BODY.mu * body.mass * r / (r2 * r2.sqrt())
    }

    fn ground_shape(&self) -> CollisionShape {
        ground_half_space(BODY.radius)
    }

    /// Advance the intact craft one substep through the live flight pipeline, capping the step
    /// near the surface (anti-tunnel). Returns `true` if the craft fractured.
    fn step_intact(&mut self, frame_dt: f64) -> bool {
        let radius = craft_bounding_radius(&self.craft.voxels).unwrap_or(0.0);
        let gap = self.body.position.y - BODY.radius - radius;
        let approach = -self.body.velocity.y;
        let dt = safe_substep_dt(gap, approach, frame_dt, CONTACT_TOL);

        let WorkshopWorld {
            body,
            craft,
            params,
            pad,
            ..
        } = self;
        flight_step(body, craft, params, pad, dt);

        let shape = craft_collision_shape(&self.craft.voxels);
        let bounds = craft_bounds(&self.craft.voxels);
        let ground = self.ground_shape();
        let (cf, _) = ground_contact_wrench(
            &self.body,
            &shape,
            bounds,
            self.craft.dry_com,
            &ground,
            &ContactParams::default(),
        );
        if let Some(frags) = fracture_on_impact(&self.craft.voxels, &self.body, cf) {
            self.fragments = frags;
            self.state = CraftState::Fractured;
            self.dirty = true;
            return true;
        }
        false
    }

    /// Advance the debris one substep: gravity + ground + pairwise contact, then integrate.
    fn step_fragments(&mut self, dt: f64) {
        let ground = self.ground_shape();
        let params = ContactParams::default();
        let n = self.fragments.len();
        let shapes: Vec<CollisionShape> = self
            .fragments
            .iter()
            .map(|(v, _)| craft_collision_shape(v))
            .collect();
        let bounds: Vec<Option<Bounds>> = self
            .fragments
            .iter()
            .map(|(v, _)| craft_bounds(v))
            .collect();
        let coms: Vec<DVec3> = self
            .fragments
            .iter()
            .map(|(v, _)| {
                v.mass_properties()
                    .map(|mp| mp.center_of_mass)
                    .unwrap_or(DVec3::ZERO)
            })
            .collect();

        let mut acc = vec![(DVec3::ZERO, DVec3::ZERO); n];
        for i in 0..n {
            let (_, b) = &self.fragments[i];
            acc[i].0 += Self::gravity_force(b);
            let (gf, gt) =
                ground_contact_wrench(b, &shapes[i], bounds[i], coms[i], &ground, &params);
            acc[i].0 += gf;
            acc[i].1 += gt;
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let (_, bi) = &self.fragments[i];
                let (_, bj) = &self.fragments[j];
                let ((fa, ta), (fb, tb)) = body_contact_wrench(
                    bi, &shapes[i], bounds[i], coms[i], bj, &shapes[j], bounds[j], coms[j], &params,
                );
                acc[i].0 += fa;
                acc[i].1 += ta;
                acc[j].0 += fb;
                acc[j].1 += tb;
            }
        }
        for (i, (_, b)) in self.fragments.iter_mut().enumerate() {
            b.integrate_wrench(acc[i].0, acc[i].1, dt);
        }
    }
}

// --- Entity markers ---

/// Tags every entity owned by Test mode (despawned on leaving Test).
#[derive(Component)]
struct TestEntity;
/// Tags every entity owned by Build mode (despawned on leaving Build).
#[derive(Component)]
struct BuildEntity;
#[derive(Component)]
struct CraftMarker;
#[derive(Component)]
struct FragmentMarker(usize);
#[derive(Component)]
struct TestHud;
#[derive(Component)]
struct BuildHud;

/// The grounded build-and-test workshop scene.
pub struct WorkshopScenePlugin;

impl Plugin for WorkshopScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .init_state::<WorkshopMode>()
            .insert_resource(WorkshopWorld::new())
            .init_resource::<EditorState>()
            .init_resource::<OrbitCam>()
            .add_systems(OnEnter(WorkshopMode::Build), enter_build)
            .add_systems(OnExit(WorkshopMode::Build), exit_build)
            .add_systems(OnEnter(WorkshopMode::Test), enter_test)
            .add_systems(OnExit(WorkshopMode::Test), exit_test)
            .add_systems(Update, toggle_mode)
            .add_systems(
                Update,
                (editor_input, draw_editor, orbit_camera, update_build_hud)
                    .run_if(in_state(WorkshopMode::Build)),
            )
            .add_systems(
                Update,
                (
                    workshop_input,
                    step_workshop,
                    reconcile_meshes,
                    track_meshes,
                    follow_camera,
                    update_test_hud,
                )
                    .chain()
                    .run_if(in_state(WorkshopMode::Test)),
            );
    }
}

/// `Enter` toggles between Build and Test (from either mode).
fn toggle_mode(
    keys: Res<ButtonInput<KeyCode>>,
    state: Res<State<WorkshopMode>>,
    mut next: ResMut<NextState<WorkshopMode>>,
) {
    if keys.just_pressed(KeyCode::Enter) {
        next.set(match state.get() {
            WorkshopMode::Build => WorkshopMode::Test,
            WorkshopMode::Test => WorkshopMode::Build,
        });
    }
}

// --- Build mode ---

fn enter_build(mut commands: Commands) {
    // The editor's orbit camera (positioned each frame by `orbit_camera`); gizmos are unlit.
    commands.spawn((Camera3d::default(), Transform::default(), BuildEntity));
    commands.spawn((
        Text::new("workshop · BUILD"),
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
        BuildHud,
        BuildEntity,
    ));
    commands.spawn((
        Text::new(
            "arrows/PgUp-Dn cursor · Space add · Backspace remove · Tab material · G engine · QE/RF/ZC camera · Enter → TEST",
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
        BuildEntity,
    ));
}

fn exit_build(mut commands: Commands, q: Query<Entity, With<BuildEntity>>) {
    for e in &q {
        commands.entity(e).despawn();
    }
}

fn update_build_hud(editor: Res<EditorState>, mut hud: Query<&mut Text, With<BuildHud>>) {
    if let Ok(mut text) = hud.single_mut() {
        let mass = editor
            .craft
            .mass_properties()
            .map(|mp| mp.mass)
            .unwrap_or(0.0);
        text.0 = format!(
            "workshop · BUILD\nvoxels:  {}\ndevices: {}\nmass:    {mass:.0} kg\ncursor:  ({}, {}, {})",
            editor.craft.voxels.len(),
            editor.craft.devices.len(),
            editor.cursor.x,
            editor.cursor.y,
            editor.cursor.z,
        );
    }
}

// --- Test mode ---

fn enter_test(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering: ResMut<Assets<ScatteringMedium>>,
    mut world: ResMut<WorkshopWorld>,
) {
    world.reset(); // fresh test craft on the pad each time we enter Test

    let ground =
        crate::ground::spawn_ground(&mut commands, &mut meshes, &mut materials, &asset_server);
    commands.entity(ground).insert(TestEntity); // so it's cleaned up on exit
    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_x(-0.5) * Quat::from_rotation_y(0.6)),
        TestEntity,
    ));
    commands.spawn((
        Text::new("workshop · TEST"),
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
        TestHud,
        TestEntity,
    ));
    commands.spawn((
        Text::new(
            "Shift/Ctrl throttle · Z/X full/cut · WSAD QE attitude · T SAS  F off · ,/. warp · Backspace reset · Enter → BUILD",
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
        TestEntity,
    ));

    let cam = world.focus() + DVec3::new(14.0, 7.0, 16.0);
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
        TestEntity,
    ));
}

#[allow(clippy::type_complexity)]
fn exit_test(
    mut commands: Commands,
    q: Query<Entity, Or<(With<TestEntity>, With<CraftMarker>, With<FragmentMarker>)>>,
) {
    for e in &q {
        commands.entity(e).despawn();
    }
}

/// Translates keys into commands (throttle/attitude/SAS), plus warp and reset.
fn workshop_input(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut world: ResMut<WorkshopWorld>,
) {
    if keys.just_pressed(KeyCode::Backspace) {
        world.reset();
        return;
    }
    if world.state != CraftState::Intact {
        return; // debris isn't controllable
    }
    let dt = time.delta_secs_f64();
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
    world
        .craft
        .apply_command(&Command::SetThrottle(throttle), orientation);

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
    if keys.just_pressed(KeyCode::KeyF) {
        world
            .craft
            .apply_command(&Command::SetSas(SasMode::Off), orientation);
    }

    if keys.just_pressed(KeyCode::Period) {
        world.warp = (world.warp * 2.0).min(MAX_WARP);
    }
    if keys.just_pressed(KeyCode::Comma) {
        world.warp = (world.warp / 2.0).max(MIN_WARP);
    }
}

fn step_workshop(time: Res<Time>, mut world: ResMut<WorkshopWorld>) {
    world.accumulator += time.delta_secs_f64() * world.warp;
    let mut n = 0;
    while world.accumulator >= SUBSTEP_DT && n < MAX_SUBSTEPS {
        match world.state {
            CraftState::Intact => {
                if world.step_intact(SUBSTEP_DT) {
                    world.accumulator = 0.0;
                    break;
                }
            }
            CraftState::Fractured => world.step_fragments(SUBSTEP_DT),
        }
        world.accumulator -= SUBSTEP_DT;
        n += 1;
    }
}

/// Rebuilds the rendered craft/debris entities when the Test world changes (enter, fracture,
/// reset). Cheap: only on `dirty` frames.
fn reconcile_meshes(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut world: ResMut<WorkshopWorld>,
    craft_q: Query<Entity, With<CraftMarker>>,
    frag_q: Query<Entity, With<FragmentMarker>>,
) {
    if !world.dirty {
        return;
    }
    for e in &craft_q {
        commands.entity(e).despawn();
    }
    for e in &frag_q {
        commands.entity(e).despawn();
    }

    let material = pbr_material(
        material_set_for(Material::COMPOSITE),
        &asset_server,
        &mut materials,
    );
    match world.state {
        CraftState::Intact => {
            let mesh = meshes.add(build_skin_mesh(&world.craft.voxels, VoxelSkin::Hull));
            let render = world.render_of(world.body.position);
            commands.spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                Transform::default(),
                WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, render)),
                CraftMarker,
                TestEntity,
            ));
        }
        CraftState::Fractured => {
            for (i, (voxels, body)) in world.fragments.iter().enumerate() {
                let mesh = meshes.add(build_skin_mesh(voxels, VoxelSkin::Hull));
                let render = world.render_of(body.position);
                commands.spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(material.clone()),
                    Transform::default(),
                    WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, render)),
                    FragmentMarker(i),
                    TestEntity,
                ));
            }
        }
    }
    world.dirty = false;
}

#[allow(clippy::type_complexity)]
fn track_meshes(
    world: Res<WorkshopWorld>,
    mut sets: ParamSet<(
        Query<(&mut WorldPlacement, &mut Transform), With<CraftMarker>>,
        Query<(&FragmentMarker, &mut WorldPlacement, &mut Transform)>,
    )>,
) {
    match world.state {
        CraftState::Intact => {
            if let Ok((mut wp, mut tf)) = sets.p0().single_mut() {
                wp.0 = WorldPos::new(FrameId::CENTRAL_BODY, world.render_of(world.body.position));
                tf.rotation = world.body.orientation.as_quat();
            }
        }
        CraftState::Fractured => {
            for (tag, mut wp, mut tf) in &mut sets.p1() {
                if let Some((_, body)) = world.fragments.get(tag.0) {
                    wp.0 = WorldPos::new(FrameId::CENTRAL_BODY, world.render_of(body.position));
                    tf.rotation = body.orientation.as_quat();
                }
            }
        }
    }
}

fn follow_camera(
    world: Res<WorkshopWorld>,
    mut camera: Query<(&mut Transform, &mut WorldPlacement), With<AnchorCamera>>,
) {
    if let Ok((mut tf, mut placement)) = camera.single_mut() {
        let target = world.focus();
        let eye = target + DVec3::new(14.0, 7.0, 16.0);
        placement.0 = WorldPos::new(FrameId::CENTRAL_BODY, eye);
        let look_dir = (target - eye).as_vec3().normalize_or_zero();
        if look_dir != Vec3::ZERO {
            tf.rotation = Transform::default().looking_to(look_dir, Vec3::Y).rotation;
        }
    }
}

fn update_test_hud(world: Res<WorkshopWorld>, mut hud: Query<&mut Text, With<TestHud>>) {
    if let Ok(mut text) = hud.single_mut() {
        match world.state {
            CraftState::Intact => {
                let speed = world.body.velocity.length();
                let resting = speed < 0.1;
                let state = if resting { "RESTING" } else { "flying" };
                let sas = match world.craft.attitude.sas.mode {
                    SasMode::Off => "off",
                    SasMode::KillRotation => "kill-rot",
                    SasMode::Hold => "hold",
                    SasMode::Point(_) => "point",
                };
                text.0 = format!(
                    "workshop · TEST: {state}\nthrottle: {:3.0}%\naltitude: {:6.2} m\nv-speed:  {:+6.2} m/s\nspeed:    {:6.2} m/s\nSAS {sas}   warp {:.0}x",
                    world.throttle * 100.0,
                    world.altitude(),
                    world.body.velocity.y,
                    speed,
                    world.warp,
                );
            }
            CraftState::Fractured => {
                text.0 = format!(
                    "workshop · TEST: CRASHED — fractured into {} pieces\nBackspace to rebuild",
                    world.fragments.len()
                );
            }
        }
    }
}
