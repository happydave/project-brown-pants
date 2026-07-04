//! Scenario — a scenario played from pure data (`-- scenario [path]`, WI 550).
//!
//! The first content-driven scene: everything it shows comes from a scenario
//! document (default `content/scenarios/first-flight.ron`) — the world, the
//! packs/settings/overrides composition, the starting blueprint, and the
//! catalog-resolved device physics. The scene itself only loads + stages the
//! validated payload, posts `Command::SpawnScenario`, and renders/routes
//! input; the sim-side director (`sounding_sim::director`) owns the spawn and
//! the flight stepping. All input is written to the **command message**
//! envelope (never applied directly), so the bus/MCP can fly this craft
//! exactly as the keyboard does.

use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::DVec3;
use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use sounding_sim::command::{Command, SasMode};
use sounding_sim::director::{DirectorPlugin, PendingSpawn, ScenarioFlight, ScenarioSpawn};
use sounding_sim::scenario::{load_scenario, ScenarioRoots};
use sounding_sim::voxel::Material;

use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};
use crate::voxel_skin::{build_skin_mesh, pbr_material, VoxelSkin};
use sounding_sim::frame::{FrameId, WorldPos};

/// The default scenario document when `-- scenario` is given no path.
const DEFAULT_SCENARIO: &str = "content/scenarios/first-flight.ron";

#[derive(Component)]
struct CraftMarker;
#[derive(Component)]
struct Hud;

/// The scenario scene: data in, flight out.
pub struct ScenarioScenePlugin;

impl Plugin for ScenarioScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .add_plugins(DirectorPlugin)
            .add_systems(Startup, load_and_stage)
            .add_systems(
                Update,
                (
                    spawn_visuals.run_if(resource_added::<ScenarioFlight>),
                    scenario_input,
                    crate::pause::toggle_pause,
                    crate::pause::step_scene,
                    track_craft,
                    follow_camera,
                    update_hud,
                )
                    .chain(),
            );
    }
}

/// Loads + validates the scenario document (arg 2 or the shipped default),
/// stages the resolved payload, and posts the spawn command. A bad document
/// fails fast with the loader's message — no half-built world.
fn load_and_stage(mut pending: ResMut<PendingSpawn>, mut commands: MessageWriter<Command>) {
    let path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| DEFAULT_SCENARIO.to_string());
    let roots = ScenarioRoots::default();
    let scenario = match load_scenario(std::path::Path::new(&path), &roots) {
        Ok(s) => s,
        Err(e) => panic!("scenario `{path}` failed to load: {e}"),
    };
    info!(
        "scenario `{}` ({}) loaded: {} catalog records, {} settings",
        scenario.id,
        scenario.name,
        scenario.catalog.ids().count(),
        scenario.catalog.settings.len(),
    );
    pending.0 = Some(ScenarioSpawn::from_scenario(&scenario));
    commands.write(Command::SpawnScenario);
}

/// Once the director has spawned the flight, dress the scene: ground, the
/// craft's real skin, light, camera, HUD.
fn spawn_visuals(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering: ResMut<Assets<ScatteringMedium>>,
    flight: Res<ScenarioFlight>,
) {
    crate::ground::spawn_ground(&mut commands, &mut meshes, &mut materials, &asset_server);

    let mesh = meshes.add(build_skin_mesh(&flight.craft.voxels, VoxelSkin::Hull));
    let material = pbr_material(Material::COMPOSITE, &asset_server, &mut materials);
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::default(),
        WorldPlacement(mesh_origin(&flight)),
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
        Text::new(format!("scenario: {}", flight.name)),
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
        Text::new("Z full throttle · X cut · T SAS hold · F SAS off · P pause · . step"),
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

    let cam = render_world(&flight) + DVec3::new(10.0, 4.0, 10.0);
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

/// The craft's render-frame position (pad-local: the world minus the surface
/// radius along +Y).
fn render_world(flight: &ScenarioFlight) -> DVec3 {
    flight.body.position - DVec3::new(0.0, flight.params.surface_radius, 0.0)
}

/// Mesh origin for the skin: the lattice origin (`position − orientation·dry_com`),
/// the same alignment every skinned scene uses (v0.1.51 fix).
fn mesh_origin(flight: &ScenarioFlight) -> WorldPos {
    let p = render_world(flight) - flight.body.orientation * flight.craft.dry_com;
    WorldPos::new(FrameId::CENTRAL_BODY, p)
}

/// Keyboard → the command envelope. Everything goes through messages: the
/// executor and the director's applicator do the rest (and the bus/MCP shares
/// this exact path).
fn scenario_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut commands: MessageWriter<Command>,
    mut attitude_active: Local<bool>,
) {
    if keys.just_pressed(KeyCode::KeyZ) {
        commands.write(Command::SetThrottle(1.0));
    }
    if keys.just_pressed(KeyCode::KeyX) {
        commands.write(Command::SetThrottle(0.0));
    }
    if keys.just_pressed(KeyCode::KeyT) {
        commands.write(Command::SetSas(SasMode::Hold));
    }
    if keys.just_pressed(KeyCode::KeyF) {
        commands.write(Command::SetSas(SasMode::Off));
    }

    // Manual attitude: WSAD/QE while held; one zero on release.
    let mut manual = DVec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        manual.x -= 1.0;
    }
    if keys.pressed(KeyCode::KeyS) {
        manual.x += 1.0;
    }
    if keys.pressed(KeyCode::KeyA) {
        manual.y += 1.0;
    }
    if keys.pressed(KeyCode::KeyD) {
        manual.y -= 1.0;
    }
    if keys.pressed(KeyCode::KeyQ) {
        manual.z += 1.0;
    }
    if keys.pressed(KeyCode::KeyE) {
        manual.z -= 1.0;
    }
    if manual != DVec3::ZERO {
        commands.write(Command::SetAttitude(manual));
        *attitude_active = true;
    } else if *attitude_active {
        commands.write(Command::SetAttitude(DVec3::ZERO));
        *attitude_active = false;
    }
}

fn track_craft(
    flight: Option<Res<ScenarioFlight>>,
    mut craft: Query<(&mut WorldPlacement, &mut Transform), With<CraftMarker>>,
) {
    let Some(flight) = flight else { return };
    if let Ok((mut wp, mut tf)) = craft.single_mut() {
        wp.0 = mesh_origin(&flight);
        tf.rotation = flight.body.orientation.as_quat();
    }
}

fn follow_camera(
    flight: Option<Res<ScenarioFlight>>,
    mut camera: Query<(&mut Transform, &mut WorldPlacement), With<AnchorCamera>>,
) {
    let Some(flight) = flight else { return };
    if let Ok((mut tf, mut placement)) = camera.single_mut() {
        let target = render_world(&flight);
        let eye = target + DVec3::new(10.0, 4.0, 10.0);
        placement.0 = WorldPos::new(FrameId::CENTRAL_BODY, eye);
        let look_dir = (target - eye).as_vec3().normalize_or_zero();
        if look_dir != Vec3::ZERO {
            tf.rotation = Transform::default().looking_to(look_dir, Vec3::Y).rotation;
        }
    }
}

fn update_hud(
    flight: Option<Res<ScenarioFlight>>,
    clock: Res<sounding_sim::sim::SimClock>,
    mut hud: Query<&mut Text, With<Hud>>,
) {
    let (Some(flight), Ok(mut text)) = (flight, hud.single_mut()) else {
        return;
    };
    let altitude = flight.pad.altitude(&flight.body);
    let speed = flight.body.velocity.length();
    let throttle = flight
        .craft
        .propulsion
        .commands
        .first()
        .map(|c| c.throttle)
        .unwrap_or(0.0);
    let prop = flight.craft.propulsion.graph.reservoirs[0].amount;
    let state = if flight.pad.released {
        "FLYING"
    } else {
        "on the pad"
    };
    let settings: String = flight
        .settings
        .iter()
        .map(|(name, s)| match &s.rationale {
            Some(r) => format!("\n  {name} ×{} — {r}", s.factor),
            None => format!("\n  {name} ×{}", s.factor),
        })
        .collect();
    // Mission lines (WI 551): name, state, and latched progress; plus the
    // most recent lore beat once one surfaces.
    let missions: String = flight
        .missions
        .iter()
        .map(|m| {
            format!(
                "\n  {} — {} ({:.0}%)",
                m.def.name,
                m.state,
                m.nodes.progress(&m.def.objective) * 100.0
            )
        })
        .collect();
    let missions = if missions.is_empty() {
        String::new()
    } else {
        format!("\nmissions:{missions}")
    };
    let lore = match &flight.lore {
        Some(beat) => format!("\n» {beat}"),
        None => String::new(),
    };
    text.0 = format!(
        "scenario: {name} — {state}{paused}\naltitude: {altitude:8.1} m\nspeed:    {speed:8.1} m/s\nthrottle: {throttle:4.2}   propellant: {prop:6.1} kg\nsettings:{settings}{missions}{lore}",
        name = flight.name,
        paused = crate::pause::paused_banner(&clock),
    );
}
