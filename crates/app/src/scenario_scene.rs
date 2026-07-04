//! Scenario — a scenario played from pure data (`-- scenario [path]`, WI 550;
//! flight-family presentation parity WI 739).
//!
//! The content-driven flight scene: everything it shows comes from a scenario
//! document — the world, the packs/settings/overrides composition, the
//! starting blueprint, the catalog-resolved device physics, and the missions.
//! The scene itself only loads + stages the validated payload, posts
//! `Command::SpawnScenario`, and renders/routes input; the sim-side director
//! (`sounding_sim::director`) owns the spawn and the flight stepping. All
//! input is written to the **command message** envelope (never applied
//! directly), so the bus/MCP can fly this craft exactly as the keyboard does.
//!
//! WI 739 folded the `-- play`/`-- launch`/`-- autopilot` scenes into this
//! presentation: those flags now alias this scene with their shipped scenario
//! documents (`content/scenarios/{play,launch,autopilot}.ron`), and the HUD/
//! controls match the old play scene — throttle ramp, manual attitude, SAS,
//! canned autopilots, gain tuning, control-tier select, time-warp, gamepad.
//!
//! Controls: Shift/Ctrl throttle up/down, Z/X full/cutoff · W/S/A/D/Q/E
//! attitude · T SAS hold toggle, R kill-rotation, F off, G recapture policy ·
//! 1/2/3/0 canned autopilots · [ ] − = SAS gains · C/V control tier · ,/.
//! warp · P pause, `.` step while paused.

use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::DVec3;
use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use sounding_sim::autopilot::{Autopilot, GravityTurn};
use sounding_sim::command::{Command, SasMode};
use sounding_sim::control::ControlTier;
use sounding_sim::director::{DirectorPlugin, PendingSpawn, ScenarioFlight, ScenarioSpawn};
use sounding_sim::fluid::MediumKind;
use sounding_sim::scenario::{load_scenario, ScenarioRoots};
use sounding_sim::session::{Outcome, Phase};
use sounding_sim::sim::SimClock;
use sounding_sim::telemetry::ActiveFlightTelemetry;

use crate::bus::ActiveFlight;
use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};
use crate::gamepad::{accumulate_chase_look, orbit_offset, ChaseLook, GamepadMap, PadSample};
use crate::voxel_skin::{pbr_material, skin_submeshes, VoxelSkin};
use sounding_sim::frame::{FrameId, WorldPos};

/// The default scenario document when the alias gives no path.
const DEFAULT_SCENARIO: &str = "content/scenarios/first-flight.ron";
/// Throttle ramp rate (per second) for Shift/Ctrl and the gamepad triggers.
const THROTTLE_RATE: f64 = 1.0;
/// Scene warp bounds (the play scene's range; the envelope clamps wider).
const MIN_WARP: f64 = 1.0;
const MAX_WARP: f64 = 16.0;

#[derive(Component)]
struct CraftMarker;
#[derive(Component)]
struct Hud;

/// The scenario document this scene loads when no explicit path is given —
/// set per launch alias (`-- play`, `-- launch`, `-- autopilot`, `-- scenario`).
#[derive(Resource)]
struct DefaultDoc(&'static str);

/// The scenario scene: data in, flight out.
pub struct ScenarioScenePlugin {
    /// The alias's shipped scenario document.
    pub default_doc: &'static str,
}

impl Default for ScenarioScenePlugin {
    fn default() -> Self {
        Self {
            default_doc: DEFAULT_SCENARIO,
        }
    }
}

impl Plugin for ScenarioScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .add_plugins(DirectorPlugin)
            .insert_resource(DefaultDoc(self.default_doc))
            .add_systems(Startup, load_and_stage)
            .add_systems(
                Update,
                (
                    spawn_visuals.run_if(resource_added::<ScenarioFlight>),
                    scenario_input,
                    crate::pause::toggle_pause,
                    crate::pause::step_scene,
                    publish_active_flight,
                    track_craft,
                    accumulate_chase_look,
                    follow_camera,
                    update_hud,
                    draw_attitude_gizmo,
                )
                    .chain(),
            );
    }
}

/// Loads + validates the scenario document (arg 2 or the alias's default),
/// stages the resolved payload, and posts the spawn command. A bad document
/// fails fast with the loader's message — no half-built world.
fn load_and_stage(
    default_doc: Res<DefaultDoc>,
    mut pending: ResMut<PendingSpawn>,
    mut commands: MessageWriter<Command>,
) {
    let path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| default_doc.0.to_string());
    let roots = ScenarioRoots::default();
    let scenario = match load_scenario(std::path::Path::new(&path), &roots) {
        Ok(s) => s,
        Err(e) => panic!("scenario `{path}` failed to load: {e}"),
    };
    info!(
        "scenario `{}` ({}) loaded: {} catalog records, {} settings, {} missions",
        scenario.id,
        scenario.name,
        scenario.catalog.ids().count(),
        scenario.catalog.settings.len(),
        scenario.missions.len(),
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
    mut look: ResMut<ChaseLook>,
    flight: Res<ScenarioFlight>,
) {
    look.reset(); // start each session at the default view (WI 665)
    crate::ground::spawn_ground(&mut commands, &mut meshes, &mut materials, &asset_server);
    // The planet: an opaque sphere whose surface is sea level (Y = 0).
    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(Sphere {
            radius: flight.params.surface_radius as f32,
        }))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.30, 0.27, 0.22),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::default(),
        WorldPlacement(WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(0.0, -flight.params.surface_radius, 0.0),
        )),
    ));

    // One submesh per distinct material, each bound to its own PBR appearance
    // (WI 821) — so a multi-material craft renders honestly in flight (previously
    // the whole hull bound the composite set) and glass cells draw translucent.
    for (material, mesh) in skin_submeshes(&flight.craft.voxels, VoxelSkin::Hull) {
        let mat = pbr_material(material, &asset_server, &mut materials);
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(mat),
            Transform::default(),
            WorldPlacement(mesh_origin(&flight)),
            CraftMarker,
        ));
    }

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
        Text::new(
            "Shift/Ctrl throttle · Z/X full/cut · WSAD QE attitude · T hold  R kill  F off · 1/2/3/0 autopilot · C/V tier · ,/. warp · P pause",
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

    let cam = render_world(&flight) + DVec3::new(16.0, 7.0, 16.0);
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

/// Keyboard + gamepad → the command envelope (WI 739, the play scene's control
/// map). Everything goes through messages: the executor and the director's
/// applicator do the rest (and the bus/MCP shares this exact path).
#[allow(clippy::too_many_arguments)]
fn scenario_input(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    pad_map: Res<GamepadMap>,
    clock: Res<SimClock>,
    flight: Option<Res<ScenarioFlight>>,
    mut commands: MessageWriter<Command>,
    mut attitude_active: Local<bool>,
) {
    let Some(flight) = flight else { return };
    let dt = time.delta_secs_f64();
    let pad = pad_map.sample(&gamepads);

    // Throttle: Shift/Ctrl or the triggers ramp the current value; Z/X or the
    // pad buttons snap it. The current value is read from the spawned craft
    // (the director applied the last command), so the ramp is stateless here.
    let current = flight
        .craft
        .propulsion
        .commands
        .first()
        .map(|c| c.throttle)
        .unwrap_or(0.0);
    let mut ramp = 0.0;
    if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
        ramp += 1.0;
    }
    if keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight) {
        ramp -= 1.0;
    }
    ramp += (pad.throttle_fwd - pad.throttle_rev) as f64;
    if ramp != 0.0 {
        commands.write(Command::SetThrottle(
            (current + THROTTLE_RATE * dt * ramp).clamp(0.0, 1.0),
        ));
    }
    if keys.just_pressed(KeyCode::KeyZ) || pad.throttle_max {
        commands.write(Command::SetThrottle(1.0));
    }
    if keys.just_pressed(KeyCode::KeyX) || pad.throttle_zero {
        commands.write(Command::SetThrottle(0.0));
    }

    // Manual attitude intent (pitch/yaw/roll about the body axes): WSAD/QE
    // while held, gamepad wins per-axis past the deadzone (WI 617); one zero
    // on release so SAS re-captures.
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
    if PadSample::active(pad.pitch) {
        manual.x = -pad.pitch as f64;
    }
    if PadSample::active(pad.roll) {
        manual.z = -pad.roll as f64;
    }
    if PadSample::active(pad.yaw) {
        manual.y = pad.yaw as f64;
    }
    if manual != DVec3::ZERO {
        commands.write(Command::SetAttitude(manual));
        *attitude_active = true;
    } else if *attitude_active {
        commands.write(Command::SetAttitude(DVec3::ZERO));
        *attitude_active = false;
    }

    // SAS mode.
    if keys.just_pressed(KeyCode::KeyT) || pad.sas_toggle {
        let mode = if flight.craft.attitude.sas.mode == SasMode::Hold {
            SasMode::Off
        } else {
            SasMode::Hold
        };
        commands.write(Command::SetSas(mode));
    }
    if keys.just_pressed(KeyCode::KeyR) {
        commands.write(Command::SetSas(SasMode::KillRotation));
    }
    if keys.just_pressed(KeyCode::KeyF) {
        commands.write(Command::SetSas(SasMode::Off));
    }
    // SAS hold-target re-capture policy (WI 564).
    if keys.just_pressed(KeyCode::KeyG) {
        commands.write(Command::SetSasRecapture(
            !flight.craft.attitude.recapture_on_release,
        ));
    }
    // Canned autopilots (WI 565): 1 prograde · 2 retrograde · 3 gravity-turn · 0 off.
    if keys.just_pressed(KeyCode::Digit1) {
        commands.write(Command::SetAutopilot(Some(Autopilot::Prograde)));
    }
    if keys.just_pressed(KeyCode::Digit2) {
        commands.write(Command::SetAutopilot(Some(Autopilot::Retrograde)));
    }
    if keys.just_pressed(KeyCode::Digit3) {
        commands.write(Command::SetAutopilot(Some(Autopilot::GravityTurn(
            GravityTurn::to_apoapsis(120_000.0),
        ))));
    }
    if keys.just_pressed(KeyCode::Digit0) {
        commands.write(Command::SetAutopilot(None));
    }
    // Live PID-gain tuning (WI 566, Tier 2): [ ] tune kp, - = tune kd.
    {
        let (kp, kd) = (flight.craft.attitude.sas.kp, flight.craft.attitude.sas.kd);
        let mut nkp = kp;
        let mut nkd = kd;
        if keys.just_pressed(KeyCode::BracketRight) {
            nkp += 1.0;
        }
        if keys.just_pressed(KeyCode::BracketLeft) {
            nkp -= 1.0;
        }
        if keys.just_pressed(KeyCode::Equal) {
            nkd += 1.0;
        }
        if keys.just_pressed(KeyCode::Minus) {
            nkd -= 1.0;
        }
        if (nkp, nkd) != (kp, kd) {
            commands.write(Command::SetSasGains(nkp.max(0.0), nkd.max(0.0)));
        }
    }

    // Player-selectable control tier (WI 571): C downshifts one rung (toward
    // Direct), V clears the downshift. Never selects below Direct.
    if keys.just_pressed(KeyCode::KeyC) {
        let avail = flight.craft.available_control();
        let current = flight.craft.control.selected.unwrap_or(avail);
        let lower = match current {
            ControlTier::Tunable => ControlTier::Canned,
            ControlTier::Canned => ControlTier::Stabilized,
            ControlTier::Stabilized => ControlTier::Direct,
            ControlTier::Direct | ControlTier::Uncontrolled => ControlTier::Direct,
        };
        commands.write(Command::SetControlTier(Some(lower)));
    }
    if keys.just_pressed(KeyCode::KeyV) {
        commands.write(Command::SetControlTier(None));
    }

    // Time-warp, via the envelope (the executor owns the clock). Scene-side
    // bounds are the play scene's 1–16×; `.` while paused is the step key.
    if !clock.paused {
        if keys.just_pressed(KeyCode::Period) {
            commands.write(Command::SetWarp(
                (clock.warp * 2.0).clamp(MIN_WARP, MAX_WARP),
            ));
        }
        if keys.just_pressed(KeyCode::Comma) {
            commands.write(Command::SetWarp(
                (clock.warp / 2.0).clamp(MIN_WARP, MAX_WARP),
            ));
        }
    }
}

/// Publishes the craft's autonomy state onto the bus bridge each frame
/// (WI 569 via WI 739), so an external client sees the live control tier —
/// exactly what the play scene published.
fn publish_active_flight(flight: Option<Res<ScenarioFlight>>, mut active: ResMut<ActiveFlight>) {
    if let Some(flight) = flight {
        active.0 = Some(ActiveFlightTelemetry::from_flight(&flight.craft));
    }
}

fn track_craft(
    flight: Option<Res<ScenarioFlight>>,
    mut craft: Query<(&mut WorldPlacement, &mut Transform), With<CraftMarker>>,
) {
    let Some(flight) = flight else { return };
    // One entity per material submesh (WI 821) — all share the craft's pose.
    for (mut wp, mut tf) in craft.iter_mut() {
        wp.0 = mesh_origin(&flight);
        tf.rotation = flight.body.orientation.as_quat();
    }
}

fn follow_camera(
    flight: Option<Res<ScenarioFlight>>,
    look: Res<ChaseLook>,
    mut camera: Query<(&mut Transform, &mut WorldPlacement), With<AnchorCamera>>,
) {
    let Some(flight) = flight else { return };
    if let Ok((mut tf, mut placement)) = camera.single_mut() {
        let target = render_world(&flight);
        // Default chase offset, orbited by the gamepad free-look (WI 665).
        let off = orbit_offset(Vec3::new(16.0, 7.0, 16.0), look.yaw, look.pitch);
        let eye = target + off.as_dvec3();
        placement.0 = WorldPos::new(FrameId::CENTRAL_BODY, eye);
        let look_dir = (target - eye).as_vec3().normalize_or_zero();
        if look_dir != Vec3::ZERO {
            tf.rotation = Transform::default().looking_to(look_dir, Vec3::Y).rotation;
        }
    }
}

fn draw_attitude_gizmo(
    flight: Option<Res<ScenarioFlight>>,
    mut gizmos: Gizmos,
    craft: Query<&Transform, With<CraftMarker>>,
) {
    let Some(flight) = flight else { return };
    // Submesh entities share one pose (WI 821) — any one anchors the gizmo.
    if let Some(tf) = craft.iter().next() {
        let pos = tf.translation;
        let nose = (flight.body.orientation.as_quat() * Vec3::Y).normalize_or_zero();
        gizmos.line(pos, pos + nose * 6.0, Color::srgb(0.3, 1.0, 0.3));
        let vel = flight.body.velocity.as_vec3();
        if vel.length() > 1.0 {
            gizmos.line(pos, pos + vel.normalize() * 5.0, Color::srgb(1.0, 0.6, 0.2));
        }
    }
}

/// Short HUD label for a control tier (WI 571).
fn tier_name(tier: ControlTier) -> &'static str {
    match tier {
        ControlTier::Uncontrolled => "UNCONTROLLED",
        ControlTier::Direct => "direct",
        ControlTier::Stabilized => "stabilized",
        ControlTier::Canned => "canned",
        ControlTier::Tunable => "tunable",
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

/// Specific orbital energy, J/kg.
fn specific_energy(flight: &ScenarioFlight) -> f64 {
    let r = flight.body.position.length();
    if r <= 0.0 {
        return 0.0;
    }
    0.5 * flight.body.velocity.length_squared() - flight.params.mu / r
}

/// (apoapsis_alt, periapsis_alt) above the surface in metres if the orbit is
/// bound, else `None`. Derived from the current 3D state.
fn apsides(flight: &ScenarioFlight) -> Option<(f64, f64)> {
    let r = flight.body.position.length();
    let energy = specific_energy(flight);
    if energy >= 0.0 || r <= 0.0 {
        return None; // unbound (escape)
    }
    let mu = flight.params.mu;
    let a = -mu / (2.0 * energy);
    let h = flight.body.position.cross(flight.body.velocity).length();
    let e = (1.0 + 2.0 * energy * h * h / (mu * mu)).max(0.0).sqrt();
    let apo = a * (1.0 + e) - flight.params.surface_radius;
    let peri = a * (1.0 - e) - flight.params.surface_radius;
    Some((apo, peri))
}

fn tilt_degrees(flight: &ScenarioFlight) -> f64 {
    let r = flight.body.position.length();
    if r <= 0.0 {
        return 0.0;
    }
    let up = flight.body.position / r;
    let nose = (flight.body.orientation * DVec3::Y).normalize_or_zero();
    nose.dot(up).clamp(-1.0, 1.0).acos().to_degrees()
}

fn update_hud(
    flight: Option<Res<ScenarioFlight>>,
    clock: Res<SimClock>,
    mut hud: Query<&mut Text, With<Hud>>,
) {
    let (Some(flight), Ok(mut text)) = (flight, hud.single_mut()) else {
        return;
    };
    let paused = crate::pause::paused_banner(&clock);
    let phase = match flight.session.phase {
        Phase::Build => "BUILD",
        Phase::Launch => "LAUNCH",
        Phase::Flight => "FLIGHT",
        Phase::Recovery => match flight.session.outcome {
            Outcome::Landed => "RECOVERY (landed)",
            Outcome::Crashed => "RECOVERY (crashed)",
            Outcome::None => "RECOVERY",
        },
    };
    let throttle = flight
        .craft
        .propulsion
        .commands
        .first()
        .map(|c| c.throttle)
        .unwrap_or(0.0);
    let fuel = flight.craft.propulsion.propellant();
    let fuel_cap = flight
        .craft
        .propulsion
        .graph
        .reservoirs
        .first()
        .map(|r| r.capacity)
        .unwrap_or(1.0);
    let flameout = throttle > 0.0 && fuel <= 1.0;
    let dv = flight.craft.propulsion.delta_v(flight.craft.dry_mass);
    let altitude = flight.pad.altitude(&flight.body);
    let medium = match flight.params.medium.sample_altitude(altitude).medium {
        MediumKind::Vacuum => "vacuum",
        MediumKind::Atmosphere => "atmosphere",
        MediumKind::Liquid => "ocean",
    };
    let r = flight.body.position.length();
    let up = if r > 0.0 {
        flight.body.position / r
    } else {
        DVec3::Y
    };
    let v_speed = flight.body.velocity.dot(up);
    let orbit_line = match apsides(&flight) {
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
    let tier = flight.craft.resolve_control();
    let sas = if !tier.allows_stabilization() {
        "unavail"
    } else {
        match flight.craft.attitude.sas.mode {
            SasMode::Off => "off",
            SasMode::KillRotation => "kill-rot",
            SasMode::Hold => "hold",
            SasMode::Point(_) => "point",
        }
    };
    let recap = if flight.craft.attitude.recapture_on_release {
        "recap"
    } else {
        "return"
    };
    let ctrl = tier_name(flight.craft.available_control());
    let sel = match flight.craft.control.selected {
        None => "auto",
        Some(t) => tier_name(t),
    };
    let eff = tier_name(tier);
    let assist = if flight.craft.assist_offline() {
        "  ASSIST OFFLINE (low power)"
    } else {
        ""
    };
    let (elec, elec_cap) = flight
        .craft
        .control
        .battery
        .and_then(|id| flight.craft.propulsion.graph.reservoirs.get(id.0))
        .map(|r| (r.amount, r.capacity))
        .unwrap_or((0.0, 0.0));
    let (kp, kd) = (flight.craft.attitude.sas.kp, flight.craft.attitude.sas.kd);
    let ap = match flight.craft.autopilot {
        None => "off".to_string(),
        Some(Autopilot::GravityTurn(_)) => "grav-turn".to_string(),
        Some(a) => format!("{a:?}").to_lowercase(),
    };
    let settings: String = flight
        .settings
        .iter()
        .map(|(name, s)| match &s.rationale {
            Some(r) => format!("\n  {name} ×{} — {r}", s.factor),
            None => format!("\n  {name} ×{}", s.factor),
        })
        .collect();
    let settings = if settings.is_empty() {
        String::new()
    } else {
        format!("\nsettings:{settings}")
    };
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
        "scenario: {name}\nphase:    {phase}{paused}\nthrottle: {tbar} {pct:3.0}%{note}\nfuel:     {fbar} {fuel:6.0} kg\npower:    {pbar} {elec:5.0} / {elec_cap:.0}\n\u{0394}v:       {dv:6.0} m/s\nG-force:  {g:5.1} g\naltitude: {alt}\nv-speed:  {v_speed:+7.0} m/s\nspeed:    {speed:7.1} m/s\n{orbit_line}\nenergy:   {energy:8.2} MJ/kg\nmedium:   {medium}   tilt {tilt:.0}\u{00b0}   SAS {sas} ({recap})\ncontrol:  avail {ctrl}  sel {sel}  eff {eff}   autopilot {ap}   gains kp={kp:.0}/kd={kd:.0}{assist}{settings}{missions}{lore}",
        name = flight.name,
        tbar = gauge(throttle),
        pct = throttle * 100.0,
        note = if flameout { "  FLAMEOUT" } else { "" },
        fbar = gauge(fuel / fuel_cap),
        pbar = gauge(if elec_cap > 0.0 { elec / elec_cap } else { 0.0 }),
        g = flight.g_force,
        alt = fmt_alt(altitude),
        speed = flight.body.velocity.length(),
        energy = specific_energy(&flight) / 1.0e6,
        tilt = tilt_degrees(&flight),
    );
}
