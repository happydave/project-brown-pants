//! Toy 9 — the dive (WI 509), the full **live chain** in SI (WI 527).
//!
//! One craft runs the whole gearbox on screen: it starts on an **SI Kepler orbit**
//! high above the surface, coasts down on rails under time warp, **auto-drops** to
//! the active gear at the atmospheric entry interface (`DiveTriggerPlugin` +
//! `HandoffPlugin`), and is then driven by the active-gear aero forces
//! (`DescentPlugin` → `glide_step`) — gravity + drag + buoyancy + lift — gliding
//! and weathervaning through vacuum → atmosphere → ocean to splashdown. All one
//! consistent SI unit system, shared with the headless app and the planet scene
//! via `CentralBody::EARTHLIKE` (no per-scene unit convention).
//!
//! Rendering uses the floating-origin flat-ground convention (sea level at world
//! Y = 0, planet centre at `(0, -R, 0)`); the craft's sim position (its orbit while
//! on rails, its `ActiveBody` once active) is converted for display each frame.

use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::{DVec2, DVec3};
use bevy::mesh::VertexAttributeValues;
use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use sounding_sim::active::{ActiveBody, Gravity};
use sounding_sim::fluid::{FluidMedium, MediumKind};
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::handoff::{orbit_state_3d, GearState, HandoffPlugin};
use sounding_sim::medium::{
    dynamic_pressure, max_cross_section, CraftThermal, DescentParams, DescentPlugin,
    DiveTriggerPlugin, DivingCraft, EntryInterface, GlideParams, DEFAULT_SLAM_COEFFICIENT,
    DIVE_HEAT_SCALE,
};
use sounding_sim::orbit::Orbit;
use sounding_sim::sim::{CentralBody, Craft, SimClock};
use sounding_sim::telemetry::ThermalTelemetry;
use sounding_sim::voxel::{Axis, Material, Voxel, VoxelCraft};

use crate::bus::DiveThermal;
use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};

/// Ambient / radiative-sink temperature for the dive, K.
const DIVE_AMBIENT: f64 = 250.0;
/// Skin temperature (K) at which the craft begins to visibly glow.
const GLOW_ONSET: f64 = 500.0;

/// The canonical Earth-like body, in SI (WI 527).
const BODY: CentralBody = CentralBody::EARTHLIKE;
/// Starting altitude of the orbit's high point, metres.
const START_ALT: f64 = 120_000.0;
/// Tangential entry speed, m/s (WI 693): a genuine **orbital** re-entry (~7 km/s,
/// just under circular at this altitude) rather than the old ~600 m/s suborbital
/// lob — so heating is physically dramatic. Periapsis falls into the atmosphere, so
/// the craft re-enters and reaches the ocean (validated headless).
const ENTRY_SPEED: f64 = 7_000.0;
/// Atmospheric-entry interface altitude, metres (where warp drops and the craft
/// wakes into active physics).
const ENTRY_ALT: f64 = 100_000.0;
/// Active-descent sub-step, seconds (the stiff entry/splashdown wants a small step).
const SUBSTEP_DT: f64 = 0.002;
/// Cap on active descent sub-steps per frame.
const MAX_SUBSTEPS: u32 = 4_000;
/// Initial time warp for the on-rails coast down to the interface.
const INITIAL_WARP: f64 = 30.0;
/// Depth (negative altitude) at which the sim is paused so the craft rests rather
/// than tunnelling the (non-collision) seabed.
const REST_DEPTH: f64 = -3_500.0;

/// A slender re-entry body along +Z (forward): a 3×3×4 composite hull with an
/// **ablative heat-shield nose tip** (WI 688) at the windward front — a positive
/// static margin so it weathervanes into the airflow, a tapered area curve for
/// transonic wave drag (WI 526), and a shield that ablates to survive re-entry.
fn dive_craft() -> VoxelCraft {
    let mut c = VoxelCraft::new(1.0);
    for z in 0..4 {
        for x in 0..3 {
            for y in 0..3 {
                c.voxels.push(Voxel {
                    cell: IVec3::new(x, y, z),
                    material: Material::COMPOSITE,
                });
            }
        }
    }
    c.voxels.push(Voxel {
        cell: IVec3::new(1, 1, 4),
        material: Material::ABLATOR,
    });
    c
}

/// The craft's flat-ground render position (sea level at Y = 0): the radial sim
/// frame shifted so the planet centre sits one radius below the origin.
fn render_world(sim_pos: DVec3) -> DVec3 {
    sim_pos - DVec3::new(0.0, BODY.radius, 0.0)
}

/// Live readout of the craft, refreshed each frame for the HUD.
#[derive(Resource, Default)]
struct DiveReadout {
    active: bool,
    altitude: f64,
    speed: f64,
    medium: MediumKind,
    pressure: f64,
    ram: f64,
    /// Hottest skin temperature, K (WI 691).
    skin_temp: f64,
    /// Any voxel at/over its material limit (overheating / burn-through).
    over_limit: bool,
    /// Remaining ablative-shield fraction, or `None` if no shield (WI 688).
    ablator_remaining: Option<f64>,
    /// Steam mass vaporised last step, kg (WI 698) — the boiling-clamp output that drives
    /// the steam VFX intensity (replaces the old skin-temperature proxy).
    steam_mass: f64,
}

/// Mouse-driven orbit/zoom state for the follow camera (WI 702): the eye sits at
/// `craft + orbit_offset(yaw, pitch, dist)`. The default reproduces the historical fixed
/// `(18, 8, 18)` view, so the scene opens unchanged.
#[derive(Resource)]
struct DiveCam {
    /// Orbit yaw about the craft (radians).
    yaw: f32,
    /// Orbit pitch above the horizon (radians), clamped away from the poles.
    pitch: f32,
    /// Eye distance from the craft (metres).
    dist: f32,
}

impl Default for DiveCam {
    fn default() -> Self {
        // Matches the previous fixed offset (18, 8, 18): yaw = π/4, a gentle downward pitch,
        // distance ≈ 26.68 m.
        Self {
            yaw: std::f32::consts::FRAC_PI_4,
            pitch: 0.305,
            dist: 26.683,
        }
    }
}

/// The camera eye offset from the orbit target for a yaw/pitch/distance (WI 702) — the
/// spherical-to-cartesian the gallery/editor orbit cameras use. Pure (unit-tested).
fn orbit_offset(yaw: f32, pitch: f32, dist: f32) -> Vec3 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    Vec3::new(sy * cp, sp, cy * cp) * dist
}

/// Marks the heads-up readout.
#[derive(Component)]
struct Hud;

/// Marks the rendered craft.
#[derive(Component)]
struct CraftMarker;

/// The Toy 9 dive scene.
pub struct DiveScenePlugin;

impl Plugin for DiveScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .init_resource::<DiveReadout>()
            // The handoff sleep path reads Gravity; the descent driver replaces the
            // pure-gravity ActivePlugin (glide_step already includes gravity).
            .insert_resource(Gravity { mu: BODY.mu })
            .add_plugins(HandoffPlugin)
            .add_plugins(DiveTriggerPlugin {
                interface: EntryInterface {
                    surface_radius: BODY.radius,
                    altitude: ENTRY_ALT,
                },
            })
            .add_plugins(DescentPlugin {
                substep_dt: SUBSTEP_DT,
                max_substeps: MAX_SUBSTEPS,
            })
            .init_resource::<SteamAssets>()
            .init_resource::<SplashAssets>()
            .init_resource::<DiveCam>()
            .add_systems(Startup, setup_scene)
            .add_systems(
                Update,
                (
                    track_craft,
                    track_thermal,
                    dive_camera_input,
                    follow_camera,
                    update_hud,
                )
                    .chain(),
            )
            // Phase-change spike (WI 695): steam when the hot craft hits the ocean.
            .add_systems(Update, (emit_steam, update_steam))
            // Water-entry splash (WI 699, temporary): a one-shot spray burst on splashdown.
            .add_systems(Update, (emit_splash, update_splash))
            // Water surface (WI 703): a local animated patch ripples + follows the camera.
            .add_systems(Update, animate_water);
    }
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering: ResMut<Assets<ScatteringMedium>>,
    mut clock: ResMut<SimClock>,
    mut craft_q: Query<(Entity, &mut Craft, &mut GearState)>,
) {
    // A genuine **orbital** re-entry from START_ALT (WI 693): a bound Kepler orbit at
    // near-circular speed whose periapsis falls into the atmosphere, so the craft
    // enters at ~7 km/s — carrying orbital kinetic energy, so re-entry heating is
    // physically dramatic (heating ∝ v³). Start at the +Y high point with a +X
    // tangential velocity so "up" ≈ +Y, matching the flat-ground render convention.
    let r0 = BODY.radius + START_ALT;
    let orbit = Orbit::from_state(
        BODY.mu,
        DVec2::new(0.0, r0),
        DVec2::new(ENTRY_SPEED, 0.0),
        0.0,
    )
    .expect("bound re-entry orbit");

    let voxels = dive_craft();
    let mp = voxels.mass_properties().expect("non-empty craft");
    let descent = DescentParams {
        medium: FluidMedium::EARTHLIKE,
        mu: BODY.mu,
        surface_radius: BODY.radius,
        drag_area: max_cross_section(&voxels),
        drag_coefficient: 1.0,
        slam_coefficient: DEFAULT_SLAM_COEFFICIENT,
    };
    let glide = GlideParams::for_craft(descent, &voxels, Axis::Z);
    let start_render = render_world(orbit_state_3d(&orbit, 0.0).0);

    // Reconfigure the single craft the OrbitPlugin spawned (main.rs): put it on the
    // re-entry orbit with a real gear-state, and attach its diving description plus
    // the render bundle. One craft runs the whole chain.
    if let Ok((entity, mut craft, mut gear)) = craft_q.single_mut() {
        craft.orbit = orbit;
        *gear = GearState::new(mp.mass, mp.inertia);
        let thermal = CraftThermal::new(&voxels, DIVE_AMBIENT, DIVE_AMBIENT, DIVE_HEAT_SCALE);
        commands.entity(entity).insert((
            DivingCraft {
                craft: voxels,
                com: mp.center_of_mass,
                glide,
            },
            thermal,
            Mesh3d(meshes.add(Mesh::from(Cuboid::new(3.0, 3.0, 5.0)))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.85, 0.84, 0.88),
                metallic: 0.6,
                perceptual_roughness: 0.3,
                ..default()
            })),
            Transform::default(),
            WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, start_render)),
            CraftMarker,
        ));
    }
    // Coast down under warp; the entry trigger drops it to 1 at the interface.
    clock.warp = INITIAL_WARP;

    // Seabed: an opaque sphere just below sea level, centred one radius down.
    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(Sphere {
            radius: (BODY.radius - 4_000.0) as f32,
        }))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.20, 0.17, 0.14),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::default(),
        WorldPlacement(WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(0.0, -BODY.radius, 0.0),
        )),
    ));

    // Distant ocean: a deep, reflective blue sphere providing the broad ocean + horizon.
    // Sunk a couple of metres below sea level so the animated near-surface patch (which
    // oscillates around Y = 0) does not z-fight its surface (WI 703).
    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(Sphere {
            radius: BODY.radius as f32,
        }))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.02, 0.12, 0.26, 0.92),
            alpha_mode: AlphaMode::Blend,
            perceptual_roughness: 0.06,
            reflectance: 0.6,
            ..default()
        })),
        Transform::default(),
        WorldPlacement(WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(0.0, -BODY.radius - 2.0, 0.0),
        )),
    ));

    // Near surface: a local animated water patch at sea level that follows the camera and
    // ripples, so the ocean reads as a moving surface and splash/steam register against it
    // (WI 703). Initial placement under the craft; `animate_water` tracks the camera each frame.
    commands.spawn((
        Mesh3d(
            meshes.add(Mesh::from(
                Plane3d::default()
                    .mesh()
                    .size(2.0 * WATER_HALF, 2.0 * WATER_HALF)
                    .subdivisions(WATER_SUBDIV),
            )),
        ),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.10, 0.30, 0.46, 0.78),
            alpha_mode: AlphaMode::Blend,
            perceptual_roughness: 0.08,
            reflectance: 0.6,
            ..default()
        })),
        Transform::default(),
        WorldPlacement(WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(start_render.x, 0.0, start_render.z),
        )),
        WaterPatch,
    ));

    // The sun: raw sunlight the atmosphere filters.
    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_x(-0.5) * Quat::from_rotation_y(0.6)),
    ));

    // Heads-up readout.
    commands.spawn((
        Text::new(
            "gear:       on-rails\naltitude:        0 m\nspeed:       0.0 m/s\nmedium:   vacuum\nhull P:        0.0 kPa\nram P:         0.0 kPa\nskin T:        250 K",
        ),
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

    // HDR camera with Bevy's physically-based atmosphere, chasing the craft.
    let cam = start_render + DVec3::new(18.0, 8.0, 18.0);
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(cam.as_vec3()).looking_at(start_render.as_vec3(), Vec3::Y),
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

/// Reads the craft's current gear (orbit while on rails, `ActiveBody` once active),
/// updates its render placement + attitude, and refreshes the HUD readout. Pauses
/// the sim once the craft is well underwater so it rests rather than tunnelling.
#[allow(clippy::type_complexity)] // a Bevy gear-agnostic craft query
fn track_craft(
    mut clock: ResMut<SimClock>,
    mut readout: ResMut<DiveReadout>,
    mut craft: Query<
        (
            Option<&Craft>,
            Option<&ActiveBody>,
            &DivingCraft,
            &mut WorldPlacement,
            &mut Transform,
        ),
        With<CraftMarker>,
    >,
) {
    let Ok((rails, active, dc, mut wp, mut tf)) = craft.single_mut() else {
        return;
    };
    let (sim_pos, velocity, rotation, is_active) = if let Some(body) = active {
        (
            body.position,
            body.velocity,
            body.orientation.as_quat(),
            true,
        )
    } else if let Some(c) = rails {
        let (p, v) = orbit_state_3d(&c.orbit, clock.time);
        (p, v, Quat::IDENTITY, false)
    } else {
        return;
    };

    wp.0 = WorldPos::new(FrameId::CENTRAL_BODY, render_world(sim_pos));
    tf.rotation = rotation;

    let altitude = sim_pos.length() - BODY.radius;
    let sample = dc.glide.descent.medium.sample_altitude(altitude);
    readout.active = is_active;
    readout.altitude = altitude;
    readout.speed = velocity.length();
    readout.medium = sample.medium;
    readout.pressure = sample.pressure;
    readout.ram = dynamic_pressure(&sample, velocity);

    // Rest on the seabed (not a collision surface this toy): pause once deep.
    if altitude <= REST_DEPTH {
        clock.paused = true;
    }
}

/// Reads the craft's thermal state (WI 691): publishes the skin-temperature readout
/// to the HUD and the bus, and makes the craft glow red→white-hot as it heats.
fn track_thermal(
    mut readout: ResMut<DiveReadout>,
    mut bridge: ResMut<DiveThermal>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    craft: Query<
        (
            &DivingCraft,
            &CraftThermal,
            &MeshMaterial3d<StandardMaterial>,
        ),
        With<CraftMarker>,
    >,
) {
    let Ok((dc, thermal, mat)) = craft.single() else {
        return;
    };
    let (max_skin, over_limit, ablator_remaining) = thermal.readout(&dc.craft);
    readout.skin_temp = max_skin;
    readout.over_limit = over_limit;
    readout.ablator_remaining = ablator_remaining;
    readout.steam_mass = thermal.state.steam_mass(); // WI 698: boiling-clamp output
    bridge.0 = Some(ThermalTelemetry {
        max_skin_temp: max_skin,
        over_limit,
        ablator_remaining,
    });
    if let Some(material) = materials.get_mut(&mat.0) {
        material.emissive = glow_color(max_skin);
    }
}

/// A heating glow: black below [`GLOW_ONSET`], ramping red → orange → white-hot as
/// skin temperature climbs. Returned over-bright (HDR) so the dive's bloom blooms.
fn glow_color(skin_temp: f64) -> LinearRgba {
    let x = (((skin_temp - GLOW_ONSET) / 2_000.0).clamp(0.0, 1.0)) as f32;
    if x <= 0.0 {
        return LinearRgba::BLACK;
    }
    // Red leads, green follows (→orange/yellow), blue last (→white); brightness grows.
    LinearRgba::rgb(x * 8.0, x * x * 8.0, x * x * x * 8.0)
}

/// Reads mouse input into the orbit camera state (WI 702): middle-drag orbits (yaw/pitch),
/// the wheel zooms (distance) — the editor/gallery convention, leaving left/right free.
fn dive_camera_input(
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut cam: ResMut<DiveCam>,
) {
    if buttons.pressed(MouseButton::Middle) {
        cam.yaw += motion.delta.x * 0.01;
        cam.pitch = (cam.pitch + motion.delta.y * 0.01).clamp(-1.4, 1.4);
    }
    if scroll.delta.y != 0.0 {
        // Zoom step scales with distance so it stays usable close in and far out.
        cam.dist = (cam.dist - scroll.delta.y * cam.dist * 0.1).clamp(5.0, 2_000.0);
    }
}

/// Keeps the anchor camera orbiting/zooming the craft's render position (WI 702): the eye is
/// `craft + orbit_offset(DiveCam)`, still tracking the craft every frame.
#[allow(clippy::type_complexity)] // disjoint Bevy queries (craft vs. camera)
fn follow_camera(
    cam: Res<DiveCam>,
    craft: Query<&WorldPlacement, (With<CraftMarker>, Without<AnchorCamera>)>,
    mut camera: Query<
        (&mut Transform, &mut WorldPlacement),
        (With<AnchorCamera>, Without<CraftMarker>),
    >,
) {
    let Ok(craft_wp) = craft.single() else {
        return;
    };
    let Ok((mut tf, mut placement)) = camera.single_mut() else {
        return;
    };
    let target = craft_wp.0.pos;
    let eye = target + orbit_offset(cam.yaw, cam.pitch, cam.dist).as_dvec3();
    placement.0 = WorldPos::new(FrameId::CENTRAL_BODY, eye);
    let look_dir = (target - eye).as_vec3().normalize_or_zero();
    if look_dir != Vec3::ZERO {
        tf.rotation = Transform::default().looking_to(look_dir, Vec3::Y).rotation;
    }
}

fn update_hud(readout: Res<DiveReadout>, mut hud: Query<&mut Text, With<Hud>>) {
    if let Ok(mut text) = hud.single_mut() {
        let gear = if readout.active { "active" } else { "on-rails" };
        let medium = match readout.medium {
            MediumKind::Vacuum => "vacuum",
            MediumKind::Atmosphere => "atmosphere",
            MediumKind::Liquid => "ocean",
        };
        let alt = readout.altitude;
        let speed = readout.speed;
        let pressure_kpa = readout.pressure / 1_000.0;
        let ram_kpa = readout.ram / 1_000.0;
        let skin = readout.skin_temp;
        let heat = if readout.over_limit {
            "  *** OVERHEAT ***"
        } else {
            ""
        };
        // Ablative-shield budget (WI 688): a percentage that drains as the nose ablates.
        let shield = match readout.ablator_remaining {
            Some(frac) => format!("\nshield:   {:7.0} %", frac * 100.0),
            None => String::new(),
        };
        text.0 = format!(
            "gear:     {gear}\naltitude: {alt:8.0} m\nspeed:    {speed:7.1} m/s\nmedium:   {medium}\nhull P:   {pressure_kpa:8.1} kPa\nram P:    {ram_kpa:8.1} kPa\nskin T:   {skin:8.0} K{heat}{shield}"
        );
    }
}

// --- Phase-change (WI 695 VFX, WI 698 coupling): water → steam from actual boiling ---

/// Steam mass (kg/step) at which the steam VFX is at full intensity (WI 698). A tuning
/// knob for the deferred general VFX pass — the visual rate/opacity scale with the
/// boiling vigour, which the sim now reports as vaporised mass.
const STEAM_MASS_FULL: f64 = 0.2;
/// Maximum concurrent steam puffs (a bounded pool).
const STEAM_MAX: usize = 80;
/// Steam puffs emitted per second at full intensity.
const STEAM_RATE: f32 = 40.0;

/// Steam emission intensity, 0–1 (pure, unit-tested): driven by the boiling clamp's
/// vaporised **steam mass** (WI 698), not a skin-temperature proxy — so steam appears
/// exactly when and as hard as the surface is actually boiling (the sim already produces
/// steam mass only when submerged and above the boiling point). The visual rate/opacity
/// scale with this.
fn steam_intensity(steam_mass: f64) -> f64 {
    (steam_mass / STEAM_MASS_FULL).clamp(0.0, 1.0)
}

/// A rising, growing, fading steam puff.
#[derive(Component)]
struct SteamPuff {
    age: f32,
    lifetime: f32,
    vel: Vec3,
}

/// The shared steam puff mesh (one sphere, instanced per puff).
#[derive(Resource)]
struct SteamAssets {
    mesh: Handle<Mesh>,
}

impl FromWorld for SteamAssets {
    fn from_world(world: &mut World) -> Self {
        let mut meshes = world.resource_mut::<Assets<Mesh>>();
        Self {
            mesh: meshes.add(Mesh::from(Sphere { radius: 1.0 })),
        }
    }
}

/// Spawns steam puffs at the craft while it is hot and in/just above the ocean,
/// at a rate proportional to the steam intensity (WI 695, prototype).
#[allow(clippy::too_many_arguments)]
fn emit_steam(
    mut commands: Commands,
    time: Res<Time>,
    readout: Res<DiveReadout>,
    assets: Res<SteamAssets>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    craft: Query<&Transform, With<CraftMarker>>,
    existing: Query<(), With<SteamPuff>>,
    mut accum: Local<f32>,
    mut seed: Local<u32>,
) {
    let intensity = steam_intensity(readout.steam_mass) as f32;
    if intensity <= 0.0 {
        *accum = 0.0;
        return;
    }
    let Ok(craft_tf) = craft.single() else {
        return;
    };
    if existing.iter().count() >= STEAM_MAX {
        return;
    }
    *accum += time.delta_secs() * intensity * STEAM_RATE;
    while *accum >= 1.0 {
        *accum -= 1.0;
        // A cheap LCG for scatter (no rng dependency; gameplay-noncritical).
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let r = |shift: u32| (((*seed >> shift) & 0xff) as f32 / 255.0) - 0.5;
        let offset = Vec3::new(r(0) * 4.0, 0.0, r(8) * 4.0);
        let vel = Vec3::new(r(16) * 2.0, 6.0 + r(4) * 3.0, r(24) * 2.0);
        let mat = materials.add(StandardMaterial {
            base_color: Color::srgba(0.9, 0.92, 0.95, 0.5),
            emissive: LinearRgba::rgb(0.25, 0.25, 0.28),
            alpha_mode: AlphaMode::Blend,
            ..default()
        });
        commands.spawn((
            Mesh3d(assets.mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_translation(craft_tf.translation + offset).with_scale(Vec3::splat(0.8)),
            SteamPuff {
                age: 0.0,
                lifetime: 1.6,
                vel,
            },
        ));
    }
}

/// Rises, grows, and fades each steam puff, despawning it (and freeing its material)
/// at the end of its life (WI 695, prototype).
fn update_steam(
    mut commands: Commands,
    time: Res<Time>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut puffs: Query<(
        Entity,
        &mut Transform,
        &mut SteamPuff,
        &MeshMaterial3d<StandardMaterial>,
    )>,
) {
    let dt = time.delta_secs();
    for (entity, mut tf, mut puff, mat) in &mut puffs {
        puff.age += dt;
        if puff.age >= puff.lifetime {
            commands.entity(entity).despawn();
            continue;
        }
        let t = puff.age / puff.lifetime;
        tf.translation += puff.vel * dt;
        tf.scale = Vec3::splat(0.8 + t * 3.0); // grow as it rises
        if let Some(m) = materials.get_mut(&mat.0) {
            m.base_color = Color::srgba(0.9, 0.92, 0.95, (1.0 - t) * 0.5); // fade out
        }
    }
}

// --- Water-entry splash (WI 699, temporary) ---
//
// A one-shot spray burst thrown up when the craft pierces the ocean surface at speed — the
// *kinetic* impact (distinct from the WI 695/698 thermal steam plume), visualising the WI 700
// water-entry slam. Reuses the steam pooling shape; explicitly a placeholder for the general
// VFX pass.

/// Entry speed (m/s) below which a splashdown throws up no spray (a gentle touch).
const SPLASH_ONSET: f64 = 20.0;
/// Entry speed (m/s) at which the splash is at full intensity.
const SPLASH_FULL: f64 = 200.0;
/// Droplets in a full-intensity burst (scaled down by intensity).
const SPLASH_DROPLETS: usize = 60;
/// Maximum concurrent splash droplets (a bounded pool).
const SPLASH_MAX: usize = 140;
/// Droplet lifetime, seconds.
const SPLASH_LIFETIME: f32 = 1.2;
/// Downward acceleration applied to droplets (exaggerated for a snappy arc), m/s².
const SPLASH_GRAVITY: f32 = 22.0;

/// Splash intensity, 0–1 (pure, unit-tested): ramps from zero at/below [`SPLASH_ONSET`] to
/// full at [`SPLASH_FULL`], by the craft's entry speed — so a fast splashdown throws up a
/// big crown and a gentle touch throws up little or none. A VFX-tuning shape for the
/// deferred general pass.
fn splash_intensity(impact_speed: f64) -> f64 {
    ((impact_speed - SPLASH_ONSET) / (SPLASH_FULL - SPLASH_ONSET)).clamp(0.0, 1.0)
}

/// A ballistic, fading splash droplet.
#[derive(Component)]
struct SplashPuff {
    age: f32,
    lifetime: f32,
    vel: Vec3,
}

/// The shared splash droplet mesh (one small sphere, instanced per droplet).
#[derive(Resource)]
struct SplashAssets {
    mesh: Handle<Mesh>,
}

impl FromWorld for SplashAssets {
    fn from_world(world: &mut World) -> Self {
        let mut meshes = world.resource_mut::<Assets<Mesh>>();
        Self {
            mesh: meshes.add(Mesh::from(Sphere { radius: 0.5 })),
        }
    }
}

/// Emits a one-shot crown of droplets when the craft transitions **into** the ocean (the
/// not-liquid → liquid medium crossing), scaled by the entry speed (WI 699). Fires once per
/// entry — distinct from the continuous steam.
#[allow(clippy::too_many_arguments)]
fn emit_splash(
    mut commands: Commands,
    readout: Res<DiveReadout>,
    assets: Res<SplashAssets>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    craft: Query<&Transform, With<CraftMarker>>,
    existing: Query<(), With<SplashPuff>>,
    mut prev_medium: Local<Option<MediumKind>>,
    mut seed: Local<u32>,
) {
    let now = readout.medium;
    let entering = *prev_medium != Some(MediumKind::Liquid) && now == MediumKind::Liquid;
    *prev_medium = Some(now);
    if !entering {
        return;
    }
    let intensity = splash_intensity(readout.speed) as f32;
    if intensity <= 0.0 {
        return; // a gentle touch makes no splash
    }
    let Ok(craft_tf) = craft.single() else {
        return;
    };
    // The contact point: the craft's horizontal position at the sea surface (render Y = 0).
    let contact = Vec3::new(craft_tf.translation.x, 0.0, craft_tf.translation.z);

    let want = (SPLASH_DROPLETS as f32 * intensity) as usize;
    let room = SPLASH_MAX.saturating_sub(existing.iter().count());
    let count = want.min(room);
    for _ in 0..count {
        // A cheap LCG for scatter (no rng dependency; gameplay-noncritical), as for steam.
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let r = |shift: u32| (((*seed >> shift) & 0xff) as f32 / 255.0) - 0.5;
        // A crown/ring: outward-and-up, the spread growing with intensity.
        let dir = Vec3::new(r(0), 0.0, r(8)).normalize_or_zero();
        let out = 6.0 + intensity * 22.0;
        let upv = 10.0 + intensity * 16.0 + r(16) * 6.0;
        let vel = dir * out + Vec3::new(0.0, upv, 0.0);
        let offset = dir * (0.5 + r(24).abs() * 1.5);
        let mat = materials.add(StandardMaterial {
            base_color: Color::srgba(0.85, 0.9, 0.97, 0.7),
            emissive: LinearRgba::rgb(0.2, 0.22, 0.26),
            alpha_mode: AlphaMode::Blend,
            ..default()
        });
        commands.spawn((
            Mesh3d(assets.mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_translation(contact + offset).with_scale(Vec3::splat(0.6)),
            SplashPuff {
                age: 0.0,
                lifetime: SPLASH_LIFETIME,
                vel,
            },
        ));
    }
}

/// Advances each splash droplet on a ballistic arc (rise then fall under [`SPLASH_GRAVITY`])
/// and fades it, despawning at end of life (WI 699).
fn update_splash(
    mut commands: Commands,
    time: Res<Time>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut puffs: Query<(
        Entity,
        &mut Transform,
        &mut SplashPuff,
        &MeshMaterial3d<StandardMaterial>,
    )>,
) {
    let dt = time.delta_secs();
    for (entity, mut tf, mut puff, mat) in &mut puffs {
        puff.age += dt;
        if puff.age >= puff.lifetime {
            commands.entity(entity).despawn();
            continue;
        }
        puff.vel.y -= SPLASH_GRAVITY * dt; // ballistic fall
        tf.translation += puff.vel * dt;
        let t = puff.age / puff.lifetime;
        tf.scale = Vec3::splat(0.6 * (1.0 - 0.4 * t)); // shrink slightly as it fades
        if let Some(m) = materials.get_mut(&mat.0) {
            m.base_color = Color::srgba(0.85, 0.9, 0.97, (1.0 - t) * 0.7); // fade out
        }
    }
}

// --- Water surface (WI 703) ---

/// Half-extent of the animated water patch, metres (the patch spans 2× this per side).
const WATER_HALF: f32 = 160.0;
/// Plane subdivisions per side — the wave grid resolution.
const WATER_SUBDIV: u32 = 64;
/// Peak wave amplitude, metres: the surface oscillates within ±this.
const WATER_AMPLITUDE: f32 = 0.55;

/// The animated near-surface water patch (WI 703).
#[derive(Component)]
struct WaterPatch;

/// Height of the water surface at local patch coordinate `(x, z)` and time `t` (WI 703) — a
/// small sum of travelling sine waves, **bounded** by [`WATER_AMPLITUDE`] (the component
/// weights sum to 1). Pure (unit-tested); computed in the patch's local frame so the surface
/// ripples in place rather than scrolling as the camera moves.
fn wave_height(x: f32, z: f32, t: f32) -> f32 {
    let w1 = (x * 0.08 + t * 1.1).sin();
    let w2 = (z * 0.11 - t * 0.9).sin();
    let w3 = ((x + z) * 0.05 + t * 0.7).sin();
    WATER_AMPLITUDE * (0.45 * w1 + 0.35 * w2 + 0.20 * w3)
}

/// Keeps the water patch under the view (follows the camera's X/Z at sea level) and ripples
/// its surface each frame by [`wave_height`], recomputing normals (WI 703).
#[allow(clippy::type_complexity)] // disjoint Bevy queries (camera vs. patch)
fn animate_water(
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    camera: Query<&WorldPlacement, (With<AnchorCamera>, Without<WaterPatch>)>,
    mut patch: Query<(&Mesh3d, &mut WorldPlacement), (With<WaterPatch>, Without<AnchorCamera>)>,
) {
    let Ok(cam_wp) = camera.single() else {
        return;
    };
    let Ok((mesh3d, mut wp)) = patch.single_mut() else {
        return;
    };
    // Follow the camera horizontally at sea level (render Y = 0).
    let c = cam_wp.0.pos;
    wp.0 = WorldPos::new(FrameId::CENTRAL_BODY, DVec3::new(c.x, 0.0, c.z));
    // Ripple the surface in the patch's local frame.
    let t = time.elapsed_secs();
    let Some(mesh) = meshes.get_mut(&mesh3d.0) else {
        return;
    };
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        for p in positions.iter_mut() {
            p[1] = wave_height(p[0], p[2], t);
        }
    }
    mesh.compute_normals();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steam_intensity_tracks_vaporised_mass() {
        // No vaporised mass → no steam (the sim produces mass only when submerged + boiling,
        // WI 698, so the in-water/hot gating now lives in the sim, not here).
        assert_eq!(steam_intensity(0.0), 0.0);
        // Positive mass ramps with how vigorously the surface boils, and clamps to 1.
        assert!(steam_intensity(STEAM_MASS_FULL * 0.25) > 0.0);
        assert!(steam_intensity(STEAM_MASS_FULL * 0.5) > steam_intensity(STEAM_MASS_FULL * 0.25));
        assert!(steam_intensity(STEAM_MASS_FULL * 10.0) <= 1.0);
    }

    #[test]
    fn splash_intensity_is_speed_gated() {
        // A gentle touch (at/below the onset) makes no splash.
        assert_eq!(splash_intensity(0.0), 0.0);
        assert_eq!(splash_intensity(SPLASH_ONSET), 0.0);
        // Above the onset it ramps, monotonically, and clamps to 1.
        assert!(splash_intensity(SPLASH_FULL * 0.4) > 0.0);
        assert!(splash_intensity(SPLASH_FULL * 0.6) > splash_intensity(SPLASH_FULL * 0.3));
        assert!(splash_intensity(SPLASH_FULL * 10.0) <= 1.0);
        // Full intensity reached by the full-speed mark.
        assert!((splash_intensity(SPLASH_FULL) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn orbit_offset_default_reproduces_the_legacy_view() {
        let cam = DiveCam::default();
        let off = orbit_offset(cam.yaw, cam.pitch, cam.dist);
        // The default reproduces the historical fixed (18, 8, 18) offset.
        assert!((off.x - 18.0).abs() < 0.1, "x = {}", off.x);
        assert!((off.y - 8.0).abs() < 0.1, "y = {}", off.y);
        assert!((off.z - 18.0).abs() < 0.1, "z = {}", off.z);
        // Magnitude equals the orbit distance.
        assert!((off.length() - cam.dist).abs() < 1e-3);

        // Pitching up raises the eye; pitching down lowers it.
        let up = orbit_offset(cam.yaw, cam.pitch + 0.3, cam.dist);
        assert!(up.y > off.y, "more pitch raises the eye");
        // Zooming changes the distance, not the direction.
        let near = orbit_offset(cam.yaw, cam.pitch, cam.dist * 0.5);
        assert!((near.length() - cam.dist * 0.5).abs() < 1e-3);
        assert!(
            off.normalize().distance(near.normalize()) < 1e-5,
            "zoom keeps the view direction"
        );
        // Orbiting yaw rotates the eye around the target (the x/z direction changes).
        let spun = orbit_offset(cam.yaw + 1.0, cam.pitch, cam.dist);
        assert!(
            (spun.x - off.x).abs() > 1.0 || (spun.z - off.z).abs() > 1.0,
            "yaw orbits horizontally"
        );
    }

    #[test]
    fn wave_height_is_bounded_and_animates() {
        // Bounded by the amplitude over a grid of positions and times, and deterministic.
        let mut moved = false;
        for &x in &[-160.0, -40.0, 0.0, 37.0, 160.0_f32] {
            for &z in &[-160.0, -12.0, 0.0, 88.0, 160.0_f32] {
                for &t in &[0.0, 0.5, 1.7, 3.3, 10.0_f32] {
                    let h = wave_height(x, z, t);
                    assert!(
                        h.abs() <= WATER_AMPLITUDE + 1e-5,
                        "wave exceeded amplitude: {h} at ({x},{z},{t})"
                    );
                    assert_eq!(h, wave_height(x, z, t), "deterministic");
                }
                // The surface actually animates: at least one time differs from t=0.
                if (wave_height(x, z, 0.0) - wave_height(x, z, 1.3)).abs() > 1e-4 {
                    moved = true;
                }
            }
        }
        assert!(moved, "the surface must animate over time");
    }
}
