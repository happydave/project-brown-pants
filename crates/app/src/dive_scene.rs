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
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::{DVec2, DVec3};
use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use sounding_sim::active::{ActiveBody, Gravity};
use sounding_sim::fluid::{FluidMedium, MediumKind};
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::handoff::{orbit_state_3d, GearState, HandoffPlugin};
use sounding_sim::medium::{
    dynamic_pressure, max_cross_section, CraftThermal, DescentParams, DescentPlugin,
    DiveTriggerPlugin, DivingCraft, EntryInterface, GlideParams, DIVE_HEAT_SCALE,
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

/// A slender re-entry body along +Z (forward): a 3×3×4 composite hull with a
/// denser, centred nose tip — a positive static margin so it weathervanes into the
/// airflow, and a tapered area curve so it shows transonic wave drag (WI 526).
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
        material: Material::ALUMINIUM,
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
            .add_systems(Startup, setup_scene)
            .add_systems(
                Update,
                (track_craft, track_thermal, follow_camera, update_hud).chain(),
            );
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

    // Ocean: a translucent blue sphere whose surface is sea level (Y = 0).
    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(Sphere {
            radius: BODY.radius as f32,
        }))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.05, 0.20, 0.40, 0.55),
            alpha_mode: AlphaMode::Blend,
            perceptual_roughness: 0.1,
            ..default()
        })),
        Transform::default(),
        WorldPlacement(WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(0.0, -BODY.radius, 0.0),
        )),
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
    let (max_skin, over_limit) = thermal.readout(&dc.craft);
    readout.skin_temp = max_skin;
    readout.over_limit = over_limit;
    bridge.0 = Some(ThermalTelemetry {
        max_skin_temp: max_skin,
        over_limit,
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

/// Keeps the anchor camera a fixed offset from the craft's render position.
#[allow(clippy::type_complexity)] // disjoint Bevy queries (craft vs. camera)
fn follow_camera(
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
    let eye = target + DVec3::new(18.0, 8.0, 18.0);
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
        text.0 = format!(
            "gear:     {gear}\naltitude: {alt:8.0} m\nspeed:    {speed:7.1} m/s\nmedium:   {medium}\nhull P:   {pressure_kpa:8.1} kPa\nram P:    {ram_kpa:8.1} kPa\nskin T:   {skin:8.0} K{heat}"
        );
    }
}
