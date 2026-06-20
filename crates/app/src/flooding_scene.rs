//! Toy: decompression / flooding (WI 520). A submerged compartmented craft.
//! Breach a compartment (`B`) and watch floodwater fill it — the added mass
//! shifts the centre of mass, so the craft tilts toward the flooded side and
//! sinks (heavier than the water it displaces). Headless flooding lives in
//! `sounding_sim::flooding`; this scene drives and draws it.
//!
//! Controls: `B` breach a compartment · `R` reset.

use bevy::math::{DVec3, Isometry3d};
use bevy::prelude::*;
use sounding_sim::active::ActiveBody;
use sounding_sim::compartments::compartments;
use sounding_sim::flooding::{flooded_mass_properties, FloodCompartment};
use sounding_sim::fluid::FluidMedium;
use sounding_sim::voxel::{Material, Voxel, VoxelCraft};

const WATER_RHO: f64 = 1_025.0;
const G: f64 = 9.81;
const ATM: f64 = 101_325.0;
const CRUSH: f64 = 5.0e6;
const START_DEPTH: f64 = 80.0;
const SURFACE_R: f64 = 6_360_000.0;
const SUBSTEP_DT: f64 = 0.01;
const MAX_SUBSTEPS: u32 = 40;
/// Water rotational drag — damps the righting oscillation to rest (N·m·s).
const ANG_DAMP: f64 = 5.0e6;
/// Water linear drag — gives the sink a bounded (terminal) rate (N·s/m).
const LIN_DAMP: f64 = 1.0e5;

/// The submerged craft, its per-compartment flood state, and the rigid body.
#[derive(Resource)]
struct FloodWorld {
    craft: VoxelCraft,
    floods: Vec<FloodCompartment>,
    body: ActiveBody,
    dry_mass: f64,
    accumulator: f64,
}

impl FloodWorld {
    fn new() -> Self {
        let craft = two_room_craft();
        let mp = craft.mass_properties().expect("non-empty");
        let set = compartments(&craft);
        let floods: Vec<FloodCompartment> = set
            .compartments
            .iter()
            .map(|c| FloodCompartment::from_compartment(c, craft.cell_size, ATM, CRUSH))
            .collect();
        // The craft starts neutrally buoyant at depth (so dry it hovers; flooding
        // is what sinks it). Place its CoM at world Y = surface − depth.
        let body = ActiveBody::new(
            DVec3::new(0.0, SURFACE_R - START_DEPTH, 0.0),
            DVec3::ZERO,
            mp.mass,
            mp.inertia,
        );
        Self {
            craft,
            floods,
            body,
            dry_mass: mp.mass,
            accumulator: 0.0,
        }
    }

    fn depth(&self) -> f64 {
        SURFACE_R - self.body.position.y
    }
}

/// A 7×5×5 shell with an internal wall at x=3 → two sealable compartments.
fn two_room_craft() -> VoxelCraft {
    let mut c = VoxelCraft::new(1.0);
    let (nx, ny, nz) = (7, 5, 5);
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let shell = x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1;
                if shell || x == 3 {
                    c.voxels.push(Voxel {
                        cell: IVec3::new(x, y, z),
                        material: Material::COMPOSITE,
                    });
                }
            }
        }
    }
    c
}

/// Marks the heads-up readout.
#[derive(Component)]
struct Hud;

/// The Toy flooding scene.
pub struct FloodingScenePlugin;

impl Plugin for FloodingScenePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(FloodWorld::new())
            .add_systems(Startup, setup_view)
            .add_systems(Update, (handle_input, step_flood, draw, update_hud).chain());
    }
}

fn setup_view(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(3.0, 6.0, 18.0).looking_at(Vec3::new(3.0, 2.0, 2.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 5_000.0,
            ..default()
        },
        Transform::from_xyz(6.0, 14.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Text::new("B breach a compartment · R reset"),
        TextFont {
            font_size: 20.0,
            ..default()
        },
        TextColor(Color::srgb(0.85, 0.92, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(12.0),
            ..default()
        },
        Hud,
    ));
}

fn handle_input(keys: Res<ButtonInput<KeyCode>>, mut world: ResMut<FloodWorld>) {
    if keys.just_pressed(KeyCode::KeyB) {
        // Breach the first still-sealed compartment.
        if let Some(f) = world.floods.iter_mut().find(|f| !f.breached) {
            f.breached = true;
        }
    }
    if keys.just_pressed(KeyCode::KeyR) {
        *world = FloodWorld::new();
    }
}

fn step_flood(time: Res<Time>, mut world: ResMut<FloodWorld>) {
    world.accumulator += time.delta_secs_f64();
    let mut n = 0;
    while world.accumulator >= SUBSTEP_DT && n < MAX_SUBSTEPS {
        let depth = world.depth();
        let sample = FluidMedium::EARTHLIKE.sample_altitude(-depth);
        for f in &mut world.floods {
            f.step(&sample, 0.6, SUBSTEP_DT);
        }
        // Flooded mass/CoM feed back into the rigid body in real time.
        let fm = flooded_mass_properties(&world.craft, &world.floods, WATER_RHO);
        world.body.mass = fm.mass;
        // Buoyancy calibrated to neutral when dry; floodwater is net ballast.
        let buoyancy = world.dry_mass * G;
        let weight = fm.mass * G;
        let mp = world.craft.mass_properties().unwrap();
        // Lever from the centre of buoyancy (≈ the dry geometric centre) to the
        // shifted CoM, **rotated into the world frame** so it tracks the craft's
        // attitude. Buoyancy acts up at the CoB and weight down at the CoM: their
        // couple is a *restoring* torque (it vanishes when the heavy flooded side
        // is straight down), not a constant spin. Water drag damps it to rest.
        let lever = world.body.orientation * (fm.center_of_mass - mp.center_of_mass);
        let righting = (-lever).cross(DVec3::new(0.0, buoyancy, 0.0));
        let torque = righting - ANG_DAMP * world.body.angular_velocity();
        let force = DVec3::new(0.0, buoyancy - weight, 0.0) - LIN_DAMP * world.body.velocity;
        world.body.integrate_wrench(force, torque, SUBSTEP_DT);
        world.accumulator -= SUBSTEP_DT;
        n += 1;
    }
}

fn draw(mut gizmos: Gizmos, world: Res<FloodWorld>) {
    let mp = world.craft.mass_properties().unwrap();
    let com = mp.center_of_mass;
    let q = world.body.orientation;
    let size = world.craft.cell_size as f32;
    // Render rover-style: anchored at the body, oriented by it.
    let place = |cell: IVec3| {
        let local = (cell.as_dvec3() + DVec3::splat(0.5)) * world.craft.cell_size - com;
        (q * local).as_vec3()
    };
    let cube = |gizmos: &mut Gizmos, cell: IVec3, s: f32, color: Color| {
        gizmos.primitive_3d(
            &Cuboid::new(s, s, s),
            Isometry3d::new(place(cell), q.as_quat()),
            color,
        );
    };

    // Hull structure.
    for v in &world.craft.voxels {
        cube(&mut gizmos, v.cell, size, Color::srgb(0.40, 0.42, 0.48));
    }
    // Floodwater: fill each compartment's cells up to its flooded fraction (the
    // lowest cells first, so it reads as a rising water level).
    let set = compartments(&world.craft);
    for (comp, flood) in set.compartments.iter().zip(&world.floods) {
        let mut cells = comp.cells.clone();
        cells.sort_by_key(|c| c.y);
        let filled = (flood.flooded_fraction() * cells.len() as f64).round() as usize;
        for &cell in cells.iter().take(filled) {
            cube(
                &mut gizmos,
                cell,
                size * 0.85,
                Color::srgb(0.15, 0.45, 0.85),
            );
        }
    }
}

fn update_hud(world: Res<FloodWorld>, mut hud: Query<&mut Text, With<Hud>>) {
    if let Ok(mut text) = hud.single_mut() {
        let frac: f64 = world
            .floods
            .iter()
            .map(|f| f.flooded_fraction())
            .sum::<f64>()
            / world.floods.len().max(1) as f64;
        let fm = flooded_mass_properties(&world.craft, &world.floods, WATER_RHO);
        let imploded = world.floods.iter().any(|f| f.imploded);
        text.0 = format!(
            "depth: {:6.1} m   flooded: {:4.0}%   mass: {:8.0} kg{}\nB breach a compartment · R reset",
            world.depth(),
            frac * 100.0,
            fm.mass,
            if imploded { "   [IMPLODED]" } else { "" }
        );
    }
}
