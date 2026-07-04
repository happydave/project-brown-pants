//! Afloat-vessel assembly + interior flooding (WI 739, moved from the harbor
//! scene so the scenario director can spawn a floating craft from data).
//!
//! A hull lattice assembles at its **real material mass** (WI 717 — no
//! auto-ballast: a light panel hull floats, a heavy/solid one sinks) into the
//! same [`ActiveBody`] + [`DivingCraft`] chain the shared
//! [`crate::medium::DescentPlugin`] steps. Until the device palette (WI 715)
//! lets players place them, every vessel gets a **synthesized** stern drive
//! (WI 708), rudder (WI 725), and ballast tank (WI 709) sized from the hull.
//!
//! Interior flooding (WI 718/520) lives here too: the per-sealed-compartment
//! flood models whose still-dry cells rebuild the hull's enclosed-buoyancy
//! set each step — **buoyancy falls as it floods**. The render side (dry-hold
//! occluders, rising water cuboids) stays presentation; this module owns only
//! the physics.
//!
//! **SCAFFOLD: much of this module is scenario scaffolding, not engine.** The
//! `synth_*` constructors invent devices a player should place and a catalog
//! should describe; the assembly hardcodes world/medium conventions the
//! scenario document should supply. Each site below carries a `SCAFFOLD:`
//! marker naming its replacement route; the inventory lives at
//! `docs/projects/sounding/scaffolding.md`. The flood physics (`FloodComps`,
//! `step_flooding`, `unflooded_cells`) is real engine and stays.

use crate::active::ActiveBody;
use crate::ballast::{Ballast, BallastCommand, BallastTank};
use crate::compartments::compartments;
use crate::flooding::FloodCompartment;
use crate::marine::{MarinePropulsion, MarineThruster, Rudder, ThrusterCommand};
use crate::medium::{
    buoyancy_wrench, enclosed_cells, max_cross_section, DescentParams, DivingCraft, GlideParams,
    DEFAULT_SLAM_COEFFICIENT,
};
use crate::resource::{Reservoir, ReservoirId, ResourceGraph, ResourceType};
use crate::voxel::{Axis, VoxelCraft};
use bevy_ecs::prelude::*;
use glam::{DQuat, DVec3, IVec3};

/// Flood relaxation rate (1/s) — a several-second flood once breached (the
/// WI 520 `step` constant).
pub const FLOOD_RATE: f64 = 0.15;

/// Marks a director-spawned floating vessel (WI 739): the scene finds it to
/// attach presentation (render children, camera target, teardown markers).
#[derive(Component)]
pub struct ScenarioVessel;

/// Builds the descent/glide parameters and the floating body for a hull
/// lattice using its **real material mass**: a light panel hull floats, a
/// heavy / solid one sinks. Spawns at sea level over the world origin with a
/// small starting list so the self-righting reads. `None` for an empty
/// lattice.
///
/// SCAFFOLD: the medium is hardcoded `EARTHLIKE` (should come from the spawn
/// payload's world), the spawn point is the world origin (the placement/world
/// data should locate the harbor), the +Z forward axis is a convention the
/// blueprint should declare, and the 0.2 rad presentation list belongs to the
/// scene. Assembly-from-lattice itself is engine; these inputs are not.
pub fn assemble_float(
    craft: &VoxelCraft,
    mu: f64,
    surface_radius: f64,
) -> Option<(ActiveBody, DivingCraft)> {
    let mp = craft.mass_properties()?;
    if mp.mass <= 0.0 {
        return None;
    }
    let descent = DescentParams {
        medium: crate::fluid::FluidMedium::EARTHLIKE,
        mu,
        surface_radius,
        drag_area: max_cross_section(craft),
        drag_coefficient: 1.0,
        slam_coefficient: DEFAULT_SLAM_COEFFICIENT,
    };
    let glide = GlideParams::for_craft(descent, craft, Axis::Z);
    let start_sim = DVec3::new(0.0, surface_radius, 0.0);
    let mut body = ActiveBody::new(start_sim, DVec3::ZERO, mp.mass, mp.inertia);
    body.orientation = DQuat::from_rotation_z(0.2); // a starting list, so the self-righting reads
    Some((
        body,
        DivingCraft::new(craft.clone(), mp.center_of_mass, glide),
    ))
}

/// The hull lattice's metre-space AABB `(min, max)`.
fn hull_aabb(craft: &VoxelCraft) -> (DVec3, DVec3) {
    let cs = craft.cell_size;
    let (mut lo, mut hi) = (IVec3::MAX, IVec3::MIN);
    for v in &craft.voxels {
        lo = lo.min(v.cell);
        hi = hi.max(v.cell);
    }
    (lo.as_dvec3() * cs, (hi.as_dvec3() + DVec3::ONE) * cs)
}

/// A synthesized marine drive for a built hull (WI 708): a port + starboard
/// screw pair low at the stern (the −Z face; forward is +Z, the glide forward
/// axis), mounted near the keel so they sit in the water. Differential
/// throttle steers (a yaw couple from the ±X offset). Sized to push the
/// editor-scale starter boat; player-placed thrusters await the WI 715
/// device palette.
///
/// SCAFFOLD: a whole invented device — thrust/draw/fuel (a hardcoded 1500 kg
/// reservoir) and mounts should come from player-placed thruster devices
/// (WI 715) backed by catalog records, assembled like engines/tanks are.
pub fn synth_marine(craft: &VoxelCraft) -> MarinePropulsion {
    let cs = craft.cell_size;
    let (min_m, max_m) = hull_aabb(craft);
    let centre_x = 0.5 * (min_m.x + max_m.x);
    let beam = (max_m.x - min_m.x).max(cs);
    let stern_z = min_m.z + 0.5 * cs; // just inside the stern (the −Z end)
    let keel_y = min_m.y + 0.5 * cs; // near the bottom row, so the screws are submerged
    let off = 0.30 * beam;
    let screw = |x: f64| MarineThruster {
        tank: ReservoirId(0),
        max_thrust: 5_000.0,
        reference_density: 1_025.0, // water surface — full thrust submerged, ~none in air
        max_draw: 4.0,
        mount: DVec3::new(x, keel_y, stern_z),
        axis: DVec3::Z, // push forward (+Z)
    };
    MarinePropulsion {
        graph: ResourceGraph {
            reservoirs: vec![Reservoir::new(ResourceType(0), 1_500.0, 1_500.0)],
            ..Default::default()
        },
        thrusters: vec![screw(centre_x + off), screw(centre_x - off)],
        commands: vec![ThrusterCommand::default(); 2],
        last_thrust: 0.0,
    }
}

/// A synthesized rudder for a built hull (WI 725): a control surface aft at
/// the stern (−Z, forward is +Z), low so it sits in the water. Area scales
/// with the hull's beam×draft so bigger boats get more steering authority.
/// Player-placed rudders await the WI 715 device palette.
///
/// SCAFFOLD: an invented device — area/slope/limits and the mount should come
/// from a player-placed rudder device (WI 715) backed by a catalog record.
pub fn synth_rudder(craft: &VoxelCraft) -> Rudder {
    let cs = craft.cell_size;
    let (min_m, max_m) = hull_aabb(craft);
    let beam = (max_m.x - min_m.x).max(cs);
    let depth = (max_m.y - min_m.y).max(cs);
    Rudder {
        mount: DVec3::new(
            0.5 * (min_m.x + max_m.x),
            min_m.y + 0.5 * cs,  // low, in the water
            min_m.z - 0.25 * cs, // just aft of the stern
        ),
        forward: DVec3::Z,
        area: 0.25 * beam * depth, // a fraction of the stern cross-section
        slope: 6.0,                // ~2π lift-curve slope per radian
        max_angle: 0.6,            // ~34° hard over
        angle: 0.0,
    }
}

/// A synthesized ballast tank for a built hull (WI 709): one tank low at the
/// keel, sized so a full flood **clearly overcomes the hull's reserve
/// buoyancy** (≈1.5× the reserve), so flooding it sinks a boat that floats
/// empty and blowing it surfaces — controllable dive/surface/hold. `None` ⇒
/// no ballast (an empty/barely-floating hull). Player-placed ballast awaits
/// the WI 715 device palette.
///
/// SCAFFOLD: an invented device — capacity/rates and the mount should come
/// from player-placed ballast-tank devices (WI 715) backed by catalog
/// records (the auto-sizing-to-reserve heuristic disappears with them).
pub fn synth_ballast(craft: &VoxelCraft, surface_radius: f64) -> Option<Ballast> {
    let mp = craft.mass_properties()?;
    let g = 9.81;
    // The hull's reserve buoyancy (fully-submerged displaced weight − real weight), as a mass.
    let deep = DVec3::new(0.0, surface_radius - 100.0, 0.0);
    let max_buoy = buoyancy_wrench(
        craft,
        mp.center_of_mass,
        deep,
        DQuat::IDENTITY,
        surface_radius,
        0.0,
        1_025.0,
        g,
        &enclosed_cells(craft),
    )
    .force
    .length();
    let reserve_mass = ((max_buoy - mp.mass * g) / g).max(0.0);
    if reserve_mass <= 0.0 {
        return None; // already sinks; ballast is meaningless
    }
    let cap_volume = 1.5 * reserve_mass / 1_025.0; // m³ of water to clearly overcome the reserve
    let cs = craft.cell_size;
    let (min_m, max_m) = hull_aabb(craft);
    let mount = DVec3::new(
        0.5 * (min_m.x + max_m.x),
        min_m.y + 0.5 * cs, // low in the hull, so a flooded tank sinks the bow evenly and trims down
        0.5 * (min_m.z + max_m.z),
    );
    let rate = cap_volume / 6.0; // ~6 s to fully flood or blow
    Some(Ballast {
        tanks: vec![BallastTank {
            capacity: cap_volume,
            mount,
            fill: 0.0,
            fill_rate: rate,
            blow_rate: rate,
        }],
        command: BallastCommand::Hold,
        dry_mass: mp.mass,
    })
}

/// One sealed compartment's flood physics (WI 718/520): its empty cells,
/// sorted by height ascending (water fills bottom-up), and the transient
/// flood model that drives buoyancy loss.
pub struct FloodComp {
    /// The compartment's empty cells, height-ascending.
    pub cells: Vec<IVec3>,
    /// The WI 520 transient flood model (volume, centroid, breach/flood state).
    pub flood: FloodCompartment,
}

/// A vessel's interior-flooding physics (WI 718, moved sim-side by WI 739):
/// the per-sealed-compartment flood models. The director's flooding system
/// steps these and rebuilds the hull's enclosed-buoyancy set from the
/// still-dry cells — the render manifest over them stays presentation.
#[derive(Component, Default)]
pub struct FloodComps {
    /// One flood model per sealed compartment.
    pub comps: Vec<FloodComp>,
    /// Whether the hull has been breached (opened to the sea). One-way.
    pub breached: bool,
}

impl FloodComps {
    /// The flood models for every sealed compartment of `craft`, dry.
    pub fn for_craft(craft: &VoxelCraft) -> FloodComps {
        let atm = 101_325.0;
        let crush = 5.0e6; // ~500 m of water (well beyond the harbor)
        let comps = compartments(craft)
            .compartments
            .iter()
            .map(|c| {
                let mut cells = c.cells.clone();
                cells.sort_by_key(|p| p.y);
                FloodComp {
                    cells,
                    flood: FloodCompartment::from_compartment(c, craft.cell_size, atm, crush),
                }
            })
            .collect();
        FloodComps {
            comps,
            breached: false,
        }
    }

    /// Breach the hull: every sealed compartment opens to the sea. One-way.
    pub fn breach(&mut self) {
        self.breached = true;
        for c in &mut self.comps {
            c.flood.breached = true;
        }
    }

    /// Overall flooded fraction across the sealed compartments, `[0, 1]`.
    pub fn flooded_fraction(&self) -> f64 {
        let (vol, water) = self.comps.iter().fold((0.0, 0.0), |(v, w), c| {
            (v + c.flood.volume, w + c.flood.floodwater)
        });
        if vol > 0.0 {
            (water / vol).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// The cells of a compartment that still hold **air** at a given flooded
/// fraction (WI 718): water fills **bottom-up**, so the lowest `fraction · n`
/// cells have flooded and the upper ones remain. `cells` must be sorted by
/// height ascending. The hull's enclosed-buoyancy set is rebuilt from these,
/// so buoyancy falls as it floods. Pure (the testable core of the feedback).
pub fn unflooded_cells(cells: &[IVec3], flooded_fraction: f64) -> Vec<IVec3> {
    let n = cells.len();
    let flooded = (flooded_fraction.clamp(0.0, 1.0) * n as f64).round() as usize;
    cells.iter().skip(flooded.min(n)).copied().collect()
}

/// Advances a vessel's interior flooding by `dt` (WI 718, the physics half of
/// the old harbor flood step): steps each compartment against the water at
/// its keel (the breach sits at/below the keel, so a breached hull always
/// takes on water, and the inflow grows as it sinks deeper), then rebuilds
/// the hull's enclosed-buoyancy set from the still-dry cells so the shared
/// descent step feels the lost buoyancy.
pub fn step_flooding(comps: &mut FloodComps, body: &ActiveBody, dc: &mut DivingCraft, dt: f64) {
    if comps.comps.is_empty() {
        return;
    }
    let surface_radius = dc.glide.descent.surface_radius;
    let medium = dc.glide.descent.medium;
    let mut enclosed: Vec<IVec3> = Vec::new();
    for comp in &mut comps.comps {
        let centroid_world = body.position + body.orientation * (comp.flood.centroid - dc.com);
        let alt = (centroid_world.length() - surface_radius).min(-0.1);
        let sample = medium.sample_altitude(alt);
        comp.flood.step(&sample, FLOOD_RATE, dt);
        enclosed.extend(unflooded_cells(&comp.cells, comp.flood.flooded_fraction()));
    }
    dc.enclosed = enclosed;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::CentralBody;
    use crate::voxel::{Material, Voxel};

    const BODY: CentralBody = CentralBody::EARTHLIKE;

    /// The harbor seed hull (also the shipped `harbor-seed` blueprint): a
    /// sealed **panel** pontoon — light, so it floats honestly under real
    /// mass; the same hull in solid cubes would sink.
    pub(crate) fn seed_hull() -> VoxelCraft {
        let mut c = VoxelCraft::new(0.5);
        let (w, h, l) = (7, 5, 11);
        for x in 0..w {
            for y in 0..h {
                for z in 0..l {
                    let on_surface =
                        x == 0 || x == w - 1 || y == 0 || y == h - 1 || z == 0 || z == l - 1;
                    if on_surface {
                        let cell = glam::IVec3::new(x, y, z);
                        c.voxels.push(Voxel {
                            cell,
                            material: Material::ALUMINIUM,
                        });
                        c.set_panel(cell, true); // a thin hull plate, not a solid cube (WI 716)
                    }
                }
            }
        }
        c
    }

    /// Regenerates the shipped harbor seed-hull blueprint (WI 739):
    /// `cargo test -p sounding_sim --lib write_harbor_seed_blueprint -- --ignored`
    #[test]
    #[ignore = "writes the shipped content/blueprints/harbor-seed.json artifact"]
    fn write_harbor_seed_blueprint() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../content/blueprints");
        let path = crate::library::save_blueprint(&dir, "Harbor Seed", &seed_hull()).unwrap();
        assert!(path.ends_with("harbor-seed.json"));
    }

    /// Net buoyancy of a craft fully submerged (max buoyancy − real weight):
    /// >0 floats, <0 sinks.
    fn net_buoyancy(craft: &VoxelCraft) -> f64 {
        let mp = craft.mass_properties().unwrap();
        let g = 9.81;
        let deep = DVec3::new(0.0, BODY.radius - 100.0, 0.0);
        let buoy = buoyancy_wrench(
            craft,
            mp.center_of_mass,
            deep,
            DQuat::IDENTITY,
            BODY.radius,
            0.0,
            1_025.0,
            g,
            &enclosed_cells(craft),
        )
        .force
        .length();
        buoy - mp.mass * g
    }

    #[test]
    fn the_panel_seed_floats_but_a_solid_hull_sinks() {
        let seed = seed_hull();
        assert!(net_buoyancy(&seed) > 0.0, "the panel pontoon floats");
        let mut solid = seed_hull();
        for v in &solid.voxels.clone() {
            solid.set_panel(v.cell, false);
        }
        assert!(
            net_buoyancy(&solid) < 0.0,
            "the same hull in solid cubes sinks"
        );
    }

    #[test]
    fn assemble_float_uses_real_mass() {
        let seed = seed_hull();
        let (body, _) = assemble_float(&seed, BODY.mu, BODY.radius).unwrap();
        let mp = seed.mass_properties().unwrap();
        assert_eq!(body.mass, mp.mass, "real material mass — no auto-ballast");
    }

    #[test]
    fn an_empty_lattice_does_not_assemble() {
        assert!(assemble_float(&VoxelCraft::new(0.5), BODY.mu, BODY.radius).is_none());
    }

    #[test]
    fn synth_marine_drives_the_seed_hull_forward_and_steers() {
        let seed = seed_hull();
        let (body, dc) = assemble_float(&seed, BODY.mu, BODY.radius).unwrap();
        let mut mp = synth_marine(&seed);
        let fuel0 = mp.fuel();

        // Full ahead: forward thrust, fuel drawn.
        mp.drive(1.0, 0.0);
        let (f, tq) = mp.thrust_step(
            &crate::fluid::FluidMedium::EARTHLIKE,
            BODY.radius,
            body.position,
            body.orientation,
            dc.com,
            0.1,
        );
        assert!(
            f.length() > 0.0,
            "the keel screws are submerged and push: {f:?}"
        );
        let forward_world = body.orientation * DVec3::Z;
        assert!(f.dot(forward_world) > 0.0, "drives forward");
        assert!(mp.fuel() < fuel0, "fuel drawn under power");
        assert!(tq.length() >= 0.0);

        // Differential throttle yaws (nonzero steering moment about up).
        mp.drive(0.4, 0.6);
        let (_f2, tq2) = mp.thrust_step(
            &crate::fluid::FluidMedium::EARTHLIKE,
            BODY.radius,
            body.position,
            body.orientation,
            dc.com,
            0.1,
        );
        let up = body.position.normalize_or(DVec3::Y);
        assert!(
            tq2.dot(up).abs() > 1e-3,
            "differential thrust steers: {tq2:?}"
        );
    }

    #[test]
    fn synth_ballast_overcomes_the_reserve() {
        let seed = seed_hull();
        let b = synth_ballast(&seed, BODY.radius).expect("a floating hull gets ballast");
        let reserve = net_buoyancy(&seed) / 9.81; // kg of reserve buoyancy
        let cap_mass = b.tanks[0].capacity * 1_025.0;
        assert!(
            cap_mass > reserve,
            "a full tank ({cap_mass} kg) overcomes the reserve ({reserve} kg)"
        );
        // A solid (sinking) hull gets none.
        let mut solid = seed_hull();
        for v in &solid.voxels.clone() {
            solid.set_panel(v.cell, false);
        }
        assert!(synth_ballast(&solid, BODY.radius).is_none());
    }

    /// WI 725: the synthesized rudder sits in the water aft of the hull and
    /// steers it **only when moving** — a yaw moment under way, nothing at a
    /// standstill.
    #[test]
    fn synth_rudder_steers_the_moving_seed_hull_only_under_way() {
        let seed = seed_hull();
        let (body, dc) = assemble_float(&seed, BODY.mu, BODY.radius).unwrap();
        let mut r = synth_rudder(&seed);
        r.set_turn(1.0); // hard over
        let m = crate::fluid::FluidMedium::EARTHLIKE;

        // At rest: no flow ⇒ no steering.
        let (_, t_rest) = r.wrench(
            &m,
            BODY.radius,
            body.position,
            body.orientation,
            DVec3::ZERO,
            dc.com,
        );
        assert_eq!(t_rest, DVec3::ZERO, "no steering at a standstill");

        // Under way (forward, the hull's +Z): a real yaw moment about the local up.
        let forward = body.orientation * DVec3::Z;
        let (_, t_move) = r.wrench(
            &m,
            BODY.radius,
            body.position,
            body.orientation,
            forward * 5.0,
            dc.com,
        );
        let up = body.position.normalize_or(DVec3::Y);
        assert!(
            t_move.dot(up).abs() > 1e-3,
            "the rudder yaws the moving hull: {t_move:?}"
        );
    }

    /// WI 709: flooding the synthesized ballast flips the seed hull's net
    /// buoyancy negative (it dives), and blowing it recovers positive net
    /// buoyancy (it surfaces) — controllable and reversible.
    #[test]
    fn ballast_flips_net_buoyancy_dive_and_surface() {
        let seed = seed_hull();
        let mut b = synth_ballast(&seed, BODY.radius).expect("the panel hull has reserve buoyancy");
        let g = 9.81;
        let enc = enclosed_cells(&seed);
        let mp = seed.mass_properties().unwrap();
        let deep = DVec3::new(0.0, BODY.radius - 100.0, 0.0);
        let buoy = buoyancy_wrench(
            &seed,
            mp.center_of_mass,
            deep,
            DQuat::IDENTITY,
            BODY.radius,
            0.0,
            1_025.0,
            g,
            &enc,
        )
        .force
        .length();
        let net = |ballast: &Ballast| buoy - ballast.wet_mass(mp.center_of_mass, 1_025.0).mass * g;

        // Blown (empty): the panel hull floats.
        assert!(net(&b) > 0.0, "blown ballast ⇒ floats");
        // Flood it fully: it sinks.
        b.command = BallastCommand::Fill;
        b.step(100.0);
        assert!(net(&b) < 0.0, "flooded ballast ⇒ sinks (dives)");
        // Blow it again: it floats once more (reversible).
        b.command = BallastCommand::Blow;
        b.step(100.0);
        assert!(net(&b) > 0.0, "blown again ⇒ floats (reversible)");
    }

    #[test]
    fn flooding_removes_buoyancy_bottom_up_and_sinks() {
        let seed = seed_hull();
        let (body, mut dc) = assemble_float(&seed, BODY.mu, BODY.radius).unwrap();
        let mut comps = FloodComps::for_craft(&seed);
        assert!(
            !comps.comps.is_empty(),
            "the sealed pontoon has a compartment"
        );
        let dry = dc.enclosed.len();
        comps.breach();
        // Step for a while: floodwater accumulates and the enclosed set shrinks.
        for _ in 0..600 {
            step_flooding(&mut comps, &body, &mut dc, 0.05);
        }
        assert!(comps.flooded_fraction() > 0.5, "breached: it floods");
        assert!(
            dc.enclosed.len() < dry,
            "enclosed-buoyancy cells shrink as it floods ({} < {dry})",
            dc.enclosed.len()
        );
        // Bottom-up: the remaining dry cells are the upper ones.
        if let Some(comp) = comps.comps.first() {
            let dry_cells = unflooded_cells(&comp.cells, comp.flood.flooded_fraction());
            if let (Some(first_all), Some(first_dry)) = (comp.cells.first(), dry_cells.first()) {
                assert!(first_dry.y >= first_all.y, "water filled from the bottom");
            }
        }
    }
}
