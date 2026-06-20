//! Decompression / flooding / implosion: the pressure-differential transient
//! (WI 520).
//!
//! The design's marquee collapse: venting to vacuum (decompression), crushing at
//! depth (implosion), and flooding are **one sign-flipped transient** across a
//! compartment boundary. A breach opens a compartment (WI 519) to the ambient
//! medium (WI 497); flow is driven by `ΔP = P_ambient − P_internal`, and the same
//! step relaxes the compartment toward the ambient-determined equilibrium:
//!
//! - gas ambient → the air equalises toward `P_ambient` (decompression when the
//!   cabin is the higher-pressure side — e.g. 1 atm vented to vacuum);
//! - liquid ambient with `ΔP > 0` → water floods in toward full, the air venting;
//! - `ΔP` past the compartment's crush strength → catastrophic flooding (implosion).
//!
//! Floodwater carries mass: [`flooded_mass_properties`] folds it into the craft's
//! mass and centre of mass each tick — the one place a resource feeds back into
//! rigid-body dynamics in real time. Bounded to active physics (tick-stepped, not
//! analytic catch-up). Headless; the flooding scene lives in the app.

use crate::compartments::Compartment;
use crate::fluid::{FluidSample, MediumKind};
use crate::voxel::VoxelCraft;
use glam::DVec3;

/// Below this free volume / pressure scale, treat as a hard boundary (avoids
/// division by zero in the internal-pressure derivation).
const EPS: f64 = 1e-9;

/// The real-time flood/air state of one compartment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloodCompartment {
    /// Total compartment volume, m³.
    pub volume: f64,
    /// Centroid of the compartment (craft local frame), m — where floodwater mass
    /// is placed.
    pub centroid: DVec3,
    /// Floodwater present, m³ (in `[0, volume]`).
    pub floodwater: f64,
    /// Trapped-air pressure–volume invariant, Pa·m³ (`P_internal = air_pv / free`).
    pub air_pv: f64,
    /// Whether a breach connects this compartment to the ambient medium.
    pub breached: bool,
    /// Inward ΔP (Pa) past which the compartment implodes (catastrophic flooding).
    pub crush_strength: f64,
    /// Set once an implosion has occurred.
    pub imploded: bool,
}

impl FloodCompartment {
    /// A dry compartment of `volume` whose air starts at `initial_pressure`.
    pub fn new(volume: f64, centroid: DVec3, initial_pressure: f64, crush_strength: f64) -> Self {
        Self {
            volume,
            centroid,
            floodwater: 0.0,
            air_pv: initial_pressure * volume,
            breached: false,
            crush_strength,
            imploded: false,
        }
    }

    /// Build from a WI 519 [`Compartment`] (its volume and cell centroid).
    pub fn from_compartment(
        comp: &Compartment,
        cell_size: f64,
        initial_pressure: f64,
        crush_strength: f64,
    ) -> Self {
        let n = comp.cells.len().max(1) as f64;
        let sum: DVec3 = comp
            .cells
            .iter()
            .map(|c| (c.as_dvec3() + DVec3::splat(0.5)) * cell_size)
            .sum();
        Self::new(comp.volume, sum / n, initial_pressure, crush_strength)
    }

    /// Free (air-filled) volume, m³.
    pub fn free_volume(&self) -> f64 {
        (self.volume - self.floodwater).max(0.0)
    }

    /// Internal air pressure, Pa.
    pub fn internal_pressure(&self) -> f64 {
        self.air_pv / self.free_volume().max(EPS)
    }

    /// Floodwater fraction `[0, 1]`.
    pub fn flooded_fraction(&self) -> f64 {
        if self.volume > 0.0 {
            self.floodwater / self.volume
        } else {
            0.0
        }
    }

    /// Advance the transient one tick against the ambient `sample`. `rate` (1/s) is
    /// the breach-flow relaxation constant. No-op when not breached. One
    /// sign-flipped model: the sign of `ΔP` and the ambient medium select the
    /// equilibrium the compartment relaxes toward, and the transient decays as it
    /// approaches (the remaining gap shrinks).
    pub fn step(&mut self, sample: &FluidSample, rate: f64, dt: f64) {
        if !self.breached {
            return;
        }
        let free = self.free_volume().max(EPS);
        let p_internal = self.air_pv / free;
        let dp = sample.pressure - p_internal;
        let relax = (rate * dt).clamp(0.0, 1.0);

        if sample.medium == MediumKind::Liquid && dp > 0.0 {
            // Implosion: inward ΔP past crush strength → catastrophic flooding.
            if dp > self.crush_strength {
                self.floodwater = self.volume;
                self.air_pv = 0.0;
                self.imploded = true;
                return;
            }
            // Flooding: water relaxes toward full; the air vents out the breach.
            self.floodwater += (self.volume - self.floodwater) * relax;
            self.air_pv += (0.0 - self.air_pv) * relax;
        } else {
            // Gas ambient (or ΔP ≤ 0): the air equalises toward ambient pressure —
            // decompression when the cabin is the higher-pressure side.
            let target = sample.pressure * free;
            self.air_pv += (target - self.air_pv) * relax;
        }
    }
}

/// A craft's mass properties with floodwater folded in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloodedMass {
    /// Total mass (dry + floodwater), kg.
    pub mass: f64,
    /// Centre of mass (craft local frame), m — shifted toward flooded compartments.
    pub center_of_mass: DVec3,
    /// Floodwater mass alone, kg.
    pub floodwater_mass: f64,
}

/// Combine a dry craft's mass properties with floodwater point-masses at each
/// flooded compartment's centroid. Recompute each tick — this is the real-time
/// feedback of a resource (floodwater) into rigid-body mass and centre of mass.
pub fn flooded_mass_properties(
    dry: &VoxelCraft,
    floods: &[FloodCompartment],
    water_density: f64,
) -> FloodedMass {
    let (dry_mass, dry_com) = match dry.mass_properties() {
        Some(mp) => (mp.mass, mp.center_of_mass),
        None => (0.0, DVec3::ZERO),
    };
    let mut mass = dry_mass;
    let mut moment = dry_mass * dry_com;
    let mut floodwater_mass = 0.0;
    for f in floods {
        let m = f.floodwater * water_density;
        floodwater_mass += m;
        mass += m;
        moment += m * f.centroid;
    }
    let center_of_mass = if mass > EPS { moment / mass } else { dry_com };
    FloodedMass {
        mass,
        center_of_mass,
        floodwater_mass,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compartments::compartments;
    use crate::fluid::FluidMedium;
    use crate::voxel::{Material, Voxel};
    use glam::IVec3;

    const ATM: f64 = 101_325.0;
    const WATER_RHO: f64 = 1_025.0;
    const CRUSH: f64 = 5.0e6; // ~ 500 m of water
    const RATE: f64 = 1.0; // relaxation rate, 1/s

    fn vacuum() -> FluidSample {
        FluidMedium::VACUUM.sample_altitude(0.0)
    }
    fn sea_level_air() -> FluidSample {
        FluidMedium::EARTHLIKE.sample_altitude(0.0)
    }
    fn deep_water(depth: f64) -> FluidSample {
        FluidMedium::EARTHLIKE.sample_altitude(-depth)
    }

    fn cabin() -> FloodCompartment {
        // A 10 m³ compartment at 1 atm, centred at the origin.
        FloodCompartment::new(10.0, DVec3::ZERO, ATM, CRUSH)
    }

    // --- I1 sign-flipped transient ---

    #[test]
    fn decompression_vents_to_vacuum_and_decays() {
        let mut c = cabin();
        c.breached = true;
        let amb = vacuum();
        let mut prev_p = c.internal_pressure();
        let mut prev_drop = f64::INFINITY;
        for _ in 0..2_000 {
            c.step(&amb, RATE, 0.05);
            let p = c.internal_pressure();
            let drop = prev_p - p;
            assert!(p <= prev_p + 1e-9, "pressure must not rise while venting");
            // The drop per step decays (relaxation).
            assert!(drop <= prev_drop + 1e-6);
            prev_p = p;
            prev_drop = drop;
        }
        assert!(c.internal_pressure() < 1.0, "vented essentially to vacuum");
        assert_eq!(c.floodwater, 0.0, "no water in a gas ambient");
    }

    #[test]
    fn flooding_fills_underwater() {
        let mut c = cabin();
        c.breached = true;
        let amb = deep_water(50.0); // ~ 6 atm, below crush
        assert!(amb.medium == MediumKind::Liquid);
        for _ in 0..5_000 {
            c.step(&amb, RATE, 0.05);
        }
        assert!(
            (c.floodwater - c.volume).abs() < 0.05,
            "compartment floods to (near) full: {}",
            c.floodwater
        );
        assert!(c.floodwater <= c.volume + 1e-9, "bounded by volume");
        assert!(!c.imploded, "50 m is below crush strength");
    }

    #[test]
    fn one_step_handles_both_signs() {
        // The SAME step function: vacuum vents air; deep water floods.
        let mut vent = cabin();
        vent.breached = true;
        let mut flood = cabin();
        flood.breached = true;
        for _ in 0..200 {
            vent.step(&vacuum(), RATE, 0.05);
            flood.step(&deep_water(50.0), RATE, 0.05);
        }
        assert!(vent.internal_pressure() < ATM, "vacuum side decompressed");
        assert!(vent.floodwater == 0.0);
        assert!(flood.floodwater > 0.0, "ocean side flooded");
    }

    #[test]
    fn deep_breach_implodes() {
        let mut c = cabin();
        c.breached = true;
        // Very deep: hydrostatic ΔP exceeds the crush strength.
        let amb = deep_water(2_000.0); // ~ 200 atm ≫ 5 MPa crush
        c.step(&amb, RATE, 0.05);
        assert!(c.imploded, "should implode at depth");
        assert!(
            (c.floodwater - c.volume).abs() < 1e-9,
            "catastrophic full flood"
        );
    }

    #[test]
    fn sealed_compartment_is_unchanged() {
        let mut c = cabin(); // breached = false
        let before = c;
        for _ in 0..100 {
            c.step(&deep_water(50.0), RATE, 0.05);
        }
        assert_eq!(c, before, "a sealed compartment does not change");
    }

    #[test]
    fn vacuum_never_floods() {
        let mut c = cabin();
        c.breached = true;
        for _ in 0..500 {
            c.step(&vacuum(), RATE, 0.05);
        }
        assert_eq!(c.floodwater, 0.0);
    }

    // --- I2 mass feedback (in real time) ---

    /// A hollow 5³ shell → one 27 m³ compartment.
    fn hollow_craft() -> VoxelCraft {
        let mut c = VoxelCraft::new(1.0);
        let n = 5;
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let shell =
                        x == 0 || x == n - 1 || y == 0 || y == n - 1 || z == 0 || z == n - 1;
                    if shell {
                        c.voxels.push(Voxel {
                            cell: IVec3::new(x, y, z),
                            material: Material::ALUMINIUM,
                        });
                    }
                }
            }
        }
        c
    }

    /// A 7×5×5 shell with an internal wall at x=3 → two compartments, off-centre
    /// from the (symmetric) craft CoM at x=3, so flooding one shifts the CoM.
    fn two_room_craft() -> VoxelCraft {
        let mut c = VoxelCraft::new(1.0);
        let (nx, ny, nz) = (7, 5, 5);
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let shell =
                        x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1;
                    if shell || x == 3 {
                        c.voxels.push(Voxel {
                            cell: IVec3::new(x, y, z),
                            material: Material::ALUMINIUM,
                        });
                    }
                }
            }
        }
        c
    }

    #[test]
    fn flooding_adds_mass_and_shifts_com_over_time() {
        let craft = two_room_craft();
        let set = compartments(&craft);
        assert_eq!(set.count(), 2);
        // Flood one room; its centroid is off the craft CoM, so the CoM shifts.
        let mut flood =
            FloodCompartment::from_compartment(&set.compartments[0], craft.cell_size, ATM, CRUSH);
        flood.breached = true;

        let dry = flooded_mass_properties(&craft, &[flood], WATER_RHO);
        let amb = deep_water(50.0);

        // Step the flood partway and re-read: mass rises, CoM moves toward the
        // compartment centroid — in real time.
        let mut prev_mass = dry.mass;
        let mut prev_dist = 0.0;
        for k in 1..=40 {
            flood.step(&amb, RATE, 0.05);
            let fm = flooded_mass_properties(&craft, &[flood], WATER_RHO);
            assert!(fm.mass >= prev_mass - 1e-9, "mass rises as it floods");
            assert!(
                (fm.mass - (dry.mass + flood.floodwater * WATER_RHO)).abs() < 1e-6,
                "mass = dry + floodwater"
            );
            let dist = (fm.center_of_mass - dry.center_of_mass).length();
            if k > 1 {
                assert!(
                    dist >= prev_dist - 1e-9,
                    "CoM shifts toward the flooded side"
                );
            }
            prev_mass = fm.mass;
            prev_dist = dist;
        }
        assert!(prev_mass > dry.mass, "the craft got heavier");
        assert!(prev_dist > 0.0, "the CoM moved");
    }

    #[test]
    fn dry_craft_keeps_its_mass_properties() {
        let craft = hollow_craft();
        let mp = craft.mass_properties().unwrap();
        let fm = flooded_mass_properties(&craft, &[], WATER_RHO);
        assert!((fm.mass - mp.mass).abs() < 1e-9);
        assert!((fm.center_of_mass - mp.center_of_mass).length() < 1e-9);
        assert_eq!(fm.floodwater_mass, 0.0);
    }

    #[test]
    fn atmosphere_breach_equalises_without_flooding() {
        // A partially-vented cabin in sea-level air re-pressurises toward 1 atm,
        // no flooding (gas ambient).
        let mut c = FloodCompartment::new(10.0, DVec3::ZERO, ATM * 0.3, CRUSH);
        c.breached = true;
        for _ in 0..2_000 {
            c.step(&sea_level_air(), RATE, 0.05);
        }
        assert!(
            (c.internal_pressure() - ATM).abs() < 500.0,
            "equalised to ~1 atm"
        );
        assert_eq!(c.floodwater, 0.0);
    }
}
