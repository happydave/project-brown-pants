//! Ballast tanks — controllable dive / surface / hold-depth (WI 709).
//!
//! The one primitive a **submarine** needs that a generic craft does not. A ballast
//! tank admits or expels **water** to vary the craft's net weight against its fixed,
//! geometry-given buoyancy (the displaced-water force `buoyancy_wrench` computes does
//! **not** change — a tank holding water does not change the hull's external
//! displacement): **flood** the tank and the craft is heavier than the water it
//! displaces and **sinks**; **blow** it (displace the water with stored gas) and it
//! is lighter and **rises**; **balance** it and it **holds depth**.
//!
//! It is built on the **floodwater mass-feedback precedent**
//! ([`crate::flooding::flooded_mass_properties`]): each tank's water mass is a point
//! mass at the tank's mount, folded into the craft's mass and centre of mass every
//! tick. Because ballast rides that same seam, it **composes with flooding for free**
//! — a breached, ballasted hull carries *both* water masses, no special case.
//!
//! Headless: the per-sub-step driver that steps fill/blow and folds the wet mass into
//! the body is [`crate::medium::advance_descent`]; this module is the model.

use bevy_ecs::prelude::Component;
use glam::DVec3;
use serde::{Deserialize, Serialize};

/// Mass below which a body is treated as massless.
const EPS_M: f64 = 1e-12;

/// One ballast tank: a bounded reservoir of **water volume** at a body-frame mount,
/// filled (flooded) and blown (expelled) at rate limits. Fill is in m³ of water,
/// always within `[0, capacity]`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BallastTank {
    /// Maximum water volume the tank holds, m³.
    pub capacity: f64,
    /// Mount (water centroid) in the craft body frame, metres.
    pub mount: DVec3,
    /// Current water volume, m³, in `[0, capacity]`.
    pub fill: f64,
    /// Flooding rate when filling, m³/s.
    pub fill_rate: f64,
    /// Expelling rate when blowing, m³/s.
    pub blow_rate: f64,
}

impl BallastTank {
    /// Advance the fill toward the commanded `target` (m³) at the appropriate rate,
    /// clamped to `[0, capacity]`.
    fn step(&mut self, command: BallastCommand, dt: f64) {
        if dt <= 0.0 || self.capacity <= 0.0 {
            self.fill = self.fill.clamp(0.0, self.capacity.max(0.0));
            return;
        }
        match command {
            BallastCommand::Hold => {}
            BallastCommand::Fill => {
                self.fill = (self.fill + self.fill_rate.max(0.0) * dt).min(self.capacity);
            }
            BallastCommand::Blow => {
                self.fill = (self.fill - self.blow_rate.max(0.0) * dt).max(0.0);
            }
        }
    }

    /// Water mass currently held, kg, at the given water density.
    fn water_mass(&self, water_density: f64) -> f64 {
        self.fill.max(0.0) * water_density.max(0.0)
    }
}

/// What the player wants the ballast to do.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BallastCommand {
    /// Admit water — flood the tanks (the craft grows heavier, dives).
    Fill,
    /// Expel water with stored gas — blow the tanks (the craft grows lighter, rises).
    Blow,
    /// Freeze the fill (hold depth).
    #[default]
    Hold,
}

/// A craft's mass with ballast water folded in (the [`crate::flooding::FloodedMass`]
/// shape).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BallastMass {
    /// Total mass (dry + ballast water), kg.
    pub mass: f64,
    /// Wet centre of mass, body frame, m — shifted toward the (filled) tanks.
    pub center_of_mass: DVec3,
    /// Ballast water mass alone, kg.
    pub ballast_mass: f64,
}

/// A craft's ballast system: the tanks plus the current command, with the craft's
/// **dry mass** cached so the wet mass folds without a per-tick eigensolve. An
/// optional ECS component; absent ⇒ the craft uses its dry mass exactly as before.
#[derive(Component, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Ballast {
    /// The tanks.
    pub tanks: Vec<BallastTank>,
    /// The current fill/blow/hold command (applied to every tank).
    pub command: BallastCommand,
    /// The craft's dry mass, kg (cached at assembly; the fold base).
    pub dry_mass: f64,
}

impl Ballast {
    /// Advance every tank one sub-step under the current command.
    pub fn step(&mut self, dt: f64) {
        let cmd = self.command;
        for t in &mut self.tanks {
            t.step(cmd, dt);
        }
    }

    /// Total ballast water mass across all tanks, kg.
    pub fn water_mass(&self, water_density: f64) -> f64 {
        self.tanks.iter().map(|t| t.water_mass(water_density)).sum()
    }

    /// Fold the ballast water (each tank's mass at its mount) into the dry mass/CoM —
    /// the [`crate::flooding::flooded_mass_properties`] shape. `dry_com` is the craft's
    /// dry centre of mass (body frame); the dry mass is the cached [`Self::dry_mass`].
    pub fn wet_mass(&self, dry_com: DVec3, water_density: f64) -> BallastMass {
        let mut mass = self.dry_mass;
        let mut moment = self.dry_mass * dry_com;
        let mut ballast_mass = 0.0;
        for t in &self.tanks {
            let m = t.water_mass(water_density);
            ballast_mass += m;
            mass += m;
            moment += m * t.mount;
        }
        let center_of_mass = if mass > EPS_M { moment / mass } else { dry_com };
        BallastMass {
            mass,
            center_of_mass,
            ballast_mass,
        }
    }

    /// Total fill fraction in `[0, 1]` across all tanks (for the HUD); zero with no
    /// capacity.
    pub fn fill_fraction(&self) -> f64 {
        let cap: f64 = self.tanks.iter().map(|t| t.capacity).sum();
        if cap > 0.0 {
            let fill: f64 = self.tanks.iter().map(|t| t.fill).sum();
            (fill / cap).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WATER_RHO: f64 = 1025.0;

    fn one_tank(capacity: f64, mount: DVec3) -> Ballast {
        Ballast {
            tanks: vec![BallastTank {
                capacity,
                mount,
                fill: 0.0,
                fill_rate: 1.0,
                blow_rate: 1.0,
            }],
            command: BallastCommand::Hold,
            dry_mass: 1_000.0,
        }
    }

    #[test]
    fn fill_blow_hold_are_rate_limited_and_clamped() {
        let mut b = one_tank(2.0, DVec3::ZERO); // 2 m³, 1 m³/s each way
        b.command = BallastCommand::Fill;
        b.step(1.0);
        assert!((b.tanks[0].fill - 1.0).abs() < 1e-9, "filled 1 m³ in 1 s");
        b.step(10.0);
        assert!((b.tanks[0].fill - 2.0).abs() < 1e-9, "clamped at capacity");
        // Hold freezes it.
        b.command = BallastCommand::Hold;
        b.step(5.0);
        assert!((b.tanks[0].fill - 2.0).abs() < 1e-9);
        // Blow empties it, clamped at 0.
        b.command = BallastCommand::Blow;
        b.step(1.0);
        assert!((b.tanks[0].fill - 1.0).abs() < 1e-9);
        b.step(10.0);
        assert!(b.tanks[0].fill.abs() < 1e-9, "clamped at empty");
    }

    #[test]
    fn wet_mass_adds_water_and_shifts_com_toward_the_tank() {
        let mut b = one_tank(3.0, DVec3::new(0.0, -1.5, 0.0)); // tank low in the hull
                                                               // Empty: wet mass == dry, CoM unchanged.
        let dry = b.wet_mass(DVec3::ZERO, WATER_RHO);
        assert!((dry.mass - 1_000.0).abs() < 1e-9);
        assert_eq!(dry.ballast_mass, 0.0);
        assert_eq!(dry.center_of_mass, DVec3::ZERO);
        // Fill it: mass rises by capacity·ρ, CoM sinks toward the tank.
        b.command = BallastCommand::Fill;
        b.step(10.0);
        let wet = b.wet_mass(DVec3::ZERO, WATER_RHO);
        assert!((wet.ballast_mass - 3.0 * WATER_RHO).abs() < 1e-6);
        assert!((wet.mass - (1_000.0 + 3.0 * WATER_RHO)).abs() < 1e-6);
        assert!(
            wet.center_of_mass.y < 0.0,
            "CoM shifted toward the low tank"
        );
    }

    /// Invariant 2/4: ballast water mass folds **additively** through the same shape as
    /// floodwater — no interaction term. (Composition with flooding is then free: both
    /// add point masses into the body mass.)
    #[test]
    fn ballast_mass_is_additive_like_floodwater() {
        let mut b = one_tank(4.0, DVec3::new(0.0, -1.0, 0.0));
        b.command = BallastCommand::Fill;
        b.step(10.0); // full
        let wet = b.wet_mass(DVec3::ZERO, WATER_RHO);
        // The increment over dry is exactly the water mass — nothing else.
        assert!((wet.mass - b.dry_mass - b.water_mass(WATER_RHO)).abs() < 1e-9);
        // A hypothetical 5 m³ of floodwater would add 5·ρ on top, independently.
        let flood = 5.0 * WATER_RHO;
        let combined = wet.mass + flood;
        assert!((combined - (b.dry_mass + 4.0 * WATER_RHO + flood)).abs() < 1e-6);
    }

    #[test]
    fn fill_fraction_tracks_the_tanks() {
        let mut b = one_tank(2.0, DVec3::ZERO);
        assert_eq!(b.fill_fraction(), 0.0);
        b.command = BallastCommand::Fill;
        b.step(1.0); // 1 of 2 m³
        assert!((b.fill_fraction() - 0.5).abs() < 1e-9);
        // No-capacity tank ⇒ inert, no panic.
        let inert = one_tank(0.0, DVec3::ZERO);
        assert_eq!(inert.fill_fraction(), 0.0);
    }

    #[test]
    fn zero_water_density_admits_no_mass() {
        let mut b = one_tank(3.0, DVec3::ZERO);
        b.command = BallastCommand::Fill;
        b.step(10.0);
        // Above the surface / vacuum: no water to weigh.
        assert_eq!(b.water_mass(0.0), 0.0);
        assert!((b.wet_mass(DVec3::ZERO, 0.0).mass - b.dry_mass).abs() < 1e-9);
    }

    #[test]
    fn ballast_round_trips_through_serde() {
        let mut b = one_tank(3.0, DVec3::new(0.1, -1.0, 0.2));
        b.command = BallastCommand::Fill;
        b.step(0.5);
        let json = serde_json::to_string(&b).unwrap();
        let back: Ballast = serde_json::from_str(&json).unwrap();
        assert_eq!(b, back);
    }
}
