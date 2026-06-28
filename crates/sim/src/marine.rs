//! Marine propulsion — a screw / ducted thruster (WI 708).
//!
//! The water-side counterpart to the rocket [`crate::propulsion::Engine`]. A
//! **screw** develops thrust by pushing against the **surrounding fluid**, so —
//! unlike a rocket (mass expelled from a tank, thrust independent of the medium,
//! works in vacuum) — its thrust **scales with the density of the medium it sits
//! in**: full in water, a negligible fraction in air (the air/water density ratio,
//! ~0.1 %), and **exactly zero in vacuum**. This falls out of a single density
//! factor — there is **no branch on medium identity**, the same discipline
//! [`crate::medium::drag_force`] and [`crate::medium::buoyancy_force`] follow.
//!
//! A thruster draws fuel/charge from a [`ResourceGraph`] reservoir while commanded
//! (the screw spins whatever medium it is in — running it clear of the water
//! wastes fuel for ~no thrust), rationed per tank exactly as [`crate::propulsion`]
//! does. Thrust enters the simulation as a **wrench** (force along the mounted axis
//! at the device's position → force + moment about the centre of mass) through
//! [`crate::active::ActiveBody::integrate_wrench`] — the one force path. Steering
//! emerges from geometry: **differential thrust** between laterally-offset thrusters
//! makes a yaw moment, no dedicated rudder.
//!
//! Headless: the per-sub-step driver that reads throttle and applies the wrench is
//! [`crate::medium::advance_descent`] (the harbor/dive share it); this module is the
//! model + the command applicator.

use crate::fluid::FluidMedium;
use crate::resource::{ReservoirId, ResourceGraph};
use bevy_ecs::prelude::Component;
use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

/// A marine thruster profile — content (a new thruster is new data, not new code).
/// Thrust acts along [`axis`](Self::axis) at [`mount`](Self::mount), both body-frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarineThruster {
    /// Reservoir this thruster draws (within the [`MarinePropulsion`] graph).
    pub tank: ReservoirId,
    /// Thrust at full throttle and the reference density, N.
    pub max_thrust: f64,
    /// The medium density at which `max_thrust` is delivered, kg/m³ (water surface,
    /// ~1025). Thrust scales linearly with the *local* density up to this value.
    pub reference_density: f64,
    /// Resource drawn per second at full throttle (independent of medium — the screw
    /// spins regardless).
    pub max_draw: f64,
    /// Mount position in the craft body frame, metres.
    pub mount: DVec3,
    /// Forward thrust direction in the body frame; normalised on use.
    pub axis: DVec3,
}

impl MarineThruster {
    /// Signed thrust magnitude, N, at a `throttle` (clamped `[-1, 1]`; negative =
    /// reverse) and the local `ambient_density`. Zero at zero density (vacuum) and
    /// negligible in air, by the density factor `min(ρ/ρ_ref, 1)`.
    pub fn thrust(&self, throttle: f64, ambient_density: f64) -> f64 {
        if self.reference_density <= 0.0 {
            return 0.0;
        }
        let factor = (ambient_density / self.reference_density).clamp(0.0, 1.0);
        throttle.clamp(-1.0, 1.0) * self.max_thrust * factor
    }
}

/// Per-thruster command: a throttle in `[-1, 1]` (negative reverses).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ThrusterCommand {
    /// Throttle setting, clamped to `[-1, 1]` on use.
    pub throttle: f64,
}

/// A craft's marine propulsion: fuel/charge tanks (a [`ResourceGraph`]), the
/// thrusters, and their per-thruster command state. An optional ECS component on a
/// floating craft; absent ⇒ no marine thrust (the dive path is unaffected).
#[derive(Component, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MarinePropulsion {
    /// Fuel/charge tanks (`graph.reservoirs[i]` is the i-th tank).
    pub graph: ResourceGraph,
    /// The thrusters.
    pub thrusters: Vec<MarineThruster>,
    /// Per-thruster command state, aligned with `thrusters`.
    pub commands: Vec<ThrusterCommand>,
    /// Last net thrust magnitude (N) from [`Self::thrust_step`] — a HUD/telemetry
    /// readout, not persisted.
    #[serde(skip)]
    pub last_thrust: f64,
}

impl MarinePropulsion {
    /// Advance one active sub-step: draw fuel/charge (throttle-scaled, rationed per
    /// tank when over-demanded, zero on an empty tank) and return the net thrust
    /// **wrench** `(force, torque)` in the **world** frame about the centre of mass
    /// `com` (body frame), given the body pose and the [`FluidMedium`] sampled at
    /// each thruster's own world position (so a screw at the surface and one deep
    /// under the keel are scaled by their own local density). Deducts the drawn
    /// resource; updates [`Self::last_thrust`].
    pub fn thrust_step(
        &mut self,
        medium: &FluidMedium,
        surface_radius: f64,
        body_position: DVec3,
        orientation: DQuat,
        com: DVec3,
        dt: f64,
    ) -> (DVec3, DVec3) {
        if dt <= 0.0 || self.thrusters.is_empty() {
            self.last_thrust = 0.0;
            return (DVec3::ZERO, DVec3::ZERO);
        }
        let n_tanks = self.graph.reservoirs.len();

        // Per-tank demand this step (|throttle|-scaled), then a ration factor so the
        // total draw never exceeds what the tank holds (shared tanks ration
        // proportionally — the resource-graph contention rule, as `Propulsion` does).
        let mut demand = vec![0.0_f64; n_tanks];
        for (t, c) in self.thrusters.iter().zip(&self.commands) {
            if let Some(d) = demand.get_mut(t.tank.0) {
                *d += c.throttle.clamp(-1.0, 1.0).abs() * t.max_draw.max(0.0) * dt;
            }
        }
        let ration: Vec<f64> = (0..n_tanks)
            .map(|i| {
                let avail = self.graph.reservoirs[i].amount;
                if demand[i] > avail && demand[i] > 0.0 {
                    (avail / demand[i]).clamp(0.0, 1.0)
                } else {
                    1.0
                }
            })
            .collect();

        let mut force = DVec3::ZERO;
        let mut torque = DVec3::ZERO;
        for (t, c) in self.thrusters.iter().zip(&self.commands) {
            let Some(&r) = ration.get(t.tank.0) else {
                continue;
            };
            let throttle = c.throttle.clamp(-1.0, 1.0);
            // Draw the rationed resource (the screw spins regardless of medium).
            let draw = throttle.abs() * t.max_draw.max(0.0) * r * dt;
            if draw > 0.0 {
                let res = &mut self.graph.reservoirs[t.tank.0];
                res.amount = (res.amount - draw).max(0.0);
            }
            // Thrust scales with the resource fraction actually delivered (an empty
            // tank, r = 0, makes no thrust) and with the local medium density.
            let arm = orientation * (t.mount - com);
            let world = body_position + arm;
            let altitude = world.length() - surface_radius;
            let density = medium.sample_altitude(altitude).density;
            let mag = t.thrust(throttle * r, density);
            if mag == 0.0 {
                continue;
            }
            let dir = (orientation * t.axis).normalize_or_zero();
            let f = mag * dir;
            force += f;
            torque += arm.cross(f);
        }
        self.last_thrust = force.length();
        (force, torque)
    }

    /// Set every thruster's throttle for a `forward` drive plus a `turn`
    /// (differential): a thruster to **starboard** (`mount.x > 0`) gets `forward +
    /// turn`, one to **port** (`mount.x < 0`) gets `forward - turn`, so a nonzero
    /// `turn` makes a yaw couple. Both inputs are taken in `[-1, 1]`; the result is
    /// clamped to `[-1, 1]`.
    pub fn drive(&mut self, forward: f64, turn: f64) {
        for (t, c) in self.thrusters.iter().zip(&mut self.commands) {
            let side = if t.mount.x >= 0.0 { 1.0 } else { -1.0 };
            c.throttle = (forward + side * turn).clamp(-1.0, 1.0);
        }
    }

    /// Total remaining fuel/charge across all tanks, in resource units.
    pub fn fuel(&self) -> f64 {
        self.graph.reservoirs.iter().map(|r| r.amount).sum()
    }

    /// Fuel/charge fill fraction in `[0, 1]` (for the HUD); zero with no capacity.
    pub fn fuel_fraction(&self) -> f64 {
        let cap: f64 = self.graph.reservoirs.iter().map(|r| r.capacity).sum();
        if cap > 0.0 {
            (self.fuel() / cap).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active::ActiveBody;
    use crate::resource::Reservoir;
    use crate::resource::ResourceType;
    use glam::DMat3;

    const FUEL: ResourceType = ResourceType(0);
    const WATER_RHO: f64 = 1025.0;
    const AIR_RHO: f64 = 1.225;

    /// One thruster on one tank, thrust along +Z (forward), mounted at the origin.
    fn single(fuel: f64, max_thrust: f64, max_draw: f64) -> MarinePropulsion {
        MarinePropulsion {
            graph: ResourceGraph {
                reservoirs: vec![Reservoir::new(FUEL, fuel, fuel.max(1.0))],
                ..Default::default()
            },
            thrusters: vec![MarineThruster {
                tank: ReservoirId(0),
                max_thrust,
                reference_density: WATER_RHO,
                max_draw,
                mount: DVec3::ZERO,
                axis: DVec3::Z,
            }],
            commands: vec![ThrusterCommand::default()],
            last_thrust: 0.0,
        }
    }

    /// A medium with an ocean of `WATER_RHO` below the surface and air above (so the
    /// altitude sign selects the medium).
    fn earthlike() -> FluidMedium {
        FluidMedium::EARTHLIKE
    }

    // --- Invariant: thrust scales with medium density, zero in vacuum ---

    #[test]
    fn thrust_scales_with_density_full_in_water_negligible_in_air_zero_in_vacuum() {
        let t = MarineThruster {
            tank: ReservoirId(0),
            max_thrust: 10_000.0,
            reference_density: WATER_RHO,
            max_draw: 1.0,
            mount: DVec3::ZERO,
            axis: DVec3::Z,
        };
        let water = t.thrust(1.0, WATER_RHO);
        let air = t.thrust(1.0, AIR_RHO);
        let vac = t.thrust(1.0, 0.0);
        assert!(
            (water - 10_000.0).abs() < 1e-9,
            "full thrust at reference density"
        );
        assert_eq!(vac, 0.0, "exactly zero in vacuum");
        assert!(
            air.abs() < 0.01 * water,
            "air thrust is a negligible fraction of water: {air} vs {water}"
        );
        // Denser-than-reference does not exceed max (clamped factor).
        assert!((t.thrust(1.0, 2.0 * WATER_RHO) - 10_000.0).abs() < 1e-9);
        // Linear in throttle.
        assert!((t.thrust(0.5, WATER_RHO) - 5_000.0).abs() < 1e-9);
        // Reverse flips the sign.
        assert!((t.thrust(-1.0, WATER_RHO) + 10_000.0).abs() < 1e-9);
        // A degenerate reference density is inert, not a panic.
        let bad = MarineThruster {
            reference_density: 0.0,
            ..t
        };
        assert_eq!(bad.thrust(1.0, WATER_RHO), 0.0);
    }

    // --- Required behaviour: drive + flame-out + rationing through thrust_step ---

    /// A thruster a few metres under the surface (so the medium samples as water)
    /// drives forward; lifting it clear (in air) develops essentially nothing.
    #[test]
    fn thrust_step_drives_in_water_and_not_in_air() {
        let mut p = single(1_000.0, 8_000.0, 1.0);
        p.commands[0].throttle = 1.0;
        let medium = earthlike();
        let r = FluidMedium::EARTHLIKE; // for clarity
        let _ = r;
        let surface = 600_000.0;
        // Body 3 m below the surface (mount at origin ⇒ the thruster is submerged).
        let under = DVec3::new(0.0, surface - 3.0, 0.0);
        let (f_w, _) = p.thrust_step(&medium, surface, under, DQuat::IDENTITY, DVec3::ZERO, 0.1);
        assert!(f_w.z > 0.0, "drives forward (+Z) in water: {f_w:?}");
        assert!(
            (f_w.length() - 8_000.0).abs() < 1.0,
            "full thrust submerged"
        );
        // Body 3 m above the surface (thruster in air).
        let above = DVec3::new(0.0, surface + 3.0, 0.0);
        let (f_a, _) = p.thrust_step(&medium, surface, above, DQuat::IDENTITY, DVec3::ZERO, 0.1);
        assert!(
            f_a.length() < 0.01 * f_w.length(),
            "negligible thrust clear of the water: {f_a:?}"
        );
    }

    #[test]
    fn flame_out_when_tank_empty_then_coasts() {
        let mut p = single(0.5, 8_000.0, 1.0); // 0.5 s of fuel at full throttle
        p.commands[0].throttle = 1.0;
        let medium = earthlike();
        let surface = 600_000.0;
        let under = DVec3::new(0.0, surface - 3.0, 0.0);
        let mut last = 0.0;
        for _ in 0..200 {
            last = p
                .thrust_step(&medium, surface, under, DQuat::IDENTITY, DVec3::ZERO, 0.01)
                .0
                .length();
        }
        assert!(p.fuel() >= 0.0, "never negative");
        assert!(p.fuel() < 1e-6, "tank drained");
        let (f, _) = p.thrust_step(&medium, surface, under, DQuat::IDENTITY, DVec3::ZERO, 0.01);
        assert_eq!(f.length(), 0.0, "flamed out; last nonzero was {last}");
        assert_eq!(p.last_thrust, 0.0);
    }

    #[test]
    fn shared_tank_rations_proportionally() {
        // Two thrusters on one tank, only enough fuel for part of the combined demand.
        let mut p = single(0.3, 8_000.0, 1.0);
        let second = p.thrusters[0];
        p.thrusters.push(second);
        p.commands.push(ThrusterCommand::default());
        p.commands[0].throttle = 1.0;
        p.commands[1].throttle = 1.0;
        let medium = earthlike();
        let surface = 600_000.0;
        let under = DVec3::new(0.0, surface - 3.0, 0.0);
        // Demand over 1 s = (1 + 1) = 2 units; only 0.3 available → tank empties, not over-drawn.
        let _ = p.thrust_step(&medium, surface, under, DQuat::IDENTITY, DVec3::ZERO, 1.0);
        assert!(p.fuel().abs() < 1e-9, "tank emptied, not over-drawn");
    }

    // --- Steering: differential thrust yields a yaw moment ---

    #[test]
    fn differential_thrust_makes_a_yaw_moment_equal_thrust_does_not() {
        // Port/starboard thrusters at ±X, thrust along +Z.
        let mut p = single(10_000.0, 8_000.0, 1.0);
        p.thrusters[0].mount = DVec3::new(1.5, 0.0, -2.0); // starboard
        let mut port = p.thrusters[0];
        port.mount = DVec3::new(-1.5, 0.0, -2.0);
        p.thrusters.push(port);
        p.commands.push(ThrusterCommand::default());
        let medium = earthlike();
        let surface = 600_000.0;
        let under = DVec3::new(0.0, surface - 3.0, 0.0);

        // Equal forward thrust: pure forward force, ~zero yaw.
        p.drive(1.0, 0.0);
        let (f, tq) = p.thrust_step(&medium, surface, under, DQuat::IDENTITY, DVec3::ZERO, 0.1);
        assert!(f.z > 0.0, "net forward");
        assert!(tq.y.abs() < 1e-6, "no yaw from symmetric thrust: {tq:?}");

        // Turn: differential throttle ⇒ a yaw couple about up (+Y).
        p.drive(0.5, 0.5);
        let (_f2, tq2) = p.thrust_step(&medium, surface, under, DQuat::IDENTITY, DVec3::ZERO, 0.1);
        assert!(tq2.y.abs() > 1e-3, "differential thrust yaws: {tq2:?}");
    }

    // --- The wrench actually accelerates a body (closes the loop) ---

    #[test]
    fn thrust_accelerates_a_floating_body_forward() {
        let mut p = single(1e6, 8_000.0, 0.1);
        p.commands[0].throttle = 1.0;
        let medium = earthlike();
        let surface = 600_000.0;
        let mut body = ActiveBody::new(
            DVec3::new(0.0, surface - 3.0, 0.0),
            DVec3::ZERO,
            1_000.0,
            DMat3::IDENTITY,
        );
        let dt = 0.01;
        for _ in 0..200 {
            let (f, tq) = p.thrust_step(
                &medium,
                surface,
                body.position,
                body.orientation,
                DVec3::ZERO,
                dt,
            );
            body.integrate_wrench(f, tq, dt);
        }
        assert!(
            body.velocity.z > 0.0,
            "accelerated forward: {:?}",
            body.velocity
        );
        assert!(body.velocity.is_finite());
    }

    #[test]
    fn drive_clamps_and_sets_per_side() {
        let mut p = single(10.0, 8_000.0, 1.0);
        p.thrusters[0].mount = DVec3::new(1.0, 0.0, 0.0);
        let mut port = p.thrusters[0];
        port.mount = DVec3::new(-1.0, 0.0, 0.0);
        p.thrusters.push(port);
        p.commands.push(ThrusterCommand::default());
        p.drive(2.0, 0.0); // over-range forward clamps to 1
        assert!((p.commands[0].throttle - 1.0).abs() < 1e-9);
        assert!((p.commands[1].throttle - 1.0).abs() < 1e-9);
        p.drive(0.0, 1.0);
        assert!(
            p.commands[0].throttle > 0.0 && p.commands[1].throttle < 0.0,
            "opposite sides"
        );
    }

    #[test]
    fn fuel_fraction_tracks_the_tanks() {
        let p = single(500.0, 8_000.0, 1.0); // capacity 500
        assert!((p.fuel_fraction() - 1.0).abs() < 1e-9);
        let empty = single(0.0, 8_000.0, 1.0);
        assert_eq!(empty.fuel_fraction(), 0.0);
    }

    #[test]
    fn marine_round_trips_through_serde() {
        let mut p = single(1_000.0, 8_000.0, 1.0);
        p.commands[0].throttle = 0.6;
        let json = serde_json::to_string(&p).unwrap();
        let back: MarinePropulsion = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }
}
