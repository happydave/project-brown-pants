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
//! [`crate::active::ActiveBody::integrate_wrench`] — the one force path. Low-speed
//! steering emerges from geometry — **differential thrust** between laterally-offset
//! thrusters makes a yaw moment — and the primary underway control is the [`Rudder`]
//! (WI 725): a hydrodynamic surface aft of the CoM whose yaw scales with speed.
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

/// A **rudder** — a hydrodynamic steering surface aft of the centre of mass (WI 725).
///
/// Unlike the screw (a powered actuator), the rudder develops a **side force** purely from the water
/// flowing past it: `½·ρ·v_fwd² · area · slope · δ` along the hull's lateral axis, where `v_fwd` is the
/// **forward** speed (the dynamic pressure on the surface) and `δ` the deflection. Acting aft of the
/// CoM it makes a **yaw moment** that scales with speed — nil at rest, sharper when faster — and
/// **reverses in reverse** (the flow hits the other face). Zero out of the water (ρ = 0). It needs no
/// power and no engine: a coasting or single-screw boat still steers. The same `½ρv²·C·area`
/// medium-agnostic shape as [`crate::aero::lift_force`].
#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rudder {
    /// Mount in the body frame, metres — **aft of the CoM** (along `−forward`) so the side force yaws.
    pub mount: DVec3,
    /// Forward (nose) unit axis in the body frame; the lateral force is ⟂ to this and to up.
    pub forward: DVec3,
    /// Plan area of the surface, m².
    pub area: f64,
    /// Lift-curve slope (side-force coefficient per radian of deflection).
    pub slope: f64,
    /// Maximum deflection, radians.
    pub max_angle: f64,
    /// Current commanded deflection, radians (signed; clamped to `±max_angle` on use).
    pub angle: f64,
}

impl Rudder {
    /// Set the deflection from a `turn` command in `[-1, 1]` (scaled to `±max_angle`).
    pub fn set_turn(&mut self, turn: f64) {
        self.angle = turn.clamp(-1.0, 1.0) * self.max_angle;
    }

    /// The rudder's steering **wrench** `(force, torque about the CoM)` in the **world** frame, given
    /// the body pose + `velocity` and the [`FluidMedium`] sampled at the rudder's own world position.
    /// Zero at rest, out of the water, or undeflected; reverses with reverse motion.
    pub fn wrench(
        &self,
        medium: &FluidMedium,
        surface_radius: f64,
        body_position: DVec3,
        orientation: DQuat,
        velocity: DVec3,
        com: DVec3,
    ) -> (DVec3, DVec3) {
        let angle = self.angle.clamp(-self.max_angle, self.max_angle);
        if angle == 0.0 || self.area <= 0.0 {
            return (DVec3::ZERO, DVec3::ZERO);
        }
        let arm = orientation * (self.mount - com);
        let world = body_position + arm;
        let density = medium
            .sample_altitude(world.length() - surface_radius)
            .density;
        if density <= 0.0 {
            return (DVec3::ZERO, DVec3::ZERO); // out of the water
        }
        let forward_world = (orientation * self.forward).normalize_or_zero();
        let v_fwd = velocity.dot(forward_world); // signed forward speed
        if v_fwd == 0.0 {
            return (DVec3::ZERO, DVec3::ZERO); // no flow over the surface ⇒ no steering
        }
        // Side force ∝ forward dynamic pressure · area · slope · deflection, reversing with reverse
        // motion (`v_fwd·|v_fwd|` is `v²` in magnitude and carries the sign of travel).
        let q_signed = 0.5 * density * v_fwd * v_fwd.abs();
        let up = if world.length() > 0.0 {
            world / world.length()
        } else {
            DVec3::Y
        };
        // The hull's lateral (starboard) axis: forward × up.
        let lateral = forward_world.cross(up).normalize_or_zero();
        let force = lateral * (q_signed * self.area * self.slope * angle);
        let torque = arm.cross(force);
        (force, torque)
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

    // --- WI 725: the rudder ---

    /// A stern rudder (aft of the CoM, low, forward +Z), deflected hard over.
    fn rudder() -> Rudder {
        Rudder {
            mount: DVec3::new(0.0, -0.5, -2.0),
            forward: DVec3::Z,
            area: 0.4,
            slope: 6.0,
            max_angle: 0.5,
            angle: 0.0,
        }
    }

    #[test]
    fn rudder_yaw_scales_with_speed_and_is_zero_at_rest() {
        let mut r = rudder();
        r.set_turn(1.0); // hard over
        let m = earthlike();
        let surface = 600_000.0;
        let pos = DVec3::new(0.0, surface - 3.0, 0.0); // rudder submerged
                                                       // At rest: no flow ⇒ no steering.
        let (_, t0) = r.wrench(&m, surface, pos, DQuat::IDENTITY, DVec3::ZERO, DVec3::ZERO);
        assert_eq!(t0, DVec3::ZERO);
        // Underway: a real yaw moment (dominant about up = +Y here), growing with v².
        let (_, t1) = r.wrench(
            &m,
            surface,
            pos,
            DQuat::IDENTITY,
            DVec3::new(0.0, 0.0, 3.0),
            DVec3::ZERO,
        );
        let (_, t2) = r.wrench(
            &m,
            surface,
            pos,
            DQuat::IDENTITY,
            DVec3::new(0.0, 0.0, 6.0),
            DVec3::ZERO,
        );
        assert!(t1.length() > 0.0, "underway the rudder steers");
        assert!(
            t1.y.abs() > t1.x.abs() && t1.y.abs() > t1.z.abs(),
            "yaw (about up) dominates: {t1:?}"
        );
        assert!(
            (t2.length() - 4.0 * t1.length()).abs() < 1e-6 * t2.length(),
            "v² scaling: {} vs {}",
            t1.length(),
            t2.length()
        );
    }

    #[test]
    fn rudder_negligible_out_of_the_water_and_zero_in_vacuum() {
        let mut r = rudder();
        r.set_turn(1.0);
        let m = earthlike();
        let surface = 600_000.0;
        let vel = DVec3::new(0.0, 0.0, 6.0);
        // Submerged: a strong bite.
        let under = DVec3::new(0.0, surface - 3.0, 0.0);
        let (fw, _) = r.wrench(&m, surface, under, DQuat::IDENTITY, vel, DVec3::ZERO);
        // In air: density-scaled like the screw ⇒ a negligible fraction of the water bite.
        let above = DVec3::new(0.0, surface + 5.0, 0.0);
        let (fa, _) = r.wrench(&m, surface, above, DQuat::IDENTITY, vel, DVec3::ZERO);
        assert!(fw.length() > 0.0);
        assert!(
            fa.length() < 0.01 * fw.length(),
            "negligible bite in air: air {} vs water {}",
            fa.length(),
            fw.length()
        );
        // True vacuum: exactly zero.
        let (fv, tv) = r.wrench(
            &FluidMedium::VACUUM,
            surface,
            under,
            DQuat::IDENTITY,
            vel,
            DVec3::ZERO,
        );
        assert_eq!(fv, DVec3::ZERO);
        assert_eq!(tv, DVec3::ZERO);
    }

    #[test]
    fn rudder_deflection_sign_and_reverse_flip_the_yaw() {
        let m = earthlike();
        let surface = 600_000.0;
        let pos = DVec3::new(0.0, surface - 3.0, 0.0);
        let fwd = DVec3::new(0.0, 0.0, 6.0);
        let yaw = |angle: f64, vel: DVec3| {
            let mut r = rudder();
            r.angle = angle;
            r.wrench(&m, surface, pos, DQuat::IDENTITY, vel, DVec3::ZERO)
                .1
                .y
        };
        // Opposite deflections ⇒ opposite yaw.
        assert!(
            yaw(0.4, fwd) * yaw(-0.4, fwd) < 0.0,
            "deflection sign sets yaw"
        );
        // Same deflection in reverse ⇒ opposite yaw to forward.
        let rev = DVec3::new(0.0, 0.0, -6.0);
        assert!(
            yaw(0.4, fwd) * yaw(0.4, rev) < 0.0,
            "reverse flips the steering"
        );
    }

    #[test]
    fn rudder_steers_a_coasting_body_no_power() {
        // A body coasting forward with a deflected rudder gains yaw — no engine, no resource.
        let mut r = rudder();
        r.set_turn(1.0);
        let m = earthlike();
        let surface = 600_000.0;
        let mut body = ActiveBody::new(
            DVec3::new(0.0, surface - 3.0, 0.0),
            DVec3::new(0.0, 0.0, 6.0),
            1_000.0,
            DMat3::IDENTITY,
        );
        for _ in 0..50 {
            let (f, t) = r.wrench(
                &m,
                surface,
                body.position,
                body.orientation,
                body.velocity,
                DVec3::ZERO,
            );
            body.integrate_wrench(f, t, 0.01);
        }
        assert!(
            body.angular_velocity().length() > 0.0,
            "the coasting hull turned under rudder alone: {:?}",
            body.angular_velocity()
        );
        assert!(body.velocity.is_finite() && body.angular_velocity().is_finite());
    }

    #[test]
    fn rudder_round_trips_through_serde() {
        let mut r = rudder();
        r.set_turn(0.5);
        let json = serde_json::to_string(&r).unwrap();
        let back: Rudder = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }
}
