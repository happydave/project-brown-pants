//! Engine thrust and propellant — devices-as-active (WI 531).
//!
//! The one new mechanic that lets a craft leave the pad. An **engine** draws
//! propellant from a [`ResourceGraph`] reservoir at a throttle-scaled rate and
//! emits a **thrust wrench** (force along its mounted, gimballed axis at the
//! device's position → force + moment about the centre of mass) into
//! [`crate::active::ActiveBody::integrate_wrench`] — the same force path drag, lift,
//! and the rover already use. Thrust is `effective_mass_flow · exhaust_velocity`;
//! when a tank empties the flow (and thrust) go to zero — flame-out. As propellant
//! drains, the wet mass and centre of mass shift in real time (the
//! [`crate::flooding`] mass-fold pattern).
//!
//! This is a **headless capability**: the ECS systems that read throttle/gimbal
//! commands and call [`Propulsion::thrust_step`] each active sub-step live in the
//! game-session / control items; here is the model + the command applicator.
//! Thrust is an **active-gear, real-time** effect (per-sub-step draw) — it forces a
//! drop out of warp; on-rails analytic (low-thrust) burns are deferred.

use crate::command::Command;
use crate::resource::{ReservoirId, ResourceGraph};
use glam::{DQuat, DVec2, DVec3};
use serde::{Deserialize, Serialize};

/// Mass threshold below which a body is treated as massless.
const EPS_M: f64 = 1e-12;

/// An engine: a propellant draw plus a thrust wrench. A *device profile* — content,
/// per the governing discipline (new engines are new data, not new code).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Engine {
    /// Propellant reservoir this engine draws (within the [`Propulsion`] graph).
    pub tank: ReservoirId,
    /// Exhaust velocity, m/s (`= Isp · g₀`). Thrust = mass-flow · this.
    pub exhaust_velocity: f64,
    /// Propellant mass-flow at full throttle, kg/s.
    pub max_mass_flow: f64,
    /// Mount position in the craft's body frame, metres.
    pub mount: DVec3,
    /// Thrust direction (the force direction) in the body frame; normalised on use.
    pub axis: DVec3,
    /// Maximum gimbal deflection, radians (`0` = fixed nozzle).
    pub max_gimbal: f64,
}

impl Engine {
    /// Maximum thrust, N (`max_mass_flow · exhaust_velocity`).
    pub fn max_thrust(&self) -> f64 {
        self.max_mass_flow * self.exhaust_velocity
    }
}

/// Per-engine command state: throttle in `[0, 1]` and a gimbal deflection (clamped
/// to the engine's `max_gimbal` on use).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EngineCommand {
    /// Throttle setting, clamped to `[0, 1]`.
    pub throttle: f64,
    /// Gimbal deflection (radians, per the two axes ⟂ to thrust).
    pub gimbal: DVec2,
}

impl Default for EngineCommand {
    fn default() -> Self {
        Self {
            throttle: 0.0,
            gimbal: DVec2::ZERO,
        }
    }
}

/// The propellant centre of mass after folding tank contents into the dry craft.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WetMass {
    /// Total (dry + propellant) mass, kg.
    pub mass: f64,
    /// Wet centre of mass, body frame, metres.
    pub center_of_mass: DVec3,
    /// Propellant mass folded in, kg.
    pub propellant_mass: f64,
}

/// A craft's propulsion: propellant tanks (reservoirs in a [`ResourceGraph`], each
/// with a body-frame mount for the mass fold), engines, and per-engine command
/// state. Tanks reuse the resource graph's bounded `[0, capacity]` semantics;
/// propellant quantity is measured in **kg** so it folds directly into mass.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Propulsion {
    /// Propellant tanks (`graph.reservoirs[i]` is the i-th tank).
    pub graph: ResourceGraph,
    /// Body-frame mount of each reservoir, aligned with `graph.reservoirs`.
    pub tank_mounts: Vec<DVec3>,
    /// Engines.
    pub engines: Vec<Engine>,
    /// Per-engine command state, aligned with `engines`.
    pub commands: Vec<EngineCommand>,
}

impl Propulsion {
    /// Advance propulsion one active sub-step: draw propellant (throttle-scaled,
    /// rationed per tank when over-demanded, zero on an empty tank), and return the
    /// net thrust **wrench** `(force, torque)` in the **world** frame about the
    /// centre of mass `com` (body frame), given the body `orientation`. Deducts the
    /// drawn propellant from the tanks.
    pub fn thrust_step(&mut self, orientation: DQuat, com: DVec3, dt: f64) -> (DVec3, DVec3) {
        if dt <= 0.0 || self.engines.is_empty() {
            return (DVec3::ZERO, DVec3::ZERO);
        }
        let n_tanks = self.graph.reservoirs.len();

        // Per-tank propellant demand this step (throttle-scaled), then a ration
        // factor so total draw never exceeds what the tank holds (shared tanks
        // ration proportionally — the resource graph's contention rule).
        let mut demand = vec![0.0_f64; n_tanks];
        for (e, c) in self.engines.iter().zip(&self.commands) {
            if let Some(d) = demand.get_mut(e.tank.0) {
                *d += c.throttle.clamp(0.0, 1.0) * e.max_mass_flow.max(0.0) * dt;
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
        for (e, c) in self.engines.iter().zip(&self.commands) {
            let Some(&r) = ration.get(e.tank.0) else {
                continue;
            };
            let mdot = c.throttle.clamp(0.0, 1.0) * e.max_mass_flow.max(0.0) * r;
            if mdot <= 0.0 || e.exhaust_velocity <= 0.0 {
                continue;
            }
            // Draw propellant.
            let drawn = mdot * dt;
            let res = &mut self.graph.reservoirs[e.tank.0];
            res.amount = (res.amount - drawn).max(0.0);

            // Thrust wrench: force along the gimballed body axis at the mount.
            let thrust = mdot * e.exhaust_velocity;
            let axis_world = orientation * deflect(e.axis, c.gimbal, e.max_gimbal);
            let f = thrust * axis_world;
            let arm = orientation * (e.mount - com);
            force += f;
            torque += arm.cross(f);
        }
        (force, torque)
    }

    /// Fold tank propellant (each tank's mass at its mount) into the dry mass/CoM —
    /// so a draining craft's wet mass falls and its CoM shifts (the
    /// [`crate::flooding::flooded_mass_properties`] pattern; mass + CoM, inertia
    /// held constant within a burn as a first approximation).
    pub fn wet_mass(&self, dry_mass: f64, dry_com: DVec3) -> WetMass {
        let mut mass = dry_mass;
        let mut moment = dry_mass * dry_com;
        let mut propellant_mass = 0.0;
        for (i, res) in self.graph.reservoirs.iter().enumerate() {
            let m = res.amount; // propellant quantity is kg
            let mount = self.tank_mounts.get(i).copied().unwrap_or(DVec3::ZERO);
            propellant_mass += m;
            mass += m;
            moment += m * mount;
        }
        let center_of_mass = if mass > EPS_M { moment / mass } else { dry_com };
        WetMass {
            mass,
            center_of_mass,
            propellant_mass,
        }
    }

    /// Applies a propulsion [`Command`] to engine command state. `SetThrottle` sets
    /// every engine's throttle; `SetGimbal` sets every gimbal-capable engine's
    /// gimbal. Other commands are ignored (returns `false`). The structural analogue
    /// of `SetGear`: applied here, not by the pure `apply_command`.
    pub fn apply_command(&mut self, cmd: &Command) -> bool {
        match *cmd {
            Command::SetThrottle(t) => {
                let t = t.clamp(0.0, 1.0);
                for c in &mut self.commands {
                    c.throttle = t;
                }
                true
            }
            Command::SetGimbal(g) => {
                for (e, c) in self.engines.iter().zip(&mut self.commands) {
                    c.gimbal = if e.max_gimbal > 0.0 {
                        g.clamp_length_max(e.max_gimbal)
                    } else {
                        DVec2::ZERO
                    };
                }
                true
            }
            _ => false,
        }
    }
}

/// Deflects a thrust `axis` by a `gimbal` deflection (clamped to `max`), rotating
/// about the two axes perpendicular to it. A zero gimbal limit or zero axis leaves
/// the (normalised) axis unchanged.
fn deflect(axis: DVec3, gimbal: DVec2, max: f64) -> DVec3 {
    let a = axis.normalize_or_zero();
    if a == DVec3::ZERO || max <= 0.0 {
        return a;
    }
    let g = gimbal.clamp_length_max(max);
    if g == DVec2::ZERO {
        return a;
    }
    let reference = if a.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
    let p1 = a.cross(reference).normalize_or_zero();
    let p2 = a.cross(p1).normalize_or_zero();
    let rot = DQuat::from_axis_angle(p1, g.x) * DQuat::from_axis_angle(p2, g.y);
    (rot * a).normalize_or_zero()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active::ActiveBody;
    use crate::resource::{Reservoir, ResourceType};
    use glam::DMat3;

    const PROP: ResourceType = ResourceType(0);

    /// One engine drawing one tank, thrust along +X, mounted at the origin.
    fn single_engine(prop_kg: f64, max_flow: f64, v_e: f64) -> Propulsion {
        Propulsion {
            graph: ResourceGraph {
                reservoirs: vec![Reservoir::new(PROP, prop_kg, prop_kg.max(1.0))],
                ..Default::default()
            },
            tank_mounts: vec![DVec3::ZERO],
            engines: vec![Engine {
                tank: ReservoirId(0),
                exhaust_velocity: v_e,
                max_mass_flow: max_flow,
                mount: DVec3::ZERO,
                axis: DVec3::X,
                max_gimbal: 0.0,
            }],
            commands: vec![EngineCommand::default()],
        }
    }

    #[test]
    fn throttle_scales_draw_and_thrust() {
        let mut p = single_engine(1000.0, 10.0, 3000.0);
        p.commands[0].throttle = 0.5;
        let (f, _) = p.thrust_step(DQuat::IDENTITY, DVec3::ZERO, 1.0);
        // mdot = 0.5·10 = 5 kg/s; thrust = 5·3000 = 15 000 N along +X.
        assert!(
            (f - DVec3::new(15_000.0, 0.0, 0.0)).length() < 1e-6,
            "{f:?}"
        );
        // Drew 5 kg.
        assert!((p.graph.reservoirs[0].amount - 995.0).abs() < 1e-9);
        // Zero throttle → no thrust, no draw.
        p.commands[0].throttle = 0.0;
        let (f0, _) = p.thrust_step(DQuat::IDENTITY, DVec3::ZERO, 1.0);
        assert_eq!(f0, DVec3::ZERO);
        assert!((p.graph.reservoirs[0].amount - 995.0).abs() < 1e-9);
    }

    #[test]
    fn flame_out_when_tank_empty() {
        let mut p = single_engine(10.0, 10.0, 3000.0); // 1 s of propellant
        p.commands[0].throttle = 1.0;
        // Burn it dry over 2 s.
        let mut last = DVec3::ZERO;
        for _ in 0..200 {
            last = p.thrust_step(DQuat::IDENTITY, DVec3::ZERO, 0.01).0;
        }
        assert!(p.graph.reservoirs[0].amount >= 0.0, "never negative");
        assert!(p.graph.reservoirs[0].amount < 1e-6, "tank drained");
        // After empty, thrust is zero (flame-out).
        let (f, _) = p.thrust_step(DQuat::IDENTITY, DVec3::ZERO, 0.01);
        assert_eq!(f, DVec3::ZERO, "flamed out; last nonzero was {last:?}");
    }

    #[test]
    fn shared_tank_rations_proportionally() {
        // Two engines on one tank with only enough for half the combined demand.
        let mut p = single_engine(3.0, 10.0, 1000.0); // tank holds 3 kg
        p.engines.push(p.engines[0]); // second engine, same tank
        p.commands.push(EngineCommand::default());
        p.commands[0].throttle = 1.0;
        p.commands[1].throttle = 1.0;
        // Demand over 1 s = (10 + 10)·1 = 20 kg; only 3 available → ration 3/20.
        let _ = p.thrust_step(DQuat::IDENTITY, DVec3::ZERO, 1.0);
        // Total draw == available; tank not negative.
        assert!(
            p.graph.reservoirs[0].amount.abs() < 1e-9,
            "tank emptied, not over-drawn"
        );
    }

    #[test]
    fn thrust_accelerates_a_body_per_tsiolkovsky() {
        // 1000 kg dry + 1000 kg propellant; v_e 3000; burn to depletion in vacuum.
        let dry = 1000.0;
        let mut p = single_engine(1000.0, 10.0, 3000.0);
        p.commands[0].throttle = 1.0;
        let mut body = ActiveBody::new(DVec3::ZERO, DVec3::ZERO, dry + 1000.0, DMat3::IDENTITY);
        let dt = 0.01;
        for _ in 0..20_000 {
            let com = DVec3::ZERO;
            let (f, tq) = p.thrust_step(body.orientation, com, dt);
            body.mass = p.wet_mass(dry, com).mass; // mass falls as propellant burns
            body.integrate_wrench(f, tq, dt);
            if p.graph.reservoirs[0].amount <= 0.0 {
                break;
            }
        }
        // Ideal Δv = v_e·ln(m0/m1) = 3000·ln(2000/1000) ≈ 2079 m/s along +X.
        let dv_ideal = 3000.0 * (2000.0_f64 / 1000.0).ln();
        assert!(body.velocity.is_finite());
        assert!(
            (body.velocity.x - dv_ideal).abs() < 0.02 * dv_ideal,
            "Tsiolkovsky: got {} vs ideal {dv_ideal}",
            body.velocity.x
        );
        assert!(body.velocity.y.abs() < 1e-6 && body.velocity.z.abs() < 1e-6);
    }

    #[test]
    fn propellant_drain_shifts_mass_and_com() {
        // Tank mounted off-origin (+X); dry CoM at origin.
        let mut p = single_engine(500.0, 10.0, 3000.0);
        p.tank_mounts[0] = DVec3::new(2.0, 0.0, 0.0);
        let full = p.wet_mass(1000.0, DVec3::ZERO);
        assert!((full.mass - 1500.0).abs() < 1e-9);
        assert!(
            full.center_of_mass.x > 0.0,
            "CoM pulled toward the full tank"
        );
        // Drain it.
        p.commands[0].throttle = 1.0;
        for _ in 0..6000 {
            p.thrust_step(DQuat::IDENTITY, DVec3::ZERO, 0.01);
        }
        let empty = p.wet_mass(1000.0, DVec3::ZERO);
        assert!(empty.mass < full.mass, "wet mass fell as propellant burned");
        assert!((empty.mass - 1000.0).abs() < 1e-3, "back to dry mass");
        assert!(
            empty.center_of_mass.x < full.center_of_mass.x,
            "CoM shifted back toward dry as the tank emptied"
        );
    }

    #[test]
    fn gimbal_deflects_axis_and_makes_a_steering_moment() {
        // Engine mounted behind the CoM (−X), thrust +X, gimbal-capable.
        let mut p = single_engine(1000.0, 10.0, 3000.0);
        p.engines[0].mount = DVec3::new(-3.0, 0.0, 0.0);
        p.engines[0].max_gimbal = 0.1; // ~5.7°
        p.commands[0].throttle = 1.0;
        p.commands[0].gimbal = DVec2::new(0.1, 0.0); // deflect about a ⟂ axis
        let (f, tq) = p.thrust_step(DQuat::IDENTITY, DVec3::ZERO, 0.01);
        assert!(f.x > 0.0, "still mostly forward");
        assert!(
            f.length() - f.x.abs() > 1e-6 || tq.length() > 1e-6,
            "gimbal produced a lateral component / steering moment: f={f:?} tq={tq:?}"
        );
        assert!(
            tq.length() > 1e-3,
            "off-axis gimballed thrust yields a moment"
        );
        // Zero-limit engine ignores gimbal (pure axial thrust, no moment about the line).
        let mut p2 = single_engine(1000.0, 10.0, 3000.0);
        p2.engines[0].mount = DVec3::new(-3.0, 0.0, 0.0); // on the thrust line
        p2.commands[0].throttle = 1.0;
        p2.commands[0].gimbal = DVec2::new(0.1, 0.0);
        let (_f2, tq2) = p2.thrust_step(DQuat::IDENTITY, DVec3::ZERO, 0.01);
        assert!(
            tq2.length() < 1e-9,
            "no gimbal → thrust on the line → no moment"
        );
    }

    #[test]
    fn command_sets_throttle_and_clamped_gimbal() {
        let mut p = single_engine(1000.0, 10.0, 3000.0);
        p.engines[0].max_gimbal = 0.05;
        assert!(p.apply_command(&Command::SetThrottle(1.5))); // clamps to 1
        assert_eq!(p.commands[0].throttle, 1.0);
        assert!(p.apply_command(&Command::SetGimbal(DVec2::new(1.0, 0.0)))); // clamps to 0.05
        assert!((p.commands[0].gimbal.length() - 0.05).abs() < 1e-9);
        // A non-propulsion command is ignored.
        assert!(!p.apply_command(&Command::SetPaused(true)));
    }

    #[test]
    fn propulsion_round_trips_through_serde() {
        let mut p = single_engine(1000.0, 10.0, 3000.0);
        p.commands[0].throttle = 0.7;
        let json = serde_json::to_string(&p).unwrap();
        let back: Propulsion = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }
}
