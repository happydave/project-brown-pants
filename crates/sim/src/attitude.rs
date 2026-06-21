//! Attitude control — RCS, reaction wheels, and SAS (WI 533).
//!
//! Actuators that steer a craft, consistent with the device/resource model:
//!
//! - **Reaction wheels** ([`ReactionWheels`]) produce torque up to a per-axis
//!   maximum, storing the *opposite* angular momentum (the body and the wheels
//!   exchange momentum — total is conserved); a wheel at its stored-momentum limit
//!   **saturates** and provides no further torque that way.
//! - **RCS** ([`Rcs`]) is an external jet: it produces torque drawing propellant
//!   from a [`ResourceGraph`] reservoir (WI 531/507) and flames out when empty.
//!
//! A **SAS** control law ([`Sas`]) — kill-rotation, hold-attitude, or point-at-a
//! -vector — and **manual** pitch/yaw/roll intent both feed a PD demand the
//! actuators try to provide; the achieved torque goes through
//! [`ActiveBody::integrate_wrench`]. Everything routes through the command envelope
//! ([`Command::SetSas`]/[`Command::SetAttitude`]) so a player, an autopilot, and the
//! AI steer identically. Headless; per-body-axis actuator authority (a balanced set,
//! not per-thruster geometry). The ECS wiring into a scene is the control items.

use crate::active::ActiveBody;
use crate::command::{Command, SasMode};
use crate::resource::{ReservoirId, ResourceGraph};
use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

/// Manual intent magnitude (per axis, in `[-1, 1]`) above which manual is treated
/// as *claiming* that axis for command arbitration (WI 563).
const MANUAL_DEADZONE: f64 = 1e-6;

/// Command arbitration (WI 563): resolve a per-body-axis attitude torque from
/// competing control sources, each `(torque, per-axis claim)`, listed
/// **highest-priority first**. The first source claiming an axis **owns** it; lower
/// sources are ignored on that axis. Demands are never summed across sources, so two
/// controllers cannot fight over the same axis. A future controller (canned
/// autopilot, player program, per-limb gait) is just another source at its priority
/// — the resolver is unchanged. Pure and order-deterministic.
fn arbitrate_axes(sources: &[(DVec3, [bool; 3])]) -> DVec3 {
    let mut out = [0.0_f64; 3];
    let mut owned = [false; 3];
    for (torque, claims) in sources {
        let t = [torque.x, torque.y, torque.z];
        for axis in 0..3 {
            if !owned[axis] && claims[axis] {
                out[axis] = t[axis];
                owned[axis] = true;
            }
        }
    }
    DVec3::new(out[0], out[1], out[2])
}

/// Reaction-wheel assembly: per-body-axis torque, saturating at a stored-momentum
/// limit. Storing the opposite momentum is how the body is torqued without an
/// external jet — until the wheels are full.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReactionWheels {
    /// Maximum torque per body axis, N·m.
    pub max_torque: f64,
    /// Stored-momentum saturation limit per body axis, N·m·s.
    pub max_momentum: f64,
    /// Current stored angular momentum (body frame), N·m·s.
    pub stored: DVec3,
}

impl ReactionWheels {
    /// A fresh assembly (no stored momentum).
    pub fn new(max_torque: f64, max_momentum: f64) -> Self {
        Self {
            max_torque,
            max_momentum,
            stored: DVec3::ZERO,
        }
    }

    /// Body torque the wheels provide toward `want` this step, limited per axis by
    /// the torque maximum and by stored-momentum headroom (the wheel spins up
    /// opposite: `stored -= torque·dt`, clamped to `±max_momentum` → saturation).
    fn provide(&mut self, want: DVec3, dt: f64) -> DVec3 {
        let t = DVec3::new(
            self.axis(want.x, self.stored.x, dt),
            self.axis(want.y, self.stored.y, dt),
            self.axis(want.z, self.stored.z, dt),
        );
        self.stored -= t * dt;
        t
    }

    fn axis(&self, want: f64, stored: f64, dt: f64) -> f64 {
        let mut t = want.clamp(-self.max_torque, self.max_torque);
        let new_stored = stored - t * dt;
        if dt > 0.0 {
            if new_stored > self.max_momentum {
                t = (stored - self.max_momentum) / dt;
            } else if new_stored < -self.max_momentum {
                t = (stored + self.max_momentum) / dt;
            }
        }
        t
    }
}

/// An RCS thruster set: per-axis torque authority, drawing propellant from a
/// reservoir proportional to the torque-impulse produced. An external torque (it
/// changes the body's total angular momentum); flames out when the tank empties.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rcs {
    /// Propellant reservoir (within the controller's [`ResourceGraph`]).
    pub propellant: ReservoirId,
    /// Maximum torque per body axis, N·m.
    pub max_torque: f64,
    /// Propellant drawn per unit torque-impulse, kg/(N·m·s).
    pub mass_flow_per_torque: f64,
}

/// A craft's attitude actuators: optional reaction wheels and optional RCS.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttitudeControl {
    /// Reaction wheels, if fitted.
    pub wheels: Option<ReactionWheels>,
    /// RCS, if fitted.
    pub rcs: Option<Rcs>,
}

impl AttitudeControl {
    /// Provide as much of the `desired` body-frame control torque as the actuators
    /// can this step: **wheels first** (free, until saturated), then **RCS**
    /// (propellant, for the remainder). Returns the actual body torque applied;
    /// updates wheel momentum and draws propellant.
    pub fn actuate(&mut self, desired: DVec3, graph: &mut ResourceGraph, dt: f64) -> DVec3 {
        let mut applied = DVec3::ZERO;
        let mut remaining = desired;

        if let Some(w) = &mut self.wheels {
            let t = w.provide(remaining, dt);
            applied += t;
            remaining -= t;
        }

        if let Some(rcs) = &self.rcs {
            if let Some(res) = graph.reservoirs.get_mut(rcs.propellant.0) {
                // Per-axis torque authority.
                let want = DVec3::new(
                    remaining.x.clamp(-rcs.max_torque, rcs.max_torque),
                    remaining.y.clamp(-rcs.max_torque, rcs.max_torque),
                    remaining.z.clamp(-rcs.max_torque, rcs.max_torque),
                );
                // Propellant needed for this torque-impulse, rationed by availability.
                let need = rcs.mass_flow_per_torque * want.length() * dt;
                let scale = if need > res.amount && need > 0.0 {
                    res.amount / need
                } else {
                    1.0
                };
                res.amount = (res.amount - need * scale).max(0.0);
                applied += want * scale;
            }
        }

        applied
    }
}

/// The stability-assist controller: a PD law over orientation/rate error producing
/// a desired body-frame control torque. Holds the captured target attitude for
/// `Hold`; aligns `nose` (a body axis) to the target for `Point`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sas {
    /// The current mode.
    pub mode: SasMode,
    /// The attitude held in `Hold` (captured on engage).
    pub target: DQuat,
    /// The body axis treated as the craft's nose (for `Point`).
    pub nose: DVec3,
    /// Proportional gain (attitude error → torque).
    pub kp: f64,
    /// Derivative gain (angular rate → damping torque).
    pub kd: f64,
}

impl Default for Sas {
    fn default() -> Self {
        Self {
            mode: SasMode::Off,
            target: DQuat::IDENTITY,
            nose: DVec3::Y,
            kp: 8.0,
            kd: 6.0,
        }
    }
}

impl Sas {
    /// Sets the mode, capturing `current` as the held attitude when entering `Hold`.
    pub fn set_mode(&mut self, mode: SasMode, current: DQuat) {
        if mode == SasMode::Hold {
            self.target = current;
        }
        self.mode = mode;
    }

    /// Desired **body-frame** control torque from the PD law, given the body
    /// `orientation` and the world-frame angular velocity `omega`.
    pub fn desired_torque(&self, orientation: DQuat, omega: DVec3) -> DVec3 {
        let omega_body = orientation.conjugate() * omega;
        match self.mode {
            SasMode::Off => DVec3::ZERO,
            SasMode::KillRotation => -self.kd * omega_body,
            SasMode::Hold => {
                // Body-frame rotation from the current attitude to the target.
                let err = rotation_vector(orientation.conjugate() * self.target);
                self.kp * err - self.kd * omega_body
            }
            SasMode::Point(dir) => {
                let nose_world = (orientation * self.nose).normalize_or_zero();
                let target = dir.normalize_or_zero();
                if nose_world == DVec3::ZERO || target == DVec3::ZERO {
                    return -self.kd * omega_body;
                }
                // World error rotation aligning the nose to the target, in the body frame.
                let err_world = rotation_vector(DQuat::from_rotation_arc(nose_world, target));
                let err_body = orientation.conjugate() * err_world;
                self.kp * err_body - self.kd * omega_body
            }
        }
    }
}

/// A complete attitude pilot: SAS + manual intent + the actuators. Bundles the
/// control law and the hardware so one `step` drives a craft; ECS-ready.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttitudePilot {
    /// The stability-assist controller.
    pub sas: Sas,
    /// Manual pitch/yaw/roll intent, each in `[-1, 1]` (scaled by `authority`).
    pub manual: DVec3,
    /// Manual full-deflection torque authority, N·m.
    pub authority: f64,
    /// The actuators.
    pub actuators: AttitudeControl,
    /// SAS hold-target re-capture policy (WI 564): when `true`, releasing a manual
    /// nudge re-captures the current attitude as the new `Hold` target (the nudge
    /// sticks); when `false`, SAS returns to the prior target. Defaults to `true`.
    #[serde(default = "default_recapture")]
    pub recapture_on_release: bool,
}

/// Default for [`AttitudePilot::recapture_on_release`] (so pre-564 saves load).
fn default_recapture() -> bool {
    true
}

impl AttitudePilot {
    /// Computes the **world-frame** control torque for this step: assembles the
    /// SAS-plus-manual demand, actuates it (wheels then RCS, updating wheel momentum
    /// and drawing propellant), and returns the achieved torque **without
    /// integrating**. This lets a unified flight step (WI 534) sum it with the other
    /// forces and integrate once. `step` is this plus the integration.
    pub fn control_torque(
        &mut self,
        body: &ActiveBody,
        graph: &mut ResourceGraph,
        dt: f64,
    ) -> DVec3 {
        self.control_torque_gated(body, graph, dt, true, true)
    }

    /// As [`control_torque`](Self::control_torque), but with the control authority
    /// **gated by the craft's resolved tier** (WI 562): `manual` enables the manual
    /// pitch/yaw/roll demand, `stabilization` enables the SAS demand. An uncontrolled
    /// craft passes `(false, false)` → zero demand → no actuation; a Direct craft
    /// passes `(true, false)` → manual only, no stabilization.
    pub fn control_torque_gated(
        &mut self,
        body: &ActiveBody,
        graph: &mut ResourceGraph,
        dt: f64,
        manual: bool,
        stabilization: bool,
    ) -> DVec3 {
        let omega = body.angular_velocity();

        // Command arbitration (WI 563): each attitude axis is owned by a single
        // controller per tick, highest priority first — manual overrides SAS on the
        // axes it claims; SAS owns the rest. Demands are *not* summed.
        let man_torque = self.manual * self.authority;
        let man_claims = if manual {
            [
                self.manual.x.abs() > MANUAL_DEADZONE,
                self.manual.y.abs() > MANUAL_DEADZONE,
                self.manual.z.abs() > MANUAL_DEADZONE,
            ]
        } else {
            [false; 3]
        };
        // The SAS PD law yields a desired angular *acceleration*; scale by the body
        // inertia so authority is inertia-aware (a heavy craft gets proportionally
        // more torque, so the same gains slew any mass). Unit-inertia → unchanged.
        let sas_torque = body.inertia * self.sas.desired_torque(body.orientation, omega);
        let sas_claims = [stabilization; 3];

        // Sources in priority order (manual over SAS).
        let desired = arbitrate_axes(&[(man_torque, man_claims), (sas_torque, sas_claims)]);
        let applied_body = self.actuators.actuate(desired, graph, dt);
        body.orientation * applied_body
    }

    /// Advance one step: apply the [`control_torque`](Self::control_torque) to the
    /// body through `integrate_wrench`.
    pub fn step(&mut self, body: &mut ActiveBody, graph: &mut ResourceGraph, dt: f64) {
        let torque_world = self.control_torque(body, graph, dt);
        body.integrate_wrench(DVec3::ZERO, torque_world, dt);
    }

    /// Applies an attitude [`Command`] (`SetAttitude`/`SetSas`/`SetSasRecapture`);
    /// other commands are ignored (returns `false`). `current` is the craft's
    /// orientation, captured as the hold target when engaging `Hold` and (per the
    /// re-capture policy, WI 564) when a manual nudge releases.
    pub fn apply_command(&mut self, cmd: &Command, current: DQuat) -> bool {
        match *cmd {
            Command::SetAttitude(intent) => {
                let clamped = intent.clamp(DVec3::splat(-1.0), DVec3::splat(1.0));
                let was_active = self.manual != DVec3::ZERO;
                self.manual = clamped;
                // Re-capture the hold target when a manual nudge fully releases, so the
                // nudge sticks instead of snapping back to the old target (WI 564). Only
                // meaningful for `Hold`.
                if self.recapture_on_release
                    && was_active
                    && clamped == DVec3::ZERO
                    && self.sas.mode == SasMode::Hold
                {
                    self.sas.target = current;
                }
                true
            }
            Command::SetSas(mode) => {
                self.sas.set_mode(mode, current);
                true
            }
            Command::SetSasRecapture(b) => {
                self.recapture_on_release = b;
                true
            }
            Command::SetSasGains(kp, kd) => {
                // Clamp to a sane non-negative range so a tuned controller cannot
                // produce divergent/NaN torque (WI 566).
                self.sas.kp = kp.clamp(0.0, 1_000.0);
                self.sas.kd = kd.clamp(0.0, 1_000.0);
                true
            }
            _ => false,
        }
    }
}

/// The shortest-rotation axis-angle vector (`axis · angle`) of a quaternion.
fn rotation_vector(q: DQuat) -> DVec3 {
    let q = if q.w < 0.0 { -q } else { q }; // shortest arc
    let (axis, angle) = q.to_axis_angle();
    if angle.abs() < 1e-12 {
        DVec3::ZERO
    } else {
        axis * angle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{Reservoir, ResourceType};
    use glam::DMat3;

    const PROP: ResourceType = ResourceType(0);

    fn body(inertia: f64) -> ActiveBody {
        ActiveBody::new(
            DVec3::ZERO,
            DVec3::ZERO,
            1.0,
            DMat3::from_diagonal(DVec3::splat(inertia)),
        )
    }

    fn wheels_only(max_torque: f64, max_momentum: f64) -> AttitudePilot {
        AttitudePilot {
            sas: Sas::default(),
            manual: DVec3::ZERO,
            authority: 5.0,
            actuators: AttitudeControl {
                wheels: Some(ReactionWheels::new(max_torque, max_momentum)),
                rcs: None,
            },
            recapture_on_release: true,
        }
    }

    fn empty_graph() -> ResourceGraph {
        ResourceGraph::default()
    }

    // --- Tier-gated control authority (WI 562) ---

    #[test]
    fn gated_torque_suppresses_manual_when_disallowed() {
        let mut pilot = wheels_only(100.0, 1e9);
        pilot.manual = DVec3::new(1.0, 0.0, 0.0); // full pitch demand
        let b = body(1.0);
        let mut g = empty_graph();
        let allowed = pilot.control_torque_gated(&b, &mut g, 0.1, true, false);
        let denied = pilot.control_torque_gated(&b, &mut g, 0.1, false, false);
        assert!(allowed.length() > 1e-6, "manual allowed produces torque");
        assert!(
            denied.length() < 1e-9,
            "manual disallowed produces no torque"
        );
    }

    #[test]
    fn gated_torque_suppresses_sas_when_disallowed() {
        // Direct (no stabilization): SAS demand is suppressed even when engaged.
        let mut pilot = wheels_only(100.0, 1e9);
        pilot
            .sas
            .set_mode(crate::command::SasMode::Hold, DQuat::IDENTITY);
        let mut b = body(1.0);
        b.orientation = DQuat::from_rotation_x(0.3); // error vs the hold target
        let mut g = empty_graph();
        let with_sas = pilot.control_torque_gated(&b, &mut g, 0.1, false, true);
        let no_sas = pilot.control_torque_gated(&b, &mut g, 0.1, false, false);
        assert!(with_sas.length() > 1e-6, "SAS allowed corrects the error");
        assert!(
            no_sas.length() < 1e-9,
            "Direct tier: no stabilization torque"
        );
    }

    // --- Command arbitration (WI 563) ---

    #[test]
    fn arbitrate_axes_single_owner_by_priority() {
        // Manual (priority 1) claims x; SAS (priority 2) claims all. Result: manual
        // owns x (5, not SAS's 1, not the sum 6); SAS owns y, z.
        let manual = (DVec3::new(5.0, 0.0, 0.0), [true, false, false]);
        let sas = (DVec3::new(1.0, 2.0, 3.0), [true, true, true]);
        let out = arbitrate_axes(&[manual, sas]);
        assert_eq!(out, DVec3::new(5.0, 2.0, 3.0));
    }

    #[test]
    fn arbitrate_axes_priority_is_positional_and_deterministic() {
        let a = (DVec3::new(5.0, 0.0, 0.0), [true, false, false]);
        let b = (DVec3::new(1.0, 2.0, 3.0), [true, true, true]);
        // Order defines priority: first claimant wins x.
        assert_eq!(arbitrate_axes(&[a, b]).x, 5.0);
        assert_eq!(arbitrate_axes(&[b, a]).x, 1.0);
        // Same inputs → same output (determinism).
        assert_eq!(arbitrate_axes(&[a, b]), arbitrate_axes(&[a, b]));
        // No claimant on an axis → zero (no actuation).
        let none = (DVec3::new(9.0, 9.0, 9.0), [false, false, false]);
        assert_eq!(arbitrate_axes(&[none]), DVec3::ZERO);
    }

    #[test]
    fn manual_overrides_sas_per_axis_through_control_torque() {
        // KillRotation SAS on a body spinning about all axes demands torque on each;
        // a manual pitch (x) demand overrides SAS on x while SAS keeps yaw/roll.
        let mut pilot = wheels_only(1e6, 1e9); // huge limits → applied ≈ desired
        pilot
            .sas
            .set_mode(crate::command::SasMode::KillRotation, DQuat::IDENTITY);
        let b = body(1.0).with_angular_velocity(DVec3::new(0.5, 0.5, 0.5)); // identity orientation
        let mut g = empty_graph();

        let mut p_sas = pilot;
        let sas_only = p_sas.control_torque_gated(&b, &mut g, 0.01, false, true);
        let mut p_man = pilot;
        p_man.manual = DVec3::new(1.0, 0.0, 0.0);
        let man_only = p_man.control_torque_gated(&b, &mut g, 0.01, true, false);
        let mut p_both = pilot;
        p_both.manual = DVec3::new(1.0, 0.0, 0.0);
        let both = p_both.control_torque_gated(&b, &mut g, 0.01, true, true);

        assert!((both.x - man_only.x).abs() < 1e-9, "manual owns pitch");
        assert!(
            (both.y - sas_only.y).abs() < 1e-9 && (both.z - sas_only.z).abs() < 1e-9,
            "SAS owns yaw/roll"
        );
        assert!(
            (both.x - (man_only.x + sas_only.x)).abs() > 1e-9,
            "arbitration, not summation"
        );
    }

    // --- SAS hold-target re-capture (WI 564) ---

    #[test]
    fn recapture_on_release_moves_hold_target_to_release_attitude() {
        let mut pilot = wheels_only(100.0, 1e9);
        pilot.recapture_on_release = true;
        let t0 = DQuat::IDENTITY;
        pilot.apply_command(&Command::SetSas(SasMode::Hold), t0);
        assert_eq!(pilot.sas.target, t0);
        // Nudge (manual active), then release at a new attitude t1.
        pilot.apply_command(&Command::SetAttitude(DVec3::new(1.0, 0.0, 0.0)), t0);
        let t1 = DQuat::from_rotation_x(0.5);
        pilot.apply_command(&Command::SetAttitude(DVec3::ZERO), t1);
        assert_eq!(pilot.sas.target, t1, "re-capture: the nudge sticks");
    }

    #[test]
    fn return_to_target_keeps_hold_target_on_release() {
        let mut pilot = wheels_only(100.0, 1e9);
        pilot.recapture_on_release = false;
        let t0 = DQuat::IDENTITY;
        pilot.apply_command(&Command::SetSas(SasMode::Hold), t0);
        pilot.apply_command(&Command::SetAttitude(DVec3::new(1.0, 0.0, 0.0)), t0);
        pilot.apply_command(
            &Command::SetAttitude(DVec3::ZERO),
            DQuat::from_rotation_x(0.5),
        );
        assert_eq!(
            pilot.sas.target, t0,
            "return-to-target: original target kept"
        );
    }

    #[test]
    fn recapture_only_applies_to_hold() {
        let mut pilot = wheels_only(100.0, 1e9);
        pilot.recapture_on_release = true;
        pilot.apply_command(&Command::SetSas(SasMode::KillRotation), DQuat::IDENTITY);
        let before = pilot.sas.target;
        pilot.apply_command(
            &Command::SetAttitude(DVec3::new(1.0, 0.0, 0.0)),
            DQuat::IDENTITY,
        );
        pilot.apply_command(
            &Command::SetAttitude(DVec3::ZERO),
            DQuat::from_rotation_x(0.5),
        );
        assert_eq!(
            pilot.sas.target, before,
            "KillRotation has no hold-target re-capture"
        );
    }

    #[test]
    fn sas_gains_tune_and_clamp() {
        let mut pilot = wheels_only(1e6, 1e9);
        assert!(pilot.apply_command(&Command::SetSasGains(50.0, 30.0), DQuat::IDENTITY));
        assert_eq!((pilot.sas.kp, pilot.sas.kd), (50.0, 30.0));
        // Negative / huge inputs clamp to a sane non-negative range.
        pilot.apply_command(&Command::SetSasGains(-5.0, 1e9), DQuat::IDENTITY);
        assert_eq!(pilot.sas.kp, 0.0);
        assert!(pilot.sas.kd <= 1_000.0);
    }

    #[test]
    fn higher_kp_gives_stronger_correction() {
        // Same attitude error, two kp values → proportionally different SAS torque
        // (the tuning is real, not cosmetic). Unit inertia, kd=0.
        let mut b = body(1.0);
        b.orientation = DQuat::from_rotation_x(0.2);
        let mut g = empty_graph();
        let mut soft = wheels_only(1e6, 1e9);
        soft.sas
            .set_mode(crate::command::SasMode::Hold, DQuat::IDENTITY);
        soft.sas.kp = 5.0;
        soft.sas.kd = 0.0;
        let mut stiff = soft;
        stiff.sas.kp = 50.0;
        let t_soft = soft
            .control_torque_gated(&b, &mut g, 0.01, false, true)
            .length();
        let t_stiff = stiff
            .control_torque_gated(&b, &mut g, 0.01, false, true)
            .length();
        assert!(
            t_stiff > t_soft * 5.0,
            "higher kp → proportionally stronger correction"
        );
    }

    #[test]
    fn set_sas_recapture_command_toggles_policy() {
        let mut pilot = wheels_only(100.0, 1e9);
        pilot.recapture_on_release = true;
        assert!(pilot.apply_command(&Command::SetSasRecapture(false), DQuat::IDENTITY));
        assert!(!pilot.recapture_on_release);
    }

    // --- Actuators ---

    #[test]
    fn reaction_wheel_provides_torque_then_saturates() {
        let mut w = ReactionWheels::new(2.0, 1.0); // 1 N·m·s of storage
        let dt = 0.1;
        let mut total_impulse = 0.0;
        let mut last = 0.0;
        for _ in 0..100 {
            let t = w.provide(DVec3::new(5.0, 0.0, 0.0), dt); // demand exceeds max_torque
            total_impulse += t.x * dt;
            last = t.x;
        }
        // Torque was capped at max_torque early, then fell to zero at saturation.
        assert!(last.abs() < 1e-9, "saturated wheel provides no more torque");
        // Stored momentum saturated at the limit; impulse delivered ≈ max_momentum.
        assert!(
            (w.stored.x.abs() - 1.0).abs() < 1e-9,
            "stored {}",
            w.stored.x
        );
        assert!(
            (total_impulse.abs() - 1.0).abs() < 1e-6,
            "impulse {total_impulse}"
        );
    }

    #[test]
    fn rcs_consumes_propellant_and_flames_out() {
        let mut a = AttitudeControl {
            wheels: None,
            rcs: Some(Rcs {
                propellant: ReservoirId(0),
                max_torque: 3.0,
                mass_flow_per_torque: 0.01,
            }),
        };
        let mut g = ResourceGraph {
            reservoirs: vec![Reservoir::new(PROP, 1.0, 1.0)],
            ..Default::default()
        };
        let dt = 0.1;
        let mut produced_after_empty = None;
        for _ in 0..500 {
            let t = a.actuate(DVec3::new(3.0, 0.0, 0.0), &mut g, dt);
            if g.reservoirs[0].amount <= 0.0 && produced_after_empty.is_none() {
                // next step after empty:
                let t2 = a.actuate(DVec3::new(3.0, 0.0, 0.0), &mut g, dt);
                produced_after_empty = Some(t2.x);
            }
            let _ = t;
        }
        assert!(g.reservoirs[0].amount >= 0.0, "never negative");
        assert!(g.reservoirs[0].amount < 1e-9, "propellant burned");
        assert_eq!(produced_after_empty, Some(0.0), "flame-out → no torque");
    }

    // --- SAS convergence ---

    #[test]
    fn kill_rotation_stops_a_tumble() {
        let mut body = body(2.0).with_angular_velocity(DVec3::new(0.4, -0.3, 0.5));
        let mut pilot = wheels_only(20.0, 1e6);
        let mut g = empty_graph();
        pilot.sas.set_mode(SasMode::KillRotation, body.orientation);
        for _ in 0..4_000 {
            pilot.step(&mut body, &mut g, 0.01);
        }
        assert!(
            body.angular_velocity().length() < 1e-3,
            "tumble killed: {}",
            body.angular_velocity().length()
        );
    }

    #[test]
    fn hold_attitude_returns_after_a_perturbation() {
        let mut body = body(2.0);
        let mut pilot = wheels_only(30.0, 1e6);
        let mut g = empty_graph();
        let target = body.orientation;
        pilot.sas.set_mode(SasMode::Hold, target);
        // Perturb: a manual kick for a moment, then release to SAS.
        pilot.manual = DVec3::new(1.0, 0.0, 0.0);
        for _ in 0..50 {
            pilot.step(&mut body, &mut g, 0.01);
        }
        pilot.manual = DVec3::ZERO;
        for _ in 0..8_000 {
            pilot.step(&mut body, &mut g, 0.01);
        }
        let err = (body.orientation.conjugate() * target).w.abs(); // 1 ⇒ aligned
        assert!(err > 0.9999, "returned to held attitude (w = {err})");
        assert!(body.angular_velocity().length() < 1e-3, "and at rest");
    }

    #[test]
    fn point_aligns_the_nose_with_a_target() {
        let mut body = body(2.0);
        let mut pilot = wheels_only(30.0, 1e6);
        let mut g = empty_graph();
        let target = DVec3::new(1.0, 0.0, 0.0); // want the +Y nose to point +X
        pilot.sas.set_mode(SasMode::Point(target), body.orientation);
        for _ in 0..8_000 {
            pilot.step(&mut body, &mut g, 0.01);
        }
        let nose = (body.orientation * pilot.sas.nose).normalize();
        assert!(
            nose.dot(target.normalize()) > 0.9999,
            "nose aligned with target: dot {}",
            nose.dot(target.normalize())
        );
        assert!(body.angular_velocity().length() < 1e-3);
    }

    #[test]
    fn manual_intent_rotates_the_craft() {
        let mut body = body(2.0);
        let mut pilot = wheels_only(20.0, 1e6); // SAS off by default
        let mut g = empty_graph();
        pilot.manual = DVec3::new(1.0, 0.0, 0.0);
        for _ in 0..100 {
            pilot.step(&mut body, &mut g, 0.01);
        }
        assert!(
            body.orientation.dot(DQuat::IDENTITY).abs() < 0.9999,
            "manual intent rotated the craft"
        );
        assert!(body.angular_velocity().length() > 0.0);
    }

    #[test]
    fn command_sets_sas_and_attitude() {
        let mut pilot = wheels_only(10.0, 1e6);
        let cur = DQuat::from_rotation_x(0.3);
        assert!(pilot.apply_command(&Command::SetSas(SasMode::Hold), cur));
        assert_eq!(pilot.sas.mode, SasMode::Hold);
        assert_eq!(pilot.sas.target, cur, "Hold captured the current attitude");
        assert!(pilot.apply_command(&Command::SetAttitude(DVec3::new(2.0, 0.0, -2.0)), cur));
        assert_eq!(
            pilot.manual,
            DVec3::new(1.0, 0.0, -1.0),
            "intent clamped to [-1,1]"
        );
        assert!(
            !pilot.apply_command(&Command::SetPaused(true), cur),
            "ignores others"
        );
    }

    #[test]
    fn pilot_round_trips_through_serde() {
        let pilot = wheels_only(10.0, 5.0);
        let json = serde_json::to_string(&pilot).unwrap();
        let back: AttitudePilot = serde_json::from_str(&json).unwrap();
        assert_eq!(pilot, back);
    }
}
