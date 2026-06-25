//! Wheels & ground contact — the rover (WI 506).
//!
//! The one genuinely new primitive: a frictional contact force. A [`Rover`] is an
//! active rigid body ([`ActiveBody`], WI 515) carrying [`Wheel`]s. Each wheel has a
//! spring-damper suspension and a simplified slip-based tire (longitudinal from
//! slip ratio, lateral from slip angle, coupled by a friction ellipse, scaled by
//! `μ·N` with `μ`/rolling-resistance from the surface material, WI 497). Each wheel
//! carries a spin DOF so the slip ratio is physical.
//!
//! Stability — not the tire law — is the hard part. The contact surface is the
//! **analytic** terrain (WI 506 `Terrain`), queried in f64, so it never pops under
//! rebasing/LOD; the body is integrated semi-implicitly (`integrate_wrench`) at a
//! capped sub-stepped fixed timestep. The result is validated by the contact-jitter
//! / no-launch test (the design's kraken detector as an automated bound).

use crate::active::ActiveBody;
use crate::powertrain::{build_powertrain, RoverPowertrain};
use crate::terrain::Terrain;
use crate::voxel::{DeviceKind, PartKind, RimSpec, SuspensionSpec, TireSpec, VoxelCraft};
use glam::{DMat3, DQuat, DVec3};

/// Longitudinal slip stiffness (shape of the slip-ratio → force curve).
const C_LONG: f64 = 5.0;
/// Lateral slip stiffness (shape of the slip-angle → force curve).
const C_LAT: f64 = 4.0;
const EPS: f64 = 1e-3;
/// Aerodynamic drag coefficient (N·s²/m²): gives the rover a finite (but high)
/// top speed of roughly 100 m/s.
const DRAG: f64 = 0.55;
/// Baseline angular-drag **rate** (1/s) under *smooth* contact (WI 611, rate-ified WI 630): a gentle
/// restoring term that keeps nominal driving tidy without preventing an intentional flip. It is scaled
/// by the body's mean moment of inertia so the damping is the same *rate* on a light editor build as on
/// a heavy one — the old absolute coefficient pinned light rovers upright (a "magical force") while
/// being negligible on heavy ones.
const BASELINE_ANGULAR_DRAG_RATE: f64 = 0.022;
/// Jitter-scaled angular-drag rate (1/s) (WI 611): added in proportion to the contact-jitter ratio
/// (per-substep normal-force change ÷ weight), so the numerical "kraken" buzz — which oscillates the
/// normal force every sub-step — is damped hard, while a genuine, smooth-contact rotation (an
/// intentional rollover) passes nearly undamped. Inertia-scaled like the baseline (WI 630).
const JITTER_ANGULAR_DRAG_RATE: f64 = 0.85;
/// Cap on the jitter ratio feeding `JITTER_ANGULAR_DRAG`, so a single hard landing spike can't make
/// the damping unboundedly large.
const JITTER_RATIO_CAP: f64 = 3.0;

/// Quarter-car unsprung mass (WI 631a). Each assembled wheel gains its own vertical degree of freedom
/// and mass (the unsprung mass: rim + tire). The contact normal force then comes from a two-mass
/// system — a suspension spring-damper between the chassis and the axle, and a stiff tire spring
/// between the axle and the ground — instead of the WI 630 single series spring (which was the
/// rigid-unsprung-mass limit of this model). This gives real ride dynamics (wheel hop, bump
/// absorption, dynamic load transfer) and is the one integrator-level change of Phase 2.
///
/// Tire-spring damping ratio: near-critical, so wheel hop settles in roughly one oscillation and the
/// dynamic ground load stays smooth (a ringing tire load would feed grip noise into the body and
/// excite tumbling over rough ground). Real tires are lighter-damped, but the strut+tire pair here is
/// the whole vertical damping path, so it carries the suspension's job too.
const TIRE_DAMPING_RATIO: f64 = 0.7;
/// Shared stability/feel ceiling on the axle springs, expressed as a maximum `ω·dt`
/// (`ω = √(k/m_unsprung)`) (WI 631a). The axle's own DOF is integrated implicitly, but its **coupling
/// to the chassis** (the suspension reaction) and the body integrator are explicit, so a spring on the
/// light unsprung mass still rings the chassis once `ω·dt` approaches ~1. Capping every axle spring
/// (tire and suspension) at this `ω·dt` — i.e. `k ≤ (ω·dt_max / dt)² · m_unsprung`, a mass-relative
/// rate (the project's scale-relative-physics convention) — keeps the contact smooth at **any** build
/// scale. It is what makes a **no-suspension ("rigid") strut** the *stiffest stable* strut: still far
/// above any tire (so the wheel rides on the tire), but smooth rather than a chassis-jolting 1e9.
const TIRE_OMEGA_DT_MAX: f64 = 0.3;
/// Minimum unsprung mass (kg) for the quarter-car path: below this a corner uses the legacy
/// series-spring contact, so a degenerate (near-massless) station never divides by ~zero or rings.
const MIN_UNSPRUNG_MASS: f64 = 0.02;
/// Minimum tire-to-suspension stiffness ratio (WI 631b ride pass). In a real vehicle the tire is
/// several times **stiffer** than the suspension spring, so the body's heave rides on the (well-damped)
/// suspension rather than bobbing on a soft tire. The suspension rate is auto-sized to the build mass,
/// so on a heavy rover it can exceed an authored (absolute) tire rate — inverting that ordering and
/// leaving the heave-on-tire mode badly under-damped (a slow wallow after a landing). Flooring the tire
/// at this multiple of the suspension rate restores the real ordering at any build scale, while a tire
/// already stiffer than the floor keeps its authored feel (the soft/stiff preset distinction on lighter
/// builds where the suspension rate is well below the authored tire rate).
const TIRE_TO_SUSPENSION_RATIO: f64 = 4.0;
/// Tire load-sensitivity cap (WI 631a): the grip a corner can generate saturates at this multiple of
/// its **static** load. The quarter-car's dynamic ground load spikes on bump/landing impacts; without a
/// cap, those spikes multiply through the friction ellipse (especially with a grippy tire) into force
/// spikes that tumble the rover over rough ground. Real tires have exactly this "load sensitivity"
/// (grip per newton falls at high load), so the cap is physical, not just a kraken guard. It bounds the
/// grip used in the friction limit only; the true dynamic load still drives the axle's vertical motion
/// and the ride feel, and normal cornering/braking load transfer (well under the cap) is unaffected.
const GRIP_LOAD_FACTOR: f64 = 3.5;
/// Cap on each corner's unsprung mass as a fraction of the craft's even per-corner share (WI 631a).
/// Only a fraction of a wheel (the rim/tire contact ring, not the hub/brake/mount) is truly unsprung;
/// real vehicles keep unsprung mass well under the sprung. Capping it keeps the sprung chassis body
/// dominant — so its rotational inertia stays high enough that hard braking/cornering does not pitch
/// or roll it over (a heavy-wheel build would otherwise flip), and the CoM shift off the WI 630 body
/// stays small (as the design intends) — while still giving every corner a real vertical DOF for hop.
/// The rim+tire's **full** mass still sets the wheel's spin inertia; only the *vertical* mass is capped.
const UNSPRUNG_FRACTION: f64 = 0.15;

/// Wheel-mount shear model (WI 618). A mount is **rated for an impact speed**: a wheel shears when the
/// closing speed of an obstacle impact (scaled by how directly the wheel faces it) exceeds its rated
/// speed. Keying off impact *speed* — not sustained contact force — means leaning on a wall (closing
/// speed ≈ 0) never shears, only genuine hits do; and it is mass-independent (a given build shears at a
/// characteristic speed regardless of how heavy it is). The rated speed scales with the chassis
/// material at the wheel: `BASE × sqrt(strength / REFERENCE)`, floored.
const BASE_SHEAR_SPEED: f64 = 8.0;
/// Reference material strength (Pa) at which a mount is rated for [`BASE_SHEAR_SPEED`] (≈ steel).
const REF_SHEAR_STRENGTH: f64 = 5.0e8;
/// Floor on a wheel's rated impact speed (m/s), so a flimsy-material chassis still survives a tap.
const MIN_SHEAR_SPEED: f64 = 3.0;
/// The share of an impact a far-side wheel still feels (WI 618). A near wheel (facing the obstacle)
/// feels ~the full closing speed and shears first; a far wheel feels only this baseline, so it shears
/// only under a much faster hit — making impact damage speed-graded.
const SHEAR_SIDE_BIAS: f64 = 0.25;

/// The mount's rated impact speed (m/s) for a chassis material of tensile `strength` (Pa) (WI 618).
fn rated_shear_speed(strength: f64) -> f64 {
    (BASE_SHEAR_SPEED * (strength / REF_SHEAR_STRENGTH).sqrt()).max(MIN_SHEAR_SPEED)
}

/// Component failure ladder (WI 631b). Each wheel station fails its components in severity order
/// *below* the catastrophic mount shear (WI 618), so a graded impact damages the soft outer parts
/// first and only a hard hit tears the wheel off. The ratings are expressed as fractions of the
/// wheel's own (chassis-material-sourced, mass-independent) shear speed, so the strict ordering
/// `tire_burst < rim_bend < damper_fail < shear` holds automatically at any build scale. (The full
/// "move shear strength onto the components" of the design is a deliberate partial here — only these
/// new sub-shear ratings are component/shear-derived; the mount shear itself is unchanged.)
const TIRE_BURST_FRACTION: f64 = 0.70;
/// Rim-bend impact rating as a fraction of the mount shear speed (WI 631b).
const RIM_BEND_FRACTION: f64 = 0.82;
/// Damper-failure impact rating as a fraction of the mount shear speed (WI 631b).
const DAMPER_FAIL_FRACTION: f64 = 0.92;
/// Residual grip multiplier of a blown tire (WI 631b): grip collapses to this fraction (running on the
/// bare rim has far less traction).
const TIRE_BLOWN_GRIP: f64 = 0.3;
/// Rolling-resistance multiplier added by a blown tire (WI 631b): a rim drags far more than rubber.
const TIRE_BLOWN_ROLLING: f64 = 4.0;
/// Rolling-resistance multiplier added by a bent rim (WI 631b): a buckled rim scrubs as it rolls.
const RIM_BENT_ROLLING: f64 = 2.0;
/// Steer/camber bias (rad) a bent rim adds to its wheel (WI 631b): a persistent pull / visible wobble.
const RIM_BENT_STEER: f64 = 0.04;
/// Remaining suspension damping fraction after a blown damper (WI 631b): that corner becomes bouncy.
const DAMPER_BLOWN_DAMPING: f64 = 0.1;
/// Tire spring rate (N/m) of a blown tire run on the bare rim (WI 631b): effectively rigid — the
/// step caps it at the sub-step-stable ceiling, so the corner stops absorbing/bouncing and thuds.
const RIGID_TIRE_STIFFNESS: f64 = 1.0e9;
/// Motor's maximum wheel speed (rad/s). Drive torque falls off as the wheel nears
/// it, so flooring the throttle cannot spin the wheels up without bound — a burnout
/// would use all the tyre's grip longitudinally and leave none for cornering,
/// making the rover spin out. ≈ top speed / wheel radius, plus slip margin.
const MAX_WHEEL_SPIN: f64 = 850.0;

/// Stable rover physics sub-step (seconds). Stiff spring-damper wheels coupled
/// through the body's moment arm require this small a step for the explicit
/// (semi-implicit) integration to stay stable — the design's "wheel sub-stepping".
/// 1/1920 s ≈ 32 sub-steps per 60 fps frame; the rover scene sub-steps to it.
pub const SUBSTEP_DT: f64 = 1.0 / 1920.0;

/// A wheel mounted on the rover. Body-frame mount; world +Z is the rover's forward.
#[derive(Clone, Copy, Debug)]
pub struct Wheel {
    /// Mount point in the body frame.
    pub mount: DVec3,
    /// Wheel radius (m).
    pub radius: f64,
    /// Suspension free length (m).
    pub rest_length: f64,
    /// Spring stiffness (N/m).
    pub stiffness: f64,
    /// Suspension damping (N·s/m).
    pub damping: f64,
    /// Maximum normal force (N) — clamps the stiff response to a hard landing.
    pub max_force: f64,
    /// Steering angle about the body up axis (rad).
    pub steer: f64,
    /// Wheel spin (rad/s) — the rolling DOF that makes slip ratio physical.
    pub spin: f64,
    /// Wheel rotational inertia (kg·m²).
    pub wheel_inertia: f64,
    /// Applied drive torque (N·m, throttle).
    pub drive_torque: f64,
    /// Applied brake torque magnitude (N·m).
    pub brake: f64,
    /// Sheared off by a hard impact (WI 618): an inert wheel exerts no contact/suspension force, takes
    /// no drive, and does not spin — the rover behaves as if that corner has no wheel.
    pub inert: bool,
    /// Impact speed (m/s) the mount is rated for before shearing (WI 618): set from the chassis
    /// material at the wheel by [`assemble_rover`] via [`rated_shear_speed`]. A stronger material holds
    /// its wheel through faster hits.
    pub shear_speed: f64,
    /// Tire grip multiplier over the surface material's friction (WI 630): the friction-ellipse limit
    /// is `surface.friction × grip_scale × normal`. 1.0 reproduces the pre-split surface-only grip.
    pub grip_scale: f64,
    /// Longitudinal slip stiffness (WI 630), per-wheel from the tire. Defaults to [`C_LONG`].
    pub slip_long: f64,
    /// Lateral slip stiffness (WI 630), per-wheel from the tire. Defaults to [`C_LAT`].
    pub slip_lat: f64,
    /// Tire compliance / spring rate (N/m), in **series** with [`Wheel::stiffness`] (WI 630): the
    /// effective contact spring is `stiffness · tire_stiffness / (stiffness + tire_stiffness)`. A high
    /// value is effectively rigid (series ≈ suspension); a low value softens the ride. Lets a
    /// rigid-strut wheel ride on tire compliance alone.
    pub tire_stiffness: f64,
    /// Whether the suspension strut is rigid (WI 630): if so, [`assemble_rover`] does not mass-size its
    /// spring (it stays very stiff) and the wheel rides on the tire's compliance.
    pub rigid_suspension: bool,
    /// Unsprung mass (kg) of this corner — rim + tire (WI 631a). Zero ⇒ the legacy series-spring
    /// contact (the rigid-unsprung-mass limit, used by hand-built fixtures); positive ⇒ the
    /// quarter-car (per-wheel vertical DOF). Set by [`assemble_rover`] from the station's rim+tire mass.
    pub unsprung_mass: f64,
    /// Suspension travel (m) before the droop stop (WI 631a): how far the axle may hang below its free
    /// length when airborne. From the suspension component; a rigid strut has zero travel.
    pub suspension_travel: f64,
    /// Axle (wheel-centre) drop below the chassis mount, measured vertically (m) (WI 631a): the
    /// quarter-car's vertical state. Free length at rest; grows as the wheel droops, shrinks as the
    /// suspension compresses. Maintained only on the quarter-car path; ignored when `unsprung_mass` 0.
    pub axle_drop: f64,
    /// Rate of change of [`Wheel::axle_drop`] (m/s) (WI 631a): the axle's vertical velocity **relative
    /// to the chassis mount**. The axle's absolute vertical velocity is `mount_velocity.y - axle_drop_vel`.
    pub axle_drop_vel: f64,
    /// Static ground load (N) this corner carries at rest (WI 631a): the reference for the tire
    /// load-sensitivity grip cap ([`GRIP_LOAD_FACTOR`]). Set by [`assemble_rover`]; zero on the legacy
    /// path (no cap, so hand-built fixtures are unchanged).
    pub static_load: f64,
    /// Rim radius (m) — the run-on-rim target if the tire blows (WI 631b). The tire profile is
    /// `radius − rim_radius`; a blown tire drops [`Wheel::radius`] to this. Set by [`assemble_rover`]
    /// from the rim component; defaults to the full radius (no drop) for hand-built fixtures.
    pub rim_radius: f64,
    /// Rolling-resistance multiplier (WI 631b): 1.0 normally; raised when the tire blows or the rim
    /// bends so a damaged corner drags more.
    pub rolling_scale: f64,
    /// Steer/camber bias (rad) added to [`Wheel::steer`] (WI 631b): 0 normally; a bent rim adds a small
    /// persistent offset (a pull / wobble).
    pub steer_bias: f64,
    /// Tire blown out (WI 631b): grip has collapsed to a residual and the wheel runs on the rim
    /// (reduced [`Wheel::radius`]). Latched.
    pub tire_blown: bool,
    /// Rim bent (WI 631b): added rolling resistance and a steer/camber bias. Latched.
    pub rim_bent: bool,
    /// Suspension damper blown (WI 631b): this corner's [`Wheel::damping`] has been cut, so it is
    /// bouncy. Latched.
    pub damper_blown: bool,
}

impl Wheel {
    /// A wheel with sensible suspension defaults at `mount`.
    pub fn new(mount: DVec3) -> Self {
        Self {
            mount,
            radius: 0.35,
            rest_length: 0.35,
            // Stiffer springs ⇒ less droop, so the wheels lose contact over crests
            // and the rover catches air at speed (rather than the chassis floating
            // while long-travel wheels stay glued to the surface).
            stiffness: 4.5e4,
            damping: 8.0e3,
            max_force: 1.0e6,
            steer: 0.0,
            spin: 0.0,
            wheel_inertia: 8.0,
            drive_torque: 0.0,
            brake: 0.0,
            inert: false,
            shear_speed: BASE_SHEAR_SPEED,
            grip_scale: 1.0,
            slip_long: C_LONG,
            slip_lat: C_LAT,
            // Effectively rigid tire by default, so the series spring equals the suspension spring
            // (no behaviour change for the hand-built wheels used by the stability tests).
            tire_stiffness: 1.0e9,
            rigid_suspension: false,
            // Zero unsprung mass ⇒ the legacy series-spring path, so hand-built `Wheel::new` fixtures
            // (and the stability suite built on them) keep the exact WI 630 behaviour (WI 631a).
            unsprung_mass: 0.0,
            suspension_travel: 0.35,
            axle_drop: 0.35,
            axle_drop_vel: 0.0,
            static_load: 0.0,
            // Failure state (WI 631b): undamaged. `rim_radius` defaults to the full radius so a blown
            // tire on a hand-built fixture drops nothing (fixtures never blow in practice — see step);
            // [`assemble_rover`] sets the real rim radius for component-built wheels.
            rim_radius: 0.35,
            rolling_scale: 1.0,
            steer_bias: 0.0,
            tire_blown: false,
            rim_bent: false,
            damper_blown: false,
        }
    }

    /// Blow this tire out (WI 631b): collapse grip to a residual, run on the rim (drop the rolling
    /// radius to [`Wheel::rim_radius`], rescaling spin inertia for the smaller wheel), add rolling
    /// resistance, and **make the contact rigid** — a flat tire run on the bare rim has essentially no
    /// compliance, so it stops absorbing and stops bouncing on a phantom tire spring (it thuds harshly
    /// instead; the suspension, if any, still does the absorbing). Latched — returns `true` only the
    /// first time (so the impact diagnostic reports a *new* blowout and re-hits don't compound).
    fn blow_tire(&mut self) -> bool {
        if self.tire_blown {
            return false;
        }
        self.tire_blown = true;
        self.grip_scale *= TIRE_BLOWN_GRIP;
        self.rolling_scale *= TIRE_BLOWN_ROLLING;
        // Run on the rim: rigid contact (the step caps this at the sub-step-stable ceiling), no give.
        self.tire_stiffness = RIGID_TIRE_STIFFNESS;
        if self.rim_radius > 0.0 && self.rim_radius < self.radius {
            let scale = self.rim_radius / self.radius;
            self.wheel_inertia = (self.wheel_inertia * scale * scale).max(0.02);
            self.radius = self.rim_radius;
        }
        true
    }

    /// Bend this rim (WI 631b): add rolling resistance and a small persistent steer/camber bias (a
    /// pull / wobble). Latched — returns `true` only the first time.
    fn bend_rim(&mut self) -> bool {
        if self.rim_bent {
            return false;
        }
        self.rim_bent = true;
        self.rolling_scale *= RIM_BENT_ROLLING;
        self.steer_bias += RIM_BENT_STEER;
        true
    }

    /// Blow this corner's damper (WI 631b): cut the suspension damping so the corner becomes bouncy
    /// (the spring rate is untouched). Latched — returns `true` only the first time.
    fn blow_damper(&mut self) -> bool {
        if self.damper_blown {
            return false;
        }
        self.damper_blown = true;
        self.damping *= DAMPER_BLOWN_DAMPING;
        true
    }

    /// Apply the graded impact-failure ladder for an effective impact `demand` speed (m/s) (WI 631b),
    /// recording any *new* component failures into `out` for the wheel at index `i`. A demand above the
    /// mount [`Wheel::shear_speed`] shears the wheel off ([`Wheel::inert`], as WI 618), superseding the
    /// lesser failures; otherwise every milder component whose rating the demand exceeds fails (the
    /// strict fraction ordering makes a damper failure imply a bent rim and a blown tire too).
    fn apply_impact(&mut self, demand: f64, i: usize, out: &mut ImpactOutcome) {
        if demand > self.shear_speed {
            self.inert = true;
            out.sheared.push(i);
            return;
        }
        if demand > DAMPER_FAIL_FRACTION * self.shear_speed && self.blow_damper() {
            out.blown_dampers.push(i);
        }
        if demand > RIM_BEND_FRACTION * self.shear_speed && self.bend_rim() {
            out.bent_rims.push(i);
        }
        if demand > TIRE_BURST_FRACTION * self.shear_speed && self.blow_tire() {
            out.blown_tires.push(i);
        }
    }
}

/// A wheeled rover: an active body plus its wheels and local gravity.
#[derive(Clone, Debug)]
pub struct Rover {
    pub body: ActiveBody,
    pub wheels: Vec<Wheel>,
    /// Downward gravitational acceleration magnitude (m/s²).
    pub gravity: f64,
    /// Last step's peak per-wheel normal-force change — a contact-jitter signal.
    pub contact_jitter: f64,
    last_total_normal: f64,
    /// Body-frame contact points on the hull's underside — the belly skids that rest on the ground
    /// when the wheels can't (sheared off or bottomed out), so the chassis doesn't tunnel through the
    /// terrain (WI 618). Spread across the footprint (one under each wheel corner) so the ends are
    /// supported and don't sink on a hard hit. Each sits at the lowest wheel-mount height.
    belly_points: Vec<DVec3>,
    /// External contact wrench to apply next step (WI 610); accumulated via [`Rover::apply_external`].
    external_force: DVec3,
    external_torque: DVec3,
}

impl Rover {
    /// Builds a rover from an active body, wheels, and gravity.
    pub fn new(body: ActiveBody, wheels: Vec<Wheel>, gravity: f64) -> Self {
        // Belly skids: one under each wheel corner, at the lowest wheel-mount height, so a wheel-less
        // chassis rests across its whole footprint (the ends don't sink) instead of tunnelling.
        let belly_y = wheels
            .iter()
            .map(|w| w.mount.y)
            .fold(f64::INFINITY, f64::min);
        let belly_y = if belly_y.is_finite() { belly_y } else { 0.0 };
        let belly_points = wheels
            .iter()
            .map(|w| DVec3::new(w.mount.x, belly_y, w.mount.z))
            .collect();
        Self {
            body,
            wheels,
            gravity,
            contact_jitter: 0.0,
            last_total_normal: 0.0,
            belly_points,
            external_force: DVec3::ZERO,
            external_torque: DVec3::ZERO,
        }
    }

    /// Advances the rover by one sub-step `dt` over `terrain` (semi-implicit).
    pub fn step(&mut self, terrain: &Terrain, dt: f64) {
        let r = DMat3::from_quat(self.body.orientation);
        let body_fwd = r * DVec3::Z;
        let body_up = r * DVec3::Y;

        let mut net_force = DVec3::new(0.0, -self.gravity * self.body.mass, 0.0);
        let mut net_torque = DVec3::ZERO;
        let mut total_normal = 0.0;
        // Live wheel count for the per-wheel inverted-slide friction cap (WI 631b playtest).
        let live_wheels = self.wheels.iter().filter(|w| !w.inert).count().max(1) as f64;

        for w in &mut self.wheels {
            // A sheared-off wheel (WI 618) exerts nothing and does not spin.
            if w.inert {
                continue;
            }
            let hub = self.body.position + r * w.mount;
            let hub_vel =
                self.body.velocity + self.body.angular_velocity().cross(hub - self.body.position);
            let ground = terrain.height(hub.x, hub.z);
            let normal = terrain.normal(hub.x, hub.z);

            // The ground normal load `n_ground` (which scales tire grip) and the vertical `support`
            // force the chassis receives. One contact model — the unsprung-mass quarter-car (WI 631a) —
            // for every wheel that has mass; the single series spring is its analytic **massless limit**
            // (`m_unsprung → 0`), used only by hand-built test fixtures and degenerate near-massless
            // stations.
            let (n_ground, support) = if w.unsprung_mass < MIN_UNSPRUNG_MASS {
                let clearance = hub.y - ground;
                let target = w.rest_length + w.radius;
                let compression = (target - clearance).clamp(0.0, target);
                if compression <= 0.0 {
                    // Airborne: the wheel spins freely under drive/brake, no contact force.
                    w.spin += motor_torque(w) / w.wheel_inertia * dt;
                    w.spin = apply_brake(w.spin, w.brake, w.wheel_inertia, dt);
                    continue;
                }
                // The massless-wheel limit: the suspension and tire springs reduce to one series
                // spring on the body (the rigid-unsprung-mass limit of the quarter-car below).
                let compression_rate = -hub_vel.dot(normal);
                let k_eff = series_stiffness(w.stiffness, w.tire_stiffness);
                let n =
                    (k_eff * compression + w.damping * compression_rate).clamp(0.0, w.max_force);
                (n, normal * n)
            } else {
                // Quarter-car (WI 631a): a suspension spring-damper between the chassis mount and the
                // axle (unsprung mass), and a tire spring-damper between the axle and the ground. The
                // tire force is the dynamic ground load (so grip follows load transfer); the chassis
                // receives only the suspension reaction. The axle's 1-D vertical DOF is integrated
                // **implicitly** (backward Euler): the scalar solve below divides by
                // `1 + (dt/m)(C + K·dt) > 1`, so it is **unconditionally stable** — a soft strut, a
                // stiff no-suspension strut (which simply locks the axle to the chassis, riding on the
                // tire), and a zero-travel strut are all the *same* code, no stiffness ever launches.
                // State is stored as a chassis-relative drop so it survives the caller repositioning
                // the body before stepping.
                let m_u = w.unsprung_mass.max(MIN_UNSPRUNG_MASS);
                let axle_y = hub.y - w.axle_drop;
                let axle_vy = hub_vel.y - w.axle_drop_vel;
                let g_ground = ground + w.radius;

                // Only compliant (sprung) wheels reach the quarter-car; a no-suspension wheel is the
                // rigid limit handled by the tire-on-body branch above. The tyre damps the unsprung hop
                // (the strut damps the body bounce). Rate capped to a mass-relative feel ceiling.
                let tire_contact = axle_y < g_ground;
                let (k_tire, c_tire) = if tire_contact {
                    let k = w.tire_stiffness.min((TIRE_OMEGA_DT_MAX / dt).powi(2) * m_u);
                    (k, 2.0 * TIRE_DAMPING_RATIO * (k * m_u).sqrt())
                } else {
                    (0.0, 0.0)
                };
                // Suspension spring (push-only, preloaded coilover): active only while compressed; a
                // kinematic droop stop (below) holds the wheel when extended, so the spring never pulls.
                let (k_susp, c_susp) = if hub.y - axle_y < w.rest_length {
                    (w.stiffness, w.damping)
                } else {
                    (0.0, 0.0)
                };

                // Backward-Euler solve for the axle's vertical velocity. Up-positive axle forces:
                //   tire    =  k_tire·(g_ground − y) − c_tire·v
                //   susp    = −k_susp·(rest − hub.y + y) − c_susp·(v − hub_vy)
                //   gravity = −m_u·g
                // ⇒ F(y,v) = A − K·y − C·v; with y_{n+1} = y + dt·v_{n+1} the solve is scalar.
                let big_k = k_tire + k_susp;
                let big_c = c_tire + c_susp;
                let a_const = k_tire * g_ground - k_susp * (w.rest_length - hub.y)
                    + c_susp * hub_vel.y
                    - m_u * self.gravity;
                let inv_m = 1.0 / m_u;
                let denom = 1.0 + dt * inv_m * (big_c + big_k * dt);
                let axle_vy_new = (axle_vy + dt * inv_m * (a_const - big_k * axle_y)) / denom;
                let mut axle_y_new = axle_y + axle_vy_new * dt;
                let mut axle_vy_fin = axle_vy_new;
                // Travel stops bound the axle to `[mount − (rest + travel), mount]`: a droop stop (the
                // strap takes the hanging load — no tension in free fall) and a bump stop (full
                // compression — a violent hit can never drive the axle above the mount).
                let max_drop = w.rest_length + w.suspension_travel;
                if hub.y - axle_y_new > max_drop {
                    axle_y_new = hub.y - max_drop;
                    axle_vy_fin = axle_vy_new.max(hub_vel.y); // don't drop faster than the chassis
                } else if axle_y_new > hub.y {
                    axle_y_new = hub.y;
                    axle_vy_fin = axle_vy_new.min(hub_vel.y); // don't rise faster than the chassis
                }
                w.axle_drop = hub.y - axle_y_new;
                w.axle_drop_vel = hub_vel.y - axle_vy_fin;

                // Ground load (tyre force) and chassis support (suspension reaction), both push-only,
                // at the resolved state.
                let n_tire = (k_tire * (g_ground - axle_y_new) - c_tire * axle_vy_fin).max(0.0);
                let susp_comp = w.rest_length - w.axle_drop;
                let f_susp =
                    (k_susp * susp_comp + c_susp * (-w.axle_drop_vel)).clamp(0.0, w.max_force);
                (n_tire, DVec3::Y * f_susp)
            };

            // The ground (tire) load feeds the contact-jitter signal (WI 611): it carries the bump/
            // landing spikes the jitter-selective angular damper cages. On the legacy path this equals
            // the support magnitude (the single series spring); on the quarter-car path it is the
            // spikier tire load (the suspension would over-smooth it and under-cage the kraken).
            total_normal += n_ground;

            // Ground tangent basis: steered heading projected perpendicular to the normal. A bent rim
            // adds a small persistent steer bias (WI 631b), so a damaged corner pulls.
            let steer_rot = DQuat::from_axis_angle(body_up, w.steer + w.steer_bias);
            let heading = steer_rot * body_fwd;
            let forward = (heading - normal * heading.dot(normal)).normalize_or_zero();
            let lateral = normal.cross(forward);

            let v_long = hub_vel.dot(forward);
            let v_lat = hub_vel.dot(lateral);
            let wheel_speed = w.spin * w.radius;

            let material = terrain.material_at(hub.x, hub.z);
            // Friction-ellipse limit scaled by the tire's grip multiplier (WI 630) and the dynamic
            // ground load (WI 631a) — so grip follows load transfer — but the load feeding grip
            // saturates at `GRIP_LOAD_FACTOR ×` the corner's static load (tire load-sensitivity), so a
            // bump/landing spike cannot multiply into a tumbling force. On the legacy path
            // (`static_load` 0) the cap is inert, reproducing the pre-split surface-only grip.
            let n_grip = if w.static_load > 0.0 {
                n_ground.min(GRIP_LOAD_FACTOR * w.static_load)
            } else {
                n_ground
            };
            let fmax = material.friction * w.grip_scale * n_grip;
            let slip_ratio = (wheel_speed - v_long) / (v_long.abs() + 1.0);
            let slip_angle = (-v_lat).atan2(v_long.abs() + EPS);
            let (fx, fy) = tire_forces(slip_ratio, slip_angle, fmax, w.slip_long, w.slip_lat);
            // Rolling resistance, scaled up for a damaged corner (blown tire / bent rim, WI 631b).
            let rolling =
                -material.rolling_resistance * w.rolling_scale * n_ground * v_long.signum();

            // Fade the tangential (drive/grip) forces only once the rover is **past upright** (tipped
            // beyond 90°, on its side/back) (WI 631a): such a wheel is not underneath the vehicle and
            // cannot lay down traction, so those forces — through a large moment arm at a bad attitude —
            // would inject rotational energy and spin a tumbled, resting rover up. Full for any normal
            // (sub-90°) attitude so bouncing/driving is unaffected, fading to zero as it inverts; the
            // normal support is never faded (no tunnelling).
            let upright = (body_up.dot(normal) * 2.0 + 1.0).clamp(0.0, 1.0);
            // Past upright, replace the faded traction with a **purely dissipative** Coulomb friction
            // opposing the contact point's tangential slip (WI 631b playtest): a friction that opposes
            // motion can only remove kinetic energy, so it cannot pump a wreck up (the reason the slip/
            // drive forces are faded) — but it does stop an upside-down rover from **sliding forever**
            // on its orientation-agnostic wheel-springs (the frictionless rest the fade left behind).
            // Blended by `upright`, so an upright rover is unaffected and a tumbled one slides to a halt.
            let v_t = hub_vel - normal * hub_vel.dot(normal);
            let v_t_speed = v_t.length();
            let coulomb = if v_t_speed > 1e-6 {
                let cap = (material.friction * n_ground)
                    .min(self.body.mass * v_t_speed / dt / live_wheels);
                -v_t / v_t_speed * cap
            } else {
                DVec3::ZERO
            };
            let contact = DVec3::new(hub.x, ground, hub.z);
            let force = support
                + (forward * (fx + rolling) + lateral * fy) * upright
                + coulomb * (1.0 - upright);
            net_force += force;
            net_torque += (contact - self.body.position).cross(force);

            // Wheel spin: motor torque accelerates (falling off near the speed limit so it never
            // burns out); ground longitudinal reaction decelerates; the brake then clamps toward zero
            // (it can lock a wheel, not reverse it — a locked wheel then brakes at the tire-grip limit).
            let ground_torque = -fx * w.radius;
            w.spin += (motor_torque(w) + ground_torque) / w.wheel_inertia * dt;
            w.spin = apply_brake(w.spin, w.brake, w.wheel_inertia, dt);
        }

        // Chassis belly contact (WI 618): penalty contacts at the hull's underside skids (one under
        // each wheel corner) so a chassis with sheared-off wheels (or bottomed-out suspension) rests
        // on its belly across its whole footprint — the ends don't sink — instead of tunnelling. The
        // skids sit below the wheeled resting height, so they're inert during normal driving.
        let n_belly = self.belly_points.len().max(1) as f64;
        // Per-skid stiffness shares the weight at ~2 cm penetration; damping near-critical for the
        // mass fraction each skid carries. Scaled to the mass so the penalty stays sub-step stable.
        let k = self.body.mass * self.gravity / 0.02 / n_belly;
        let c = 2.0 * 0.7 * (k * self.body.mass / n_belly).sqrt();
        let mut belly_contact = false;
        for &bp in &self.belly_points {
            let p = self.body.position + r * bp;
            let ground = terrain.height(p.x, p.z);
            if p.y >= ground {
                continue;
            }
            belly_contact = true;
            let normal = terrain.normal(p.x, p.z);
            let pen = ground - p.y;
            let arm = p - self.body.position;
            let v = self.body.velocity + self.body.angular_velocity().cross(arm);
            let closing = -v.dot(normal);
            let nf = (k * pen + c * closing).max(0.0);
            // Coulomb friction opposing tangential slip (so a wreck slides to a stop), capped at μN
            // and at the impulse that would just arrest this skid's share of the slip this sub-step.
            let v_t = v - normal * v.dot(normal);
            let material = terrain.material_at(p.x, p.z);
            let f_t = if v_t.length() > 1e-6 {
                let cap =
                    (material.friction * nf).min(self.body.mass * v_t.length() / dt / n_belly);
                -v_t.normalize() * cap
            } else {
                DVec3::ZERO
            };
            let force = normal * nf + f_t;
            net_force += force;
            net_torque += arm.cross(force);
        }

        // Aerodynamic drag → a finite top speed, keeping the rover in the stable band.
        net_force -= DRAG * self.body.velocity * self.body.velocity.length();

        // Jitter-selective angular drag (WI 611): a contact-stabilisation term (it cages the kraken buzz
        // and gently rights the rover), not real aerodynamics. It scales with the contact-jitter signal
        // so the per-substep buzz is caged while a genuine smooth-contact rotation (an intentional flip)
        // passes nearly undamped — and it applies only under substantial contact (see below) so it never
        // arrests a mid-air tumble.
        let jitter = (total_normal - self.last_total_normal).abs();
        let weight = self.body.mass * self.gravity;
        // Apply the drag only under **substantial** ground contact — solidly supported (a good fraction
        // of the rover's weight on the wheels) or resting on a belly skid. A mere graze (a low-hanging
        // wheel brushing the ground while the rover is really airborne, e.g. just off the ramp) does not
        // count, so a tumble keeps spinning until it truly lands (WI 631a). Inverted resting is caged
        // instead by the past-upright tangential-force fade, not by this drag.
        if total_normal > 0.1 * weight || belly_contact {
            let jitter_ratio = if weight > 1e-9 {
                (jitter / weight).min(JITTER_RATIO_CAP)
            } else {
                0.0
            };
            // Scale the drag by the body's mean moment of inertia (trace/3, rotation-invariant) so the
            // damping is a consistent *rate* across build sizes (WI 630): the same gentle restoring on a
            // light editor build as on a heavy test rover, not an absolute torque that pins light ones.
            let i = &self.body.inertia;
            let i_mean = (i.x_axis.x + i.y_axis.y + i.z_axis.z) / 3.0;
            let angular_drag =
                i_mean * (BASELINE_ANGULAR_DRAG_RATE + JITTER_ANGULAR_DRAG_RATE * jitter_ratio);
            net_torque -= angular_drag * self.body.angular_velocity();
        }

        // External contact wrench (WI 610): obstacle contacts accumulated by the caller this step.
        net_force += self.external_force;
        net_torque += self.external_torque;
        self.external_force = DVec3::ZERO;
        self.external_torque = DVec3::ZERO;

        self.contact_jitter = jitter;
        self.last_total_normal = total_normal;
        self.body.integrate_wrench(net_force, net_torque, dt);
    }

    /// Accumulate an external contact wrench (force + torque about the CoM, world frame) to apply on
    /// the next [`step`] (WI 610). Multiple obstacle contacts sum; `step` clears it after applying.
    pub fn apply_external(&mut self, force: DVec3, torque: DVec3) {
        self.external_force += force;
        self.external_torque += torque;
    }

    /// Height of the body origin above the terrain directly beneath it.
    pub fn height_above_terrain(&self, terrain: &Terrain) -> f64 {
        self.body.position.y - terrain.height(self.body.position.x, self.body.position.z)
    }

    /// Whether the rover is airborne (WI 630): every still-live wheel is clear of the ground (beyond
    /// its full suspension reach). Used by the app to detect a hard landing for fall damage. A rover
    /// with no live wheels is treated as airborne (nothing holds it up).
    pub fn airborne(&self, terrain: &Terrain) -> bool {
        let r = DMat3::from_quat(self.body.orientation);
        self.wheels.iter().filter(|w| !w.inert).all(|w| {
            let hub = self.body.position + r * w.mount;
            let clearance = hub.y - terrain.height(hub.x, hub.z);
            clearance > w.rest_length + w.radius + 0.02
        })
    }

    /// Coordinated **counter-steer** for the wheels in `steer` (indices). Each steered wheel's angle
    /// is proportional to its longitudinal (body +Z) offset from the CoM (`mount.z`):
    /// `δ_i = atan(κ · mount.z_i)`, with the gain `κ` scaled so the **farthest** steered wheel
    /// reaches `max_angle` at `input = ±1`. Wheels behind the CoM (negative `mount.z`) therefore
    /// steer the **opposite** way to those ahead, a wheel on the CoM barely steers, and the result is
    /// scale-independent (the gain absorbs the build size). Wheels not listed are set straight.
    pub fn set_steer(&mut self, input: f64, max_angle: f64, steer: &[usize]) {
        for w in &mut self.wheels {
            w.steer = 0.0;
        }
        let max_z = steer
            .iter()
            .filter_map(|&i| self.wheels.get(i))
            .map(|w| w.mount.z.abs())
            .fold(0.0_f64, f64::max);
        if max_z <= 1e-9 || max_angle == 0.0 {
            return;
        }
        let kappa = input * max_angle.tan() / max_z;
        let limit = max_angle.abs() + 0.2;
        for &i in steer {
            if let Some(w) = self.wheels.get_mut(i) {
                w.steer = (kappa * w.mount.z).atan().clamp(-limit, limit);
            }
        }
    }

    /// Apply an obstacle impact to the wheels (WI 618). `closing_speed` is the rover's approach speed
    /// into the obstacle (m/s, ≥ 0 — zero when leaning/separating, so leaning never shears) and
    /// `into_obstacle` is the (horizontal, unit) direction toward it. Each still-live wheel feels an
    /// effective impact speed `closing_speed × share`, where a near wheel (facing the obstacle) gets
    /// ~the full speed and a far one only [`SHEAR_SIDE_BIAS`]; it shears (marked [`Wheel::inert`])
    /// when that exceeds its rated [`Wheel::shear_speed`]. So a slow nudge shears nothing, a fast hit
    /// shears the near wheels, and a very fast one can take them all. Returns an [`ImpactOutcome`] for
    /// diagnostics. The caller drops the sheared indices from its drive/steer groups.
    pub fn shear_on_impact(&mut self, closing_speed: f64, into_obstacle: DVec3) -> ImpactOutcome {
        let mut out = ImpactOutcome::default();
        if closing_speed <= 0.0 {
            return out;
        }
        let into = DVec3::new(into_obstacle.x, 0.0, into_obstacle.z).normalize_or_zero();
        let r = DMat3::from_quat(self.body.orientation);
        for i in 0..self.wheels.len() {
            if self.wheels[i].inert {
                continue;
            }
            let m = r * self.wheels[i].mount;
            let dir = DVec3::new(m.x, 0.0, m.z).normalize_or_zero();
            let facing = dir.dot(into).max(0.0);
            let demand = closing_speed * (SHEAR_SIDE_BIAS + (1.0 - SHEAR_SIDE_BIAS) * facing);
            if demand > out.peak_demand {
                out.peak_demand = demand;
                out.peak_wheel = Some(i);
                out.peak_capacity = self.wheels[i].shear_speed;
            }
            // The graded failure ladder (WI 631b): below the mount shear, the soft components fail
            // first (tire, then rim, then damper); at/above it the wheel shears clean off (WI 618).
            self.wheels[i].apply_impact(demand, i, &mut out);
        }
        out
    }

    /// Apply a **hard-landing** impact to the wheels (WI 630). A vertical fall hits every wheel
    /// roughly equally, so — unlike [`Rover::shear_on_impact`], which is horizontal and side-graded —
    /// each still-live wheel feels the full `closing_speed` (the rover's downward speed at touchdown)
    /// and shears when that exceeds its rated [`Wheel::shear_speed`]. A gentle landing shears nothing;
    /// a hard enough drop tears the wheels off. Returns an [`ImpactOutcome`] for the diagnostic.
    pub fn shear_on_landing(&mut self, closing_speed: f64) -> ImpactOutcome {
        let mut out = ImpactOutcome::default();
        if closing_speed <= 0.0 {
            return out;
        }
        for i in 0..self.wheels.len() {
            if self.wheels[i].inert {
                continue;
            }
            // Track the most-stressed (lowest-rated) live wheel for the diagnostic.
            if out.peak_wheel.is_none() || self.wheels[i].shear_speed < out.peak_capacity {
                out.peak_demand = closing_speed;
                out.peak_wheel = Some(i);
                out.peak_capacity = self.wheels[i].shear_speed;
            }
            // A hard landing applies the same graded ladder as a horizontal impact (WI 631b): a heavy
            // touchdown can blow tires / bend rims before the drop is fatal enough to shear the wheels.
            self.wheels[i].apply_impact(closing_speed, i, &mut out);
        }
        out
    }
}

/// What an obstacle impact did to the wheels (WI 618): which wheels sheared, plus the most-stressed
/// live wheel and the effective impact speed it felt vs. its rated speed — the data behind the
/// impact diagnostic. All speeds in m/s.
#[derive(Clone, Debug, Default)]
pub struct ImpactOutcome {
    /// Indices of wheels that sheared off this impact.
    pub sheared: Vec<usize>,
    /// Indices of wheels whose tire newly blew out this impact (WI 631b).
    pub blown_tires: Vec<usize>,
    /// Indices of wheels whose rim newly bent this impact (WI 631b).
    pub bent_rims: Vec<usize>,
    /// Indices of wheels whose damper newly blew this impact (WI 631b).
    pub blown_dampers: Vec<usize>,
    /// The most-stressed still-live wheel at the moment of impact (if any).
    pub peak_wheel: Option<usize>,
    /// The effective impact speed that wheel felt (closing × share), m/s.
    pub peak_demand: f64,
    /// That wheel's rated shear speed, m/s — shears when `peak_demand` exceeds it.
    pub peak_capacity: f64,
}

/// The result of assembling a rover from a built lattice (WI 607): the rover plus
/// its drivetrain binding (which wheels drive / steer, by index into `rover.wheels`)
/// and the powertrain that feeds the drive wheels (WI 609).
#[derive(Clone, Debug)]
pub struct RoverAssembly {
    /// The assembled rover (chassis body + wheels).
    pub rover: Rover,
    /// Indices into `rover.wheels` that receive drive/motor torque.
    pub drive: Vec<usize>,
    /// Indices into `rover.wheels` that turn with steering input.
    pub steer: Vec<usize>,
    /// The drive power source (combustion / electric) derived from the build's devices/parts
    /// (WI 609): an `Engine`+`Tank` build burns fuel, a `Battery` build draws charge (solar from
    /// panels), and a build with neither gets a self-sustaining default.
    pub powertrain: RoverPowertrain,
}

/// The tensile strength (Pa) of the chassis voxel nearest `point` (craft frame) — the material a
/// wheel mount inherits for its shear strength (WI 618). Zero when the craft has no voxels.
fn nearest_material_strength(craft: &VoxelCraft, point: DVec3) -> f64 {
    let s = craft.cell_size;
    craft
        .voxels
        .iter()
        .map(|v| {
            let center = (v.cell.as_dvec3() + DVec3::splat(0.5)) * s;
            ((center - point).length_squared(), v.material.strength)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, strength)| strength)
        .unwrap_or(0.0)
}

/// The components of one wheel station gathered during assembly (WI 630): the optional suspension, the
/// rim (with its mass), the tire (with its mass), and the station mount (set by any component, which
/// are co-located). A station is a wheel once it has a rim and a tire.
type StationParts = (
    Option<SuspensionSpec>,
    Option<(RimSpec, f64)>,
    Option<(TireSpec, f64)>,
    DVec3,
);

/// Compose the three wheel-station components into a single runtime [`Wheel`] (WI 630). The effective
/// rolling radius is `rim.radius + tire.profile`; ride height comes from the suspension; spin inertia
/// (½·m·r², solid disk) from the unsprung `inertia_mass` (rim + tire); grip and slip from the tire;
/// and the rated shear speed from the chassis material at `mount_local` (WI 618). The suspension
/// spring / damping / max force are placeholders — [`assemble_rover`] re-sizes them to the assembled
/// mass. Returns `None` for a degenerate component set (non-positive radius or mass), so a bad station
/// is skipped rather than producing a non-physical wheel. With the pre-split tire defaults (grip 1.0,
/// slip = `C_LONG`/`C_LAT`) and a migrated radius split, this reproduces the pre-split wheel exactly.
fn compose_wheel(
    susp: SuspensionSpec,
    rim: RimSpec,
    tire: TireSpec,
    mount_local: DVec3,
    inertia_mass: f64,
    com: DVec3,
    craft: &VoxelCraft,
) -> Option<Wheel> {
    let radius = rim.radius + tire.profile;
    if radius <= 0.0 || inertia_mass <= 0.0 {
        return None;
    }
    // The rover core mounts wheels relative to the body's CoM (`body.position`).
    let mut w = Wheel::new(mount_local - com);
    w.radius = radius;
    // The rim radius is the run-on-rim target if the tire blows (WI 631b).
    w.rim_radius = rim.radius;
    w.rest_length = susp.rest_length;
    w.wheel_inertia = (0.5 * inertia_mass * radius * radius).max(0.02);
    w.shear_speed = rated_shear_speed(nearest_material_strength(craft, mount_local));
    w.grip_scale = tire.grip_scale;
    w.slip_long = tire.slip_long;
    w.slip_lat = tire.slip_lat;
    w.tire_stiffness = tire.stiffness;
    w.rigid_suspension = susp.rigid;
    // Quarter-car unsprung mass (WI 631a): the rim + tire mass becomes this corner's own vertical DOF.
    // The axle starts at the suspension free length and settles under load over the first frames.
    w.unsprung_mass = inertia_mass;
    w.suspension_travel = susp.travel;
    w.axle_drop = susp.rest_length;
    Some(w)
}

/// Assemble a [`Rover`] from a built `craft` (WI 607), placing its centre of mass at
/// world `position` under `gravity`. Mass / inertia / CoM come from the chassis voxels
/// **and** attached parts ([`VoxelCraft::mass_properties`]). Each wheel comes from either a legacy
/// monolithic [`PartKind::Wheel`] part (migrated to components) or a **complete component station**
/// (a [`PartKind::Suspension`] + [`PartKind::Rim`] + [`PartKind::Tire`] sharing a station id), composed
/// into a [`Wheel`] by [`compose_wheel`] (WI 630); drive/steer groups come from the rim flags.
///
/// Returns `None` when the craft has no mass or **no wheels** (no legacy wheel parts and no complete
/// stations) — a lattice without wheels is not a rover (the rocket assembly path handles those). This
/// is the deterministic rocket-vs-rover discriminator: wheels ⇒ rover.
pub fn assemble_rover(craft: &VoxelCraft, position: DVec3, gravity: f64) -> Option<RoverAssembly> {
    let mp = craft.mass_properties()?;

    // The component specs of one wheel gathered before composition (WI 631a): so the unsprung masses
    // can be reclassified off the sprung body before wheels are mounted CoM-relative to it.
    struct RawWheel {
        susp: SuspensionSpec,
        rim: RimSpec,
        tire: TireSpec,
        mount: DVec3,
        unsprung: f64,
    }
    let mut raw: Vec<RawWheel> = Vec::new();

    // Pass 1a — legacy monolithic wheels (still authored by the editor, and loaded from pre-630 saves)
    // migrate to the three components (WI 630). Iterated in part order so wheel indices are stable.
    for part in &craft.parts {
        if let PartKind::Wheel(spec) = part.kind {
            let (susp, rim, tire) = spec.to_components();
            raw.push(RawWheel {
                susp,
                rim,
                tire,
                mount: part.mount,
                unsprung: part.mass,
            });
        }
    }

    // Pass 1b — component stations: group the components by station id. A station becomes a wheel when
    // it has a **rim and a tire**; the **suspension is optional** — when absent the wheel rides on the
    // tire via a rigid strut (WI 630). Sorted by id for a deterministic wheel order; incomplete
    // stations are skipped. The unsprung mass of each wheel is its rim + tire mass.
    let mut stations: std::collections::BTreeMap<u32, StationParts> =
        std::collections::BTreeMap::new();
    for part in &craft.parts {
        let Some(id) = part.station else { continue };
        let entry = stations.entry(id).or_default();
        entry.3 = part.mount;
        match part.kind {
            PartKind::Suspension(s) if entry.0.is_none() => entry.0 = Some(s),
            PartKind::Rim(r) if entry.1.is_none() => entry.1 = Some((r, part.mass)),
            PartKind::Tire(t) if entry.2.is_none() => entry.2 = Some((t, part.mass)),
            _ => {}
        }
    }
    for (_, (susp, rim, tire, mount)) in stations {
        let (Some((rim, rim_mass)), Some((tire, tire_mass))) = (rim, tire) else {
            continue; // incomplete station — needs at least a rim and a tire
        };
        let susp = susp.unwrap_or_else(|| SuspensionSpec::rigid(0.3 * (rim.radius + tire.profile)));
        raw.push(RawWheel {
            susp,
            rim,
            tire,
            mount,
            unsprung: rim_mass + tire_mass,
        });
    }

    if raw.is_empty() {
        return None; // a lattice without wheels is not a rover (the rocket path handles those)
    }

    // Reclassify the unsprung masses off the sprung chassis body (WI 631a): each corner's (capped)
    // unsprung mass gains its own vertical DOF, so the chassis body's **translational** mass is reduced
    // by the same amount — weight is then conserved (the axles carry their own gravity through the tire)
    // rather than double-counted. The body keeps the **full** CoM and inertia tensor: the wheels are
    // still rigidly mounted horizontally and resist body rotation, so stripping their inertia (they sit
    // far out at the corners, contributing disproportionately) would make a heavy-wheel build tumble.
    // Keeping full inertia is the conservative, stable choice and leaves wheel mounts as in WI 630.
    // A **compliant** (sprung) wheel gets its own axle DOF, so its (capped) rim+tire mass is
    // reclassified onto the axle. A **no-suspension** wheel is the rigid limit (`k_susp → ∞`): the axle
    // is locked to the chassis and the body rides directly on the tyre, so there is no independent axle
    // — its mass stays on the sprung body (contributes 0 unsprung).
    let cap = UNSPRUNG_FRACTION * mp.mass / raw.len() as f64;
    let unsprung: Vec<(f64, DVec3)> = raw
        .iter()
        .map(|r| {
            let m = if r.susp.rigid {
                0.0
            } else {
                r.unsprung.min(cap)
            };
            (m, r.mount)
        })
        .collect();
    let total_unsprung: f64 = unsprung.iter().map(|(m, _)| *m).sum();
    let sprung_mass = (mp.mass - total_unsprung).max(1e-9);
    let com = mp.center_of_mass;
    let sprung_inertia = mp.inertia;

    // Pass 2 — compose each raw wheel into a runtime wheel; drive/steer membership from the rim flags.
    let mut wheels = Vec::new();
    let mut drive = Vec::new();
    let mut steer = Vec::new();
    for (rw, &(u_eff, _)) in raw.iter().zip(&unsprung) {
        if let Some(mut w) =
            compose_wheel(rw.susp, rw.rim, rw.tire, rw.mount, rw.unsprung, com, craft)
        {
            // The capped vertical unsprung mass (the spin inertia keeps the full rim+tire mass).
            w.unsprung_mass = u_eff;
            let i = wheels.len();
            if rw.rim.drive {
                drive.push(i);
            }
            if rw.rim.steer {
                steer.push(i);
            }
            wheels.push(w);
        }
    }
    if wheels.is_empty() {
        return None;
    }

    // Size each wheel. Two regimes of the one quarter-car model:
    //  - **Compliant** (has an axle DOF): the suspension spring carries the sprung corner at ~25%
    //    compression (damped near-critical); the tire spring carries the rest.
    //  - **No axle DOF** (`unsprung_mass` 0 — a no-suspension/rigid wheel, or a massless fixture): the
    //    rigid limit. The axle is locked to the chassis and the body rides directly on the tyre, via the
    //    tire-on-body contact (a `1e9` strut makes the series rate ≈ the tyre's). A no-suspension wheel
    //    is **critically damped** for the load it carries (no spawn bounce) and grip-capped; a bare
    //    fixture keeps the lighter legacy damping and no cap.
    let n = wheels.len() as f64;
    let load = (sprung_mass * gravity / n).max(1.0);
    let m_corner = sprung_mass / n;
    // Total static load each corner's tire carries at rest (the grip load-sensitivity reference).
    let corner_static = (mp.mass * gravity / n).max(1.0);
    for w in &mut wheels {
        w.max_force = (load * 6.0).max(1.0e3);
        if w.unsprung_mass < MIN_UNSPRUNG_MASS {
            w.stiffness = 1.0e9;
            let k_eff = series_stiffness(w.stiffness, w.tire_stiffness);
            if w.rigid_suspension {
                // No-suspension wheel: body on the tyre. Critical (ζ = 1) for the carried corner load.
                w.damping = 2.0 * (k_eff * m_corner).sqrt();
                w.static_load = corner_static;
            } else {
                w.damping = 2.0 * 0.7 * (k_eff * m_corner).sqrt(); // massless fixture
                w.static_load = 0.0;
            }
        } else {
            let target_comp = (0.25 * w.rest_length).max(1e-3);
            let k_ceiling = (TIRE_OMEGA_DT_MAX / SUBSTEP_DT).powi(2) * w.unsprung_mass;
            w.stiffness = (load / target_comp).clamp(1.0e3, 5.0e5).min(k_ceiling);
            w.damping = 2.0 * 0.7 * (w.stiffness * m_corner).sqrt();
            // Keep the tire stiffer than the suspension (real ordering), so the body heave rides on the
            // well-damped suspension instead of wallowing on a soft tire (WI 631b ride pass). Honour an
            // authored tire that is already stiffer; clamp to the sub-step ceiling.
            w.tire_stiffness = w
                .tire_stiffness
                .max(TIRE_TO_SUSPENSION_RATIO * w.stiffness)
                .min(k_ceiling);
            w.static_load = corner_static;
        }
    }

    // Powertrain (WI 609): an `Engine` device is a combustion engine for the rover (drivetrain),
    // not thrust; `Tank` feeds it, `Battery` + `SolarPanel` parts are the electric path. Sized to the
    // total assembled mass (the whole rover the drivetrain has to move), not the sprung fraction.
    let count_dev = |k: DeviceKind| craft.devices.iter().filter(|d| d.kind == k).count();
    let solar = craft
        .parts
        .iter()
        .filter(|p| matches!(p.kind, PartKind::SolarPanel))
        .count();
    let powertrain = build_powertrain(
        count_dev(DeviceKind::Engine),
        count_dev(DeviceKind::Tank),
        count_dev(DeviceKind::Battery),
        solar,
        mp.mass,
        drive.len(),
    );

    let body = ActiveBody::new(position, DVec3::ZERO, sprung_mass, sprung_inertia);
    Some(RoverAssembly {
        rover: Rover::new(body, wheels, gravity),
        drive,
        steer,
        powertrain,
    })
}

/// Two springs in series (WI 630): the combined rate `ks·kt / (ks + kt)`, always softer than the
/// softer of the two. Used to combine the suspension and tire springs into one effective contact
/// spring, so a no-suspension (rigid-strut) wheel rides on the tire and a soft tire softens the ride.
fn series_stiffness(ks: f64, kt: f64) -> f64 {
    let sum = ks + kt;
    if sum <= 0.0 {
        0.0
    } else {
        ks * kt / sum
    }
}

/// Drive torque after the motor's speed limit: it falls to zero as the wheel
/// approaches [`MAX_WHEEL_SPIN`], so the wheels cannot spin up without bound.
fn motor_torque(w: &Wheel) -> f64 {
    let scale = (1.0 - w.spin.abs() / MAX_WHEEL_SPIN).clamp(0.0, 1.0);
    w.drive_torque * scale
}

/// Apply `brake` torque to a wheel's `spin` over `dt` **without overshooting zero** (WI 618). A brake
/// can stop a wheel, not spin it backwards — clamping at zero avoids the huge-brake / low-inertia
/// chatter (spin flipping sign every sub-step) that made straight-line braking feel mushy. Once the
/// wheel locks (`spin == 0`) the tire brakes at the grip limit via its (large) longitudinal slip.
fn apply_brake(spin: f64, brake: f64, inertia: f64, dt: f64) -> f64 {
    if brake <= 0.0 || spin == 0.0 {
        return spin;
    }
    let delta = brake / inertia * dt; // magnitude the brake would shed this sub-step
    if spin.abs() <= delta {
        0.0
    } else {
        spin - spin.signum() * delta
    }
}

/// Simplified slip-based tire forces (longitudinal, lateral), saturating at the
/// friction-ellipse limit `fmax`. Zero at zero slip; tanh-saturating with slip.
fn tire_forces(slip_ratio: f64, slip_angle: f64, fmax: f64, c_long: f64, c_lat: f64) -> (f64, f64) {
    let fx = fmax * (c_long * slip_ratio).tanh();
    // `slip_angle = atan2(-v_lat, |v_long|)`, so `fy` here points to **oppose** the
    // lateral slip (a restoring force). Getting this sign wrong makes the lateral
    // force amplify sliding → oversteer spin-out.
    let fy = fmax * (c_lat * slip_angle).tanh();
    let mag = (fx * fx + fy * fy).sqrt();
    if mag > fmax && mag > 0.0 {
        let s = fmax / mag;
        (fx * s, fy * s)
    } else {
        (fx, fy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::SurfaceMaterial;
    use crate::voxel::{
        Device, DeviceKind, Material, Part, PartKind, RimSpec, SuspensionSpec, TireSpec, Voxel,
        VoxelCraft, WheelPart,
    };
    use glam::IVec3;

    /// A 3×5 voxel chassis (the rover-scene block) with `n_parts` wheel parts mounted
    /// at the four corners (front pair steering, all driving).
    fn chassis_with_wheels() -> VoxelCraft {
        let mut craft = VoxelCraft::new(0.5);
        for x in 0..3 {
            for z in 0..5 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        let mounts = [
            (DVec3::new(-1.0, -0.2, -2.0), false), // rear-left: drive only
            (DVec3::new(1.0, -0.2, -2.0), false),  // rear-right: drive only
            (DVec3::new(-1.0, -0.2, 2.0), true),   // front-left: drive + steer
            (DVec3::new(1.0, -0.2, 2.0), true),    // front-right: drive + steer
        ];
        for (mount, steer) in mounts {
            craft.parts.push(Part {
                mount,
                mass: 40.0,
                kind: PartKind::Wheel(WheelPart::new(true, steer)),
                station: None,
            });
        }
        craft
    }

    /// The same 3×5 chassis as [`chassis_with_wheels`], but each wheel authored as a **component
    /// station** (suspension + rim + tire) instead of a legacy monolithic wheel (WI 630). The
    /// component split is behaviour-preserving: each wheel's total mass (40 kg, all on the rim+tire so
    /// it counts toward inertia exactly like the legacy single-mass part) and rolling radius match.
    fn chassis_with_stations() -> VoxelCraft {
        let mut craft = VoxelCraft::new(0.5);
        for x in 0..3 {
            for z in 0..5 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        let mounts = [
            (DVec3::new(-1.0, -0.2, -2.0), false),
            (DVec3::new(1.0, -0.2, -2.0), false),
            (DVec3::new(-1.0, -0.2, 2.0), true),
            (DVec3::new(1.0, -0.2, 2.0), true),
        ];
        // Components mirror `WheelPart::new(true, steer).to_components()` so the composed wheel equals
        // the legacy one; suspension carries no mass, rim+tire split the 40 kg.
        let (susp, _rim, _tire) = WheelPart::new(true, false).to_components();
        for (id, (mount, steer)) in mounts.into_iter().enumerate() {
            let (_, rim, tire) = WheelPart::new(true, steer).to_components();
            let id = id as u32;
            craft.parts.push(Part {
                mount,
                mass: 0.0,
                kind: PartKind::Suspension(susp),
                station: Some(id),
            });
            craft.parts.push(Part {
                mount,
                mass: 20.0,
                kind: PartKind::Rim(rim),
                station: Some(id),
            });
            craft.parts.push(Part {
                mount,
                mass: 20.0,
                kind: PartKind::Tire(tire),
                station: Some(id),
            });
        }
        craft
    }

    #[test]
    fn tire_spec_defaults_match_core_slip_constants() {
        // The TireSpec defaults must reproduce the rover core's slip constants and unit grip, or a
        // migrated wheel would not drive identically (WI 630 behaviour preservation).
        let t = TireSpec::new(0.1);
        assert_eq!(t.slip_long, C_LONG);
        assert_eq!(t.slip_lat, C_LAT);
        assert_eq!(t.grip_scale, 1.0);
    }

    #[test]
    fn migration_equivalence_legacy_wheel_and_station_compose_identically() {
        // A pre-split (legacy WheelPart) build and the same build authored as component stations must
        // assemble to numerically identical wheels — the WI 630 migration-equivalence criterion.
        let legacy =
            assemble_rover(&chassis_with_wheels(), DVec3::new(0.0, 5.0, 0.0), 9.81).unwrap();
        let split =
            assemble_rover(&chassis_with_stations(), DVec3::new(0.0, 5.0, 0.0), 9.81).unwrap();
        assert_eq!(legacy.rover.wheels.len(), split.rover.wheels.len());
        assert_eq!(legacy.drive, split.drive);
        assert_eq!(legacy.steer, split.steer);
        for (a, b) in legacy.rover.wheels.iter().zip(&split.rover.wheels) {
            assert!((a.mount - b.mount).length() < 1e-12, "mount {a:?} {b:?}");
            assert!(
                (a.radius - b.radius).abs() < 1e-12,
                "radius {} {}",
                a.radius,
                b.radius
            );
            assert!((a.rest_length - b.rest_length).abs() < 1e-12);
            assert!((a.wheel_inertia - b.wheel_inertia).abs() < 1e-9, "inertia");
            assert!((a.stiffness - b.stiffness).abs() < 1e-6, "stiffness");
            assert!((a.damping - b.damping).abs() < 1e-6, "damping");
            assert!((a.max_force - b.max_force).abs() < 1e-6, "max_force");
            assert!((a.shear_speed - b.shear_speed).abs() < 1e-12, "shear");
            assert_eq!(a.grip_scale, b.grip_scale);
            assert_eq!(a.slip_long, b.slip_long);
            assert_eq!(a.slip_lat, b.slip_lat);
        }
    }

    #[test]
    fn incomplete_station_is_not_a_wheel() {
        // A station missing its tire is not a complete wheel; a build whose only "wheels" are
        // incomplete stations is not a rover (WI 630 invalid-station handling).
        let mut craft = VoxelCraft::new(0.5);
        for x in 0..3 {
            for z in 0..5 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        let mount = DVec3::new(-1.0, -0.2, -2.0);
        let (susp, rim, _tire) = WheelPart::new(true, false).to_components();
        craft.parts.push(Part {
            mount,
            mass: 0.0,
            kind: PartKind::Suspension(susp),
            station: Some(0),
        });
        craft.parts.push(Part {
            mount,
            mass: 20.0,
            kind: PartKind::Rim(rim),
            station: Some(0),
        });
        // No tire → station incomplete → no wheel → not a rover.
        assert!(assemble_rover(&craft, DVec3::new(0.0, 5.0, 0.0), 9.81).is_none());
    }

    #[test]
    fn station_without_suspension_is_a_valid_wheel() {
        // A station with just a rim + tire (no suspension placed) is a valid wheel that rides on the
        // tire's compliance — suspension is optional (WI 630, the user's no-suspension authoring).
        let mut craft = VoxelCraft::new(0.5);
        for x in 0..3 {
            for z in 0..5 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        for (id, (mount, steer)) in [
            (DVec3::new(-1.0, -0.2, -2.0), false),
            (DVec3::new(1.0, -0.2, -2.0), false),
            (DVec3::new(-1.0, -0.2, 2.0), true),
            (DVec3::new(1.0, -0.2, 2.0), true),
        ]
        .into_iter()
        .enumerate()
        {
            let id = id as u32;
            craft.parts.push(Part {
                mount,
                mass: 20.0,
                kind: PartKind::Rim(RimSpec {
                    radius: 0.25,
                    drive: true,
                    steer,
                }),
                station: Some(id),
            });
            craft.parts.push(Part {
                mount,
                mass: 20.0,
                kind: PartKind::Tire(TireSpec {
                    stiffness: 2.0e5,
                    ..TireSpec::new(0.1)
                }),
                station: Some(id),
            });
        }
        let asm = assemble_rover(&craft, DVec3::new(0.0, 5.0, 0.0), 9.81)
            .expect("rim+tire stations ⇒ rover even without suspension");
        assert_eq!(asm.rover.wheels.len(), 4);
        assert!(asm.rover.wheels.iter().all(|w| w.rigid_suspension));
    }

    /// A 3×5 chassis built as four component stations with the given suspension + tire (rim default,
    /// all-drive, front-steer), assembled resting just above flat terrain (WI 630 Phase B helper).
    fn station_assembly(susp: SuspensionSpec, tire: TireSpec, terrain: &Terrain) -> RoverAssembly {
        let mut craft = VoxelCraft::new(0.5);
        for x in 0..3 {
            for z in 0..5 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        let mounts = [
            (DVec3::new(-1.0, -0.2, -2.0), false),
            (DVec3::new(1.0, -0.2, -2.0), false),
            (DVec3::new(-1.0, -0.2, 2.0), true),
            (DVec3::new(1.0, -0.2, 2.0), true),
        ];
        for (id, (mount, steer)) in mounts.into_iter().enumerate() {
            let id = id as u32;
            craft.parts.push(Part {
                mount,
                mass: 0.0,
                kind: PartKind::Suspension(susp),
                station: Some(id),
            });
            craft.parts.push(Part {
                mount,
                mass: 20.0,
                kind: PartKind::Rim(RimSpec {
                    radius: 0.25,
                    drive: true,
                    steer,
                }),
                station: Some(id),
            });
            craft.parts.push(Part {
                mount,
                mass: 20.0,
                kind: PartKind::Tire(tire),
                station: Some(id),
            });
        }
        let ground = terrain.height(0.0, 0.0);
        assemble_rover(&craft, DVec3::new(0.0, ground + 0.9, 0.0), 9.81).expect("stations ⇒ rover")
    }

    #[test]
    fn series_stiffness_is_softer_than_either_spring() {
        // The series combination is always softer than the softer spring, and an effectively rigid
        // tire leaves the suspension spring essentially unchanged (WI 630).
        let s = series_stiffness(1.0e5, 3.0e4);
        // Softer than the softer of the two springs (3e4, which is already < 1e5).
        assert!(s < 3.0e4);
        let rigid = series_stiffness(1.0e5, 1.0e9);
        assert!((rigid - 1.0e5).abs() / 1.0e5 < 1e-3);
    }

    #[test]
    fn softer_tire_settles_lower_than_a_stiff_tire() {
        // A softer (but still physical) tire compresses more under load, so the body rests a little
        // lower — the felt difference of "change the tires" (WI 630). But the tire is floored at
        // TIRE_TO_SUSPENSION_RATIO× the suspension rate (WI 631b ride pass: a real tire is stiffer than
        // the spring), so an *absurdly* soft tire does not sink/wallow — it clamps to the floor and
        // rides the same as any other below-floor tire. Both settle finite.
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let settle = |tire: TireSpec| {
            let mut rover = station_assembly(SuspensionSpec::new(), tire, &terrain).rover;
            for _ in 0..6_000 {
                rover.step(&terrain, DT);
            }
            assert!(rover.body.position.is_finite());
            rover.body.position.y
        };
        // Within the physical range (both above the floor), a softer tire rides lower.
        let stiff = settle(TireSpec::new(0.1)); // near-rigid
        let soft = settle(TireSpec {
            stiffness: 4.0e5,
            ..TireSpec::new(0.1)
        });
        assert!(
            soft < stiff - 0.01,
            "a softer (physical) tire did not ride lower: soft {soft:.3} vs stiff {stiff:.3}"
        );
        // Below the floor, tires clamp to the same rate (no wallow): two very soft tires ride alike.
        let floored_a = settle(TireSpec {
            stiffness: 3.0e4,
            ..TireSpec::new(0.1)
        });
        let floored_b = settle(TireSpec {
            stiffness: 1.0e4,
            ..TireSpec::new(0.1)
        });
        assert!(
            (floored_a - floored_b).abs() < 0.01,
            "below-floor tires should ride alike (floored): {floored_a:.3} vs {floored_b:.3}"
        );
    }

    #[test]
    fn assembled_tire_is_stiffer_than_the_suspension() {
        // WI 631b ride pass: the real-world ordering (tire stiffer than the spring) must hold after
        // assembly at any build scale, even when the authored tire is softer than the mass-sized
        // suspension — otherwise the body wallows on a soft tire. A too-soft authored tire is floored.
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let soft_tire = TireSpec {
            stiffness: 1.0e4, // absurdly soft — must be floored above the suspension
            ..TireSpec::new(0.1)
        };
        let rover = station_assembly(SuspensionSpec::new(), soft_tire, &terrain).rover;
        for w in &rover.wheels {
            let k_ceiling = (TIRE_OMEGA_DT_MAX / SUBSTEP_DT).powi(2) * w.unsprung_mass;
            let expected = (TIRE_TO_SUSPENSION_RATIO * w.stiffness).min(k_ceiling);
            assert!(
                w.tire_stiffness >= expected - 1.0,
                "tire not floored above the spring: k_tire {} < {}× k_susp {}",
                w.tire_stiffness,
                TIRE_TO_SUSPENSION_RATIO,
                w.stiffness
            );
        }
    }

    #[test]
    fn rigid_suspension_rides_on_tire_without_tunnelling() {
        // A rigid strut has no spring travel of its own; the wheel must still rest on the tire's
        // compliance rather than tunnelling through the ground (WI 630, the user's no-suspension case).
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let tire = TireSpec {
            stiffness: 2.0e5,
            ..TireSpec::new(0.1)
        };
        let mut rover = station_assembly(SuspensionSpec::rigid(0.35), tire, &terrain).rover;
        for _ in 0..6_000 {
            rover.step(&terrain, DT);
        }
        assert!(rover.body.position.is_finite());
        assert!(
            rover.height_above_terrain(&terrain) > 0.0,
            "rigid-strut rover sank through the ground: h = {}",
            rover.height_above_terrain(&terrain)
        );
    }

    #[test]
    fn grippier_tire_corners_harder_than_a_slick() {
        // More tire grip → a higher sustainable lateral force → a tighter turn (higher yaw rate) at
        // the same speed and steer (WI 630). Both stay finite and upright (no spin-out launch).
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let peak_yaw = |grip: f64| {
            let tire = TireSpec {
                grip_scale: grip,
                ..TireSpec::new(0.1)
            };
            let asm = station_assembly(SuspensionSpec::new(), tire, &terrain);
            let (mut rover, drive) = (asm.rover, asm.drive);
            let mass = rover.body.mass;
            for &i in &drive {
                rover.wheels[i].drive_torque = mass * 1.5;
            }
            for _ in 0..4_000 {
                rover.step(&terrain, DT); // reach a moderate speed
            }
            // A gentle steady steer (a wide turn) — well below the provoked-rollover regime, so the
            // only variable is how hard the tire can hold the corner.
            rover.wheels[2].steer = 0.1;
            rover.wheels[3].steer = 0.1;
            let mut peak = 0.0_f64;
            for _ in 0..8_000 {
                rover.step(&terrain, DT);
                assert!(rover.body.position.is_finite());
                peak = peak.max(rover.body.angular_velocity().y.abs());
            }
            assert!(body_up(&rover).y > 0.8, "rover rolled during the grip test");
            peak
        };
        let grippy = peak_yaw(1.3);
        let slick = peak_yaw(0.5);
        assert!(
            grippy > slick * 1.15,
            "grippier tire did not corner harder: grippy yaw {grippy:.3} vs slick {slick:.3}"
        );
    }

    #[test]
    fn extreme_tire_grip_stays_within_kraken_bounds() {
        // The grip multiplier must not break the kraken bounds at either extreme: driving hard over
        // bumps with very low and very high grip stays finite with bounded spin (WI 630 stability gate).
        let terrain = Terrain {
            amplitude: 0.5,
            ..Default::default()
        };
        for grip in [0.3, 2.0] {
            let tire = TireSpec {
                grip_scale: grip,
                ..TireSpec::new(0.1)
            };
            let asm = station_assembly(SuspensionSpec::new(), tire, &terrain);
            let (mut rover, drive) = (asm.rover, asm.drive);
            for &i in &drive {
                rover.wheels[i].drive_torque = 2_500.0;
            }
            let mut max_omega = 0.0_f64;
            for step in 0..20_000 {
                rover.step(&terrain, DT);
                assert!(
                    rover.body.position.is_finite() && rover.body.velocity.is_finite(),
                    "grip {grip}: non-finite at step {step}"
                );
                if step > 1_000 {
                    max_omega = max_omega.max(rover.body.angular_velocity().length());
                }
            }
            assert!(
                max_omega < 6.0,
                "grip {grip}: tumbled over bumps, max_omega = {max_omega}"
            );
        }
    }

    #[test]
    fn light_rover_drives_roughly_straight_and_can_be_rolled() {
        // The inertia-scaled angular drag (WI 630) must leave a *light* editor-scale build both
        // drivable (tracks roughly straight under symmetric throttle) and rollable (a hard turn at
        // speed tips it) — the old absolute drag did the first by pinning the second ("magical force").
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let light = || {
            let s = 0.1;
            let mut craft = VoxelCraft::new(s);
            for x in 0..4 {
                for z in 0..6 {
                    craft.voxels.push(Voxel {
                        cell: IVec3::new(x, 0, z),
                        material: Material::COMPOSITE,
                    });
                }
            }
            for (cx, cz, st) in [(0, 0, false), (3, 0, false), (0, 5, true), (3, 5, true)] {
                let mount = DVec3::new((cx as f64 + 0.5) * s, -0.05, (cz as f64 + 0.5) * s);
                let id = craft.parts.len() as u32;
                craft.parts.push(Part {
                    mount,
                    mass: 1.0,
                    kind: PartKind::Suspension(SuspensionSpec::for_cell_size(s)),
                    station: Some(id),
                });
                craft.parts.push(Part {
                    mount,
                    mass: 1.0,
                    kind: PartKind::Rim(RimSpec {
                        radius: 0.1,
                        drive: true,
                        steer: st,
                    }),
                    station: Some(id),
                });
                craft.parts.push(Part {
                    mount,
                    mass: 1.0,
                    kind: PartKind::Tire(TireSpec::new(0.05)),
                    station: Some(id),
                });
            }
            let asm = assemble_rover(&craft, DVec3::ZERO, 9.81).unwrap();
            let drop = asm
                .rover
                .wheels
                .iter()
                .map(|w| w.rest_length + w.radius - w.mount.y)
                .fold(0.0_f64, f64::max);
            let mut rover = asm.rover;
            rover.body.position = DVec3::new(0.0, terrain.height(0.0, 0.0) + drop, 0.0);
            (rover, asm.drive, asm.steer)
        };

        // Drivable: symmetric throttle tracks roughly straight (small lateral drift) and stays upright.
        let (mut rover, drive, _) = light();
        let mass = rover.body.mass;
        for _ in 0..16_000 {
            for &i in &drive {
                rover.wheels[i].drive_torque = mass * 2.0;
            }
            rover.step(&terrain, DT);
        }
        assert!(
            rover.body.position.z > 1.0,
            "light rover did not drive forward"
        );
        assert!(
            rover.body.position.x.abs() < 0.5 * rover.body.position.z,
            "light rover wandered badly: x={:.2} z={:.2}",
            rover.body.position.x,
            rover.body.position.z
        );
        assert!(
            body_up(&rover).y > 0.9,
            "light rover should stay upright cruising"
        );

        // Rollable: a *tippy* light build (tall + narrow, grippy slicks) tips on a hard turn at speed.
        // Before the inertia-scaled drag this same light build was pinned upright (the "magical force").
        let s = 0.1;
        let mut craft = VoxelCraft::new(s);
        for y in 0..4 {
            for z in 0..5 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(0, y, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        for (cz, st) in [(0, false), (4, true)] {
            for dx in [-0.25, 0.25] {
                let mount = DVec3::new(dx, -0.05, (cz as f64 + 0.5) * s);
                let id = craft.parts.len() as u32;
                // Sprung (suspended) wheels: a light, tall-narrow build on real suspension is the
                // roll-prone case the WI 611 relaxed damper must still allow to tip.
                craft.parts.push(Part {
                    mount,
                    mass: 1.0,
                    kind: PartKind::Suspension(SuspensionSpec::for_cell_size(s)),
                    station: Some(id),
                });
                craft.parts.push(Part {
                    mount,
                    mass: 1.0,
                    kind: PartKind::Rim(RimSpec {
                        radius: 0.1,
                        drive: true,
                        steer: st,
                    }),
                    station: Some(id),
                });
                craft.parts.push(Part {
                    mount,
                    mass: 1.0,
                    // Grippy slick: high grip generates the lateral force that tips a tall build.
                    kind: PartKind::Tire(TireSpec {
                        grip_scale: 1.6,
                        ..TireSpec::new(0.05)
                    }),
                    station: Some(id),
                });
            }
        }
        let asm = assemble_rover(&craft, DVec3::ZERO, 9.81).unwrap();
        let drop = asm
            .rover
            .wheels
            .iter()
            .map(|w| w.rest_length + w.radius - w.mount.y)
            .fold(0.0_f64, f64::max);
        let mut rover = asm.rover;
        rover.body.position = DVec3::new(0.0, terrain.height(0.0, 0.0) + drop, 0.0);
        let tip_mass = rover.body.mass;
        for &i in &asm.drive {
            rover.wheels[i].drive_torque = tip_mass * 6.0;
        }
        for _ in 0..6_000 {
            rover.step(&terrain, DT);
        }
        for &i in &asm.steer {
            rover.wheels[i].steer = 0.6;
        }
        let mut min_up_y = 1.0_f64;
        for _ in 0..16_000 {
            rover.step(&terrain, DT);
            min_up_y = min_up_y.min(body_up(&rover).y);
        }
        assert!(rover.body.position.is_finite());
        assert!(
            min_up_y < 0.6,
            "tippy light rover could not be rolled by a hard turn at speed: min up.y = {min_up_y}"
        );
    }

    #[test]
    fn shear_on_landing_is_speed_graded() {
        // A gentle touchdown shears nothing; a hard fall shears every wheel (WI 630 fall damage). The
        // default rated shear speed is BASE_SHEAR_SPEED (8 m/s), so 2 m/s is safe and 50 m/s is fatal.
        let make = || {
            let body = ActiveBody::new(DVec3::ZERO, DVec3::ZERO, 100.0, DMat3::IDENTITY);
            let wheels = vec![
                Wheel::new(DVec3::new(-1.0, -0.2, -1.0)),
                Wheel::new(DVec3::new(1.0, -0.2, -1.0)),
                Wheel::new(DVec3::new(-1.0, -0.2, 1.0)),
                Wheel::new(DVec3::new(1.0, -0.2, 1.0)),
            ];
            Rover::new(body, wheels, 9.81)
        };
        let mut soft = make();
        assert!(soft.shear_on_landing(2.0).sheared.is_empty());
        let mut hard = make();
        assert_eq!(hard.shear_on_landing(50.0).sheared.len(), 4);
        assert!(hard.wheels.iter().all(|w| w.inert));
    }

    #[test]
    fn graded_impact_ladder_fails_components_in_severity_order() {
        // WI 631b: below the catastrophic mount shear (WI 618), an impact fails the soft components
        // first — tire, then rim, then damper — and only a hard enough hit shears the wheel off. A
        // hard landing applies the full (un-side-graded) demand, so the speed thresholds are crisp on
        // default fixtures (shear = BASE_SHEAR_SPEED).
        let make = || {
            let body = ActiveBody::new(DVec3::ZERO, DVec3::ZERO, 100.0, DMat3::IDENTITY);
            let wheels = vec![
                Wheel::new(DVec3::new(-1.0, -0.2, -1.0)),
                Wheel::new(DVec3::new(1.0, -0.2, -1.0)),
            ];
            Rover::new(body, wheels, 9.81)
        };
        let s = BASE_SHEAR_SPEED;

        // Tire only (demand in (tire_burst, rim_bend) = (0.70, 0.82) · shear).
        let mut r = make();
        let o = r.shear_on_landing(0.76 * s);
        assert_eq!(o.blown_tires.len(), 2);
        assert!(o.bent_rims.is_empty() && o.blown_dampers.is_empty() && o.sheared.is_empty());
        assert!(r
            .wheels
            .iter()
            .all(|w| w.tire_blown && !w.rim_bent && !w.damper_blown && !w.inert));

        // Tire + rim (demand in (rim_bend, damper_fail) = (0.82, 0.92) · shear).
        let mut r = make();
        let o = r.shear_on_landing(0.87 * s);
        assert_eq!(o.blown_tires.len(), 2);
        assert_eq!(o.bent_rims.len(), 2);
        assert!(o.blown_dampers.is_empty() && o.sheared.is_empty());
        assert!(r
            .wheels
            .iter()
            .all(|w| w.tire_blown && w.rim_bent && !w.damper_blown && !w.inert));

        // Tire + rim + damper (demand in (damper_fail, shear) = (0.92, 1.0) · shear).
        let mut r = make();
        let o = r.shear_on_landing(0.96 * s);
        assert_eq!(o.blown_tires.len(), 2);
        assert_eq!(o.bent_rims.len(), 2);
        assert_eq!(o.blown_dampers.len(), 2);
        assert!(o.sheared.is_empty());
        assert!(r
            .wheels
            .iter()
            .all(|w| w.tire_blown && w.rim_bent && w.damper_blown && !w.inert));

        // Catastrophic: above the mount shear the wheel shears clean off (WI 618 unchanged).
        let mut r = make();
        let o = r.shear_on_landing(1.5 * s);
        assert_eq!(o.sheared.len(), 2);
        assert!(
            o.blown_tires.is_empty(),
            "a clean shear supersedes lesser failures"
        );
        assert!(r.wheels.iter().all(|w| w.inert));
    }

    #[test]
    fn impact_failures_latch_and_do_not_compound() {
        // A failed component stays failed: re-hitting it at the same severity is a no-op (no compounding
        // grip loss, no duplicate report), but a harder hit can still escalate to the next component.
        let body = ActiveBody::new(DVec3::ZERO, DVec3::ZERO, 100.0, DMat3::IDENTITY);
        let mut r = Rover::new(body, vec![Wheel::new(DVec3::new(-1.0, -0.2, -1.0))], 9.81);
        let s = BASE_SHEAR_SPEED;
        assert_eq!(r.shear_on_landing(0.76 * s).blown_tires.len(), 1);
        let grip_after = r.wheels[0].grip_scale;
        let again = r.shear_on_landing(0.76 * s);
        assert!(again.blown_tires.is_empty(), "no second blowout report");
        assert_eq!(r.wheels[0].grip_scale, grip_after, "grip did not compound");
        // A harder hit escalates to the rim without re-blowing the (already blown) tire.
        let harder = r.shear_on_landing(0.87 * s);
        assert!(harder.blown_tires.is_empty());
        assert_eq!(harder.bent_rims.len(), 1);
    }

    #[test]
    fn blown_tire_loses_grip_and_runs_on_the_rim() {
        // WI 631b: on an assembled (component) rover the rim radius is real, so a blown tire collapses
        // grip to a residual, drops the rolling radius to the rim (runs on the rim), and drags more.
        let craft = chassis_with_stations();
        let mut rover = assemble_rover(&craft, DVec3::new(0.0, 5.0, 0.0), 9.81)
            .unwrap()
            .rover;
        let s = rover.wheels[0].shear_speed;
        let (grip0, radius0, rim, k0) = {
            let w = &rover.wheels[0];
            (w.grip_scale, w.radius, w.rim_radius, w.tire_stiffness)
        };
        assert!(rim < radius0, "rim radius is below the rolling radius");
        let o = rover.shear_on_landing(0.76 * s); // tire only
        assert_eq!(o.blown_tires.len(), 4);
        let w = &rover.wheels[0];
        assert!(w.tire_blown);
        assert!(
            w.grip_scale < grip0 && w.grip_scale > 0.0,
            "grip collapsed to a residual: {} -> {}",
            grip0,
            w.grip_scale
        );
        assert!((w.radius - rim).abs() < 1e-9, "now runs on the rim radius");
        assert!(w.rolling_scale > 1.0, "more rolling resistance");
        assert!(
            w.tire_stiffness > k0,
            "a blown tire runs rigid on the rim (no compliance): {k0} -> {}",
            w.tire_stiffness
        );
    }

    #[test]
    fn bent_rim_and_blown_damper_change_the_corner() {
        // WI 631b: a hit between damper_fail and shear bends the rim (adds a steer/camber bias) and
        // blows the damper (cuts that corner's damping) without removing the wheel.
        let craft = chassis_with_stations();
        let mut rover = assemble_rover(&craft, DVec3::new(0.0, 5.0, 0.0), 9.81)
            .unwrap()
            .rover;
        let s = rover.wheels[0].shear_speed;
        let damp0 = rover.wheels[0].damping;
        let o = rover.shear_on_landing(0.93 * s);
        assert_eq!(o.bent_rims.len(), 4);
        assert_eq!(o.blown_dampers.len(), 4);
        assert!(o.sheared.is_empty());
        let w = &rover.wheels[0];
        assert!(
            w.rim_bent && w.steer_bias.abs() > 0.0,
            "bent rim adds a steer bias"
        );
        assert!(
            w.damper_blown && w.damping < damp0 * 0.5,
            "damper cut: {} -> {}",
            damp0,
            w.damping
        );
    }

    #[test]
    fn a_damaged_rover_stays_within_kraken_bounds() {
        // A rover with every corner damaged (tire blown — running on the rims — rim bent, damper blown)
        // must still drive over bumps finite and untumbled: a failure must never wake the kraken.
        let terrain = Terrain {
            amplitude: 0.3,
            ..Default::default()
        };
        let asm = station_assembly(SuspensionSpec::new(), TireSpec::new(0.1), &terrain);
        let (mut rover, drive) = (asm.rover, asm.drive);
        for _ in 0..3_000 {
            rover.step(&terrain, DT); // settle
        }
        let s = rover.wheels[0].shear_speed;
        let o = rover.shear_on_landing(0.93 * s); // damage every corner short of shearing
        assert!(o.sheared.is_empty());
        assert!(rover
            .wheels
            .iter()
            .all(|w| w.tire_blown && w.rim_bent && w.damper_blown));
        for &i in &drive {
            rover.wheels[i].drive_torque = rover.body.mass;
        }
        let mut max_omega = 0.0_f64;
        for step in 0..16_000 {
            rover.step(&terrain, DT);
            assert!(
                rover.body.position.is_finite() && rover.body.velocity.is_finite(),
                "damaged rover non-finite at step {step}"
            );
            if step > 1_000 {
                max_omega = max_omega.max(rover.body.angular_velocity().length());
            }
        }
        assert!(
            max_omega < 6.0,
            "damaged rover tumbled: max_omega = {max_omega}"
        );
        assert!(body_up(&rover).y > 0.5, "damaged rover flipped");
    }

    #[test]
    fn driving_off_the_ramp_launches_the_rover_into_the_air() {
        // With a short (cell-scaled) suspension, a rover that drives up the wedge and off the lip
        // actually leaves the ground (WI 630 playtest fix + test ramp) — the wheels can no longer
        // reach down to stay glued. The rover catches real air and stays finite.
        use crate::terrain::Ramp;
        let terrain = Terrain {
            amplitude: 0.0,
            ramp: Some(Ramp {
                center_x: 0.0,
                half_width: 3.0,
                start_z: 4.0,
                run: 3.0,
                angle: 30.0_f64.to_radians(),
            }),
            ..Default::default()
        };
        let asm = station_assembly(
            SuspensionSpec::for_cell_size(0.5),
            TireSpec::new(0.1),
            &terrain,
        );
        let (mut rover, drive) = (asm.rover, asm.drive);
        let mass = rover.body.mass;
        rover.body.velocity = DVec3::new(0.0, 0.0, 8.0); // a running start toward the ramp
        for &i in &drive {
            rover.wheels[i].drive_torque = mass * 5.0;
        }
        let mut max_air = 0.0_f64;
        for step in 0..8_000 {
            rover.step(&terrain, DT);
            assert!(rover.body.position.is_finite(), "non-finite at step {step}");
            if step > 1_000 {
                max_air = max_air.max(rover.height_above_terrain(&terrain));
            }
        }
        assert!(
            max_air > 0.4,
            "rover did not launch off the ramp (max height above terrain = {max_air:.2} m)"
        );
    }

    // ---- WI 631a: unsprung-mass quarter-car ----

    /// A 0.1 m four-corner rover whose front and rear axles take the given (optional) suspension strut
    /// — `None` ⇒ a no-suspension station (rim + tire only) — and the given tire `stiffness` (the
    /// editor's off-road/road/slick presets). For the mixed / zero-travel stability tests.
    fn corner_rover(
        front: Option<SuspensionSpec>,
        rear: Option<SuspensionSpec>,
        stiffness: f64,
    ) -> VoxelCraft {
        let s = 0.1;
        let mut craft = VoxelCraft::new(s);
        for x in 0..4 {
            for z in 0..6 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        for (id, (cx, cz, st, susp)) in [
            (0, 0, false, rear),
            (3, 0, false, rear),
            (0, 5, true, front),
            (3, 5, true, front),
        ]
        .into_iter()
        .enumerate()
        {
            let id = id as u32;
            let mount = DVec3::new((cx as f64 + 0.5) * s, -0.05, (cz as f64 + 0.5) * s);
            if let Some(sp) = susp {
                craft.parts.push(Part {
                    mount,
                    mass: 1.0,
                    kind: PartKind::Suspension(sp),
                    station: Some(id),
                });
            }
            craft.parts.push(Part {
                mount,
                mass: 1.0,
                kind: PartKind::Rim(RimSpec {
                    radius: 0.1,
                    drive: true,
                    steer: st,
                }),
                station: Some(id),
            });
            craft.parts.push(Part {
                mount,
                mass: 1.0,
                kind: PartKind::Tire(TireSpec {
                    stiffness,
                    ..TireSpec::new(0.05)
                }),
                station: Some(id),
            });
        }
        craft
    }

    #[test]
    fn mixed_and_zero_travel_suspension_stay_stable() {
        // The implicit quarter-car is one model for every strut stiffness, so the configurations the
        // two-path version got wrong are all stable: a vehicle with BOTH sprung and no-suspension axles,
        // and a zero-travel suspension. None launches; all drive over bumps finite and untumbled.
        let terrain = Terrain {
            amplitude: 0.2,
            ..Default::default()
        };
        let sprung = SuspensionSpec::for_cell_size(0.1);
        let zero_travel = SuspensionSpec {
            rest_length: 0.05,
            travel: 0.0,
            rigid: false,
        };
        // Off-road (5e4) and slick (4e5) — the editor's authored tire presets a player actually uses.
        let cases = [
            ("off-road, no suspension", corner_rover(None, None, 5.0e4)),
            ("slick, no suspension", corner_rover(None, None, 4.0e5)),
            (
                "off-road, front sprung + rear rigid",
                corner_rover(Some(sprung), None, 5.0e4),
            ),
            (
                "off-road, front rigid + rear sprung",
                corner_rover(None, Some(sprung), 5.0e4),
            ),
            (
                "off-road, zero-travel front + rear",
                corner_rover(Some(zero_travel), Some(zero_travel), 5.0e4),
            ),
        ];
        for (name, craft) in cases {
            let asm = assemble_rover(&craft, DVec3::ZERO, 9.81).unwrap();
            let drop = asm
                .rover
                .wheels
                .iter()
                .map(|w| w.rest_length + w.radius - w.mount.y)
                .fold(0.0_f64, f64::max);
            let (mut rover, drive) = (asm.rover, asm.drive);
            let spawn_y = terrain.height(0.0, 0.0) + drop;
            rover.body.position = DVec3::new(0.0, spawn_y, 0.0);
            for _ in 0..3_000 {
                rover.step(&terrain, DT); // settle
            }
            // A governed, steady cruise (no-suspension can't safely be floored over bumps — that is
            // the intentional high-speed-bump-fly regime — but at a sane speed it stays planted).
            let target = 8.0;
            let mut max_omega = 0.0_f64;
            let mut min_up = 1.0_f64;
            for step in 0..12_000 {
                let throttle = if rover.body.velocity.length() < target {
                    rover.body.mass * 1.5
                } else {
                    0.0
                };
                for &i in &drive {
                    rover.wheels[i].drive_torque = throttle;
                }
                rover.step(&terrain, DT);
                assert!(
                    rover.body.position.is_finite(),
                    "{name}: non-finite at step {step}"
                );
                if step > 1_000 {
                    max_omega = max_omega.max(rover.body.angular_velocity().length());
                    min_up = min_up.min(body_up(&rover).y);
                }
            }
            assert!(
                rover.height_above_terrain(&terrain) < 3.0,
                "{name}: rover launched (height {:.1} m)",
                rover.height_above_terrain(&terrain)
            );
            assert!(
                rover.body.position.z > 0.5,
                "{name}: did not drive forward (z {:.2})",
                rover.body.position.z
            );
            assert!(
                min_up > 0.6 && max_omega < 4.0,
                "{name}: not planted at cruise (min up.y {min_up:.2}, max_omega {max_omega:.2})"
            );
        }
    }

    #[test]
    fn asymmetric_mixed_rover_settles_after_a_drop() {
        // The user's build: one rear-left suspension, three no-suspension corners, all off-road tires.
        // Dropped, it must come to REST — no indefinite one-wheel hop (playtest bug).
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let s = 0.1;
        let mut craft = VoxelCraft::new(s);
        for x in 0..4 {
            for z in 0..6 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        for (id, (cx, cz, st, has_susp)) in [
            (0, 0, false, true),  // rear-left: the only suspension
            (3, 0, false, false), // rear-right
            (0, 5, true, false),  // front-left
            (3, 5, true, false),  // front-right
        ]
        .into_iter()
        .enumerate()
        {
            let id = id as u32;
            let mount = DVec3::new((cx as f64 + 0.5) * s, -0.05, (cz as f64 + 0.5) * s);
            if has_susp {
                craft.parts.push(Part {
                    mount,
                    mass: 1.0,
                    kind: PartKind::Suspension(SuspensionSpec::for_cell_size(s)),
                    station: Some(id),
                });
            }
            craft.parts.push(Part {
                mount,
                mass: 1.0,
                kind: PartKind::Rim(RimSpec {
                    radius: 0.1,
                    drive: true,
                    steer: st,
                }),
                station: Some(id),
            });
            craft.parts.push(Part {
                mount,
                mass: 1.0,
                kind: PartKind::Tire(TireSpec {
                    stiffness: 5.0e4,
                    ..TireSpec::new(0.05)
                }),
                station: Some(id),
            });
        }
        let asm = assemble_rover(&craft, DVec3::ZERO, 9.81).unwrap();
        let drop = asm
            .rover
            .wheels
            .iter()
            .map(|w| w.rest_length + w.radius - w.mount.y)
            .fold(0.0_f64, f64::max);
        // Drop the rover from `orientation`, settle, and return the tail (vertical-hop, angular) speeds.
        // The hop bug is a sustained *vertical* bounce and/or a building spin — a brakeless wheeled
        // rover legitimately *coasts* horizontally, so horizontal speed is not the signal.
        let run = |orientation: DQuat| {
            let mut rover = asm.rover.clone();
            rover.body.orientation = orientation.normalize();
            rover.body.position = DVec3::new(0.0, terrain.height(0.0, 0.0) + drop + 0.3, 0.0);
            let (mut tail_vy, mut tail_om) = (0.0_f64, 0.0_f64);
            for step in 0..40_000 {
                rover.step(&terrain, DT);
                assert!(rover.body.position.is_finite(), "non-finite at step {step}");
                if step > 36_000 {
                    tail_vy = tail_vy.max(rover.body.velocity.y.abs());
                    tail_om = tail_om.max(rover.body.angular_velocity().length());
                }
            }
            (tail_vy, tail_om, rover.height_above_terrain(&terrain))
        };

        // Upright but tipped onto a front corner (the user's "hops on one front tire"): no sustained
        // vertical hop, no rocking.
        let (vy, om, _) = run(DQuat::from_euler(glam::EulerRot::XYZ, 0.25, 0.0, 0.15));
        assert!(
            vy < 0.1 && om < 0.3,
            "tipped-upright rover keeps hopping/rocking: vy {vy:.3}, omega {om:.2}"
        );

        // Hard inverted (tumble off the ramp at an angle): must NOT spin up — rotation decays — and
        // must not fall through the world. (A fully inverted rover can't self-right; the bugs were it
        // accelerating its spin and sliding forever.)
        let (vy, om, h) = run(DQuat::from_euler(glam::EulerRot::XYZ, 0.35, 0.2, 0.5));
        assert!(
            om < 0.3 && vy < 0.2,
            "inverted rover spun up / keeps hopping: vy {vy:.3}, omega {om:.2}"
        );
        assert!(
            h > -1.0,
            "inverted rover tunnelled through the world: h {h:.2}"
        );
    }

    #[test]
    fn an_upside_down_rover_slides_to_a_stop() {
        // WI 631b playtest: a rover resting upside down must **stop sliding**, not coast forever. It
        // rests on its (orientation-agnostic) wheel-springs; the past-upright traction fade left that
        // contact frictionless, so a tumbled rover kept ~all its horizontal speed indefinitely. The
        // dissipative past-upright Coulomb friction must bleed that off (without tunnelling).
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let craft = chassis_with_stations();
        let mut rover = assemble_rover(&craft, DVec3::ZERO, 9.81).unwrap().rover;
        rover.body.orientation = DQuat::from_rotation_x(std::f64::consts::PI); // fully inverted
        rover.body.position = DVec3::new(0.0, terrain.height(0.0, 0.0) + 1.0, 0.0);
        rover.body.velocity = DVec3::new(6.0, 0.0, 0.0);
        for _ in 0..40_000 {
            rover.step(&terrain, DT);
        }
        assert!(
            body_up(&rover).y < -0.9,
            "test precondition: stays inverted"
        );
        let v_horiz = DVec3::new(rover.body.velocity.x, 0.0, rover.body.velocity.z).length();
        assert!(
            v_horiz < 0.3,
            "inverted rover still sliding: |v_horiz| = {v_horiz:.3}"
        );
        assert!(
            rover.height_above_terrain(&terrain) > -1.0,
            "inverted rover tunnelled"
        );
    }

    #[test]
    fn airborne_rover_conserves_its_spin() {
        // Mid-air, a wheel exerts nothing on the chassis and the angular drag does not act — both are
        // contact effects, not aerodynamics (WI 631a fix). A rover tumbling off the ramp keeps spinning
        // until it lands. Tested on a **no-suspension** build (the playtest failure): its stiff strut
        // would otherwise apply a large spurious force/torque to the chassis mid-air.
        let terrain = Terrain::default();
        let craft = corner_rover(None, None, 5.0e4); // all no-suspension, off-road tires
        let asm = assemble_rover(&craft, DVec3::ZERO, 9.81).unwrap();
        let mut rover = asm.rover;
        rover.body.position = DVec3::new(0.0, terrain.height(0.0, 0.0) + 50.0, 0.0);
        rover.body = rover.body.with_angular_velocity(DVec3::new(0.4, 0.0, 3.0));
        let om0 = rover.body.angular_velocity().length();
        let v0 = rover.body.velocity.length();
        for _ in 0..2_000 {
            rover.step(&terrain, DT); // ~1 s of fall — stays well airborne
        }
        assert!(
            rover.height_above_terrain(&terrain) > 1.0,
            "rover should still be airborne for this test"
        );
        let om1 = rover.body.angular_velocity().length();
        assert!(
            om1 > 0.95 * om0,
            "airborne spin was damped (phantom aerodynamics): {om0:.3} -> {om1:.3}"
        );
        // Horizontal speed is conserved in free fall (only gravity adds downward) — no spurious slowing.
        let v_horiz = (rover.body.velocity.x.powi(2) + rover.body.velocity.z.powi(2)).sqrt();
        assert!(
            v_horiz >= v0 - 0.05,
            "airborne rover was slowed horizontally mid-air"
        );
    }

    #[test]
    fn no_suspension_wheels_ride_on_the_tire_without_launching() {
        // A station with no suspension (rim + tire only) must rest on the tire's compliance and stay
        // grounded — not launch. A no-suspension wheel has no quarter-car axle DOF (a 1e9 "rigid strut"
        // between chassis and a light unsprung mass would ring past the explicit-integration limit); it
        // rides the tire via the legacy series spring, with its mass on the sprung body.
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        for stiffness in [5.0e4, 4.0e5] {
            // off-road, then slick (the presets the user drove)
            let s = 0.1;
            let mut craft = VoxelCraft::new(s);
            for x in 0..4 {
                for z in 0..6 {
                    craft.voxels.push(Voxel {
                        cell: IVec3::new(x, 0, z),
                        material: Material::COMPOSITE,
                    });
                }
            }
            for (cx, cz, st) in [(0, 0, false), (3, 0, false), (0, 5, true), (3, 5, true)] {
                let mount = DVec3::new((cx as f64 + 0.5) * s, -0.05, (cz as f64 + 0.5) * s);
                let id = craft.parts.len() as u32;
                // No suspension part — just rim + tire (the "no suspension" build).
                craft.parts.push(Part {
                    mount,
                    mass: 1.0,
                    kind: PartKind::Rim(RimSpec {
                        radius: 0.1,
                        drive: true,
                        steer: st,
                    }),
                    station: Some(id),
                });
                craft.parts.push(Part {
                    mount,
                    mass: 1.0,
                    kind: PartKind::Tire(TireSpec {
                        stiffness,
                        ..TireSpec::new(0.05)
                    }),
                    station: Some(id),
                });
            }
            let asm = assemble_rover(&craft, DVec3::ZERO, 9.81).unwrap();
            assert!(
                asm.rover.wheels.iter().all(|w| w.rigid_suspension),
                "no-suspension stations should be rigid struts"
            );
            let drop = asm
                .rover
                .wheels
                .iter()
                .map(|w| w.rest_length + w.radius - w.mount.y)
                .fold(0.0_f64, f64::max);
            let mut rover = asm.rover;
            let spawn_y = terrain.height(0.0, 0.0) + drop;
            rover.body.position = DVec3::new(0.0, spawn_y, 0.0);
            let mut max_y = spawn_y;
            let mut max_vy = 0.0_f64;
            for step in 0..6_000 {
                rover.step(&terrain, DT);
                assert!(
                    rover.body.position.is_finite(),
                    "stiffness {stiffness}: non-finite at step {step}"
                );
                max_y = max_y.max(rover.body.position.y);
                if step > 200 {
                    max_vy = max_vy.max(rover.body.velocity.y.abs());
                }
            }
            // It settles onto the tyre (sagging slightly *below* spawn), with **no bounce** — the tyre
            // is critically damped, so it never rises above where it spawned and stops cleanly.
            assert!(
                max_y < spawn_y + 0.01,
                "no-suspension rover bounced on spawn: spawn {spawn_y:.3} → max {max_y:.3} (stiffness {stiffness})"
            );
            assert!(
                max_vy < 0.3,
                "no-suspension rover did not settle on spawn (bouncing): max |vy| {max_vy:.2}"
            );
            assert!(
                rover.height_above_terrain(&terrain) > 0.0,
                "no-suspension rover sank through the ground"
            );
        }
    }

    /// A multi-axle rover: a `3 × (2·axle_pairs+1)` chassis with a left/right wheel station at each of
    /// `axle_pairs` axles (front axle steers, all drive), assembled resting just above flat terrain.
    /// Exercises how the quarter-car scales past four corners (6×6, 8×8, an 18-wheeler).
    fn multi_axle_assembly(axle_pairs: usize, terrain: &Terrain) -> RoverAssembly {
        let s = 0.5;
        let mut craft = VoxelCraft::new(s);
        let len = 2 * axle_pairs + 1;
        for x in 0..3 {
            for z in 0..len as i32 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        let mut id = 0u32;
        for a in 0..axle_pairs {
            let cz = 1 + 2 * a;
            let z = (cz as f64 + 0.5) * s;
            let steer = a == 0; // front axle steers
            for dx in [-0.5, 0.5] {
                let mount = DVec3::new(0.75 + dx, -0.2, z); // outboard of the 3-wide chassis
                craft.parts.push(Part {
                    mount,
                    mass: 0.0,
                    kind: PartKind::Suspension(SuspensionSpec::new()),
                    station: Some(id),
                });
                craft.parts.push(Part {
                    mount,
                    mass: 20.0,
                    kind: PartKind::Rim(RimSpec {
                        radius: 0.25,
                        drive: true,
                        steer,
                    }),
                    station: Some(id),
                });
                craft.parts.push(Part {
                    mount,
                    mass: 20.0,
                    kind: PartKind::Tire(TireSpec::new(0.1)),
                    station: Some(id),
                });
                id += 1;
            }
        }
        let ground = terrain.height(0.0, 0.0);
        assemble_rover(&craft, DVec3::new(0.0, ground + 0.9, 0.0), 9.81)
            .expect("multi-axle ⇒ rover")
    }

    #[test]
    fn many_axle_rovers_stay_stable_and_conserve_mass() {
        // The quarter-car is per-wheel and N-invariant: a 6-wheel (3 axles), 8-wheel (4 axles), and
        // 18-wheel (9 axles) rover all assemble, conserve mass (sprung + Σ unsprung == craft), settle
        // roughly level, and drive over bumps finite and untumbled — no four-corner assumption.
        let terrain = Terrain {
            amplitude: 0.3,
            ..Default::default()
        };
        for pairs in [3, 4, 9] {
            let n_wheels = pairs * 2;
            let flat = Terrain {
                amplitude: 0.0,
                ..Default::default()
            };
            let craft_mass = {
                // Mass conservation on the flat-built rover (independent of the bumpy run below).
                let asm = multi_axle_assembly(pairs, &flat);
                assert_eq!(asm.rover.wheels.len(), n_wheels, "{n_wheels}-wheel count");
                let unsprung: f64 = asm.rover.wheels.iter().map(|w| w.unsprung_mass).sum();
                (asm.rover.body.mass + unsprung, unsprung)
            };
            assert!(
                craft_mass.1 > 0.0 && craft_mass.0 > 0.0,
                "{n_wheels}-wheel: positive masses"
            );

            // Settle on flat ground, then drive over bumps.
            let asm = multi_axle_assembly(pairs, &terrain);
            let (mut rover, drive) = (asm.rover, asm.drive);
            for _ in 0..4_000 {
                rover.step(&terrain, DT);
            }
            assert!(
                body_up(&rover).y > 0.85,
                "{n_wheels}-wheel rover did not rest level: up.y = {}",
                body_up(&rover).y
            );
            for &i in &drive {
                rover.wheels[i].drive_torque = rover.body.mass * 1.5;
            }
            let mut max_omega = 0.0_f64;
            for step in 0..16_000 {
                rover.step(&terrain, DT);
                assert!(
                    rover.body.position.is_finite() && rover.body.velocity.is_finite(),
                    "{n_wheels}-wheel: non-finite at step {step}"
                );
                if step > 1_000 {
                    max_omega = max_omega.max(rover.body.angular_velocity().length());
                }
            }
            assert!(
                max_omega < 6.0,
                "{n_wheels}-wheel rover tumbled over bumps: max_omega = {max_omega}"
            );
            assert!(
                rover.body.position.z > 0.5,
                "{n_wheels}-wheel rover did not drive forward: z = {}",
                rover.body.position.z
            );
        }
    }

    #[test]
    fn quarter_car_reclassifies_unsprung_mass_conserving_total() {
        // Each assembled wheel gains a quarter-car vertical DOF with its own (capped) unsprung mass,
        // reclassified off the sprung body so the total is conserved: sprung + Σ unsprung == craft mass.
        let craft = chassis_with_stations();
        let mp = craft.mass_properties().unwrap();
        let asm = assemble_rover(&craft, DVec3::new(0.0, 5.0, 0.0), 9.81).unwrap();
        let total_unsprung: f64 = asm.rover.wheels.iter().map(|w| w.unsprung_mass).sum();
        assert!(total_unsprung > 0.0, "wheels carry unsprung mass");
        assert!(
            (asm.rover.body.mass + total_unsprung - mp.mass).abs() < 1e-9,
            "sprung {} + unsprung {} != craft {}",
            asm.rover.body.mass,
            total_unsprung,
            mp.mass
        );
        // Every assembled wheel is on the quarter-car path (positive unsprung mass).
        assert!(asm
            .rover
            .wheels
            .iter()
            .all(|w| w.unsprung_mass >= MIN_UNSPRUNG_MASS));
    }

    #[test]
    fn quarter_car_settles_with_the_suspension_under_load() {
        // At rest the suspension carries the sprung weight, so each axle sits compressed below its
        // free length (axle_drop < rest_length) — the quarter-car bears load and settles finite.
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let mut rover = station_assembly(SuspensionSpec::new(), TireSpec::new(0.1), &terrain).rover;
        for _ in 0..6_000 {
            rover.step(&terrain, DT);
        }
        assert!(rover.body.position.is_finite());
        assert!(
            rover.body.velocity.length() < 0.3,
            "did not settle: {:?}",
            rover.body.velocity
        );
        assert!(
            body_up(&rover).y > 0.9,
            "rover should rest roughly level: up.y = {}",
            body_up(&rover).y
        );
        assert!(
            rover.wheels.iter().all(|w| w.unsprung_mass > 0.0),
            "expected the quarter-car path"
        );
        // The suspension bears the sprung weight, so on average each strut sits compressed below its
        // free length (an individual corner may droop in the over-constrained four-spring equilibrium).
        let mean_drop =
            rover.wheels.iter().map(|w| w.axle_drop).sum::<f64>() / rover.wheels.len() as f64;
        assert!(
            mean_drop < rover.wheels[0].rest_length,
            "suspension should carry load (mean compressed): mean drop {mean_drop:.3} vs rest {:.3}",
            rover.wheels[0].rest_length
        );
    }

    #[test]
    fn the_axle_travels_over_bumps_real_wheel_hop() {
        // The payoff: over bumps the wheels visibly travel (the axle compresses and rebounds) rather
        // than tracking the chassis rigidly — real ride dynamics from the per-wheel vertical DOF.
        let terrain = Terrain {
            amplitude: 0.4,
            ..Default::default()
        };
        let tire = TireSpec {
            stiffness: 5.0e4,
            ..TireSpec::new(0.1)
        };
        let asm = station_assembly(SuspensionSpec::new(), tire, &terrain);
        let (mut rover, drive) = (asm.rover, asm.drive);
        for _ in 0..2_000 {
            rover.step(&terrain, DT); // settle
        }
        for &i in &drive {
            rover.wheels[i].drive_torque = rover.body.mass * 2.0;
        }
        let mut min_drop = f64::INFINITY;
        let mut max_drop = f64::NEG_INFINITY;
        for _ in 0..10_000 {
            rover.step(&terrain, DT);
            for w in &rover.wheels {
                min_drop = min_drop.min(w.axle_drop);
                max_drop = max_drop.max(w.axle_drop);
            }
        }
        assert!(rover.body.position.is_finite());
        assert!(
            max_drop - min_drop > 0.05,
            "wheels did not travel over the bumps (glued shadow): range {:.3}",
            max_drop - min_drop
        );
    }

    #[test]
    fn wheel_hop_settles_after_a_drop() {
        // After a disturbance (a small drop) the body and the per-wheel axle hop both decay — no
        // sustained self-excited oscillation.
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let tire = TireSpec {
            stiffness: 5.0e4,
            ..TireSpec::new(0.1)
        };
        let asm = station_assembly(SuspensionSpec::new(), tire, &terrain);
        let mut rover = asm.rover;
        rover.body.position.y += 0.3; // drop it above its settled height
        for _ in 0..8_000 {
            rover.step(&terrain, DT);
        }
        assert!(rover.body.position.is_finite());
        assert!(
            rover.body.velocity.length() < 0.3,
            "body did not settle after the drop: {:?}",
            rover.body.velocity
        );
        assert!(
            rover.wheels.iter().all(|w| w.axle_drop_vel.abs() < 0.2),
            "axle hop did not settle: {:?}",
            rover
                .wheels
                .iter()
                .map(|w| w.axle_drop_vel)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn quarter_car_is_stable_across_tire_rates_at_editor_scale() {
        // The scale-relative kraken guard: at a light editor scale, the quarter-car stays finite and
        // bounded while driving hard over bumps across the whole tire-rate range — from a soft tire up
        // to the effectively-rigid default (which the mass-relative ceiling caps to a stable rate).
        let terrain = Terrain {
            amplitude: 0.3,
            ..Default::default()
        };
        for stiffness in [2.0e4, 1.0e5, 4.0e5, 1.0e9] {
            let tire = TireSpec {
                stiffness,
                ..TireSpec::new(0.1)
            };
            let asm = station_assembly(SuspensionSpec::for_cell_size(0.5), tire, &terrain);
            let (mut rover, drive) = (asm.rover, asm.drive);
            for &i in &drive {
                rover.wheels[i].drive_torque = rover.body.mass * 3.0;
            }
            let mut max_omega = 0.0_f64;
            for step in 0..15_000 {
                rover.step(&terrain, DT);
                assert!(
                    rover.body.position.is_finite() && rover.body.velocity.is_finite(),
                    "stiffness {stiffness}: non-finite at step {step}"
                );
                if step > 1_000 {
                    max_omega = max_omega.max(rover.body.angular_velocity().length());
                }
            }
            assert!(
                max_omega < 6.0,
                "stiffness {stiffness}: tumbled over bumps, max_omega = {max_omega}"
            );
        }
    }

    #[test]
    fn assemble_rover_builds_wheels_and_groups() {
        let craft = chassis_with_wheels();
        let mp = craft.mass_properties().unwrap();
        let asm = assemble_rover(&craft, DVec3::new(0.0, 5.0, 0.0), 9.81).unwrap();

        assert_eq!(asm.rover.wheels.len(), 4);
        assert_eq!(asm.drive, vec![0, 1, 2, 3]); // all four drive
        assert_eq!(asm.steer, vec![2, 3]); // only the front pair steer
                                           // No power devices ⇒ the default (electric) powertrain.
        assert_eq!(asm.powertrain.label(), "charge");
        // The body is the **sprung** mass (WI 631a): the per-corner unsprung masses are reclassified
        // off it onto the axles, conserving total mass (sprung + Σ unsprung == craft mass).
        let total_unsprung: f64 = asm.rover.wheels.iter().map(|w| w.unsprung_mass).sum();
        assert!(total_unsprung > 0.0, "wheels should carry unsprung mass");
        assert!(
            (asm.rover.body.mass + total_unsprung - mp.mass).abs() < 1e-9,
            "sprung + unsprung must equal the craft mass"
        );
        // Wheel mounts are CoM-relative; the sprung CoM shifts only slightly off the full CoM once the
        // low-mounted wheels are reclassified, so the mount is near the full-CoM-relative position.
        let expected = DVec3::new(-1.0, -0.2, -2.0) - mp.center_of_mass;
        assert!((asm.rover.wheels[0].mount - expected).length() < 0.1);
    }

    #[test]
    fn assemble_rover_is_none_without_wheels() {
        let mut craft = VoxelCraft::new(0.5);
        craft.voxels.push(Voxel {
            cell: IVec3::new(0, 0, 0),
            material: Material::COMPOSITE,
        });
        assert!(assemble_rover(&craft, DVec3::ZERO, 9.81).is_none());
        // Empty lattice (no mass) is also None.
        assert!(assemble_rover(&VoxelCraft::new(0.5), DVec3::ZERO, 9.81).is_none());
    }

    #[test]
    fn assembled_rover_drives_forward_without_wheelie() {
        // A small (0.1 m cell) rover, assembled the way the workshop builds one: a flat chassis with
        // four corner wheels just below it, sized to the cell and auto-tuned to the mass. It must
        // actually drive forward (+Z) on flat ground and not flip into a perpetual wheelie.
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let s = 0.1;
        let mut craft = VoxelCraft::new(s);
        for x in 0..4 {
            for z in 0..6 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        for (cx, cz, steer) in [(0, 0, false), (3, 0, false), (0, 5, true), (3, 5, true)] {
            let mount = DVec3::new((cx as f64 + 0.5) * s, -0.1, (cz as f64 + 0.5) * s);
            craft.parts.push(Part {
                mount,
                mass: 3.0,
                kind: PartKind::Wheel(WheelPart::for_cell_size(s, true, steer)),
                station: None,
            });
        }
        let mass = craft.mass_properties().unwrap().mass;
        let asm = assemble_rover(&craft, DVec3::ZERO, 9.81).unwrap();
        let mut rover = asm.rover;
        // Rest it on the ground (lowest wheel at free length), then settle.
        let drop = rover
            .wheels
            .iter()
            .map(|w| w.rest_length + w.radius - w.mount.y)
            .fold(0.0_f64, f64::max);
        rover.body.position = DVec3::new(0.0, terrain.height(0.0, 0.0) + drop, 0.0);
        for _ in 0..4_000 {
            rover.step(&terrain, SUBSTEP_DT);
        }
        let z0 = rover.body.position.z;

        for &i in &asm.drive {
            rover.wheels[i].drive_torque = mass * 4.0;
        }
        let mut max_pitch = 0.0_f64;
        for _ in 0..8_000 {
            rover.step(&terrain, SUBSTEP_DT);
            max_pitch = max_pitch.max(rover.body.angular_velocity().x.abs());
        }
        assert!(rover.body.position.is_finite());
        assert!(
            rover.body.position.z - z0 > 0.3,
            "rover did not drive forward: Δz = {}",
            rover.body.position.z - z0
        );
        // Some nose-up under acceleration is fine; a perpetual wheelie/flip is not.
        assert!(max_pitch < 3.0, "excessive pitch (wheelie): {max_pitch}");
    }

    #[test]
    fn rover_stops_at_an_obstacle_without_tunnelling() {
        use crate::collision::{craft_bounds, craft_collision_shape, BoxShape, CollisionShape};
        use crate::contact::{body_contact_wrench, ContactParams};

        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let s = 0.1;
        let mut craft = VoxelCraft::new(s);
        for x in 0..4 {
            for z in 0..6 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        for (cx, cz, st) in [(0, 0, false), (3, 0, false), (0, 5, true), (3, 5, true)] {
            craft.parts.push(Part {
                mount: DVec3::new((cx as f64 + 0.5) * s, -0.1, (cz as f64 + 0.5) * s),
                mass: 3.0,
                kind: PartKind::Wheel(WheelPart::for_cell_size(s, true, st)),
                station: None,
            });
        }
        let mp = craft.mass_properties().unwrap();
        let com = mp.center_of_mass;
        let asm = assemble_rover(&craft, DVec3::ZERO, 9.81).unwrap();
        let mut rover = asm.rover;
        let drop = rover
            .wheels
            .iter()
            .map(|w| w.rest_length + w.radius - w.mount.y)
            .fold(0.0_f64, f64::max);
        rover.body.position = DVec3::new(0.0, terrain.height(0.0, 0.0) + drop, 0.0);

        let rover_shape = craft_collision_shape(&craft);
        let rover_bounds = craft_bounds(&craft);

        // A wall across the path at z ≈ 3 m.
        let wall_z = 3.0;
        let obstacle = ActiveBody::new(
            DVec3::new(0.0, 0.6, wall_z),
            DVec3::ZERO,
            1.0e12,
            DMat3::IDENTITY,
        );
        let obs_shape = CollisionShape::CuboidCompound(vec![BoxShape {
            center: DVec3::ZERO,
            half_extents: DVec3::new(3.0, 1.2, 0.2),
        }]);
        let obs_bounds = Some(crate::collision::Bounds {
            aabb_min: DVec3::new(-3.0, -1.2, -0.2),
            aabb_max: DVec3::new(3.0, 1.2, 0.2),
            sphere_center: DVec3::ZERO,
            sphere_radius: DVec3::new(3.0, 1.2, 0.2).length(),
        });
        let params = ContactParams::default();

        let mut max_vy = 0.0_f64;
        let mut max_z = f64::NEG_INFINITY; // closest approach to the wall over the run
        for step in 0..40_000 {
            if step > 3_000 {
                for &i in &asm.drive {
                    rover.wheels[i].drive_torque = mp.mass * 3.0; // drive into the wall
                }
            }
            let (wrench, _) = body_contact_wrench(
                &rover.body,
                &rover_shape,
                rover_bounds,
                com,
                &obstacle,
                &obs_shape,
                obs_bounds,
                DVec3::ZERO,
                &params,
            );
            rover.apply_external(wrench.0, wrench.1);
            rover.step(&terrain, SUBSTEP_DT);
            assert!(rover.body.position.is_finite(), "non-finite at {step}");
            max_z = max_z.max(rover.body.position.z);
            if step > 5_000 {
                max_vy = max_vy.max(rover.body.velocity.y.abs());
            }
        }
        // It advanced to the wall but did **not** tunnel through it (front face ≈ 2.8 m), and the
        // contact never launched it. (Tracked by closest approach, so a later yaw/spin-out — a light
        // rover ramming an off-centre wall is now free to slew, no longer pinned by the old absolute
        // angular drag — does not mask whether it tunnelled.)
        assert!(max_z > 0.3, "rover did not drive toward the wall: {max_z}");
        assert!(max_z < 2.9, "rover tunnelled through the wall: {max_z}");
        assert!(
            max_vy < 5.0,
            "obstacle contact launched the rover: {max_vy}"
        );
    }

    #[test]
    fn set_steer_counter_steers_behind_com() {
        let s = 0.1;
        let mut craft = VoxelCraft::new(s);
        for x in 0..4 {
            for z in 0..6 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        // Four corner wheels, all steerable (front at z≈max, rear at z≈min).
        for (cx, cz) in [(0, 0), (3, 0), (0, 5), (3, 5)] {
            craft.parts.push(Part {
                mount: DVec3::new((cx as f64 + 0.5) * s, -0.1, (cz as f64 + 0.5) * s),
                mass: 3.0,
                kind: PartKind::Wheel(WheelPart::for_cell_size(s, true, true)),
                station: None,
            });
        }
        let mut rover = assemble_rover(&craft, DVec3::ZERO, 9.81).unwrap().rover;
        let steer: Vec<usize> = (0..rover.wheels.len()).collect();
        let max_angle = 0.4;

        rover.set_steer(1.0, max_angle, &steer);
        for w in &rover.wheels {
            // Steer sign follows the longitudinal offset → rear (z<0) inverts vs front (z>0).
            assert!((w.steer.signum() - w.mount.z.signum()).abs() < 1e-9);
            assert!(w.steer.abs() <= max_angle + 1e-9);
        }
        let max = rover
            .wheels
            .iter()
            .map(|w| w.steer.abs())
            .fold(0.0, f64::max);
        assert!(
            (max - max_angle).abs() < 1e-6,
            "farthest wheel hits max: {max}"
        );
        let front = rover.wheels.iter().find(|w| w.mount.z > 0.0).unwrap().steer;
        let rear = rover.wheels.iter().find(|w| w.mount.z < 0.0).unwrap().steer;
        assert!(front > 0.0 && rear < 0.0, "front {front}, rear {rear}");

        // Zero input → all straight.
        rover.set_steer(0.0, max_angle, &steer);
        assert!(rover.wheels.iter().all(|w| w.steer == 0.0));
    }

    #[test]
    fn engine_and_tank_make_a_combustion_powertrain() {
        // An Engine device on a rover is a combustion engine (drivetrain), fed by a Tank (WI 609).
        let mut craft = chassis_with_wheels();
        craft.devices.push(Device::structural(
            IVec3::new(1, 0, 2),
            100.0,
            DeviceKind::Engine,
        ));
        craft.devices.push(Device::structural(
            IVec3::new(1, 0, 3),
            80.0,
            DeviceKind::Tank,
        ));
        let asm = assemble_rover(&craft, DVec3::ZERO, 9.81).unwrap();
        assert_eq!(asm.rover.wheels.len(), 4);
        assert_eq!(asm.powertrain.label(), "fuel");
        assert!(asm.powertrain.reservoir.amount > 0.0);
    }

    /// The production sub-step (see [`super::SUBSTEP_DT`]).
    const DT: f64 = SUBSTEP_DT;

    /// A modest four-wheel rover from a voxel chassis, placed `drop` metres above
    /// the terrain at world `(ox, _, oz)`.
    fn rover_at(terrain: &Terrain, ox: f64, oz: f64, drop: f64) -> Rover {
        let mut craft = VoxelCraft::new(0.5);
        for x in 0..3 {
            for z in 0..5 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        let mp = craft.mass_properties().unwrap();
        let ground = terrain.height(ox, oz);
        let body = ActiveBody::from_mass_properties(
            DVec3::new(ox, ground + 0.9 + drop, oz),
            DVec3::ZERO,
            &mp,
        );
        let wheels = vec![
            Wheel::new(DVec3::new(-1.0, -0.2, -2.0)),
            Wheel::new(DVec3::new(1.0, -0.2, -2.0)),
            Wheel::new(DVec3::new(-1.0, -0.2, 2.0)),
            Wheel::new(DVec3::new(1.0, -0.2, 2.0)),
        ];
        Rover::new(body, wheels, 9.81)
    }

    #[test]
    fn tire_is_zero_at_zero_slip_and_saturates() {
        assert_eq!(tire_forces(0.0, 0.0, 1_000.0, C_LONG, C_LAT), (0.0, 0.0));
        let (fx, _) = tire_forces(5.0, 0.0, 1_000.0, C_LONG, C_LAT); // large slip → saturates near fmax
        assert!(fx > 900.0 && fx <= 1_000.0 + 1e-9);
        // Friction ellipse: combined never exceeds fmax.
        let (fx, fy) = tire_forces(5.0, 1.2, 1_000.0, C_LONG, C_LAT);
        assert!((fx * fx + fy * fy).sqrt() <= 1_000.0 + 1e-6);
    }

    #[test]
    fn tire_force_scales_with_surface_material() {
        // fmax = μ·N, so ice (low μ) yields a smaller saturated force than bedrock.
        let n = 5_000.0;
        let ice = tire_forces(5.0, 0.0, SurfaceMaterial::ICE.friction * n, C_LONG, C_LAT).0;
        let bedrock = tire_forces(
            5.0,
            0.0,
            SurfaceMaterial::BEDROCK.friction * n,
            C_LONG,
            C_LAT,
        )
        .0;
        assert!(ice < bedrock);
    }

    #[test]
    fn rover_settles_on_suspension_without_blowing_up() {
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let mut rover = rover_at(&terrain, 0.0, 0.0, 0.3);
        for _ in 0..4_000 {
            rover.step(&terrain, DT);
        }
        // Comes to rest at a finite height, no launch, no NaN.
        assert!(
            rover.body.velocity.length() < 0.2,
            "did not settle: {:?}",
            rover.body.velocity
        );
        let h = rover.height_above_terrain(&terrain);
        assert!(
            h.is_finite() && h > 0.0 && h < 2.0,
            "resting height off: {h}"
        );
    }

    #[test]
    fn airborne_rover_is_in_free_fall() {
        let terrain = Terrain::default();
        let mut rover = rover_at(&terrain, 0.0, 0.0, 100.0); // high above ground
        let v0 = rover.body.velocity.y;
        rover.step(&terrain, DT);
        // Only gravity acts (no contact force); downward velocity increases.
        assert!(rover.body.velocity.y < v0);
        assert!(rover.body.velocity.is_finite());
    }

    #[test]
    fn drive_torque_accelerates_the_rover_forward() {
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let mut rover = rover_at(&terrain, 0.0, 0.0, 0.0);
        for _ in 0..1_500 {
            rover.step(&terrain, DT); // settle
        }
        for w in &mut rover.wheels {
            w.drive_torque = 4_000.0;
        }
        for _ in 0..3_000 {
            rover.step(&terrain, DT);
        }
        assert!(
            rover.body.velocity.z > 0.5,
            "rover did not drive forward: {:?}",
            rover.body.velocity
        );
        assert!(
            rover.height_above_terrain(&terrain) < 2.0,
            "rover left the ground"
        );
    }

    #[test]
    fn no_launch_driving_over_bumps_at_planetary_offset() {
        // The kraken test: drive over varied terrain at a large world offset (where
        // rendering would rebase) and assert the contact never launches the rover.
        let terrain = Terrain {
            amplitude: 0.3,
            ..Default::default()
        };
        let (ox, oz) = (6_378_000.0, -1_200_000.0);
        let mut rover = rover_at(&terrain, ox, oz, 0.2);
        // Cruise at a governed, modest speed so the test isolates contact
        // stability from a fast rover legitimately jumping off crests.
        let target_speed = 6.0;
        let dt = DT;
        let mut max_vy = 0.0_f64;
        let mut max_h = f64::MIN;
        let mut max_jitter = 0.0_f64;
        for step in 0..20_000 {
            let throttle = if rover.body.velocity.z < target_speed {
                500.0
            } else {
                0.0
            };
            for w in &mut rover.wheels {
                w.drive_torque = throttle;
            }
            rover.step(&terrain, dt);
            let h = rover.height_above_terrain(&terrain);
            assert!(h.is_finite(), "non-finite rover height");
            // Ignore the initial settle-in; then the cruise must hug the terrain.
            if step > 4_000 {
                max_vy = max_vy.max(rover.body.velocity.y.abs());
                max_h = max_h.max(h);
                max_jitter = max_jitter.max(rover.contact_jitter);
            }
        }
        // No launch: vertical speed stays small while cruising over the bumps
        // (a kraken launch sends it to tens or hundreds of m/s). The rover hugs
        // the terrain. (Tumbling under steady throttle is covered separately.)
        assert!(max_vy < 3.0, "rover was launched: max |v_y| = {max_vy}");
        assert!(max_h < 3.5, "rover left the terrain: max height {max_h}");
        assert!(max_jitter.is_finite());
    }

    #[test]
    fn steering_does_not_cause_continuous_spin() {
        // Mimic the app: floor the throttle AND hold steer. The rover may turn,
        // but it must not spin out into a continuous loop.
        let terrain = Terrain {
            amplitude: 0.6,
            ..Default::default()
        };
        let mut rover = rover_at(&terrain, 6_378_000.0, -1_200_000.0, 0.2);
        for w in &mut rover.wheels {
            w.drive_torque = 2_500.0;
        }
        rover.wheels[2].steer = 0.3;
        rover.wheels[3].steer = 0.3;
        let mut max_wx = 0.0_f64;
        let mut max_wy = 0.0_f64;
        let mut max_wz = 0.0_f64;
        for step in 0..20_000 {
            rover.step(&terrain, DT);
            if step > 2_000 {
                let w = rover.body.angular_velocity();
                max_wx = max_wx.max(w.x.abs());
                max_wy = max_wy.max(w.y.abs());
                max_wz = max_wz.max(w.z.abs());
            }
        }
        // Held steer makes the rover circle (a controlled turn), but the per-axis
        // angular velocity must stay bounded — the oversteer spin-out bug ran the
        // yaw rate (w.y) away to ~5 rad/s; a controlled turn keeps it well under.
        assert!(max_wx < 2.0, "roll runaway: wx={max_wx}");
        assert!(max_wy < 2.5, "yaw runaway (spin-out): wy={max_wy}");
        assert!(max_wz < 2.0, "pitch runaway: wz={max_wz}");
    }

    #[test]
    fn high_speed_over_bumps_stays_finite() {
        // The app scenario: floor it on the gentle rolling terrain. At ~100 m/s the
        // rover flies off crests (intended craziness), but it must stay finite and
        // not spin out of control — it recovers under angular drag.
        let terrain = Terrain {
            amplitude: 0.7,
            wavelength: 55.0,
            ..Default::default()
        };
        let mut rover = rover_at(&terrain, 6_378_000.0, -1_200_000.0, 0.2);
        for w in &mut rover.wheels {
            w.drive_torque = 2_500.0;
        }
        let mut max_omega = 0.0_f64;
        let mut max_air = 0.0_f64;
        let mut top_speed = 0.0_f64;
        for step in 0..40_000 {
            rover.step(&terrain, DT);
            assert!(
                rover.body.position.is_finite() && rover.body.velocity.is_finite(),
                "rover state went non-finite at step {step}"
            );
            if step > 1_000 {
                max_omega = max_omega.max(rover.body.angular_velocity().length());
                max_air = max_air.max(rover.height_above_terrain(&terrain));
                top_speed = top_speed.max(rover.body.velocity.length());
            }
        }
        // It reaches a high speed and catches real air over the crests…
        assert!(top_speed > 75.0, "top speed too low: {top_speed}");
        assert!(max_air > 0.8, "rover did not catch air: {max_air}");
        // …but stays finite and recovers rather than spinning endlessly: the jitter-selective damper
        // (WI 611) still caps the rough-landing buzz, so even reckless straight-line bump-flying keeps
        // bounded angular velocity. (A *steered* hard turn is the intentional-rollover case.)
        assert!(
            max_omega < 6.0,
            "rover tumbled at high speed over bumps: {max_omega}"
        );
    }

    #[test]
    fn full_throttle_reaches_high_speed_without_tumbling() {
        // On flat ground, flooring it accelerates to a high top speed (~100 m/s)
        // and stays stable — no tumbling on any axis.
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let mut rover = rover_at(&terrain, 6_378_000.0, -1_200_000.0, 0.2);
        for w in &mut rover.wheels {
            w.drive_torque = 2_500.0;
        }
        let mut max_omega = 0.0_f64;
        for step in 0..60_000 {
            rover.step(&terrain, DT);
            if step > 1_000 {
                max_omega = max_omega.max(rover.body.angular_velocity().length());
            }
        }
        let speed = rover.body.velocity.length();
        let w = rover.body.angular_velocity();
        // Genuinely fast, and stable: bounded per-axis angular velocity, no NaN.
        assert!(speed > 60.0 && speed < 130.0, "top speed off: {speed}");
        assert!(
            max_omega < 2.0,
            "rover tumbled at speed: max |omega| = {max_omega}"
        );
        assert!(
            w.x.abs() < 1.0 && w.y.abs() < 1.0 && w.z.abs() < 1.0,
            "per-axis spin at speed: {w:?}"
        );
    }

    /// A tall, narrow (roll-prone) four-wheel rover resting on flat ground at the origin: high CoM
    /// over a short track, so a hard turn at speed can tip it (WI 611).
    fn roll_prone_rover() -> (Rover, Vec<usize>) {
        let s = 0.5;
        let mut craft = VoxelCraft::new(s);
        // 1 cell wide (x), 3 long (z), 4 tall (y): a high centre of mass.
        for y in 0..4 {
            for z in 0..3 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(0, y, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        // Four wheels on a narrow track (x ≈ ±0.15 about the chassis centre x≈0.25), at the bottom.
        let mounts = [
            (DVec3::new(0.1, -0.3, 0.25), false),
            (DVec3::new(0.4, -0.3, 0.25), false),
            (DVec3::new(0.1, -0.3, 1.25), true),
            (DVec3::new(0.4, -0.3, 1.25), true),
        ];
        for (mount, steer) in mounts {
            craft.parts.push(Part {
                mount,
                mass: 8.0,
                kind: PartKind::Wheel(WheelPart::for_cell_size(s, true, steer)),
                station: None,
            });
        }
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let asm = assemble_rover(&craft, DVec3::ZERO, 9.81).unwrap();
        let mut rover = asm.rover;
        let drop = rover
            .wheels
            .iter()
            .map(|w| w.rest_length + w.radius - w.mount.y)
            .fold(0.0_f64, f64::max);
        rover.body.position = DVec3::new(0.0, terrain.height(0.0, 0.0) + drop, 0.0);
        for _ in 0..4_000 {
            rover.step(&terrain, DT);
        }
        (rover, asm.drive)
    }

    fn body_up(rover: &Rover) -> DVec3 {
        DMat3::from_quat(rover.body.orientation) * DVec3::Y
    }

    #[test]
    fn sharp_turn_at_speed_can_roll() {
        // Provoked driving (WI 611): bring a roll-prone rover up to speed, then jam a hard steer.
        // The lateral force at the tyres, acting through the high CoM over the narrow track, must be
        // able to tip it past upright — the whole point of relaxing the blanket angular damper.
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let (mut rover, drive) = roll_prone_rover();
        let mass = rover.body.mass;
        // Accelerate to speed.
        for &i in &drive {
            rover.wheels[i].drive_torque = mass * 6.0;
        }
        for _ in 0..6_000 {
            rover.step(&terrain, DT);
        }
        // Jam a hard steer on the front pair and hold the throttle.
        rover.wheels[2].steer = 0.6;
        rover.wheels[3].steer = 0.6;
        let mut min_up_y = 1.0_f64;
        for _ in 0..20_000 {
            rover.step(&terrain, DT);
            min_up_y = min_up_y.min(body_up(&rover).y);
        }
        assert!(rover.body.position.is_finite());
        assert!(
            min_up_y < 0.5,
            "roll-prone rover did not tip on a hard turn at speed: min up.y = {min_up_y}"
        );
    }

    #[test]
    fn nominal_cruise_does_not_spuriously_roll() {
        // The other side (WI 611): a normal low-wide rover cruising at a moderate speed with a gentle
        // steer over mild terrain must stay upright — no spurious flip from the relaxed damper.
        let terrain = Terrain {
            amplitude: 0.25,
            ..Default::default()
        };
        let mut rover = rover_at(&terrain, 0.0, 0.0, 0.2);
        for _ in 0..2_000 {
            rover.step(&terrain, DT); // settle
        }
        let target_speed = 8.0;
        let mut min_up_y = 1.0_f64;
        for step in 0..20_000 {
            let throttle = if rover.body.velocity.length() < target_speed {
                1_500.0
            } else {
                0.0
            };
            for w in &mut rover.wheels {
                w.drive_torque = throttle;
            }
            // A gentle, steady steer (a wide easy turn).
            rover.wheels[2].steer = 0.08;
            rover.wheels[3].steer = 0.08;
            rover.step(&terrain, DT);
            if step > 2_000 {
                min_up_y = min_up_y.min(body_up(&rover).y);
            }
        }
        assert!(
            min_up_y > 0.85,
            "nominal cruise spuriously rolled: min up.y = {min_up_y}"
        );
    }

    #[test]
    fn braking_locks_wheels_and_stops_the_rover() {
        // WI 618: a strong brake must cleanly stop the rover (lock the wheels), not chatter the spin.
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let s = 0.1;
        let mut craft = VoxelCraft::new(s);
        for x in 0..4 {
            for z in 0..6 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        for (cx, cz, st) in [(0, 0, false), (3, 0, false), (0, 5, true), (3, 5, true)] {
            craft.parts.push(Part {
                mount: DVec3::new((cx as f64 + 0.5) * s, -0.1, (cz as f64 + 0.5) * s),
                mass: 3.0,
                kind: PartKind::Wheel(WheelPart::for_cell_size(s, true, st)),
                station: None,
            });
        }
        let mass = craft.mass_properties().unwrap().mass;
        let asm = assemble_rover(&craft, DVec3::ZERO, 9.81).unwrap();
        let mut rover = asm.rover;
        let drop = rover
            .wheels
            .iter()
            .map(|w| w.rest_length + w.radius - w.mount.y)
            .fold(0.0_f64, f64::max);
        rover.body.position = DVec3::new(0.0, terrain.height(0.0, 0.0) + drop, 0.0);

        // Drive up to speed.
        for &i in &asm.drive {
            rover.wheels[i].drive_torque = mass * 4.0;
        }
        for _ in 0..6_000 {
            rover.step(&terrain, SUBSTEP_DT);
        }
        assert!(
            rover.body.velocity.z > 1.0,
            "rover should be moving before braking: {}",
            rover.body.velocity.z
        );

        // Cut throttle, slam the brake (as the app does).
        for w in &mut rover.wheels {
            w.drive_torque = 0.0;
            w.brake = mass * 35.0;
        }
        for _ in 0..4_000 {
            rover.step(&terrain, SUBSTEP_DT);
        }
        // It came to a near-stop, and the (locked) wheels are not chattering at high spin.
        assert!(
            rover.body.velocity.length() < 0.3,
            "brake did not stop the rover: v = {:?}",
            rover.body.velocity
        );
        assert!(
            rover.wheels.iter().all(|w| w.spin.abs() < 5.0),
            "wheels still spinning under full brake (chatter): {:?}",
            rover.wheels.iter().map(|w| w.spin).collect::<Vec<_>>()
        );
    }

    #[test]
    fn shear_on_impact_is_speed_and_side_graded() {
        // WI 618: shearing is keyed to impact (closing) speed vs. each mount's rated speed, and to
        // which side faces the obstacle. `into_obstacle = +Z` means the front wheels (mount.z > 0)
        // face the impact.
        let craft = chassis_with_wheels();
        let mut rover = assemble_rover(&craft, DVec3::new(0.0, 5.0, 0.0), 9.81)
            .unwrap()
            .rover;
        // Pin every mount's rated speed so the test is independent of the calibration constants.
        for w in &mut rover.wheels {
            w.shear_speed = 6.0;
        }
        let into = DVec3::Z;

        // Leaning (zero closing speed) never shears — the key fix.
        let out = rover.shear_on_impact(0.0, into);
        assert!(out.sheared.is_empty());

        // Slow nudge (below rating even for a head-on wheel): nothing shears.
        let out = rover.shear_on_impact(4.0, into);
        assert!(out.sheared.is_empty());
        assert!(rover.wheels.iter().all(|w| !w.inert));
        assert!(out.peak_wheel.is_some());
        assert!((out.peak_capacity - 6.0).abs() < 1e-9);

        // Fast forward hit: the front wheels (facing +Z) exceed their rating; rear survive.
        let out = rover.shear_on_impact(10.0, into);
        assert!(!out.sheared.is_empty());
        assert!(
            out.sheared.iter().all(|&i| rover.wheels[i].mount.z > 0.0),
            "only front wheels shear on a fast forward hit"
        );
        assert!(rover.wheels.iter().any(|w| !w.inert), "rear wheels survive");

        // Very fast hit: even the far (rear, share = bias) wheels exceed their rating → all shear.
        let out = rover.shear_on_impact(40.0, into);
        assert!(!out.sheared.is_empty());
        assert!(
            rover.wheels.iter().all(|w| w.inert),
            "a very fast hit shears every wheel"
        );
    }

    #[test]
    fn wheelless_chassis_rests_on_its_belly_without_tunnelling() {
        // WI 618: with every wheel sheared inert, the chassis settles onto its belly contact rather
        // than tunnelling through the terrain — it drops a little (wheels gone), then comes to rest at
        // a finite, bounded height instead of falling away.
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let mut rover = rover_at(&terrain, 0.0, 0.0, 0.0);
        for _ in 0..2_000 {
            rover.step(&terrain, DT); // settle on its wheels
        }
        let h0 = rover.height_above_terrain(&terrain);
        for w in &mut rover.wheels {
            w.inert = true;
        }
        for _ in 0..4_000 {
            rover.step(&terrain, DT);
        }
        let h1 = rover.height_above_terrain(&terrain);
        assert!(rover.body.position.is_finite());
        // It sat down (lost its wheels) but did NOT tunnel away.
        assert!(h1 < h0, "chassis did not sit down: h0={h0}, h1={h1}");
        assert!(h1 > -0.5, "chassis tunnelled through the ground: h1={h1}");
        // And it came to rest (belly damping + friction arrested it).
        assert!(
            rover.body.velocity.length() < 0.5,
            "wheelless chassis did not settle: v={:?}",
            rover.body.velocity
        );
    }

    #[test]
    fn stepping_is_deterministic() {
        let terrain = Terrain::default();
        let mut a = rover_at(&terrain, 100.0, 50.0, 0.2);
        let mut b = rover_at(&terrain, 100.0, 50.0, 0.2);
        for _ in 0..1_000 {
            a.step(&terrain, DT);
            b.step(&terrain, DT);
        }
        assert_eq!(a.body.position, b.body.position);
    }

    #[test]
    fn mass_and_inertia_come_from_the_voxel_lattice() {
        let terrain = Terrain::default();
        let mut craft = VoxelCraft::new(0.5);
        for x in 0..3 {
            for z in 0..5 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        let mp = craft.mass_properties().unwrap();
        let rover = rover_at(&terrain, 0.0, 0.0, 0.0);
        assert!((rover.body.mass - mp.mass).abs() < 1e-9);
    }
}
