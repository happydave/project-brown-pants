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
use crate::voxel::{DeviceKind, PartKind, VoxelCraft};
use glam::{DMat3, DQuat, DVec3};

/// Longitudinal slip stiffness (shape of the slip-ratio → force curve).
const C_LONG: f64 = 5.0;
/// Lateral slip stiffness (shape of the slip-angle → force curve).
const C_LAT: f64 = 4.0;
const EPS: f64 = 1e-3;
/// Aerodynamic drag coefficient (N·s²/m²): gives the rover a finite (but high)
/// top speed of roughly 100 m/s.
const DRAG: f64 = 0.55;
/// Baseline angular drag (N·m·s) under *smooth* contact (WI 611): a gentle restoring term that keeps
/// nominal driving tidy without preventing an intentional flip — far weaker than the old blanket
/// damper, so genuine rotation survives.
const BASELINE_ANGULAR_DRAG: f64 = 40.0;
/// Jitter-scaled angular drag (N·m·s) (WI 611): added in proportion to the contact-jitter ratio
/// (per-substep normal-force change ÷ weight), so the numerical "kraken" buzz — which oscillates the
/// normal force every sub-step — is damped hard, while a genuine, smooth-contact rotation (an
/// intentional rollover) passes nearly undamped. Replaces the old blanket `ANGULAR_DRAG`.
const JITTER_ANGULAR_DRAG: f64 = 1_500.0;
/// Cap on the jitter ratio feeding `JITTER_ANGULAR_DRAG`, so a single hard landing spike can't make
/// the damping unboundedly large.
const JITTER_RATIO_CAP: f64 = 3.0;

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

        for w in &mut self.wheels {
            // A sheared-off wheel (WI 618) exerts nothing and does not spin.
            if w.inert {
                continue;
            }
            let hub = self.body.position + r * w.mount;
            let hub_vel =
                self.body.velocity + self.body.angular_velocity().cross(hub - self.body.position);

            let ground = terrain.height(hub.x, hub.z);
            let clearance = hub.y - ground;
            let target = w.rest_length + w.radius;
            let compression = (target - clearance).clamp(0.0, target);

            if compression <= 0.0 {
                // Airborne: the wheel spins freely under drive/brake, no contact force.
                w.spin += motor_torque(w) / w.wheel_inertia * dt;
                w.spin = apply_brake(w.spin, w.brake, w.wheel_inertia, dt);
                continue;
            }

            let normal = terrain.normal(hub.x, hub.z);
            // Closing speed along the contact normal — positive when the hub
            // approaches the surface. This captures both vertical motion and the
            // compression induced by driving forward over a slope (∇h · v), so the
            // damping sees the forward-over-bump rate and the contact stays stable.
            let compression_rate = -hub_vel.dot(normal);
            let n =
                (w.stiffness * compression + w.damping * compression_rate).clamp(0.0, w.max_force);
            total_normal += n;

            // Ground tangent basis: steered heading projected perpendicular to the normal.
            let steer_rot = DQuat::from_axis_angle(body_up, w.steer);
            let heading = steer_rot * body_fwd;
            let forward = (heading - normal * heading.dot(normal)).normalize_or_zero();
            let lateral = normal.cross(forward);

            let v_long = hub_vel.dot(forward);
            let v_lat = hub_vel.dot(lateral);
            let wheel_speed = w.spin * w.radius;

            let material = terrain.material_at(hub.x, hub.z);
            let fmax = material.friction * n;
            let slip_ratio = (wheel_speed - v_long) / (v_long.abs() + 1.0);
            let slip_angle = (-v_lat).atan2(v_long.abs() + EPS);
            let (fx, fy) = tire_forces(slip_ratio, slip_angle, fmax);
            let rolling = -material.rolling_resistance * n * v_long.signum();

            let contact = DVec3::new(hub.x, ground, hub.z);
            let force = normal * n + forward * (fx + rolling) + lateral * fy;
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
        for &bp in &self.belly_points {
            let p = self.body.position + r * bp;
            let ground = terrain.height(p.x, p.z);
            if p.y >= ground {
                continue;
            }
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

        // Jitter-selective angular drag (WI 611): scale the damper by the contact-jitter signal so
        // the numerical kraken buzz (which rattles the total normal force every sub-step) is caged,
        // while a genuine, smooth-contact rotation — an intentional flip from a sharp turn or an
        // obstacle clip — passes nearly undamped. Replaces the old blanket angular damper.
        let jitter = (total_normal - self.last_total_normal).abs();
        let weight = self.body.mass * self.gravity;
        let jitter_ratio = if weight > 1e-9 {
            (jitter / weight).min(JITTER_RATIO_CAP)
        } else {
            0.0
        };
        let angular_drag = BASELINE_ANGULAR_DRAG + JITTER_ANGULAR_DRAG * jitter_ratio;
        net_torque -= angular_drag * self.body.angular_velocity();

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
            if demand > self.wheels[i].shear_speed {
                self.wheels[i].inert = true;
                out.sheared.push(i);
            }
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

/// Assemble a [`Rover`] from a built `craft` (WI 607), placing its centre of mass at
/// world `position` under `gravity`. Mass / inertia / CoM come from the chassis voxels
/// **and** attached parts ([`VoxelCraft::mass_properties`]); each [`PartKind::Wheel`]
/// part becomes a [`Wheel`] at its CoM-relative mount, carrying its suspension/tire
/// parameters; drive/steer groups come from the wheel parts' flags.
///
/// Returns `None` when the craft has no mass or **no wheel parts** — a lattice without
/// wheels is not a rover (the rocket assembly path handles those). This is the
/// deterministic rocket-vs-rover discriminator: wheels ⇒ rover.
pub fn assemble_rover(craft: &VoxelCraft, position: DVec3, gravity: f64) -> Option<RoverAssembly> {
    let mp = craft.mass_properties()?;
    let com = mp.center_of_mass;

    let mut wheels = Vec::new();
    let mut drive = Vec::new();
    let mut steer = Vec::new();
    for part in &craft.parts {
        let PartKind::Wheel(spec) = part.kind else {
            continue;
        };
        let i = wheels.len();
        // Skip a degenerate wheel rather than producing a non-physical one.
        if spec.radius <= 0.0 || spec.wheel_inertia <= 0.0 {
            continue;
        }
        wheels.push(Wheel {
            // The rover core mounts wheels relative to the body's CoM (`body.position`).
            mount: part.mount - com,
            radius: spec.radius,
            rest_length: spec.rest_length,
            stiffness: spec.stiffness,
            damping: spec.damping,
            max_force: spec.max_force,
            steer: 0.0,
            spin: 0.0,
            // Physical wheel inertia from the actual wheel-part mass (½·m·r², solid disk) — not the
            // old oversized constant, which stored so much spin energy that a stopped rover's wheels
            // flung it forward again after a wall bounce (WI 618 user feedback).
            wheel_inertia: (0.5 * part.mass * spec.radius * spec.radius).max(0.02),
            drive_torque: 0.0,
            brake: 0.0,
            inert: false,
            // Rated impact speed from the chassis material at the wheel (WI 618).
            shear_speed: rated_shear_speed(nearest_material_strength(craft, part.mount)),
        });
        if spec.drive {
            drive.push(i);
        }
        if spec.steer {
            steer.push(i);
        }
    }

    if wheels.is_empty() {
        return None;
    }

    // Size each wheel's suspension to the assembled mass (WI 612 feedback): a spring stiff enough to
    // carry its share of the weight at ~25% compression, damped near-critical, with headroom for
    // bumps. Without this the fixed `WheelPart` springs were mismatched to the build (sagging or
    // flinging), which read as "won't move / nose lifts".
    let n = wheels.len() as f64;
    let load = (mp.mass * gravity / n).max(1.0);
    let m_wheel = mp.mass / n;
    for w in &mut wheels {
        let target_comp = (0.25 * w.rest_length).max(1e-3);
        w.stiffness = (load / target_comp).clamp(1.0e3, 5.0e5);
        w.max_force = (load * 6.0).max(1.0e3);
        w.damping = 2.0 * 0.7 * (w.stiffness * m_wheel).sqrt();
    }

    // Powertrain (WI 609): an `Engine` device is a combustion engine for the rover (drivetrain),
    // not thrust; `Tank` feeds it, `Battery` + `SolarPanel` parts are the electric path.
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

    let body = ActiveBody::from_mass_properties(position, DVec3::ZERO, &mp);
    Some(RoverAssembly {
        rover: Rover::new(body, wheels, gravity),
        drive,
        steer,
        powertrain,
    })
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
fn tire_forces(slip_ratio: f64, slip_angle: f64, fmax: f64) -> (f64, f64) {
    let fx = fmax * (C_LONG * slip_ratio).tanh();
    // `slip_angle = atan2(-v_lat, |v_long|)`, so `fy` here points to **oppose** the
    // lateral slip (a restoring force). Getting this sign wrong makes the lateral
    // force amplify sliding → oversteer spin-out.
    let fy = fmax * (C_LAT * slip_angle).tanh();
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
        Device, DeviceKind, Material, Part, PartKind, Voxel, VoxelCraft, WheelPart,
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
            });
        }
        craft
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
        // Body mass equals the chassis-plus-parts mass.
        assert!((asm.rover.body.mass - mp.mass).abs() < 1e-9);
        // Wheel mounts are CoM-relative.
        let expected = DVec3::new(-1.0, -0.2, -2.0) - mp.center_of_mass;
        assert!((asm.rover.wheels[0].mount - expected).length() < 1e-12);
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
        for step in 0..40_000 {
            if step > 3_000 {
                for &i in &asm.drive {
                    rover.wheels[i].drive_torque = mp.mass * 6.0; // floor it into the wall
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
            if step > 5_000 {
                max_vy = max_vy.max(rover.body.velocity.y.abs());
            }
        }
        // It advanced toward the wall but did **not** tunnel through it (front face ≈ 2.8 m), and
        // the contact never launched it.
        assert!(
            rover.body.position.z > 0.3,
            "rover did not drive toward the wall: {}",
            rover.body.position.z
        );
        assert!(
            rover.body.position.z < 2.9,
            "rover tunnelled through the wall: {}",
            rover.body.position.z
        );
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
        assert_eq!(tire_forces(0.0, 0.0, 1_000.0), (0.0, 0.0));
        let (fx, _) = tire_forces(5.0, 0.0, 1_000.0); // large slip → saturates near fmax
        assert!(fx > 900.0 && fx <= 1_000.0 + 1e-9);
        // Friction ellipse: combined never exceeds fmax.
        let (fx, fy) = tire_forces(5.0, 1.2, 1_000.0);
        assert!((fx * fx + fy * fy).sqrt() <= 1_000.0 + 1e-6);
    }

    #[test]
    fn tire_force_scales_with_surface_material() {
        // fmax = μ·N, so ice (low μ) yields a smaller saturated force than bedrock.
        let n = 5_000.0;
        let ice = tire_forces(5.0, 0.0, SurfaceMaterial::ICE.friction * n).0;
        let bedrock = tire_forces(5.0, 0.0, SurfaceMaterial::BEDROCK.friction * n).0;
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
