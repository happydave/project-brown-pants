//! FAR-style aerodynamics: lift, stability, and transonic area-ruling (WI 521).
//!
//! Completes the design's cross-section-based aero module. The dive (WI 509)
//! already takes **drag** from the voxel `area_curve`; this adds **lift** and a
//! **pitching/stability** moment (thin-airfoil, validated against the 2π
//! lift-slope) and a **transonic area-ruling** wave drag — all from the *one* area
//! curve and the *one* [`FluidSample`], the medium parameterising them:
//!
//! - vacuum (ρ=0) → no aero;
//! - any dense fluid → lift (hydrodynamic lift works in water too);
//! - a **compressible** medium (gas) → Mach and the transonic wave-drag rise;
//!   water is treated incompressible (the design disables it) so wave drag is
//!   exactly zero there.
//!
//! Parameterized thin-airfoil / area-rule aero — not a panel or CFD solve (those
//! stay quarantined spikes). Headless; the wind-tunnel scene lives in the app.

use crate::fluid::{FluidSample, MediumKind};
use crate::voxel::VoxelCraft;
use glam::{DVec3, IVec3};
use std::collections::HashSet;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// The six axis-aligned face offsets / outward normals of a cubic cell.
const FACE_OFFSETS: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

/// One voxel's exposed-face geometry for a given flow direction (WI 687) — the
/// directional generalization of [`VoxelCraft::area_curve`] /
/// [`crate::medium::max_cross_section`], and the aero-derived output the thermal
/// model consumes for convective heating (so thermal does not re-derive the
/// lattice itself — design resolution T3/T4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelExposure {
    /// The voxel's lattice cell.
    pub cell: IVec3,
    /// Total exposed skin area (faces bordering empty space), m².
    pub exposed_area: f64,
    /// Windward-projected exposed area, m²: `Σ face_area · max(0, n · flow_dir)`
    /// over the voxel's exposed faces. Zero for interior or fully-leeward voxels.
    pub windward_area: f64,
}

/// Per-voxel exposed and windward-projected areas for a flow direction expressed
/// in the craft's **local** frame (the direction the craft moves through the
/// medium). A face is *exposed* when its neighbouring cell is empty, and *windward*
/// in proportion to how directly its outward normal faces the oncoming flow
/// (`n · flow_dir > 0`). `flow_dir` need not be unit; a zero/degenerate direction
/// yields zero windward area everywhere (no convective loading at rest).
///
/// **Windward shadowing (WI 697).** A voxel standing in the aerodynamic shadow of an
/// upstream cell — a shield in front of it, possibly across a gap — has its windward
/// area attenuated by [`OCCLUSION_RESIDUAL`], so a body behind a heat shield is
/// physically cooler rather than reliant on a calibrated heat scale. *Adjacent*
/// shadowing is already handled by face culling (a directly-trailing voxel's leading
/// face is interior, so it carries no windward area); this adds the non-adjacent /
/// standoff case via an upstream ray-march. Pure geometry over the lattice — the
/// cross-section aero already owns this domain.
pub fn windward_faces(craft: &VoxelCraft, flow_dir: DVec3) -> Vec<VoxelExposure> {
    let occupied: HashSet<IVec3> = craft.voxels.iter().map(|v| v.cell).collect();
    let cell_area = craft.cell_size * craft.cell_size;
    let f = flow_dir.normalize_or_zero();
    // The occlusion march can cross the craft in at most this many cell steps, so it
    // always terminates.
    let max_steps = lattice_span(&occupied);
    craft
        .voxels
        .iter()
        .map(|v| {
            let mut exposed_area = 0.0;
            let mut windward_area = 0.0;
            for off in FACE_OFFSETS {
                if occupied.contains(&(v.cell + off)) {
                    continue; // interior face between two occupied cells
                }
                exposed_area += cell_area;
                windward_area += cell_area * off.as_dvec3().dot(f).max(0.0);
            }
            // WI 697: attenuate the windward heating of a voxel that sits behind an
            // upstream blocker along the flow (its leading surface is in the wake).
            if windward_area > 0.0 && occluded_upstream(v.cell, f, &occupied, max_steps) {
                windward_area *= OCCLUSION_RESIDUAL;
            }
            VoxelExposure {
                cell: v.cell,
                exposed_area,
                windward_area,
            }
        })
        .collect()
}

/// Residual share of windward heating retained by a fully flow-shadowed voxel (WI 697) —
/// a wake is cooler than the stagnation surface, but not perfectly cold.
const OCCLUSION_RESIDUAL: f64 = 0.1;

/// A safe upper bound (in cell steps) on crossing the occupied lattice along any straight
/// line: the summed bounding-box span over the three axes. Guarantees the occlusion march
/// terminates. Zero for an empty lattice.
fn lattice_span(occupied: &HashSet<IVec3>) -> i32 {
    if occupied.is_empty() {
        return 0;
    }
    let mut lo = IVec3::splat(i32::MAX);
    let mut hi = IVec3::splat(i32::MIN);
    for c in occupied {
        lo = lo.min(*c);
        hi = hi.max(*c);
    }
    let span = hi - lo;
    span.x + span.y + span.z + 3
}

/// Whether an occupied cell lies upstream of `cell` along the flow (the `+f` direction) —
/// i.e. `cell` is in that cell's aerodynamic shadow (WI 697). Marches cell-by-cell toward
/// the oncoming flow with a 3D DDA, bounded by `max_steps`. `f` is assumed normalized;
/// a zero direction (no flow) is never shadowed.
fn occluded_upstream(cell: IVec3, f: DVec3, occupied: &HashSet<IVec3>, max_steps: i32) -> bool {
    let step = IVec3::new(axis_step(f.x), axis_step(f.y), axis_step(f.z));
    if step == IVec3::ZERO {
        return false;
    }
    // Parametric distance to cross one cell on each axis (∞ where the flow has no
    // component); from the cell centre the first boundary is half a cell away.
    let t_delta = DVec3::new(axis_tdelta(f.x), axis_tdelta(f.y), axis_tdelta(f.z));
    let mut t_max = t_delta * 0.5;
    let mut cur = cell;
    for _ in 0..max_steps {
        if t_max.x <= t_max.y && t_max.x <= t_max.z {
            cur.x += step.x;
            t_max.x += t_delta.x;
        } else if t_max.y <= t_max.z {
            cur.y += step.y;
            t_max.y += t_delta.y;
        } else {
            cur.z += step.z;
            t_max.z += t_delta.z;
        }
        if occupied.contains(&cur) {
            return true;
        }
    }
    false
}

/// Integer step direction for a flow component: ±1, or 0 when the component is zero
/// (`f64::signum` returns ±1 even for 0.0, so it cannot be used here).
fn axis_step(c: f64) -> i32 {
    if c > 0.0 {
        1
    } else if c < 0.0 {
        -1
    } else {
        0
    }
}

/// Parametric distance to cross one cell along an axis with flow component `c` (`1/|c|`),
/// or infinity when the flow has no component on that axis.
fn axis_tdelta(c: f64) -> f64 {
    if c != 0.0 {
        1.0 / c.abs()
    } else {
        f64::INFINITY
    }
}

/// Ratio of specific heats for air (diatomic), for the speed of sound.
const GAMMA: f64 = 1.4;
/// Stall angle of attack, radians (~15°).
const STALL: f64 = 0.26;
/// Critical Mach below which there is no wave drag.
const M_CRIT: f64 = 0.8;

/// Thin-airfoil lift coefficient as a function of angle of attack (radians).
/// Odd in `α`, with slope `2π` near zero (the classic thin-airfoil result) and a
/// stall rollover past [`STALL`]. Bounded for all `α`.
pub fn lift_coefficient(alpha: f64) -> f64 {
    let a = alpha.abs().min(FRAC_PI_2);
    let peak = TAU * STALL; // 2π·α at the stall angle
    let cl = if a <= STALL {
        TAU * a
    } else {
        // Post-stall: decline from the peak toward 40% of it by 90°.
        let t = ((a - STALL) / (FRAC_PI_2 - STALL)).clamp(0.0, 1.0);
        peak * (1.0 - 0.6 * t)
    };
    cl * alpha.signum()
}

/// Angle of attack (radians, `[0, π/2]`) between the relative velocity and the
/// body's forward axis — the acute angle, so reversed flow is handled.
fn angle_of_attack(velocity: DVec3, body_forward: DVec3) -> f64 {
    let v = velocity.normalize_or_zero();
    let f = body_forward.normalize_or_zero();
    if v == DVec3::ZERO || f == DVec3::ZERO {
        return 0.0;
    }
    let ang = f.dot(v).clamp(-1.0, 1.0).acos();
    if ang > FRAC_PI_2 {
        PI - ang
    } else {
        ang
    }
}

/// Aero/hydro **lift**: `½ρv²·Cl(α)·A` perpendicular to the relative velocity, in
/// the plane of the velocity and the body axis, toward the body's forward side.
/// Zero in vacuum or at rest; present in any dense fluid (water included).
pub fn lift_force(sample: &FluidSample, velocity: DVec3, body_forward: DVec3, area: f64) -> DVec3 {
    let speed = velocity.length();
    if speed <= 0.0 || sample.density <= 0.0 {
        return DVec3::ZERO;
    }
    let vd = velocity / speed;
    let f = body_forward.normalize_or_zero();
    // Component of the body axis perpendicular to the flow = the lift direction.
    let perp = f - f.dot(vd) * vd;
    let lift_dir = perp.normalize_or_zero();
    if lift_dir == DVec3::ZERO {
        return DVec3::ZERO; // axis aligned with flow: no lift
    }
    let alpha = angle_of_attack(velocity, body_forward);
    let cl = lift_coefficient(alpha);
    0.5 * sample.density * speed * speed * cl * area * lift_dir
}

/// Speed of sound in the medium, m/s — `√(γP/ρ)` for a **compressible gas**
/// (atmosphere). `None` for liquid (incompressible by design) and vacuum, which
/// is what gates wave drag off there.
pub fn sound_speed(sample: &FluidSample) -> Option<f64> {
    match sample.medium {
        MediumKind::Atmosphere if sample.density > 0.0 && sample.pressure > 0.0 => {
            Some((GAMMA * sample.pressure / sample.density).sqrt())
        }
        _ => None,
    }
}

/// Mach number, or `None` in an incompressible/vacuum medium.
pub fn mach(sample: &FluidSample, speed: f64) -> Option<f64> {
    sound_speed(sample).map(|a| speed / a)
}

/// A measure of the area curve's **abruptness** — the normalised squared
/// variation of the cross-sectional area along the body. Small for a smooth
/// (area-ruled) taper, large for an abrupt body. Drives the wave-drag magnitude.
pub fn area_ruling_factor(area_curve: &[(i32, f64)]) -> f64 {
    if area_curve.len() < 2 {
        return 0.0;
    }
    let amax = area_curve.iter().map(|&(_, a)| a).fold(0.0_f64, f64::max);
    if amax <= 0.0 {
        return 0.0;
    }
    let var: f64 = area_curve
        .windows(2)
        .map(|w| (w[1].1 - w[0].1).powi(2))
        .sum();
    var / (amax * amax)
}

/// Transonic wave-drag coefficient: zero below [`M_CRIT`], a bump peaking near
/// Mach 1.1, declining supersonically, scaled by the area-ruling factor.
pub fn wave_drag_coefficient(mach: f64, area_ruling_factor: f64) -> f64 {
    if mach < M_CRIT {
        return 0.0;
    }
    let bump = (-((mach - 1.1) / 0.3).powi(2)).exp();
    area_ruling_factor * bump
}

/// Wave-drag force from transonic area-ruling: opposes the velocity, and is
/// **exactly zero** in an incompressible (water) or vacuum medium (no Mach).
pub fn wave_drag_force(
    sample: &FluidSample,
    velocity: DVec3,
    area_ruling_factor: f64,
    area: f64,
) -> DVec3 {
    let speed = velocity.length();
    if speed <= 0.0 {
        return DVec3::ZERO;
    }
    let Some(m) = mach(sample, speed) else {
        return DVec3::ZERO; // incompressible/vacuum: no wave drag
    };
    let cd = wave_drag_coefficient(m, area_ruling_factor);
    -0.5 * sample.density * speed * speed * cd * area * (velocity / speed)
}

/// Centre of pressure along the body axis, m — the area-weighted centroid of the
/// area curve (`Σ station·area / Σ area`, scaled to metres). The cross-section's
/// "balance point", where the aero force effectively acts.
pub fn center_of_pressure(area_curve: &[(i32, f64)], cell_size: f64) -> f64 {
    let total: f64 = area_curve.iter().map(|&(_, a)| a).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let weighted: f64 = area_curve.iter().map(|&(s, a)| (s as f64 + 0.5) * a).sum();
    (weighted / total) * cell_size
}

/// Pitching moment about the centre of mass from an aero force acting at the
/// centre of pressure: `(cop − com) × force`. Restoring (weathervaning) when the
/// centre of pressure is aft of the centre of mass.
pub fn pitching_moment(cop_offset: DVec3, aero_force: DVec3) -> DVec3 {
    cop_offset.cross(aero_force)
}

/// Aerodynamic **pitch-damping** moment — a torque opposing the body's angular
/// velocity, scaled by the aero loading (`½ρ·v`), a reference area, a
/// characteristic length², and a damping coefficient. This is the `Cm_q`-style
/// damping derivative that makes a statically-stable craft *converge* to trim
/// rather than oscillate about it undamped. Medium-parameterised like every other
/// aero force: **zero in vacuum** (ρ=0) and **zero at rest** (v=0). The product
/// `ρ·v·A·L²` has units of `N·m·s`, so multiplied by `ω` (1/s) it is a torque.
pub fn pitch_damping_moment(
    sample: &FluidSample,
    speed: f64,
    angular_velocity: DVec3,
    area: f64,
    length: f64,
    coeff: f64,
) -> DVec3 {
    if speed <= 0.0 || sample.density <= 0.0 {
        return DVec3::ZERO;
    }
    -coeff * 0.5 * sample.density * speed * area * length * length * angular_velocity
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fluid::FluidMedium;

    fn air() -> FluidSample {
        FluidMedium::EARTHLIKE.sample_altitude(0.0)
    }
    fn water() -> FluidSample {
        FluidMedium::EARTHLIKE.sample_altitude(-10.0)
    }
    fn vacuum() -> FluidSample {
        FluidMedium::VACUUM.sample_altitude(0.0)
    }

    // --- I1 lift (thin-airfoil) ---

    #[test]
    fn lift_slope_is_two_pi_near_zero() {
        // dCl/dα ≈ 2π per radian (thin-airfoil theory).
        let slope = (lift_coefficient(0.02) - lift_coefficient(-0.02)) / 0.04;
        assert!((slope - TAU).abs() < 1e-6, "lift slope {slope} ≠ 2π");
        // Odd function.
        assert!((lift_coefficient(0.1) + lift_coefficient(-0.1)).abs() < 1e-12);
    }

    #[test]
    fn lift_stalls_past_the_stall_angle() {
        let peak = lift_coefficient(STALL);
        assert!(lift_coefficient(0.6) < peak, "post-stall Cl should drop");
        assert!(lift_coefficient(0.5) < peak);
        // The peak is near the stall angle.
        assert!(peak > lift_coefficient(0.1));
    }

    #[test]
    fn lift_is_zero_in_vacuum_present_in_air_and_water() {
        // Body axis +x, flow mostly +x with a little +y → small AoA.
        let v = DVec3::new(100.0, 8.0, 0.0);
        let fwd = DVec3::X;
        assert_eq!(lift_force(&vacuum(), v, fwd, 4.0), DVec3::ZERO);
        let air_lift = lift_force(&air(), v, fwd, 4.0);
        let water_lift = lift_force(&water(), v, fwd, 4.0);
        assert!(air_lift.length() > 0.0, "air produces lift");
        assert!(
            water_lift.length() > air_lift.length(),
            "denser water, more lift"
        );
        // Lift is perpendicular to the velocity.
        assert!(air_lift.dot(v).abs() < 1e-6 * air_lift.length() * v.length() + 1e-6);
    }

    #[test]
    fn lift_is_zero_when_axis_aligned_with_flow() {
        let v = DVec3::new(100.0, 0.0, 0.0);
        assert_eq!(lift_force(&air(), v, DVec3::X, 4.0), DVec3::ZERO);
    }

    // --- I3 transonic area-ruling, compressible-only ---

    #[test]
    fn sound_speed_only_in_a_gas() {
        let a = sound_speed(&air()).expect("air has a sound speed");
        assert!(
            (a - 340.0).abs() < 30.0,
            "sea-level sound speed ~340 m/s, got {a}"
        );
        assert!(
            sound_speed(&water()).is_none(),
            "water incompressible (by design)"
        );
        assert!(sound_speed(&vacuum()).is_none());
    }

    #[test]
    fn wave_drag_peaks_transonic_zero_subsonic() {
        let f = 0.5;
        assert_eq!(wave_drag_coefficient(0.5, f), 0.0, "subsonic: no wave drag");
        let at1 = wave_drag_coefficient(1.0, f);
        let at2 = wave_drag_coefficient(2.0, f);
        assert!(at1 > 0.0);
        assert!(at1 > at2, "declines supersonically: {at1} vs {at2}");
        assert!(
            at1 > wave_drag_coefficient(0.85, f),
            "rises through transonic"
        );
    }

    #[test]
    fn smooth_body_has_less_wave_drag_than_abrupt() {
        // Smooth taper vs an abrupt step, same peak area.
        let smooth = [(0, 1.0), (1, 2.0), (2, 3.0), (3, 2.0), (4, 1.0)];
        let abrupt = [(0, 0.0), (1, 3.0), (2, 3.0), (3, 0.0), (4, 0.0)];
        let fs = area_ruling_factor(&smooth);
        let fa = area_ruling_factor(&abrupt);
        assert!(
            fa > fs,
            "abrupt body has a larger area-ruling factor: {fa} vs {fs}"
        );
        // → larger wave drag at the same Mach.
        assert!(wave_drag_coefficient(1.1, fa) > wave_drag_coefficient(1.1, fs));
    }

    #[test]
    fn wave_drag_force_zero_in_water_and_vacuum() {
        // A speed that is supersonic in air.
        let v = DVec3::new(600.0, 0.0, 0.0);
        let f = 0.5;
        assert!(
            wave_drag_force(&air(), v, f, 4.0).length() > 0.0,
            "air: wave drag"
        );
        assert_eq!(
            wave_drag_force(&water(), v, f, 4.0),
            DVec3::ZERO,
            "water: no wave drag (incompressible)"
        );
        assert_eq!(wave_drag_force(&vacuum(), v, f, 4.0), DVec3::ZERO);
    }

    // --- I2 stability ---

    #[test]
    fn center_of_pressure_is_the_area_centroid() {
        // Symmetric area curve → CoP at the middle.
        let curve = [(0, 1.0), (1, 2.0), (2, 1.0)];
        // centroid = (0.5·1 + 1.5·2 + 2.5·1)/4 = (0.5+3+2.5)/4 = 1.5
        assert!((center_of_pressure(&curve, 1.0) - 1.5).abs() < 1e-9);
    }

    #[test]
    fn pitching_moment_is_restoring_when_cop_aft_of_com() {
        // Body axis +x; CoP aft of CoM means toward −x (behind, for a craft moving
        // +x). A positive-AoA lift (+y) at the aft CoP makes a moment that pitches
        // the nose toward the flow (reduces AoA) — weathervaning.
        let cop_offset = DVec3::new(-2.0, 0.0, 0.0); // aft of CoM
        let lift = DVec3::new(0.0, 100.0, 0.0); // upward lift
        let moment = pitching_moment(cop_offset, lift);
        // Moment about z; an aft upward force pitches the nose down (−z here),
        // opposing the nose-up that created the +AoA → restoring.
        assert!(
            moment.z < 0.0,
            "aft CoP gives a restoring (nose-down) moment"
        );
        // A forward CoP would be destabilising (opposite sign).
        let fwd_moment = pitching_moment(DVec3::new(2.0, 0.0, 0.0), lift);
        assert!(fwd_moment.z > 0.0);
    }

    #[test]
    fn pitch_damping_opposes_spin_and_vanishes_in_vacuum() {
        let omega = DVec3::new(0.0, 0.0, 0.5);
        let air_damp = pitch_damping_moment(&air(), 100.0, omega, 4.0, 2.0, 0.1);
        // Opposes the spin (anti-parallel) and is nonzero in a dense medium.
        assert!(
            air_damp.dot(omega) < 0.0,
            "damping must oppose angular velocity"
        );
        assert!(air_damp.length() > 0.0);
        // Denser water damps harder at the same spin/speed.
        let water_damp = pitch_damping_moment(&water(), 100.0, omega, 4.0, 2.0, 0.1);
        assert!(water_damp.length() > air_damp.length());
        // Vacuum and rest produce no damping (medium-parameterised).
        assert_eq!(
            pitch_damping_moment(&vacuum(), 100.0, omega, 4.0, 2.0, 0.1),
            DVec3::ZERO
        );
        assert_eq!(
            pitch_damping_moment(&air(), 0.0, omega, 4.0, 2.0, 0.1),
            DVec3::ZERO
        );
    }

    #[test]
    fn windward_faces_expose_only_the_oncoming_direction() {
        use crate::voxel::{Material, Voxel, VoxelCraft};
        use glam::IVec3;

        // A single voxel: all six faces exposed; only the +x face is windward to +x flow.
        let mut single = VoxelCraft::new(1.0);
        single.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        let exp = windward_faces(&single, DVec3::X);
        assert_eq!(exp.len(), 1);
        assert!(
            (exp[0].exposed_area - 6.0).abs() < 1e-9,
            "6 faces of area 1"
        );
        assert!(
            (exp[0].windward_area - 1.0).abs() < 1e-9,
            "one face faces +x"
        );

        // A 2-voxel bar along x: the shared interior face is culled (each has 5 exposed),
        // and only the leading (+x) voxel takes windward area for +x flow.
        let mut bar = VoxelCraft::new(1.0);
        bar.voxels.push(Voxel {
            cell: IVec3::new(0, 0, 0),
            material: Material::ALUMINIUM,
        });
        bar.voxels.push(Voxel {
            cell: IVec3::new(1, 0, 0),
            material: Material::ALUMINIUM,
        });
        let exp = windward_faces(&bar, DVec3::X);
        let lead = exp.iter().find(|e| e.cell.x == 1).unwrap();
        let trail = exp.iter().find(|e| e.cell.x == 0).unwrap();
        assert!(
            (lead.exposed_area - 5.0).abs() < 1e-9,
            "interior face culled"
        );
        assert!(
            (lead.windward_area - 1.0).abs() < 1e-9,
            "leading face windward"
        );
        assert!(
            trail.windward_area.abs() < 1e-9,
            "trailing voxel not windward"
        );

        // At rest (zero flow) nothing is windward.
        let rest = windward_faces(&bar, DVec3::ZERO);
        assert!(rest.iter().all(|e| e.windward_area.abs() < 1e-9));
    }

    #[test]
    fn windward_shadowing_attenuates_a_body_behind_a_blocker() {
        use crate::voxel::{Material, Voxel, VoxelCraft};
        use glam::IVec3;

        // Flow +Z: a body at z=0, a gap at z=1, a blocker (shield) at z=2.
        let body = IVec3::new(0, 0, 0);
        let shield = IVec3::new(0, 0, 2);
        let mut craft = VoxelCraft::new(1.0);
        craft.voxels.push(Voxel {
            cell: body,
            material: Material::ALUMINIUM,
        });
        craft.voxels.push(Voxel {
            cell: shield,
            material: Material::ALUMINIUM,
        });
        let exp = windward_faces(&craft, DVec3::Z);
        let body_w = exp.iter().find(|e| e.cell == body).unwrap().windward_area;
        let shield_w = exp.iter().find(|e| e.cell == shield).unwrap().windward_area;
        assert!(shield_w > 0.0, "the shield faces the flow");
        assert!(
            body_w < shield_w,
            "the shadowed body is attenuated: body {body_w} vs shield {shield_w}"
        );
        assert!(
            body_w <= 0.2,
            "the shadowed body keeps only a little windward area: {body_w}"
        );

        // Remove the shield: the body is directly exposed and recovers full windward area.
        let mut solo = VoxelCraft::new(1.0);
        solo.voxels.push(Voxel {
            cell: body,
            material: Material::ALUMINIUM,
        });
        let body_solo = windward_faces(&solo, DVec3::Z)[0].windward_area;
        assert!(
            (body_solo - shield_w).abs() < 1e-9,
            "an exposed body has full windward area: {body_solo}"
        );
        assert!(
            body_w < body_solo,
            "shadowing reduced the body's windward area: {body_w} < {body_solo}"
        );
    }
}
