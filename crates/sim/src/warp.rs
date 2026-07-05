//! Anti-tunnelling time-warp cap (WI 595) — keep a fast active body from passing through a
//! surface under time-warp.
//!
//! The active gear integrates at a fixed substep `FIXED_DT`; time-warp scales how far simulated
//! time advances. The tunnelling risk is a step large enough that a body crosses a surface
//! before the contact stage runs. The defence is a **surface-proximity cap**: near a surface,
//! limit the warp (equivalently, the effective step) so one step advances the body no farther
//! than its current gap to the surface (plus the contact `tolerance`). Far from any surface, or
//! receding from it, warp is untouched.
//!
//! These are **pure policy functions** (no stored state): warp lives in `SimClock`, and the cap
//! is applied as a `min` wherever an active body's warp is resolved, derived from the body's
//! state and the surface — consistent with the existing warp model. CCD/swept tests are
//! deferred (see the work item's recorded decision); the proximity cap covers the project's
//! warp range.

use crate::active::ActiveBody;
use crate::collision::CollisionShape;

/// The largest warp factor for which one warp-scaled step (`base_dt · warp`) moves a body no
/// more than `gap + tolerance` toward a surface — so the fixed-step integrator cannot pass
/// through it before contact is resolved. `approach_speed` is the speed component **toward** the
/// surface (≤ 0 means receding/stationary → the `requested` warp is returned unchanged). The
/// result never exceeds `requested`. When already touching/penetrating (`gap ≤ 0`) the body may
/// still advance up to `tolerance`, so the cap shrinks toward zero rather than locking up.
pub fn max_safe_warp(
    requested: f64,
    gap: f64,
    approach_speed: f64,
    base_dt: f64,
    tolerance: f64,
) -> f64 {
    if approach_speed <= 0.0 || base_dt <= 0.0 {
        return requested; // receding, stationary, or no step → nothing to cap
    }
    let allowed = gap.max(0.0) + tolerance; // how far it may close this step
    let cap = allowed / (approach_speed * base_dt);
    requested.min(cap.max(0.0))
}

/// The largest **substep size** for which a body closing on a surface at `approach_speed` moves
/// no more than `gap + tolerance` — the substep-size view of [`max_safe_warp`] (the two are the
/// same policy: `safe_substep_dt = base_dt · max_safe_warp(1, …)`). Returns `base_dt` when
/// receding.
pub fn safe_substep_dt(gap: f64, approach_speed: f64, base_dt: f64, tolerance: f64) -> f64 {
    if approach_speed <= 0.0 {
        return base_dt;
    }
    let allowed = gap.max(0.0) + tolerance;
    base_dt.min(allowed / approach_speed)
}

/// The warp cap for an active `body` approaching a flat-ground `ground` half-space, given the
/// body's CoM-relative `bounding_radius` (see `collision::craft_bounding_radius`). The gap is
/// the nearest-point distance above the plane (`CoM·n − offset − radius`), conservative for any
/// orientation; the approach speed is the body's closing speed along the plane normal. For a
/// non-half-space ground this returns `requested` unchanged (broader surfaces are future work).
pub fn ground_safe_warp(
    requested: f64,
    body: &ActiveBody,
    bounding_radius: f64,
    ground: &CollisionShape,
    base_dt: f64,
    tolerance: f64,
) -> f64 {
    match ground {
        CollisionShape::HalfSpace { normal, offset } => {
            let gap = body.position.dot(*normal) - offset - bounding_radius;
            let approach = -body.velocity.dot(*normal);
            max_safe_warp(requested, gap, approach, base_dt, tolerance)
        }
        // No proximity model for a compound here; leave warp unchanged.
        CollisionShape::CuboidCompound(_) | CollisionShape::Compound { .. } => requested,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::{craft_bounding_radius, ground_half_space};
    use crate::voxel::{Material, Voxel, VoxelCraft};
    use glam::{DMat3, DVec3, IVec3};

    const BASE_DT: f64 = 1.0 / 64.0;
    const TOL: f64 = 0.05;

    fn unit_body(pos: DVec3, vel: DVec3) -> ActiveBody {
        ActiveBody::new(pos, vel, 1.0, DMat3::IDENTITY)
    }

    #[test]
    fn receding_or_far_body_keeps_requested_warp() {
        // Receding (moving up) → no cap.
        assert_eq!(max_safe_warp(1000.0, 1.0, -50.0, BASE_DT, TOL), 1000.0);
        // Far away and slow → the cap exceeds the request, so the request stands.
        assert_eq!(max_safe_warp(10.0, 1_000.0, 5.0, BASE_DT, TOL), 10.0);
    }

    #[test]
    fn near_and_fast_body_is_capped_below_request() {
        // 1 m gap, closing at 500 m/s: a single FIXED_DT step at warp 1000 would move
        // 500·(1/64)·1000 ≈ 7800 m — far through the surface. The cap brings it back.
        let w = max_safe_warp(1000.0, 1.0, 500.0, BASE_DT, TOL);
        assert!(w < 1000.0, "warp should be capped: {w}");
        // At the capped warp, the step displacement is exactly gap + tolerance.
        let displacement = 500.0 * (BASE_DT * w);
        assert!(
            (displacement - (1.0 + TOL)).abs() < 1e-9,
            "step closes gap+tol: {displacement}"
        );
    }

    #[test]
    fn substep_size_view_matches_warp_view() {
        let gap = 2.0;
        let speed = 120.0;
        let dt = safe_substep_dt(gap, speed, BASE_DT, TOL);
        let w = max_safe_warp(1.0, gap, speed, BASE_DT, TOL);
        assert!((dt - BASE_DT * w).abs() < 1e-12, "the two views agree");
    }

    #[test]
    fn ground_safe_warp_caps_a_descending_craft() {
        let ground = ground_half_space(0.0);
        // A craft 0.5 m above the plane (CoM at 1.0, radius 0.5) diving at 400 m/s.
        let body = unit_body(DVec3::new(0.0, 1.0, 0.0), DVec3::new(0.0, -400.0, 0.0));
        let capped = ground_safe_warp(1_000.0, &body, 0.5, &ground, BASE_DT, TOL);
        assert!(
            capped < 1_000.0,
            "descending near the plane is capped: {capped}"
        );
        // A craft climbing away is not capped.
        let up = unit_body(DVec3::new(0.0, 1.0, 0.0), DVec3::new(0.0, 400.0, 0.0));
        assert_eq!(
            ground_safe_warp(1_000.0, &up, 0.5, &ground, BASE_DT, TOL),
            1_000.0
        );
    }

    #[test]
    fn cap_prevents_tunnelling_while_uncapped_warp_does_not() {
        // The headless anti-tunnel guarantee: a body diving at 500 m/s from 10 m up, stepped
        // with warp-scaled substeps. With the cap it never crosses the plane beyond the contact
        // tolerance; without it, one step passes straight through.
        let ground = ground_half_space(0.0);
        let radius = 0.5;
        let requested = 1_000.0;

        // Capped: each step advances at most gap+tol toward the plane.
        let mut body = unit_body(
            DVec3::new(0.0, 10.0 + radius, 0.0),
            DVec3::new(0.0, -500.0, 0.0),
        );
        let mut min_gap = f64::INFINITY;
        for _ in 0..100_000 {
            let w = ground_safe_warp(requested, &body, radius, &ground, BASE_DT, TOL);
            let dt = BASE_DT * w;
            body.position += body.velocity * dt;
            let gap = body.position.y - radius; // distance of nearest point above plane (offset 0)
            min_gap = min_gap.min(gap);
            if gap <= TOL {
                break; // reached the contact band — the integrator's contact stage takes over
            }
        }
        assert!(
            min_gap >= -TOL - 1e-9,
            "capped warp never tunnels past the tolerance: min_gap={min_gap}"
        );

        // Uncapped control: a single full-warp step tunnels far through the plane.
        let mut body = unit_body(
            DVec3::new(0.0, 10.0 + radius, 0.0),
            DVec3::new(0.0, -500.0, 0.0),
        );
        body.position += body.velocity * (BASE_DT * requested);
        let gap = body.position.y - radius;
        assert!(
            gap < -100.0,
            "uncapped warp tunnels straight through: gap={gap}"
        );
    }

    #[test]
    fn bounding_radius_drives_the_gap() {
        // A 2-cell-tall craft has a larger bounding radius, so its surface is "near" sooner.
        let mut tall = VoxelCraft::new(1.0);
        tall.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        tall.voxels.push(Voxel {
            cell: IVec3::new(0, 1, 0),
            material: Material::ALUMINIUM,
        });
        let r = craft_bounding_radius(&tall).unwrap();
        assert!(r > 1.0, "spans more than a unit cell from the CoM: {r}");
        assert!(craft_bounding_radius(&VoxelCraft::new(1.0)).is_none());
    }
}
