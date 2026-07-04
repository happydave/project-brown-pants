//! Connected-component structural breakage (WI 518).
//!
//! The voxel model's third role (after mass/inertia and the aero cross-section):
//! structural failure splits the connectivity graph into **connected components**,
//! each becoming its own craft. The design's deliberate choice over stress-field
//! FEA — cheap and dramatic, "split along structural lines".
//!
//! Three layers, all headless:
//!
//! - [`connected_components`] — partition a (possibly severed) lattice into
//!   maximal **face-connected** fragment crafts.
//! - [`failing_cut`] / [`break_craft`] — a **stress proxy** (not FEA): a candidate
//!   axis-aligned cut fails when the inertial load across it (the force to keep the
//!   outboard side moving with the rigid body) exceeds the crossing bonds'
//!   strength (cross-section × material tensile strength). One cross-section
//!   comparison — no stress field, no equilibrium solve.
//! - [`split_active`] / [`fracture`] — a **momentum-conserving** rigid split: each
//!   fragment inherits the parent's linear and angular velocity (`v = v_cm + ω×r`),
//!   re-deriving its own mass/inertia, conserving total momentum.

use crate::active::ActiveBody;
use crate::collision::{craft_bounds, craft_collision_shape, Bounds, CollisionShape};
use crate::contact::{body_contact_wrench, ground_contact_wrench, ContactParams};
use crate::voxel::{Axis, VoxelCraft};
use glam::{DVec3, IVec3};
use std::collections::{HashMap, HashSet};

/// A severed bond: the unordered pair of adjacent cells whose face connection is
/// cut. Stored canonically (smaller cell first) so lookups are order-independent.
pub type Severed = HashSet<(IVec3, IVec3)>;

/// Canonical ordering of a cell pair, so `(a, b)` and `(b, a)` are the same bond.
fn bond(a: IVec3, b: IVec3) -> (IVec3, IVec3) {
    if (a.x, a.y, a.z) <= (b.x, b.y, b.z) {
        (a, b)
    } else {
        (b, a)
    }
}

/// The six face-neighbour offsets.
const NEIGHBOURS: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

/// Partition the craft's voxels into maximal **face-connected** components,
/// excluding any `severed` bonds from adjacency. Each component becomes a fragment
/// [`VoxelCraft`] (same `cell_size`, its voxels plus the devices and attachment
/// points whose cell lies in it). A connected craft yields one fragment; an
/// already-disconnected lattice yields its true components.
pub fn connected_components(craft: &VoxelCraft, severed: &Severed) -> Vec<VoxelCraft> {
    // Index occupied cells.
    let occupied: HashMap<IVec3, usize> = craft
        .voxels
        .iter()
        .enumerate()
        .map(|(i, v)| (v.cell, i))
        .collect();

    // Flood-fill face-connected components over the occupied cells.
    let mut component_of: HashMap<IVec3, usize> = HashMap::new();
    let mut next_component = 0;
    for v in &craft.voxels {
        if component_of.contains_key(&v.cell) {
            continue;
        }
        // BFS from this seed.
        let id = next_component;
        next_component += 1;
        let mut stack = vec![v.cell];
        component_of.insert(v.cell, id);
        while let Some(cell) = stack.pop() {
            for off in NEIGHBOURS {
                let n = cell + off;
                if !occupied.contains_key(&n) || component_of.contains_key(&n) {
                    continue;
                }
                if severed.contains(&bond(cell, n)) {
                    continue;
                }
                component_of.insert(n, id);
                stack.push(n);
            }
        }
    }

    // Build one fragment craft per component.
    let mut fragments: Vec<VoxelCraft> = (0..next_component)
        .map(|_| VoxelCraft::new(craft.cell_size))
        .collect();
    for v in &craft.voxels {
        fragments[component_of[&v.cell]].voxels.push(*v);
    }
    for d in &craft.devices {
        if let Some(&id) = component_of.get(&d.cell) {
            fragments[id].devices.push(*d);
        }
    }
    for a in &craft.attachments {
        if let Some(&id) = component_of.get(&a.cell) {
            fragments[id].attachments.push(*a);
        }
    }
    fragments
}

/// The rigid-body acceleration of a point at world offset `r` from the centre of
/// mass, for a body accelerating its CoM at `a_cm` and spinning at `omega`
/// (torque-free, so no angular-acceleration term): `a = a_cm + ω×(ω×r)`.
fn point_acceleration(a_cm: DVec3, omega: DVec3, r: DVec3) -> DVec3 {
    a_cm + omega.cross(omega.cross(r))
}

/// The world-frame centre of cell `c` in the craft's local lattice frame.
fn cell_center(cell_size: f64, c: IVec3) -> DVec3 {
    (c.as_dvec3() + DVec3::splat(0.5)) * cell_size
}

/// Evaluate axis-aligned cut planes under a rigid-body load (`a_cm` applied
/// acceleration and `omega` angular velocity, both in the craft's **local lattice
/// frame**) and return the [`Severed`] bonds of the **weakest** cut whose inertial
/// load exceeds its bond strength — or `None` if the structure holds. Pure proxy:
/// one cross-section comparison per candidate plane, no stress field. ([`fracture`]
/// converts a world-frame body load into this local frame.)
pub fn failing_cut(craft: &VoxelCraft, a_cm: DVec3, omega: DVec3) -> Option<Severed> {
    let mp = craft.mass_properties()?;
    let com = mp.center_of_mass;
    let cell_size = craft.cell_size;
    let cell_volume = craft.cell_volume();
    let face_area = cell_size * cell_size;
    let occupied: HashSet<IVec3> = craft.voxels.iter().map(|v| v.cell).collect();

    let mut best: Option<(f64, Severed)> = None; // (load/strength ratio, bonds)

    for axis in [Axis::X, Axis::Y, Axis::Z] {
        let coord = |c: IVec3| match axis {
            Axis::X => c.x,
            Axis::Y => c.y,
            Axis::Z => c.z,
        };
        let off = match axis {
            Axis::X => IVec3::new(1, 0, 0),
            Axis::Y => IVec3::new(0, 1, 0),
            Axis::Z => IVec3::new(0, 0, 1),
        };
        let (lo, hi) = match craft
            .voxels
            .iter()
            .map(|v| coord(v.cell))
            .fold(None, |acc, k| {
                let (mn, mx) = acc.unwrap_or((k, k));
                Some((mn.min(k), mx.max(k)))
            }) {
            Some(range) => range,
            None => continue,
        };

        // Each plane sits between layer k and k+1.
        for k in lo..hi {
            // Bonds crossing the plane: occupied cells at layer ≤k adjacent to an
            // occupied cell one step up the axis.
            let mut bonds: Severed = HashSet::new();
            let mut strength = 0.0;
            for v in &craft.voxels {
                if coord(v.cell) != k {
                    continue;
                }
                let n = v.cell + off;
                if occupied.contains(&n) {
                    bonds.insert(bond(v.cell, n));
                    // Cells are solid cubes since WI 824 (plates live on faces;
                    // face-panel bonds join this graph in WI 828).
                    strength += v.material.strength * face_area;
                }
            }
            if bonds.is_empty() {
                continue;
            }

            // Outboard side = the layers farther from the CoM along this axis.
            let com_layer = match axis {
                Axis::X => com.x,
                Axis::Y => com.y,
                Axis::Z => com.z,
            } / cell_size
                - 0.5;
            let high_is_outboard =
                (k as f64 + 1.0 - com_layer).abs() > (k as f64 - com_layer).abs();

            // Net inertial force the bonds must transmit to the outboard side.
            let mut load = DVec3::ZERO;
            for v in &craft.voxels {
                let on_high = coord(v.cell) > k;
                if on_high != high_is_outboard {
                    continue;
                }
                let m = v.material.density * cell_volume;
                let r = cell_center(cell_size, v.cell) - com;
                load += m * point_acceleration(a_cm, omega, r);
            }
            for d in &craft.devices {
                let on_high = coord(d.cell) > k;
                if on_high != high_is_outboard {
                    continue;
                }
                let r = cell_center(cell_size, d.cell) - com;
                load += d.mass * point_acceleration(a_cm, omega, r);
            }

            let ratio = load.length() / strength;
            if ratio > 1.0 && best.as_ref().is_none_or(|(b, _)| ratio > *b) {
                best = Some((ratio, bonds));
            }
        }
    }

    best.map(|(_, bonds)| bonds)
}

/// Break the craft under the given load: if it fails, return the connected-
/// component fragment crafts; otherwise `None` (it holds). Geometry only — see
/// [`fracture`] for the active-body kinematics.
pub fn break_craft(craft: &VoxelCraft, a_cm: DVec3, omega: DVec3) -> Option<Vec<VoxelCraft>> {
    let severed = failing_cut(craft, a_cm, omega)?;
    Some(connected_components(craft, &severed))
}

/// Split an active craft's rigid state across a set of fragment crafts (a
/// partition of the parent), producing one momentum-conserving [`ActiveBody`] per
/// fragment: CoM world position from the parent's rigid transform, velocity
/// `v_cm + ω×r`, inherited orientation, and angular momentum `I_world·ω`. Total
/// linear and angular momentum about the parent CoM is conserved.
pub fn split_active(
    parent_craft: &VoxelCraft,
    parent_body: &ActiveBody,
    fragments: &[VoxelCraft],
) -> Vec<ActiveBody> {
    let parent_com = match parent_craft.mass_properties() {
        Some(mp) => mp.center_of_mass,
        None => return Vec::new(),
    };
    let omega = parent_body.angular_velocity();
    fragments
        .iter()
        .filter_map(|frag| {
            let mp = frag.mass_properties()?;
            // Offset of this fragment's CoM from the parent CoM, taken into the
            // world frame through the parent's orientation.
            let r_world = parent_body.orientation * (mp.center_of_mass - parent_com);
            let com_world = parent_body.position + r_world;
            let velocity = parent_body.velocity + omega.cross(r_world);
            let mut body = ActiveBody::new(com_world, velocity, mp.mass, mp.inertia);
            body.orientation = parent_body.orientation;
            Some(body.with_angular_velocity(omega))
        })
        .collect()
}

/// Break an active craft under an applied acceleration (e.g. drag deceleration;
/// `DVec3::ZERO` for a pure-spin break — the spin is read from the body), and
/// return momentum-conserving fragment `(VoxelCraft, ActiveBody)` pairs, or `None`
/// if the craft holds.
pub fn fracture(
    craft: &VoxelCraft,
    body: &ActiveBody,
    applied_accel: DVec3,
) -> Option<Vec<(VoxelCraft, ActiveBody)>> {
    // The stress proxy works in the craft's local frame; bring the world-frame
    // load there. The kinematic split stays in the world frame.
    let to_local = body.orientation.inverse();
    let accel_local = to_local * applied_accel;
    let omega_local = to_local * body.angular_velocity();
    let fragments = break_craft(craft, accel_local, omega_local)?;
    let bodies = split_active(craft, body, &fragments);
    Some(fragments.into_iter().zip(bodies).collect())
}

/// Break an active craft under a **contact force** (WI 594) — the collision→fracture coupling.
/// A contact force `F` transmitted through the structure decelerates the craft's CoM at `F/m`,
/// which is exactly the applied acceleration the [`fracture`] stress proxy consumes; so a hard
/// impact (large contact force) fractures the craft and a gentle touchdown (small force) leaves
/// it intact, using the same material-strength threshold as every other break. Returns
/// momentum-conserving fragment `(VoxelCraft, ActiveBody)` pairs, or `None` if it holds
/// (including a massless body, which cannot transmit a load).
pub fn fracture_on_impact(
    craft: &VoxelCraft,
    body: &ActiveBody,
    contact_force: DVec3,
) -> Option<Vec<(VoxelCraft, ActiveBody)>> {
    if body.mass <= 0.0 {
        return None;
    }
    fracture(craft, body, contact_force / body.mass)
}

/// A static (immovable) collider the debris stepper resolves fragments against — e.g. a pad obstacle
/// (WI 674). Borrows the collider's authoritative body/shape so the caller's own obstacle type need
/// not be visible here; the collider does not move (its contact reaction is discarded).
pub struct StaticCollider<'a> {
    pub body: &'a ActiveBody,
    pub shape: &'a CollisionShape,
    pub bounds: Option<Bounds>,
}

/// Advance a set of fracture fragments one substep in a **local frame** (WI 629): each fragment is a
/// rigid body under a constant gravity acceleration, a static `ground`, and pairwise inter-fragment
/// contact, integrated via [`ActiveBody::integrate_wrench`]. This is the headless, rover-frame
/// analogue of the rocket Test's app-side `step_fragments`: the rover Test calls it with
/// `gravity = (0, -g, 0)` and a flat-pad `HalfSpace` ground. Fragments with degenerate mass
/// properties contribute no shape/contact but are still integrated (gravity only), never panicking.
pub fn step_debris(
    fragments: &mut [(VoxelCraft, ActiveBody)],
    ground: &CollisionShape,
    obstacles: &[StaticCollider<'_>],
    gravity: DVec3,
    params: &ContactParams,
    dt: f64,
) {
    let n = fragments.len();
    let shapes: Vec<CollisionShape> = fragments
        .iter()
        .map(|(v, _)| craft_collision_shape(v))
        .collect();
    let bounds: Vec<Option<Bounds>> = fragments.iter().map(|(v, _)| craft_bounds(v)).collect();
    let coms: Vec<DVec3> = fragments
        .iter()
        .map(|(v, _)| {
            v.mass_properties()
                .map(|mp| mp.center_of_mass)
                .unwrap_or(DVec3::ZERO)
        })
        .collect();

    let mut acc = vec![(DVec3::ZERO, DVec3::ZERO); n];
    for (i, item) in acc.iter_mut().enumerate() {
        let (_, b) = &fragments[i];
        item.0 += gravity * b.mass;
        let (gf, gt) = ground_contact_wrench(b, &shapes[i], bounds[i], coms[i], ground, params);
        item.0 += gf;
        item.1 += gt;
        // Static obstacle contact (WI 674): each fragment is pushed out of the pad obstacles so the
        // debris piles up against a wall instead of ghosting through it. The obstacle is immovable —
        // its equal-and-opposite reaction is discarded.
        for ob in obstacles {
            let ((f, t), _) = body_contact_wrench(
                b,
                &shapes[i],
                bounds[i],
                coms[i],
                ob.body,
                ob.shape,
                ob.bounds,
                DVec3::ZERO,
                params,
            );
            item.0 += f;
            item.1 += t;
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let (_, bi) = &fragments[i];
            let (_, bj) = &fragments[j];
            let ((fa, ta), (fb, tb)) = body_contact_wrench(
                bi, &shapes[i], bounds[i], coms[i], bj, &shapes[j], bounds[j], coms[j], params,
            );
            acc[i].0 += fa;
            acc[i].1 += ta;
            acc[j].0 += fb;
            acc[j].1 += tb;
        }
    }
    for (i, (_, b)) in fragments.iter_mut().enumerate() {
        b.integrate_wrench(acc[i].0, acc[i].1, dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::{Material, Thermal, Voxel};

    /// A straight bar of `n` unit cells along +x, all one material.
    fn bar(n: i32, material: Material) -> VoxelCraft {
        let mut c = VoxelCraft::new(1.0);
        for x in 0..n {
            c.voxels.push(Voxel {
                cell: IVec3::new(x, 0, 0),
                material,
            });
        }
        c
    }

    fn total_mass(frags: &[VoxelCraft]) -> f64 {
        frags
            .iter()
            .filter_map(|f| f.mass_properties().map(|mp| mp.mass))
            .sum()
    }

    /// A single unit-cell fragment + its body at `pos` with velocity `vel`.
    fn cube_fragment(pos: DVec3, vel: DVec3) -> (VoxelCraft, ActiveBody) {
        let mut c = VoxelCraft::new(1.0);
        c.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        let mp = c.mass_properties().expect("one voxel has mass");
        let body = ActiveBody::new(pos, vel, mp.mass, mp.inertia);
        (c, body)
    }

    fn flat_ground(y: f64) -> CollisionShape {
        CollisionShape::HalfSpace {
            normal: DVec3::Y,
            offset: y,
        }
    }

    // --- WI 629 debris stepper ---

    #[test]
    fn debris_fragment_falls_and_comes_to_rest() {
        // A cube released at y=5 over a ground at y=0 settles near the surface with ~zero speed.
        let mut frags = vec![cube_fragment(DVec3::new(0.0, 5.0, 0.0), DVec3::ZERO)];
        let ground = flat_ground(0.0);
        let params = ContactParams::default();
        let g = DVec3::new(0.0, -9.81, 0.0);
        for _ in 0..3000 {
            step_debris(&mut frags, &ground, &[], g, &params, 0.004);
        }
        let b = &frags[0].1;
        assert!(b.position.is_finite(), "position went non-finite");
        assert!(
            b.position.y < 5.0,
            "fragment did not fall: y={}",
            b.position.y
        );
        // Resting on the surface: the CoM (cube centre, ~0.5 above its base) hovers near +0.5, neither
        // sunk through the plane nor floating, and the speed has bled off.
        assert!(
            (-0.2..1.2).contains(&b.position.y),
            "fragment not resting on the ground: y={}",
            b.position.y
        );
        assert!(
            b.velocity.length() < 1.0,
            "fragment not at rest: |v|={}",
            b.velocity.length()
        );
    }

    #[test]
    fn debris_overlapping_fragments_separate_without_sinking() {
        // Two cubes overlapping along x (ground far below, to isolate inter-fragment contact) push
        // apart: their x-separation grows and neither sinks through the distant ground.
        let mut frags = vec![
            cube_fragment(DVec3::new(0.0, 0.0, 0.0), DVec3::ZERO),
            cube_fragment(DVec3::new(0.5, 0.0, 0.0), DVec3::ZERO),
        ];
        let ground = flat_ground(-1000.0);
        let params = ContactParams::default();
        let sep0 = (frags[1].1.position - frags[0].1.position).length();
        for _ in 0..50 {
            step_debris(&mut frags, &ground, &[], DVec3::ZERO, &params, 0.004);
        }
        let sep1 = (frags[1].1.position - frags[0].1.position).length();
        assert!(
            sep1 >= sep0,
            "overlapping fragments did not separate: {sep0} -> {sep1}"
        );
        for (_, b) in &frags {
            assert!(b.position.is_finite());
            assert!(b.position.y > -1.5, "a fragment sank: y={}", b.position.y);
        }
    }

    #[test]
    fn debris_free_fall_is_gravity_impulse_and_finite() {
        // No contact (ground far below): one step changes vertical velocity by exactly g·dt.
        let mut frags = vec![cube_fragment(DVec3::new(0.0, 0.0, 0.0), DVec3::ZERO)];
        let ground = flat_ground(-1000.0);
        let params = ContactParams::default();
        let dt = 0.01;
        step_debris(
            &mut frags,
            &ground,
            &[],
            DVec3::new(0.0, -9.81, 0.0),
            &params,
            dt,
        );
        let b = &frags[0].1;
        assert!(b.velocity.is_finite() && b.position.is_finite());
        assert!(
            (b.velocity.y - (-9.81 * dt)).abs() < 1e-9,
            "free-fall Δv≠g·dt: vy={}",
            b.velocity.y
        );
        assert!(
            b.velocity.x.abs() < 1e-12 && b.velocity.z.abs() < 1e-12,
            "free fall gained lateral velocity"
        );
    }

    #[test]
    fn debris_is_stopped_by_a_static_obstacle() {
        // A fragment sliding toward a static box (ground far below, no gravity) is pushed back, not
        // through it. Mirrors the app's `Obstacle` setup — box centred at the body via `shape_pose`
        // (local centre ZERO, dry_com ZERO), with broad-phase bounds.
        use crate::collision::BoxShape;
        use glam::DMat3;
        let mut frags = vec![cube_fragment(
            DVec3::new(0.0, 0.5, 0.0),
            DVec3::new(8.0, 0.0, 0.0),
        )];
        // A wall centred at x=3, spanning x∈[2.5,3.5].
        let half = DVec3::new(0.5, 1.0, 1.0);
        let wall_body = ActiveBody::new(
            DVec3::new(3.0, 0.5, 0.0),
            DVec3::ZERO,
            1.0e12,
            DMat3::IDENTITY,
        );
        let wall_shape = CollisionShape::CuboidCompound(vec![BoxShape {
            center: DVec3::ZERO,
            half_extents: half,
        }]);
        let obstacles = [StaticCollider {
            body: &wall_body,
            shape: &wall_shape,
            bounds: Some(Bounds {
                aabb_min: -half,
                aabb_max: half,
                sphere_center: DVec3::ZERO,
                sphere_radius: half.length(),
            }),
        }];
        let ground = flat_ground(-1000.0);
        let params = ContactParams::default();
        for _ in 0..400 {
            step_debris(&mut frags, &ground, &obstacles, DVec3::ZERO, &params, 0.004);
        }
        let b = &frags[0].1;
        assert!(b.position.is_finite());
        // The fragment (half-width 0.5) cannot have crossed to the wall's far face (x ≥ 3.5 − 0.5).
        assert!(
            b.position.x < 3.0,
            "fragment ghosted through the wall: x={}",
            b.position.x
        );
    }

    // --- I1 partition ---

    #[test]
    fn connected_bar_is_one_component() {
        let c = bar(4, Material::ALUMINIUM);
        let frags = connected_components(&c, &Severed::new());
        assert_eq!(frags.len(), 1);
        assert_eq!(frags[0].voxels.len(), 4);
    }

    #[test]
    fn severing_a_bond_splits_in_two() {
        let c = bar(4, Material::ALUMINIUM);
        // Cut between x=1 and x=2.
        let mut severed = Severed::new();
        severed.insert(bond(IVec3::new(1, 0, 0), IVec3::new(2, 0, 0)));
        let frags = connected_components(&c, &severed);
        assert_eq!(frags.len(), 2);
        let mut sizes: Vec<usize> = frags.iter().map(|f| f.voxels.len()).collect();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![2, 2]);
    }

    #[test]
    fn fragment_without_control_point_is_uncontrolled() {
        // A bar with a Command device at x=0. Cut it in two: the fragment holding
        // the command device retains control; the other is uncontrolled debris (WI 562).
        use crate::voxel::{Device, DeviceKind};
        let mut c = bar(4, Material::ALUMINIUM);
        c.devices.push(Device::structural(
            IVec3::new(0, 0, 0),
            10.0,
            DeviceKind::Command,
        ));
        let mut severed = Severed::new();
        severed.insert(bond(IVec3::new(1, 0, 0), IVec3::new(2, 0, 0)));
        let frags = connected_components(&c, &severed);
        assert_eq!(frags.len(), 2);
        let controllable = frags.iter().filter(|f| f.has_control_point()).count();
        assert_eq!(
            controllable, 1,
            "exactly one fragment keeps the control point"
        );
        // The fragment containing x=0 is the controllable one; the other is inert.
        let with = frags
            .iter()
            .find(|f| f.voxels.iter().any(|v| v.cell.x == 0))
            .unwrap();
        let without = frags
            .iter()
            .find(|f| f.voxels.iter().all(|v| v.cell.x != 0))
            .unwrap();
        assert!(with.has_control_point());
        assert!(!without.has_control_point());
    }

    #[test]
    fn non_separating_cut_stays_connected() {
        // An L: (0,0,0)-(1,0,0)-(1,1,0). Severing the x bond at y=0 leaves the
        // path around through (1,1,0)? No — (0,0,0) only connects via (1,0,0).
        // Build a loop so a single severed bond does not disconnect it.
        let mut c = VoxelCraft::new(1.0);
        for cell in [
            IVec3::new(0, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(1, 1, 0),
            IVec3::new(0, 1, 0),
        ] {
            c.voxels.push(Voxel {
                cell,
                material: Material::STEEL,
            });
        }
        // Cut one edge of the loop; the ring keeps it connected.
        let mut severed = Severed::new();
        severed.insert(bond(IVec3::new(0, 0, 0), IVec3::new(1, 0, 0)));
        assert_eq!(connected_components(&c, &severed).len(), 1);
    }

    #[test]
    fn devices_follow_their_component() {
        use crate::voxel::{Device, DeviceKind};
        let mut c = bar(4, Material::ALUMINIUM);
        c.devices.push(Device::structural(
            IVec3::new(0, 0, 0),
            50.0,
            DeviceKind::Tank,
        ));
        c.devices.push(Device::structural(
            IVec3::new(3, 0, 0),
            50.0,
            DeviceKind::Engine,
        ));
        let mut severed = Severed::new();
        severed.insert(bond(IVec3::new(1, 0, 0), IVec3::new(2, 0, 0)));
        let frags = connected_components(&c, &severed);
        assert_eq!(frags.len(), 2);
        // Each fragment kept exactly one device.
        assert!(frags.iter().all(|f| f.devices.len() == 1));
    }

    // --- I2 mass conservation ---

    #[test]
    fn severing_conserves_mass() {
        let c = bar(6, Material::STEEL);
        let before = c.mass_properties().unwrap().mass;
        let mut severed = Severed::new();
        severed.insert(bond(IVec3::new(2, 0, 0), IVec3::new(3, 0, 0)));
        let frags = connected_components(&c, &severed);
        assert!((total_mass(&frags) - before).abs() < 1e-9);
    }

    // --- I4 stress proxy ---

    #[test]
    fn strong_axial_load_breaks_weak_load_does_not() {
        // A long aluminium bar decelerating along its length.
        let c = bar(11, Material::ALUMINIUM);
        // Gentle: 1 m/s² — bonds (3.1e8 Pa × 1 m²) trivially hold.
        assert!(break_craft(&c, DVec3::new(-1.0, 0.0, 0.0), DVec3::ZERO).is_none());
        // Brutal: each cell is ~2700 kg; a huge deceleration overwhelms a bond.
        // Required across the mid-cut ≈ (outboard mass) · a; pick a that exceeds
        // 3.1e8 N. Outboard ~5 cells ≈ 13_500 kg → a ≈ 3e8/13500 ≈ 2.3e4 enough.
        let frags = break_craft(&c, DVec3::new(-5.0e4, 0.0, 0.0), DVec3::ZERO)
            .expect("brutal load should break it");
        assert!(frags.len() >= 2, "expected a break, got {}", frags.len());
    }

    #[test]
    fn stronger_material_resists_the_same_load() {
        // A load in the window that breaks aluminium (3.1e8 Pa) but not titanium
        // (9.0e8 Pa). Mid-cut outboard ≈ 5 cells; aluminium 2700 kg/cell, titanium
        // 4500. At a = 3.0e4: Al 5·2700·3e4 = 4.05e8 > 3.1e8 (breaks); Ti
        // 5·4500·3e4 = 6.75e8 < 9.0e8 (holds).
        let load = DVec3::new(-3.0e4, 0.0, 0.0);
        let weak = bar(11, Material::ALUMINIUM);
        let strong = bar(11, Material::TITANIUM);
        assert!(break_craft(&weak, load, DVec3::ZERO).is_some());
        assert!(break_craft(&strong, load, DVec3::ZERO).is_none());
    }

    #[test]
    fn single_voxel_never_breaks() {
        let c = bar(1, Material::ALUMINIUM);
        assert!(break_craft(&c, DVec3::new(0.0, -1.0e9, 0.0), DVec3::ZERO).is_none());
    }

    // --- I3 momentum conservation (the kinematic split) ---

    #[test]
    fn split_conserves_linear_and_angular_momentum() {
        // A spinning bar broken in the middle.
        let c = bar(6, Material::ALUMINIUM);
        let mp = c.mass_properties().unwrap();
        let parent = ActiveBody::new(
            DVec3::new(100.0, 0.0, 0.0),
            DVec3::new(0.0, 10.0, 0.0),
            mp.mass,
            mp.inertia,
        )
        .with_angular_velocity(DVec3::new(0.0, 0.0, 2.0));

        // Parent momenta (about the parent CoM). The active body stores angular
        // momentum (I_world·ω) directly as a public field.
        let p_parent = parent.mass * parent.velocity;
        let l_parent = parent.angular_momentum;

        let mut severed = Severed::new();
        severed.insert(bond(IVec3::new(2, 0, 0), IVec3::new(3, 0, 0)));
        let frags = connected_components(&c, &severed);
        let bodies = split_active(&c, &parent, &frags);
        assert_eq!(bodies.len(), 2);

        // Sum fragment momenta about the parent CoM.
        let mut p_sum = DVec3::ZERO;
        let mut l_sum = DVec3::ZERO;
        for b in &bodies {
            p_sum += b.mass * b.velocity;
            let r = b.position - parent.position;
            // spin (stored) + orbital about the parent CoM
            l_sum += b.angular_momentum + b.mass * r.cross(b.velocity - parent.velocity);
        }
        assert!(
            (p_sum - p_parent).length() < 1e-6,
            "linear: {p_sum} vs {p_parent}"
        );
        assert!(
            (l_sum - l_parent).length() < 1e-6,
            "angular: {l_sum} vs {l_parent}"
        );
    }

    #[test]
    fn fracture_flings_fragments_apart_when_spinning() {
        // A fast-spinning bar: centripetal load snaps it; the tips fly off with
        // different velocities (the ω×r fling).
        let c = bar(9, Material::ALUMINIUM);
        let mp = c.mass_properties().unwrap();
        let body = ActiveBody::new(DVec3::ZERO, DVec3::ZERO, mp.mass, mp.inertia)
            .with_angular_velocity(DVec3::new(0.0, 0.0, 200.0)); // very fast spin
        let pieces = fracture(&c, &body, DVec3::ZERO).expect("spin should snap it");
        assert!(pieces.len() >= 2);
        // The fragment bodies do not all share one velocity (tangential fling).
        let v0 = pieces[0].1.velocity;
        assert!(
            pieces.iter().any(|(_, b)| (b.velocity - v0).length() > 1.0),
            "fragments should fly apart"
        );
    }

    #[test]
    fn intact_craft_does_not_fracture() {
        let c = bar(4, Material::TITANIUM);
        let mp = c.mass_properties().unwrap();
        let body = ActiveBody::new(DVec3::ZERO, DVec3::ZERO, mp.mass, mp.inertia);
        assert!(fracture(&c, &body, DVec3::ZERO).is_none());
    }

    // --- WI 594: breakage-on-impact (contact force → fracture) ---

    /// A frangible (low-strength) material so an achievable impact force crosses the threshold.
    const FRANGIBLE: Material = Material {
        density: 2700.0,
        strength: 2.0e6,
        thermal: Thermal::INERT,
    };

    #[test]
    fn hard_impact_force_fractures_but_gentle_does_not() {
        // A 6-cell frangible bar. Axial contact force along its length.
        let c = bar(6, FRANGIBLE);
        let mp = c.mass_properties().unwrap();
        let body = ActiveBody::new(DVec3::ZERO, DVec3::new(50.0, 0.0, 0.0), mp.mass, mp.inertia);
        // Gentle: a small contact force (CoM decel ~30 m/s²) — bonds hold.
        assert!(
            fracture_on_impact(&c, &body, DVec3::new(-5.0e5, 0.0, 0.0)).is_none(),
            "a light touch should not fracture"
        );
        // Hard: a large contact force (CoM decel ~300 m/s²) overruns the 2 MPa bonds.
        let pieces = fracture_on_impact(&c, &body, DVec3::new(-5.0e6, 0.0, 0.0))
            .expect("a hard impact should fracture");
        assert!(pieces.len() >= 2, "expected a break, got {}", pieces.len());
    }

    #[test]
    fn impact_fragments_are_collidable() {
        // Every fragment from an impact has a non-empty collision shape and broad-phase bounds,
        // so it participates in collision thereafter (WI 593).
        use crate::collision::{craft_bounds, craft_collision_shape, CollisionShape};
        let c = bar(6, FRANGIBLE);
        let mp = c.mass_properties().unwrap();
        let body = ActiveBody::new(DVec3::ZERO, DVec3::ZERO, mp.mass, mp.inertia);
        let pieces = fracture_on_impact(&c, &body, DVec3::new(-5.0e6, 0.0, 0.0)).unwrap();
        for (frag, _) in &pieces {
            let CollisionShape::CuboidCompound(boxes) = craft_collision_shape(frag) else {
                panic!("fragment shape is a cuboid compound");
            };
            assert!(!boxes.is_empty(), "fragment has collision boxes");
            assert!(
                craft_bounds(frag).is_some(),
                "fragment has broad-phase bounds"
            );
        }
    }

    #[test]
    fn massless_body_does_not_fracture() {
        let c = bar(4, FRANGIBLE);
        let body = ActiveBody::new(DVec3::ZERO, DVec3::ZERO, 0.0, glam::DMat3::ZERO);
        assert!(fracture_on_impact(&c, &body, DVec3::new(1.0e9, 0.0, 0.0)).is_none());
    }

    #[test]
    fn strong_craft_survives_the_same_impact() {
        // The same hard impact that shatters the frangible bar leaves a titanium bar intact —
        // the threshold is the material's, not a fixed force.
        let c = bar(6, Material::TITANIUM);
        let mp = c.mass_properties().unwrap();
        let body = ActiveBody::new(DVec3::ZERO, DVec3::ZERO, mp.mass, mp.inertia);
        assert!(fracture_on_impact(&c, &body, DVec3::new(-5.0e6, 0.0, 0.0)).is_none());
    }

    // --- durable-format backward compatibility (the serde default) ---

    #[test]
    fn material_without_strength_loads_with_default() {
        let m: Material = serde_json::from_str(r#"{"density": 1234.0}"#).unwrap();
        assert_eq!(m.density, 1234.0);
        assert_eq!(m.strength, Material::default_strength());
    }

    #[test]
    fn fragment_inertia_is_finite_for_single_voxel_fragment() {
        // A 2-cell bar split into two single-voxel fragments: each has a defined,
        // finite inertia (guarded as in WI 505/515).
        let c = bar(2, Material::ALUMINIUM);
        let mp = c.mass_properties().unwrap();
        let body = ActiveBody::new(DVec3::ZERO, DVec3::ZERO, mp.mass, mp.inertia)
            .with_angular_velocity(DVec3::new(1.0, 0.0, 0.0));
        let mut severed = Severed::new();
        severed.insert(bond(IVec3::new(0, 0, 0), IVec3::new(1, 0, 0)));
        let frags = connected_components(&c, &severed);
        let bodies = split_active(&c, &body, &frags);
        for b in &bodies {
            assert!(b.position.is_finite() && b.velocity.is_finite());
            assert!(b.angular_velocity().is_finite());
        }
    }

    /// WI 824 interim (the panels design's stage-5 gap, accepted in the plan):
    /// face panels are **not yet** in the connectivity graph, so an all-panel
    /// beam is unbreakable until WI 828 adds face bonds — this test pins the
    /// interim so 828 has a red/green seam to flip, and asserts the solid beam's
    /// behavior is untouched by the panel-model change. (The WI 716 R2 property —
    /// a panel is never stronger than the solid it replaces — returns as a
    /// face-bond test in WI 828.)
    #[test]
    fn face_panels_are_outside_the_breakage_graph_until_wi_828() {
        let mut solid = VoxelCraft::new(0.5);
        for x in 0..6 {
            solid.voxels.push(Voxel {
                cell: IVec3::new(x, 0, 0),
                material: Material::ALUMINIUM,
            });
        }
        // The converted form of a legacy panel beam: plates only, no voxels.
        let mut panel = solid.clone();
        for v in solid.voxels.clone() {
            panel.set_panel(v.cell, true);
        }
        panel.convert_legacy_panels();
        assert!(panel.voxels.is_empty(), "converted beam is all plates");

        let heavy = DVec3::new(0.0, 200_000.0, 0.0); // a bending load past the beam's strength
        let light = DVec3::new(0.0, 1_000.0, 0.0); // a load neither beam fails under
        assert!(
            failing_cut(&solid, heavy, DVec3::ZERO).is_some(),
            "solid fractures under heavy load (unchanged by WI 824)"
        );
        assert!(failing_cut(&solid, light, DVec3::ZERO).is_none());
        assert!(
            failing_cut(&panel, heavy, DVec3::ZERO).is_none(),
            "plates carry no bonds yet — the WI 828 seam"
        );
    }
}
