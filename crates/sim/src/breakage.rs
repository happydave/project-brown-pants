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
use crate::panel_mesh::{edge_cells, panel_edges};
use crate::voxel::{Axis, VoxelCraft, PANEL_FILL};
use glam::{DVec3, IVec3};
use std::collections::{HashMap, HashSet};

/// A node in the structural connectivity graph (WI 828): a solid cell, or a face
/// panel keyed exactly as it is stored (owner cell + normal axis). Pre-828 the
/// graph was cells only; panels join it bonded along their four lattice edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BreakNode {
    /// An occupied structural cell.
    Cell(IVec3),
    /// A face panel on the boundary owned by `(cell, axis)`.
    Panel(IVec3, Axis),
}

impl BreakNode {
    /// Total ordering key (cells before panels, then coordinates), used for the
    /// canonical bond ordering and deterministic bond-list sorting.
    fn key(self) -> (u8, i32, i32, i32, u8) {
        match self {
            BreakNode::Cell(c) => (0, c.x, c.y, c.z, 0),
            BreakNode::Panel(c, a) => (1, c.x, c.y, c.z, a as u8),
        }
    }
}

/// A severed bond: the unordered pair of graph nodes whose connection is cut.
/// Stored canonically (smaller key first) so lookups are order-independent.
pub type Severed = HashSet<(BreakNode, BreakNode)>;

/// Canonical ordering of a node pair, so `(a, b)` and `(b, a)` are the same bond.
fn bond_nodes(a: BreakNode, b: BreakNode) -> (BreakNode, BreakNode) {
    if a.key() <= b.key() {
        (a, b)
    } else {
        (b, a)
    }
}

/// Canonical cell–cell bond (the pre-828 shape, kept for callers/tests that sever
/// between two solid cells).
fn bond(a: IVec3, b: IVec3) -> (BreakNode, BreakNode) {
    bond_nodes(BreakNode::Cell(a), BreakNode::Cell(b))
}

/// One structural bond: an unordered node pair (canonical order) and the tensile
/// strength it withstands, N.
struct StructuralBond {
    a: BreakNode,
    b: BreakNode,
    strength: f64,
}

/// Every structural bond of the craft with its strength, deterministically ordered
/// (WI 828). Three bond kinds:
///
/// - **cell–cell**: face-adjacent occupied cells; strength = the negative-side
///   cell's material × `cell²` (the pre-828 convention, kept so voxel-only crafts
///   evaluate exactly as before).
/// - **panel–cell** and **panel–panel**: a panel bonds along its four lattice
///   edges ([`panel_edges`]) to every occupied cell ([`edge_cells`]) and every
///   other panel sharing an edge. Each shared edge contributes the plate's edge
///   cross-section (`PANEL_FILL × cell²`) × the **weaker** endpoint's material —
///   the WI 716 R2 discipline: a plate joint is never stronger than the solid face
///   bond it stands in for (a skin plate reaches its backing cell through all four
///   edges: 0.2 of a face bond, still 5× weaker).
fn structural_bonds(craft: &VoxelCraft) -> Vec<StructuralBond> {
    let face_area = craft.cell_size * craft.cell_size;
    let edge_area = PANEL_FILL * face_area;
    let material_of: HashMap<IVec3, f64> = craft
        .voxels
        .iter()
        .map(|v| (v.cell, v.material.strength))
        .collect();
    let mut acc: HashMap<(BreakNode, BreakNode), f64> = HashMap::new();

    // Cell–cell face bonds (positive offsets only: each pair once, negative-side
    // cell's material — the pre-828 numbers).
    for v in &craft.voxels {
        for off in [IVec3::X, IVec3::Y, IVec3::Z] {
            let n = v.cell + off;
            if material_of.contains_key(&n) {
                *acc.entry(bond(v.cell, n)).or_default() += v.material.strength * face_area;
            }
        }
    }

    // Panel–cell edge bonds.
    for p in &craft.face_panels {
        let node = BreakNode::Panel(p.cell, p.axis);
        for (origin, axis) in panel_edges(p) {
            for c in edge_cells(origin, axis) {
                if let Some(&cell_strength) = material_of.get(&c) {
                    let s = p.material.strength.min(cell_strength);
                    *acc.entry(bond_nodes(node, BreakNode::Cell(c))).or_default() += s * edge_area;
                }
            }
        }
    }

    // Panel–panel edge bonds: pairwise among the panels sharing each lattice edge.
    let mut edge_panels: HashMap<(i32, i32, i32, u8), Vec<usize>> = HashMap::new();
    for (i, p) in craft.face_panels.iter().enumerate() {
        for (origin, axis) in panel_edges(p) {
            edge_panels
                .entry((origin.x, origin.y, origin.z, axis as u8))
                .or_default()
                .push(i);
        }
    }
    for list in edge_panels.values() {
        for (i, &pi) in list.iter().enumerate() {
            for &pj in &list[i + 1..] {
                let (a, b) = (&craft.face_panels[pi], &craft.face_panels[pj]);
                let s = a.material.strength.min(b.material.strength);
                *acc.entry(bond_nodes(
                    BreakNode::Panel(a.cell, a.axis),
                    BreakNode::Panel(b.cell, b.axis),
                ))
                .or_default() += s * edge_area;
            }
        }
    }

    let mut bonds: Vec<StructuralBond> = acc
        .into_iter()
        .map(|((a, b), strength)| StructuralBond { a, b, strength })
        .collect();
    bonds.sort_by_key(|b| (b.a.key(), b.b.key()));
    bonds
}

/// Partition the craft's structural nodes — occupied cells **and face panels**
/// (WI 828) — into maximal connected components, excluding any `severed` bonds
/// from adjacency (cell–cell face bonds; panel edge bonds). Each component becomes
/// a fragment [`VoxelCraft`] (same `cell_size`, its voxels **and panels** plus the
/// devices and attachment points whose cell lies in it). A connected craft yields
/// one fragment; an already-disconnected lattice yields its true components — so a
/// torn-off plate is a fragment like any other, and a plate-only hull partitions
/// by edge adjacency.
pub fn connected_components(craft: &VoxelCraft, severed: &Severed) -> Vec<VoxelCraft> {
    // Adjacency from the structural bond list, minus the severed pairs.
    // Bond order is deterministic, so neighbour lists (and thus fragment
    // numbering) are too.
    let mut adjacency: HashMap<BreakNode, Vec<BreakNode>> = HashMap::new();
    for b in structural_bonds(craft) {
        if severed.contains(&(b.a, b.b)) {
            continue;
        }
        adjacency.entry(b.a).or_default().push(b.b);
        adjacency.entry(b.b).or_default().push(b.a);
    }

    // Flood-fill components; seeds in store order (voxels, then panels).
    let seeds: Vec<BreakNode> = craft
        .voxels
        .iter()
        .map(|v| BreakNode::Cell(v.cell))
        .chain(
            craft
                .face_panels
                .iter()
                .map(|p| BreakNode::Panel(p.cell, p.axis)),
        )
        .collect();
    let mut component_of: HashMap<BreakNode, usize> = HashMap::new();
    let mut next_component = 0;
    for &seed in &seeds {
        if component_of.contains_key(&seed) {
            continue;
        }
        let id = next_component;
        next_component += 1;
        let mut stack = vec![seed];
        component_of.insert(seed, id);
        while let Some(node) = stack.pop() {
            if let Some(neighbours) = adjacency.get(&node) {
                for &n in neighbours {
                    if let std::collections::hash_map::Entry::Vacant(e) = component_of.entry(n) {
                        e.insert(id);
                        stack.push(n);
                    }
                }
            }
        }
    }

    // Build one fragment craft per component.
    let mut fragments: Vec<VoxelCraft> = (0..next_component)
        .map(|_| VoxelCraft::new(craft.cell_size))
        .collect();
    for v in &craft.voxels {
        fragments[component_of[&BreakNode::Cell(v.cell)]]
            .voxels
            .push(*v);
    }
    for p in &craft.face_panels {
        fragments[component_of[&BreakNode::Panel(p.cell, p.axis)]]
            .face_panels
            .push(*p);
    }
    for d in &craft.devices {
        if let Some(&id) = component_of.get(&BreakNode::Cell(d.cell)) {
            fragments[id].devices.push(*d);
        }
    }
    for a in &craft.attachments {
        if let Some(&id) = component_of.get(&BreakNode::Cell(a.cell)) {
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
///
/// **Panels (WI 828)**: plates are graph nodes; their edge bonds cross planes like
/// any other, plate masses load the outboard side at their face centres, and the
/// candidate planes span panel extents (a plate-only hull has planes to fail at).
/// Side rule: a cell sides by its layer, a panel by its owner cell's layer —
/// except a **normal-axis panel lying exactly on the cut plane, which sides
/// outboard** (the skin peels outward; without this a skin plate could never be
/// cut from its backing).
pub fn failing_cut(craft: &VoxelCraft, a_cm: DVec3, omega: DVec3) -> Option<Severed> {
    let mp = craft.mass_properties()?;
    let com = mp.center_of_mass;
    let cell_size = craft.cell_size;
    let cell_volume = craft.cell_volume();
    let panel_mass = |p: &crate::voxel::FacePanel| p.material.density * craft.panel_volume();
    let bonds = structural_bonds(craft);
    if bonds.is_empty() {
        return None; // a single node cannot be cut apart
    }

    let mut best: Option<(f64, Severed)> = None; // (load/strength ratio, bonds)

    for axis in [Axis::X, Axis::Y, Axis::Z] {
        let coord = |c: IVec3| match axis {
            Axis::X => c.x,
            Axis::Y => c.y,
            Axis::Z => c.z,
        };
        // Candidate planes span the voxel layers and the panel extents (a
        // normal-axis panel touches both boundary layers).
        let range = craft
            .voxels
            .iter()
            .map(|v| coord(v.cell))
            .chain(
                craft
                    .face_panels
                    .iter()
                    .flat_map(|p| [coord(p.cell), coord(p.cell + p.axis.unit())]),
            )
            .fold(None, |acc: Option<(i32, i32)>, k| {
                let (mn, mx) = acc.unwrap_or((k, k));
                Some((mn.min(k), mx.max(k)))
            });
        let Some((lo, hi)) = range else { continue };

        // Each plane sits between layer k and k+1.
        for k in lo..hi {
            // Outboard side = the layers farther from the CoM along this axis.
            let com_layer = match axis {
                Axis::X => com.x,
                Axis::Y => com.y,
                Axis::Z => com.z,
            } / cell_size
                - 0.5;
            let high_is_outboard =
                (k as f64 + 1.0 - com_layer).abs() > (k as f64 - com_layer).abs();

            // Which side of the plane a node is on (`true` = the high side).
            let on_high = |n: BreakNode| -> bool {
                match n {
                    BreakNode::Cell(c) => coord(c) > k,
                    BreakNode::Panel(c, a) => {
                        if a == axis && coord(c) == k {
                            high_is_outboard // on-plane skin peels outward
                        } else {
                            coord(c) > k
                        }
                    }
                }
            };

            // Bonds crossing the plane, and their total strength.
            let mut crossing: Severed = HashSet::new();
            let mut strength = 0.0;
            for b in &bonds {
                if on_high(b.a) != on_high(b.b) {
                    crossing.insert((b.a, b.b));
                    strength += b.strength;
                }
            }
            if crossing.is_empty() {
                continue;
            }

            // Net inertial force the bonds must transmit to the outboard side.
            let mut load = DVec3::ZERO;
            for v in &craft.voxels {
                if on_high(BreakNode::Cell(v.cell)) != high_is_outboard {
                    continue;
                }
                let m = v.material.density * cell_volume;
                let r = cell_center(cell_size, v.cell) - com;
                load += m * point_acceleration(a_cm, omega, r);
            }
            for p in &craft.face_panels {
                if on_high(BreakNode::Panel(p.cell, p.axis)) != high_is_outboard {
                    continue;
                }
                let r = craft.face_center(p) - com;
                load += panel_mass(p) * point_acceleration(a_cm, omega, r);
            }
            for d in &craft.devices {
                if (coord(d.cell) > k) != high_is_outboard {
                    continue;
                }
                let r = cell_center(cell_size, d.cell) - com;
                load += d.mass * point_acceleration(a_cm, omega, r);
            }

            let ratio = load.length() / strength;
            if ratio > 1.0 && best.as_ref().is_none_or(|(b, _)| ratio > *b) {
                best = Some((ratio, crossing));
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

    // --- WI 828: face panels in the breakage graph ---

    /// The WI 824 interim seam, flipped by WI 828: a converted all-plate beam is
    /// a structure now — its mullion edge bonds carry load and fail like any
    /// other, while the solid beam's behaviour is untouched.
    #[test]
    fn a_converted_plate_beam_breaks_like_a_structure() {
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
            "solid fractures under heavy load (unchanged)"
        );
        assert!(failing_cut(&solid, light, DVec3::ZERO).is_none());
        assert!(
            failing_cut(&panel, heavy, DVec3::ZERO).is_some(),
            "the plate tube's edge bonds fail under the heavy load (WI 828)"
        );
        assert!(
            failing_cut(&panel, light, DVec3::ZERO).is_none(),
            "the plate tube holds a light load"
        );
    }

    #[test]
    fn a_plate_joint_is_never_stronger_than_the_solid_it_replaces() {
        // The R2 comparison with the outboard mass held constant: two cells with a
        // one-cell gap, bridged (a) by a solid cell (full face bonds, 1.0·S·s²)
        // and (b) by a plate spanning the gap's top boundary (one lattice edge to
        // each side, 0.05·S·s²). Pulling the far cell at the same acceleration:
        // the plate bridge tears where the solid bridge holds.
        //   plate bond: 0.05 · 3.1e8 = 1.55e7 N; far cell 2700 kg
        //   solid bond: 3.1e8 N
        //   a = 2.0e4 → load 5.4e7 N: plate ratio 3.5 (breaks), solid 0.17 (holds)
        let far = IVec3::new(2, 0, 0);
        let mut solid_bridge = VoxelCraft::new(1.0);
        for cell in [IVec3::ZERO, IVec3::new(1, 0, 0), far] {
            solid_bridge.voxels.push(Voxel {
                cell,
                material: Material::ALUMINIUM,
            });
        }
        let mut plate_bridge = VoxelCraft::new(1.0);
        for cell in [IVec3::ZERO, far] {
            plate_bridge.voxels.push(Voxel {
                cell,
                material: Material::ALUMINIUM,
            });
        }
        // The plate lies on the gap cell's +Y boundary; its x=1 and x=2 lattice
        // edges seat into the two cells (one edge each).
        plate_bridge.set_face_panel(IVec3::new(1, 0, 0), IVec3::Y, Some(Material::ALUMINIUM));

        let pull = DVec3::new(2.0e4, 0.0, 0.0);
        assert!(
            failing_cut(&plate_bridge, pull, DVec3::ZERO).is_some(),
            "the plate bridge tears"
        );
        assert!(
            failing_cut(&solid_bridge, pull, DVec3::ZERO).is_none(),
            "the solid bridge holds the same pull"
        );
    }

    #[test]
    fn an_overloaded_plate_sail_sheds_while_the_frame_survives() {
        // A 3-cell beam continued by a 5-plate coplanar sail (each joint one
        // mullion edge, 0.05·S·s² = 1.55e7 N). Per unit acceleration:
        //   shed joint:  5 plates · 135 kg = 675 kg  / 1.55e7 → fails above a ≈ 2.3e4
        //   beam plane:  (1 cell + sail) = 3375 kg   / 3.1e8  → fails above a ≈ 9.2e4
        //   all-solid 8-cell beam, worst plane: 4·2700 / 3.1e8 → fails above a ≈ 2.87e4
        // a = 2.6e4 sits in the window: the sail sheds, both beams' solid joints hold.
        let mut framed = VoxelCraft::new(1.0);
        for x in 0..3 {
            framed.voxels.push(Voxel {
                cell: IVec3::new(x, 0, 0),
                material: Material::ALUMINIUM,
            });
        }
        for x in 3..8 {
            framed.set_face_panel(IVec3::new(x, 0, 0), IVec3::Y, Some(Material::ALUMINIUM));
        }
        let mut all_solid = VoxelCraft::new(1.0);
        for x in 0..8 {
            all_solid.voxels.push(Voxel {
                cell: IVec3::new(x, 0, 0),
                material: Material::ALUMINIUM,
            });
        }

        let pull = DVec3::new(2.6e4, 0.0, 0.0);
        let frags = break_craft(&framed, pull, DVec3::ZERO).expect("the sail sheds");
        assert_eq!(frags.len(), 2);
        let sail = frags
            .iter()
            .find(|f| f.voxels.is_empty())
            .expect("one fragment is the plate sail");
        let frame = frags
            .iter()
            .find(|f| !f.voxels.is_empty())
            .expect("one fragment is the solid frame");
        assert_eq!(
            sail.face_panels.len(),
            5,
            "the whole sail sheds as one piece"
        );
        assert_eq!(frame.voxels.len(), 3, "the frame survives intact");
        assert!(frame.face_panels.is_empty());
        assert!(
            break_craft(&all_solid, pull, DVec3::ZERO).is_none(),
            "the equivalent all-solid beam holds the same pull"
        );
    }

    #[test]
    fn fragments_inherit_their_plates() {
        // A two-cell beam with an end-cap plate on each cell: severing the middle
        // yields two fragments, each carrying its own plate — nothing dropped.
        let mut c = VoxelCraft::new(1.0);
        for cell in [IVec3::ZERO, IVec3::new(1, 0, 0)] {
            c.voxels.push(Voxel {
                cell,
                material: Material::ALUMINIUM,
            });
        }
        c.set_face_panel(IVec3::ZERO, IVec3::NEG_X, Some(Material::ALUMINIUM));
        c.set_face_panel(IVec3::new(1, 0, 0), IVec3::X, Some(Material::GLASS));
        let before = c.mass_properties().unwrap().mass;

        let mut severed = Severed::new();
        severed.insert(bond(IVec3::ZERO, IVec3::new(1, 0, 0)));
        let frags = connected_components(&c, &severed);
        assert_eq!(frags.len(), 2);
        assert!(frags.iter().all(|f| f.face_panels.len() == 1));
        assert!(
            (total_mass(&frags) - before).abs() < 1e-9,
            "plates keep their mass"
        );
    }

    #[test]
    fn plate_only_crafts_partition_by_edge_adjacency() {
        // Two coplanar plates sharing a lattice edge are one component; two
        // disjoint plates are two — a plate-only hull is breakable structure.
        let mut joined = VoxelCraft::new(1.0);
        joined.set_face_panel(IVec3::ZERO, IVec3::Y, Some(Material::ALUMINIUM));
        joined.set_face_panel(IVec3::new(1, 0, 0), IVec3::Y, Some(Material::ALUMINIUM));
        assert_eq!(connected_components(&joined, &Severed::new()).len(), 1);

        let mut apart = VoxelCraft::new(1.0);
        apart.set_face_panel(IVec3::ZERO, IVec3::Y, Some(Material::ALUMINIUM));
        apart.set_face_panel(IVec3::new(3, 0, 0), IVec3::Y, Some(Material::ALUMINIUM));
        assert_eq!(connected_components(&apart, &Severed::new()).len(), 2);
    }

    #[test]
    fn a_lone_plate_never_breaks() {
        // The panel analogue of `single_voxel_never_breaks`: one node, no bonds,
        // nothing to cut.
        let mut c = VoxelCraft::new(1.0);
        c.set_face_panel(IVec3::ZERO, IVec3::Y, Some(Material::ALUMINIUM));
        assert!(break_craft(&c, DVec3::new(0.0, -1.0e9, 0.0), DVec3::ZERO).is_none());
    }
}
