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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::{Material, Voxel};

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
}
