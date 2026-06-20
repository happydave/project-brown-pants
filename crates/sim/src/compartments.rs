//! Airtight compartments: the voxel model's fourth role (WI 519).
//!
//! A craft is hollow structure — solid voxels are walls/hull; the empty cells
//! they enclose are air space. A **compartment** is a connected component of
//! *enclosed* empty cells: the empty space the exterior cannot reach. Derived by
//! flood-fill over the lattice, the substrate for decompression/flooding (WI 520).
//!
//! Mechanism: expand the craft's bounding box by one cell; the empty cells on
//! that border are the exterior. Flood the exterior inward through empty cells
//! (open doors are passable; solid voxels and **closed** doors are barriers). The
//! empty cells the exterior cannot reach are enclosed; their face-connected
//! components are the sealed compartments. A solid block has none; a hollow shell
//! has one; an internal wall makes two; an open door in that wall merges them; a
//! hull breach lets the exterior in and removes that compartment from the set.
//!
//! [`CompartmentCache`] holds the set and recomputes **only** when marked dirty
//! (a build/break/door-toggle event), never per frame.

use crate::voxel::VoxelCraft;
use glam::IVec3;
use std::collections::HashSet;

/// One sealed interior volume: its enclosed empty cells and their total volume.
#[derive(Clone, Debug, PartialEq)]
pub struct Compartment {
    /// The enclosed empty cells making up this compartment.
    pub cells: Vec<IVec3>,
    /// Enclosed volume, m³ (`cells.len() × cell_volume`).
    pub volume: f64,
}

/// The set of sealed compartments of a craft.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompartmentSet {
    /// The sealed compartments (vented/exterior space is not included).
    pub compartments: Vec<Compartment>,
}

impl CompartmentSet {
    /// Number of sealed compartments.
    pub fn count(&self) -> usize {
        self.compartments.len()
    }

    /// Total sealed volume, m³.
    pub fn total_volume(&self) -> f64 {
        self.compartments.iter().map(|c| c.volume).sum()
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

/// Compute the sealed compartments of `craft` given its current door states.
/// Pure and deterministic.
pub fn compartments(craft: &VoxelCraft) -> CompartmentSet {
    // Solid (air barrier) cells: structural voxels plus closed doors.
    let mut solid: HashSet<IVec3> = craft.voxels.iter().map(|v| v.cell).collect();
    for d in &craft.doors {
        if !d.open {
            solid.insert(d.cell);
        }
    }
    if solid.is_empty() {
        return CompartmentSet::default();
    }

    // Bounding box of the structure, expanded by one cell so the border is all
    // exterior air.
    let (mut lo, mut hi) = (IVec3::MAX, IVec3::MIN);
    for &c in &solid {
        lo = lo.min(c);
        hi = hi.max(c);
    }
    lo -= IVec3::ONE;
    hi += IVec3::ONE;

    let in_bbox = |c: IVec3| {
        c.x >= lo.x && c.x <= hi.x && c.y >= lo.y && c.y <= hi.y && c.z >= lo.z && c.z <= hi.z
    };
    let is_air = |c: IVec3| in_bbox(c) && !solid.contains(&c);

    // Flood the exterior from the expanded-bbox border (all air there is outside
    // the hull) inward through air cells.
    let mut exterior: HashSet<IVec3> = HashSet::new();
    let mut stack: Vec<IVec3> = Vec::new();
    for x in lo.x..=hi.x {
        for y in lo.y..=hi.y {
            for z in lo.z..=hi.z {
                let c = IVec3::new(x, y, z);
                let on_border =
                    x == lo.x || x == hi.x || y == lo.y || y == hi.y || z == lo.z || z == hi.z;
                if on_border && is_air(c) && exterior.insert(c) {
                    stack.push(c);
                }
            }
        }
    }
    while let Some(c) = stack.pop() {
        for off in NEIGHBOURS {
            let n = c + off;
            if is_air(n) && exterior.insert(n) {
                stack.push(n);
            }
        }
    }

    // Interior air = air cells the exterior never reached. Collect in a sorted
    // order for deterministic component numbering.
    let mut interior: Vec<IVec3> = Vec::new();
    for x in lo.x..=hi.x {
        for y in lo.y..=hi.y {
            for z in lo.z..=hi.z {
                let c = IVec3::new(x, y, z);
                if is_air(c) && !exterior.contains(&c) {
                    interior.push(c);
                }
            }
        }
    }
    let interior_set: HashSet<IVec3> = interior.iter().copied().collect();

    // Face-connected components of the interior air = the compartments.
    let cell_volume = craft.cell_volume();
    let mut seen: HashSet<IVec3> = HashSet::new();
    let mut compartments = Vec::new();
    for &seed in &interior {
        if seen.contains(&seed) {
            continue;
        }
        let mut cells = Vec::new();
        let mut stack = vec![seed];
        seen.insert(seed);
        while let Some(c) = stack.pop() {
            cells.push(c);
            for off in NEIGHBOURS {
                let n = c + off;
                if interior_set.contains(&n) && seen.insert(n) {
                    stack.push(n);
                }
            }
        }
        let volume = cells.len() as f64 * cell_volume;
        compartments.push(Compartment { cells, volume });
    }

    CompartmentSet { compartments }
}

/// A cached compartment set that recomputes **only** when marked dirty — the
/// design's "recomputed only on structural-change events, not per frame". Mark it
/// dirty on build/break/door-toggle; `get` recomputes lazily, otherwise reuses.
#[derive(Clone, Debug)]
pub struct CompartmentCache {
    set: CompartmentSet,
    dirty: bool,
    recomputes: u64,
}

impl CompartmentCache {
    /// Build the cache, computing the set once.
    pub fn new(craft: &VoxelCraft) -> Self {
        Self {
            set: compartments(craft),
            dirty: false,
            recomputes: 1,
        }
    }

    /// Signal a structural change; the next [`get`](Self::get) will recompute.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// The current compartment set, recomputing first iff a structural change was
    /// signalled since the last computation.
    pub fn get(&mut self, craft: &VoxelCraft) -> &CompartmentSet {
        if self.dirty {
            self.set = compartments(craft);
            self.dirty = false;
            self.recomputes += 1;
        }
        &self.set
    }

    /// How many times the set has been (re)computed — for verifying that reuse,
    /// not recomputation, happens when nothing changed.
    pub fn recomputes(&self) -> u64 {
        self.recomputes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::{Door, Material, Voxel};

    /// A solid `n³` block of voxels.
    fn solid_block(n: i32) -> VoxelCraft {
        let mut c = VoxelCraft::new(1.0);
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    c.voxels.push(Voxel {
                        cell: IVec3::new(x, y, z),
                        material: Material::ALUMINIUM,
                    });
                }
            }
        }
        c
    }

    /// A hollow `n³` shell: a solid block with its interior `(1..n-1)³` removed.
    fn hollow_shell(n: i32) -> VoxelCraft {
        let mut c = solid_block(n);
        c.voxels.retain(|v| {
            let p = v.cell;
            !(p.x > 0 && p.x < n - 1 && p.y > 0 && p.y < n - 1 && p.z > 0 && p.z < n - 1)
        });
        c
    }

    // --- I1 / I2 topology ---

    #[test]
    fn solid_block_has_no_compartments() {
        assert_eq!(compartments(&solid_block(4)).count(), 0);
    }

    #[test]
    fn hollow_shell_has_one_compartment() {
        // 5³ shell → 3³ = 27 enclosed cells.
        let set = compartments(&hollow_shell(5));
        assert_eq!(set.count(), 1);
        assert_eq!(set.compartments[0].cells.len(), 27);
        assert!((set.total_volume() - 27.0).abs() < 1e-9);
    }

    #[test]
    fn internal_wall_makes_two_compartments() {
        // A 7×5×5 shell with a solid wall at x=3 splits the interior in two.
        let mut c = VoxelCraft::new(1.0);
        let (nx, ny, nz) = (7, 5, 5);
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let on_shell =
                        x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1;
                    let on_wall = x == 3;
                    if on_shell || on_wall {
                        c.voxels.push(Voxel {
                            cell: IVec3::new(x, y, z),
                            material: Material::ALUMINIUM,
                        });
                    }
                }
            }
        }
        let set = compartments(&c);
        assert_eq!(set.count(), 2, "internal wall should make two compartments");
        // Each side: 2×3×3 = 18 cells (x in {1,2} and {4,5}).
        let mut sizes: Vec<usize> = set.compartments.iter().map(|c| c.cells.len()).collect();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![18, 18]);
    }

    /// A 7×5×5 shell, internal wall at x=3 with a one-cell gap at (3,2,2) where a
    /// door sits.
    fn two_rooms_with_door(open: bool) -> VoxelCraft {
        let mut c = VoxelCraft::new(1.0);
        let (nx, ny, nz) = (7, 5, 5);
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let on_shell =
                        x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1;
                    let on_wall = x == 3 && !(y == 2 && z == 2); // gap at (3,2,2)
                    if on_shell || on_wall {
                        c.voxels.push(Voxel {
                            cell: IVec3::new(x, y, z),
                            material: Material::ALUMINIUM,
                        });
                    }
                }
            }
        }
        c.doors.push(Door {
            cell: IVec3::new(3, 2, 2),
            open,
        });
        c
    }

    #[test]
    fn open_door_merges_closed_door_splits() {
        // Closed door → the gap is solid → two rooms.
        assert_eq!(compartments(&two_rooms_with_door(false)).count(), 2);
        // Open door → air passes through the gap → one compartment.
        assert_eq!(compartments(&two_rooms_with_door(true)).count(), 1);
    }

    #[test]
    fn hull_breach_vents_the_compartment() {
        // Remove a face voxel from a hollow shell → exterior floods in → the
        // interior is no longer sealed.
        let mut c = hollow_shell(5);
        assert_eq!(compartments(&c).count(), 1);
        // Punch a hole in the +x face (at the centre of that face).
        c.voxels.retain(|v| v.cell != IVec3::new(4, 2, 2));
        assert_eq!(
            compartments(&c).count(),
            0,
            "a breached compartment vents to exterior"
        );
    }

    #[test]
    fn empty_and_single_voxel_have_no_compartments() {
        assert_eq!(compartments(&VoxelCraft::new(1.0)).count(), 0);
        let mut one = VoxelCraft::new(1.0);
        one.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::STEEL,
        });
        assert_eq!(compartments(&one).count(), 0);
    }

    #[test]
    fn one_cell_compartment_is_valid() {
        // A 3³ shell encloses exactly one cell.
        let set = compartments(&hollow_shell(3));
        assert_eq!(set.count(), 1);
        assert_eq!(set.compartments[0].cells, vec![IVec3::new(1, 1, 1)]);
    }

    #[test]
    fn volume_uses_cell_size() {
        let mut c = hollow_shell(5);
        c.cell_size = 2.0; // each cell is 8 m³
        let set = compartments(&c);
        assert!((set.total_volume() - 27.0 * 8.0).abs() < 1e-9);
    }

    #[test]
    fn partition_is_deterministic() {
        let c = two_rooms_with_door(false);
        assert_eq!(compartments(&c), compartments(&c));
    }

    // --- I3 recompute discipline ---

    #[test]
    fn cache_recomputes_only_when_dirty() {
        let craft = two_rooms_with_door(false);
        let mut cache = CompartmentCache::new(&craft);
        assert_eq!(cache.recomputes(), 1);
        assert_eq!(cache.get(&craft).count(), 2);
        // No structural change → reused, not recomputed.
        let _ = cache.get(&craft);
        let _ = cache.get(&craft);
        assert_eq!(cache.recomputes(), 1, "reused the cached set");

        // A door toggle is a structural change → recompute on next get.
        let open = two_rooms_with_door(true);
        cache.mark_dirty();
        assert_eq!(cache.get(&open).count(), 1, "open door merged the rooms");
        assert_eq!(cache.recomputes(), 2);
        // And again no change → no further recompute.
        let _ = cache.get(&open);
        assert_eq!(cache.recomputes(), 2);
    }

    #[test]
    fn doors_round_trip_through_serde() {
        let craft = two_rooms_with_door(true);
        let json = serde_json::to_string(&craft).unwrap();
        let back: VoxelCraft = serde_json::from_str(&json).unwrap();
        assert_eq!(craft, back);
        // A craft serialized without `doors` (pre-WI 519) still loads.
        let no_doors = r#"{"cell_size":1.0,"voxels":[],"devices":[],"attachments":[]}"#;
        let loaded: VoxelCraft = serde_json::from_str(no_doors).unwrap();
        assert!(loaded.doors.is_empty());
    }
}
