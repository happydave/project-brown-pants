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
use std::collections::{HashMap, HashSet};

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

/// Node semantics for the exterior fill (WI 832, the staged-rollout seam): the
/// **compartment** fill walks every cell with any air fraction (an empty cell is
/// all air; a shaped solid cell carries its form's remainder — the R1 admission
/// invariant guarantees that remainder is one region reaching every open face,
/// which is what licenses cell-granular through-flow); the **aero envelope**
/// (WI 827) keeps boolean empty-cell nodes until WI 834 generalizes it.
enum NodeMode<'a> {
    /// Air = strictly-unoccupied cells (the pre-832 view; the aero envelope).
    EmptyCells,
    /// Air = any cell with air fraction > 0 (the map holds shaped solid cells'
    /// remainders; unoccupied cells are implicitly fraction 1).
    AirFractions(&'a HashMap<IVec3, f64>),
}

impl NodeMode<'_> {
    /// Whether the fill may occupy `cell` as an air node.
    fn is_air(&self, solid: &HashSet<IVec3>, cell: IVec3) -> bool {
        match self {
            NodeMode::EmptyCells => !solid.contains(&cell),
            NodeMode::AirFractions(shaped) => {
                !solid.contains(&cell) || shaped.get(&cell).is_some_and(|f| *f > 0.0)
            }
        }
    }
}

/// The air remainders of a craft's shaped **voxel** cells (`1 − form volume`,
/// omitting zero-air forms): the extra nodes the compartment fill walks. Keyed
/// off voxels, so an orphan shape entry (or one on a door cell) contributes
/// nothing.
fn shaped_air_fractions(craft: &VoxelCraft) -> HashMap<IVec3, f64> {
    let mut out = HashMap::new();
    for v in &craft.voxels {
        if let Some(s) = craft.shape_at(v.cell) {
            let air = 1.0 - crate::shape::constants(s.form).volume;
            if air > 0.0 {
                out.insert(v.cell, air);
            }
        }
    }
    out
}

/// The exterior flood-fill shared by compartment derivation and the WI 827 aero
/// **sealed envelope**: the expanded structure bounding box (`lo..=hi`, border all
/// air by construction) and the air cells the exterior can reach through it,
/// crossing a boundary only where **the** per-face coverage predicate
/// ([`VoxelCraft::boundary_sealed`], WI 824/832) says it is not sealed. `solid`
/// is the caller's barrier set (compartments: voxels + *closed* doors; the aero
/// envelope: voxels + *all* doors) and `mode` its node semantics (see
/// [`NodeMode`]).
struct ExteriorFill {
    /// Expanded bounding-box minimum (inclusive).
    lo: IVec3,
    /// Expanded bounding-box maximum (inclusive).
    hi: IVec3,
    /// The exterior-reachable air cells within the box.
    exterior: HashSet<IVec3>,
}

fn exterior_fill(craft: &VoxelCraft, solid: &HashSet<IVec3>, mode: &NodeMode<'_>) -> ExteriorFill {
    // Bounding box of the structure — solids and paneled boundaries (a craft can
    // be all plates, WI 824) — expanded by one cell so the border is all
    // exterior air.
    let (mut lo, mut hi) = (IVec3::MAX, IVec3::MIN);
    for &c in solid {
        lo = lo.min(c);
        hi = hi.max(c);
    }
    for p in &craft.face_panels {
        lo = lo.min(p.cell);
        hi = hi.max(p.cell + p.axis.unit());
    }
    lo -= IVec3::ONE;
    hi += IVec3::ONE;

    let in_bbox = |c: IVec3| {
        c.x >= lo.x && c.x <= hi.x && c.y >= lo.y && c.y <= hi.y && c.z >= lo.z && c.z <= hi.z
    };
    let is_air = |c: IVec3| in_bbox(c) && mode.is_air(solid, c);

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
            if is_air(n) && !craft.boundary_sealed(solid, c, off) && exterior.insert(n) {
                stack.push(n);
            }
        }
    }
    ExteriorFill { lo, hi, exterior }
}

/// The **sealed envelope** (WI 827, panels design stage 4): every cell the exterior
/// flood-fill cannot reach — occupied structure plus the enclosed air sealed behind
/// it. This is the aero cross-section input: a closed plated hull presents its full
/// body to the flow (the air inside goes around). Derived on demand from the same
/// fill machinery compartments use — never stored.
///
/// **Doors are structure for aero**: every door cell is a barrier (and an envelope
/// member) *regardless of open state*, so the area curve is a build-time property —
/// an open hatch does not delete the fuselage's cross-section. (Compartments keep
/// their open/closed door semantics; only the aero input treats doors as fixed.)
pub fn sealed_envelope(craft: &VoxelCraft) -> HashSet<IVec3> {
    let mut solid: HashSet<IVec3> = craft.voxels.iter().map(|v| v.cell).collect();
    for d in &craft.doors {
        solid.insert(d.cell);
    }
    if solid.is_empty() && craft.face_panels.is_empty() {
        return HashSet::new();
    }
    // Boolean nodes until WI 834: shaped cells stay full-cube blockers for aero.
    let fill = exterior_fill(craft, &solid, &NodeMode::EmptyCells);
    let mut envelope = HashSet::new();
    for x in fill.lo.x..=fill.hi.x {
        for y in fill.lo.y..=fill.hi.y {
            for z in fill.lo.z..=fill.hi.z {
                let c = IVec3::new(x, y, z);
                if !fill.exterior.contains(&c) {
                    envelope.insert(c);
                }
            }
        }
    }
    envelope
}

/// Compute the sealed compartments of `craft` given its current door states.
/// Pure and deterministic. Passability between air cells is decided by **the**
/// per-face coverage predicate ([`VoxelCraft::boundary_sealed`], WI 824/832) —
/// solid occupancy (voxels + closed doors), face panels, and shaped-cell
/// coverage all seal through it; no other seal logic exists here. **Nodes are
/// cells with any air fraction** (WI 832): an empty cell is all air, a shaped
/// solid cell carries its form's remainder (per R1, one region reaching every
/// open face), so air flows *through* a wedge cell where its faces are open,
/// and a compartment's volume sums its members' air fractions.
pub fn compartments(craft: &VoxelCraft) -> CompartmentSet {
    // Solid (air barrier) cells: structural voxels plus closed doors.
    let mut solid: HashSet<IVec3> = craft.voxels.iter().map(|v| v.cell).collect();
    for d in &craft.doors {
        if !d.open {
            solid.insert(d.cell);
        }
    }
    if solid.is_empty() && craft.face_panels.is_empty() {
        return CompartmentSet::default();
    }

    let shaped_air = shaped_air_fractions(craft);
    let mode = NodeMode::AirFractions(&shaped_air);
    let fill = exterior_fill(craft, &solid, &mode);
    let (lo, hi) = (fill.lo, fill.hi);
    let exterior = &fill.exterior;
    let air_fraction = |c: IVec3| -> f64 {
        if !solid.contains(&c) {
            1.0
        } else {
            shaped_air.get(&c).copied().unwrap_or(0.0)
        }
    };

    // Interior air = air-carrying cells the exterior never reached. Collect in
    // a sorted order for deterministic component numbering.
    let mut interior: Vec<IVec3> = Vec::new();
    for x in lo.x..=hi.x {
        for y in lo.y..=hi.y {
            for z in lo.z..=hi.z {
                let c = IVec3::new(x, y, z);
                if air_fraction(c) > 0.0 && !exterior.contains(&c) {
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
        let mut volume = 0.0;
        let mut stack = vec![seed];
        seen.insert(seed);
        while let Some(c) = stack.pop() {
            cells.push(c);
            volume += air_fraction(c) * cell_volume;
            for off in NEIGHBOURS {
                let n = c + off;
                if interior_set.contains(&n)
                    && !craft.boundary_sealed(&solid, c, off)
                    && seen.insert(n)
                {
                    stack.push(n);
                }
            }
        }
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

    // --- WI 832: shaped-cell coverage through the one predicate ---

    use crate::shape::{rotations, FillMode, Form, ShapedCell};
    use glam::{DMat3, DVec3};

    /// The rotation index whose matrix equals `want` (the table is frozen, so
    /// the found index is stable).
    fn orientation_of(want: DMat3) -> u8 {
        rotations()
            .iter()
            .position(|r| r.abs_diff_eq(want, 1e-12))
            .expect("rotation in the table") as u8
    }

    fn wedge_at(craft: &mut VoxelCraft, cell: IVec3, orientation: u8) {
        craft.voxels.push(Voxel {
            cell,
            material: Material::ALUMINIUM,
        });
        craft.set_shape(ShapedCell {
            cell,
            form: Form::Wedge,
            orientation,
            fill: FillMode::Solid,
        });
    }

    #[test]
    fn mated_wedges_seal_and_identical_wedges_vent() {
        // Canonical wedge: solid {y ≤ z}; both x faces cover the triangle
        // {y ≤ z}. Rotating by diag(1,−1,−1) complements it ({y ≥ z}), so the
        // pair jointly covers the shared boundary — sealed. Two identically
        // oriented wedges cover the same half — open.
        let complement = orientation_of(DMat3::from_diagonal(DVec3::new(1.0, -1.0, -1.0)));
        let a = IVec3::ZERO;
        let b = IVec3::new(1, 0, 0);

        let mut mated = VoxelCraft::new(1.0);
        wedge_at(&mut mated, a, 0);
        wedge_at(&mut mated, b, complement);
        let solid: HashSet<IVec3> = [a, b].into_iter().collect();
        assert!(
            mated.boundary_sealed(&solid, a, IVec3::X),
            "complementary wedges seal their boundary"
        );

        let mut same = VoxelCraft::new(1.0);
        wedge_at(&mut same, a, 0);
        wedge_at(&mut same, b, 0);
        assert!(
            !same.boundary_sealed(&solid, a, IVec3::X),
            "identical wedges leave the boundary open"
        );
    }

    #[test]
    fn a_paneled_half_face_seals() {
        // A lone wedge's half-open boundary vents bare; a face panel on that
        // boundary covers it fully.
        let a = IVec3::ZERO;
        let mut craft = VoxelCraft::new(1.0);
        wedge_at(&mut craft, a, 0);
        let solid: HashSet<IVec3> = [a].into_iter().collect();
        assert!(
            !craft.boundary_sealed(&solid, a, IVec3::X),
            "a half-covered face is open"
        );
        craft.set_face_panel(a, IVec3::X, Some(Material::GLASS));
        assert!(
            craft.boundary_sealed(&solid, a, IVec3::X),
            "a panel completes the coverage"
        );
    }

    #[test]
    fn a_hull_keeps_or_loses_its_compartment_per_the_wedge_geometry() {
        // A 3³ hollow hull (cavity at (1,1,1)); the z=0 wall cell (1,1,0)
        // becomes a wedge in three orientations:
        //   (a) full face outward  → the cavity GAINS the wedge's air remainder
        //       (volume 1.5 cells);
        //   (b) full face inward   → the cavity stays sealed alone (1.0), the
        //       remainder joins the exterior;
        //   (c) half faces on the wall axis → the cavity VENTS (0 compartments).
        let wall = IVec3::new(1, 1, 0);
        let hull_with = |orientation: Option<u8>| {
            let mut c = hollow_shell(3);
            if let Some(o) = orientation {
                c.set_shape(ShapedCell {
                    cell: wall,
                    form: Form::Wedge,
                    orientation: o,
                    fill: FillMode::Solid,
                });
            }
            c
        };

        // (a) rotate by 180° about Y: solid {y ≤ 1−z} — z0 full, z1 empty.
        let out = orientation_of(DMat3::from_diagonal(DVec3::new(-1.0, 1.0, -1.0)));
        let set = compartments(&hull_with(Some(out)));
        assert_eq!(set.count(), 1, "(a) sealed");
        assert!(
            (set.total_volume() - 1.5).abs() < 1e-9,
            "(a) cavity + the wedge's air remainder: {}",
            set.total_volume()
        );
        assert!(
            set.compartments[0].cells.contains(&wall),
            "(a) the wedge cell is a member"
        );

        // (b) canonical: z1 (inward) full, z0 (outward) empty.
        let set = compartments(&hull_with(Some(0)));
        assert_eq!(set.count(), 1, "(b) still sealed");
        assert!(
            (set.total_volume() - 1.0).abs() < 1e-9,
            "(b) the cavity alone: {}",
            set.total_volume()
        );
        assert!(
            !set.compartments[0].cells.contains(&wall),
            "(b) the wedge's remainder joined the exterior"
        );

        // (c) an orientation whose two z faces are both partial: air crosses
        // the wall through the wedge — vented.
        let c = crate::shape::constants(Form::Wedge);
        let venting = c
            .distinct_orientations
            .iter()
            .copied()
            .find(|&o| {
                let m = crate::shape::face_masks(Form::Wedge, o);
                let partial = |f: usize| {
                    let n = crate::shape::mask_popcount(&m[f]);
                    n > 0 && n < 256
                };
                partial(4) && partial(5)
            })
            .expect("a wedge orientation with both z faces partial");
        let set = compartments(&hull_with(Some(venting)));
        assert_eq!(set.count(), 0, "(c) vented through the wedge");

        // Control: the unshaped hull has its one full-cavity compartment.
        let set = compartments(&hull_with(None));
        assert_eq!(set.count(), 1);
        assert!((set.total_volume() - 1.0).abs() < 1e-9);
    }

    // --- WI 824: face-panel sealing through the one coverage predicate ---

    /// An `n³` box of **face panels** around an empty interior: plates on every
    /// boundary between the inside cells `(0..n)³` and the outside.
    fn panel_box(n: i32) -> VoxelCraft {
        let mut c = VoxelCraft::new(1.0);
        for a in 0..n {
            for b in 0..n {
                for (cell, dir) in [
                    (IVec3::new(0, a, b), IVec3::NEG_X),
                    (IVec3::new(n - 1, a, b), IVec3::X),
                    (IVec3::new(a, 0, b), IVec3::NEG_Y),
                    (IVec3::new(a, n - 1, b), IVec3::Y),
                    (IVec3::new(a, b, 0), IVec3::NEG_Z),
                    (IVec3::new(a, b, n - 1), IVec3::Z),
                ] {
                    c.set_face_panel(cell, dir, Some(Material::ALUMINIUM));
                }
            }
        }
        c
    }

    #[test]
    fn a_panel_box_encloses_and_one_removed_plate_vents_it() {
        // A pure-plate 3³ box (no voxels at all) seals 27 cells.
        let mut c = panel_box(3);
        let set = compartments(&c);
        assert_eq!(set.count(), 1, "an all-plate craft has a compartment");
        assert_eq!(set.compartments[0].cells.len(), 27);
        // Removing one plate vents the whole box to the exterior.
        c.set_face_panel(IVec3::new(0, 0, 0), IVec3::NEG_X, None);
        assert_eq!(compartments(&c).count(), 0, "one missing plate vents it");
    }

    #[test]
    fn an_interior_panel_bulkhead_splits_a_cavity() {
        // A 5³ solid shell whose cavity is split by a plate wall at the x=1|2
        // boundary — a *paneled* bulkhead, no voxels added.
        let mut c = hollow_shell(5);
        assert_eq!(compartments(&c).count(), 1);
        for y in 1..4 {
            for z in 1..4 {
                c.set_face_panel(IVec3::new(1, y, z), IVec3::X, Some(Material::ALUMINIUM));
            }
        }
        let set = compartments(&c);
        assert_eq!(set.count(), 2, "the panel bulkhead splits the cavity");
        let mut sizes: Vec<usize> = set.compartments.iter().map(|c| c.cells.len()).collect();
        sizes.sort();
        assert_eq!(sizes, vec![9, 18], "3×3 slab one side, 2×3×3 the other");
    }

    #[test]
    fn converting_a_legacy_panel_shell_preserves_separation() {
        // R1 topology preservation end-to-end: a flagged shell encloses before
        // conversion and still encloses after (the cavity never reaches the
        // exterior), with the converted wall cells joining sealed space.
        let mut legacy = hollow_shell(5);
        for v in legacy.voxels.clone() {
            legacy.set_panel(v.cell, true);
        }
        assert_eq!(compartments(&legacy).count(), 1, "pre-conversion: sealed");
        legacy.convert_legacy_panels();
        assert!(legacy.voxels.is_empty());
        let set = compartments(&legacy);
        assert!(set.count() >= 1, "post-conversion: still sealed");
        // The 3³ cavity is still enclosed, plus the double-hull wall voids.
        assert!(
            set.total_volume() >= 27.0,
            "cavity (27) plus wall voids enclosed: {}",
            set.total_volume()
        );
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
