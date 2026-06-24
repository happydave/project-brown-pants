//! The voxel ship: the single craft representation (WI 505).
//!
//! A craft is a sparse voxel lattice (occupied integer cells, each of a
//! structural material) plus mounted devices. The design's central bet is that
//! this one representation answers mass/inertia, aero cross-section, breakage, and
//! compartments; Toy 5 validates the first two — **mass + inertia tensor** and the
//! **aero cross-sectional-area curve** are derived here, from the same voxels.
//!
//! All derivations are pure functions of the craft data, so they are unit-tested
//! headless. Materials are data-driven (a new material is new constants, not new
//! code), consistent with the WI 497 field pattern.

use glam::{DMat3, DVec3, IVec3};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A structural material: the data the discipline says to model — density and
/// tensile strength. A new material is a new value, not a new code path.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Material {
    /// Density, kg/m³.
    pub density: f64,
    /// Tensile strength, Pa — the structural stress a bond of this material
    /// withstands before breaking (consumed by connected-component breakage,
    /// WI 518). Defaulted on load so pre-strength saves stay backward-loadable.
    #[serde(default = "Material::default_strength")]
    pub strength: f64,
}

impl Material {
    /// Aluminium-like structural material.
    pub const ALUMINIUM: Material = Material {
        density: 2_700.0,
        strength: 3.1e8,
    };
    /// Steel-like structural material.
    pub const STEEL: Material = Material {
        density: 7_850.0,
        strength: 5.0e8,
    };
    /// Titanium-like structural material.
    pub const TITANIUM: Material = Material {
        density: 4_500.0,
        strength: 9.0e8,
    };
    /// Light composite.
    pub const COMPOSITE: Material = Material {
        density: 1_600.0,
        strength: 6.0e8,
    };

    /// The strength assumed for a material loaded from a pre-strength save: high
    /// enough to be effectively unbreakable, so old craft do not spontaneously
    /// shatter.
    pub fn default_strength() -> f64 {
        1.0e12
    }
}

/// A single occupied cell of the lattice.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Voxel {
    /// Integer grid coordinate (the cell's minimum corner).
    pub cell: IVec3,
    /// The cell's structural material.
    pub material: Material,
}

/// The coarse lattice tag for a device — what kind of part it is. Used by the
/// lattice/breakage path (e.g. `has_control_point` keys on `Command`) and by
/// serialization. A device's *flight behaviour* is the separate, optional
/// [`crate::control::DeviceFunction`] it carries (WI 570); `kind` and `function` are
/// kept consistent by the constructors below.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    /// A control point (cockpit / command pod / probe core) — the lattice's command
    /// device; the only kind `has_control_point` recognises.
    Command,
    Engine,
    Tank,
    Rcs,
    /// A control computer (SAS / autopilot / tuning) — distinct from a control point,
    /// so a computer-only fragment is not controllable on its own (WI 570).
    Computer,
    /// An electrical battery — an electricity reservoir powering the computers (WI 570).
    Battery,
}

impl DeviceKind {
    /// Bulk density of this device kind, kg/m³ (WI 615). A device fills roughly one cell, so its mass
    /// is `density × cell_size³` — the same shape as a structural voxel's `material.density × cell³`.
    /// Densities are chosen comparable to the structural materials (≈1600–7850 kg/m³) so a device
    /// weighs about as much as the voxels around it at any cell size, instead of a fixed mass that
    /// dominates a small build. Tank is light (an empty shell; propellant is modelled separately).
    pub fn density(self) -> f64 {
        match self {
            DeviceKind::Command => 800.0,
            DeviceKind::Computer => 1_200.0,
            DeviceKind::Battery => 2_500.0,
            DeviceKind::Engine => 3_000.0,
            DeviceKind::Tank => 600.0,
            DeviceKind::Rcs => 1_000.0,
        }
    }
}

/// The mass (kg) of a device of `kind` filling a cell of side `cell_size` m (WI 615):
/// `density(kind) × cell_size³`. Use this at every placement site so device mass tracks build scale.
pub fn device_mass(kind: DeviceKind, cell_size: f64) -> f64 {
    kind.density() * cell_size * cell_size * cell_size
}

/// A mounted functional device: a mass at a cell. Contributes to mass and inertia
/// (never to the voxel-occupancy area curve). Beyond mass, a device may carry a
/// [`crate::control::DeviceFunction`] giving it real flight behaviour assembled into
/// the craft's control system (WI 570); a device without one is structural mass only.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Device {
    /// Mounting cell.
    pub cell: IVec3,
    /// Device mass, kg.
    pub mass: f64,
    /// Device type (coarse lattice tag).
    pub kind: DeviceKind,
    /// Optional flight function (WI 570). Defaulted absent so pre-570 saves load.
    #[serde(default)]
    pub function: Option<crate::control::DeviceFunction>,
}

impl Device {
    /// A structural / inert-mass device of `kind` (no flight function) — the pre-570
    /// shape.
    pub fn structural(cell: IVec3, mass: f64, kind: DeviceKind) -> Self {
        Self {
            cell,
            mass,
            kind,
            function: None,
        }
    }

    /// A control point device (kind `Command` + a control-point function), crewed or
    /// uncrewed (WI 570).
    pub fn control_point(cell: IVec3, mass: f64, crewed: bool) -> Self {
        use crate::control::{ControlPoint, DeviceFunction};
        let point = if crewed {
            ControlPoint::crewed()
        } else {
            ControlPoint::uncrewed()
        };
        Self {
            cell,
            mass,
            kind: DeviceKind::Command,
            function: Some(DeviceFunction::ControlPoint(point)),
        }
    }

    /// A control-computer device (kind `Computer` + a computer function) granting
    /// `computer`'s tier while powered (WI 570).
    pub fn computer(cell: IVec3, mass: f64, computer: crate::control::ControlComputer) -> Self {
        use crate::control::DeviceFunction;
        Self {
            cell,
            mass,
            kind: DeviceKind::Computer,
            function: Some(DeviceFunction::Computer(computer)),
        }
    }

    /// A battery device (kind `Battery` + a battery function) providing an electricity
    /// reservoir (WI 570).
    pub fn battery(cell: IVec3, mass: f64, spec: crate::control::BatterySpec) -> Self {
        use crate::control::DeviceFunction;
        Self {
            cell,
            mass,
            kind: DeviceKind::Battery,
            function: Some(DeviceFunction::Battery(spec)),
        }
    }
}

/// A face direction, used by subassembly attachment points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Face {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

/// A subassembly attachment point: a cell and the face it mates on.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttachmentPoint {
    pub cell: IVec3,
    pub face: Face,
}

/// A door / hatch occupying an (empty) cell in the structure. When `open` the cell
/// is passable air; when closed it is an air barrier (solid for the compartment
/// flood-fill, WI 519). Doors affect air topology only — not mass, aero, or
/// breakage (a thin door is structurally negligible here).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Door {
    /// The empty cell the door occupies (a gap in a wall).
    pub cell: IVec3,
    /// Open (passable air) or closed (a barrier).
    pub open: bool,
}

/// An axis along which the cross-sectional-area curve is sliced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

/// Derived mass properties of a craft, in its own local frame (metres, kg).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MassProperties {
    /// Total mass, kg.
    pub mass: f64,
    /// Centre of mass, metres.
    pub center_of_mass: DVec3,
    /// Inertia tensor about the centre of mass, kg·m² (symmetric, PSD).
    pub inertia: DMat3,
    /// Principal moments of inertia (eigenvalues of `inertia`).
    pub principal_moments: DVec3,
    /// Principal axes (columns), the eigenvectors of `inertia`.
    pub principal_axes: DMat3,
}

/// A non-voxel **catalog part** attached to a chassis at a continuous body-frame
/// pose (WI 607). Unlike a [`Device`] (locked to an integer lattice cell), a part
/// mounts at an arbitrary sub-cell offset — which is why wheels (and seat / antenna /
/// solar / bumper) are parts, not devices: the rover core mounts wheels at sub-cell,
/// outboard positions a cell grid cannot express. A part contributes mass and
/// (point-mass, parallel-axis) inertia like a device, and is render-relevant; a
/// [`PartKind::Wheel`] is additionally physics-relevant (it becomes a rover wheel).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Part {
    /// Mount position in the craft's local (lattice) frame, metres — the same frame
    /// as [`Voxel::cell`] centres. Assembly subtracts the centre of mass to get the
    /// body-frame (CoM-relative) mount the rover core expects.
    pub mount: DVec3,
    /// Part mass, kg.
    pub mass: f64,
    /// What the part is (and its kind-specific parameters).
    pub kind: PartKind,
}

/// The kind of a catalog [`Part`] (WI 607). `Wheel` carries the physical and
/// group parameters that assembly turns into a rover wheel; the remaining kinds are
/// inert mass with a recognisable role (their meshes land in WI 608).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartKind {
    /// A wheel + suspension + tire — the one physics-relevant part.
    Wheel(WheelPart),
    /// A crew seat (recognisability; crew/control comes from a control point device).
    Seat,
    /// A communications antenna (cosmetic now; a comms model is later).
    Antenna,
    /// A solar panel (electric charging is wired in the powertrain WI 609).
    SolarPanel,
    /// A bumper (structure that reads as a rover and breaks on impact, WI 610).
    Bumper,
}

/// Physical and drivetrain parameters of a [`PartKind::Wheel`] (WI 607). The
/// physical fields mirror [`crate::rover::Wheel`]'s suspension/tire parameters so
/// assembly maps a wheel part straight onto a rover wheel; the group flags record
/// the drivetrain membership the rover Test path commands (drive torque / steering)
/// by group rather than by hard-coded index.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WheelPart {
    /// Wheel radius (m).
    pub radius: f64,
    /// Suspension free length (m).
    pub rest_length: f64,
    /// Spring stiffness (N/m).
    pub stiffness: f64,
    /// Suspension damping (N·s/m).
    pub damping: f64,
    /// Maximum suspension normal force (N).
    pub max_force: f64,
    /// Wheel rotational inertia (kg·m²).
    pub wheel_inertia: f64,
    /// In the drive group (receives engine/motor torque).
    pub drive: bool,
    /// In the steer group (turns with steering input).
    pub steer: bool,
}

impl WheelPart {
    /// A wheel part with the rover core's sensible defaults (mirrors
    /// [`crate::rover::Wheel::new`]); `drive`/`steer` choose the drivetrain groups.
    pub fn new(drive: bool, steer: bool) -> Self {
        Self {
            radius: 0.35,
            rest_length: 0.35,
            stiffness: 4.5e4,
            damping: 8.0e3,
            max_force: 1.0e6,
            wheel_inertia: 8.0,
            drive,
            steer,
        }
    }

    /// A wheel part sized to the build's `cell_size` (WI 612 feedback): radius and suspension
    /// travel scale with the cell so a 0.1 m build gets small wheels on short suspension, not
    /// metre-scale stilts. The force parameters (`stiffness`/`damping`/`max_force`) are placeholders
    /// — [`crate::rover::assemble_rover`] re-sizes them to the assembled rover's mass.
    pub fn for_cell_size(cell_size: f64, drive: bool, steer: bool) -> Self {
        let radius = 1.5 * cell_size;
        Self {
            radius,
            rest_length: 0.5 * cell_size,
            stiffness: 4.5e4,
            damping: 8.0e3,
            max_force: 1.0e6,
            // ~ m·r² for a light wheel; keeps spin-up responsive at small scale.
            wheel_inertia: (radius * radius * 40.0).max(0.05),
            drive,
            steer,
        }
    }
}

/// A craft: a sparse voxel lattice with devices and subassembly attachment points.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VoxelCraft {
    /// Edge length of one cell, metres.
    pub cell_size: f64,
    /// Occupied structural cells.
    pub voxels: Vec<Voxel>,
    /// Mounted devices.
    pub devices: Vec<Device>,
    /// Subassembly attachment points (empty for a standalone craft).
    pub attachments: Vec<AttachmentPoint>,
    /// Doors / hatches occupying gaps in the structure (compartment flood-fill,
    /// WI 519). Defaulted on load so pre-doors saves stay backward-loadable.
    #[serde(default)]
    pub doors: Vec<Door>,
    /// Attached catalog parts (wheels, seat, antenna, solar, bumper — WI 607).
    /// Defaulted on load so pre-parts saves stay backward-loadable.
    #[serde(default)]
    pub parts: Vec<Part>,
}

impl Default for VoxelCraft {
    fn default() -> Self {
        Self {
            cell_size: 1.0,
            voxels: Vec::new(),
            devices: Vec::new(),
            attachments: Vec::new(),
            doors: Vec::new(),
            parts: Vec::new(),
        }
    }
}

impl VoxelCraft {
    /// An empty craft with the given cell size.
    pub fn new(cell_size: f64) -> Self {
        Self {
            cell_size,
            ..Default::default()
        }
    }

    /// Volume of one cell, m³.
    pub fn cell_volume(&self) -> f64 {
        self.cell_size * self.cell_size * self.cell_size
    }

    /// Total occupied voxel volume, m³ (excludes devices).
    pub fn occupied_volume(&self) -> f64 {
        self.voxels.len() as f64 * self.cell_volume()
    }

    /// Whether this craft (or breakage fragment) carries a control point — a
    /// `DeviceKind::Command` device through which commands can enter it. A fragment
    /// without one is uncontrolled (inert debris); the flight-level autonomy tier of
    /// one that has it is resolved by `crate::control::ControlSystem` (WI 562).
    pub fn has_control_point(&self) -> bool {
        self.devices.iter().any(|d| d.kind == DeviceKind::Command)
    }

    /// World-frame centre of cell `c` (the cell's geometric centre), metres.
    fn cell_center(&self, c: IVec3) -> DVec3 {
        (c.as_dvec3() + DVec3::splat(0.5)) * self.cell_size
    }

    /// Derived mass properties, or `None` for an empty craft (no mass).
    pub fn mass_properties(&self) -> Option<MassProperties> {
        // Accumulate mass and first moment for the centre of mass.
        let mut mass = 0.0;
        let mut moment = DVec3::ZERO;
        let cell_volume = self.cell_volume();
        for v in &self.voxels {
            let m = v.material.density * cell_volume;
            mass += m;
            moment += m * self.cell_center(v.cell);
        }
        for d in &self.devices {
            mass += d.mass;
            moment += d.mass * self.cell_center(d.cell);
        }
        for p in &self.parts {
            mass += p.mass;
            moment += p.mass * p.mount;
        }
        if mass <= 0.0 {
            return None;
        }
        let com = moment / mass;

        // Inertia about the centre of mass.
        let mut ixx = 0.0;
        let mut iyy = 0.0;
        let mut izz = 0.0;
        let mut ixy = 0.0;
        let mut ixz = 0.0;
        let mut iyz = 0.0;
        // Solid-cube self inertia (per diagonal): m·s²/6.
        let cube_self = self.cell_size * self.cell_size / 6.0;
        for v in &self.voxels {
            let m = v.material.density * cell_volume;
            let r = self.cell_center(v.cell) - com;
            ixx += m * (cube_self + r.y * r.y + r.z * r.z);
            iyy += m * (cube_self + r.x * r.x + r.z * r.z);
            izz += m * (cube_self + r.x * r.x + r.y * r.y);
            ixy -= m * r.x * r.y;
            ixz -= m * r.x * r.z;
            iyz -= m * r.y * r.z;
        }
        for d in &self.devices {
            let m = d.mass;
            let r = self.cell_center(d.cell) - com;
            ixx += m * (r.y * r.y + r.z * r.z);
            iyy += m * (r.x * r.x + r.z * r.z);
            izz += m * (r.x * r.x + r.y * r.y);
            ixy -= m * r.x * r.y;
            ixz -= m * r.x * r.z;
            iyz -= m * r.y * r.z;
        }
        // Attached parts as point masses at their continuous mount (parallel-axis).
        for p in &self.parts {
            let m = p.mass;
            let r = p.mount - com;
            ixx += m * (r.y * r.y + r.z * r.z);
            iyy += m * (r.x * r.x + r.z * r.z);
            izz += m * (r.x * r.x + r.y * r.y);
            ixy -= m * r.x * r.y;
            ixz -= m * r.x * r.z;
            iyz -= m * r.y * r.z;
        }
        let inertia = DMat3::from_cols(
            DVec3::new(ixx, ixy, ixz),
            DVec3::new(ixy, iyy, iyz),
            DVec3::new(ixz, iyz, izz),
        );
        let (principal_moments, principal_axes) = jacobi_symmetric(inertia);

        Some(MassProperties {
            mass,
            center_of_mass: com,
            inertia,
            principal_moments,
            principal_axes,
        })
    }

    /// The cross-sectional-area curve along `axis`: `(station, area_m2)` pairs,
    /// sorted by station. Area = occupied cells in the slice × cell². Derived from
    /// voxel occupancy only (devices excluded). Integrates (× cell_size) to the
    /// occupied voxel volume.
    pub fn area_curve(&self, axis: Axis) -> Vec<(i32, f64)> {
        let mut counts: BTreeMap<i32, usize> = BTreeMap::new();
        for v in &self.voxels {
            let station = match axis {
                Axis::X => v.cell.x,
                Axis::Y => v.cell.y,
                Axis::Z => v.cell.z,
            };
            *counts.entry(station).or_default() += 1;
        }
        let cell_area = self.cell_size * self.cell_size;
        counts
            .into_iter()
            .map(|(s, c)| (s, c as f64 * cell_area))
            .collect()
    }

    /// Inserts another craft's voxels and devices, offset by `offset` cells (used
    /// to place a reusable subassembly). Attachment points are not copied.
    pub fn insert(&mut self, other: &VoxelCraft, offset: IVec3) {
        for v in &other.voxels {
            self.voxels.push(Voxel {
                cell: v.cell + offset,
                material: v.material,
            });
        }
        for d in &other.devices {
            self.devices.push(Device {
                cell: d.cell + offset,
                ..*d
            });
        }
    }
}

/// Cyclic Jacobi eigensolve for a symmetric 3×3 matrix. Returns the eigenvalues
/// and the eigenvectors as the columns of the returned matrix.
// The rotation loops index two fixed columns/rows by a running index; the range
// loop is the clearest expression of that, so the range-loop lint is silenced.
#[allow(clippy::needless_range_loop)]
fn jacobi_symmetric(mat: DMat3) -> (DVec3, DMat3) {
    let c0 = mat.col(0);
    let c1 = mat.col(1);
    let c2 = mat.col(2);
    let mut a = [[c0.x, c1.x, c2.x], [c0.y, c1.y, c2.y], [c0.z, c1.z, c2.z]];
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for _ in 0..100 {
        // Largest off-diagonal magnitude.
        let mut p = 0;
        let mut q = 1;
        let mut max = a[0][1].abs();
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            if a[i][j].abs() >= max {
                max = a[i][j].abs();
                p = i;
                q = j;
            }
        }
        if max < 1e-20 {
            break;
        }
        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = if theta == 0.0 {
            1.0
        } else {
            theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
        };
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;

        // A <- Jᵀ A J (rotate columns p,q then rows p,q).
        for k in 0..3 {
            let akp = a[k][p];
            let akq = a[k][q];
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
        }
        for k in 0..3 {
            let apk = a[p][k];
            let aqk = a[q][k];
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
        }
        // Accumulate eigenvectors.
        for row in v.iter_mut() {
            let vp = row[p];
            let vq = row[q];
            row[p] = c * vp - s * vq;
            row[q] = s * vp + c * vq;
        }
    }

    let evals = DVec3::new(a[0][0], a[1][1], a[2][2]);
    let evecs = DMat3::from_cols(
        DVec3::new(v[0][0], v[1][0], v[2][0]),
        DVec3::new(v[0][1], v[1][1], v[2][1]),
        DVec3::new(v[0][2], v[1][2], v[2][2]),
    );
    (evals, evecs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a solid rectangular block of cells, all of `material`.
    fn block(nx: i32, ny: i32, nz: i32, cell_size: f64, material: Material) -> VoxelCraft {
        let mut craft = VoxelCraft::new(cell_size);
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    craft.voxels.push(Voxel {
                        cell: IVec3::new(x, y, z),
                        material,
                    });
                }
            }
        }
        craft
    }

    #[test]
    fn device_mass_scales_with_cell_volume_and_is_voxel_comparable() {
        // Mass scales with the cube of cell size (WI 615): doubling the cell ⇒ 8× mass.
        for kind in [
            DeviceKind::Command,
            DeviceKind::Computer,
            DeviceKind::Battery,
            DeviceKind::Engine,
            DeviceKind::Tank,
        ] {
            let m01 = device_mass(kind, 0.1);
            let m02 = device_mass(kind, 0.2);
            assert!(
                (m02 - 8.0 * m01).abs() < 1e-9,
                "device mass must scale with cell³"
            );
            // At 0.1 m a device is comparable to a FRAME-ish structural voxel (~1.6 kg), not 10–100×.
            let frame_voxel = 1_600.0 * 0.1 * 0.1 * 0.1; // ≈1.6 kg
            assert!(
                m01 > 0.2 * frame_voxel && m01 < 5.0 * frame_voxel,
                "device {kind:?} mass {m01} kg not voxel-comparable at 0.1 m"
            );
        }
    }

    #[test]
    fn single_voxel_mass_and_com() {
        let craft = block(
            1,
            1,
            1,
            2.0,
            Material {
                density: 1_000.0,
                strength: 1.0e9,
            },
        );
        let mp = craft.mass_properties().unwrap();
        // mass = density × cell³ = 1000 × 8 = 8000 kg.
        assert!((mp.mass - 8_000.0).abs() < 1e-6);
        // centre at the cell centre (1,1,1) for a 2 m cell at origin.
        assert!((mp.center_of_mass - DVec3::splat(1.0)).length() < 1e-9);
    }

    #[test]
    fn attached_part_shifts_com_and_inertia() {
        // A symmetric block: CoM at its geometric centre, finite inertia.
        let base = block(2, 2, 2, 1.0, Material::COMPOSITE);
        let mp0 = base.mass_properties().unwrap();

        // Mount a heavy part outboard (+x) and low (−y) of the block.
        let mut craft = base.clone();
        let mount = DVec3::new(4.0, -1.0, 1.0);
        craft.parts.push(Part {
            mount,
            mass: 500.0,
            kind: PartKind::Wheel(WheelPart::new(true, false)),
        });
        let mp = craft.mass_properties().unwrap();

        // Mass grows by exactly the part mass.
        assert!((mp.mass - (mp0.mass + 500.0)).abs() < 1e-9);
        // CoM shifts toward the part: +x and −y of the bare centre.
        assert!(mp.center_of_mass.x > mp0.center_of_mass.x);
        assert!(mp.center_of_mass.y < mp0.center_of_mass.y);
        // The outboard mass raises the moments about the axes orthogonal to its offset
        // (parallel-axis): an outboard +x/−y mass increases izz (about z).
        assert!(mp.inertia.col(2).z > mp0.inertia.col(2).z);
    }

    #[test]
    fn uniform_block_inertia_matches_analytic_box() {
        let (nx, ny, nz) = (4, 2, 6);
        let s = 0.5;
        let density = 1_200.0;
        let craft = block(
            nx,
            ny,
            nz,
            s,
            Material {
                density,
                strength: 1.0e9,
            },
        );
        let mp = craft.mass_properties().unwrap();

        let (lx, ly, lz) = (nx as f64 * s, ny as f64 * s, nz as f64 * s);
        let m = mp.mass;
        let exp_ixx = m / 12.0 * (ly * ly + lz * lz);
        let exp_iyy = m / 12.0 * (lx * lx + lz * lz);
        let exp_izz = m / 12.0 * (lx * lx + ly * ly);

        // A filled box is exactly the sum of its unit cubes, so this matches tightly.
        assert!((mp.inertia.col(0).x - exp_ixx).abs() < 1e-6 * exp_ixx);
        assert!((mp.inertia.col(1).y - exp_iyy).abs() < 1e-6 * exp_iyy);
        assert!((mp.inertia.col(2).z - exp_izz).abs() < 1e-6 * exp_izz);
        // Off-diagonals vanish for an axis-aligned box about its centre.
        assert!(mp.inertia.col(0).y.abs() < 1e-6 * exp_ixx);
        assert!(mp.inertia.col(0).z.abs() < 1e-6 * exp_ixx);
        // Symmetric.
        assert!((mp.inertia.col(0).y - mp.inertia.col(1).x).abs() < 1e-12);
    }

    #[test]
    fn principal_moments_recover_box_diagonal() {
        let craft = block(
            4,
            2,
            6,
            0.5,
            Material {
                density: 1_200.0,
                strength: 1.0e9,
            },
        );
        let mp = craft.mass_properties().unwrap();
        // For an axis-aligned box the principal moments equal the diagonal entries.
        let mut got = [
            mp.principal_moments.x,
            mp.principal_moments.y,
            mp.principal_moments.z,
        ];
        let mut diag = [
            mp.inertia.col(0).x,
            mp.inertia.col(1).y,
            mp.inertia.col(2).z,
        ];
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        diag.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (g, d) in got.iter().zip(diag.iter()) {
            assert!((g - d).abs() < 1e-6 * d);
        }
    }

    #[test]
    fn center_of_mass_offsets_toward_heavier_region() {
        // Two equal-size cells, one denser; CoM shifts toward the dense one.
        let mut craft = VoxelCraft::new(1.0);
        craft.voxels.push(Voxel {
            cell: IVec3::new(0, 0, 0),
            material: Material {
                density: 1_000.0,
                strength: 1.0e9,
            },
        });
        craft.voxels.push(Voxel {
            cell: IVec3::new(1, 0, 0),
            material: Material {
                density: 3_000.0,
                strength: 1.0e9,
            },
        });
        let mp = craft.mass_properties().unwrap();
        // Cell centres at x=0.5 and x=1.5; mass-weighted mean > 1.0 (midpoint).
        assert!(mp.center_of_mass.x > 1.0);
        assert!(mp.center_of_mass.x < 1.5);
    }

    #[test]
    fn area_curve_conserves_volume() {
        let craft = block(3, 4, 5, 0.5, Material::ALUMINIUM);
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let curve = craft.area_curve(axis);
            let integral: f64 = curve.iter().map(|(_, a)| a * craft.cell_size).sum();
            assert!((integral - craft.occupied_volume()).abs() < 1e-9);
        }
    }

    #[test]
    fn empty_craft_has_no_mass_properties() {
        let craft = VoxelCraft::new(1.0);
        assert!(craft.mass_properties().is_none());
        assert!(craft.area_curve(Axis::X).is_empty());
    }

    #[test]
    fn device_only_craft_has_mass() {
        let mut craft = VoxelCraft::new(1.0);
        craft
            .devices
            .push(Device::structural(IVec3::ZERO, 50.0, DeviceKind::Command));
        let mp = craft.mass_properties().unwrap();
        assert!((mp.mass - 50.0).abs() < 1e-9);
    }

    #[test]
    fn control_point_presence_tracks_command_device() {
        // A bare lattice has no control point; adding a Command device gives it one
        // (the lattice-level controllability breakage reads, WI 562).
        let mut craft = block(1, 1, 1, 1.0, Material::STEEL);
        assert!(!craft.has_control_point());
        craft
            .devices
            .push(Device::structural(IVec3::ZERO, 5.0, DeviceKind::Engine));
        assert!(
            !craft.has_control_point(),
            "an engine is not a control point"
        );
        craft
            .devices
            .push(Device::structural(IVec3::ZERO, 10.0, DeviceKind::Command));
        assert!(craft.has_control_point());
    }

    #[test]
    fn computer_or_battery_alone_is_not_a_control_point() {
        // A fragment carrying only a computer/battery (no control point) is
        // uncontrolled debris — `has_control_point` keys on `Command` only (WI 570).
        let mut craft = block(1, 1, 1, 1.0, Material::STEEL);
        craft.devices.push(Device::computer(
            IVec3::ZERO,
            10.0,
            crate::control::ControlComputer::command_core(0.5),
        ));
        craft.devices.push(Device::battery(
            IVec3::ZERO,
            20.0,
            crate::control::BatterySpec::full(100.0),
        ));
        assert!(
            !craft.has_control_point(),
            "computer + battery without a control point is not controllable"
        );
        // Adding a control point makes it controllable.
        craft
            .devices
            .push(Device::control_point(IVec3::ZERO, 50.0, true));
        assert!(craft.has_control_point());
    }

    #[test]
    fn inserting_a_subassembly_sums_mass() {
        let a = block(2, 2, 2, 1.0, Material::STEEL);
        let b = block(1, 1, 3, 1.0, Material::TITANIUM);
        let ma = a.mass_properties().unwrap().mass;
        let mb = b.mass_properties().unwrap().mass;
        let mut combined = a.clone();
        combined.insert(&b, IVec3::new(5, 0, 0));
        let mc = combined.mass_properties().unwrap().mass;
        assert!((mc - (ma + mb)).abs() < 1e-6);
    }

    #[test]
    fn material_is_data_driven() {
        // A new material is a value, not a code change.
        let exotic = Material {
            density: 19_300.0,
            strength: 1.0e9,
        }; // tungsten-like
        let mut craft = VoxelCraft::new(1.0);
        craft.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: exotic,
        });
        let mp = craft.mass_properties().unwrap();
        assert!((mp.mass - 19_300.0).abs() < 1e-6);
    }
}
