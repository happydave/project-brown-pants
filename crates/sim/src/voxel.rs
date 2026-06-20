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

/// What a device is (a type tag only — devices are inert mass in Toy 5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    Command,
    Engine,
    Tank,
    Rcs,
}

/// A mounted functional device: a mass at a cell. Contributes to mass and inertia
/// only; it does not add to the voxel-occupancy area curve (Toy 5).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Device {
    /// Mounting cell.
    pub cell: IVec3,
    /// Device mass, kg.
    pub mass: f64,
    /// Device type.
    pub kind: DeviceKind,
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
}

impl Default for VoxelCraft {
    fn default() -> Self {
        Self {
            cell_size: 1.0,
            voxels: Vec::new(),
            devices: Vec::new(),
            attachments: Vec::new(),
            doors: Vec::new(),
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
        craft.devices.push(Device {
            cell: IVec3::ZERO,
            mass: 50.0,
            kind: DeviceKind::Command,
        });
        let mp = craft.mass_properties().unwrap();
        assert!((mp.mass - 50.0).abs() < 1e-9);
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
