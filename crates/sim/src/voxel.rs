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
use std::collections::{BTreeMap, HashSet};

/// A structural material: the data the discipline says to model — density,
/// tensile strength, and thermal properties. A new material is a new value, not a
/// new code path.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Material {
    /// Density, kg/m³.
    pub density: f64,
    /// Tensile strength, Pa — the structural stress a bond of this material
    /// withstands before breaking (consumed by connected-component breakage,
    /// WI 518). Defaulted on load so pre-strength saves stay backward-loadable.
    #[serde(default = "Material::default_strength")]
    pub strength: f64,
    /// Thermal properties (WI 687): heat capacity, conductivity, emissivity, and
    /// failure temperature, consumed by the two-node thermal model
    /// ([`crate::thermal`]). Defaulted to [`Thermal::INERT`] on load so
    /// pre-thermal saves stay backward-loadable (and never spontaneously melt).
    #[serde(default)]
    pub thermal: Thermal,
}

impl Material {
    /// Aluminium-like structural material.
    pub const ALUMINIUM: Material = Material {
        density: 2_700.0,
        strength: 3.1e8,
        thermal: Thermal::ALUMINIUM,
    };
    /// Steel-like structural material.
    pub const STEEL: Material = Material {
        density: 7_850.0,
        strength: 5.0e8,
        thermal: Thermal::STEEL,
    };
    /// Titanium-like structural material.
    pub const TITANIUM: Material = Material {
        density: 4_500.0,
        strength: 9.0e8,
        thermal: Thermal::TITANIUM,
    };
    /// Light composite — also the slice's reference heat-resistant material
    /// (low conductivity, high emissivity, high failure temperature).
    pub const COMPOSITE: Material = Material {
        density: 1_600.0,
        strength: 6.0e8,
        thermal: Thermal::COMPOSITE,
    };
    /// Carbon-phenolic **ablative heat shield** (WI 688): light, and it protects by
    /// vaporising ablator rather than by a high bare failure temperature.
    pub const ABLATOR: Material = Material {
        density: 1_400.0,
        strength: 5.0e7,
        thermal: Thermal::ABLATOR,
    };
    /// Glass (WI 821): the first **transparent** material — windows, canopies,
    /// portholes. Transparency is a render-side property (the sim treats glass as
    /// ordinary data, per this module's discipline); physically it is dense-ish and
    /// **brittle** — the weakest structural material, so glass hulls break honestly.
    pub const GLASS: Material = Material {
        density: 2_500.0,
        strength: 4.0e7,
        thermal: Thermal::GLASS,
    };

    /// The strength assumed for a material loaded from a pre-strength save: high
    /// enough to be effectively unbreakable, so old craft do not spontaneously
    /// shatter.
    pub fn default_strength() -> f64 {
        1.0e12
    }
}

/// The thermal properties of a [`Material`] (WI 687) — the data the two-node
/// thermal model integrates. Per the governing discipline these are fields, not
/// code paths: a heat-shield material is a value with a high failure temperature
/// and low conductivity, not a special case.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Thermal {
    /// Specific heat capacity, J·kg⁻¹·K⁻¹ — energy to raise 1 kg by 1 K.
    pub specific_heat: f64,
    /// Thermal conductivity, W·m⁻¹·K⁻¹ — skin↔core and voxel↔voxel heat transfer.
    pub conductivity: f64,
    /// Emissivity, 0–1 — radiative efficiency toward the environment.
    pub emissivity: f64,
    /// Maximum temperature, K — the skin temperature at which the voxel fails
    /// (consumed by the thermal→breakage failure path).
    pub max_temp: f64,
    /// Ablation set-point, K (WI 688): while the skin is above this and ablator
    /// remains, the surface vaporises ablator to hold near this temperature. `0`
    /// (the default) = non-ablative. Defaulted on load so pre-ablation saves stay
    /// backward-loadable.
    #[serde(default)]
    pub ablation_temp: f64,
    /// Latent heat of ablation, J·kg⁻¹ (WI 688) — energy carried away per kg of
    /// ablator vaporised. `0` = non-ablative.
    #[serde(default)]
    pub latent_heat: f64,
    /// Ablator fraction, 0–1 (WI 688) — the share of a voxel's mass that is
    /// consumable ablator (the rest is the bare structural material that remains
    /// after burn-through). `0` = non-ablative.
    #[serde(default)]
    pub ablator_fraction: f64,
}

impl Default for Thermal {
    fn default() -> Self {
        Self::INERT
    }
}

impl Thermal {
    /// The thermal properties assumed for a pre-thermal save: a high failure
    /// temperature (so legacy/structural craft never melt), modest capacity, and
    /// low emissivity. Backward-load default and the value structural-only
    /// fixtures use.
    pub const INERT: Thermal = Thermal {
        specific_heat: 900.0,
        conductivity: 200.0,
        emissivity: 0.1,
        max_temp: 1.0e9,
        ablation_temp: 0.0,
        latent_heat: 0.0,
        ablator_fraction: 0.0,
    };
    /// Aluminium-like: high conductivity, low emissivity, ~melting at 900 K.
    pub const ALUMINIUM: Thermal = Thermal {
        specific_heat: 900.0,
        conductivity: 237.0,
        emissivity: 0.15,
        max_temp: 900.0,
        ablation_temp: 0.0,
        latent_heat: 0.0,
        ablator_fraction: 0.0,
    };
    /// Steel-like: moderate conductivity, higher failure temperature.
    pub const STEEL: Thermal = Thermal {
        specific_heat: 490.0,
        conductivity: 50.0,
        emissivity: 0.30,
        max_temp: 1_700.0,
        ablation_temp: 0.0,
        latent_heat: 0.0,
        ablator_fraction: 0.0,
    };
    /// Titanium-like: low conductivity, high failure temperature.
    pub const TITANIUM: Thermal = Thermal {
        specific_heat: 520.0,
        conductivity: 22.0,
        emissivity: 0.30,
        max_temp: 1_940.0,
        ablation_temp: 0.0,
        latent_heat: 0.0,
        ablator_fraction: 0.0,
    };
    /// Carbon-composite-like: a poor conductor, strong radiator, very high
    /// failure temperature — the reference heat-shield material.
    pub const COMPOSITE: Thermal = Thermal {
        specific_heat: 1_000.0,
        conductivity: 5.0,
        emissivity: 0.80,
        max_temp: 3_000.0,
        ablation_temp: 0.0,
        latent_heat: 0.0,
        ablator_fraction: 0.0,
    };
    /// Carbon-phenolic-like **ablative heat shield** (WI 688): a poor conductor and
    /// strong radiator that, above its ablation set-point, vaporises ablator to carry
    /// heat away — protecting the craft until the ablator is spent, then reverting to
    /// a bare char that fails at `max_temp`.
    pub const ABLATOR: Thermal = Thermal {
        specific_heat: 1_500.0,
        conductivity: 0.5,
        emissivity: 0.85,
        max_temp: 3_500.0,
        ablation_temp: 1_300.0,
        latent_heat: 5.0e6,
        ablator_fraction: 0.6,
    };
    /// Glass-like (WI 821): a poor conductor and strong radiator that softens (fails)
    /// near 1000 K — fine for a cabin window, hopeless as a re-entry windshield.
    pub const GLASS: Thermal = Thermal {
        specific_heat: 840.0,
        conductivity: 1.0,
        emissivity: 0.90,
        max_temp: 1_000.0,
        ablation_temp: 0.0,
        latent_heat: 0.0,
        ablator_fraction: 0.0,
    };
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
    /// Selected motor tier (WI 652) — only meaningful on an `Engine` device; sizes the rover
    /// drivetrain (torque / top-speed / draw). Defaulted absent so the mass-derived default applies
    /// and pre-652 saves load.
    #[serde(default)]
    pub motor: Option<crate::powertrain::MotorTier>,
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
            motor: None,
        }
    }

    /// A rover drive motor (kind `Engine` + a selected [`MotorTier`], WI 652): its mass is the
    /// motor's, and the assembly sizes the drivetrain torque/top-speed/draw from the tier.
    pub fn engine(cell: IVec3, motor: crate::powertrain::MotorTier) -> Self {
        Self {
            cell,
            mass: motor.spec().mass,
            kind: DeviceKind::Engine,
            function: None,
            motor: Some(motor),
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
            motor: None,
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
            motor: None,
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
            motor: None,
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

/// An axis along which the cross-sectional-area curve is sliced (and the normal
/// axis of a [`FacePanel`]'s boundary, WI 824).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    /// The positive unit cell-step along this axis.
    pub fn unit(self) -> IVec3 {
        match self {
            Axis::X => IVec3::X,
            Axis::Y => IVec3::Y,
            Axis::Z => IVec3::Z,
        }
    }
}

/// A **face panel** (WI 824, the panels design): a thin structural plate on the
/// boundary between lattice cell `cell` and `cell + axis.unit()` — the canonical
/// key is always the negative-side cell, so one boundary is stored exactly once.
/// Panels mass and displace their plate ([`PANEL_FILL`] thickness), seal their
/// boundary (the coverage predicate), and carry their own material — a glass face
/// panel is a window.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FacePanel {
    /// The cell on the boundary's negative side (the canonical owner).
    pub cell: IVec3,
    /// The boundary's normal axis: the panel sits between `cell` and
    /// `cell + axis.unit()`.
    pub axis: Axis,
    /// The plate's material.
    pub material: Material,
}

impl FacePanel {
    /// Deterministic sort key (WI 820 discipline: ordered encode).
    fn key(&self) -> (i32, i32, i32, u8) {
        (self.cell.x, self.cell.y, self.cell.z, self.axis as u8)
    }
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
    /// Wheel-station id (WI 630): groups a [`PartKind::Suspension`] + [`PartKind::Rim`] +
    /// [`PartKind::Tire`] into one wheel. `None` for parts that are not part of a station
    /// (seat, antenna, solar, bumper, and legacy monolithic [`PartKind::Wheel`]). A station is
    /// **complete** when all three component kinds share an id; only complete stations become rover
    /// wheels. Defaulted on load so pre-component saves stay backward-loadable.
    #[serde(default)]
    pub station: Option<u32>,
}

/// The kind of a catalog [`Part`] (WI 607). `Wheel` carries the physical and
/// group parameters that assembly turns into a rover wheel; the remaining kinds are
/// inert mass with a recognisable role (their meshes land in WI 608).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartKind {
    /// A **legacy** monolithic wheel + suspension + tire (WI 607). Still authored by the editor and
    /// loadable from old saves; assembly migrates it to a [`Suspension`]/[`Rim`]/[`Tire`] station via
    /// [`WheelPart::to_components`] (WI 630). New authoring places the three components instead.
    Wheel(WheelPart),
    /// The spring-damper strut of a wheel station (WI 630): anchors the station and owns ride height
    /// and travel. Optional — a station with a rigid/zero-travel strut still rides on tire compliance.
    Suspension(SuspensionSpec),
    /// The rim/hub of a wheel station (WI 630): owns the rim radius and the drivetrain (drive/steer)
    /// membership of the wheel.
    Rim(RimSpec),
    /// The tire of a wheel station (WI 630): owns grip (rubber compound), slip stiffness, and the
    /// profile that — with the rim radius — sets the effective rolling radius.
    Tire(TireSpec),
    /// A crew seat (recognisability; crew/control comes from a control point device).
    Seat,
    /// A communications antenna (cosmetic now; a comms model is later).
    Antenna,
    /// A solar panel (electric charging is wired in the powertrain WI 609).
    SolarPanel,
    /// A bumper (structure that reads as a rover and breaks on impact, WI 610).
    Bumper,
}

/// Spring-damper strut parameters of a [`PartKind::Suspension`] (WI 630). Owns ride height and
/// travel; the spring stiffness / damping / max force are sized to the assembled rover's mass by
/// [`crate::rover::assemble_rover`] (so a heavy build does not sag), not authored here.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SuspensionSpec {
    /// Suspension free length (m) — ride height.
    pub rest_length: f64,
    /// Suspension travel (m) — usable compression before bottoming.
    pub travel: f64,
    /// A rigid strut (WI 630): no spring travel of its own, so the wheel rides on the tire's
    /// compliance alone — the user's "remove suspension travel, keep some from the rubber/air".
    /// Defaulted `false` (a normal sprung strut) so pre-630 saves load as before.
    #[serde(default)]
    pub rigid: bool,
}

impl SuspensionSpec {
    /// A strut with the rover core's sensible defaults (mirrors [`crate::rover::Wheel::new`]).
    pub fn new() -> Self {
        Self {
            rest_length: 0.35,
            travel: 0.35,
            rigid: false,
        }
    }

    /// A rigid strut: no suspension travel; the wheel rides on tire compliance only (WI 630).
    pub fn rigid(rest_length: f64) -> Self {
        Self {
            rest_length,
            travel: 0.0,
            rigid: true,
        }
    }

    /// A strut sized to the build's `cell_size` (WI 630). Ride height and travel scale with the cell so
    /// a small build gets a short strut, not a fixed 0.35 m one that reaches far below the hub — which
    /// kept a tipping rover's raised wheels glued to the ground instead of letting it tumble.
    pub fn for_cell_size(cell_size: f64) -> Self {
        Self {
            rest_length: 0.35 * cell_size,
            travel: 0.25 * cell_size,
            rigid: false,
        }
    }
}

impl Default for SuspensionSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// Rim/hub parameters of a [`PartKind::Rim`] (WI 630). Owns the rim radius (the effective rolling
/// radius is `rim radius + tire profile`) and the wheel's drivetrain group membership.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RimSpec {
    /// Rim radius (m); the tire's profile is added to get the effective rolling radius.
    pub radius: f64,
    /// In the drive group (receives engine/motor torque).
    pub drive: bool,
    /// In the steer group (turns with steering input).
    pub steer: bool,
}

impl RimSpec {
    /// A rim with the rover core's default radius; `drive`/`steer` choose the drivetrain groups.
    pub fn new(drive: bool, steer: bool) -> Self {
        Self {
            radius: 0.25,
            drive,
            steer,
        }
    }
}

/// Tire parameters of a [`PartKind::Tire`] (WI 630). Owns grip (rubber compound, scaling surface
/// friction), slip stiffness (how sharply force builds with slip), and the profile (section height
/// added to the rim radius). Defaults reproduce the pre-split behaviour (grip 1.0, slip = the rover
/// core's slip constants), so a migrated wheel drives identically until the tire is changed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TireSpec {
    /// Section height (m) added to the rim radius for the effective rolling radius.
    pub profile: f64,
    /// Grip multiplier over the surface material's friction (rubber compound). 1.0 = baseline.
    pub grip_scale: f64,
    /// Longitudinal slip stiffness (shape of the slip-ratio → force curve).
    pub slip_long: f64,
    /// Lateral slip stiffness (shape of the slip-angle → force curve).
    pub slip_lat: f64,
    /// Tire compliance — the rubber/air spring rate (N/m), in **series** with the suspension spring
    /// (WI 630). A high value (the default) is effectively rigid, so a migrated wheel rides as before;
    /// a lower value softens the ride and lets a no-suspension build ride on the tire. Defaulted high
    /// on load so pre-630 saves are unchanged.
    #[serde(default = "TireSpec::default_stiffness")]
    pub stiffness: f64,
}

impl TireSpec {
    /// A tire whose parameters reproduce the pre-split rover wheel (WI 630). The slip defaults match
    /// the rover core's `C_LONG` / `C_LAT`, and the stiffness is effectively rigid so the series
    /// spring equals the suspension spring (no behaviour change); `profile` is the section height.
    pub fn new(profile: f64) -> Self {
        Self {
            profile,
            grip_scale: 1.0,
            slip_long: 5.0,
            slip_lat: 4.0,
            stiffness: Self::default_stiffness(),
        }
    }

    /// The default (effectively rigid) tire spring rate, N/m: high enough that the series combination
    /// with any sized suspension spring is the suspension spring to within f64 tolerance (WI 630).
    pub fn default_stiffness() -> f64 {
        1.0e9
    }
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

    /// Migrate a legacy monolithic wheel to the three station components (WI 630). The split is
    /// behaviour-preserving: the effective rolling radius (rim + tire profile) equals the old radius
    /// exactly, the tire reproduces the pre-split grip/slip, and `rest_length` / `drive` / `steer`
    /// carry over. The component masses are the caller's concern (the legacy part's single mass is
    /// used directly for inertia by [`crate::rover::assemble_rover`]).
    pub fn to_components(&self) -> (SuspensionSpec, RimSpec, TireSpec) {
        // 30 % of the radius is tire profile; the rest is rim. Computing rim as `radius - profile`
        // keeps `rim + profile == radius` exact in IEEE arithmetic.
        let profile = 0.3 * self.radius;
        (
            SuspensionSpec {
                rest_length: self.rest_length,
                travel: self.rest_length,
                rigid: false,
            },
            RimSpec {
                radius: self.radius - profile,
                drive: self.drive,
                steer: self.steer,
            },
            TireSpec::new(profile),
        )
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
    /// **Face panels** (WI 824): thin plates on cell boundaries — the first-class
    /// panel model (cells are solid cubes; plates live on faces). Kept sorted by
    /// [`FacePanel::key`] for deterministic encode; defaulted on load so pre-824
    /// saves stay backward-loadable.
    #[serde(default)]
    pub face_panels: Vec<FacePanel>,
    /// **Shaped cells** (WI 831, the shape catalog): a sorted sidecar keyed by
    /// cell — an occupied cell with an entry takes its [`crate::shape::Form`] +
    /// orientation (+ fill mode, WI 836); one without is a full solid cube.
    /// **Skipped when empty** so pre-shape saves re-serialize byte-identically;
    /// an entry whose cell is unoccupied is inert (the folds are voxel-driven).
    /// Kept sorted by cell for deterministic encode (the WI 820 discipline).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shapes: Vec<crate::shape::ShapedCell>,
}

/// A **plate's thickness** as a fraction of the cell size (WI 716 → WI 824): a
/// [`FacePanel`] masses and displaces `face_area × PANEL_FILL × cell_size` of its
/// material — the lever that lets a real hull float without a magic-light material.
/// `0.05` (a ~5 mm plate on a 0.1 m build cell). Halved from the legacy 0.1 at the
/// WI 824 conversion: a legacy hull wall converts to an inner *and* outer skin
/// (the R1 topology rule), so half-thickness plates keep a converted wall's total
/// plate mass at the legacy value while the sealed inter-skin void displaces
/// honestly (the double-hull accounting audited in the WI 824 plan).
pub const PANEL_FILL: f64 = 0.05;

impl Default for VoxelCraft {
    fn default() -> Self {
        Self {
            cell_size: 1.0,
            voxels: Vec::new(),
            devices: Vec::new(),
            attachments: Vec::new(),
            doors: Vec::new(),
            parts: Vec::new(),
            face_panels: Vec::new(),
            shapes: Vec::new(),
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

    /// Canonicalize the boundary between `cell` and `cell + dir` (`dir` a ±unit
    /// step) to its face key: the negative-side cell and the normal axis.
    pub fn canonical_face(cell: IVec3, dir: IVec3) -> (IVec3, Axis) {
        let axis = if dir.x != 0 {
            Axis::X
        } else if dir.y != 0 {
            Axis::Y
        } else {
            Axis::Z
        };
        let owner = if dir.x + dir.y + dir.z > 0 {
            cell
        } else {
            cell + dir
        };
        (owner, axis)
    }

    /// The face panel on the boundary owned by `(cell, axis)`, if any (WI 824).
    pub fn face_panel_at(&self, cell: IVec3, axis: Axis) -> Option<&FacePanel> {
        let key = (cell.x, cell.y, cell.z, axis as u8);
        self.face_panels
            .binary_search_by(|p| p.key().cmp(&key))
            .ok()
            .map(|i| &self.face_panels[i])
    }

    /// The face panel on the boundary between `cell` and `cell + dir` (either
    /// side's view; `dir` a ±unit step), if any (WI 824).
    pub fn face_panel_between(&self, cell: IVec3, dir: IVec3) -> Option<&FacePanel> {
        let (owner, axis) = Self::canonical_face(cell, dir);
        self.face_panel_at(owner, axis)
    }

    /// Place (`Some(material)`) or remove (`None`) the face panel on the boundary
    /// between `cell` and `cell + dir` (WI 824). Keeps the store sorted (the
    /// deterministic-encode invariant); placing over an existing panel replaces
    /// its material.
    pub fn set_face_panel(&mut self, cell: IVec3, dir: IVec3, material: Option<Material>) {
        let (owner, axis) = Self::canonical_face(cell, dir);
        let key = (owner.x, owner.y, owner.z, axis as u8);
        match self.face_panels.binary_search_by(|p| p.key().cmp(&key)) {
            Ok(i) => match material {
                Some(m) => self.face_panels[i].material = m,
                None => {
                    self.face_panels.remove(i);
                }
            },
            Err(i) => {
                if let Some(m) = material {
                    self.face_panels.insert(
                        i,
                        FacePanel {
                            cell: owner,
                            axis,
                            material: m,
                        },
                    );
                }
            }
        }
    }

    /// Restore the face-panel sort invariant (defensive, for decoded documents
    /// whose producer did not order the store).
    pub fn normalize_face_panels(&mut self) {
        self.face_panels.sort_by_key(|p| p.key());
        self.face_panels.dedup_by_key(|p| p.key());
    }

    /// The shape entry for `cell`, if any (WI 831). An occupied cell without one
    /// is a full solid cube.
    pub fn shape_at(&self, cell: IVec3) -> Option<&crate::shape::ShapedCell> {
        let key = (cell.x, cell.y, cell.z);
        self.shapes
            .binary_search_by(|s| s.key().cmp(&key))
            .ok()
            .map(|i| &self.shapes[i])
    }

    /// Set (or replace) the shape of a cell, keeping the store sorted (the
    /// deterministic-encode invariant). A `Cube` entry at identity is stored as
    /// given — the caller decides whether to canonicalize.
    pub fn set_shape(&mut self, shape: crate::shape::ShapedCell) {
        match self.shapes.binary_search_by(|s| s.key().cmp(&shape.key())) {
            Ok(i) => self.shapes[i] = shape,
            Err(i) => self.shapes.insert(i, shape),
        }
    }

    /// Remove the shape entry of `cell` (back to a full solid cube), if any.
    pub fn clear_shape(&mut self, cell: IVec3) {
        let key = (cell.x, cell.y, cell.z);
        if let Ok(i) = self.shapes.binary_search_by(|s| s.key().cmp(&key)) {
            self.shapes.remove(i);
        }
    }

    /// Restore the shape-store sort invariant (defensive, for decoded documents
    /// whose producer did not order the store).
    pub fn normalize_shapes(&mut self) {
        self.shapes.sort_by_key(|s| s.key());
        self.shapes.dedup_by_key(|s| s.key());
    }

    /// A voxel's material volume, m³ (WI 831): the cell volume × its form's
    /// volume fraction (1 for an unshaped cell). A **shell** (WI 836) is its
    /// boundary skin at plate thickness — `shell_area × PANEL_FILL` of the cell
    /// (the multiply by [`PANEL_FILL`] happens here, keeping the form constants
    /// pure geometry). Mass *and* displacement both fold through this volume,
    /// so a shell masses and displaces only its skin (the design's
    /// plate-for-mass collapse); every admitted form's skin volume is well
    /// below its solid volume at `PANEL_FILL` 0.05, so no clamp is needed.
    pub fn voxel_volume(&self, v: &Voxel) -> f64 {
        match self.shape_at(v.cell) {
            Some(s) => {
                let c = crate::shape::constants(s.form);
                let fraction = match s.fill {
                    crate::shape::FillMode::Solid => c.volume,
                    crate::shape::FillMode::Shell => c.shell_area * PANEL_FILL,
                };
                self.cell_volume() * fraction
            }
            None => self.cell_volume(),
        }
    }

    /// A voxel's mass, kg (WI 831): `density × voxel_volume`.
    pub fn voxel_mass(&self, v: &Voxel) -> f64 {
        v.material.density * self.voxel_volume(v)
    }

    /// A voxel's mass centroid in the craft's local frame, metres: the cell
    /// centre for an unshaped cell; the rotated form centroid for a shaped one;
    /// the rotated **skin** centroid for a shell (WI 836).
    pub fn voxel_centroid(&self, v: &Voxel) -> DVec3 {
        match self.shape_at(v.cell) {
            Some(s) => {
                let fc = crate::shape::constants(s.form);
                let c = match s.fill {
                    crate::shape::FillMode::Solid => fc.centroid_oriented(s.orientation),
                    crate::shape::FillMode::Shell => fc.shell_centroid_oriented(s.orientation),
                };
                (v.cell.as_dvec3() + c) * self.cell_size
            }
            None => self.cell_center(v.cell),
        }
    }

    /// A voxel's inertia tensor about its own centroid, kg·m² (WI 831): the
    /// solid-cube diagonal for an unshaped cell; `ρ·s⁵·R·I_unit·Rᵀ` for a shaped
    /// one; **zero** for a shell (WI 836) — a point mass at its skin centroid,
    /// exactly the face-panel pattern (slab self-inertia stays an open
    /// refinement there and here alike).
    pub fn voxel_self_inertia(&self, v: &Voxel) -> DMat3 {
        match self.shape_at(v.cell) {
            Some(s) if s.fill == crate::shape::FillMode::Shell => DMat3::ZERO,
            Some(s) => {
                let unit = crate::shape::constants(s.form).unit_inertia_oriented(s.orientation);
                let scale = v.material.density * self.cell_size.powi(5);
                unit * scale
            }
            None => {
                // m·s²/6 per diagonal — the exact pre-831 cube self-inertia.
                let m = v.material.density * self.cell_volume();
                DMat3::from_diagonal(DVec3::splat(m * self.cell_size * self.cell_size / 6.0))
            }
        }
    }

    /// **The per-face coverage predicate** (WI 824; partial coverage since
    /// WI 832): the boundary between `cell` and `cell + dir` is sealed iff
    /// fully covered by the two sides' occupancy and/or a face panel. A side
    /// contributes **no** coverage when not in `solid`, **full** coverage when
    /// an unshaped `solid` member (voxels and closed doors alike), and its
    /// form's **oriented face mask** when shaped — so mated complementary
    /// wedges seal their shared boundary while a lone wedge's half-open face
    /// does not. `solid` is the caller's occupancy set (the compartment
    /// flood-fill passes voxels + closed doors; the aero fill passes voxels +
    /// all doors). Callers must not inline their own seal logic — this is the
    /// one function shaped coverage flows through.
    pub fn boundary_sealed(&self, solid: &HashSet<IVec3>, cell: IVec3, dir: IVec3) -> bool {
        if self.face_panel_between(cell, dir).is_some() {
            return true;
        }
        if self.shapes.is_empty() {
            // Fast path: no shaped cells — the exact pre-832 rule.
            return solid.contains(&cell) || solid.contains(&(cell + dir));
        }
        let axis = if dir.x != 0 {
            0
        } else if dir.y != 0 {
            1
        } else {
            2
        };
        // Own face toward the neighbour, neighbour's face back toward us
        // (face order [x0, x1, y0, y1, z0, z1]).
        let positive = dir.x + dir.y + dir.z > 0;
        let (face_a, face_b) = if positive {
            (2 * axis + 1, 2 * axis)
        } else {
            (2 * axis, 2 * axis + 1)
        };
        let coverage = |c: IVec3, face: usize| -> crate::shape::FaceMask {
            if !solid.contains(&c) {
                crate::shape::MASK_EMPTY
            } else {
                match self.shape_at(c) {
                    Some(s) => crate::shape::face_masks(s.form, s.orientation)[face],
                    None => crate::shape::MASK_FULL,
                }
            }
        };
        let a = coverage(cell, face_a);
        if a == crate::shape::MASK_FULL {
            return true;
        }
        let b = coverage(cell + dir, face_b);
        crate::shape::masks_seal(&a, &b)
    }

    /// Plate a shell **directly** (WI 820): for every cell of `cells` (the
    /// intended shell occupancy), add a face panel of `material` on each face
    /// adjacent to a cell *outside* that occupancy — the WI 824 R1 double-skin
    /// topology (a hull wall gains an inner and an outer skin around its void),
    /// as a builder rather than a migration (the legacy flag converter this
    /// replaces was retired with format v1). The shell cells themselves hold no
    /// voxels; callers building mixed hulls add solid cells separately.
    /// Deterministic regardless of `cells` order (the panel store is sorted).
    pub fn plate_shell(&mut self, cells: &[IVec3], material: Material) {
        let occupied: HashSet<IVec3> = cells.iter().copied().collect();
        let dirs = [
            IVec3::X,
            IVec3::NEG_X,
            IVec3::Y,
            IVec3::NEG_Y,
            IVec3::Z,
            IVec3::NEG_Z,
        ];
        for &cell in cells {
            for d in dirs {
                if !occupied.contains(&(cell + d)) {
                    self.set_face_panel(cell, d, Some(material));
                }
            }
        }
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

    /// World-frame centre of a face panel's plate, metres (WI 824): the centre of
    /// the boundary between `p.cell` and its positive-`axis` neighbour.
    pub fn face_center(&self, p: &FacePanel) -> DVec3 {
        self.cell_center(p.cell) + p.axis.unit().as_dvec3() * (0.5 * self.cell_size)
    }

    /// A face panel's plate volume, m³ (WI 824): face area × plate thickness.
    pub fn panel_volume(&self) -> f64 {
        self.cell_size * self.cell_size * (PANEL_FILL * self.cell_size)
    }

    /// Derived mass properties, or `None` for an empty craft (no mass).
    /// Cells are solid cubes — or their catalog form when shaped (WI 831:
    /// `voxel_mass`/`voxel_centroid`/`voxel_self_inertia` fold the form's
    /// derived volume, centroid, and inertia; a **shell** cell masses its
    /// plate-thickness skin as a point mass at the skin centroid — WI 836);
    /// **face panels** fold as their plate mass at their face centre
    /// (point-mass inertia, the device/part pattern) — WI 824.
    pub fn mass_properties(&self) -> Option<MassProperties> {
        // Accumulate mass and first moment for the centre of mass.
        let mut mass = 0.0;
        let mut moment = DVec3::ZERO;
        for v in &self.voxels {
            let m = self.voxel_mass(v);
            mass += m;
            moment += m * self.voxel_centroid(v);
        }
        for p in &self.face_panels {
            let m = p.material.density * self.panel_volume();
            mass += m;
            moment += m * self.face_center(p);
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
        // Per-voxel self inertia about its own centroid (solid-cube diagonal, or
        // the rotated form tensor — WI 831) plus the parallel-axis point terms.
        for v in &self.voxels {
            let m = self.voxel_mass(v);
            let r = self.voxel_centroid(v) - com;
            let si = self.voxel_self_inertia(v);
            ixx += si.col(0).x + m * (r.y * r.y + r.z * r.z);
            iyy += si.col(1).y + m * (r.x * r.x + r.z * r.z);
            izz += si.col(2).z + m * (r.x * r.x + r.y * r.y);
            ixy += si.col(1).x - m * r.x * r.y;
            ixz += si.col(2).x - m * r.x * r.z;
            iyz += si.col(2).y - m * r.y * r.z;
        }
        // Face panels as point masses at their face centres (WI 824; the
        // device/part pattern — slab self-inertia is a refinement left open).
        for p in &self.face_panels {
            let m = p.material.density * self.panel_volume();
            let r = self.face_center(p) - com;
            ixx += m * (r.y * r.y + r.z * r.z);
            iyy += m * (r.x * r.x + r.z * r.z);
            izz += m * (r.x * r.x + r.y * r.y);
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
    /// sorted by station. **The sealed envelope, fractionally** (WI 827 → 834,
    /// design R2): area = Σ presented fraction × cell² over envelope cells in
    /// the slice (devices excluded), from
    /// [`crate::compartments::envelope_fractions`] — everything the exterior
    /// flood-fill cannot reach presents fully (structure *and* enclosed air,
    /// including a shaped cell's sealed remainder), while a shaped cell whose
    /// remainder the exterior reached presents its **solid volume fraction**
    /// (the mean-slice identity: average cross-section = volume / length —
    /// orientation-independent by design; sub-station profiles stay a
    /// door-open refinement). A wedge-faired shoulder thus spreads one area
    /// step into two smaller ones, which the area-ruling factor rewards. A
    /// closed plated hull presents its full body to the flow; a breached hull
    /// presents only its structure (the cavity vents). A **free-standing
    /// plate** (neither side presenting) presents itself: its full face area
    /// sliced along its normal (attributed to its owning cell's station), its
    /// [`PANEL_FILL`] edge sliced tangentially; a plate on the envelope's skin
    /// adds nothing (the cell behind it already presents). The fold runs in
    /// sorted cell order — mixed fractional sums must be order-deterministic.
    pub fn area_curve(&self, axis: Axis) -> Vec<(i32, f64)> {
        let envelope = crate::compartments::envelope_fractions(self);
        let cell_area = self.cell_size * self.cell_size;
        let station = |c: IVec3| match axis {
            Axis::X => c.x,
            Axis::Y => c.y,
            Axis::Z => c.z,
        };
        let mut members: Vec<(IVec3, f64)> = envelope.iter().map(|(&c, &f)| (c, f)).collect();
        members.sort_unstable_by_key(|(c, _)| (c.x, c.y, c.z));
        let mut areas: BTreeMap<i32, f64> = BTreeMap::new();
        for (cell, fraction) in members {
            *areas.entry(station(cell)).or_default() += fraction * cell_area;
        }
        let presents = |c: IVec3| envelope.get(&c).is_some_and(|f| *f > 0.0);
        for p in &self.face_panels {
            let free_standing = !presents(p.cell) && !presents(p.cell + p.axis.unit());
            if !free_standing {
                continue;
            }
            let projected = if axis == p.axis {
                cell_area // face-on: the full plate face
            } else {
                PANEL_FILL * cell_area // edge-on: the plate's thin edge
            };
            *areas.entry(station(p.cell)).or_default() += projected;
        }
        areas.into_iter().collect()
    }

    /// Inserts another craft's voxels, devices, and cell shapes, offset by
    /// `offset` cells (used to place a reusable subassembly). Attachment points
    /// are not copied.
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
        for s in &other.shapes {
            self.set_shape(crate::shape::ShapedCell {
                cell: s.cell + offset,
                ..*s
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

    #[test]
    fn cell_scaled_suspension_is_short_and_sprung() {
        // The strut's ride height and travel scale with the build (WI 630): a small build gets a short
        // strut, not the fixed 0.35 m one that reached far below the hub and glued a tipping rover's
        // raised wheels to the ground. A 0.1 m build's strut is far shorter than that old fixed value.
        let s = SuspensionSpec::for_cell_size(0.1);
        assert!(!s.rigid);
        assert!(
            s.rest_length < 0.1,
            "rest_length should scale down with the cell"
        );
        assert!(
            s.travel < s.rest_length,
            "travel is a fraction of the ride height"
        );
        // Doubling the cell doubles the ride height.
        let big = SuspensionSpec::for_cell_size(0.2);
        assert!((big.rest_length - 2.0 * s.rest_length).abs() < 1e-12);
    }

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
                thermal: Thermal::INERT,
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
            station: None,
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
                thermal: Thermal::INERT,
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
                thermal: Thermal::INERT,
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
                thermal: Thermal::INERT,
            },
        });
        craft.voxels.push(Voxel {
            cell: IVec3::new(1, 0, 0),
            material: Material {
                density: 3_000.0,
                strength: 1.0e9,
                thermal: Thermal::INERT,
            },
        });
        let mp = craft.mass_properties().unwrap();
        // Cell centres at x=0.5 and x=1.5; mass-weighted mean > 1.0 (midpoint).
        assert!(mp.center_of_mass.x > 1.0);
        assert!(mp.center_of_mass.x < 1.5);
    }

    #[test]
    fn a_wedge_cell_masses_half_a_cube_at_its_centroid() {
        // WI 831: one aluminium wedge at the origin (canonical: the ramp y ≤ z).
        // Mass = ρ·s³/2; CoM at the wedge centroid (1/2, 1/3, 2/3)·s.
        use crate::shape::{FillMode, Form, ShapedCell};
        let s = 0.5;
        let mut craft = VoxelCraft::new(s);
        craft.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        let full = craft.mass_properties().unwrap();
        craft.set_shape(ShapedCell {
            cell: IVec3::ZERO,
            form: Form::Wedge,
            orientation: 0,
            fill: FillMode::Solid,
        });
        let mp = craft.mass_properties().unwrap();
        assert!(
            (mp.mass - 0.5 * full.mass).abs() < 1e-9,
            "half a cube's mass"
        );
        let want = DVec3::new(0.5, 1.0 / 3.0, 2.0 / 3.0) * s;
        assert!(
            (mp.center_of_mass - want).length() < 1e-12,
            "CoM at the wedge centroid: {} vs {want}",
            mp.center_of_mass
        );
        // Clearing the shape restores the full cube exactly.
        craft.clear_shape(IVec3::ZERO);
        let back = craft.mass_properties().unwrap();
        assert!((back.mass - full.mass).abs() < 1e-12);
    }

    #[test]
    fn rotating_a_wedge_moves_com_and_inertia_as_the_rotation() {
        use crate::shape::{constants, rotations, FillMode, Form, ShapedCell};
        let s = 1.0;
        let base = |orientation: u8| {
            let mut craft = VoxelCraft::new(s);
            craft.voxels.push(Voxel {
                cell: IVec3::ZERO,
                material: Material::STEEL,
            });
            craft.set_shape(ShapedCell {
                cell: IVec3::ZERO,
                form: Form::Wedge,
                orientation,
                fill: FillMode::Solid,
            });
            craft.mass_properties().unwrap()
        };
        let c = constants(Form::Wedge);
        for &o in &c.distinct_orientations {
            let mp = base(o);
            let r = rotations()[o as usize];
            let want_com = r * (c.centroid - DVec3::splat(0.5)) + DVec3::splat(0.5);
            assert!(
                (mp.center_of_mass - want_com).length() < 1e-12,
                "orientation {o}"
            );
            // The craft inertia about the CoM is the rotated form inertia
            // (single voxel at its own centroid — no parallel-axis term).
            let want_i = r * c.unit_inertia * r.transpose() * Material::STEEL.density;
            assert!(
                (mp.inertia - want_i).abs_diff_eq(DMat3::ZERO, 1e-9),
                "orientation {o}: inertia mismatch"
            );
        }
    }

    // --- WI 834: shaped cells present fractional station areas (design R2).
    // The WI 832 staging pin (`the_aero_envelope_ignores_shaped_cells_until_
    // wi_834`) was deleted by this stage, as designed, and replaced by the
    // positive tests below.

    /// The 3×3×3 hull with a one-cell central cavity (the WI 832 fixture).
    fn cavity_hull() -> VoxelCraft {
        let mut c = VoxelCraft::new(1.0);
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    if !(x == 1 && y == 1 && z == 1) {
                        c.voxels.push(Voxel {
                            cell: IVec3::new(x, y, z),
                            material: Material::ALUMINIUM,
                        });
                    }
                }
            }
        }
        c
    }

    fn shape_wedge(craft: &mut VoxelCraft, cell: IVec3, orientation: u8) {
        use crate::shape::{FillMode, Form, ShapedCell};
        craft.set_shape(ShapedCell {
            cell,
            form: Form::Wedge,
            orientation,
            fill: FillMode::Solid,
        });
    }

    #[test]
    fn a_wedge_faired_shoulder_smooths_the_area_curve() {
        // The workitem's headline AC: a 2×2 body with a 1×1 nose stub steps
        // [4, 4, 4, 1]; wedging the three shoulder cells fairs it to
        // [4, 4, 2.5, 1] — the same occupancy is strictly smoother (one
        // 3-cell² step spread into 1.5 + 1.5) over the same peak area. At
        // mean-slice granularity shaping smooths *transitions* — the
        // area-ruling factor scores interior steps only.
        use crate::aero::area_ruling_factor;
        let build = |shaped: bool| {
            let mut c = VoxelCraft::new(1.0);
            for x in 0..3 {
                for y in 0..2 {
                    for z in 0..2 {
                        c.voxels.push(Voxel {
                            cell: IVec3::new(x, y, z),
                            material: Material::ALUMINIUM,
                        });
                    }
                }
            }
            c.voxels.push(Voxel {
                cell: IVec3::new(3, 0, 0),
                material: Material::ALUMINIUM,
            });
            if shaped {
                for cell in [
                    IVec3::new(2, 1, 0),
                    IVec3::new(2, 0, 1),
                    IVec3::new(2, 1, 1),
                ] {
                    shape_wedge(&mut c, cell, 0);
                }
            }
            c
        };
        let stepped = build(false).area_curve(Axis::X);
        let faired = build(true).area_curve(Axis::X);
        assert_eq!(
            stepped,
            vec![(0, 4.0), (1, 4.0), (2, 4.0), (3, 1.0)],
            "the unshaped stub steps"
        );
        assert_eq!(
            faired,
            vec![(0, 4.0), (1, 4.0), (2, 2.5), (3, 1.0)],
            "the wedged shoulder fairs the transition"
        );
        let (fs, ff) = (area_ruling_factor(&stepped), area_ruling_factor(&faired));
        assert!(ff < fs, "faired {ff} strictly smoother than stepped {fs}");
        let peak = |c: &[(i32, f64)]| c.iter().map(|&(_, a)| a).fold(0.0_f64, f64::max);
        assert_eq!(peak(&stepped), peak(&faired), "same peak area");
    }

    #[test]
    fn an_enclosed_wedge_remainder_counts_fully_and_an_exterior_one_does_not() {
        // Design R2, both halves on the WI 832 hull: a wedge whose open half
        // sits inside the sealed hull presents the full cell (the curve is the
        // unshaped hull's, exactly); a wedge whose open half faces the
        // exterior presents only its solid half (its station drops by the
        // remainder × cell²) — and the cavity stays sealed either way.
        use crate::shape::rotations;
        use glam::DMat3;
        let orientation_of = |want: DMat3| -> u8 {
            rotations()
                .iter()
                .position(|r| r.abs_diff_eq(want, 1e-12))
                .expect("rotation in the table") as u8
        };
        let plain = cavity_hull();
        let wedge_cell = IVec3::new(1, 1, 0);

        // Full face outward (−Z): the remainder joins the sealed cavity.
        let mut enclosed = cavity_hull();
        let out = orientation_of(DMat3::from_diagonal(glam::DVec3::new(-1.0, 1.0, -1.0)));
        shape_wedge(&mut enclosed, wedge_cell, out);
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            assert_eq!(
                enclosed.area_curve(axis),
                plain.area_curve(axis),
                "enclosed remainder counts fully"
            );
        }

        // Identity: the full face points at the cavity (+Z), the open half at
        // the exterior — the remainder vents, the solid half presents.
        let mut vented = cavity_hull();
        shape_wedge(&mut vented, wedge_cell, 0);
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let station = match axis {
                Axis::X => wedge_cell.x,
                Axis::Y => wedge_cell.y,
                Axis::Z => wedge_cell.z,
            };
            let expected: Vec<(i32, f64)> = plain
                .area_curve(axis)
                .into_iter()
                .map(|(s, a)| (s, if s == station { a - 0.5 } else { a }))
                .collect();
            assert_eq!(vented.area_curve(axis), expected);
        }
    }

    #[test]
    fn a_cavity_behind_a_half_open_wall_vents_from_the_curve() {
        // The mode flip's topology: the exterior flows *through* a shaped
        // cell's open faces (exactly as the compartment fill does — one flood,
        // one semantics), so the WI 832 vented orientation empties the cavity
        // from the envelope: structure presents, the cavity and the wedge's
        // remainder do not.
        use crate::shape::{constants, face_masks, mask_popcount, Form};
        // A wedge orientation whose two Z faces are both partial (the WI 832
        // compartments fixture's mask-searched vented case).
        let partial = |m: &crate::shape::FaceMask| {
            let n = mask_popcount(m);
            n > 0 && n < 256
        };
        let vented_o = constants(Form::Wedge)
            .distinct_orientations
            .iter()
            .copied()
            .find(|&o| {
                let masks = face_masks(Form::Wedge, o);
                partial(&masks[4]) && partial(&masks[5])
            })
            .expect("a both-z-partial orientation exists");
        let mut c = cavity_hull();
        shape_wedge(&mut c, IVec3::new(1, 1, 0), vented_o);
        // Along X: stations 0 and 2 are solid 3×3 walls (9); station 1 loses
        // the cavity (1.0) and the wedge's remainder (0.5).
        assert_eq!(
            c.area_curve(Axis::X),
            vec![(0, 9.0), (1, 7.5), (2, 9.0)],
            "the cavity and the remainder vent; the structure presents"
        );
    }

    #[test]
    fn a_lone_shaped_cell_presents_its_mean_slice_everywhere() {
        // The mean-slice identity, exhaustively: one shaped cell contributes
        // its solid volume fraction × cell² at its station — identical for
        // every catalog form, every distinct orientation, and every axis
        // (orientation-independence is the design invariant).
        use crate::shape::{constants, FillMode, ShapedCell, FORMS};
        for form in FORMS {
            let c = constants(form);
            for &o in &c.distinct_orientations {
                let mut craft = VoxelCraft::new(1.0);
                craft.voxels.push(Voxel {
                    cell: IVec3::ZERO,
                    material: Material::ALUMINIUM,
                });
                craft.set_shape(ShapedCell {
                    cell: IVec3::ZERO,
                    form,
                    orientation: o,
                    fill: FillMode::Solid,
                });
                for axis in [Axis::X, Axis::Y, Axis::Z] {
                    let curve = craft.area_curve(axis);
                    assert_eq!(curve.len(), 1, "{form:?} o{o} {axis:?}");
                    assert_eq!(curve[0].0, 0);
                    assert!(
                        (curve[0].1 - c.volume).abs() < 1e-12,
                        "{form:?} o{o} {axis:?}: presents {} want {}",
                        curve[0].1,
                        c.volume
                    );
                }
            }
        }
    }

    // --- WI 836: shell fill-variants — plate-for-mass, solid-for-topology.

    #[test]
    fn a_glass_wedge_shell_masses_its_skin_and_seals_like_the_solid_wedge() {
        // AC 1 (sim half): the shell's mass is the exact skin identity —
        // shell_area × cell² × (PANEL_FILL × cell) × density, sitting at the
        // oriented skin centroid — while the seal predicate answers exactly as
        // the solid wedge's on every boundary (fill is never consulted).
        use crate::shape::{constants, FillMode, Form, ShapedCell};
        let build = |fill: FillMode| {
            let mut c = VoxelCraft::new(1.0);
            c.voxels.push(Voxel {
                cell: IVec3::ZERO,
                material: Material::GLASS,
            });
            c.set_shape(ShapedCell {
                cell: IVec3::ZERO,
                form: Form::Wedge,
                orientation: 0,
                fill,
            });
            c
        };
        let shell = build(FillMode::Shell);
        let fc = constants(Form::Wedge);
        let v = &shell.voxels[0];
        let want_mass = Material::GLASS.density * fc.shell_area * PANEL_FILL;
        assert!((shell.voxel_mass(v) - want_mass).abs() < 1e-12);
        assert!((shell.voxel_centroid(v) - fc.shell_centroid_oriented(0)).length() < 1e-12);
        assert_eq!(shell.voxel_self_inertia(v), DMat3::ZERO, "point mass");
        // Solid-for-topology: every boundary of the cell seals identically.
        let solid_variant = build(FillMode::Solid);
        let occupied: HashSet<IVec3> = [IVec3::ZERO].into_iter().collect();
        for dir in [
            IVec3::X,
            IVec3::NEG_X,
            IVec3::Y,
            IVec3::NEG_Y,
            IVec3::Z,
            IVec3::NEG_Z,
        ] {
            assert_eq!(
                shell.boundary_sealed(&occupied, IVec3::ZERO, dir),
                solid_variant.boundary_sealed(&occupied, IVec3::ZERO, dir),
                "{dir:?}"
            );
        }
    }

    #[test]
    fn a_shell_presents_its_solid_forms_station_areas() {
        // Topology pin, aero half: the WI 834 wedge-faired shoulder with its
        // three shoulder wedges as *shells* presents the identical area curve —
        // the envelope never consults fill.
        use crate::shape::{FillMode, Form, ShapedCell};
        let build = |fill: FillMode| {
            let mut c = VoxelCraft::new(1.0);
            for x in 0..4 {
                for y in 0..2 {
                    for z in 0..2 {
                        if x == 3 && (y, z) != (0, 0) {
                            continue;
                        }
                        c.voxels.push(Voxel {
                            cell: IVec3::new(x, y, z),
                            material: Material::ALUMINIUM,
                        });
                    }
                }
            }
            for (cell, o) in [
                (IVec3::new(2, 1, 0), 0),
                (IVec3::new(2, 1, 1), 0),
                (IVec3::new(2, 0, 1), 0),
            ] {
                c.set_shape(ShapedCell {
                    cell,
                    form: Form::Wedge,
                    orientation: o,
                    fill,
                });
            }
            c
        };
        let solid = build(FillMode::Solid);
        let shell = build(FillMode::Shell);
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            assert_eq!(
                solid.area_curve(axis),
                shell.area_curve(axis),
                "{axis:?}: a shell is its solid form to the envelope"
            );
        }
    }

    #[test]
    fn a_shell_canopy_keeps_the_compartment_sealed_and_lightens_the_draft() {
        // AC 2: a 9×9×9 ablator hull (buoyant — walls ≈ 386 cells over a 343-
        // cell cavity) roofed with one glass cube canopy cell. Solid vs shell:
        // the enclosure is identical (a cube shell is the full cube to the
        // flood-fill — the bubble-canopy collapse), the shell craft is lighter
        // by exactly (1 − shell fraction) × cell³ × ρ_glass, and the harbor
        // float measure (weight ÷ fully-submerged displaced weight, enclosed
        // air included — `would_float`'s model) strictly improves.
        use crate::medium::enclosed_cells;
        use crate::shape::{constants, FillMode, Form, ShapedCell};
        let n = 9;
        let canopy = IVec3::new(4, 8, 4);
        let build = |fill: FillMode| {
            let mut c = VoxelCraft::new(1.0);
            for x in 0..n {
                for y in 0..n {
                    for z in 0..n {
                        let interior = (1..n - 1).contains(&x)
                            && (1..n - 1).contains(&y)
                            && (1..n - 1).contains(&z);
                        if interior {
                            continue;
                        }
                        let cell = IVec3::new(x, y, z);
                        c.voxels.push(Voxel {
                            cell,
                            material: if cell == canopy {
                                Material::GLASS
                            } else {
                                Material::ABLATOR
                            },
                        });
                    }
                }
            }
            c.set_shape(ShapedCell {
                cell: canopy,
                form: Form::Cube,
                orientation: 0,
                fill,
            });
            c
        };
        let solid = build(FillMode::Solid);
        let shell = build(FillMode::Shell);
        let mut enc_solid = enclosed_cells(&solid);
        let mut enc_shell = enclosed_cells(&shell);
        enc_solid.sort_by_key(|c| (c.x, c.y, c.z));
        enc_shell.sort_by_key(|c| (c.x, c.y, c.z));
        assert_eq!(enc_solid, enc_shell, "the compartment stays sealed");
        // Exactly the 7³ cavity — in particular the canopy cell itself is NOT
        // an air node (the cube's float-fold volume is 1 − ~1e-16; the ledger
        // thresholds at geometric epsilon, not > 0).
        assert_eq!(enc_solid.len(), (n as usize - 2).pow(3));
        let mass = |c: &VoxelCraft| c.mass_properties().unwrap().mass;
        let want_delta =
            (1.0 - constants(Form::Cube).shell_area * PANEL_FILL) * Material::GLASS.density;
        assert!((mass(&solid) - mass(&shell) - want_delta).abs() < 1e-9);
        // Weight ÷ fully-submerged displaced weight (hull + enclosed air), the
        // model the harbor prediction runs. Both variants float; the shell
        // floats strictly higher (glass is denser than water, so shedding its
        // interior sheds proportionally more weight than displacement).
        let ratio = |c: &VoxelCraft, enclosed: &[IVec3]| {
            let displaced: f64 =
                c.voxels.iter().map(|v| c.voxel_volume(v)).sum::<f64>() + enclosed.len() as f64;
            mass(c) / (1_025.0 * displaced)
        };
        let rs = ratio(&solid, &enc_solid);
        let rh = ratio(&shell, &enc_shell);
        assert!(rs < 1.0 && rh < 1.0, "both float: {rs} {rh}");
        assert!(rh < rs, "the shell canopy lightens the draft: {rh} vs {rs}");
    }

    #[test]
    fn shapes_ride_subassembly_insertion() {
        use crate::shape::{FillMode, Form, ShapedCell};
        let mut sub = VoxelCraft::new(1.0);
        sub.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        sub.set_shape(ShapedCell {
            cell: IVec3::ZERO,
            form: Form::OuterCorner,
            orientation: 3,
            fill: FillMode::Solid,
        });
        let mut craft = VoxelCraft::new(1.0);
        craft.insert(&sub, IVec3::new(5, 0, 0));
        let s = craft
            .shape_at(IVec3::new(5, 0, 0))
            .expect("shape offset with its cell");
        assert_eq!(s.form, Form::OuterCorner);
        assert_eq!(s.orientation, 3);
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
            thermal: Thermal::INERT,
        }; // tungsten-like
        let mut craft = VoxelCraft::new(1.0);
        craft.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: exotic,
        });
        let mp = craft.mass_properties().unwrap();
        assert!((mp.mass - 19_300.0).abs() < 1e-6);
    }

    #[test]
    fn pre_thermal_material_loads_with_inert_thermal_defaults() {
        // A material serialized before WI 687 (no `thermal` field) must still load,
        // defaulting to inert thermal properties so legacy craft never melt.
        let json = r#"{"density":2700.0,"strength":3.1e8}"#;
        let m: Material = serde_json::from_str(json).unwrap();
        assert_eq!(m.density, 2_700.0);
        assert_eq!(m.thermal, Thermal::INERT);
        assert!(m.thermal.max_temp > 1.0e8, "inert default never fails");
    }

    // --- WI 824: face panels (cells are cubes; plates live on faces) ---

    #[test]
    fn face_panel_store_canonicalizes_both_sides_and_round_trips() {
        let mut craft = VoxelCraft::new(0.5);
        assert!(craft.face_panels.is_empty());
        // The same boundary addressed from either side is one canonical entry.
        let a = IVec3::new(1, 0, 0);
        craft.set_face_panel(a, IVec3::X, Some(Material::ALUMINIUM));
        assert_eq!(craft.face_panels.len(), 1);
        assert!(craft.face_panel_between(a, IVec3::X).is_some());
        assert!(craft
            .face_panel_between(a + IVec3::X, IVec3::NEG_X)
            .is_some());
        // Placing from the other side replaces the material, never doubles.
        craft.set_face_panel(a + IVec3::X, IVec3::NEG_X, Some(Material::GLASS));
        assert_eq!(craft.face_panels.len(), 1);
        assert_eq!(
            craft.face_panel_between(a, IVec3::X).unwrap().material,
            Material::GLASS
        );
        // Removal from either side clears the one entry.
        craft.set_face_panel(a, IVec3::X, None);
        assert!(craft.face_panels.is_empty());
    }

    #[test]
    fn a_face_panel_masses_its_plate_and_an_all_panel_craft_has_mass() {
        let cs = 0.5;
        let mut craft = VoxelCraft::new(cs);
        craft.set_face_panel(IVec3::ZERO, IVec3::Y, Some(Material::ALUMINIUM));
        let mp = craft.mass_properties().expect("plates alone carry mass");
        let plate = Material::ALUMINIUM.density * cs * cs * (PANEL_FILL * cs);
        assert!(
            (mp.mass - plate).abs() < 1e-9,
            "plate mass: {} vs {plate}",
            mp.mass
        );
        // The plate sits at its face centre (one cell up from the cell centre's y).
        assert!((mp.center_of_mass.y - cs).abs() < 1e-9);
    }

    #[test]
    fn a_plated_shell_presents_the_solid_cross_section_via_the_envelope() {
        // WI 827 sealed envelope (fixture rebuilt at WI 820, converter retired):
        // a plated hull is plates around sealed air, and the envelope
        // (everything the exterior cannot reach) is exactly the solid region —
        // so the curve equals the solid hull's (the flow goes around a closed
        // hull, not through it).
        let solid = block(2, 2, 2, 0.5, Material::ALUMINIUM);
        let cells: Vec<IVec3> = solid.voxels.iter().map(|v| v.cell).collect();
        let mut panel = VoxelCraft::new(0.5);
        panel.plate_shell(&cells, Material::ALUMINIUM);
        assert!(panel.voxels.is_empty());
        assert_eq!(panel.area_curve(Axis::X), solid.area_curve(Axis::X));
        assert_eq!(panel.area_curve(Axis::Y), solid.area_curve(Axis::Y));
        assert_eq!(panel.area_curve(Axis::Z), solid.area_curve(Axis::Z));
    }

    /// A solid-walled `n×n×n` box with a hollow interior (walls one cell thick).
    fn hollow_box(n: i32, cell_size: f64) -> VoxelCraft {
        let mut craft = VoxelCraft::new(cell_size);
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let wall = x == 0 || x == n - 1 || y == 0 || y == n - 1 || z == 0 || z == n - 1;
                    if wall {
                        craft.voxels.push(Voxel {
                            cell: IVec3::new(x, y, z),
                            material: Material::ALUMINIUM,
                        });
                    }
                }
            }
        }
        craft
    }

    #[test]
    fn a_hollow_hull_presents_its_full_body_cross_section() {
        // The R2 comparison fixture (WI 827): before the envelope, a hollow
        // hull's mid-body slice counted only its walls (4×4 − 2×2 = 12 cells);
        // with the envelope the enclosed cavity presents too, so hollow == solid
        // (16 cells per mid slice). The air inside goes around with the hull.
        let hollow = hollow_box(4, 0.5);
        let solid = block(4, 4, 4, 0.5, Material::ALUMINIUM);
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            assert_eq!(hollow.area_curve(axis), solid.area_curve(axis));
        }
        let cell_area = 0.5 * 0.5;
        let mid = hollow
            .area_curve(Axis::X)
            .iter()
            .find(|&&(s, _)| s == 1)
            .map(|&(_, a)| a)
            .unwrap();
        assert!(
            (mid - 16.0 * cell_area).abs() < 1e-9,
            "mid slice presents the full 4×4 body, was walls-only 12 pre-827 (got {mid})"
        );
    }

    #[test]
    fn a_breached_hull_presents_only_its_structure() {
        // Reachability does the work: knock one wall cell out and the cavity
        // vents — only the remaining structure presents cross-section.
        let mut breached = hollow_box(4, 0.5);
        breached.voxels.retain(|v| v.cell != IVec3::new(0, 1, 1));
        let curve = breached.area_curve(Axis::X);
        let integral: f64 = curve.iter().map(|(_, a)| a * breached.cell_size).sum();
        assert!(
            (integral - breached.occupied_volume()).abs() < 1e-9,
            "a vented cavity contributes nothing: curve integrates to structure volume alone"
        );
    }

    #[test]
    fn a_lone_plate_presents_its_plate() {
        // A free-standing plate presents its full face sliced along its normal
        // (at its owning cell's station) and its thin edge sliced tangentially.
        let cs = 0.5;
        let mut craft = VoxelCraft::new(cs);
        craft.set_face_panel(IVec3::ZERO, IVec3::Y, Some(Material::ALUMINIUM));
        let cell_area = cs * cs;
        assert_eq!(craft.area_curve(Axis::Y), vec![(0, cell_area)]);
        assert_eq!(craft.area_curve(Axis::X), vec![(0, PANEL_FILL * cell_area)]);
        assert_eq!(craft.area_curve(Axis::Z), vec![(0, PANEL_FILL * cell_area)]);
    }

    #[test]
    fn a_plate_on_the_envelope_skin_adds_nothing() {
        // A plate flush against a solid cell is the hull's skin: the cell behind
        // it already presents that area — no double counting.
        let solid = block(1, 1, 1, 0.5, Material::ALUMINIUM);
        let mut skinned = solid.clone();
        skinned.set_face_panel(IVec3::ZERO, IVec3::Y, Some(Material::ALUMINIUM));
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            assert_eq!(skinned.area_curve(axis), solid.area_curve(axis));
        }
    }

    #[test]
    fn an_open_door_does_not_vent_the_cross_section() {
        // Doors are structure for aero (WI 827 decision): an open hatch does not
        // delete the fuselage's drag area, so the curve is a build-time property
        // independent of door state — and equal to the fully sealed hull's.
        let sealed = hollow_box(4, 0.5);
        let mut doored = sealed.clone();
        let gap = IVec3::new(0, 1, 1);
        doored.voxels.retain(|v| v.cell != gap);
        doored.doors.push(Door {
            cell: gap,
            open: false,
        });
        let mut open = doored.clone();
        open.doors[0].open = true;
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            assert_eq!(doored.area_curve(axis), open.area_curve(axis));
            assert_eq!(doored.area_curve(axis), sealed.area_curve(axis));
        }
    }

    #[test]
    fn plate_shell_follows_the_r1_double_skin_rule() {
        // WI 820 (the builder that replaced the retired legacy converter): a
        // 3×3×3 shell around a cavity plates every face adjacent to a cell
        // outside the shell — inner and outer skins (the WI 824 audited
        // double-skin accounting), no voxels. Face cells (not corners/edges)
        // have exactly two such faces: exterior + cavity. Total plates:
        // 6×(1+1) + 12×2 + 8×3 = 60.
        let mut cells = Vec::new();
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    if !(x == 1 && y == 1 && z == 1) {
                        cells.push(IVec3::new(x, y, z));
                    }
                }
            }
        }
        let mut shell = VoxelCraft::new(0.5);
        shell.plate_shell(&cells, Material::ALUMINIUM);
        assert!(shell.voxels.is_empty(), "a plated shell holds no voxels");
        assert_eq!(shell.face_panels.len(), 60);
        // Deterministic regardless of input order: the sorted panel store makes
        // reversed input build the identical craft.
        let mut reversed = VoxelCraft::new(0.5);
        let mut rev = cells.clone();
        rev.reverse();
        reversed.plate_shell(&rev, Material::ALUMINIUM);
        assert_eq!(reversed, shell);
    }
}
