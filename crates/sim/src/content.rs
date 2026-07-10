//! Content catalog + RON asset packs: loader, override model, merge ladder (WIs 547/548).
//!
//! The foundation of the content layer (content aspect, Slice 0): a flat
//! namespace of **typed, identified records** — devices, materials, resources,
//! and body *references* — that the simulation will look up by id instead of
//! hardcoding instances. Records carry **real physical parameters only** (the
//! quantities the sim already consumes: a material *is* a [`Material`], an
//! engine spec mirrors [`crate::propulsion::Engine`], a motor spec mirrors
//! [`crate::powertrain::MotorSpec`]); balance scalars arrive with WI 549 and
//! are **never** stored here.
//!
//! An **asset pack** is a versioned, hand-authored RON document: a manifest
//! (id + version + content-format version + optional `depends`) plus records.
//! Records support ParentName-style **abstract-base inheritance** (the
//! CDDA/RimWorld pattern): a record may name a `parent` of the same kind;
//! unset fields inherit the parent's resolved values, declared fields
//! override, and resolution happens **after** the whole composition is read,
//! so declaration order never matters. A parent may be abstract *or* concrete
//! — `abstract: true` controls only whether the record appears in the
//! resolved catalog.
//!
//! **The override model + merge ladder (WI 548).** Every tunable field is
//! reachable by one field-operation mechanism — **set / multiply / extend /
//! delete** over an inherited base, with the proportional **multiply** as the
//! first-class tuning primitive; **suppress** (WI 880, design I4) is the
//! body-recipe-specific "do not generate" marker, distinct from delete in
//! both semantics and provenance (a suppressed drawn field is supplied by its
//! same-name explicit scalar instead of its seeded draw). Overrides live in **override sets** (RON
//! documents declaring a source id + a **phase**) and resolve through the
//! deterministic **named-phase ladder**: settings (reserved for WI 549) →
//! base packs (record *definitions*) → pack-to-pack patches → scenario →
//! player/local — never by file-discovery or input order. Within a phase,
//! sources order by declared dependencies (topological) then stable id;
//! later writes win and every displaced value is recorded. **The ladder runs
//! on raw (pre-inheritance) records**, so an override targeting an abstract
//! base *re-flows to every variant* (the design's bulk-tuning-by-one-line
//! property); inheritance then resolves, then validation. Structural fields
//! (`id`, `parent`, `abstract`, `class`) are not override targets — overrides
//! tune values, never topology.
//!
//! **Per-value provenance:** every resolved value traces to the source +
//! phase that produced it, with the ordered chain of values it shadowed
//! (newest displaced first, down to the pack-authored original or an explicit
//! `Unset`). Inherited fields carry the provenance of the value they
//! inherited, so a variant tuned via its base still traces to that override.
//!
//! **Content is data, never code** (untrusted-pack safety): the loader
//! deserializes and validates; nothing in a pack can execute. Failures are
//! **loud and typed** — parse errors, an unknown format version, duplicate
//! ids/sources, unknown dependencies or dependency cycles, unknown or cyclic
//! or kind-mismatched parents, unknown/structural/type-mismatched override
//! targets, and missing or inapplicable fields all fail the load naming the
//! offender; records are never silently skipped. Unknown fields and unknown
//! record kinds are rejected (authored artifacts deserve typo-loudness;
//! forward-compat relaxation is a deliberate later migration decision, unlike
//! the reserved-container idiom of [`crate::persist`], which serves
//! machine-written saves).
//!
//! The pack **format version** ([`CONTENT_FORMAT_VERSION`]) is a separate
//! line from [`crate::persist::FORMAT_VERSION`]: authored packs and
//! machine-written saves are different artifact classes with different
//! migration pressure. Adding the override-set document type followed
//! `persist`'s additive rule (existing pack files are unchanged). A pack's
//! own `version` field is an opaque identity string in this slice.
//!
//! **Body records reference, never define** (content design Currency Addendum
//! 2026-07-02): a body record carries the identity of a world-building
//! library asset ([`crate::body_library`] slug); world-building owns body
//! data, content composes it.
//!
//! Merging is **deterministic**: ordering is a function of declared phases,
//! dependencies, and ids only, and the catalog is ordered storage — the same
//! documents produce the same catalog in any input order (asserted in test).
//! Headless and rendering-free. The sim does not consume the catalog yet
//! (WI 550's scope).

use crate::body_asset::{BodyAsset, Rotation, SurfaceRecipe};
use crate::body_derive;
use crate::bodygen::{self, Archetype, ArchetypeBands};
use crate::fluid::FluidMedium;
use crate::voxel::{Material, Thermal};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::path::Path;
use std::sync::OnceLock;

/// Version of the authored content-document *format* (packs and override
/// sets; not of any pack's content). Rejected loudly when a document declares
/// anything else. Increments only on a schema change to the document shapes.
///
/// History: 1 = through WI 892. 2 = WI 889 — the archetype band vocabulary
/// swapped to the drawn independent set (`nominal_insolation` / `bond_albedo`
/// / `greenhouse_delta_t` / `mean_molar_mass` band pairs replace the direct
/// `atmosphere_surface_density`/`_scale_height`/`_temperature` pairs), the
/// project's first non-additive pack-grammar change. Stale packs are refused
/// **by version** via the header-first probe in `merge_composition` (the
/// [`crate::persist`] two-stage-probe pattern) — never by an incidental
/// unknown-field parse error. WI 880 (the `suppress` op + record field) was
/// **additive** — every format-2 document resolves unchanged — so the
/// version did not move.
pub const CONTENT_FORMAT_VERSION: u32 = 2;

/// Field names that overrides may never target: record identity and
/// inheritance topology are definitions, not tunables.
const STRUCTURAL_FIELDS: [&str; 6] = ["id", "parent", "abstract", "class", "shape", "layer_type"];

/// The largest integer an authored (f64 Num) `surface_seed` may carry: 2^53,
/// the ceiling of exact integer representation in an f64. Above it the slot
/// silently loses precision, so validation rejects it loudly instead (WI 891,
/// parked decision (b) from WIs 881/883).
const MAX_AUTHORED_SEED: f64 = 9_007_199_254_740_992.0;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a composition failed to load. Every variant names its offender — a
/// merge never completes partially and records are never silently dropped.
#[derive(Debug)]
pub enum ContentError {
    /// A document file could not be read.
    Io(std::io::Error),
    /// RON text failed to parse (includes position context from `ron`).
    Parse(String),
    /// A document declares a content-format version this build does not know.
    Format { found: u32 },
    /// Two records share an id (the namespace is flat across kinds and packs).
    DuplicateId { id: String },
    /// Two source documents (packs / override sets) share an id.
    DuplicateSource { id: String },
    /// A source names a dependency that is not among the supplied documents
    /// (packs depend on packs; override sets on same-phase sets).
    UnknownDependency { source: String, depends_on: String },
    /// Source dependencies form a cycle.
    DependencyCycle { source: String },
    /// A record names a parent id that does not exist in the composition.
    UnknownParent { child: String, parent: String },
    /// A record names a parent of a different record kind.
    KindMismatch { child: String, parent: String },
    /// A record's inheritance chain loops (including naming itself).
    Cycle { id: String },
    /// A concrete (non-abstract) record is missing a required physical field
    /// after inheritance resolution.
    MissingField { id: String, field: &'static str },
    /// A device record carries a field that does not apply to its class
    /// (an authoring error caught loudly, like unknown fields).
    InapplicableField { id: String, field: &'static str },
    /// A derivation input is outside its physical domain (WI 886: a bond
    /// albedo above 1, a non-positive molar mass/radius, a negative
    /// insolation, a non-positive surface temperature feeding the gas
    /// relations) — no non-finite or unphysical value may enter a resolved
    /// body.
    UnphysicalValue { id: String, field: &'static str },
    /// A body recipe's `surface_stack` names a record that does not exist as
    /// a concrete record (unknown id, or an abstract base — those are
    /// inheritance targets, not content) (WI 892).
    UnknownStackLayer { body: String, layer: String },
    /// A body recipe's `surface_stack` names a record of another kind.
    StackLayerWrongKind { body: String, layer: String },
    /// A body recipe's `surface_stack` names the same layer twice — by-id
    /// addressing would be ambiguous.
    DuplicateStackLayer { body: String, layer: String },
    /// A surface layer declares a `layer_type` this build does not know.
    UnknownLayerType { id: String, layer_type: String },
    /// A `suppress` names a field that is not generator-drawn (WI 880) —
    /// suppression applies to the drawn independents only (design I2);
    /// derived quantities and non-body fields have nothing to suppress.
    NotSuppressible { record: String, field: String },
    /// An override's target matches nothing (unknown record id or base id).
    UnknownTarget { target: String },
    /// An override names a field the targeted record's kind does not have.
    UnknownField { record: String, field: String },
    /// An override targets a structural field (identity/topology — not tunable).
    StructuralField { record: String, field: String },
    /// An override's operation does not fit the field's type.
    TypeMismatch {
        record: String,
        field: String,
        op: &'static str,
    },
    /// `multiply`/`extend`/`delete` hit a field with no value at that point in
    /// the ladder (the value flows from a parent later — target the defining
    /// base instead, which re-flows to every variant).
    UnsetField { record: String, field: String },
    /// `delete` named a list element that is not present.
    AbsentElement {
        record: String,
        field: String,
        element: String,
    },
    /// Two settings scalars share a name (one frozen value per name; the
    /// scenario is a *source of* settings documents and same-phase name
    /// collisions stay a loud authoring error — design doctrine, WI 800).
    DuplicateScalar { name: String },
    /// A settings scalar's factor is not finite and strictly positive — ×0
    /// and negatives cannot express a physical modification in a
    /// multiply-only grammar (WI 550).
    InvalidScalarFactor { name: String, factor: f64 },
    /// A balance scalar's bound field has no content-defined value when the
    /// settings bake applies — the scalar would *originate* a physical
    /// quantity rather than modify one (the physical-truth seam, WI 549).
    SeamViolation {
        scalar: String,
        record: String,
        field: String,
    },
    /// A record references something outside the catalog that does not
    /// resolve (today: a body record's world-building library slug; WI 550
    /// extends this to scenario → pack/blueprint references).
    UnresolvedReference { record: String, reference: String },
}

impl fmt::Display for ContentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContentError::Io(e) => write!(f, "content document io error: {e}"),
            ContentError::Parse(msg) => write!(f, "content document parse error: {msg}"),
            ContentError::Format { found } => write!(
                f,
                "unsupported content format version {found} (this build reads {CONTENT_FORMAT_VERSION})"
            ),
            ContentError::DuplicateId { id } => write!(f, "duplicate record id `{id}`"),
            ContentError::DuplicateSource { id } => write!(f, "duplicate source id `{id}`"),
            ContentError::UnknownDependency { source, depends_on } => {
                write!(f, "source `{source}` depends on unknown source `{depends_on}`")
            }
            ContentError::DependencyCycle { source } => {
                write!(f, "dependency cycle through source `{source}`")
            }
            ContentError::UnknownParent { child, parent } => {
                write!(f, "record `{child}` names unknown parent `{parent}`")
            }
            ContentError::KindMismatch { child, parent } => {
                write!(f, "record `{child}` names parent `{parent}` of a different kind")
            }
            ContentError::Cycle { id } => write!(f, "inheritance cycle through record `{id}`"),
            ContentError::MissingField { id, field } => {
                write!(f, "record `{id}` is missing required field `{field}`")
            }
            ContentError::InapplicableField { id, field } => {
                write!(f, "record `{id}` carries field `{field}` inapplicable to its class")
            }
            ContentError::UnknownStackLayer { body, layer } => write!(
                f,
                "body recipe `{body}` surface_stack names `{layer}`, which is not a concrete \
                 record of the composition (abstract bases are inheritance targets, not content)"
            ),
            ContentError::StackLayerWrongKind { body, layer } => write!(
                f,
                "body recipe `{body}` surface_stack names `{layer}`, which is not a SurfaceLayer \
                 record"
            ),
            ContentError::DuplicateStackLayer { body, layer } => write!(
                f,
                "body recipe `{body}` surface_stack names `{layer}` twice — by-id addressing \
                 would be ambiguous"
            ),
            ContentError::UnknownLayerType { id, layer_type } => write!(
                f,
                "surface layer `{id}` declares unknown layer_type `{layer_type}` \
                 (known: terrain, crater, material)"
            ),
            ContentError::UnphysicalValue { id, field } => {
                write!(
                    f,
                    "record `{id}` field `{field}` is outside its physical domain for derivation"
                )
            }
            ContentError::NotSuppressible { record, field } => write!(
                f,
                "suppress on record `{record}` names `{field}`, which is not a generator-drawn \
                 field — suppression applies to drawn independents only"
            ),
            ContentError::UnknownTarget { target } => {
                write!(f, "override targets unknown {target}")
            }
            ContentError::UnknownField { record, field } => {
                write!(f, "override on record `{record}` names unknown field `{field}`")
            }
            ContentError::StructuralField { record, field } => write!(
                f,
                "override on record `{record}` targets structural field `{field}` (identity/topology is not tunable)"
            ),
            ContentError::TypeMismatch { record, field, op } => {
                write!(f, "override op `{op}` does not fit field `{field}` of record `{record}`")
            }
            ContentError::UnsetField { record, field } => write!(
                f,
                "override on record `{record}` field `{field}` has no value at this point in the ladder \
                 (it flows from a parent; target the defining base instead — a base override re-flows to every variant)"
            ),
            ContentError::AbsentElement { record, field, element } => write!(
                f,
                "override delete on record `{record}` field `{field}`: element `{element}` is not present"
            ),
            ContentError::DuplicateScalar { name } => {
                write!(f, "duplicate balance-scalar name `{name}`")
            }
            ContentError::InvalidScalarFactor { name, factor } => write!(
                f,
                "balance scalar `{name}` has factor {factor}: a factor must be finite and \
                 strictly positive (a multiply-only grammar cannot express ×0 or negatives)"
            ),
            ContentError::SeamViolation { scalar, record, field } => write!(
                f,
                "balance scalar `{scalar}` would originate (not modify) record `{record}` field `{field}`: \
                 no content record defines that quantity — a scalar may only multiply a physically-defined value"
            ),
            ContentError::UnresolvedReference { record, reference } => {
                write!(f, "record `{record}` references unresolved `{reference}`")
            }
        }
    }
}

impl std::error::Error for ContentError {}

// ---------------------------------------------------------------------------
// Raw (authored) forms — permissive all-`Option` shapes the loader reads.
// ---------------------------------------------------------------------------

/// The pack document as authored: manifest + record definitions.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPack {
    /// Document *format* version — must equal [`CONTENT_FORMAT_VERSION`].
    format: u32,
    /// The pack's identity.
    id: String,
    /// The pack's own version — an opaque identity string in this slice.
    version: String,
    /// Other pack ids this pack orders after (intra-phase determinism).
    #[serde(default)]
    depends: Vec<String>,
    /// The authored records, any order (resolution is order-independent).
    records: Vec<RawRecord>,
}

/// One authored record. The RON kind tag is the enum variant name.
#[derive(Debug, Clone, Deserialize)]
enum RawRecord {
    Device(RawDevice),
    Material(RawMaterial),
    Resource(RawResource),
    Body(RawBody),
    /// Boxed — the flattened body recipe carries many more fields than the other
    /// raw records (fixed scalars + WI 883 archetype bands), so an unboxed variant
    /// would bloat every `RawRecord`.
    BodyRecipe(Box<RawBodyRecipe>),
    /// One element definition of a body's surface-layer stack (WI 892).
    SurfaceLayer(RawSurfaceLayer),
}

/// A record kind — the override target-selector vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum RecordKind {
    Device,
    Material,
    Resource,
    Body,
    BodyRecipe,
    SurfaceLayer,
}

impl RawRecord {
    fn id(&self) -> &str {
        match self {
            RawRecord::Device(r) => &r.id,
            RawRecord::Material(r) => &r.id,
            RawRecord::Resource(r) => &r.id,
            RawRecord::Body(r) => &r.id,
            RawRecord::BodyRecipe(r) => &r.id,
            RawRecord::SurfaceLayer(r) => &r.id,
        }
    }
    fn parent(&self) -> Option<&str> {
        match self {
            RawRecord::Device(r) => r.parent.as_deref(),
            RawRecord::Material(r) => r.parent.as_deref(),
            RawRecord::Resource(r) => r.parent.as_deref(),
            RawRecord::Body(r) => r.parent.as_deref(),
            RawRecord::BodyRecipe(r) => r.parent.as_deref(),
            RawRecord::SurfaceLayer(r) => r.parent.as_deref(),
        }
    }
    fn is_abstract(&self) -> bool {
        match self {
            RawRecord::Device(r) => r.is_abstract,
            RawRecord::Material(r) => r.is_abstract,
            RawRecord::Resource(r) => r.is_abstract,
            RawRecord::Body(r) => r.is_abstract,
            RawRecord::BodyRecipe(r) => r.is_abstract,
            RawRecord::SurfaceLayer(r) => r.is_abstract,
        }
    }
    fn kind(&self) -> RecordKind {
        match self {
            RawRecord::Device(_) => RecordKind::Device,
            RawRecord::Material(_) => RecordKind::Material,
            RawRecord::Resource(_) => RecordKind::Resource,
            RawRecord::Body(_) => RecordKind::Body,
            RawRecord::BodyRecipe(_) => RecordKind::BodyRecipe,
            RawRecord::SurfaceLayer(_) => RecordKind::SurfaceLayer,
        }
    }

    /// Child-over-parent field merge. `parent` must already be fully merged.
    /// Fails with [`ContentError::KindMismatch`] when the kinds differ.
    fn merge_over(self, parent: &RawRecord) -> Result<RawRecord, ContentError> {
        match (self, parent) {
            (RawRecord::Device(c), RawRecord::Device(p)) => Ok(RawRecord::Device(c.merge_over(p))),
            (RawRecord::Material(c), RawRecord::Material(p)) => {
                Ok(RawRecord::Material(c.merge_over(p)))
            }
            (RawRecord::Resource(c), RawRecord::Resource(p)) => {
                Ok(RawRecord::Resource(c.merge_over(p)))
            }
            (RawRecord::Body(c), RawRecord::Body(p)) => Ok(RawRecord::Body(c.merge_over(p))),
            (RawRecord::BodyRecipe(c), RawRecord::BodyRecipe(p)) => {
                Ok(RawRecord::BodyRecipe(Box::new(c.merge_over(p))))
            }
            (RawRecord::SurfaceLayer(c), RawRecord::SurfaceLayer(p)) => {
                Ok(RawRecord::SurfaceLayer(c.merge_over(p)))
            }
            (c, p) => Err(ContentError::KindMismatch {
                child: c.id().to_string(),
                parent: p.id().to_string(),
            }),
        }
    }
}

/// The functional class a device record describes — the discriminator that
/// decides which physical fields apply. Ordered/hashable so it can key a
/// scenario's device-binding map (WI 550).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DeviceClass {
    Engine,
    Tank,
    Battery,
    Motor,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDevice {
    id: String,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default, rename = "abstract")]
    is_abstract: bool,
    #[serde(default)]
    class: Option<DeviceClass>,
    /// Bulk density, kg/m³ — cell-scaled mass, the WI 615 shape.
    #[serde(default)]
    density: Option<f64>,
    // Engine (mirrors `propulsion::Engine`).
    #[serde(default)]
    exhaust_velocity: Option<f64>,
    #[serde(default)]
    max_mass_flow: Option<f64>,
    // Tank (kg) / Battery (J).
    #[serde(default)]
    capacity: Option<f64>,
    // Motor (mirrors `powertrain::MotorSpec`).
    #[serde(default)]
    max_torque: Option<f64>,
    #[serde(default)]
    top_speed: Option<f64>,
    #[serde(default)]
    motor_mass: Option<f64>,
    #[serde(default)]
    draw: Option<f64>,
    /// Inert metadata tags (WI 548) — the first list field; a content-filter
    /// hook for later work. `None` inherits; an authored list replaces.
    #[serde(default)]
    tags: Option<Vec<String>>,
}

impl RawDevice {
    fn merge_over(self, p: &RawDevice) -> RawDevice {
        RawDevice {
            id: self.id,
            parent: self.parent,
            is_abstract: self.is_abstract,
            class: self.class.or(p.class),
            density: self.density.or(p.density),
            exhaust_velocity: self.exhaust_velocity.or(p.exhaust_velocity),
            max_mass_flow: self.max_mass_flow.or(p.max_mass_flow),
            capacity: self.capacity.or(p.capacity),
            max_torque: self.max_torque.or(p.max_torque),
            top_speed: self.top_speed.or(p.top_speed),
            motor_mass: self.motor_mass.or(p.motor_mass),
            draw: self.draw.or(p.draw),
            tags: self.tags.or_else(|| p.tags.clone()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMaterial {
    id: String,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default, rename = "abstract")]
    is_abstract: bool,
    /// kg/m³.
    #[serde(default)]
    density: Option<f64>,
    /// Tensile strength, Pa.
    #[serde(default)]
    strength: Option<f64>,
    // The thermal quad (required for concrete materials)…
    #[serde(default)]
    specific_heat: Option<f64>,
    #[serde(default)]
    conductivity: Option<f64>,
    #[serde(default)]
    emissivity: Option<f64>,
    #[serde(default)]
    max_temp: Option<f64>,
    // …plus the optional ablation trio (default 0 = non-ablative).
    #[serde(default)]
    ablation_temp: Option<f64>,
    #[serde(default)]
    latent_heat: Option<f64>,
    #[serde(default)]
    ablator_fraction: Option<f64>,
    /// Inert metadata tags (WI 548).
    #[serde(default)]
    tags: Option<Vec<String>>,
}

impl RawMaterial {
    fn merge_over(self, p: &RawMaterial) -> RawMaterial {
        RawMaterial {
            id: self.id,
            parent: self.parent,
            is_abstract: self.is_abstract,
            density: self.density.or(p.density),
            strength: self.strength.or(p.strength),
            specific_heat: self.specific_heat.or(p.specific_heat),
            conductivity: self.conductivity.or(p.conductivity),
            emissivity: self.emissivity.or(p.emissivity),
            max_temp: self.max_temp.or(p.max_temp),
            ablation_temp: self.ablation_temp.or(p.ablation_temp),
            latent_heat: self.latent_heat.or(p.latent_heat),
            ablator_fraction: self.ablator_fraction.or(p.ablator_fraction),
            tags: self.tags.or_else(|| p.tags.clone()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawResource {
    id: String,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default, rename = "abstract")]
    is_abstract: bool,
    /// kg/m³ where meaningful (a massless resource like charge omits it).
    #[serde(default)]
    density: Option<f64>,
    /// Economy-facing tradability (inert until the economy sibling lands).
    #[serde(default)]
    tradable: Option<bool>,
    /// Inert metadata tags (WI 548).
    #[serde(default)]
    tags: Option<Vec<String>>,
}

impl RawResource {
    fn merge_over(self, p: &RawResource) -> RawResource {
        RawResource {
            id: self.id,
            parent: self.parent,
            is_abstract: self.is_abstract,
            density: self.density.or(p.density),
            tradable: self.tradable.or(p.tradable),
            tags: self.tags.or_else(|| p.tags.clone()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBody {
    id: String,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default, rename = "abstract")]
    is_abstract: bool,
    /// World-building body-library slug this record points at. A body record
    /// **references** a library asset; it never defines body physical data.
    #[serde(default)]
    body: Option<String>,
    /// Inert metadata tags (WI 548).
    #[serde(default)]
    tags: Option<Vec<String>>,
}

impl RawBody {
    fn merge_over(self, p: &RawBody) -> RawBody {
        RawBody {
            id: self.id,
            parent: self.parent,
            is_abstract: self.is_abstract,
            body: self.body.or_else(|| p.body.clone()),
            tags: self.tags.or_else(|| p.tags.clone()),
        }
    }
}

/// One element definition of a body's **surface-layer stack** (WI 892): a
/// stable-id, typed, switchable parameter carrier the ladder tunes like any
/// record. `layer_type` is **structural** (like `class`/`shape` — a
/// definition, not a tunable); `enabled` is the design's `disable` op
/// expressed as an ordinary Flag field (`set enabled=false` via the ladder);
/// the param fields are the union of the well-known types' keys, and
/// authoring an off-type param is loud (`InapplicableField`, the WI 884
/// posture). Bodies reference these records by id from `surface_stack`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSurfaceLayer {
    id: String,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default, rename = "abstract")]
    is_abstract: bool,
    /// Structural: which well-known type this layer is (`terrain`/`crater`/
    /// `material`).
    #[serde(default)]
    layer_type: Option<String>,
    /// Default true; `false` = carried but read as absent (the kill switch).
    #[serde(default)]
    enabled: Option<bool>,
    /// Crater: global density multiplier.
    #[serde(default)]
    density: Option<f64>,
    /// Crater: global depth multiplier.
    #[serde(default)]
    depth: Option<f64>,
    /// Material: classifier base-temperature offset, K.
    #[serde(default)]
    temperature: Option<f64>,
    /// Material: moisture midpoint offset.
    #[serde(default)]
    moisture: Option<f64>,
    /// Material: moisture deviation multiplier.
    #[serde(default)]
    moisture_scale: Option<f64>,
    /// Inert metadata tags.
    #[serde(default)]
    tags: Option<Vec<String>>,
}

impl RawSurfaceLayer {
    fn merge_over(self, p: &RawSurfaceLayer) -> RawSurfaceLayer {
        RawSurfaceLayer {
            id: self.id,
            parent: self.parent,
            is_abstract: self.is_abstract,
            layer_type: self.layer_type.or_else(|| p.layer_type.clone()),
            enabled: self.enabled.or(p.enabled),
            density: self.density.or(p.density),
            depth: self.depth.or(p.depth),
            temperature: self.temperature.or(p.temperature),
            moisture: self.moisture.or(p.moisture),
            moisture_scale: self.moisture_scale.or(p.moisture_scale),
            tags: self.tags.or_else(|| p.tags.clone()),
        }
    }
}

/// A **body recipe** (WI 881): the *intrinsic* definition of a celestial body as
/// composable content data — the flattened, ladder-tunable form of a
/// [`BodyAsset`]. Distinct from [`RawBody`], which is a placement-side *reference*
/// to a `body_library` slug (the asset ⊕ placement split, in the content layer).
/// Nested `BodyAsset` structure is flattened onto scalar slots so the ladder can
/// inherit/override each field; `validate` reassembles a `BodyAsset`. Axis is
/// fixed to +Z this slice (both `bodygen` and `earthlike` rotate about +Z);
/// axial tilt, moisture, and the ordered surface-layer stack are later slices.
///
/// Two resolve modes, discriminated by `shape` (WI 883):
/// - **Fixed** (no `shape`): the scalar fields below are the body directly (the
///   WI-881 path — `earthlike`, `earthlike_ice_age`).
/// - **Sampled** (`shape` set): an archetype. The `*_min`/`*_max` **band** fields
///   carry the parameter ranges; `validate` samples them at `surface_seed` via
///   [`bodygen::sample`], reproducing `bodygen::generate` for that shape. The
///   scalar fields are unused in this mode; only `name`, `surface_seed`, and the
///   shape's drawn bands are required. `shape` is structural (not an override
///   target) and inherits — so a concrete body `parent`-ing an archetype base
///   inherits its shape + bands and adds a seed.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBodyRecipe {
    id: String,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default, rename = "abstract")]
    is_abstract: bool,
    /// Archetype shape (WI 883). Absent ⇒ fixed body (scalar fields); present ⇒
    /// sampled body (band fields). Structural: selects the sampler branch and is
    /// never an override target.
    #[serde(default)]
    shape: Option<Archetype>,
    /// Human-facing display name.
    #[serde(default)]
    name: Option<String>,
    /// Gravitational parameter μ = G·M, m³/s².
    #[serde(default)]
    mu: Option<f64>,
    /// Surface (sea-level) radius, metres.
    #[serde(default)]
    radius: Option<f64>,
    /// Sidereal rotation period, seconds (`0.0` ⇒ non-rotating); axis is +Z.
    #[serde(default)]
    rotation_period: Option<f64>,
    // Fluid medium (flattened) — see `fluid::FluidMedium`.
    #[serde(default)]
    atmosphere_surface_density: Option<f64>,
    #[serde(default)]
    atmosphere_surface_pressure: Option<f64>,
    #[serde(default)]
    atmosphere_scale_height: Option<f64>,
    #[serde(default)]
    ocean_surface_density: Option<f64>,
    #[serde(default)]
    ocean_surface_pressure: Option<f64>,
    #[serde(default)]
    ocean_density_gradient: Option<f64>,
    #[serde(default)]
    gravity: Option<f64>,
    #[serde(default)]
    atmosphere_temperature: Option<f64>,
    #[serde(default)]
    ocean_temperature: Option<f64>,
    /// Master surface seed. A non-negative whole number (named-body seeds are
    /// small; stored as a number, cast to `u64` at validate).
    #[serde(default)]
    surface_seed: Option<f64>,
    /// Biome classifier base-temperature offset, K (WI 870/875). Absent ⇒ the
    /// surface `material` resolves to JSON `null` (no override), matching a body
    /// with no per-asset offset; present ⇒ `{"temperature": offset}`.
    #[serde(default)]
    surface_temperature_offset: Option<f64>,
    /// The ordered surface-layer stack, as `SurfaceLayer` record ids (WI 892).
    /// Application order is list order; the WI 879 id-keyed list ops splice it.
    #[serde(default)]
    surface_stack: Option<Vec<String>>,
    /// Suppressed generator-drawn fields (WI 880, design I4 "do-not-generate"):
    /// recipe-vocabulary names from [`bodygen::SUPPRESSIBLE_FIELDS`]. Shaped
    /// recipes only. Each listed field is not drawn; its same-name scalar —
    /// admitted through the WI 884 mode wall for exactly these names — supplies
    /// the value. Inherits whole-value like every field; the `Suppress` ladder
    /// op appends to it (generic list ops cannot touch it).
    #[serde(default)]
    suppress: Option<Vec<String>>,
    // Derivation inputs (WI 886) — the independent thermal/gas intent a fixed
    // recipe may author *instead of* pinning the derived medium fields:
    // T_surf = T_eq(nominal_insolation, bond_albedo) + greenhouse_delta_t;
    // density/scale-height follow by ideal gas / hydrostatics with
    // mean_molar_mass. Inapplicable on shaped recipes (bands own the medium).
    /// Intent-level insolation, W/m² ("as if at nominal orbit", design C1).
    #[serde(default)]
    nominal_insolation: Option<f64>,
    /// Bond albedo, in `[0, 1)` — `1 − A` must stay positive for T_eq.
    #[serde(default)]
    bond_albedo: Option<f64>,
    /// Greenhouse warming above equilibrium, K.
    #[serde(default)]
    greenhouse_delta_t: Option<f64>,
    /// Mean molar mass of the atmosphere, kg/mol.
    #[serde(default)]
    mean_molar_mass: Option<f64>,
    // Archetype bands (WI 883, re-vocabularied WI 889) — the `[min, max)`
    // ranges the sampler draws when `shape` is set. Each bound is an
    // independent, ladder-tunable field; a shape requires only the bounds it
    // draws (Moon: radius/gravity/rotation + insolation/albedo; Rocky adds
    // pressure/greenhouse/molar-mass; Ocean adds the two ocean bands). The
    // medium is **derived** from the drawn independents via `body_derive` —
    // there are no direct temperature/density/scale-height bands.
    #[serde(default)]
    radius_min: Option<f64>,
    #[serde(default)]
    radius_max: Option<f64>,
    #[serde(default)]
    gravity_min: Option<f64>,
    #[serde(default)]
    gravity_max: Option<f64>,
    #[serde(default)]
    rotation_period_min: Option<f64>,
    #[serde(default)]
    rotation_period_max: Option<f64>,
    #[serde(default)]
    atmosphere_surface_pressure_min: Option<f64>,
    #[serde(default)]
    atmosphere_surface_pressure_max: Option<f64>,
    #[serde(default)]
    nominal_insolation_min: Option<f64>,
    #[serde(default)]
    nominal_insolation_max: Option<f64>,
    #[serde(default)]
    bond_albedo_min: Option<f64>,
    #[serde(default)]
    bond_albedo_max: Option<f64>,
    #[serde(default)]
    greenhouse_delta_t_min: Option<f64>,
    #[serde(default)]
    greenhouse_delta_t_max: Option<f64>,
    #[serde(default)]
    mean_molar_mass_min: Option<f64>,
    #[serde(default)]
    mean_molar_mass_max: Option<f64>,
    #[serde(default)]
    ocean_surface_density_min: Option<f64>,
    #[serde(default)]
    ocean_surface_density_max: Option<f64>,
    #[serde(default)]
    ocean_temperature_min: Option<f64>,
    #[serde(default)]
    ocean_temperature_max: Option<f64>,
    /// Inert metadata tags.
    #[serde(default)]
    tags: Option<Vec<String>>,
}

impl RawBodyRecipe {
    fn merge_over(self, p: &RawBodyRecipe) -> RawBodyRecipe {
        RawBodyRecipe {
            id: self.id,
            parent: self.parent,
            is_abstract: self.is_abstract,
            shape: self.shape.or(p.shape),
            name: self.name.or_else(|| p.name.clone()),
            mu: self.mu.or(p.mu),
            radius: self.radius.or(p.radius),
            rotation_period: self.rotation_period.or(p.rotation_period),
            atmosphere_surface_density: self
                .atmosphere_surface_density
                .or(p.atmosphere_surface_density),
            atmosphere_surface_pressure: self
                .atmosphere_surface_pressure
                .or(p.atmosphere_surface_pressure),
            atmosphere_scale_height: self.atmosphere_scale_height.or(p.atmosphere_scale_height),
            ocean_surface_density: self.ocean_surface_density.or(p.ocean_surface_density),
            ocean_surface_pressure: self.ocean_surface_pressure.or(p.ocean_surface_pressure),
            ocean_density_gradient: self.ocean_density_gradient.or(p.ocean_density_gradient),
            gravity: self.gravity.or(p.gravity),
            atmosphere_temperature: self.atmosphere_temperature.or(p.atmosphere_temperature),
            ocean_temperature: self.ocean_temperature.or(p.ocean_temperature),
            surface_seed: self.surface_seed.or(p.surface_seed),
            surface_temperature_offset: self
                .surface_temperature_offset
                .or(p.surface_temperature_offset),
            surface_stack: self.surface_stack.or_else(|| p.surface_stack.clone()),
            suppress: self.suppress.or_else(|| p.suppress.clone()),
            nominal_insolation: self.nominal_insolation.or(p.nominal_insolation),
            bond_albedo: self.bond_albedo.or(p.bond_albedo),
            greenhouse_delta_t: self.greenhouse_delta_t.or(p.greenhouse_delta_t),
            mean_molar_mass: self.mean_molar_mass.or(p.mean_molar_mass),
            radius_min: self.radius_min.or(p.radius_min),
            radius_max: self.radius_max.or(p.radius_max),
            gravity_min: self.gravity_min.or(p.gravity_min),
            gravity_max: self.gravity_max.or(p.gravity_max),
            rotation_period_min: self.rotation_period_min.or(p.rotation_period_min),
            rotation_period_max: self.rotation_period_max.or(p.rotation_period_max),
            atmosphere_surface_pressure_min: self
                .atmosphere_surface_pressure_min
                .or(p.atmosphere_surface_pressure_min),
            atmosphere_surface_pressure_max: self
                .atmosphere_surface_pressure_max
                .or(p.atmosphere_surface_pressure_max),
            nominal_insolation_min: self.nominal_insolation_min.or(p.nominal_insolation_min),
            nominal_insolation_max: self.nominal_insolation_max.or(p.nominal_insolation_max),
            bond_albedo_min: self.bond_albedo_min.or(p.bond_albedo_min),
            bond_albedo_max: self.bond_albedo_max.or(p.bond_albedo_max),
            greenhouse_delta_t_min: self.greenhouse_delta_t_min.or(p.greenhouse_delta_t_min),
            greenhouse_delta_t_max: self.greenhouse_delta_t_max.or(p.greenhouse_delta_t_max),
            mean_molar_mass_min: self.mean_molar_mass_min.or(p.mean_molar_mass_min),
            mean_molar_mass_max: self.mean_molar_mass_max.or(p.mean_molar_mass_max),
            ocean_surface_density_min: self
                .ocean_surface_density_min
                .or(p.ocean_surface_density_min),
            ocean_surface_density_max: self
                .ocean_surface_density_max
                .or(p.ocean_surface_density_max),
            ocean_temperature_min: self.ocean_temperature_min.or(p.ocean_temperature_min),
            ocean_temperature_max: self.ocean_temperature_max.or(p.ocean_temperature_max),
            tags: self.tags.or_else(|| p.tags.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// Override sets (WI 548) — authored tuning documents on the merge ladder.
// ---------------------------------------------------------------------------

/// The ladder phase an override set belongs to. Settings (WI 549) precedes
/// these; base packs (definitions) sit between settings and `Patch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum OverridePhase {
    /// Pack-to-pack adjustments (a compatibility pack tuning another pack).
    Patch,
    /// A scenario's tuning of the packs it enables.
    Scenario,
    /// House rules — reserved last; local edits win and are never depended on.
    Local,
}

/// An override-set document as authored.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOverrideSet {
    /// Document *format* version — must equal [`CONTENT_FORMAT_VERSION`].
    format: u32,
    /// The source's identity (shares one namespace with pack ids).
    id: String,
    /// Which ladder phase this set applies in.
    phase: OverridePhase,
    /// Same-phase set ids this source orders after.
    #[serde(default)]
    depends: Vec<String>,
    /// The overrides, applied in authored order.
    overrides: Vec<RawOverride>,
}

/// One field operation.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOverride {
    target: Target,
    field: String,
    op: Op,
}

/// What an override applies to.
#[derive(Debug, Deserialize)]
enum Target {
    /// One record by id (abstract bases are legal targets — the canonical
    /// family-tuning move: a base write re-flows to every variant).
    Id(String),
    /// Every record of a kind (zero matches is legal).
    Kind(RecordKind),
    /// Every record whose inheritance chain reaches the named base
    /// (the base itself excluded; zero inheritors is legal).
    Base(String),
}

impl Target {
    fn describe(&self) -> String {
        match self {
            Target::Id(id) => format!("record id `{id}`"),
            Target::Kind(k) => format!("kind {k:?}"),
            Target::Base(id) => format!("base `{id}`"),
        }
    }
}

/// The field operations. `Set` operands are explicitly typed (RON-safe). List
/// fields carry ordered, **id-keyed** ops (WI 879): `Extend` appends, `Delete`
/// removes by id, and `InsertAfter`/`Replace` splice relative to a named element
/// so an ordered list (e.g. a body's surface-layer stack) composes across the
/// ladder by element identity, never by index. `Suppress` (WI 880, design I4)
/// marks a shaped `BodyRecipe`'s generator-drawn field "do not generate" —
/// `field` names the *drawn field* (recipe vocabulary), and the mark lands in
/// the record's `suppress` list; the field's value must then come from the
/// same-name explicit scalar. Distinct from `Delete` in semantics (a positive
/// marker, never an absence trick) and in provenance (ledgered under the
/// `suppress` key). The dedicated op is the **only** ladder spelling: generic
/// ops aimed at the `suppress` list itself are rejected.
#[derive(Debug, Deserialize)]
enum Op {
    Set(SetValue),
    Multiply(f64),
    Extend(Vec<String>),
    Delete(String),
    /// Mark `field` suppressed on the target `BodyRecipe` (WI 880).
    Suppress,
    /// Insert `items` immediately after the first element equal to `anchor`.
    InsertAfter {
        anchor: String,
        items: Vec<String>,
    },
    /// Replace the first element equal to `target` with `items` (splice in place).
    Replace {
        target: String,
        items: Vec<String>,
    },
}

/// A typed `set` operand.
#[derive(Debug, Deserialize)]
enum SetValue {
    Number(f64),
    Bool(bool),
    Text(String),
    List(Vec<String>),
}

// ---------------------------------------------------------------------------
// Settings (WI 549) — named balance scalars, frozen first, baked at merge.
// ---------------------------------------------------------------------------

/// A settings document as authored: named balance scalars. Its grammar can
/// express **only multiplication factors** — the physical-truth seam in
/// structural form (a scalar cannot set, extend, or delete; it can only
/// modify a physically-defined quantity, and baking rejects the residual
/// originate-by-multiplying-nothing case as a [`ContentError::SeamViolation`]).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSettings {
    /// Document *format* version — must equal [`CONTENT_FORMAT_VERSION`].
    format: u32,
    /// The source's identity (shares one namespace with pack/set ids).
    id: String,
    /// The named scalars, applied in declaration order (documents in id order).
    scalars: Vec<RawScalar>,
}

/// One named balance scalar: a proportional factor bound to a target field.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawScalar {
    /// The scalar's name — frozen once per composition, surfaced in
    /// provenance and [`Catalog::settings`] for telemetry ("real × modifier").
    ///
    /// Naming convention (WI 550): names are a **semi-public contract** — they
    /// appear in telemetry and will be recorded in saves (WI 553). Use
    /// lowercase `snake_case`; renaming a scalar is an outward-facing change.
    name: String,
    /// The proportional factor. Must be finite and strictly positive (×0 and
    /// negatives are nonsense in a multiply-only grammar); 1.0 is a legal
    /// identity knob.
    factor: f64,
    /// What it multiplies: the 548 target vocabulary plus a field name.
    target: Target,
    field: String,
    /// Optional human-readable justification, surfaced with the factor in
    /// [`Catalog::settings`] → telemetry (WI 550) — the educational trust
    /// line beside "real × modifier" ("×50 — fuel logistics isn't this
    /// lesson's focus").
    rationale: Option<String>,
}

/// A frozen balance scalar as exposed on [`Catalog::settings`]: the factor
/// plus its optional authored rationale (WI 550). Serde-able so telemetry can
/// carry the composed settings (names are a semi-public contract).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Setting {
    /// The frozen proportional factor.
    pub factor: f64,
    /// Optional authored justification for telemetry/UI.
    pub rationale: Option<String>,
}

// ---------------------------------------------------------------------------
// Provenance — where every resolved value came from, and what it displaced.
// ---------------------------------------------------------------------------

/// A value source: the defining pack, an override set at a phase, or a
/// settings scalar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceRef {
    /// Authored in a pack's record definition.
    Pack { id: String },
    /// Written by an override set.
    Override {
        source: String,
        phase: OverridePhase,
    },
    /// Multiplied by a named balance scalar during the settings bake (WI 549).
    Setting { source: String, scalar: String },
}

/// A value as recorded in provenance shadows.
#[derive(Debug, Clone, PartialEq)]
pub enum ProvValue {
    Number(f64),
    Bool(bool),
    Text(String),
    List(Vec<String>),
    /// The field had no value (it was defined by the shadowing write).
    Unset,
}

/// A displaced value and the source that had produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct Shadow {
    pub value: ProvValue,
    pub source: SourceRef,
}

/// Per-field provenance: the winning source plus the ordered chain of
/// displaced values (newest displaced first, ending at the pack-authored
/// original or an [`ProvValue::Unset`]).
#[derive(Debug, Clone, PartialEq)]
pub struct FieldProvenance {
    pub source: SourceRef,
    pub shadows: Vec<Shadow>,
}

// ---------------------------------------------------------------------------
// Resolved records — fully-typed, real physical parameters.
// ---------------------------------------------------------------------------

/// Where a resolved record was defined — the record-level origin (per-field
/// detail lives in [`Entry::field_provenance`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// Id of the pack that defined the record.
    pub pack_id: String,
    /// That pack's (opaque) version string.
    pub pack_version: String,
}

/// A device's class-specific physical spec. Real quantities only — the shapes
/// the sim already consumes.
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceSpec {
    /// Mirrors [`crate::propulsion::Engine`]: thrust = mass-flow × exhaust velocity.
    Engine {
        /// Exhaust velocity, m/s (`= Isp · g₀`).
        exhaust_velocity: f64,
        /// Maximum propellant mass flow, kg/s.
        max_mass_flow: f64,
    },
    /// A propellant tank: capacity in kg.
    Tank { capacity: f64 },
    /// A battery: capacity in J.
    Battery { capacity: f64 },
    /// Mirrors [`crate::powertrain::MotorSpec`].
    Motor {
        /// Per-drive-wheel torque at full throttle, N·m.
        max_torque: f64,
        /// Wheel-spin top-speed cap, rad/s.
        top_speed: f64,
        /// Motor mass, kg.
        mass: f64,
        /// Consumption multiplier on the source's base draw.
        draw: f64,
    },
}

/// A resolved device record.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceRecord {
    pub id: String,
    /// Bulk density, kg/m³ — mass is `density × cell³` (WI 615 shape).
    pub density: f64,
    pub spec: DeviceSpec,
    /// Inert metadata tags.
    pub tags: Vec<String>,
}

/// A resolved material record — the payload is the *actual* sim material type.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialRecord {
    pub id: String,
    pub material: Material,
    /// Inert metadata tags.
    pub tags: Vec<String>,
}

/// A resolved resource record.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceRecord {
    pub id: String,
    /// kg/m³ where meaningful; `None` for massless resources (e.g. charge).
    pub density: Option<f64>,
    /// Economy-facing flag; inert until the economy sibling (WI 552).
    pub tradable: bool,
    /// Inert metadata tags.
    pub tags: Vec<String>,
}

/// A resolved body-reference record: identity linkage into the world-building
/// body library — no physical body data (world-building owns that).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BodyRefRecord {
    pub id: String,
    /// [`crate::body_library`] slug of the referenced body asset.
    pub body_slug: String,
    /// Inert metadata tags.
    pub tags: Vec<String>,
}

/// A resolved surface-layer record (WI 892): the stack element bodies
/// reference by id from `surface_stack`. The embedded element's `id` equals
/// the record id (one namespace — the ladder's and the stack's addressing
/// key are the same string).
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceLayerRecord {
    pub id: String,
    /// The resolved element, ready to join a body's stack.
    pub layer: crate::body_asset::SurfaceLayer,
    /// Inert metadata tags.
    pub tags: Vec<String>,
}

/// A resolved body-recipe record (WI 881): the intrinsic body definition,
/// reassembled from the flattened recipe fields into a [`BodyAsset`] the sim
/// reads. Contrast [`BodyRefRecord`], which only links to a library slug.
///
/// A **shaped** record (WI 883/884) additionally retains its archetype `shape`
/// and its ladder-resolved parameter bands, so a consumer can re-sample the
/// family at arbitrary seeds (`bodygen::generate` reads the canonical archetype
/// records' bands this way); `body` is then the record's own-`surface_seed`
/// sample. Fixed records carry `None` for both.
#[derive(Debug, Clone, PartialEq)]
pub struct BodyRecipeRecord {
    pub id: String,
    /// The resolved intrinsic body.
    pub body: BodyAsset,
    /// The archetype shape, for sampled records (WI 883); `None` = fixed.
    pub shape: Option<Archetype>,
    /// The ladder-resolved bands a shaped record was sampled from (WI 884).
    pub(crate) bands: Option<ArchetypeBands>,
    /// Inert metadata tags.
    pub tags: Vec<String>,
}

/// A resolved, concrete content record.
#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    Device(DeviceRecord),
    Material(MaterialRecord),
    Resource(ResourceRecord),
    Body(BodyRefRecord),
    /// Boxed — a resolved body is much larger than the other records.
    BodyRecipe(Box<BodyRecipeRecord>),
    /// A resolved surface-layer element (WI 892).
    SurfaceLayer(SurfaceLayerRecord),
}

/// A catalog entry: the record, its defining pack, and per-field provenance
/// (each authored-or-overridden value's winning source + shadow chain;
/// inherited fields carry the provenance of the value they inherited).
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub record: Record,
    pub provenance: Provenance,
    pub field_provenance: BTreeMap<String, FieldProvenance>,
}

/// The resolved catalog of a merged composition: concrete records by id, in
/// deterministic (ordered) storage. Abstract bases are resolution inputs, not
/// content — they do not appear here.
///
/// `pack_id`/`pack_version` carry the single pack on the 547 single-pack
/// path; on a multi-pack merge they carry the **first base pack in resolution
/// order** (compat surface). `sources` lists every source id in ladder order.
#[derive(Debug, Clone, PartialEq)]
pub struct Catalog {
    pub pack_id: String,
    pub pack_version: String,
    /// The composed base packs as (id, version) pairs in resolution order —
    /// the save-recordable half of the catalog's identity (WI 553; the
    /// content design's "resolved-catalog identity"). `sources` names every
    /// ladder input; this names the versioned pack artifacts specifically.
    pub packs: Vec<(String, String)>,
    /// All source ids (settings, then packs, then override sets — the
    /// design's ladder numbering) in resolution order.
    pub sources: Vec<String>,
    /// The frozen balance scalars (name → factor + rationale, WI 549/550):
    /// resolved before any data phase, immutable, readable by later
    /// layers/telemetry. The records below already have them baked in — the
    /// sim never sees these.
    pub settings: BTreeMap<String, Setting>,
    records: BTreeMap<String, Entry>,
}

impl Catalog {
    /// Load a single pack from RON text (no overrides — the WI 547 path).
    pub fn from_ron_str(source: &str) -> Result<Catalog, ContentError> {
        Catalog::merge(&[source], &[])
    }

    /// Load a single pack from a RON file on disk.
    pub fn load(path: &Path) -> Result<Catalog, ContentError> {
        let text = std::fs::read_to_string(path).map_err(ContentError::Io)?;
        Catalog::from_ron_str(&text)
    }

    /// Merge a composition: pack documents (record definitions) plus override
    /// sets, resolved through the named-phase ladder. Input order of either
    /// slice is irrelevant — ordering is derived from declared phases,
    /// dependencies, and ids only.
    pub fn merge(pack_texts: &[&str], override_texts: &[&str]) -> Result<Catalog, ContentError> {
        merge_composition(pack_texts, &[], override_texts)
    }

    /// Merge a full composition: packs, **settings documents** (named balance
    /// scalars, WI 549), and override sets. Scalar values freeze before any
    /// data phase; their baked multiplies apply right after definitions,
    /// before every override phase (the design's ladder order). Input order
    /// of any slice is irrelevant.
    pub fn compose(
        pack_texts: &[&str],
        settings_texts: &[&str],
        override_texts: &[&str],
    ) -> Result<Catalog, ContentError> {
        merge_composition(pack_texts, settings_texts, override_texts)
    }

    /// Detector (WI 549): every body record's world-building library slug
    /// must be among `known_slugs` (the caller lists the library — the merge
    /// itself stays filesystem-free). Fails loudly naming record + slug.
    pub fn validate_body_refs(
        &self,
        known_slugs: &std::collections::BTreeSet<String>,
    ) -> Result<(), ContentError> {
        for (id, entry) in &self.records {
            if let Record::Body(b) = &entry.record {
                if !known_slugs.contains(&b.body_slug) {
                    return Err(ContentError::UnresolvedReference {
                        record: id.clone(),
                        reference: b.body_slug.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Look up a concrete record by id.
    pub fn get(&self, id: &str) -> Option<&Entry> {
        self.records.get(id)
    }

    /// All concrete record ids, in deterministic (sorted) order.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.records.keys().map(String::as_str)
    }

    /// Number of concrete records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the catalog holds no concrete records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Field access — one table drives ops, shadow capture, and provenance walks.
// ---------------------------------------------------------------------------

/// A mutable, typed view of one raw-record field.
enum Slot<'a> {
    Num(&'a mut Option<f64>),
    Flag(&'a mut Option<bool>),
    Text(&'a mut Option<String>),
    List(&'a mut Option<Vec<String>>),
}

impl Slot<'_> {
    fn value(&self) -> Option<ProvValue> {
        match self {
            Slot::Num(v) => v.map(ProvValue::Number),
            Slot::Flag(v) => v.map(ProvValue::Bool),
            Slot::Text(v) => (**v).clone().map(ProvValue::Text),
            Slot::List(v) => (**v).clone().map(ProvValue::List),
        }
    }
}

/// The overridable fields of a record, as `(name, slot)` accessors. One table
/// per kind; drives op application and the per-field provenance walk alike.
fn slot<'a>(rec: &'a mut RawRecord, field: &str) -> Option<Slot<'a>> {
    match rec {
        RawRecord::Device(d) => match field {
            "density" => Some(Slot::Num(&mut d.density)),
            "exhaust_velocity" => Some(Slot::Num(&mut d.exhaust_velocity)),
            "max_mass_flow" => Some(Slot::Num(&mut d.max_mass_flow)),
            "capacity" => Some(Slot::Num(&mut d.capacity)),
            "max_torque" => Some(Slot::Num(&mut d.max_torque)),
            "top_speed" => Some(Slot::Num(&mut d.top_speed)),
            "motor_mass" => Some(Slot::Num(&mut d.motor_mass)),
            "draw" => Some(Slot::Num(&mut d.draw)),
            "tags" => Some(Slot::List(&mut d.tags)),
            _ => None,
        },
        RawRecord::Material(m) => match field {
            "density" => Some(Slot::Num(&mut m.density)),
            "strength" => Some(Slot::Num(&mut m.strength)),
            "specific_heat" => Some(Slot::Num(&mut m.specific_heat)),
            "conductivity" => Some(Slot::Num(&mut m.conductivity)),
            "emissivity" => Some(Slot::Num(&mut m.emissivity)),
            "max_temp" => Some(Slot::Num(&mut m.max_temp)),
            "ablation_temp" => Some(Slot::Num(&mut m.ablation_temp)),
            "latent_heat" => Some(Slot::Num(&mut m.latent_heat)),
            "ablator_fraction" => Some(Slot::Num(&mut m.ablator_fraction)),
            "tags" => Some(Slot::List(&mut m.tags)),
            _ => None,
        },
        RawRecord::Resource(r) => match field {
            "density" => Some(Slot::Num(&mut r.density)),
            "tradable" => Some(Slot::Flag(&mut r.tradable)),
            "tags" => Some(Slot::List(&mut r.tags)),
            _ => None,
        },
        RawRecord::Body(b) => match field {
            "body" => Some(Slot::Text(&mut b.body)),
            "tags" => Some(Slot::List(&mut b.tags)),
            _ => None,
        },
        RawRecord::BodyRecipe(b) => match field {
            "name" => Some(Slot::Text(&mut b.name)),
            "mu" => Some(Slot::Num(&mut b.mu)),
            "radius" => Some(Slot::Num(&mut b.radius)),
            "rotation_period" => Some(Slot::Num(&mut b.rotation_period)),
            "atmosphere_surface_density" => Some(Slot::Num(&mut b.atmosphere_surface_density)),
            "atmosphere_surface_pressure" => Some(Slot::Num(&mut b.atmosphere_surface_pressure)),
            "atmosphere_scale_height" => Some(Slot::Num(&mut b.atmosphere_scale_height)),
            "ocean_surface_density" => Some(Slot::Num(&mut b.ocean_surface_density)),
            "ocean_surface_pressure" => Some(Slot::Num(&mut b.ocean_surface_pressure)),
            "ocean_density_gradient" => Some(Slot::Num(&mut b.ocean_density_gradient)),
            "gravity" => Some(Slot::Num(&mut b.gravity)),
            "atmosphere_temperature" => Some(Slot::Num(&mut b.atmosphere_temperature)),
            "ocean_temperature" => Some(Slot::Num(&mut b.ocean_temperature)),
            "surface_seed" => Some(Slot::Num(&mut b.surface_seed)),
            "surface_temperature_offset" => Some(Slot::Num(&mut b.surface_temperature_offset)),
            "surface_stack" => Some(Slot::List(&mut b.surface_stack)),
            "suppress" => Some(Slot::List(&mut b.suppress)),
            "nominal_insolation" => Some(Slot::Num(&mut b.nominal_insolation)),
            "bond_albedo" => Some(Slot::Num(&mut b.bond_albedo)),
            "greenhouse_delta_t" => Some(Slot::Num(&mut b.greenhouse_delta_t)),
            "mean_molar_mass" => Some(Slot::Num(&mut b.mean_molar_mass)),
            "radius_min" => Some(Slot::Num(&mut b.radius_min)),
            "radius_max" => Some(Slot::Num(&mut b.radius_max)),
            "gravity_min" => Some(Slot::Num(&mut b.gravity_min)),
            "gravity_max" => Some(Slot::Num(&mut b.gravity_max)),
            "rotation_period_min" => Some(Slot::Num(&mut b.rotation_period_min)),
            "rotation_period_max" => Some(Slot::Num(&mut b.rotation_period_max)),
            "atmosphere_surface_pressure_min" => {
                Some(Slot::Num(&mut b.atmosphere_surface_pressure_min))
            }
            "atmosphere_surface_pressure_max" => {
                Some(Slot::Num(&mut b.atmosphere_surface_pressure_max))
            }
            "nominal_insolation_min" => Some(Slot::Num(&mut b.nominal_insolation_min)),
            "nominal_insolation_max" => Some(Slot::Num(&mut b.nominal_insolation_max)),
            "bond_albedo_min" => Some(Slot::Num(&mut b.bond_albedo_min)),
            "bond_albedo_max" => Some(Slot::Num(&mut b.bond_albedo_max)),
            "greenhouse_delta_t_min" => Some(Slot::Num(&mut b.greenhouse_delta_t_min)),
            "greenhouse_delta_t_max" => Some(Slot::Num(&mut b.greenhouse_delta_t_max)),
            "mean_molar_mass_min" => Some(Slot::Num(&mut b.mean_molar_mass_min)),
            "mean_molar_mass_max" => Some(Slot::Num(&mut b.mean_molar_mass_max)),
            "ocean_surface_density_min" => Some(Slot::Num(&mut b.ocean_surface_density_min)),
            "ocean_surface_density_max" => Some(Slot::Num(&mut b.ocean_surface_density_max)),
            "ocean_temperature_min" => Some(Slot::Num(&mut b.ocean_temperature_min)),
            "ocean_temperature_max" => Some(Slot::Num(&mut b.ocean_temperature_max)),
            "tags" => Some(Slot::List(&mut b.tags)),
            _ => None,
        },
        RawRecord::SurfaceLayer(l) => match field {
            "enabled" => Some(Slot::Flag(&mut l.enabled)),
            "density" => Some(Slot::Num(&mut l.density)),
            "depth" => Some(Slot::Num(&mut l.depth)),
            "temperature" => Some(Slot::Num(&mut l.temperature)),
            "moisture" => Some(Slot::Num(&mut l.moisture)),
            "moisture_scale" => Some(Slot::Num(&mut l.moisture_scale)),
            "tags" => Some(Slot::List(&mut l.tags)),
            _ => None,
        },
    }
}

/// The overridable field names of a kind (drives the provenance walk).
fn field_names(kind: RecordKind) -> &'static [&'static str] {
    match kind {
        RecordKind::Device => &[
            "density",
            "exhaust_velocity",
            "max_mass_flow",
            "capacity",
            "max_torque",
            "top_speed",
            "motor_mass",
            "draw",
            "tags",
        ],
        RecordKind::Material => &[
            "density",
            "strength",
            "specific_heat",
            "conductivity",
            "emissivity",
            "max_temp",
            "ablation_temp",
            "latent_heat",
            "ablator_fraction",
            "tags",
        ],
        RecordKind::Resource => &["density", "tradable", "tags"],
        RecordKind::Body => &["body", "tags"],
        RecordKind::BodyRecipe => &[
            "name",
            "mu",
            "radius",
            "rotation_period",
            "atmosphere_surface_density",
            "atmosphere_surface_pressure",
            "atmosphere_scale_height",
            "ocean_surface_density",
            "ocean_surface_pressure",
            "ocean_density_gradient",
            "gravity",
            "atmosphere_temperature",
            "ocean_temperature",
            "surface_seed",
            "surface_temperature_offset",
            "nominal_insolation",
            "bond_albedo",
            "greenhouse_delta_t",
            "mean_molar_mass",
            "radius_min",
            "radius_max",
            "gravity_min",
            "gravity_max",
            "rotation_period_min",
            "rotation_period_max",
            "atmosphere_surface_pressure_min",
            "atmosphere_surface_pressure_max",
            "nominal_insolation_min",
            "nominal_insolation_max",
            "bond_albedo_min",
            "bond_albedo_max",
            "greenhouse_delta_t_min",
            "greenhouse_delta_t_max",
            "mean_molar_mass_min",
            "mean_molar_mass_max",
            "ocean_surface_density_min",
            "ocean_surface_density_max",
            "ocean_temperature_min",
            "ocean_temperature_max",
            "surface_stack",
            "suppress",
            "tags",
        ],
        RecordKind::SurfaceLayer => &[
            "enabled",
            "density",
            "depth",
            "temperature",
            "moisture",
            "moisture_scale",
            "tags",
        ],
    }
}

/// Read a field's current raw value without mutating (provenance walk).
fn peek(rec: &RawRecord, field: &str) -> Option<Option<ProvValue>> {
    // Safe: `slot` only hands out references; we clone the value and drop it.
    let mut clone = rec.clone();
    slot(&mut clone, field).map(|s| s.value())
}

// ---------------------------------------------------------------------------
// The merge ladder (WI 548).
// ---------------------------------------------------------------------------

/// Order a set of sources topologically by their declared dependencies,
/// smallest-id-first among ready nodes (deterministic Kahn).
fn topo_order(
    deps: &BTreeMap<String, Vec<String>>, // source id -> depends-on ids
) -> Result<Vec<String>, ContentError> {
    for (source, ds) in deps {
        for d in ds {
            if !deps.contains_key(d) {
                return Err(ContentError::UnknownDependency {
                    source: source.clone(),
                    depends_on: d.clone(),
                });
            }
        }
    }
    let mut remaining: BTreeMap<String, HashSet<String>> = deps
        .iter()
        .map(|(id, ds)| (id.clone(), ds.iter().cloned().collect()))
        .collect();
    let mut order = Vec::with_capacity(deps.len());
    while !remaining.is_empty() {
        // Smallest id whose dependencies are all placed.
        let ready = remaining
            .iter()
            .find(|(_, ds)| ds.is_empty())
            .map(|(id, _)| id.clone());
        let Some(id) = ready else {
            // Every remaining node waits on another: a cycle exists. Name the
            // smallest still-blocked source for determinism (it is either a
            // cycle member or depends on one).
            let source = remaining.keys().next().expect("non-empty").clone();
            return Err(ContentError::DependencyCycle { source });
        };
        remaining.remove(&id);
        for ds in remaining.values_mut() {
            ds.remove(&id);
        }
        order.push(id);
    }
    Ok(order)
}

/// Does `id`'s inheritance chain reach `base`? (Existence of all parents has
/// been validated; a visited set guards against cycles, which error later.)
fn chain_reaches(id: &str, base: &str, raws: &BTreeMap<String, RawRecord>) -> bool {
    let mut seen = HashSet::new();
    let mut cur = id;
    while let Some(parent) = raws.get(cur).and_then(|r| r.parent()) {
        if !seen.insert(parent.to_string()) {
            return false; // cycle — reported by inheritance resolution
        }
        if parent == base {
            return true;
        }
        cur = parent;
    }
    false
}

/// Expand an override/scalar target to concrete record ids, deterministically
/// (sorted-id order — `raws` is an ordered map). Shared by the settings bake
/// and the override ladder so selector semantics cannot drift apart.
fn expand_target(
    target: &Target,
    raws: &BTreeMap<String, RawRecord>,
) -> Result<Vec<String>, ContentError> {
    Ok(match target {
        Target::Id(id) => {
            if !raws.contains_key(id) {
                return Err(ContentError::UnknownTarget {
                    target: target.describe(),
                });
            }
            vec![id.clone()]
        }
        Target::Kind(kind) => raws
            .iter()
            .filter(|(_, r)| r.kind() == *kind)
            .map(|(id, _)| id.clone())
            .collect(),
        Target::Base(base) => {
            if !raws.contains_key(base) {
                return Err(ContentError::UnknownTarget {
                    target: target.describe(),
                });
            }
            raws.keys()
                .filter(|id| chain_reaches(id, base, raws))
                .cloned()
                .collect()
        }
    })
}

/// Apply one op to one record's field, updating the provenance ledger.
fn apply_op(
    rec: &mut RawRecord,
    field: &str,
    op: &Op,
    source: &SourceRef,
    defining_pack: &str,
    prov: &mut HashMap<(String, String), FieldProvenance>,
) -> Result<(), ContentError> {
    let record_id = rec.id().to_string();
    if STRUCTURAL_FIELDS.contains(&field) {
        return Err(ContentError::StructuralField {
            record: record_id,
            field: field.to_string(),
        });
    }
    // WI 880: the `suppress` list composes only through the dedicated op —
    // its `field` names the *drawn field to suppress*, and the mark lands in
    // the record's suppress list with its own provenance key. Generic ops
    // aimed at the list itself are rejected below (design I4: the marker is
    // never an ordinary list edit, so it cannot be conflated with `delete`).
    if let Op::Suppress = op {
        let RawRecord::BodyRecipe(b) = rec else {
            return Err(ContentError::UnknownField {
                record: record_id,
                field: field.to_string(),
            });
        };
        if !bodygen::SUPPRESSIBLE_FIELDS.contains(&field) {
            return Err(ContentError::NotSuppressible {
                record: record_id,
                field: field.to_string(),
            });
        }
        let old = b
            .suppress
            .clone()
            .map(ProvValue::List)
            .unwrap_or(ProvValue::Unset);
        let list = b.suppress.get_or_insert_with(Vec::new);
        if !list.iter().any(|e| e == field) {
            list.push(field.to_string());
        }
        // Same ledger discipline as every op, keyed under `suppress` (the
        // list the mark lives in), so the marker's chain is queryable and
        // distinct from any drawn field's value provenance.
        let key = (record_id, "suppress".to_string());
        let prior_source = prov
            .get(&key)
            .map(|p| p.source.clone())
            .unwrap_or(SourceRef::Pack {
                id: defining_pack.to_string(),
            });
        let entry = prov.entry(key).or_insert_with(|| FieldProvenance {
            source: prior_source.clone(),
            shadows: Vec::new(),
        });
        entry.shadows.insert(
            0,
            Shadow {
                value: old,
                source: prior_source,
            },
        );
        entry.source = source.clone();
        return Ok(());
    }
    if field == "suppress" {
        let op_name = match op {
            Op::Set(_) => "set",
            Op::Multiply(_) => "multiply",
            Op::Extend(_) => "extend",
            Op::Delete(_) => "delete",
            Op::InsertAfter { .. } => "insert_after",
            Op::Replace { .. } => "replace",
            Op::Suppress => "suppress",
        };
        return Err(ContentError::TypeMismatch {
            record: record_id,
            field: field.to_string(),
            op: op_name,
        });
    }
    let parentless = rec.parent().is_none();
    let Some(mut s) = slot(rec, field) else {
        return Err(ContentError::UnknownField {
            record: record_id,
            field: field.to_string(),
        });
    };
    let old = s.value();
    let mismatch = |op: &'static str| ContentError::TypeMismatch {
        record: record_id.clone(),
        field: field.to_string(),
        op,
    };
    let unset = || ContentError::UnsetField {
        record: record_id.clone(),
        field: field.to_string(),
    };
    match (&mut s, op) {
        (Slot::Num(v), Op::Set(SetValue::Number(n))) => **v = Some(*n),
        (Slot::Flag(v), Op::Set(SetValue::Bool(b))) => **v = Some(*b),
        (Slot::Text(v), Op::Set(SetValue::Text(t))) => **v = Some(t.clone()),
        (Slot::List(v), Op::Set(SetValue::List(l))) => **v = Some(l.clone()),
        (_, Op::Set(_)) => return Err(mismatch("set")),
        (Slot::Num(v), Op::Multiply(factor)) => match **v {
            Some(cur) => **v = Some(cur * factor),
            None => return Err(unset()),
        },
        (_, Op::Multiply(_)) => return Err(mismatch("multiply")),
        (Slot::List(v), Op::Extend(items)) => match v {
            Some(list) => list.extend(items.iter().cloned()),
            None if parentless => **v = Some(items.clone()),
            None => return Err(unset()),
        },
        (_, Op::Extend(_)) => return Err(mismatch("extend")),
        (Slot::List(v), Op::Delete(item)) => {
            let list = match v {
                Some(list) => list,
                None if parentless => {
                    return Err(ContentError::AbsentElement {
                        record: record_id,
                        field: field.to_string(),
                        element: item.clone(),
                    })
                }
                None => return Err(unset()),
            };
            match list.iter().position(|e| e == item) {
                Some(i) => {
                    list.remove(i);
                }
                None => {
                    return Err(ContentError::AbsentElement {
                        record: record_id,
                        field: field.to_string(),
                        element: item.clone(),
                    })
                }
            }
        }
        (_, Op::Delete(_)) => return Err(mismatch("delete")),
        (Slot::List(v), Op::InsertAfter { anchor, items }) => {
            let list = match v {
                Some(list) => list,
                None if parentless => {
                    return Err(ContentError::AbsentElement {
                        record: record_id,
                        field: field.to_string(),
                        element: anchor.clone(),
                    })
                }
                None => return Err(unset()),
            };
            match list.iter().position(|e| e == anchor) {
                Some(i) => {
                    for (k, item) in items.iter().enumerate() {
                        list.insert(i + 1 + k, item.clone());
                    }
                }
                None => {
                    return Err(ContentError::AbsentElement {
                        record: record_id,
                        field: field.to_string(),
                        element: anchor.clone(),
                    })
                }
            }
        }
        (_, Op::InsertAfter { .. }) => return Err(mismatch("insert_after")),
        (Slot::List(v), Op::Replace { target, items }) => {
            let list = match v {
                Some(list) => list,
                None if parentless => {
                    return Err(ContentError::AbsentElement {
                        record: record_id,
                        field: field.to_string(),
                        element: target.clone(),
                    })
                }
                None => return Err(unset()),
            };
            match list.iter().position(|e| e == target) {
                Some(i) => {
                    list.remove(i);
                    for (k, item) in items.iter().enumerate() {
                        list.insert(i + k, item.clone());
                    }
                }
                None => {
                    return Err(ContentError::AbsentElement {
                        record: record_id,
                        field: field.to_string(),
                        element: target.clone(),
                    })
                }
            }
        }
        (_, Op::Replace { .. }) => return Err(mismatch("replace")),
        // Handled by the early arm above (never reaches the slot machinery).
        (_, Op::Suppress) => unreachable!("suppress is applied before slot dispatch"),
    }
    // Provenance: the displaced value's source is whoever last wrote it (or
    // the defining pack for an authored/unset original).
    let key = (record_id, field.to_string());
    let prior_source = prov
        .get(&key)
        .map(|p| p.source.clone())
        .unwrap_or(SourceRef::Pack {
            id: defining_pack.to_string(),
        });
    let entry = prov.entry(key).or_insert_with(|| FieldProvenance {
        source: prior_source.clone(),
        shadows: Vec::new(),
    });
    entry.shadows.insert(
        0,
        Shadow {
            value: old.unwrap_or(ProvValue::Unset),
            source: prior_source,
        },
    );
    entry.source = source.clone();
    Ok(())
}

/// The full merge pipeline: parse → freeze settings → order sources →
/// collect definitions → bake settings scalars → apply the override ladder
/// (on raw records) → resolve inheritance → validate → build per-field
/// provenance.
fn merge_composition(
    pack_texts: &[&str],
    settings_texts: &[&str],
    override_texts: &[&str],
) -> Result<Catalog, ContentError> {
    // Parse + format-check every document. Packs get a **header-first**
    // format probe (WI 889): the typed `RawPack` parse is `deny_unknown_fields`
    // over the record grammar, so a stale-format pack authoring since-retired
    // fields would otherwise die as an incidental unknown-field parse error
    // without ever reaching the version gate — for exactly the document class
    // the version exists to refuse. The lenient probe reads only `format`
    // (the `persist` two-stage-probe pattern), so refusal is by version,
    // loud and named, regardless of what fields the document authors.
    #[derive(Deserialize)]
    struct FormatProbe {
        format: u32,
    }
    let mut packs = Vec::with_capacity(pack_texts.len());
    for text in pack_texts {
        let probe: FormatProbe =
            ron::from_str(text).map_err(|e| ContentError::Parse(e.to_string()))?;
        if probe.format != CONTENT_FORMAT_VERSION {
            return Err(ContentError::Format {
                found: probe.format,
            });
        }
        let p: RawPack = ron::from_str(text).map_err(|e| ContentError::Parse(e.to_string()))?;
        // The typed gate stays authoritative (the probe read the same bytes,
        // so this can only agree — it exists so the typed field is the one
        // the contract hangs on).
        if p.format != CONTENT_FORMAT_VERSION {
            return Err(ContentError::Format { found: p.format });
        }
        packs.push(p);
    }
    let mut settings_docs = Vec::with_capacity(settings_texts.len());
    for text in settings_texts {
        let s: RawSettings = ron::from_str(text).map_err(|e| ContentError::Parse(e.to_string()))?;
        if s.format != CONTENT_FORMAT_VERSION {
            return Err(ContentError::Format { found: s.format });
        }
        settings_docs.push(s);
    }
    let mut sets = Vec::with_capacity(override_texts.len());
    for text in override_texts {
        let s: RawOverrideSet =
            ron::from_str(text).map_err(|e| ContentError::Parse(e.to_string()))?;
        if s.format != CONTENT_FORMAT_VERSION {
            return Err(ContentError::Format { found: s.format });
        }
        sets.push(s);
    }

    // One source-id namespace across settings, packs, and override sets.
    let mut source_ids = HashSet::new();
    for id in settings_docs
        .iter()
        .map(|s| &s.id)
        .chain(packs.iter().map(|p| &p.id))
        .chain(sets.iter().map(|s| &s.id))
    {
        if !source_ids.insert(id.clone()) {
            return Err(ContentError::DuplicateSource { id: id.clone() });
        }
    }

    // Ladder step 1 — settings: resolve and FREEZE scalar values before any
    // data phase (documents in id order; duplicate names are an error — one
    // frozen value per name).
    settings_docs.sort_by(|a, b| a.id.cmp(&b.id));
    let mut settings: BTreeMap<String, Setting> = BTreeMap::new();
    for doc in &settings_docs {
        for scalar in &doc.scalars {
            // WI 550: a factor must be finite and strictly positive — ×0 and
            // negatives are nonsense in a multiply-only grammar (extreme
            // values belong in packs, as deliberate physical truths).
            if !(scalar.factor.is_finite() && scalar.factor > 0.0) {
                return Err(ContentError::InvalidScalarFactor {
                    name: scalar.name.clone(),
                    factor: scalar.factor,
                });
            }
            if settings
                .insert(
                    scalar.name.clone(),
                    Setting {
                        factor: scalar.factor,
                        rationale: scalar.rationale.clone(),
                    },
                )
                .is_some()
            {
                return Err(ContentError::DuplicateScalar {
                    name: scalar.name.clone(),
                });
            }
        }
    }

    // Phase: base packs. Order by declared dependencies, then id.
    let pack_deps: BTreeMap<String, Vec<String>> = packs
        .iter()
        .map(|p| (p.id.clone(), p.depends.clone()))
        .collect();
    let pack_order = topo_order(&pack_deps)?;
    let mut packs_by_id: HashMap<String, RawPack> =
        packs.into_iter().map(|p| (p.id.clone(), p)).collect();

    // Collect record definitions (defined once, anywhere).
    let mut raws: BTreeMap<String, RawRecord> = BTreeMap::new();
    let mut defining: HashMap<String, Provenance> = HashMap::new();
    let mut first_pack: Option<(String, String)> = None;
    let mut pack_identities: Vec<(String, String)> = Vec::with_capacity(pack_order.len());
    for pack_id in &pack_order {
        let pack = packs_by_id.remove(pack_id).expect("ordered from this set");
        if first_pack.is_none() {
            first_pack = Some((pack.id.clone(), pack.version.clone()));
        }
        pack_identities.push((pack.id.clone(), pack.version.clone()));
        for r in pack.records {
            let id = r.id().to_string();
            if raws.insert(id.clone(), r).is_some() {
                return Err(ContentError::DuplicateId { id });
            }
            defining.insert(
                id,
                Provenance {
                    pack_id: pack.id.clone(),
                    pack_version: pack.version.clone(),
                },
            );
        }
    }

    // Parent existence is needed before base-selector expansion.
    for (id, r) in &raws {
        if let Some(parent) = r.parent() {
            if !raws.contains_key(parent) {
                return Err(ContentError::UnknownParent {
                    child: id.clone(),
                    parent: parent.to_string(),
                });
            }
        }
    }

    // Settings BAKE (WI 549): the frozen scalars multiply physical content as
    // the first writes after definitions — before every override phase — on
    // raw records, so a base binding re-flows like any 548 override. The
    // grammar admits only multiplication; the residual seam case (a scalar
    // multiplying a field no content record defines = originating) is
    // rejected here with the scalar's name attached.
    let mut prov: HashMap<(String, String), FieldProvenance> = HashMap::new();
    for doc in &settings_docs {
        for scalar in &doc.scalars {
            let source = SourceRef::Setting {
                source: doc.id.clone(),
                scalar: scalar.name.clone(),
            };
            let op = Op::Multiply(scalar.factor);
            for id in expand_target(&scalar.target, &raws)? {
                let defining_pack = defining[&id].pack_id.clone();
                let rec = raws.get_mut(&id).expect("expanded from this map");
                apply_op(rec, &scalar.field, &op, &source, &defining_pack, &mut prov).map_err(
                    |e| match e {
                        ContentError::UnsetField { record, field } => ContentError::SeamViolation {
                            scalar: scalar.name.clone(),
                            record,
                            field,
                        },
                        other => other,
                    },
                )?;
            }
        }
    }

    // Phases: patches → scenario → local. Within a phase: topo by declared
    // same-phase depends, then id.
    let mut set_order: Vec<usize> = Vec::with_capacity(sets.len());
    for phase in [
        OverridePhase::Patch,
        OverridePhase::Scenario,
        OverridePhase::Local,
    ] {
        let phase_deps: BTreeMap<String, Vec<String>> = sets
            .iter()
            .filter(|s| s.phase == phase)
            .map(|s| (s.id.clone(), s.depends.clone()))
            .collect();
        for id in topo_order(&phase_deps)? {
            let idx = sets
                .iter()
                .position(|s| s.id == id)
                .expect("ordered from this set");
            set_order.push(idx);
        }
    }
    // Every set has one of the three phases, so every set was ordered.
    debug_assert_eq!(set_order.len(), sets.len());

    // Apply the override ladder on raw (pre-inheritance) records.
    for &idx in &set_order {
        let set = &sets[idx];
        let source = SourceRef::Override {
            source: set.id.clone(),
            phase: set.phase,
        };
        for ov in &set.overrides {
            for id in expand_target(&ov.target, &raws)? {
                let defining_pack = defining[&id].pack_id.clone();
                let rec = raws.get_mut(&id).expect("expanded from this map");
                apply_op(rec, &ov.field, &ov.op, &source, &defining_pack, &mut prov)?;
            }
        }
    }

    // Resolve inheritance (after the ladder — base writes re-flow), validate.
    let mut merged: HashMap<String, RawRecord> = HashMap::new();
    for id in raws.keys() {
        let mut visiting = HashSet::new();
        merge_chain(id, &raws, &mut merged, &mut visiting)?;
    }

    let mut records = BTreeMap::new();
    for id in raws.keys() {
        let m = &merged[id];
        if m.is_abstract() {
            continue; // bases are inheritance targets, not content
        }
        let record = validate(m, &merged)?;
        // Per-field provenance: each present field traces to the chain member
        // whose (post-ladder) raw value supplied it.
        let mut field_provenance = BTreeMap::new();
        for &field in field_names(m.kind()) {
            let mut cur = id.as_str();
            loop {
                let raw = &raws[cur];
                match peek(raw, field) {
                    Some(Some(_)) => {
                        let fp = prov
                            .get(&(cur.to_string(), field.to_string()))
                            .cloned()
                            .unwrap_or_else(|| FieldProvenance {
                                source: SourceRef::Pack {
                                    id: defining[cur].pack_id.clone(),
                                },
                                shadows: Vec::new(),
                            });
                        field_provenance.insert(field.to_string(), fp);
                        break;
                    }
                    _ => match raw.parent() {
                        Some(p) => cur = p,
                        None => break, // unset everywhere (optional field)
                    },
                }
            }
        }
        records.insert(
            id.clone(),
            Entry {
                record,
                provenance: defining[id].clone(),
                field_provenance,
            },
        );
    }

    let (pack_id, pack_version) = first_pack.unwrap_or_default();
    // Ladder numbering: settings, then packs, then override sets.
    let mut sources: Vec<String> = settings_docs.iter().map(|s| s.id.clone()).collect();
    sources.extend(pack_order);
    sources.extend(set_order.iter().map(|&i| sets[i].id.clone()));
    Ok(Catalog {
        pack_id,
        pack_version,
        packs: pack_identities,
        sources,
        settings,
        records,
    })
}

// ---------------------------------------------------------------------------
// Inheritance resolution + validation (WI 547).
// ---------------------------------------------------------------------------

/// Compute the fully-merged raw form of `id` (its fields with every ancestor
/// folded in), memoizing results and detecting cycles/unknown parents.
fn merge_chain(
    id: &str,
    ordered: &BTreeMap<String, RawRecord>,
    merged: &mut HashMap<String, RawRecord>,
    visiting: &mut HashSet<String>,
) -> Result<(), ContentError> {
    if merged.contains_key(id) {
        return Ok(());
    }
    if !visiting.insert(id.to_string()) {
        return Err(ContentError::Cycle { id: id.to_string() });
    }
    let record = ordered
        .get(id)
        .expect("merge_chain called only with known ids")
        .clone();
    let result = match record.parent() {
        None => record,
        Some(parent_id) => {
            let parent_id = parent_id.to_string();
            if !ordered.contains_key(&parent_id) {
                return Err(ContentError::UnknownParent {
                    child: id.to_string(),
                    parent: parent_id,
                });
            }
            merge_chain(&parent_id, ordered, merged, visiting)?;
            record.merge_over(&merged[&parent_id])?
        }
    };
    visiting.remove(id);
    merged.insert(id.to_string(), result);
    Ok(())
}

/// Validate a fully-merged concrete record into its resolved, fully-typed form.
/// Resolves one raw surface-layer record into its stack element (WI 892):
/// `layer_type` is required and must be a known slug; off-type params are
/// loud (`InapplicableField`); params JSON carries exactly the authored
/// keys (none authored ⇒ `Null`, the lenient readers' defaults spelling).
fn resolve_surface_layer(
    l: &RawSurfaceLayer,
) -> Result<crate::body_asset::SurfaceLayer, ContentError> {
    use crate::body_asset::SurfaceLayerType;
    let slug = l.layer_type.clone().ok_or(ContentError::MissingField {
        id: l.id.clone(),
        field: "layer_type",
    })?;
    let layer_type =
        SurfaceLayerType::from_slug(&slug).ok_or_else(|| ContentError::UnknownLayerType {
            id: l.id.clone(),
            layer_type: slug,
        })?;
    // Off-type params are loud, per-key (the WI 884 mode-exclusivity posture).
    let keys: [(&'static str, Option<f64>, SurfaceLayerType); 5] = [
        ("density", l.density, SurfaceLayerType::Crater),
        ("depth", l.depth, SurfaceLayerType::Crater),
        ("temperature", l.temperature, SurfaceLayerType::Material),
        ("moisture", l.moisture, SurfaceLayerType::Material),
        (
            "moisture_scale",
            l.moisture_scale,
            SurfaceLayerType::Material,
        ),
    ];
    let mut params = serde_json::Map::new();
    for (field, value, applies_to) in keys {
        if let Some(v) = value {
            if layer_type != applies_to {
                return Err(ContentError::InapplicableField {
                    id: l.id.clone(),
                    field,
                });
            }
            params.insert(field.to_string(), serde_json::json!(v));
        }
    }
    Ok(crate::body_asset::SurfaceLayer {
        id: l.id.clone(),
        layer_type,
        enabled: l.enabled.unwrap_or(true),
        params: if params.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::Value::Object(params)
        },
    })
}

fn validate(m: &RawRecord, merged: &HashMap<String, RawRecord>) -> Result<Record, ContentError> {
    let missing = |id: &str, field: &'static str| ContentError::MissingField {
        id: id.to_string(),
        field,
    };
    match m {
        RawRecord::Device(d) => {
            let class = d.class.ok_or_else(|| missing(&d.id, "class"))?;
            let density = d.density.ok_or_else(|| missing(&d.id, "density"))?;
            let spec = device_spec(d, class)?;
            Ok(Record::Device(DeviceRecord {
                id: d.id.clone(),
                density,
                spec,
                tags: d.tags.clone().unwrap_or_default(),
            }))
        }
        RawRecord::Material(mt) => {
            let material = Material {
                density: mt.density.ok_or_else(|| missing(&mt.id, "density"))?,
                strength: mt.strength.ok_or_else(|| missing(&mt.id, "strength"))?,
                thermal: Thermal {
                    specific_heat: mt
                        .specific_heat
                        .ok_or_else(|| missing(&mt.id, "specific_heat"))?,
                    conductivity: mt
                        .conductivity
                        .ok_or_else(|| missing(&mt.id, "conductivity"))?,
                    emissivity: mt.emissivity.ok_or_else(|| missing(&mt.id, "emissivity"))?,
                    max_temp: mt.max_temp.ok_or_else(|| missing(&mt.id, "max_temp"))?,
                    ablation_temp: mt.ablation_temp.unwrap_or(0.0),
                    latent_heat: mt.latent_heat.unwrap_or(0.0),
                    ablator_fraction: mt.ablator_fraction.unwrap_or(0.0),
                },
            };
            Ok(Record::Material(MaterialRecord {
                id: mt.id.clone(),
                material,
                tags: mt.tags.clone().unwrap_or_default(),
            }))
        }
        RawRecord::Resource(r) => Ok(Record::Resource(ResourceRecord {
            id: r.id.clone(),
            density: r.density,
            tradable: r.tradable.unwrap_or(false),
            tags: r.tags.clone().unwrap_or_default(),
        })),
        RawRecord::Body(b) => Ok(Record::Body(BodyRefRecord {
            id: b.id.clone(),
            body_slug: b.body.clone().ok_or_else(|| missing(&b.id, "body"))?,
            tags: b.tags.clone().unwrap_or_default(),
        })),
        RawRecord::SurfaceLayer(l) => Ok(Record::SurfaceLayer(SurfaceLayerRecord {
            id: l.id.clone(),
            layer: resolve_surface_layer(l)?,
            tags: l.tags.clone().unwrap_or_default(),
        })),
        RawRecord::BodyRecipe(b) => {
            let req = |v: Option<f64>, field: &'static str| v.ok_or_else(|| missing(&b.id, field));
            // WI 891 (parked decision (b)): the authored seed is a Num slot
            // cast to u64 — validate the cast instead of letting a negative
            // saturate, a fraction truncate, or a value above 2^53 silently
            // lose integer precision. Full-width u64 seeds travel only through
            // persisted refs / `bodygen::generate`, never through this slot.
            let surface_seed = || -> Result<u64, ContentError> {
                let raw = req(b.surface_seed, "surface_seed")?;
                let valid =
                    raw.is_finite() && raw >= 0.0 && raw.fract() == 0.0 && raw <= MAX_AUTHORED_SEED;
                if !valid {
                    return Err(ContentError::UnphysicalValue {
                        id: b.id.clone(),
                        field: "surface_seed",
                    });
                }
                Ok(raw as u64)
            };
            // WI 884: a recipe is either **sampled** (`shape` + band fields) or
            // **fixed** (scalar fields) — authoring the other mode's fields is a
            // loud error, mirroring `device_spec`'s class-dependent rejection
            // (previously the off-mode fields were silently ignored). Inheritance
            // cannot mix the modes indirectly: `shape` inherits together with the
            // bands, so a child of a shaped parent is itself shaped.
            let inapplicable = |field: &'static str| ContentError::InapplicableField {
                id: b.id.clone(),
                field,
            };
            // WI 880: the resolved suppress set (BTreeSet — deterministic
            // iteration, duplicates idempotent). Only shaped recipes have a
            // generator to veto; the fixed path is pin-or-derive-or-loud
            // (WI 886), so any suppression there is inapplicable.
            let suppressed: BTreeSet<&str> =
                b.suppress.iter().flatten().map(String::as_str).collect();
            if b.shape.is_none() && !suppressed.is_empty() {
                return Err(inapplicable("suppress"));
            }
            let fixed_scalars = [
                (b.mu.is_some(), "mu"),
                (b.radius.is_some(), "radius"),
                (b.rotation_period.is_some(), "rotation_period"),
                (
                    b.atmosphere_surface_density.is_some(),
                    "atmosphere_surface_density",
                ),
                (
                    b.atmosphere_surface_pressure.is_some(),
                    "atmosphere_surface_pressure",
                ),
                (
                    b.atmosphere_scale_height.is_some(),
                    "atmosphere_scale_height",
                ),
                (b.ocean_surface_density.is_some(), "ocean_surface_density"),
                (b.ocean_surface_pressure.is_some(), "ocean_surface_pressure"),
                (b.ocean_density_gradient.is_some(), "ocean_density_gradient"),
                (b.gravity.is_some(), "gravity"),
                (b.atmosphere_temperature.is_some(), "atmosphere_temperature"),
                (b.ocean_temperature.is_some(), "ocean_temperature"),
                // The scalar derivation inputs (WI 886) are fixed-mode-only
                // too: a shaped recipe authors the same independents as
                // `_min`/`_max` BANDS (WI 889) and both modes feed the same
                // `body_derive` relations.
                (b.nominal_insolation.is_some(), "nominal_insolation"),
                (b.bond_albedo.is_some(), "bond_albedo"),
                (b.greenhouse_delta_t.is_some(), "greenhouse_delta_t"),
                (b.mean_molar_mass.is_some(), "mean_molar_mass"),
            ];
            let band_fields = [
                (b.radius_min.is_some(), "radius_min"),
                (b.radius_max.is_some(), "radius_max"),
                (b.gravity_min.is_some(), "gravity_min"),
                (b.gravity_max.is_some(), "gravity_max"),
                (b.rotation_period_min.is_some(), "rotation_period_min"),
                (b.rotation_period_max.is_some(), "rotation_period_max"),
                (
                    b.atmosphere_surface_pressure_min.is_some(),
                    "atmosphere_surface_pressure_min",
                ),
                (
                    b.atmosphere_surface_pressure_max.is_some(),
                    "atmosphere_surface_pressure_max",
                ),
                (b.nominal_insolation_min.is_some(), "nominal_insolation_min"),
                (b.nominal_insolation_max.is_some(), "nominal_insolation_max"),
                (b.bond_albedo_min.is_some(), "bond_albedo_min"),
                (b.bond_albedo_max.is_some(), "bond_albedo_max"),
                (b.greenhouse_delta_t_min.is_some(), "greenhouse_delta_t_min"),
                (b.greenhouse_delta_t_max.is_some(), "greenhouse_delta_t_max"),
                (b.mean_molar_mass_min.is_some(), "mean_molar_mass_min"),
                (b.mean_molar_mass_max.is_some(), "mean_molar_mass_max"),
                (
                    b.ocean_surface_density_min.is_some(),
                    "ocean_surface_density_min",
                ),
                (
                    b.ocean_surface_density_max.is_some(),
                    "ocean_surface_density_max",
                ),
                (b.ocean_temperature_min.is_some(), "ocean_temperature_min"),
                (b.ocean_temperature_max.is_some(), "ocean_temperature_max"),
            ];
            let off_mode = if b.shape.is_some() {
                &fixed_scalars[..]
            } else {
                &band_fields[..]
            };
            // A suppressed field's same-name scalar is the sanctioned explicit
            // source (WI 880) — the mode wall admits exactly those names; every
            // other off-mode field stays as loud as it was (WI 884).
            if let Some((_, field)) = off_mode
                .iter()
                .find(|(present, name)| *present && !suppressed.contains(name))
            {
                return Err(inapplicable(field));
            }
            // The surface-layer stack (WI 892). The stack starts from the
            // recipe's resolved `surface_stack` (Phase B — empty until then),
            // and the `surface_temperature_offset` sugar composes onto it per
            // the pinned rule (see `apply_temperature_sugar`): absent offset +
            // empty stack ⇒ no layers, matching a body that reads its medium
            // directly (WI 875); a bare offset ⇒ one synthesized material
            // layer carrying the JSON the biome classifier reads.
            let mut surface_layers: Vec<crate::body_asset::SurfaceLayer> = Vec::new();
            if let Some(stack) = &b.surface_stack {
                let mut seen: BTreeSet<&str> = BTreeSet::new();
                for layer_id in stack {
                    if !seen.insert(layer_id) {
                        return Err(ContentError::DuplicateStackLayer {
                            body: b.id.clone(),
                            layer: layer_id.clone(),
                        });
                    }
                    match merged.get(layer_id) {
                        Some(RawRecord::SurfaceLayer(l)) if !l.is_abstract => {
                            surface_layers.push(resolve_surface_layer(l)?);
                        }
                        // An abstract layer is an inheritance target, not
                        // content — same refusal as an absent record.
                        Some(RawRecord::SurfaceLayer(_)) | None => {
                            return Err(ContentError::UnknownStackLayer {
                                body: b.id.clone(),
                                layer: layer_id.clone(),
                            });
                        }
                        Some(_) => {
                            return Err(ContentError::StackLayerWrongKind {
                                body: b.id.clone(),
                                layer: layer_id.clone(),
                            });
                        }
                    }
                }
            }
            apply_temperature_sugar(&mut surface_layers, b.surface_temperature_offset);
            let (body, bands) = match b.shape {
                // Sampled (WI 883): an archetype. Draw the shape's bands at the
                // seed via the shared sampler, reproducing `bodygen::generate`.
                // The scalar fields are unused here; identity + classifier offset
                // come from the recipe, everything else from the seeded draw.
                Some(shape) => {
                    let unphysical = |field: &'static str| ContentError::UnphysicalValue {
                        id: b.id.clone(),
                        field,
                    };
                    // A used band requires both bounds; every used band must be
                    // finite and ordered (min ≤ max), and the drawn-independent
                    // bands respect their physical domains (WI 889 — the WI 886
                    // `UnphysicalValue` posture extended to bounds; a violating
                    // band names the record and the offending bound).
                    let band = |lo: Option<f64>,
                                hi: Option<f64>,
                                fmin: &'static str,
                                fmax: &'static str| {
                        let lo = req(lo, fmin)?;
                        let hi = req(hi, fmax)?;
                        if !lo.is_finite() {
                            return Err(unphysical(fmin));
                        }
                        if !hi.is_finite() {
                            return Err(unphysical(fmax));
                        }
                        if lo > hi {
                            return Err(unphysical(fmin));
                        }
                        Ok::<(f64, f64), ContentError>((lo, hi))
                    };
                    let physical = |(lo, hi): (f64, f64),
                                    ok: &dyn Fn(f64) -> bool,
                                    fmin: &'static str,
                                    fmax: &'static str| {
                        if !ok(lo) {
                            return Err(unphysical(fmin));
                        }
                        if !ok(hi) {
                            return Err(unphysical(fmax));
                        }
                        Ok(())
                    };
                    // The drawn independents every shape needs (the medium
                    // derives from them, WI 889): insolation ≥ 0, albedo in
                    // [0, 1) — `1 − A` must stay positive for T_eq.
                    let insolation = band(
                        b.nominal_insolation_min,
                        b.nominal_insolation_max,
                        "nominal_insolation_min",
                        "nominal_insolation_max",
                    )?;
                    physical(
                        insolation,
                        &|v| v >= 0.0,
                        "nominal_insolation_min",
                        "nominal_insolation_max",
                    )?;
                    let albedo = band(
                        b.bond_albedo_min,
                        b.bond_albedo_max,
                        "bond_albedo_min",
                        "bond_albedo_max",
                    )?;
                    physical(
                        albedo,
                        &|v| (0.0..1.0).contains(&v),
                        "bond_albedo_min",
                        "bond_albedo_max",
                    )?;
                    // The atmosphere trio (Rocky/Ocean): greenhouse ≥ 0,
                    // molar mass > 0.
                    let atmosphere = |b: &RawBodyRecipe| {
                        let pressure = band(
                            b.atmosphere_surface_pressure_min,
                            b.atmosphere_surface_pressure_max,
                            "atmosphere_surface_pressure_min",
                            "atmosphere_surface_pressure_max",
                        )?;
                        let greenhouse = band(
                            b.greenhouse_delta_t_min,
                            b.greenhouse_delta_t_max,
                            "greenhouse_delta_t_min",
                            "greenhouse_delta_t_max",
                        )?;
                        physical(
                            greenhouse,
                            &|v| v >= 0.0,
                            "greenhouse_delta_t_min",
                            "greenhouse_delta_t_max",
                        )?;
                        let molar_mass = band(
                            b.mean_molar_mass_min,
                            b.mean_molar_mass_max,
                            "mean_molar_mass_min",
                            "mean_molar_mass_max",
                        )?;
                        physical(
                            molar_mass,
                            &|v| v > 0.0,
                            "mean_molar_mass_min",
                            "mean_molar_mass_max",
                        )?;
                        Ok::<_, ContentError>((pressure, greenhouse, molar_mass))
                    };
                    let common = ArchetypeBands {
                        radius: band(b.radius_min, b.radius_max, "radius_min", "radius_max")?,
                        gravity: band(b.gravity_min, b.gravity_max, "gravity_min", "gravity_max")?,
                        sidereal_period: band(
                            b.rotation_period_min,
                            b.rotation_period_max,
                            "rotation_period_min",
                            "rotation_period_max",
                        )?,
                        nominal_insolation: insolation,
                        bond_albedo: albedo,
                        ..ArchetypeBands::default()
                    };
                    let bands = match shape {
                        Archetype::Moon => common,
                        Archetype::RockyPlanet => {
                            let (pressure, greenhouse, molar_mass) = atmosphere(b)?;
                            ArchetypeBands {
                                atmosphere_surface_pressure: pressure,
                                greenhouse_delta_t: greenhouse,
                                mean_molar_mass: molar_mass,
                                ..common
                            }
                        }
                        Archetype::OceanWorld => {
                            let (pressure, greenhouse, molar_mass) = atmosphere(b)?;
                            ArchetypeBands {
                                atmosphere_surface_pressure: pressure,
                                greenhouse_delta_t: greenhouse,
                                mean_molar_mass: molar_mass,
                                ocean_surface_density: band(
                                    b.ocean_surface_density_min,
                                    b.ocean_surface_density_max,
                                    "ocean_surface_density_min",
                                    "ocean_surface_density_max",
                                )?,
                                ocean_temperature: band(
                                    b.ocean_temperature_min,
                                    b.ocean_temperature_max,
                                    "ocean_temperature_min",
                                    "ocean_temperature_max",
                                )?,
                                ..common
                            }
                        }
                    };
                    // WI 880: suppressed fields — validate each name against
                    // the drawn vocabulary (all shapes, then this shape),
                    // require its same-name explicit scalar, hold the scalar
                    // to the same physical domain its band bounds obey, and
                    // pin it into the sample (that field's stream is never
                    // opened; every other draw is bit-unchanged, WI 889).
                    let mut pins: BTreeMap<&str, f64> = BTreeMap::new();
                    for &name in &suppressed {
                        let Some(&stat) = bodygen::SUPPRESSIBLE_FIELDS.iter().find(|&&f| f == name)
                        else {
                            return Err(ContentError::NotSuppressible {
                                record: b.id.clone(),
                                field: name.to_string(),
                            });
                        };
                        if !bodygen::suppressible_fields(shape).contains(&stat) {
                            return Err(inapplicable(stat));
                        }
                        let scalar = match stat {
                            "radius" => b.radius,
                            "gravity" => b.gravity,
                            "rotation_period" => b.rotation_period,
                            "nominal_insolation" => b.nominal_insolation,
                            "bond_albedo" => b.bond_albedo,
                            "atmosphere_surface_pressure" => b.atmosphere_surface_pressure,
                            "greenhouse_delta_t" => b.greenhouse_delta_t,
                            "mean_molar_mass" => b.mean_molar_mass,
                            "ocean_surface_density" => b.ocean_surface_density,
                            "ocean_temperature" => b.ocean_temperature,
                            other => unreachable!("suppressible field {other} has no scalar"),
                        };
                        let v = req(scalar, stat)?;
                        let in_domain = v.is_finite()
                            && match stat {
                                "nominal_insolation" | "greenhouse_delta_t" => v >= 0.0,
                                "bond_albedo" => (0.0..1.0).contains(&v),
                                "mean_molar_mass" => v > 0.0,
                                _ => true,
                            };
                        if !in_domain {
                            return Err(unphysical(stat));
                        }
                        pins.insert(stat, v);
                    }
                    let seed = surface_seed()?;
                    let mut body = bodygen::sample(seed, shape, &bands, &pins);
                    // Identity + classifier offset are the recipe's, not the
                    // generator's synthesized ones; the physics is the draw's.
                    body.id = b.id.clone();
                    body.name = b.name.clone().ok_or_else(|| missing(&b.id, "name"))?;
                    body.surface.layers = surface_layers;
                    // Retain the resolved bands (WI 884) so consumers can
                    // re-sample the family at other seeds.
                    (body, Some(bands))
                }
                // Fixed (WI 881/886): independents are authored; each derived
                // medium field is **pin-or-derive** — an authored value holds
                // exactly (the design's `pin:`, keyed by the derived field's
                // own name), an absent one is computed by its named relation
                // (`body_derive`), and neither-pin-nor-inputs is loud.
                None => {
                    let unphysical = |field: &'static str| ContentError::UnphysicalValue {
                        id: b.id.clone(),
                        field,
                    };
                    // Independent set (required, as they always were).
                    let mu = req(b.mu, "mu")?;
                    let radius = req(b.radius, "radius")?;
                    let p_atm = req(b.atmosphere_surface_pressure, "atmosphere_surface_pressure")?;
                    let ocean_density = req(b.ocean_surface_density, "ocean_surface_density")?;
                    let ocean_gradient = req(b.ocean_density_gradient, "ocean_density_gradient")?;
                    let ocean_temperature = req(b.ocean_temperature, "ocean_temperature")?;

                    // g = μ/R² (pin wins; relation needs a physical radius).
                    let gravity = match b.gravity {
                        Some(pin) => pin,
                        None => {
                            if !(radius.is_finite() && radius > 0.0) {
                                return Err(unphysical("radius"));
                            }
                            body_derive::surface_gravity(mu, radius)
                        }
                    };
                    // T_surf = T_eq(S, A) + ΔT_greenhouse (C1). The relation
                    // errors name the missing/unphysical *input* — strictly
                    // more actionable than naming the derived field.
                    let t_surf = match b.atmosphere_temperature {
                        Some(pin) => pin,
                        None => {
                            let s = req(b.nominal_insolation, "nominal_insolation")?;
                            let a = req(b.bond_albedo, "bond_albedo")?;
                            let dt = req(b.greenhouse_delta_t, "greenhouse_delta_t")?;
                            if !(s.is_finite() && s >= 0.0) {
                                return Err(unphysical("nominal_insolation"));
                            }
                            // Same domain as the band validation: `[0, 1)` —
                            // `1 − A` must stay positive for T_eq (WI 893).
                            if !(0.0..1.0).contains(&a) {
                                return Err(unphysical("bond_albedo"));
                            }
                            body_derive::surface_temperature(
                                body_derive::equilibrium_temperature(s, a),
                                dt,
                            )
                        }
                    };
                    // A positive molar mass, where a gas relation needs it.
                    let molar_mass = |field: &'static str| {
                        let m = req(b.mean_molar_mass, "mean_molar_mass")?;
                        if !(m.is_finite() && m > 0.0) {
                            return Err(unphysical("mean_molar_mass"));
                        }
                        if !(t_surf.is_finite() && t_surf > 0.0) {
                            return Err(unphysical(field));
                        }
                        Ok(m)
                    };
                    // Airless skeleton (P₀ = 0): density 0, the positive
                    // placeholder scale height `generate` has always used.
                    let atmosphere_surface_density = match b.atmosphere_surface_density {
                        Some(pin) => pin,
                        None if p_atm == 0.0 => 0.0,
                        None => body_derive::atmosphere_surface_density(
                            p_atm,
                            molar_mass("atmosphere_temperature")?,
                            t_surf,
                        ),
                    };
                    let atmosphere_scale_height = match b.atmosphere_scale_height {
                        Some(pin) => pin,
                        None if p_atm == 0.0 => 1.0,
                        None => body_derive::scale_height(
                            t_surf,
                            molar_mass("atmosphere_temperature")?,
                            gravity,
                        ),
                    };
                    // Ocean surface pressure: continuous with the atmosphere
                    // when an ocean is present (authored density > 0).
                    let ocean_present = ocean_density > 0.0;
                    let ocean_surface_pressure = match b.ocean_surface_pressure {
                        Some(pin) => pin,
                        None if ocean_present => p_atm,
                        None => 0.0,
                    };
                    // Ocean gating — a *presence* decision, applied last, wins
                    // over pins: frozen (medium T_surf at/below the classifier
                    // freeze point; never the WI-875 presentation offset) or
                    // airless ⇒ no liquid ocean (the trio zeroes; temperatures
                    // stay, matching the airless `generate` skeleton). The
                    // decision itself is `body_derive::gate_ocean` — the ONE
                    // implementation both resolve arms share (WI 889).
                    let (ocean_surface_density, ocean_surface_pressure, ocean_density_gradient) =
                        body_derive::gate_ocean(
                            t_surf,
                            p_atm,
                            ocean_density,
                            ocean_surface_pressure,
                            ocean_gradient,
                        );

                    let body = BodyAsset {
                        id: b.id.clone(),
                        name: b.name.clone().ok_or_else(|| missing(&b.id, "name"))?,
                        mu,
                        radius,
                        rotation: Rotation {
                            axis: DVec3::Z,
                            sidereal_period: req(b.rotation_period, "rotation_period")?,
                        },
                        fluid_medium: FluidMedium {
                            atmosphere_surface_density,
                            atmosphere_surface_pressure: p_atm,
                            atmosphere_scale_height,
                            ocean_surface_density,
                            ocean_surface_pressure,
                            ocean_density_gradient,
                            gravity,
                            atmosphere_temperature: t_surf,
                            ocean_temperature,
                        },
                        surface: SurfaceRecipe {
                            layers: surface_layers,
                            ..SurfaceRecipe::from_seed(surface_seed()?)
                        },
                        render: serde_json::Value::Null,
                    };
                    (body, None)
                }
            };
            Ok(Record::BodyRecipe(Box::new(BodyRecipeRecord {
                id: b.id.clone(),
                body,
                shape: b.shape,
                bands,
                tags: b.tags.clone().unwrap_or_default(),
            })))
        }
    }
}

/// Build the class-specific spec, requiring the class's fields and rejecting
/// fields that do not apply to it (loud authoring-typo detection).
fn device_spec(d: &RawDevice, class: DeviceClass) -> Result<DeviceSpec, ContentError> {
    let missing = |field: &'static str| ContentError::MissingField {
        id: d.id.clone(),
        field,
    };
    let inapplicable = |field: &'static str| ContentError::InapplicableField {
        id: d.id.clone(),
        field,
    };
    let engine = matches!(class, DeviceClass::Engine);
    let tanklike = matches!(class, DeviceClass::Tank | DeviceClass::Battery);
    let motor = matches!(class, DeviceClass::Motor);
    if d.exhaust_velocity.is_some() && !engine {
        return Err(inapplicable("exhaust_velocity"));
    }
    if d.max_mass_flow.is_some() && !engine {
        return Err(inapplicable("max_mass_flow"));
    }
    if d.capacity.is_some() && !tanklike {
        return Err(inapplicable("capacity"));
    }
    if d.max_torque.is_some() && !motor {
        return Err(inapplicable("max_torque"));
    }
    if d.top_speed.is_some() && !motor {
        return Err(inapplicable("top_speed"));
    }
    if d.motor_mass.is_some() && !motor {
        return Err(inapplicable("motor_mass"));
    }
    if d.draw.is_some() && !motor {
        return Err(inapplicable("draw"));
    }
    Ok(match class {
        DeviceClass::Engine => DeviceSpec::Engine {
            exhaust_velocity: d
                .exhaust_velocity
                .ok_or_else(|| missing("exhaust_velocity"))?,
            max_mass_flow: d.max_mass_flow.ok_or_else(|| missing("max_mass_flow"))?,
        },
        DeviceClass::Tank => DeviceSpec::Tank {
            capacity: d.capacity.ok_or_else(|| missing("capacity"))?,
        },
        DeviceClass::Battery => DeviceSpec::Battery {
            capacity: d.capacity.ok_or_else(|| missing("capacity"))?,
        },
        DeviceClass::Motor => DeviceSpec::Motor {
            max_torque: d.max_torque.ok_or_else(|| missing("max_torque"))?,
            top_speed: d.top_speed.ok_or_else(|| missing("top_speed"))?,
            mass: d.motor_mass.ok_or_else(|| missing("motor_mass"))?,
            draw: d.draw.ok_or_else(|| missing("draw"))?,
        },
    })
}

// ---------------------------------------------------------------------------
// The canonical bodies (WI 884) — the embedded recipe pack.
// ---------------------------------------------------------------------------

/// The canonical bodies pack, embedded at compile time (WI 884): the single
/// authored source of the canonical fixed bodies (`earthlike`,
/// `earthlike-ice-age`) and the archetype band records (`moon`, `rocky`,
/// `ocean`). `BodyAsset::earthlike`/`earthlike_ice_age` and `bodygen::generate`
/// resolve from here — the hardcoded constructor assemblies and generator band
/// literals they replaced are gone.
const CANONICAL_BODIES_RON: &str = include_str!("../content/bodies.ron");

/// The resolved canonical-bodies catalog, parsed once per process. The embedded
/// pack is a compile-time asset gated by a unit test, so the `expect`s here are
/// build-integrity assertions, never reachable input failures.
fn canonical_catalog() -> &'static Catalog {
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        Catalog::from_ron_str(CANONICAL_BODIES_RON)
            .expect("embedded canonical bodies pack (crates/sim/content/bodies.ron) must resolve")
    })
}

/// The `surface_temperature_offset` sugar composed onto a resolved layer
/// stack (WI 892, plan-pinned rule): the offset applies to the **first
/// enabled material layer** in stack order, overwriting that layer's own
/// `temperature` param (body-local authoring is the more specific); if
/// material layers exist but **none is enabled**, the sugar is **discarded**
/// (an explicit disable is the stronger authoring — the kill switch keeps its
/// teeth); if **no material layer exists** (the pre-stack universal case), a
/// synthesized enabled layer with the well-known id `material` is appended at
/// the tail. Non-object params on the receiving layer are replaced by the
/// offset object (they were unreadable to the classifier anyway).
fn apply_temperature_sugar(layers: &mut Vec<crate::body_asset::SurfaceLayer>, offset: Option<f64>) {
    use crate::body_asset::{SurfaceLayer, SurfaceLayerType};
    let Some(offset) = offset else {
        return;
    };
    if let Some(layer) = layers
        .iter_mut()
        .find(|l| l.enabled && l.layer_type == SurfaceLayerType::Material)
    {
        match &mut layer.params {
            serde_json::Value::Object(map) => {
                map.insert("temperature".to_string(), serde_json::json!(offset));
            }
            other => *other = serde_json::json!({ "temperature": offset }),
        }
    } else if layers
        .iter()
        .any(|l| l.layer_type == SurfaceLayerType::Material)
    {
        // Material layers exist but every one is disabled: discarded.
    } else {
        layers.push(SurfaceLayer::well_known(
            SurfaceLayerType::Material,
            serde_json::json!({ "temperature": offset }),
        ));
    }
}

/// Resolve a canonical body by record id from the embedded pack (WI 884).
/// Panics if the id is absent or not a body recipe — a build-integrity error
/// (the embedded pack is fixed at compile time and covered by a unit test).
pub(crate) fn canonical_body(id: &str) -> BodyAsset {
    let entry = canonical_catalog()
        .get(id)
        .unwrap_or_else(|| panic!("canonical body `{id}` missing from the embedded bodies pack"));
    match &entry.record {
        Record::BodyRecipe(r) => r.body.clone(),
        other => panic!("canonical record `{id}` is not a body recipe: {other:?}"),
    }
}

/// Non-panicking sibling of [`canonical_body`] for **persisted input** (WI 891):
/// a saved catalog ref names a recipe id this build may simply not have, which
/// is a load error to report, never a build-integrity panic. `None` when the id
/// is absent from the embedded pack or is not a body recipe.
pub(crate) fn try_canonical_body(id: &str) -> Option<BodyAsset> {
    match &canonical_catalog().get(id)?.record {
        Record::BodyRecipe(r) => Some(r.body.clone()),
        _ => None,
    }
}

/// The embedded canonical pack's identity — `(pack id, pack version)` — for
/// recording provenance into persisted catalog refs (WI 891). The embedded
/// pack is exactly one pack by construction; covered by the build-integrity
/// tests over `canonical_catalog`.
pub(crate) fn canonical_pack_identity() -> (String, String) {
    canonical_catalog()
        .packs
        .first()
        .cloned()
        .expect("embedded canonical bodies pack records its identity")
}

/// The ladder-resolved parameter bands of the canonical archetype record for
/// `archetype` (record id = its slug), from the embedded pack (WI 884). Drives
/// [`crate::bodygen::generate`]. Same build-integrity panic contract as
/// [`canonical_body`].
pub(crate) fn canonical_bands(archetype: Archetype) -> ArchetypeBands {
    let id = archetype.slug();
    let entry = canonical_catalog().get(id).unwrap_or_else(|| {
        panic!("canonical archetype `{id}` missing from the embedded bodies pack")
    });
    match &entry.record {
        Record::BodyRecipe(r) => r
            .bands
            .unwrap_or_else(|| panic!("canonical archetype `{id}` retains no bands (not shaped?)")),
        other => panic!("canonical record `{id}` is not a body recipe: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biome::OCEAN_FREEZE_THRESHOLD_K;

    /// A minimal valid pack around the given records list (RON snippet).
    fn pack(records: &str) -> String {
        format!(
            "#![enable(implicit_some)]\n(format: 2, id: \"test\", version: \"1\", records: [{records}])"
        )
    }

    /// A minimal pack with an explicit id (multi-pack tests).
    fn pack_named(id: &str, extra: &str, records: &str) -> String {
        format!(
            "#![enable(implicit_some)]\n(format: 2, id: \"{id}\", version: \"1\", {extra} records: [{records}])"
        )
    }

    /// A minimal override set.
    fn override_set(id: &str, phase: &str, extra: &str, overrides: &str) -> String {
        format!("(format: 2, id: \"{id}\", phase: {phase}, {extra} overrides: [{overrides}])")
    }

    fn engine_base() -> &'static str {
        r#"Device(( id: "engine_base", abstract: true, class: Engine, exhaust_velocity: 3200.0 )),"#
    }

    fn engine_variant() -> &'static str {
        r#"Device(( id: "v", parent: "engine_base", density: 3000.0, max_mass_flow: 1.0 )),"#
    }

    fn engine_ev(cat: &Catalog, id: &str) -> f64 {
        match &cat.get(id).unwrap().record {
            Record::Device(d) => match d.spec {
                DeviceSpec::Engine {
                    exhaust_velocity, ..
                } => exhaust_velocity,
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }

    // ------------------------------------------------------------------
    // WI 547 — single-pack loading, inheritance, validation.
    // ------------------------------------------------------------------

    #[test]
    fn shipped_core_pack_loads_and_resolves() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../content/packs/core.ron");
        let cat = Catalog::load(&path).expect("shipped core pack loads");
        assert_eq!(cat.pack_id, "core");
        // The concrete engine inherits exhaust velocity from the abstract base
        // and carries its own mass flow + density.
        let entry = cat.get("lf_engine_small").expect("engine present");
        assert_eq!(entry.provenance.pack_id, "core");
        match &entry.record {
            Record::Device(d) => {
                assert_eq!(d.density, 3000.0);
                assert_eq!(
                    d.spec,
                    DeviceSpec::Engine {
                        exhaust_velocity: 3200.0,
                        max_mass_flow: 1.5
                    }
                );
            }
            other => panic!("expected device, got {other:?}"),
        }
        // The abstract base is not content.
        assert!(cat.get("engine_base").is_none());
        // Material and resource records resolve.
        match &cat.get("aluminium").expect("material present").record {
            Record::Material(m) => {
                assert_eq!(m.material.density, 2700.0);
                assert_eq!(m.material.thermal.max_temp, 900.0);
            }
            other => panic!("expected material, got {other:?}"),
        }
        match &cat.get("liquid_fuel").expect("resource present").record {
            Record::Resource(r) => {
                assert_eq!(r.density, Some(800.0));
                assert!(r.tradable);
            }
            other => panic!("expected resource, got {other:?}"),
        }
    }

    #[test]
    fn inheritance_chain_resolves_and_overrides() {
        let src = pack(&format!(
            r#"{base}
            Device(( id: "mid", parent: "engine_base", abstract: true, density: 2500.0 )),
            Device(( id: "leaf", parent: "mid", density: 3100.0, max_mass_flow: 2.0 )),"#,
            base = engine_base()
        ));
        let cat = Catalog::from_ron_str(&src).unwrap();
        match &cat.get("leaf").unwrap().record {
            Record::Device(d) => {
                // Grandparent's exhaust velocity, own density override, own flow.
                assert_eq!(d.density, 3100.0);
                assert_eq!(
                    d.spec,
                    DeviceSpec::Engine {
                        exhaust_velocity: 3200.0,
                        max_mass_flow: 2.0
                    }
                );
            }
            other => panic!("expected device, got {other:?}"),
        }
        // Both abstracts are absent from the catalog.
        assert!(cat.get("engine_base").is_none() && cat.get("mid").is_none());
        assert_eq!(cat.len(), 1);
    }

    #[test]
    fn editing_a_base_reflows_to_variants() {
        let a = pack(&format!("{}{}", engine_base(), engine_variant()));
        let b = pack(&format!(
            "{}{}",
            engine_base().replace("3200.0", "4400.0"),
            engine_variant()
        ));
        let va = Catalog::from_ron_str(&a).unwrap();
        let vb = Catalog::from_ron_str(&b).unwrap();
        assert_eq!(engine_ev(&va, "v"), 3200.0);
        assert_eq!(engine_ev(&vb, "v"), 4400.0);
    }

    #[test]
    fn declaration_order_does_not_matter() {
        let fwd = pack(&format!("{}{}", engine_base(), engine_variant()));
        let rev = pack(&format!("{}{}", engine_variant(), engine_base()));
        assert_eq!(
            Catalog::from_ron_str(&fwd).unwrap(),
            Catalog::from_ron_str(&rev).unwrap()
        );
    }

    #[test]
    fn duplicate_id_rejected_across_kinds() {
        let src = pack(
            r#"Resource(( id: "x" )),
               Material(( id: "x", density: 1.0, strength: 1.0, specific_heat: 1.0, conductivity: 1.0, emissivity: 0.5, max_temp: 100.0 )),"#,
        );
        match Catalog::from_ron_str(&src) {
            Err(ContentError::DuplicateId { id }) => assert_eq!(id, "x"),
            other => panic!("expected DuplicateId, got {other:?}"),
        }
    }

    #[test]
    fn unknown_parent_rejected_naming_both() {
        let src = pack(r#"Device(( id: "child", parent: "ghost" )),"#);
        match Catalog::from_ron_str(&src) {
            Err(ContentError::UnknownParent { child, parent }) => {
                assert_eq!((child.as_str(), parent.as_str()), ("child", "ghost"));
            }
            other => panic!("expected UnknownParent, got {other:?}"),
        }
    }

    #[test]
    fn cycles_rejected_including_self_parent() {
        let self_p = pack(r#"Device(( id: "a", parent: "a" )),"#);
        assert!(matches!(
            Catalog::from_ron_str(&self_p),
            Err(ContentError::Cycle { .. })
        ));
        let two = pack(
            r#"Device(( id: "a", parent: "b" )),
               Device(( id: "b", parent: "a" )),"#,
        );
        assert!(matches!(
            Catalog::from_ron_str(&two),
            Err(ContentError::Cycle { .. })
        ));
    }

    #[test]
    fn parent_of_different_kind_rejected() {
        let src = pack(
            r#"Resource(( id: "base", abstract: true )),
               Device(( id: "child", parent: "base" )),"#,
        );
        match Catalog::from_ron_str(&src) {
            Err(ContentError::KindMismatch { child, parent }) => {
                assert_eq!((child.as_str(), parent.as_str()), ("child", "base"));
            }
            other => panic!("expected KindMismatch, got {other:?}"),
        }
    }

    #[test]
    fn wrong_format_version_rejected() {
        let src = "(format: 99, id: \"p\", version: \"1\", records: [])";
        match Catalog::from_ron_str(src) {
            Err(ContentError::Format { found }) => assert_eq!(found, 99),
            other => panic!("expected Format, got {other:?}"),
        }
    }

    #[test]
    fn unknown_field_and_unknown_kind_rejected() {
        let field = pack(r#"Resource(( id: "r", flavour: "grape" )),"#);
        assert!(matches!(
            Catalog::from_ron_str(&field),
            Err(ContentError::Parse(_))
        ));
        let kind = pack(r#"Widget(( id: "w" )),"#);
        assert!(matches!(
            Catalog::from_ron_str(&kind),
            Err(ContentError::Parse(_))
        ));
    }

    #[test]
    fn missing_required_field_rejected() {
        // Concrete engine without a mass flow (base supplies velocity only).
        let src = pack(&format!(
            "{}Device(( id: \"v\", parent: \"engine_base\", density: 1.0 )),",
            engine_base()
        ));
        match Catalog::from_ron_str(&src) {
            Err(ContentError::MissingField { id, field }) => {
                assert_eq!((id.as_str(), field), ("v", "max_mass_flow"));
            }
            other => panic!("expected MissingField, got {other:?}"),
        }
    }

    #[test]
    fn inapplicable_field_rejected() {
        let src = pack(
            r#"Device(( id: "t", class: Tank, density: 500.0, capacity: 100.0, exhaust_velocity: 3000.0 )),"#,
        );
        match Catalog::from_ron_str(&src) {
            Err(ContentError::InapplicableField { id, field }) => {
                assert_eq!((id.as_str(), field), ("t", "exhaust_velocity"));
            }
            other => panic!("expected InapplicableField, got {other:?}"),
        }
    }

    #[test]
    fn motor_and_battery_records_resolve() {
        let src = pack(
            r#"Device(( id: "motor_std", class: Motor, density: 4000.0, max_torque: 90.0, top_speed: 60.0, motor_mass: 45.0, draw: 1.0 )),
               Device(( id: "battery_s", class: Battery, density: 2500.0, capacity: 5.0e6 )),
               Body(( id: "start_moon", body: "training-moon" )),"#,
        );
        let cat = Catalog::from_ron_str(&src).unwrap();
        assert!(matches!(
            &cat.get("motor_std").unwrap().record,
            Record::Device(DeviceRecord {
                spec: DeviceSpec::Motor { .. },
                ..
            })
        ));
        assert!(matches!(
            &cat.get("battery_s").unwrap().record,
            Record::Device(DeviceRecord {
                spec: DeviceSpec::Battery { .. },
                ..
            })
        ));
        match &cat.get("start_moon").unwrap().record {
            Record::Body(b) => assert_eq!(b.body_slug, "training-moon"),
            other => panic!("expected body ref, got {other:?}"),
        }
    }

    #[test]
    fn body_reference_requires_slug() {
        let src = pack(r#"Body(( id: "start_moon" )),"#);
        assert!(matches!(
            Catalog::from_ron_str(&src),
            Err(ContentError::MissingField { field: "body", .. })
        ));
    }

    #[test]
    fn empty_pack_loads_to_empty_catalog() {
        let cat =
            Catalog::from_ron_str("(format: 2, id: \"p\", version: \"1\", records: [])").unwrap();
        assert!(cat.is_empty());
    }

    #[test]
    fn loading_is_deterministic() {
        let src = pack(&format!(
            "{}{}Resource(( id: \"fuel\", density: 800.0 )),",
            engine_base(),
            engine_variant()
        ));
        let a = Catalog::from_ron_str(&src).unwrap();
        let b = Catalog::from_ron_str(&src).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.ids().collect::<Vec<_>>(), vec!["fuel", "v"]);
    }

    // ------------------------------------------------------------------
    // WI 548 — override model + merge ladder + provenance.
    // ------------------------------------------------------------------

    #[test]
    fn base_multiply_reflows_to_variant_with_provenance() {
        let p = pack(&format!("{}{}", engine_base(), engine_variant()));
        let s = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("engine_base"), field: "exhaust_velocity", op: Multiply(0.5) ),"#,
        );
        let cat = Catalog::merge(&[&p], &[&s]).unwrap();
        assert_eq!(engine_ev(&cat, "v"), 1600.0);
        // The inherited field's provenance traces to the base's override.
        let fp = &cat.get("v").unwrap().field_provenance["exhaust_velocity"];
        assert_eq!(
            fp.source,
            SourceRef::Override {
                source: "scn".into(),
                phase: OverridePhase::Scenario
            }
        );
        assert_eq!(
            fp.shadows,
            vec![Shadow {
                value: ProvValue::Number(3200.0),
                source: SourceRef::Pack { id: "test".into() }
            }]
        );
    }

    #[test]
    fn later_phase_shadows_earlier_and_chain_is_recorded() {
        let p = pack(&format!("{}{}", engine_base(), engine_variant()));
        let scn = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("engine_base"), field: "exhaust_velocity", op: Multiply(0.5) ),"#,
        );
        let local = override_set(
            "house",
            "Local",
            "",
            r#"( target: Id("engine_base"), field: "exhaust_velocity", op: Set(Number(9000.0)) ),"#,
        );
        let cat = Catalog::merge(&[&p], &[&local, &scn]).unwrap();
        assert_eq!(engine_ev(&cat, "v"), 9000.0);
        let fp = &cat.get("v").unwrap().field_provenance["exhaust_velocity"];
        assert_eq!(
            fp.source,
            SourceRef::Override {
                source: "house".into(),
                phase: OverridePhase::Local
            }
        );
        // Newest displaced first: the scenario's 1600, then the pack's 3200.
        assert_eq!(fp.shadows.len(), 2);
        assert_eq!(fp.shadows[0].value, ProvValue::Number(1600.0));
        assert_eq!(
            fp.shadows[0].source,
            SourceRef::Override {
                source: "scn".into(),
                phase: OverridePhase::Scenario
            }
        );
        assert_eq!(fp.shadows[1].value, ProvValue::Number(3200.0));
        assert_eq!(fp.shadows[1].source, SourceRef::Pack { id: "test".into() });
    }

    #[test]
    fn multiply_on_unset_variant_field_errors_with_guidance() {
        let p = pack(&format!("{}{}", engine_base(), engine_variant()));
        let s = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("v"), field: "exhaust_velocity", op: Multiply(0.5) ),"#,
        );
        match Catalog::merge(&[&p], &[&s]) {
            Err(ContentError::UnsetField { record, field }) => {
                assert_eq!((record.as_str(), field.as_str()), ("v", "exhaust_velocity"));
            }
            other => panic!("expected UnsetField, got {other:?}"),
        }
    }

    #[test]
    fn intra_phase_dependency_order_wins() {
        let p = pack(&format!("{}{}", engine_base(), engine_variant()));
        let a = override_set(
            "patch_a",
            "Patch",
            "",
            r#"( target: Id("engine_base"), field: "exhaust_velocity", op: Set(Number(1000.0)) ),"#,
        );
        // patch_b declares it comes after patch_a, so its write wins.
        let b = override_set(
            "patch_b",
            "Patch",
            "depends: [\"patch_a\"],",
            r#"( target: Id("engine_base"), field: "exhaust_velocity", op: Set(Number(2000.0)) ),"#,
        );
        // Input order must not matter.
        let cat = Catalog::merge(&[&p], &[&b, &a]).unwrap();
        assert_eq!(engine_ev(&cat, "v"), 2000.0);
        let fp = &cat.get("v").unwrap().field_provenance["exhaust_velocity"];
        assert_eq!(fp.shadows[0].value, ProvValue::Number(1000.0));
    }

    #[test]
    fn merge_is_deterministic_under_input_permutation() {
        let p1 = pack_named("alpha", "", engine_base());
        let p2 = pack_named(
            "beta",
            "depends: [\"alpha\"],",
            r#"Device(( id: "v", parent: "engine_base", density: 3000.0, max_mass_flow: 1.0 )),"#,
        );
        let s1 = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("engine_base"), field: "exhaust_velocity", op: Multiply(2.0) ),"#,
        );
        let s2 = override_set(
            "house",
            "Local",
            "",
            r#"( target: Kind(Device), field: "tags", op: Set(List(["tuned"])) ),"#,
        );
        let a = Catalog::merge(&[&p1, &p2], &[&s1, &s2]).unwrap();
        let b = Catalog::merge(&[&p2, &p1], &[&s2, &s1]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.sources, vec!["alpha", "beta", "scn", "house"]);
        assert_eq!(a.pack_id, "alpha");
    }

    #[test]
    fn set_defines_an_unset_field() {
        // The variant has no authored capacity... but capacity is not an
        // engine field; use a tank whose density comes from an override.
        let p = pack(r#"Device(( id: "t", class: Tank, capacity: 100.0 )),"#);
        let s = override_set(
            "patch",
            "Patch",
            "",
            r#"( target: Id("t"), field: "density", op: Set(Number(500.0)) ),"#,
        );
        let cat = Catalog::merge(&[&p], &[&s]).unwrap();
        match &cat.get("t").unwrap().record {
            Record::Device(d) => assert_eq!(d.density, 500.0),
            other => panic!("expected device, got {other:?}"),
        }
        let fp = &cat.get("t").unwrap().field_provenance["density"];
        assert_eq!(fp.shadows[0].value, ProvValue::Unset);
    }

    #[test]
    fn extend_and_delete_on_tags() {
        let p = pack(r#"Resource(( id: "fuel", density: 800.0 )),"#);
        // Parentless + unset tags: extend starts from empty.
        let s1 = override_set(
            "patch",
            "Patch",
            "",
            r#"( target: Id("fuel"), field: "tags", op: Extend(["starter", "cheap"]) ),"#,
        );
        let s2 = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("fuel"), field: "tags", op: Delete("cheap") ),"#,
        );
        let cat = Catalog::merge(&[&p], &[&s1, &s2]).unwrap();
        match &cat.get("fuel").unwrap().record {
            Record::Resource(r) => assert_eq!(r.tags, vec!["starter"]),
            other => panic!("expected resource, got {other:?}"),
        }
        // Deleting an absent element is loud.
        let s3 = override_set(
            "house",
            "Local",
            "",
            r#"( target: Id("fuel"), field: "tags", op: Delete("missing") ),"#,
        );
        match Catalog::merge(&[&p], &[&s1, &s3]) {
            Err(ContentError::AbsentElement { element, .. }) => assert_eq!(element, "missing"),
            other => panic!("expected AbsentElement, got {other:?}"),
        }
    }

    // WI 879: ordered, id-keyed list ops compose across the ladder without index
    // dependence — insert-after and replace splice relative to a named element.
    fn tags_of(cat: &Catalog, id: &str) -> Vec<String> {
        match &cat.get(id).unwrap().record {
            Record::Resource(r) => r.tags.clone(),
            other => panic!("expected resource, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // WI 892 — the surface-layer stack.
    // -----------------------------------------------------------------

    /// A fixed recipe body carrying the derive_recipe independents plus
    /// `extra` (e.g. a surface_stack) — the WI 892 stack fixtures' base.
    fn stack_body(extra: &str) -> String {
        format!(
            r#"BodyRecipe(( id: "sb", name: "SB",
                mu: 4.0e14, radius: 6.0e6, rotation_period: 86400.0,
                atmosphere_surface_pressure: 100000.0,
                nominal_insolation: 1361.0, bond_albedo: 0.3, greenhouse_delta_t: 33.0,
                mean_molar_mass: 0.029,
                ocean_surface_density: 1000.0, ocean_density_gradient: 0.0,
                ocean_temperature: 285.0, surface_seed: 7, {extra} )),"#
        )
    }

    fn stack_layers_of(cat: &Catalog, id: &str) -> Vec<(String, String, bool)> {
        resolved_body(cat, id)
            .surface
            .layers
            .iter()
            .map(|l| (l.id.clone(), l.layer_type.slug().to_string(), l.enabled))
            .collect()
    }

    /// WI 892 Phase B scenario 1: an authored stack resolves in order; layer
    /// params reach the body; a ladder `multiply` on the layer record's param
    /// re-flows into the resolved stack (per-layer-field provenance for free).
    #[test]
    fn authored_stack_resolves_in_order_and_ladder_tunes_layer_params() {
        let p = pack(&format!(
            r#"SurfaceLayer(( id: "big-craters", layer_type: "crater", density: 1.5, depth: 2.0 )),
               SurfaceLayer(( id: "climate", layer_type: "material", temperature: -10.0 )),
               {}"#,
            stack_body(r#"surface_stack: ["big-craters", "climate"],"#)
        ));
        let cat = Catalog::merge(&[&p], &[]).unwrap();
        assert_eq!(
            stack_layers_of(&cat, "sb"),
            vec![
                ("big-craters".to_string(), "crater".to_string(), true),
                ("climate".to_string(), "material".to_string(), true),
            ],
            "stack order is list order; enabled defaults true"
        );
        let body = resolved_body(&cat, "sb");
        assert_eq!(
            body.surface
                .params_of(crate::body_asset::SurfaceLayerType::Crater)["density"]
                .as_f64(),
            Some(1.5)
        );

        // Ladder-multiply the layer record's density: the body re-resolves
        // with the tuned value (layers are ordinary records to the ladder).
        let ov = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("big-craters"), field: "density", op: Multiply(2.0) ),"#,
        );
        let cat2 = Catalog::merge(&[&p], &[&ov]).unwrap();
        assert_eq!(
            resolved_body(&cat2, "sb")
                .surface
                .params_of(crate::body_asset::SurfaceLayerType::Crater)["density"]
                .as_f64(),
            Some(3.0),
            "multiply on the layer record re-flows into the resolved stack"
        );
        // Provenance recorded on the layer record's field.
        let fp = &cat2.get("big-craters").unwrap().field_provenance["density"];
        assert_eq!(fp.shadows[0].value, ProvValue::Number(1.5));
    }

    /// WI 892 Phase B scenario 3 + edge cases: unknown / wrong-kind /
    /// duplicate / abstract stack references and unknown layer types are
    /// loud and named; off-type params are `InapplicableField`.
    #[test]
    fn stack_reference_failures_are_loud_and_named() {
        // Unknown id.
        let p = pack(&stack_body(r#"surface_stack: ["ghost"],"#));
        match Catalog::merge(&[&p], &[]) {
            Err(ContentError::UnknownStackLayer { body, layer }) => {
                assert_eq!((body.as_str(), layer.as_str()), ("sb", "ghost"));
            }
            other => panic!("expected UnknownStackLayer, got {other:?}"),
        }
        // Wrong kind (a Resource is not a layer).
        let p = pack(&format!(
            r#"Resource(( id: "fuel", density: 800.0 )), {}"#,
            stack_body(r#"surface_stack: ["fuel"],"#)
        ));
        match Catalog::merge(&[&p], &[]) {
            Err(ContentError::StackLayerWrongKind { body, layer }) => {
                assert_eq!((body.as_str(), layer.as_str()), ("sb", "fuel"));
            }
            other => panic!("expected StackLayerWrongKind, got {other:?}"),
        }
        // Duplicate id in one stack.
        let p = pack(&format!(
            r#"SurfaceLayer(( id: "c", layer_type: "crater" )), {}"#,
            stack_body(r#"surface_stack: ["c", "c"],"#)
        ));
        match Catalog::merge(&[&p], &[]) {
            Err(ContentError::DuplicateStackLayer { body, layer }) => {
                assert_eq!((body.as_str(), layer.as_str()), ("sb", "c"));
            }
            other => panic!("expected DuplicateStackLayer, got {other:?}"),
        }
        // An abstract layer is an inheritance target, not content.
        let p = pack(&format!(
            r#"SurfaceLayer(( id: "base", abstract: true, layer_type: "crater" )), {}"#,
            stack_body(r#"surface_stack: ["base"],"#)
        ));
        assert!(matches!(
            Catalog::merge(&[&p], &[]),
            Err(ContentError::UnknownStackLayer { .. })
        ));
        // Unknown layer type.
        let p = pack(r#"SurfaceLayer(( id: "x", layer_type: "volcano" )),"#);
        match Catalog::merge(&[&p], &[]) {
            Err(ContentError::UnknownLayerType { id, layer_type }) => {
                assert_eq!((id.as_str(), layer_type.as_str()), ("x", "volcano"));
            }
            other => panic!("expected UnknownLayerType, got {other:?}"),
        }
        // Off-type param (temperature on a crater layer) — Phase B scenario 4.
        let p = pack(r#"SurfaceLayer(( id: "x", layer_type: "crater", temperature: -5.0 )),"#);
        match Catalog::merge(&[&p], &[]) {
            Err(ContentError::InapplicableField { id, field }) => {
                assert_eq!((id.as_str(), field), ("x", "temperature"));
            }
            other => panic!("expected InapplicableField, got {other:?}"),
        }
        // Missing layer_type.
        let p = pack(r#"SurfaceLayer(( id: "x", density: 1.0 )),"#);
        assert!(matches!(
            Catalog::merge(&[&p], &[]),
            Err(ContentError::MissingField {
                field: "layer_type",
                ..
            })
        ));
        // layer_type is structural (a definition, not a tunable): an override
        // targeting it is rejected via the existing mechanism (plan invariant).
        let p = pack(r#"SurfaceLayer(( id: "x", layer_type: "crater" )),"#);
        let bad = override_set(
            "bad",
            "Patch",
            "",
            r#"( target: Id("x"), field: "layer_type", op: Set(Text("material")) ),"#,
        );
        assert!(matches!(
            Catalog::merge(&[&p], &[&bad]),
            Err(ContentError::StructuralField { .. })
        ));
    }

    /// WI 892 pinned sugar rule: the offset lands on the first ENABLED
    /// material layer (overriding its own temperature); all-disabled ⇒
    /// discarded; none ⇒ a synthesized well-known `material` layer appends.
    #[test]
    fn temperature_sugar_composes_per_the_pinned_rule() {
        use crate::body_asset::SurfaceLayerType;
        // First enabled material layer receives (and is overridden by) the sugar.
        let p = pack(&format!(
            r#"SurfaceLayer(( id: "climate", layer_type: "material", temperature: -10.0, moisture: 0.2 )),
               {}"#,
            stack_body(r#"surface_stack: ["climate"], surface_temperature_offset: -40.0,"#)
        ));
        let cat = Catalog::merge(&[&p], &[]).unwrap();
        let body = resolved_body(&cat, "sb");
        let params = body.surface.params_of(SurfaceLayerType::Material);
        assert_eq!(params["temperature"].as_f64(), Some(-40.0), "sugar wins");
        assert_eq!(params["moisture"].as_f64(), Some(0.2), "other keys kept");
        assert_eq!(body.surface.layers.len(), 1, "no synthesized layer");

        // No material layer at all ⇒ synthesized well-known layer appended.
        let p = pack(&format!(
            r#"SurfaceLayer(( id: "c", layer_type: "crater" )), {}"#,
            stack_body(r#"surface_stack: ["c"], surface_temperature_offset: -40.0,"#)
        ));
        let body = resolved_body(&Catalog::merge(&[&p], &[]).unwrap(), "sb");
        assert_eq!(
            body.surface
                .layers
                .iter()
                .map(|l| l.id.as_str())
                .collect::<Vec<_>>(),
            vec!["c", "material"],
            "synthesized layer appended at the tail with the well-known id"
        );

        // Phase C scenario 2 (second half): all material layers disabled ⇒
        // the sugar is DISCARDED (explicit disable is the stronger authoring).
        let p = pack(&format!(
            r#"SurfaceLayer(( id: "climate", layer_type: "material", temperature: -10.0, enabled: false )),
               {}"#,
            stack_body(r#"surface_stack: ["climate"], surface_temperature_offset: -40.0,"#)
        ));
        let body = resolved_body(&Catalog::merge(&[&p], &[]).unwrap(), "sb");
        assert!(
            body.surface.params_of(SurfaceLayerType::Material).is_null(),
            "disabled layer reads as absent and the sugar does not resurrect it"
        );
        assert_eq!(body.surface.layers.len(), 1, "carried, not removed");
    }

    /// WI 892 Phase C: the WI 879 id-keyed ops splice `surface_stack`, and
    /// `set enabled=false` on a layer record via the ladder is the design's
    /// `disable` — the classifier returns to defaults, stack order untouched.
    #[test]
    fn stack_ops_splice_by_id_and_disable_is_a_field_set() {
        use crate::body_asset::SurfaceLayerType;
        let p = pack(&format!(
            r#"SurfaceLayer(( id: "craters", layer_type: "crater", density: 1.5 )),
               SurfaceLayer(( id: "climate", layer_type: "material", temperature: -40.0 )),
               SurfaceLayer(( id: "climate2", layer_type: "material", temperature: 5.0 )),
               {}"#,
            stack_body(r#"surface_stack: ["craters"],"#)
        ));
        // InsertAfter by id, then Replace by id (Phase C scenario 1).
        let ins = override_set(
            "ins",
            "Patch",
            "",
            r#"( target: Id("sb"), field: "surface_stack", op: InsertAfter(anchor: "craters", items: ["climate"]) ),"#,
        );
        let cat = Catalog::merge(&[&p], &[&ins]).unwrap();
        assert_eq!(
            stack_layers_of(&cat, "sb")
                .iter()
                .map(|(id, _, _)| id.as_str())
                .collect::<Vec<_>>(),
            vec!["craters", "climate"]
        );
        let rep = override_set(
            "rep",
            "Scenario",
            "",
            r#"( target: Id("sb"), field: "surface_stack", op: Replace(target: "climate", items: ["climate2"]) ),"#,
        );
        let cat = Catalog::merge(&[&p], &[&ins, &rep]).unwrap();
        assert_eq!(
            stack_layers_of(&cat, "sb")
                .iter()
                .map(|(id, _, _)| id.as_str())
                .collect::<Vec<_>>(),
            vec!["craters", "climate2"]
        );
        // Whole-list provenance shows the transitions.
        let fp = &cat.get("sb").unwrap().field_provenance["surface_stack"];
        assert_eq!(
            fp.shadows[0].value,
            ProvValue::List(vec!["craters".into(), "climate".into()])
        );

        // Disable via the ladder (Phase C scenario 2, first half): the layer
        // stays in the stack but reads as absent.
        let off = override_set(
            "off",
            "Local",
            "",
            r#"( target: Id("climate2"), field: "enabled", op: Set(Bool(false)) ),"#,
        );
        let cat = Catalog::merge(&[&p], &[&ins, &rep, &off]).unwrap();
        let body = resolved_body(&cat, "sb");
        assert_eq!(
            stack_layers_of(&cat, "sb")
                .iter()
                .map(|(id, _, enabled)| (id.as_str(), *enabled))
                .collect::<Vec<_>>(),
            vec![("craters", true), ("climate2", false)],
            "order untouched; the layer is carried disabled"
        );
        assert!(
            body.surface.params_of(SurfaceLayerType::Material).is_null(),
            "disabled ⇒ classifier defaults"
        );
    }

    #[test]
    fn insert_after_and_replace_are_id_keyed() {
        let p = pack(r#"Resource(( id: "fuel", density: 800.0 )),"#);
        // Seed an ordered list [a, b, c].
        let seed = override_set(
            "seed",
            "Patch",
            "",
            r#"( target: Id("fuel"), field: "tags", op: Set(List(["a", "b", "c"])) ),"#,
        );
        // Insert after `a`, then replace `b` with two elements — id-keyed, so the
        // earlier insert shifting positions must not affect the later target.
        let ins = override_set(
            "ins",
            "Scenario",
            "",
            r#"( target: Id("fuel"), field: "tags", op: InsertAfter(anchor: "a", items: ["x"]) ),"#,
        );
        let rep = override_set(
            "rep",
            "Local",
            "",
            r#"( target: Id("fuel"), field: "tags", op: Replace(target: "b", items: ["y", "z"]) ),"#,
        );
        let cat = Catalog::merge(&[&p], &[&seed, &ins, &rep]).unwrap();
        assert_eq!(tags_of(&cat, "fuel"), vec!["a", "x", "y", "z", "c"]);
        // Determinism: an independent merge of the same inputs resolves identically.
        let cat_again = Catalog::merge(&[&p], &[&seed, &ins, &rep]).unwrap();
        assert_eq!(tags_of(&cat, "fuel"), tags_of(&cat_again, "fuel"));

        // Insert after the last element appends at the tail.
        let tail = override_set(
            "tail",
            "Local",
            "",
            r#"( target: Id("fuel"), field: "tags", op: InsertAfter(anchor: "c", items: ["end"]) ),"#,
        );
        let cat = Catalog::merge(&[&p], &[&seed, &tail]).unwrap();
        assert_eq!(tags_of(&cat, "fuel"), vec!["a", "b", "c", "end"]);

        // Provenance: the list op records the displaced whole list.
        let cat = Catalog::merge(&[&p], &[&seed, &ins]).unwrap();
        let fp = &cat.get("fuel").unwrap().field_provenance["tags"];
        assert_eq!(
            fp.shadows[0].value,
            ProvValue::List(vec!["a".into(), "b".into(), "c".into()])
        );
    }

    #[test]
    fn insert_after_and_replace_absent_anchor_is_loud() {
        let p = pack(r#"Resource(( id: "fuel", density: 800.0 )),"#);
        let seed = override_set(
            "seed",
            "Patch",
            "",
            r#"( target: Id("fuel"), field: "tags", op: Set(List(["a"])) ),"#,
        );
        let ins = override_set(
            "ins",
            "Scenario",
            "",
            r#"( target: Id("fuel"), field: "tags", op: InsertAfter(anchor: "nope", items: ["x"]) ),"#,
        );
        match Catalog::merge(&[&p], &[&seed, &ins]) {
            Err(ContentError::AbsentElement { element, .. }) => assert_eq!(element, "nope"),
            other => panic!("expected AbsentElement, got {other:?}"),
        }
        let rep = override_set(
            "rep",
            "Scenario",
            "",
            r#"( target: Id("fuel"), field: "tags", op: Replace(target: "nope", items: ["x"]) ),"#,
        );
        match Catalog::merge(&[&p], &[&seed, &rep]) {
            Err(ContentError::AbsentElement { element, .. }) => assert_eq!(element, "nope"),
            other => panic!("expected AbsentElement, got {other:?}"),
        }
    }

    #[test]
    fn insert_after_on_wrong_slot_type_is_loud() {
        // `density` is a numeric slot, not a list.
        let p = pack(r#"Resource(( id: "fuel", density: 800.0 )),"#);
        let bad = override_set(
            "bad",
            "Patch",
            "",
            r#"( target: Id("fuel"), field: "density", op: InsertAfter(anchor: "a", items: ["x"]) ),"#,
        );
        match Catalog::merge(&[&p], &[&bad]) {
            Err(ContentError::TypeMismatch { op, .. }) => assert_eq!(op, "insert_after"),
            other => panic!("expected TypeMismatch, got {other:?}"),
        }
    }

    // WI 881: BodyRecipe as a content record kind — the canonical bodies authored
    // as composable RON, characterization-equal to the constructors. Since WI 884
    // the constructors *resolve the shipped pack*, so these inline fixtures are
    // the **independent literal pins**: they no longer share a source with the
    // constructors, and a drift in the shipped bodies.ron fails here.

    /// The earthlike body as a recipe record — independent literal pin (WI 884).
    /// WI 887: gravity and scale height are un-pinned here exactly as in the
    /// shipped pack (they derive: μ/R² and R·T/(M·g)); mean molar mass supplies
    /// the gas relation.
    fn earthlike_recipe() -> &'static str {
        r#"BodyRecipe((
            id: "earthlike", name: "Earth-like",
            mu: 3.986e14, radius: 6360000.0, rotation_period: 86164.0905,
            atmosphere_surface_density: 1.225, atmosphere_surface_pressure: 101325.0,
            mean_molar_mass: 0.0289644,
            ocean_surface_density: 1025.0, ocean_surface_pressure: 101325.0,
            ocean_density_gradient: 0.0,
            atmosphere_temperature: 288.15, ocean_temperature: 290.0,
            surface_seed: 0,
        )),"#
    }

    fn resolved_body(cat: &Catalog, id: &str) -> BodyAsset {
        match &cat.get(id).unwrap().record {
            Record::BodyRecipe(b) => b.body.clone(),
            other => panic!("expected body recipe, got {other:?}"),
        }
    }

    fn body_offset(b: &BodyAsset) -> Option<f64> {
        b.surface
            .params_of(crate::body_asset::SurfaceLayerType::Material)
            .get("temperature")
            .and_then(|v| v.as_f64())
    }

    /// Field-for-field approximate equality (RON-decimal → f64 is not bit-exact).
    fn assert_body_approx(got: &BodyAsset, want: &BodyAsset) {
        assert_eq!(got.id, want.id);
        assert_eq!(got.name, want.name);
        assert_physics_approx(got, want);
    }

    /// Approximate equality of everything the sampler/derivation computes, but
    /// **not** the authoring identity (`id`/`name`). Used to characterize a
    /// sampled recipe body against `bodygen::generate`, whose id/name are
    /// synthesized and differ from the recipe's authored ones by construction.
    fn assert_physics_approx(got: &BodyAsset, want: &BodyAsset) {
        let close = |x: f64, y: f64| (x - y).abs() <= 1e-6 * x.abs().max(1.0);
        assert!(close(got.mu, want.mu), "mu {} vs {}", got.mu, want.mu);
        assert!(close(got.radius, want.radius));
        assert_eq!(got.rotation.axis, want.rotation.axis);
        assert!(close(
            got.rotation.sidereal_period,
            want.rotation.sidereal_period
        ));
        let (g, w) = (&got.fluid_medium, &want.fluid_medium);
        for (x, y) in [
            (g.atmosphere_surface_density, w.atmosphere_surface_density),
            (g.atmosphere_surface_pressure, w.atmosphere_surface_pressure),
            (g.atmosphere_scale_height, w.atmosphere_scale_height),
            (g.ocean_surface_density, w.ocean_surface_density),
            (g.ocean_surface_pressure, w.ocean_surface_pressure),
            (g.ocean_density_gradient, w.ocean_density_gradient),
            (g.gravity, w.gravity),
            (g.atmosphere_temperature, w.atmosphere_temperature),
            (g.ocean_temperature, w.ocean_temperature),
        ] {
            assert!(close(x, y), "medium field {x} vs {y}");
        }
        assert_eq!(got.surface.seed, want.surface.seed);
        match (body_offset(got), body_offset(want)) {
            (None, None) => {}
            (Some(a), Some(b)) => assert!(close(a, b), "offset {a} vs {b}"),
            (a, b) => panic!("material offset presence mismatch: {a:?} vs {b:?}"),
        }
    }

    #[test]
    fn earthlike_recipe_resolves_equal_to_constructor() {
        let p = pack(earthlike_recipe());
        let cat = Catalog::merge(&[&p], &[]).unwrap();
        assert_body_approx(&resolved_body(&cat, "earthlike"), &BodyAsset::earthlike());
        // No offset authored ⇒ material is JSON null (matches the WI 875 earthlike).
        assert!(
            resolved_body(&cat, "earthlike").surface.layers.is_empty(),
            "no offset authored => empty stack (matches the WI 875 earthlike)"
        );
    }

    #[test]
    fn ice_age_recipe_is_inheritance_plus_offset() {
        // Ice-age = parent earthlike + a single cold surface offset; nothing else
        // re-authored. Physics must match the temperate twin exactly (inherited).
        use crate::body_asset::EARTHLIKE_ICE_AGE_OFFSET;
        let ice_age = r#"BodyRecipe(( id: "earthlike-ice-age", name: "Earth-like (Ice Age)",
            parent: "earthlike", surface_temperature_offset: -40.15 )),"#;
        let p = pack(&format!("{}{}", earthlike_recipe(), ice_age));
        let cat = Catalog::merge(&[&p], &[]).unwrap();
        let temperate = resolved_body(&cat, "earthlike");
        let cold = resolved_body(&cat, "earthlike-ice-age");

        // Same physics: medium / mu / radius / rotation inherited unchanged.
        assert_eq!(cold.fluid_medium, temperate.fluid_medium);
        assert_eq!(cold.mu, temperate.mu);
        assert_eq!(cold.radius, temperate.radius);
        assert_eq!(cold.rotation, temperate.rotation);
        // Only the classifier offset differs, and it matches the constructor.
        assert!(temperate.surface.layers.is_empty());
        assert!(
            (body_offset(&cold).unwrap() - EARTHLIKE_ICE_AGE_OFFSET).abs() < 0.05,
            "ice-age offset {:?}",
            body_offset(&cold)
        );
        // And the whole body matches the ice-age constructor.
        assert_body_approx(&cold, &BodyAsset::earthlike_ice_age());
    }

    #[test]
    fn body_recipe_participates_in_the_ladder() {
        let p = pack(earthlike_recipe());
        // A scenario override re-flows onto a recipe field, provenance-tracked.
        let ov = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("earthlike"), field: "radius", op: Multiply(2.0) ),"#,
        );
        let cat = Catalog::merge(&[&p], &[&ov]).unwrap();
        assert!((resolved_body(&cat, "earthlike").radius - 2.0 * 6_360_000.0).abs() < 1.0);
        let fp = &cat.get("earthlike").unwrap().field_provenance["radius"];
        assert_eq!(fp.shadows[0].value, ProvValue::Number(6_360_000.0));

        // Overriding a structural field is rejected.
        let bad_struct = override_set(
            "bad",
            "Patch",
            "",
            r#"( target: Id("earthlike"), field: "id", op: Set(Text("x")) ),"#,
        );
        assert!(matches!(
            Catalog::merge(&[&p], &[&bad_struct]),
            Err(ContentError::StructuralField { .. })
        ));

        // A type-mismatched op is rejected (multiply on the text `name`).
        let bad_type = override_set(
            "bad",
            "Patch",
            "",
            r#"( target: Id("earthlike"), field: "name", op: Multiply(2.0) ),"#,
        );
        assert!(matches!(
            Catalog::merge(&[&p], &[&bad_type]),
            Err(ContentError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn body_recipe_missing_required_field_is_loud() {
        let p = pack(r#"BodyRecipe(( id: "bare", name: "Bare" )),"#);
        match Catalog::merge(&[&p], &[]) {
            Err(ContentError::MissingField { id, .. }) => assert_eq!(id, "bare"),
            other => panic!("expected MissingField, got {other:?}"),
        }
    }

    // WI 883: archetypes as sampled recipes — an abstract base carrying a `shape`
    // + parameter bands, a concrete child supplying a seed; resolving samples the
    // bands and reproduces `bodygen::generate` for that shape. Since WI 884
    // `generate` reads the *shipped* bands, so these inline literal fixtures are
    // the independent pin welding the shipped bands to these values (the
    // stream itself is pinned by bodygen's golden test). Re-authored to the
    // WI 889 independent-set vocabulary, band-identical to the shipped pack.

    /// The three archetype base recipes (abstract), bands = independent literal
    /// copies of the shipped canonical bands (the WI 884 weld, re-authored at
    /// WI 889), paired with the `Archetype` the oracle `generate` uses.
    fn archetype_bases() -> [(&'static str, &'static str, Archetype); 3] {
        [
            (
                "moon",
                r#"BodyRecipe(( id: "moon", abstract: true, shape: moon,
                    radius_min: 2.0e5, radius_max: 2.0e6,
                    gravity_min: 0.5, gravity_max: 3.0,
                    rotation_period_min: 20000.0, rotation_period_max: 200000.0,
                    nominal_insolation_min: 200.0, nominal_insolation_max: 3000.0,
                    bond_albedo_min: 0.05, bond_albedo_max: 0.35 )),"#,
                Archetype::Moon,
            ),
            (
                "rocky",
                r#"BodyRecipe(( id: "rocky", abstract: true, shape: rocky_planet,
                    radius_min: 2.5e6, radius_max: 8.0e6,
                    gravity_min: 3.0, gravity_max: 12.0,
                    rotation_period_min: 20000.0, rotation_period_max: 200000.0,
                    atmosphere_surface_pressure_min: 50000.0, atmosphere_surface_pressure_max: 150000.0,
                    nominal_insolation_min: 800.0, nominal_insolation_max: 2000.0,
                    bond_albedo_min: 0.1, bond_albedo_max: 0.4,
                    greenhouse_delta_t_min: 5.0, greenhouse_delta_t_max: 40.0,
                    mean_molar_mass_min: 0.02, mean_molar_mass_max: 0.045 )),"#,
                Archetype::RockyPlanet,
            ),
            (
                "ocean",
                r#"BodyRecipe(( id: "ocean", abstract: true, shape: ocean_world,
                    radius_min: 3.0e6, radius_max: 9.0e6,
                    gravity_min: 5.0, gravity_max: 12.0,
                    rotation_period_min: 20000.0, rotation_period_max: 200000.0,
                    atmosphere_surface_pressure_min: 80000.0, atmosphere_surface_pressure_max: 180000.0,
                    nominal_insolation_min: 1400.0, nominal_insolation_max: 2000.0,
                    bond_albedo_min: 0.15, bond_albedo_max: 0.3,
                    greenhouse_delta_t_min: 25.0, greenhouse_delta_t_max: 40.0,
                    mean_molar_mass_min: 0.018, mean_molar_mass_max: 0.03,
                    ocean_surface_density_min: 950.0, ocean_surface_density_max: 1100.0,
                    ocean_temperature_min: 275.0, ocean_temperature_max: 300.0 )),"#,
                Archetype::OceanWorld,
            ),
        ]
    }

    #[test]
    fn archetype_recipe_reproduces_generate() {
        // Seeds chosen to be exactly representable as f64 (the recipe seed field is
        // numeric — WI 881's Num-seed tradeoff), so the recipe seed round-trips to
        // the same u64 the oracle draws from. (u64::MAX would not round-trip.)
        for (base_id, base_ron, arch) in archetype_bases() {
            for seed in [0u64, 1, 42, 7777, 4_503_599_627_370_496] {
                let child = format!(
                    r#"BodyRecipe(( id: "body", name: "Body", parent: "{base_id}", surface_seed: {seed} )),"#
                );
                let p = pack(&format!("{base_ron}{child}"));
                let cat = Catalog::merge(&[&p], &[]).unwrap();
                let got = resolved_body(&cat, "body");
                // Physics is the seeded draw; id/name are the recipe's, not the
                // generator's synthesized ones (excluded from the comparison).
                assert_physics_approx(&got, &bodygen::generate(seed, arch));
                assert_eq!(got.id, "body");
            }
        }
    }

    #[test]
    fn sampled_recipe_is_deterministic() {
        let (base_id, base_ron, _) = archetype_bases()[2]; // ocean (most draws)
        let child = format!(
            r#"BodyRecipe(( id: "body", name: "Body", parent: "{base_id}", surface_seed: 314159 )),"#
        );
        let p = pack(&format!("{base_ron}{child}"));
        let a = resolved_body(&Catalog::merge(&[&p], &[]).unwrap(), "body");
        let b = resolved_body(&Catalog::merge(&[&p], &[]).unwrap(), "body");
        assert_eq!(a, b, "same recipe + seed ⇒ byte-identical body");
    }

    /// A concrete sampled body authoring its own shape + bands (no parent), so the
    /// band fields sit directly on a catalog record and are overridable there.
    fn standalone_rocky() -> &'static str {
        r#"BodyRecipe(( id: "r1", name: "R1", shape: rocky_planet, surface_seed: 5,
            radius_min: 2.5e6, radius_max: 8.0e6,
            gravity_min: 3.0, gravity_max: 12.0,
            rotation_period_min: 20000.0, rotation_period_max: 200000.0,
            atmosphere_surface_pressure_min: 50000.0, atmosphere_surface_pressure_max: 150000.0,
            nominal_insolation_min: 800.0, nominal_insolation_max: 2000.0,
            bond_albedo_min: 0.1, bond_albedo_max: 0.4,
            greenhouse_delta_t_min: 5.0, greenhouse_delta_t_max: 40.0,
            mean_molar_mass_min: 0.02, mean_molar_mass_max: 0.045 )),"#
    }

    #[test]
    fn archetype_bands_compose_with_overrides() {
        let p = pack(standalone_rocky());
        // `set` both radius bounds to a fixed value (collapse the band); `multiply`
        // the gravity upper bound. Both are ordinary scalar ops on band fields.
        let ov = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("r1"), field: "radius_min", op: Set(Number(4000000.0)) ),
               ( target: Id("r1"), field: "radius_max", op: Set(Number(4000000.0)) ),
               ( target: Id("r1"), field: "gravity_max", op: Multiply(0.5) ),
               ( target: Id("r1"), field: "nominal_insolation_max", op: Multiply(0.5) ),"#,
        );
        let cat = Catalog::merge(&[&p], &[&ov]).unwrap();
        let body = resolved_body(&cat, "r1");
        // Zero-width radius band ⇒ the draw returns exactly the pinned value.
        assert!((body.radius - 4.0e6).abs() < 1.0, "radius {}", body.radius);
        // gravity band is now [3, 6]; the sampled g (= medium.gravity) lies within.
        assert!(
            (3.0..=6.0).contains(&body.fluid_medium.gravity),
            "g {}",
            body.fluid_medium.gravity
        );
        // Provenance: the multiplies shadowed the authored upper bounds — the
        // WI 889 independent-set bands are ordinary ladder-tunable fields.
        let fp = &cat.get("r1").unwrap().field_provenance["gravity_max"];
        assert_eq!(fp.shadows[0].value, ProvValue::Number(12.0));
        let fp = &cat.get("r1").unwrap().field_provenance["nominal_insolation_max"];
        assert_eq!(fp.shadows[0].value, ProvValue::Number(2000.0));
    }

    #[test]
    fn overriding_shape_is_rejected() {
        let p = pack(standalone_rocky());
        let bad = override_set(
            "bad",
            "Patch",
            "",
            r#"( target: Id("r1"), field: "shape", op: Set(Text("moon")) ),"#,
        );
        assert!(matches!(
            Catalog::merge(&[&p], &[&bad]),
            Err(ContentError::StructuralField { .. })
        ));
    }

    #[test]
    fn sampled_recipe_missing_band_is_loud() {
        // A rocky-shaped recipe that omits the atmosphere bands its shape draws.
        let p = pack(
            r#"BodyRecipe(( id: "r1", name: "R1", shape: rocky_planet, surface_seed: 1,
                radius_min: 2.5e6, radius_max: 8.0e6, gravity_min: 3.0, gravity_max: 12.0,
                rotation_period_min: 20000.0, rotation_period_max: 200000.0 )),"#,
        );
        match Catalog::merge(&[&p], &[]) {
            Err(ContentError::MissingField { id, field }) => {
                assert_eq!(id, "r1");
                // The drawn independents are checked first (WI 889); the first
                // missing band its shape requires is named.
                assert_eq!(field, "nominal_insolation_min");
            }
            other => panic!("expected MissingField, got {other:?}"),
        }
    }

    /// WI 889: the drawn-independent bands respect their physical domains —
    /// a violating bound is loud, typed, and names record + bound.
    #[test]
    fn band_domain_violations_are_loud_and_named() {
        let moon_with = |albedo_max: &str, insolation: &str| {
            pack(&format!(
                r#"BodyRecipe(( id: "m1", name: "M", shape: moon, surface_seed: 1,
                    radius_min: 2.0e5, radius_max: 2.0e6,
                    gravity_min: 0.5, gravity_max: 3.0,
                    rotation_period_min: 20000.0, rotation_period_max: 200000.0,
                    {insolation},
                    bond_albedo_min: 0.05, bond_albedo_max: {albedo_max} )),"#
            ))
        };
        let expect_unphysical = |p: String, want: &str| match Catalog::merge(&[&p], &[]) {
            Err(ContentError::UnphysicalValue { id, field }) => {
                assert_eq!(id, "m1");
                assert_eq!(field, want);
            }
            other => panic!("expected UnphysicalValue({want}), got {other:?}"),
        };
        // Albedo touching 1.0: `1 − A` must stay positive for T_eq.
        expect_unphysical(
            moon_with(
                "1.0",
                "nominal_insolation_min: 200.0, nominal_insolation_max: 3000.0",
            ),
            "bond_albedo_max",
        );
        // Negative insolation.
        expect_unphysical(
            moon_with(
                "0.35",
                "nominal_insolation_min: -1.0, nominal_insolation_max: 3000.0",
            ),
            "nominal_insolation_min",
        );
        // Inverted band (min > max).
        expect_unphysical(
            moon_with(
                "0.35",
                "nominal_insolation_min: 3000.0, nominal_insolation_max: 200.0",
            ),
            "nominal_insolation_min",
        );
        // Rocky's atmosphere trio: negative greenhouse; non-positive molar mass.
        let rocky_with = |greenhouse_min: &str, molar_min: &str| {
            pack(&format!(
                r#"BodyRecipe(( id: "m1", name: "R", shape: rocky_planet, surface_seed: 1,
                    radius_min: 2.5e6, radius_max: 8.0e6,
                    gravity_min: 3.0, gravity_max: 12.0,
                    rotation_period_min: 20000.0, rotation_period_max: 200000.0,
                    atmosphere_surface_pressure_min: 50000.0, atmosphere_surface_pressure_max: 150000.0,
                    nominal_insolation_min: 800.0, nominal_insolation_max: 2000.0,
                    bond_albedo_min: 0.1, bond_albedo_max: 0.4,
                    greenhouse_delta_t_min: {greenhouse_min}, greenhouse_delta_t_max: 40.0,
                    mean_molar_mass_min: {molar_min}, mean_molar_mass_max: 0.045 )),"#
            ))
        };
        expect_unphysical(rocky_with("-5.0", "0.02"), "greenhouse_delta_t_min");
        expect_unphysical(rocky_with("5.0", "0.0"), "mean_molar_mass_min");
    }

    /// WI 889: ocean gating is one shared decision across both resolve arms —
    /// a sampled ocean world drawn into the frozen regime loses its liquid
    /// exactly as a fixed recipe does, while the canonical ocean bands clear
    /// the gate at every corner of their band box.
    #[test]
    fn sampled_ocean_gating_is_uniform_with_the_fixed_arm() {
        let p = pack(
            r#"BodyRecipe(( id: "cold", name: "Cold", shape: ocean_world, surface_seed: 3,
                radius_min: 3.0e6, radius_max: 9.0e6,
                gravity_min: 5.0, gravity_max: 12.0,
                rotation_period_min: 20000.0, rotation_period_max: 200000.0,
                atmosphere_surface_pressure_min: 80000.0, atmosphere_surface_pressure_max: 180000.0,
                nominal_insolation_min: 50.0, nominal_insolation_max: 60.0,
                bond_albedo_min: 0.25, bond_albedo_max: 0.3,
                greenhouse_delta_t_min: 0.0, greenhouse_delta_t_max: 1.0,
                mean_molar_mass_min: 0.018, mean_molar_mass_max: 0.03,
                ocean_surface_density_min: 950.0, ocean_surface_density_max: 1100.0,
                ocean_temperature_min: 275.0, ocean_temperature_max: 300.0 )),"#,
        );
        let body = resolved_body(&Catalog::merge(&[&p], &[]).unwrap(), "cold");
        let m = &body.fluid_medium;
        assert!(
            m.atmosphere_temperature < OCEAN_FREEZE_THRESHOLD_K,
            "fixture must land frozen (T {})",
            m.atmosphere_temperature
        );
        assert_eq!(m.ocean_surface_density, 0.0, "frozen ⇒ no liquid ocean");
        assert_eq!(m.ocean_surface_pressure, 0.0);
        assert_eq!(m.ocean_density_gradient, 0.0);
        assert!(
            m.atmosphere_surface_density > 0.0,
            "the atmosphere survives the gate"
        );

        // Canonical ocean bands: every (S, A, ΔT) corner resolves above the
        // freeze gate — the shipped population always keeps its ocean.
        let bands = canonical_bands(Archetype::OceanWorld);
        for s in [bands.nominal_insolation.0, bands.nominal_insolation.1] {
            for a in [bands.bond_albedo.0, bands.bond_albedo.1] {
                for dt in [bands.greenhouse_delta_t.0, bands.greenhouse_delta_t.1] {
                    let t = body_derive::surface_temperature(
                        body_derive::equilibrium_temperature(s, a),
                        dt,
                    );
                    assert!(
                        t > OCEAN_FREEZE_THRESHOLD_K,
                        "canonical corner (S={s}, A={a}, dT={dt}) freezes at {t} K"
                    );
                }
            }
        }
    }

    /// WI 889 (plan-review H1): a stale `format: 1` pack authoring the
    /// RETIRED band vocabulary — exactly the document class the grammar bump
    /// exists for — is refused **by version** via the header-first probe, not
    /// by an incidental unknown-field parse error.
    #[test]
    fn stale_format_pack_is_refused_by_version_not_parse() {
        let src = r#"(
            format: 1, id: "old", version: "1",
            records: [ BodyRecipe(( id: "r1", name: "R1", shape: rocky_planet, surface_seed: 1,
                radius_min: 2.5e6, radius_max: 8.0e6, gravity_min: 3.0, gravity_max: 12.0,
                rotation_period_min: 20000.0, rotation_period_max: 200000.0,
                atmosphere_surface_pressure_min: 50000.0, atmosphere_surface_pressure_max: 150000.0,
                atmosphere_surface_density_min: 0.2, atmosphere_surface_density_max: 2.0,
                atmosphere_scale_height_min: 5000.0, atmosphere_scale_height_max: 12000.0,
                atmosphere_temperature_min: 220.0, atmosphere_temperature_max: 300.0 )) ],
        )"#;
        match Catalog::merge(&[src], &[]) {
            Err(ContentError::Format { found: 1 }) => {}
            other => panic!("expected Format {{ found: 1 }}, got {other:?}"),
        }
    }

    // WI 884: the shipped canonical bodies — the embedded pack is the single
    // authored source; these tests weld it to the engine physics constants and
    // gate the build-integrity `expect`s in the canonical accessors.

    #[test]
    fn embedded_canonical_pack_welds_to_the_physics_consts() {
        use crate::body_asset::EARTHLIKE_ICE_AGE_OFFSET;
        use crate::sim::CentralBody;

        // The temperate earthlike: recipe values must equal the engine physics
        // constants **exactly** (the RON transcribes the same digit strings, and
        // decimal→f64 parsing is correctly rounded on both sides).
        let e = canonical_body("earthlike");
        assert_eq!(e.id, "earthlike");
        assert_eq!(e.mu, CentralBody::EARTHLIKE.mu);
        assert_eq!(e.radius, CentralBody::EARTHLIKE.radius);
        assert_eq!(e.rotation, Rotation::EARTHLIKE);
        assert_eq!(e.fluid_medium, FluidMedium::EARTHLIKE);
        assert_eq!(e.surface.seed, 0);
        assert!(e.surface.layers.is_empty());

        // The ice-age sibling: inherited physics identical to the twin; its
        // authored offset pins to the WI-875 derived constant (tolerance covers
        // the authored `-40.15` vs the computed `248 - 288.15` — ≲1e-13 apart).
        let i = canonical_body("earthlike-ice-age");
        assert_eq!(i.fluid_medium, e.fluid_medium);
        assert_eq!(i.mu, e.mu);
        assert_eq!(i.radius, e.radius);
        assert_eq!(i.rotation, e.rotation);
        let offset = i
            .surface
            .params_of(crate::body_asset::SurfaceLayerType::Material)["temperature"]
            .as_f64()
            .unwrap();
        assert!(
            (offset - EARTHLIKE_ICE_AGE_OFFSET).abs() < 1e-9,
            "shipped ice-age offset {offset} drifted from the derived constant \
             {EARTHLIKE_ICE_AGE_OFFSET} — update bodies.ron"
        );

        // Every archetype record retains ladder-resolved bands (spot-check the
        // moon's historic radius band exactly).
        for arch in Archetype::ALL {
            let _ = canonical_bands(arch); // panics if missing/not shaped
        }
        assert_eq!(canonical_bands(Archetype::Moon).radius, (2.0e5, 2.0e6));
    }

    #[test]
    fn shaped_recipe_rejects_authored_fixed_scalars() {
        // A shaped recipe authoring a fixed scalar is a loud authoring error
        // (WI 884) — the sampled body's fields come from the draw, not scalars.
        let p = pack(
            r#"BodyRecipe(( id: "r1", name: "R1", shape: moon, surface_seed: 1, mu: 1.0e12,
                radius_min: 2.0e5, radius_max: 2.0e6, gravity_min: 0.5, gravity_max: 3.0,
                rotation_period_min: 20000.0, rotation_period_max: 200000.0 )),"#,
        );
        match Catalog::merge(&[&p], &[]) {
            Err(ContentError::InapplicableField { id, field }) => {
                assert_eq!(id, "r1");
                assert_eq!(field, "mu");
            }
            other => panic!("expected InapplicableField, got {other:?}"),
        }

        // …and the same rejection fires when an override writes the scalar onto
        // a shaped record (validation runs after the ladder).
        let p = pack(standalone_rocky());
        let ov = override_set(
            "bad",
            "Scenario",
            "",
            r#"( target: Id("r1"), field: "mu", op: Set(Number(1.0e12)) ),"#,
        );
        assert!(matches!(
            Catalog::merge(&[&p], &[&ov]),
            Err(ContentError::InapplicableField { .. })
        ));
    }

    #[test]
    fn fixed_recipe_rejects_authored_band_fields() {
        // Bands are meaningless without a sampler: a fixed (shapeless) recipe
        // authoring a band bound is rejected symmetrically (WI 884).
        let p = pack(
            r#"BodyRecipe(( id: "b1", name: "B1", radius_min: 1.0,
                mu: 3.986e14, radius: 6360000.0, rotation_period: 86164.0905,
                atmosphere_surface_density: 1.225, atmosphere_surface_pressure: 101325.0,
                atmosphere_scale_height: 8500.0, ocean_surface_density: 1025.0,
                ocean_surface_pressure: 101325.0, ocean_density_gradient: 0.0,
                gravity: 9.81, atmosphere_temperature: 288.15, ocean_temperature: 290.0,
                surface_seed: 0 )),"#,
        );
        match Catalog::merge(&[&p], &[]) {
            Err(ContentError::InapplicableField { id, field }) => {
                assert_eq!(id, "b1");
                assert_eq!(field, "radius_min");
            }
            other => panic!("expected InapplicableField, got {other:?}"),
        }
    }

    // WI 886: the derivation pass — fixed recipes author independents (or pins);
    // derived medium fields are computed by the named relations in
    // `body_derive`, pins hold exactly, gating decides ocean presence last.

    /// A fixed recipe with the full independent set and NO pins — every derived
    /// field must come from its relation. `{extra}` appends/overrides fields.
    fn derive_recipe(extra: &str) -> String {
        format!(
            r#"BodyRecipe(( id: "d1", name: "D1",
                mu: 4.0e14, radius: 6.0e6, rotation_period: 86400.0,
                atmosphere_surface_pressure: 100000.0,
                nominal_insolation: 1361.0, bond_albedo: 0.3, greenhouse_delta_t: 33.0,
                mean_molar_mass: 0.029,
                ocean_surface_density: 1000.0, ocean_density_gradient: 0.0,
                ocean_temperature: 285.0, surface_seed: 7, {extra} )),"#
        )
    }

    // ------------------------------------------------------------------
    // WI 880 — `suppress`: do-not-generate on the sampled path (design I4).
    // ------------------------------------------------------------------

    fn shaped_child(base_id: &str, extra: &str) -> String {
        format!(
            r#"BodyRecipe(( id: "body", name: "Body", parent: "{base_id}", surface_seed: 42, {extra} )),"#
        )
    }

    fn shaped_body(base_ron: &str, base_id: &str, extra: &str) -> BodyAsset {
        let p = pack(&format!("{base_ron}{}", shaped_child(base_id, extra)));
        resolved_body(&Catalog::merge(&[&p], &[]).unwrap(), "body")
    }

    #[test]
    fn suppress_pins_the_scalar_and_shifts_no_other_draw() {
        let (base_id, base_ron, _) = archetype_bases()[1]; // rocky
        let baseline = shaped_body(base_ron, base_id, "");
        let pinned = shaped_body(
            base_ron,
            base_id,
            r#"suppress: ["bond_albedo"], bond_albedo: 0.123,"#,
        );
        // Every unsuppressed stream is bit-untouched (WI 889 per-field
        // seeding: the vetoed field's stream is simply never opened)...
        assert_eq!(pinned.radius, baseline.radius);
        assert_eq!(pinned.mu, baseline.mu);
        assert_eq!(pinned.rotation, baseline.rotation);
        assert_eq!(
            pinned.fluid_medium.atmosphere_surface_pressure,
            baseline.fluid_medium.atmosphere_surface_pressure
        );
        // ...while the albedo-derived temperature moved.
        assert_ne!(
            pinned.fluid_medium.atmosphere_temperature,
            baseline.fluid_medium.atmosphere_temperature
        );
        // Suppressing the full thermal set determines T_surf exactly: the
        // explicit values feed the same shared relations, verbatim.
        let all = shaped_body(
            base_ron,
            base_id,
            r#"suppress: ["nominal_insolation", "bond_albedo", "greenhouse_delta_t"],
               nominal_insolation: 1361.0, bond_albedo: 0.3, greenhouse_delta_t: 33.0,"#,
        );
        assert_eq!(
            all.fluid_medium.atmosphere_temperature,
            body_derive::surface_temperature(
                body_derive::equilibrium_temperature(1361.0, 0.3),
                33.0
            )
        );
        assert_eq!(all.radius, baseline.radius);
    }

    #[test]
    fn suppressed_zero_pressure_is_a_coherent_airless_body() {
        // SpaceEngine's `NoAtmosphere`, spelled as data: suppress the drawn
        // pressure and author 0 — the derivation seam does the rest (airless
        // density, and `gate_ocean` closes the ocean at p_atm == 0).
        let (base_id, base_ron, _) = archetype_bases()[2]; // ocean world
        let baseline = shaped_body(base_ron, base_id, "");
        assert!(
            baseline.fluid_medium.ocean_surface_density > 0.0,
            "fixture bands must gate the baseline ocean open"
        );
        let airless = shaped_body(
            base_ron,
            base_id,
            r#"suppress: ["atmosphere_surface_pressure"], atmosphere_surface_pressure: 0.0,"#,
        );
        let m = &airless.fluid_medium;
        assert_eq!(m.atmosphere_surface_pressure, 0.0);
        assert_eq!(m.atmosphere_surface_density, 0.0);
        assert_eq!(m.ocean_surface_density, 0.0);
        assert_eq!(m.ocean_surface_pressure, 0.0);
        assert_eq!(m.ocean_density_gradient, 0.0);
        // The unsuppressed streams didn't move; the inert ocean temperature
        // is still its own draw.
        assert_eq!(airless.radius, baseline.radius);
        assert_eq!(m.ocean_temperature, baseline.fluid_medium.ocean_temperature);
    }

    #[test]
    fn suppress_op_rides_the_ladder_with_its_own_provenance() {
        let (base_id, base_ron, _) = archetype_bases()[1];
        let p = pack(&format!("{base_ron}{}", shaped_child(base_id, "")));
        // A base-targeted suppress fans out to the variant's raw record; the
        // explicit value arrives as an ordinary ladder Set on the scalar the
        // wall now admits.
        let s = override_set(
            "scn",
            "Scenario",
            "",
            &format!(
                r#"( target: Base("{base_id}"), field: "bond_albedo", op: Suppress ),
                   ( target: Id("body"), field: "bond_albedo", op: Set(Number(0.2)) ),"#
            ),
        );
        let via_ladder = resolved_body(&Catalog::merge(&[&p], &[&s]).unwrap(), "body");
        // Semantics identical to authoring the suppression on the record.
        let via_record = shaped_body(
            base_ron,
            base_id,
            r#"suppress: ["bond_albedo"], bond_albedo: 0.2,"#,
        );
        assert_eq!(via_ladder, via_record);
        // The marker's provenance: its own `suppress` key, sourced to the
        // override set — never `delete`'s pathway, never a value shadow of
        // the drawn field itself (AC: no conflation).
        let cat = Catalog::merge(&[&p], &[&s]).unwrap();
        let fp = &cat.get("body").unwrap().field_provenance["suppress"];
        assert_eq!(
            fp.source,
            SourceRef::Override {
                source: "scn".into(),
                phase: OverridePhase::Scenario
            }
        );
        assert_eq!(
            fp.shadows,
            vec![Shadow {
                value: ProvValue::Unset,
                source: SourceRef::Pack { id: "test".into() }
            }]
        );
    }

    #[test]
    fn suppress_inherits_whole_value_like_every_field() {
        let (base_id, base_ron, _) = archetype_bases()[1];
        // The base is abstract, so hang the suppression on a concrete middle
        // record and inherit from it.
        let mid = format!(
            r#"BodyRecipe(( id: "mid", name: "Mid", parent: "{base_id}", surface_seed: 42,
                suppress: ["bond_albedo"], bond_albedo: 0.2 )),"#
        );
        let child = r#"BodyRecipe(( id: "body", name: "Body", parent: "mid" )),"#;
        let p = pack(&format!("{base_ron}{mid}{child}"));
        let cat = Catalog::merge(&[&p], &[]).unwrap();
        // The child inherits list + scalar together and resolves identically
        // to authoring both directly.
        assert_eq!(
            resolved_body(&cat, "body").fluid_medium,
            shaped_body(
                base_ron,
                base_id,
                r#"suppress: ["bond_albedo"], bond_albedo: 0.2,"#
            )
            .fluid_medium
        );
        // A child declaring its OWN suppress list replaces the parent's
        // whole-value (the uniform field rule) — the still-inherited albedo
        // scalar is then no longer admitted, and the wall says so loudly
        // rather than half-merging the two lists.
        let child2 = r#"BodyRecipe(( id: "body2", name: "Body2", parent: "mid",
            suppress: ["greenhouse_delta_t"], greenhouse_delta_t: 10.0 )),"#;
        let p2 = pack(&format!("{base_ron}{mid}{child2}"));
        match Catalog::merge(&[&p2], &[]) {
            Err(ContentError::InapplicableField { id, field }) => {
                assert_eq!(id, "body2");
                assert_eq!(field, "bond_albedo");
            }
            other => panic!("expected InapplicableField(bond_albedo), got {other:?}"),
        }
    }

    #[test]
    fn suppress_twice_is_idempotent() {
        // Plan edge case: marking an already-suppressed field (authored + op,
        // with a duplicated authored entry for good measure) is one marker,
        // no error, and both sources stay visible in the ledger chain.
        let (base_id, base_ron, _) = archetype_bases()[1];
        let p = pack(&format!(
            "{base_ron}{}",
            shaped_child(
                base_id,
                r#"suppress: ["bond_albedo", "bond_albedo"], bond_albedo: 0.2,"#
            )
        ));
        let s = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("body"), field: "bond_albedo", op: Suppress ),"#,
        );
        let cat = Catalog::merge(&[&p], &[&s]).unwrap();
        assert_eq!(
            resolved_body(&cat, "body"),
            shaped_body(
                base_ron,
                base_id,
                r#"suppress: ["bond_albedo"], bond_albedo: 0.2,"#
            )
        );
        // The redundant op still ledgers: the pack-authored list is the
        // shadow, the override set is the current source.
        let fp = &cat.get("body").unwrap().field_provenance["suppress"];
        assert_eq!(
            fp.source,
            SourceRef::Override {
                source: "scn".into(),
                phase: OverridePhase::Scenario
            }
        );
        assert_eq!(
            fp.shadows,
            vec![Shadow {
                value: ProvValue::List(vec!["bond_albedo".into(), "bond_albedo".into()]),
                source: SourceRef::Pack { id: "test".into() }
            }]
        );
    }

    #[test]
    fn suppress_misuse_is_loud_and_typed() {
        let (base_id, base_ron, _) = archetype_bases()[1];
        let (moon_id, moon_ron, _) = archetype_bases()[0];
        // (a) A fixed recipe has no generator to veto.
        let fixed = pack(&derive_recipe(r#"suppress: ["bond_albedo"],"#));
        match Catalog::merge(&[&fixed], &[]) {
            Err(ContentError::InapplicableField { id, field }) => {
                assert_eq!((id.as_str(), field), ("d1", "suppress"));
            }
            other => panic!("(a) expected InapplicableField(suppress), got {other:?}"),
        }
        // (b) The shape never draws it (recipe-authored path).
        let p = pack(&format!(
            "{moon_ron}{}",
            shaped_child(
                moon_id,
                r#"suppress: ["ocean_temperature"], ocean_temperature: 280.0,"#
            )
        ));
        match Catalog::merge(&[&p], &[]) {
            Err(ContentError::InapplicableField { id, field }) => {
                assert_eq!((id.as_str(), field), ("body", "ocean_temperature"));
            }
            other => panic!("(b) expected InapplicableField(ocean_temperature), got {other:?}"),
        }
        // (c) Not a drawn field anywhere: derived quantities are not
        // suppressible (design I2) — recipe-authored path.
        let p = pack(&format!(
            "{base_ron}{}",
            shaped_child(base_id, r#"suppress: ["mu"],"#)
        ));
        match Catalog::merge(&[&p], &[]) {
            Err(ContentError::NotSuppressible { record, field }) => {
                assert_eq!((record.as_str(), field.as_str()), ("body", "mu"));
            }
            other => panic!("(c) expected NotSuppressible(mu), got {other:?}"),
        }
        // (d) Same refusal at ladder time.
        let p = pack(&format!("{base_ron}{}", shaped_child(base_id, "")));
        let s = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("body"), field: "mu", op: Suppress ),"#,
        );
        match Catalog::merge(&[&p], &[&s]) {
            Err(ContentError::NotSuppressible { record, field }) => {
                assert_eq!((record.as_str(), field.as_str()), ("body", "mu"));
            }
            other => panic!("(d) expected NotSuppressible(mu), got {other:?}"),
        }
        // (e) Suppress aimed at a kind with no generator at all.
        let p = pack(&format!(
            "{base_ron}{}Resource(( id: \"fuel\", density: 800.0 )),",
            shaped_child(base_id, "")
        ));
        let s = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("fuel"), field: "bond_albedo", op: Suppress ),"#,
        );
        match Catalog::merge(&[&p], &[&s]) {
            Err(ContentError::UnknownField { record, field }) => {
                assert_eq!((record.as_str(), field.as_str()), ("fuel", "bond_albedo"));
            }
            other => panic!("(e) expected UnknownField, got {other:?}"),
        }
        // (f) Suppressed with no explicit source: the scalar is now required.
        let p = pack(&format!(
            "{base_ron}{}",
            shaped_child(base_id, r#"suppress: ["bond_albedo"],"#)
        ));
        match Catalog::merge(&[&p], &[]) {
            Err(ContentError::MissingField { id, field }) => {
                assert_eq!((id.as_str(), field), ("body", "bond_albedo"));
            }
            other => panic!("(f) expected MissingField(bond_albedo), got {other:?}"),
        }
        // (g) The explicit scalar obeys the band domain — one albedo domain
        // across every arm (WI 889/893 posture).
        let p = pack(&format!(
            "{base_ron}{}",
            shaped_child(base_id, r#"suppress: ["bond_albedo"], bond_albedo: 1.0,"#)
        ));
        match Catalog::merge(&[&p], &[]) {
            Err(ContentError::UnphysicalValue { id, field }) => {
                assert_eq!((id.as_str(), field), ("body", "bond_albedo"));
            }
            other => panic!("(g) expected UnphysicalValue(bond_albedo), got {other:?}"),
        }
        // (h) A scalar with NO matching suppress stays exactly as loud as
        // WI 884 left it — presence never implies suppression (design I4).
        let p = pack(&format!(
            "{base_ron}{}",
            shaped_child(base_id, r#"bond_albedo: 0.3,"#)
        ));
        match Catalog::merge(&[&p], &[]) {
            Err(ContentError::InapplicableField { id, field }) => {
                assert_eq!((id.as_str(), field), ("body", "bond_albedo"));
            }
            other => panic!("(h) expected InapplicableField(bond_albedo), got {other:?}"),
        }
        // (i) The suppress list itself is not a generic op target: the
        // dedicated op is the only ladder spelling (no delete conflation).
        let p = pack(&format!("{base_ron}{}", shaped_child(base_id, "")));
        let s = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("body"), field: "suppress", op: Extend(["bond_albedo"]) ),"#,
        );
        match Catalog::merge(&[&p], &[&s]) {
            Err(ContentError::TypeMismatch { record, field, op }) => {
                assert_eq!(
                    (record.as_str(), field.as_str(), op),
                    ("body", "suppress", "extend")
                );
            }
            other => panic!("(i) expected TypeMismatch(suppress/extend), got {other:?}"),
        }
    }

    #[test]
    fn fixed_recipe_derives_medium_from_independents() {
        let p = pack(&derive_recipe(""));
        let m = resolved_body(&Catalog::merge(&[&p], &[]).unwrap(), "d1").fluid_medium;
        // Bit-exact against the relations (validate runs the same functions on
        // the same inputs).
        let g = crate::body_derive::surface_gravity(4.0e14, 6.0e6);
        let t = crate::body_derive::surface_temperature(
            crate::body_derive::equilibrium_temperature(1361.0, 0.3),
            33.0,
        );
        assert_eq!(m.gravity, g);
        assert_eq!(m.atmosphere_temperature, t);
        assert!(t > OCEAN_FREEZE_THRESHOLD_K, "fixture must not gate");
        assert_eq!(
            m.atmosphere_surface_density,
            crate::body_derive::atmosphere_surface_density(100_000.0, 0.029, t)
        );
        assert_eq!(
            m.atmosphere_scale_height,
            crate::body_derive::scale_height(t, 0.029, g)
        );
        // Ocean pressure derives by continuity; independents pass through.
        assert_eq!(m.ocean_surface_pressure, 100_000.0);
        assert_eq!(m.ocean_surface_density, 1_000.0);
        assert_eq!(m.ocean_temperature, 285.0);
    }

    /// WI 891 (parked decision (b), settled): the authored Num seed is
    /// validated at its u64 cast — negative, fractional, and beyond-2^53
    /// values are loud, never silently saturated/truncated/imprecise. The
    /// ceiling itself (2^53, exactly representable) is allowed.
    #[test]
    fn unphysical_surface_seeds_are_loud() {
        let recipe = |seed: &str| {
            format!(
                r#"BodyRecipe(( id: "s1", name: "S1",
                    mu: 4.0e14, radius: 6.0e6, rotation_period: 86400.0,
                    atmosphere_surface_pressure: 100000.0,
                    nominal_insolation: 1361.0, bond_albedo: 0.3, greenhouse_delta_t: 33.0,
                    mean_molar_mass: 0.029,
                    ocean_surface_density: 1000.0, ocean_density_gradient: 0.0,
                    ocean_temperature: 285.0, surface_seed: {seed} )),"#
            )
        };
        for bad in ["-1", "1.5", "9007199254740994"] {
            let p = pack(&recipe(bad));
            match Catalog::merge(&[&p], &[]) {
                Err(ContentError::UnphysicalValue { id, field }) => {
                    assert_eq!(id, "s1");
                    assert_eq!(field, "surface_seed");
                }
                other => panic!("seed {bad}: expected UnphysicalValue, got {other:?}"),
            }
        }
        let ceiling = pack(&recipe("9007199254740992"));
        let cat = Catalog::merge(&[&ceiling], &[]).expect("2^53 exactly is allowed");
        assert_eq!(resolved_body(&cat, "s1").surface.seed, 1u64 << 53);
    }

    #[test]
    fn derivation_is_deterministic() {
        let p = pack(&derive_recipe(""));
        let a = resolved_body(&Catalog::merge(&[&p], &[]).unwrap(), "d1");
        let b = resolved_body(&Catalog::merge(&[&p], &[]).unwrap(), "d1");
        assert_eq!(a, b, "same recipe ⇒ bit-identical derived body");
    }

    #[test]
    fn pins_win_and_chain_through_derivation() {
        // Pin gravity and temperature; density/scale-height must derive from
        // the *effective* (pinned) upstream values — pin precedence composes
        // through the relation chain.
        let p = pack(&derive_recipe(
            "gravity: 5.0, atmosphere_temperature: 280.0,",
        ));
        let m = resolved_body(&Catalog::merge(&[&p], &[]).unwrap(), "d1").fluid_medium;
        assert_eq!(m.gravity, 5.0);
        assert_eq!(m.atmosphere_temperature, 280.0);
        assert_eq!(
            m.atmosphere_scale_height,
            crate::body_derive::scale_height(280.0, 0.029, 5.0)
        );

        // A ladder op on a pin is a re-pin, provenance-tracked; the chain
        // re-flows from the new pin.
        let ov = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("d1"), field: "gravity", op: Multiply(2.0) ),"#,
        );
        let cat = Catalog::merge(&[&p], &[&ov]).unwrap();
        let m2 = resolved_body(&cat, "d1").fluid_medium;
        assert_eq!(m2.gravity, 10.0);
        assert_eq!(
            m2.atmosphere_scale_height,
            crate::body_derive::scale_height(280.0, 0.029, 10.0)
        );
        let fp = &cat.get("d1").unwrap().field_provenance["gravity"];
        assert_eq!(fp.shadows[0].value, ProvValue::Number(5.0));
    }

    #[test]
    fn derivation_missing_inputs_are_loud() {
        // No temperature pin and an incomplete insolation chain: the error
        // names the missing *input* (more actionable than the derived field).
        let p = pack(
            r#"BodyRecipe(( id: "d1", name: "D1",
                mu: 4.0e14, radius: 6.0e6, rotation_period: 86400.0,
                atmosphere_surface_pressure: 100000.0,
                nominal_insolation: 1361.0, greenhouse_delta_t: 33.0,
                mean_molar_mass: 0.029,
                ocean_surface_density: 1000.0, ocean_density_gradient: 0.0,
                ocean_temperature: 285.0, surface_seed: 7 )),"#,
        );
        match Catalog::merge(&[&p], &[]) {
            Err(ContentError::MissingField { id, field }) => {
                assert_eq!(id, "d1");
                assert_eq!(field, "bond_albedo");
            }
            other => panic!("expected MissingField, got {other:?}"),
        }
    }

    #[test]
    fn unphysical_derivation_inputs_are_loud() {
        // A bond albedo above 1 would put a negative number under the fourth
        // root — rejected before any relation runs.
        let p = pack(
            r#"BodyRecipe(( id: "d1", name: "D1",
                mu: 4.0e14, radius: 6.0e6, rotation_period: 86400.0,
                atmosphere_surface_pressure: 100000.0,
                nominal_insolation: 1361.0, bond_albedo: 1.5, greenhouse_delta_t: 33.0,
                mean_molar_mass: 0.029,
                ocean_surface_density: 1000.0, ocean_density_gradient: 0.0,
                ocean_temperature: 285.0, surface_seed: 7 )),"#,
        );
        match Catalog::merge(&[&p], &[]) {
            Err(ContentError::UnphysicalValue { id, field }) => {
                assert_eq!(id, "d1");
                assert_eq!(field, "bond_albedo");
            }
            other => panic!("expected UnphysicalValue, got {other:?}"),
        }
    }

    #[test]
    fn fixed_albedo_domain_is_half_open_like_the_bands() {
        // WI 893: both resolve arms share one albedo domain, `[0, 1)`. The
        // fixture varies only the thermal fields; everything else is valid.
        let recipe = |thermal: &str| {
            pack(&format!(
                r#"BodyRecipe(( id: "d1", name: "D1",
                    mu: 4.0e14, radius: 6.0e6, rotation_period: 86400.0,
                    atmosphere_surface_pressure: 100000.0,
                    nominal_insolation: 1361.0, {thermal} greenhouse_delta_t: 33.0,
                    mean_molar_mass: 0.029,
                    ocean_surface_density: 1000.0, ocean_density_gradient: 0.0,
                    ocean_temperature: 285.0, surface_seed: 7 )),"#
            ))
        };
        // Exactly 1.0 on the derive path: refused loudly, naming the input —
        // not a silent T_eq = 0 K body (band parity: the shaped arm already
        // refuses any bound touching 1.0).
        match Catalog::merge(&[&recipe("bond_albedo: 1.0,")], &[]) {
            Err(ContentError::UnphysicalValue { id, field }) => {
                assert_eq!(id, "d1");
                assert_eq!(field, "bond_albedo");
            }
            other => panic!("expected UnphysicalValue(bond_albedo), got {other:?}"),
        }
        // Just inside the boundary: resolves as before (the domain is
        // half-open at exactly 1.0, nothing below it moved).
        Catalog::merge(&[&recipe("bond_albedo: 0.999,")], &[])
            .expect("albedo just below 1 resolves");
        // A pinned temperature short-circuits the derivation, so the unused
        // albedo is never consumed — the WI 886 posture, preserved.
        Catalog::merge(
            &[&recipe("bond_albedo: 1.0, atmosphere_temperature: 285.0,")],
            &[],
        )
        .expect("pinned T_surf never consumes the albedo");
    }

    #[test]
    fn ocean_gating_zeroes_the_trio() {
        // Frozen (pinned T at/below the freeze point): the ocean trio zeroes —
        // even a *pinned* ocean pressure (gating is a presence decision and
        // applies last) — while temperatures stay.
        let p = pack(&derive_recipe(
            "atmosphere_temperature: 200.0, ocean_surface_pressure: 100000.0,",
        ));
        let m = resolved_body(&Catalog::merge(&[&p], &[]).unwrap(), "d1").fluid_medium;
        assert_eq!(m.ocean_surface_density, 0.0);
        assert_eq!(m.ocean_surface_pressure, 0.0);
        assert_eq!(m.ocean_density_gradient, 0.0);
        assert_eq!(m.ocean_temperature, 285.0);

        // The WI-875 classifier offset is presentation-only: a warm medium
        // with a cold offset keeps its ocean (the ice-age contract).
        let p = pack(&derive_recipe(
            "atmosphere_temperature: 288.15, surface_temperature_offset: -40.15,",
        ));
        let m = resolved_body(&Catalog::merge(&[&p], &[]).unwrap(), "d1").fluid_medium;
        assert_eq!(m.ocean_surface_density, 1_000.0);

        // Airless (P₀ = 0): the skeleton (density 0, placeholder scale height)
        // plus a gated-off ocean; no molar mass or insolation chain needed
        // when the remaining derived fields are pinned or skeleton-supplied.
        let p = pack(
            r#"BodyRecipe(( id: "d1", name: "D1",
                mu: 4.0e14, radius: 6.0e6, rotation_period: 86400.0,
                atmosphere_surface_pressure: 0.0, atmosphere_temperature: 200.0,
                ocean_surface_density: 1000.0, ocean_density_gradient: 0.0,
                ocean_temperature: 200.0, surface_seed: 7 )),"#,
        );
        let m = resolved_body(&Catalog::merge(&[&p], &[]).unwrap(), "d1").fluid_medium;
        assert_eq!(m.atmosphere_surface_density, 0.0);
        assert_eq!(m.atmosphere_scale_height, 1.0);
        assert_eq!(m.ocean_surface_density, 0.0);
        assert_eq!(m.ocean_surface_pressure, 0.0);
    }

    #[test]
    fn derivation_inputs_inapplicable_on_shaped() {
        let p = pack(
            r#"BodyRecipe(( id: "r1", name: "R1", shape: moon, surface_seed: 1,
                nominal_insolation: 1361.0,
                radius_min: 2.0e5, radius_max: 2.0e6, gravity_min: 0.5, gravity_max: 3.0,
                rotation_period_min: 20000.0, rotation_period_max: 200000.0 )),"#,
        );
        match Catalog::merge(&[&p], &[]) {
            Err(ContentError::InapplicableField { id, field }) => {
                assert_eq!(id, "r1");
                assert_eq!(field, "nominal_insolation");
            }
            other => panic!("expected InapplicableField, got {other:?}"),
        }
    }

    #[test]
    fn extend_on_unset_with_parent_errors() {
        let p = pack(&format!("{}{}", engine_base(), engine_variant()));
        let s = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("v"), field: "tags", op: Extend(["x"]) ),"#,
        );
        assert!(matches!(
            Catalog::merge(&[&p], &[&s]),
            Err(ContentError::UnsetField { .. })
        ));
    }

    #[test]
    fn structural_and_unknown_fields_rejected() {
        let p = pack(&format!("{}{}", engine_base(), engine_variant()));
        let structural = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("v"), field: "parent", op: Set(Text("engine_base")) ),"#,
        );
        assert!(matches!(
            Catalog::merge(&[&p], &[&structural]),
            Err(ContentError::StructuralField { .. })
        ));
        let unknown = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("v"), field: "warp_factor", op: Set(Number(9.0)) ),"#,
        );
        match Catalog::merge(&[&p], &[&unknown]) {
            Err(ContentError::UnknownField { record, field }) => {
                assert_eq!((record.as_str(), field.as_str()), ("v", "warp_factor"));
            }
            other => panic!("expected UnknownField, got {other:?}"),
        }
    }

    #[test]
    fn type_mismatches_rejected() {
        let p = pack(r#"Resource(( id: "fuel", density: 800.0, tradable: true )),"#);
        let mul_bool = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("fuel"), field: "tradable", op: Multiply(2.0) ),"#,
        );
        match Catalog::merge(&[&p], &[&mul_bool]) {
            Err(ContentError::TypeMismatch { field, op, .. }) => {
                assert_eq!((field.as_str(), op), ("tradable", "multiply"));
            }
            other => panic!("expected TypeMismatch, got {other:?}"),
        }
        let set_wrong = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("fuel"), field: "density", op: Set(Bool(true)) ),"#,
        );
        assert!(matches!(
            Catalog::merge(&[&p], &[&set_wrong]),
            Err(ContentError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn selectors_by_kind_and_by_base() {
        let p = pack(&format!(
            "{}{}Device(( id: \"w\", parent: \"engine_base\", density: 1.0, max_mass_flow: 2.0 )),
             Resource(( id: \"fuel\", density: 800.0 )),",
            engine_base(),
            engine_variant()
        ));
        // By base: both inheritors get the tag (base excluded, kind-wide not touched).
        let by_base = override_set(
            "patch",
            "Patch",
            "",
            r#"( target: Base("engine_base"), field: "tags", op: Set(List(["engine"])) ),"#,
        );
        // By kind: every resource (one) gets a density bump.
        let by_kind = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Kind(Resource), field: "density", op: Multiply(1.25) ),"#,
        );
        let cat = Catalog::merge(&[&p], &[&by_base, &by_kind]).unwrap();
        for id in ["v", "w"] {
            match &cat.get(id).unwrap().record {
                Record::Device(d) => assert_eq!(d.tags, vec!["engine"]),
                other => panic!("expected device, got {other:?}"),
            }
        }
        match &cat.get("fuel").unwrap().record {
            Record::Resource(r) => assert_eq!(r.density, Some(1000.0)),
            other => panic!("expected resource, got {other:?}"),
        }
        // Zero-match by-kind is legal.
        let zero = override_set(
            "house",
            "Local",
            "",
            r#"( target: Kind(Body), field: "tags", op: Set(List(["x"])) ),"#,
        );
        assert!(Catalog::merge(&[&p], &[&zero]).is_ok());
    }

    #[test]
    fn unknown_targets_rejected() {
        let p = pack(engine_base());
        for target in [r#"Id("ghost")"#, r#"Base("ghost")"#] {
            let s = override_set(
                "scn",
                "Scenario",
                "",
                &format!(r#"( target: {target}, field: "tags", op: Set(List([])) ),"#),
            );
            assert!(matches!(
                Catalog::merge(&[&p], &[&s]),
                Err(ContentError::UnknownTarget { .. })
            ));
        }
    }

    #[test]
    fn source_errors_rejected() {
        let p1 = pack_named("alpha", "", "");
        // Duplicate source id (pack vs override set).
        let s_dup = override_set("alpha", "Scenario", "", "");
        assert!(matches!(
            Catalog::merge(&[&p1], &[&s_dup]),
            Err(ContentError::DuplicateSource { .. })
        ));
        // Duplicate record definition across packs.
        let pa = pack_named("alpha", "", r#"Resource(( id: "fuel" )),"#);
        let pb = pack_named("beta", "", r#"Resource(( id: "fuel" )),"#);
        assert!(matches!(
            Catalog::merge(&[&pa, &pb], &[]),
            Err(ContentError::DuplicateId { .. })
        ));
        // Unknown + cyclic dependencies.
        let p_unknown = pack_named("gamma", "depends: [\"ghost\"],", "");
        assert!(matches!(
            Catalog::merge(&[&p_unknown], &[]),
            Err(ContentError::UnknownDependency { .. })
        ));
        let c1 = pack_named("c1", "depends: [\"c2\"],", "");
        let c2 = pack_named("c2", "depends: [\"c1\"],", "");
        assert!(matches!(
            Catalog::merge(&[&c1, &c2], &[]),
            Err(ContentError::DependencyCycle { .. })
        ));
    }

    #[test]
    fn shipped_example_scenario_applies_over_core() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../content");
        let core = std::fs::read_to_string(root.join("packs/core.ron")).unwrap();
        let scn = std::fs::read_to_string(root.join("overrides/example-scenario.ron")).unwrap();
        let cat = Catalog::merge(&[&core], &[&scn]).expect("shipped composition merges");
        // 3200 × 0.9, re-flowed through the abstract base to the variant.
        assert_eq!(engine_ev(&cat, "lf_engine_small"), 3200.0 * 0.9);
        match &cat.get("liquid_fuel").unwrap().record {
            Record::Resource(r) => assert_eq!(r.tags, vec!["starter"]),
            other => panic!("expected resource, got {other:?}"),
        }
        assert_eq!(cat.sources, vec!["core", "example-scenario"]);
    }

    // ------------------------------------------------------------------
    // WI 549 — settings stage, physical-truth seam, detectors.
    // ------------------------------------------------------------------

    /// A minimal settings document.
    fn settings_doc(id: &str, scalars: &str) -> String {
        format!("(format: 2, id: \"{id}\", scalars: [{scalars}])")
    }

    #[test]
    fn scalar_factor_must_be_finite_and_positive() {
        // WI 550: ×0, negatives, and non-finite factors are rejected at
        // freeze with the scalar named — a multiply-only grammar cannot
        // express them as a physical modification.
        let p = pack(&format!("{}{}", engine_base(), engine_variant()));
        for factor in ["0.0", "-0.5", "inf", "NaN"] {
            let s = settings_doc(
                "bal",
                &format!(
                    r#"( name: "eff", factor: {factor}, target: Id("engine_base"), field: "exhaust_velocity" ),"#
                ),
            );
            match Catalog::compose(&[&p], &[&s], &[]) {
                Err(ContentError::InvalidScalarFactor { name, .. }) => assert_eq!(name, "eff"),
                other => panic!("factor {factor}: expected InvalidScalarFactor, got {other:?}"),
            }
        }
    }

    #[test]
    fn scalar_rationale_surfaces_on_the_frozen_setting() {
        // WI 550: the authored rationale rides the frozen setting (the
        // telemetry trust line beside "real × modifier"); absent = None.
        let p = pack(&format!("{}{}", engine_base(), engine_variant()));
        let s = settings_doc(
            "bal",
            r#"( name: "eff", factor: 0.5, target: Id("engine_base"), field: "exhaust_velocity",
                 rationale: Some("engines simplified; orbits are the lesson") ),
               ( name: "plain", factor: 2.0, target: Id("engine_base"), field: "exhaust_velocity" ),"#,
        );
        let cat = Catalog::compose(&[&p], &[&s], &[]).unwrap();
        let eff = cat.settings.get("eff").unwrap();
        assert_eq!(eff.factor, 0.5);
        assert_eq!(
            eff.rationale.as_deref(),
            Some("engines simplified; orbits are the lesson")
        );
        assert_eq!(cat.settings.get("plain").unwrap().rationale, None);
        // Both scalars multiply the same field — distinct names stack.
        assert_eq!(engine_ev(&cat, "v"), 3200.0 * 0.5 * 2.0);
    }

    #[test]
    fn scalar_bakes_at_merge_with_frozen_set_and_provenance() {
        let p = pack(&format!("{}{}", engine_base(), engine_variant()));
        let s = settings_doc(
            "bal",
            r#"( name: "fuel_efficiency", factor: 0.7, target: Id("engine_base"), field: "exhaust_velocity" ),"#,
        );
        let cat = Catalog::compose(&[&p], &[&s], &[]).unwrap();
        // Baked into the resolved physical value, re-flowed to the variant;
        // the sim-facing record carries no scalar anywhere.
        assert_eq!(engine_ev(&cat, "v"), 3200.0 * 0.7);
        // Frozen set exposed by name.
        assert_eq!(
            cat.settings.get("fuel_efficiency").map(|s| s.factor),
            Some(0.7)
        );
        // Provenance: winning source names document + scalar; shadow holds
        // the authored physical value (real × modifier for telemetry).
        let fp = &cat.get("v").unwrap().field_provenance["exhaust_velocity"];
        assert_eq!(
            fp.source,
            SourceRef::Setting {
                source: "bal".into(),
                scalar: "fuel_efficiency".into()
            }
        );
        assert_eq!(fp.shadows[0].value, ProvValue::Number(3200.0));
        assert_eq!(fp.shadows[0].source, SourceRef::Pack { id: "test".into() });
        // Settings resolve before packs in the ladder listing.
        assert_eq!(cat.sources, vec!["bal", "test"]);
    }

    #[test]
    fn stacked_scalars_multiply_cumulatively_with_chain() {
        let p = pack(&format!("{}{}", engine_base(), engine_variant()));
        let s = settings_doc(
            "bal",
            r#"( name: "a", factor: 0.5, target: Id("engine_base"), field: "exhaust_velocity" ),
               ( name: "b", factor: 0.5, target: Id("engine_base"), field: "exhaust_velocity" ),"#,
        );
        let cat = Catalog::compose(&[&p], &[&s], &[]).unwrap();
        assert_eq!(engine_ev(&cat, "v"), 3200.0 * 0.25);
        let fp = &cat.get("v").unwrap().field_provenance["exhaust_velocity"];
        // Newest displaced first: a's product (1600), then the authored 3200.
        assert_eq!(fp.shadows.len(), 2);
        assert_eq!(fp.shadows[0].value, ProvValue::Number(1600.0));
        assert_eq!(fp.shadows[1].value, ProvValue::Number(3200.0));
    }

    #[test]
    fn scalars_apply_before_override_phases() {
        let p = pack(&format!("{}{}", engine_base(), engine_variant()));
        let s = settings_doc(
            "bal",
            r#"( name: "eff", factor: 0.5, target: Id("engine_base"), field: "exhaust_velocity" ),"#,
        );
        // A local override multiplies the already-scaled value.
        let local = override_set(
            "house",
            "Local",
            "",
            r#"( target: Id("engine_base"), field: "exhaust_velocity", op: Multiply(2.0) ),"#,
        );
        let cat = Catalog::compose(&[&p], &[&s], &[&local]).unwrap();
        assert_eq!(engine_ev(&cat, "v"), 3200.0); // 3200 × 0.5 × 2.0
        let fp = &cat.get("v").unwrap().field_provenance["exhaust_velocity"];
        assert_eq!(
            fp.source,
            SourceRef::Override {
                source: "house".into(),
                phase: OverridePhase::Local
            }
        );
        assert_eq!(fp.shadows[0].value, ProvValue::Number(1600.0));
        assert!(matches!(fp.shadows[0].source, SourceRef::Setting { .. }));
    }

    #[test]
    fn seam_violation_scalar_cannot_originate() {
        // The variant's exhaust velocity is unset (flows from the base) — a
        // scalar bound to it would originate, not modify.
        let p = pack(&format!("{}{}", engine_base(), engine_variant()));
        let s = settings_doc(
            "bal",
            r#"( name: "eff", factor: 0.7, target: Id("v"), field: "exhaust_velocity" ),"#,
        );
        match Catalog::compose(&[&p], &[&s], &[]) {
            Err(ContentError::SeamViolation {
                scalar,
                record,
                field,
            }) => {
                assert_eq!(
                    (scalar.as_str(), record.as_str(), field.as_str()),
                    ("eff", "v", "exhaust_velocity")
                );
            }
            other => panic!("expected SeamViolation, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_scalar_name_rejected_across_documents() {
        let p = pack(engine_base());
        let s1 = settings_doc(
            "bal_a",
            r#"( name: "eff", factor: 0.7, target: Id("engine_base"), field: "exhaust_velocity" ),"#,
        );
        let s2 = settings_doc(
            "bal_b",
            r#"( name: "eff", factor: 0.9, target: Id("engine_base"), field: "exhaust_velocity" ),"#,
        );
        match Catalog::compose(&[&p], &[&s1, &s2], &[]) {
            Err(ContentError::DuplicateScalar { name }) => assert_eq!(name, "eff"),
            other => panic!("expected DuplicateScalar, got {other:?}"),
        }
    }

    #[test]
    fn scalar_bindings_validate_like_overrides() {
        let p = pack(engine_base());
        let ghost = settings_doc(
            "bal",
            r#"( name: "eff", factor: 0.7, target: Id("ghost"), field: "exhaust_velocity" ),"#,
        );
        assert!(matches!(
            Catalog::compose(&[&p], &[&ghost], &[]),
            Err(ContentError::UnknownTarget { .. })
        ));
        let bad_field = settings_doc(
            "bal",
            r#"( name: "eff", factor: 0.7, target: Id("engine_base"), field: "warp_factor" ),"#,
        );
        assert!(matches!(
            Catalog::compose(&[&p], &[&bad_field], &[]),
            Err(ContentError::UnknownField { .. })
        ));
    }

    #[test]
    fn unresolved_body_reference_detector() {
        use std::collections::BTreeSet;
        let p = pack(r#"Body(( id: "start_moon", body: "training-moon" )),"#);
        let cat = Catalog::from_ron_str(&p).unwrap();
        let known: BTreeSet<String> = ["training-moon".to_string()].into();
        assert!(cat.validate_body_refs(&known).is_ok());
        let empty = BTreeSet::new();
        match cat.validate_body_refs(&empty) {
            Err(ContentError::UnresolvedReference { record, reference }) => {
                assert_eq!(
                    (record.as_str(), reference.as_str()),
                    ("start_moon", "training-moon")
                );
            }
            other => panic!("expected UnresolvedReference, got {other:?}"),
        }
    }

    #[test]
    fn compose_is_deterministic_under_input_permutation() {
        let p1 = pack_named("alpha", "", engine_base());
        let p2 = pack_named("beta", "depends: [\"alpha\"],", engine_variant());
        let s1 = settings_doc(
            "bal_a",
            r#"( name: "eff", factor: 0.9, target: Id("engine_base"), field: "exhaust_velocity" ),"#,
        );
        let s2 = settings_doc(
            "bal_b",
            r#"( name: "dens", factor: 1.1, target: Id("v"), field: "density" ),"#,
        );
        let ov = override_set(
            "scn",
            "Scenario",
            "",
            r#"( target: Id("engine_base"), field: "exhaust_velocity", op: Multiply(2.0) ),"#,
        );
        let a = Catalog::compose(&[&p1, &p2], &[&s1, &s2], &[&ov]).unwrap();
        let b = Catalog::compose(&[&p2, &p1], &[&s2, &s1], &[&ov]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.sources, vec!["bal_a", "bal_b", "alpha", "beta", "scn"]);
    }

    #[test]
    fn shipped_example_settings_composes_over_core() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../content");
        let core = std::fs::read_to_string(root.join("packs/core.ron")).unwrap();
        let bal = std::fs::read_to_string(root.join("settings/example-settings.ron")).unwrap();
        let scn = std::fs::read_to_string(root.join("overrides/example-scenario.ron")).unwrap();
        let cat = Catalog::compose(&[&core], &[&bal], &[&scn]).expect("full composition merges");
        // 3200 × 0.85 (settings bake) × 0.9 (scenario multiply), via the base.
        assert_eq!(engine_ev(&cat, "lf_engine_small"), 3200.0 * 0.85 * 0.9);
        assert_eq!(
            cat.settings.get("engine_efficiency").map(|s| s.factor),
            Some(0.85)
        );
        assert_eq!(
            cat.sources,
            vec!["example-settings", "core", "example-scenario"]
        );
    }
}
