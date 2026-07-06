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
//! first-class tuning primitive. Overrides live in **override sets** (RON
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

use crate::voxel::{Material, Thermal};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::Path;

/// Version of the authored content-document *format* (packs and override
/// sets; not of any pack's content). Rejected loudly when a document declares
/// anything else. Increments only on a schema change to the document shapes.
pub const CONTENT_FORMAT_VERSION: u32 = 1;

/// Field names that overrides may never target: record identity and
/// inheritance topology are definitions, not tunables.
const STRUCTURAL_FIELDS: [&str; 4] = ["id", "parent", "abstract", "class"];

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
}

/// A record kind — the override target-selector vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum RecordKind {
    Device,
    Material,
    Resource,
    Body,
}

impl RawRecord {
    fn id(&self) -> &str {
        match self {
            RawRecord::Device(r) => &r.id,
            RawRecord::Material(r) => &r.id,
            RawRecord::Resource(r) => &r.id,
            RawRecord::Body(r) => &r.id,
        }
    }
    fn parent(&self) -> Option<&str> {
        match self {
            RawRecord::Device(r) => r.parent.as_deref(),
            RawRecord::Material(r) => r.parent.as_deref(),
            RawRecord::Resource(r) => r.parent.as_deref(),
            RawRecord::Body(r) => r.parent.as_deref(),
        }
    }
    fn is_abstract(&self) -> bool {
        match self {
            RawRecord::Device(r) => r.is_abstract,
            RawRecord::Material(r) => r.is_abstract,
            RawRecord::Resource(r) => r.is_abstract,
            RawRecord::Body(r) => r.is_abstract,
        }
    }
    fn kind(&self) -> RecordKind {
        match self {
            RawRecord::Device(_) => RecordKind::Device,
            RawRecord::Material(_) => RecordKind::Material,
            RawRecord::Resource(_) => RecordKind::Resource,
            RawRecord::Body(_) => RecordKind::Body,
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

/// The four field operations. `Set` operands are explicitly typed (RON-safe).
#[derive(Debug, Deserialize)]
enum Op {
    Set(SetValue),
    Multiply(f64),
    Extend(Vec<String>),
    Delete(String),
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

/// A resolved, concrete content record.
#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    Device(DeviceRecord),
    Material(MaterialRecord),
    Resource(ResourceRecord),
    Body(BodyRefRecord),
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
    // Parse + format-check every document.
    let mut packs = Vec::with_capacity(pack_texts.len());
    for text in pack_texts {
        let p: RawPack = ron::from_str(text).map_err(|e| ContentError::Parse(e.to_string()))?;
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
        let record = validate(m)?;
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
fn validate(m: &RawRecord) -> Result<Record, ContentError> {
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal valid pack around the given records list (RON snippet).
    fn pack(records: &str) -> String {
        format!(
            "#![enable(implicit_some)]\n(format: 1, id: \"test\", version: \"1\", records: [{records}])"
        )
    }

    /// A minimal pack with an explicit id (multi-pack tests).
    fn pack_named(id: &str, extra: &str, records: &str) -> String {
        format!(
            "#![enable(implicit_some)]\n(format: 1, id: \"{id}\", version: \"1\", {extra} records: [{records}])"
        )
    }

    /// A minimal override set.
    fn override_set(id: &str, phase: &str, extra: &str, overrides: &str) -> String {
        format!("(format: 1, id: \"{id}\", phase: {phase}, {extra} overrides: [{overrides}])")
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
            Catalog::from_ron_str("(format: 1, id: \"p\", version: \"1\", records: [])").unwrap();
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
        format!("(format: 1, id: \"{id}\", scalars: [{scalars}])")
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
