//! Versioned persistence format (WI 498).
//!
//! The single, explicitly versioned serialization format for a craft, a
//! subassembly, a blueprint, or (scaled up) a whole-world save — **one envelope,
//! one version line, several uses.** It is versioned *from the first commit*
//! because every saved artifact becomes a migration liability the moment the
//! schema drifts.
//!
//! This is **durable persistence**, deliberately distinct from the ephemeral
//! runtime bus (`Command`/`Telemetry`, WI 502): those are unversioned wire shapes
//! for live clients; these are versioned saves. Do not route saves through the
//! telemetry types.
//!
//! At format version 1 the content model does not exist yet — the structural
//! lattice, devices, resource graph, and crew arrive with later toys. So the
//! payload here is **skeletal and extensible**: it embeds the real WI 497
//! world-coordinate types plus metadata, and reserves empty, opaque containers
//! that later toys fill in (a future format-version change).

use crate::frame::WorldPos;
use crate::voxel::VoxelCraft;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Current on-disk format version. Increments **only** on a schema change — it is
/// deliberately independent of the crate's semantic version (which bumps one
/// patch per work item). A monotonic integer is what makes it a migration signal.
pub const FORMAT_VERSION: u32 = 1;

/// What a serialized artifact is used as. One format, several uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Craft,
    Subassembly,
    Blueprint,
    WorldSave,
}

/// A craft-scope serialized subgraph. A craft, a subassembly, and a blueprint are
/// the **same shape** at different scopes; the [`Payload`] kind distinguishes them.
///
/// At format version 1 the voxel/device contents are real (WI 505), filled in
/// place over WI 498's previously-opaque placeholders. The `resources` and `crew`
/// containers remain reserved (opaque `Value`s, empty) until their toys.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CraftSubgraph {
    /// Stable identifier.
    pub id: String,
    /// Human-facing display name.
    pub name: String,
    /// A WI 497 world-coordinate value — the craft's reference placement.
    pub reference_position: WorldPos,
    /// The voxel lattice + devices + attachment interface (WI 505).
    #[serde(default)]
    pub craft: VoxelCraft,
    /// Reserved: resource reservoirs / converters / conduits (Toy 7).
    #[serde(default)]
    pub resources: Vec<serde_json::Value>,
    /// Reserved: assigned crew (later).
    #[serde(default)]
    pub crew: Vec<serde_json::Value>,
}

impl CraftSubgraph {
    /// Builds a craft subgraph carrying `craft`.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        reference_position: WorldPos,
        craft: VoxelCraft,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            reference_position,
            craft,
            resources: Vec::new(),
            crew: Vec::new(),
        }
    }
}

/// Reserved, skeletal world-save payload — the same machinery scaled to the
/// universe, **distinct** from a craft subgraph. Empty at format version 1.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct WorldPayload {
    /// On-rails vessels and their orbits, converter timestamps, terrain patches
    /// (world persistence, later).
    #[serde(default)]
    pub vessels: Vec<serde_json::Value>,
}

/// The payload, internally tagged by `kind`. The three craft-scope kinds share
/// [`CraftSubgraph`]; world-save carries its own [`WorldPayload`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Payload {
    Craft(CraftSubgraph),
    Subassembly(CraftSubgraph),
    Blueprint(CraftSubgraph),
    WorldSave(WorldPayload),
}

impl Payload {
    /// The kind tag for this payload.
    pub fn kind(&self) -> Kind {
        match self {
            Payload::Craft(_) => Kind::Craft,
            Payload::Subassembly(_) => Kind::Subassembly,
            Payload::Blueprint(_) => Kind::Blueprint,
            Payload::WorldSave(_) => Kind::WorldSave,
        }
    }
}

/// A complete versioned document: the format envelope.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedDocument {
    /// The format version this document was written at.
    pub format_version: u32,
    /// The payload (and, via its tag, the kind).
    pub payload: Payload,
}

impl SavedDocument {
    /// Wraps a payload at the current [`FORMAT_VERSION`].
    pub fn new(payload: Payload) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            payload,
        }
    }

    /// The kind of this document.
    pub fn kind(&self) -> Kind {
        self.payload.kind()
    }

    /// Serializes to pretty JSON (human-inspectable).
    pub fn to_json(&self) -> Result<String, FormatError> {
        serde_json::to_string_pretty(self).map_err(|e| FormatError::Malformed(e.to_string()))
    }

    /// Loads from JSON with a **two-stage parse**: a version-stable [`VersionProbe`]
    /// reads the format version *first* and rejects an unsupported one, so a newer
    /// file is rejected by version rather than by an incidental payload-shape
    /// mismatch. The `match` below is the **migration seam** — call sites do not
    /// change when future migrations are added; new arms attach here.
    pub fn from_json(s: &str) -> Result<SavedDocument, FormatError> {
        let probe: VersionProbe =
            serde_json::from_str(s).map_err(|e| FormatError::Malformed(e.to_string()))?;
        match probe.format_version {
            FORMAT_VERSION => {
                serde_json::from_str(s).map_err(|e| FormatError::Malformed(e.to_string()))
            }
            // Future: an older supported version parses into its vN shape here and
            // is migrated up to the current `SavedDocument` before returning. No
            // older versions exist at format version 1, so nothing is migrated yet.
            other => Err(FormatError::UnsupportedVersion(other)),
        }
    }
}

/// Version-stable header for the first load stage. Deserializes only the format
/// version, ignoring the payload, so it parses regardless of how a future
/// version's payload is shaped. **Its fields must not change across versions.**
#[derive(Deserialize)]
struct VersionProbe {
    format_version: u32,
}

/// A persistence-format error. Typed and non-panicking, so malformed or foreign
/// input is rejected cleanly at the boundary.
#[derive(Debug, Clone, PartialEq)]
pub enum FormatError {
    /// Malformed/truncated input, a missing required field (including a missing
    /// version), or an unknown kind — and, on save, a serialize failure.
    Malformed(String),
    /// The document declares a format version this build does not support.
    UnsupportedVersion(u32),
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormatError::Malformed(m) => write!(f, "malformed serialized document: {m}"),
            FormatError::UnsupportedVersion(v) => write!(
                f,
                "unsupported format version {v} (this build supports {FORMAT_VERSION})"
            ),
        }
    }
}

impl std::error::Error for FormatError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameId;
    use crate::voxel::{Material, Voxel};
    use glam::{DVec3, IVec3};

    fn sample_pos() -> WorldPos {
        WorldPos::new(FrameId::CENTRAL_BODY, DVec3::new(1.0, 2.0, 3.0))
    }

    fn sample_craft() -> VoxelCraft {
        let mut craft = VoxelCraft::new(1.0);
        craft.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        craft
    }

    #[test]
    fn craft_round_trips_preserving_version_and_kind() {
        let pos = sample_pos();
        let doc = SavedDocument::new(Payload::Craft(CraftSubgraph::new(
            "ship-1",
            "Ranger",
            pos,
            sample_craft(),
        )));
        let json = doc.to_json().unwrap();
        let back = SavedDocument::from_json(&json).unwrap();
        assert_eq!(doc, back);
        assert_eq!(back.format_version, FORMAT_VERSION);
        assert_eq!(back.kind(), Kind::Craft);
    }

    #[test]
    fn all_four_kinds_round_trip_through_one_envelope() {
        let pos = sample_pos();
        let payloads = [
            Payload::Craft(CraftSubgraph::new("c", "C", pos, sample_craft())),
            Payload::Subassembly(CraftSubgraph::new("s", "S", pos, sample_craft())),
            Payload::Blueprint(CraftSubgraph::new("b", "B", pos, sample_craft())),
            Payload::WorldSave(WorldPayload::default()),
        ];
        for p in payloads {
            let kind = p.kind();
            let doc = SavedDocument::new(p);
            let back = SavedDocument::from_json(&doc.to_json().unwrap()).unwrap();
            assert_eq!(doc, back);
            assert_eq!(back.kind(), kind);
        }
    }

    #[test]
    fn craft_subassembly_blueprint_share_the_subgraph_payload() {
        let pos = sample_pos();
        let cs = CraftSubgraph::new("x", "X", pos, sample_craft());
        // Same shape carried by three kinds.
        for payload in [
            Payload::Craft(cs.clone()),
            Payload::Subassembly(cs.clone()),
            Payload::Blueprint(cs.clone()),
        ] {
            if let Payload::Craft(c) | Payload::Subassembly(c) | Payload::Blueprint(c) = &payload {
                assert_eq!(c, &cs);
            } else {
                unreachable!();
            }
        }
    }

    #[test]
    fn newer_version_rejected_by_version_not_payload_parse() {
        // Alien payload shape (a bare number) a v1 build cannot parse — the
        // version-stable probe still reads the version and rejects by version.
        let newer = r#"{ "format_version": 2, "payload": 12345 }"#;
        assert_eq!(
            SavedDocument::from_json(newer),
            Err(FormatError::UnsupportedVersion(2))
        );
    }

    #[test]
    fn missing_version_field_is_rejected_not_assumed_v1() {
        let no_version = r#"{ "payload": { "kind": "craft" } }"#;
        assert!(matches!(
            SavedDocument::from_json(no_version),
            Err(FormatError::Malformed(_))
        ));
    }

    #[test]
    fn malformed_input_is_rejected_without_panic() {
        for bad in ["{ not json", "", "[1,2,3", "null"] {
            assert!(matches!(
                SavedDocument::from_json(bad),
                Err(FormatError::Malformed(_))
            ));
        }
    }

    #[test]
    fn unknown_kind_is_rejected() {
        let s = r#"{ "format_version": 1, "payload": { "kind": "spaceship", "id": "a", "name": "b" } }"#;
        assert!(matches!(
            SavedDocument::from_json(s),
            Err(FormatError::Malformed(_))
        ));
    }

    #[test]
    fn payload_embeds_worldpos_voxels_and_reserved_containers() {
        let pos = sample_pos();
        let doc = SavedDocument::new(Payload::Craft(CraftSubgraph::new(
            "id",
            "Name",
            pos,
            sample_craft(),
        )));
        let back = SavedDocument::from_json(&doc.to_json().unwrap()).unwrap();
        let Payload::Craft(c) = &back.payload else {
            panic!("expected craft");
        };
        assert_eq!(c.reference_position, pos);
        // Real voxel content round-trips; resources/crew stay reserved (empty).
        assert_eq!(c.craft.voxels.len(), 1);
        assert!(c.resources.is_empty());
        assert!(c.crew.is_empty());
    }

    #[test]
    fn json_is_human_inspectable_with_version_and_kind() {
        let doc = SavedDocument::new(Payload::Craft(CraftSubgraph::new(
            "id",
            "Name",
            sample_pos(),
            sample_craft(),
        )));
        let json = doc.to_json().unwrap();
        assert!(json.contains("format_version"));
        assert!(json.contains("\"kind\""));
        assert!(json.contains("\"craft\""));
    }

    #[test]
    fn format_version_is_one() {
        assert_eq!(FORMAT_VERSION, 1);
    }
}
