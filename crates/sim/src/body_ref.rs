//! Persisted catalog references — "recipe+seed is truth" (WI 891, design I3/D1).
//!
//! A **body ref** is the catalog's persisted spelling of a body: instead of a
//! resolved [`BodyAsset`] snapshot, the document records *how to regenerate*
//! it — a canonical recipe id (plus the embedded pack's identity, provenance
//! only) or a generator archetype + seed — together with the
//! [`BODY_OUTPUT_VERSION`] and digest recorded at save time. Loading a ref
//! regenerates through the ordinary resolve/derive pipeline and verifies the
//! digest; the WI 888 golden harness is what makes that regeneration
//! bit-identical across builds of one output version.
//!
//! **The digest is the sole verifier.** Pack identity enriches diagnostics,
//! but a pack-version difference is never, by itself, an error — a pack bump
//! that leaves a body byte-identical keeps every ref to it valid. A missing
//! id is refused loudly (no fallback body, no tombstone — the modding story
//! is deferred with parked decision (a)).

use crate::body_asset::BodyAsset;
use crate::body_digest::{digest_hex, BODY_OUTPUT_VERSION};
use crate::bodygen::{generate, Archetype};
use serde::{Deserialize, Serialize};
use std::fmt;

/// How a ref's body regenerates.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BodyRefSource {
    /// A canonical recipe from the embedded bodies pack, by record id.
    Recipe {
        /// The recipe record id (`content/bodies.ron`).
        recipe_id: String,
        /// Embedded pack id at save time (provenance/diagnostics only).
        pack_id: String,
        /// Embedded pack version at save time (provenance/diagnostics only).
        pack_version: String,
    },
    /// A generated body: [`Archetype`] slug; the seed rides the ref itself.
    Generated {
        /// The archetype slug (`moon`/`rocky`/`ocean`, the WI 888 golden-label
        /// spelling).
        archetype: String,
    },
}

/// A persisted catalog reference: enough to regenerate a body bit-identically
/// and to prove it happened. One new additive [`crate::persist::Payload`]
/// kind — no format bump (the documented additive-variant rule).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BodyRef {
    /// Display name (the regenerated body's name; list/diagnostic surface).
    pub name: String,
    /// Regeneration source.
    pub source: BodyRefSource,
    /// The body's surface seed. For a generated ref this is the `generate`
    /// input; for a recipe ref it is recorded from the resolved body for
    /// diagnostics (the recipe authors its own seed). A true u64 — never the
    /// authored f64 Num slot, so the 2^53 ceiling does not apply here.
    pub seed: u64,
    /// [`BODY_OUTPUT_VERSION`] at save time.
    pub output_version: u32,
    /// FNV-1a 64 digest of the resolved body at save time, as 16 lowercase
    /// hex digits (the golden-file spelling). Compared as a string.
    pub digest: String,
}

/// A typed ref failure — loud, naming the offender (design I5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BodyRefError {
    /// The recipe id is not a canonical body recipe in this build.
    UnknownRecipe {
        /// The unresolvable id.
        recipe_id: String,
        /// Recorded pack identity (the likely-cause diagnostic).
        pack_id: String,
        /// Recorded pack version.
        pack_version: String,
    },
    /// The archetype slug is not one this build's generator knows.
    UnknownArchetype {
        /// The unresolvable slug.
        slug: String,
    },
    /// The ref was recorded under a different [`BODY_OUTPUT_VERSION`] — the
    /// deliberate-reroll case, surfaced distinguishably so the caller decides
    /// (regenerate anyway via [`BodyRef::regenerate`], or keep the old body
    /// from wherever it was snapshotted).
    OutputVersionMoved {
        /// The ref's display name.
        name: String,
        /// Version recorded at save time.
        recorded: u32,
        /// This build's version.
        current: u32,
    },
    /// Same output version, different digest: unintended generator drift or a
    /// corrupt/hand-edited ref.
    DigestMismatch {
        /// The ref's display name.
        name: String,
        /// Digest recorded at save time (hex).
        recorded: String,
        /// Digest of the regenerated body (hex).
        computed: String,
        /// Likely-cause note (pack-version delta), empty when none applies.
        context: String,
    },
}

impl fmt::Display for BodyRefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BodyRefError::UnknownRecipe {
                recipe_id,
                pack_id,
                pack_version,
            } => write!(
                f,
                "body ref names recipe `{recipe_id}` (recorded from pack `{pack_id}` \
                 version {pack_version}), which this build's canonical pack does not provide"
            ),
            BodyRefError::UnknownArchetype { slug } => {
                write!(f, "body ref names unknown archetype `{slug}`")
            }
            BodyRefError::OutputVersionMoved {
                name,
                recorded,
                current,
            } => write!(
                f,
                "body ref `{name}` was recorded at output version {recorded}; this build \
                 generates version {current} — regenerating would reroll the body"
            ),
            BodyRefError::DigestMismatch {
                name,
                recorded,
                computed,
                context,
            } => {
                write!(
                    f,
                    "body ref `{name}` regenerated with digest {computed}, but {recorded} \
                     was recorded — unintended generator drift or a corrupt ref"
                )?;
                if !context.is_empty() {
                    write!(f, " ({context})")?;
                }
                Ok(())
            }
        }
    }
}

impl BodyRef {
    /// A ref to a canonical recipe body, recorded from this build's resolution
    /// of it. `None` when the id is not a canonical body recipe.
    pub fn for_canonical(recipe_id: &str) -> Option<BodyRef> {
        let asset = crate::content::try_canonical_body(recipe_id)?;
        let (pack_id, pack_version) = crate::content::canonical_pack_identity();
        Some(BodyRef {
            name: asset.name.clone(),
            source: BodyRefSource::Recipe {
                recipe_id: recipe_id.to_string(),
                pack_id,
                pack_version,
            },
            seed: asset.surface.seed,
            output_version: BODY_OUTPUT_VERSION,
            digest: digest_hex(&asset),
        })
    }

    /// A ref to a generated body, recorded from this build's generation of it.
    pub fn for_generated(archetype: Archetype, seed: u64) -> BodyRef {
        let asset = generate(seed, archetype);
        BodyRef {
            name: asset.name.clone(),
            source: BodyRefSource::Generated {
                archetype: archetype.slug().to_string(),
            },
            seed,
            output_version: BODY_OUTPUT_VERSION,
            digest: digest_hex(&asset),
        }
    }

    /// Regenerates the body **without** version/digest verification — the
    /// deliberate-reroll path, for a caller that saw
    /// [`BodyRefError::OutputVersionMoved`] and chose to proceed. Still fails
    /// loudly on an unresolvable source.
    pub fn regenerate(&self) -> Result<BodyAsset, BodyRefError> {
        match &self.source {
            BodyRefSource::Recipe {
                recipe_id,
                pack_id,
                pack_version,
            } => crate::content::try_canonical_body(recipe_id).ok_or_else(|| {
                BodyRefError::UnknownRecipe {
                    recipe_id: recipe_id.clone(),
                    pack_id: pack_id.clone(),
                    pack_version: pack_version.clone(),
                }
            }),
            BodyRefSource::Generated { archetype } => {
                let arch = Archetype::from_slug(archetype).ok_or_else(|| {
                    BodyRefError::UnknownArchetype {
                        slug: archetype.clone(),
                    }
                })?;
                Ok(generate(self.seed, arch))
            }
        }
    }

    /// The load path: regenerates and verifies. Ordered so the outcomes stay
    /// distinguishable — a moved output version is reported as such (digest
    /// mismatch is then *expected*), and only a same-version mismatch reads as
    /// unintended drift.
    pub fn resolve(&self) -> Result<BodyAsset, BodyRefError> {
        if self.output_version != BODY_OUTPUT_VERSION {
            return Err(BodyRefError::OutputVersionMoved {
                name: self.name.clone(),
                recorded: self.output_version,
                current: BODY_OUTPUT_VERSION,
            });
        }
        let asset = self.regenerate()?;
        let computed = digest_hex(&asset);
        if computed != self.digest {
            return Err(BodyRefError::DigestMismatch {
                name: self.name.clone(),
                recorded: self.digest.clone(),
                computed,
                context: self.pack_context(),
            });
        }
        Ok(asset)
    }

    /// The likely-cause diagnostic for a recipe-ref mismatch: the recorded
    /// pack identity against this build's. Empty for generated refs and for
    /// an unchanged pack.
    fn pack_context(&self) -> String {
        match &self.source {
            BodyRefSource::Recipe {
                pack_id,
                pack_version,
                ..
            } => {
                let (cur_id, cur_version) = crate::content::canonical_pack_identity();
                if *pack_id != cur_id || *pack_version != cur_version {
                    format!(
                        "recorded from pack `{pack_id}` version {pack_version}; this build \
                         embeds pack `{cur_id}` version {cur_version}"
                    )
                } else {
                    String::new()
                }
            }
            BodyRefSource::Generated { .. } => String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A canonical ref regenerates the exact body it was recorded from —
    /// AC 1's digest-verified round trip, at the type level.
    #[test]
    fn canonical_ref_resolves_bit_identical() {
        let r = BodyRef::for_canonical("earthlike").expect("earthlike is canonical");
        assert!(
            matches!(&r.source, BodyRefSource::Recipe { recipe_id, .. } if recipe_id == "earthlike")
        );
        let body = r.resolve().expect("clean resolve");
        assert_eq!(digest_hex(&body), r.digest, "digest-verified regeneration");
        assert_eq!(body, crate::body_asset::BodyAsset::earthlike());
    }

    /// A generated ref does the same through `bodygen::generate`.
    #[test]
    fn generated_ref_resolves_bit_identical() {
        let r = BodyRef::for_generated(Archetype::RockyPlanet, 42);
        let body = r.resolve().expect("clean resolve");
        assert_eq!(body, generate(42, Archetype::RockyPlanet));
        assert_eq!(digest_hex(&body), r.digest);
    }

    /// Tampered digest ⇒ a loud mismatch naming the body; recorded/computed
    /// both surfaced (scenario A2).
    #[test]
    fn tampered_digest_is_a_named_mismatch() {
        let mut r = BodyRef::for_generated(Archetype::Moon, 7);
        let good = r.digest.clone();
        r.digest = "0000000000000000".to_string();
        match r.resolve() {
            Err(BodyRefError::DigestMismatch {
                name,
                recorded,
                computed,
                ..
            }) => {
                assert_eq!(name, r.name);
                assert_eq!(recorded, "0000000000000000");
                assert_eq!(computed, good);
            }
            other => panic!("expected DigestMismatch, got {other:?}"),
        }
    }

    /// A moved output version is its own distinguishable outcome — and the
    /// deliberate-reroll path (`regenerate`) still works past it.
    #[test]
    fn moved_output_version_is_distinguishable_and_reroll_still_possible() {
        let mut r = BodyRef::for_generated(Archetype::OceanWorld, 3);
        r.output_version = BODY_OUTPUT_VERSION + 1;
        match r.resolve() {
            Err(BodyRefError::OutputVersionMoved {
                recorded, current, ..
            }) => {
                assert_eq!(recorded, BODY_OUTPUT_VERSION + 1);
                assert_eq!(current, BODY_OUTPUT_VERSION);
            }
            other => panic!("expected OutputVersionMoved, got {other:?}"),
        }
        assert!(r.regenerate().is_ok(), "the caller may still reroll");
    }

    /// WI 889 (kept-body semantics, workitem AC 3): a ref recorded at output
    /// version **2** — the real predecessor of the batched stream break —
    /// resolves to the designed deliberate-reroll fork, and `regenerate()`
    /// past it yields the version-3 body. Regeneration at the old values is
    /// retired: nothing can recreate the version-2 output from the ref.
    #[test]
    fn version_two_era_refs_reroll_to_the_version_three_body() {
        assert_eq!(BODY_OUTPUT_VERSION, 3, "this test narrates the 2 → 3 break");
        let mut r = BodyRef::for_generated(Archetype::RockyPlanet, 11);
        r.output_version = 2;
        r.digest = "feedfacefeedface".to_string(); // v2-era digest, unverifiable now
        match r.resolve() {
            Err(BodyRefError::OutputVersionMoved {
                recorded, current, ..
            }) => {
                assert_eq!(recorded, 2);
                assert_eq!(current, 3);
            }
            other => panic!("expected OutputVersionMoved, got {other:?}"),
        }
        let rerolled = r.regenerate().expect("deliberate reroll");
        assert_eq!(
            rerolled,
            crate::bodygen::generate(11, Archetype::RockyPlanet),
            "the reroll is the current-generator body"
        );
    }

    /// Unresolvable sources are typed and name the offender (scenarios A4 and
    /// the unknown-archetype edge case).
    #[test]
    fn unknown_sources_are_typed_and_named() {
        assert!(BodyRef::for_canonical("no-such-recipe").is_none());
        let r = BodyRef {
            name: "Ghost".into(),
            source: BodyRefSource::Recipe {
                recipe_id: "no-such-recipe".into(),
                pack_id: "bodies".into(),
                pack_version: "0".into(),
            },
            seed: 0,
            output_version: BODY_OUTPUT_VERSION,
            digest: "0000000000000000".into(),
        };
        match r.resolve() {
            Err(BodyRefError::UnknownRecipe { recipe_id, .. }) => {
                assert_eq!(recipe_id, "no-such-recipe")
            }
            other => panic!("expected UnknownRecipe, got {other:?}"),
        }
        let g = BodyRef {
            name: "Ghost".into(),
            source: BodyRefSource::Generated {
                archetype: "gasbag".into(),
            },
            seed: 1,
            output_version: BODY_OUTPUT_VERSION,
            digest: "0000000000000000".into(),
        };
        match g.resolve() {
            Err(BodyRefError::UnknownArchetype { slug }) => assert_eq!(slug, "gasbag"),
            other => panic!("expected UnknownArchetype, got {other:?}"),
        }
    }

    /// Boundary seeds round-trip exactly through the ref's JSON spelling —
    /// the u64 path never transits f64 (parked decision (b), settled).
    #[test]
    fn boundary_seeds_round_trip_exactly() {
        for seed in [0u64, 1 << 53, u64::MAX] {
            let r = BodyRef::for_generated(Archetype::Moon, seed);
            let json = serde_json::to_string(&r).unwrap();
            let back: BodyRef = serde_json::from_str(&json).unwrap();
            assert_eq!(back.seed, seed, "seed {seed} survives JSON exactly");
            assert_eq!(back, r);
            assert_eq!(back.resolve().unwrap(), r.resolve().unwrap());
        }
    }
}
