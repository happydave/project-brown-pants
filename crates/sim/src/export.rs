//! `sounding export-body` — re-author a resolved body as `parent + overrides`
//! (WI 897, design item 7's write side — the last bodies-as-recipes item).
//!
//! Turns a sampled body into authored pack content without ever emitting a
//! flat scalar dump (the design's SpaceEngine `snowLevel` guard): the emitted
//! record is `parent` + either `surface_seed` (the **seed spelling** — tracks
//! the family, re-samples under future band tuning) or the WI 880 suppress
//! vocabulary pinning every drawn value (the **freeze spelling** — pins the
//! draws; parent stack/offset/layer tuning still flows to both).
//!
//! **Self-verifying**: nothing is returned until the export proves itself —
//! both spellings are constructed internally with the same identity, each is
//! composed (one composition per spelling) through the real
//! parse→ladder→inherit→validate pipeline, and their digests must agree
//! (cross-spelling equivalence — two maximally different documents converging
//! bit-identically, covering inherited stacks/offsets/pins because both
//! spellings inherit them identically). A generated id additionally anchors
//! against `generate(seed, archetype)` verbatim — the keep-loop contract,
//! self-guarding: if the canonical slug records ever gain a stack, this
//! comparison fails typed rather than passing silently.

use crate::body_digest::digest_body;
use crate::bodygen;
use crate::content::{Catalog, ContentError, Record};
use std::collections::BTreeMap;
use std::fmt;

/// The authored-seed ceiling (WI 891 decision (b), mirrored from the
/// validation bound): `surface_seed` is a numeric slot, exact only through
/// 2⁵³ inclusive. The surface field consumes the seed even under `--freeze`
/// (suppression pins draws, not terrain), so a larger generated seed has no
/// lossless authored spelling at all. Welded to `content::MAX_AUTHORED_SEED`
/// (the f64 validation bound) by test — the two constants cannot drift.
const MAX_AUTHORED_SEED: u64 = 9_007_199_254_740_992;

/// A verified export: the emitted pack document, the record id it defines,
/// and the digest the verification proved.
#[derive(Debug)]
pub struct Export {
    pub pack: String,
    pub record_id: String,
    pub digest: String,
}

/// Typed export failures (design I5 — loud, naming the offender).
#[derive(Debug)]
pub enum ExportError {
    /// The target id is neither a body recipe in the composed catalog nor a
    /// parseable generated id.
    UnknownRecipe(String),
    /// The target is a fixed recipe — already authored content.
    NotShaped(String),
    /// The seed exceeds the authored 2⁵³ ceiling; no lossless authored
    /// spelling exists (the terrain consumes the seed even frozen).
    SeedCeiling { id: String, seed: u64 },
    /// A verification composition failed (bad `--id` collision, etc.).
    Content(ContentError),
    /// The round-trip verification failed: the named comparison did not
    /// converge. Nothing was emitted.
    RoundTrip {
        comparison: &'static str,
        expected: String,
        got: String,
    },
}

impl fmt::Display for ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExportError::UnknownRecipe(id) => {
                write!(f, "no body recipe `{id}` in the composed catalog (and not a generated id)")
            }
            ExportError::NotShaped(id) => write!(
                f,
                "`{id}` is a fixed recipe — already authored content; export re-authors sampled bodies"
            ),
            ExportError::SeedCeiling { id, seed } => write!(
                f,
                "`{id}`: seed {seed} exceeds the authored ceiling (2^53) — a recipe's \
                 surface_seed is numeric and the terrain consumes the seed even frozen, \
                 so no lossless authored spelling exists"
            ),
            ExportError::Content(e) => write!(f, "export verification composition failed: {e}"),
            ExportError::RoundTrip {
                comparison,
                expected,
                got,
            } => write!(
                f,
                "round-trip verification failed ({comparison}): expected digest {expected}, \
                 got {got} — nothing emitted"
            ),
        }
    }
}

impl std::error::Error for ExportError {}

impl From<ContentError> for ExportError {
    fn from(e: ContentError) -> Self {
        ExportError::Content(e)
    }
}

/// Export `target` (a shaped catalog record id, or a generated body id) as a
/// one-record pack. `pack_texts` is the full source composition (the caller
/// puts the embedded canonical pack first); `seed` overrides the default
/// (the record's own seed / the generated id's encoded seed); `freeze`
/// selects the suppress-spelling for the *emitted* document (both spellings
/// are always verified); `new_id` overrides the default record id.
pub fn export_body(
    pack_texts: &[&str],
    target: &str,
    seed: Option<u64>,
    freeze: bool,
    new_id: Option<&str>,
) -> Result<Export, ExportError> {
    let catalog = Catalog::compose(pack_texts, &[], &[])?;

    // Resolve the target: catalog lookup wins (a real record named `gen-…`
    // is addressed as itself); the generated-id parse covers ids that exist
    // nowhere but a save or keep-loop.
    let (parent_id, shape, default_seed, generated) = match catalog.get(target) {
        Some(entry) => match &entry.record {
            Record::BodyRecipe(r) => match r.shape {
                Some(shape) => (target.to_string(), shape, r.body.surface.seed, false),
                None => return Err(ExportError::NotShaped(target.to_string())),
            },
            _ => return Err(ExportError::UnknownRecipe(target.to_string())),
        },
        None => match bodygen::parse_generated_id(target) {
            Some((archetype, encoded_seed)) => {
                (archetype.slug().to_string(), archetype, encoded_seed, true)
            }
            None => return Err(ExportError::UnknownRecipe(target.to_string())),
        },
    };
    let seed = seed.unwrap_or(default_seed);
    if seed > MAX_AUTHORED_SEED {
        return Err(ExportError::SeedCeiling {
            id: target.to_string(),
            seed,
        });
    }

    // The parent record supplies the bands and retained pins the freeze
    // spelling must honor (an inherited suppression freezes at the parent's
    // explicit value, not a phantom draw).
    let parent = match &catalog
        .get(&parent_id)
        .ok_or_else(|| ExportError::UnknownRecipe(parent_id.clone()))?
        .record
    {
        Record::BodyRecipe(r) => r.as_ref().clone(),
        _ => return Err(ExportError::UnknownRecipe(parent_id.clone())),
    };
    // Typed, not a panic: reachable only through the pub API when a caller's
    // composition omits the canonical pack and defines a *fixed* record under
    // an archetype slug — the gen id then has no shaped family to re-author.
    let bands = parent
        .bands
        .ok_or_else(|| ExportError::NotShaped(parent_id.clone()))?;
    let parent_pins: BTreeMap<&str, f64> = parent
        .derivation_inputs
        .iter()
        .map(|(k, v)| (*k, *v))
        .collect();

    // Identity defaults: the synthesized generated identity for this
    // family + seed (for gen ids this makes the export reproduce
    // `generate()` verbatim — id and name included).
    let (default_id, default_name) = bodygen::generated_identity(seed, shape);
    let record_id = new_id.unwrap_or(&default_id).to_string();
    let name = default_name;

    // Both spellings, same identity.
    let seed_spelling = emit(&record_id, &name, &parent_id, seed, None);
    let drawn = bodygen::drawn_independents(seed, shape, &bands, &parent_pins);
    let freeze_fields: Vec<(&str, f64)> = drawn.iter().map(|(n, v, _)| (*n, *v)).collect();
    let freeze_spelling = emit(&record_id, &name, &parent_id, seed, Some(&freeze_fields));

    // Verification: one composition per spelling, digests must converge.
    let resolve = |spelling: &str| -> Result<u64, ExportError> {
        let mut texts: Vec<&str> = pack_texts.to_vec();
        texts.push(spelling);
        let composed = Catalog::compose(&texts, &[], &[])?;
        match &composed
            .get(&record_id)
            .expect("the emitted record composes under its own id")
            .record
        {
            Record::BodyRecipe(r) => Ok(digest_body(&r.body)),
            _ => unreachable!("the emitted record is a body recipe"),
        }
    };
    let via_seed = resolve(&seed_spelling)?;
    let via_freeze = resolve(&freeze_spelling)?;
    if via_seed != via_freeze {
        return Err(ExportError::RoundTrip {
            comparison: "cross-spelling: seed vs freeze",
            expected: format!("{via_seed:016x}"),
            got: format!("{via_freeze:016x}"),
        });
    }
    if generated && record_id == default_id {
        let oracle = digest_body(&bodygen::generate(seed, shape));
        if via_seed != oracle {
            return Err(ExportError::RoundTrip {
                comparison: "generated-body anchor: export vs generate()",
                expected: format!("{oracle:016x}"),
                got: format!("{via_seed:016x}"),
            });
        }
    }

    Ok(Export {
        pack: if freeze {
            freeze_spelling
        } else {
            seed_spelling
        },
        record_id,
        digest: format!("{via_seed:016x}"),
    })
}

/// Render the one-record pack document. `frozen` = the drawn fields to
/// suppress + author explicitly (the freeze spelling); `None` = the seed
/// spelling. Deterministic: fixed layout, `{}` float display
/// (shortest-round-trip exact), fields in the drawn order.
fn emit(
    record_id: &str,
    name: &str,
    parent: &str,
    seed: u64,
    frozen: Option<&[(&str, f64)]>,
) -> String {
    let mut out = String::from("#![enable(implicit_some)]\n");
    out.push_str("// Exported by `sounding export-body` (WI 897): parent + overrides,\n");
    out.push_str("// never a flat dump of derived fields.\n");
    out.push_str(&format!(
        "(format: 2, id: {:?}, version: \"1\", records: [\n    BodyRecipe((\n",
        format!("export-{record_id}")
    ));
    out.push_str(&format!("        id: {record_id:?},\n"));
    out.push_str(&format!("        name: {name:?},\n"));
    out.push_str(&format!("        parent: {parent:?},\n"));
    out.push_str(&format!("        surface_seed: {seed},\n"));
    if let Some(fields) = frozen {
        let names: Vec<String> = fields.iter().map(|(n, _)| format!("{n:?}")).collect();
        out.push_str(&format!("        suppress: [{}],\n", names.join(", ")));
        for (field, value) in fields {
            out.push_str(&format!("        {field}: {value},\n"));
        }
    }
    out.push_str("    )),\n])\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body_digest::digest_hex;
    use crate::bodygen::Archetype;
    use crate::bodygen::{generate, suppressible_fields};
    use crate::content;

    fn embedded() -> Vec<&'static str> {
        vec![content::embedded_pack_source()]
    }

    fn resolved_digest(texts: &[&str], id: &str) -> String {
        let catalog = Catalog::compose(texts, &[], &[]).unwrap();
        match &catalog.get(id).unwrap().record {
            Record::BodyRecipe(r) => digest_hex(&r.body),
            other => panic!("expected body recipe, got {other:?}"),
        }
    }

    #[test]
    fn generated_id_export_reproduces_generate_verbatim() {
        // The keep-loop contract: both spellings, all archetypes.
        for arch in Archetype::ALL {
            for seed in [0u64, 1, 42, 7777] {
                let (gen_id, _) = bodygen::generated_identity(seed, arch);
                for freeze in [false, true] {
                    let export = export_body(&embedded(), &gen_id, None, freeze, None).unwrap();
                    assert_eq!(export.record_id, gen_id);
                    assert_eq!(
                        export.digest,
                        digest_hex(&generate(seed, arch)),
                        "{arch:?}@{seed} freeze={freeze}"
                    );
                    // Literal content assertions (plan review: the
                    // cross-spelling check alone cannot catch a coherent
                    // wrong seed — the emitted text pins it).
                    assert!(
                        export
                            .pack
                            .contains(&format!("parent: \"{}\"", arch.slug())),
                        "{}",
                        export.pack
                    );
                    assert!(
                        export.pack.contains(&format!("surface_seed: {seed},")),
                        "{}",
                        export.pack
                    );
                    // Never a flat dump: no derived medium field is authored.
                    for derived in ["atmosphere_scale_height", "atmosphere_surface_density"] {
                        assert!(
                            !export.pack.contains(&format!("{derived}:")),
                            "{}",
                            export.pack
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn freeze_spelling_suppresses_exactly_the_drawn_set() {
        let (gen_id, _) = bodygen::generated_identity(3, Archetype::Moon);
        let export = export_body(&embedded(), &gen_id, None, true, None).unwrap();
        // The Moon's five drawn fields, no more (per-archetype, not union).
        for field in suppressible_fields(Archetype::Moon) {
            assert!(
                export.pack.contains(&format!("\"{field}\"")),
                "{field} missing from suppress list:\n{}",
                export.pack
            );
        }
        assert!(
            !export.pack.contains("mean_molar_mass"),
            "a Moon draws no atmosphere trio:\n{}",
            export.pack
        );
        // The emitted pack is standalone-composable and resolves to the
        // verified digest.
        let texts = [content::embedded_pack_source(), export.pack.as_str()];
        assert_eq!(resolved_digest(&texts, &export.record_id), export.digest);
    }

    #[test]
    fn shaped_catalog_export_verifies_cross_spelling() {
        // The catalog record's own seed by default; an explicit seed too.
        for seed in [None, Some(9u64)] {
            let by_seed = export_body(&embedded(), "rocky", seed, false, None).unwrap();
            let frozen = export_body(&embedded(), "rocky", seed, true, None).unwrap();
            assert_eq!(by_seed.digest, frozen.digest, "spellings converge");
            assert_eq!(by_seed.record_id, frozen.record_id);
            assert!(by_seed.pack.contains("parent: \"rocky\""));
        }
    }

    #[test]
    fn band_tuning_moves_the_seed_export_but_not_the_frozen_one() {
        let (gen_id, _) = bodygen::generated_identity(11, Archetype::RockyPlanet);
        let by_seed = export_body(&embedded(), &gen_id, None, false, None).unwrap();
        let frozen = export_body(&embedded(), &gen_id, None, true, None).unwrap();
        assert_eq!(by_seed.digest, frozen.digest);
        // Tune the parent's gravity band and re-resolve both emitted packs.
        let tune = r#"(format: 2, id: "tune", phase: Scenario,
            overrides: [( target: Id("rocky"), field: "gravity_max", op: Multiply(1.5) )])"#;
        let digest_under = |pack: &str| {
            let texts = [content::embedded_pack_source(), pack];
            let catalog = Catalog::compose(&texts, &[], &[tune]).unwrap();
            match &catalog.get(&by_seed.record_id).unwrap().record {
                Record::BodyRecipe(r) => digest_hex(&r.body),
                other => panic!("{other:?}"),
            }
        };
        assert_ne!(
            digest_under(&by_seed.pack),
            by_seed.digest,
            "the seed spelling tracks the family: band tuning re-samples"
        );
        assert_eq!(
            digest_under(&frozen.pack),
            frozen.digest,
            "the frozen spelling pins the draws: band tuning changes nothing"
        );
    }

    #[test]
    fn inherited_pins_and_stacks_ride_both_spellings() {
        // A tuned parent from a "--pack file": its own suppression AND an
        // authored layer stack (the plan-review B1 case).
        let tuned_parent = r#"#![enable(implicit_some)]
(format: 2, id: "mods", version: "1", records: [
    SurfaceLayer(( id: "big-craters", layer_type: "crater", density: 1.5, depth: 2.0 )),
    BodyRecipe(( id: "tuned", name: "Tuned", parent: "rocky",
        suppress: ["bond_albedo"], bond_albedo: 0.2,
        surface_stack: ["big-craters"],
        surface_seed: 5 )),
])"#;
        let texts = vec![content::embedded_pack_source(), tuned_parent];
        let by_seed = export_body(&texts, "tuned", Some(13), false, None).unwrap();
        let frozen = export_body(&texts, "tuned", Some(13), true, None).unwrap();
        assert_eq!(by_seed.digest, frozen.digest, "B1: stacks + pins covered");
        // The inherited pin freezes at the parent's explicit value...
        assert!(frozen.pack.contains("bond_albedo: 0.2,"), "{}", frozen.pack);
        // ...and neither spelling re-authors the stack (inheritance carries it).
        for pack in [&by_seed.pack, &frozen.pack] {
            assert!(!pack.contains("surface_stack"), "{pack}");
        }
        // The resolved export carries the inherited layer (digest-covered).
        let all = [
            content::embedded_pack_source(),
            tuned_parent,
            by_seed.pack.as_str(),
        ];
        let catalog = Catalog::compose(&all, &[], &[]).unwrap();
        match &catalog.get(&by_seed.record_id).unwrap().record {
            Record::BodyRecipe(r) => {
                assert_eq!(r.body.surface.layers.len(), 1);
                assert_eq!(r.body.surface.layers[0].id, "big-craters");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn refusals_are_loud_and_typed() {
        // Unknown id (not gen-formed either).
        assert!(matches!(
            export_body(&embedded(), "nope", None, false, None),
            Err(ExportError::UnknownRecipe(id)) if id == "nope"
        ));
        // Fixed record: already authored content.
        assert!(matches!(
            export_body(&embedded(), "earthlike", None, false, None),
            Err(ExportError::NotShaped(id)) if id == "earthlike"
        ));
        // Seed ceiling: a generated id encoding a seed above 2^53.
        let (big, _) = bodygen::generated_identity(MAX_AUTHORED_SEED + 1, Archetype::Moon);
        assert!(matches!(
            export_body(&embedded(), &big, None, false, None),
            Err(ExportError::SeedCeiling { seed, .. }) if seed == MAX_AUTHORED_SEED + 1
        ));
        // Exactly 2^53 is allowed (inclusive bound).
        let (edge, _) = bodygen::generated_identity(MAX_AUTHORED_SEED, Archetype::Moon);
        export_body(&embedded(), &edge, None, false, None).unwrap();
        // --id collision with a composed record: the verification
        // composition's typed DuplicateId, surfaced.
        assert!(matches!(
            export_body(&embedded(), "rocky", Some(1), false, Some("earthlike")),
            Err(ExportError::Content(_))
        ));
        // The generate() anchor is self-guarding: replace the canonical pack
        // with retuned bands (same ids) and the gen-id export must refuse —
        // the emitted recipe cannot reproduce the real generate() output.
        let retuned =
            content::embedded_pack_source().replace("gravity_min: 3.0", "gravity_min: 4.0");
        assert_ne!(
            retuned,
            content::embedded_pack_source(),
            "fixture edits a band"
        );
        let (gen_id, _) = bodygen::generated_identity(11, Archetype::RockyPlanet);
        match export_body(&[&retuned], &gen_id, None, false, None) {
            Err(ExportError::RoundTrip { comparison, .. }) => {
                assert!(comparison.contains("generate()"), "{comparison}");
            }
            other => panic!("expected RoundTrip refusal, got {other:?}"),
        }
    }

    #[test]
    fn export_is_deterministic_and_parse_generated_id_round_trips() {
        let (gen_id, _) = bodygen::generated_identity(42, Archetype::OceanWorld);
        let a = export_body(&embedded(), &gen_id, None, true, None).unwrap();
        let b = export_body(&embedded(), &gen_id, None, true, None).unwrap();
        assert_eq!(a.pack, b.pack, "same inputs => byte-identical document");
        // The id parser is the exact inverse of the synthesizer.
        for arch in Archetype::ALL {
            for seed in [0u64, 42, u64::MAX] {
                let (id, _) = bodygen::generated_identity(seed, arch);
                assert_eq!(bodygen::parse_generated_id(&id), Some((arch, seed)));
            }
        }
        assert_eq!(bodygen::parse_generated_id("gen-rocky-XYZ"), None);
        assert_eq!(bodygen::parse_generated_id("rocky"), None);
        assert_eq!(
            bodygen::parse_generated_id("gen-nope-0000000000000000"),
            None
        );
        // Strictly 16 *lowercase* hex: uppercase, wrong length, and a dashed
        // pseudo-slug (rsplit takes the LAST dash, so "rocky-extra" is no
        // archetype) are all rejected.
        assert_eq!(
            bodygen::parse_generated_id("gen-rocky-000000000000002A"),
            None
        );
        assert_eq!(
            bodygen::parse_generated_id("gen-rocky-000000000000002"),
            None
        );
        assert_eq!(
            bodygen::parse_generated_id("gen-rocky-00000000000000002a"),
            None
        );
        assert_eq!(
            bodygen::parse_generated_id("gen-rocky-extra-000000000000002a"),
            None
        );
    }

    #[test]
    fn seed_ceiling_welds_to_the_validation_bound() {
        // The u64 ceiling here and the f64 validation bound in `content` are
        // the same number (2^53) — a drift in either direction breaks this.
        assert_eq!(MAX_AUTHORED_SEED as f64, content::MAX_AUTHORED_SEED);
        assert_eq!(content::MAX_AUTHORED_SEED as u64, MAX_AUTHORED_SEED);
    }
}
