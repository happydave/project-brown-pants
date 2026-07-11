//! `sounding check` — the recipe-authoring report engine (WI 896, design
//! item 7's read side).
//!
//! Resolves nothing itself: the caller hands a **resolved** [`Catalog`] (and
//! optionally a world save's identity + body records), and this module renders
//! what the authoring actually produced — the derived medium beside its pins,
//! the drawn independents beside their stream tags, layer stacks, suppress
//! marks, and the save-vs-catalog digest classification — deterministically
//! (sorted record order, `BTreeMap` iteration, no timestamps or paths).
//!
//! **Warnings ≠ errors** (the design's Phase-3 "loud warnings" tier, first
//! consumer): findings that are legal-but-worth-eyes (a pin deviating from
//! its relation, an input a pin makes dead, ocean intent the gate zeroed, a
//! save record that would reroll) are `WARN` lines and a count — the caller
//! still exits 0. Invalid inputs never reach this module: composition already
//! failed loudly and typed. One deliberate divergence from the load path: a
//! snapshot record failing its integrity digest is a *warning line* here (a
//! diagnostic tool shows every problem at once) where load refuses typed —
//! `check` is read-only and never applies the record anywhere.
//!
//! The reports each consume a seam parked by an earlier WI: pin-vs-relation
//! deviation + pin-shadowed inputs (WI 886), ocean-gating + drawn
//! independents with domain tags (WI 889), layer stack (WI 892), suppress
//! marks (WI 880), save digests (WI 891). The authored-seed lint became a
//! hard `UnphysicalValue` in WI 891, so it surfaces as a composition error
//! before any report runs.

use crate::body_derive;
use crate::body_digest::{digest_hex, BODY_OUTPUT_VERSION};
use crate::bodygen;
use crate::content::{BodyRecipeRecord, Catalog, Entry, Record, SourceRef};
use crate::world_save::{ContentIdentity, PackIdentity, SavedBodyRecord};
use std::collections::BTreeMap;
use std::fmt;

/// A world save's check-relevant slice: the recorded content identity and the
/// body records. Extracted by the caller (the CLI reads a `WorldPayload`);
/// keeping the engine off the full save shape makes it testable without a
/// flight state.
pub struct SaveCheck<'a> {
    pub identity: &'a ContentIdentity,
    pub bodies: &'a [SavedBodyRecord],
}

/// The rendered report plus its warning count (the caller's exit contract:
/// warnings exit 0, only typed errors are nonzero).
#[derive(Debug)]
pub struct CheckOutput {
    pub text: String,
    pub warnings: usize,
}

/// The engine's own (tiny) failure surface — everything else fails earlier,
/// typed, at composition or save parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckError {
    /// The requested record id is not in the composed catalog (or is not a
    /// body recipe).
    UnknownRecipe(String),
}

impl fmt::Display for CheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckError::UnknownRecipe(id) => {
                write!(f, "no body recipe `{id}` in the composed catalog")
            }
        }
    }
}

impl std::error::Error for CheckError {}

/// The derived-medium fields the fixed arm treats as pin-or-derive (WI 886);
/// presence in `field_provenance` on a fixed record means pinned.
const PINNABLE: [&str; 5] = [
    "gravity",
    "atmosphere_temperature",
    "atmosphere_surface_density",
    "atmosphere_scale_height",
    "ocean_surface_pressure",
];

/// Relative tolerance below which a pin is considered to *match* its
/// relation — silences representation noise only; any real authored pin
/// (an anchor like the ISA 1.225) deviates far above this and warns, which
/// is the report's purpose.
const PIN_MATCH_RTOL: f64 = 1e-9;

fn relative_deviation(a: f64, b: f64) -> f64 {
    let scale = a.abs().max(b.abs());
    if scale == 0.0 {
        0.0
    } else {
        (a - b).abs() / scale
    }
}

fn source_label(s: &SourceRef) -> String {
    match s {
        SourceRef::Pack { id } => format!("pack `{id}`"),
        SourceRef::Override { source, phase } => format!("override `{source}` ({phase:?})"),
        SourceRef::Setting { source, scalar } => format!("setting `{source}` (`{scalar}`)"),
    }
}

/// Render the check report over a resolved catalog: every body recipe (or
/// `only` one), plus the save-vs-catalog section when a save is given.
pub fn check_report(
    catalog: &Catalog,
    only: Option<&str>,
    save: Option<&SaveCheck<'_>>,
) -> Result<CheckOutput, CheckError> {
    let recipes: Vec<(&str, &Entry, &BodyRecipeRecord)> = catalog
        .ids()
        .filter_map(|id| {
            let entry = catalog.get(id)?;
            match &entry.record {
                Record::BodyRecipe(r) => Some((id, entry, r.as_ref())),
                _ => None,
            }
        })
        .collect();
    if let Some(id) = only {
        if !recipes.iter().any(|(rid, _, _)| *rid == id) {
            return Err(CheckError::UnknownRecipe(id.to_string()));
        }
    }

    let mut out = String::new();
    let mut warnings = 0usize;

    out.push_str(&format!(
        "sounding check — catalog `{}` ({} records, {} body recipes)\n",
        catalog.pack_id,
        catalog.len(),
        recipes.len()
    ));
    out.push_str(&format!("sources: {}\n", catalog.sources.join(", ")));

    for (id, entry, r) in &recipes {
        if only.is_some_and(|o| o != *id) {
            continue;
        }
        let body = &r.body;
        let m = &body.fluid_medium;
        let mode = match r.shape {
            Some(shape) => format!("shaped {}", shape.slug()),
            None => "fixed".to_string(),
        };
        out.push_str(&format!(
            "\n== {id} — {mode} · defined by pack `{}` ==\n",
            entry.provenance.pack_id
        ));
        out.push_str(&format!(
            "  body: radius {} m · mu {} · rotation {} s · surface seed {}\n",
            body.radius, body.mu, body.rotation.sidereal_period, body.surface.seed
        ));

        // The derived medium, with pin annotations (fixed arm only — a
        // shaped record's medium is never pinned).
        out.push_str("  medium:\n");
        let medium_fields: [(&str, f64); 9] = [
            ("gravity", m.gravity),
            ("atmosphere_temperature", m.atmosphere_temperature),
            ("atmosphere_surface_density", m.atmosphere_surface_density),
            ("atmosphere_surface_pressure", m.atmosphere_surface_pressure),
            ("atmosphere_scale_height", m.atmosphere_scale_height),
            ("ocean_surface_density", m.ocean_surface_density),
            ("ocean_surface_pressure", m.ocean_surface_pressure),
            ("ocean_density_gradient", m.ocean_density_gradient),
            ("ocean_temperature", m.ocean_temperature),
        ];
        for (name, value) in medium_fields {
            let pin = r.shape.is_none() && PINNABLE.contains(&name);
            match (pin, entry.field_provenance.get(name)) {
                (true, Some(fp)) => out.push_str(&format!(
                    "    {name} = {value}  (pinned ← {})\n",
                    source_label(&fp.source)
                )),
                _ => out.push_str(&format!("    {name} = {value}\n")),
            }
        }

        if r.shape.is_none() {
            fixed_reports(&mut out, &mut warnings, id, entry, r);
        } else {
            shaped_reports(&mut out, &mut warnings, id, entry, r);
        }

        // Report 5 (WI 892): the resolved layer stack, with the stack
        // field's winning source.
        if body.surface.layers.is_empty() {
            out.push_str("  layers: (none)\n");
        } else {
            let stack = body
                .surface
                .layers
                .iter()
                .map(|l| {
                    format!(
                        "{} ({:?}, {})",
                        l.id,
                        l.layer_type,
                        if l.enabled { "enabled" } else { "disabled" }
                    )
                })
                .collect::<Vec<_>>()
                .join(" -> ");
            let src = entry
                .field_provenance
                .get("surface_stack")
                .map(|fp| format!("  [stack ← {}]", source_label(&fp.source)))
                .unwrap_or_default();
            out.push_str(&format!("  layers: {stack}{src}\n"));
        }
    }

    if let Some(save) = save {
        warnings += save_report(&mut out, catalog, save, &recipes);
    }

    out.push_str(&format!("\n{warnings} warning(s)\n"));
    Ok(CheckOutput {
        text: out,
        warnings,
    })
}

/// Reports 1–3 on a fixed record: pin-vs-relation deviation, pin-shadowed
/// inputs, and ocean intent the gate zeroed (WIs 886/893/889).
fn fixed_reports(
    out: &mut String,
    warnings: &mut usize,
    id: &str,
    entry: &Entry,
    r: &BodyRecipeRecord,
) {
    let mut warn = |out: &mut String, text: String| {
        out.push_str(&format!("  WARN {text}\n"));
        *warnings += 1;
    };
    let body = &r.body;
    let m = &body.fluid_medium;
    let inputs = &r.derivation_inputs;
    let pinned = |name: &str| entry.field_provenance.contains_key(name);

    // Authored inputs, echoed (the retained raw values).
    if !inputs.is_empty() {
        let list = inputs
            .iter()
            .map(|(k, v)| format!("{k} {v}"))
            .collect::<Vec<_>>()
            .join(" · ");
        out.push_str(&format!("  authored inputs: {list}\n"));
    }

    // Report 1 — pin-vs-relation deviation, where the relation's inputs are
    // available. The relation values are computed by the same `body_derive`
    // functions the resolver runs (design I2: one physics).
    let mut deviation = |out: &mut String, field: &str, pin: f64, relation: f64| {
        if relative_deviation(pin, relation) > PIN_MATCH_RTOL {
            warn(
                out,
                format!("{id}: pinned {field} = {pin} deviates from its relation ({relation})"),
            );
        }
    };
    if pinned("gravity") {
        deviation(
            out,
            "gravity",
            m.gravity,
            body_derive::surface_gravity(body.mu, body.radius),
        );
    }
    if pinned("atmosphere_temperature") {
        if let (Some(s), Some(a), Some(dt)) = (
            inputs.get("nominal_insolation"),
            inputs.get("bond_albedo"),
            inputs.get("greenhouse_delta_t"),
        ) {
            deviation(
                out,
                "atmosphere_temperature",
                m.atmosphere_temperature,
                body_derive::surface_temperature(body_derive::equilibrium_temperature(*s, *a), *dt),
            );
        }
    }
    if let Some(mm) = inputs.get("mean_molar_mass") {
        if m.atmosphere_surface_pressure > 0.0 {
            if pinned("atmosphere_surface_density") {
                deviation(
                    out,
                    "atmosphere_surface_density",
                    m.atmosphere_surface_density,
                    body_derive::atmosphere_surface_density(
                        m.atmosphere_surface_pressure,
                        *mm,
                        m.atmosphere_temperature,
                    ),
                );
            }
            if pinned("atmosphere_scale_height") {
                deviation(
                    out,
                    "atmosphere_scale_height",
                    m.atmosphere_scale_height,
                    body_derive::scale_height(m.atmosphere_temperature, *mm, m.gravity),
                );
            }
        }
    }

    // Report 2 — inputs a pin makes dead. The consumption rules mirror the
    // fixed arm exactly: S/A/ΔT feed only the T_surf relation; molar mass
    // feeds the density and scale-height relations (only when an atmosphere
    // exists).
    if pinned("atmosphere_temperature") {
        for name in ["nominal_insolation", "bond_albedo", "greenhouse_delta_t"] {
            if inputs.contains_key(name) {
                warn(
                    out,
                    format!("{id}: authored {name} is dead — atmosphere_temperature is pinned"),
                );
            }
        }
    }
    if inputs.contains_key("mean_molar_mass") {
        let airless = m.atmosphere_surface_pressure == 0.0;
        let both_pinned = pinned("atmosphere_surface_density") && pinned("atmosphere_scale_height");
        if airless || both_pinned {
            warn(
                out,
                format!(
                    "{id}: authored mean_molar_mass is dead — {}",
                    if airless {
                        "the body is airless"
                    } else {
                        "density and scale height are both pinned"
                    }
                ),
            );
        }
    }

    // Report 3 — ocean intent the gate zeroed: authored ocean values survive
    // only in the retained inputs (gating applies last, even over pins).
    let intent = inputs.get("ocean_surface_density").copied().unwrap_or(0.0) > 0.0
        || inputs.get("ocean_surface_pressure").copied().unwrap_or(0.0) > 0.0;
    let gated = m.ocean_surface_density == 0.0 && m.ocean_surface_pressure == 0.0;
    if intent && gated {
        warn(
            out,
            format!(
                "{id}: ocean intent gated off (authored density {}, resolved T_surf {} K, \
                 surface pressure {} Pa)",
                inputs.get("ocean_surface_density").copied().unwrap_or(0.0),
                m.atmosphere_temperature,
                m.atmosphere_surface_pressure
            ),
        );
    }
}

/// Reports 3/4/8 on a shaped record: the drawn independents with their
/// stream tags (suppressed fields as explicit pins), the suppress marks with
/// provenance, and a gated-off ocean on an ocean shape (WIs 889/880).
fn shaped_reports(
    out: &mut String,
    warnings: &mut usize,
    id: &str,
    entry: &Entry,
    r: &BodyRecipeRecord,
) {
    let mut warn = |out: &mut String, text: String| {
        out.push_str(&format!("  WARN {text}\n"));
        *warnings += 1;
    };
    let shape = r.shape.expect("shaped_reports called for a shaped record");
    let body = &r.body;

    // Report 4 — what the sampler consumed, welded to `sample` by test.
    let bands = r.bands.expect("a resolved shaped record retains its bands");
    let pins: BTreeMap<&str, f64> = r.derivation_inputs.iter().map(|(k, v)| (*k, *v)).collect();
    out.push_str(&format!(
        "  drawn independents (seed {}):\n",
        body.surface.seed
    ));
    for (name, value, tag) in bodygen::drawn_independents(body.surface.seed, shape, &bands, &pins) {
        if tag.is_empty() {
            out.push_str(&format!("    {name} = {value}  (suppressed — explicit)\n"));
        } else {
            out.push_str(&format!("    {name} = {value}  [stream {tag}]\n"));
        }
    }

    // Report 8 — the suppress marks and their provenance chain.
    if !r.suppress.is_empty() {
        let src = entry
            .field_provenance
            .get("suppress")
            .map(|fp| {
                let shadows = fp.shadows.len();
                if shadows == 0 {
                    format!(" ← {}", source_label(&fp.source))
                } else {
                    format!(" ← {} (+{shadows} shadowed)", source_label(&fp.source))
                }
            })
            .unwrap_or_default();
        out.push_str(&format!("  suppress: [{}]{src}\n", r.suppress.join(", ")));
    }

    // Report 3, shaped spelling: an ocean shape whose gate closed at this
    // seed (frozen or airless corner of the bands).
    if shape == crate::bodygen::Archetype::OceanWorld
        && body.fluid_medium.ocean_surface_density == 0.0
    {
        warn(
            out,
            format!(
                "{id}: ocean gated off at this seed (T_surf {} K, surface pressure {} Pa)",
                body.fluid_medium.atmosphere_temperature,
                body.fluid_medium.atmosphere_surface_pressure
            ),
        );
    }
}

/// Report 6 (WI 891): the save's identity drift plus each body record
/// classified against the composed catalog — read-only (the load path's
/// taxonomy, without applying anything). Returns the warnings added.
fn save_report(
    out: &mut String,
    catalog: &Catalog,
    save: &SaveCheck<'_>,
    recipes: &[(&str, &Entry, &BodyRecipeRecord)],
) -> usize {
    let mut warnings = 0usize;
    out.push_str("\n== world save vs catalog ==\n");
    out.push_str(&format!(
        "  scenario: {} (recorded)\n",
        save.identity.scenario_id
    ));

    // Identity drift (the WI 891 honesty layer, reused verbatim). The
    // scenario fields are echoed from the record — `check` composes packs,
    // not scenarios — so only pack/settings drift can fire.
    let current = ContentIdentity {
        scenario_id: save.identity.scenario_id.clone(),
        scenario_version: save.identity.scenario_version.clone(),
        packs: catalog
            .packs
            .iter()
            .map(|(id, version)| PackIdentity {
                id: id.clone(),
                version: version.clone(),
            })
            .collect(),
        settings: catalog.settings.clone(),
    };
    for line in crate::world_save::drift_report(save.identity, &current) {
        out.push_str(&format!("  WARN identity drift: {line}\n"));
        warnings += 1;
    }

    // Body records, classified read-only with the load path's taxonomy.
    for record in save.bodies {
        let current = recipes
            .iter()
            .find(|(rid, _, _)| *rid == record.id())
            .map(|(_, _, r)| &r.body);
        match (record, current) {
            (_, None) => out.push_str(&format!(
                "  body `{}`: not in the composed catalog (generated or foreign — \
                 resolved by its own ref at load)\n",
                record.id()
            )),
            (
                SavedBodyRecord::Snapshot {
                    id,
                    output_version,
                    digest,
                    body,
                },
                Some(_),
            ) => {
                if digest_hex(body) != *digest {
                    out.push_str(&format!(
                        "  WARN body `{id}`: snapshot integrity FAILURE — the record's \
                         digest does not match its own body (load would refuse)\n"
                    ));
                    warnings += 1;
                } else if *output_version != BODY_OUTPUT_VERSION {
                    out.push_str(&format!(
                        "  WARN body `{id}`: snapshot pins output version {output_version} \
                         (this build generates {BODY_OUTPUT_VERSION})\n"
                    ));
                    warnings += 1;
                } else {
                    out.push_str(&format!("  body `{id}`: snapshot (wins verbatim)\n"));
                }
            }
            (
                SavedBodyRecord::Digest {
                    id,
                    output_version,
                    digest,
                },
                Some(body),
            ) => {
                if *output_version != BODY_OUTPUT_VERSION {
                    out.push_str(&format!(
                        "  WARN body `{id}`: would reroll — saved at output version \
                         {output_version}, this build generates {BODY_OUTPUT_VERSION}\n"
                    ));
                    warnings += 1;
                } else {
                    let computed = digest_hex(body);
                    if computed == *digest {
                        out.push_str(&format!("  body `{id}`: digest current\n"));
                    } else {
                        out.push_str(&format!(
                            "  WARN body `{id}`: STALE — catalog resolves digest \
                             {computed}, save recorded {digest} (unintended drift)\n"
                        ));
                        warnings += 1;
                    }
                }
            }
        }
    }
    warnings
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body_asset::BodyAsset;
    use crate::body_digest::digest_hex;
    use crate::content;
    use crate::world_save::body_records;
    use std::collections::BTreeSet;

    fn pack(records: &str) -> String {
        format!(
            "#![enable(implicit_some)]\n(format: 2, id: \"test\", version: \"1\", records: [{records}])"
        )
    }

    fn report(pack_text: &str) -> CheckOutput {
        let catalog = Catalog::merge(&[pack_text], &[]).unwrap();
        check_report(&catalog, None, None).unwrap()
    }

    /// A valid fixed recipe with `extra` fields spliced in.
    fn fixed_recipe(extra: &str) -> String {
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

    #[test]
    fn canonical_report_is_deterministic_and_flags_the_isa_anchors() {
        // The shipped catalog's two fixed records pin the ISA density anchor
        // (1.225), which deviates from the ideal-gas relation at the pinned
        // 288.15 K — the deviation report exists to SHOW exactly that, so the
        // canonical render carries those two warnings and no others.
        let catalog = content::embedded_catalog();
        let a = check_report(catalog, None, None).unwrap();
        let b = check_report(catalog, None, None).unwrap();
        assert_eq!(a.text, b.text, "same inputs => byte-identical report");
        assert_eq!(a.warnings, 2, "the two ISA density anchors:\n{}", a.text);
        assert_eq!(
            a.text.matches("pinned atmosphere_surface_density").count(),
            2,
            "{}",
            a.text
        );
        // The pins are annotated with their source, and the shaped records
        // list their drawn independents with stream tags.
        assert!(a.text.contains("(pinned ← pack `"), "{}", a.text);
        assert!(
            a.text.contains("[stream rocky/sidereal_period]"),
            "{}",
            a.text
        );
    }

    #[test]
    fn deviating_gravity_pin_warns_with_both_values() {
        // mu/R² = 11.11...; the pin says 5.0 — the report names both.
        let out = report(&pack(&fixed_recipe("gravity: 5.0,")));
        assert!(
            out.text
                .contains("pinned gravity = 5 deviates from its relation"),
            "{}",
            out.text
        );
    }

    #[test]
    fn temperature_pin_deviation_and_shadowed_inputs_warn() {
        // A pinned temperature beside authored thermal inputs: the relation
        // disagrees (deviation) AND the inputs are dead (shadowed).
        let out = report(&pack(&fixed_recipe("atmosphere_temperature: 500.0,")));
        assert!(
            out.text
                .contains("pinned atmosphere_temperature = 500 deviates"),
            "{}",
            out.text
        );
        for name in ["nominal_insolation", "bond_albedo", "greenhouse_delta_t"] {
            assert!(
                out.text.contains(&format!("authored {name} is dead")),
                "{name}:\n{}",
                out.text
            );
        }
        // Molar mass stays alive (scale height + density still derive).
        assert!(
            !out.text.contains("mean_molar_mass is dead"),
            "{}",
            out.text
        );
    }

    #[test]
    fn molar_mass_is_dead_when_both_gas_relations_are_pinned() {
        let out = report(&pack(&fixed_recipe(
            "atmosphere_surface_density: 1.2, atmosphere_scale_height: 8000.0,",
        )));
        assert!(
            out.text.contains(
                "authored mean_molar_mass is dead — density and scale height are both pinned"
            ),
            "{}",
            out.text
        );
    }

    #[test]
    fn gated_ocean_intent_warns_on_both_arms() {
        // Fixed: a frozen pin (200 K) zeroes the authored ocean.
        let out = report(&pack(&fixed_recipe("atmosphere_temperature: 200.0,")));
        assert!(
            out.text
                .contains("ocean intent gated off (authored density 1000"),
            "{}",
            out.text
        );
        // Shaped: an OceanWorld whose bands are entirely below the freeze
        // point gates off at every seed.
        let cold = r#"BodyRecipe(( id: "cold", name: "Cold", shape: ocean_world,
            radius_min: 3.0e6, radius_max: 9.0e6,
            gravity_min: 5.0, gravity_max: 12.0,
            rotation_period_min: 20000.0, rotation_period_max: 200000.0,
            atmosphere_surface_pressure_min: 80000.0, atmosphere_surface_pressure_max: 180000.0,
            nominal_insolation_min: 200.0, nominal_insolation_max: 300.0,
            bond_albedo_min: 0.1, bond_albedo_max: 0.2,
            greenhouse_delta_t_min: 0.0, greenhouse_delta_t_max: 1.0,
            mean_molar_mass_min: 0.018, mean_molar_mass_max: 0.03,
            ocean_surface_density_min: 950.0, ocean_surface_density_max: 1100.0,
            ocean_temperature_min: 275.0, ocean_temperature_max: 300.0,
            surface_seed: 3 )),"#;
        let out = report(&pack(cold));
        assert!(
            out.text.contains("cold: ocean gated off at this seed"),
            "{}",
            out.text
        );
    }

    #[test]
    fn shaped_report_shows_tags_suppress_marks_and_layers() {
        let layers = r#"SurfaceLayer(( id: "big-craters", layer_type: "crater", density: 1.5, depth: 2.0 )),"#;
        let shaped = r#"BodyRecipe(( id: "r1", name: "R1", shape: rocky_planet,
            radius_min: 2.5e6, radius_max: 8.0e6,
            gravity_min: 3.0, gravity_max: 12.0,
            rotation_period_min: 20000.0, rotation_period_max: 200000.0,
            atmosphere_surface_pressure_min: 50000.0, atmosphere_surface_pressure_max: 150000.0,
            nominal_insolation_min: 800.0, nominal_insolation_max: 2000.0,
            bond_albedo_min: 0.1, bond_albedo_max: 0.4,
            greenhouse_delta_t_min: 5.0, greenhouse_delta_t_max: 40.0,
            mean_molar_mass_min: 0.02, mean_molar_mass_max: 0.045,
            suppress: ["bond_albedo"], bond_albedo: 0.21,
            surface_stack: ["big-craters"],
            surface_seed: 42 )),"#;
        let out = report(&pack(&format!("{layers}{shaped}")));
        // Report 4: drawn fields carry stream tags; the suppressed field is
        // an explicit value with no stream.
        assert!(
            out.text
                .contains("bond_albedo = 0.21  (suppressed — explicit)"),
            "{}",
            out.text
        );
        assert!(out.text.contains("[stream rocky/radius]"), "{}", out.text);
        assert!(
            out.text.contains("[stream rocky/sidereal_period]"),
            "{}",
            out.text
        );
        // Report 8: the marks and their provenance.
        assert!(
            out.text.contains("suppress: [bond_albedo] ← pack `test`"),
            "{}",
            out.text
        );
        // Report 5: the stack with its source.
        assert!(
            out.text
                .contains("layers: big-craters (Crater, enabled)  [stack ← pack `test`]"),
            "{}",
            out.text
        );
        assert_eq!(out.warnings, 0, "{}", out.text);
    }

    #[test]
    fn save_report_classifies_all_four_tiers() {
        let catalog = content::embedded_catalog();
        let earthlike = match &catalog.get("earthlike").unwrap().record {
            Record::BodyRecipe(r) => r.body.clone(),
            _ => unreachable!(),
        };
        let ice_age = match &catalog.get("earthlike-ice-age").unwrap().record {
            Record::BodyRecipe(r) => r.body.clone(),
            _ => unreachable!(),
        };
        let assets: Vec<BodyAsset> = vec![earthlike, ice_age];
        let snapshot_ids: BTreeSet<String> = ["earthlike".to_string()].into();
        let mut records = body_records(&assets, &snapshot_ids);
        // Tamper tier by tier: the ice-age digest goes stale; add a
        // version-moved record and a foreign id.
        for record in &mut records {
            if let SavedBodyRecord::Digest { id, digest, .. } = record {
                if id == "earthlike-ice-age" {
                    *digest = "0000000000000000".into();
                }
            }
        }
        records.push(SavedBodyRecord::Digest {
            id: "earthlike".into(), // duplicate id is a LOAD error; use moon
            output_version: BODY_OUTPUT_VERSION - 1,
            digest: "1111111111111111".into(),
        });
        // Rename the duplicate to a distinct catalog record for the
        // version-moved tier, and add a generated id for the foreign tier.
        if let Some(SavedBodyRecord::Digest { id, .. }) = records.last_mut() {
            *id = "moon".into();
        }
        records.push(SavedBodyRecord::Digest {
            id: "gen-rocky-000000000000002a".into(),
            output_version: BODY_OUTPUT_VERSION,
            digest: "2222222222222222".into(),
        });
        let identity = ContentIdentity {
            scenario_id: "fixture".into(),
            scenario_version: None,
            packs: vec![PackIdentity {
                id: catalog.packs[0].0.clone(),
                version: "stale-version".into(),
            }],
            settings: BTreeMap::new(),
        };
        let save = SaveCheck {
            identity: &identity,
            bodies: &records,
        };
        let out = check_report(catalog, None, Some(&save)).unwrap();
        assert!(
            out.text
                .contains("body `earthlike`: snapshot (wins verbatim)"),
            "{}",
            out.text
        );
        assert!(
            out.text.contains("WARN body `earthlike-ice-age`: STALE"),
            "{}",
            out.text
        );
        assert!(
            out.text
                .contains("WARN body `moon`: would reroll — saved at output version"),
            "{}",
            out.text
        );
        assert!(
            out.text
                .contains("body `gen-rocky-000000000000002a`: not in the composed catalog"),
            "{}",
            out.text
        );
        assert!(out.text.contains("WARN identity drift:"), "{}", out.text);
        // Integrity tier: corrupt the snapshot's own digest.
        let mut corrupt = body_records(&assets, &snapshot_ids);
        if let Some(SavedBodyRecord::Snapshot { digest, .. }) = corrupt
            .iter_mut()
            .find(|r| matches!(r, SavedBodyRecord::Snapshot { .. }))
        {
            *digest = "dead000000000000".into();
        }
        let save = SaveCheck {
            identity: &identity,
            bodies: &corrupt,
        };
        let out = check_report(catalog, None, Some(&save)).unwrap();
        assert!(
            out.text
                .contains("WARN body `earthlike`: snapshot integrity FAILURE"),
            "{}",
            out.text
        );
    }

    #[test]
    fn only_id_narrows_and_unknown_id_is_loud() {
        let catalog = content::embedded_catalog();
        let out = check_report(catalog, Some("moon"), None).unwrap();
        assert!(out.text.contains("== moon —"), "{}", out.text);
        assert!(!out.text.contains("== earthlike —"), "{}", out.text);
        assert_eq!(
            check_report(catalog, Some("nope"), None).unwrap_err(),
            CheckError::UnknownRecipe("nope".into())
        );
    }

    #[test]
    fn quiet_fixed_recipe_renders_without_warnings() {
        // No pins beyond requirements, warm ocean, all inputs consumed.
        let out = report(&pack(&fixed_recipe("")));
        assert_eq!(out.warnings, 0, "{}", out.text);
        assert!(out.text.contains("== d1 — fixed"), "{}", out.text);
        assert!(out.text.contains("authored inputs:"), "{}", out.text);
        assert!(out.text.contains("layers: (none)"), "{}", out.text);
    }

    #[test]
    fn snapshot_digest_verifies_via_digest_hex() {
        // The save fixture above relies on body_records stamping digest_hex —
        // pin that relationship so the fixture cannot silently weaken.
        let catalog = content::embedded_catalog();
        let earthlike = match &catalog.get("earthlike").unwrap().record {
            Record::BodyRecipe(r) => r.body.clone(),
            _ => unreachable!(),
        };
        let records = body_records(std::slice::from_ref(&earthlike), &BTreeSet::new());
        match &records[0] {
            SavedBodyRecord::Digest { digest, .. } => {
                assert_eq!(digest, &digest_hex(&earthlike));
            }
            other => panic!("expected digest tier, got {other:?}"),
        }
    }
}
