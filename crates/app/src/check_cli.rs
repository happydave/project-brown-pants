//! `sounding check` — the headless authoring-report subcommand (WI 896).
//!
//! Dispatched from `main` *before* any Bevy `App` exists: parse flags, compose
//! the catalog (the embedded canonical pack first, then any `--pack` files, so
//! external packs can parent from the canonical bases), run the sim-side
//! report engine, print, exit. Warnings exit 0 — only typed content/persist/IO
//! failures (and a usage error) are nonzero. Read-only by construction: the
//! engine renders strings; nothing here writes.

use sounding_sim::check::{check_report, SaveCheck};
use sounding_sim::content::{self, Catalog};
use sounding_sim::world_save;
use std::path::PathBuf;

/// Parsed `check` arguments: `sounding check [recipe-id] [--pack FILE ...]
/// [--save FILE]`.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct CheckArgs {
    /// Narrow the report to one body recipe.
    pub recipe: Option<String>,
    /// Extra pack files composed after the embedded canonical pack.
    pub packs: Vec<PathBuf>,
    /// A world save to classify against the composed catalog.
    pub save: Option<PathBuf>,
}

pub const USAGE: &str = "usage: sounding check [recipe-id] [--pack FILE ...] [--save FILE]";

/// Parse the arguments after `check`. Pure (unit-tested); unknown flags and
/// duplicate positionals are usage errors (nonzero exit, per review).
pub fn parse_args(args: &[String]) -> Result<CheckArgs, String> {
    let mut parsed = CheckArgs::default();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--pack" => match it.next() {
                Some(path) => parsed.packs.push(PathBuf::from(path)),
                None => return Err("--pack requires a file path".into()),
            },
            "--save" => match it.next() {
                Some(path) if parsed.save.is_none() => parsed.save = Some(PathBuf::from(path)),
                Some(_) => return Err("--save given twice".into()),
                None => return Err("--save requires a file path".into()),
            },
            flag if flag.starts_with("--") => return Err(format!("unknown flag `{flag}`")),
            id if parsed.recipe.is_none() => parsed.recipe = Some(id.to_string()),
            extra => return Err(format!("unexpected argument `{extra}`")),
        }
    }
    Ok(parsed)
}

/// Run the subcommand; returns the process exit code.
pub fn run(args: &[String]) -> i32 {
    let parsed = match parse_args(args) {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("{e}\n{USAGE}");
            return 2;
        }
    };

    // Compose: the embedded canonical pack, then the --pack files in argv
    // order (all base-phase; a record-id collision is the existing typed
    // DuplicateId refusal).
    let mut texts: Vec<String> = vec![content::embedded_pack_source().to_string()];
    for path in &parsed.packs {
        match std::fs::read_to_string(path) {
            Ok(text) => texts.push(text),
            Err(e) => {
                eprintln!("cannot read pack {}: {e}", path.display());
                return 1;
            }
        }
    }
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let catalog = if parsed.packs.is_empty() {
        // The parse-once embedded catalog (no recomposition needed).
        content::embedded_catalog().clone()
    } else {
        match Catalog::compose(&refs, &[], &[]) {
            Ok(catalog) => catalog,
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        }
    };

    // The save's check-relevant slice, when asked for.
    let payload = match &parsed.save {
        None => None,
        Some(path) => match world_save::load_world(path) {
            Ok(payload) => Some(payload),
            Err(e) => {
                eprintln!("cannot load save {}: {e}", path.display());
                return 1;
            }
        },
    };
    let save_slice = payload
        .as_ref()
        .and_then(|p| p.scenario.as_ref())
        .map(|s| SaveCheck {
            identity: &s.content,
            bodies: &s.bodies,
        });

    match check_report(&catalog, parsed.recipe.as_deref(), save_slice.as_ref()) {
        Ok(output) => {
            print!("{}", output.text);
            if payload.is_some() && save_slice.is_none() {
                println!("world save: no scenario state (no body records to check)");
            }
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_parse_the_documented_grammar() {
        let args: Vec<String> = [
            "earthlike",
            "--pack",
            "a.ron",
            "--pack",
            "b.ron",
            "--save",
            "w.json",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(
            parse_args(&args).unwrap(),
            CheckArgs {
                recipe: Some("earthlike".into()),
                packs: vec!["a.ron".into(), "b.ron".into()],
                save: Some("w.json".into()),
            }
        );
        assert_eq!(parse_args(&[]).unwrap(), CheckArgs::default());
    }

    #[test]
    fn bad_args_are_usage_errors() {
        for bad in [
            vec!["--nope".to_string()],
            vec!["--pack".to_string()],
            vec!["--save".to_string()],
            vec!["a".to_string(), "b".to_string()],
            vec!["--save".into(), "x".into(), "--save".into(), "y".into()],
        ] {
            assert!(parse_args(&bad).is_err(), "{bad:?}");
        }
    }
}
