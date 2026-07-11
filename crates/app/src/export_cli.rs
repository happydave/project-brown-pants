//! `sounding export-body` — the authored-content export subcommand (WI 897).
//!
//! Same shape as its `check` sibling: dispatched from `main` before any Bevy
//! `App` exists, pure unit-tested arg parsing, embedded-canonical-pack-first
//! composition. The emitted (self-verified) pack goes to stdout — pipeable —
//! or to `--out FILE`; a one-line verification receipt goes to stderr. Exit
//! 0 on success, 1 on typed errors, 2 on usage.

use sounding_sim::bodygen::parse_generated_id;
use sounding_sim::content;
use sounding_sim::export::export_body;
use std::path::PathBuf;

/// Parsed arguments: `sounding export-body <id> [--seed N] [--freeze]
/// [--id NEW-ID] [--pack FILE ...] [--out FILE]`.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ExportArgs {
    pub target: String,
    pub seed: Option<u64>,
    pub freeze: bool,
    pub new_id: Option<String>,
    pub packs: Vec<PathBuf>,
    pub out: Option<PathBuf>,
}

pub const USAGE: &str =
    "usage: sounding export-body <id> [--seed N] [--freeze] [--id NEW-ID] [--pack FILE ...] [--out FILE]";

/// Parse the arguments after `export-body`. Usage errors include a `--seed`
/// that contradicts a generated id's encoded seed (the id *is* the seed —
/// ambiguity is an error, not a precedence rule).
pub fn parse_args(args: &[String]) -> Result<ExportArgs, String> {
    let mut target: Option<String> = None;
    let mut parsed = ExportArgs::default();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--seed" => match it.next().map(|s| s.parse::<u64>()) {
                Some(Ok(n)) if parsed.seed.is_none() => parsed.seed = Some(n),
                Some(Ok(_)) => return Err("--seed given twice".into()),
                Some(Err(_)) => return Err("--seed requires a non-negative integer".into()),
                None => return Err("--seed requires a value".into()),
            },
            "--freeze" => parsed.freeze = true,
            "--id" => match it.next() {
                Some(id) if parsed.new_id.is_none() => parsed.new_id = Some(id.clone()),
                Some(_) => return Err("--id given twice".into()),
                None => return Err("--id requires a record id".into()),
            },
            "--pack" => match it.next() {
                Some(path) => parsed.packs.push(PathBuf::from(path)),
                None => return Err("--pack requires a file path".into()),
            },
            "--out" => match it.next() {
                Some(path) if parsed.out.is_none() => parsed.out = Some(PathBuf::from(path)),
                Some(_) => return Err("--out given twice".into()),
                None => return Err("--out requires a file path".into()),
            },
            flag if flag.starts_with("--") => return Err(format!("unknown flag `{flag}`")),
            id if target.is_none() => target = Some(id.to_string()),
            extra => return Err(format!("unexpected argument `{extra}`")),
        }
    }
    parsed.target = target.ok_or("a body id is required")?;
    if let (Some(seed), Some((_, encoded))) = (parsed.seed, parse_generated_id(&parsed.target)) {
        if seed != encoded {
            return Err(format!(
                "--seed {seed} contradicts the id's encoded seed {encoded} — a generated id IS its seed"
            ));
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
    match export_body(
        &refs,
        &parsed.target,
        parsed.seed,
        parsed.freeze,
        parsed.new_id.as_deref(),
    ) {
        Ok(export) => {
            eprintln!(
                "verified: record `{}` resolves with digest {}",
                export.record_id, export.digest
            );
            match &parsed.out {
                None => print!("{}", export.pack),
                Some(path) => {
                    if let Err(e) = std::fs::write(path, &export.pack) {
                        eprintln!("cannot write {}: {e}", path.display());
                        return 1;
                    }
                }
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
        let args: Vec<String> = ["rocky", "--seed", "42", "--freeze", "--id", "my-rock"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            parse_args(&args).unwrap(),
            ExportArgs {
                target: "rocky".into(),
                seed: Some(42),
                freeze: true,
                new_id: Some("my-rock".into()),
                packs: vec![],
                out: None,
            }
        );
        // A matching --seed on a gen id is redundant but consistent — allowed.
        let ok: Vec<String> = ["gen-rocky-000000000000002a", "--seed", "42"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(parse_args(&ok).is_ok());
    }

    #[test]
    fn bad_args_are_usage_errors() {
        let cases: Vec<Vec<String>> = vec![
            vec![],                                            // id required
            vec!["--seed".into(), "x".into(), "rocky".into()], // non-integer
            vec!["rocky".into(), "--nope".into()],             // unknown flag
            vec!["a".into(), "b".into()],                      // double positional
            // a --seed contradicting the gen id's encoded seed
            vec![
                "gen-rocky-000000000000002a".into(),
                "--seed".into(),
                "7".into(),
            ],
        ];
        for bad in cases {
            assert!(parse_args(&bad).is_err(), "{bad:?}");
        }
    }
}
