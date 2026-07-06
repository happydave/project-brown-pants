//! The resident-participant binary (WI 858): a presence-only headless peer.
//!
//! Usage:
//!   sounding_participant --server http://host:8790 --invite <token>
//!       [--player peer] [--name Presence]
//!       [--blueprint content/blueprints/first-flight.json]
//!       [--content "scenario:first-flight:First Flight"]
//!       [--at dx,dz[,h]] [--orbit] [--for <seconds>]
//!
//! Defaults present as a craft parked a few tens of metres from the shipped
//! first-flight scenario's pad; `--orbit` publishes a canned LEO conic instead
//! (the record then propagates on its orbit for every observer). Exiting (or
//! being killed) simply lets the lease lapse — the vessel goes
//! stale-but-claimable (R5); protocol v1 has no logout by design.

use sounding_netclient::participant::{ParticipantConfig, ParticipantHandle};
use sounding_netclient::NetConfig;
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::orbit::Orbit;
use sounding_sim::sim::CentralBody;
use sounding_sim::vessel::MotionState;
use std::time::Duration;

/// Parsed command line (pure; unit-tested below).
#[derive(Debug, Clone, PartialEq)]
struct Opts {
    server: String,
    invite: String,
    player: String,
    name: String,
    blueprint: String,
    content: String,
    /// Pad offset (dx, dz, height above the surface radius), metres.
    at: (f64, f64, f64),
    orbit: bool,
    run_for: Option<f64>,
}

fn parse_args(args: &[String]) -> Result<Opts, String> {
    let mut opts = Opts {
        server: String::new(),
        invite: String::new(),
        player: "peer".to_string(),
        name: "Presence".to_string(),
        blueprint: "content/blueprints/first-flight.json".to_string(),
        content: "scenario:first-flight:First Flight".to_string(),
        at: (15.0, 0.0, 0.0),
        orbit: false,
        run_for: None,
    };
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = |name: &str| -> Result<String, String> {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match flag.as_str() {
            "--server" => opts.server = value("--server")?,
            "--invite" => opts.invite = value("--invite")?,
            "--player" => opts.player = value("--player")?,
            "--name" => opts.name = value("--name")?,
            "--blueprint" => opts.blueprint = value("--blueprint")?,
            "--content" => opts.content = value("--content")?,
            "--at" => {
                let v = value("--at")?;
                let parts: Vec<f64> = v
                    .split(',')
                    .map(|p| p.trim().parse::<f64>())
                    .collect::<Result<_, _>>()
                    .map_err(|_| format!("--at: not numbers: {v}"))?;
                opts.at = match parts.as_slice() {
                    [dx, dz] => (*dx, *dz, 0.0),
                    [dx, dz, h] => (*dx, *dz, *h),
                    _ => return Err(format!("--at expects dx,dz[,h], got {v}")),
                };
            }
            "--orbit" => opts.orbit = true,
            "--for" => {
                let v = value("--for")?;
                let secs: f64 = v.parse().map_err(|_| format!("--for: not a number: {v}"))?;
                if secs <= 0.0 {
                    return Err(format!("--for must be positive, got {v}"));
                }
                opts.run_for = Some(secs);
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if opts.server.is_empty() || opts.invite.is_empty() {
        return Err("--server and --invite are required".to_string());
    }
    Ok(opts)
}

/// The participant's rails motion from the parsed options (pure).
fn motion_from(opts: &Opts) -> MotionState {
    if opts.orbit {
        // A canned circular LEO conic about the earthlike body (R + 400 km).
        let mu = CentralBody::EARTHLIKE.mu;
        let r = CentralBody::EARTHLIKE.radius + 400_000.0;
        let orbit = Orbit::from_state(
            mu,
            glam::DVec2::new(r, 0.0),
            glam::DVec2::new(0.0, (mu / r).sqrt()),
            0.0,
        )
        .expect("a circular orbit is bound");
        MotionState::Conic {
            frame: FrameId::CENTRAL_BODY,
            orbit,
        }
    } else {
        let (dx, dz, h) = opts.at;
        MotionState::SurfaceFix {
            position: WorldPos::new(
                FrameId::CENTRAL_BODY,
                glam::DVec3::new(dx, CentralBody::EARTHLIKE.radius + h, dz),
            ),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = match parse_args(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            eprintln!(
                "usage: sounding_participant --server <url> --invite <token> [--player p] \
                 [--name n] [--blueprint path] [--content identity] [--at dx,dz[,h]] \
                 [--orbit] [--for seconds]"
            );
            std::process::exit(2);
        }
    };

    // Fail fast on the blueprint before any network (legible, named error).
    let craft = match sounding_sim::library::load_craft(std::path::Path::new(&opts.blueprint)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("blueprint {}: {e}", opts.blueprint);
            std::process::exit(1);
        }
    };

    let motion = motion_from(&opts);
    let handle = ParticipantHandle::start(ParticipantConfig {
        net: NetConfig::new(&opts.server, &opts.invite, &opts.player, &opts.content),
        vessel_name: opts.name.clone(),
        craft,
        motion,
        tick: Duration::from_millis(250),
        log: true,
    });
    println!(
        "participant: \"{}\" as {} on {} — vessel {}",
        opts.name, opts.player, opts.server, handle.vessel_id
    );
    println!("participant: presence-only; exit/kill lets the lease lapse (R5)");

    match opts.run_for {
        Some(secs) => {
            std::thread::sleep(Duration::from_secs_f64(secs));
            handle.shutdown();
            println!("participant: done ({secs}s run)");
        }
        None => loop {
            std::thread::park();
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parse_defaults_requirements_and_overrides() {
        // Required pair enforced.
        assert!(parse_args(&strs(&["--server", "http://h:1"])).is_err());
        assert!(parse_args(&[]).is_err());

        // Defaults.
        let o = parse_args(&strs(&["--server", "http://h:1", "--invite", "t"])).unwrap();
        assert_eq!(o.player, "peer");
        assert_eq!(o.name, "Presence");
        assert_eq!(o.blueprint, "content/blueprints/first-flight.json");
        assert_eq!(o.content, "scenario:first-flight:First Flight");
        assert_eq!(o.at, (15.0, 0.0, 0.0));
        assert!(!o.orbit && o.run_for.is_none());

        // Overrides + --at forms.
        let o = parse_args(&strs(&[
            "--server",
            "http://h:1",
            "--invite",
            "t",
            "--player",
            "bot",
            "--name",
            "Buoy",
            "--at",
            "3, -4, 2.5",
            "--orbit",
            "--for",
            "10",
        ]))
        .unwrap();
        assert_eq!(o.player, "bot");
        assert_eq!(o.at, (3.0, -4.0, 2.5));
        assert!(o.orbit);
        assert_eq!(o.run_for, Some(10.0));

        // Rejections are legible.
        assert!(
            parse_args(&strs(&["--server", "h", "--invite", "t", "--for", "0"]))
                .unwrap_err()
                .contains("positive")
        );
        assert!(
            parse_args(&strs(&["--server", "h", "--invite", "t", "--at", "x,y"]))
                .unwrap_err()
                .contains("--at")
        );
        assert!(parse_args(&strs(&["--bogus"]))
            .unwrap_err()
            .contains("--bogus"));
    }

    #[test]
    fn motion_modes_map_to_the_right_rails_shapes() {
        let base = parse_args(&strs(&["--server", "h", "--invite", "t"])).unwrap();
        match motion_from(&base) {
            MotionState::SurfaceFix { position } => {
                assert_eq!(position.pos.x, 15.0);
                assert_eq!(position.pos.y, CentralBody::EARTHLIKE.radius);
            }
            _ => panic!("default is a parked surface fix"),
        }
        let mut orbit = base.clone();
        orbit.orbit = true;
        match motion_from(&orbit) {
            MotionState::Conic { orbit, .. } => {
                assert!(orbit.is_bound());
                let (p, _) = orbit.position_velocity(0.0);
                assert!((p.length() - (CentralBody::EARTHLIKE.radius + 400_000.0)).abs() < 1.0);
            }
            _ => panic!("--orbit is a conic"),
        }
    }
}
