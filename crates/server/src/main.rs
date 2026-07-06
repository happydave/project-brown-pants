//! The universe-server binary: parse flags, start, serve until interrupted.
//!
//! Usage:
//!   sounding_server --invite-token <token> [--addr 127.0.0.1:8790]
//!                   [--content <identity>] [--ttl <seconds>]
//!                   [--save <world-save.json>]

use sounding_server::{start, ServerOptions};

fn main() {
    let mut options = ServerOptions {
        addr: "127.0.0.1:8790".to_string(),
        ..Default::default()
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = |name: &str| {
            args.next().unwrap_or_else(|| {
                eprintln!("{name} requires a value");
                std::process::exit(2);
            })
        };
        match flag.as_str() {
            "--addr" => options.addr = value("--addr"),
            "--invite-token" => options.store.invite_token = value("--invite-token"),
            "--content" => options.store.content_identity = Some(value("--content")),
            "--ttl" => {
                let v = value("--ttl");
                options.store.lease_ttl = v.parse().unwrap_or_else(|_| {
                    eprintln!("--ttl: not a number: {v}");
                    std::process::exit(2);
                });
            }
            "--save" => options.save_path = Some(value("--save").into()),
            other => {
                eprintln!("unknown flag {other}");
                eprintln!(
                    "usage: sounding_server --invite-token <token> [--addr host:port] \
                     [--content <identity>] [--ttl <seconds>] [--save <path>]"
                );
                std::process::exit(2);
            }
        }
    }
    if options.store.invite_token.is_empty() {
        eprintln!("an --invite-token is required (the LAN trust boundary)");
        std::process::exit(2);
    }
    match start(options) {
        Ok(handle) => {
            println!("universe server listening on http://{}", handle.addr);
            // Serve until the process is interrupted.
            loop {
                std::thread::park();
            }
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
