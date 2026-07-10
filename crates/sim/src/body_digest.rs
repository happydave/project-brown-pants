//! Body output digests + the output version (WI 888, slice 5 of
//! bodies-as-recipes): "same recipe + seed ⇒ same body, forever" as enforced
//! machinery, not hope.
//!
//! [`digest_body`] hashes a resolved [`BodyAsset`] over a **canonical byte
//! layout** (documented below) with an in-crate FNV-1a 64 — deliberately *not*
//! `std`'s `DefaultHasher`, whose algorithm is unstable across Rust releases,
//! and not an external hash crate (the sim stays dependency-free). The
//! committed golden file (`crates/sim/golden/body_digests.txt`) pins the
//! digests of N archetypes × M seeds plus the canonical fixed bodies, so **any
//! unintended bit change to generator output is a red test naming the body**
//! (the Elite 800-ly-typo lesson), and every kept/saved body stays
//! reproducible.
//!
//! **The version taxonomy** (three constants, three jobs):
//! - [`crate::content::CONTENT_FORMAT_VERSION`] — the authored RON pack
//!   grammar (migrate/reject on load).
//! - [`crate::persist::FORMAT_VERSION`] — the machine-written save envelope.
//! - [`BODY_OUTPUT_VERSION`] (here) — the **bit-level resolved-body output**:
//!   bump it deliberately when generator output is *meant* to change; the
//!   golden regeneration path refuses to rewrite digests without it. Future
//!   consumers: the world-save body digest (design D1/N1) and the
//!   surface-chunk cache key. Granularity is deliberately **global** for now
//!   (design-review N2): per-subsystem versions exist to protect persistent
//!   caches from over-invalidation, and no persistent cache exists yet —
//!   revisit at the persistence slice.
//!
//! **Digest layout contract** (changing it is itself an output change —
//! regenerate + bump): id, name, mu, radius, rotation axis (x, y, z), sidereal
//! period, the nine `FluidMedium` fields in declaration order, surface seed,
//! then the `terrain`/`crater`/`material`/`render` JSON areas. Strings and
//! JSON are length-prefixed UTF-8 (JSON via `serde_json::to_string`, which is
//! deterministic here: the default feature set stores maps sorted); `f64` as
//! `to_bits` little-endian; integers little-endian.
//!
//! **Deliberate-break policy** (recorded WI 888): the two known intentional
//! stream breaks — hash-derived child seeds (vs today's `seed ^ salt`) and
//! sampled-path derivation — are deferred, **batched**, and will land together
//! under one `BODY_OUTPUT_VERSION` bump alongside the slice that adds
//! generation steps (the surface-layer stack), with this harness auditing the
//! regeneration. Note the boundary this harness does *not* yet cover: surface
//! **heights** are excluded from goldens because the height/ejecta path still
//! uses libm transcendentals (`exp`, `acos`, trig — a known, tracked violation
//! of the design's no-libm noise rule; see the follow-up work item), so height
//! bits are not cross-platform-stable yet. Body *resolution* is clean:
//! `bodygen`/`body_derive` are integer-splitmix + sqrt only.

use crate::body_asset::BodyAsset;

/// The bit-level resolved-body output version. Bump **deliberately** when
/// generator output is meant to change, and regenerate the golden file in the
/// same commit (the regeneration test refuses digest changes without a bump).
pub const BODY_OUTPUT_VERSION: u32 = 1;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Minimal FNV-1a 64 accumulator over the canonical layout.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(FNV_OFFSET)
    }
    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(FNV_PRIME);
        }
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.bytes(&v.to_bits().to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.bytes(s.as_bytes());
    }
    fn json(&mut self, v: &serde_json::Value) {
        self.str(&v.to_string());
    }
}

/// [`digest_body`] as 16 lowercase hex digits — the golden file's spelling,
/// and the one persisted surface (WI 891: catalog refs and world-save body
/// records store and compare this string form).
pub fn digest_hex(a: &BodyAsset) -> String {
    format!("{:016x}", digest_body(a))
}

/// The stable digest of a resolved body — a pure function of the asset's
/// canonical byte layout (module docs). Feeds the golden harness now; the
/// world-save digest and surface-chunk cache key later.
///
/// Exhaustive destructuring (no `..`) is deliberate: adding a field to
/// [`BodyAsset`] (or its parts) is a **compile error here**, so a new field
/// can never be silently omitted from the digest — extending the layout is a
/// conscious act (and an output change: regenerate + bump).
pub fn digest_body(a: &BodyAsset) -> u64 {
    let BodyAsset {
        id,
        name,
        mu,
        radius,
        rotation,
        fluid_medium,
        surface,
        render,
    } = a;
    let crate::body_asset::Rotation {
        axis,
        sidereal_period,
    } = rotation;
    let crate::fluid::FluidMedium {
        atmosphere_surface_density,
        atmosphere_surface_pressure,
        atmosphere_scale_height,
        ocean_surface_density,
        ocean_surface_pressure,
        ocean_density_gradient,
        gravity,
        atmosphere_temperature,
        ocean_temperature,
    } = fluid_medium;
    let crate::body_asset::SurfaceRecipe {
        seed,
        terrain,
        crater,
        material,
    } = surface;

    let mut h = Fnv::new();
    h.str(id);
    h.str(name);
    h.f64(*mu);
    h.f64(*radius);
    h.f64(axis.x);
    h.f64(axis.y);
    h.f64(axis.z);
    h.f64(*sidereal_period);
    h.f64(*atmosphere_surface_density);
    h.f64(*atmosphere_surface_pressure);
    h.f64(*atmosphere_scale_height);
    h.f64(*ocean_surface_density);
    h.f64(*ocean_surface_pressure);
    h.f64(*ocean_density_gradient);
    h.f64(*gravity);
    h.f64(*atmosphere_temperature);
    h.f64(*ocean_temperature);
    h.u64(*seed);
    h.json(terrain);
    h.json(crater);
    h.json(material);
    h.json(render);
    h.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bodygen::{generate, Archetype};

    /// The committed golden matrix (compile-time copy — for the regeneration
    /// test this is exactly the "old"/committed side of the comparison).
    const GOLDEN: &str = include_str!("../golden/body_digests.txt");

    /// Seeds spanning the u64 range (0 and `u64::MAX` are legal here —
    /// `generate` takes a true u64; the WI-881 numeric-seed ceiling is a RON
    /// authoring concern, not a generator one).
    const SEEDS: [u64; 8] = [
        0,
        1,
        42,
        7777,
        123_456_789,
        987_654_321,
        4_503_599_627_370_496, // 2^52
        u64::MAX,
    ];

    /// The full golden matrix: every archetype × seed via the generator, plus
    /// the canonical fixed bodies via the embedded pack (covering the fixed
    /// path incl. the WI-886/887 derivation and the ladder).
    fn golden_matrix() -> Vec<(String, u64)> {
        let mut rows = Vec::new();
        for arch in Archetype::ALL {
            for seed in SEEDS {
                let body = generate(seed, arch);
                rows.push((format!("gen:{}:{seed}", arch.slug()), digest_body(&body)));
            }
        }
        for id in ["earthlike", "earthlike-ice-age"] {
            let body = crate::content::canonical_body(id);
            rows.push((format!("canonical:{id}"), digest_body(&body)));
        }
        rows
    }

    fn render_golden(rows: &[(String, u64)]) -> String {
        let mut out = format!("output_version: {BODY_OUTPUT_VERSION}\n");
        for (label, digest) in rows {
            out.push_str(&format!("{label} {digest:016x}\n"));
        }
        out
    }

    /// Strict parse of the committed file: `(version, [(label, digest)])`.
    fn parse_golden(text: &str) -> (u32, Vec<(String, u64)>) {
        let mut lines = text.lines();
        let header = lines.next().expect("golden file: missing header");
        let version = header
            .strip_prefix("output_version: ")
            .and_then(|v| v.parse().ok())
            .expect("golden file: malformed header");
        let entries = lines
            .map(|l| {
                let (label, hex) = l.rsplit_once(' ').expect("golden file: malformed line");
                let digest =
                    u64::from_str_radix(hex, 16).expect("golden file: malformed digest hex");
                (label.to_string(), digest)
            })
            .collect();
        (version, entries)
    }

    #[test]
    fn digest_is_deterministic_and_field_sensitive() {
        let a = crate::content::canonical_body("earthlike");
        assert_eq!(digest_body(&a), digest_body(&a.clone()));
        // Sensitivity: any single field change moves the digest.
        let mut b = a.clone();
        b.mu += 1.0;
        assert_ne!(digest_body(&a), digest_body(&b));
        let mut c = a.clone();
        c.surface.material = serde_json::json!({ "temperature": -1.0 });
        assert_ne!(digest_body(&a), digest_body(&c));
        // Distinct bodies differ.
        let ice = crate::content::canonical_body("earthlike-ice-age");
        assert_ne!(digest_body(&a), digest_body(&ice));
    }

    /// THE harness: every resolved body in the matrix matches its committed
    /// digest, and the file was generated at the current output version. A
    /// failure here means generator output changed — if unintended, fix the
    /// drift; if deliberate, bump [`BODY_OUTPUT_VERSION`] and regenerate
    /// (`regenerate_golden_body_digests`).
    #[test]
    fn golden_body_digests_match() {
        let (version, committed) = parse_golden(GOLDEN);
        assert_eq!(
            version, BODY_OUTPUT_VERSION,
            "golden file generated at output_version {version}, but the build \
             is {BODY_OUTPUT_VERSION} — regenerate the goldens"
        );
        let fresh = golden_matrix();
        assert_eq!(
            committed.len(),
            fresh.len(),
            "golden matrix size changed — regenerate deliberately"
        );
        for ((c_label, c_digest), (f_label, f_digest)) in committed.iter().zip(&fresh) {
            assert_eq!(c_label, f_label, "golden matrix order/labels drifted");
            assert_eq!(
                c_digest, f_digest,
                "resolved-body digest drifted for `{c_label}` — unintended \
                 generator change, or bump BODY_OUTPUT_VERSION deliberately"
            );
        }
    }

    /// Regeneration path (run explicitly:
    /// `cargo test -p sounding_sim regenerate_golden_body_digests -- --ignored`).
    /// **Refuses** to rewrite changed digests while the version is unchanged —
    /// the mechanical "a deliberate break requires a deliberate bump" gate.
    /// Bootstrap (empty committed file) is allowed.
    #[test]
    #[ignore = "regenerates the committed golden file; run explicitly"]
    fn regenerate_golden_body_digests() {
        let fresh = golden_matrix();
        if !GOLDEN.trim().is_empty() {
            let (old_version, old_entries) = parse_golden(GOLDEN);
            let digests_changed = old_entries
                .iter()
                .zip(&fresh)
                .any(|((ol, od), (fl, fd))| ol != fl || od != fd)
                || old_entries.len() != fresh.len();
            if digests_changed && old_version == BODY_OUTPUT_VERSION {
                let first = old_entries
                    .iter()
                    .zip(&fresh)
                    .find(|((ol, od), (fl, fd))| ol != fl || od != fd)
                    .map(|((ol, _), _)| ol.clone())
                    .unwrap_or_else(|| "matrix size".to_string());
                panic!(
                    "refusing to regenerate: digests changed (first: `{first}`) but \
                     BODY_OUTPUT_VERSION is still {BODY_OUTPUT_VERSION} — a deliberate \
                     output change requires a deliberate version bump first"
                );
            }
        }
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("golden/body_digests.txt");
        std::fs::write(&path, render_golden(&fresh)).expect("write golden file");
        println!("wrote {} entries to {}", fresh.len(), path.display());
    }
}
