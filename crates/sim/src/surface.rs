//! The surface-material field abstraction (WI 497).
//!
//! The ground-contact parallel to the fluid-medium field: *do not hardcode
//! traction — model the terrain's surface material as a field.* A
//! [`SurfaceMaterial`] is plain coefficient data, so regolith, ice, and bedrock
//! are content, not code. It is consumed later by the wheel/ground-contact model
//! (the rover, WI 506); this work item defines and verifies the data only.

use serde::{Deserialize, Serialize};

/// Friction and rolling-resistance coefficients of a terrain surface material.
///
/// A new material is new constants, never new control flow. Coefficients are
/// dimensionless and non-negative.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceMaterial {
    /// Coefficient of friction at the contact patch (scales the tire force law).
    pub friction: f64,
    /// Rolling-resistance coefficient.
    pub rolling_resistance: f64,
}

impl SurfaceMaterial {
    /// Loose regolith: moderate friction, high rolling resistance.
    pub const REGOLITH: SurfaceMaterial = SurfaceMaterial {
        friction: 0.6,
        rolling_resistance: 0.08,
    };

    /// Solid bedrock: high friction, low rolling resistance.
    pub const BEDROCK: SurfaceMaterial = SurfaceMaterial {
        friction: 0.9,
        rolling_resistance: 0.02,
    };

    /// Ice: very low friction, low rolling resistance.
    pub const ICE: SurfaceMaterial = SurfaceMaterial {
        friction: 0.1,
        rolling_resistance: 0.01,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_materials_are_distinct() {
        assert_ne!(SurfaceMaterial::REGOLITH, SurfaceMaterial::BEDROCK);
        assert_ne!(SurfaceMaterial::BEDROCK, SurfaceMaterial::ICE);
        // Ordering matches intuition: ice < regolith < bedrock in grip.
        let (ice, regolith, bedrock) = (
            SurfaceMaterial::ICE.friction,
            SurfaceMaterial::REGOLITH.friction,
            SurfaceMaterial::BEDROCK.friction,
        );
        assert!(ice < regolith);
        assert!(regolith < bedrock);
    }

    #[test]
    fn coefficients_are_finite_and_bounded() {
        for m in [
            SurfaceMaterial::REGOLITH,
            SurfaceMaterial::BEDROCK,
            SurfaceMaterial::ICE,
        ] {
            assert!(m.friction.is_finite() && (0.0..=2.0).contains(&m.friction));
            assert!(m.rolling_resistance.is_finite() && m.rolling_resistance >= 0.0);
        }
    }

    #[test]
    fn ad_hoc_material_from_data_is_data_driven() {
        // A new material is new constants, no new code path (I3).
        let mud = SurfaceMaterial {
            friction: 0.35,
            rolling_resistance: 0.25,
        };
        assert!(mud.friction.is_finite() && mud.rolling_resistance >= 0.0);
    }
}
