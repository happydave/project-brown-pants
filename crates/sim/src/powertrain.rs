//! Rover powertrain (WI 609): what turns throttle into drive torque while consuming a resource.
//!
//! Two sources, both gating the existing [`crate::rover::Wheel`] drive input rather than adding new
//! physics: a **combustion** engine burning fuel from a tank, or an **electric** battery optionally
//! recharged by solar. The powertrain owns a single [`Reservoir`] directly (no converter graph is
//! needed for one tank) and withdraws from it proportional to applied drive torque × time; when the
//! reservoir empties, drive torque falls to zero and the rover coasts.

use crate::control::ELECTRICITY;
use crate::resource::{Reservoir, ResourceType};

/// Resource tag for combustion fuel (self-contained; the tag is cosmetic here).
const FUEL: ResourceType = ResourceType(0);

/// Per-drive-wheel drive torque at full throttle, per kg of rover mass (N·m/kg). Scales drive
/// authority with mass so any build pulls away.
pub const DRIVE_TORQUE_PER_KG: f64 = 4.0;

/// Fuel/charge consumed per (N·m of total drive torque · second).
const COMBUSTION_CONSUMPTION: f64 = 0.004;
const ELECTRIC_CONSUMPTION: f64 = 0.006;
/// Capacity contributed per tank / battery device.
const FUEL_PER_TANK: f64 = 800.0;
const BATTERY_CAP: f64 = 300.0;
/// Charge per second contributed per solar-panel part.
const SOLAR_PER_PANEL: f64 = 5.0;
/// Default battery capacity for a build with no power devices.
const DEFAULT_BATTERY_CAP: f64 = 200.0;

/// Where a rover's drive power comes from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PowerSource {
    /// Burns fuel; no recharge — run it dry and it coasts.
    Combustion,
    /// Draws charge; recharged at `solar_per_s` (0 if no panels).
    Electric { solar_per_s: f64 },
}

/// A rover's powertrain: a resource reservoir plus the torque it can deliver while supplied.
#[derive(Clone, Debug)]
pub struct RoverPowertrain {
    pub source: PowerSource,
    /// Fuel (combustion) or charge (electric).
    pub reservoir: Reservoir,
    /// Drive torque per drive wheel at full throttle, N·m (when supplied).
    pub max_torque: f64,
    /// Number of driven wheels (for total-torque consumption).
    pub drive_wheels: f64,
    /// Reservoir units consumed per (N·m of total drive torque · second).
    pub consumption: f64,
}

impl RoverPowertrain {
    /// Advance one frame and return the realized **per-wheel** drive torque. Solar (electric) tops up
    /// the reservoir first; then the throttle-commanded torque is delivered, scaled down to whatever
    /// resource is available over `dt` (zero when empty → the rover coasts). `throttle` ∈ [-1, 1].
    pub fn drive_torque(&mut self, throttle: f64, dt: f64) -> f64 {
        if let PowerSource::Electric { solar_per_s } = self.source {
            self.reservoir.amount =
                (self.reservoir.amount + solar_per_s * dt).min(self.reservoir.capacity);
        }
        let t = throttle.clamp(-1.0, 1.0);
        let desired = t.abs() * self.max_torque;
        if desired <= 0.0 || dt <= 0.0 {
            return 0.0;
        }
        let need = desired * self.drive_wheels * dt * self.consumption;
        let avail = need.min(self.reservoir.amount.max(0.0));
        self.reservoir.amount -= avail;
        let frac = if need > 1e-12 { avail / need } else { 1.0 };
        desired * frac * t.signum()
    }

    /// Reservoir fill fraction in `[0, 1]` (for the HUD).
    pub fn fraction(&self) -> f64 {
        if self.reservoir.capacity > 0.0 {
            (self.reservoir.amount / self.reservoir.capacity).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// HUD label for the resource.
    pub fn label(&self) -> &'static str {
        match self.source {
            PowerSource::Combustion => "fuel",
            PowerSource::Electric { .. } => "charge",
        }
    }
}

/// Build a rover powertrain from the build's device/part counts, mass, and drive-wheel count.
///
/// **Engine + Tank ⇒ combustion** (fuel from tanks); else **Battery ⇒ electric** (charge from the
/// batteries, solar from panels); else a **default electric** powertrain whose solar fully sustains
/// full-throttle driving, so a minimal build (wheels + control point) is never stranded.
pub fn build_powertrain(
    engines: usize,
    tanks: usize,
    batteries: usize,
    solar_panels: usize,
    mass: f64,
    drive_wheels: usize,
) -> RoverPowertrain {
    let max_torque = mass * DRIVE_TORQUE_PER_KG;
    let drive_wheels = (drive_wheels.max(1)) as f64;
    if engines > 0 && tanks > 0 {
        let cap = tanks as f64 * FUEL_PER_TANK;
        RoverPowertrain {
            source: PowerSource::Combustion,
            reservoir: Reservoir::new(FUEL, cap, cap),
            max_torque,
            drive_wheels,
            consumption: COMBUSTION_CONSUMPTION,
        }
    } else if batteries > 0 {
        let cap = batteries as f64 * BATTERY_CAP;
        RoverPowertrain {
            source: PowerSource::Electric {
                solar_per_s: solar_panels as f64 * SOLAR_PER_PANEL,
            },
            reservoir: Reservoir::new(ELECTRICITY, cap, cap),
            max_torque,
            drive_wheels,
            consumption: ELECTRIC_CONSUMPTION,
        }
    } else {
        // Default: solar exactly sustains full-throttle drive (drain == recharge), so it never strands.
        let sustain = ELECTRIC_CONSUMPTION * max_torque * drive_wheels;
        RoverPowertrain {
            source: PowerSource::Electric {
                solar_per_s: sustain,
            },
            reservoir: Reservoir::new(ELECTRICITY, DEFAULT_BATTERY_CAP, DEFAULT_BATTERY_CAP),
            max_torque,
            drive_wheels,
            consumption: ELECTRIC_CONSUMPTION,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combustion_depletes_then_coasts() {
        let mut pt = build_powertrain(1, 1, 0, 0, 100.0, 4);
        assert_eq!(pt.label(), "fuel");
        assert!(pt.drive_torque(1.0, 0.01) > 0.0, "drives while fuelled");
        let mut last = 1.0;
        for _ in 0..30_000 {
            last = pt.drive_torque(1.0, 0.01);
        }
        assert!(pt.fraction() < 1e-6, "tank empties: {}", pt.fraction());
        assert_eq!(last, 0.0, "empty ⇒ coasts");
    }

    #[test]
    fn electric_recharges_from_solar_when_idle() {
        let mut pt = build_powertrain(0, 0, 1, 1, 100.0, 4);
        assert_eq!(pt.label(), "charge");
        pt.reservoir.amount = 10.0;
        pt.drive_torque(0.0, 1.0); // idle: solar adds, nothing drawn
        assert!(pt.reservoir.amount > 10.0, "solar recharges when idle");
    }

    #[test]
    fn default_source_never_strands() {
        let mut pt = build_powertrain(0, 0, 0, 0, 100.0, 4);
        let mut last = 0.0;
        for _ in 0..50_000 {
            last = pt.drive_torque(1.0, 0.01); // floor it indefinitely
        }
        assert!(pt.fraction() > 0.0, "default solar sustains full throttle");
        assert!(last > 0.0, "still driving");
    }

    #[test]
    fn empty_battery_cuts_drive_without_solar() {
        let mut pt = build_powertrain(0, 0, 1, 0, 100.0, 4); // battery, no solar
        let mut last = 1.0;
        for _ in 0..30_000 {
            last = pt.drive_torque(1.0, 0.01);
        }
        assert!(pt.fraction() < 1e-6 && last == 0.0, "drains to a coast");
    }
}
