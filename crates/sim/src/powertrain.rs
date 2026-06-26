//! Rover powertrain (WI 609): what turns throttle into drive torque while consuming a resource.
//!
//! Two sources, both gating the existing [`crate::rover::Wheel`] drive input rather than adding new
//! physics: a **combustion** engine burning fuel from a tank, or an **electric** battery optionally
//! recharged by solar. The powertrain owns a single [`Reservoir`] directly (no converter graph is
//! needed for one tank) and withdraws from it proportional to applied drive torque × time; when the
//! reservoir empties, drive torque falls to zero and the rover coasts.

use crate::control::ELECTRICITY;
use crate::resource::{Reservoir, ResourceType};
use serde::{Deserialize, Serialize};

/// Resource tag for combustion fuel (self-contained; the tag is cosmetic here).
const FUEL: ResourceType = ResourceType(0);

/// Per-drive-wheel drive torque at full throttle, per kg of rover mass (N·m/kg). Scales drive
/// authority with mass so any build pulls away. The mass-derived **default** when no motor is
/// selected (WI 652).
pub const DRIVE_TORQUE_PER_KG: f64 = 4.0;

/// The default wheel-spin top-speed cap (rad/s) when no motor is selected (mirrors
/// `rover::MAX_WHEEL_SPIN`); a selected motor overrides it (WI 652).
pub const DEFAULT_TOP_SPEED: f64 = 850.0;

/// A selectable rover motor (WI 652): the player sizes the drivetrain by tier instead of relying on
/// the mass-derived default. Each tier maps to a [`MotorSpec`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotorTier {
    /// Low torque, modest top-speed, light, frugal — for small/light builds.
    Economy,
    /// A balanced general-purpose motor.
    Standard,
    /// High top-speed and torque, heavier and thirstier.
    Performance,
    /// Maximum torque for heavy haulers; lower top-speed, heavy, thirsty.
    Heavy,
}

/// The stats a [`MotorTier`] delivers (WI 652).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotorSpec {
    /// Per-drive-wheel drive torque at full throttle, N·m.
    pub max_torque: f64,
    /// Wheel-spin top-speed cap, rad/s.
    pub top_speed: f64,
    /// Motor mass, kg.
    pub mass: f64,
    /// Consumption multiplier on the source's base draw (1.0 = nominal).
    pub draw: f64,
}

impl MotorTier {
    pub const ALL: [MotorTier; 4] = [
        MotorTier::Economy,
        MotorTier::Standard,
        MotorTier::Performance,
        MotorTier::Heavy,
    ];

    /// The stats for this tier.
    pub fn spec(self) -> MotorSpec {
        match self {
            MotorTier::Economy => MotorSpec { max_torque: 1.2e3, top_speed: 600.0, mass: 30.0, draw: 0.8 },
            MotorTier::Standard => MotorSpec { max_torque: 2.5e3, top_speed: 850.0, mass: 60.0, draw: 1.0 },
            MotorTier::Performance => MotorSpec { max_torque: 4.0e3, top_speed: 1200.0, mass: 90.0, draw: 1.4 },
            MotorTier::Heavy => MotorSpec { max_torque: 6.0e3, top_speed: 500.0, mass: 160.0, draw: 1.6 },
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            MotorTier::Economy => "Economy",
            MotorTier::Standard => "Standard",
            MotorTier::Performance => "Performance",
            MotorTier::Heavy => "Heavy",
        }
    }
}

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
    /// Wheel-spin top-speed cap (rad/s) — the selected motor's, or [`DEFAULT_TOP_SPEED`] (WI 652).
    pub top_speed: f64,
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
    motor: Option<MotorTier>,
    engines: usize,
    tanks: usize,
    batteries: usize,
    solar_panels: usize,
    mass: f64,
    drive_wheels: usize,
) -> RoverPowertrain {
    // A selected motor sizes torque + top-speed; multiple engines run parallel motors (sum torque).
    // No motor → the mass-derived default (the pre-652 behaviour).
    let (max_torque, top_speed, draw) = match motor {
        Some(t) => {
            let s = t.spec();
            (s.max_torque * (engines.max(1) as f64), s.top_speed, s.draw)
        }
        None => (mass * DRIVE_TORQUE_PER_KG, DEFAULT_TOP_SPEED, 1.0),
    };
    let drive_wheels = (drive_wheels.max(1)) as f64;
    if engines > 0 && tanks > 0 {
        let cap = tanks as f64 * FUEL_PER_TANK;
        RoverPowertrain {
            source: PowerSource::Combustion,
            reservoir: Reservoir::new(FUEL, cap, cap),
            max_torque,
            drive_wheels,
            consumption: COMBUSTION_CONSUMPTION * draw,
            top_speed,
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
            consumption: ELECTRIC_CONSUMPTION * draw,
            top_speed,
        }
    } else {
        // Default: solar exactly sustains full-throttle drive (drain == recharge), so it never strands.
        let consumption = ELECTRIC_CONSUMPTION * draw;
        let sustain = consumption * max_torque * drive_wheels;
        RoverPowertrain {
            source: PowerSource::Electric {
                solar_per_s: sustain,
            },
            reservoir: Reservoir::new(ELECTRICITY, DEFAULT_BATTERY_CAP, DEFAULT_BATTERY_CAP),
            max_torque,
            drive_wheels,
            consumption,
            top_speed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combustion_depletes_then_coasts() {
        let mut pt = build_powertrain(None, 1, 1, 0, 0, 100.0, 4);
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
        let mut pt = build_powertrain(None, 0, 0, 1, 1, 100.0, 4);
        assert_eq!(pt.label(), "charge");
        pt.reservoir.amount = 10.0;
        pt.drive_torque(0.0, 1.0); // idle: solar adds, nothing drawn
        assert!(pt.reservoir.amount > 10.0, "solar recharges when idle");
    }

    #[test]
    fn default_source_never_strands() {
        let mut pt = build_powertrain(None, 0, 0, 0, 0, 100.0, 4);
        let mut last = 0.0;
        for _ in 0..50_000 {
            last = pt.drive_torque(1.0, 0.01); // floor it indefinitely
        }
        assert!(pt.fraction() > 0.0, "default solar sustains full throttle");
        assert!(last > 0.0, "still driving");
    }

    #[test]
    fn motor_tiers_size_torque_and_top_speed_monotonically() {
        // Torque rises across the performance tiers; Heavy trades top-speed for torque.
        let e = MotorTier::Economy.spec();
        let s = MotorTier::Standard.spec();
        let p = MotorTier::Performance.spec();
        assert!(e.max_torque < s.max_torque && s.max_torque < p.max_torque);
        assert!(e.top_speed < s.top_speed && s.top_speed < p.top_speed);
        assert!(MotorTier::Heavy.spec().max_torque > p.max_torque);
        assert!(MotorTier::Heavy.spec().top_speed < e.top_speed);
    }

    #[test]
    fn selected_motor_overrides_mass_derived_torque_and_top_speed() {
        // No motor → mass-derived default; a motor → its spec (independent of mass).
        let default = build_powertrain(None, 1, 1, 0, 0, 100.0, 4);
        assert_eq!(default.max_torque, 100.0 * DRIVE_TORQUE_PER_KG);
        assert_eq!(default.top_speed, DEFAULT_TOP_SPEED);
        let perf = build_powertrain(Some(MotorTier::Performance), 1, 1, 0, 0, 100.0, 4);
        assert_eq!(perf.max_torque, MotorTier::Performance.spec().max_torque);
        assert_eq!(perf.top_speed, MotorTier::Performance.spec().top_speed);
        // Two engines run parallel motors → double torque.
        let twin = build_powertrain(Some(MotorTier::Performance), 2, 1, 0, 0, 100.0, 4);
        assert_eq!(twin.max_torque, 2.0 * MotorTier::Performance.spec().max_torque);
    }

    #[test]
    fn empty_battery_cuts_drive_without_solar() {
        let mut pt = build_powertrain(None, 0, 0, 1, 0, 100.0, 4); // battery, no solar
        let mut last = 1.0;
        for _ in 0..30_000 {
            last = pt.drive_torque(1.0, 0.01);
        }
        assert!(pt.fraction() < 1e-6 && last == 0.0, "drains to a coast");
    }
}
