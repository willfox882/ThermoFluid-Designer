"""
units.py
--------
Display-unit layer for results output (Improvement #5).

The solver and ALL stored model data are always SI internally
(m, m³/s, Pa, m/s, W, °C).  This module converts those SI values to the
user-selected display system for *output only* — results tables, the pump
plot, CSV exports, canvas overlays and the results read-out.  Component INPUT
forms remain SI; toggling units never mutates the model.

Two systems
───────────
  SI        – metric engineering display :  m,  L/s,  kPa,  m/s,  kW,  °C
  Imperial  – US customary               :  ft, gpm,  psi,  ft/s, hp,  °F

Each quantity exposes a ``*_value(si)`` converter (scalar or NumPy array) and a
``*_label()`` returning the current unit string, so a header and its column of
values can never drift apart.
"""

from __future__ import annotations

SI       = "SI"
IMPERIAL = "Imperial"

# ── Conversion factors (multiply an SI value to get the target unit) ──────────
_M_TO_FT    = 3.280839895013123          # metre        → foot
_M3S_TO_GPM = 15850.323141488905         # m³/s         → US gallon/minute
_PA_TO_PSI  = 1.0 / 6894.757293168361    # pascal       → psi
_W_TO_HP    = 1.0 / 745.6998715822702    # watt         → mechanical horsepower

_system = SI


# ── System state ──────────────────────────────────────────────────────────────

def set_system(name: str) -> None:
    """Select the active display system (``units.SI`` or ``units.IMPERIAL``)."""
    global _system
    if name not in (SI, IMPERIAL):
        raise ValueError(f"Unknown unit system: {name!r}")
    _system = name


def get_system() -> str:
    return _system


def is_imperial() -> bool:
    return _system == IMPERIAL


def toggle() -> str:
    """Flip SI ↔ Imperial; return the new system."""
    set_system(IMPERIAL if _system == SI else SI)
    return _system


# ── Length / head (m) ─────────────────────────────────────────────────────────

def head_value(m):
    return m * _M_TO_FT if is_imperial() else m

def head_label() -> str:
    return "ft" if is_imperial() else "m"


# ── Volumetric flow (m³/s) → L/s or gpm ───────────────────────────────────────

def flow_value(m3s):
    return m3s * _M3S_TO_GPM if is_imperial() else m3s * 1000.0

def flow_label() -> str:
    return "gpm" if is_imperial() else "L/s"


# ── Velocity (m/s) ────────────────────────────────────────────────────────────

def velocity_value(mps):
    return mps * _M_TO_FT if is_imperial() else mps

def velocity_label() -> str:
    return "ft/s" if is_imperial() else "m/s"


# ── Pressure (Pa) → kPa or psi ────────────────────────────────────────────────

def pressure_value(pa):
    return pa * _PA_TO_PSI if is_imperial() else pa / 1000.0

def pressure_label() -> str:
    return "psi" if is_imperial() else "kPa"


# ── Power (W) → kW or hp ──────────────────────────────────────────────────────

def power_value(w):
    return w * _W_TO_HP if is_imperial() else w / 1000.0

def power_label() -> str:
    return "hp" if is_imperial() else "kW"


def power_value_from_kw(kw):
    """Convenience for call sites that already hold power in kW."""
    return power_value(kw * 1000.0)


# ── Temperature (°C) → °C or °F ───────────────────────────────────────────────

def temperature_value(c):
    return c * 9.0 / 5.0 + 32.0 if is_imperial() else c

def temperature_label() -> str:
    return "°F" if is_imperial() else "°C"
