"""
fluid_props.py
--------------
Physical constants and fluid property functions.
Default fluid: water at 20°C (SI units throughout).

Future: add temperature-dependent properties and multi-fluid support.
"""

# ── Water at 20 °C ────────────────────────────────────────────────────────────
DENSITY      = 998.2       # kg/m³
VISCOSITY    = 1.002e-3    # Pa·s  (dynamic)
KIN_VISC     = VISCOSITY / DENSITY   # m²/s  (kinematic)
GRAVITY      = 9.81        # m/s²
SPEC_WEIGHT  = DENSITY * GRAVITY     # N/m³  (γ = ρg)

# ── Reynolds regime boundaries ────────────────────────────────────────────────
RE_LAMINAR     = 2300.0
RE_TURBULENT   = 4000.0

# ── Pipe material roughness (absolute ε in metres) ───────────────────────────
# Keyed by (material, condition).  Values from Munson, Young & Okiishi Table 6.1
ROUGHNESS_TABLE = {
    # Steel
    ("Steel", "Sheet metal, new"):    0.05e-3,
    ("Steel", "Stainless, new"):      0.002e-3,
    ("Steel", "Commercial, new"):     0.046e-3,
    ("Steel", "Riveted"):             3.0e-3,
    ("Steel", "Rusted"):              2.0e-3,
    # Iron
    ("Iron", "Cast, new"):            0.26e-3,
    ("Iron", "Wrought, new"):         0.046e-3,
    ("Iron", "Galvanized, new"):      0.15e-3,
    ("Iron", "Asphalted cast"):       0.12e-3,
    # Brass
    ("Brass", "Drawn, new"):          0.002e-3,
    # Plastic
    ("Plastic", "Drawn tubing"):      0.0015e-3,
    # Glass
    ("Glass", "Smooth"):              0.0,
    # Concrete
    ("Concrete", "Smoothed"):         0.04e-3,
    ("Concrete", "Rough"):            2.0e-3,
    # Rubber
    ("Rubber", "Smoothed"):           0.01e-3,
    # Wood
    ("Wood", "Stave"):                0.5e-3,
}

# Organised per-material for dropdown menus
MATERIAL_CONDITIONS = {
    "Steel":    ["Commercial, new", "Sheet metal, new", "Stainless, new", "Riveted", "Rusted"],
    "Iron":     ["Cast, new", "Wrought, new", "Galvanized, new", "Asphalted cast"],
    "Brass":    ["Drawn, new"],
    "Plastic":  ["Drawn tubing"],
    "Glass":    ["Smooth"],
    "Concrete": ["Smoothed", "Rough"],
    "Rubber":   ["Smoothed"],
    "Wood":     ["Stave"],
}

# Legacy flat dict kept for backward compat
ROUGHNESS = {
    "commercial_steel":    46e-6,
    "galvanized_iron":    150e-6,
    "cast_iron":          260e-6,
    "concrete":          1200e-6,
    "drawn_tubing":         1.5e-6,
    "pvc":                  1.5e-6,
    "smooth":               0.0,
}

DEFAULT_ROUGHNESS = ROUGHNESS["commercial_steel"]  # 4.6×10⁻⁵ m
DEFAULT_MATERIAL  = "Steel"
DEFAULT_CONDITION = "Commercial, new"


# ── Fitting K-value lookup (Table 6.5, Munson/Young/Okiishi) ─────────────────
# Structure: FITTING_K[connection_type][fitting_subtype] = {diameter_in: K}
# Diameter is nominal in inches.  Use nearest available.

FITTING_K = {
    "Screwed": {
        "Globe valve":        {0.5: 14,   1: 8.2, 2: 6.9, 4: 5.7},
        "Gate valve":         {0.5: 0.30, 1: 0.24, 2: 0.16, 4: 0.11},
        "Swing check valve":  {0.5: 5.1,  1: 2.9, 2: 2.1, 4: 2.0},
        "Angle valve":        {0.5: 9.0,  1: 4.7, 2: 2.0, 4: 1.0},
        "45° elbow, regular": {0.5: 0.39, 1: 0.32, 2: 0.30, 4: 0.29},
        "90° elbow, regular": {0.5: 2.0,  1: 1.5, 2: 0.95, 4: 0.64},
        "90° elbow, long radius": {0.5: 1.0, 1: 0.72, 2: 0.41, 4: 0.23},
        "180° elbow, regular":{0.5: 2.0,  1: 1.5, 2: 0.95, 4: 0.64},
        "Tee, line flow":     {0.5: 0.90, 1: 0.90, 2: 0.90, 4: 0.90},
        "Tee, branch flow":   {0.5: 2.4,  1: 1.8, 2: 1.4, 4: 1.1},
    },
    "Flanged": {
        "Globe valve":        {1: 13,   2: 8.5,  4: 6.0,  8: 5.8,  20: 5.5},
        "Gate valve":         {1: 0.80, 2: 0.35, 4: 0.16, 8: 0.07, 20: 0.03},
        "Swing check valve":  {1: 2.0,  2: 2.0,  4: 2.0,  8: 2.0,  20: 2.0},
        "Angle valve":        {1: 4.5,  2: 2.4,  4: 2.0,  8: 2.0,  20: 2.0},
        "45° elbow, long radius": {1: 0.21, 2: 0.20, 4: 0.19, 8: 0.16, 20: 0.14},
        "90° elbow, regular": {1: 0.50, 2: 0.39, 4: 0.30, 8: 0.26, 20: 0.21},
        "90° elbow, long radius": {1: 0.40, 2: 0.30, 4: 0.19, 8: 0.15, 20: 0.10},
        "180° elbow, regular":{1: 0.41, 2: 0.35, 4: 0.30, 8: 0.25, 20: 0.20},
        "180° elbow, long radius":{1: 0.40, 2: 0.30, 4: 0.21, 8: 0.15, 20: 0.10},
        "Tee, line flow":     {1: 0.24, 2: 0.19, 4: 0.14, 8: 0.10, 20: 0.07},
        "Tee, branch flow":   {1: 1.0,  2: 0.80, 4: 0.64, 8: 0.58, 20: 0.41},
    },
}

# All fitting subtypes grouped by category (for dropdown menus)
FITTING_CATEGORIES = {
    "Valves": ["Globe valve", "Gate valve", "Swing check valve", "Angle valve"],
    "Elbows": ["45° elbow, regular", "45° elbow, long radius",
               "90° elbow, regular", "90° elbow, long radius",
               "180° elbow, regular", "180° elbow, long radius"],
    "Tees":   ["Tee, line flow", "Tee, branch flow"],
}

# Nominal diameters available per connection type (inches)
FITTING_DIAMETERS = {
    "Screwed": [0.5, 1, 2, 4],
    "Flanged": [1, 2, 4, 8, 20],
}

# Conversion: nominal pipe diameter (inches) → approximate inner diameter (metres)
# Using Schedule 40 standard pipe dimensions
NOMINAL_TO_METRES = {
    0.5:  0.01580,   # 15.8 mm
    1:    0.02664,   # 26.6 mm
    2:    0.05250,   # 52.5 mm
    4:    0.10226,   # 102.3 mm
    8:    0.20272,   # 202.7 mm
    20:   0.48890,   # 488.9 mm
}


def lookup_fitting_k(connection_type: str, fitting_subtype: str,
                     nominal_diameter_in: float) -> float:
    """
    Look up K-value for a fitting from Table 6.5.
    Uses nearest available diameter if exact match not found.
    Returns 0.0 if fitting/connection combo not in table.
    """
    conn_table = FITTING_K.get(connection_type, {})
    diam_table = conn_table.get(fitting_subtype, {})
    if not diam_table:
        return 0.0
    # Find nearest available diameter
    available = sorted(diam_table.keys())
    nearest = min(available, key=lambda d: abs(d - nominal_diameter_in))
    return diam_table[nearest]


def lookup_roughness(material: str, condition: str) -> float:
    """Look up absolute roughness [m] for a material+condition pair."""
    return ROUGHNESS_TABLE.get((material, condition), DEFAULT_ROUGHNESS)


def reynolds_number(velocity: float, diameter: float,
                    rho: float = DENSITY, mu: float = VISCOSITY) -> float:
    """Re = ρ·V·D / μ"""
    return rho * abs(velocity) * diameter / mu


def friction_factor_laminar(Re: float) -> float:
    """Darcy friction factor for fully developed laminar flow: f = 64/Re"""
    if Re < 1e-12:
        return 0.0
    return 64.0 / Re


def friction_factor_haaland(Re: float, eps_over_D: float) -> float:
    """
    Haaland (1983) explicit approximation of the Colebrook-White equation.
    Accurate to ±2% for Re > 3000.

        1/√f = -1.8 · log₁₀[(ε/D / 3.7)^1.11 + 6.9/Re]

    Args:
        Re          : Reynolds number (> 0)
        eps_over_D  : relative roughness ε/D

    Returns:
        Darcy friction factor f
    """
    import math
    inner = (eps_over_D / 3.7) ** 1.11 + 6.9 / Re
    f_inv = -1.8 * math.log10(inner)
    return (1.0 / f_inv) ** 2


def friction_factor(Re: float, eps_over_D: float) -> float:
    """
    Compute Darcy friction factor with smooth transition between regimes.

    Regimes:
        Re < 2300               → Laminar   : f = 64/Re
        2300 ≤ Re < 4000        → Transition : linear interpolation
        Re ≥ 4000               → Turbulent  : Haaland explicit equation

    Args:
        Re          : Reynolds number
        eps_over_D  : relative roughness (ε/D)

    Returns:
        Darcy friction factor f  (dimensionless)
    """
    if Re < 1e-10:
        return 0.0

    f_lam = 64.0 / max(Re, 1e-10)

    if Re < RE_LAMINAR:
        return f_lam

    # Turbulent value (evaluated at actual Re or transition endpoint)
    f_turb = friction_factor_haaland(max(Re, RE_TURBULENT), eps_over_D)

    if Re >= RE_TURBULENT:
        return f_turb

    # Smooth linear interpolation through transition zone
    t = (Re - RE_LAMINAR) / (RE_TURBULENT - RE_LAMINAR)   # 0 → 1
    f_lam_at_2300 = 64.0 / RE_LAMINAR
    f_turb_at_4000 = friction_factor_haaland(RE_TURBULENT, eps_over_D)
    return f_lam_at_2300 + t * (f_turb_at_4000 - f_lam_at_2300)
