"""
components.py
-------------
Thermofluid network component hierarchy.

Class tree:
    FluidComponent  (abstract base)
    ├── Pipe        – Darcy-Weisbach with Haaland friction, minor losses
    ├── Pump        – quadratic characteristic curve hp = A·Q² + B·Q + C
    ├── Valve       – K-value model (or Cv if needed later)
    ├── Junction    – internal network node with optional demand
    └── Reservoir   – fixed-head boundary condition

All head-loss functions are signed:
    positive  → energy dissipated in the positive-flow direction
    negative  → energy added   (pumps only)

Units throughout: SI  (m, m³/s, Pa, kg, s, W)
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Optional

from fluid_props import (
    DENSITY, VISCOSITY, GRAVITY,
    DEFAULT_ROUGHNESS, DEFAULT_MATERIAL, DEFAULT_CONDITION,
    friction_factor, reynolds_number,
    lookup_roughness, lookup_fitting_k,
    NOMINAL_TO_METRES, FITTING_K, FITTING_CATEGORIES, FITTING_DIAMETERS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════════════════════════════════════════

class FluidComponent:
    """
    Abstract base class for all thermofluid network components.

    State variables (updated after each solver iteration):
        pressure        Pa      gauge pressure at component reference point
        mass_flow_rate  kg/s    mass flow rate through component
        velocity        m/s     mean flow velocity (for edge components)

    Sub-classes must implement:
        compute_head_loss(Q)    → float   [m]
        compute_reynolds(Q)     → float   [-]
        compute_friction_factor(Q) → float [-]
        dhead_loss_dQ(Q)        → float   [m / (m³/s)]
        validate()              → List[str]
    """

    def __init__(self, component_id: str, name: str = ""):
        self.id: str = component_id
        self.name: str = name or component_id

        # ── Post-solve state ───────────────────────────────────────────
        self.pressure: float        = 0.0   # Pa
        self.mass_flow_rate: float  = 0.0   # kg/s
        self.velocity: float        = 0.0   # m/s

    # ── Required interface ─────────────────────────────────────────────
    def compute_head_loss(self, Q: float) -> float:
        raise NotImplementedError

    def compute_reynolds(self, Q: float) -> float:
        raise NotImplementedError

    def compute_friction_factor(self, Q: float) -> float:
        raise NotImplementedError

    def dhead_loss_dQ(self, Q: float, eps: float = 1e-8) -> float:
        """
        Derivative dh_L/dQ.
        Default: central finite difference.
        Override with analytic expression in sub-classes for speed.
        """
        h_plus  = self.compute_head_loss(Q + eps)
        h_minus = self.compute_head_loss(Q - eps)
        return (h_plus - h_minus) / (2.0 * eps)

    def validate(self) -> List[str]:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict) -> "FluidComponent":
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id!r})"


# ═══════════════════════════════════════════════════════════════════════════════
# Pipe
# ═══════════════════════════════════════════════════════════════════════════════

class Pipe(FluidComponent):
    """
    Circular pipe with Darcy-Weisbach major losses and lumped minor losses.

    Head loss (signed, direction-aware):

        h_L = sign(Q) · [ f(Re, ε/D) · L/D · v²/(2g)  +  Σ(K) · v²/(2g) ]

    where v = |Q| / A is the mean velocity.

    Derivative:
        dh_L/dQ = [ f·L/D + K_total ] · 2·|Q| / (2g · A²)
                  + df/dQ · L/D · |Q|² / (2g · A²)   (≈ small, included numerically)

    Args:
        diameter        m   inner diameter
        length          m   pipe length
        roughness       m   absolute roughness ε
        elevation_change m  (z_out − z_in); positive = uphill
        K_minor         -   sum of minor-loss coefficients (elbows, tees, …)
    """

    def __init__(self,
                 component_id: str,
                 diameter: float      = 0.1,
                 length: float        = 100.0,
                 roughness: float     = DEFAULT_ROUGHNESS,
                 elevation_change: float = 0.0,
                 K_minor: float       = 0.0,
                 material: str        = DEFAULT_MATERIAL,
                 condition: str       = DEFAULT_CONDITION,
                 name: str            = ""):
        super().__init__(component_id, name)
        self.diameter          = diameter
        self.length            = length
        self.roughness         = roughness
        self.elevation_change  = elevation_change
        self.K_minor           = K_minor
        self.material          = material
        self.condition         = condition

    @property
    def area(self) -> float:
        """Cross-sectional flow area [m²]."""
        return math.pi * self.diameter ** 2 / 4.0

    @property
    def eps_over_D(self) -> float:
        return self.roughness / self.diameter

    def compute_reynolds(self, Q: float) -> float:
        """Re = ρ·|V|·D / μ"""
        if abs(Q) < 1e-14:
            return 0.0
        V = abs(Q) / self.area
        return reynolds_number(V, self.diameter)

    def compute_friction_factor(self, Q: float) -> float:
        """Darcy friction factor via fluid_props.friction_factor()."""
        Re = self.compute_reynolds(Q)
        return friction_factor(Re, self.eps_over_D)

    def compute_head_loss(self, Q: float) -> float:
        """
        Total head loss [m], signed by flow direction.
        Elevation change is NOT included here — it is embedded in the
        piezometric head difference at the solver level:
            H_from − H_to = h_friction + h_minor  (elevation baked into H)
        """
        if abs(Q) < 1e-14:
            return 0.0

        V_abs = abs(Q) / self.area
        sign  = math.copysign(1.0, Q)
        f     = self.compute_friction_factor(Q)

        h_major = f * (self.length / self.diameter) * V_abs**2 / (2.0 * GRAVITY)
        h_minor = self.K_minor * V_abs**2 / (2.0 * GRAVITY)

        return sign * (h_major + h_minor)

    def dhead_loss_dQ(self, Q: float, eps: float = 1e-8) -> float:
        """
        Full analytic dh_L/dQ including df/dQ via Haaland chain rule.

        h_L = [f(Q)·L/D + K] · Q·|Q| / (2g·A²)

        dh_L/dQ = [f·L/D + K]·|Q|/(g·A²)    [term 1: constant-f]
                + df/dQ · L/D · |Q|²/(2g·A²) [term 2: Jacobian correction]

        df/dQ = df/dRe * dRe/dQ  via Haaland.
        Laminar / transition zones use finite difference (negligible error).
        """
        if abs(Q) < 1e-14:
            # Near-zero: symmetric FD avoids 0/0
            return (self.compute_head_loss(eps) - self.compute_head_loss(-eps)) / (2.0 * eps)

        A_cs  = self.area          # cross-sectional area  [m²]
        A2    = A_cs ** 2
        L_D   = self.length / self.diameter
        K     = self.K_minor
        f     = self.compute_friction_factor(Q)
        Re    = self.compute_reynolds(Q)

        # Term 1: dh_L/dQ with f treated as constant (dominant term)
        #   h_L = (f*L/D + K) * Q*|Q| / (2g*A²)
        #   d/dQ = (f*L/D + K) * 2|Q| / (2g*A²) = (f*L/D + K)*|Q| / (g*A²)
        term1 = (f * L_D + K) * abs(Q) / (GRAVITY * A2)

        # Term 2: df/dQ correction via chain rule  df/dQ = df/dRe * dRe/dQ
        #   dRe/dQ = rho*D / (mu * A)   (Re = rho*|Q|*D / (mu*A), derivative is unsigned)
        dRe_dQ = DENSITY * self.diameter / (VISCOSITY * A_cs)

        delta_Re = max(abs(Re) * 1e-5, 0.5)       # FD step on Re
        f_plus   = friction_factor(Re + delta_Re, self.roughness / self.diameter)
        f_minus  = friction_factor(Re - delta_Re, self.roughness / self.diameter)
        df_dRe   = (f_plus - f_minus) / (2.0 * delta_Re)

        #   h_L term2 = df/dQ * L/D * Q*|Q| / (2g*A²)
        #   d/dQ(Q*|Q|) = 2|Q|, so d/dQ(df * L/D * Q*|Q| / 2gA²) = df_dRe*dRe_dQ * L/D * |Q|/gA²
        term2 = df_dRe * dRe_dQ * L_D * Q * abs(Q) / (2.0 * GRAVITY * A2)

        return term1 + term2

    def validate(self) -> List[str]:
        errs = []
        if self.diameter <= 0:
            errs.append(f"[{self.id}] diameter must be > 0 m (got {self.diameter})")
        if self.length < 0:
            errs.append(f"[{self.id}] length must be ≥ 0 m (got {self.length})")
        if self.roughness < 0:
            errs.append(f"[{self.id}] roughness must be ≥ 0 m (got {self.roughness})")
        if self.K_minor < 0:
            errs.append(f"[{self.id}] K_minor must be ≥ 0 (got {self.K_minor})")
        return errs

    def to_dict(self) -> dict:
        return {
            "type": "Pipe",
            "id": self.id,
            "name": self.name,
            "diameter": self.diameter,
            "length": self.length,
            "roughness": self.roughness,
            "elevation_change": self.elevation_change,
            "K_minor": self.K_minor,
            "material": self.material,
            "condition": self.condition,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Pipe":
        return cls(
            component_id    = d["id"],
            diameter        = d.get("diameter", 0.1),
            length          = d.get("length", 100.0),
            roughness       = d.get("roughness", DEFAULT_ROUGHNESS),
            elevation_change= d.get("elevation_change", 0.0),
            K_minor         = d.get("K_minor", 0.0),
            material        = d.get("material", DEFAULT_MATERIAL),
            condition       = d.get("condition", DEFAULT_CONDITION),
            name            = d.get("name", d["id"]),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pump
# ═══════════════════════════════════════════════════════════════════════════════

class Pump(FluidComponent):
    """
    Centrifugal pump with quadratic characteristic curve:

        hp(Q) = A·Q² + B·Q + C        [m]

    where Q is volumetric flow rate [m³/s].

    Convention:
        A < 0   (head decreases with flow — physically correct for most pumps)
        C > 0   (shut-off head at Q = 0)
        B ≥ 0   (optional rising slope near Q=0, uncommon)

    In the network solver the pump is an edge component.  Its head loss is:
        h_L = −hp(Q)
    so that the energy equation H_from − H_to − h_L = 0 becomes
        H_to = H_from + hp(Q)   ✓

    The pump is always oriented from the suction node to the discharge node.
    Reverse flow (Q < 0) is allowed but yields h_L > 0 (pump resists backflow).

    Args:
        A, B, C     Quadratic curve coefficients  [m/(m³/s)², m/(m³/s), m]
        diameter    Reference diameter for velocity/Re calculation [m]
        is_on       If False, pump is bypassed (hp = 0, h_L = 0)
    """

    def __init__(self,
                 component_id: str,
                 A: float       = -8000.0,
                 B: float       = 0.0,
                 C: float       = 25.0,
                 diameter: float = 0.1,
                 is_on: bool    = True,
                 name: str      = ""):
        super().__init__(component_id, name)
        self.A         = A
        self.B         = B
        self.C         = C
        self.diameter  = diameter
        self.is_on     = is_on

    @property
    def area(self) -> float:
        return math.pi * self.diameter**2 / 4.0

    def compute_pump_head(self, Q: float) -> float:
        """hp = A·Q² + B·Q + C  [m].  Only valid for Q ≥ 0."""
        if not self.is_on:
            return 0.0
        return self.A * Q**2 + self.B * Q + self.C

    def compute_head_loss(self, Q: float) -> float:
        """
        Network sign convention: h_L = −hp(Q).
        For forward flow: h_L < 0 (energy is ADDED).
        """
        if not self.is_on:
            return 0.0
        hp = self.compute_pump_head(Q)
        return -hp

    def dhead_loss_dQ(self, Q: float, eps: float = 1e-8) -> float:
        """Analytic derivative: d(−hp)/dQ = −(2A·Q + B)"""
        if not self.is_on:
            return 0.0
        return -(2.0 * self.A * Q + self.B)

    def compute_reynolds(self, Q: float) -> float:
        if abs(Q) < 1e-14:
            return 0.0
        V = abs(Q) / self.area
        return reynolds_number(V, self.diameter)

    def compute_friction_factor(self, Q: float) -> float:
        return 0.0   # not applicable for pump

    def get_operating_range(self) -> tuple[float, float]:
        """
        Return (Q_min, Q_max) where hp > 0, found by solving A·Q² + B·Q + C = 0.
        Returns (0, 0) if no positive-head range exists.
        """
        if self.A == 0.0:
            return (0.0, max(0.0, -self.C / self.B) if self.B != 0 else 0.0)
        disc = self.B**2 - 4.0 * self.A * self.C
        if disc < 0:
            return (0.0, 0.0)
        Q_max = (-self.B - math.sqrt(disc)) / (2.0 * self.A)
        return (0.0, max(0.0, Q_max))

    def curve_data(self, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Return (Q_array, hp_array) for plotting."""
        _, Q_max = self.get_operating_range()
        if Q_max <= 0:
            Q_max = 0.05
        Q = np.linspace(0.0, Q_max * 1.2, n_points)
        hp = np.array([self.compute_pump_head(q) for q in Q])
        return Q, hp

    def validate(self) -> List[str]:
        errs = []
        if self.C < 0:
            errs.append(f"[{self.id}] shut-off head C should be ≥ 0 (got {self.C})")
        if self.diameter <= 0:
            errs.append(f"[{self.id}] diameter must be > 0 m (got {self.diameter})")
        if self.A > 0:
            errs.append(f"[{self.id}] curve coeff A should be ≤ 0 for stable operation "
                        f"(got {self.A}); head would increase with flow — unusual")
        return errs

    def to_dict(self) -> dict:
        return {
            "type": "Pump",
            "id": self.id,
            "name": self.name,
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "diameter": self.diameter,
            "is_on": self.is_on,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Pump":
        return cls(
            component_id = d["id"],
            A            = d.get("A", -8000.0),
            B            = d.get("B", 0.0),
            C            = d.get("C", 25.0),
            diameter     = d.get("diameter", 0.1),
            is_on        = d.get("is_on", True),
            name         = d.get("name", d["id"]),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Valve
# ═══════════════════════════════════════════════════════════════════════════════

class Valve(FluidComponent):
    """
    Control valve modelled as a lumped minor loss:

        h_L = sign(Q) · K · v²/(2g)

    where v = |Q|/A is mean velocity.

    A fully closed valve is represented by setting is_open = False,
    which returns h_L = +∞ equivalent (a very large number in practice
    — the solver will drive Q → 0 through that branch).

    Args:
        diameter    m   valve bore (for velocity / Re)
        K           -   loss coefficient (fully open; user-supplied)
        is_open     bool
    """

    CLOSED_K = 1e8    # effective K for closed valve

    def __init__(self,
                 component_id: str,
                 diameter: float  = 0.1,
                 K: float         = 5.0,
                 is_open: bool    = True,
                 name: str        = ""):
        super().__init__(component_id, name)
        self.diameter = diameter
        self.K        = K
        self.is_open  = is_open

    @property
    def area(self) -> float:
        return math.pi * self.diameter**2 / 4.0

    @property
    def effective_K(self) -> float:
        return self.K if self.is_open else self.CLOSED_K

    def compute_reynolds(self, Q: float) -> float:
        if abs(Q) < 1e-14:
            return 0.0
        V = abs(Q) / self.area
        return reynolds_number(V, self.diameter)

    def compute_friction_factor(self, Q: float) -> float:
        return 0.0   # not applicable; captured in K

    def compute_head_loss(self, Q: float) -> float:
        if abs(Q) < 1e-14:
            return 0.0
        V_abs = abs(Q) / self.area
        sign  = math.copysign(1.0, Q)
        return sign * self.effective_K * V_abs**2 / (2.0 * GRAVITY)

    def dhead_loss_dQ(self, Q: float, eps: float = 1e-8) -> float:
        if abs(Q) < 1e-10:
            eps_fd = 1e-7
            return (self.compute_head_loss(eps_fd) -
                    self.compute_head_loss(-eps_fd)) / (2.0 * eps_fd)
        return self.effective_K * abs(Q) / (GRAVITY * self.area**2)

    def validate(self) -> List[str]:
        errs = []
        if self.diameter <= 0:
            errs.append(f"[{self.id}] diameter must be > 0 m (got {self.diameter})")
        if self.K < 0:
            errs.append(f"[{self.id}] K must be ≥ 0 (got {self.K})")
        return errs

    def to_dict(self) -> dict:
        return {
            "type": "Valve",
            "id": self.id,
            "name": self.name,
            "diameter": self.diameter,
            "K": self.K,
            "is_open": self.is_open,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Valve":
        return cls(
            component_id = d["id"],
            diameter     = d.get("diameter", 0.1),
            K            = d.get("K", 5.0),
            is_open      = d.get("is_open", True),
            name         = d.get("name", d["id"]),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Fitting  (edge component — valve, elbow, tee with tabulated K)
# ═══════════════════════════════════════════════════════════════════════════════

class Fitting(FluidComponent):
    """
    Pipe fitting with K-value from standard tables (Table 6.5).

    Physics identical to Valve:  h_L = sign(Q) · K · v²/(2g)

    The K-value is auto-looked-up from fitting_subtype + connection_type +
    nominal_diameter but can be manually overridden.

    Args:
        fitting_subtype     e.g. "Globe valve", "90° elbow, regular"
        connection_type     "Screwed" or "Flanged"
        nominal_diameter_in Nominal pipe diameter in inches
        K                   Loss coefficient (auto-populated, overridable)
        diameter            Inner diameter in metres (auto from nominal)
    """

    def __init__(self,
                 component_id: str,
                 fitting_subtype: str  = "90° elbow, regular",
                 connection_type: str  = "Screwed",
                 nominal_diameter_in: float = 1.0,
                 K: Optional[float]    = None,
                 diameter: Optional[float] = None,
                 name: str             = ""):
        super().__init__(component_id, name)
        self.fitting_subtype    = fitting_subtype
        self.connection_type    = connection_type
        self.nominal_diameter_in = nominal_diameter_in

        # Auto-lookup diameter from nominal size
        if diameter is not None:
            self.diameter = diameter
        else:
            self.diameter = NOMINAL_TO_METRES.get(
                nominal_diameter_in, 0.02664)  # default 1"

        # Auto-lookup K from table; user can override
        if K is not None:
            self.K = K
        else:
            self.K = lookup_fitting_k(
                connection_type, fitting_subtype, nominal_diameter_in)

    @property
    def area(self) -> float:
        return math.pi * self.diameter**2 / 4.0

    def compute_reynolds(self, Q: float) -> float:
        if abs(Q) < 1e-14:
            return 0.0
        V = abs(Q) / self.area
        return reynolds_number(V, self.diameter)

    def compute_friction_factor(self, Q: float) -> float:
        return 0.0

    def compute_head_loss(self, Q: float) -> float:
        if abs(Q) < 1e-14:
            return 0.0
        V_abs = abs(Q) / self.area
        sign  = math.copysign(1.0, Q)
        return sign * self.K * V_abs**2 / (2.0 * GRAVITY)

    def dhead_loss_dQ(self, Q: float, eps: float = 1e-8) -> float:
        if abs(Q) < 1e-10:
            eps_fd = 1e-7
            return (self.compute_head_loss(eps_fd) -
                    self.compute_head_loss(-eps_fd)) / (2.0 * eps_fd)
        return self.K * abs(Q) / (GRAVITY * self.area**2)

    def update_k_from_table(self):
        """Re-lookup K from table (call after changing subtype/connection/diameter)."""
        self.K = lookup_fitting_k(
            self.connection_type, self.fitting_subtype, self.nominal_diameter_in)
        self.diameter = NOMINAL_TO_METRES.get(
            self.nominal_diameter_in, self.diameter)

    def validate(self) -> List[str]:
        errs = []
        if self.diameter <= 0:
            errs.append(f"[{self.id}] diameter must be > 0 m (got {self.diameter})")
        if self.K < 0:
            errs.append(f"[{self.id}] K must be ≥ 0 (got {self.K})")
        return errs

    def to_dict(self) -> dict:
        return {
            "type": "Fitting",
            "id": self.id,
            "name": self.name,
            "fitting_subtype": self.fitting_subtype,
            "connection_type": self.connection_type,
            "nominal_diameter_in": self.nominal_diameter_in,
            "K": self.K,
            "diameter": self.diameter,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fitting":
        return cls(
            component_id       = d["id"],
            fitting_subtype    = d.get("fitting_subtype", "90° elbow, regular"),
            connection_type    = d.get("connection_type", "Screwed"),
            nominal_diameter_in= d.get("nominal_diameter_in", 1.0),
            K                  = d.get("K"),
            diameter           = d.get("diameter"),
            name               = d.get("name", d["id"]),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Junction  (network node)
# ═══════════════════════════════════════════════════════════════════════════════

class Junction(FluidComponent):
    """
    Internal network node.

    Piezometric head H = P/(ρg) + z is the unknown solved for at each junction.

    Args:
        elevation   m   node elevation above datum (z in Bernoulli equation)
        demand      m³/s  net withdrawal at this node (positive = withdrawing)
    """

    def __init__(self,
                 component_id: str,
                 elevation: float = 0.0,
                 demand: float    = 0.0,
                 name: str        = ""):
        super().__init__(component_id, name)
        self.elevation = elevation
        self.demand    = demand
        self.head: float = 0.0   # piezometric head — set by solver

    @property
    def pressure_head(self) -> float:
        """P/(ρg) = H − z  [m]"""
        return self.head - self.elevation

    @property
    def pressure_Pa(self) -> float:
        return self.pressure_head * DENSITY * GRAVITY

    def compute_head_loss(self, Q: float) -> float:
        return 0.0

    def compute_reynolds(self, Q: float) -> float:
        return 0.0

    def compute_friction_factor(self, Q: float) -> float:
        return 0.0

    def dhead_loss_dQ(self, Q: float, eps: float = 1e-8) -> float:
        return 0.0

    def validate(self) -> List[str]:
        errs = []
        if self.demand < 0:
            errs.append(f"[{self.id}] demand < 0 (injection); ensure this is intentional")
        return errs

    def to_dict(self) -> dict:
        return {
            "type": "Junction",
            "id": self.id,
            "name": self.name,
            "elevation": self.elevation,
            "demand": self.demand,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Junction":
        return cls(
            component_id = d["id"],
            elevation    = d.get("elevation", 0.0),
            demand       = d.get("demand", 0.0),
            name         = d.get("name", d["id"]),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Reservoir  (fixed-head boundary)
# ═══════════════════════════════════════════════════════════════════════════════

class Reservoir(FluidComponent):
    """
    Fixed-head boundary condition node.

    The total piezometric head H = P_surface/(ρg) + z_surface is prescribed.
    For an open free-surface reservoir at elevation z:  H = z.

    Args:
        total_head  m   prescribed piezometric head (fixed boundary condition)
    """

    def __init__(self,
                 component_id: str,
                 total_head: float = 10.0,
                 name: str         = ""):
        super().__init__(component_id, name)
        self.total_head = total_head
        self.head       = total_head    # alias — kept in sync

    def compute_head_loss(self, Q: float) -> float:
        return 0.0

    def compute_reynolds(self, Q: float) -> float:
        return 0.0

    def compute_friction_factor(self, Q: float) -> float:
        return 0.0

    def dhead_loss_dQ(self, Q: float, eps: float = 1e-8) -> float:
        return 0.0

    def validate(self) -> List[str]:
        return []

    def to_dict(self) -> dict:
        return {
            "type": "Reservoir",
            "id": self.id,
            "name": self.name,
            "total_head": self.total_head,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Reservoir":
        return cls(
            component_id = d["id"],
            total_head   = d.get("total_head", 10.0),
            name         = d.get("name", d["id"]),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Factory helper
# ═══════════════════════════════════════════════════════════════════════════════

_COMPONENT_REGISTRY = {
    "Pipe":      Pipe,
    "Pump":      Pump,
    "Valve":     Valve,
    "Fitting":   Fitting,
    "Junction":  Junction,
    "Reservoir": Reservoir,
}

def component_from_dict(d: dict) -> FluidComponent:
    """Deserialise any component from its dict representation."""
    kind = d.get("type")
    cls  = _COMPONENT_REGISTRY.get(kind)
    if cls is None:
        raise ValueError(f"Unknown component type: {kind!r}")
    return cls.from_dict(d)

EDGE_COMPONENT_TYPES = {"Pipe", "Pump", "Valve", "Fitting"}
NODE_COMPONENT_TYPES = {"Junction", "Reservoir"}
