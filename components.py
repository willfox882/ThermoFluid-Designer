"""
components.py
-------------
Thermofluid network component hierarchy.

Class tree:
    FluidComponent  (abstract base)
    ├── Pipe        – Darcy-Weisbach with Haaland friction, minor losses via fittings
    ├── Pump        – quadratic characteristic curve hp = A·Q² + B·Q + C
    ├── Valve       – K-value model
    ├── Junction    – internal network node with optional demand
    ├── Reservoir   – fixed-head boundary condition
    └── Fitting     – legacy standalone fitting (kept for backward compat; see FittingAttachment)

FittingAttachment – child object of a Pipe (NOT a FluidComponent).
    Fittings are now attached to pipes, not placed as standalone edges.

Units throughout: SI  (m, m³/s, Pa, kg, s, W)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import fluid_props as fp
from fluid_props import (
    GRAVITY, ATMOSPHERIC_PRESSURE,
    DEFAULT_ROUGHNESS, DEFAULT_MATERIAL, DEFAULT_CONDITION,
    friction_factor, reynolds_number,
    lookup_roughness, lookup_fitting_k,
    NOMINAL_TO_METRES, FITTING_K, FITTING_CATEGORIES, FITTING_DIAMETERS,
)
# NOTE: DENSITY, VISCOSITY and VAPOR_PRESSURE are *temperature-dependent* state
# that the solver updates per-solve, so they are read live via the `fp` module
# (fp.DENSITY / fp.VISCOSITY / fp.VAPOR_PRESSURE) rather than imported by name
# (which would freeze them at import time).  GRAVITY / ATMOSPHERIC_PRESSURE are
# true constants and are imported directly.


# ═══════════════════════════════════════════════════════════════════════════════
# FittingAttachment  (child of Pipe — not a FluidComponent)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FittingAttachment:
    """
    A pipe fitting attached to a Pipe segment as a child object.

    Position along the pipe (position_t in [0,1]) is cosmetic only —
    it does not affect calculations. The pipe's total minor loss is
    the sum of all attached fittings' effective_K values.

    K_override takes precedence over K_default when set.
    """
    fitting_id: str
    fitting_subtype: str = "90° elbow, regular"
    connection_type: str = "Screwed"
    nominal_diameter_in: float = 1.0
    K_default: float = 0.3
    K_override: Optional[float] = None
    position_t: float = 0.5       # 0–1 along pipe length (cosmetic)
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.fitting_id

    @property
    def effective_K(self) -> float:
        """K coefficient used in head-loss calculations."""
        return self.K_override if self.K_override is not None else self.K_default

    def to_dict(self) -> dict:
        return {
            "fitting_id":          self.fitting_id,
            "fitting_subtype":     self.fitting_subtype,
            "connection_type":     self.connection_type,
            "nominal_diameter_in": self.nominal_diameter_in,
            "K_default":           self.K_default,
            "K_override":          self.K_override,
            "position_t":          self.position_t,
            "name":                self.name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FittingAttachment":
        return cls(
            fitting_id          = d["fitting_id"],
            fitting_subtype     = d.get("fitting_subtype", "90° elbow, regular"),
            connection_type     = d.get("connection_type", "Screwed"),
            nominal_diameter_in = d.get("nominal_diameter_in", 1.0),
            K_default           = d.get("K_default", 0.3),
            K_override          = d.get("K_override"),
            position_t          = d.get("position_t", 0.5),
            name                = d.get("name", d["fitting_id"]),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════════════════════════════════════════

class FluidComponent:
    """Abstract base class for all thermofluid network components."""

    def __init__(self, component_id: str, name: str = ""):
        self.id: str   = component_id
        self.name: str = name or component_id

        self.pressure: float        = 0.0
        self.mass_flow_rate: float  = 0.0
        self.velocity: float        = 0.0

    def compute_head_loss(self, Q: float) -> float:
        raise NotImplementedError

    def compute_reynolds(self, Q: float) -> float:
        raise NotImplementedError

    def compute_friction_factor(self, Q: float) -> float:
        raise NotImplementedError

    def dhead_loss_dQ(self, Q: float, eps: float = 1e-8) -> float:
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
    Circular pipe with Darcy-Weisbach major losses and per-fitting minor losses.

    Minor loss coefficient
    ──────────────────────
    K_total is determined in priority order:
        1. K_minor_override  — if explicitly set (non-None), used directly.
        2. sum(f.effective_K for f in self.fittings)  — sum of child fittings.

    Head loss (signed):
        h_L = sign(Q) · [ f·L/D · v²/(2g)  +  K_total · v²/(2g) ]
    """

    def __init__(self,
                 component_id: str,
                 diameter: float          = 0.1,
                 length: float            = 100.0,
                 roughness: float         = DEFAULT_ROUGHNESS,
                 K_minor: float           = 0.0,
                 material: str            = DEFAULT_MATERIAL,
                 condition: str           = DEFAULT_CONDITION,
                 name: str                = ""):
        super().__init__(component_id, name)
        self.diameter         = diameter
        self.length           = length
        self.roughness        = roughness
        self.material         = material
        self.condition        = condition

        # K_minor_override: explicit override (takes priority over fittings).
        # For backward compat: if K_minor > 0 on construction, treat as override.
        self._K_minor_override: Optional[float] = K_minor if K_minor > 0.0 else None
        self.fittings: List[FittingAttachment] = []

    # ── K_minor property ──────────────────────────────────────────────────────

    @property
    def K_minor(self) -> float:
        """Total minor loss K. Override takes precedence over fittings sum."""
        if self._K_minor_override is not None:
            return self._K_minor_override
        return sum(f.effective_K for f in self.fittings)

    @K_minor.setter
    def K_minor(self, value: float):
        self._K_minor_override = float(value) if value is not None else None

    @property
    def K_minor_override(self) -> Optional[float]:
        return self._K_minor_override

    @K_minor_override.setter
    def K_minor_override(self, value):
        self._K_minor_override = float(value) if value is not None else None

    @property
    def K_minor_computed(self) -> float:
        """Sum of fitting K values (regardless of override)."""
        return sum(f.effective_K for f in self.fittings)

    # ── Geometry ──────────────────────────────────────────────────────────────

    @property
    def area(self) -> float:
        return math.pi * self.diameter ** 2 / 4.0

    @property
    def eps_over_D(self) -> float:
        return self.roughness / self.diameter

    # ── Physics ───────────────────────────────────────────────────────────────

    def compute_reynolds(self, Q: float) -> float:
        if abs(Q) < 1e-14:
            return 0.0
        V = abs(Q) / self.area
        return reynolds_number(V, self.diameter)

    def compute_friction_factor(self, Q: float) -> float:
        Re = self.compute_reynolds(Q)
        return friction_factor(Re, self.eps_over_D)

    def compute_head_loss(self, Q: float) -> float:
        if abs(Q) < 1e-14:
            return 0.0
        V_abs = abs(Q) / self.area
        sign  = math.copysign(1.0, Q)
        f     = self.compute_friction_factor(Q)
        h_major = f * (self.length / self.diameter) * V_abs**2 / (2.0 * GRAVITY)
        h_minor = self.K_minor * V_abs**2 / (2.0 * GRAVITY)
        return sign * (h_major + h_minor)

    def dhead_loss_dQ(self, Q: float, eps: float = 1e-8) -> float:
        if abs(Q) < 1e-14:
            return (self.compute_head_loss(eps) - self.compute_head_loss(-eps)) / (2.0 * eps)

        A_cs  = self.area
        A2    = A_cs ** 2
        L_D   = self.length / self.diameter
        K     = self.K_minor
        f     = self.compute_friction_factor(Q)
        Re    = self.compute_reynolds(Q)

        term1 = (f * L_D + K) * abs(Q) / (GRAVITY * A2)

        # Re depends on |Q|, so dRe/d|Q| = ρD/(μ·A) (sign-independent).
        dRe_dQ   = fp.DENSITY * self.diameter / (fp.VISCOSITY * A_cs)
        delta_Re = max(abs(Re) * 1e-5, 0.5)
        f_plus   = friction_factor(Re + delta_Re, self.roughness / self.diameter)
        f_minus  = friction_factor(Re - delta_Re, self.roughness / self.diameter)
        df_dRe   = (f_plus - f_minus) / (2.0 * delta_Re)
        # h_L = (f·L/D + K)·Q·|Q| / (2g·A²) is ODD in Q, so dh_L/dQ must be EVEN.
        # The friction-derivative term therefore carries Q² (= |Q|·|Q|), NOT
        # Q·|Q| — the latter would flip sign for reverse flow and corrupt the
        # Jacobian (up to ~28% error at low Re) on any edge with negative flow.
        term2    = df_dRe * dRe_dQ * L_D * (Q * Q) / (2.0 * GRAVITY * A2)

        return term1 + term2

    def validate(self) -> List[str]:
        errs = []
        if self.diameter <= 0:
            errs.append(f"[{self.id}] diameter must be > 0 m (got {self.diameter})")
        if self.length < 0:
            errs.append(f"[{self.id}] length must be ≥ 0 m (got {self.length})")
        if self.roughness < 0:
            errs.append(f"[{self.id}] roughness must be ≥ 0 m (got {self.roughness})")
        if self._K_minor_override is not None and self._K_minor_override < 0:
            errs.append(f"[{self.id}] K_minor override must be ≥ 0 (got {self._K_minor_override})")
        return errs

    def to_dict(self) -> dict:
        return {
            "type":             "Pipe",
            "id":               self.id,
            "name":             self.name,
            "diameter":         self.diameter,
            "length":           self.length,
            "roughness":        self.roughness,
            "K_minor":          self.K_minor,          # legacy compat field
            "K_minor_override": self._K_minor_override,
            "fittings":         [f.to_dict() for f in self.fittings],
            "material":         self.material,
            "condition":        self.condition,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Pipe":
        obj = cls(
            component_id    = d["id"],
            diameter        = d.get("diameter", 0.1),
            length          = d.get("length", 100.0),
            roughness       = d.get("roughness", DEFAULT_ROUGHNESS),
            K_minor         = d.get("K_minor", 0.0),   # backward compat
            material        = d.get("material", DEFAULT_MATERIAL),
            condition       = d.get("condition", DEFAULT_CONDITION),
            name            = d.get("name", d["id"]),
        )
        # New-format explicit override wins over K_minor heuristic
        if "K_minor_override" in d:
            obj._K_minor_override = d["K_minor_override"]
        # Load attached fittings
        for fd in d.get("fittings", []):
            obj.fittings.append(FittingAttachment.from_dict(fd))
        return obj


# ═══════════════════════════════════════════════════════════════════════════════
# Pump
# ═══════════════════════════════════════════════════════════════════════════════

class Pump(FluidComponent):
    """
    Centrifugal pump:  hp(Q) = A·Q² + B·Q + C   [m]
    Network sign convention:  h_L = −hp(Q)

    When switched OFF the pump is modelled as a CLOSED LINK (check-valve
    behaviour, matching EPANET): a very large hydraulic resistance ``OFF_K``
    drives the flow through it to ≈0.  This is more physical than a lossless
    pass-through and gives the energy row a non-zero Q-derivative, avoiding a
    singular Jacobian when an idle pump is the only edge at a free node.
    """

    OFF_K = 1e8   # closed-link loss coefficient (same magnitude as a shut valve)

    def __init__(self,
                 component_id: str,
                 A: float        = -8000.0,
                 B: float        = 0.0,
                 C: float        = 25.0,
                 diameter: float = 0.1,
                 is_on: bool     = True,
                 desired_flow_rate: float = 0.001,
                 npsh_required: float     = 2.0,
                 name: str       = ""):
        super().__init__(component_id, name)
        self.A                 = A
        self.B                 = B
        self.C                 = C
        self.diameter          = diameter
        self.is_on             = is_on
        self.desired_flow_rate = desired_flow_rate   # [m³/s]  used for power estimate
        self.npsh_required     = npsh_required       # [m]  NPSHr from manufacturer
        self.npsh_available: float = 0.0
        self.is_cavitating:  bool  = False

    @property
    def area(self) -> float:
        return math.pi * self.diameter**2 / 4.0

    def compute_npsha(self, P_suction: float, V_suction: float) -> float:
        """
        Available Net Positive Suction Head (NPSHa).

            NPSHa = (P_abs - P_vapor) / (ρg) + V_suction² / (2g)

        ``P_suction`` is the solver's node pressure, which is GAUGE (an open
        reservoir surface has P = 0).  NPSHa requires the ABSOLUTE suction
        pressure, so atmospheric pressure is added.  Omitting it understates
        NPSHa by ~10.3 m and raises a false cavitation alarm for essentially
        every normal installation.
        """
        P_abs       = P_suction + ATMOSPHERIC_PRESSURE
        head_static = (P_abs - fp.VAPOR_PRESSURE) / (fp.DENSITY * GRAVITY)
        head_vel    = V_suction**2 / (2.0 * GRAVITY)
        self.npsh_available = head_static + head_vel
        self.is_cavitating  = self.npsh_available < self.npsh_required
        return self.npsh_available

    def compute_pump_head(self, Q: float) -> float:
        if not self.is_on:
            return 0.0
        return self.A * Q**2 + self.B * Q + self.C

    def compute_head_loss(self, Q: float) -> float:
        if not self.is_on:
            # OFF → closed link: high-resistance loss that blocks the flow.
            if abs(Q) < 1e-14:
                return 0.0
            V_abs = abs(Q) / self.area
            return math.copysign(1.0, Q) * self.OFF_K * V_abs**2 / (2.0 * GRAVITY)
        return -self.compute_pump_head(Q)

    def dhead_loss_dQ(self, Q: float, eps: float = 1e-8) -> float:
        if not self.is_on:
            # Derivative of the closed-link resistance (same form as a shut valve).
            if abs(Q) < 1e-10:
                eps_fd = 1e-7
                return (self.compute_head_loss(eps_fd)
                        - self.compute_head_loss(-eps_fd)) / (2.0 * eps_fd)
            return self.OFF_K * abs(Q) / (GRAVITY * self.area**2)
        return -(2.0 * self.A * Q + self.B)

    def compute_reynolds(self, Q: float) -> float:
        if abs(Q) < 1e-14:
            return 0.0
        V = abs(Q) / self.area
        return reynolds_number(V, self.diameter)

    def compute_friction_factor(self, Q: float) -> float:
        return 0.0

    def get_operating_range(self) -> tuple[float, float]:
        if self.A == 0.0:
            return (0.0, max(0.0, -self.C / self.B) if self.B != 0 else 0.0)
        disc = self.B**2 - 4.0 * self.A * self.C
        if disc < 0:
            return (0.0, 0.0)
        Q_max = (-self.B - math.sqrt(disc)) / (2.0 * self.A)
        return (0.0, max(0.0, Q_max))

    def curve_data(self, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
        _, Q_max = self.get_operating_range()
        if Q_max <= 0:
            Q_max = 0.05
        Q  = np.linspace(0.0, Q_max * 1.2, n_points)
        hp = np.array([self.compute_pump_head(q) for q in Q])
        return Q, hp

    def validate(self) -> List[str]:
        errs = []
        if self.C < 0:
            errs.append(f"[{self.id}] shut-off head C should be ≥ 0 (got {self.C})")
        if self.diameter <= 0:
            errs.append(f"[{self.id}] diameter must be > 0 m (got {self.diameter})")
        if self.A > 0:
            errs.append(f"[{self.id}] curve coeff A should be ≤ 0 for stable operation")
        if self.npsh_required < 0:
            errs.append(f"[{self.id}] NPSHr must be ≥ 0 (got {self.npsh_required})")
        return errs

    def to_dict(self) -> dict:
        return {
            "type":              "Pump",
            "id":                self.id,
            "name":              self.name,
            "A":                 self.A,
            "B":                 self.B,
            "C":                 self.C,
            "diameter":          self.diameter,
            "is_on":             self.is_on,
            "desired_flow_rate": self.desired_flow_rate,
            "npsh_required":     self.npsh_required,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Pump":
        return cls(
            component_id       = d["id"],
            A                  = d.get("A", -8000.0),
            B                  = d.get("B", 0.0),
            C                  = d.get("C", 25.0),
            diameter           = d.get("diameter", 0.1),
            is_on              = d.get("is_on", True),
            desired_flow_rate  = d.get("desired_flow_rate", 0.001),
            npsh_required      = d.get("npsh_required", 2.0),
            name               = d.get("name", d["id"]),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Valve
# ═══════════════════════════════════════════════════════════════════════════════

class Valve(FluidComponent):
    """Control valve: h_L = sign(Q) · K · v²/(2g)"""

    CLOSED_K = 1e8

    def __init__(self,
                 component_id: str,
                 diameter: float = 0.1,
                 K: float        = 5.0,
                 is_open: bool   = True,
                 name: str       = ""):
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
        return reynolds_number(abs(Q) / self.area, self.diameter)

    def compute_friction_factor(self, Q: float) -> float:
        return 0.0

    def compute_head_loss(self, Q: float) -> float:
        if abs(Q) < 1e-14:
            return 0.0
        V_abs = abs(Q) / self.area
        return math.copysign(1.0, Q) * self.effective_K * V_abs**2 / (2.0 * GRAVITY)

    def dhead_loss_dQ(self, Q: float, eps: float = 1e-8) -> float:
        if abs(Q) < 1e-10:
            eps_fd = 1e-7
            return (self.compute_head_loss(eps_fd) - self.compute_head_loss(-eps_fd)) / (2*eps_fd)
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
            "type":     "Valve",
            "id":       self.id,
            "name":     self.name,
            "diameter": self.diameter,
            "K":        self.K,
            "is_open":  self.is_open,
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
# Fitting  (legacy standalone edge — kept for backward compat file loading)
# ═══════════════════════════════════════════════════════════════════════════════

class Fitting(FluidComponent):
    """
    Legacy standalone fitting component.
    New networks attach fittings to pipes via FittingAttachment instead.
    This class is kept for loading old save files.
    """

    def __init__(self,
                 component_id: str,
                 fitting_subtype: str       = "90° elbow, regular",
                 connection_type: str       = "Screwed",
                 nominal_diameter_in: float = 1.0,
                 K: Optional[float]         = None,
                 diameter: Optional[float]  = None,
                 name: str                  = ""):
        super().__init__(component_id, name)
        self.fitting_subtype     = fitting_subtype
        self.connection_type     = connection_type
        self.nominal_diameter_in = nominal_diameter_in

        self.diameter = (diameter if diameter is not None
                         else NOMINAL_TO_METRES.get(nominal_diameter_in, 0.02664))
        self.K = (K if K is not None
                  else lookup_fitting_k(connection_type, fitting_subtype, nominal_diameter_in))

    @property
    def area(self) -> float:
        return math.pi * self.diameter**2 / 4.0

    def compute_reynolds(self, Q: float) -> float:
        if abs(Q) < 1e-14:
            return 0.0
        return reynolds_number(abs(Q) / self.area, self.diameter)

    def compute_friction_factor(self, Q: float) -> float:
        return 0.0

    def compute_head_loss(self, Q: float) -> float:
        if abs(Q) < 1e-14:
            return 0.0
        V_abs = abs(Q) / self.area
        return math.copysign(1.0, Q) * self.K * V_abs**2 / (2.0 * GRAVITY)

    def dhead_loss_dQ(self, Q: float, eps: float = 1e-8) -> float:
        if abs(Q) < 1e-10:
            eps_fd = 1e-7
            return (self.compute_head_loss(eps_fd) - self.compute_head_loss(-eps_fd)) / (2*eps_fd)
        return self.K * abs(Q) / (GRAVITY * self.area**2)

    def validate(self) -> List[str]:
        errs = []
        if self.diameter <= 0:
            errs.append(f"[{self.id}] diameter must be > 0 m")
        if self.K < 0:
            errs.append(f"[{self.id}] K must be ≥ 0")
        return errs

    def to_dict(self) -> dict:
        return {
            "type":                "Fitting",
            "id":                  self.id,
            "name":                self.name,
            "fitting_subtype":     self.fitting_subtype,
            "connection_type":     self.connection_type,
            "nominal_diameter_in": self.nominal_diameter_in,
            "K":                   self.K,
            "diameter":            self.diameter,
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

    def to_fitting_attachment(self, pipe_position_t: float = 0.5) -> FittingAttachment:
        """Convert this legacy fitting to a FittingAttachment for migration."""
        return FittingAttachment(
            fitting_id          = self.id,
            fitting_subtype     = self.fitting_subtype,
            connection_type     = self.connection_type,
            nominal_diameter_in = self.nominal_diameter_in,
            K_default           = self.K,
            K_override          = None,
            position_t          = pipe_position_t,
            name                = self.name,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Junction  (network node)
# ═══════════════════════════════════════════════════════════════════════════════

class Junction(FluidComponent):
    """Internal network node — head is an unknown solved by the Newton-Raphson solver."""

    def __init__(self,
                 component_id: str,
                 elevation: float = 0.0,
                 demand: float    = 0.0,
                 name: str        = ""):
        super().__init__(component_id, name)
        self.elevation = elevation
        self.demand    = demand
        self.head: float = 0.0

    @property
    def pressure_head(self) -> float:
        return self.head - self.elevation

    @property
    def pressure_Pa(self) -> float:
        return self.pressure_head * fp.DENSITY * GRAVITY

    def compute_head_loss(self, Q: float) -> float:   return 0.0
    def compute_reynolds(self, Q: float) -> float:    return 0.0
    def compute_friction_factor(self, Q: float) -> float: return 0.0
    def dhead_loss_dQ(self, Q: float, eps: float = 1e-8) -> float: return 0.0

    def validate(self) -> List[str]:
        errs = []
        if self.demand < 0:
            errs.append(f"[{self.id}] demand < 0 (injection); ensure this is intentional")
        return errs

    def to_dict(self) -> dict:
        return {
            "type":      "Junction",
            "id":        self.id,
            "name":      self.name,
            "elevation": self.elevation,
            "demand":    self.demand,
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
    """Fixed-head boundary condition node.  H = z + P/(ρg)"""

    def __init__(self,
                 component_id: str,
                 total_head: Optional[float] = None,
                 name: str                   = "",
                 elevation: Optional[float]  = None,
                 surface_pressure_Pa: float  = 0.0):
        super().__init__(component_id, name)
        if elevation is not None:
            self._elevation           = float(elevation)
            self._surface_pressure_Pa = float(surface_pressure_Pa)
        elif total_head is not None:
            self._elevation           = float(total_head)
            self._surface_pressure_Pa = 0.0
        else:
            self._elevation           = 10.0
            self._surface_pressure_Pa = 0.0
        self.head = self.total_head

    @property
    def elevation(self) -> float:
        return self._elevation

    @elevation.setter
    def elevation(self, value: float):
        self._elevation = float(value)
        self.head = self.total_head

    @property
    def surface_pressure_Pa(self) -> float:
        return self._surface_pressure_Pa

    @surface_pressure_Pa.setter
    def surface_pressure_Pa(self, value: float):
        self._surface_pressure_Pa = float(value)
        self.head = self.total_head

    @property
    def total_head(self) -> float:
        return self._elevation + self._surface_pressure_Pa / (fp.DENSITY * GRAVITY)

    @total_head.setter
    def total_head(self, value: float):
        self._elevation           = float(value)
        self._surface_pressure_Pa = 0.0
        self.head                 = float(value)

    def compute_head_loss(self, Q: float) -> float:   return 0.0
    def compute_reynolds(self, Q: float) -> float:    return 0.0
    def compute_friction_factor(self, Q: float) -> float: return 0.0
    def dhead_loss_dQ(self, Q: float, eps: float = 1e-8) -> float: return 0.0
    def validate(self) -> List[str]:                  return []

    def to_dict(self) -> dict:
        return {
            "type":                "Reservoir",
            "id":                  self.id,
            "name":                self.name,
            "elevation":           self._elevation,
            "surface_pressure_Pa": self._surface_pressure_Pa,
            "total_head":          self.total_head,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Reservoir":
        if "elevation" in d:
            return cls(
                component_id        = d["id"],
                elevation           = d["elevation"],
                surface_pressure_Pa = d.get("surface_pressure_Pa", 0.0),
                name                = d.get("name", d["id"]),
            )
        return cls(
            component_id = d["id"],
            total_head   = d.get("total_head", 10.0),
            name         = d.get("name", d["id"]),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PressurizedSource  (fixed-head boundary with explicit pressure)
# ═══════════════════════════════════════════════════════════════════════════════

class PressurizedSource(Reservoir):
    """
    Pressurized source boundary node.

    Behaves identically to a Reservoir (fixed-head boundary condition) but is
    labelled distinctly in the UI to indicate it represents a pressurized
    supply line or closed tank rather than an open-surface reservoir.

    Total head:  H = elevation + surface_pressure_Pa / (ρ·g)

    Optional known_flow_rate
    ────────────────────────
    If known_flow_rate > 0, the node switches from a fixed-head boundary to a
    fixed-flow injection boundary (the solver treats it as a Junction with
    demand = −known_flow_rate).  This is useful when the supply line has a
    metered flow rate that is known independently of system pressure.
    """

    def __init__(self, *args, known_flow_rate: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.known_flow_rate: float = known_flow_rate   # [m³/s]; 0 = pressure BC

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["type"]             = "PressurizedSource"
        d["known_flow_rate"]  = self.known_flow_rate
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PressurizedSource":
        kwargs = dict(known_flow_rate=d.get("known_flow_rate", 0.0))
        if "elevation" in d:
            return cls(
                component_id        = d["id"],
                elevation           = d["elevation"],
                surface_pressure_Pa = d.get("surface_pressure_Pa", 0.0),
                name                = d.get("name", d["id"]),
                **kwargs,
            )
        return cls(
            component_id = d["id"],
            total_head   = d.get("total_head", 10.0),
            name         = d.get("name", d["id"]),
            **kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PRV  (Pressure-Reducing Valve)
# ═══════════════════════════════════════════════════════════════════════════════

class PRV(FluidComponent):
    """
    Pressure-Reducing Valve edge component.

    Maintains downstream pressure at ``setpoint_Pa``.  The head-loss model
    used here is a Cv-based hydraulic resistance:

        h_loss = sign(Q) · Q² / (Cv² · g)      [m]

    where Cv is in SI units (m³/s per Pa^0.5).  When the upstream pressure
    exceeds the setpoint the PRV throttles; this is captured in the solver by
    a higher effective head loss.  Full active-setpoint enforcement requires a
    dedicated solver constraint (future work — currently the PRV acts as a
    variable-resistance valve whose Cv can be adjusted interactively).

    Properties
    ──────────
    setpoint_Pa : downstream pressure setpoint [Pa]
    Cv          : flow coefficient [m³/s / Pa^0.5]  (default 1e-4 ≈ typical ½″ PRV)
    max_flow    : maximum rated flow [m³/s]
    diameter    : nominal bore [m]
    """

    def __init__(self,
                 component_id: str,
                 diameter:    float = 0.05,
                 setpoint_Pa: float = 200_000.0,
                 Cv:          float = 1e-4,
                 max_flow:    float = 0.005,
                 name:        str   = ""):
        super().__init__(component_id, name)
        self.diameter    = diameter
        self.setpoint_Pa = setpoint_Pa
        self.Cv          = Cv
        self.max_flow    = max_flow

    @property
    def area(self) -> float:
        return math.pi * self.diameter ** 2 / 4.0

    @property
    def setpoint_head(self) -> float:
        return self.setpoint_Pa / (fp.DENSITY * GRAVITY)

    def compute_head_loss(self, Q: float) -> float:
        if abs(Q) < 1e-14:
            return 0.0
        # h = Q² / (Cv² · g)
        h = Q ** 2 / (self.Cv ** 2 * GRAVITY)
        return math.copysign(1.0, Q) * h

    def dhead_loss_dQ(self, Q: float, eps: float = 1e-8) -> float:
        if abs(Q) < 1e-10:
            return 2.0 * eps / (self.Cv ** 2 * GRAVITY)
        return 2.0 * abs(Q) / (self.Cv ** 2 * GRAVITY)

    def compute_reynolds(self, Q: float) -> float:
        if abs(Q) < 1e-14:
            return 0.0
        return reynolds_number(abs(Q) / self.area, self.diameter)

    def compute_friction_factor(self, Q: float) -> float:
        return 0.0

    def validate(self) -> List[str]:
        errs = []
        if self.setpoint_Pa < 0:
            errs.append(f"[{self.id}] setpoint_Pa must be ≥ 0 (got {self.setpoint_Pa})")
        if self.diameter <= 0:
            errs.append(f"[{self.id}] diameter must be > 0 m (got {self.diameter})")
        if self.Cv <= 0:
            errs.append(f"[{self.id}] Cv must be > 0 (got {self.Cv})")
        return errs

    def to_dict(self) -> dict:
        return {
            "type":        "PRV",
            "id":          self.id,
            "name":        self.name,
            "diameter":    self.diameter,
            "setpoint_Pa": self.setpoint_Pa,
            "Cv":          self.Cv,
            "max_flow":    self.max_flow,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PRV":
        return cls(
            component_id = d["id"],
            diameter     = d.get("diameter", 0.05),
            setpoint_Pa  = d.get("setpoint_Pa", 200_000.0),
            Cv           = d.get("Cv", 1e-4),
            max_flow     = d.get("max_flow", 0.005),
            name         = d.get("name", d["id"]),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Factory helper
# ═══════════════════════════════════════════════════════════════════════════════

_COMPONENT_REGISTRY = {
    "Pipe":              Pipe,
    "Pump":              Pump,
    "Valve":             Valve,
    "PRV":               PRV,
    "Fitting":           Fitting,
    "Junction":          Junction,
    "Reservoir":         Reservoir,
    "PressurizedSource": PressurizedSource,
}

def component_from_dict(d: dict) -> FluidComponent:
    kind = d.get("type")
    cls  = _COMPONENT_REGISTRY.get(kind)
    if cls is None:
        raise ValueError(f"Unknown component type: {kind!r}")
    return cls.from_dict(d)

EDGE_COMPONENT_TYPES = {"Pipe", "Pump", "Valve", "Fitting", "PRV"}
NODE_COMPONENT_TYPES = {"Junction", "Reservoir", "PressurizedSource"}
