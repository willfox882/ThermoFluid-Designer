"""
solver.py
---------
Newton-Raphson pipe network solver.

Why Newton-Raphson over Hardy-Cross / Global Gradient Method?
─────────────────────────────────────────────────────────────
NR with an explicit analytical Jacobian converges quadratically near the
solution, handles multiple loops and pumps naturally, and maps cleanly onto
scipy.optimize.fsolve — which internally uses MINPACK's hybrd routine
(a trust-region NR variant) and accepts an analytical Jacobian (fprime).
This gives us:
  • Quadratic convergence
  • Robust handling of ill-conditioned networks
  • No loop identification step required (unlike Hardy-Cross)

Unknown vector  x  (length N + P)
──────────────────────────────────
  x[0 : N]   — piezometric head H_i [m] at each of N free nodes (Junctions)
  x[N : N+P] — volumetric flow rate Q_j [m³/s] in each of P edges

Residual equations  F(x) = 0
─────────────────────────────
For i = 0 … N-1  (continuity at free node i):
    F_i = Σ_j A[i,j]·Q_j  −  D_i  = 0
    where A[i,j] = ±1 is the incidence matrix and D_i is the nodal demand.

For j = 0 … P-1  (energy equation on edge j):
    F_{N+j} = H_from(j) − H_to(j) − h_L(Q_j)  = 0
    h_L > 0 for losses (pipes, valves, closed pumps)
    h_L < 0 for energy addition (running pump: h_L = −hp)

Jacobian  J = ∂F/∂x  (shape (N+P) × (N+P))
────────────────────────────────────────────
Partition x into [H_free | Q]:

    J = ┌  0_{N×N}        │  A_{N×P}              ┐
        │─────────────────┼───────────────────────│
        │  B_{P×N}        │  −diag(dh_L/dQ)_{P×P} ┘

where B[j, node_idx(from)] = +1,  B[j, node_idx(to)] = −1
(only for free nodes; reservoir rows are absent from B because their
 heads are known constants, not unknowns).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import fsolve

from network import PipeNetwork, NetworkEdge
from components import Junction, Reservoir, Pump


# ═══════════════════════════════════════════════════════════════════════════════

class SolverResult:
    """Container for a completed solve."""

    def __init__(self):
        self.converged:       bool  = False
        self.residual_norm:   float = float("inf")
        self.iterations:      int   = 0
        self.message:         str   = ""
        self.errors:          List[str] = []

        # Results keyed by ID
        self.heads:            Dict[str, float] = {}   # all nodes   [m]
        self.flows:            Dict[str, float] = {}   # all edges   [m³/s]
        self.velocities:       Dict[str, float] = {}   # all edges   [m/s]
        self.head_losses:      Dict[str, float] = {}   # all edges   [m]
        self.reynolds:         Dict[str, float] = {}   # all edges   [-]
        self.friction_factors: Dict[str, float] = {}   # all edges   [-]
        self.pressures:        Dict[str, float] = {}   # all nodes   [Pa]

    def __bool__(self):
        return self.converged

    def summary_lines(self) -> List[str]:
        lines = [
            f"Converged : {self.converged}",
            f"Residual  : {self.residual_norm:.3e}",
            f"Message   : {self.message}",
            "",
            "── Node heads ──────────────────────────────",
        ]
        for nid, H in self.heads.items():
            lines.append(f"  {nid:<20s}  H = {H:8.3f} m")
        lines += ["", "── Edge flows ──────────────────────────────"]
        for eid, Q in self.flows.items():
            V  = self.velocities.get(eid, 0.0)
            Re = self.reynolds.get(eid, 0.0)
            hL = self.head_losses.get(eid, 0.0)
            lines.append(
                f"  {eid:<20s}  Q = {Q*1000:7.3f} L/s  "
                f"V = {V:6.3f} m/s  Re = {Re:8.0f}  ΔH = {hL:7.3f} m"
            )
        return lines


# ═══════════════════════════════════════════════════════════════════════════════

class NetworkSolver:
    """
    Solves a PipeNetwork using Newton-Raphson via scipy.optimize.fsolve.

    Usage
    -----
        solver = NetworkSolver(network)
        result = solver.solve()
        if result.converged:
            print(result.heads)
    """

    def __init__(self, network: PipeNetwork) -> None:
        self.network = network

        # Set during _build_state_maps()
        self._free_node_ids: List[str] = []
        self._edge_ids:       List[str] = []
        self._N:  int = 0   # number of free nodes
        self._P:  int = 0   # number of edges
        self._A:  np.ndarray = np.zeros((0, 0))
        self._node_idx: Dict[str, int] = {}
        self._edge_idx: Dict[str, int] = {}

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _build_state_maps(self) -> None:
        """Cache index maps and incidence matrix."""
        A, free_ids, edge_ids = self.network.build_incidence_matrix()
        self._A            = A
        self._free_node_ids = free_ids
        self._edge_ids      = edge_ids
        self._N  = len(free_ids)
        self._P  = len(edge_ids)
        self._node_idx = {nid: i for i, nid in enumerate(free_ids)}
        self._edge_idx = {eid: j for j, eid in enumerate(edge_ids)}

    def _node_head(self, node_id: str, H_free: np.ndarray) -> float:
        """Head at any node — free (unknown) or reservoir (fixed)."""
        if node_id in self._node_idx:
            return H_free[self._node_idx[node_id]]
        return self.network.nodes[node_id].component.head  # Reservoir

    def _demand(self, node_id: str) -> float:
        comp = self.network.nodes[node_id].component
        return comp.demand if isinstance(comp, Junction) else 0.0

    # ── Residuals ─────────────────────────────────────────────────────────────

    def residuals(self, x: np.ndarray) -> np.ndarray:
        """
        Compute F(x) — the residual vector of length N + P.

        x[:N]  = piezometric heads at free nodes [m]
        x[N:]  = volumetric flow rates in edges  [m³/s]
        """
        H_free = x[:self._N]
        Q      = x[self._N:]
        F      = np.zeros(self._N + self._P)

        # ── Continuity (rows 0 … N-1) ────────────────────────────────
        for i, node_id in enumerate(self._free_node_ids):
            F[i] = np.dot(self._A[i, :], Q) - self._demand(node_id)

        # ── Energy (rows N … N+P-1) ──────────────────────────────────
        for j, edge_id in enumerate(self._edge_ids):
            edge  = self.network.edges[edge_id]
            H_from = self._node_head(edge.from_node_id, H_free)
            H_to   = self._node_head(edge.to_node_id,   H_free)
            h_L    = edge.component.compute_head_loss(Q[j])
            F[self._N + j] = H_from - H_to - h_L

        return F

    # ── Jacobian ──────────────────────────────────────────────────────────────

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Analytical Jacobian  J = ∂F/∂x,  shape (N+P, N+P).

        Block structure:
            J[:N,  :N]   = 0              (continuity doesn't depend on H)
            J[:N,  N:]   = A              (incidence matrix)
            J[N:,  :N]   = B  (±1 where from/to nodes are free)
            J[N:,  N:]   = −diag(dh_L/dQ)
        """
        Q = x[self._N:]
        J = np.zeros((self._N + self._P, self._N + self._P))

        # Upper-right block: ∂F_cont / ∂Q  = A
        J[:self._N, self._N:] = self._A

        # Lower blocks: energy equations
        for j, edge_id in enumerate(self._edge_ids):
            edge    = self.network.edges[edge_id]
            row     = self._N + j

            # ∂F_energy / ∂H_from = +1  (if from-node is free)
            if edge.from_node_id in self._node_idx:
                J[row, self._node_idx[edge.from_node_id]] = +1.0

            # ∂F_energy / ∂H_to   = −1  (if to-node is free)
            if edge.to_node_id in self._node_idx:
                J[row, self._node_idx[edge.to_node_id]] = -1.0

            # ∂F_energy / ∂Q_j   = −dh_L/dQ_j
            dh = edge.component.dhead_loss_dQ(Q[j])
            J[row, self._N + j] = -dh

        return J

    # ── Initial guess ─────────────────────────────────────────────────────────

    def _initial_guess(self) -> np.ndarray:
        """
        Heuristic starting point designed to land near the physical root.

        Node heads
        ----------
        • Reservoir nodes: their fixed heads (known BCs).
        • Free nodes (Junctions):
            - Base estimate = average of ALL reservoir heads.  This puts every
              junction in the interior of the feasible head range and avoids the
              pathological case where BFS copies the upstream reservoir head
              directly (which makes the downstream energy residual ≈ ΔH_total
              and fsolve struggles to escape).
            - If the junction is reachable *only* through a pump (no passive
              path from a reservoir), its head is raised by the pump shut-off
              head C so the initial point sits above the delivery reservoir,
              preventing a spurious backward-flow root.

        Edge flows
        ----------
        • Pumps:   1×10⁻³ m³/s  positive (in declared flow direction)
        • Others:  1×10⁻⁴ m³/s  positive
        """
        x0 = np.zeros(self._N + self._P)

        reservoirs = self.network.get_reservoir_nodes()
        avg_head   = (float(np.mean([r.component.head for r in reservoirs]))
                      if reservoirs else 10.0)

        # ── Identify nodes reachable only through pumps ───────────────────────
        # BFS: track which nodes have a passive (non-pump) path from any reservoir.
        passively_reachable: set[str] = set()
        for rn in reservoirs:
            passively_reachable.add(rn.node_id)

        changed = True
        while changed:
            changed = False
            for eid, edge in self.network.edges.items():
                comp = edge.component
                if isinstance(comp, Pump):
                    continue   # skip pump edges for passive reachability
                fn, tn = edge.from_node_id, edge.to_node_id
                if fn in passively_reachable and tn not in passively_reachable:
                    passively_reachable.add(tn); changed = True
                if tn in passively_reachable and fn not in passively_reachable:
                    passively_reachable.add(fn); changed = True

        # ── Pump boost: find max pump shut-off head feeding each free node ────
        pump_boost: dict[str, float] = {}
        for eid, edge in self.network.edges.items():
            comp = edge.component
            if isinstance(comp, Pump) and comp.is_on:
                # The to_node gets boosted
                tn = edge.to_node_id
                if tn not in passively_reachable:
                    pump_boost[tn] = max(pump_boost.get(tn, 0.0), abs(comp.C))

        # ── Assign free-node heads ────────────────────────────────────────────
        for i, nid in enumerate(self._free_node_ids):
            h = avg_head
            if nid in pump_boost:
                # Boost above avg_head so we're clearly above delivery reservoirs
                h = avg_head + pump_boost[nid]
            x0[i] = h

        # ── Edge flow estimates ───────────────────────────────────────────────
        for j, eid in enumerate(self._edge_ids):
            comp = self.network.edges[eid].component
            x0[self._N + j] = 1e-3 if isinstance(comp, Pump) else 1e-4

        return x0

    # ── Main solve ────────────────────────────────────────────────────────────

    def solve(self,
              x0:      Optional[np.ndarray] = None,
              tol:     float = 1e-9,
              max_iter: int  = 200) -> SolverResult:
        """
        Solve the pipe network.

        Parameters
        ----------
        x0       : Initial guess vector (auto-generated if None)
        tol      : Convergence tolerance for residuals
        max_iter : Maximum Newton iterations

        Returns
        -------
        SolverResult — always returned, check .converged
        """
        result = SolverResult()

        # 1. Validate
        errors = self.network.validate()
        if errors:
            result.errors  = errors
            result.message = "Validation failed — cannot solve."
            return result

        # 2. Build index maps
        self._build_state_maps()

        if self._N == 0:
            # All nodes are reservoirs — trivially solved
            result.converged = True
            result.message   = "No free nodes; network is trivially defined by reservoirs."
            result.heads     = {nid: n.component.head
                                for nid, n in self.network.nodes.items()}
            return result

        # 3. Solve
        if x0 is None:
            x0 = self._initial_guess()

        # ── Newton-Raphson with backtracking line search ───────────────────────
        # scipy.fsolve internally calls the Jacobian with a finite-difference
        # perturbation to verify correctness, which fails when Q values are very
        # small (1e-4) and the system is stiff. A custom NR loop converges
        # reliably across all network topologies.
        x   = x0.copy()
        converged  = False
        last_norm  = float("inf")
        nfev       = 0

        for iteration in range(max_iter):
            F    = self.residuals(x)
            norm = float(np.linalg.norm(F))
            nfev += 1

            if norm < tol:
                converged = True
                break

            if norm > last_norm * 1e6 and iteration > 2:
                # Diverging badly — stop
                break
            last_norm = norm

            try:
                J  = self.jacobian(x)
                dx = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                break

            # Backtracking Armijo line search (max 8 halvings)
            alpha = 1.0
            for _ in range(8):
                x_new  = x + alpha * dx
                F_new  = self.residuals(x_new)
                nfev  += 1
                if np.linalg.norm(F_new) < norm:
                    break
                alpha *= 0.5

            x = x + alpha * dx

        sol = x
        result.converged     = converged
        result.message       = ("The solution converged."
                                if converged else
                                "Newton-Raphson did not converge within the iteration limit.")
        result.residual_norm = float(np.linalg.norm(self.residuals(sol)))
        result.iterations    = nfev

        # 4. Unpack and store results
        H_sol = sol[:self._N]
        Q_sol = sol[self._N:]

        # Update free-node heads in the model
        for i, nid in enumerate(self._free_node_ids):
            self.network.nodes[nid].component.head = H_sol[i]

        # Update edge flow rates in the model
        for j, eid in enumerate(self._edge_ids):
            self.network.edges[eid].flow_rate = Q_sol[j]

        # Build result dicts
        from fluid_props import DENSITY, GRAVITY
        for nid, node in self.network.nodes.items():
            H = node.component.head
            result.heads[nid] = H
            z = getattr(node.component, "elevation", 0.0)
            result.pressures[nid] = (H - z) * DENSITY * GRAVITY

        for j, eid in enumerate(self._edge_ids):
            edge = self.network.edges[eid]
            comp = edge.component
            Q    = Q_sol[j]

            result.flows[eid]      = Q
            result.head_losses[eid] = comp.compute_head_loss(Q)
            result.reynolds[eid]   = comp.compute_reynolds(Q)
            result.friction_factors[eid] = comp.compute_friction_factor(Q)

            A = getattr(comp, "area", None)
            result.velocities[eid] = (Q / A) if (A and A > 0) else 0.0

        return result

    # ── System curve computation ───────────────────────────────────────────────

    def compute_system_curve(self,
                             pump_edge_id:  str,
                             last_result:   SolverResult,
                             n_points:      int = 150
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate system curve:  h_system(Q) = h_static + R_eff · Q²

        Method
        ──────
        At the operating point (Q*, H*) we know:
            H* = h_pump(Q*)  →  h_pump(Q*) = h_system(Q*)

        The static head h_static is the elevation/head difference the system
        must overcome at zero flow.  For a system with source reservoir H_src
        and sink reservoir H_snk:
            h_static = H_snk − H_src   (head to be overcome)

        Effective resistance:
            R_eff = (H* − h_static) / Q*²

        System curve (parabola through origin of the friction losses):
            h_sys(Q) = h_static + R_eff · Q²

        This approximation is valid near the operating point.  It breaks down
        for highly non-linear networks but is accurate enough for design.

        Returns (Q_array, h_system_array).
        """
        if pump_edge_id not in last_result.flows:
            return np.array([]), np.array([])

        Q_star = abs(last_result.flows[pump_edge_id])
        if Q_star < 1e-10:
            return np.array([]), np.array([])

        # Pump's head at operating point (from energy equation on pump edge)
        pump_edge  = self.network.edges[pump_edge_id]
        pump_comp  = pump_edge.component
        H_pump_star = abs(pump_comp.compute_pump_head(Q_star))

        # Static head: difference between sink and source reservoirs
        reservoirs = self.network.get_reservoir_nodes()
        if len(reservoirs) >= 2:
            heads = sorted([r.component.total_head for r in reservoirs])
            h_static = max(0.0, heads[-1] - heads[0])
        else:
            h_static = 0.0

        # Friction component at operating point
        h_fric_star = H_pump_star - h_static
        if h_fric_star < 0:
            h_fric_star = 0.0

        R_eff = h_fric_star / Q_star**2

        # Build curve
        Q_max = Q_star * 1.8
        Q_arr = np.linspace(0.0, Q_max, n_points)
        h_sys = h_static + R_eff * Q_arr**2

        return Q_arr, h_sys
