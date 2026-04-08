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

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import fsolve

from network import PipeNetwork, NetworkEdge
from components import Junction, Reservoir, Pump, PressurizedSource


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
        if isinstance(comp, Junction):
            return comp.demand
        # PressurizedSource with a known flow rate acts as a fixed-injection node
        if isinstance(comp, PressurizedSource):
            kfr = getattr(comp, 'known_flow_rate', 0.0)
            if kfr > 0:
                return -kfr   # negative = injection into the network
        return 0.0

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
                # Robustness: clamp Newton step to prevent wild divergence
                dx = np.clip(dx, -1e3, 1e3)
            except np.linalg.LinAlgError:
                break

            # Backtracking Armijo line search (max 8 halvings)
            alpha = 1.0
            for _ in range(8):
                x_new  = x + alpha * dx
                # Prevent negative flows/heads from causing math errors in residuals
                # (especially for friction factor log calculations)
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
            V = (Q / A) if (A and A > 0) else 0.0
            result.velocities[eid] = V

        # 5. POST-SOLVE: NPSH and Cavitation checks
        for eid, edge in self.network.edges.items():
            comp = edge.component
            if isinstance(comp, Pump) and comp.is_on:
                P_suc = result.pressures.get(edge.from_node_id, 0.0)
                V_suc = result.velocities.get(eid, 0.0)
                comp.compute_npsha(P_suc, V_suc)
                if comp.is_cavitating:
                    result.errors.append(f"CAVITATION WARNING: Pump {comp.id} "
                                         f"NPSHa ({comp.npsh_available:.2f} m) < "
                                         f"NPSHr ({comp.npsh_required:.2f} m)")

        return result

    # ── System curve computation ───────────────────────────────────────────────

    def _find_reservoir_pair(self, pump_edge_id: str) -> Tuple[float, float]:
        """
        BFS from each side of a pump edge to locate the supply (suction-side)
        and delivery (discharge-side) reservoir total heads.

        Returns (H_source, H_delivery).
        Fallback: if one side finds no reservoir, uses global min/max.
        """
        pump_edge = self.network.edges.get(pump_edge_id)
        if pump_edge is None:
            return 0.0, 0.0

        def bfs_head(start_id: str) -> Optional[float]:
            visited = {start_id}
            queue   = [start_id]
            while queue:
                nid  = queue.pop(0)
                node = self.network.nodes.get(nid)
                if node is None:
                    continue
                if isinstance(node.component, Reservoir):
                    return node.component.total_head
                for eid in node.connected_edge_ids:
                    if eid == pump_edge_id:
                        continue          # do not cross the pump itself
                    edge = self.network.edges.get(eid)
                    if edge is None:
                        continue
                    fn, tn = edge.from_node_id, edge.to_node_id
                    nbr = tn if fn == nid else fn
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            return None

        H_src = bfs_head(pump_edge.from_node_id)
        H_del = bfs_head(pump_edge.to_node_id)

        # Fallback: use global min / max reservoir heads
        reservoirs = self.network.get_reservoir_nodes()
        if reservoirs:
            hs = sorted(r.component.total_head for r in reservoirs)
            if H_src is None:
                H_src = hs[0]
            if H_del is None:
                H_del = hs[-1]

        return (H_src or 0.0), (H_del or 0.0)

    def compute_system_curve(self,
                             pump_edge_id:  str,
                             last_result:   Optional[SolverResult] = None,
                             n_points:      int = 150
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute system curve  h_system(Q) = h_static + Σ h_loss(Q)
        by sweeping Q values.

        This method is INDEPENDENT of pump state — identical curve whether
        the pump is ON or OFF.

        h_static  = H_delivery − H_source
                    (BFS from pump ports to nearest reservoir on each side;
                     > 0 for uphill pumping, < 0 when gravity helps)

        h_loss(Q) = Σ compute_head_loss(Q) over all non-pump edges
                    (series network assumption; reasonable approx for parallel)

        Returns (Q_arr [m³/s], h_sys_arr [m]).
        """
        if pump_edge_id not in self.network.edges:
            return np.array([]), np.array([])

        pump_comp = self.network.edges[pump_edge_id].component
        if not isinstance(pump_comp, Pump):
            return np.array([]), np.array([])

        H_src, H_del = self._find_reservoir_pair(pump_edge_id)
        h_static = H_del - H_src   # > 0 for typical uphill pump system

        non_pump = [e for e in self.network.edges.values()
                    if not isinstance(e.component, Pump)]

        # Q range based on pump operating range + desired flow
        Q_des = max(getattr(pump_comp, 'desired_flow_rate', 0.0), 0.0)
        _, Q_pump_max = pump_comp.get_operating_range()
        Q_max = max(Q_pump_max * 1.5, Q_des * 2.0, 0.05)

        Q_arr = np.linspace(0.0, Q_max, n_points)
        h_sys = np.empty(n_points)

        for i, Q in enumerate(Q_arr):
            if Q < 1e-12:
                h_sys[i] = h_static
            else:
                h_loss = sum(e.component.compute_head_loss(Q) for e in non_pump)
                h_sys[i] = h_static + h_loss

        return Q_arr, h_sys

    def compute_system_curve_standalone(self,
                                        last_result: Optional[SolverResult] = None,
                                        n_points:    int = 150
                                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        System curve for networks that contain no pump.

        h_static  = H_outlet − H_inlet
                    (negative for gravity-driven flow where inlet head > outlet)

        h_loss(Q) = Σ compute_head_loss(Q) over all edges

        Returns (Q_arr, h_sys), or empty arrays if insufficient data.
        """
        reservoirs = self.network.get_reservoir_nodes()
        if len(reservoirs) < 2:
            return np.array([]), np.array([])

        hs = sorted(r.component.total_head for r in reservoirs)
        H_src  = hs[-1]     # highest-head reservoir = gravity source
        H_del  = hs[0]      # lowest-head reservoir = destination
        h_static = H_del - H_src   # negative for gravity-driven flow

        all_edges = [e for e in self.network.edges.values()
                     if not isinstance(e.component, Pump)]
        if not all_edges:
            return np.array([]), np.array([])

        if last_result and last_result.flows:
            Q_ref = max(abs(q) for q in last_result.flows.values())
            Q_max = max(Q_ref * 2.0, 0.01)
        else:
            Q_max = 0.05

        Q_arr = np.linspace(0.0, Q_max, n_points)
        h_sys = np.empty(n_points)

        for i, Q in enumerate(Q_arr):
            if Q < 1e-12:
                h_sys[i] = h_static
            else:
                h_loss = sum(e.component.compute_head_loss(Q) for e in all_edges)
                h_sys[i] = h_static + h_loss

        return Q_arr, h_sys

    @staticmethod
    def generate_pump_curve(Q_des: float, h_req: float,
                            pump_type: str = "centrifugal"
                            ) -> Tuple[float, float, float]:
        """
        Generate a synthetic quadratic pump curve  h(Q) = H_shutoff − a·Q²
        whose BEP coincides with (Q_des, h_req).

        Shutoff-head multiplier per pump type:
          centrifugal  → 1.25 × h_req
          mixed-flow   → 1.15 × h_req
          axial        → 1.10 × h_req

        Returns (A, B, C) such that  h_pump = A·Q² + B·Q + C  (A ≤ 0).
        """
        if Q_des <= 0 or h_req <= 0:
            return -8000.0, 0.0, 25.0

        factors = {"centrifugal": 1.25, "mixed-flow": 1.15, "axial": 1.10}
        shutoff_mult = factors.get(pump_type, 1.25)
        H_shutoff    = shutoff_mult * h_req
        a            = (H_shutoff - h_req) / Q_des ** 2   # > 0
        return -a, 0.0, H_shutoff

    # ── Multi-pump topology + combined curves ─────────────────────────────────

    def _pumps_are_series(self, pid1: str, pid2: str) -> bool:
        """
        Return True if pump1_out can reach pump2_in (or vice versa)
        via BFS across junctions/pipes without crossing a Reservoir or
        either pump edge itself.
        """
        e1 = self.network.edges.get(pid1)
        e2 = self.network.edges.get(pid2)
        if e1 is None or e2 is None:
            return False

        exclude = {pid1, pid2}

        def reachable(start: str, target: str) -> bool:
            visited = {start}
            queue   = [start]
            while queue:
                nid  = queue.pop(0)
                if nid == target:
                    return True
                node = self.network.nodes.get(nid)
                if node is None:
                    continue
                for eid in node.connected_edge_ids:
                    if eid in exclude:
                        continue
                    edge = self.network.edges.get(eid)
                    if edge is None:
                        continue
                    fn, tn = edge.from_node_id, edge.to_node_id
                    nbr = tn if fn == nid else fn
                    if nbr in visited:
                        continue
                    nbr_node = self.network.nodes.get(nbr)
                    if nbr_node and isinstance(nbr_node.component, Reservoir):
                        continue   # do not traverse through fixed-head boundaries
                    visited.add(nbr)
                    queue.append(nbr)
            return False

        return (reachable(e1.to_node_id,   e2.from_node_id) or
                reachable(e2.to_node_id,   e1.from_node_id))

    def _pumps_are_parallel(self, pid1: str, pid2: str) -> bool:
        """
        Return True if the two pumps share both a common inlet node
        cluster and a common outlet node cluster (i.e., the flow splits
        at a junction before the pumps and rejoins after them).
        """
        e1 = self.network.edges.get(pid1)
        e2 = self.network.edges.get(pid2)
        if e1 is None or e2 is None:
            return False

        exclude = {pid1, pid2}

        def reachable(start: str, target: str) -> bool:
            visited = {start}
            queue   = [start]
            while queue:
                nid  = queue.pop(0)
                if nid == target:
                    return True
                node = self.network.nodes.get(nid)
                if node is None:
                    continue
                for eid in node.connected_edge_ids:
                    if eid in exclude:
                        continue
                    edge = self.network.edges.get(eid)
                    if edge is None:
                        continue
                    fn, tn = edge.from_node_id, edge.to_node_id
                    nbr = tn if fn == nid else fn
                    if nbr in visited:
                        continue
                    nbr_node = self.network.nodes.get(nbr)
                    if nbr_node and isinstance(nbr_node.component, Reservoir):
                        continue
                    visited.add(nbr)
                    queue.append(nbr)
            return False

        same_inlet  = (e1.from_node_id == e2.from_node_id or
                       reachable(e1.from_node_id, e2.from_node_id))
        same_outlet = (e1.to_node_id == e2.to_node_id or
                       reachable(e1.to_node_id, e2.to_node_id))
        return same_inlet and same_outlet

    def detect_pump_groups(self) -> List[Dict]:
        """
        Inspect network topology (Option A — canvas-topology-driven) and
        cluster the pump edges into series / parallel / independent groups.

        A group entry has the shape:
            {
              "type":     "series" | "parallel" | "single",
              "pump_ids": [<edge_id>, ...],
            }

        Detection rules
        ───────────────
        • Two pumps are **series**   if one's outlet-side can reach the
          other's inlet-side through junctions/pipes without crossing a
          reservoir or either pump.
        • Two pumps are **parallel** if they share both an inlet-side
          junction cluster and an outlet-side junction cluster.
        • Otherwise they are **independent** (each in its own group).
        """
        from typing import Dict as _Dict
        pump_edges = self.network.get_pumps()
        if not pump_edges:
            return []
        if len(pump_edges) == 1:
            return [{"type": "single", "pump_ids": [pump_edges[0].edge_id]}]

        # Union-Find for grouping
        parent: Dict[str, str] = {pe.edge_id: pe.edge_id for pe in pump_edges}
        config: Dict[str, str] = {}   # edge_id → "series" | "parallel"

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str, cfg: str):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx
                config[rx] = cfg

        ids = [pe.edge_id for pe in pump_edges]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if self._pumps_are_series(ids[i], ids[j]):
                    union(ids[i], ids[j], "series")
                elif self._pumps_are_parallel(ids[i], ids[j]):
                    union(ids[i], ids[j], "parallel")

        # Collect groups
        buckets: Dict[str, List[str]] = {}
        for pid in ids:
            root = find(pid)
            buckets.setdefault(root, []).append(pid)

        groups = []
        for root, members in buckets.items():
            if len(members) == 1:
                groups.append({"type": "single", "pump_ids": members})
            else:
                groups.append({"type": config.get(root, "single"),
                                "pump_ids": members})
        return groups

    def compute_combined_pump_curve(self,
                                    pump_ids: List[str],
                                    config:   str,
                                    n_points: int = 300
                                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the combined pump curve for a series or parallel group.

        Series
        ──────
        Same Q flows through every pump; heads add:
            h_total(Q) = Σ h_i(Q)

        Parallel
        ────────
        Same head across every pump; flows add.  For each head value h,
        invert each pump curve to find Q_i(h), then sum:
            Q_total(h) = Σ Q_i(h)
        Returns (Q_arr, h_arr) sorted by Q ascending.

        Returns empty arrays if no running pumps in the group.
        """
        pump_comps: List[Pump] = []
        for pid in pump_ids:
            edge = self.network.edges.get(pid)
            if edge and isinstance(edge.component, Pump) and edge.component.is_on:
                pump_comps.append(edge.component)

        if not pump_comps:
            return np.array([]), np.array([])

        if config == "series":
            # Q_max limited by the pump with the smallest free-delivery point
            Q_maxes = [pc.get_operating_range()[1] for pc in pump_comps]
            Q_max   = min(Q_maxes) if Q_maxes else 0.05
            if Q_max <= 0:
                Q_max = 0.05
            Q_arr = np.linspace(0.0, Q_max * 1.2, n_points)
            h_arr = np.array([sum(pc.compute_pump_head(Q) for pc in pump_comps)
                               for Q in Q_arr])
            mask  = h_arr >= 0
            return Q_arr[mask], h_arr[mask]

        elif config == "parallel":
            # Sweep head from 0 up to the minimum shutoff head
            h_max = min(pc.C for pc in pump_comps)
            if h_max <= 0:
                return np.array([]), np.array([])
            h_sweep = np.linspace(0.0, h_max, n_points)
            Q_total = np.zeros(n_points)

            for i, h in enumerate(h_sweep):
                for pc in pump_comps:
                    # Solve  A·Q² + B·Q + (C − h) = 0  for Q ≥ 0
                    a, b, c = pc.A, pc.B, pc.C - h
                    if abs(a) < 1e-30:
                        Qi = max(0.0, -c / b) if abs(b) > 1e-30 else 0.0
                    else:
                        disc = b * b - 4.0 * a * c
                        if disc < 0:
                            Qi = 0.0
                        else:
                            sq  = math.sqrt(disc)
                            Qi  = max(0.0,
                                      (-b + sq) / (2.0 * a),
                                      (-b - sq) / (2.0 * a))
                    Q_total[i] += max(0.0, Qi)

            # Return sorted by Q ascending (head is monotone decreasing in Q)
            order = np.argsort(Q_total)
            return Q_total[order], h_sweep[order]

        return np.array([]), np.array([])

    @staticmethod
    def find_curve_intersection(Q1: np.ndarray, h1: np.ndarray,
                                Q2: np.ndarray, h2: np.ndarray
                                ) -> Optional[Tuple[float, float]]:
        """
        Find the first intersection of two head-vs-flow curves by
        interpolating both onto a common Q grid and locating the
        sign-change of their difference.

        Returns (Q_op, h_op) or None if no intersection is found.
        """
        if len(Q1) < 2 or len(Q2) < 2:
            return None

        Q_lo = max(Q1[0],  Q2[0])
        Q_hi = min(Q1[-1], Q2[-1])
        if Q_lo >= Q_hi:
            return None

        Q_common = np.linspace(Q_lo, Q_hi, 500)
        h_a = np.interp(Q_common, Q1, h1)
        h_b = np.interp(Q_common, Q2, h2)
        diff = h_a - h_b

        crossings = np.where(np.diff(np.sign(diff)))[0]
        if not len(crossings):
            return None

        idx = crossings[0]
        d0, d1 = diff[idx], diff[idx + 1]
        q0, q1 = Q_common[idx], Q_common[idx + 1]
        Q_op = q0 - d0 * (q1 - q0) / (d1 - d0) if (d1 - d0) != 0 else q0
        h_op = float(np.interp(Q_op, Q_common, h_a))
        return Q_op, h_op
