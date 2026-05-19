"""
test_parallel_physics.py
------------------------
Physics fidelity tests for parallel pipe networks.

Parallel-branch governing equations
───────────────────────────────────
Two pipes between common junctions Ja and Jb both see the SAME piezometric
head drop:

        ΔH = H_Ja − H_Jb  =  h_L,a(Q_a)  =  h_L,b(Q_b)              (1)

Mass balance at Ja:

        Q_total = Q_a + Q_b                                         (2)

These two equations + the upstream/downstream loop close the network.  The
solver must satisfy (1) and (2) simultaneously.

This file checks:
  • forward solve:  Δp_a == Δp_b across the branches
  • mass balance:   Q_total = sum(Q_branches)
  • system curve:   sweeping Q must produce h_static + h_loss(Q) that
                    matches a *hand-resolved* parallel-equivalent loss,
                    NOT the naive sum of branch losses at Q_total
                    (currently a known issue in compute_system_curve).
  • pump-sizing round-trip:   solve → read h_req → generate pump curve →
                              re-solve → Q must equal Q_des.
"""

import math
import numpy as np
import pytest

from fluid_props import friction_factor, DENSITY, VISCOSITY, GRAVITY
from components import Pipe, Pump, Junction, Reservoir
from network   import PipeNetwork
from solver    import NetworkSolver


# ─────────────────────────────────────────────────────────────────────────────
# Reference solver for a single (parallel-branch-equivalent) head loss
# ─────────────────────────────────────────────────────────────────────────────

def _pipe_hL(Q, D, L, eps, K):
    A  = math.pi * D**2 / 4.0
    V  = abs(Q) / A
    Re = DENSITY * V * D / VISCOSITY
    f  = friction_factor(Re, eps / D)
    return math.copysign(1.0, Q) * (f * L / D + K) * V**2 / (2.0 * GRAVITY)


def parallel_equivalent_head_loss(Q_total, branches, tol=1e-12, max_iter=200):
    """
    Solve   h_L,a(Q_a) = h_L,b(Q_b)  with  Σ Q_i = Q_total
    using bisection on the shared head loss h*.

    branches : list of dicts {'D','L','eps','K'}.
    Returns (h_star, [Q_branch_i, ...]).
    """
    if Q_total <= 0:
        return 0.0, [0.0] * len(branches)

    def Q_given_h(h, br):
        """Invert single-pipe h_L(Q) = h for Q > 0 by bisection."""
        if h <= 0:
            return 0.0
        lo, hi = 1e-12, max(Q_total * 10, 1e-3)
        # Expand hi until h_L(hi) > h
        for _ in range(60):
            if _pipe_hL(hi, **br) >= h:
                break
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _pipe_hL(mid, **br) > h:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-15:
                break
        return 0.5 * (lo + hi)

    # Bisection on h*
    h_lo, h_hi = 1e-12, 1.0
    # Expand h_hi until Σ Q_i(h_hi) ≥ Q_total
    for _ in range(80):
        S = sum(Q_given_h(h_hi, br) for br in branches)
        if S >= Q_total:
            break
        h_hi *= 2.0

    for _ in range(max_iter):
        h_mid = 0.5 * (h_lo + h_hi)
        S = sum(Q_given_h(h_mid, br) for br in branches)
        if abs(S - Q_total) < tol * max(Q_total, 1e-9):
            break
        if S > Q_total:
            h_hi = h_mid
        else:
            h_lo = h_mid

    h_star = 0.5 * (h_lo + h_hi)
    return h_star, [Q_given_h(h_star, br) for br in branches]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Forward solve — Δp equality & mass balance
# ─────────────────────────────────────────────────────────────────────────────

class TestParallelForwardSolve:

    def _build(self, H_A=20.0, H_B=0.0):
        net = PipeNetwork()
        net.add_node(Reservoir('RA', total_head=H_A))
        net.add_node(Reservoir('RB', total_head=H_B))
        net.add_node(Junction('Ja',  elevation=0.0))
        net.add_node(Junction('Jb',  elevation=0.0))
        net.add_edge(Pipe('Pin',  diameter=0.10, length=50.0,  roughness=4.6e-5),
                     'RA', 'Ja')
        net.add_edge(Pipe('PA',   diameter=0.06, length=200.0, roughness=4.6e-5,
                          K_minor=1.5), 'Ja', 'Jb')
        net.add_edge(Pipe('PB',   diameter=0.08, length=200.0, roughness=4.6e-5,
                          K_minor=0.5), 'Ja', 'Jb')
        net.add_edge(Pipe('Pout', diameter=0.10, length=50.0,  roughness=4.6e-5),
                     'Jb', 'RB')
        return net

    def test_branches_share_head_loss(self):
        net = self._build()
        r = NetworkSolver(net).solve()
        assert r.converged, r.message

        h_PA = r.head_losses['PA']
        h_PB = r.head_losses['PB']
        # Both branches share Ja → Jb head drop
        assert abs(h_PA - h_PB) < 1e-9, \
            f"Branch head-loss imbalance: hPA={h_PA:.6f}m, hPB={h_PB:.6f}m"

        # Hand-check: each branch h_L matches H_Ja - H_Jb
        dH = r.heads['Ja'] - r.heads['Jb']
        assert abs(h_PA - dH) < 1e-9 and abs(h_PB - dH) < 1e-9

    def test_branches_mass_balance(self):
        net = self._build()
        r = NetworkSolver(net).solve()
        assert r.converged

        Q_in = r.flows['Pin']
        Q_a  = r.flows['PA']
        Q_b  = r.flows['PB']
        Q_out= r.flows['Pout']

        # Junction-level conservation
        assert abs(Q_in - (Q_a + Q_b)) < 1e-10, \
            f"Ja balance: Q_in={Q_in*1e3:.4f}L/s, sum={Q_a*1e3+Q_b*1e3:.4f}L/s"
        assert abs((Q_a + Q_b) - Q_out) < 1e-10, "Jb balance fails"

    def test_total_energy_balance(self):
        H_A, H_B = 20.0, 0.0
        net = self._build(H_A, H_B)
        r = NetworkSolver(net).solve()
        assert r.converged

        # Following ONE branch from RA → Ja → (PA) → Jb → RB:
        path_A = r.head_losses['Pin'] + r.head_losses['PA'] + r.head_losses['Pout']
        assert abs(path_A - (H_A - H_B)) < 1e-6

        # And the other path RA → Ja → (PB) → Jb → RB:
        path_B = r.head_losses['Pin'] + r.head_losses['PB'] + r.head_losses['Pout']
        assert abs(path_B - (H_A - H_B)) < 1e-6

    def test_against_hand_parallel_equivalent(self):
        """
        Forward solve must reproduce the hand-calculated branch split:
            given Q_total, the bisected h_star and per-branch Q_i.
        """
        net = self._build()
        r = NetworkSolver(net).solve()
        assert r.converged

        Q_total = r.flows['Pin']
        branches = [
            {'D':0.06, 'L':200.0, 'eps':4.6e-5, 'K':1.5},
            {'D':0.08, 'L':200.0, 'eps':4.6e-5, 'K':0.5},
        ]
        h_star, Qs = parallel_equivalent_head_loss(Q_total, branches)
        Q_a_hand, Q_b_hand = Qs

        # Solver values should match to within bisection tolerance
        assert abs(r.flows['PA'] - Q_a_hand) / Q_total < 1e-5
        assert abs(r.flows['PB'] - Q_b_hand) / Q_total < 1e-5
        assert abs((r.heads['Ja']-r.heads['Jb']) - h_star) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 2.  compute_system_curve — must use the parallel-equivalent loss, NOT
#     a naive Σ over all non-pump edges
# ─────────────────────────────────────────────────────────────────────────────

class TestParallelSystemCurve:
    """
    With a parallel pair PA‖PB inserted in the loop, the system head loss
    at a given total Q is

        h_sys(Q) = h_pin(Q) + h_parallel_eq(Q) + h_pout(Q)

    where h_parallel_eq is found by solving  h_L,a(Q_a) = h_L,b(Q_b),
    Q_a + Q_b = Q.  This is much LESS than the naive sum  h_L,a(Q) +
    h_L,b(Q) that the current implementation produces.
    """

    def _build_with_pump(self):
        net = PipeNetwork()
        net.add_node(Reservoir('RA', total_head=0.0))
        net.add_node(Reservoir('RB', total_head=15.0))
        net.add_node(Junction('Jp',  elevation=0.0))
        net.add_node(Junction('Ja',  elevation=0.0))
        net.add_node(Junction('Jb',  elevation=0.0))

        net.add_edge(Pump('Pu', A=-6000.0, B=0.0, C=35.0, diameter=0.10),
                     'RA', 'Jp')
        net.add_edge(Pipe('Pin',  diameter=0.10, length=50.0,  roughness=4.6e-5),
                     'Jp', 'Ja')
        net.add_edge(Pipe('PA',   diameter=0.06, length=200.0, roughness=4.6e-5,
                          K_minor=1.5), 'Ja', 'Jb')
        net.add_edge(Pipe('PB',   diameter=0.08, length=200.0, roughness=4.6e-5,
                          K_minor=0.5), 'Ja', 'Jb')
        net.add_edge(Pipe('Pout', diameter=0.10, length=50.0,  roughness=4.6e-5),
                     'Jb', 'RB')
        net.edges['Pu'].component.desired_flow_rate = 0.012
        return net

    def test_system_curve_at_operating_point_matches_pump_head(self):
        """
        After solve, evaluating the system curve at the *actual* solved Q
        should produce h_sys(Q*) ≈ hp(Q*).  This is the consistency check
        between the operating point and the system curve.
        """
        net = self._build_with_pump()
        solver = NetworkSolver(net)
        r = solver.solve()
        assert r.converged, r.message

        Q_op = r.flows['Pu']
        pump = net.edges['Pu'].component
        hp_op = pump.compute_pump_head(Q_op)

        Q_arr, h_arr = solver.compute_system_curve('Pu', r)
        h_sys_at_op = float(np.interp(Q_op, Q_arr, h_arr))

        rel_err = abs(h_sys_at_op - hp_op) / abs(hp_op)
        assert rel_err < 1e-3, \
            f"System curve does NOT pass through operating point: " \
            f"hp(Q*)={hp_op:.4f}m, h_sys(Q*)={h_sys_at_op:.4f}m, " \
            f"rel_err={rel_err*100:.3f}%"

    def test_system_curve_matches_hand_parallel_resolution(self):
        """
        Direct test of compute_system_curve at a sample Q:
        compare against hand-resolved parallel-equivalent loss + static head.
        """
        net = self._build_with_pump()
        solver = NetworkSolver(net)
        r = solver.solve()
        assert r.converged

        Q_test = r.flows['Pu']         # use the actual converged total Q

        # Hand-calc:
        h_static = 15.0 - 0.0
        h_pin    = _pipe_hL(Q_test, D=0.10, L=50.0, eps=4.6e-5, K=0.0)
        h_pout   = _pipe_hL(Q_test, D=0.10, L=50.0, eps=4.6e-5, K=0.0)
        h_par, _ = parallel_equivalent_head_loss(
            Q_test, [
                {'D':0.06,'L':200.0,'eps':4.6e-5,'K':1.5},
                {'D':0.08,'L':200.0,'eps':4.6e-5,'K':0.5},
            ])
        h_sys_hand = h_static + h_pin + h_par + h_pout

        Q_arr, h_arr = solver.compute_system_curve('Pu', r)
        h_sys_code   = float(np.interp(Q_test, Q_arr, h_arr))

        rel_err = abs(h_sys_code - h_sys_hand) / abs(h_sys_hand)
        assert rel_err < 5e-3, \
            f"compute_system_curve mishandles parallel branches.\n" \
            f"   hand-resolved h_sys = {h_sys_hand:.4f} m\n" \
            f"   code      h_sys    = {h_sys_code:.4f} m\n" \
            f"   rel_err = {rel_err*100:.2f}%"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Pump-sizing round-trip on a parallel network
# ─────────────────────────────────────────────────────────────────────────────

class TestParallelPumpSizingRoundTrip:
    """
    UI flow:
        1. Build network with parallel branches + placeholder pump.
        2. Solve.
        3. Read h_req at desired_flow_rate from the system curve.
        4. Call generate_pump_curve(Q_des, h_req).
        5. Apply new pump curve, re-solve.
        6. Solved Q must equal Q_des (within tolerance).

    This is the user's question #3 in its parallel-pipe form.
    """

    def _build(self, pump_A=-6000.0, pump_B=0.0, pump_C=35.0):
        net = PipeNetwork()
        net.add_node(Reservoir('RA', total_head=0.0))
        net.add_node(Reservoir('RB', total_head=12.0))
        net.add_node(Junction('Jp',  elevation=0.0))
        net.add_node(Junction('Ja',  elevation=0.0))
        net.add_node(Junction('Jb',  elevation=0.0))
        net.add_edge(Pump('Pu', A=pump_A, B=pump_B, C=pump_C, diameter=0.10),
                     'RA', 'Jp')
        net.add_edge(Pipe('Pin',  diameter=0.10, length=50.0,  roughness=4.6e-5),
                     'Jp', 'Ja')
        net.add_edge(Pipe('PA',   diameter=0.06, length=200.0, roughness=4.6e-5,
                          K_minor=1.5), 'Ja', 'Jb')
        net.add_edge(Pipe('PB',   diameter=0.08, length=200.0, roughness=4.6e-5,
                          K_minor=0.5), 'Ja', 'Jb')
        net.add_edge(Pipe('Pout', diameter=0.10, length=50.0,  roughness=4.6e-5),
                     'Jb', 'RB')
        return net

    def test_roundtrip_parallel(self):
        Q_des = 0.012   # 12 L/s

        # Phase 1: placeholder pump, solve once to populate the system curve
        net = self._build()
        net.edges['Pu'].component.desired_flow_rate = Q_des
        solver = NetworkSolver(net)
        r = solver.solve()
        assert r.converged

        # Read h_req at Q_des from the system curve  ─ exactly what main_window does
        Q_arr, h_arr = solver.compute_system_curve('Pu', r)
        h_req = float(np.interp(Q_des, Q_arr, h_arr))
        assert h_req > 0

        # Phase 2: generate pump curve and re-solve
        A_c, B_c, C_c = NetworkSolver.generate_pump_curve(Q_des, h_req,
                                                          "centrifugal")
        net2 = self._build(pump_A=A_c, pump_B=B_c, pump_C=C_c)
        r2 = NetworkSolver(net2).solve()
        assert r2.converged

        Q_solved = r2.flows['Pu']
        rel_err  = abs(Q_solved - Q_des) / Q_des
        # If the system curve correctly handles parallel branches, this
        # round-trip is exact (to solver tolerance).
        assert rel_err < 1e-3, \
            f"Parallel-network pump sizing round-trip mismatch.  " \
            f"Q_des={Q_des*1e3:.3f}L/s, Q_solved={Q_solved*1e3:.3f}L/s, " \
            f"rel_err={rel_err*100:.2f}%.  This indicates the system curve " \
            f"used for sizing does NOT match the actual network resistance."
