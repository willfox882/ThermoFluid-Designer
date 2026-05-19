"""
test_series_bernoulli.py
------------------------
Hand-calculated Bernoulli energy-balance checks for series pipe networks.

Scenario family
───────────────
Reservoir A (H_A) → [pump] → pipe (L, D, eps, ΣK) → Reservoir B (H_B)

Steady, incompressible, no demand.  By Bernoulli between free surfaces:

    H_A + h_p(Q)  =  H_B + h_L(Q)

where
    h_L(Q) = ( f(Re, eps/D) * L/D + ΣK ) * V² / (2g),    V = Q / (πD²/4)

For a *known pump curve*  hp(Q) = A·Q² + B·Q + C  the solver finds Q at
the intersection.  For an *unknown pump*  (sizing mode) we specify Q_des and
back-calculate

    h_p,required(Q_des)  =  (H_B - H_A) + h_L(Q_des).
"""

import math
import pytest

from fluid_props import friction_factor, DENSITY, VISCOSITY, GRAVITY
from components import Pipe, Pump, Junction, Reservoir
from network   import PipeNetwork
from solver    import NetworkSolver


# ─────────────────────────────────────────────────────────────────────────────
# Hand-calculation helpers (used as the "truth" reference)
# ─────────────────────────────────────────────────────────────────────────────

def hand_head_loss(Q, D, L, eps, K_sum):
    """Single pipe head loss [m] from textbook Darcy-Weisbach + minor losses."""
    A   = math.pi * D**2 / 4.0
    V   = Q / A
    Re  = DENSITY * V * D / VISCOSITY
    f   = friction_factor(Re, eps / D)
    return (f * L / D + K_sum) * V**2 / (2.0 * GRAVITY)


def hand_required_pump_head(Q_des, H_A, H_B, D, L, eps, K_sum):
    """
    Required pump head to deliver Q_des through the series pipe between
    reservoirs A and B.  Bernoulli A → B:

        H_A + h_p = H_B + h_L(Q_des)
        => h_p = (H_B - H_A) + h_L(Q_des)
    """
    return (H_B - H_A) + hand_head_loss(Q_des, D, L, eps, K_sum)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  No-pump gravity-driven series (sanity: solver matches hand-calc)
# ─────────────────────────────────────────────────────────────────────────────

class TestSeriesNoPump:
    """A→B with H_A > H_B, no pump.  Q found by solver must match hand-calc."""

    def _build(self, H_A=10.0, H_B=0.0, D=0.05, L=100.0, eps=4.6e-5, K_sum=0.0):
        net = PipeNetwork()
        net.add_node(Reservoir('RA', total_head=H_A))
        net.add_node(Reservoir('RB', total_head=H_B))
        p = Pipe('P1', diameter=D, length=L, roughness=eps, K_minor=K_sum)
        net.add_edge(p, 'RA', 'RB')
        return net

    def test_single_pipe_gravity(self):
        """Solver Q must satisfy h_L(Q) = H_A - H_B exactly."""
        H_A, H_B = 10.0, 0.0
        D, L, eps = 0.05, 100.0, 4.6e-5
        net = self._build(H_A, H_B, D, L, eps, 0.0)
        r = NetworkSolver(net).solve()
        assert r.converged, r.message

        Q = r.flows['P1']
        h_L_hand = hand_head_loss(Q, D, L, eps, 0.0)
        assert abs(h_L_hand - (H_A - H_B)) < 1e-6, \
            f"Bernoulli imbalance: h_L_hand={h_L_hand} vs ΔH={H_A-H_B}"

    def test_single_pipe_with_three_elbows(self):
        """Same as above but with ΣK = 3·1.5 = 4.5 (three k=1.5 elbows)."""
        H_A, H_B = 10.0, 0.0
        D, L, eps, K = 0.05, 100.0, 4.6e-5, 4.5
        net = self._build(H_A, H_B, D, L, eps, K)
        r = NetworkSolver(net).solve()
        assert r.converged, r.message

        Q = r.flows['P1']
        h_L_hand = hand_head_loss(Q, D, L, eps, K)
        assert abs(h_L_hand - (H_A - H_B)) < 1e-6
        # Higher K must give lower Q than the no-K case
        net2 = self._build(H_A, H_B, D, L, eps, 0.0)
        r2 = NetworkSolver(net2).solve()
        assert r.flows['P1'] < r2.flows['P1']


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Pump REQUIRED to lift fluid uphill — exact Bernoulli closure
# ─────────────────────────────────────────────────────────────────────────────

class TestSeriesPumpRequired:
    """
    A→pump→pipe→B, H_B > H_A.  For an explicit pump curve, the solver Q
    must close the energy equation:

        H_A + hp(Q) = H_B + h_L(Q)
    """

    def _build(self, H_A, H_B, D, L, eps, K_sum, A_c, B_c, C_c):
        net = PipeNetwork()
        net.add_node(Reservoir('RA', total_head=H_A))
        net.add_node(Reservoir('RB', total_head=H_B))
        net.add_node(Junction ('J1', elevation=0.0))
        pump = Pump('Pu', A=A_c, B=B_c, C=C_c, diameter=D)
        net.add_edge(pump, 'RA', 'J1')
        net.add_edge(Pipe('P1', diameter=D, length=L,
                          roughness=eps, K_minor=K_sum),
                     'J1', 'RB')
        return net

    def test_pump_closes_bernoulli(self):
        """User's exact scenario: H_A=0, H_B=10, pipe + elbows + a pump."""
        H_A, H_B = 0.0, 10.0
        D, L, eps = 0.075, 150.0, 4.6e-5
        K_sum = 3 * 1.5      # three k=1.5 elbows
        # Pump:  hp = -8000·Q² + 25
        A_c, B_c, C_c = -8000.0, 0.0, 25.0
        net = self._build(H_A, H_B, D, L, eps, K_sum, A_c, B_c, C_c)
        r = NetworkSolver(net).solve()
        assert r.converged, r.message

        Q = r.flows['Pu']
        assert abs(r.flows['P1'] - Q) < 1e-12, "mass balance series"

        hp   = A_c*Q**2 + B_c*Q + C_c
        h_L  = hand_head_loss(Q, D, L, eps, K_sum)
        resid = H_A + hp - (H_B + h_L)
        assert abs(resid) < 1e-6, f"Bernoulli residual = {resid:.3e}"

    def test_required_pump_head_matches_back_calc(self):
        """
        Inverse problem (user's main concern):
            Pick a target Q_des, hand-compute h_p,required, build a pump
            whose curve passes through (Q_des, h_req), re-solve, verify
            Q ≈ Q_des to high precision.

        This is the round-trip the UI's "Pump Sizing Mode" performs.
        """
        H_A, H_B = 0.0, 10.0
        D, L, eps = 0.075, 150.0, 4.6e-5
        K_sum = 3 * 1.5
        Q_des = 0.005   # 5 L/s

        # Hand-calc required head
        h_req_hand = hand_required_pump_head(Q_des, H_A, H_B, D, L, eps, K_sum)
        assert h_req_hand > 0, "Need uphill scenario"

        # Build a pump curve hp(Q) = -a·Q² + h_shut with hp(Q_des)=h_req
        # and shut-off head = 1.25 * h_req (matches generate_pump_curve default).
        from solver import NetworkSolver as NS
        A_c, B_c, C_c = NS.generate_pump_curve(Q_des, h_req_hand,
                                               pump_type="centrifugal")
        # Verify the generated curve actually passes through (Q_des, h_req)
        hp_at_Qdes = A_c*Q_des**2 + B_c*Q_des + C_c
        assert abs(hp_at_Qdes - h_req_hand) < 1e-9

        # Solve with this pump
        net = self._build(H_A, H_B, D, L, eps, K_sum, A_c, B_c, C_c)
        r   = NetworkSolver(net).solve()
        assert r.converged, r.message
        Q_solved = r.flows['Pu']

        rel_err = abs(Q_solved - Q_des) / Q_des
        assert rel_err < 1e-4, \
            f"Round-trip mismatch: Q_des={Q_des:.6f}, Q_solved={Q_solved:.6f}, " \
            f"rel_err={rel_err*100:.4f}%"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Multiple pipes in series, with and without pump
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiPipeSeries:
    """Energy balance over an N-pipe series chain."""

    @pytest.mark.parametrize("config", [
        # (H_A, H_B, n_pipes, with_pump)
        (10.0,  0.0, 3, False),    # gravity, no pump
        ( 0.0, 15.0, 3, True),     # uphill, pump
        ( 5.0,  5.0, 4, True),     # flat, pump pushes through losses only
    ])
    def test_series_chain_energy_balance(self, config):
        H_A, H_B, n, with_pump = config
        net = PipeNetwork()
        net.add_node(Reservoir('RA', total_head=H_A))
        net.add_node(Reservoir('RB', total_head=H_B))

        prev = 'RA'
        if with_pump:
            net.add_node(Junction('Jp', elevation=0.0))
            net.add_edge(Pump('Pu', A=-5000.0, B=0.0, C=40.0, diameter=0.05),
                         'RA', 'Jp')
            prev = 'Jp'

        for i in range(n):
            jid = f'J{i}'
            if i < n - 1:
                net.add_node(Junction(jid, elevation=0.0))
                target = jid
            else:
                target = 'RB'
            net.add_edge(Pipe(f'P{i}', diameter=0.05, length=80.0 + 10*i,
                              roughness=4.6e-5, K_minor=0.3 * (i+1)),
                         prev, target)
            prev = target

        r = NetworkSolver(net).solve()
        assert r.converged, r.message

        # Mass balance: same Q on every edge (series)
        flows = list(r.flows.values())
        for q in flows:
            assert abs(q - flows[0]) < 1e-10

        # Energy balance:  H_A + Σhp − Σh_L − H_B = 0
        sum_hp = 0.0
        sum_hL = 0.0
        for eid, hL in r.head_losses.items():
            comp = net.edges[eid].component
            if isinstance(comp, Pump):
                sum_hp += -hL    # h_L = -hp by convention
            else:
                sum_hL += hL

        resid = H_A + sum_hp - sum_hL - H_B
        assert abs(resid) < 1e-6, f"Energy residual = {resid:.3e}"
