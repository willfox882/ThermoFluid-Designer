"""
test_pump_sizing_consistency.py
-------------------------------
The user's question (#3):

    "Have a series and parallel network with a pump (unknown), when I press
     solve, are the pump characteristics the EXACT same as if I had the same
     network with known pump characteristics but UNKNOWN FLOW?"

What this means operationally
─────────────────────────────
1. Build a network with an arbitrary pump curve and a target desired_flow_rate Q_des.
2. Solve forward → records system curve at Q_des → reports h_req.
3. Take (Q_des, h_req), generate a pump curve via NetworkSolver.generate_pump_curve.
4. Replace the pump with the new one and solve again.
5. The new operating point Q' MUST equal Q_des (because the new pump's curve
   is constructed to pass through (Q_des, h_req) and the system curve also
   passes through (Q_des, h_req) — they intersect at exactly Q_des).

If step 5 fails, the system curve handed to the user (and to generate_pump_curve)
is inconsistent with the actual network resistance.
"""

import math
import numpy as np
import pytest

from components import Pipe, Pump, Junction, Reservoir
from network   import PipeNetwork
from solver    import NetworkSolver


def _read_h_req_at(solver, pump_id, Q_des, result):
    Q_arr, h_arr = solver.compute_system_curve(pump_id, result)
    return float(np.interp(Q_des, Q_arr, h_arr))


# ─────────────────────────────────────────────────────────────────────────────
# SERIES networks
# ─────────────────────────────────────────────────────────────────────────────

class TestSeriesRoundTrip:

    def _net(self, A=-8000.0, B=0.0, C=25.0, with_elbows=True):
        net = PipeNetwork()
        net.add_node(Reservoir('RS', total_head=0.0))
        net.add_node(Reservoir('RD', total_head=12.0))
        net.add_node(Junction('J1', elevation=0.0))
        net.add_node(Junction('J2', elevation=0.0))
        K = 3 * 1.5 if with_elbows else 0.0
        net.add_edge(Pump('Pu', A=A, B=B, C=C, diameter=0.075), 'RS', 'J1')
        net.add_edge(Pipe('P1', diameter=0.075, length=100.0,
                          roughness=4.6e-5, K_minor=K), 'J1', 'J2')
        net.add_edge(Pipe('P2', diameter=0.075, length= 80.0,
                          roughness=4.6e-5), 'J2', 'RD')
        return net

    @pytest.mark.parametrize("Q_des", [0.003, 0.005, 0.008, 0.012])
    def test_round_trip_series(self, Q_des):
        # Phase 1
        net = self._net()
        net.edges['Pu'].component.desired_flow_rate = Q_des
        solver = NetworkSolver(net)
        r = solver.solve()
        assert r.converged
        h_req = _read_h_req_at(solver, 'Pu', Q_des, r)
        if h_req <= 0:
            pytest.skip(f"Q_des={Q_des} requires no pump head (gravity supplies it)")

        # Phase 2
        A_c, B_c, C_c = NetworkSolver.generate_pump_curve(Q_des, h_req)
        net2 = self._net(A=A_c, B=B_c, C=C_c)
        r2 = NetworkSolver(net2).solve()
        assert r2.converged
        Q_solved = r2.flows['Pu']
        rel_err  = abs(Q_solved - Q_des) / Q_des
        assert rel_err < 1e-3, \
            f"Series round-trip mismatch at Q_des={Q_des}: " \
            f"Q_solved={Q_solved}, rel_err={rel_err*100:.3f}%"


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL networks
# ─────────────────────────────────────────────────────────────────────────────

class TestParallelRoundTrip:

    def _net(self, A=-6000.0, B=0.0, C=35.0):
        net = PipeNetwork()
        net.add_node(Reservoir('RA', total_head=0.0))
        net.add_node(Reservoir('RB', total_head=10.0))
        net.add_node(Junction('Jp',  elevation=0.0))
        net.add_node(Junction('Ja',  elevation=0.0))
        net.add_node(Junction('Jb',  elevation=0.0))
        net.add_edge(Pump('Pu', A=A, B=B, C=C, diameter=0.10),  'RA', 'Jp')
        net.add_edge(Pipe('Pin',  diameter=0.10, length=40.0,
                          roughness=4.6e-5),                    'Jp', 'Ja')
        net.add_edge(Pipe('PA',   diameter=0.06, length=150.0,
                          roughness=4.6e-5, K_minor=1.5),       'Ja', 'Jb')
        net.add_edge(Pipe('PB',   diameter=0.08, length=150.0,
                          roughness=4.6e-5, K_minor=0.5),       'Ja', 'Jb')
        net.add_edge(Pipe('Pout', diameter=0.10, length=40.0,
                          roughness=4.6e-5),                    'Jb', 'RB')
        return net

    @pytest.mark.parametrize("Q_des", [0.005, 0.010, 0.015])
    def test_round_trip_parallel(self, Q_des):
        net = self._net()
        net.edges['Pu'].component.desired_flow_rate = Q_des
        solver = NetworkSolver(net)
        r = solver.solve()
        assert r.converged
        h_req = _read_h_req_at(solver, 'Pu', Q_des, r)
        if h_req <= 0:
            pytest.skip("Static head supplies flow without pump work.")

        A_c, B_c, C_c = NetworkSolver.generate_pump_curve(Q_des, h_req)
        net2 = self._net(A=A_c, B=B_c, C=C_c)
        r2 = NetworkSolver(net2).solve()
        assert r2.converged
        Q_solved = r2.flows['Pu']
        rel_err  = abs(Q_solved - Q_des) / Q_des
        assert rel_err < 1e-3, \
            f"Parallel round-trip mismatch at Q_des={Q_des}: " \
            f"Q_solved={Q_solved} rel_err={rel_err*100:.3f}%"


# ─────────────────────────────────────────────────────────────────────────────
# SERIES + PARALLEL combined
# ─────────────────────────────────────────────────────────────────────────────

class TestSeriesPlusParallelRoundTrip:

    def _net(self, A=-7000.0, B=0.0, C=40.0):
        """RS → pump → J1 → P1 → Ja → (PA‖PB) → Jb → P2 → RD."""
        net = PipeNetwork()
        net.add_node(Reservoir('RS', total_head=0.0))
        net.add_node(Reservoir('RD', total_head=8.0))
        for nid in ('J1','Ja','Jb'):
            net.add_node(Junction(nid, elevation=0.0))
        net.add_edge(Pump('Pu', A=A, B=B, C=C, diameter=0.10), 'RS', 'J1')
        net.add_edge(Pipe('P1',  diameter=0.10, length=60.0,
                          roughness=4.6e-5, K_minor=2.0), 'J1', 'Ja')
        net.add_edge(Pipe('PA',  diameter=0.05, length=180.0,
                          roughness=4.6e-5, K_minor=1.0), 'Ja', 'Jb')
        net.add_edge(Pipe('PB',  diameter=0.07, length=180.0,
                          roughness=4.6e-5, K_minor=0.5), 'Ja', 'Jb')
        net.add_edge(Pipe('P2',  diameter=0.10, length=70.0,
                          roughness=4.6e-5, K_minor=1.0), 'Jb', 'RD')
        return net

    def test_round_trip_combined(self):
        Q_des = 0.007
        net = self._net()
        net.edges['Pu'].component.desired_flow_rate = Q_des
        solver = NetworkSolver(net)
        r = solver.solve()
        assert r.converged
        h_req = _read_h_req_at(solver, 'Pu', Q_des, r)
        if h_req <= 0:
            pytest.skip("Gravity supplies flow.")

        A_c, B_c, C_c = NetworkSolver.generate_pump_curve(Q_des, h_req)
        net2 = self._net(A=A_c, B=B_c, C=C_c)
        r2 = NetworkSolver(net2).solve()
        assert r2.converged
        Q_solved = r2.flows['Pu']
        rel_err  = abs(Q_solved - Q_des) / Q_des
        assert rel_err < 2e-3, \
            f"Series+parallel round-trip mismatch: Q_solved={Q_solved}, " \
            f"Q_des={Q_des}, rel_err={rel_err*100:.3f}%"
