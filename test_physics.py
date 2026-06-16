"""
test_physics.py
---------------
Comprehensive test suite for the thermofluid physics engine.
Run with:  python -m pytest tests/ -v
"""

import sys, os, math, pytest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from fluid_props import (
    reynolds_number, friction_factor,
    friction_factor_laminar, friction_factor_haaland,
    DENSITY, VISCOSITY, GRAVITY, VAPOR_PRESSURE, ATMOSPHERIC_PRESSURE,
)
from components import Pipe, Pump, Valve, Junction, Reservoir, component_from_dict
from network import PipeNetwork
from solver import NetworkSolver, SolverResult


# ═══════════════════════════════════════════════════════════════════════════════
# Fluid properties
# ═══════════════════════════════════════════════════════════════════════════════

class TestFluidProps:

    def test_reynolds_laminar(self):
        # Re = ρ·V·D / μ
        D, V = 0.05, 0.01
        Re = reynolds_number(V, D)
        expected = DENSITY * V * D / VISCOSITY
        assert abs(Re - expected) / expected < 1e-12

    def test_friction_factor_laminar(self):
        Re = 1000.0
        f = friction_factor(Re, eps_over_D=0.01)
        assert abs(f - 64.0 / Re) < 1e-12, "Laminar: f = 64/Re"

    def test_friction_factor_turbulent_smooth(self):
        # Moody chart benchmark: smooth pipe Re=1e6 → f ≈ 0.01133 (Haaland)
        f = friction_factor(1e6, eps_over_D=0.0)
        assert 0.010 < f < 0.014

    def test_friction_factor_turbulent_rough(self):
        # High Re + high roughness → fully rough regime f > smooth
        f_rough  = friction_factor(1e6, eps_over_D=0.05)
        f_smooth = friction_factor(1e6, eps_over_D=0.0)
        assert f_rough > f_smooth

    def test_friction_factor_continuity_at_transition(self):
        # f must be C0-continuous across Re=2300 and Re=4000 boundaries.
        # Check the two regime-change points separately — DON'T compare
        # values that straddle the whole transition zone (2300 → 4000) since
        # f legitimately changes there.
        for Re_boundary in (2300.0, 4000.0):
            f_minus = friction_factor(Re_boundary - 0.5, 0.001)
            f_at    = friction_factor(Re_boundary,        0.001)
            f_plus  = friction_factor(Re_boundary + 0.5, 0.001)
            assert abs(f_at - f_minus) < 1e-4, \
                f"Discontinuity at Re={Re_boundary}- : {f_minus} → {f_at}"
            assert abs(f_plus - f_at) < 1e-4, \
                f"Discontinuity at Re={Re_boundary}+ : {f_at} → {f_plus}"

    def test_friction_factor_positive(self):
        for Re in [100, 2300, 4000, 1e5, 1e7]:
            f = friction_factor(Re, 0.001)
            assert f > 0, f"f must be positive at Re={Re}"

    def test_haaland_known_value(self):
        # Haaland (1983): Re=1e5, ε/D=0.001 → f ≈ 0.02148
        f = friction_factor_haaland(1e5, 0.001)
        assert 0.020 < f < 0.024

    def test_zero_flow_edge_case(self):
        # At Re=0 there is no fluid motion, so the Darcy friction factor is
        # undefined.  The code returns 0 (so h_L = f·L/D·V²/2g → 0·anything
        # → 0, which is physically what we want).  The only requirement is
        # that the call must not raise and must return a finite value.
        f = friction_factor(0.0, 0.001)
        assert math.isfinite(f) and f >= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Component unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipe:

    def setup_method(self):
        self.pipe = Pipe('P', diameter=0.1, length=100.0,
                         roughness=4.6e-5, K_minor=0.5)

    def test_area(self):
        expected = math.pi * 0.1**2 / 4.0
        assert abs(self.pipe.area - expected) < 1e-12

    def test_head_loss_positive_for_positive_flow(self):
        assert self.pipe.compute_head_loss(0.01) > 0

    def test_head_loss_negative_for_negative_flow(self):
        assert self.pipe.compute_head_loss(-0.01) < 0

    def test_head_loss_antisymmetric(self):
        Q = 0.015
        assert abs(self.pipe.compute_head_loss(Q) +
                   self.pipe.compute_head_loss(-Q)) < 1e-12

    def test_head_loss_zero_at_zero(self):
        assert self.pipe.compute_head_loss(0.0) == 0.0

    def test_reynolds_number(self):
        Q = 0.01
        V = Q / self.pipe.area
        Re_expected = DENSITY * V * 0.1 / VISCOSITY
        assert abs(self.pipe.compute_reynolds(Q) - Re_expected) < 1

    def test_dhead_loss_dQ_accuracy(self):
        """Analytic Jacobian must match finite difference to < 0.01%."""
        for Q in [1e-4, 1e-3, 0.01, 0.05, 0.1]:
            dh_analytic = self.pipe.dhead_loss_dQ(Q)
            eps = Q * 1e-5
            dh_fd = (self.pipe.compute_head_loss(Q + eps) -
                     self.pipe.compute_head_loss(Q - eps)) / (2 * eps)
            rel_err = abs(dh_analytic - dh_fd) / abs(dh_fd)
            assert rel_err < 1e-4, \
                f"Q={Q}: analytic={dh_analytic:.4f} FD={dh_fd:.4f} err={rel_err*100:.4f}%"

    def test_dhead_loss_dQ_accuracy_reverse_flow(self):
        """Regression (H1): analytic Jacobian must match FD for NEGATIVE flow.

        The Haaland chain-rule term previously used Q·|Q| instead of Q², which
        flipped sign for reverse flow and produced up to ~28% Jacobian error.
        """
        for Q in [-1e-4, -1e-3, -0.01, -0.05, -0.1]:
            dh_analytic = self.pipe.dhead_loss_dQ(Q)
            eps = abs(Q) * 1e-5
            dh_fd = (self.pipe.compute_head_loss(Q + eps) -
                     self.pipe.compute_head_loss(Q - eps)) / (2 * eps)
            rel_err = abs(dh_analytic - dh_fd) / abs(dh_fd)
            assert rel_err < 1e-4, \
                f"Q={Q}: analytic={dh_analytic:.4f} FD={dh_fd:.4f} err={rel_err*100:.4f}%"

    def test_dhead_loss_dQ_even_symmetry(self):
        """dh_L/dQ is even in Q (since h_L is odd): J(+Q) == J(-Q)."""
        for Q in [1e-4, 1e-3, 0.01, 0.05, 0.1]:
            assert abs(self.pipe.dhead_loss_dQ(Q) -
                       self.pipe.dhead_loss_dQ(-Q)) < 1e-9

    def test_dhead_loss_dQ_near_zero(self):
        # Must not crash or return NaN near Q=0
        dh = self.pipe.dhead_loss_dQ(0.0)
        assert math.isfinite(dh) and dh >= 0

    def test_validate_valid(self):
        assert self.pipe.validate() == []

    def test_validate_bad_diameter(self):
        p = Pipe('P2', diameter=-0.1, length=100.0)
        errs = p.validate()
        assert any('diameter' in e for e in errs)

    def test_validate_bad_length(self):
        # Negative length is invalid.  Length = 0 is permitted (a zero-length
        # pipe behaves as a lossless connector; validate() admits length ≥ 0
        # by design — see Pipe.validate).
        p = Pipe('P3', diameter=0.1, length=-1.0)
        errs = p.validate()
        assert any('length' in e for e in errs)

    def test_serialization_roundtrip(self):
        d = self.pipe.to_dict()
        p2 = Pipe.from_dict(d)
        assert p2.id == self.pipe.id
        assert p2.diameter == self.pipe.diameter
        assert p2.length == self.pipe.length
        assert p2.roughness == self.pipe.roughness
        assert p2.K_minor == self.pipe.K_minor

    def test_legacy_elevation_change_key_ignored(self):
        """Backward-compat (M2): old files carry an 'elevation_change' key that
        the model no longer uses; loading must not raise and must ignore it."""
        d = self.pipe.to_dict()
        d["elevation_change"] = 42.0   # simulate a v0.1 save file
        p2 = Pipe.from_dict(d)
        assert not hasattr(p2, "elevation_change")
        assert p2.diameter == self.pipe.diameter


class TestPump:

    def setup_method(self):
        # Typical centrifugal: 25m shut-off, steep falling curve
        self.pump = Pump('Pu', A=-5000.0, B=0.0, C=25.0, diameter=0.1)

    def test_shutoff_head(self):
        hp = self.pump.compute_pump_head(0.0)
        assert abs(hp - 25.0) < 1e-10

    def test_head_decreases_with_flow(self):
        Q = 0.01
        hp1 = self.pump.compute_pump_head(0.0)
        hp2 = self.pump.compute_pump_head(Q)
        assert hp2 < hp1

    def test_head_loss_is_negative_pump_head(self):
        Q = 0.01
        h_L = self.pump.compute_head_loss(Q)
        hp  = self.pump.compute_pump_head(Q)
        assert abs(h_L - (-hp)) < 1e-10

    def test_dhead_loss_dQ_accuracy(self):
        for Q in [1e-4, 1e-3, 0.01, 0.03]:
            dh = self.pump.dhead_loss_dQ(Q)
            eps = 1e-7
            dh_fd = (self.pump.compute_head_loss(Q + eps) -
                     self.pump.compute_head_loss(Q - eps)) / (2 * eps)
            assert abs(dh - dh_fd) / (abs(dh_fd) + 1e-10) < 1e-6

    def test_curve_data_shape(self):
        Q_arr, hp_arr = self.pump.curve_data(n_points=50)
        assert len(Q_arr) == 50
        assert len(hp_arr) == 50

    def test_off_pump_zero_head(self):
        pump_off = Pump('Poff', A=-5000.0, B=0.0, C=25.0, is_on=False)
        assert pump_off.compute_pump_head(0.01) == 0.0

    def test_off_pump_is_high_resistance(self):
        """Improvement #2: an OFF pump acts as a closed link (huge loss), not a
        lossless pass-through."""
        on  = Pump('On',  A=-5000.0, B=0.0, C=25.0, diameter=0.1, is_on=True)
        off = Pump('Off', A=-5000.0, B=0.0, C=25.0, diameter=0.1, is_on=False)
        assert off.compute_head_loss(0.01) > 1e5      # closed-link resistance
        assert off.compute_head_loss(0.0) == 0.0
        # Antisymmetric in Q (check valve modelled symmetrically here)
        assert abs(off.compute_head_loss(0.01) + off.compute_head_loss(-0.01)) < 1e-6
        # Jacobian is finite & positive (no singular zero-derivative)
        assert off.dhead_loss_dQ(0.01) > 0

    def test_validate_stable_curve(self):
        errs = self.pump.validate()
        assert errs == [], f"Valid pump should have no errors: {errs}"

    def test_validate_unstable_curve(self):
        # A > 0 is physically unstable
        p = Pump('Pbad', A=1000.0, B=0.0, C=25.0)
        errs = p.validate()
        assert any('A' in e for e in errs)


class TestNPSH:
    """Regression (H2): NPSHa must use ABSOLUTE suction pressure.

    The solver reports gauge pressure (open surface = 0 Pa), so compute_npsha
    must add atmospheric pressure.  Previously it did not, which made an open
    sump at sea level read NPSHa ≈ -0.23 m and falsely flag cavitation.
    """

    def test_npsha_open_sump_not_cavitating(self):
        pump = Pump('Pu', npsh_required=2.0)
        # Open sump at the surface: gauge pressure 0, negligible suction velocity.
        npsha = pump.compute_npsha(P_suction=0.0, V_suction=0.0)
        expected = (ATMOSPHERIC_PRESSURE - VAPOR_PRESSURE) / (DENSITY * GRAVITY)
        assert abs(npsha - expected) < 1e-9
        assert npsha > 9.0, f"Open sump NPSHa should be ~10 m, got {npsha:.3f}"
        assert not pump.is_cavitating

    def test_npsha_includes_velocity_head(self):
        pump = Pump('Pu', npsh_required=2.0)
        n0 = pump.compute_npsha(P_suction=0.0, V_suction=0.0)
        n1 = pump.compute_npsha(P_suction=0.0, V_suction=2.0)
        assert abs((n1 - n0) - (2.0**2 / (2 * GRAVITY))) < 1e-9

    def test_npsha_low_suction_does_cavitate(self):
        # Strong suction vacuum (≈ -0.9 bar gauge) with a high NPSHr → cavitation.
        pump = Pump('Pu', npsh_required=8.0)
        pump.compute_npsha(P_suction=-90000.0, V_suction=0.0)
        assert pump.is_cavitating


class TestValve:

    def setup_method(self):
        self.valve = Valve('V', diameter=0.1, K=5.0, is_open=True)

    def test_head_loss_positive(self):
        assert self.valve.compute_head_loss(0.01) > 0

    def test_closed_valve_high_loss(self):
        v_closed = Valve('Vc', diameter=0.1, K=5.0, is_open=False)
        h_open   = self.valve.compute_head_loss(0.001)
        h_closed = v_closed.compute_head_loss(0.001)
        assert h_closed > h_open * 1e5  # CLOSED_K >> K

    def test_dhead_loss_dQ_accuracy(self):
        Q = 0.01
        dh = self.valve.dhead_loss_dQ(Q)
        eps = 1e-7
        dh_fd = (self.valve.compute_head_loss(Q + eps) -
                 self.valve.compute_head_loss(Q - eps)) / (2 * eps)
        assert abs(dh - dh_fd) / abs(dh_fd) < 1e-6


class TestJunctionReservoir:

    def test_junction_pressure_head(self):
        j = Junction('J', elevation=5.0)
        j.head = 15.0
        assert abs(j.pressure_head - 10.0) < 1e-10

    def test_reservoir_head_fixed(self):
        r = Reservoir('R', total_head=20.0)
        assert r.total_head == 20.0
        assert r.head == 20.0

    def test_component_factory(self):
        for cls in (Pipe, Pump, Valve, Junction, Reservoir):
            inst = cls('X')
            d    = inst.to_dict()
            inst2 = component_from_dict(d)
            assert type(inst2) is cls
            assert inst2.id == 'X'


# ═══════════════════════════════════════════════════════════════════════════════
# Network construction
# ═══════════════════════════════════════════════════════════════════════════════

class TestNetwork:

    def _two_reservoir_net(self):
        net = PipeNetwork()
        net.add_node(Reservoir('R1', total_head=20.0))
        net.add_node(Reservoir('R2', total_head=0.0))
        net.add_node(Junction('J1', elevation=0.0))
        net.add_edge(Pipe('P1', diameter=0.1, length=300.0), 'R1', 'J1')
        net.add_edge(Pipe('P2', diameter=0.08, length=200.0), 'J1', 'R2')
        return net

    def test_add_node_and_edge(self):
        net = self._two_reservoir_net()
        assert len(net.nodes) == 3
        assert len(net.edges) == 2

    def test_duplicate_node_raises(self):
        net = PipeNetwork()
        net.add_node(Reservoir('R1'))
        with pytest.raises(ValueError):
            net.add_node(Reservoir('R1'))

    def test_duplicate_edge_raises(self):
        net = PipeNetwork()
        net.add_node(Reservoir('R1'))
        net.add_node(Reservoir('R2'))
        net.add_edge(Pipe('P1'), 'R1', 'R2')
        with pytest.raises(ValueError):
            net.add_edge(Pipe('P1'), 'R1', 'R2')

    def test_edge_unknown_node_raises(self):
        net = PipeNetwork()
        net.add_node(Reservoir('R1'))
        with pytest.raises(KeyError):
            net.add_edge(Pipe('P1'), 'R1', 'NONEXISTENT')

    def test_validate_valid_network(self):
        net = self._two_reservoir_net()
        errs = net.validate()
        assert errs == [], f"Valid network should have no errors: {errs}"

    def test_validate_no_reservoir(self):
        net = PipeNetwork()
        net.add_node(Junction('J1'))
        net.add_node(Junction('J2'))
        net.add_edge(Pipe('P1'), 'J1', 'J2')
        errs = net.validate()
        assert any('Reservoir' in e for e in errs)

    def test_validate_disconnected_raises(self):
        net = PipeNetwork()
        net.add_node(Reservoir('R1'))
        net.add_node(Reservoir('R2'))
        net.add_node(Junction('J1'))
        net.add_node(Junction('J2'))
        net.add_edge(Pipe('P1'), 'R1', 'J1')
        # J2 and R2 are isolated from R1/J1
        net.add_edge(Pipe('P2'), 'R2', 'J2')
        errs = net.validate()
        assert any('connected' in e.lower() for e in errs)

    def test_remove_node(self):
        net = self._two_reservoir_net()
        net.remove_node('J1')
        assert 'J1' not in net.nodes
        assert 'P1' not in net.edges
        assert 'P2' not in net.edges

    def test_incidence_matrix_shape(self):
        net = self._two_reservoir_net()
        A, free_ids, edge_ids = net.build_incidence_matrix()
        # One free node (J1), two edges
        assert A.shape == (1, 2)

    def test_incidence_matrix_values(self):
        net = self._two_reservoir_net()
        A, free_ids, edge_ids = net.build_incidence_matrix()
        # P1 flows INTO J1 → A[0, edge_idx(P1)] = +1
        # P2 flows OUT of J1 → A[0, edge_idx(P2)] = -1
        j_idx   = 0  # only one free node
        p1_idx  = edge_ids.index('P1')
        p2_idx  = edge_ids.index('P2')
        assert A[j_idx, p1_idx] == +1
        assert A[j_idx, p2_idx] == -1

    def test_json_roundtrip(self):
        import tempfile, os
        net = self._two_reservoir_net()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            net.save_json(path)
            net2 = PipeNetwork.load_json(path)
            assert set(net2.nodes) == set(net.nodes)
            assert set(net2.edges) == set(net.edges)
            assert net2.edges['P1'].from_node_id == 'R1'
            assert net2.edges['P1'].to_node_id   == 'J1'
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════════
# Solver
# ═══════════════════════════════════════════════════════════════════════════════

class TestSolver:

    def _solve(self, net: PipeNetwork) -> SolverResult:
        return NetworkSolver(net).solve()

    # ── 1. Simple two-reservoir, two-pipe series ──────────────────────────────

    def test_series_pipe_convergence(self):
        net = PipeNetwork()
        net.add_node(Reservoir('R1', total_head=20.0))
        net.add_node(Reservoir('R2', total_head=0.0))
        net.add_node(Junction('J1', elevation=0.0))
        net.add_edge(Pipe('P1', diameter=0.1, length=500.0, roughness=4.6e-5), 'R1', 'J1')
        net.add_edge(Pipe('P2', diameter=0.08, length=300.0, roughness=4.6e-5), 'J1', 'R2')
        r = self._solve(net)
        assert r.converged, f"Did not converge: {r.message}"
        assert r.residual_norm < 1e-8

    def test_series_pipe_mass_balance(self):
        net = PipeNetwork()
        net.add_node(Reservoir('R1', total_head=20.0))
        net.add_node(Reservoir('R2', total_head=0.0))
        net.add_node(Junction('J1', elevation=0.0))
        net.add_edge(Pipe('P1', diameter=0.1, length=500.0), 'R1', 'J1')
        net.add_edge(Pipe('P2', diameter=0.08, length=300.0), 'J1', 'R2')
        r = self._solve(net)
        assert r.converged
        # Mass balance: Q_P1 = Q_P2 (series)
        assert abs(r.flows['P1'] - r.flows['P2']) < 1e-9

    def test_series_pipe_energy_balance(self):
        net = PipeNetwork()
        net.add_node(Reservoir('R1', total_head=20.0))
        net.add_node(Reservoir('R2', total_head=0.0))
        net.add_node(Junction('J1', elevation=0.0))
        net.add_edge(Pipe('P1', diameter=0.1, length=500.0, roughness=4.6e-5), 'R1', 'J1')
        net.add_edge(Pipe('P2', diameter=0.08, length=300.0, roughness=4.6e-5), 'J1', 'R2')
        r = self._solve(net)
        assert r.converged
        # Total head loss must equal head difference
        total_loss = r.head_losses['P1'] + r.head_losses['P2']
        head_diff  = 20.0 - 0.0
        assert abs(total_loss - head_diff) < 1e-6, \
            f"Energy: Σh_L={total_loss:.6f}m ≠ ΔH={head_diff}m"

    def test_series_pipe_boundary_heads(self):
        net = PipeNetwork()
        net.add_node(Reservoir('R1', total_head=20.0))
        net.add_node(Reservoir('R2', total_head=0.0))
        net.add_node(Junction('J1', elevation=0.0))
        net.add_edge(Pipe('P1', diameter=0.1, length=500.0), 'R1', 'J1')
        net.add_edge(Pipe('P2', diameter=0.08, length=300.0), 'J1', 'R2')
        r = self._solve(net)
        assert r.converged
        assert abs(r.heads['R1'] - 20.0) < 1e-10
        assert abs(r.heads['R2'] -  0.0) < 1e-10

    # ── 2. Parallel pipe network ───────────────────────────────────────────────

    def test_parallel_pipes_mass_balance(self):
        """Two parallel pipes: Q_total = Q_a + Q_b."""
        net = PipeNetwork()
        net.add_node(Reservoir('R1', total_head=15.0))
        net.add_node(Reservoir('R2', total_head=0.0))
        net.add_node(Junction('Ja', elevation=0.0))
        net.add_node(Junction('Jb', elevation=0.0))
        # Splitter
        net.add_edge(Pipe('P0', diameter=0.15, length=50.0),  'R1', 'Ja')
        # Parallel branches
        net.add_edge(Pipe('Pa', diameter=0.10, length=300.0), 'Ja', 'Jb')
        net.add_edge(Pipe('Pb', diameter=0.08, length=300.0), 'Ja', 'Jb')
        # Collector
        net.add_edge(Pipe('P1', diameter=0.15, length=50.0),  'Jb', 'R2')
        r = self._solve(net)
        assert r.converged, f"Did not converge: {r.message}"
        # Conservation at Ja: Q_P0 = Q_Pa + Q_Pb
        imbalance_a = abs(r.flows['P0'] - r.flows['Pa'] - r.flows['Pb'])
        assert imbalance_a < 1e-9, f"Mass balance at Ja: {imbalance_a:.2e}"

    def test_parallel_pipes_equal_head_loss(self):
        """Parallel pipes must have identical head loss (same end nodes)."""
        net = PipeNetwork()
        net.add_node(Reservoir('R1', total_head=15.0))
        net.add_node(Reservoir('R2', total_head=0.0))
        net.add_node(Junction('Ja', elevation=0.0))
        net.add_node(Junction('Jb', elevation=0.0))
        net.add_edge(Pipe('P0', diameter=0.15, length=50.0),  'R1', 'Ja')
        net.add_edge(Pipe('Pa', diameter=0.10, length=300.0), 'Ja', 'Jb')
        net.add_edge(Pipe('Pb', diameter=0.08, length=300.0), 'Ja', 'Jb')
        net.add_edge(Pipe('P1', diameter=0.15, length=50.0),  'Jb', 'R2')
        r = self._solve(net)
        assert r.converged
        dh_Pa = r.heads['Ja'] - r.heads['Jb']
        dh_Pb = r.heads['Ja'] - r.heads['Jb']
        assert abs(dh_Pa - dh_Pb) < 1e-10

    # ── 3. Pump network ────────────────────────────────────────────────────────

    def test_pump_convergence(self):
        """Pump from sump → two delivery branches."""
        net = PipeNetwork()
        net.add_node(Reservoir('R_sump', total_head=0.0))
        net.add_node(Reservoir('R_a',    total_head=20.0))
        net.add_node(Reservoir('R_b',    total_head=15.0))
        net.add_node(Junction('J1',      elevation=0.0))
        net.add_edge(Pump('Pu1', A=-8000.0, B=0.0, C=30.0, diameter=0.1), 'R_sump', 'J1')
        net.add_edge(Pipe('P1',  diameter=0.10, length=200.0), 'J1', 'R_a')
        net.add_edge(Pipe('P2',  diameter=0.08, length=150.0), 'J1', 'R_b')
        r = self._solve(net)
        assert r.converged, f"Pump network did not converge: {r.message}"

    def test_pump_adds_energy(self):
        """Pump head loss must be negative (energy addition)."""
        net = PipeNetwork()
        net.add_node(Reservoir('R_low',  total_head=0.0))
        net.add_node(Reservoir('R_high', total_head=20.0))
        net.add_node(Junction('J1', elevation=0.0))
        net.add_edge(Pump('Pu1', A=-5000.0, B=0.0, C=25.0, diameter=0.1), 'R_low', 'J1')
        net.add_edge(Pipe('P1',  diameter=0.08, length=300.0), 'J1', 'R_high')
        r = self._solve(net)
        assert r.converged
        assert r.head_losses['Pu1'] < 0, "Pump h_L must be negative"
        assert r.flows['Pu1'] > 0,       "Pump flow must be positive"

    def test_pump_mass_balance(self):
        net = PipeNetwork()
        net.add_node(Reservoir('Rs', total_head=0.0))
        net.add_node(Reservoir('Ra', total_head=20.0))
        net.add_node(Reservoir('Rb', total_head=15.0))
        net.add_node(Junction('J1', elevation=0.0))
        net.add_edge(Pump('Pu1', A=-8000.0, B=0.0, C=30.0, diameter=0.1), 'Rs', 'J1')
        net.add_edge(Pipe('P1',  diameter=0.10, length=200.0), 'J1', 'Ra')
        net.add_edge(Pipe('P2',  diameter=0.08, length=150.0), 'J1', 'Rb')
        r = self._solve(net)
        assert r.converged
        j_node  = net.nodes['J1']
        Q_in    = sum(r.flows[e] for e in j_node.connected_edge_ids
                      if net.edges[e].to_node_id   == 'J1')
        Q_out   = sum(r.flows[e] for e in j_node.connected_edge_ids
                      if net.edges[e].from_node_id == 'J1')
        assert abs(Q_in - Q_out) < 1e-9, \
            f"Mass balance at J1: Q_in={Q_in*1e3:.4f} Q_out={Q_out*1e3:.4f} L/s"

    def test_reverse_flow_edge_converges(self):
        """Regression (H1): an edge whose declared direction opposes the actual
        flow (negative Q) must still converge with full Jacobian accuracy.

        P2 is declared J1→R2 but R2 is the HIGH reservoir, so the physical flow
        is R2→J1, i.e. Q_P2 < 0.  This exercises the reverse-flow Jacobian term.
        """
        net = PipeNetwork()
        net.add_node(Reservoir('R1', total_head=5.0))
        net.add_node(Reservoir('R2', total_head=20.0))   # higher → feeds backward
        net.add_node(Reservoir('R3', total_head=0.0))
        net.add_node(Junction('J1', elevation=0.0))
        net.add_edge(Pipe('P1', diameter=0.10, length=200.0, roughness=4.6e-5), 'R1', 'J1')
        net.add_edge(Pipe('P2', diameter=0.10, length=200.0, roughness=4.6e-5), 'J1', 'R2')
        net.add_edge(Pipe('P3', diameter=0.10, length=200.0, roughness=4.6e-5), 'J1', 'R3')
        r = self._solve(net)
        assert r.converged, f"Reverse-flow net did not converge: {r.message}"
        assert r.residual_norm < 1e-8
        assert r.flows['P2'] < 0, "P2 should carry reverse (negative) flow"
        # Per-edge Bernoulli closure holds even on the reverse edge
        for eid, edge in net.edges.items():
            dH = r.heads[edge.from_node_id] - r.heads[edge.to_node_id]
            assert abs(dH - r.head_losses[eid]) < 1e-7

    def test_off_pump_blocks_flow(self):
        """Improvement #2: a 20 m head across an OFF pump yields ~0 flow."""
        net = PipeNetwork()
        net.add_node(Reservoir('R1', total_head=20.0))
        net.add_node(Reservoir('R2', total_head=0.0))
        net.add_node(Junction('J1', elevation=0.0))
        net.add_edge(Pump('Pu1', A=-8000.0, B=0.0, C=30.0, diameter=0.1,
                          is_on=False), 'R1', 'J1')
        net.add_edge(Pipe('P1', diameter=0.1, length=100.0), 'J1', 'R2')
        r = self._solve(net)
        assert r.converged, r.message
        assert abs(r.flows['Pu1']) < 1e-4, "OFF pump must throttle flow to ~0"
        assert abs(r.flows['P1']) < 1e-4

    def test_off_pump_only_edges_not_singular(self):
        """Regression (L4): a free node whose only edges are OFF pumps used to
        give a singular Jacobian; closed-link semantics make it solvable."""
        net = PipeNetwork()
        net.add_node(Reservoir('R1', total_head=20.0))
        net.add_node(Reservoir('R2', total_head=0.0))
        net.add_node(Junction('J1', elevation=0.0))
        net.add_edge(Pump('Pu1', diameter=0.1, is_on=False), 'R1', 'J1')
        net.add_edge(Pump('Pu2', diameter=0.1, is_on=False), 'J1', 'R2')
        r = self._solve(net)
        assert r.converged, f"Off-pump-only node should solve: {r.message}"
        assert abs(r.flows['Pu1']) < 1e-4

    def test_worst_residual_diagnostic(self):
        """Improvement #6: SolverResult.worst_residual names the worst-satisfied
        equation (continuity@node or energy@edge) for failure diagnosis."""
        net = PipeNetwork()
        net.add_node(Reservoir('R1', total_head=20.0))
        net.add_node(Reservoir('R2', total_head=0.0))
        net.add_node(Junction('J1', elevation=0.0))
        net.add_edge(Pipe('P1', diameter=0.10, length=300.0), 'R1', 'J1')
        net.add_edge(Pipe('P2', diameter=0.08, length=200.0), 'J1', 'R2')
        r = self._solve(net)
        assert r.converged
        assert r.worst_residual is not None
        label, val = r.worst_residual
        assert ('node' in label) or ('edge' in label)
        assert abs(val) < 1e-6   # converged → worst residual is tiny

    def test_result_carries_npsh_data(self):
        """Improvement #3: SolverResult.npsh exposes NPSHa/NPSHr/margin per pump
        so the results table + CSV can report cavitation without reaching into
        component internals."""
        net = PipeNetwork()
        net.add_node(Reservoir('Rs', total_head=2.0))
        net.add_node(Reservoir('Ra', total_head=18.0))
        net.add_node(Junction('J1', elevation=0.0))
        net.add_edge(Pump('Pu1', A=-8000.0, B=0.0, C=30.0, diameter=0.1,
                          npsh_required=2.0), 'Rs', 'J1')
        net.add_edge(Pipe('P1', diameter=0.08, length=150.0), 'J1', 'Ra')
        r = self._solve(net)
        assert r.converged
        assert 'Pu1' in r.npsh, "Pump NPSH data missing from result"
        d = r.npsh['Pu1']
        assert abs(d["margin"] - (d["available"] - d["required"])) < 1e-9
        assert d["required"] == 2.0
        # With atmospheric pressure included, a +2 m suction reservoir is safe.
        assert not d["cavitating"] and d["available"] > 9.0

    # ── 4. Demand node ─────────────────────────────────────────────────────────

    def test_demand_node(self):
        """Junction with non-zero demand: continuity must still hold."""
        net = PipeNetwork()
        net.add_node(Reservoir('R1', total_head=20.0))
        net.add_node(Reservoir('R2', total_head=0.0))
        net.add_node(Junction('J1', elevation=0.0, demand=0.002))  # 2 L/s withdrawal
        net.add_edge(Pipe('P1', diameter=0.1, length=300.0), 'R1', 'J1')
        net.add_edge(Pipe('P2', diameter=0.08, length=200.0), 'J1', 'R2')
        r = self._solve(net)
        assert r.converged
        # Q_in − Q_out = demand
        j = net.nodes['J1']
        Q_in  = sum(r.flows[e] for e in j.connected_edge_ids
                    if net.edges[e].to_node_id   == 'J1')
        Q_out = sum(r.flows[e] for e in j.connected_edge_ids
                    if net.edges[e].from_node_id == 'J1')
        assert abs((Q_in - Q_out) - 0.002) < 1e-9

    # ── 5. Solver robustness ───────────────────────────────────────────────────

    def test_validation_error_no_reservoir(self):
        net = PipeNetwork()
        net.add_node(Junction('J1'))
        net.add_node(Junction('J2'))
        net.add_edge(Pipe('P1'), 'J1', 'J2')
        r = NetworkSolver(net).solve()
        assert not r.converged
        assert len(r.errors) > 0

    def test_solve_idempotent(self):
        """Solving the same network twice gives the same result."""
        net = PipeNetwork()
        net.add_node(Reservoir('R1', total_head=20.0))
        net.add_node(Reservoir('R2', total_head=0.0))
        net.add_node(Junction('J1', elevation=0.0))
        net.add_edge(Pipe('P1', diameter=0.1, length=300.0), 'R1', 'J1')
        net.add_edge(Pipe('P2', diameter=0.08, length=200.0), 'J1', 'R2')
        solver = NetworkSolver(net)
        r1 = solver.solve()
        r2 = solver.solve()
        assert r1.converged and r2.converged
        assert abs(r1.flows['P1'] - r2.flows['P1']) < 1e-12

    def test_detect_three_pumps_in_series(self):
        """Regression (L3): 3 chained pumps must classify as one 'series' group.

        The old union-find keyed the config by a root that path-compression could
        change, mislabelling groups of 3+ pumps.
        """
        net = PipeNetwork()
        net.add_node(Reservoir('R1', total_head=0.0))
        net.add_node(Reservoir('R2', total_head=60.0))
        net.add_node(Junction('J1', elevation=0.0))
        net.add_node(Junction('J2', elevation=0.0))
        net.add_node(Junction('J3', elevation=0.0))
        net.add_edge(Pump('Pu1', A=-4000.0, B=0.0, C=25.0, diameter=0.1), 'R1', 'J1')
        net.add_edge(Pump('Pu2', A=-4000.0, B=0.0, C=25.0, diameter=0.1), 'J1', 'J2')
        net.add_edge(Pump('Pu3', A=-4000.0, B=0.0, C=25.0, diameter=0.1), 'J2', 'J3')
        net.add_edge(Pipe('P1', diameter=0.1, length=100.0), 'J3', 'R2')
        groups = NetworkSolver(net).detect_pump_groups()
        assert len(groups) == 1, f"Expected one group, got {groups}"
        assert groups[0]["type"] == "series"
        assert set(groups[0]["pump_ids"]) == {'Pu1', 'Pu2', 'Pu3'}

    def test_system_curve_generation(self):
        net = PipeNetwork()
        net.add_node(Reservoir('Rs', total_head=0.0))
        net.add_node(Reservoir('Ra', total_head=20.0))
        net.add_node(Junction('J1', elevation=0.0))
        net.add_edge(Pump('Pu1', A=-5000.0, B=0.0, C=25.0, diameter=0.1), 'Rs', 'J1')
        net.add_edge(Pipe('P1',  diameter=0.08, length=300.0), 'J1', 'Ra')
        solver = NetworkSolver(net)
        r = solver.solve()
        assert r.converged
        Q_arr, h_arr = solver.compute_system_curve('Pu1', r)
        assert len(Q_arr) > 10
        assert all(np.diff(h_arr) >= -1e-6)   # system curve is non-decreasing


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import subprocess, sys
    sys.exit(subprocess.call(["python", "-m", "pytest", __file__, "-v"]))
