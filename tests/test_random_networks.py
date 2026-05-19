"""
test_random_networks.py
-----------------------
Randomly-generated network audit.  For every seed we build a network with
plausible-but-random geometry and verify:

  • mass balance at every junction        (Σ in − Σ out = demand)
  • Bernoulli on every edge               (H_from − H_to = h_L(Q))
  • residual norm                          (||F(x*)|| < 1e-6)
  • round-trip pump sizing consistency     (Q_solved == Q_des)

We restrict to series and series+parallel topologies (user's questions 1-3)
because those are the cases the README claims are validated.
"""

import math
import random
import numpy as np
import pytest

from components import Pipe, Pump, Junction, Reservoir
from network   import PipeNetwork
from solver    import NetworkSolver


# ─────────────────────────────────────────────────────────────────────────────
# Random-network factory
# ─────────────────────────────────────────────────────────────────────────────

def random_series_network(seed, with_pump):
    rng = random.Random(seed)
    n_pipes = rng.randint(2, 5)
    H_low   = rng.uniform( 0.0,  5.0)
    H_high  = rng.uniform(10.0, 30.0)

    net = PipeNetwork()
    if with_pump:
        # Pump lifts from low → high reservoir
        net.add_node(Reservoir('RA', total_head=H_low))
        net.add_node(Reservoir('RB', total_head=H_high))
        H_lift = H_high - H_low
        # Pump curve scaled so that its operating range plausibly covers
        # this lift.  Shutoff head ~ 1.5 × lift, free-delivery Q ~ 0.02 m³/s.
        C = 1.5 * H_lift + 5.0
        A_c = -(C - H_lift) / 0.02**2
        net.add_node(Junction('J0', elevation=0.0))
        net.add_edge(Pump('Pu', A=A_c, B=0.0, C=C, diameter=0.075),
                     'RA', 'J0')
        prev = 'J0'
    else:
        # Gravity-driven from high → low
        net.add_node(Reservoir('RA', total_head=H_high))
        net.add_node(Reservoir('RB', total_head=H_low))
        prev = 'RA'

    for i in range(n_pipes):
        nxt = 'RB' if i == n_pipes - 1 else f'J{i+1}'
        if nxt != 'RB':
            net.add_node(Junction(nxt, elevation=0.0))
        net.add_edge(
            Pipe(f'P{i}',
                 diameter  = rng.uniform(0.05, 0.12),
                 length    = rng.uniform(40.0, 200.0),
                 roughness = rng.choice([4.6e-5, 1.5e-6, 1.5e-4]),
                 K_minor   = rng.uniform(0.0, 4.0)),
            prev, nxt)
        prev = nxt
    return net


def random_parallel_network(seed, with_pump):
    rng = random.Random(seed)
    H_low   = rng.uniform( 0.0,  3.0)
    H_high  = rng.uniform(10.0, 25.0)
    n_branches = rng.randint(2, 3)

    net = PipeNetwork()
    if with_pump:
        net.add_node(Reservoir('RA', total_head=H_low))
        net.add_node(Reservoir('RB', total_head=H_high))
        H_lift = H_high - H_low
        C = 1.5 * H_lift + 5.0
        A_c = -(C - H_lift) / 0.02**2
        net.add_node(Junction('Jp', elevation=0.0))
        net.add_edge(Pump('Pu', A=A_c, B=0.0, C=C, diameter=0.10), 'RA', 'Jp')
        upstream = 'Jp'
    else:
        net.add_node(Reservoir('RA', total_head=H_high))
        net.add_node(Reservoir('RB', total_head=H_low))
        upstream = 'RA'

    net.add_node(Junction('Ja', elevation=0.0))
    net.add_node(Junction('Jb', elevation=0.0))
    net.add_edge(Pipe('Pin',  diameter=0.10, length=40.0,
                      roughness=4.6e-5), upstream, 'Ja')
    for i in range(n_branches):
        net.add_edge(
            Pipe(f'PB{i}',
                 diameter  = rng.uniform(0.04, 0.09),
                 length    = rng.uniform(80.0, 200.0),
                 roughness = 4.6e-5,
                 K_minor   = rng.uniform(0.0, 2.5)),
            'Ja', 'Jb')
    net.add_edge(Pipe('Pout', diameter=0.10, length=40.0,
                      roughness=4.6e-5), 'Jb', 'RB')
    return net


# ─────────────────────────────────────────────────────────────────────────────
# Generic invariants — mass balance, energy balance, residual
# ─────────────────────────────────────────────────────────────────────────────

def check_invariants(net, r):
    # Per-junction continuity
    for nid, node in net.nodes.items():
        if not node.is_junction():
            continue
        flow_in = sum(r.flows[e] for e in node.connected_edge_ids
                      if net.edges[e].to_node_id == nid)
        flow_out = sum(r.flows[e] for e in node.connected_edge_ids
                       if net.edges[e].from_node_id == nid)
        demand = node.component.demand
        imbal = flow_in - flow_out - demand
        assert abs(imbal) < 1e-9, \
            f"Mass balance at {nid}: imbal={imbal:.3e}"

    # Per-edge Bernoulli: H_from − H_to == h_L
    for eid, edge in net.edges.items():
        H_from = r.heads[edge.from_node_id]
        H_to   = r.heads[edge.to_node_id]
        h_L    = r.head_losses[eid]
        assert abs((H_from - H_to) - h_L) < 1e-7, \
            f"Energy on {eid}: ΔH={H_from-H_to:.6f}, h_L={h_L:.6f}"


@pytest.mark.parametrize("seed", range(8))
def test_random_series_no_pump(seed):
    net = random_series_network(seed, with_pump=False)
    r = NetworkSolver(net).solve()
    assert r.converged, f"seed {seed} failed: {r.message}"
    assert r.residual_norm < 1e-6
    check_invariants(net, r)


@pytest.mark.parametrize("seed", range(8))
def test_random_series_with_pump(seed):
    net = random_series_network(seed, with_pump=True)
    r = NetworkSolver(net).solve()
    assert r.converged, f"seed {seed} failed: {r.message}"
    assert r.residual_norm < 1e-6
    check_invariants(net, r)


@pytest.mark.parametrize("seed", range(8))
def test_random_parallel_no_pump(seed):
    net = random_parallel_network(seed, with_pump=False)
    r = NetworkSolver(net).solve()
    assert r.converged, f"seed {seed} failed: {r.message}"
    assert r.residual_norm < 1e-6
    check_invariants(net, r)


@pytest.mark.parametrize("seed", range(8))
def test_random_parallel_with_pump(seed):
    net = random_parallel_network(seed, with_pump=True)
    r = NetworkSolver(net).solve()
    assert r.converged, f"seed {seed} failed: {r.message}"
    assert r.residual_norm < 1e-6
    check_invariants(net, r)


# ─────────────────────────────────────────────────────────────────────────────
# Round-trip pump-sizing consistency on randomly generated networks
#   (user's question #3, across many topologies)
# ─────────────────────────────────────────────────────────────────────────────

def _round_trip(net_factory, seed):
    net = net_factory(seed, with_pump=True)
    solver = NetworkSolver(net)
    # Pick a Q_des that's a fraction of the converged forward-solve flow
    r = solver.solve()
    if not r.converged:
        pytest.skip("forward solve failed")

    pump_id = next(eid for eid, e in net.edges.items()
                   if e.component.__class__.__name__ == "Pump")
    Q_forward = r.flows[pump_id]
    if Q_forward <= 0:
        pytest.skip("non-positive forward flow")
    Q_des = 0.7 * Q_forward

    net.edges[pump_id].component.desired_flow_rate = Q_des
    Q_arr, h_arr = solver.compute_system_curve(pump_id, r)
    if Q_arr.size < 2:
        pytest.skip("could not build system curve")
    h_req = float(np.interp(Q_des, Q_arr, h_arr))
    if h_req <= 0:
        pytest.skip("static head supplies flow")

    A_c, B_c, C_c = NetworkSolver.generate_pump_curve(Q_des, h_req)

    # Rebuild same topology with new pump
    net2 = net_factory(seed, with_pump=True)
    pid2 = next(eid for eid, e in net2.edges.items()
                if e.component.__class__.__name__ == "Pump")
    net2.edges[pid2].component.A = A_c
    net2.edges[pid2].component.B = B_c
    net2.edges[pid2].component.C = C_c
    r2 = NetworkSolver(net2).solve()
    assert r2.converged
    Q_solved = r2.flows[pid2]
    return Q_des, Q_solved


@pytest.mark.parametrize("seed", range(5))
def test_roundtrip_random_series(seed):
    Q_des, Q_solved = _round_trip(random_series_network, seed)
    rel = abs(Q_solved - Q_des) / Q_des
    assert rel < 2e-3, \
        f"seed {seed}: Q_des={Q_des:.5f}, Q_solved={Q_solved:.5f}, " \
        f"rel_err={rel*100:.2f}%"


@pytest.mark.parametrize("seed", range(5))
def test_roundtrip_random_parallel(seed):
    Q_des, Q_solved = _round_trip(random_parallel_network, seed)
    rel = abs(Q_solved - Q_des) / Q_des
    assert rel < 2e-3, \
        f"seed {seed}: Q_des={Q_des:.5f}, Q_solved={Q_solved:.5f}, " \
        f"rel_err={rel*100:.2f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Reservoir-only topology (exercises the N==0 short-circuit path)
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoReservoirSinglePipe:
    """
    Edge-case the user is likely to draw: just two reservoirs and one pipe.
    The Q on that pipe is well-defined by Bernoulli:  h_L(Q) = H_A − H_B.
    """

    def test_solver_computes_flow(self):
        net = PipeNetwork()
        net.add_node(Reservoir('RA', total_head=10.0))
        net.add_node(Reservoir('RB', total_head= 0.0))
        net.add_edge(Pipe('P0', diameter=0.05, length=100.0,
                          roughness=4.6e-5), 'RA', 'RB')
        r = NetworkSolver(net).solve()
        assert r.converged
        assert 'P0' in r.flows, \
            "Solver returned no flow on the only pipe (N==0 short-circuit)."
        Q = r.flows['P0']
        # Hand-check via the pipe's own head-loss model
        h_L = net.edges['P0'].component.compute_head_loss(Q)
        assert abs(h_L - 10.0) < 1e-6
