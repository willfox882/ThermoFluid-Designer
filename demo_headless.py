#!/usr/bin/env python3
"""
demo_headless.py
────────────────
ThermoFluid Designer — Physics Engine Demo  (no GUI required)

This script runs the full Newton-Raphson pipe network solver on the
demonstration network and prints a formatted report proving that the
physics engine works correctly.

Requirements:  numpy, scipy   (no PyQt6 needed)
Run with:      python demo_headless.py
"""

import sys
import os
import math

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The report uses Unicode box-drawing characters (═, ✓, →).  On consoles whose
# default encoding cannot represent them — notably Windows cp1252 when stdout is
# piped or redirected (e.g. in CI) — print() would raise UnicodeEncodeError and
# crash the demo.  Reconfigure the streams to UTF-8 so the demo always runs.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass   # stream may not be a reconfigurable TextIOWrapper

from fluid_props import DENSITY, VISCOSITY, GRAVITY
from components import Pipe, Pump, Valve, Junction, Reservoir
from network import PipeNetwork
from solver import NetworkSolver


def build_demo_network() -> PipeNetwork:
    """
    Construct the demonstration network:

        R_sump (H=0 m)
            │
        [Pu1: A=-8000, C=30 m]       ← centrifugal pump
            │
           J1  ──[P1: 200 m, D=0.10 m]──► R_a (H=20 m)   Tank A
            └───[P2: 150 m, D=0.08 m]────► R_b (H=15 m)   Tank B
    """
    net = PipeNetwork()

    # Nodes
    net.add_node(Reservoir("R_sump", total_head=0.0,  name="Sump"))
    net.add_node(Reservoir("R_a",    total_head=20.0, name="Tank A"))
    net.add_node(Reservoir("R_b",    total_head=15.0, name="Tank B"))
    net.add_node(Junction("J1",      elevation=0.0,   demand=0.0))

    # Edges
    net.add_edge(Pump("Pu1", A=-8000.0, B=0.0, C=30.0, diameter=0.1, name="Pump"),
                 from_node_id="R_sump", to_node_id="J1")
    net.add_edge(Pipe("P1",  diameter=0.10, length=200.0, name="Pipe A"),
                 from_node_id="J1", to_node_id="R_a")
    net.add_edge(Pipe("P2",  diameter=0.08, length=150.0, name="Pipe B"),
                 from_node_id="J1", to_node_id="R_b")

    return net


def run_demo():
    """Solve the demo network and print a formatted report."""
    W  = 52  # report width

    print()
    print("═" * W)
    print(" ThermoFluid Designer — Physics Demo".center(W))
    print("═" * W)
    print()
    print(" Network: Pump → Junction → 2 delivery tanks")
    print()
    print(" Topology:")
    print("   R_sump(H=0) → [Pu1] → J1 → [P1: 200m, D=100mm] → R_a(H=20)")
    print("                           └─→ [P2: 150m, D=80mm]  → R_b(H=15)")
    print()

    # Build and validate
    net = build_demo_network()
    errors = net.validate()
    if errors:
        print(" ✗ Network validation FAILED:")
        for e in errors:
            print(f"   • {e}")
        return False

    print(" ✓ Network valid — solving with Newton-Raphson...")
    print()

    # Solve
    solver = NetworkSolver(net)
    result = solver.solve(tol=1e-9, max_iter=200)

    if not result.converged:
        print(f" ✗ Solver did NOT converge (residual={result.residual_norm:.2e})")
        return False

    # ── Node heads ────────────────────────────────────────────────────────
    print(" Node heads:")
    for nid, node in net.nodes.items():
        H    = result.heads[nid]
        comp = node.component
        label = ""
        if isinstance(comp, Reservoir):
            label = f"({comp.name})" if comp.name != nid else ""
        print(f"   {nid:<10s} H = {H:7.3f} m  {label}")
    print()

    # ── Edge results ──────────────────────────────────────────────────────
    print(" Edge results:")
    for eid, edge in net.edges.items():
        comp = edge.component
        Q    = result.flows[eid]
        Q_Ls = Q * 1000.0

        if isinstance(comp, Pump):
            hp = comp.compute_pump_head(Q)
            print(f"   {eid:<5s} Q = {Q_Ls:5.2f} L/s  hp = {hp:.2f} m  [PUMP]")
        else:
            V   = result.velocities[eid]
            Re  = result.reynolds[eid]
            f   = result.friction_factors[eid]
            print(f"   {eid:<5s} Q = {Q_Ls:5.2f} L/s  "
                  f"V = {V:.2f} m/s  Re = {Re:.0f}  f = {f:.5f}")
    print()

    # ── Mass balance check at J1 ──────────────────────────────────────────
    Q_pu1 = result.flows["Pu1"]
    Q_p1  = result.flows["P1"]
    Q_p2  = result.flows["P2"]
    mass_err = Q_pu1 - Q_p1 - Q_p2   # inflow - outflows at J1
    mass_err_nLs = mass_err * 1e9     # convert to nL/s

    # ── Energy balance check ──────────────────────────────────────────────
    # Along each path from sump to delivery tank, energy must balance:
    #   H_sump + hp_pump - h_L_pipe = H_tank
    # Path 1: sump → pump → J1 → P1 → R_a
    H_sump = result.heads["R_sump"]
    H_J1   = result.heads["J1"]
    H_Ra   = result.heads["R_a"]
    hp_pu1 = net.edges["Pu1"].component.compute_pump_head(Q_pu1)
    hL_p1  = net.edges["P1"].component.compute_head_loss(Q_p1)

    energy_err_1 = (H_sump + hp_pu1) - H_J1  # pump lifts to J1
    energy_err_2 = H_J1 - H_Ra - hL_p1       # J1 → R_a via P1
    energy_err = max(abs(energy_err_1), abs(energy_err_2))

    # ── Print verification ────────────────────────────────────────────────
    mass_ok   = abs(mass_err) < 1e-8
    energy_ok = energy_err < 1e-6
    solver_ok = result.residual_norm < 1e-8

    print(f" Mass balance at J1:  {mass_err_nLs:.2f} nL/s error "
          f"{'✓' if mass_ok else '✗'}")
    print(f" Energy balance:      exact to {energy_err:.1e} m "
          f"{'✓' if energy_ok else '✗'}")
    print(f" Solver residual:     {result.residual_norm:.2e} "
          f"{'✓' if solver_ok else '✗'}")
    print()

    all_pass = mass_ok and energy_ok and solver_ok
    if all_pass:
        print(" ══════════════════════════════════════════")
        print("  ALL CHECKS PASSED — Physics engine OK ✓")
        print(" ══════════════════════════════════════════")
    else:
        print(" ══════════════════════════════════════════")
        print("  SOME CHECKS FAILED — see above          ")
        print(" ══════════════════════════════════════════")
    print()

    return all_pass


def test_parallel_two_inlet_two_outlet() -> bool:
    """
    TASK A: Parallel system with 2 inlets feeding through a tee to 2 outlets.

        R_source1 (H=30m) ──[P1]──► J1
        R_source2 (H=25m) ──[P2]──► J1
                               J1 ──[P3]──► J2
                               J2 ──[P4]──► R_sink1 (H=5m)
                               J2 ──[P5]──► R_sink2 (H=10m)

    Checks:
      • Solver converges
      • Mass balance at J1: Q_P1 + Q_P2 = Q_P3
      • Mass balance at J2: Q_P3 = Q_P4 + Q_P5
      • Energy balance on every path source→sink
      • Flow directions physically correct (high→low head)
    """
    W = 56
    print()
    print("═" * W)
    print(" Parallel Network — 2 Inlets, 2 Outlets".center(W))
    print("═" * W)
    print()
    print(" Topology:")
    print("   Rs1(H=30) ──[P1]──► J1 ──[P3]──► J2 ──[P4]──► Rk1(H=5)")
    print("   Rs2(H=25) ──[P2]──►    ◄         └──[P5]──► Rk2(H=10)")
    print()

    net = PipeNetwork()

    net.add_node(Reservoir("Rs1", total_head=30.0, name="Source 1"))
    net.add_node(Reservoir("Rs2", total_head=25.0, name="Source 2"))
    net.add_node(Reservoir("Rk1", total_head=5.0,  name="Sink 1"))
    net.add_node(Reservoir("Rk2", total_head=10.0, name="Sink 2"))
    net.add_node(Junction("J1",   elevation=0.0,   demand=0.0))
    net.add_node(Junction("J2",   elevation=0.0,   demand=0.0))

    net.add_edge(Pipe("P1", diameter=0.10, length=100.0, name="Inlet 1"),
                 from_node_id="Rs1", to_node_id="J1")
    net.add_edge(Pipe("P2", diameter=0.10, length=100.0, name="Inlet 2"),
                 from_node_id="Rs2", to_node_id="J1")
    net.add_edge(Pipe("P3", diameter=0.12, length=200.0, name="Main"),
                 from_node_id="J1",  to_node_id="J2")
    net.add_edge(Pipe("P4", diameter=0.10, length=100.0, name="Outlet 1"),
                 from_node_id="J2",  to_node_id="Rk1")
    net.add_edge(Pipe("P5", diameter=0.08, length=100.0, name="Outlet 2"),
                 from_node_id="J2",  to_node_id="Rk2")

    errors = net.validate()
    if errors:
        print(" ✗ Validation FAILED:")
        for e in errors:
            print(f"   • {e}")
        return False

    print(" ✓ Network valid — solving with Newton-Raphson...")
    print()

    solver = NetworkSolver(net)
    result = solver.solve(tol=1e-9, max_iter=200)

    if not result.converged:
        print(f" ✗ Solver did NOT converge  (residual = {result.residual_norm:.2e})")
        return False

    print(f" ✓ Converged  ({result.iterations} function evaluations)")
    print(f"   Residual norm: {result.residual_norm:.2e}")
    print()

    # ── Print node heads ──────────────────────────────────────────────
    print(" Node heads:")
    for nid, H in result.heads.items():
        print(f"   {nid:<8s}  H = {H:9.4f} m")
    print()

    # ── Print edge flows ──────────────────────────────────────────────
    print(" Edge flows:")
    for eid, Q in result.flows.items():
        V  = result.velocities.get(eid, 0.0)
        Re = result.reynolds.get(eid, 0.0)
        print(f"   {eid:<4s}  Q = {Q*1000:8.4f} L/s  V = {V:.3f} m/s  Re = {Re:,.0f}")
    print()

    Q_P1 = result.flows["P1"]
    Q_P2 = result.flows["P2"]
    Q_P3 = result.flows["P3"]
    Q_P4 = result.flows["P4"]
    Q_P5 = result.flows["P5"]

    H = result.heads
    hL = {eid: net.edges[eid].component.compute_head_loss(result.flows[eid])
          for eid in net.edges}

    # ── Mass balances ─────────────────────────────────────────────────
    mass_err_J1 = (Q_P1 + Q_P2) - Q_P3      # both inlets → J1, one outlet P3
    mass_err_J2 = Q_P3 - (Q_P4 + Q_P5)      # one inlet P3, two outlets
    mass_ok_J1  = abs(mass_err_J1) < 1e-8
    mass_ok_J2  = abs(mass_err_J2) < 1e-8

    print(f" Mass balance at J1: {mass_err_J1*1e9:+.3f} nL/s "
          f"{'✓' if mass_ok_J1 else '✗'}")
    print(f" Mass balance at J2: {mass_err_J2*1e9:+.3f} nL/s "
          f"{'✓' if mass_ok_J2 else '✗'}")

    # ── Energy balances on every path ─────────────────────────────────
    # Each edge: H_from − H_to − h_L = 0
    energy_errors = {
        "P1 (Rs1→J1)":  abs(H["Rs1"] - H["J1"] - hL["P1"]),
        "P2 (Rs2→J1)":  abs(H["Rs2"] - H["J1"] - hL["P2"]),
        "P3 (J1→J2)":   abs(H["J1"]  - H["J2"] - hL["P3"]),
        "P4 (J2→Rk1)":  abs(H["J2"]  - H["Rk1"] - hL["P4"]),
        "P5 (J2→Rk2)":  abs(H["J2"]  - H["Rk2"] - hL["P5"]),
    }
    energy_max = max(energy_errors.values())
    energy_ok  = energy_max < 1e-6
    print(f" Energy balance (all paths): max error = {energy_max:.2e} m "
          f"{'✓' if energy_ok else '✗'}")

    # ── Flow direction check (flow from high to low head) ─────────────
    dir_ok = all([
        Q_P1 > 0,              # Rs1(H=30) → J1 (lower head)
        Q_P2 > 0,              # Rs2(H=25) → J1
        Q_P3 > 0,              # J1 → J2 → sinks (lower head)
        Q_P4 > 0,              # J2 → Rk1(H=5)
        Q_P5 > 0,              # J2 → Rk2(H=10)
        H["J1"] > H["J2"],     # head drops along main path
        H["J2"] > H["Rk2"],    # J2 still above higher sink
    ])
    print(f" Flow directions physically correct: {'✓' if dir_ok else '✗'}")
    print(f"   J1 head = {H['J1']:.4f} m  |  J2 head = {H['J2']:.4f} m")

    solver_ok = result.residual_norm < 1e-8
    print(f" Solver residual: {result.residual_norm:.2e} "
          f"{'✓' if solver_ok else '✗'}")
    print()

    all_pass = mass_ok_J1 and mass_ok_J2 and energy_ok and dir_ok and solver_ok
    if all_pass:
        print(" ════════════════════════════════════════════════════")
        print("  ALL CHECKS PASSED — Parallel network OK ✓")
        print(" ════════════════════════════════════════════════════")
    else:
        print(" ════════════════════════════════════════════════════")
        print("  SOME CHECKS FAILED — see above                    ")
        print(" ════════════════════════════════════════════════════")
    print()

    return all_pass


if __name__ == "__main__":
    ok1 = run_demo()
    ok2 = test_parallel_two_inlet_two_outlet()
    sys.exit(0 if (ok1 and ok2) else 1)
