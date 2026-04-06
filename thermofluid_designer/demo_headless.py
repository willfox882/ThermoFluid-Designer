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


if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
