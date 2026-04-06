"""
network.py
----------
Directed graph representation of a thermofluid pipe network.

Graph model
───────────
  Nodes   →  Junction, Reservoir
  Edges   →  Pipe, Pump, Valve

Storage
───────
  self.nodes : dict[node_id, NetworkNode]
  self.edges : dict[edge_id, NetworkEdge]

Solver traversal
────────────────
  build_incidence_matrix() returns the N_free × N_edge signed incidence matrix A
  where A[i,j] = +1  if edge j flows INTO free node i
               = −1  if edge j flows OUT of free node i
               =  0  otherwise

Boundary conditions
───────────────────
  Reservoir nodes carry a fixed piezometric head.
  They do NOT appear in the unknown vector — their head is substituted
  directly into the energy residual for edges that touch them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from components import (
    FluidComponent, Junction, Reservoir, Pipe, Pump, Valve, Fitting,
    component_from_dict, EDGE_COMPONENT_TYPES, NODE_COMPONENT_TYPES,
)


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class NetworkNode:
    """A node in the pipe network (Junction or Reservoir)."""
    node_id:   str
    component: FluidComponent          # Junction | Reservoir
    connected_edge_ids: List[str] = field(default_factory=list)

    def is_reservoir(self) -> bool:
        return isinstance(self.component, Reservoir)

    def is_junction(self) -> bool:
        return isinstance(self.component, Junction)


@dataclass
class NetworkEdge:
    """A directed edge in the pipe network (Pipe, Pump, or Valve)."""
    edge_id:      str
    component:    FluidComponent       # Pipe | Pump | Valve
    from_node_id: str
    to_node_id:   str
    flow_rate: float = 0.0            # m³/s — positive = from→to; set by solver


# ── Main network class ─────────────────────────────────────────────────────────

class PipeNetwork:
    """
    Directed graph of a thermofluid pipe network.

    Typical usage:
        net = PipeNetwork()

        r1 = Reservoir("R1", total_head=20.0)
        j1 = Junction("J1")
        p1 = Pipe("P1", diameter=0.1, length=500.0)

        net.add_node(r1)
        net.add_node(j1)
        net.add_edge(p1, from_node_id="R1", to_node_id="J1")

        errors = net.validate()
        # → solver.solve(net)
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, NetworkNode] = {}
        self.edges: Dict[str, NetworkEdge] = {}
        # Canvas positions (UI layer; stored here so save/load is unified)
        self.canvas_positions: Dict[str, Tuple[float, float]] = {}

    # ── Construction ──────────────────────────────────────────────────────────

    def add_node(self, component: FluidComponent,
                 canvas_x: float = 0.0, canvas_y: float = 0.0) -> str:
        """Register a Junction or Reservoir as a network node."""
        if component.id in self.nodes:
            raise ValueError(f"Node '{component.id}' already exists in network.")
        node = NetworkNode(node_id=component.id, component=component)
        self.nodes[component.id] = node
        self.canvas_positions[component.id] = (canvas_x, canvas_y)
        return component.id

    def add_edge(self, component: FluidComponent,
                 from_node_id: str, to_node_id: str,
                 canvas_mid_x: float = 0.0, canvas_mid_y: float = 0.0) -> str:
        """Register a Pipe/Pump/Valve as a directed network edge."""
        if component.id in self.edges:
            raise ValueError(f"Edge '{component.id}' already exists in network.")
        if from_node_id not in self.nodes:
            raise KeyError(f"from_node '{from_node_id}' not in network.")
        if to_node_id not in self.nodes:
            raise KeyError(f"to_node '{to_node_id}' not in network.")

        edge = NetworkEdge(
            edge_id      = component.id,
            component    = component,
            from_node_id = from_node_id,
            to_node_id   = to_node_id,
        )
        self.edges[component.id] = edge
        self.nodes[from_node_id].connected_edge_ids.append(component.id)
        self.nodes[to_node_id].connected_edge_ids.append(component.id)
        return component.id

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all connected edges."""
        if node_id not in self.nodes:
            return
        # remove all edges that touch this node
        connected = list(self.nodes[node_id].connected_edge_ids)
        for eid in connected:
            self.remove_edge(eid)
        del self.nodes[node_id]
        self.canvas_positions.pop(node_id, None)

    def remove_edge(self, edge_id: str) -> None:
        """Remove an edge and update node connection lists."""
        if edge_id not in self.edges:
            return
        e = self.edges[edge_id]
        for nid in (e.from_node_id, e.to_node_id):
            if nid in self.nodes:
                try:
                    self.nodes[nid].connected_edge_ids.remove(edge_id)
                except ValueError:
                    pass
        del self.edges[edge_id]

    def clear(self) -> None:
        """Remove everything."""
        self.nodes.clear()
        self.edges.clear()
        self.canvas_positions.clear()

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_free_nodes(self) -> List[NetworkNode]:
        """Nodes whose head is UNKNOWN (Junctions)."""
        return [n for n in self.nodes.values() if n.is_junction()]

    def get_reservoir_nodes(self) -> List[NetworkNode]:
        """Nodes whose head is FIXED (Reservoirs)."""
        return [n for n in self.nodes.values() if n.is_reservoir()]

    def get_edges_list(self) -> List[NetworkEdge]:
        return list(self.edges.values())

    def get_pumps(self) -> List[NetworkEdge]:
        return [e for e in self.edges.values() if isinstance(e.component, Pump)]

    def node_head(self, node_id: str) -> float:
        """Current piezometric head at any node (free or fixed)."""
        return self.nodes[node_id].component.head

    # ── Incidence matrix ─────────────────────────────────────────────────────

    def build_incidence_matrix(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Construct the signed incidence matrix A  (N_free × N_edges).

            A[i, j] = +1   edge j flows INTO free node i
            A[i, j] = −1   edge j flows OUT of free node i
            A[i, j] =  0   otherwise

        Returns
        -------
        A               ndarray shape (N, P)
        free_node_ids   list of N free-node IDs   (row order)
        edge_ids        list of P edge IDs         (column order)
        """
        free_nodes = self.get_free_nodes()
        edges      = self.get_edges_list()

        free_node_ids = [n.node_id for n in free_nodes]
        edge_ids      = [e.edge_id for e in edges]

        N = len(free_node_ids)
        P = len(edge_ids)

        A        = np.zeros((N, P), dtype=float)
        node_idx = {nid: i for i, nid in enumerate(free_node_ids)}

        for j, edge in enumerate(edges):
            if edge.to_node_id in node_idx:
                A[node_idx[edge.to_node_id], j] = +1.0
            if edge.from_node_id in node_idx:
                A[node_idx[edge.from_node_id], j] = -1.0

        return A, free_node_ids, edge_ids

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """
        Check network topology and component parameters.
        Returns a list of error strings (empty = valid).
        """
        errs: List[str] = []

        # Topology checks
        if len(self.nodes) < 2:
            errs.append("Network needs at least 2 nodes.")
        if len(self.edges) == 0:
            errs.append("Network has no edges (pipes/pumps/valves).")
        if len(self.get_reservoir_nodes()) == 0:
            errs.append("Network needs at least one Reservoir (fixed-head boundary).")

        # Edge connectivity
        for eid, edge in self.edges.items():
            if edge.from_node_id not in self.nodes:
                errs.append(f"Edge '{eid}': from-node '{edge.from_node_id}' not found.")
            if edge.to_node_id not in self.nodes:
                errs.append(f"Edge '{eid}': to-node '{edge.to_node_id}' not found.")
            if edge.from_node_id == edge.to_node_id:
                errs.append(f"Edge '{eid}': self-loop (from == to).")

        # Component parameter validation
        for node in self.nodes.values():
            errs.extend(node.component.validate())
        for edge in self.edges.values():
            errs.extend(edge.component.validate())

        # Connectivity: every free node must be reachable from at least one reservoir
        #   (simple check: all nodes in one connected component)
        if len(self.nodes) > 0 and len(self.edges) > 0:
            adj: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
            for e in self.edges.values():
                adj[e.from_node_id].append(e.to_node_id)
                adj[e.to_node_id].append(e.from_node_id)

            start = next(iter(self.nodes))
            visited: set = {start}
            queue = [start]
            while queue:
                cur = queue.pop()
                for nb in adj[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)

            if len(visited) != len(self.nodes):
                disconnected = set(self.nodes) - visited
                errs.append(f"Disconnected nodes: {disconnected}. "
                            f"All nodes must be connected.")

        return errs

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        nodes_list = []
        for nid, node in self.nodes.items():
            d = node.component.to_dict()
            x, y = self.canvas_positions.get(nid, (0.0, 0.0))
            d["canvas_x"] = x
            d["canvas_y"] = y
            nodes_list.append(d)

        edges_list = []
        for eid, edge in self.edges.items():
            d = edge.component.to_dict()
            d["from_node"] = edge.from_node_id
            d["to_node"]   = edge.to_node_id
            edges_list.append(d)

        return {
            "version": "1.0",
            "nodes": nodes_list,
            "edges": edges_list,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PipeNetwork":
        """Reconstruct a PipeNetwork from a serialised dict."""
        net = cls()

        for nd in data.get("nodes", []):
            comp = component_from_dict(nd)
            cx   = nd.get("canvas_x", 0.0)
            cy   = nd.get("canvas_y", 0.0)
            net.add_node(comp, canvas_x=cx, canvas_y=cy)

        for ed in data.get("edges", []):
            comp      = component_from_dict(ed)
            from_node = ed["from_node"]
            to_node   = ed["to_node"]
            net.add_edge(comp, from_node_id=from_node, to_node_id=to_node)

        return net

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "PipeNetwork":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"PipeNetwork — {len(self.nodes)} nodes, {len(self.edges)} edges",
            f"  Reservoirs : {len(self.get_reservoir_nodes())}",
            f"  Junctions  : {len(self.get_free_nodes())}",
            f"  Pumps      : {len(self.get_pumps())}",
        ]
        return "\n".join(lines)
