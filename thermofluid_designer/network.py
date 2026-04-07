"""
network.py
----------
Directed graph representation of a thermofluid pipe network.

Graph model
───────────
  Nodes   →  Junction, Reservoir
            (+ hidden phantom Junctions that serve as pump/valve ports)
  Edges   →  Pipe, Pump, Valve

Inline components (pumps / valves placed freely on the canvas)
──────────────────────────────────────────────────────────────
  Each freely-placed Pump or Valve is stored as a regular edge, but its
  two endpoint nodes are "phantom" Junctions that exist only to give the
  solver anchor points.  The UI renders them as port circles on the
  pump/valve icon rather than as standalone junction icons.

  phantom_nodes : Dict[phantom_node_id, edge_id]
  inline_positions : Dict[edge_id, (canvas_x, canvas_y)]   ← centre of icon

Serialisation
─────────────
  Version "2.0" files contain an "inline_components" list that stores
  pump/valve icon position and their phantom-node IDs.
  Version "1.0" files are loaded with automatic migration:
  - Fitting edges are converted to FittingAttachment children of the
    first available pipe.
  - Pump/Valve edges remain as regular edges (old-style rendering).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from components import (
    FluidComponent, Junction, Reservoir, Pipe, Pump, Valve, Fitting,
    FittingAttachment,
    component_from_dict, EDGE_COMPONENT_TYPES, NODE_COMPONENT_TYPES,
)


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class NetworkNode:
    node_id:   str
    component: FluidComponent
    connected_edge_ids: List[str] = field(default_factory=list)

    def is_reservoir(self) -> bool:
        return isinstance(self.component, Reservoir)

    def is_junction(self) -> bool:
        return isinstance(self.component, Junction)


@dataclass
class NetworkEdge:
    edge_id:      str
    component:    FluidComponent
    from_node_id: str
    to_node_id:   str
    flow_rate: float = 0.0


# ── Main network class ─────────────────────────────────────────────────────────

class PipeNetwork:

    def __init__(self) -> None:
        self.nodes:   Dict[str, NetworkNode] = {}
        self.edges:   Dict[str, NetworkEdge] = {}
        self.canvas_positions: Dict[str, Tuple[float, float]] = {}

        # Phantom node tracking (populated by add_inline_component)
        # phantom_nodes[phantom_node_id] = pump_or_valve_edge_id
        self.phantom_nodes:   Dict[str, str]                  = {}
        # inline_positions[edge_id] = (canvas_x, canvas_y) of the icon centre
        self.inline_positions: Dict[str, Tuple[float, float]] = {}

    # ── Construction ──────────────────────────────────────────────────────────

    def add_node(self, component: FluidComponent,
                 canvas_x: float = 0.0, canvas_y: float = 0.0) -> str:
        if component.id in self.nodes:
            raise ValueError(f"Node '{component.id}' already exists in network.")
        self.nodes[component.id] = NetworkNode(node_id=component.id, component=component)
        self.canvas_positions[component.id] = (canvas_x, canvas_y)
        return component.id

    def add_edge(self, component: FluidComponent,
                 from_node_id: str, to_node_id: str,
                 canvas_mid_x: float = 0.0, canvas_mid_y: float = 0.0) -> str:
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

    def add_inline_component(self, component: FluidComponent,
                              canvas_x: float, canvas_y: float,
                              port_offset: float = 40.0
                              ) -> Tuple[str, str, str]:
        """
        Place a Pump or Valve freely on the canvas.

        Creates two phantom Junction nodes (inlet / outlet) at ±port_offset
        from (canvas_x, canvas_y), registers them as phantom, and adds the
        component as a network edge between them.

        Returns (edge_id, inlet_node_id, outlet_node_id).
        """
        edge_id   = component.id
        inlet_id  = f"{edge_id}_in"
        outlet_id = f"{edge_id}_out"

        inlet_junc  = Junction(inlet_id,  name=f"{component.name} Inlet")
        outlet_junc = Junction(outlet_id, name=f"{component.name} Outlet")

        self.add_node(inlet_junc,  canvas_x - port_offset, canvas_y)
        self.add_node(outlet_junc, canvas_x + port_offset, canvas_y)
        self.add_edge(component, from_node_id=inlet_id, to_node_id=outlet_id)

        self.phantom_nodes[inlet_id]  = edge_id
        self.phantom_nodes[outlet_id] = edge_id
        self.inline_positions[edge_id] = (canvas_x, canvas_y)

        return edge_id, inlet_id, outlet_id

    # ── Removal ───────────────────────────────────────────────────────────────

    def remove_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            return
        connected = list(self.nodes[node_id].connected_edge_ids)
        for eid in connected:
            self.remove_edge(eid)
        del self.nodes[node_id]
        self.canvas_positions.pop(node_id, None)
        self.phantom_nodes.pop(node_id, None)

    def remove_edge(self, edge_id: str) -> None:
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
        self.inline_positions.pop(edge_id, None)

    def remove_inline_component(self, edge_id: str) -> List[str]:
        """
        Remove an inline pump/valve and its phantom nodes.
        Returns list of all edge IDs that were also removed (pipes attached
        to the phantom nodes).
        """
        if edge_id not in self.edges:
            return []
        edge = self.edges[edge_id]
        fn, tn = edge.from_node_id, edge.to_node_id

        # Collect all edges that will vanish with the phantom nodes
        removed_edges: List[str] = []
        for nid in (fn, tn):
            if nid in self.nodes:
                removed_edges.extend(self.nodes[nid].connected_edge_ids)
        removed_edges = list(dict.fromkeys(removed_edges))  # deduplicate

        # Remove phantom nodes (this cascades to connected edges)
        self.remove_node(fn)
        if tn in self.nodes:
            self.remove_node(tn)

        return removed_edges

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.canvas_positions.clear()
        self.phantom_nodes.clear()
        self.inline_positions.clear()

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_free_nodes(self) -> List[NetworkNode]:
        return [n for n in self.nodes.values() if n.is_junction()]

    def get_reservoir_nodes(self) -> List[NetworkNode]:
        return [n for n in self.nodes.values() if n.is_reservoir()]

    def get_edges_list(self) -> List[NetworkEdge]:
        return list(self.edges.values())

    def get_pumps(self) -> List[NetworkEdge]:
        return [e for e in self.edges.values() if isinstance(e.component, Pump)]

    def node_head(self, node_id: str) -> float:
        return self.nodes[node_id].component.head

    def is_phantom(self, node_id: str) -> bool:
        return node_id in self.phantom_nodes

    # ── Incidence matrix ─────────────────────────────────────────────────────

    def build_incidence_matrix(self) -> Tuple[np.ndarray, List[str], List[str]]:
        free_nodes    = self.get_free_nodes()
        edges         = self.get_edges_list()
        free_node_ids = [n.node_id for n in free_nodes]
        edge_ids      = [e.edge_id for e in edges]

        N = len(free_node_ids)
        P = len(edge_ids)
        A = np.zeros((N, P), dtype=float)
        node_idx = {nid: i for i, nid in enumerate(free_node_ids)}

        for j, edge in enumerate(edges):
            if edge.to_node_id in node_idx:
                A[node_idx[edge.to_node_id], j] = +1.0
            if edge.from_node_id in node_idx:
                A[node_idx[edge.from_node_id], j] = -1.0

        return A, free_node_ids, edge_ids

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        errs: List[str] = []

        # Count non-phantom nodes
        real_nodes = {nid: n for nid, n in self.nodes.items()
                      if nid not in self.phantom_nodes}

        if len(real_nodes) < 2:
            errs.append("Network needs at least 2 nodes.")
        if len(self.edges) == 0:
            errs.append("Network has no edges (pipes/pumps/valves).")
        if not any(n.is_reservoir() for n in real_nodes.values()):
            errs.append("Network needs at least one Reservoir (fixed-head boundary).")

        # Edge connectivity
        for eid, edge in self.edges.items():
            if edge.from_node_id not in self.nodes:
                errs.append(f"Edge '{eid}': from-node '{edge.from_node_id}' not found.")
            if edge.to_node_id not in self.nodes:
                errs.append(f"Edge '{eid}': to-node '{edge.to_node_id}' not found.")
            if edge.from_node_id == edge.to_node_id:
                errs.append(f"Edge '{eid}': self-loop (from == to).")

        # Component validation
        for node in self.nodes.values():
            errs.extend(node.component.validate())
        for edge in self.edges.values():
            errs.extend(edge.component.validate())

        # Inline pump/valve connectivity check
        for eid, edge in self.edges.items():
            if isinstance(edge.component, (Pump, Valve)) and eid in self.inline_positions:
                fn = edge.from_node_id  # inlet phantom
                tn = edge.to_node_id    # outlet phantom
                fn_conns = len(self.nodes[fn].connected_edge_ids) if fn in self.nodes else 0
                tn_conns = len(self.nodes[tn].connected_edge_ids) if tn in self.nodes else 0
                # Each phantom node should have the pump edge + at least 1 pipe
                if fn_conns < 2:
                    errs.append(
                        f"Error: {type(edge.component).__name__} '{edge.component.name}' "
                        f"inlet is not connected to any pipe.")
                if tn_conns < 2:
                    errs.append(
                        f"Error: {type(edge.component).__name__} '{edge.component.name}' "
                        f"outlet is not connected to any pipe.")

        # Network connectivity (BFS over all nodes)
        if len(self.nodes) > 0 and len(self.edges) > 0:
            adj: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
            for e in self.edges.values():
                adj[e.from_node_id].append(e.to_node_id)
                adj[e.to_node_id].append(e.from_node_id)

            start   = next(iter(self.nodes))
            visited: set = {start}
            queue   = [start]
            while queue:
                cur = queue.pop()
                for nb in adj[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)

            if len(visited) != len(self.nodes):
                disconnected = {nid for nid in self.nodes
                                if nid not in visited and nid not in self.phantom_nodes}
                if disconnected:
                    errs.append(f"Disconnected nodes: {disconnected}. "
                                f"All nodes must be connected.")

        return errs

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        # Regular (non-phantom) nodes
        nodes_list = []
        for nid, node in self.nodes.items():
            if nid in self.phantom_nodes:
                continue
            d = node.component.to_dict()
            x, y = self.canvas_positions.get(nid, (0.0, 0.0))
            d["canvas_x"] = x
            d["canvas_y"] = y
            nodes_list.append(d)

        # Regular (non-inline) edges (pipes only now; pump/valve in inline_components)
        edges_list = []
        for eid, edge in self.edges.items():
            if eid in self.inline_positions:
                continue  # saved in inline_components
            d = edge.component.to_dict()
            d["from_node"] = edge.from_node_id
            d["to_node"]   = edge.to_node_id
            edges_list.append(d)

        # Inline components (pumps / valves)
        inline_list = []
        for eid, (cx, cy) in self.inline_positions.items():
            if eid not in self.edges:
                continue
            edge = self.edges[eid]
            d = edge.component.to_dict()
            d["from_node"]  = edge.from_node_id
            d["to_node"]    = edge.to_node_id
            d["canvas_x"]   = cx
            d["canvas_y"]   = cy
            inline_list.append(d)

        return {
            "version":           "2.0",
            "nodes":             nodes_list,
            "edges":             edges_list,
            "inline_components": inline_list,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PipeNetwork":
        net     = cls()
        version = data.get("version", "1.0")

        # ── Nodes ────────────────────────────────────────────────────────────
        for nd in data.get("nodes", []):
            comp = component_from_dict(nd)
            net.add_node(comp, canvas_x=nd.get("canvas_x", 0.0),
                         canvas_y=nd.get("canvas_y", 0.0))

        # ── Inline components (v2.0) — must be before regular edges so phantom
        #    nodes exist when pipes reference them ────────────────────────────
        for ic in data.get("inline_components", []):
            comp = component_from_dict(ic)
            cx   = ic.get("canvas_x", 0.0)
            cy   = ic.get("canvas_y", 0.0)
            # Reconstruct phantom nodes using stored from/to IDs
            fn   = ic["from_node"]
            tn   = ic["to_node"]

            # Add phantom junctions if they don't already exist
            if fn not in net.nodes:
                net.add_node(Junction(fn, name=f"{comp.name} Inlet"),
                             canvas_x=cx - 40, canvas_y=cy)
            if tn not in net.nodes:
                net.add_node(Junction(tn, name=f"{comp.name} Outlet"),
                             canvas_x=cx + 40, canvas_y=cy)

            net.add_edge(comp, from_node_id=fn, to_node_id=tn)
            net.phantom_nodes[fn]       = comp.id
            net.phantom_nodes[tn]       = comp.id
            net.inline_positions[comp.id] = (cx, cy)

        # ── Regular edges ─────────────────────────────────────────────────
        fitting_edges: List[dict] = []   # collected for migration
        for ed in data.get("edges", []):
            comp = component_from_dict(ed)
            if isinstance(comp, Fitting):
                fitting_edges.append(ed)
                continue   # migrate below
            net.add_edge(comp, from_node_id=ed["from_node"], to_node_id=ed["to_node"])

        # ── Migrate legacy fitting edges onto pipes ────────────────────────
        if fitting_edges:
            # Find the first available pipe edge to attach to
            pipe_ids = [eid for eid, e in net.edges.items()
                        if isinstance(e.component, Pipe)]
            target_pipe_id = pipe_ids[0] if pipe_ids else None

            for i, fed in enumerate(fitting_edges):
                f_comp = Fitting.from_dict(fed)
                t = (i + 1) / (len(fitting_edges) + 1)  # spread along pipe
                fa = f_comp.to_fitting_attachment(pipe_position_t=t)
                if target_pipe_id and isinstance(
                        net.edges[target_pipe_id].component, Pipe):
                    net.edges[target_pipe_id].component.fittings.append(fa)

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
        real_nodes = sum(1 for nid in self.nodes if nid not in self.phantom_nodes)
        lines = [
            f"PipeNetwork — {real_nodes} nodes, {len(self.edges)} edges",
            f"  Reservoirs : {len(self.get_reservoir_nodes())}",
            f"  Junctions  : {len(self.get_free_nodes()) - len(self.phantom_nodes)}",
            f"  Pumps      : {len(self.get_pumps())}",
        ]
        return "\n".join(lines)
