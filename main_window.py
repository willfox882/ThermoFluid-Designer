"""
main_window.py
--------------
QMainWindow wiring together canvas, properties panel, solver, and file I/O.

Key UX changes vs v1.0
───────────────────────
• Pumps and valves are placed freely (click toolbar → click canvas).
  They appear as movable icons with inlet/outlet port circles.
• Fittings are attached to pipes (click toolbar → click a pipe on canvas).
  They are shown as small diamonds along the pipe line.
• Pipes remain simple arrows drawn by clicking source node → target node.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QTabWidget,
    QToolBar, QStatusBar, QLabel, QFileDialog,
    QMessageBox, QDockWidget, QSizePolicy,
    QVBoxLayout, QApplication, QDoubleSpinBox,
)
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSlot
from PyQt6.QtGui import (QAction, QActionGroup, QIcon, QKeySequence, QFont, QColor,
                         QUndoStack, QUndoCommand)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components import (
    Pipe, Pump, Valve, Fitting, Junction, Reservoir, PressurizedSource,
    FittingAttachment, component_from_dict,
)
from network import PipeNetwork
from solver import NetworkSolver, SolverResult
import units
from canvas import (
    ThermofluidCanvas, ThermofluidView, CanvasSignals,
    NodeGraphicsItem, EdgeGraphicsItem, InlineComponentItem, PortItem,
    PipeEdgeItem, FittingIconItem,
)
from sidebar import PropertiesPanel
from plotting_widget import PlottingWidget
from fluid_props import lookup_fitting_k, NOMINAL_TO_METRES


# ── Configuration ─────────────────────────────────────────────────────────────
# Set environment variable LOAD_DEMO_ON_START=1 (or edit this line to True)
# to auto-load the demo network on startup.  Default is False (blank canvas).
LOAD_DEMO_ON_START: bool = os.environ.get("LOAD_DEMO_ON_START", "0") == "1"


# ── Tooltip builder ────────────────────────────────────────────────────────────

def _component_tooltip(comp) -> str:
    if isinstance(comp, PressurizedSource):
        return (f"Pressurized Source: {comp.name}\nH = {comp.total_head:.3f} m\n"
                f"z = {comp.elevation:.3f} m\nP_supply = {comp.surface_pressure_Pa:.0f} Pa")
    elif isinstance(comp, Reservoir):
        return (f"Reservoir: {comp.name}\nH = {comp.total_head:.3f} m\n"
                f"z = {comp.elevation:.3f} m\nP_surface = {comp.surface_pressure_Pa:.0f} Pa")
    elif isinstance(comp, Junction):
        return (f"Junction: {comp.name}\nElevation: {comp.elevation:.3f} m\n"
                f"Demand: {comp.demand*1000:.4f} L/s")
    elif isinstance(comp, Pipe):
        return (f"Pipe: {comp.name}\nD = {comp.diameter*1000:.1f} mm  L = {comp.length:.1f} m\n"
                f"Material: {comp.material} / {comp.condition}\n"
                f"ε = {comp.roughness*1e6:.2f} µm  K = {comp.K_minor:.3f}")
    elif isinstance(comp, Pump):
        state = "Running" if comp.is_on else "Off"
        return (f"Pump: {comp.name}  [{state}]\n"
                f"A = {comp.A:.0f}  B = {comp.B:.3f}  C = {comp.C:.2f} m")
    elif isinstance(comp, Valve):
        state = "Open" if comp.is_open else "Closed"
        return f"Valve: {comp.name}  [{state}]\nK = {comp.K:.3f}"
    elif isinstance(comp, Fitting):
        return (f"Fitting: {comp.name}\n{comp.fitting_subtype}\nK = {comp.K:.4f}")
    return comp.name


# ═══════════════════════════════════════════════════════════════════════════════
# Undo / Redo commands
# ═══════════════════════════════════════════════════════════════════════════════

class AddNodeCommand(QUndoCommand):
    def __init__(self, mw, comp_type: str, comp, x: float, y: float):
        super().__init__(f"Add {comp_type.capitalize()}")
        self._mw = mw; self._comp_type = comp_type
        self._comp = comp; self._x = x; self._y = y

    def redo(self):
        mw = self._mw
        if self._comp.id in mw._network.nodes:
            return
        mw._network.add_node(self._comp, canvas_x=self._x, canvas_y=self._y)
        item = mw._scene.add_node(self._comp.id, self._comp_type, self._x, self._y)
        item.setToolTip(_component_tooltip(self._comp))
        item.set_display_name(self._comp.name)
        if isinstance(self._comp, Reservoir):
            item.set_elevation(self._comp.elevation)
        mw._mark_dirty(); mw._refresh_status()

    def undo(self):
        mw = self._mw
        if self._comp.id not in mw._network.nodes:
            return
        for eid in list(mw._network.nodes[self._comp.id].connected_edge_ids):
            if eid in mw._scene.edge_items:
                mw._scene.removeItem(mw._scene.edge_items.pop(eid))
        mw._network.remove_node(self._comp.id)
        mw._scene.remove_component(self._comp.id)
        mw._sidebar.clear_selection()
        mw._last_result = None; mw._scene.clear_results()
        mw._mark_dirty(); mw._refresh_status()


class AddInlineComponentCommand(QUndoCommand):
    def __init__(self, mw, comp_type: str, comp, x: float, y: float):
        super().__init__(f"Add {comp_type.capitalize()}")
        self._mw = mw; self._comp_type = comp_type
        self._comp = comp; self._x = x; self._y = y
        self._edge_id   = comp.id
        self._inlet_id  = f"{comp.id}_in"
        self._outlet_id = f"{comp.id}_out"

    def redo(self):
        mw = self._mw
        if self._edge_id in mw._network.edges:
            return
        edge_id, inlet_id, outlet_id = mw._network.add_inline_component(
            self._comp, self._x, self._y)
        item = mw._scene.add_inline_component(
            edge_id, self._comp_type, inlet_id, outlet_id, self._x, self._y)
        item.set_display_name(self._comp.name)
        item.setToolTip(_component_tooltip(self._comp))
        mw._mark_dirty(); mw._refresh_status()

    def undo(self):
        mw = self._mw
        if self._edge_id not in mw._network.edges:
            return
        net = mw._network
        edge = net.edges[self._edge_id]
        fn, tn = edge.from_node_id, edge.to_node_id
        doomed = set([self._edge_id])
        for nid in (fn, tn):
            if nid in net.nodes:
                doomed.update(net.nodes[nid].connected_edge_ids)
        net.remove_inline_component(self._edge_id)
        for eid in doomed:
            if eid in mw._scene.edge_items:
                mw._scene.removeItem(mw._scene.edge_items.pop(eid))
        mw._scene.remove_inline_component(self._edge_id)
        mw._sidebar.clear_selection()
        mw._last_result = None; mw._scene.clear_results()
        mw._mark_dirty(); mw._refresh_status()


class AddEdgeCommand(QUndoCommand):
    def __init__(self, mw, comp, from_id: str, to_id: str, edge_type: str):
        super().__init__(f"Add {edge_type.capitalize()}")
        self._mw = mw; self._comp = comp
        self._from_id = from_id; self._to_id = to_id
        self._edge_type = edge_type

    def redo(self):
        mw = self._mw
        if self._comp.id in mw._network.edges:
            return
        if (self._from_id not in mw._network.nodes
                or self._to_id not in mw._network.nodes):
            return
        try:
            mw._network.add_edge(self._comp,
                                  from_node_id=self._from_id,
                                  to_node_id=self._to_id)
        except (KeyError, ValueError):
            return
        edge_item = mw._scene.add_edge(
            self._comp.id, self._edge_type, self._from_id, self._to_id)
        if edge_item:
            edge_item.setToolTip(_component_tooltip(self._comp))
            edge_item.set_display_name(self._comp.name)
            if isinstance(self._comp, Pipe) and isinstance(edge_item, PipeEdgeItem):
                edge_item.set_fittings(self._comp.fittings)
        mw._mark_dirty(); mw._refresh_status()
        mw._signals.edge_selected.emit(self._comp.id)

    def undo(self):
        mw = self._mw
        if self._comp.id in mw._scene.edge_items:
            mw._scene.removeItem(mw._scene.edge_items.pop(self._comp.id))
        mw._network.remove_edge(self._comp.id)
        mw._sidebar.clear_selection()
        mw._last_result = None; mw._scene.clear_results()
        mw._mark_dirty(); mw._refresh_status()


class AddFittingCommand(QUndoCommand):
    def __init__(self, mw, pipe_edge_id: str, fa):
        super().__init__("Add Fitting")
        self._mw = mw; self._pipe_edge_id = pipe_edge_id; self._fa = fa

    def redo(self):
        mw = self._mw
        if self._pipe_edge_id not in mw._network.edges:
            return
        pipe_comp = mw._network.edges[self._pipe_edge_id].component
        if not isinstance(pipe_comp, Pipe):
            return
        if not any(f.fitting_id == self._fa.fitting_id for f in pipe_comp.fittings):
            pipe_comp.fittings.append(self._fa)
        pipe_item = mw._scene.edge_items.get(self._pipe_edge_id)
        if isinstance(pipe_item, PipeEdgeItem):
            pipe_item.set_fittings(pipe_comp.fittings)
        mw._sidebar.load_component(pipe_comp)
        mw._right_tabs.setCurrentIndex(0)
        mw._mark_dirty(); mw._refresh_status()

    def undo(self):
        mw = self._mw
        if self._pipe_edge_id not in mw._network.edges:
            return
        pipe_comp = mw._network.edges[self._pipe_edge_id].component
        if not isinstance(pipe_comp, Pipe):
            return
        pipe_comp.fittings = [
            f for f in pipe_comp.fittings if f.fitting_id != self._fa.fitting_id]
        pipe_item = mw._scene.edge_items.get(self._pipe_edge_id)
        if isinstance(pipe_item, PipeEdgeItem):
            pipe_item.set_fittings(pipe_comp.fittings)
        mw._sidebar.clear_selection()
        mw._mark_dirty(); mw._refresh_status()


class DeleteNodeCommand(QUndoCommand):
    def __init__(self, mw, node_id: str):
        super().__init__("Delete Node")
        self._mw = mw; self._node_id = node_id
        net  = mw._network
        node = net.nodes[node_id]
        self._comp_dict = node.component.to_dict()
        self._pos       = net.canvas_positions.get(node_id, (0.0, 0.0))
        self._saved_edges = []
        for eid in list(node.connected_edge_ids):
            if eid in net.edges:
                e = net.edges[eid]
                self._saved_edges.append({
                    "comp": e.component.to_dict(),
                    "from": e.from_node_id,
                    "to":   e.to_node_id,
                })

    def redo(self):
        mw = self._mw
        if self._node_id not in mw._network.nodes:
            return
        for eid in list(mw._network.nodes[self._node_id].connected_edge_ids):
            if eid in mw._scene.edge_items:
                mw._scene.removeItem(mw._scene.edge_items.pop(eid))
        mw._network.remove_node(self._node_id)
        mw._scene.remove_component(self._node_id)
        mw._sidebar.clear_selection()
        mw._last_result = None; mw._scene.clear_results()
        mw._mark_dirty(); mw._refresh_status()

    def undo(self):
        mw = self._mw
        if self._node_id in mw._network.nodes:
            return
        comp  = component_from_dict(self._comp_dict)
        ctype = "reservoir" if isinstance(comp, Reservoir) else "junction"
        mw._network.add_node(comp, canvas_x=self._pos[0], canvas_y=self._pos[1])
        item = mw._scene.add_node(comp.id, ctype, self._pos[0], self._pos[1])
        item.set_display_name(comp.name)
        if isinstance(comp, Reservoir):
            item.set_elevation(comp.elevation)
        for edge_info in self._saved_edges:
            ecomp = component_from_dict(edge_info["comp"])
            fid, tid = edge_info["from"], edge_info["to"]
            if fid in mw._network.nodes and tid in mw._network.nodes:
                if ecomp.id not in mw._network.edges:
                    try:
                        mw._network.add_edge(ecomp, from_node_id=fid, to_node_id=tid)
                        etype = {"Pipe": "pipe", "Pump": "pump", "Valve": "valve"}.get(
                            type(ecomp).__name__, "pipe")
                        edge_item = mw._scene.add_edge(ecomp.id, etype, fid, tid)
                        if isinstance(ecomp, Pipe) and isinstance(edge_item, PipeEdgeItem):
                            edge_item.set_fittings(ecomp.fittings)
                    except (KeyError, ValueError):
                        pass
        mw._last_result = None; mw._scene.clear_results()
        mw._mark_dirty(); mw._refresh_status()


class DeleteEdgeCommand(QUndoCommand):
    def __init__(self, mw, edge_id: str):
        super().__init__("Delete Edge")
        self._mw = mw; self._edge_id = edge_id
        edge = mw._network.edges[edge_id]
        self._comp_dict = edge.component.to_dict()
        self._from_id   = edge.from_node_id
        self._to_id     = edge.to_node_id

    def redo(self):
        mw = self._mw
        if self._edge_id in mw._scene.edge_items:
            mw._scene.removeItem(mw._scene.edge_items.pop(self._edge_id))
        mw._network.remove_edge(self._edge_id)
        mw._sidebar.clear_selection()
        mw._last_result = None; mw._scene.clear_results()
        mw._mark_dirty(); mw._refresh_status()

    def undo(self):
        mw = self._mw
        if self._edge_id in mw._network.edges:
            return
        if (self._from_id not in mw._network.nodes
                or self._to_id not in mw._network.nodes):
            return
        comp = component_from_dict(self._comp_dict)
        try:
            mw._network.add_edge(comp, from_node_id=self._from_id,
                                  to_node_id=self._to_id)
        except (KeyError, ValueError):
            return
        etype = {"Pipe": "pipe", "Pump": "pump", "Valve": "valve"}.get(
            type(comp).__name__, "pipe")
        edge_item = mw._scene.add_edge(comp.id, etype, self._from_id, self._to_id)
        if isinstance(comp, Pipe) and isinstance(edge_item, PipeEdgeItem):
            edge_item.set_fittings(comp.fittings)
        mw._last_result = None; mw._scene.clear_results()
        mw._mark_dirty(); mw._refresh_status()


class DeleteInlineComponentCommand(QUndoCommand):
    def __init__(self, mw, edge_id: str):
        super().__init__("Delete Inline Component")
        self._mw = mw; self._edge_id = edge_id
        net = mw._network; edge = net.edges[edge_id]
        self._comp_dict  = edge.component.to_dict()
        self._x, self._y = net.inline_positions.get(edge_id, (0.0, 0.0))
        self._inlet_id   = edge.from_node_id
        self._outlet_id  = edge.to_node_id
        self._comp_type  = ("pump" if isinstance(edge.component, Pump) else "valve")
        self._saved_pipes = []
        for nid in (self._inlet_id, self._outlet_id):
            if nid in net.nodes:
                for eid in net.nodes[nid].connected_edge_ids:
                    if eid != edge_id and eid in net.edges:
                        e = net.edges[eid]
                        self._saved_pipes.append({
                            "comp": e.component.to_dict(),
                            "from": e.from_node_id,
                            "to":   e.to_node_id,
                        })

    def redo(self):
        mw = self._mw
        if self._edge_id not in mw._network.edges:
            return
        net = mw._network; edge = net.edges[self._edge_id]
        fn, tn = edge.from_node_id, edge.to_node_id
        doomed = set([self._edge_id])
        for nid in (fn, tn):
            if nid in net.nodes:
                doomed.update(net.nodes[nid].connected_edge_ids)
        net.remove_inline_component(self._edge_id)
        for eid in doomed:
            if eid in mw._scene.edge_items:
                mw._scene.removeItem(mw._scene.edge_items.pop(eid))
        mw._scene.remove_inline_component(self._edge_id)
        mw._sidebar.clear_selection()
        mw._last_result = None; mw._scene.clear_results()
        mw._mark_dirty(); mw._refresh_status()

    def undo(self):
        mw = self._mw
        if self._edge_id in mw._network.edges:
            return
        comp = component_from_dict(self._comp_dict)
        mw._network.add_inline_component(comp, self._x, self._y)
        item = mw._scene.add_inline_component(
            self._edge_id, self._comp_type,
            self._inlet_id, self._outlet_id, self._x, self._y)
        item.set_display_name(comp.name)
        item.setToolTip(_component_tooltip(comp))
        for pipe_info in self._saved_pipes:
            ecomp = component_from_dict(pipe_info["comp"])
            fid, tid = pipe_info["from"], pipe_info["to"]
            if (fid in mw._network.nodes and tid in mw._network.nodes
                    and ecomp.id not in mw._network.edges):
                try:
                    mw._network.add_edge(ecomp, from_node_id=fid, to_node_id=tid)
                    edge_item = mw._scene.add_edge(ecomp.id, "pipe", fid, tid)
                    if edge_item and isinstance(ecomp, Pipe):
                        edge_item.set_fittings(ecomp.fittings)
                except (KeyError, ValueError):
                    pass
        mw._last_result = None; mw._scene.clear_results()
        mw._mark_dirty(); mw._refresh_status()


class DeleteFittingCommand(QUndoCommand):
    def __init__(self, mw, pipe_id: str, fa):
        super().__init__("Delete Fitting")
        self._mw = mw; self._pipe_id = pipe_id; self._fa = fa

    def redo(self):
        mw = self._mw
        if self._pipe_id not in mw._network.edges:
            return
        pipe_comp = mw._network.edges[self._pipe_id].component
        if not isinstance(pipe_comp, Pipe):
            return
        pipe_comp.fittings = [
            f for f in pipe_comp.fittings if f.fitting_id != self._fa.fitting_id]
        pipe_item = mw._scene.edge_items.get(self._pipe_id)
        if isinstance(pipe_item, PipeEdgeItem):
            pipe_item.set_fittings(pipe_comp.fittings)
        mw._sidebar.load_component(pipe_comp)
        mw._right_tabs.setCurrentIndex(0)
        mw._mark_dirty(); mw._refresh_status()

    def undo(self):
        mw = self._mw
        if self._pipe_id not in mw._network.edges:
            return
        pipe_comp = mw._network.edges[self._pipe_id].component
        if not isinstance(pipe_comp, Pipe):
            return
        if not any(f.fitting_id == self._fa.fitting_id for f in pipe_comp.fittings):
            pipe_comp.fittings.append(self._fa)
        pipe_item = mw._scene.edge_items.get(self._pipe_id)
        if isinstance(pipe_item, PipeEdgeItem):
            pipe_item.set_fittings(pipe_comp.fittings)
        mw._sidebar.load_component(pipe_comp)
        mw._right_tabs.setCurrentIndex(0)
        mw._mark_dirty(); mw._refresh_status()


class MoveNodeCommand(QUndoCommand):
    def __init__(self, mw, node_id: str,
                 old_x: float, old_y: float, new_x: float, new_y: float):
        super().__init__("Move Node")
        self._mw = mw; self._node_id = node_id
        self._old_x = old_x; self._old_y = old_y
        self._new_x = new_x; self._new_y = new_y

    def redo(self):
        self._apply(self._new_x, self._new_y)

    def undo(self):
        self._apply(self._old_x, self._old_y)

    def _apply(self, x: float, y: float):
        mw = self._mw
        item = mw._scene.node_items.get(self._node_id)
        if item:
            item.setPos(x, y)
            # itemChange → node_moved → _on_node_moved updates canvas_positions


class MoveInlineCommand(QUndoCommand):
    def __init__(self, mw, edge_id: str,
                 old_x: float, old_y: float, new_x: float, new_y: float):
        super().__init__("Move Component")
        self._mw = mw; self._edge_id = edge_id
        self._old_x = old_x; self._old_y = old_y
        self._new_x = new_x; self._new_y = new_y

    def redo(self):
        self._apply(self._new_x, self._new_y)

    def undo(self):
        self._apply(self._old_x, self._old_y)

    def _apply(self, x: float, y: float):
        mw = self._mw
        item = mw._scene.inline_items.get(self._edge_id)
        if item:
            item.setPos(x, y)
            mw._network.inline_positions[self._edge_id] = (x, y)


# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    APP_NAME = "ThermoFluid Designer"

    def __init__(self):
        super().__init__()
        self._network      = PipeNetwork()
        self._solver       = NetworkSolver(self._network)
        self._last_result: Optional[SolverResult] = None
        self._current_file: Optional[str]          = None
        self._dirty        = False

        self._counters = {k: 0 for k in
                          ("reservoir", "junction", "pipe", "pump", "valve", "fitting",
                           "pressurized_source")}
        self._last_system_curves: dict = {}   # populated after each successful solve

        # Pipe connection mode state
        self._pending_edge_type: Optional[str] = None
        self._connect_mode_active = False
        self._conn_step   = 0
        self._conn_from_id: Optional[str] = None

        # Fitting placement state
        self._fitting_mode_active = False
        self._fitting_defaults: dict = {}   # subtype, connection, nominal, K

        self._undo_stack = QUndoStack(self)

        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._connect_signals()

        self.setWindowTitle(self.APP_NAME)
        self.resize(1300, 820)

        if LOAD_DEMO_ON_START:
            QTimer.singleShot(100, self._load_demo_network)
        else:
            QTimer.singleShot(100, self._view.zoom_fit)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        self._signals = CanvasSignals()
        self._scene   = ThermofluidCanvas(self._signals)
        self._view    = ThermofluidView(self._scene)
        splitter.addWidget(self._view)

        self._right_tabs = QTabWidget()
        self._right_tabs.setMinimumWidth(290)
        self._right_tabs.setMaximumWidth(420)
        self._sidebar = PropertiesPanel()
        self._plotter = PlottingWidget()
        self._right_tabs.addTab(self._sidebar, "Properties")
        self._right_tabs.addTab(self._plotter,  "Plots")
        splitter.addWidget(self._right_tabs)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)

        self._status_net    = QLabel("  Network: empty")
        self._status_solver = QLabel("Solver: —  ")
        self._status_net.setFont(QFont("Segoe UI", 8))
        self._status_solver.setFont(QFont("Segoe UI", 8))
        sb = self.statusBar()
        sb.addWidget(self._status_net, stretch=1)
        sb.addPermanentWidget(self._status_solver)

    # ── Menus ─────────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        def _fa(text, shortcut=None, slot=None):
            act = QAction(text, self)
            if shortcut: act.setShortcut(shortcut)
            if slot:     act.triggered.connect(slot)
            file_menu.addAction(act)

        _fa("&New",           QKeySequence.StandardKey.New,  self._on_new)
        _fa("&Open…",         QKeySequence.StandardKey.Open, self._on_open)
        file_menu.addSeparator()
        _fa("&Save",          QKeySequence.StandardKey.Save,    self._on_save)
        _fa("Save &As…",      QKeySequence("Ctrl+Shift+S"),      self._on_save_as)
        file_menu.addSeparator()
        _fa("&Export Results to CSV…", QKeySequence("Ctrl+E"),   self._on_export_csv)
        file_menu.addSeparator()
        _fa("E&xit",          QKeySequence.StandardKey.Quit, self.close)

        view_menu = mb.addMenu("&View")
        act_fit = QAction("Fit to View", self)
        act_fit.setShortcut(QKeySequence("F"))
        act_fit.triggered.connect(self._view.zoom_fit)
        view_menu.addAction(act_fit)
        act_reset = QAction("Reset Zoom", self)
        act_reset.triggered.connect(self._view.zoom_reset)
        view_menu.addAction(act_reset)

        view_menu.addSeparator()
        units_menu = view_menu.addMenu("Result &Units")
        self._units_group = QActionGroup(self)
        self._units_group.setExclusive(True)
        for label, system in (("SI  (m, L/s, kPa)", units.SI),
                              ("Imperial  (ft, gpm, psi)", units.IMPERIAL)):
            act = QAction(label, self, checkable=True)
            act.setChecked(units.get_system() == system)
            act.triggered.connect(lambda _checked, s=system: self._on_set_units(s))
            self._units_group.addAction(act)
            units_menu.addAction(act)

        help_menu = mb.addMenu("&Help")
        act_about = QAction("About…", self)
        act_about.triggered.connect(self._on_about)
        help_menu.addAction(act_about)

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self):
        tb = QToolBar("Main")
        self._toolbar = tb
        tb.setIconSize(QSize(18, 18))
        tb.setMovable(False)
        tb.setStyleSheet(
            "QToolBar { background:#2d3040; spacing:4px; padding:4px; }"
            "QToolBar::separator { background:#555; width:1px; margin:4px 6px; }"
            "QToolButton { color:#ccc; padding:4px 8px; border-radius:3px; font-size:11px; }"
            "QToolButton:hover  { background:#404560; color:white; }"
            "QToolButton:checked{ background:#3a7bd5; color:white; }"
        )
        self.addToolBar(tb)

        def btn(text, tip, slot=None, checkable=False):
            act = QAction(text, self)
            act.setToolTip(tip)
            act.setCheckable(checkable)
            if slot: act.triggered.connect(slot)
            tb.addAction(act)
            return act

        # Node placement
        btn("⬡ Reservoir",        "Place reservoir (fixed head)",          self._place_reservoir)
        btn("⬡ Press. Source",    "Place pressurized source (fixed head)", self._place_pressurized_source)
        btn("● Junction",         "Place junction node",                   self._place_junction)
        tb.addSeparator()

        # Pipe: click-to-connect between two nodes/ports
        self._act_pipe = btn("━ Pipe",
            "Click source node/port then target node/port to draw a pipe",
            self._connect_pipe, checkable=True)

        # Pump/Valve: click canvas to place freely
        self._act_pump  = btn("⊛ Pump",
            "Click canvas to place a pump (then connect pipes to its ports)",
            self._place_pump_mode, checkable=True)
        self._act_valve = btn("⊠ Valve",
            "Click canvas to place a valve (then connect pipes to its ports)",
            self._place_valve_mode, checkable=True)

        # Fitting: click a pipe on canvas to attach
        self._act_fitting = btn("◇ Fitting",
            "Click a pipe on canvas to attach a fitting to it",
            self._fitting_attach_mode, checkable=True)

        self._connect_actions = [self._act_pipe]
        self._placement_actions = [self._act_pump, self._act_valve, self._act_fitting]
        tb.addSeparator()

        self._act_solve = btn("▶ Solve",
            "Run Newton-Raphson solver  (Ctrl+Enter)", self._on_solve)
        solve_btn = tb.widgetForAction(self._act_solve)
        if solve_btn:
            solve_btn.setStyleSheet("color:#7fff7f; font-weight:bold;")
        btn("✕ Clear",  "Clear all results",    self._on_clear_results)
        btn("⊡ New",    "New empty network",    self._on_new)
        tb.addSeparator()
        btn("⟲ Fit",   "Zoom to fit",           self._view.zoom_fit)

        # Fluid temperature control — drives ρ(T)/μ(T)/P_vapor(T) at solve time.
        tb.addSeparator()
        temp_lbl = QLabel(" Fluid T ")
        temp_lbl.setStyleSheet("color:#ccc; font-size:11px;")
        tb.addWidget(temp_lbl)
        self._temp_spin = QDoubleSpinBox()
        self._temp_spin.setRange(0.0, 100.0)
        self._temp_spin.setSingleStep(5.0)
        self._temp_spin.setDecimals(1)
        self._temp_spin.setSuffix(" °C")
        self._temp_spin.setValue(getattr(self._network, "temperature_c", 20.0))
        self._temp_spin.setToolTip(
            "Water temperature (0–100 °C). Sets density, viscosity and vapor "
            "pressure used by the solver. Re-solve to apply.")
        self._temp_spin.valueChanged.connect(self._on_temperature_changed)
        tb.addWidget(self._temp_spin)

        self._act_solve.setShortcut(QKeySequence("Ctrl+Return"))

    def _on_temperature_changed(self, value: float):
        """Update the network's fluid temperature (applied on the next solve)."""
        self._network.temperature_c = float(value)
        self._mark_dirty()
        self._refresh_status()

    def _sync_temperature_control(self):
        """Reflect the current network's temperature in the toolbar spinbox
        without re-triggering the change handler (used after File→Open)."""
        if hasattr(self, "_temp_spin"):
            self._temp_spin.blockSignals(True)
            self._temp_spin.setValue(getattr(self._network, "temperature_c", 20.0))
            self._temp_spin.blockSignals(False)

    def _on_set_units(self, system: str):
        """Switch the results display unit system (SI ↔ Imperial).  This is a
        pure display change — the model and solver remain SI, so no re-solve is
        needed; every results surface is simply re-rendered."""
        if units.get_system() == system:
            return
        units.set_system(system)
        self._plotter.refresh_units()    # tables + pump plot
        self._sidebar.refresh_units()    # per-component results read-out
        self._scene.update()             # canvas head / flow overlays
        self._refresh_status()

    # ── Signal wiring ─────────────────────────────────────────────────────────

    def _connect_signals(self):
        s = self._signals
        s.node_selected.connect(self._on_node_selected)
        s.edge_selected.connect(self._on_edge_selected)
        s.nothing_selected.connect(self._on_nothing_selected)
        s.node_moved.connect(self._on_node_moved)
        s.connection_requested.connect(self._on_connection_requested)
        s.delete_requested.connect(self._on_delete_requested)
        s.fitting_selected.connect(self._on_fitting_selected)
        s.fitting_placement_requested.connect(self._on_fitting_placement_requested)
        s.move_finished.connect(self._on_move_finished)
        s.inline_move_finished.connect(self._on_inline_move_finished)
        s.escape_pressed.connect(self._cancel_all_modes)

        self._view.placement_requested.connect(self._on_place_component)
        self._sidebar.apply_requested.connect(self._on_properties_apply)
        self._sidebar.fitting_apply_requested.connect(self._on_fitting_apply)
        self._sidebar.fitting_action_requested.connect(self._on_fitting_action)

    # ── Toolbar slots ─────────────────────────────────────────────────────────

    def _place_reservoir(self):
        self._cancel_all_modes()
        self._view.set_placement_mode("reservoir")
        self._status_net.setText("  Click canvas to place Reservoir")

    def _place_pressurized_source(self):
        self._cancel_all_modes()
        self._view.set_placement_mode("pressurized_source")
        self._status_net.setText("  Click canvas to place Pressurized Source")

    def _place_junction(self):
        self._cancel_all_modes()
        self._view.set_placement_mode("junction")
        self._status_net.setText("  Click canvas to place Junction")

    def _connect_pipe(self):
        if self._connect_mode_active:
            self._cancel_all_modes()
            return
        self._cancel_all_modes()
        self._pending_edge_type   = "pipe"
        self._connect_mode_active = True
        self._conn_step           = 0
        self._conn_from_id        = None
        self._act_pipe.setChecked(True)
        self._status_net.setText("  [Pipe] Click SOURCE node or port…")

    def _place_pump_mode(self):
        if self._view._placement_mode == "pump":
            self._cancel_all_modes()
            return
        self._cancel_all_modes()
        self._view.set_placement_mode("pump")
        self._act_pump.setChecked(True)
        self._status_net.setText("  Click canvas to place Pump")

    def _place_valve_mode(self):
        if self._view._placement_mode == "valve":
            self._cancel_all_modes()
            return
        self._cancel_all_modes()
        self._view.set_placement_mode("valve")
        self._act_valve.setChecked(True)
        self._status_net.setText("  Click canvas to place Valve")

    def _fitting_attach_mode(self):
        if self._fitting_mode_active:
            self._cancel_all_modes()
            return
        self._cancel_all_modes()
        self._fitting_mode_active = True
        self._act_fitting.setChecked(True)
        self._view.set_fitting_mode(True)
        self._status_net.setText("  [Fitting] Hover over a pipe and click to attach")

    def _cancel_all_modes(self):
        """Exit any active placement / connect / fitting mode cleanly."""
        self._connect_mode_active = False
        self._pending_edge_type   = None
        self._conn_step           = 0
        self._conn_from_id        = None
        self._fitting_mode_active = False
        self._scene._abort_connection()
        self._view.set_placement_mode(None)
        self._view.set_fitting_mode(False)
        for act in self._connect_actions + self._placement_actions:
            act.setChecked(False)

    # ── Canvas placement events ───────────────────────────────────────────────

    @pyqtSlot(str, float, float)
    def _on_place_component(self, comp_type: str, x: float, y: float):
        """Called when user clicks canvas during placement mode."""
        if comp_type in ("reservoir", "junction", "pressurized_source"):
            self._add_node_to_network(comp_type, x, y)
        elif comp_type in ("pump", "valve"):
            self._add_inline_component(comp_type, x, y)
        # Uncheck the relevant action
        action_map = {"pump": self._act_pump, "valve": self._act_valve}
        if comp_type in action_map:
            action_map[comp_type].setChecked(False)

    def _add_node_to_network(self, comp_type: str, x: float, y: float) -> Optional[str]:
        self._counters[comp_type] += 1
        n      = self._counters[comp_type]
        prefix = {"reservoir": "R", "junction": "J", "pressurized_source": "PS"}
        comp_id = f"{prefix[comp_type]}{n}"

        if comp_type == "reservoir":
            comp = Reservoir(comp_id, total_head=15.0)
        elif comp_type == "pressurized_source":
            comp = PressurizedSource(comp_id,
                                     elevation=0.0,
                                     surface_pressure_Pa=150000.0,  # 1.5 bar gauge default
                                     name=f"Press.Source {n}")
        else:
            comp = Junction(comp_id, elevation=0.0, demand=0.0)

        self._undo_stack.push(AddNodeCommand(self, comp_type, comp, x, y))
        return comp_id

    def _add_inline_component(self, comp_type: str, x: float, y: float) -> Optional[str]:
        """Place a pump or valve as a freely-placed inline icon with phantom nodes."""
        self._counters[comp_type] += 1
        n = self._counters[comp_type]
        prefix = {"pump": "Pu", "valve": "V"}
        comp_id = f"{prefix[comp_type]}{n}"

        if comp_type == "pump":
            comp = Pump(comp_id, A=-8000.0, B=0.0, C=25.0, diameter=0.1,
                        name=f"Pump {n}")
        else:
            comp = Valve(comp_id, diameter=0.1, K=5.0, name=f"Valve {n}")

        self._undo_stack.push(AddInlineComponentCommand(self, comp_type, comp, x, y))
        return comp_id

    # ── Pipe connection ───────────────────────────────────────────────────────

    @pyqtSlot(str, str, str)
    def _on_connection_requested(self, edge_type: str, from_id: str, to_id: str):
        self._connect_mode_active = False
        self._pending_edge_type   = None
        self._conn_step           = 0
        self._conn_from_id        = None
        for act in self._connect_actions:
            act.setChecked(False)

        self._counters[edge_type] += 1
        n       = self._counters[edge_type]
        prefix  = {"pipe": "P", "pump": "Pu", "valve": "V"}
        edge_id = f"{prefix.get(edge_type, 'E')}{n}"

        if edge_type == "pipe":
            comp = Pipe(edge_id, diameter=0.1, length=100.0)
        elif edge_type == "pump":
            comp = Pump(edge_id)
        else:
            comp = Valve(edge_id)

        # Validate before pushing (prevents bad state on undo-redo)
        if from_id not in self._network.nodes or to_id not in self._network.nodes:
            self._show_error(f"Node '{from_id}' or '{to_id}' not found.")
            return

        self._undo_stack.push(AddEdgeCommand(self, comp, from_id, to_id, edge_type))

    # ── Fitting attachment ────────────────────────────────────────────────────

    @pyqtSlot(str, float)
    def _on_fitting_placement_requested(self, pipe_edge_id: str, position_t: float):
        """Attach a new fitting to the given pipe at position_t."""
        if pipe_edge_id not in self._network.edges:
            return
        pipe_comp = self._network.edges[pipe_edge_id].component
        if not isinstance(pipe_comp, Pipe):
            return

        self._counters["fitting"] += 1
        n          = self._counters["fitting"]
        fitting_id = f"Ft{n}"

        # Default: 90° elbow screwed 1"
        subtype = self._fitting_defaults.get("fitting_subtype", "90° elbow, regular")
        conn    = self._fitting_defaults.get("connection_type", "Screwed")
        nom     = self._fitting_defaults.get("nominal_diameter_in", 1.0)
        K_def   = lookup_fitting_k(conn, subtype, nom)

        fa = FittingAttachment(
            fitting_id          = fitting_id,
            fitting_subtype     = subtype,
            connection_type     = conn,
            nominal_diameter_in = nom,
            K_default           = K_def,
            K_override          = None,
            position_t          = position_t,
            name                = f"Fitting {n}",
        )

        self._undo_stack.push(AddFittingCommand(self, pipe_edge_id, fa))

    @pyqtSlot(str, str)
    def _on_fitting_selected(self, pipe_edge_id: str, fitting_id: str):
        """User clicked a fitting icon on a pipe — show fitting properties."""
        if pipe_edge_id not in self._network.edges:
            return
        pipe_comp = self._network.edges[pipe_edge_id].component
        if not isinstance(pipe_comp, Pipe):
            return
        fa = next((f for f in pipe_comp.fittings if f.fitting_id == fitting_id), None)
        if fa is None:
            return
        self._sidebar.load_fitting_attachment(pipe_edge_id, fa)
        self._right_tabs.setCurrentIndex(0)

    @pyqtSlot(str, str, dict)
    def _on_fitting_apply(self, pipe_id: str, fitting_id: str, params: dict):
        """Apply fitting property changes."""
        if pipe_id not in self._network.edges:
            return
        pipe_comp = self._network.edges[pipe_id].component
        if not isinstance(pipe_comp, Pipe):
            return
        fa = next((f for f in pipe_comp.fittings if f.fitting_id == fitting_id), None)
        if fa is None:
            return

        for key, val in params.items():
            if hasattr(fa, key):
                setattr(fa, key, val)

        # Update canvas pipe item fitting data
        pipe_item = self._scene.edge_items.get(pipe_id)
        if isinstance(pipe_item, PipeEdgeItem):
            pipe_item.set_fittings(pipe_comp.fittings)

        self._mark_dirty()
        self._last_result = None
        self._scene.clear_results()
        self._sidebar.hide_results()
        self._status_solver.setText("Solver: — (modified, re-solve needed)")
        self._status_solver.setStyleSheet("color:#e0a030;")
        self._refresh_status()

    @pyqtSlot(str, str, str)
    def _on_fitting_action(self, action: str, pipe_id: str, fitting_id: str):
        """Handle fitting list actions (currently: delete)."""
        if action != "delete":
            return
        if pipe_id not in self._network.edges:
            return
        pipe_comp = self._network.edges[pipe_id].component
        if not isinstance(pipe_comp, Pipe):
            return
        fa = next((f for f in pipe_comp.fittings if f.fitting_id == fitting_id), None)
        if fa is None:
            return
        self._undo_stack.push(DeleteFittingCommand(self, pipe_id, fa))

    # ── Node / edge selection ─────────────────────────────────────────────────

    @pyqtSlot(str)
    def _on_node_selected(self, node_id: str):
        # ── Pipe connect mode ─────────────────────────────────────────────
        if self._connect_mode_active:
            if self._conn_step == 0:
                self._conn_from_id = node_id
                self._conn_step    = 1
                self._scene.begin_connection(node_id, self._pending_edge_type or "pipe")
                self._status_net.setText(
                    f"  [Pipe] Source: {node_id} — Click TARGET node/port…")
            # Step 2 is handled by scene → connection_requested signal
            return

        # ── Remap phantom nodes → pump/valve edge selection ───────────────
        if node_id in self._network.phantom_nodes:
            edge_id = self._network.phantom_nodes[node_id]
            self._on_edge_selected(edge_id)
            return

        # ── Normal node selection ─────────────────────────────────────────
        if node_id not in self._network.nodes:
            return
        comp = self._network.nodes[node_id].component
        self._sidebar.load_component(comp)
        self._right_tabs.setCurrentIndex(0)

        if self._last_result and self._last_result.converged:
            H = self._last_result.heads.get(node_id)
            P = self._last_result.pressures.get(node_id)
            if H is not None:
                self._sidebar.show_results({"head": H, "pressure_Pa": P})
        else:
            self._sidebar.hide_results()

    @pyqtSlot(str)
    def _on_edge_selected(self, edge_id: str):
        if edge_id not in self._network.edges:
            return
        comp = self._network.edges[edge_id].component
        self._sidebar.load_component(comp)
        self._right_tabs.setCurrentIndex(0)

        if self._last_result and self._last_result.converged:
            result_dict = {
                "flow":            self._last_result.flows.get(edge_id),
                "velocity":        self._last_result.velocities.get(edge_id),
                "head_loss":       self._last_result.head_losses.get(edge_id),
                "reynolds":        self._last_result.reynolds.get(edge_id),
                "friction_factor": self._last_result.friction_factors.get(edge_id),
            }
            if isinstance(comp, Pump):
                result_dict["npsh_available"] = comp.npsh_available
                result_dict["is_cavitating"]  = comp.is_cavitating
            # Compute required pump head + power at desired flow rate
            if isinstance(comp, Pump) and self._last_system_curves:
                h_req, P_kW = self._compute_pump_sizing(comp, edge_id)
                if h_req is not None:
                    result_dict["pump_req_head"] = h_req
                if P_kW is not None:
                    result_dict["pump_power"] = P_kW
            self._sidebar.show_results(result_dict)
        else:
            self._sidebar.hide_results()

    def _compute_pump_sizing(self, comp: Pump, edge_id: str
                            ) -> tuple:
        """
        Compute required pump head and hydraulic power at the pump's
        desired_flow_rate using the stored system curve.

            h_req = h_system(Q_des)
            P_req = ρ · g · Q_des · h_req   [W]  → returned in kW

        Returns (h_req [m], P_req [kW]).  Either value may be None if data
        are insufficient or if the system curve indicates no pump work is
        needed (gravity/pressure drives the flow at Q_des).
        """
        from fluid_props import DENSITY, GRAVITY
        import numpy as np

        Q_des = comp.desired_flow_rate   # [m³/s]
        if Q_des <= 0:
            return None, None

        sc = (self._last_system_curves.get(edge_id)
              or self._last_system_curves.get("__standalone__"))
        if sc is None:
            return None, None

        Q_arr, h_arr = sc
        if len(Q_arr) < 2:
            return None, None

        h_req = float(np.interp(Q_des, Q_arr, h_arr))
        if not np.isfinite(h_req):
            # System-curve sampling failed (e.g. a sub-solve diverged → NaN);
            # don't propagate a meaningless head/power into the UI.
            return None, None

        if h_req <= 0:
            # Gravity / upstream pressure provides sufficient head — no pump needed
            return h_req, None

        P_kW = DENSITY * GRAVITY * Q_des * h_req / 1000.0
        return h_req, P_kW

    @pyqtSlot()
    def _on_nothing_selected(self):
        self._sidebar.clear_selection()
        if self._connect_mode_active:
            self._cancel_all_modes()

    @pyqtSlot(str, float, float)
    def _on_node_moved(self, node_id: str, x: float, y: float):
        self._network.canvas_positions[node_id] = (x, y)
        self._mark_dirty()

    @pyqtSlot(str, float, float, float, float)
    def _on_move_finished(self, node_id: str,
                          old_x: float, old_y: float,
                          new_x: float, new_y: float):
        """Push a move command when a node drag ends (for undo support)."""
        self._undo_stack.push(
            MoveNodeCommand(self, node_id, old_x, old_y, new_x, new_y))

    @pyqtSlot(str, float, float, float, float)
    def _on_inline_move_finished(self, edge_id: str,
                                  old_x: float, old_y: float,
                                  new_x: float, new_y: float):
        """Push a move command when an inline component drag ends."""
        self._undo_stack.push(
            MoveInlineCommand(self, edge_id, old_x, old_y, new_x, new_y))

    # ── Delete ────────────────────────────────────────────────────────────────

    @pyqtSlot(str)
    def _on_delete_requested(self, comp_id: str):
        # ── Fitting delete (compound key from context menu) ───────────────
        if comp_id.startswith("fitting::"):
            _, pipe_id, fitting_id = comp_id.split("::", 2)
            self._on_fitting_action("delete", pipe_id, fitting_id)
            return

        # ── Inline pump/valve ─────────────────────────────────────────────
        if (comp_id in self._network.edges
                and comp_id in self._network.inline_positions):
            self._undo_stack.push(DeleteInlineComponentCommand(self, comp_id))
            return

        # ── Regular node ──────────────────────────────────────────────────
        if comp_id in self._network.nodes:
            self._undo_stack.push(DeleteNodeCommand(self, comp_id))
            return

        # ── Regular edge ──────────────────────────────────────────────────
        if comp_id in self._network.edges:
            self._undo_stack.push(DeleteEdgeCommand(self, comp_id))

    # ── Properties apply ──────────────────────────────────────────────────────

    @pyqtSlot(str, dict)
    def _on_properties_apply(self, comp_id: str, params: dict):
        comp = None
        if comp_id in self._network.nodes:
            comp = self._network.nodes[comp_id].component
        elif comp_id in self._network.edges:
            comp = self._network.edges[comp_id].component
        if comp is None:
            return

        for key, val in params.items():
            if key == "K_minor_override":
                # Can be None (use fittings) or float (override)
                if isinstance(comp, Pipe):
                    comp._K_minor_override = val
            elif hasattr(comp, key):
                setattr(comp, key, val)

        if isinstance(comp, Reservoir):
            comp.head = comp.total_head

        # Update pipe canvas item if fittings list changed
        if isinstance(comp, Pipe):
            pipe_item = self._scene.edge_items.get(comp_id)
            if isinstance(pipe_item, PipeEdgeItem):
                pipe_item.set_fittings(comp.fittings)

        self._sync_canvas_item(comp_id)
        self._mark_dirty()
        self._last_result        = None
        self._last_system_curves = {}
        self._scene.clear_results()
        self._sidebar.hide_results()
        self._status_solver.setText("Solver: — (modified, re-solve needed)")
        self._status_solver.setStyleSheet("color:#e0a030;")
        self._refresh_status()

    def _sync_canvas_item(self, comp_id: str):
        comp = None
        if comp_id in self._network.nodes:
            comp = self._network.nodes[comp_id].component
            item = self._scene.node_items.get(comp_id)
        elif comp_id in self._network.edges:
            comp = self._network.edges[comp_id].component
            item = (self._scene.edge_items.get(comp_id)
                    or self._scene.inline_items.get(comp_id))
        else:
            return
        if item is None or comp is None:
            return
        item.setToolTip(_component_tooltip(comp))
        item.set_display_name(comp.name)
        if isinstance(comp, Reservoir):
            item.set_elevation(comp.elevation)

    # ── Solve ─────────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_solve(self):
        try:
            self._status_solver.setText("Solver: running…")
            QApplication.processEvents()

            solver = NetworkSolver(self._network)
            result = solver.solve(tol=1e-9, max_iter=200)
            self._last_result = result

            if not result.converged and result.errors:
                self._sidebar.show_validation_error(result.errors)
                self._status_solver.setText(
                    f"Solver: ✗ validation failed ({len(result.errors)} errors)")
                self._status_solver.setStyleSheet("color:#e04040;")
                self._refresh_status()
                return

            if result.converged:
                # Check for cavitation warnings specifically
                cavitating_pumps = [e.component.id for e in self._network.get_pumps()
                                    if e.component.is_on and e.component.is_cavitating]
                if cavitating_pumps:
                    self._status_solver.setText(
                        f"Solver: ✓ converged (⚠ CAVITATION in {len(cavitating_pumps)} pumps)")
                    self._status_solver.setStyleSheet("color:#e0a030; font-weight:bold;")
                else:
                    self._status_solver.setText(
                        f"Solver: ✓ converged  (residual = {result.residual_norm:.2e})")
                    self._status_solver.setStyleSheet("color:#50cc80;")
            else:
                self._status_solver.setText(
                    f"Solver: ⚠ did not converge  (residual = {result.residual_norm:.2e})")
                self._status_solver.setStyleSheet("color:#e07030;")

            # Do NOT paint a non-converged iterate onto the canvas/plots — those
            # numbers are not a valid solution and would mislead the user.
            if not result.converged:
                self._scene.clear_results()
                self._plotter.clear()
                self._sidebar.hide_results()
                self._last_system_curves = {}
                diag = ""
                if result.worst_residual:
                    lbl, val = result.worst_residual
                    diag = f"  Worst-satisfied equation: {lbl} (residual {val:.2e})."
                self._sidebar.show_validation_error([
                    "Solver did not converge — results are not shown." + diag,
                    "Check the network for unstable pumps, closed loops with no "
                    "outlet, or extreme parameter values."])
                self._refresh_status()
                return

            # ── Results processing (converged solutions only) ─────────────────────
            elevations = {nid: node.component.elevation
                          for nid, node in self._network.nodes.items()
                          if isinstance(node.component, Reservoir)}
            self._scene.apply_results(result.heads, result.flows, elevations)

            system_curves = {}
            all_pumps = self._network.get_pumps()

            # Compute system curve for EVERY pump, regardless of on/off state.
            # The system curve is a property of the network, not the pump.
            for pump_edge in all_pumps:
                eid = pump_edge.edge_id
                Q_arr, h_arr = solver.compute_system_curve(eid, result)
                if len(Q_arr):
                    system_curves[eid] = (Q_arr, h_arr)

            # Standalone curve only when the network has no pumps at all
            if not all_pumps:
                Q_sa, h_sa = solver.compute_system_curve_standalone(result)
                if len(Q_sa):
                    system_curves["__standalone__"] = (Q_sa, h_sa)

            self._last_system_curves = system_curves

            # ── Multi-pump: detect topology groups + build combined curves ────────
            pump_groups = solver.detect_pump_groups()
            for grp in pump_groups:
                cfg      = grp["type"]
                pids     = grp["pump_ids"]
                grp["on_pumps"] = [pid for pid in pids
                                    if self._network.edges[pid].component.is_on]
                if cfg in ("series", "parallel") and len(grp["on_pumps"]) >= 2:
                    cQ, ch = solver.compute_combined_pump_curve(pids, cfg)
                    grp["combined_Q"] = cQ
                    grp["combined_h"] = ch

                    # Find operating point: intersection of combined curve + system curve
                    ref_pid = pids[0]
                    if ref_pid in system_curves:
                        sQ, sh = system_curves[ref_pid]
                        pt = solver.find_curve_intersection(cQ, ch, sQ, sh)
                        grp["op_point"] = pt   # (Q, h) or None
                    else:
                        grp["op_point"] = None
                else:
                    grp["combined_Q"] = None
                    grp["combined_h"] = None
                    grp["op_point"]   = None

            self._plotter.update_results(self._network, result, system_curves,
                                         pump_groups)
            self._right_tabs.setCurrentIndex(1)

            errs = self._network.validate()
            if result.errors: # Include cavitation warnings from solver
                errs.extend([e for e in result.errors if "CAVITATION" in e])
            self._sidebar.show_validation_error(errs)

        except Exception as e:
            self._status_solver.setText("Solver: CRASHED")
            self._status_solver.setStyleSheet("color:#e04040; font-weight:bold;")
            QMessageBox.critical(self, "Simulation Error",
                                 f"An unexpected error occurred during simulation:\n\n{str(e)}")
            import traceback
            traceback.print_exc()

    @pyqtSlot()
    def _on_clear_results(self):
        self._last_result = None
        self._last_system_curves = {}
        self._scene.clear_results()
        self._plotter.clear()
        self._sidebar.hide_results()
        self._status_solver.setText("Solver: —  ")
        self._status_solver.setStyleSheet("")

    # ── File operations ───────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_export_csv(self):
        if not self._last_result or not self._last_result.converged:
            QMessageBox.information(self, "Export CSV",
                "No solved results to export.\nRun the solver first.")
            return
        import csv
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results as CSV", "", "CSV files (*.csv)")
        if not path:
            return
        result = self._last_result
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                u_h, u_p = units.head_label(), units.pressure_label()
                u_q, u_v = units.flow_label(), units.velocity_label()
                writer.writerow(["=== NODE RESULTS ==="])
                writer.writerow(["Node ID", "Type", f"Head ({u_h})",
                                 f"Elevation ({u_h})", f"Pressure ({u_p})"])
                for nid, node in self._network.nodes.items():
                    if self._network.is_phantom(nid):
                        continue
                    comp  = node.component
                    H     = units.head_value(result.heads.get(nid, 0.0))
                    P     = units.pressure_value(result.pressures.get(nid, 0.0))
                    z     = units.head_value(getattr(comp, "elevation", 0.0))
                    writer.writerow([nid, type(comp).__name__,
                                     f"{H:.4f}", f"{z:.4f}", f"{P:.4f}"])
                writer.writerow([])
                writer.writerow(["=== EDGE RESULTS ==="])
                writer.writerow(["Edge ID", "Type", f"Flow ({u_q})", f"Velocity ({u_v})",
                                  "Reynolds", "f (Darcy)", f"Head Loss ({u_h})"])
                for eid, edge in self._network.edges.items():
                    comp = edge.component
                    Q    = units.flow_value(result.flows.get(eid, 0.0))
                    V    = units.velocity_value(result.velocities.get(eid, 0.0))
                    Re   = result.reynolds.get(eid, 0.0)
                    ff   = result.friction_factors.get(eid, 0.0)
                    hL   = units.head_value(result.head_losses.get(eid, 0.0))
                    writer.writerow([eid, type(comp).__name__,
                                     f"{Q:.4f}", f"{V:.4f}",
                                     f"{Re:.0f}", f"{ff:.6f}", f"{hL:.4f}"])

                npsh = getattr(result, "npsh", {}) or {}
                if npsh:
                    writer.writerow([])
                    writer.writerow(["=== PUMP NPSH / CAVITATION ==="])
                    writer.writerow(["Pump ID", f"NPSHa ({u_h})", f"NPSHr ({u_h})",
                                     f"Margin ({u_h})", "Status"])
                    for eid, d in npsh.items():
                        status = "CAVITATING" if d.get("cavitating") else "OK"
                        writer.writerow([eid,
                                         f"{units.head_value(d.get('available', 0.0)):.4f}",
                                         f"{units.head_value(d.get('required', 0.0)):.4f}",
                                         f"{units.head_value(d.get('margin', 0.0)):.4f}",
                                         status])
            self._status_net.setText(f"  Exported results to {os.path.basename(path)}")
        except Exception as e:
            self._show_error(f"Export failed:\n{e}")

    def _on_new(self):
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes", "Discard current network and start new?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return

        self._network.clear()
        self._scene.clear()
        self._scene.node_items.clear()
        self._scene.edge_items.clear()
        self._scene.inline_items.clear()
        self._scene.port_items.clear()
        self._scene._draw_grid()

        self._counters = {k: 0 for k in self._counters}
        self._last_result       = None
        self._last_system_curves = {}
        self._current_file = None
        self._dirty        = False
        self._undo_stack.clear()
        self._cancel_all_modes()
        self._sidebar.clear_selection()
        self._plotter.clear()
        self._status_solver.setText("Solver: —  ")
        self._status_solver.setStyleSheet("")
        self.setWindowTitle(self.APP_NAME)
        self._refresh_status()

    def _on_open(self):
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes", "Discard current network?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Network", "", "ThermoFluid Network (*.tfn *.json)")
        if not path:
            return
        try:
            net = PipeNetwork.load_json(path)
        except Exception as e:
            self._show_error(f"Failed to load file:\n{e}")
            return

        self._on_new()
        self._network      = net
        self._solver       = NetworkSolver(net)
        self._current_file = path

        self._sync_temperature_control()
        self._rebuild_canvas_from_network(net)
        self._undo_stack.clear()   # File load is not undoable
        self._plotter.set_network(net)
        self._dirty = False
        self.setWindowTitle(f"{self.APP_NAME} — {os.path.basename(path)}")
        QTimer.singleShot(200, self._view.zoom_fit)
        self._refresh_status()

    def _rebuild_canvas_from_network(self, net: PipeNetwork):
        """Rebuild all canvas items from a loaded PipeNetwork."""
        # 1. Regular (non-phantom) nodes
        for nid, node in net.nodes.items():
            if net.is_phantom(nid):
                continue
            x, y     = net.canvas_positions.get(nid, (0.0, 0.0))
            comp_type = "reservoir" if node.is_reservoir() else "junction"
            item = self._scene.add_node(nid, comp_type, x, y)
            item.setToolTip(_component_tooltip(node.component))
            item.set_display_name(node.component.name)
            if node.is_reservoir():
                item.set_elevation(node.component.elevation)

        # 2. Inline components (pumps/valves with phantom nodes) — BEFORE pipes
        for eid, edge in net.edges.items():
            if eid not in net.inline_positions:
                continue
            comp   = edge.component
            x, y   = net.inline_positions[eid]
            fn, tn = edge.from_node_id, edge.to_node_id
            ctype  = "pump" if isinstance(comp, Pump) else "valve"
            item   = self._scene.add_inline_component(eid, ctype, fn, tn, x, y)
            item.set_display_name(comp.name)
            item.setToolTip(_component_tooltip(comp))

        # 3. Regular edges (pipes, plus legacy pump/valve between real junctions)
        for eid, edge in net.edges.items():
            if eid in net.inline_positions:
                continue   # already handled above
            comp = edge.component
            if isinstance(comp, Pump):   etype = "pump"
            elif isinstance(comp, Valve): etype = "valve"
            elif isinstance(comp, Fitting): continue   # migrated to fittings
            else: etype = "pipe"
            edge_item = self._scene.add_edge(eid, etype,
                                              edge.from_node_id, edge.to_node_id)
            if edge_item:
                edge_item.setToolTip(_component_tooltip(comp))
                edge_item.set_display_name(comp.name)
                if isinstance(comp, Pipe) and isinstance(edge_item, PipeEdgeItem):
                    edge_item.set_fittings(comp.fittings)

    def _on_save(self):
        if self._current_file:
            self._save_to(self._current_file)
        else:
            self._on_save_as()

    def _on_save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Network", "", "ThermoFluid Network (*.tfn);;JSON (*.json)")
        if not path:
            return
        # Sync canvas positions before saving
        for nid, item in self._scene.node_items.items():
            p = item.pos()
            self._network.canvas_positions[nid] = (p.x(), p.y())
        for eid, item in self._scene.inline_items.items():
            p = item.pos()
            self._network.inline_positions[eid] = (p.x(), p.y())
        self._save_to(path)
        self._current_file = path

    def _save_to(self, path: str):
        try:
            self._network.save_json(path)
            self._dirty = False
            self.setWindowTitle(f"{self.APP_NAME} — {os.path.basename(path)}")
        except Exception as e:
            self._show_error(f"Save failed:\n{e}")

    # ── Status bar ────────────────────────────────────────────────────────────

    def _refresh_status(self):
        n    = len(self._network.nodes) - len(self._network.phantom_nodes)
        e    = len(self._network.edges)
        errs = self._network.validate()
        if n == 0:
            self._status_net.setText("  Network: empty")
            self._status_net.setStyleSheet("color:#888;")
        elif errs:
            self._status_net.setText(
                f"  Network: {n} nodes, {e} edges  ✗ {len(errs)} error(s)")
            self._status_net.setStyleSheet("color:#e04040;")
        elif self._last_result and self._last_result.converged:
            self._status_net.setText(
                f"  Network: {n} nodes, {e} edges  ✓ solved")
            self._status_net.setStyleSheet("color:#50cc80;")
        else:
            self._status_net.setText(
                f"  Network: {n} nodes, {e} edges  ● ready to solve")
            self._status_net.setStyleSheet("color:#e0a030;")

    def _mark_dirty(self):
        self._dirty = True
        title = self.windowTitle()
        if not title.startswith("*"):
            self.setWindowTitle("* " + title)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _show_error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)

    def _on_about(self):
        QMessageBox.about(self, "About ThermoFluid Designer",
            "<b>ThermoFluid Designer v0.2</b><br><br>"
            "A production-grade pipe network simulator.<br><br>"
            "Physics:<br>"
            "• Darcy-Weisbach with Haaland friction factor<br>"
            "• Newton-Raphson solver with explicit Jacobian<br>"
            "• Quadratic pump characteristic curves<br>"
            "• Per-fitting minor losses (K-value method)<br>"
            "• Full SI units throughout<br><br>"
            "Built with Python, PyQt6, NumPy, SciPy, Matplotlib.")

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        mod = event.modifiers()

        if key == Qt.Key.Key_Delete:
            sel = self._scene.selectedItems()
            if sel:
                # Loop through all selected items instead of just the first one
                for item in list(sel):
                    if isinstance(item, PortItem):
                        parent = item.parentItem()
                        if isinstance(parent, InlineComponentItem):
                            self._on_delete_requested(parent.component_edge_id)
                    elif isinstance(item, InlineComponentItem):
                        self._on_delete_requested(item.component_edge_id)
                    elif isinstance(item, NodeGraphicsItem):
                        self._on_delete_requested(item.node_id)
                    elif isinstance(item, EdgeGraphicsItem):
                        self._on_delete_requested(item.edge_id)
                    elif isinstance(item, FittingIconItem):
                        # Handle direct deletion of fitting icons
                        self._on_delete_requested(f"fitting::{item.pipe_edge_id}::{item.fitting_id}")
            return

        if key == Qt.Key.Key_Escape:
            self._cancel_all_modes()
            return

        if key == Qt.Key.Key_Z and mod & Qt.KeyboardModifier.ControlModifier:
            if mod & Qt.KeyboardModifier.ShiftModifier:
                self._undo_stack.redo()
            else:
                self._undo_stack.undo()
            return

        super().keyPressEvent(event)

    # ── Close ─────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes", "You have unsaved changes. Exit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
        event.accept()

    # ── Demo network ──────────────────────────────────────────────────────────

    def _load_demo_network(self):
        """
        Pre-built demo using the new inline-pump architecture:

            R_sump (H=0)
               |
           [P_feed: 50m, D=0.10m]
               |
           [Pu1 — inline pump A=-8000, C=30m]
               |
           [P_rise: 30m, D=0.10m]  (has a 90° elbow fitting)
               |
              J1 ──[P1: 200m, D=0.10m]──► R_a (H=20m)
               └──[P2: 150m, D=0.08m]──────► R_b (H=15m)

        Physics identical to v0.1 demo — same network topology,
        just the pump is now a freely-placed inline component.
        """
        net   = self._network
        scene = self._scene
        CX, CY = 400, 300

        # ── Nodes ─────────────────────────────────────────────────────────
        layout_nodes = {
            "R_sump": (-320,  0),
            "J1":     (  60,  0),
            "R_a":    ( 280, -80),
            "R_b":    ( 280,  80),
        }
        for comp in [
            Reservoir("R_sump", total_head=0.0,  name="Sump"),
            Junction( "J1",     elevation=0.0,    demand=0.0),
            Reservoir("R_a",    total_head=20.0,  name="Tank A"),
            Reservoir("R_b",    total_head=15.0,  name="Tank B"),
        ]:
            px, py = layout_nodes[comp.id]
            net.add_node(comp, canvas_x=CX+px, canvas_y=CY+py)
            ctype = "reservoir" if isinstance(comp, Reservoir) else "junction"
            item  = scene.add_node(comp.id, ctype, CX+px, CY+py)
            item.setToolTip(_component_tooltip(comp))
            item.set_display_name(comp.name)
            if isinstance(comp, Reservoir):
                item.set_elevation(comp.elevation)

        # ── Inline pump ────────────────────────────────────────────────────
        pump_cx = CX - 140
        pump_cy = CY
        pu1 = Pump("Pu1", A=-8000.0, B=0.0, C=30.0, diameter=0.1, name="Pump")
        net.add_inline_component(pu1, pump_cx, pump_cy)
        pump_item = scene.add_inline_component(
            "Pu1", "pump", "Pu1_in", "Pu1_out", pump_cx, pump_cy)
        pump_item.set_display_name("Pump")
        pump_item.setToolTip(_component_tooltip(pu1))

        # ── Pipe edges ─────────────────────────────────────────────────────
        p_feed = Pipe("P_feed",  diameter=0.10, length=50.0,  name="Feed")
        p_rise = Pipe("P_rise",  diameter=0.10, length=30.0,  name="Rise")
        p1     = Pipe("P1",      diameter=0.10, length=200.0, name="Pipe A")
        p2     = Pipe("P2",      diameter=0.08, length=150.0, name="Pipe B")

        # Add a demo fitting (90° elbow) to the rise pipe
        from fluid_props import lookup_fitting_k
        K_elbow = lookup_fitting_k("Screwed", "90° elbow, regular", 4.0)
        p_rise.fittings.append(FittingAttachment(
            fitting_id="Ft1", fitting_subtype="90° elbow, regular",
            connection_type="Screwed", nominal_diameter_in=4.0,
            K_default=K_elbow, position_t=0.5, name="Elbow"))

        for comp, fn, tn in [
            (p_feed, "R_sump", "Pu1_in"),
            (p_rise, "Pu1_out", "J1"),
            (p1,    "J1",     "R_a"),
            (p2,    "J1",     "R_b"),
        ]:
            net.add_edge(comp, from_node_id=fn, to_node_id=tn)
            edge_item = scene.add_edge(comp.id, "pipe", fn, tn)
            if edge_item:
                edge_item.setToolTip(_component_tooltip(comp))
                edge_item.set_display_name(comp.name)
                if isinstance(edge_item, PipeEdgeItem):
                    edge_item.set_fittings(comp.fittings)

        # Sync counters
        self._counters.update(
            reservoir=2, junction=1, pipe=4, pump=1, fitting=1)

        self._plotter.set_network(net)
        self._refresh_status()
        self._undo_stack.clear()   # Demo load is not undoable

        QTimer.singleShot(150, self._view.zoom_fit)
        QTimer.singleShot(400, self._on_solve)
