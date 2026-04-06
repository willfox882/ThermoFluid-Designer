"""
main_window.py
--------------
QMainWindow that wires together:
    • ThermofluidCanvas / ThermofluidView  (central widget)
    • PropertiesPanel                      (right dock)
    • PlottingWidget                       (right dock, second tab)
    • NetworkSolver                        (model layer)
    • File menu (New / Open / Save / Save As)
    • Toolbar  (component palette + tools)
    • Status bar (validation + solver state)
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QTabWidget,
    QToolBar, QStatusBar, QLabel, QFileDialog,
    QMessageBox, QDockWidget, QSizePolicy,
    QVBoxLayout, QApplication,
)
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSlot
from PyQt6.QtGui import (
    QAction, QIcon, QKeySequence, QFont, QColor,
)

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components import Pipe, Pump, Valve, Fitting, Junction, Reservoir, component_from_dict
from network import PipeNetwork
from solver import NetworkSolver, SolverResult
from canvas import ThermofluidCanvas, ThermofluidView, CanvasSignals
from sidebar import PropertiesPanel
from plotting_widget import PlottingWidget


# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    APP_NAME = "ThermoFluid Designer"

    def __init__(self):
        super().__init__()
        self._network     = PipeNetwork()
        self._solver      = NetworkSolver(self._network)
        self._last_result : Optional[SolverResult] = None
        self._current_file: Optional[str]          = None
        self._dirty       = False          # unsaved changes?

        # counters for auto-IDs
        self._counters = {k: 0 for k in
                          ("reservoir", "junction", "pipe", "pump", "valve", "fitting")}

        # pending edge-type while connecting two nodes
        self._pending_edge_type: Optional[str] = None
        self._connect_mode_active = False
        self._conn_step = 0
        self._conn_from_id: Optional[str] = None

        # Undo stack for delete operations
        self._undo_stack: list = []  # list of (type, data) tuples

        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._connect_signals()

        self.setWindowTitle(self.APP_NAME)
        self.resize(1300, 820)

        # Load a demo network after event loop starts
        QTimer.singleShot(100, self._load_demo_network)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Central splitter: canvas (left 70%) | right panel (30%)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # Canvas
        self._signals = CanvasSignals()
        self._scene   = ThermofluidCanvas(self._signals)
        self._view    = ThermofluidView(self._scene)
        splitter.addWidget(self._view)

        # Right panel: Properties + Plots tabs
        self._right_tabs = QTabWidget()
        self._right_tabs.setMinimumWidth(290)
        self._right_tabs.setMaximumWidth(400)

        self._sidebar  = PropertiesPanel()
        self._plotter  = PlottingWidget()

        self._right_tabs.addTab(self._sidebar, "Properties")
        self._right_tabs.addTab(self._plotter, "Plots")
        splitter.addWidget(self._right_tabs)

        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)

        # Status bar
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

        # ── File ──────────────────────────────────────────────────────
        file_menu = mb.addMenu("&File")

        act_new = QAction("&New", self)
        act_new.setShortcut(QKeySequence.StandardKey.New)
        act_new.triggered.connect(self._on_new)
        file_menu.addAction(act_new)

        act_open = QAction("&Open…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._on_open)
        file_menu.addAction(act_open)

        file_menu.addSeparator()

        act_save = QAction("&Save", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self._on_save)
        file_menu.addAction(act_save)

        act_saveas = QAction("Save &As…", self)
        act_saveas.setShortcut(QKeySequence("Ctrl+Shift+S"))
        act_saveas.triggered.connect(self._on_save_as)
        file_menu.addAction(act_saveas)

        file_menu.addSeparator()

        act_exit = QAction("E&xit", self)
        act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # ── View ──────────────────────────────────────────────────────
        view_menu = mb.addMenu("&View")
        act_fit = QAction("Fit to View", self)
        act_fit.setShortcut(QKeySequence("F"))
        act_fit.triggered.connect(self._view.zoom_fit)
        view_menu.addAction(act_fit)

        act_reset = QAction("Reset Zoom", self)
        act_reset.triggered.connect(self._view.zoom_reset)
        view_menu.addAction(act_reset)

        # ── Help ──────────────────────────────────────────────────────
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
            "QToolButton { color:#ccc; padding:4px 8px; border-radius:3px; "
            "              font-size:11px; }"
            "QToolButton:hover  { background:#404560; color:white; }"
            "QToolButton:checked{ background:#3a7bd5; color:white; }"
        )
        self.addToolBar(tb)

        def btn(text: str, tip: str, slot=None, checkable=False) -> QAction:
            act = QAction(text, self)
            act.setToolTip(tip)
            act.setCheckable(checkable)
            if slot:
                act.triggered.connect(slot)
            tb.addAction(act)
            return act

        # Node palette
        btn("⬡ Reservoir",  "Place reservoir (fixed head)", self._place_reservoir)
        btn("● Junction",   "Place junction node",           self._place_junction)
        tb.addSeparator()

        # Edge palette — checkable so user can see they're in connect mode
        self._act_pipe    = btn("━ Pipe",    "Connect two nodes with a pipe",    self._connect_pipe,    checkable=True)
        self._act_pump    = btn("⊛ Pump",    "Connect two nodes with a pump",    self._connect_pump,    checkable=True)
        self._act_valve   = btn("⊠ Valve",   "Connect two nodes with a valve",   self._connect_valve,   checkable=True)
        self._act_fitting = btn("◇ Fitting", "Connect two nodes with a fitting", self._connect_fitting, checkable=True)
        self._connect_actions = [self._act_pipe, self._act_pump, self._act_valve, self._act_fitting]
        tb.addSeparator()

        # Actions
        self._act_solve = btn("▶ Solve",  "Run Newton-Raphson solver", self._on_solve)
        # QAction has no setStyleSheet; style the underlying QToolButton
        solve_btn = tb.widgetForAction(self._act_solve)
        if solve_btn:
            solve_btn.setStyleSheet(
                "color:#7fff7f; font-weight:bold;")
        btn("✕ Clear",   "Clear all results",             self._on_clear_results)
        btn("⊡ New",     "New empty network",             self._on_new)
        tb.addSeparator()
        btn("⟲ Fit",     "Zoom to fit",                   self._view.zoom_fit)

        # Keyboard shortcut
        self._act_solve.setShortcut(QKeySequence("Ctrl+Return"))

    # ── Signal wiring ─────────────────────────────────────────────────────────

    def _connect_signals(self):
        s = self._signals
        s.node_selected.connect(self._on_node_selected)
        s.edge_selected.connect(self._on_edge_selected)
        s.nothing_selected.connect(self._on_nothing_selected)
        s.node_moved.connect(self._on_node_moved)
        s.connection_requested.connect(self._on_connection_requested)
        s.delete_requested.connect(self._on_delete_requested)

        self._view.placement_requested.connect(self._on_place_component)
        self._sidebar.apply_requested.connect(self._on_properties_apply)

    # ── Toolbar slots: placement mode ─────────────────────────────────────────

    def _place_reservoir(self):
        self._view.set_placement_mode("reservoir")
        self._status_net.setText("  Click canvas to place Reservoir")

    def _place_junction(self):
        self._view.set_placement_mode("junction")
        self._status_net.setText("  Click canvas to place Junction")

    def _connect_pipe(self):
        self._enter_connect_mode("pipe")

    def _connect_pump(self):
        self._enter_connect_mode("pump")

    def _connect_valve(self):
        self._enter_connect_mode("valve")

    def _connect_fitting(self):
        self._enter_connect_mode("fitting")

    def _enter_connect_mode(self, edge_type: str):
        """
        Click-to-connect mode: first click selects source node,
        second click selects destination and creates the edge.
        Uses a state flag — no signal re-routing needed.
        """
        self._pending_edge_type = edge_type
        self._connect_mode_active = True
        self._conn_step = 0
        self._conn_from_id = None
        self._status_net.setText(
            f"  [{edge_type.title()}] Click SOURCE node…")

        # Check the right toolbar button, uncheck others
        action_map = {"pipe": self._act_pipe, "pump": self._act_pump,
                      "valve": self._act_valve, "fitting": self._act_fitting}
        for k, act in action_map.items():
            act.setChecked(k == edge_type)

    def _exit_connect_mode(self):
        """Clean exit from connect mode — always safe to call."""
        self._connect_mode_active = False
        self._pending_edge_type = None
        self._conn_step = 0
        self._conn_from_id = None
        self._scene._abort_connection()
        # Uncheck all connect-mode toolbar buttons
        for act in self._connect_actions:
            act.setChecked(False)
        self._refresh_status()

    # ── Canvas event handlers ─────────────────────────────────────────────────

    @pyqtSlot(str, float, float)
    def _on_place_component(self, comp_type: str, x: float, y: float):
        """Called when user clicks canvas during placement mode."""
        self._add_node_to_network(comp_type, x, y)

    def _add_node_to_network(self, comp_type: str,
                              x: float, y: float) -> Optional[str]:
        self._counters[comp_type] += 1
        n = self._counters[comp_type]
        prefix = {"reservoir": "R", "junction": "J"}
        comp_id = f"{prefix[comp_type]}{n}"

        if comp_type == "reservoir":
            comp = Reservoir(comp_id, total_head=15.0)
        else:
            comp = Junction(comp_id, elevation=0.0, demand=0.0)

        self._network.add_node(comp, canvas_x=x, canvas_y=y)
        self._scene.add_node(comp_id, comp_type, x, y)
        self._mark_dirty()
        self._refresh_status()
        return comp_id

    @pyqtSlot(str, str, str)
    def _on_connection_requested(self, edge_type: str, from_id: str, to_id: str):
        """Called by canvas when user completes a connection."""
        # Exit connect mode cleanly
        self._connect_mode_active = False
        self._pending_edge_type = None
        self._conn_step = 0
        self._conn_from_id = None
        for act in self._connect_actions:
            act.setChecked(False)

        self._counters[edge_type] += 1
        n = self._counters[edge_type]
        prefix = {"pipe": "P", "pump": "Pu", "valve": "V", "fitting": "Ft"}
        edge_id = f"{prefix.get(edge_type, 'E')}{n}"

        # Default component
        if edge_type == "pipe":
            comp = Pipe(edge_id, diameter=0.1, length=100.0)
        elif edge_type == "pump":
            comp = Pump(edge_id, A=-8000.0, B=0.0, C=25.0, diameter=0.1)
        elif edge_type == "fitting":
            comp = Fitting(edge_id)
        else:
            comp = Valve(edge_id, diameter=0.1, K=5.0)

        try:
            self._network.add_edge(comp, from_node_id=from_id, to_node_id=to_id)
        except (KeyError, ValueError) as e:
            self._show_error(str(e))
            return

        self._scene.add_edge(edge_id, edge_type, from_id, to_id)
        self._mark_dirty()
        self._refresh_status()
        self._signals.edge_selected.emit(edge_id)

    @pyqtSlot(str)
    def _on_node_selected(self, node_id: str):
        # ── Connect mode: route to two-click connection handler ───────
        if self._connect_mode_active:
            if self._conn_step == 0:
                self._conn_from_id = node_id
                self._conn_step = 1
                self._scene.begin_connection(node_id, self._pending_edge_type or "pipe")
                self._status_net.setText(
                    f"  [{(self._pending_edge_type or 'pipe').title()}] "
                    f"Source: {node_id} — Click TARGET node…")
            # Step 2 is handled by canvas → connection_requested signal
            return

        # ── Normal selection ──────────────────────────────────────────
        if node_id not in self._network.nodes:
            return
        comp = self._network.nodes[node_id].component
        self._sidebar.load_component(comp)
        self._right_tabs.setCurrentIndex(0)

        # Show results if available
        if self._last_result and self._last_result.converged:
            H   = self._last_result.heads.get(node_id)
            P   = self._last_result.pressures.get(node_id)
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
            self._sidebar.show_results({
                "flow":           self._last_result.flows.get(edge_id),
                "velocity":       self._last_result.velocities.get(edge_id),
                "head_loss":      self._last_result.head_losses.get(edge_id),
                "reynolds":       self._last_result.reynolds.get(edge_id),
                "friction_factor":self._last_result.friction_factors.get(edge_id),
            })
        else:
            self._sidebar.hide_results()

    @pyqtSlot()
    def _on_nothing_selected(self):
        self._sidebar.clear_selection()
        # If we were in connect mode, exit cleanly
        if self._connect_mode_active:
            self._exit_connect_mode()

    @pyqtSlot(str, float, float)
    def _on_node_moved(self, node_id: str, x: float, y: float):
        self._network.canvas_positions[node_id] = (x, y)
        self._mark_dirty()

    @pyqtSlot(str)
    def _on_delete_requested(self, comp_id: str):
        # Save undo info before deleting
        if comp_id in self._network.nodes:
            node = self._network.nodes[comp_id]
            comp_dict = node.component.to_dict()
            pos = self._network.canvas_positions.get(comp_id, (0, 0))
            # Also save connected edges so they can be restored
            connected_edges = []
            for eid in list(node.connected_edge_ids):
                if eid in self._network.edges:
                    edge = self._network.edges[eid]
                    connected_edges.append({
                        "comp": edge.component.to_dict(),
                        "from": edge.from_node_id,
                        "to":   edge.to_node_id,
                    })
            self._undo_stack.append(("node", comp_dict, pos, connected_edges))
            self._network.remove_node(comp_id)
        elif comp_id in self._network.edges:
            edge = self._network.edges[comp_id]
            edge_info = {
                "comp": edge.component.to_dict(),
                "from": edge.from_node_id,
                "to":   edge.to_node_id,
            }
            self._undo_stack.append(("edge", edge_info))
            self._network.remove_edge(comp_id)
        else:
            return

        self._scene.remove_component(comp_id)
        self._sidebar.clear_selection()
        self._last_result = None
        self._scene.clear_results()
        self._mark_dirty()
        self._refresh_status()

    # ── Properties apply ──────────────────────────────────────────────────────

    @pyqtSlot(str, dict)
    def _on_properties_apply(self, comp_id: str, params: dict):
        """Update component parameters from sidebar form."""
        comp = None
        if comp_id in self._network.nodes:
            comp = self._network.nodes[comp_id].component
        elif comp_id in self._network.edges:
            comp = self._network.edges[comp_id].component

        if comp is None:
            return

        for key, val in params.items():
            if hasattr(comp, key):
                setattr(comp, key, val)

        # Special: keep Reservoir.head in sync with total_head
        if isinstance(comp, Reservoir):
            comp.head = comp.total_head

        self._mark_dirty()
        self._last_result = None
        self._scene.clear_results()
        self._sidebar.hide_results()
        self._status_solver.setText("Solver: — (modified, re-solve needed)")
        self._status_solver.setStyleSheet("color:#e0a030;")
        self._refresh_status()

    # ── Solve ─────────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_solve(self):
        self._status_solver.setText("Solver: running…")
        QApplication.processEvents()

        solver = NetworkSolver(self._network)
        result = solver.solve(tol=1e-9, max_iter=200)

        self._last_result = result

        if not result.converged and result.errors:
            self._sidebar.show_validation_error(result.errors)
            self._status_solver.setText(
                f"Solver: ✗ validation failed ({len(result.errors)} errors)")
            self._refresh_status()
            return

        if result.converged:
            self._status_solver.setText(
                f"Solver: ✓ converged  "
                f"(residual = {result.residual_norm:.2e})")
            self._status_solver.setStyleSheet("color:#50cc80;")
        else:
            self._status_solver.setText(
                f"Solver: ⚠ did not converge  "
                f"(residual = {result.residual_norm:.2e})")
            self._status_solver.setStyleSheet("color:#e07030;")

        # Apply results to canvas
        self._scene.apply_results(result.heads, result.flows)

        # Compute system curves for all pumps
        system_curves = {}
        for pump_edge in self._network.get_pumps():
            eid = pump_edge.edge_id
            Q_arr, h_arr = solver.compute_system_curve(eid, result)
            if len(Q_arr):
                system_curves[eid] = (Q_arr, h_arr)

        # Update plotter
        self._plotter.update_results(self._network, result, system_curves)
        self._right_tabs.setCurrentIndex(1)   # switch to Plots tab

        # Validate and show errors/success in sidebar
        errs = self._network.validate()
        self._sidebar.show_validation_error(errs)

        # Refresh selected component results if any
        sel = self._scene.selectedItems()

    @pyqtSlot()
    def _on_clear_results(self):
        self._last_result = None
        self._scene.clear_results()
        self._plotter.clear()
        self._sidebar.hide_results()
        self._status_solver.setText("Solver: —  ")
        self._status_solver.setStyleSheet("")

    # ── File operations ───────────────────────────────────────────────────────

    def _on_new(self):
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "Discard current network and start new?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return

        self._network.clear()
        # scene.clear() removes all QGraphicsItems; clear our dicts afterwards
        # so we never hold dangling item references
        self._scene.clear()
        self._scene.node_items.clear()
        self._scene.edge_items.clear()
        self._scene._draw_grid()

        self._counters = {k: 0 for k in self._counters}
        self._last_result  = None
        self._current_file = None
        self._dirty        = False
        self._undo_stack.clear()
        self._connect_mode_active = False
        self._pending_edge_type = None
        self._sidebar.clear_selection()
        self._plotter.clear()
        self._status_solver.setText("Solver: —  ")
        self._status_solver.setStyleSheet("")
        self.setWindowTitle(self.APP_NAME)
        self._refresh_status()

    def _on_open(self):
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "Discard current network?",
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

        self._on_new()   # clear current state
        self._network    = net
        self._solver     = NetworkSolver(net)
        self._current_file = path

        # Rebuild canvas from loaded network
        for nid, node in net.nodes.items():
            x, y = net.canvas_positions.get(nid, (0.0, 0.0))
            comp_type = "reservoir" if node.is_reservoir() else "junction"
            self._scene.add_node(nid, comp_type, x, y)

        for eid, edge in net.edges.items():
            from components import Pipe, Pump, Valve, Fitting
            if isinstance(edge.component, Pump):
                etype = "pump"
            elif isinstance(edge.component, Valve):
                etype = "valve"
            elif isinstance(edge.component, Fitting):
                etype = "fitting"
            else:
                etype = "pipe"
            self._scene.add_edge(eid, etype, edge.from_node_id, edge.to_node_id)

        self._plotter.set_network(net)
        self._dirty = False
        self.setWindowTitle(f"{self.APP_NAME} — {os.path.basename(path)}")
        QTimer.singleShot(200, self._view.zoom_fit)
        self._refresh_status()

    def _on_save(self):
        if self._current_file:
            self._save_to(self._current_file)
        else:
            self._on_save_as()

    def _on_save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Network", "", "ThermoFluid Network (*.tfn);;JSON (*.json)")
        if path:
            # Update canvas positions before saving
            for nid, item in self._scene.node_items.items():
                p = item.pos()
                self._network.canvas_positions[nid] = (p.x(), p.y())
            self._save_to(path)
            self._current_file = path

    def _save_to(self, path: str):
        try:
            self._network.save_json(path)
            self._dirty = False
            self.setWindowTitle(
                f"{self.APP_NAME} — {os.path.basename(path)}")
        except Exception as e:
            self._show_error(f"Save failed:\n{e}")

    # ── Status bar ────────────────────────────────────────────────────────────

    def _refresh_status(self):
        n  = len(self._network.nodes)
        e  = len(self._network.edges)
        errs = self._network.validate()
        if n == 0:
            self._status_net.setText("  Network: empty")
            self._status_net.setStyleSheet("color:#888;")
        elif errs:
            # Red for errors (disconnected nodes, missing reservoirs, etc.)
            self._status_net.setText(
                f"  Network: {n} nodes, {e} edges  ✗ {len(errs)} error(s)")
            self._status_net.setStyleSheet("color:#e04040;")
        elif self._last_result and self._last_result.converged:
            # Green: valid AND solved
            self._status_net.setText(
                f"  Network: {n} nodes, {e} edges  ✓ solved")
            self._status_net.setStyleSheet("color:#50cc80;")
        else:
            # Yellow: valid but not yet solved
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
            "<b>ThermoFluid Designer v0.1</b><br><br>"
            "A production-grade pipe network simulator.<br><br>"
            "Physics:<br>"
            "• Darcy-Weisbach with Haaland friction factor<br>"
            "• Newton-Raphson solver with explicit Jacobian<br>"
            "• Quadratic pump characteristic curves<br>"
            "• Full SI units throughout<br><br>"
            "Built with Python, PyQt6, NumPy, SciPy, Matplotlib.")

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        mod = event.modifiers()

        # Delete key — delete selected component
        if key == Qt.Key.Key_Delete:
            sel = self._scene.selectedItems()
            if sel:
                from canvas import NodeGraphicsItem, EdgeGraphicsItem
                item = sel[0]
                if isinstance(item, NodeGraphicsItem):
                    self._on_delete_requested(item.node_id)
                elif isinstance(item, EdgeGraphicsItem):
                    self._on_delete_requested(item.edge_id)
            return

        # Escape — abort connection mode or placement mode
        if key == Qt.Key.Key_Escape:
            self._exit_connect_mode()
            self._view.set_placement_mode(None)
            return

        # Ctrl+Z — undo last delete
        if key == Qt.Key.Key_Z and mod & Qt.KeyboardModifier.ControlModifier:
            self._undo_last_delete()
            return

        super().keyPressEvent(event)

    def _undo_last_delete(self):
        """Undo the most recent delete operation."""
        if not self._undo_stack:
            return

        entry = self._undo_stack.pop()

        if entry[0] == "edge":
            _, edge_info = entry
            comp = component_from_dict(edge_info["comp"])
            from_id, to_id = edge_info["from"], edge_info["to"]
            # Both endpoint nodes must still exist
            if from_id not in self._network.nodes or to_id not in self._network.nodes:
                return
            try:
                self._network.add_edge(comp, from_node_id=from_id, to_node_id=to_id)
            except (KeyError, ValueError):
                return
            etype = {"Pipe": "pipe", "Pump": "pump", "Valve": "valve"}.get(
                type(comp).__name__, "pipe")
            self._scene.add_edge(comp.id, etype, from_id, to_id)

        elif entry[0] == "node":
            _, comp_dict, pos, connected_edges = entry
            comp = component_from_dict(comp_dict)
            ctype = "reservoir" if isinstance(comp, Reservoir) else "junction"
            self._network.add_node(comp, canvas_x=pos[0], canvas_y=pos[1])
            self._scene.add_node(comp.id, ctype, pos[0], pos[1])

            # Restore connected edges
            for edge_info in connected_edges:
                ecomp = component_from_dict(edge_info["comp"])
                from_id, to_id = edge_info["from"], edge_info["to"]
                if from_id in self._network.nodes and to_id in self._network.nodes:
                    try:
                        self._network.add_edge(ecomp, from_node_id=from_id,
                                               to_node_id=to_id)
                        etype = {"Pipe": "pipe", "Pump": "pump", "Valve": "valve"}.get(
                            type(ecomp).__name__, "pipe")
                        self._scene.add_edge(ecomp.id, etype, from_id, to_id)
                    except (KeyError, ValueError):
                        pass

        self._last_result = None
        self._scene.clear_results()
        self._mark_dirty()
        self._refresh_status()

    # ── Closing ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Exit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
        event.accept()

    # ── Demo network ──────────────────────────────────────────────────────────

    def _load_demo_network(self):
        """
        Pre-built demonstration network (pump + two delivery branches):

            R_sump (H=0m)
                |
            [Pu1: A=-8000, C=30m]       ← centrifugal pump
                |
               J1  ──[P1: 200m, D=0.10m]──► R_a (H=20m)
                └──[P2: 150m, D=0.08m]──────► R_b (H=15m)

        Physics:
          • Pump lifts fluid from the sump (H=0) to J1 (≈25 m after solve)
          • Two parallel pipe branches deliver to elevated reservoirs
          • Expected Q_pump ≈ 24.7 L/s, J1 head ≈ 25.1 m
          • Mass balance at J1:  Q_P1 + Q_P2 = Q_pump  (exact to 10⁻¹⁰ m³/s)
        """
        net   = self._network
        scene = self._scene

        # ── Nodes ─────────────────────────────────────────────────────────────
        r_sump = Reservoir("R_sump", total_head=0.0,  name="Sump")
        r_a    = Reservoir("R_a",    total_head=20.0, name="Tank A")
        r_b    = Reservoir("R_b",    total_head=15.0, name="Tank B")
        j1     = Junction("J1",      elevation=0.0,   demand=0.0)

        CX, CY = 400, 300
        layout = {
            "R_sump": (-280,  140),
            "J1":     (-100,    0),
            "R_a":    ( 160,  -80),
            "R_b":    ( 160,   80),
        }
        node_order = [r_sump, j1, r_a, r_b]

        for comp in node_order:
            px, py = layout[comp.id]
            net.add_node(comp, canvas_x=CX + px, canvas_y=CY + py)
            ctype = "reservoir" if isinstance(comp, Reservoir) else "junction"
            scene.add_node(comp.id, ctype, CX + px, CY + py)

        # ── Edges ─────────────────────────────────────────────────────────────
        pu1 = Pump("Pu1", A=-8000.0, B=0.0, C=30.0, diameter=0.1, name="Pump")
        p1  = Pipe("P1",  diameter=0.10, length=200.0, name="Pipe A")
        p2  = Pipe("P2",  diameter=0.08, length=150.0, name="Pipe B")

        for comp, from_id, to_id, etype in [
            (pu1, "R_sump", "J1",   "pump"),
            (p1,  "J1",    "R_a",  "pipe"),
            (p2,  "J1",    "R_b",  "pipe"),
        ]:
            net.add_edge(comp, from_node_id=from_id, to_node_id=to_id)
            scene.add_edge(comp.id, etype, from_id, to_id)

        # Sync counters
        self._counters["reservoir"] = 3
        self._counters["junction"]  = 1
        self._counters["pipe"]      = 2
        self._counters["pump"]      = 1

        self._plotter.set_network(net)
        self._refresh_status()

        # Auto-fit view and auto-solve
        QTimer.singleShot(150, self._view.zoom_fit)
        QTimer.singleShot(400, self._on_solve)
