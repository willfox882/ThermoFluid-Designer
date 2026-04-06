"""
plotting_widget.py
------------------
Embedded Matplotlib figures for the thermofluid designer.

Tabs
────
  1. Pump Curve    – pump characteristic, system curve (parabolic approximation),
                     operating point, and efficiency annotation.
  2. Results Table – node heads and edge flow/velocity/Re/f in a styled table.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QLabel, QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
except ImportError:          # older matplotlib
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar

from matplotlib.figure import Figure
import matplotlib.patches as mpatches

from solver import SolverResult
from network import PipeNetwork
from components import Pump


# ── Style constants ────────────────────────────────────────────────────────────
BG   = "#f5f7fa"
FG   = "#2d3040"
BLUE = "#3a7bd5"
RED  = "#d94040"
GRN  = "#2ecc71"
GREY = "#909090"


class PlottingWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._network:  Optional[PipeNetwork]  = None
        self._result:   Optional[SolverResult] = None
        self._build_ui()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        # Tab 1: Pump curve plot
        self._pump_tab = QWidget()
        pump_layout = QVBoxLayout(self._pump_tab)
        pump_layout.setContentsMargins(0, 0, 0, 0)

        self._pump_fig    = Figure(figsize=(5, 3.5), facecolor=BG, tight_layout=True)
        self._pump_canvas = FigureCanvas(self._pump_fig)
        self._pump_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pump_ax = self._pump_fig.add_subplot(111)
        self._pump_toolbar = NavToolbar(self._pump_canvas, self._pump_tab)

        pump_layout.addWidget(self._pump_toolbar)
        pump_layout.addWidget(self._pump_canvas)
        self._tabs.addTab(self._pump_tab, "Pump Curve")

        # Tab 2: Results table
        self._table_tab = QWidget()
        table_layout = QVBoxLayout(self._table_tab)
        table_layout.setContentsMargins(4, 4, 4, 4)
        table_layout.setSpacing(6)

        lbl_nodes = QLabel("Node Heads")
        lbl_nodes.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        table_layout.addWidget(lbl_nodes)

        self._node_table = self._make_table(
            ["Node", "Type", "Head (m)", "Elev (m)", "Pressure (kPa)"])
        table_layout.addWidget(self._node_table)

        lbl_edges = QLabel("Edge Results")
        lbl_edges.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        table_layout.addWidget(lbl_edges)

        self._edge_table = self._make_table(
            ["Edge", "Type", "Flow (L/s)", "Vel (m/s)", "Reynolds", "f", "ΔH (m)"])
        table_layout.addWidget(self._edge_table)

        self._tabs.addTab(self._table_tab, "Results Table")

        self._draw_empty_pump_plot()

    @staticmethod
    def _make_table(headers: list[str]) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        t.setAlternatingRowColors(True)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        t.verticalHeader().setVisible(False)
        t.setFont(QFont("Consolas", 8))
        t.setMaximumHeight(200)
        return t

    # ── Public API ────────────────────────────────────────────────────────────

    def set_network(self, network: PipeNetwork):
        self._network = network
        self._draw_empty_pump_plot()

    def update_results(self, network: PipeNetwork, result: SolverResult,
                       system_curves: dict = None):
        """
        Refresh all plots and tables with solver results.

        system_curves : { pump_edge_id: (Q_arr, h_arr) }   (from solver)
        """
        self._network = network
        self._result  = result
        self._draw_pump_plot(system_curves or {})
        self._populate_tables(result)

    def clear(self):
        self._result = None
        self._draw_empty_pump_plot()
        self._node_table.setRowCount(0)
        self._edge_table.setRowCount(0)

    # ── Pump plot ─────────────────────────────────────────────────────────────

    def _draw_empty_pump_plot(self):
        ax = self._pump_ax
        ax.clear()
        ax.set_facecolor(BG)
        ax.text(0.5, 0.5, "Solve the network to see\npump curve & system curve",
                transform=ax.transAxes, ha="center", va="center",
                color=GREY, fontsize=10, style="italic")
        ax.set_xlabel("Flow rate Q  [L/s]", color=FG)
        ax.set_ylabel("Head  h  [m]", color=FG)
        ax.set_title("Pump Curve & System Curve", color=FG, fontweight="bold")
        ax.tick_params(colors=GREY)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        self._pump_fig.patch.set_facecolor(BG)
        self._pump_canvas.draw()

    def _draw_pump_plot(self, system_curves: dict):
        if self._network is None or self._result is None:
            return

        pump_edges = self._network.get_pumps()
        if not pump_edges:
            self._draw_empty_pump_plot()
            return

        ax = self._pump_ax
        ax.clear()
        ax.set_facecolor(BG)
        ax.set_facecolor("#fafbfd")
        ax.grid(True, linestyle="--", linewidth=0.5, color="#dde0e8", alpha=0.8)

        legend_handles = []

        for pump_edge in pump_edges:
            pump_comp: Pump = pump_edge.component
            eid = pump_edge.edge_id

            # Pump characteristic curve
            Q_arr, hp_arr = pump_comp.curve_data(n_points=300)
            Q_Ls  = Q_arr * 1000.0          # convert to L/s for readability

            line_pump, = ax.plot(Q_Ls, hp_arr, color=RED, lw=2.2,
                                 label=f"Pump: {eid}")
            legend_handles.append(line_pump)

            # System curve
            sys_key = eid
            if sys_key in system_curves:
                Qs_arr, hs_arr = system_curves[sys_key]
                Qs_Ls = Qs_arr * 1000.0
                line_sys, = ax.plot(Qs_Ls, hs_arr, color=BLUE, lw=2.0,
                                    linestyle="--", label="System curve")
                legend_handles.append(line_sys)

            # Operating point
            if eid in self._result.flows:
                Q_op  = abs(self._result.flows[eid])
                hp_op = abs(pump_comp.compute_pump_head(Q_op))
                Q_op_Ls = Q_op * 1000.0
                ax.scatter([Q_op_Ls], [hp_op], color=GRN, s=80, zorder=5,
                           edgecolors="white", linewidths=1.5)
                ax.annotate(
                    f" Operating point\n Q = {Q_op_Ls:.2f} L/s\n h = {hp_op:.2f} m",
                    xy=(Q_op_Ls, hp_op),
                    xytext=(Q_op_Ls + max(Q_Ls)*0.05, hp_op + max(hp_arr)*0.05),
                    fontsize=7.5, color=FG,
                    arrowprops=dict(arrowstyle="->", color=GRN, lw=1.2),
                )
                op_patch = mpatches.Patch(color=GRN, label="Operating point")
                legend_handles.append(op_patch)

        ax.set_xlabel("Flow rate Q  [L/s]", color=FG, fontsize=9)
        ax.set_ylabel("Head  h  [m]", color=FG, fontsize=9)
        ax.set_title("Pump Curve & System Curve", color=FG,
                     fontweight="bold", fontsize=10)
        ax.tick_params(colors=GREY, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")

        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=7.5,
                      framealpha=0.9, loc="upper right")

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        self._pump_fig.patch.set_facecolor(BG)
        self._pump_canvas.draw()

    # ── Results tables ────────────────────────────────────────────────────────

    def _populate_tables(self, result: SolverResult):
        if self._network is None:
            return

        from fluid_props import DENSITY, GRAVITY

        # ── Node table ────────────────────────────────────────────────
        self._node_table.setRowCount(0)
        for nid, node in self._network.nodes.items():
            comp = node.component
            H    = result.heads.get(nid, 0.0)
            P_kPa = result.pressures.get(nid, 0.0) / 1000.0
            z    = getattr(comp, "elevation", 0.0)

            row = self._node_table.rowCount()
            self._node_table.insertRow(row)
            data = [nid, type(comp).__name__,
                    f"{H:.3f}", f"{z:.2f}", f"{P_kPa:.2f}"]
            for col, val in enumerate(data):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter |
                    Qt.AlignmentFlag.AlignVCenter)
                self._node_table.setItem(row, col, item)

        # ── Edge table ────────────────────────────────────────────────
        self._edge_table.setRowCount(0)
        for eid, edge in self._network.edges.items():
            comp  = edge.component
            Q     = result.flows.get(eid, 0.0)
            V     = result.velocities.get(eid, 0.0)
            Re    = result.reynolds.get(eid, 0.0)
            f     = result.friction_factors.get(eid, 0.0)
            hL    = result.head_losses.get(eid, 0.0)

            row = self._edge_table.rowCount()
            self._edge_table.insertRow(row)
            data = [eid, type(comp).__name__,
                    f"{Q*1000:.3f}", f"{V:.3f}",
                    f"{Re:,.0f}", f"{f:.5f}", f"{hL:.3f}"]
            for col, val in enumerate(data):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter |
                    Qt.AlignmentFlag.AlignVCenter)
                self._edge_table.setItem(row, col, item)

            # Colour-code high-Re rows (turbulent)
            if Re > 4000:
                for c in range(self._edge_table.columnCount()):
                    it = self._edge_table.item(row, c)
                    if it:
                        it.setForeground(QColor("#2a5faa"))
