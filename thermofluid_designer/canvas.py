"""
canvas.py
---------
QGraphicsScene-based canvas for the thermofluid network designer.

Architecture
────────────
ThermofluidCanvas  (QGraphicsScene)
 ├── NodeGraphicsItem    (base)
 │    ├── JunctionItem
 │    └── ReservoirItem
 └── EdgeGraphicsItem    (drawn between two NodeGraphicsItems)
      ├── PipeEdgeItem
      ├── PumpEdgeItem
      └── ValveEdgeItem

ThermofluidView   (QGraphicsView)
  Wraps the canvas; adds zoom, pan, placement-mode cursor management.

CanvasSignals  (QObject)
  Centralised PyQt signals used by MainWindow to respond to canvas events.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from PyQt6.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsLineItem,
    QGraphicsRectItem, QGraphicsPathItem,
    QGraphicsTextItem, QMenu,
)
from PyQt6.QtCore import (
    Qt, QPointF, QRectF, pyqtSignal, QObject, QLineF,
)
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont,
    QPolygonF, QPainterPath, QCursor, QTransform,
)

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    "junction":     QColor(90, 95, 110),
    "reservoir":    QColor(25, 105, 185),
    "pipe":         QColor(55, 125, 210),
    "pump":         QColor(210, 60, 55),
    "valve":        QColor(50, 175, 90),
    "selected":     QColor(255, 195, 0),
    "invalid":      QColor(220, 50, 50),
    "grid_line":    QColor(230, 232, 238),
    "grid_major":   QColor(210, 215, 225),
    "edge_line":    QColor(100, 110, 130),
    "flow_hot":     QColor(240, 80, 40),
    "flow_cold":    QColor(40, 120, 220),
    "port":         QColor(255, 255, 255, 200),
    "port_border":  QColor(80, 90, 110),
    "text_dark":    QColor(40, 44, 52),
    "text_light":   QColor(245, 247, 250),
    "bg":           QColor(245, 247, 250),
}

GRID_MINOR = 20    # px
GRID_MAJOR = 100   # px
PORT_R     = 6.0   # port circle radius


# ═══════════════════════════════════════════════════════════════════════════════
# Signals
# ═══════════════════════════════════════════════════════════════════════════════

class CanvasSignals(QObject):
    node_selected       = pyqtSignal(str)          # node_id
    edge_selected       = pyqtSignal(str)          # edge_id
    nothing_selected    = pyqtSignal()
    node_moved          = pyqtSignal(str, float, float)  # id, x, y
    connection_requested = pyqtSignal(str, str, str)     # edge_type, from_id, to_id
    delete_requested    = pyqtSignal(str)          # component_id
    canvas_right_clicked = pyqtSignal(float, float)


# ═══════════════════════════════════════════════════════════════════════════════
# Node graphics items
# ═══════════════════════════════════════════════════════════════════════════════

class NodeGraphicsItem(QGraphicsItem):
    """Abstract base for Junction and Reservoir items."""

    def __init__(self, node_id: str, signals: CanvasSignals,
                 x: float = 0, y: float = 0):
        super().__init__()
        self.node_id = node_id
        self.signals = signals
        self.is_valid = True

        self.setPos(x, y)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.setZValue(2)

        self._hover = False
        self._size  = 48.0          # override in sub-classes

    # Results overlay (set after solve)
    _result_head: Optional[float] = None

    def set_result(self, head: float):
        self._result_head = head
        self.update()

    def clear_result(self):
        self._result_head = None
        self.update()

    def primary_color(self) -> QColor:
        raise NotImplementedError

    def boundingRect(self) -> QRectF:
        s = self._size
        return QRectF(-s/2 - PORT_R - 2, -s/2 - PORT_R - 2,
                       s + 2*(PORT_R + 2), s + 2*(PORT_R + 2) + 18)

    def center(self) -> QPointF:
        return self.scenePos()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.signals.node_moved.emit(
                self.node_id, self.pos().x(), self.pos().y())
            # Update connected edges (handled by scene)
            if self.scene():
                self.scene().refresh_edges_for_node(self.node_id)
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.signals.node_selected.emit(self.node_id)
        super().mousePressEvent(event)

    def hoverEnterEvent(self, event):
        self._hover = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hover = False
        self.update()
        super().hoverLeaveEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu()
        conn_action   = menu.addAction(f"Connect from '{self.node_id}'…")
        menu.addSeparator()
        delete_action = menu.addAction(f"Delete '{self.node_id}'")
        action = menu.exec(event.screenPos())
        if action == conn_action:
            # Notify scene to start connection mode
            if self.scene():
                self.scene().begin_connection(self.node_id)
        elif action == delete_action:
            self.signals.delete_requested.emit(self.node_id)

    def _draw_selection_ring(self, painter: QPainter):
        if self.isSelected() or self._hover:
            r = self._size / 2 + PORT_R + 1
            glow_color = C["selected"] if self.isSelected() else QColor(200, 210, 255, 120)
            pen = QPen(glow_color, 2.5)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(0, 0), r, r)

    def _draw_label(self, painter: QPainter, text: str):
        painter.setPen(QPen(C["text_dark"]))
        font = QFont("Segoe UI", 7, QFont.Weight.Bold)
        painter.setFont(font)
        label_rect = QRectF(-self._size/2, self._size/2 + 4, self._size, 14)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_result_overlay(self, painter: QPainter):
        if self._result_head is None:
            return
        painter.setPen(QPen(QColor(30, 30, 30)))
        font = QFont("Segoe UI", 6)
        painter.setFont(font)
        rect = QRectF(-self._size/2, -self._size/2 - 14, self._size, 14)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                         f"H={self._result_head:.2f}m")


class JunctionItem(NodeGraphicsItem):
    def __init__(self, node_id: str, signals: CanvasSignals,
                 x: float = 0, y: float = 0):
        super().__init__(node_id, signals, x, y)
        self._size = 34.0

    def primary_color(self) -> QColor:
        return C["invalid"] if not self.is_valid else C["junction"]

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.primary_color()
        r = self._size / 2

        # Body circle
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(140), 1.5))
        painter.drawEllipse(QPointF(0, 0), r, r)

        # Cross symbol
        painter.setPen(QPen(C["text_light"], 2.0))
        offset = r * 0.45
        painter.drawLine(QPointF(-offset, 0), QPointF(offset, 0))
        painter.drawLine(QPointF(0, -offset), QPointF(0, offset))

        self._draw_selection_ring(painter)
        self._draw_label(painter, self.node_id)
        self._draw_result_overlay(painter)


class ReservoirItem(NodeGraphicsItem):
    def __init__(self, node_id: str, signals: CanvasSignals,
                 x: float = 0, y: float = 0):
        super().__init__(node_id, signals, x, y)
        self._size = 60.0

    def primary_color(self) -> QColor:
        return C["invalid"] if not self.is_valid else C["reservoir"]

    def boundingRect(self) -> QRectF:
        w = self._size
        h = self._size * 0.72
        return QRectF(-w/2 - PORT_R - 2, -h/2 - PORT_R - 2,
                       w + 2*(PORT_R+2), h + 2*(PORT_R+2) + 18)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.primary_color()
        w = self._size
        h = self._size * 0.72
        x0, y0 = -w/2, -h/2

        # Outer tank wall
        painter.setBrush(QBrush(color.lighter(170)))
        painter.setPen(QPen(color.darker(130), 1.8))
        painter.drawRect(QRectF(x0, y0, w, h))

        # Water fill gradient
        water_top = y0 + h * 0.28
        water_h   = h * 0.55
        water_color = QColor(30, 110, 220, 190)
        painter.setBrush(QBrush(water_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(x0 + 2, water_top, w - 4, water_h))

        # Water surface ripple lines
        painter.setPen(QPen(QColor(90, 160, 255, 180), 1.5))
        for dx in (-8, 0, 8):
            painter.drawArc(
                QRectF(x0 + w/2 - 10 + dx, water_top - 4, 20, 8), 0, 180 * 16)

        # Outer border again (on top)
        if self.isSelected():
            painter.setPen(QPen(C["selected"], 2.5))
        else:
            painter.setPen(QPen(color.darker(130), 1.8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(x0, y0, w, h))

        # Port dot at bottom
        py = y0 + h
        painter.setBrush(QBrush(C["port"]))
        painter.setPen(QPen(C["port_border"], 1.2))
        painter.drawEllipse(QPointF(0, py), PORT_R, PORT_R)

        # Side ports
        painter.drawEllipse(QPointF(x0, 0), PORT_R, PORT_R)
        painter.drawEllipse(QPointF(x0 + w, 0), PORT_R, PORT_R)

        self._draw_selection_ring(painter)
        self._draw_label(painter, self.node_id)
        self._draw_result_overlay(painter)


# ═══════════════════════════════════════════════════════════════════════════════
# Edge graphics items
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeGraphicsItem(QGraphicsItem):
    """
    Directed edge drawn between two NodeGraphicsItems.
    Sub-classes draw a component symbol at the midpoint.
    """

    def __init__(self, edge_id: str, edge_type: str,
                 from_item: NodeGraphicsItem, to_item: NodeGraphicsItem,
                 signals: CanvasSignals):
        super().__init__()
        self.edge_id   = edge_id
        self.edge_type = edge_type
        self.from_item = from_item
        self.to_item   = to_item
        self.signals   = signals
        self.is_valid  = True
        self._flow_rate: Optional[float] = None

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)
        self._hover = False

    def set_flow(self, Q: float):
        self._flow_rate = Q
        self.update()

    def clear_flow(self):
        self._flow_rate = None
        self.update()

    def _endpoints(self) -> tuple[QPointF, QPointF]:
        return self.from_item.center(), self.to_item.center()

    def _midpoint(self) -> QPointF:
        p1, p2 = self._endpoints()
        return QPointF((p1.x()+p2.x())/2, (p1.y()+p2.y())/2)

    def _angle_rad(self) -> float:
        p1, p2 = self._endpoints()
        return math.atan2(p2.y() - p1.y(), p2.x() - p1.x())

    def _line_color(self) -> QColor:
        if not self.is_valid:
            return C["invalid"]
        if self._flow_rate is not None:
            # Use the scene's max_flow for a relative color ramp (cold→hot)
            max_q = 0.02  # fallback
            if self.scene() and hasattr(self.scene(), 'max_flow'):
                max_q = self.scene().max_flow
            t = min(abs(self._flow_rate) / max_q, 1.0)
            r = int(C["flow_cold"].red()   + t*(C["flow_hot"].red()   - C["flow_cold"].red()))
            g = int(C["flow_cold"].green() + t*(C["flow_hot"].green() - C["flow_cold"].green()))
            b = int(C["flow_cold"].blue()  + t*(C["flow_hot"].blue()  - C["flow_cold"].blue()))
            return QColor(r, g, b)
        return C["edge_line"]

    def boundingRect(self) -> QRectF:
        p1, p2 = self._endpoints()
        pad = 40
        return QRectF(
            min(p1.x(), p2.x()) - pad, min(p1.y(), p2.y()) - pad,
            abs(p2.x()-p1.x()) + 2*pad, abs(p2.y()-p1.y()) + 2*pad
        )

    def _draw_line_and_arrow(self, painter: QPainter):
        p1, p2 = self._endpoints()
        lc = self._line_color()
        lw = 2.5 if self.isSelected() else 2.0
        pen = QPen(C["selected"] if self.isSelected() else lc, lw)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(p1, p2)

        # Directional arrow at 65% along the line
        t   = 0.65
        ap  = QPointF(p1.x() + t*(p2.x()-p1.x()), p1.y() + t*(p2.y()-p1.y()))
        ang = self._angle_rad()
        arr = 10
        ax1 = QPointF(ap.x() - arr*math.cos(ang-0.38),
                      ap.y() - arr*math.sin(ang-0.38))
        ax2 = QPointF(ap.x() - arr*math.cos(ang+0.38),
                      ap.y() - arr*math.sin(ang+0.38))
        painter.drawLine(ap, ax1)
        painter.drawLine(ap, ax2)

    def _draw_flow_label(self, painter: QPainter):
        if self._flow_rate is None:
            return
        mid = self._midpoint()
        ang = self._angle_rad()
        # Offset label perpendicular to edge
        perp_x = -math.sin(ang) * 18
        perp_y =  math.cos(ang) * 18
        lx = mid.x() + perp_x - 28
        ly = mid.y() + perp_y - 7

        painter.setPen(QPen(C["text_dark"]))
        font = QFont("Consolas", 6)
        painter.setFont(font)
        Q_Ls = self._flow_rate * 1000.0
        painter.drawText(QRectF(lx, ly, 56, 14),
                         Qt.AlignmentFlag.AlignCenter,
                         f"{Q_Ls:.2f} L/s")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.signals.edge_selected.emit(self.edge_id)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu()
        del_action = menu.addAction(f"Delete '{self.edge_id}'")
        action = menu.exec(event.screenPos())
        if action == del_action:
            self.signals.delete_requested.emit(self.edge_id)

    def hoverEnterEvent(self, event):
        self._hover = True; self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hover = False; self.update()
        super().hoverLeaveEvent(event)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_line_and_arrow(painter)
        self._draw_symbol(painter)
        self._draw_edge_label(painter)
        self._draw_flow_label(painter)

    def _draw_symbol(self, painter: QPainter):
        """Override in sub-class to draw component symbol at midpoint."""
        pass

    def _draw_edge_label(self, painter: QPainter):
        mid = self._midpoint()
        ang = self._angle_rad()
        perp_x = -math.sin(ang) * 14
        perp_y =  math.cos(ang) * 14
        lx = mid.x() + perp_x - 25
        ly = mid.y() + perp_y - 14

        painter.setPen(QPen(C["text_dark"], 1))
        font = QFont("Segoe UI", 6, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QRectF(lx, ly, 50, 12),
                         Qt.AlignmentFlag.AlignCenter,
                         self.edge_id)


class PipeEdgeItem(EdgeGraphicsItem):
    def __init__(self, edge_id, from_item, to_item, signals):
        super().__init__(edge_id, "pipe", from_item, to_item, signals)

    def _draw_symbol(self, painter: QPainter):
        # Draw small pipe cross-section indicator at midpoint
        mid = self._midpoint()
        ang = self._angle_rad()
        painter.save()
        painter.translate(mid)
        painter.rotate(math.degrees(ang))

        color = C["pipe"]
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(140), 1))
        painter.drawRect(QRectF(-12, -6, 24, 12))

        # Inner bore
        painter.setBrush(QBrush(C["bg"]))
        painter.drawRect(QRectF(-10, -3, 20, 6))

        painter.restore()


class PumpEdgeItem(EdgeGraphicsItem):
    def __init__(self, edge_id, from_item, to_item, signals):
        super().__init__(edge_id, "pump", from_item, to_item, signals)

    def _draw_symbol(self, painter: QPainter):
        mid = self._midpoint()
        ang = self._angle_rad()
        painter.save()
        painter.translate(mid)
        painter.rotate(math.degrees(ang))

        # Circle body
        color = C["pump"]
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(140), 1.5))
        painter.drawEllipse(QPointF(0, 0), 14, 14)

        # Impeller triangle
        path = QPainterPath()
        path.moveTo(-7, 0)
        path.lineTo(5, -6)
        path.lineTo(5, 6)
        path.closeSubpath()
        painter.fillPath(path, QBrush(C["text_light"]))

        painter.restore()


class ValveEdgeItem(EdgeGraphicsItem):
    def __init__(self, edge_id, from_item, to_item, signals):
        super().__init__(edge_id, "valve", from_item, to_item, signals)

    def _draw_symbol(self, painter: QPainter):
        mid = self._midpoint()
        ang = self._angle_rad()
        painter.save()
        painter.translate(mid)
        painter.rotate(math.degrees(ang))

        color = C["valve"]
        painter.setPen(QPen(color.darker(140), 1.5))
        painter.setBrush(QBrush(color))

        # Bowtie shape
        poly1 = QPolygonF([QPointF(-13, -8), QPointF(0, 0), QPointF(-13, 8)])
        poly2 = QPolygonF([QPointF(13, -8),  QPointF(0, 0), QPointF(13, 8)])
        painter.drawPolygon(poly1)
        painter.drawPolygon(poly2)

        # Stem
        painter.setPen(QPen(color.darker(170), 2))
        painter.drawLine(QPointF(0, 0), QPointF(0, -14))
        painter.drawLine(QPointF(-4, -14), QPointF(4, -14))

        painter.restore()


class FittingEdgeItem(EdgeGraphicsItem):
    """Fitting (elbow/valve/tee) drawn as a diamond at the midpoint."""
    def __init__(self, edge_id, from_item, to_item, signals):
        super().__init__(edge_id, "fitting", from_item, to_item, signals)

    def _draw_symbol(self, painter: QPainter):
        mid = self._midpoint()
        ang = self._angle_rad()
        painter.save()
        painter.translate(mid)
        painter.rotate(math.degrees(ang))

        color = QColor(180, 100, 220)  # purple for fittings
        painter.setPen(QPen(color.darker(140), 1.5))
        painter.setBrush(QBrush(color))

        # Diamond shape
        poly = QPolygonF([
            QPointF(0, -10), QPointF(12, 0),
            QPointF(0, 10),  QPointF(-12, 0),
        ])
        painter.drawPolygon(poly)

        # K label
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        font = QFont("Segoe UI", 5, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QRectF(-8, -6, 16, 12),
                         Qt.AlignmentFlag.AlignCenter, "K")

        painter.restore()


_EDGE_ITEM_CLASSES = {
    "pipe":    PipeEdgeItem,
    "pump":    PumpEdgeItem,
    "valve":   ValveEdgeItem,
    "fitting": FittingEdgeItem,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Temp connection line (shown while user is connecting)
# ═══════════════════════════════════════════════════════════════════════════════

class TempConnectionLine(QGraphicsLineItem):
    def __init__(self):
        super().__init__()
        pen = QPen(QColor(255, 165, 0), 2, Qt.PenStyle.DashLine)
        pen.setDashPattern([6, 4])
        self.setPen(pen)
        self.setZValue(10)


# ═══════════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════════

class ThermofluidCanvas(QGraphicsScene):
    """
    Central canvas for component placement and connection.

    Workflow for connections:
        1. User right-clicks a node → "Connect from …"
        2. Scene enters connection mode: _connecting = True, source set
        3. Mouse moves → temp dashed line follows cursor
        4. User clicks a target node → connection_requested signal fired
        5. MainWindow creates the edge in both model and canvas
    """

    def __init__(self, signals: CanvasSignals, parent=None):
        super().__init__(parent)
        self.signals = signals

        self.node_items: Dict[str, NodeGraphicsItem] = {}
        self.edge_items: Dict[str, EdgeGraphicsItem] = {}

        self._connecting = False
        self._conn_source: Optional[str] = None
        self._temp_line: Optional[TempConnectionLine] = None
        self._pending_edge_type: Optional[str] = None   # "pipe" | "pump" | "valve"
        self.max_flow: float = 0.02   # m³/s — updated by apply_results for color ramp

        self.setSceneRect(-3000, -3000, 6000, 6000)
        self._draw_grid()

    # ── Grid ─────────────────────────────────────────────────────────────────

    def _draw_grid(self):
        pen_minor = QPen(C["grid_line"], 0.5)
        pen_major = QPen(C["grid_major"], 0.8)

        for x in range(-3000, 3001, GRID_MINOR):
            pen = pen_major if x % GRID_MAJOR == 0 else pen_minor
            self.addLine(x, -3000, x, 3000, pen)

        for y in range(-3000, 3001, GRID_MINOR):
            pen = pen_major if y % GRID_MAJOR == 0 else pen_minor
            self.addLine(-3000, y, 3000, y, pen)

    # ── Component management ──────────────────────────────────────────────────

    def add_node(self, node_id: str, node_type: str,
                 x: float = 0, y: float = 0) -> NodeGraphicsItem:
        cls = JunctionItem if node_type == "junction" else ReservoirItem
        item = cls(node_id, self.signals, x, y)
        self.addItem(item)
        self.node_items[node_id] = item
        return item

    def add_edge(self, edge_id: str, edge_type: str,
                 from_id: str, to_id: str) -> Optional[EdgeGraphicsItem]:
        if from_id not in self.node_items or to_id not in self.node_items:
            return None
        cls  = _EDGE_ITEM_CLASSES.get(edge_type, PipeEdgeItem)
        item = cls(edge_id, self.node_items[from_id],
                   self.node_items[to_id], self.signals)
        self.addItem(item)
        self.edge_items[edge_id] = item
        return item

    def remove_component(self, comp_id: str):
        if comp_id in self.node_items:
            self.removeItem(self.node_items.pop(comp_id))
        elif comp_id in self.edge_items:
            self.removeItem(self.edge_items.pop(comp_id))

    def mark_validity(self, comp_id: str, valid: bool):
        item = (self.node_items.get(comp_id) or self.edge_items.get(comp_id))
        if item:
            item.is_valid = valid
            item.update()

    def refresh_edges_for_node(self, node_id: str):
        """Trigger repaint on all edges touching a moved node."""
        for ei in self.edge_items.values():
            if ei.from_item.node_id == node_id or ei.to_item.node_id == node_id:
                ei.prepareGeometryChange()
                ei.update()

    # ── Solver results display ────────────────────────────────────────────────

    def apply_results(self, heads: dict, flows: dict):
        # Compute max flow for the color ramp (use at least 0.001 to avoid /0)
        if flows:
            self.max_flow = max(abs(q) for q in flows.values()) or 0.02
            self.max_flow = max(self.max_flow, 1e-6)
        for nid, H in heads.items():
            if nid in self.node_items:
                self.node_items[nid].set_result(H)
        for eid, Q in flows.items():
            if eid in self.edge_items:
                self.edge_items[eid].set_flow(Q)

    def clear_results(self):
        self.max_flow = 0.02   # reset to default
        for item in self.node_items.values():
            item.clear_result()
        for item in self.edge_items.values():
            item.clear_flow()

    # ── Connection mode ───────────────────────────────────────────────────────

    def begin_connection(self, source_id: str, edge_type: str = "pipe"):
        self._connecting       = True
        self._conn_source      = source_id
        self._pending_edge_type = edge_type
        src = self.node_items.get(source_id)
        if src:
            p = src.center()
            self._temp_line = TempConnectionLine()
            self._temp_line.setLine(p.x(), p.y(), p.x(), p.y())
            self.addItem(self._temp_line)

    def _abort_connection(self):
        self._connecting = False
        self._conn_source = None
        self._pending_edge_type = None
        if self._temp_line:
            self.removeItem(self._temp_line)
            self._temp_line = None

    # ── Mouse events ──────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event):
        if self._connecting and self._temp_line:
            src = self.node_items.get(self._conn_source)
            if src:
                p = src.center()
                self._temp_line.setLine(p.x(), p.y(),
                                        event.scenePos().x(),
                                        event.scenePos().y())
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if self._connecting and event.button() == Qt.MouseButton.LeftButton:
            # Find if we clicked a node
            items  = self.items(event.scenePos())
            target = next((i for i in items
                           if isinstance(i, NodeGraphicsItem)), None)
            if target and target.node_id != self._conn_source:
                from_id    = self._conn_source
                to_id      = target.node_id
                edge_type  = self._pending_edge_type or "pipe"
                self._abort_connection()
                self.signals.connection_requested.emit(edge_type, from_id, to_id)
            else:
                self._abort_connection()
                # Notify MainWindow so it can restore normal signal routing
                self.signals.nothing_selected.emit()
            return

        super().mousePressEvent(event)

        # If nothing selected, emit deselect signal
        if not self.selectedItems():
            self.signals.nothing_selected.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._abort_connection()
        super().keyPressEvent(event)


# ═══════════════════════════════════════════════════════════════════════════════
# View
# ═══════════════════════════════════════════════════════════════════════════════

class ThermofluidView(QGraphicsView):
    """
    QGraphicsView wrapping ThermofluidCanvas.
    Adds:
    • Smooth zoom with Ctrl+scroll
    • Pan with middle-mouse drag
    • Placement mode: click-to-place a component type
    """

    placement_requested = pyqtSignal(str, float, float)   # type, scene_x, scene_y

    def __init__(self, scene: ThermofluidCanvas, parent=None):
        super().__init__(scene, parent)
        self._scene = scene
        self._placement_mode: Optional[str] = None   # component type

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(C["bg"]))
        self.setMinimumSize(400, 300)

    # ── Placement mode ────────────────────────────────────────────────────────

    def set_placement_mode(self, component_type: Optional[str]):
        self._placement_mode = component_type
        if component_type:
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if (self._placement_mode
                and event.button() == Qt.MouseButton.LeftButton
                and not self._scene._connecting):
            scene_pos = self.mapToScene(event.pos())
            comp_type = self._placement_mode
            self.set_placement_mode(None)
            self.placement_requested.emit(comp_type, scene_pos.x(), scene_pos.y())
            return

        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            # Simulate left-click to start pan
            fake = event
            super().mousePressEvent(event)
            return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(
                QGraphicsView.DragMode.NoDrag
                if self._placement_mode else
                QGraphicsView.DragMode.RubberBandDrag)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)

    # ── Zoom helpers ──────────────────────────────────────────────────────────

    def zoom_fit(self):
        items = [i for i in self._scene.items()
                 if not isinstance(i, QGraphicsLineItem)]
        if not items:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            return
        bounds = items[0].mapToScene(items[0].boundingRect()).boundingRect()
        for it in items[1:]:
            bounds = bounds.united(
                it.mapToScene(it.boundingRect()).boundingRect())
        self.fitInView(bounds.adjusted(-60, -60, 60, 60),
                       Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_reset(self):
        self.setTransform(QTransform())
