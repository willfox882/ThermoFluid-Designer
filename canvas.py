"""
canvas.py
---------
QGraphicsScene-based canvas for the thermofluid network designer.

Architecture
────────────
ThermofluidCanvas  (QGraphicsScene)
 ├── NodeGraphicsItem    (base for junctions / reservoirs)
 │    ├── JunctionItem
 │    ├── ReservoirItem
 │    └── PortItem          ← inlet / outlet port on an inline component
 ├── InlineComponentItem    ← freely-placed pump / valve icon
 │    ├── PumpCanvasItem
 │    └── ValveCanvasItem
 └── EdgeGraphicsItem       ← pipe drawn between two node/port items
      └── PipeEdgeItem      ← carries fitting attachment icons

ThermofluidView   (QGraphicsView)
  Wraps the canvas; adds zoom, pan, placement-mode, fitting-mode.

CanvasSignals  (QObject)
  Central PyQt signals for canvas ↔ MainWindow communication.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

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

from components import FittingAttachment

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    "junction":     QColor(90, 95, 110),
    "reservoir":    QColor(25, 105, 185),
    "pipe":         QColor(55, 125, 210),
    "pump":         QColor(210, 60, 55),
    "valve":        QColor(50, 175, 90),
    "fitting":      QColor(160, 80, 200),
    "selected":     QColor(255, 195, 0),
    "invalid":      QColor(220, 50, 50),
    "grid_line":    QColor(230, 232, 238),
    "grid_major":   QColor(210, 215, 225),
    "edge_line":    QColor(100, 110, 130),
    "flow_hot":     QColor(240, 80, 40),
    "flow_cold":    QColor(40, 120, 220),
    "port_in":      QColor(255, 140, 0),
    "port_out":     QColor(50, 200, 100),
    "port":         QColor(255, 255, 255, 200),
    "port_border":  QColor(80, 90, 110),
    "text_dark":    QColor(40, 44, 52),
    "text_light":   QColor(245, 247, 250),
    "bg":           QColor(245, 247, 250),
    "fitting_hl":   QColor(255, 220, 0),   # pipe highlight in fitting mode
}

GRID_MINOR = 20    # px
GRID_MAJOR = 100   # px
PORT_R     = 6.0   # port circle radius


# ═══════════════════════════════════════════════════════════════════════════════
# Signals
# ═══════════════════════════════════════════════════════════════════════════════

class CanvasSignals(QObject):
    node_selected             = pyqtSignal(str)           # node_id
    edge_selected             = pyqtSignal(str)           # edge_id
    nothing_selected          = pyqtSignal()
    node_moved                = pyqtSignal(str, float, float)
    connection_requested      = pyqtSignal(str, str, str) # edge_type, from_id, to_id
    delete_requested          = pyqtSignal(str)           # component_id
    canvas_right_clicked      = pyqtSignal(float, float)
    fitting_selected          = pyqtSignal(str, str)      # pipe_edge_id, fitting_id
    fitting_placement_requested = pyqtSignal(str, float)  # pipe_edge_id, position_t
    escape_pressed            = pyqtSignal()
    # Emitted when a drag ends and position actually changed (for undo tracking)
    move_finished             = pyqtSignal(str, float, float, float, float)  # node_id, old_x, old_y, new_x, new_y
    inline_move_finished      = pyqtSignal(str, float, float, float, float)  # edge_id, old_x, old_y, new_x, new_y


# ═══════════════════════════════════════════════════════════════════════════════
# Node graphics items
# ═══════════════════════════════════════════════════════════════════════════════

class NodeGraphicsItem(QGraphicsItem):
    """Abstract base for Junction / Reservoir items."""

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
        self._size  = 48.0

    _result_head: Optional[float]  = None
    _elevation:   Optional[float]  = None
    _display_name: Optional[str]   = None

    def set_result(self, head: float):
        self._result_head = head; self.update()

    def clear_result(self):
        self._result_head = None; self.update()

    def set_elevation(self, z: float):
        self._elevation = z; self.update()

    def set_display_name(self, name: str):
        self._display_name = name if name and name != self.node_id else None
        self.update()

    def primary_color(self) -> QColor:
        raise NotImplementedError

    def boundingRect(self) -> QRectF:
        s = self._size
        return QRectF(-s/2 - PORT_R - 2, -s/2 - PORT_R - 2 - 20,
                       s + 2*(PORT_R + 2), s + 2*(PORT_R + 2) + 18 + 20)

    def center(self) -> QPointF:
        return self.scenePos()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            x = round(value.x() / GRID_MINOR) * GRID_MINOR
            y = round(value.y() / GRID_MINOR) * GRID_MINOR
            return QPointF(x, y)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.signals.node_moved.emit(
                self.node_id, self.pos().x(), self.pos().y())
            if self.scene():
                self.scene().refresh_edges_for_node(self.node_id)
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = self.pos()
            self.signals.node_selected.emit(self.node_id)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            start = getattr(self, '_drag_start_pos', None)
            if start is not None:
                end = self.pos()
                if end != start:
                    self.signals.move_finished.emit(
                        self.node_id,
                        start.x(), start.y(),
                        end.x(), end.y(),
                    )
                self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    def hoverEnterEvent(self, event):
        self._hover = True;  self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hover = False; self.update()
        super().hoverLeaveEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu()
        conn_action   = menu.addAction(f"Connect from '{self.node_id}'…")
        menu.addSeparator()
        delete_action = menu.addAction(f"Delete '{self.node_id}'")
        action = menu.exec(event.screenPos())
        if action == conn_action:
            if self.scene():
                self.scene().begin_connection(self.node_id)
        elif action == delete_action:
            self.signals.delete_requested.emit(self.node_id)

    def _draw_selection_ring(self, painter: QPainter):
        if self.isSelected() or self._hover:
            r = self._size / 2 + PORT_R + 1
            glow = C["selected"] if self.isSelected() else QColor(200, 210, 255, 120)
            painter.setPen(QPen(glow, 2.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(0, 0), r, r)

    def _draw_label(self, painter: QPainter, text: str):
        painter.setPen(QPen(C["text_dark"]))
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        label_rect = QRectF(-self._size/2, self._size/2 + 4, self._size, 14)
        display = self._display_name if self._display_name else text
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, display)

    def _draw_result_overlay(self, painter: QPainter):
        if self._result_head is None:
            return
        painter.setPen(QPen(QColor(30, 30, 30)))
        painter.setFont(QFont("Segoe UI", 6))
        if self._elevation is not None:
            rect1 = QRectF(-self._size/2, -self._size/2 - 26, self._size, 13)
            rect2 = QRectF(-self._size/2, -self._size/2 - 13, self._size, 13)
            painter.drawText(rect1, Qt.AlignmentFlag.AlignCenter,
                             f"H={self._result_head:.2f}m")
            painter.drawText(rect2, Qt.AlignmentFlag.AlignCenter,
                             f"z={self._elevation:.2f}m")
        else:
            rect = QRectF(-self._size/2, -self._size/2 - 14, self._size, 14)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"H={self._result_head:.2f}m")


class JunctionItem(NodeGraphicsItem):
    def __init__(self, node_id, signals, x=0, y=0):
        super().__init__(node_id, signals, x, y)
        self._size = 34.0

    def primary_color(self) -> QColor:
        return C["invalid"] if not self.is_valid else C["junction"]

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.primary_color()
        r = self._size / 2
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(140), 1.5))
        painter.drawEllipse(QPointF(0, 0), r, r)
        painter.setPen(QPen(C["text_light"], 2.0))
        offset = r * 0.45
        painter.drawLine(QPointF(-offset, 0), QPointF(offset, 0))
        painter.drawLine(QPointF(0, -offset), QPointF(0, offset))
        self._draw_selection_ring(painter)
        self._draw_label(painter, self.node_id)
        self._draw_result_overlay(painter)


class ReservoirItem(NodeGraphicsItem):
    def __init__(self, node_id, signals, x=0, y=0):
        super().__init__(node_id, signals, x, y)
        self._size = 60.0

    def primary_color(self) -> QColor:
        return C["invalid"] if not self.is_valid else C["reservoir"]

    def boundingRect(self) -> QRectF:
        w = self._size; h = self._size * 0.72
        return QRectF(-w/2 - PORT_R - 2, -h/2 - PORT_R - 2 - 20,
                       w + 2*(PORT_R+2), h + 2*(PORT_R+2) + 18 + 20)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.primary_color()
        w = self._size; h = self._size * 0.72
        x0, y0 = -w/2, -h/2

        painter.setBrush(QBrush(color.lighter(170)))
        painter.setPen(QPen(color.darker(130), 1.8))
        painter.drawRect(QRectF(x0, y0, w, h))

        water_top = y0 + h * 0.28; water_h = h * 0.55
        painter.setBrush(QBrush(QColor(30, 110, 220, 190)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(x0 + 2, water_top, w - 4, water_h))

        painter.setPen(QPen(QColor(90, 160, 255, 180), 1.5))
        for dx in (-8, 0, 8):
            painter.drawArc(QRectF(x0 + w/2 - 10 + dx, water_top - 4, 20, 8), 0, 180*16)

        painter.setPen(QPen(C["selected"] if self.isSelected() else color.darker(130), 1.8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(x0, y0, w, h))

        py = y0 + h
        painter.setBrush(QBrush(C["port"]))
        painter.setPen(QPen(C["port_border"], 1.2))
        painter.drawEllipse(QPointF(0, py), PORT_R, PORT_R)
        painter.drawEllipse(QPointF(x0, 0), PORT_R, PORT_R)
        painter.drawEllipse(QPointF(x0 + w, 0), PORT_R, PORT_R)

        self._draw_selection_ring(painter)
        self._draw_label(painter, self.node_id)
        self._draw_result_overlay(painter)


# ═══════════════════════════════════════════════════════════════════════════════
# Port item  (inlet / outlet of an inline component)
# ═══════════════════════════════════════════════════════════════════════════════

class PortItem(NodeGraphicsItem):
    """
    Inlet or outlet port attached to a PumpCanvasItem / ValveCanvasItem.

    • Not independently movable (fixed offset from parent).
    • Not selectable (clicking emits node_selected for pipe connection).
    • Parent InlineComponentItem handles all position-change notifications.
    """

    def __init__(self, node_id: str, signals: CanvasSignals,
                 offset_x: float, offset_y: float,
                 port_type: str, parent_item: QGraphicsItem):
        super().__init__(node_id, signals, 0, 0)
        # Disable independent movement and selection
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setPos(offset_x, offset_y)
        self.setParentItem(parent_item)
        self.port_type = port_type   # "inlet" | "outlet"
        self._size = 16.0

    def center(self) -> QPointF:
        return self.scenePos()

    def primary_color(self) -> QColor:
        return C["port_in"] if self.port_type == "inlet" else C["port_out"]

    def boundingRect(self) -> QRectF:
        r = PORT_R + 4
        return QRectF(-r, -r, 2*r, 2*r + 12)

    def itemChange(self, change, value):
        # Skip NodeGraphicsItem grid-snap + signal logic entirely;
        # parent InlineComponentItem handles all this.
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            return value
        # Call QGraphicsItem base (bypass NodeGraphicsItem)
        return super(NodeGraphicsItem, self).itemChange(change, value)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.primary_color()
        r = PORT_R + 1

        painter.setBrush(QBrush(QColor(255, 255, 255, 220)))
        painter.setPen(QPen(color, 2))
        painter.drawEllipse(QPointF(0, 0), r, r)

        painter.setPen(QPen(color.darker(150)))
        painter.setFont(QFont("Segoe UI", 4, QFont.Weight.Bold))
        label = "IN" if self.port_type == "inlet" else "OUT"
        painter.drawText(QRectF(-10, r + 1, 20, 9),
                         Qt.AlignmentFlag.AlignCenter, label)

        if self._hover:
            painter.setPen(QPen(C["selected"], 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(0, 0), r + 2, r + 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Inline component items (Pump / Valve freely placed on canvas)
# ═══════════════════════════════════════════════════════════════════════════════

class InlineComponentItem(QGraphicsItem):
    """
    Base class for a freely-placed pump or valve icon.

    Owns two PortItem children (inlet at -PORT_OFFSET, outlet at +PORT_OFFSET).
    When the body is moved, both ports follow and their canvas positions are
    broadcast via node_moved signals so the network model stays in sync.
    """

    BODY_R      = 20.0
    PORT_OFFSET = 36.0

    def __init__(self, edge_id: str,
                 inlet_id: str, outlet_id: str,
                 signals: CanvasSignals,
                 x: float = 0, y: float = 0):
        super().__init__()
        self.component_edge_id = edge_id
        self.signals   = signals
        self._display_name: Optional[str]  = None
        self._flow_rate: Optional[float]   = None
        self._hover  = False
        self.is_valid = True

        self.setPos(x, y)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.setZValue(2.5)

        self.inlet_port  = PortItem(inlet_id,  signals, -self.PORT_OFFSET, 0, "inlet",  self)
        self.outlet_port = PortItem(outlet_id, signals,  self.PORT_OFFSET, 0, "outlet", self)

    def set_display_name(self, name: str):
        self._display_name = name if name and name != self.component_edge_id else None
        self.update()

    def set_flow(self, Q: float):
        self._flow_rate = Q; self.update()

    def clear_flow(self):
        self._flow_rate = None; self.update()

    def boundingRect(self) -> QRectF:
        px = self.PORT_OFFSET + PORT_R + 6
        return QRectF(-px, -self.BODY_R - 6,
                      2 * px, self.BODY_R + 6 + PORT_R + 14 + 18)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            x = round(value.x() / GRID_MINOR) * GRID_MINOR
            y = round(value.y() / GRID_MINOR) * GRID_MINOR
            return QPointF(x, y)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            sc = self.scene()
            if sc:
                sc.refresh_edges_for_node(self.inlet_port.node_id)
                sc.refresh_edges_for_node(self.outlet_port.node_id)
            in_sp  = self.inlet_port.scenePos()
            out_sp = self.outlet_port.scenePos()
            self.signals.node_moved.emit(self.inlet_port.node_id,  in_sp.x(),  in_sp.y())
            self.signals.node_moved.emit(self.outlet_port.node_id, out_sp.x(), out_sp.y())
        return super().itemChange(change, value)

    def _draw_symbol(self, painter: QPainter):
        pass   # implemented by subclasses

    def _draw_label(self, painter: QPainter):
        label = self._display_name or self.component_edge_id
        painter.setPen(QPen(C["text_dark"]))
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        y = self.BODY_R + PORT_R + 10
        painter.drawText(QRectF(-self.PORT_OFFSET, y, 2*self.PORT_OFFSET, 14),
                         Qt.AlignmentFlag.AlignCenter, label)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_symbol(painter)
        self._draw_label(painter)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = self.pos()
            self.signals.edge_selected.emit(self.component_edge_id)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            start = getattr(self, '_drag_start_pos', None)
            if start is not None:
                end = self.pos()
                if end != start:
                    self.signals.inline_move_finished.emit(
                        self.component_edge_id,
                        start.x(), start.y(),
                        end.x(), end.y(),
                    )
                self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    def hoverEnterEvent(self, event):
        self._hover = True;  self.update(); super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hover = False; self.update(); super().hoverLeaveEvent(event)

    def contextMenuEvent(self, event):
        menu    = QMenu()
        del_act = menu.addAction(f"Delete '{self.component_edge_id}'")
        action  = menu.exec(event.screenPos())
        if action == del_act:
            self.signals.delete_requested.emit(self.component_edge_id)


class PumpCanvasItem(InlineComponentItem):
    """Freely-placed centrifugal pump icon with inlet/outlet ports."""

    def _draw_symbol(self, painter: QPainter):
        color  = C["invalid"] if not self.is_valid else C["pump"]
        is_sel = self.isSelected() or self._hover

        # Connecting lines body → ports
        painter.setPen(QPen(color.darker(120), 3,
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(-self.BODY_R, 0), QPointF(-self.PORT_OFFSET, 0))
        painter.drawLine(QPointF( self.BODY_R, 0), QPointF( self.PORT_OFFSET, 0))

        # Body circle
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(C["selected"] if is_sel else color.darker(140),
                            2.5 if is_sel else 1.5))
        painter.drawEllipse(QPointF(0, 0), self.BODY_R, self.BODY_R)

        # Impeller triangle
        r = self.BODY_R * 0.55
        path = QPainterPath()
        path.moveTo(-r, 0)
        path.lineTo(r * 0.65, -r * 0.75)
        path.lineTo(r * 0.65,  r * 0.75)
        path.closeSubpath()
        painter.fillPath(path, QBrush(C["text_light"]))


class ValveCanvasItem(InlineComponentItem):
    """Freely-placed valve icon (bowtie) with inlet/outlet ports."""

    def _draw_symbol(self, painter: QPainter):
        color  = C["invalid"] if not self.is_valid else C["valve"]
        is_sel = self.isSelected() or self._hover
        bw     = self.BODY_R          # half-width of bowtie

        # Connecting lines body → ports
        painter.setPen(QPen(color.darker(120), 3,
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(-bw - 2, 0), QPointF(-self.PORT_OFFSET, 0))
        painter.drawLine(QPointF( bw + 2, 0), QPointF( self.PORT_OFFSET, 0))

        # Bowtie halves
        painter.setPen(QPen(C["selected"] if is_sel else color.darker(140),
                            2.5 if is_sel else 1.5))
        painter.setBrush(QBrush(color))
        hw = bw * 0.7
        poly1 = QPolygonF([QPointF(-bw, -hw), QPointF(0, 0), QPointF(-bw,  hw)])
        poly2 = QPolygonF([QPointF( bw, -hw), QPointF(0, 0), QPointF( bw,  hw)])
        painter.drawPolygon(poly1)
        painter.drawPolygon(poly2)

        # Stem
        painter.setPen(QPen(color.darker(160), 2))
        painter.drawLine(QPointF(0, 0), QPointF(0, -(bw + 6)))
        painter.drawLine(QPointF(-4, -(bw + 6)), QPointF(4, -(bw + 6)))


# ═══════════════════════════════════════════════════════════════════════════════
# Edge graphics items
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeGraphicsItem(QGraphicsItem):
    """Directed edge drawn between two NodeGraphicsItems (or PortItems)."""

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
        self._display_name: Optional[str] = None

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)
        self._hover = False

    def set_display_name(self, name: str):
        self._display_name = name if name and name != self.edge_id else None
        self.update()

    def set_flow(self, Q: float):
        self._flow_rate = Q; self.update()

    def clear_flow(self):
        self._flow_rate = None; self.update()

    def _endpoints(self) -> tuple[QPointF, QPointF]:
        return self.from_item.center(), self.to_item.center()

    def _midpoint(self) -> QPointF:
        p1, p2 = self._endpoints()
        return QPointF((p1.x()+p2.x())/2, (p1.y()+p2.y())/2)

    def _angle_rad(self) -> float:
        p1, p2 = self._endpoints()
        return math.atan2(p2.y()-p1.y(), p2.x()-p1.x())

    def _line_color(self) -> QColor:
        if not self.is_valid:
            return C["invalid"]
        if self._flow_rate is not None:
            max_q = 0.02
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
            abs(p2.x()-p1.x()) + 2*pad, abs(p2.y()-p1.y()) + 2*pad)

    def _draw_line_and_arrow(self, painter: QPainter,
                              color_override: Optional[QColor] = None):
        p1, p2 = self._endpoints()
        lc = color_override or self._line_color()
        lw = 2.5 if self.isSelected() else 2.0
        pen = QPen(C["selected"] if self.isSelected() else lc, lw)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(p1, p2)

        # Directional arrow at 65 % along the line
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
        mid  = self._midpoint()
        ang  = self._angle_rad()
        perp_x = -math.sin(ang) * 18
        perp_y =  math.cos(ang) * 18
        painter.setPen(QPen(C["text_dark"]))
        painter.setFont(QFont("Consolas", 6))
        Q_Ls = self._flow_rate * 1000.0
        painter.drawText(QRectF(mid.x()+perp_x-28, mid.y()+perp_y-7, 56, 14),
                         Qt.AlignmentFlag.AlignCenter, f"{Q_Ls:.2f} L/s")

    def _draw_edge_label(self, painter: QPainter):
        mid  = self._midpoint()
        ang  = self._angle_rad()
        perp_x = -math.sin(ang) * 14
        perp_y =  math.cos(ang) * 14
        painter.setPen(QPen(C["text_dark"], 1))
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        display = self._display_name if self._display_name else self.edge_id
        painter.drawText(QRectF(mid.x()+perp_x-25, mid.y()+perp_y-14, 50, 12),
                         Qt.AlignmentFlag.AlignCenter, display)

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
        self._hover = True;  self.update(); super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hover = False; self.update(); super().hoverLeaveEvent(event)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_line_and_arrow(painter)
        self._draw_symbol(painter)
        self._draw_edge_label(painter)
        self._draw_flow_label(painter)

    def _draw_symbol(self, painter: QPainter):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Fitting icon item  (standalone, fully interactive scene item)
# ═══════════════════════════════════════════════════════════════════════════════

class FittingIconItem(QGraphicsItem):
    """
    A small interactive diamond icon representing one FittingAttachment on a pipe.

    Each fitting gets its own scene item so that:
    - Every fitting is independently clickable (fixes multi-fitting selection)
    - Each has its own bounding box and hover area
    - z-value (1.5) keeps fittings above the pipe line (1) but below nodes (2)
    """

    RADIUS = 8.0   # half-size of the diamond
    HIT_R  = 12.0  # click hit radius

    def __init__(self, fitting_id: str, pipe_edge_id: str,
                 signals: CanvasSignals):
        super().__init__()
        self.fitting_id   = fitting_id
        self.pipe_edge_id = pipe_edge_id
        self.signals      = signals
        self._hover       = False

        self.setAcceptHoverEvents(True)
        self.setZValue(1.5)
        # Enable standard selection
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

    def boundingRect(self) -> QRectF:
        r = self.HIT_R + 2
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        sz   = self.RADIUS
        # Use isSelected() instead of custom flag
        fill = C["selected"] if (self.isSelected() or self._hover) else C["fitting"]
        painter.setPen(QPen(fill.darker(140), 1.5))
        painter.setBrush(QBrush(fill))
        poly = QPolygonF([
            QPointF(0, -sz), QPointF(sz, 0),
            QPointF(0,  sz), QPointF(-sz, 0),
        ])
        painter.drawPolygon(poly)
        painter.setPen(QPen(C["text_light"]))
        painter.setFont(QFont("Segoe UI", 4, QFont.Weight.Bold))
        painter.drawText(QRectF(-6, -5, 12, 10), Qt.AlignmentFlag.AlignCenter, "K")

    def mousePressEvent(self, event):
        # Let standard selection logic work (selection handled by scene)
        super().mousePressEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            if value:
                self.signals.fitting_selected.emit(self.pipe_edge_id, self.fitting_id)
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):
        self._hover = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hover = False
        self.update()
        super().hoverLeaveEvent(event)

    def contextMenuEvent(self, event):
        menu    = QMenu()
        del_act = menu.addAction(f"Remove fitting '{self.fitting_id}'")
        action  = menu.exec(event.screenPos())
        if action == del_act:
            self.signals.delete_requested.emit(
                f"fitting::{self.pipe_edge_id}::{self.fitting_id}")


class PipeEdgeItem(EdgeGraphicsItem):
    """
    Pipe rendered as a straight arrow line.

    Fitting attachments are represented as independent FittingIconItem scene items
    positioned along the pipe.  Each fitting is fully clickable / hoverable.
    """

    def __init__(self, edge_id, from_item, to_item, signals):
        super().__init__(edge_id, "pipe", from_item, to_item, signals)
        self._fitting_attachments: List[FittingAttachment] = []
        self._fitting_items:       List[FittingIconItem]   = []
        self._fitting_highlight: bool = False

    def set_fittings(self, fittings: List[FittingAttachment]):
        """Replace all fitting icons.  Manages scene membership automatically."""
        sc = self.scene()
        for fi in self._fitting_items:
            if fi.scene():
                fi.scene().removeItem(fi)
        self._fitting_items.clear()

        self._fitting_attachments = list(fittings)

        for fa in self._fitting_attachments:
            fi = FittingIconItem(fa.fitting_id, self.edge_id, self.signals)
            self._fitting_items.append(fi)
            if sc:
                sc.addItem(fi)

        self._reposition_fittings()
        self.update()

    def _reposition_fittings(self):
        """Move each FittingIconItem to its position along the pipe."""
        if not self._fitting_items:
            return
        try:
            p1, p2 = self._endpoints()
        except Exception:
            return
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        for fi, fa in zip(self._fitting_items, self._fitting_attachments):
            fi.setPos(p1.x() + fa.position_t * dx,
                      p1.y() + fa.position_t * dy)

    def prepareGeometryChange(self):
        """Also reposition fitting icons whenever the pipe geometry changes."""
        super().prepareGeometryChange()
        self._reposition_fittings()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSceneChange:
            new_scene = value
            old_scene = self.scene()
            if old_scene and new_scene is None:
                # Being removed from scene — take fitting items with us
                for fi in self._fitting_items:
                    old_scene.removeItem(fi)
            elif new_scene and old_scene is None:
                # Being re-added to a scene — add fitting items too
                for fi in self._fitting_items:
                    new_scene.addItem(fi)
        return super().itemChange(change, value)

    def _draw_symbol(self, painter: QPainter):
        pass   # No midpoint box icon — just the arrow line

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        lc = C["fitting_hl"] if self._fitting_highlight else None
        self._draw_line_and_arrow(painter, color_override=lc)
        self._draw_edge_label(painter)
        self._draw_flow_label(painter)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Deselect all fittings on this pipe and show pipe properties
            for fi in self._fitting_items:
                fi._selected = False
                fi.update()
            self.signals.edge_selected.emit(self.edge_id)
        super(EdgeGraphicsItem, self).mousePressEvent(event)

    def contextMenuEvent(self, event):
        super().contextMenuEvent(event)


# Legacy edge items kept for backward compat (old-format files where pump/valve
# were placed between two regular junction nodes, not as inline components).

class PumpEdgeItem(EdgeGraphicsItem):
    def __init__(self, edge_id, from_item, to_item, signals):
        super().__init__(edge_id, "pump", from_item, to_item, signals)

    def _draw_symbol(self, painter: QPainter):
        mid = self._midpoint(); ang = self._angle_rad()
        painter.save()
        painter.translate(mid); painter.rotate(math.degrees(ang))
        color = C["pump"]
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(140), 1.5))
        painter.drawEllipse(QPointF(0, 0), 14, 14)
        path = QPainterPath()
        path.moveTo(-7, 0); path.lineTo(5, -6); path.lineTo(5, 6)
        path.closeSubpath()
        painter.fillPath(path, QBrush(C["text_light"]))
        painter.restore()


class ValveEdgeItem(EdgeGraphicsItem):
    def __init__(self, edge_id, from_item, to_item, signals):
        super().__init__(edge_id, "valve", from_item, to_item, signals)

    def _draw_symbol(self, painter: QPainter):
        mid = self._midpoint(); ang = self._angle_rad()
        painter.save()
        painter.translate(mid); painter.rotate(math.degrees(ang))
        color = C["valve"]
        painter.setPen(QPen(color.darker(140), 1.5))
        painter.setBrush(QBrush(color))
        poly1 = QPolygonF([QPointF(-13, -8), QPointF(0, 0), QPointF(-13, 8)])
        poly2 = QPolygonF([QPointF(13,  -8), QPointF(0, 0), QPointF(13,  8)])
        painter.drawPolygon(poly1); painter.drawPolygon(poly2)
        painter.setPen(QPen(color.darker(170), 2))
        painter.drawLine(QPointF(0, 0), QPointF(0, -14))
        painter.drawLine(QPointF(-4, -14), QPointF(4, -14))
        painter.restore()


_EDGE_ITEM_CLASSES = {
    "pipe":  PipeEdgeItem,
    "pump":  PumpEdgeItem,
    "valve": ValveEdgeItem,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Temp connection line
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

    Dictionaries maintained:
        node_items   : Dict[node_id, NodeGraphicsItem]
        edge_items   : Dict[edge_id, EdgeGraphicsItem]
        inline_items : Dict[edge_id, InlineComponentItem]   ← pumps / valves
        port_items   : Dict[node_id, PortItem]              ← inlet / outlet ports
    """

    def __init__(self, signals: CanvasSignals, parent=None):
        super().__init__(parent)
        self.signals = signals

        self.node_items:   Dict[str, NodeGraphicsItem]     = {}
        self.edge_items:   Dict[str, EdgeGraphicsItem]     = {}
        self.inline_items: Dict[str, InlineComponentItem]  = {}
        self.port_items:   Dict[str, PortItem]             = {}

        self._connecting         = False
        self._conn_source:       Optional[str]   = None
        self._temp_line:         Optional[TempConnectionLine] = None
        self._pending_edge_type: Optional[str]   = None
        self._fitting_mode:      bool            = False

        self.max_flow: float = 0.02

        self.setSceneRect(-3000, -3000, 6000, 6000)
        self._draw_grid()

    # ── Grid ─────────────────────────────────────────────────────────────────

    def _draw_grid(self):
        pen_minor = QPen(C["grid_line"],  0.5)
        pen_major = QPen(C["grid_major"], 0.8)
        for x in range(-3000, 3001, GRID_MINOR):
            self.addLine(x, -3000, x, 3000,
                         pen_major if x % GRID_MAJOR == 0 else pen_minor)
        for y in range(-3000, 3001, GRID_MINOR):
            self.addLine(-3000, y, 3000, y,
                         pen_major if y % GRID_MAJOR == 0 else pen_minor)

    # ── Node / edge management ────────────────────────────────────────────────

    def _lookup_node_item(self, node_id: str) -> Optional[NodeGraphicsItem]:
        """Find a node/port item by ID (checks both dicts)."""
        return self.node_items.get(node_id) or self.port_items.get(node_id)

    def add_node(self, node_id: str, node_type: str,
                 x: float = 0, y: float = 0) -> NodeGraphicsItem:
        cls  = JunctionItem if node_type == "junction" else ReservoirItem
        item = cls(node_id, self.signals, x, y)
        self.addItem(item)
        self.node_items[node_id] = item
        return item

    def add_edge(self, edge_id: str, edge_type: str,
                 from_id: str, to_id: str) -> Optional[EdgeGraphicsItem]:
        from_item = self._lookup_node_item(from_id)
        to_item   = self._lookup_node_item(to_id)
        if from_item is None or to_item is None:
            return None
        cls  = _EDGE_ITEM_CLASSES.get(edge_type, PipeEdgeItem)
        item = cls(edge_id, from_item, to_item, self.signals)
        self.addItem(item)
        self.edge_items[edge_id] = item
        return item

    def add_inline_component(self, edge_id: str, comp_type: str,
                              inlet_id: str, outlet_id: str,
                              x: float, y: float) -> InlineComponentItem:
        """Create and register a PumpCanvasItem or ValveCanvasItem."""
        cls  = PumpCanvasItem if comp_type == "pump" else ValveCanvasItem
        item = cls(edge_id, inlet_id, outlet_id, self.signals, x, y)
        self.addItem(item)
        self.inline_items[edge_id]   = item
        self.port_items[inlet_id]    = item.inlet_port
        self.port_items[outlet_id]   = item.outlet_port
        return item

    def remove_component(self, comp_id: str):
        if comp_id in self.node_items:
            self.removeItem(self.node_items.pop(comp_id))
        elif comp_id in self.edge_items:
            self.removeItem(self.edge_items.pop(comp_id))

    def remove_inline_component(self, edge_id: str):
        if edge_id not in self.inline_items:
            return
        item = self.inline_items.pop(edge_id)
        self.port_items.pop(item.inlet_port.node_id, None)
        self.port_items.pop(item.outlet_port.node_id, None)
        self.removeItem(item)   # children (ports) are removed automatically

    def mark_validity(self, comp_id: str, valid: bool):
        item = (self.node_items.get(comp_id)
                or self.edge_items.get(comp_id)
                or self.inline_items.get(comp_id))
        if item:
            item.is_valid = valid
            item.update()

    def refresh_edges_for_node(self, node_id: str):
        for ei in self.edge_items.values():
            if (hasattr(ei.from_item, 'node_id') and ei.from_item.node_id == node_id
                    or hasattr(ei.to_item, 'node_id') and ei.to_item.node_id == node_id):
                ei.prepareGeometryChange()
                ei.update()

    # ── Fitting highlight (used during fitting-placement mode) ────────────────

    def set_fitting_mode(self, active: bool):
        self._fitting_mode = active
        if not active:
            self._clear_pipe_highlight()

    def _clear_pipe_highlight(self):
        for item in self.edge_items.values():
            if isinstance(item, PipeEdgeItem):
                item._fitting_highlight = False
                item.update()

    def _highlight_pipe_at(self, scene_pos: QPointF):
        items = self.items(scene_pos)
        pipe  = next((i for i in items if isinstance(i, PipeEdgeItem)), None)
        for eid, item in self.edge_items.items():
            if isinstance(item, PipeEdgeItem):
                new_hl = (item is pipe)
                if item._fitting_highlight != new_hl:
                    item._fitting_highlight = new_hl
                    item.update()

    # ── Solver results ────────────────────────────────────────────────────────

    def apply_results(self, heads: dict, flows: dict, elevations: dict = None):
        if flows:
            self.max_flow = max(abs(q) for q in flows.values()) or 0.02
            self.max_flow = max(self.max_flow, 1e-6)
        for nid, H in heads.items():
            if nid in self.node_items:
                self.node_items[nid].set_result(H)
        for eid, Q in flows.items():
            if eid in self.edge_items:
                self.edge_items[eid].set_flow(Q)
            elif eid in self.inline_items:
                self.inline_items[eid].set_flow(Q)
        if elevations:
            for nid, z in elevations.items():
                if nid in self.node_items:
                    self.node_items[nid].set_elevation(z)

    def clear_results(self):
        self.max_flow = 0.02
        for item in self.node_items.values():
            item.clear_result()
        for item in self.edge_items.values():
            item.clear_flow()
        for item in self.inline_items.values():
            item.clear_flow()

    # ── Connection mode ───────────────────────────────────────────────────────

    def begin_connection(self, source_id: str, edge_type: str = "pipe"):
        self._connecting        = True
        self._conn_source       = source_id
        self._pending_edge_type = edge_type
        src = self._lookup_node_item(source_id)
        if src:
            p = src.center()
            self._temp_line = TempConnectionLine()
            self._temp_line.setLine(p.x(), p.y(), p.x(), p.y())
            self.addItem(self._temp_line)

    def _abort_connection(self):
        self._connecting        = False
        self._conn_source       = None
        self._pending_edge_type = None
        if self._temp_line:
            self.removeItem(self._temp_line)
            self._temp_line = None

    # ── Mouse events ──────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event):
        pos = event.scenePos()

        if self._fitting_mode:
            self._highlight_pipe_at(pos)

        if self._connecting and self._temp_line:
            src = self._lookup_node_item(self._conn_source)
            if src:
                p = src.center()
                self._temp_line.setLine(p.x(), p.y(), pos.x(), pos.y())

        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        pos = event.scenePos()

        # ── Fitting placement mode ────────────────────────────────────────
        if self._fitting_mode and event.button() == Qt.MouseButton.LeftButton:
            items = self.items(pos)
            pipe  = next((i for i in items if isinstance(i, PipeEdgeItem)), None)
            if pipe:
                p1, p2 = pipe._endpoints()
                dx = p2.x() - p1.x(); dy = p2.y() - p1.y()
                lsq = dx*dx + dy*dy
                t = (((pos.x()-p1.x())*dx + (pos.y()-p1.y())*dy) / lsq
                     if lsq > 0 else 0.5)
                t = max(0.1, min(0.9, t))
                self.signals.fitting_placement_requested.emit(pipe.edge_id, t)
            # Stay in fitting mode; MainWindow exits after each attachment
            return

        # ── Connection mode ───────────────────────────────────────────────
        if self._connecting and event.button() == Qt.MouseButton.LeftButton:
            items  = self.items(pos)
            target = next((i for i in items
                           if isinstance(i, NodeGraphicsItem)
                           and not isinstance(i, PortItem)   # skip ports as target nodes — handled via scenePos
                           or isinstance(i, PortItem)),
                          None)
            # Re-filter: accept both regular nodes and ports
            target = next((i for i in items if isinstance(i, NodeGraphicsItem)), None)
            if target and target.node_id != self._conn_source:
                from_id   = self._conn_source
                to_id     = target.node_id
                edge_type = self._pending_edge_type or "pipe"
                self._abort_connection()
                self.signals.connection_requested.emit(edge_type, from_id, to_id)
            else:
                self._abort_connection()
                self.signals.nothing_selected.emit()
            return

        super().mousePressEvent(event)

        if not self.selectedItems():
            self.signals.nothing_selected.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._abort_connection()
            if self._fitting_mode:
                self._fitting_mode = False
                self._clear_pipe_highlight()
            self.signals.escape_pressed.emit()
            self.signals.nothing_selected.emit()
        super().keyPressEvent(event)


# ═══════════════════════════════════════════════════════════════════════════════
# View
# ═══════════════════════════════════════════════════════════════════════════════

class ThermofluidView(QGraphicsView):
    """
    QGraphicsView wrapping ThermofluidCanvas.
    Adds: smooth zoom (Ctrl+scroll), pan (middle-mouse), placement mode.
    """

    placement_requested = pyqtSignal(str, float, float)   # type, scene_x, scene_y

    def __init__(self, scene: ThermofluidCanvas, parent=None):
        super().__init__(scene, parent)
        self._scene           = scene
        self._placement_mode: Optional[str] = None
        self._fitting_mode    = False

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

    def set_placement_mode(self, component_type: Optional[str]):
        self._placement_mode = component_type
        if component_type:
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

    def set_fitting_mode(self, active: bool):
        """Enter / leave fitting-attachment mode."""
        self._fitting_mode = active
        self._scene.set_fitting_mode(active)
        if active:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if (self._placement_mode
                and event.button() == Qt.MouseButton.LeftButton
                and not self._scene._connecting
                and not self._scene._fitting_mode):
            scene_pos = self.mapToScene(event.pos())
            comp_type = self._placement_mode
            self.set_placement_mode(None)
            self.placement_requested.emit(comp_type, scene_pos.x(), scene_pos.y())
            return

        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            super().mousePressEvent(event)
            return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)

    # ── Zoom helpers ──────────────────────────────────────────────────────────

    def zoom_fit(self):
        # Exclude grid lines (QGraphicsLineItem) and fitting icons
        # (FittingIconItem positions are included via their parent pipe's bounding rect)
        items = [i for i in self._scene.items()
                 if not isinstance(i, (QGraphicsLineItem, FittingIconItem))]
        if not items:
            # Empty canvas — show a modest centred region
            self.fitInView(QRectF(-200, -200, 400, 400),
                           Qt.AspectRatioMode.KeepAspectRatio)
            return

        # Build a bounding rect from each item's scene-space bounding rect
        xs_min = xs_max = ys_min = ys_max = None
        for item in items:
            sbr = item.mapToScene(item.boundingRect()).boundingRect()
            if xs_min is None:
                xs_min, xs_max = sbr.left(), sbr.right()
                ys_min, ys_max = sbr.top(),  sbr.bottom()
            else:
                xs_min = min(xs_min, sbr.left())
                xs_max = max(xs_max, sbr.right())
                ys_min = min(ys_min, sbr.top())
                ys_max = max(ys_max, sbr.bottom())

        if xs_min is None or xs_max <= xs_min or ys_max <= ys_min:
            return

        w = xs_max - xs_min
        h = ys_max - ys_min
        pad_x = max(w * 0.08, 40)
        pad_y = max(h * 0.08, 40)
        rect = QRectF(xs_min - pad_x, ys_min - pad_y,
                      w + 2 * pad_x, h + 2 * pad_y)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_reset(self):
        self.resetTransform()
