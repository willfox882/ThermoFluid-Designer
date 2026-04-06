"""
sidebar.py
----------
Dynamic properties panel.  Rebuilds its form fields whenever a different
component is selected on the canvas.

Supports all component types:
  Reservoir  → total_head
  Junction   → elevation, demand
  Pipe       → material, condition, diameter, length, roughness, K_minor
  Pump       → A, B, C, diameter, is_on
  Valve      → diameter, K, is_open
  Fitting    → subtype, connection, nominal diameter, K (auto+override)
"""

from __future__ import annotations
from typing import Optional, Callable, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QDoubleSpinBox, QCheckBox,
    QPushButton, QGroupBox, QScrollArea, QSizePolicy,
    QFrame, QSpacerItem, QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette

from components import (
    FluidComponent, Pipe, Pump, Valve, Junction, Reservoir, Fitting,
)
from fluid_props import (
    MATERIAL_CONDITIONS, ROUGHNESS_TABLE, lookup_roughness,
    FITTING_CATEGORIES, FITTING_DIAMETERS, FITTING_K,
    lookup_fitting_k, NOMINAL_TO_METRES,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_spinbox(minimum: float, maximum: float, value: float,
                  decimals: int = 4, step: float = None) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(minimum, maximum)
    sb.setDecimals(decimals)
    sb.setValue(value)
    if step:
        sb.setSingleStep(step)
    sb.setMinimumWidth(110)
    return sb


def _section(title: str) -> QGroupBox:
    gb = QGroupBox(title)
    gb.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
    return gb


# ═══════════════════════════════════════════════════════════════════════════════

class PropertiesPanel(QWidget):
    """
    Right-hand properties panel.

    Signals
    -------
    apply_requested(comp_id, prop_dict)  — emitted when user hits Apply
    """

    apply_requested = pyqtSignal(str, dict)   # component_id, new_params

    def __init__(self, parent=None):
        super().__init__(parent)
        self._comp_id: Optional[str]       = None
        self._comp_type: Optional[str]     = None
        self._fields: dict[str, QWidget]   = {}

        self._build_ui()

    # ── Layout skeleton ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── Header ────────────────────────────────────────────────────
        hdr = QLabel("Properties")
        hdr.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.setStyleSheet(
            "background:#2d3040; color:#e8eaf0; "
            "padding:6px; border-radius:4px;")
        root.addWidget(hdr)

        # ── Type/ID badge ─────────────────────────────────────────────
        badge_row = QHBoxLayout()
        self._type_label = QLabel("—")
        self._type_label.setFont(QFont("Segoe UI", 8))
        self._type_label.setStyleSheet(
            "background:#4a5060; color:#b0c4de; "
            "padding:3px 8px; border-radius:3px;")
        self._id_label = QLabel("")
        self._id_label.setFont(QFont("Consolas", 8))
        badge_row.addWidget(self._type_label)
        badge_row.addStretch()
        badge_row.addWidget(self._id_label)
        root.addLayout(badge_row)

        # ── Scrollable form area ──────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._form_widget = QWidget()
        self._form_layout = QVBoxLayout(self._form_widget)
        self._form_layout.setContentsMargins(0, 0, 0, 0)
        self._form_layout.setSpacing(8)
        scroll.setWidget(self._form_widget)
        root.addWidget(scroll, stretch=1)

        # ── Apply button ──────────────────────────────────────────────
        self._apply_btn = QPushButton("Apply Changes")
        self._apply_btn.setEnabled(False)
        self._apply_btn.setStyleSheet(
            "QPushButton { background:#3a7bd5; color:white; "
            "  padding:6px; border-radius:4px; font-weight:bold; }"
            "QPushButton:hover { background:#5090e8; }"
            "QPushButton:disabled { background:#666; color:#999; }"
        )
        self._apply_btn.clicked.connect(self._on_apply)
        root.addWidget(self._apply_btn)

        # ── Validation / result section ───────────────────────────────
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setFont(QFont("Segoe UI", 7))
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        root.addWidget(self._status_label)

        # Results display
        self._result_box = _section("Solver Results")
        res_layout = QFormLayout()
        res_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        res_layout.setSpacing(4)
        self._res_labels: dict[str, QLabel] = {}
        for key in ("Head (m)", "Flow (L/s)", "Velocity (m/s)",
                    "Reynolds", "f (Darcy)", "ΔH (m)"):
            lbl = QLabel("—")
            lbl.setFont(QFont("Consolas", 8))
            res_layout.addRow(key + ":", lbl)
            self._res_labels[key] = lbl
        self._result_box.setLayout(res_layout)
        self._result_box.setVisible(False)
        root.addWidget(self._result_box)

        self._empty_label = QLabel(
            "Select a component\non the canvas\nto view properties.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color:#888; font-style:italic;")
        self._form_layout.addWidget(self._empty_label)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_component(self, comp: FluidComponent):
        """Populate form for the given component."""
        self._comp_id   = comp.id
        self._comp_type = type(comp).__name__

        self._type_label.setText(self._comp_type)
        self._id_label.setText(comp.id)
        self._apply_btn.setEnabled(True)
        self._status_label.setText("")

        self._clear_form()
        self._fields.clear()

        builder = {
            "Reservoir": self._build_reservoir_form,
            "Junction":  self._build_junction_form,
            "Pipe":      self._build_pipe_form,
            "Pump":      self._build_pump_form,
            "Valve":     self._build_valve_form,
            "Fitting":   self._build_fitting_form,
        }.get(self._comp_type)

        if builder:
            builder(comp)

        self._form_layout.addStretch()

    def clear_selection(self):
        self._comp_id   = None
        self._comp_type = None
        self._type_label.setText("—")
        self._id_label.setText("")
        self._apply_btn.setEnabled(False)
        self._status_label.setText("")
        self._result_box.setVisible(False)
        self._clear_form()
        self._form_layout.addWidget(self._empty_label)
        self._empty_label.setVisible(True)

    def show_validation_error(self, errors: list[str]):
        if errors:
            txt = "\n".join(f"⚠ {e}" for e in errors)
            self._status_label.setStyleSheet("color:#e05050;")
            self._status_label.setText(txt)
        else:
            self._status_label.setStyleSheet("color:#50b050;")
            self._status_label.setText("✓ Network valid")

    def show_results(self, result_dict: dict):
        """Display post-solve results for the selected component."""
        self._result_box.setVisible(True)
        mapping = {
            "Head (m)":      result_dict.get("head"),
            "Flow (L/s)":    _fmt(result_dict.get("flow"), scale=1000),
            "Velocity (m/s)":result_dict.get("velocity"),
            "Reynolds":      result_dict.get("reynolds"),
            "f (Darcy)":     result_dict.get("friction_factor"),
            "ΔH (m)":        result_dict.get("head_loss"),
        }
        for key, val in mapping.items():
            lbl = self._res_labels[key]
            if val is None:
                lbl.setText("—")
            elif key == "Reynolds":
                lbl.setText(f"{val:,.0f}")
            else:
                lbl.setText(f"{val:.4f}")

    def hide_results(self):
        self._result_box.setVisible(False)
        for lbl in self._res_labels.values():
            lbl.setText("—")

    # ── Form builders ─────────────────────────────────────────────────────────

    def _clear_form(self):
        while self._form_layout.count():
            item = self._form_layout.takeAt(0)
            w = item.widget()
            if w and w is not self._empty_label:
                w.deleteLater()
        # _empty_label is removed from layout by takeAt but NOT destroyed
        # clear_selection() will re-add it when needed

    def _add_group(self, title: str, rows: list[tuple]) -> QGroupBox:
        """Add a QGroupBox with a QFormLayout to the panel."""
        gb     = _section(title)
        layout = QFormLayout()
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        layout.setSpacing(6)
        for label, widget in rows:
            layout.addRow(label, widget)
        gb.setLayout(layout)
        self._form_layout.addWidget(gb)
        return gb

    def _build_reservoir_form(self, comp: Reservoir):
        self._empty_label.setVisible(False)
        sb = _make_spinbox(0, 10000, comp.total_head, 3, 1.0)
        sb.setSuffix(" m")
        self._fields["total_head"] = sb
        self._add_group("Reservoir", [("Total head:", sb)])

    def _build_junction_form(self, comp: Junction):
        self._empty_label.setVisible(False)
        elev = _make_spinbox(-1000, 10000, comp.elevation, 3, 0.5)
        elev.setSuffix(" m")
        dem  = _make_spinbox(-1, 1, comp.demand, 6, 0.0001)
        dem.setSuffix(" m³/s")
        self._fields["elevation"] = elev
        self._fields["demand"]    = dem
        self._add_group("Junction", [
            ("Elevation:", elev),
            ("Demand:", dem),
        ])

    def _build_pipe_form(self, comp: Pipe):
        self._empty_label.setVisible(False)

        # Material / Condition dropdowns
        mat_combo = QComboBox()
        mat_combo.addItems(list(MATERIAL_CONDITIONS.keys()))
        mat_combo.setCurrentText(comp.material)
        mat_combo.setMinimumWidth(110)

        cond_combo = QComboBox()
        self._update_condition_combo(cond_combo, comp.material)
        cond_combo.setCurrentText(comp.condition)
        cond_combo.setMinimumWidth(110)

        # When material changes → update condition list and roughness
        rough = _make_spinbox(0, 0.1, comp.roughness, 7, 0.000001)
        rough.setSuffix(" m")

        def _on_material_changed(mat_text):
            self._update_condition_combo(cond_combo, mat_text)
            cond = cond_combo.currentText()
            r = lookup_roughness(mat_text, cond)
            rough.setValue(r)

        def _on_condition_changed(cond_text):
            mat = mat_combo.currentText()
            r = lookup_roughness(mat, cond_text)
            rough.setValue(r)

        mat_combo.currentTextChanged.connect(_on_material_changed)
        cond_combo.currentTextChanged.connect(_on_condition_changed)

        diam  = _make_spinbox(0.001, 5.0, comp.diameter, 4, 0.01)
        diam.setSuffix(" m")
        leng  = _make_spinbox(0, 100000, comp.length, 2, 10.0)
        leng.setSuffix(" m")
        elev  = _make_spinbox(-5000, 5000, comp.elevation_change, 3, 0.5)
        elev.setSuffix(" m")
        Km    = _make_spinbox(0, 1000, comp.K_minor, 3, 0.1)

        self._fields.update({
            "material": mat_combo, "condition": cond_combo,
            "diameter": diam, "length": leng, "roughness": rough,
            "elevation_change": elev, "K_minor": Km,
        })
        self._add_group("Pipe Material", [
            ("Material:", mat_combo),
            ("Condition:", cond_combo),
            ("Roughness ε:", rough),
        ])
        self._add_group("Pipe Geometry", [
            ("Diameter D:", diam),
            ("Length L:", leng),
        ])
        self._add_group("Head Loss", [
            ("Elev. change Δz:", elev),
            ("Minor loss ΣK:", Km),
        ])

    @staticmethod
    def _update_condition_combo(combo: QComboBox, material: str):
        combo.blockSignals(True)
        combo.clear()
        conditions = MATERIAL_CONDITIONS.get(material, [])
        combo.addItems(conditions)
        combo.blockSignals(False)

    def _build_fitting_form(self, comp: Fitting):
        self._empty_label.setVisible(False)

        # Connection type
        conn_combo = QComboBox()
        conn_combo.addItems(["Screwed", "Flanged"])
        conn_combo.setCurrentText(comp.connection_type)
        conn_combo.setMinimumWidth(110)

        # Fitting subtype — flat list from all categories
        sub_combo = QComboBox()
        all_subtypes = []
        for cat, items in FITTING_CATEGORIES.items():
            for item in items:
                all_subtypes.append(item)
        sub_combo.addItems(all_subtypes)
        sub_combo.setCurrentText(comp.fitting_subtype)
        sub_combo.setMinimumWidth(110)

        # Nominal diameter
        nom_combo = QComboBox()
        self._update_nom_diameters(nom_combo, comp.connection_type)
        # Set to nearest available
        nom_combo.setCurrentText(str(comp.nominal_diameter_in))
        nom_combo.setMinimumWidth(110)

        # K value (auto-populated, editable)
        k_spin = _make_spinbox(0, 1e6, comp.K, 4, 0.1)

        # Diameter in metres (auto from nominal, shown read-only-ish)
        diam_spin = _make_spinbox(0.001, 5.0, comp.diameter, 5, 0.001)
        diam_spin.setSuffix(" m")

        # Auto-update K and diameter when subtype/connection/nominal changes
        def _update_k():
            ct = conn_combo.currentText()
            st = sub_combo.currentText()
            try:
                nd = float(nom_combo.currentText())
            except ValueError:
                return
            k = lookup_fitting_k(ct, st, nd)
            k_spin.setValue(k)
            d = NOMINAL_TO_METRES.get(nd, comp.diameter)
            diam_spin.setValue(d)

        conn_combo.currentTextChanged.connect(lambda _: self._update_nom_diameters(nom_combo, conn_combo.currentText()))
        conn_combo.currentTextChanged.connect(lambda _: _update_k())
        sub_combo.currentTextChanged.connect(lambda _: _update_k())
        nom_combo.currentTextChanged.connect(lambda _: _update_k())

        self._fields.update({
            "connection_type": conn_combo,
            "fitting_subtype": sub_combo,
            "nominal_diameter_in": nom_combo,
            "K": k_spin,
            "diameter": diam_spin,
        })

        self._add_group("Fitting Type", [
            ("Connection:", conn_combo),
            ("Fitting:", sub_combo),
            ("Nom. diameter (in):", nom_combo),
        ])
        self._add_group("Loss Coefficient", [
            ("K (auto):", k_spin),
            ("Inner diam.:", diam_spin),
        ])

        note = QLabel(
            "<small>K is auto-populated from Table 6.5.<br>"
            "You can override it manually.</small>")
        note.setTextFormat(Qt.TextFormat.RichText)
        note.setStyleSheet("color:#888; padding:2px;")
        self._form_layout.addWidget(note)

    @staticmethod
    def _update_nom_diameters(combo: QComboBox, connection_type: str):
        combo.blockSignals(True)
        prev = combo.currentText()
        combo.clear()
        diams = FITTING_DIAMETERS.get(connection_type, [1, 2, 4])
        combo.addItems([str(d) for d in diams])
        # Restore previous if available
        idx = combo.findText(prev)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _build_pump_form(self, comp: Pump):
        self._empty_label.setVisible(False)
        sa = _make_spinbox(-1e9, 0, comp.A, 1, 100)
        sb = _make_spinbox(-1e6, 1e6, comp.B, 3, 1.0)
        sc = _make_spinbox(0, 10000, comp.C, 2, 1.0)
        sc.setSuffix(" m")
        diam = _make_spinbox(0.001, 5.0, comp.diameter, 4, 0.01)
        diam.setSuffix(" m")
        is_on = QCheckBox("Pump running")
        is_on.setChecked(comp.is_on)
        self._fields.update({
            "A": sa, "B": sb, "C": sc,
            "diameter": diam, "is_on": is_on,
        })

        note = QLabel(
            "<small>hp = A·Q² + B·Q + C &nbsp;[m]<br>"
            "A ≤ 0 for stable operation</small>")
        note.setTextFormat(Qt.TextFormat.RichText)
        note.setStyleSheet("color:#888; padding:2px;")

        gb = _section("Pump Curve  hp = A·Q² + B·Q + C")
        fl = QFormLayout()
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        fl.setSpacing(6)
        fl.addRow("A (m/(m³/s)²):", sa)
        fl.addRow("B (m/(m³/s)):", sb)
        fl.addRow("C — shut-off head:", sc)
        fl.addRow("", note)
        gb.setLayout(fl)
        self._form_layout.addWidget(gb)

        self._add_group("Physical", [
            ("Reference diam.:", diam),
            ("", is_on),
        ])

    def _build_valve_form(self, comp: Valve):
        self._empty_label.setVisible(False)
        diam    = _make_spinbox(0.001, 5.0, comp.diameter, 4, 0.01)
        diam.setSuffix(" m")
        K       = _make_spinbox(0, 1e6, comp.K, 3, 0.5)
        is_open = QCheckBox("Valve open")
        is_open.setChecked(comp.is_open)
        self._fields.update({
            "diameter": diam, "K": K, "is_open": is_open,
        })
        self._add_group("Valve", [
            ("Bore diameter:", diam),
            ("Loss coeff. K:", K),
            ("", is_open),
        ])

    # ── Apply handler ─────────────────────────────────────────────────────────

    def _on_apply(self):
        if not self._comp_id:
            return
        params: dict = {}
        for key, widget in self._fields.items():
            if isinstance(widget, QDoubleSpinBox):
                params[key] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                val = widget.currentText()
                # Convert nominal diameter string to float
                if key == "nominal_diameter_in":
                    try:
                        val = float(val)
                    except ValueError:
                        val = 1.0
                params[key] = val
            elif isinstance(widget, QLineEdit):
                params[key] = widget.text()
        self.apply_requested.emit(self._comp_id, params)


# ── tiny helper ────────────────────────────────────────────────────────────────

def _fmt(val, scale=1.0):
    if val is None:
        return None
    return val * scale
