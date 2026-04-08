"""
sidebar.py
----------
Dynamic properties panel.  Rebuilt whenever a different component (or fitting
attachment) is selected on the canvas.

Supports:
  Reservoir          → elevation, surface pressure
  Junction           → elevation, demand
  Pipe               → material, geometry, minor-loss (fittings list + override)
  Pump               → A, B, C, diameter, is_on
  Valve              → diameter, K, is_open
  FittingAttachment  → connection type, subtype, nominal diameter, K override
"""

from __future__ import annotations
from typing import Optional, Callable, Any, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QDoubleSpinBox, QCheckBox,
    QPushButton, QGroupBox, QScrollArea, QSizePolicy,
    QFrame, QSpacerItem, QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from components import (
    FluidComponent, Pipe, Pump, Valve, PRV, Junction, Reservoir, Fitting,
    FittingAttachment, PressurizedSource,
)
from fluid_props import (
    DENSITY, GRAVITY,
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
    apply_requested(comp_id, prop_dict)
        Emitted when Apply is clicked for a standard component.

    fitting_apply_requested(pipe_id, fitting_id, prop_dict)
        Emitted when Apply is clicked while a FittingAttachment is shown.

    fitting_action_requested(action, pipe_id, fitting_id)
        Emitted by fitting-list delete buttons.  action = "delete".
    """

    apply_requested         = pyqtSignal(str, dict)        # comp_id, params
    fitting_apply_requested = pyqtSignal(str, str, dict)   # pipe_id, fitting_id, params
    fitting_action_requested = pyqtSignal(str, str, str)   # action, pipe_id, fitting_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._comp_id:   Optional[str] = None
        self._comp_type: Optional[str] = None
        self._context:   str           = "component"   # "component" | "fitting"
        self._fitting_pipe_id: Optional[str] = None

        self._fields:         dict[str, QWidget]      = {}
        self._computed_fields: dict[str, Callable]    = {}

        self._build_ui()

    # ── Layout skeleton ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        hdr = QLabel("Properties")
        hdr.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.setStyleSheet(
            "background:#2d3040; color:#e8eaf0; padding:6px; border-radius:4px;")
        root.addWidget(hdr)

        badge_row = QHBoxLayout()
        self._type_label = QLabel("—")
        self._type_label.setFont(QFont("Segoe UI", 8))
        self._type_label.setStyleSheet(
            "background:#4a5060; color:#b0c4de; padding:3px 8px; border-radius:3px;")
        self._id_label = QLabel("")
        self._id_label.setFont(QFont("Consolas", 8))
        badge_row.addWidget(self._type_label)
        badge_row.addStretch()
        badge_row.addWidget(self._id_label)
        root.addLayout(badge_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._form_widget = QWidget()
        self._form_layout = QVBoxLayout(self._form_widget)
        self._form_layout.setContentsMargins(0, 0, 0, 0)
        self._form_layout.setSpacing(8)
        scroll.setWidget(self._form_widget)
        root.addWidget(scroll, stretch=1)

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

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setFont(QFont("Segoe UI", 7))
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        root.addWidget(self._status_label)

        # Results section
        self._result_box = _section("Solver Results")
        res_layout = QFormLayout()
        res_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        res_layout.setSpacing(4)
        self._res_labels: dict[str, QLabel] = {}
        for key in ("Head (m)", "Flow (L/s)", "Velocity (m/s)",
                    "Reynolds", "f (Darcy)", "ΔH (m)",
                    "Req. Head (m)", "P_req (kW)"):
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
        self._comp_id   = comp.id
        self._comp_type = type(comp).__name__
        self._context   = "component"
        self._fitting_pipe_id = None

        self._type_label.setText(self._comp_type)
        self._id_label.setText(comp.id)
        self._apply_btn.setEnabled(True)
        self._status_label.setText("")

        self._clear_form()

        builder = {
            "Reservoir":         self._build_reservoir_form,
            "PressurizedSource": self._build_pressurized_source_form,
            "Junction":          self._build_junction_form,
            "Pipe":              self._build_pipe_form,
            "Pump":              self._build_pump_form,
            "Valve":             self._build_valve_form,
            "PRV":               self._build_prv_form,
            "Fitting":           self._build_fitting_form,
        }.get(self._comp_type)
        if builder:
            builder(comp)

        self._form_layout.addStretch()

    def load_fitting_attachment(self, pipe_id: str, fitting: FittingAttachment):
        """Show form for a FittingAttachment (child of a pipe)."""
        self._comp_id         = fitting.fitting_id
        self._comp_type       = "FittingAttachment"
        self._context         = "fitting"
        self._fitting_pipe_id = pipe_id

        self._type_label.setText("Fitting")
        self._id_label.setText(fitting.fitting_id)
        self._apply_btn.setEnabled(True)
        self._status_label.setText("")

        self._clear_form()
        self._build_fitting_attachment_form(fitting)
        self._form_layout.addStretch()

    def clear_selection(self):
        self._comp_id   = None
        self._comp_type = None
        self._context   = "component"
        self._fitting_pipe_id = None
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
            self._status_label.setStyleSheet("color:#e05050;")
            self._status_label.setText("\n".join(f"⚠ {e}" for e in errors))
        else:
            self._status_label.setStyleSheet("color:#50b050;")
            self._status_label.setText("✓ Network valid")

    def show_results(self, result_dict: dict):
        self._result_box.setVisible(True)
        mapping = {
            "Head (m)":       result_dict.get("head"),
            "Flow (L/s)":     _fmt(result_dict.get("flow"), scale=1000),
            "Velocity (m/s)": result_dict.get("velocity"),
            "Reynolds":       result_dict.get("reynolds"),
            "f (Darcy)":      result_dict.get("friction_factor"),
            "ΔH (m)":         result_dict.get("head_loss"),
            "Req. Head (m)":  result_dict.get("pump_req_head"),
            "P_req (kW)":     result_dict.get("pump_power"),
        }
        for key, val in mapping.items():
            lbl = self._res_labels[key]
            if val is None:
                lbl.setText("—")
            elif key == "Reynolds":
                lbl.setText(f"{val:,.0f}")
            else:
                lbl.setText(f"{val:.4f}")

        # Update inline pump-sizing labels if visible
        h_req = result_dict.get("pump_req_head")
        p_req = result_dict.get("pump_power")
        if hasattr(self, '_pump_h_req_label'):
            self._pump_h_req_label.setText(
                f"{h_req:.3f} m" if h_req is not None else "—")
        if hasattr(self, '_pump_p_req_label'):
            self._pump_p_req_label.setText(
                f"{p_req:.4f} kW" if p_req is not None else "—")

        # NPSH: compute NPSHa from suction-side head if available
        npsh_a = result_dict.get("npsh_available")
        npsh_r = self._fields.get("npsh_required").value() if "npsh_required" in self._fields else None
        
        if hasattr(self, '_npsha_label') and npsh_a is not None:
            self._npsha_label.setText(f"{npsh_a:.3f} m")
            if npsh_r is not None and hasattr(self, '_npsh_warn_label'):
                if npsh_a < npsh_r:
                    self._npsh_warn_label.setText(
                        f"⚠ Cavitation risk! NPSHa ({npsh_a:.2f} m) < NPSHr ({npsh_r:.2f} m)")
                    self._npsh_warn_label.setStyleSheet("color:#e04040; font-weight:bold;")
                else:
                    margin = npsh_a - npsh_r
                    self._npsh_warn_label.setText(f"✓ Margin = {margin:.2f} m")
                    self._npsh_warn_label.setStyleSheet("color:#50b050;")

    def hide_results(self):
        self._result_box.setVisible(False)
        for lbl in self._res_labels.values():
            lbl.setText("—")
        if hasattr(self, '_pump_h_req_label'):
            self._pump_h_req_label.setText("Solve network first")
        if hasattr(self, '_pump_p_req_label'):
            self._pump_p_req_label.setText("—")
        if hasattr(self, '_npsha_label'):
            self._npsha_label.setText("—")
        if hasattr(self, '_npsh_warn_label'):
            self._npsh_warn_label.setText("")

    # ── Internal form helpers ─────────────────────────────────────────────────

    def _clear_form(self):
        self._fields.clear()
        self._computed_fields.clear()
        
        # Clear specific result-linked attributes to avoid RuntimeError
        for attr in ("_pump_h_req_label", "_pump_p_req_label", 
                      "_npsha_label", "_npsh_warn_label"):
            if hasattr(self, attr):
                delattr(self, attr)

        while self._form_layout.count():
            item = self._form_layout.takeAt(0)
            w = item.widget()
            if w and w is not self._empty_label:
                w.deleteLater()

    def _add_group(self, title: str, rows: list) -> QGroupBox:
        gb     = _section(title)
        layout = QFormLayout()
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        layout.setSpacing(6)
        for label, widget in rows:
            layout.addRow(label, widget)
        gb.setLayout(layout)
        self._form_layout.addWidget(gb)
        return gb

    def _add_name_field(self, comp: "FluidComponent"):
        name_edit = QLineEdit(comp.name)
        name_edit.setPlaceholderText("Component name…")
        name_edit.setMinimumWidth(110)
        self._fields["name"] = name_edit
        self._add_group("Identity", [("Name:", name_edit)])

    # ── Form builders ─────────────────────────────────────────────────────────

    def _build_reservoir_form(self, comp: Reservoir):
        self._empty_label.setVisible(False)
        self._add_name_field(comp)

        elev  = _make_spinbox(-1000, 10000, comp.elevation, 3, 0.5)
        elev.setSuffix(" m")
        press = _make_spinbox(0, 1e7, comp.surface_pressure_Pa, 1, 1000.0)
        press.setSuffix(" Pa")
        h_label = QLabel(f"{comp.total_head:.3f} m")
        h_label.setFont(QFont("Consolas", 8))
        h_label.setStyleSheet(
            "color:#3a7bd5; padding:2px 4px; font-weight:bold; "
            "background:#eef3fb; border-radius:3px;")

        def _update_head():
            H = elev.value() + press.value() / (DENSITY * GRAVITY)
            h_label.setText(f"{H:.3f} m")

        elev.valueChanged.connect(lambda _: _update_head())
        press.valueChanged.connect(lambda _: _update_head())

        self._fields["elevation"]           = elev
        self._fields["surface_pressure_Pa"] = press
        self._add_group("Reservoir", [
            ("Elevation z:", elev),
            ("Surface pressure:", press),
            ("Total head H:", h_label),
        ])
        note = QLabel(
            "<small>Datum: z = 0 at reference plane.<br>"
            "H = z + P/(ρg)  •  Open tanks: P = 0 Pa.</small>")
        note.setTextFormat(Qt.TextFormat.RichText)
        note.setStyleSheet("color:#888; padding:2px;")
        self._form_layout.addWidget(note)

    def _build_pressurized_source_form(self, comp: PressurizedSource):
        """Form for a PressurizedSource — same fields as Reservoir but emphasises pressure."""
        self._empty_label.setVisible(False)
        self._add_name_field(comp)

        elev  = _make_spinbox(-1000, 10000, comp.elevation, 3, 0.5)
        elev.setSuffix(" m")
        press = _make_spinbox(0, 1e8, comp.surface_pressure_Pa, 1, 5000.0)
        press.setSuffix(" Pa")
        h_label = QLabel(f"{comp.total_head:.3f} m")
        h_label.setFont(QFont("Consolas", 8))
        h_label.setStyleSheet(
            "color:#3a7bd5; padding:2px 4px; font-weight:bold; "
            "background:#eef3fb; border-radius:3px;")

        def _update_head():
            H = elev.value() + press.value() / (DENSITY * GRAVITY)
            h_label.setText(f"{H:.3f} m")

        elev.valueChanged.connect(lambda _: _update_head())
        press.valueChanged.connect(lambda _: _update_head())

        self._fields["elevation"]           = elev
        self._fields["surface_pressure_Pa"] = press
        self._add_group("Pressurized Source", [
            ("Elevation z:", elev),
            ("Supply pressure:", press),
            ("Total head H:", h_label),
        ])

        # Optional known flow rate — switches BC from fixed-head to fixed-flow
        kfr_Ls = getattr(comp, 'known_flow_rate', 0.0) * 1000.0
        kfr_spin = _make_spinbox(0, 10000, kfr_Ls, 3, 0.1)
        kfr_spin.setSuffix(" L/s")
        kfr_spin.setSpecialValueText("— (pressure BC)")
        # Store as m³/s via computed_fields
        self._computed_fields["known_flow_rate"] = lambda: kfr_spin.value() / 1000.0

        kfr_note = QLabel(
            "<small><b>0 L/s</b> → fixed-head BC (pressure drives flow).<br>"
            "<b>> 0 L/s</b> → fixed-flow BC (flow rate is prescribed).</small>")
        kfr_note.setTextFormat(Qt.TextFormat.RichText)
        kfr_note.setStyleSheet("color:#888; padding:2px;")

        self._add_group("Known Flow Rate (optional)", [
            ("Flow rate:", kfr_spin),
        ])
        self._form_layout.addWidget(kfr_note)

        note = QLabel(
            "<small>H = z + P/(ρg)  •  Flow occurs when H is higher<br>"
            "than the outlet. No pump required for pressure-driven flow.</small>")
        note.setTextFormat(Qt.TextFormat.RichText)
        note.setStyleSheet("color:#888; padding:2px;")
        self._form_layout.addWidget(note)

    def _build_junction_form(self, comp: Junction):
        self._empty_label.setVisible(False)
        self._add_name_field(comp)
        elev = _make_spinbox(-1000, 10000, comp.elevation, 3, 0.5); elev.setSuffix(" m")
        dem  = _make_spinbox(-1, 1, comp.demand, 6, 0.0001);         dem.setSuffix(" m³/s")
        self._fields["elevation"] = elev
        self._fields["demand"]    = dem
        self._add_group("Junction", [("Elevation:", elev), ("Demand:", dem)])

    def _build_pipe_form(self, comp: Pipe):
        self._empty_label.setVisible(False)
        self._add_name_field(comp)

        mat_combo = QComboBox(); mat_combo.addItems(list(MATERIAL_CONDITIONS.keys()))
        mat_combo.setCurrentText(comp.material); mat_combo.setMinimumWidth(110)

        cond_combo = QComboBox()
        self._update_condition_combo(cond_combo, comp.material)
        cond_combo.setCurrentText(comp.condition); cond_combo.setMinimumWidth(110)

        rough = _make_spinbox(0, 0.1, comp.roughness, 7, 0.000001); rough.setSuffix(" m")

        def _on_mat(mat):
            self._update_condition_combo(cond_combo, mat)
            rough.setValue(lookup_roughness(mat, cond_combo.currentText()))

        def _on_cond(cond):
            rough.setValue(lookup_roughness(mat_combo.currentText(), cond))

        mat_combo.currentTextChanged.connect(_on_mat)
        cond_combo.currentTextChanged.connect(_on_cond)

        diam = _make_spinbox(0.001, 5.0, comp.diameter, 4, 0.01); diam.setSuffix(" m")
        leng = _make_spinbox(0, 100000, comp.length, 2, 10.0);    leng.setSuffix(" m")
        elev = _make_spinbox(-5000, 5000, comp.elevation_change, 3, 0.5); elev.setSuffix(" m")

        self._fields.update({
            "material": mat_combo, "condition": cond_combo,
            "diameter": diam, "length": leng, "roughness": rough,
            "elevation_change": elev,
        })
        self._add_group("Pipe Material", [
            ("Material:", mat_combo), ("Condition:", cond_combo), ("Roughness ε:", rough),
        ])
        self._add_group("Pipe Geometry", [("Diameter D:", diam), ("Length L:", leng)])

        # ── Head loss / minor losses ──────────────────────────────────────
        k_computed_lbl = QLabel(f"{comp.K_minor_computed:.4f}")
        k_computed_lbl.setFont(QFont("Consolas", 8))
        k_computed_lbl.setStyleSheet(
            "color:#3a7bd5; padding:2px 4px; background:#eef3fb; border-radius:3px;")

        use_override = QCheckBox("Manual override K")
        use_override.setChecked(comp._K_minor_override is not None)

        k_override_spin = _make_spinbox(0, 1000,
                                        comp._K_minor_override if comp._K_minor_override is not None
                                        else comp.K_minor_computed, 3, 0.1)
        k_override_spin.setEnabled(comp._K_minor_override is not None)
        use_override.toggled.connect(k_override_spin.setEnabled)

        self._computed_fields["K_minor_override"] = (
            lambda: k_override_spin.value() if use_override.isChecked() else None
        )

        self._add_group("Head Loss", [
            ("Elev. change Δz:", elev),
            ("Fittings ΣK:", k_computed_lbl),
            ("", use_override),
            ("Override K:", k_override_spin),
        ])

        # ── Attached fittings list ────────────────────────────────────────
        self._build_pipe_fittings_section(comp)

    def _build_pipe_fittings_section(self, comp: Pipe):
        """Append the fittings list group to the pipe form."""
        gb     = _section(f"Fittings  ({len(comp.fittings)} attached)")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        if not comp.fittings:
            lbl = QLabel("<i>No fittings — click a pipe in fitting mode to add one.</i>")
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setStyleSheet("color:#888; padding:4px;")
            layout.addWidget(lbl)
        else:
            for fa in comp.fittings:
                layout.addWidget(self._make_fitting_row(comp.id, fa))

        gb.setLayout(layout)
        self._form_layout.addWidget(gb)

    def _make_fitting_row(self, pipe_id: str, fa: FittingAttachment) -> QWidget:
        """Create a compact one-line widget for a fitting in the list."""
        row = QWidget()
        h   = QHBoxLayout(row)
        h.setContentsMargins(2, 2, 2, 2)
        h.setSpacing(6)

        # Type label
        type_lbl = QLabel(fa.fitting_subtype[:22] + ("…" if len(fa.fitting_subtype) > 22 else ""))
        type_lbl.setFont(QFont("Segoe UI", 7))
        type_lbl.setMinimumWidth(120)

        # K display
        k_lbl = QLabel(f"K={fa.effective_K:.3f}")
        k_lbl.setFont(QFont("Consolas", 7))
        k_lbl.setStyleSheet("color:#555;")

        # Delete button
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(20, 20)
        del_btn.setToolTip(f"Remove {fa.name}")
        del_btn.setStyleSheet(
            "QPushButton { background:#e05050; color:white; "
            "  border-radius:3px; padding:0; font-size:9px; }"
            "QPushButton:hover { background:#c03030; }")
        _pipe_id  = pipe_id
        _fit_id   = fa.fitting_id
        del_btn.clicked.connect(
            lambda _, p=_pipe_id, f=_fit_id:
            self.fitting_action_requested.emit("delete", p, f))

        h.addWidget(type_lbl, stretch=1)
        h.addWidget(k_lbl)
        h.addWidget(del_btn)
        return row

    @staticmethod
    def _update_condition_combo(combo: QComboBox, material: str):
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(MATERIAL_CONDITIONS.get(material, []))
        combo.blockSignals(False)

    def _build_fitting_form(self, comp: Fitting):
        """Legacy standalone Fitting form (old-format files)."""
        self._empty_label.setVisible(False)
        self._add_name_field(comp)
        conn_combo = QComboBox(); conn_combo.addItems(["Screwed", "Flanged"])
        conn_combo.setCurrentText(comp.connection_type); conn_combo.setMinimumWidth(110)
        sub_combo  = QComboBox()
        for cat, items in FITTING_CATEGORIES.items():
            for item in items:
                sub_combo.addItem(item)
        sub_combo.setCurrentText(comp.fitting_subtype); sub_combo.setMinimumWidth(110)
        nom_combo  = QComboBox()
        self._update_nom_diameters(nom_combo, comp.connection_type)
        nom_combo.setCurrentText(str(comp.nominal_diameter_in))
        k_spin     = _make_spinbox(0, 1e6, comp.K, 4, 0.1)

        def _update_k():
            try: nd = float(nom_combo.currentText())
            except ValueError: return
            k_spin.setValue(lookup_fitting_k(conn_combo.currentText(), sub_combo.currentText(), nd))

        conn_combo.currentTextChanged.connect(
            lambda _: self._update_nom_diameters(nom_combo, conn_combo.currentText()))
        conn_combo.currentTextChanged.connect(lambda _: _update_k())
        sub_combo.currentTextChanged.connect(lambda _: _update_k())
        nom_combo.currentTextChanged.connect(lambda _: _update_k())

        self._fields.update({
            "connection_type": conn_combo, "fitting_subtype": sub_combo,
            "nominal_diameter_in": nom_combo, "K": k_spin,
        })
        self._add_group("Fitting Type", [
            ("Connection:", conn_combo), ("Fitting:", sub_combo),
            ("Nom. diam. (in):", nom_combo),
        ])
        self._add_group("Loss Coefficient", [("K:", k_spin)])

    def _build_fitting_attachment_form(self, fa: FittingAttachment):
        """Form for editing a FittingAttachment (child of a pipe)."""
        self._empty_label.setVisible(False)

        name_edit = QLineEdit(fa.name); name_edit.setMinimumWidth(110)
        self._fields["name"] = name_edit
        self._add_group("Identity", [("Name:", name_edit)])

        conn_combo = QComboBox(); conn_combo.addItems(["Screwed", "Flanged"])
        conn_combo.setCurrentText(fa.connection_type); conn_combo.setMinimumWidth(110)

        sub_combo = QComboBox()
        for cat, items in FITTING_CATEGORIES.items():
            for item in items:
                sub_combo.addItem(item)
        sub_combo.setCurrentText(fa.fitting_subtype); sub_combo.setMinimumWidth(110)

        nom_combo = QComboBox()
        self._update_nom_diameters(nom_combo, fa.connection_type)
        nom_combo.setCurrentText(str(fa.nominal_diameter_in))

        k_default_lbl = QLabel(f"{fa.K_default:.4f}")
        k_default_lbl.setFont(QFont("Consolas", 8))
        k_default_lbl.setStyleSheet(
            "color:#3a7bd5; padding:2px 4px; background:#eef3fb; border-radius:3px;")

        use_override    = QCheckBox("Override K value")
        use_override.setChecked(fa.K_override is not None)
        k_override_spin = _make_spinbox(
            0, 1000,
            fa.K_override if fa.K_override is not None else fa.K_default,
            4, 0.1)
        k_override_spin.setEnabled(fa.K_override is not None)
        use_override.toggled.connect(k_override_spin.setEnabled)

        def _update_k_default():
            try: nd = float(nom_combo.currentText())
            except ValueError: return
            k = lookup_fitting_k(conn_combo.currentText(), sub_combo.currentText(), nd)
            k_default_lbl.setText(f"{k:.4f}")

        conn_combo.currentTextChanged.connect(
            lambda _: self._update_nom_diameters(nom_combo, conn_combo.currentText()))
        conn_combo.currentTextChanged.connect(lambda _: _update_k_default())
        sub_combo.currentTextChanged.connect(lambda _: _update_k_default())
        nom_combo.currentTextChanged.connect(lambda _: _update_k_default())

        self._fields.update({
            "connection_type":     conn_combo,
            "fitting_subtype":     sub_combo,
            "nominal_diameter_in": nom_combo,
        })
        self._computed_fields["K_default"]  = lambda: float(k_default_lbl.text())
        self._computed_fields["K_override"] = (
            lambda: k_override_spin.value() if use_override.isChecked() else None
        )

        self._add_group("Fitting Type", [
            ("Connection:", conn_combo), ("Fitting:", sub_combo),
            ("Nom. diam. (in):", nom_combo),
        ])
        self._add_group("Loss Coefficient", [
            ("Default K (table):", k_default_lbl),
            ("", use_override),
            ("Override K:", k_override_spin),
        ])

        note = QLabel(
            "<small>Default K auto-populated from Table 6.5.<br>"
            "Override only if you have a measured value.</small>")
        note.setTextFormat(Qt.TextFormat.RichText)
        note.setStyleSheet("color:#888; padding:2px;")
        self._form_layout.addWidget(note)

    def _build_pump_form(self, comp: Pump):
        self._empty_label.setVisible(False)
        self._add_name_field(comp)
        sa = _make_spinbox(-1e9, 0, comp.A, 1, 100)
        sb = _make_spinbox(-1e6, 1e6, comp.B, 3, 1.0)
        sc = _make_spinbox(0, 10000, comp.C, 2, 1.0); sc.setSuffix(" m")
        diam  = _make_spinbox(0.001, 5.0, comp.diameter, 4, 0.01); diam.setSuffix(" m")
        is_on = QCheckBox("Pump running"); is_on.setChecked(comp.is_on)
        self._fields.update({"A": sa, "B": sb, "C": sc, "diameter": diam, "is_on": is_on})

        note = QLabel("<small>hp = A·Q² + B·Q + C &nbsp;[m]<br>A ≤ 0 for stable operation</small>")
        note.setTextFormat(Qt.TextFormat.RichText)
        note.setStyleSheet("color:#888; padding:2px;")

        gb = _section("Pump Curve  hp = A·Q² + B·Q + C")
        fl = QFormLayout()
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight); fl.setSpacing(6)
        fl.addRow("A (m/(m³/s)²):", sa)
        fl.addRow("B (m/(m³/s)):",  sb)
        fl.addRow("C — shut-off head:", sc)
        fl.addRow("", note)
        gb.setLayout(fl)
        self._form_layout.addWidget(gb)
        self._add_group("Physical", [("Reference diam.:", diam), ("", is_on)])

        # ── Pump Sizing / Power Estimation ────────────────────────────────────
        qd_Ls = comp.desired_flow_rate * 1000.0
        qd_spin = _make_spinbox(0, 10000, qd_Ls, 3, 0.1)
        qd_spin.setSuffix(" L/s")
        self._computed_fields["desired_flow_rate"] = lambda: qd_spin.value() / 1000.0

        # Read-only labels updated after solve via show_results()
        h_req_lbl = QLabel("Solve network first")
        h_req_lbl.setFont(QFont("Consolas", 8))
        h_req_lbl.setStyleSheet(
            "color:#3a7bd5; padding:2px 4px; background:#eef3fb; border-radius:3px;")
        p_req_lbl = QLabel("—")
        p_req_lbl.setFont(QFont("Consolas", 8))
        p_req_lbl.setStyleSheet(
            "color:#3a7bd5; padding:2px 4px; background:#eef3fb; border-radius:3px;")

        # Store for external update from main_window
        self._pump_h_req_label = h_req_lbl
        self._pump_p_req_label = p_req_lbl

        # Pump type selector for curve generator
        pump_type_combo = QComboBox()
        pump_type_combo.addItems(["centrifugal", "mixed-flow", "axial"])
        pump_type_combo.setMinimumWidth(110)

        gen_btn = QPushButton("Generate Curve from Sizing")
        gen_btn.setStyleSheet(
            "QPushButton { background:#3a7bd5; color:white; padding:4px 8px; "
            "  border-radius:4px; font-size:9px; }"
            "QPushButton:hover { background:#5090e8; }")
        gen_btn.clicked.connect(
            lambda: self._on_generate_pump_curve(sa, sb, sc, qd_spin, pump_type_combo))

        sizing_gb = _section("Pump Sizing Mode")
        sizing_fl = QFormLayout()
        sizing_fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        sizing_fl.setSpacing(6)
        sizing_fl.addRow("Desired flow Q_des:", qd_spin)
        sizing_fl.addRow("Req. head h_req:", h_req_lbl)
        sizing_fl.addRow("Req. power P_req:", p_req_lbl)
        sizing_fl.addRow("Pump type:", pump_type_combo)
        sizing_fl.addRow("", gen_btn)
        sizing_gb.setLayout(sizing_fl)
        self._form_layout.addWidget(sizing_gb)

        # ── NPSH section ──────────────────────────────────────────────────────
        npsh_r_spin = _make_spinbox(0, 100, comp.npsh_required, 2, 0.5)
        npsh_r_spin.setSuffix(" m")
        self._fields["npsh_required"] = npsh_r_spin

        npsha_lbl = QLabel("—")
        npsha_lbl.setFont(QFont("Consolas", 8))
        npsha_lbl.setStyleSheet(
            "color:#3a7bd5; padding:2px 4px; background:#eef3fb; border-radius:3px;")
        npsh_warn_lbl = QLabel("")
        npsh_warn_lbl.setFont(QFont("Segoe UI", 7))
        npsh_warn_lbl.setStyleSheet("color:#e04040;")
        self._npsha_label = npsha_lbl
        self._npsh_warn_label = npsh_warn_lbl

        npsh_gb = _section("NPSH — Cavitation Check")
        npsh_fl = QFormLayout()
        npsh_fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        npsh_fl.setSpacing(6)
        npsh_fl.addRow("NPSHr (required):", npsh_r_spin)
        npsh_fl.addRow("NPSHa (available):", npsha_lbl)
        npsh_fl.addRow("", npsh_warn_lbl)
        npsh_gb.setLayout(npsh_fl)
        self._form_layout.addWidget(npsh_gb)

        npsh_note = QLabel(
            "<small>NPSHa = (p_atm − p_vapor)/(ρg) + H_suction<br>"
            "Solve network to compute NPSHa.  Warning if NPSHa &lt; NPSHr.</small>")
        npsh_note.setTextFormat(Qt.TextFormat.RichText)
        npsh_note.setStyleSheet("color:#888; padding:2px;")
        self._form_layout.addWidget(npsh_note)

    def _on_generate_pump_curve(self, sa, sb, sc, qd_spin, pump_type_combo):
        """
        Compute optimal pump curve coefficients (A, B, C) for the current
        desired flow rate + required head, then fill in the spinboxes.

        Uses h_req from the Solver Results label if available.
        """
        from solver import NetworkSolver
        import re

        Q_des_m3s = qd_spin.value() / 1000.0
        if Q_des_m3s <= 0:
            return

        # Try to read h_req from the stored label
        h_req = None
        if hasattr(self, '_pump_h_req_label'):
            txt = self._pump_h_req_label.text()
            try:
                h_req = float(txt)
            except (ValueError, AttributeError):
                pass

        if h_req is None or h_req <= 0:
            from PyQt6.QtWidgets import QInputDialog
            val, ok = QInputDialog.getDouble(
                self, "Required Head",
                "Enter required system head h_req [m]:",
                25.0, 0.1, 10000.0, 2)
            if not ok:
                return
            h_req = val

        pump_type = pump_type_combo.currentText()
        A, B, C = NetworkSolver.generate_pump_curve(Q_des_m3s, h_req, pump_type)

        sa.setValue(A)
        sb.setValue(B)
        sc.setValue(C)

        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self, "Pump Curve Generated",
            f"Pump type : {pump_type}\n"
            f"BEP       : Q = {Q_des_m3s*1000:.2f} L/s,  h = {h_req:.2f} m\n"
            f"Shutoff   : {C:.2f} m\n\n"
            f"A = {A:.1f}   B = {B:.3f}   C = {C:.2f}\n\n"
            "Press Apply Changes to save."
        )

    def _build_prv_form(self, comp: PRV):
        """Form for a Pressure-Reducing Valve."""
        self._empty_label.setVisible(False)
        self._add_name_field(comp)

        diam     = _make_spinbox(0.001, 5.0, comp.diameter, 4, 0.01); diam.setSuffix(" m")
        sp_pa    = _make_spinbox(0, 2e7, comp.setpoint_Pa, 0, 10_000.0); sp_pa.setSuffix(" Pa")
        cv_spin  = _make_spinbox(1e-8, 1.0, comp.Cv, 8, 1e-5)
        mf_spin  = _make_spinbox(0, 100, comp.max_flow * 1000, 3, 0.1); mf_spin.setSuffix(" L/s")

        # Computed field: max_flow in m³/s
        self._computed_fields["max_flow"] = lambda: mf_spin.value() / 1000.0

        sp_head_lbl = QLabel(f"{comp.setpoint_head:.3f} m")
        sp_head_lbl.setFont(QFont("Consolas", 8))
        sp_head_lbl.setStyleSheet(
            "color:#3a7bd5; padding:2px 4px; background:#eef3fb; border-radius:3px;")

        def _update_sp_head():
            h = sp_pa.value() / (DENSITY * GRAVITY)
            sp_head_lbl.setText(f"{h:.3f} m")

        sp_pa.valueChanged.connect(lambda _: _update_sp_head())

        self._fields.update({"diameter": diam, "setpoint_Pa": sp_pa, "Cv": cv_spin})
        self._add_group("PRV", [
            ("Bore diameter:", diam),
            ("Setpoint pressure:", sp_pa),
            ("Setpoint head:", sp_head_lbl),
            ("Flow coeff. Cv:", cv_spin),
            ("Max flow:", mf_spin),
        ])

        note = QLabel(
            "<small>PRV maintains downstream pressure ≤ setpoint.<br>"
            "Head loss: h = Q² / (Cv² · g)  (Cv in m³/s per √Pa).</small>")
        note.setTextFormat(Qt.TextFormat.RichText)
        note.setStyleSheet("color:#888; padding:2px;")
        self._form_layout.addWidget(note)

    def _build_valve_form(self, comp: Valve):
        self._empty_label.setVisible(False)
        self._add_name_field(comp)
        diam    = _make_spinbox(0.001, 5.0, comp.diameter, 4, 0.01); diam.setSuffix(" m")
        K       = _make_spinbox(0, 1e6, comp.K, 3, 0.5)
        is_open = QCheckBox("Valve open"); is_open.setChecked(comp.is_open)
        self._fields.update({"diameter": diam, "K": K, "is_open": is_open})
        self._add_group("Valve", [("Bore diameter:", diam), ("Loss coeff. K:", K), ("", is_open)])

    @staticmethod
    def _update_nom_diameters(combo: QComboBox, connection_type: str):
        combo.blockSignals(True)
        prev = combo.currentText()
        combo.clear()
        diams = FITTING_DIAMETERS.get(connection_type, [1, 2, 4])
        combo.addItems([str(d) for d in diams])
        idx = combo.findText(prev)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    # ── Apply handler ─────────────────────────────────────────────────────────

    def _on_apply(self):
        if not self._comp_id:
            return

        params: dict = {}

        # Collect from standard field widgets
        for key, widget in self._fields.items():
            if isinstance(widget, QDoubleSpinBox):
                params[key] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                val = widget.currentText()
                if key == "nominal_diameter_in":
                    try:
                        val = float(val)
                    except ValueError:
                        val = 1.0
                params[key] = val
            elif isinstance(widget, QLineEdit):
                params[key] = widget.text()

        # Collect from computed fields (lambdas that may return None)
        for key, getter in self._computed_fields.items():
            params[key] = getter()

        if self._context == "fitting":
            self.fitting_apply_requested.emit(
                self._fitting_pipe_id, self._comp_id, params)
        else:
            self.apply_requested.emit(self._comp_id, params)


# ── tiny helper ────────────────────────────────────────────────────────────────

def _fmt(val, scale=1.0):
    return None if val is None else val * scale
