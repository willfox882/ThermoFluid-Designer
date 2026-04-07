"""
main.py
-------
ThermoFluid Designer — entry point.

Run with:
    python main.py

Requirements (install once):
    pip install PyQt6 numpy scipy matplotlib
"""

import sys
import os

# Ensure the project root is on the path so all imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore    import Qt
from PyQt6.QtGui     import QFont

from main_window import MainWindow


def main():
    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName("ThermoFluid Designer")
    app.setOrganizationName("thermofluid")
    app.setStyle("Fusion")

    # Base font
    font = QFont("Segoe UI", 9)
    app.setFont(font)

    # Global stylesheet
    app.setStyleSheet("""
        QMainWindow { background: #f0f2f5; }
        QMenuBar    { background: #2d3040; color: #ccc; }
        QMenuBar::item:selected { background: #3a7bd5; color: white; }
        QMenu       { background: #2d3040; color: #ddd; border: 1px solid #555; }
        QMenu::item:selected { background: #3a7bd5; }
        QGroupBox   { border: 1px solid #ccd; border-radius: 4px;
                      margin-top: 8px; padding-top: 6px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px;
                           color: #555; font-size: 8pt; }
        QScrollBar:vertical { width: 8px; background: #eee; }
        QScrollBar::handle:vertical { background: #bbb; border-radius: 4px; }
        QTabWidget::pane { border: 1px solid #ccd; }
        QTabBar::tab   { padding: 5px 12px; background: #e0e4ec; color: #555; }
        QTabBar::tab:selected { background: white; color: #2d3040; font-weight: bold; }
        QDoubleSpinBox { padding: 2px; border: 1px solid #bbb; border-radius: 3px; }
        QPushButton    { padding: 4px 10px; border-radius: 3px;
                         background: #e8eaf0; border: 1px solid #bbb; }
        QPushButton:hover { background: #d0d8f0; }
        QStatusBar   { background: #2d3040; color: #aaa; }
    """)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
