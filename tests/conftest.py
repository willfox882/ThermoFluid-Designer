"""
Make project root importable for tests in tests/.
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest


@pytest.fixture(autouse=True)
def _reset_global_state():
    """Isolate tests from the global fluid-temperature (#4) and display-unit
    (#5) state: reset to 20 °C / SI before and after each test."""
    import fluid_props, units
    fluid_props.set_fluid_temperature(20.0)
    units.set_system(units.SI)
    yield
    fluid_props.set_fluid_temperature(20.0)
    units.set_system(units.SI)
