"""
Make project root importable for tests in tests/.
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest


@pytest.fixture(autouse=True)
def _reset_fluid_temperature():
    """Isolate tests from the global fluid-temperature state (#4): reset to
    20 °C before and after each test so a solve at a non-default temperature
    cannot leak its ρ/μ into an unrelated test."""
    import fluid_props
    fluid_props.set_fluid_temperature(20.0)
    yield
    fluid_props.set_fluid_temperature(20.0)
