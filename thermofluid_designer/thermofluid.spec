# -*- mode: python ; coding: utf-8 -*-
"""
thermofluid.spec
----------------
PyInstaller spec for building ThermoFluid Designer into a single Windows .exe

Build command (run from project root):
    pyinstaller thermofluid.spec --clean

Output:
    dist/ThermoFluidDesigner.exe   (single-file, ~40-80 MB)
"""

import sys
from pathlib import Path

block_cipher = None
ROOT = Path(SPECPATH)          # project root

a = Analysis(
    [str(ROOT / 'main.py')],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=[
        # Project modules (flat layout — help PyInstaller find them)
        'fluid_props', 'components', 'network', 'solver',
        'canvas', 'sidebar', 'plotting_widget', 'main_window',
        # PyQt6 sub-modules that auto-import detection often misses
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.QtPrintSupport',
        # Matplotlib backends
        'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_qt5agg',   # fallback
        'matplotlib.backends.backend_pdf',
        'matplotlib.backends.backend_svg',
        # SciPy internals
        'scipy.optimize._minpack_py',
        'scipy.optimize._zeros_py',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_lapack',
        'scipy.sparse.csgraph._tools',
        'scipy.sparse.linalg._dsolve.linsolve',
        # NumPy
        'numpy.core._dtype_ctypes',
    ],
    excludes=[
        # Trim unused large packages
        'tkinter', 'unittest', 'email', 'html', 'http', 'xml',
        'xmlrpc', 'curses', 'doctest', 'pdb', 'pydoc',
        'IPython', 'jupyter', 'notebook',
        'PIL',          # Pillow (not used)
        'cv2',          # OpenCV (not used)
        'pandas',       # pandas (not used)
        'sklearn',      # scikit-learn (not used)
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ThermoFluidDesigner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                  # compress with UPX if available
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,             # no console window (pure GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,                 # add .ico path here when available
    version=None,              # add version_info file here when needed
)
