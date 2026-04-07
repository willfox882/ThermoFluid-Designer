# Building ThermoFluid Designer as a Standalone .exe

## Prerequisites

- **Python 3.10+** installed and on PATH
- All dependencies installed:
  ```
  pip install PyQt6 numpy scipy matplotlib
  ```

## Quick Build (3 steps)

```bash
# 1. Install PyInstaller
pip install pyinstaller

# 2. Navigate to the project folder
cd thermofluid_designer

# 3. Build using the included spec file
pyinstaller thermofluid.spec --clean
```

## Output

The compiled executable will be at:

```
dist/ThermoFluidDesigner.exe
```

This is a single-folder bundle. Distribute the entire `dist/ThermoFluidDesigner/` folder.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'PyQt6'` | Run `pip install PyQt6` first |
| `FileNotFoundError` during build | Make sure all `.py` files are in the same folder as `main.py` |
| `.exe` crashes on launch | Run from command prompt to see the error: `dist\ThermoFluidDesigner\ThermoFluidDesigner.exe` |
| Anti-virus flags the .exe | PyInstaller executables are often false-positived; add an exclusion |
| Missing DLLs on another PC | Install the [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |

## Building a Single-File .exe (optional)

If you prefer one `.exe` file instead of a folder, edit `thermofluid.spec` and change:

```python
exe = EXE(
    ...
    console=False,  # already set
)
```

Then build with `--onefile`:

```bash
pyinstaller thermofluid.spec --clean --onefile
```

Note: Single-file builds are slower to launch (unpacks to temp on each run).

## Spec File

The included `thermofluid.spec` is pre-configured with:
- Entry point: `main.py`
- Hidden imports for NumPy, SciPy, Matplotlib backends
- PyQt6 plugin collection
- Console disabled (windowed mode)
