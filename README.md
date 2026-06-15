# ThermoFluid Designer

A production-grade pipe network simulation tool with a graphical drag-and-drop canvas, full-fidelity physics, and Newton-Raphson solver.

---

## Features

| Category | Details |
|---|---|
| **Components** | Pipe, Centrifugal Pump, Valve, Junction, Reservoir |
| **Friction** | Darcy-Weisbach + Haaland (turbulent), 64/Re (laminar), smooth transition |
| **Solver** | Custom Newton-Raphson with backtracking line search, analytical Jacobian |
| **Pump model** | Quadratic curve `hp = A·Q² + B·Q + C`, on/off state |
| **Valve model** | K-value with open/closed toggle (K_closed = 10⁸) |
| **Topology** | Any number of loops, parallel branches, demand nodes |
| **Results** | Head, flow, velocity, Reynolds number, friction factor, pressure at every node/edge |
| **Plots** | Pump curve + system curve + operating point; results table |
| **File I/O** | Save/load networks as `.tfn` (JSON) |
| **Units** | SI only (m, m³/s, Pa, kg, s) |

---

## Quick Start (Windows)

### Option A — One-click launcher (recommended)
1. Extract the archive
2. Double-click **`install_and_run.bat`**
3. It will check for Python, install dependencies automatically, and launch the app

### Option B — Pre-built executable
1. Follow the steps in `BUILD_EXE.md` to build `ThermoFluidDesigner.exe`
2. Double-click to run — no Python installation required

### Option C — Run from source manually

**Requirements**
- Python 3.10+
- PyQt6, NumPy, SciPy, Matplotlib

```
pip install PyQt6 numpy scipy matplotlib
cd thermofluid_designer
python main.py
```

### Option D — Headless physics demo (no GUI needed)
Verify the physics engine works without installing PyQt6:
```
pip install numpy scipy
python demo_headless.py
```

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Solve network |
| `Delete` | Delete selected component |
| `Escape` | Cancel connection / placement mode |
| `Ctrl+Z` | Undo last delete |
| `Ctrl+N` | New network |
| `Ctrl+O` | Open file |
| `Ctrl+S` | Save |
| `F` | Fit to view |

---

## Building the .exe (Windows)

```
pip install pyinstaller
cd thermofluid_designer
pyinstaller thermofluid.spec --clean
```

Output: `dist/ThermoFluidDesigner.exe`

---

## Usage Guide

### Building a network

| Action | How |
|---|---|
| Place a reservoir | Click **⬡ Reservoir** in toolbar, then click canvas |
| Place a junction | Click **● Junction**, then click canvas |
| Add a pipe | Click **━ Pipe**, click source node, click destination node |
| Add a pump | Click **⊛ Pump**, click source node, click destination node |
| Add a valve | Click **⊠ Valve**, click source node, click destination node |
| Delete component | Select it, press `Delete` key (or right-click → Delete) |
| Edit properties | Click any component to open Properties panel on the right |

### Solving

- Click **▶ Solve** (or press `Ctrl+Enter`)
- Results appear as colour-coded overlays on the canvas
- Flow rates are displayed on each edge (L/s)
- Node heads are shown above each node
- Switch to the **Results Table** tab for full numerical output

### Pump curve

After solving a network with a pump:
- Switch to the **Pump Curve** tab (bottom-right panel)
- Red curve = pump characteristic `hp(Q)`
- Blue dashed = system curve `h_sys = h_static + R_eff·Q²`
- Green dot = operating point

### Properties panel

Each component type exposes its full parameter set:

| Component | Parameters |
|---|---|
| Reservoir | Total head H [m] |
| Junction | Elevation z [m], demand D [m³/s] |
| Pipe | Diameter D [m], Length L [m], Roughness ε [m], Minor loss ΣK (fittings or override) |
| Pump | Curve coefficients A, B, C; Reference diameter; On/Off |
| Valve | Bore diameter [m], Loss coefficient K; Open/Closed |

After editing, click **Apply Changes** — results are cleared and the network must be re-solved.

---

## Physics Reference

### Energy equation (per edge)

```
H_from − H_to = h_L(Q)
```

where `H = P/(ρg) + z` is piezometric head.

### Pipe head loss (Darcy-Weisbach)

```
h_L = sign(Q) · [f(Re, ε/D)·L/D + ΣK] · Q·|Q| / (2g·A²)
```

- `f` = Darcy friction factor (Haaland explicit approximation for Re > 4000, 64/Re for Re < 2300, linear interpolation in 2300–4000 transition)
- `A` = pipe cross-sectional area = π·D²/4

### Pump head loss

```
h_L = −hp(Q) = −(A·Q² + B·Q + C)
```

Negative `h_L` means energy is added to the fluid.
Stable operation requires `A ≤ 0`.

### Continuity at each junction

```
Σ_j A[i,j]·Q_j = D_i
```

where `A` is the signed incidence matrix and `D_i` is nodal demand.

### Newton-Raphson solver

Unknown vector: `x = [H₀ … H_{N-1}, Q₀ … Q_{P-1}]`

Residuals:
- Continuity (N equations): `F_i = Σ A[i,j]·Q_j − D_i`
- Energy (P equations): `F_{N+j} = H_from − H_to − h_L(Q_j)`

Jacobian blocks:
```
J = ┌  0_{N×N}   │  A_{N×P}          ┐
    │─────────────┼───────────────────│
    │  B_{P×N}   │  −diag(dh_L/dQ)   ┘
```

`dh_L/dQ` is computed analytically including the Haaland chain-rule correction `df/dQ = (df/dRe)·(dRe/dQ)`.

Convergence: backtracking Armijo line search, tolerance `‖F‖ < 10⁻⁹`.

---

## Project Structure

```
thermofluid_designer/
├── main.py                  Entry point
├── install_and_run.bat      One-click Windows setup & launch
├── demo_headless.py         Physics demo (no GUI needed)
├── thermofluid.spec         PyInstaller build spec
├── requirements.txt         pip dependencies
├── BUILD_EXE.md             How to build the .exe
│
├── fluid_props.py           Physical constants, friction factor functions
├── components.py            Pipe, Pump, Valve, Junction, Reservoir classes
├── network.py               PipeNetwork graph model, incidence matrix
├── solver.py                Newton-Raphson solver, SolverResult
│
├── canvas.py                QGraphicsScene canvas, node/edge graphics items
├── sidebar.py               Dynamic properties form panel
├── plotting_widget.py       Pump curve + results table (Matplotlib embedded)
├── main_window.py           QMainWindow controller, toolbar, menus
│
└── test_physics.py          52-test physics suite (52/52 passing)
```

---

## Test Suite

```
cd thermofluid_designer
python -m pytest test_physics.py -v
```

52 tests covering: fluid properties, component physics (Pipe/Pump/Valve/Junction/Reservoir), network construction, and solver correctness across series, parallel, pump, demand, and idempotency scenarios.

Key verified properties:
- Jacobian accuracy vs. finite-difference: < 0.001% error across all flow regimes
- Series network energy balance: `|Σh_L − ΔH| < 10⁻⁶ m`
- Mass balance at all junctions: `< 10⁻⁹ m³/s` error
- Pump operating point: physically correct (h_L < 0, Q > 0)
- Solver residual: `< 10⁻¹¹` at convergence

---

## Roadmap

- [ ] Imperial units toggle (ft, gal/min, psi)
- [ ] Minor loss library (elbows, tees, reducers with standard K values)
- [ ] Variable-speed pump (affinity laws)
- [ ] Transient simulation (water hammer)
- [ ] Export results to CSV/Excel
- [ ] Pressure contour overlay on canvas
- [ ] Multi-fluid support (density/viscosity selector)
- [ ] Network auto-layout algorithm
