# Cleuson dam-break example

This example downloads real Swiss terrain, initializes Lac de Cleuson at rest,
removes its modeled dam at **30 s**, and follows the release down the Printze
valley through Siviez until **600 s**. The movie uses a plan view with water
depth in colour, shaded terrain, elevation contours and a fixed colour scale.

## Terrain and assumptions

- Source: [swisstopo swissALTI3D](https://www.swisstopo.admin.ch/en/height-model-swissalti3d),
  32 tiles from the 2024 release, downloaded at 2 m and averaged onto a 25 m grid.
  Coordinates are LV95 / EPSG:2056; elevations use LN02.
- Domain: easting 2,589,000–2,592,500 m and northing 1,105,000–1,112,500 m;
  140 × 300 cells including one ghost cell on each boundary.
- The lake outline and elevation (2186 m) come from the FOEN VECTOR25 lake
  inventory. Download URLs, dates and published tile checksums are retained in
  [the source manifest](../data/cleuson/sources.json).
- [The dam operator](https://www.grande-dixence.ch/en/the-complex/dams/cleuson-82/)
  gives a capacity of 20 million m³ and a crest length of 420 m.
  Underwater bathymetry is **idealized**, using distance from the mapped shore.
  The synthetic lake has maximum initial depth 61.18 m and initial volume
  19.6846 million m³ after excluding the modeled dam footprint.
- The dam is a 100 m wide grid footprint along the northwestern lake shore.
  Initially its bed is at least 2188 m. Removal replaces only this footprint
  with an idealized curved foundation channel, approximately 2114 m at its
  centre. Water depth and both momenta remain unchanged at removal.
- Gravity is 9.81 m/s². Uniform Manning roughness is 0.025 s/m^(1/3), treated
  by a semi-implicit momentum damping step. The downstream valley starts dry.
  The domain edges use the solver's radiative boundary conditions.

## Solver and verification

`examples/cleuson/run.jl` calls the existing first-order XPU well-balanced
wet/dry kernels. It uses the reconstructed face wave speeds, a CFL of 0.45,
donor-limited transport and the hydrostatic pressure corrections. A 1 mm
velocity regularization scale is specified independently of the physical grid
spacing. The run uses the CPU backend on this machine.

The runner checks the intact lake after 30 s, verifies that removing the dam
does not change its water volume, checks raw depths before dry-cell cleanup,
and integrates boundary fluxes to verify the full-domain mass balance.
It writes depth snapshots every 1.5 simulated seconds, maximum depth,
first arrival above 5 cm, a diagnostic CSV and a verification TOML file.

The completed run took 3903 steps and produced 401 snapshots. The intact lake
had zero measured change in depth or momentum. The minimum raw depth was
0 m; all saved states were finite. Maximum volume-balance error, including
boundary outflow, was 1.565 × 10⁻⁷ m³ (relative error 7.948 × 10⁻¹⁵).
At 600 s, 7.1873 million m³ remained inside the domain and 12.4973 million m³
had left it; 5.3313 million m³ remained within the initial reservoir mask.
The cell containing the mapped Siviez reference point first exceeded 5 cm at
99.87 s and reached a maximum modeled depth of 11.58 m. These values belong
to this idealized setup, including its assumed lake bed and dam foundation.

The saved run report and diagnostic history are also copied to
`data/cleuson/verification.toml` and `data/cleuson/diagnostics.csv`.

## Reproduce

From the repository root, with its Julia environment installed:

```sh
python3 -m venv --system-site-packages cache/swiss_dam_venv
cache/swiss_dam_venv/bin/python -m pip install numpy scipy matplotlib requests rasterio
cache/swiss_dam_venv/bin/python examples/cleuson/prepare_terrain.py
julia --project --startup-file=no -t 4 examples/cleuson/run.jl
cache/swiss_dam_venv/bin/python plotting/plot_depth.py
```

Prepared inputs are in `data/cleuson/`; the original GeoTIFF tiles and
snapshots are cached under `cache/swiss_dam/`. Binary fields use little-endian
Float64 inputs or Float32 snapshots with x as the fastest-varying index.
The plotting helper accepts `--case`, `--frames`, `--output` and `--fps`.
It uses a fixed 0–80 m square-root colour scale and hides depths below 5 cm.
The movie runs at 30 times simulation speed, with a one-second hold at each end.

Output: [`cleuson_dam_break.mp4`](animations/cleuson_dam_break.mp4).

## Resolution and runtime

The preparation script accepts `--dx` in metres and `--output` for the case
directory. It resamples the original 2 m GeoTIFF tiles and rebuilds the lake and
dam masks at the requested resolution. A separate 2.5 m case is prepared with:

```sh
cache/swiss_dam_venv/bin/python examples/cleuson/prepare_terrain.py \
  --dx 2.5 --output cache/swiss_dam/cleuson_2p5m
```

This creates a 1400 × 3000 grid (4.2 million cells), with 1398 × 2998 updated
interior cells. Inputs occupy approximately 202 MB. Saving the same 401 Float32
depth snapshots would occupy approximately 6.74 GB before MP4 rendering.

To measure timestep cost without writing snapshots:

```sh
julia --project --startup-file=no -t 4 examples/cleuson/run.jl \
  cache/swiss_dam/cleuson_2p5m cache/swiss_dam/benchmark_2p5m \
  --benchmark-steps 30
```

Benchmark mode runs five intact-lake warm-up steps, checks the equilibrium,
removes the dam early and measures the requested number of moving steps. It
includes the usual fluxes, friction, boundary updates, mass checks, maximum-depth
and arrival-time bookkeeping. It writes a `benchmark.toml` report and leaves
normal-run diagnostics untouched. These are short-run timings; they do not
verify the full 600 s fine-grid simulation.

For a fixed domain and similar wave speeds, reducing cell width from 25 m to
2.5 m gives about 100 times as many cells and ten times as many CFL steps.
The first scaling estimate is therefore 1000 times the original 36.85 s,
or approximately 10.2 hours. A measured per-step benchmark can refine this
estimate because cache behaviour, diagnostics and warm-up costs do not scale
uniformly with cell count.

On this machine with four CPU threads, 30 measured moving steps after warm-up
gave **7.07 ms/step at 25 m** and **523 ms/step at 2.5 m**. The fine-grid
per-step range was 462–633 ms. Scaling the original 3903 steps by ten gives
39,030 steps and a refined estimate of **5.67 hours for 600 simulated seconds**.
Allow roughly 6–8 hours for changes in fine-grid wave speeds and snapshot output.
The combined benchmark process used 1.16 GiB peak resident memory. The fine
benchmark preserved the initial equilibrium exactly, had nonnegative raw depths,
and a relative mass balance error of 3.01 × 10⁻¹⁴.

The reports are retained as `data/cleuson/benchmark_25m.toml` and
`data/cleuson/benchmark_2p5m.toml`. The prepared fine-grid case is in
`cache/swiss_dam/cleuson_2p5m/`; the full 600 s fine-grid run has not been
executed. To run it and render a separate movie:

```sh
julia --project --startup-file=no -t 4 examples/cleuson/run.jl \
  cache/swiss_dam/cleuson_2p5m cache/swiss_dam/frames_2p5m
cache/swiss_dam_venv/bin/python plotting/plot_depth.py \
  --case cache/swiss_dam/cleuson_2p5m --frames cache/swiss_dam/frames_2p5m \
  --output docs/animations/cleuson_dam_break_2p5m.mp4
```

## 5 m simulation and animations

`examples/cleuson/run_5m.sh` prepares a **700 × 1500** grid, runs **200 simulated
seconds** with dam removal at 30 s, saves snapshots every 0.5 s and renders both
an oblique terrain/water view and a top-down depth map:

```sh
bash examples/cleuson/run_5m.sh
```

The script reuses the cached original terrain tiles. Set `SWE_THREADS` to change
the Julia thread count and `SWE_PYTHON` to select a Python environment with the
required libraries. It saves the input configuration and run diagnostics as
`data/cleuson/case_5m.toml`, `verification_5m.toml` and `diagnostics_5m.csv`.
Large prepared arrays and depth snapshots remain under `cache/swiss_dam/`.

[`plotting/plot_depth_3d.py`](../plotting/plot_depth_3d.py) draws the bed as a shaded
grey surface and the water free surface `z+h`, with colour showing water depth.
It uses an orthographic view, fixed 0–80 m square-root colour scale and 1.5×
vertical exaggeration. The display mesh is 10 m along the flooded corridor and
50 m elsewhere; these are rendering settings and do not change the 5 m solver
grid. Depths below 5 cm are hidden. The script accepts different case/frame
directories, camera angles, mesh spacing, exaggeration and a `--preview-time`
for a still image.

Movies: [oblique view](animations/cleuson_dam_break_5m_3d.mp4) and
[top-down view](animations/cleuson_dam_break_5m_2d.mp4).

The top-down view uses the same grey hillshade, elevation contours and fixed
0–80 m depth colour scale as the original 25 m animation. To render it again
from the saved 5 m snapshots without rerunning the solver:

```sh
cache/swiss_dam_venv/bin/python plotting/plot_depth.py \
  --case cache/swiss_dam/cleuson_5m --frames cache/swiss_dam/frames_5m \
  --output docs/animations/cleuson_dam_break_5m_2d.mp4
```

The completed 5 m run used **8723 timesteps** and produced **401 snapshots**.
The solver loop took **1271.69 s (21.2 minutes)** on four CPU threads, averaging
145.8 ms per step including diagnostics and snapshot output. The whole Julia
process took 1320 s with 632 MiB peak resident memory. The intact lake stayed
exactly at rest; raw depth never went below zero and all saved states were finite.
Maximum volume-balance error was 7.674 × 10⁻⁷ m³ (relative 3.878 × 10⁻¹⁴).
At 200 s, the domain retained 17.6300 million m³, cumulative net boundary outflow
was 2.1584 million m³, and 10.1934 million m³ remained in the initial reservoir
mask.

The oblique MP4 is **1536 × 960**, H.264, 20 fps and 22.05 s long, including
one-second holds at its start and end. Rendering took 299 s. Its fixed terrain
projection is cached; terrain and water polygons are sorted together before
drawing. The complete file was decoded successfully with FFmpeg, and initial,
100 s and 200 s preview frames are saved next to it.

The top-down MP4 is **1080 × 1200**, H.264, 20 fps and 22.05 s long, with the
same start/end holds. Its complete file was decoded successfully with FFmpeg;
initial, 150 s and 200 s preview frames are saved next to it.
