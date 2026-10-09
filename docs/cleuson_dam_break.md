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
