# First-order wet/dry treatment

Both final solvers retain Rusanov fluxes and forward Euler stepping. The supplied
Hwang–Lynett–Son paper motivates the key requirements: non-negative reconstructed
depths, independent x/y shoreline treatment, and pressure/source cancellation at
rest. The concrete implementation uses the first-order hydrostatic reconstruction
of [Audusse et al. (2004)](https://publications.imp.fu-berlin.de/478/1/file_2004_siam.pdf),
equations (2.5) and (2.8)–(2.11). It does not reproduce the supplied paper's
second-order subcell reconstruction or directional draining formula.

## Reconstruction and update

At each face, set the common bed barrier to `max(z_left, z_right)` and clip both
free surfaces against that barrier. Reconstruct momentum as corrected depth times
regularized cell velocity. Use the corrected depth jump in the Rusanov mass flux.

For a lake at rest, corrected depths coincide on both sides of every face. At a
wet/dry interface below the exposed bed, both depths vanish. The mass flux is
therefore zero, including on dry interfaces.

The normal momentum flux stores transport and pressure separately. The pressure
residual at each cell side is `P_face - g*h_face^2/2`, using that cell's corrected
face depth. Each residual vanishes at rest. This implements the bed correction
without assuming that both faces are flooded.

The draining limiter bounds the total outgoing volume by the cell's stored water
volume. Each shared face uses its donor's limit. Mass and momentum transport use
the same effective timestep; pressure and its bed correction use the global
timestep. This keeps the update conservative and retains hydrostatic balance.

Positive shallow depths are retained during cleanup; only their momentum is
zeroed below `h_eps = 1e-10`. Non-finite states are no longer silently replaced by
dry cells. Radiative boundaries extrapolate free surface with an exposed-bed
correction, so non-flat boundaries preserve rest states. Rectangular grids use
explicit boundary launch ranges.

MPI exchanges initial state/bathymetry and donor draining limits. Each step
exchanges the state after cleanup and physical boundary conditions. Both solvers
use CFL = 0.45, with global MPI speed reductions every step. These choices add
communication compared with the previous ten-step timestep reuse and overlapping
state update. Performance has not been benchmarked.

## Reproduce

From the repository root:

```bash
julia --project --startup-file=no test/test_wet_dry_wb.jl
```

The driver includes and calls the actual solver files. It uses a 34 × 30 grid,
including ghost cells, on `[-1,1] × [-1,1]`, 200 steps, `g = 1`, Float64, and two CPU threads
per process. Sponge damping is disabled for comparison. MPI layouts are 1 × 1,
2 × 1, and 2 × 2; shorelines and moving fronts cross decomposition boundaries.
The fully dry case exits immediately because every wave speed is zero.

Seven equilibrium cases cover smooth submerged bed, an emergent island, an
oblique shoreline, a discontinuous bed step, rough shorelines, a fully dry domain,
and a uniform depth of `5e-12`. Tests include initially dry and shallow cells,
momentum, finite states, and total interior mass. Two moving cases check a dry-bed
dam break and a small shoreline perturbation inside a basin with dry walls.
An additional deliberately excessive timestep activates the draining limiter;
positivity and conservation are checked before cleanup.

## Results

Single-XPU results from 2026-10-09:

| Equilibrium | Maximum depth change | Maximum momentum |
|---|---:|---:|
| Fully wet | 2.78e-17 | 0 |
| Island | 1.67e-16 | 0 |
| Oblique shoreline | 2.22e-16 | 4.14e-16 |
| Bed step | 0 | 0 |
| Rough shoreline | 7.36e-16 | 0 |
| Fully dry | 0 | 0 |
| Shallow water | 0 | 0 |

The original island reconstruction produced momentum of `3.24e-4` after a single
step (`dt = 0.005`); the corrected solver remains at roundoff over 200 steps.
The moving tests remain non-negative, wet previously dry cells, and conserve
interior mass to roundoff. The reported mass changes are below `6e-13` in the
single-XPU cases (unscaled sums of depth).

All three MPI layouts pass the same cases. Their largest equilibrium depth change
is `9.16e-16` (rough shoreline), and maximum equilibrium momentum is `3.46e-16`
(oblique shoreline). Full-state comparisons against single-XPU pass at tolerance
`1e-13` for `h`, `hu`, `hv`, and `z`, including the two moving cases. Simulation
times agree at relative/absolute tolerance `1e-13`.

The new suite passes 431 assertions across the single-XPU run, all MPI ranks,
and decomposition comparisons. The existing 1D reference and sequential/XPU flux
tests also pass all 80 assertions.

Validation runs use the Threads backend. GPU execution requires a separate run
on GPU hardware; no GPU device is available on this machine.
