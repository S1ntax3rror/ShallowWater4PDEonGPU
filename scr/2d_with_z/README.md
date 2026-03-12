# 2D Shallow Water Solver with Bottom Topography 

This source code implements a **2D shallow water equation (SWE) solver**
with **bottom topography** in Julia. The solver uses a **finite-volume
method with a Rusanov (local Lax--Friedrichs) flux** and explicit time
stepping.

The simulation models shallow water flow over terrain composed of
several smooth hills and visualizes the result using **Makie**.

## Features

-   2D shallow water equations
-   Bottom topography (terrain)
-   Explicit time integration
-   Rusanov numerical flux
-   CFL stability condition
-   Reflective boundary conditions
-   Real-time visualization with Makie
-   Heatmap visualization of the water surface

------------------------------------------------------------------------

# Mathematical Model

## Conserved Variables

The solver uses the conserved state vector

U = (h, hu, hv)

where

-   **h(x,y,t)** = water depth
-   **u(x,y,t)** = velocity in the x-direction
-   **v(x,y,t)** = velocity in the y-direction
-   **hu, hv** = momentum components

------------------------------------------------------------------------

## Governing Equations

The 2D shallow water equations are written as a balance law

∂t U + ∂x F(U) + ∂y G(U) = S(U)

### Flux in x-direction

F(U) = \[ hu, (hu)\^2 / h + ½ g h², hu hv / h\]

### Flux in y-direction

G(U) = \[ hv, hu hv / h, (hv)\^2 / h + ½ g h²\]

### Source Term

Bottom topography introduces source terms

S(U) = \[ 0, - g h ∂x z, - g h ∂y z\]

where **z(x,y)** is the bottom elevation.

------------------------------------------------------------------------

# Numerical Method

## Finite Volume Grid

The computational domain is discretized on a structured grid

(x_i, y_j)

Each cell stores the state

U\_{i,j} = (h, hu, hv)

------------------------------------------------------------------------

## Reconstruction

The solver uses **piecewise constant reconstruction**

U_L = U\[i,j\] U_R = U\[i+1,j\]

and similarly in the y-direction.

------------------------------------------------------------------------

## Rusanov Flux

The Rusanov flux is

F = ½ (F(U_L) + F(U_R)) - ½ a (U_R - U_L)

where

a = max(\|u\| + sqrt(g h))

is the maximum local wave speed.

------------------------------------------------------------------------

## Time Integration

Explicit Euler time stepping is used

U\^{n+1} = U\^n - dt/dx (F\_{i+1/2} - F\_{i-1/2}) - dt/dy (G\_{j+1/2} -
G\_{j-1/2}) + dt S(U)

------------------------------------------------------------------------

## CFL Condition

The time step is computed using

dt = CFL / max( λx/dx + λy/dy )

where

λx = \|u\| + sqrt(g h) λy = \|v\| + sqrt(g h)

Typical CFL value:

CFL ≈ 0.99

------------------------------------------------------------------------

# Bottom Topography

The terrain is constructed from several Gaussian hills

Example:

z = 0.04 \* exp(-((x+2)\^2 + (y+2)\^2)/0.7) + 0.03 \* exp(-((x-1)\^2 +
(y-1)\^2)/0.5) + 0.02 \* exp(-(x\^2 + (y-2)\^2)/0.9)

This produces a smooth landscape over which water flows.

------------------------------------------------------------------------

# Initial Condition

Water initially starts nearly at rest.

The free surface is

γ = h + z

A perturbation can be added to generate waves.

Example:

h(x,y) = γ0 - z + perturbation

------------------------------------------------------------------------

# Code Structure

## State Representation

Each grid cell stores a small static vector

``` julia
S = zeros(SVector{3,Float64}, nx, ny)
```

Each entry represents

S\[i,j\] = (h, hu, hv)

Using **StaticArrays** improves performance by avoiding heap
allocations.

------------------------------------------------------------------------


## Visualization

The simulation visualizes

-   a 1D cross-section of the water surface
-   a 2D heatmap of the water height

Makie **Observables** are used to update plots during the simulation.

Example:

``` julia
h_obs = Observable(getindex.(S,1))
heatmap!(ax, xs, ys, h_obs)
```

------------------------------------------------------------------------

# Running the Simulation

Run the main function

``` julia
swe2d_topography()
```

The simulation will generate an animation showing the evolution of the
water surface.


------------------------------------------------------------------------

# References

LeVeque --- *Finite Volume Methods for Hyperbolic Problems*

Toro --- *Riemann Solvers and Numerical Methods for Fluid Dynamics*

Audusse et al. --- *Well-balanced schemes for shallow water equations*