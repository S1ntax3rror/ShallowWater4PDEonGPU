# Well-Balanced First-Order SWE Solver with Wet/Dry Handling

This note summarizes the numerical changes made to the 2D shallow-water solver, motivated by the paper:

> S. Hwang, P. J. Lynett, and S. Son,  
> *A second-order well-balanced reconstruction for the shallow flows with wet/dry fronts*,  
> Computers and Mathematics with Applications, 208 (2026), 33–54.

The paper develops a **second-order central-upwind**, **well-balanced**, **positivity-preserving** finite-volume method for the 2D shallow-water equations with wet/dry fronts.  
Our code currently implements a **first-order Rusanov analogue** of the same core ideas:

1. reconstruct using the **free surface**
   $ \eta = h + z, $
2. evaluate water depths at interfaces using the **face bathymetry**,
3. use a **flux/source pairing** that preserves the lake-at-rest equilibrium,
4. apply a preliminary **wet/dry treatment**,
5. add a **draining timestep** to prevent cells from losing more water than they contain.

---

## 1. Governing equations

The 2D shallow-water equations over fixed bathymetry $z(x,y)$ are

```math
\partial_t h
+
\partial_x(hu)
+
\partial_y(hv)
=
0,

\partial_t(hu)
+
\partial_x\left(hu^2 + \frac{1}{2}gh^2\right)
+
\partial_y(huv)
=
-gh\,\partial_x z,
```

$$
\partial_t(hv)
+
\partial_x(huv)
+
\partial_y\left(hv^2 + \frac{1}{2}gh^2\right)
=
-gh\,\partial_y z.
$$

The lake-at-rest equilibrium is

$$
u=v=0,
\qquad
\eta := h+z = \text{constant}.
$$

A method is **well-balanced** if it preserves this state exactly, up to roundoff.

---

## 2. Why the original scheme was not well-balanced

The original mass flux used Rusanov diffusion on $h$:

$$
F_1^{\text{old}}
=
\frac{1}{2}(hu_L+hu_R)
-
\frac{1}{2}a(h_R-h_L).
$$

At lake-at-rest,

$$
hu_L=hu_R=0,
$$

but the water depth generally varies over topography:

$$
h_R-h_L \neq 0.
$$

Therefore,

$$
F_1^{\text{old}} \neq 0,
$$

so the solver generated artificial mass transport even when the exact solution should be stationary.

The momentum equation also failed to balance because the discrete hydrostatic pressure flux was not paired with a source term that cancels it exactly.

---

# 3. Paper idea: evolve/reconstruct the free surface

The paper rewrites the conserved variable as

$$
U =
\begin{bmatrix}
w\\ hu\\ hv
\end{bmatrix},
\qquad
w=h+B,
$$

where $B$ is the bottom elevation.  
In our notation,

$$
w \equiv \eta,
\qquad
B \equiv z.
$$

The key observation is:

$$
\eta=\text{constant}
$$

at lake-at-rest, whereas $h$ is not constant over variable bathymetry.

---

# 4. First-order well-balanced interface reconstruction

Our current code uses a **first-order piecewise-constant reconstruction**.

At the $x$-interface $i+1/2,j$,

$$
z_{i+1/2,j}
=
\frac{z_{i,j}+z_{i+1,j}}{2}.
$$

The free-surface values are

$$
\eta_L = h_{i,j}+z_{i,j},
\qquad
\eta_R = h_{i+1,j}+z_{i+1,j}.
$$

The reconstructed interface depths are

$$
h_L
=
\max\left(0,\eta_L-z_{i+1/2,j}\right),
$$

$$
h_R
=
\max\left(0,\eta_R-z_{i+1/2,j}\right).
$$

Similarly, at the $y$-interface $i,j+1/2$,

$$
z_{i,j+1/2}
=
\frac{z_{i,j}+z_{i,j+1}}{2},
$$

$$
h_L
=
\max\left(0,\eta_{i,j}-z_{i,j+1/2}\right),
$$

$$
h_R
=
\max\left(0,\eta_{i,j+1}-z_{i,j+1/2}\right).
$$

---

# 5. Velocity treatment near wet/dry fronts

A direct evaluation

$$
u = \frac{hu}{h},
\qquad
v = \frac{hv}{h}
$$

becomes numerically dangerous when $h$ is very small. Even tiny momenta can generate extremely large velocities, which inflate the local wave speed

$$
|u|+\sqrt{gh},
\qquad
|v|+\sqrt{gh},
$$

and can collapse the global CFL timestep.

To avoid this, the current implementation uses a **desingularized velocity formula**:

$$
\boxed{
u =
\frac{
\sqrt{2}\,h\,(hu)
}{
\sqrt{h^4+\max(h^4,\varepsilon_{\mathrm{vel}})}
}
}
$$

$$
\boxed{
v =
\frac{
\sqrt{2}\,h\,(hv)
}{
\sqrt{h^4+\max(h^4,\varepsilon_{\mathrm{vel}})}
}
}
$$

with

$$
\varepsilon_{\mathrm{vel}}
=
\min(\Delta x,\Delta y)^4.
$$

For well-resolved wet cells,

$$
h^4 \gg \varepsilon_{\mathrm{vel}},
$$

so the formula reduces to

$$
u \approx \frac{hu}{h},
\qquad
v \approx \frac{hv}{h}.
$$

For nearly dry cells,

$$
h^4 \lesssim \varepsilon_{\mathrm{vel}},
$$

the velocity is smoothly regularized instead of becoming unbounded.

In code:

```julia
@inline function desing_velocity(hval, qval, vel_eps)
    if hval <= 0.0
        return 0.0
    end

    return sqrt(2.0) * hval * qval /
           sqrt(hval^4 + max(hval^4, vel_eps))
end
```

The current run additionally uses a dry-cell threshold

$$
h_\varepsilon = 10^{-2},
$$

below which dry cleanup sets

$$
h=0,\qquad hu=0,\qquad hv=0.
$$

---

# 6. First-order well-balanced Rusanov flux

The physical fluxes are evaluated using reconstructed interface depths.

For the $x$-direction,

$$
F(U)
=
\begin{bmatrix}
hu\\[1ex]
hu\,u + \frac{1}{2}gh^2\\[1ex]
hu\,v
\end{bmatrix}.
$$

For the $y$-direction,

$$
G(U)
=
\begin{bmatrix}
hv\\[1ex]
hv\,u\\[1ex]
hv\,v + \frac{1}{2}gh^2
\end{bmatrix}.
$$

---

## 6.1. $x$-direction Rusanov flux

Define

$$
a_{i+1/2,j}
=
\max
\left(
|u_L|+\sqrt{gh_L},
|u_R|+\sqrt{gh_R}
\right).
$$

The reconstructed momenta are

$$
(hu)_L = h_L u_L,
\qquad
(hu)_R = h_R u_R,
$$

$$
(hv)_L = h_L v_L,
\qquad
(hv)_R = h_R v_R.
$$

### Mass flux

The well-balanced modification is

$$
\boxed{
F_{1,i+1/2,j}
=
\frac{1}{2}\left[(hu)_L+(hu)_R\right]
-
\frac{1}{2}a_{i+1/2,j}(\eta_R-\eta_L)
}
$$

rather than diffusing $h_R-h_L$.

At lake-at-rest,

$$
\eta_R-\eta_L = 0,
$$

so

$$
F_{1,i+1/2,j}=0.
$$

### $x$-momentum flux

$$
F_{2,i+1/2,j}
=
\frac{1}{2}
\left[
(hu)_L u_L + \frac{1}{2}g h_L^2
+
(hu)_R u_R + \frac{1}{2}g h_R^2
\right]
-
\frac{1}{2}a_{i+1/2,j}\left[(hu)_R-(hu)_L\right].
$$

### Cross-momentum flux

$$
F_{3,i+1/2,j}
=
\frac{1}{2}
\left[
(hu)_L v_L
+
(hu)_R v_R
\right]
-
\frac{1}{2}a_{i+1/2,j}\left[(hv)_R-(hv)_L\right].
$$

---

## 6.2. $y$-direction Rusanov flux

Define

$$
b_{i,j+1/2}
=
\max
\left(
|v_L|+\sqrt{gh_L},
|v_R|+\sqrt{gh_R}
\right).
$$

### Mass flux

$$
\boxed{
G_{1,i,j+1/2}
=
\frac{1}{2}\left[(hv)_L+(hv)_R\right]
-
\frac{1}{2}b_{i,j+1/2}(\eta_R-\eta_L)
}
$$

### $x$-momentum transported in $y$

$$
G_{2,i,j+1/2}
=
\frac{1}{2}
\left[
(hv)_L u_L
+
(hv)_R u_R
\right]
-
\frac{1}{2}b_{i,j+1/2}
\left[(hu)_R-(hu)_L\right].
$$

### $y$-momentum flux

$$
G_{3,i,j+1/2}
=
\frac{1}{2}
\left[
(hv)_L v_L+\frac{1}{2}g h_L^2
+
(hv)_R v_R+\frac{1}{2}g h_R^2
\right]
-
\frac{1}{2}b_{i,j+1/2}
\left[(hv)_R-(hv)_L\right].
$$

---

# 7. Well-balanced source term

The source term must cancel the hydrostatic pressure flux difference exactly at lake-at-rest.

At cell $(i,j)$, define face bathymetry values

$$
z_E = \frac{z_{i,j}+z_{i+1,j}}{2},
\qquad
z_W = \frac{z_{i-1,j}+z_{i,j}}{2},
$$

$$
z_N = \frac{z_{i,j}+z_{i,j+1}}{2},
\qquad
z_S = \frac{z_{i,j-1}+z_{i,j}}{2}.
$$

Define local face depths using the cell-centered free surface

$$
\eta_C = h_{i,j}+z_{i,j},
$$

$$
h_E = \max(0,\eta_C-z_E),
\qquad
h_W = \max(0,\eta_C-z_W),
$$

$$
h_N = \max(0,\eta_C-z_N),
\qquad
h_S = \max(0,\eta_C-z_S).
$$

Then

$$
h^{\text{src}}_x
=
\frac{h_E+h_W}{2},
\qquad
h^{\text{src}}_y
=
\frac{h_N+h_S}{2}.
$$

The source terms used in the code are

$$
S^{(2)}_{i,j}
=
-g\,h^{\text{src}}_x
\frac{z_E-z_W}{\Delta x},
$$

$$
S^{(3)}_{i,j}
=
-g\,h^{\text{src}}_y
\frac{z_N-z_S}{\Delta y}.
$$

For a fully wet lake-at-rest state, these source terms exactly cancel the face-based hydrostatic pressure flux differences.

---

# 8. Finite-volume update

The momentum updates are

$$
(hu)^{n+1}_{i,j}
=
(hu)^n_{i,j}
-
\Delta t
\left[
\frac{F_{2,i+1/2,j}-F_{2,i-1/2,j}}{\Delta x}
+
\frac{G_{2,i,j+1/2}-G_{2,i,j-1/2}}{\Delta y}
\right]
+
\Delta t\,S^{(2)}_{i,j},
$$

$$
(hv)^{n+1}_{i,j}
=
(hv)^n_{i,j}
-
\Delta t
\left[
\frac{F_{3,i+1/2,j}-F_{3,i-1/2,j}}{\Delta x}
+
\frac{G_{3,i,j+1/2}-G_{3,i,j-1/2}}{\Delta y}
\right]
+
\Delta t\,S^{(3)}_{i,j}.
$$

The water-depth update is modified by the draining timestep, described below.

---

# 9. Wet/dry treatment

## 9.1. What the paper does

The paper distinguishes:

- dry cells,
- fully flooded cells,
- partially flooded cells.

It reconstructs the free surface and corrects interface values when naive reconstruction would produce negative local water depths. In dry cells, reconstructed depths at interfaces are set to zero. In partially flooded cells, the free-surface reconstruction is modified to remain nonnegative and preserve the cell-average water content.

---

## 9.2. What the current first-order code does

The current code uses a simpler first-order wet/dry procedure:

### Interface non-negativity

$$
h_L = \max(0,\eta_L-z_f),
\qquad
h_R = \max(0,\eta_R-z_f).
$$

### Desingularized velocity evaluation

Instead of using raw division $hu/h$ and $hv/h$ close to shore, the code uses

$$
u =
\frac{
\sqrt{2}\,h\,(hu)
}{
\sqrt{h^4+\max(h^4,\varepsilon_{\mathrm{vel}})}
},
$$

$$
v =
\frac{
\sqrt{2}\,h\,(hv)
}{
\sqrt{h^4+\max(h^4,\varepsilon_{\mathrm{vel}})}
}.
$$

This suppresses unphysically large velocities in almost-dry cells and prevents severe timestep collapse when the wave reaches the coastline.

### Dry-cell cleanup

After each update,

$$
h<h_\varepsilon
\quad\Longrightarrow\quad
h=0,\;hu=0,\;hv=0.
$$

This is a practical first-order approximation of the paper’s more elaborate wet/dry reconstruction.

---

# 10. Draining timestep

The purpose of a draining timestep is to prevent one cell from losing more water than it contains during a single step.

---

## 10.1. Paper concept

The paper computes directional draining timesteps in partially flooded cells. Because the effective water amount can differ in the $x$- and $y$-directions, the draining constraint is direction-dependent.

For outgoing mass fluxes,

$$
Q^{\text{out}}
=
\max(F_{1,E},0)
+
\max(-F_{1,W},0)
+
\max(G_{1,N},0)
+
\max(-G_{1,S},0).
$$

The paper defines local drain times and uses

$$
\Delta t_{\text{face}}
=
\min(\Delta t,\Delta t_{\text{drain}})
$$

on outgoing fluxes. This suppresses excessive outflow while preserving conservation.

---

## 10.2. Current first-order implementation

Our current code uses the cell-average approximation

$$
R_{i,j}
=
\frac{
\max(F_{1,E},0)+\max(-F_{1,W},0)
}{\Delta x}
+
\frac{
\max(G_{1,N},0)+\max(-G_{1,S},0)
}{\Delta y}.
$$

Then

$$
\Delta t^{\text{drain}}_{i,j}
=
\begin{cases}
\min\left(\Delta t,\dfrac{h_{i,j}}{R_{i,j}}\right),
& R_{i,j}>0,\\[2ex]
\Delta t,
& R_{i,j}=0.
\end{cases}
$$

For each face, the effective timestep is assigned from the **upwind draining cell**:

- if $F_{1,i+1/2,j}>0$, use the drain time of cell $(i,j)$,
- if $F_{1,i+1/2,j}<0$, use the drain time of cell $(i+1,j)$,

and similarly in the $y$-direction.

The water-depth update becomes

$$
h^{n+1}_{i,j}
=
h^n_{i,j}
-
\frac{
F_{1,E}\Delta t_E
-
F_{1,W}\Delta t_W
}{\Delta x}
-
\frac{
G_{1,N}\Delta t_N
-
G_{1,S}\Delta t_S
}{\Delta y}.
$$

This is the current positivity-preserving stabilization added to the first-order solver.

---

# 11. Boundary conditions and sponge layer

The paper studies several benchmark-specific boundary conditions such as periodic, solid-wall, and open boundaries.  
Our solver currently uses a separate absorbing strategy:

1. an outer enlarged computational domain,
2. a sponge layer that damps momentum:
   $
hu \leftarrow (1-\sigma)hu,
   \qquad
   hv \leftarrow (1-\sigma)hv.
$

This sponge layer is **not** the wet/dry treatment.  
It is an outer-domain absorbing mechanism intended to reduce artificial reflections.

The boundary update must still be made dry-safe:

- avoid divisions by $h=0$,
- avoid $\sqrt{gh}$ with negative depths,
- avoid parallel race conditions from duplicate boundary writes.

---

# 12. Steady-state diagnostics

For a lake-at-rest test, define

$$
e_{i,j}
=
\left|
(h_{i,j}+z_{i,j})-\eta_{0,i,j}
\right|.
$$

A useful error metric is the wet-cell $L^\infty$ error:

$$
\|e\|_{L^\infty(\Omega_{\text{wet}})}
=
\max_{(i,j)\in\Omega_{\text{wet}}}
e_{i,j},
$$

where

$$
\Omega_{\text{wet}}
=
\{(i,j): h_{i,j}>h_\varepsilon\}.
$$

This measures the largest steady-state free-surface error only where water is actually present.

---

# 13. Current status

## Implemented

- first-order free-surface-based reconstruction,
- interface depths from face bathymetry,
- Rusanov diffusion on free-surface differences,
- compatible face-based source term,
- nonnegative reconstructed interface depths,
- desingularized velocity evaluation near dry cells,
- dry-cell cleanup,
- first-order draining timestep approximation,
- wet-cell $L^\infty$ steady-state error diagnostic.


---

# 14. Summary of the numerical modification

The core well-balanced replacement is

$$
\boxed{
\text{diffuse }\eta=h+z\text{ instead of }h
}
$$

and

$$
\boxed{
\text{compute face pressure terms from face-reconstructed depths}
}
$$

with a source term designed to cancel the hydrostatic pressure contribution exactly at rest.

The wet/dry and draining modifications then prevent invalid negative depths and preserve numerical stability near moving shorelines. The additional velocity desingularization is essential in practice: it prevents nearly dry cells from generating unrealistically large velocities and therefore avoids catastrophic CFL timestep collapse during shoreline interaction.
