
# 1D and 2D shallow water solvers

#------------------------------------------------------------------------------
# 1D flux and characteristic speed
#------------------------------------------------------------------------------
f(S) = SA[S[2], S[2]^2/S[1]+0.5*g*S[1]^2]

# characteristic speed magnitude
# λ = u ± √gh
λ(S) = abs(S[2] / S[1]) + sqrt(g * S[1])

#------------------------------------------------------------------------------
# 2D fluxes and characteristic speeds
#------------------------------------------------------------------------------

# x-flux
fx(S) = SA[
    S[2],
    S[2]^2 / S[1] + 0.5 * g * S[1]^2,
    S[2] * S[3] / S[1]
]

# y-flux
fy(S) = SA[
    S[3],
    S[2] * S[3] / S[1],
    S[3]^2 / S[1] + 0.5 * g * S[1]^2
]

# characteristic wave speed estimates
λx(S) = abs(S[2] / S[1]) + sqrt(g * S[1])
λy(S) = abs(S[3] / S[1]) + sqrt(g * S[1])


#-----------------------------------------------------------------------------
# 1D shallow water solvers
#-----------------------------------------------------------------------------

# reference solver without topography from Ivan Utkin

@views function solve(S,Sᴸ,Sᴿ,F,dx)
    # reconstruction step (piecewise constant)
    @. Sᴸ = S[1:end-1]
    @. Sᴿ = S[2:end]
    # Rusanov flux (diffusion is locally proportional to characteristic speed)
    @. F = 0.5 * (f(Sᴸ) + f(Sᴿ)) - 0.5 * max(λ(Sᴸ), λ(Sᴿ)) * (Sᴿ - Sᴸ)
    # time step from CFL condition
    dt = 0.99 * dx / maximum(λ.(S))
    # state update
    @. S[2:end-1] -= dt * (F[2:end] - F[1:end-1]) / dx
    # boundary conditions
    # mass is copied:       dh/dx = 0
    # momentum is mirrored: hu    = 0
    S[1] = S[2][1], -S[2][2]
    S[end] = S[end-1][1], -S[end-1][2]
    return nothing
end

# solver with topography source term

@views function solve_z(S,Sᴸ,Sᴿ,F,dx,dzdx, nx)
    # reconstruction step (piecewise constant)
    @. Sᴸ = S[1:end-1]
    @. Sᴿ = S[2:end]

    # Rusanov flux
    @. F = 0.5 * (f(Sᴸ) + f(Sᴿ)) - 0.5 * max(λ(Sᴸ), λ(Sᴿ)) * (Sᴿ - Sᴸ)

    # CFL time step
    dt = 0.99 * dx / maximum(λ.(S))

    # conservative flux update
    @. S[2:end-1] -= dt * (F[2:end] - F[1:end-1]) / dx

    # source term in momentum equation: -(g h z_x)
    for i in 2:nx-1
        h  = S[i][1]
        hu = S[i][2] - dt * g * h * dzdx[i]
        S[i] = SVector(h, hu)
    end

    # reflective boundary conditions
    S[1]   = SVector(S[2][1],   -S[2][2])
    S[end] = SVector(S[end-1][1], -S[end-1][2])

    # safety: keep h positive
    for i in eachindex(S)
        if S[i][1] <= 0
            S[i] = SVector(1e-8, S[i][2])
        end
    end

    return nothing
end


# -----------------------------------------------------------------------------
# 2D shallow water with topography
# -----------------------------------------------------------------------------

@views function solve_z_2d(S, Sᴸ, Sᴿ, Sᴮ, Sᵀ, F, G, dx, dy, dzdx, dzdy, nx, ny)
    # reconstruction step (piecewise constant)
    @. Sᴸ = S[1:end-1, :]
    @. Sᴿ = S[2:end, :]
    @. Sᴮ = S[:, 1:end-1]
    @. Sᵀ = S[:, 2:end]

    # Rusanov fluxes
    @. F = 0.5 * (fx(Sᴸ) + fx(Sᴿ)) - 0.5 * max(λx(Sᴸ), λx(Sᴿ)) * (Sᴿ - Sᴸ)
    @. G = 0.5 * (fy(Sᴮ) + fy(Sᵀ)) - 0.5 * max(λy(Sᴮ), λy(Sᵀ)) * (Sᵀ - Sᴮ)

    # CFL time step
    dt = 0.99 / maximum(λx.(S) ./ dx .+ λy.(S) ./ dy)

    # conservative update
    @. S[2:end-1, 2:end-1] -= dt * (
        (F[2:end,   2:end-1] - F[1:end-1, 2:end-1]) / dx +
        (G[2:end-1, 2:end]   - G[2:end-1, 1:end-1]) / dy
    )

    # source term update
    for i in 2:nx-1, j in 2:ny-1
        h  = S[i, j][1]
        hu = S[i, j][2] - dt * g * h * dzdx[i, j]
        hv = S[i, j][3] - dt * g * h * dzdy[i, j]
        S[i, j] = SVector(h, hu, hv)
    end

    # reflective boundary conditions
    for j in 1:ny
        S[1, j]   = SVector(S[2, j][1],      -S[2, j][2],      S[2, j][3])
        S[end, j] = SVector(S[end-1, j][1],  -S[end-1, j][2],  S[end-1, j][3])
    end

    for i in 1:nx
        S[i, 1]   = SVector(S[i, 2][1],      S[i, 2][2],      -S[i, 2][3])
        S[i, end] = SVector(S[i, end-1][1],  S[i, end-1][2],  -S[i, end-1][3])
    end

    # positivity fix
    for i in 1:nx, j in 1:ny
        if S[i, j][1] <= 0
            S[i, j] = SVector(hmin, S[i, j][2], S[i, j][3])
        end
    end
    return nothing
end

# Kernel functions 



@inline avx_comp(hv1, hv2, h, ix, iy) = 0.5 * (hv1[ix, iy] * hv2[ix, iy] / h[ix, iy] + hv1[ix+1, iy] * hv2[ix+1, iy] / h[ix+1, iy])
@inline avy_comp(hv1, hv2, h, ix, iy) = 0.5 * (hv1[ix, iy] * hv2[ix, iy] / h[ix, iy] + hv1[ix, iy+1] * hv2[ix, iy+1] / h[ix, iy+1])
@inline avx_simp(h, ix, iy) = 0.5 * (h[ix, iy] * h[ix, iy] + h[ix+1, iy] * h[ix+1, iy])
@inline avy_simp(h, ix, iy) = 0.5 * (h[ix, iy] * h[ix, iy] + h[ix, iy+1] * h[ix, iy+1])
@inline dxa(h, ix, iy) = h[ix+1, iy] - h[ix, iy]
@inline dya(h, ix, iy) = h[ix, iy+1] - h[ix, iy]

@inline dxb(h, ix, iy) = h[ix, iy] - h[ix-1, iy]
@inline dyb(h, ix, iy) = h[ix, iy] - h[ix, iy-1]

@inline eta(h, z, ix, iy) = h[ix, iy] + z[ix, iy]

@inline zx_face(z, ix, iy) = 0.5 * (z[ix, iy] + z[ix+1, iy])
@inline zy_face(z, ix, iy) = 0.5 * (z[ix, iy] + z[ix, iy+1])

@inline hx_L(h, z, ix, iy) =
    max(0.0, eta(h, z, ix, iy) - zx_face(z, ix, iy))

@inline hx_R(h, z, ix, iy) =
    max(0.0, eta(h, z, ix+1, iy) - zx_face(z, ix, iy))

@inline hy_L(h, z, ix, iy) =
    max(0.0, eta(h, z, ix, iy) - zy_face(z, ix, iy))

@inline hy_R(h, z, ix, iy) =
    max(0.0, eta(h, z, ix, iy+1) - zy_face(z, ix, iy))

@inline function desing_velocity(hval, qval, vel_eps)
    if hval <= 0.0
        return 0.0
    end

    return sqrt(2.0) * hval * qval /
           sqrt(hval^4 + max(hval^4, vel_eps))
end

@inline vel_u(h, hu, ix, iy, vel_eps) =
    desing_velocity(h[ix, iy], hu[ix, iy], vel_eps)

@inline vel_v(h, hv, ix, iy, vel_eps) =
    desing_velocity(h[ix, iy], hv[ix, iy], vel_eps)

@inline bc_speed_x(h, hu, ix, iy, g) =
    h[ix, iy] > h_eps ?
        abs(hu[ix, iy] / h[ix, iy]) + sqrt(g * h[ix, iy]) :
        0.0

@inline bc_speed_y(h, hv, ix, iy, g) =
    h[ix, iy] > h_eps ?
        abs(hv[ix, iy] / h[ix, iy]) + sqrt(g * h[ix, iy]) :
        0.0




# Kernel flux function for the 2D SWE well balanced xpu solver

@parallel_indices (ix, iy) function compute_1st_2nd_and_3th_flux_wb!(
    F₁, F₂, F₃,
    G₁, G₂, G₃,
    hu, hv, h, z, g,
    max_speed_x, max_speed_y,
    vel_eps
)
    nx, ny = size(h)

    # -------------------------------------------------------------------------
    # x-direction fluxes
    # -------------------------------------------------------------------------
    if ix <= nx - 1 && iy <= ny
        hL = hx_L(h, z, ix, iy)
        hR = hx_R(h, z, ix, iy)

        ηL = eta(h, z, ix, iy)
        ηR = eta(h, z, ix+1, iy)

        uL = vel_u(h, hu, ix, iy, vel_eps)
        uR = vel_u(h, hu, ix+1, iy, vel_eps)

        vL = vel_v(h, hv, ix, iy, vel_eps)
        vR = vel_v(h, hv, ix+1, iy, vel_eps)

        # Reconstruct momenta consistently:
        # momentum = reconstructed depth × cell velocity
        huL = hL * uL
        huR = hR * uR
        hvL = hL * vL
        hvR = hR * vR

        ax = max_speed_x[ix, iy]

        # Mass / free-surface flux
        F₁[ix, iy] =
            0.5 * (huL + huR) -
            0.5 * ax * (ηR - ηL)

        # x-momentum flux
        F₂[ix, iy] =
            0.5 * (
                huL * uL + 0.5 * g * hL^2 +
                huR * uR + 0.5 * g * hR^2
            ) -
            0.5 * ax * (huR - huL)

        # y-momentum transported in x
        F₃[ix, iy] =
            0.5 * (
                huL * vL +
                huR * vR
            ) -
            0.5 * ax * (hvR - hvL)
    end

    # -------------------------------------------------------------------------
    # y-direction fluxes
    # -------------------------------------------------------------------------
    if ix <= nx && iy <= ny - 1
        hL = hy_L(h, z, ix, iy)
        hR = hy_R(h, z, ix, iy)

        ηL = eta(h, z, ix, iy)
        ηR = eta(h, z, ix, iy+1)

        uL = vel_u(h, hu, ix, iy, vel_eps)
        uR = vel_u(h, hu, ix, iy+1, vel_eps)

        vL = vel_v(h, hv, ix, iy, vel_eps)
        vR = vel_v(h, hv, ix, iy+1, vel_eps)

        huL = hL * uL
        huR = hR * uR
        hvL = hL * vL
        hvR = hR * vR

        ay = max_speed_y[ix, iy]

        # Mass / free-surface flux
        G₁[ix, iy] =
            0.5 * (hvL + hvR) -
            0.5 * ay * (ηR - ηL)

        # x-momentum transported in y
        G₂[ix, iy] =
            0.5 * (
                hvL * uL +
                hvR * uR
            ) -
            0.5 * ay * (huR - huL)

        # y-momentum flux
        G₃[ix, iy] =
            0.5 * (
                hvL * vL + 0.5 * g * hL^2 +
                hvR * vR + 0.5 * g * hR^2
            ) -
            0.5 * ay * (hvR - hvL)
    end

    return nothing
end

# Kernel flux function for the 2d SWE cpu solver (not well balanced, for testing only)

@parallel_indices (ix, iy) function compute_1st_2nd_and_3th_flux!(F₁, F₂, F₃, G₁, G₂, G₃, hu, hv, h, g, max_speed_x, max_speed_y)
    nx, ny = size(h)
    if (ix <= nx - 1 && iy <= ny)
        F₁[ix, iy] = 0.5 * (hu[ix, iy] + hu[ix+1, iy]) - 0.5 * max_speed_x[ix, iy] * dxa(h, ix, iy)
        F₂[ix, iy] = avx_comp(hu, hu, h, ix, iy) + 0.5 * g * avx_simp(h, ix, iy) - 0.5 * max_speed_x[ix, iy] * dxa(hu, ix, iy)
        F₃[ix, iy] = avx_comp(hu, hv, h, ix, iy) - 0.5 * max_speed_x[ix, iy] * dxa(hv, ix, iy)
    end
    if (ix <= nx && iy <= ny - 1)
        G₁[ix, iy] = 0.5 * (hv[ix, iy] + hv[ix, iy+1]) - 0.5 * max_speed_y[ix, iy] * dya(h, ix, iy)
        G₂[ix, iy] = avy_comp(hv, hu, h, ix, iy) - 0.5 * max_speed_y[ix, iy] * dya(hu, ix, iy)
        G₃[ix, iy] = avy_comp(hv, hv, h, ix, iy) + 0.5 * g * avy_simp(h, ix, iy) - 0.5 * max_speed_y[ix, iy] * dya(hv, ix, iy)
    end
    return nothing
end
