using CairoMakie   # for visualisation
using StaticArrays # for small fixed-size state vectors

const g = 1.0

# -----------------------------------------------------------------------------
# Fluxes
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 2D shallow water with topography
# -----------------------------------------------------------------------------

@views function swe2d_topography()
    # physics
    lx = 50.0
    ly = 50.0

    # numerics
    nx   = 250
    ny   = 250
    nt   = 1.5 * nx
    nvis = 5

    # derived numerics
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)

    xs = LinRange(-lx / 2, lx / 2, nx)
    ys = LinRange(-ly / 2, ly / 2, ny)

    # -------------------------------------------------------------------------
    # Array initialisation
    # -------------------------------------------------------------------------

    # conservative state: S = (h, hu, hv)
    S = zeros(SVector{3, Float64}, nx, ny)

    # interface states
    Sᴸ = zeros(SVector{3, Float64}, nx - 1, ny)
    Sᴿ = zeros(SVector{3, Float64}, nx - 1, ny)
    Sᴮ = zeros(SVector{3, Float64}, nx, ny - 1)
    Sᵀ = zeros(SVector{3, Float64}, nx, ny - 1)

    # numerical fluxes
    F = zeros(SVector{3, Float64}, nx - 1, ny)
    G = zeros(SVector{3, Float64}, nx, ny - 1)

    # -------------------------------------------------------------------------
    # Bottom topography
    # -------------------------------------------------------------------------

    # smooth island
    x0    = -10.0
    y0    = 0.0
    zmax  = 0.12
    rflat = 3.0
    redge = 4.5

    R = sqrt.((xs .- x0).^2 .+ (ys' .- y0).^2)
    island = similar(R)

    for j in axes(R, 1), i in axes(R, 2)
        r = R[j, i]
        if r <= rflat
            island[j, i] = zmax
        elseif r <= redge
            s = (r - rflat) / (redge - rflat)
            island[j, i] = zmax * 0.5 * (1 + cos(pi * s))
        else
            island[j, i] = 0.0
        end
    end

    # total bottom topography
    z =
        island .+
        0.07 .* exp.(-((xs .- 7).^2 .+ (ys').^2)) .+
        0.02 .* exp.(-(ys.^2 .+ (xs' .- 2).^2))

    # bottom slopes
    dzdx = zeros(nx, ny)
    dzdy = zeros(nx, ny)

    dzdx[2:end-1, :] .= (z[3:end, :] .- z[1:end-2, :]) ./ (2dx)
    dzdx[1, :]       .= dzdx[2, :]
    dzdx[end, :]     .= dzdx[end-1, :]

    dzdy[:, 2:end-1] .= (z[:, 3:end] .- z[:, 1:end-2]) ./ (2dy)
    dzdy[:, 1]       .= dzdy[:, 2]
    dzdy[:, end]     .= dzdy[:, end-1]

    # -------------------------------------------------------------------------
    # Initial condition
    # -------------------------------------------------------------------------

    h_in  = 0.20
    h_out = 0.10
    r0    = 2.5

    # to avoid exact dry states
    hmin = 1e-6

    η0 = [((x^2 + y^2) < r0^2) ? h_in : h_out for x in xs, y in ys]
    h0 = max.(hmin, η0 .- z)

    @. S = SVector(h0, 0.0, 0.0)

    # -------------------------------------------------------------------------
    # Visualisation
    # -------------------------------------------------------------------------

    midj = ny ÷ 2

    h_obs  = Observable(getindex.(S, 1))
    hu_obs = Observable(getindex.(S, 2))
    γ_obs  = Observable(h_obs[] .+ z)

    h_slice  = @lift($h_obs[:, midj])
    hu_slice = @lift($hu_obs[:, midj])
    γ_slice  = @lift($γ_obs[:, midj])

    fig = Figure(size = (900, 700))
    ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = "height")
    ax2 = Axis(fig[2, 1], xlabel = "x", ylabel = "y", aspect = DataAspect())

    lines!(ax1, xs, z[:, midj], label = "bottom z")
    lines!(ax1, xs, h_slice,    label = "depth h")
    lines!(ax1, xs, γ_slice,    label = "free surface")
    ylims!(ax1, 0, 0.3)
    axislegend(ax1)

    hm = heatmap!(
        ax2, xs, ys, γ_obs;
        colormap   = :curl,
        colorrange = (0.05, 0.15)
    )

    Colorbar(fig[2, 2], hm, label = "free surface height")

    # -------------------------------------------------------------------------
    # Time stepping
    # -------------------------------------------------------------------------

    record(fig, "docs/swe2d_topo_absorbing_debug.mp4"; fps = 20) do io
        for it in 1:nt
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

            # absorbing / radiation-type boundary conditions
            # normal wave speed estimate: c = sqrt(g*h)
            # normal velocity: u = hu/h, v = hv/h

            # left and right boundaries
            for j in 1:ny
                # left boundary
                hL  = S[1, j][1]
                huL = S[1, j][2]
                hvL = S[1, j][3]
                uL  = huL / hL
                cL  = (abs(uL) + sqrt(g * hL)) * dt / dx

                S[1, j] = SVector(
                    S[2, j][1] + ((cL - 1) / (cL + 1)) * (S[2, j][1] - S[1, j][1]),
                    S[2, j][2] + ((cL - 1) / (cL + 1)) * (S[2, j][2] - S[1, j][2]),
                    S[2, j][3] + ((cL - 1) / (cL + 1)) * (S[2, j][3] - S[1, j][3])
                )

                # right boundary
                hR  = S[end, j][1]
                huR = S[end, j][2]
                hvR = S[end, j][3]
                uR  = huR / hR
                cR  = (abs(uR) + sqrt(g * hR)) * dt / dx

                S[end, j] = SVector(
                    S[end-1, j][1] + ((cR - 1) / (cR + 1)) * (S[end-1, j][1] - S[end, j][1]),
                    S[end-1, j][2] + ((cR - 1) / (cR + 1)) * (S[end-1, j][2] - S[end, j][2]),
                    S[end-1, j][3] + ((cR - 1) / (cR + 1)) * (S[end-1, j][3] - S[end, j][3])
                )
            end

            # bottom and top boundaries
            for i in 1:nx
                # bottom boundary
                hB  = S[i, 1][1]
                huB = S[i, 1][2]
                hvB = S[i, 1][3]
                vB  = hvB / hB
                cB  = (abs(vB) + sqrt(g * hB)) * dt / dy

                S[i, 1] = SVector(
                    S[i, 2][1] + ((cB - 1) / (cB + 1)) * (S[i, 2][1] - S[i, 1][1]),
                    S[i, 2][2] + ((cB - 1) / (cB + 1)) * (S[i, 2][2] - S[i, 1][2]),
                    S[i, 2][3] + ((cB - 1) / (cB + 1)) * (S[i, 2][3] - S[i, 1][3])
                )

                # top boundary
                hT  = S[i, end][1]
                huT = S[i, end][2]
                hvT = S[i, end][3]
                vT  = hvT / hT
                cT  = (abs(vT) + sqrt(g * hT)) * dt / dy

                S[i, end] = SVector(
                    S[i, end-1][1] + ((cT - 1) / (cT + 1)) * (S[i, end-1][1] - S[i, end][1]),
                    S[i, end-1][2] + ((cT - 1) / (cT + 1)) * (S[i, end-1][2] - S[i, end][2]),
                    S[i, end-1][3] + ((cT - 1) / (cT + 1)) * (S[i, end-1][3] - S[i, end][3])
                )
            end

            # simple sponge layer near boundaries
            layers = 20
            σmax = 0.15

            for i in 1:nx, j in 1:ny
                di = min(i-1, nx-i)
                dj = min(j-1, ny-j)
                d  = min(di, dj)

                if d < layers
                    σ = σmax * (1 - d / layers)^2
                    h  = S[i,j][1]
                    hu = S[i,j][2] * (1 - σ)
                    hv = S[i,j][3] * (1 - σ)
                    S[i,j] = SVector(h, hu, hv)
                end
            end

            # positivity fix
            for i in 1:nx, j in 1:ny
                if S[i, j][1] <= 0
                    S[i, j] = SVector(hmin, S[i, j][2], S[i, j][3])
                end
            end

            # visualisation
            if it % nvis == 0
                h_obs[]  = getindex.(S, 1)
                hu_obs[] = getindex.(S, 2)
                γ_obs[]  = h_obs[] .+ z
                recordframe!(io)
            end

            # progression info
            percent = 100 * it / nt
            print("\rProgress: $(round(percent, digits=1)) %")
            flush(stdout)
        end
    end

    return nothing
end

swe2d_topography()