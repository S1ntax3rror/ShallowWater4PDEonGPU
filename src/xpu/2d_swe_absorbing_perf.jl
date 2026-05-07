# 
using CairoMakie   # for visualisation
using StaticArrays # for small fixed-size state vectors


const g = 1.0

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

    # inverse of grid spacing precomputed for efficiency
    _dx = 1.0 / dx
    _dy = 1.0 / dy

    _2dx = 1.0 / (2dx)
    _2dy = 1.0 / (2dy)

    xs = LinRange(-lx / 2, lx / 2, nx)
    ys = LinRange(-ly / 2, ly / 2, ny)

    # -------------------------------------------------------------------------
    # Array initialisation
    # -------------------------------------------------------------------------

    # conservative state: S = (h, hu, hv)
    h = zeros(nx, ny)
    hu = zeros(nx, ny)
    hv = zeros(nx, ny)

    # numerical fluxes
    F = zeros(SVector{3, Float64}, nx - 1, ny)
    F₁ = zeros(nx - 1, ny)
    F₂ = zeros(nx - 1, ny)
    F₃ = zeros(nx - 1, ny)


    G = zeros(SVector{3, Float64}, nx, ny - 1)
    G₁ = zeros(nx, ny - 1)
    G₂ = zeros(nx, ny - 1)
    G₃ = zeros(nx, ny - 1)


    # -------------------------------------------------------------------------
    # Bottom topography Initialisation
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

    dzdx[2:end-1, :] .= (z[3:end, :] .- z[1:end-2, :]) .* _2dx
    dzdx[1, :]       .= dzdx[2, :]
    dzdx[end, :]     .= dzdx[end-1, :]

    dzdy[:, 2:end-1] .= (z[:, 3:end] .- z[:, 1:end-2]) .* _2dy
    dzdy[:, 1]       .= dzdy[:, 2]
    dzdy[:, end]     .= dzdy[:, end-1]

    # -------------------------------------------------------------------------
    # Absorbing boundary layer Initialisation
    # -------------------------------------------------------------------------

    uL = zeros(ny)
    cL = zeros(ny)
    uR = zeros(ny)
    cR = zeros(ny)

    uB = zeros(nx)
    cB = zeros(nx)
    uT = zeros(nx)
    cT = zeros(nx)

    d = zeros(nx, ny)
    σ = zeros(nx, ny)
    for i in 1:nx, j in 1:ny
        di = min(i-1, nx-i)
        dj = min(j-1, ny-j)
        d[i, j]  = min(di, dj)

    end

    σ = zeros(nx, ny)
    layers = 20
    _layers = 1.0 / layers
    σmax = 0.15

    for i in 1:nx, j in 1:ny
        if d[i, j] < layers
            σ[i, j] = σmax * (1 - d[i, j] * _layers)
        else
            σ[i, j] = 0.0
        end
    end

    # -------------------------------------------------------------------------
    # Initial condition for height field
    # -------------------------------------------------------------------------

    h_in  = 0.20
    h_out = 0.10
    r0    = 2.5

    # to avoid exact dry states
    hmin = 1e-6

    η0 = [((x^2 + y^2) < r0^2) ? h_in : h_out for x in xs, y in ys]
    h0 = max.(hmin, η0 .- z)

    h = h0
    # -------------------------------------------------------------------------
    # Visualisation
    # -------------------------------------------------------------------------

    midj = ny ÷ 2

    h_obs  = Observable(h)
    hu_obs = Observable(hu)
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

    record(fig, "docs/swe2d_topo_absorbing_perf.mp4"; fps = 20) do io
        for it in 1:nt


            # global maximum wave speed in x and y directions
            max_speed_x  = max.(abs.(hu[1:end-1, :] ./ h[1:end-1, :]) + sqrt.(g .* h[1:end-1, :]), abs.(hu[2:end, :] ./ h[2:end, :]) + sqrt.(g .* h[2:end, :]))
            max_speed_y  = max.(abs.(hv[:, 1:end-1] ./ h[:, 1:end-1]) + sqrt.(g .* h[:, 1:end-1]), abs.(hv[:, 2:end] ./ h[:, 2:end]) + sqrt.(g .* h[:, 2:end]))

            # Rusanov fluxes computed component-wise for futher parallelisation using the primative arrays
            @. F₁ = 0.5 * (hu[1:end-1, :] + hu[2:end, :]) - 0.5 * max_speed_x * (h[2:end, :] - h[1:end-1, :])
            @. G₁ = 0.5 * (hv[:, 1:end-1] + hv[:, 2:end]) - 0.5 * max_speed_y * (h[:, 2:end] - h[:, 1:end-1])

            @. F₂ = 0.5 * (hu[1:end-1, :] * hu[1:end-1, :] / h[1:end-1, :] + 0.5 * g * h[1:end-1, :]^2 + hu[2:end, :] * hu[2:end, :] / h[2:end, :] + 0.5 * g * h[2:end, :]^2)
                    - 0.5 * max_speed_x * (hu[2:end, :] - hu[1:end-1, :])
            @. G₃ = 0.5 * (hv[:, 1:end-1] * hv[:, 1:end-1] / h[:, 1:end-1] + 0.5 * g * h[:, 1:end-1]^2 + hv[:, 2:end] * hv[:, 2:end] / h[:, 2:end] + 0.5 * g * h[:, 2:end]^2)
                    - 0.5 * max_speed_y * (hv[:, 2:end] - hv[:, 1:end-1])


            @. F₃ = 0.5 * (hu[1:end-1, :] * hv[1:end-1, :] / h[1:end-1, :] + hu[2:end, :] * hv[2:end, :] / h[2:end, :]) - 0.5 * max_speed_x * (hv[2:end, :] - hv[1:end-1, :])
            @. G₂ = 0.5 * (hu[:, 1:end-1] * hv[:, 1:end-1] / h[:, 1:end-1] + hu[:, 2:end] * hv[:, 2:end] / h[:, 2:end]) - 0.5 * max_speed_y * (hu[:, 2:end] - hu[:, 1:end-1])
            


            # CFL time step computed from the primative arrays for futher parallelisation
            dt = 0.99 / (maximum(max_speed_x) * _dx + maximum(max_speed_y) * _dy)

            # conservative update component-wise for futher parallelisation using the primative arrays
            @. h[2:end-1, 2:end-1] -= dt * (
                (F₁[2:end,   2:end-1] - F₁[1:end-1, 2:end-1]) * _dx +
                (G₁[2:end-1, 2:end]   - G₁[2:end-1, 1:end-1]) * _dy
            )

            @. hu[2:end-1, 2:end-1] -= dt * (
                (F₂[2:end,   2:end-1] - F₂[1:end-1, 2:end-1]) * _dx +
                (G₂[2:end-1, 2:end]   - G₂[2:end-1, 1:end-1]) * _dy
            )

            @. hv[2:end-1, 2:end-1] -= dt * (
                (F₃[2:end,   2:end-1] - F₃[1:end-1, 2:end-1]) * _dx +
                (G₃[2:end-1, 2:end]   - G₃[2:end-1, 1:end-1]) * _dy
            )


            # source term update component-wise for futher parallelisation using the primative arrays
            @. hu[2:end-1, 2:end-1] -= dt * g * h[2:end-1, 2:end-1] * dzdx[2:end-1, 2:end-1]
            @. hv[2:end-1, 2:end-1] -= dt * g * h[2:end-1, 2:end-1] * dzdy[2:end-1, 2:end-1]


            # left and right boundaries computed component-wise for futher parallelisation using the primative arrays

            uL = hu[1, :] ./ h[1, :]
            cL = (abs.(uL) + sqrt.(g .* h[1, :])) .* dt * _dx

            uR = hu[end, :] ./ h[end, :]
            cR = (abs.(uR) + sqrt.(g .* h[end, :])) .* dt * _dx

            h[1, :]   = h[2, :] + ((cL .- 1) ./ (cL .+ 1)) .* (h[2, :] .- h[1, :])
            hu[1, :]  = hu[2, :] + ((cL .- 1) ./ (cL .+ 1)) .* (hu[2, :] .- hu[1, :])
            hv[1, :]  = hv[2, :] + ((cL .- 1) ./ (cL .+ 1)) .* (hv[2, :] .- hv[1, :])

            h[end, :]  = h[end-1, :] + ((cR .- 1) ./ (cR .+ 1)) .* (h[end-1, :] .- h[end, :])
            hu[end, :] = hu[end-1, :] + ((cR .- 1) ./ (cR .+ 1)) .* (hu[end-1, :] .- hu[end, :])
            hv[end, :] = hv[end-1, :] + ((cR .- 1) ./ (cR .+ 1)) .* (hv[end-1, :] .- hv[end, :])


            # bottom and top boundaries computed component-wise for futher parallelisation using the primative arrays
            uB = hv[:, 1] ./ h[:, 1]
            cB = (abs.(uB) + sqrt.(g .* h[:, 1])) .* dt * _dy

            uT = hv[:, end] ./ h[:, end]
            cT = (abs.(uT) + sqrt.(g .* h[:, end])) .* dt * _dy

            h[:, 1]   = h[:, 2] + ((cB .- 1) ./ (cB .+ 1)) .* (h[:, 2] .- h[:, 1])
            hu[:, 1]  = hu[:, 2] + ((cB .- 1) ./ (cB .+ 1)) .* (hu[:, 2] .- hu[:, 1])
            hv[:, 1]  = hv[:, 2] + ((cB .- 1) ./ (cB .+ 1)) .* (hv[:, 2] .- hv[:, 1])

            h[:, end]  = h[:, end-1] + ((cT .- 1) ./ (cT .+ 1)) .* (h[:, end-1] .- h[:, end])
            hu[:, end] = hu[:, end-1] + ((cT .- 1) ./ (cT .+ 1)) .* (hu[:, end-1] .- hu[:, end])
            hv[:, end] = hv[:, end-1] + ((cT .- 1) ./ (cT .+ 1)) .* (hv[:, end-1] .- hv[:, end])

            # simple sponge layer near boundaries
            hu .= hu .* (1 .- σ)
            hv .= hv .* (1 .- σ)

            # positivity fix
            @. h = max(h, hmin)

            # visualisation
            if it % nvis == 0
                h_obs[]  = h
                hu_obs[] = hu
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