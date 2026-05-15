using CairoMakie   # for visualisation
using StaticArrays # for small fixed-size state vectors

include("../solvers.jl")

const g = 1.0



@views function swe2d_topography()
    # physics
    lx = 50.0
    ly = 50.0

    # numerics
    nx   = 500
    ny   = 500
    nt   = nx
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
        colormap   = :viridis,
        colorrange = (0, 0.20)
    )

    Colorbar(fig[2, 2], hm, label = "free surface height")

    # -------------------------------------------------------------------------
    # Time stepping
    # -------------------------------------------------------------------------

    record(fig, "docs/swe2d_topo.mp4"; fps = 20) do io
        for it in 1:nt
            solve_z_2d(S, Sᴸ, Sᴿ, Sᴮ, Sᵀ, F, G, dx, dy, dzdx, dzdy, nx, ny)
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