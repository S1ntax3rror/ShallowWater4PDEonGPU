using CairoMakie, Printf, StaticArrays

include("../solvers.jl")

# gravity
const g = 1.0


@views function swe1d_topography()
    # physics
    lx = 10.0

    # numerics
    nx = 250
    nt = 2nx
    nvis = 5

    # preprocessing
    dx = lx / (nx - 1)
    xs = LinRange(-lx / 2, lx / 2, nx)

    # state vector: S = (h, hu)
    S = zeros(SVector{2, Float64}, nx)

    # left and right states
    Sᴸ = zeros(SVector{2, Float64}, nx - 1)
    Sᴿ = zeros(SVector{2, Float64}, nx - 1)

    # numerical flux
    F = zeros(SVector{2, Float64}, nx - 1)

    # ------------------------------------------------------------------
    # bottom topography z(x): small hill
    # ------------------------------------------------------------------
    z = 0.035 .* exp.(-(xs .+ 1.5).^2 / 0.5) .+
    0.025 .* exp.(-(xs .- 1.8).^2 / 0.8)

    # slope dz/dx
    dzdx = zeros(nx)
    dzdx[2:end-1] .= (z[3:end] .- z[1:end-2]) ./ (2 * dx)
    dzdx[1] = dzdx[2]
    dzdx[end] = dzdx[end-1]

    # ------------------------------------------------------------------
    # initial condition: lake at rest
    # gamma = h + z = const
    # so h = gamma - z, hu = 0
    # ------------------------------------------------------------------
    γ0 = 0.2
    h0 = γ0 .+ 0.05 .* exp.(-xs.^2)

    @assert minimum(h0) > 0 "Initial water depth became non-positive."

    @. S = SVector(h0, 0.0)

    # ------------------------------------------------------------------
    # visualization
    # top: h and bottom z
    # bottom: hu
    # ------------------------------------------------------------------

    h_obs  = Observable(getindex.(S,1))
    hu_obs = Observable(getindex.(S,2))
    γ_obs  = Observable(h_obs[] .+ z)

    fig = Figure(size = (900,700))

    ax1 = Axis(fig[1,1], xlabel="x", ylabel="height")
    ax2 = Axis(fig[2,1], xlabel="x", ylabel="hu")

    lines!(ax1, xs, z, label="bottom z")
    lines!(ax1, xs, h_obs, label="depth h")
    lines!(ax1, xs, γ_obs, label="free surface")

    axislegend(ax1)

    lines!(ax2, xs, hu_obs)

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    record(fig, "docs/1d_swe_topo.mp4"; fps=20) do io
        for it in 1:nt

            solve_z(S, Sᴸ, Sᴿ, F, dx, dzdx, nx)
            # update plots
            if it % nvis == 0
                h_obs[]  = getindex.(S,1)
                hu_obs[] = getindex.(S,2)
                γ_obs[]  = h_obs[] .+ z

                #display(fig)
                recordframe!(io)
            end
        end
    end

    return nothing
end

swe1d_topography()