using CairoMakie, Printf, StaticArrays


# gravity
const g = 1.0

# flux function for S = (h, hu)
f(S) = SA[S[2], S[2]^2 / S[1] + 0.5 * g * S[1]^2]

# characteristic speed magnitude
λ(S) = abs(S[2] / S[1]) + sqrt(g * S[1])

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

    display(fig)
    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    for it in 1:nt
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

        # update plots
        if it % nvis == 0
            h_obs[]  = getindex.(S,1)
            hu_obs[] = getindex.(S,2)
            γ_obs[]  = h_obs[] .+ z

            display(fig)
        end
    end

    return nothing
end

swe1d_topography()