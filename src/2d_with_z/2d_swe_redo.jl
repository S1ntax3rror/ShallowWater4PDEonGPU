using CairoMakie, Printf, StaticArrays


# gravity
const gravity = 1.0

# flux functions for S = (h, hu, hv)
f(S) = SA[S[2], S[2]^2 / S[1] + 0.5 * gravity * S[1]^2, (S[2]*S[3])/S[1]]
g(S) = SA[S[3], (S[2]*S[3])/S[1], S[3]^2 / S[1] + 0.5 * gravity * S[1]^2]

# characteristic speed magnitude
λx(S) = abs(S[2] / S[1]) + sqrt(gravity * S[1])
λy(S) = abs(S[3] / S[1]) + sqrt(gravity * S[1])

@views function swe1d_topography()
    # physics
    lx = 10.0
    ly = 10.0

    # numerics
    nx = 250
    ny = 250
    nt = 2nx
    nvis = 5

    # preprocessing
    dx = lx / (nx - 1)
    xs = LinRange(-lx / 2, lx / 2, nx)
    dy = ly / (ny - 1)
    ys = LinRange(-ly / 2, ly / 2, ny)

    # state vector: S = (h, hu, hv)
    S = zeros(SVector{3, Float64}, nx, ny)

    # left and right states
    Sᴸ = zeros(SVector{3, Float64}, nx - 1, ny)
    Sᴿ = zeros(SVector{3, Float64}, nx - 1, ny)
    ST = zeros(SVector{3, Float64}, nx, ny - 1)
    SB = zeros(SVector{3, Float64}, nx, ny - 1)


    # numerical flux
    F = zeros(SVector{3, Float64}, nx - 1, ny)
    G = zeros(SVector{3, Float64}, nx, ny - 1)
    
    # ------------------------------------------------------------------
    # bottom topography z(x, y): small hill
    # ------------------------------------------------------------------
    z = 0.035 .* exp.(-((xs .+ 1.5).^2 .+ ys'.^2) / 0.5) .+
        0.025 .* exp.(-((xs .- 1.8).^2 .+ ys'.^2)/ 0.8)

    # slope dz/dx
    dzdx = zeros(nx, ny)
    dzdy = zeros(nx, ny)
    dzdx[2:end-1, :] .= (z[3:end, :] .- z[1:end-2, :]) ./ (2 * dx)
    dzdx[1, :] = dzdx[2, :]
    dzdx[end, :] = dzdx[end-1, :]

    dzdy[:, 2:end-1] .= (z[:, 3:end] .- z[:, 1:end-2]) ./ (2 * dy)
    dzdy[:, 1] = dzdy[:, 2]
    dzdy[:, end] = dzdy[:, end-1]

    # ------------------------------------------------------------------
    # initial condition: lake at rest
    # gamma = h + z = const
    # so h = gamma - z, hu = 0
    # ------------------------------------------------------------------
    γ0 = 0.2
    h0 = γ0 .+ 0.05 .* exp.(-(xs.^2 .+ ys'.^2))
    @assert minimum(h0) > 0 "Initial water depth became non-positive."

    @. S = SVector(h0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # visualization
    # top: h and bottom z
    # bottom: hu
    # ------------------------------------------------------------------

    h_obs  = Observable(getindex.(S,1))
    hu_obs = Observable(getindex.(S,2))
    γ_obs  = Observable(h_obs[] .+ z)

    fig = Figure(size = (900,700))

    ax1 = Axis(fig[1, 1], xlabel="x",  ylabel = "y", aspect = DataAspect())
    
    hm = heatmap!(
        ax1, xs, ys, γ_obs;
        colormap   = :viridis,
        colorrange = (0.15, 0.26)
    )
    # lines!(ax2, xs, hu_obs)
    Colorbar(fig[1, 2], hm, label = "free surface height")
    
    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    record(fig, "docs/2d_swe_topo.mp4"; fps=20) do io
        for it in 1:nt
            # reconstruction step (piecewise constant)
            @. Sᴸ = S[1:end-1, :]
            @. Sᴿ = S[2:end, :]
            @. SB = S[:, 1:end-1]
            @. ST = S[:, 2:end]
            
            # Rusanov flux
            @. F = 0.5 * (f(Sᴸ) + f(Sᴿ)) - 0.5 * max(λx(Sᴸ), λx(Sᴿ)) * (Sᴿ - Sᴸ)
            @. G = 0.5 * (g(SB) + g(ST)) - 0.5 * max(λy(SB), λy(ST)) * (ST - SB)

            # CFL time step
            dt = 0.99 / maximum(λx.(S) ./ dx .+ λy.(S) ./ dy)

            # conservative flux update
            @. S[2:end-1, 2:end-1] -= dt * (
                (F[2:end, 2:end-1] - F[1:end-1, 2:end-1]) / dx +
                (G[2:end-1, 2:end] - G[2:end-1, 1:end-1]) / dy
            )

            # source term in momentum equation: -(g h z_x)
            for i in 2:nx-1
                for j in 2:ny-1
                    h  = S[i, j][1]
                    hu = S[i,j][2] - dt * gravity * h * dzdx[i,j]
                    hv = S[i,j][3] - dt * gravity * h * dzdy[i,j]
                    S[i,j] = SVector(h, hu, hv)
                end
            end

            # reflective boundary conditions
            for j in 1:ny
                S[1, j]   = SVector(S[2, j][1],     -S[2, j][2],     S[2, j][3])
                S[end, j] = SVector(S[end-1, j][1], -S[end-1, j][2], S[end-1, j][3])
            end

            for i in 1:nx
                S[i, 1]   = SVector(S[i, 2][1],     -S[i, 2][2],     S[i, 2][3])
                S[i, end] = SVector(S[i, end-1][1], -S[i, end-1][2], S[i, end-1][3])
            end

            # safety: keep h positive
            for i in 1:nx
                for j in 1:ny
                    if S[i, j][1] <= 0
                        S[i, j] = SVector(1e-8, S[i, j][2], S[i, j][3])
                    end
                end
            end

            # update plots
            if it % nvis == 0
                h_obs[]  = getindex.(S,1)
                hu_obs[] = getindex.(S,2)
                γ_obs[]  = h_obs[] .+ z

                #display(fig)
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

swe1d_topography()