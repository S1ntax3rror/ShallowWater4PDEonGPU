using CairoMakie, StaticArrays

const g = 1.0

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

# wave speed estimates
λx(S) = abs(S[2] / S[1]) + sqrt(g * S[1])
λy(S) = abs(S[3] / S[1]) + sqrt(g * S[1])

@views function swe2d_topography()
    lx, ly = 50.0, 50.0
    nx, ny = 500, 500
    nt = 2nx
    nvis = 5

    dx = lx / (nx - 1)
    dy = ly / (ny - 1)

    xs = LinRange(-lx/2, lx/2, nx)
    ys = LinRange(-ly/2, ly/2, ny)

    # state: (h, hu, hv)
    S = zeros(SVector{3,Float64}, nx, ny)

    # interface states
    Sᴸ = zeros(SVector{3,Float64}, nx-1, ny)
    Sᴿ = zeros(SVector{3,Float64}, nx-1, ny)
    Sᴮ = zeros(SVector{3,Float64}, nx, ny-1)
    Sᵀ = zeros(SVector{3,Float64}, nx, ny-1)

    # numerical fluxes
    F = zeros(SVector{3,Float64}, nx-1, ny)
    G = zeros(SVector{3,Float64}, nx, ny-1)

    # bottom topography
    z =
        0.04 .* exp.(-((xs .+ 15).^2 .+ (ys').^2)./0.6) .+
        0.07 .* exp.(-((xs .- 7).^2 .+ (ys').^2)) .+
        0.02 .* exp.(-(ys.^2 .+ (xs' .- 2).^2))

    # slopes
    dzdx = zeros(nx, ny)
    dzdy = zeros(nx, ny)

    dzdx[2:end-1, :] .= (z[3:end, :] .- z[1:end-2, :]) ./ (2dx)
    dzdx[1, :] .= dzdx[2, :]
    dzdx[end, :] .= dzdx[end-1, :]

    dzdy[:, 2:end-1] .= (z[:, 3:end] .- z[:, 1:end-2]) ./ (2dy)
    dzdy[:, 1] .= dzdy[:, 2]
    dzdy[:, end] .= dzdy[:, end-1]

    # initial condition
    h_in = 0.20
    h_out = 0.10
    r0 = 5.0

    y0 = [((x^2 + y^2) < r0^2) ? h_in : h_out for x in xs, y in ys]

    h0 = y0 .- z

    @assert minimum(h0) > 0
    @. S = SVector(h0, 0.0, 0.0)
    # ---------- visualization ----------
    midj = ny ÷ 2

    h_obs = Observable(getindex.(S, 1))
    hu_obs = Observable(getindex.(S, 2))
    γ_obs = Observable(h_obs[] .+ z)

    h_slice  = @lift($h_obs[:, midj])
    hu_slice = @lift($hu_obs[:, midj])
    γ_slice  = @lift($γ_obs[:, midj])

    fig = Figure(size = (900, 700))
    ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = "height")
    ax2 = Axis(fig[2,1], xlabel="x", ylabel="y", aspect=DataAspect())

    lines!(ax1, xs, z[:, midj], label = "bottom z")
    lines!(ax1, xs, h_slice, label = "depth h")
    lines!(ax1, xs, γ_slice, label = "free surface")
    ylims!(ax1, 0, 0.3)

    axislegend(ax1)

    hm = heatmap!(ax2, xs, ys, γ_obs,
              colormap=:viridis,
              colorrange=(0,0.25))

    Colorbar(fig[2,2], hm, label="free surface height")

    # ---------- time stepping ----------
    record(fig, "docs/swe2d_topo.mp4"; fps = 20) do io
        for it in 1:nt
            # piecewise constant reconstruction
            @. Sᴸ = S[1:end-1, :]
            @. Sᴿ = S[2:end, :]
            @. Sᴮ = S[:, 1:end-1]
            @. Sᵀ = S[:, 2:end]

            # Rusanov fluxes
            @. F = 0.5 * (fx(Sᴸ) + fx(Sᴿ)) - 0.5 * max(λx(Sᴸ), λx(Sᴿ)) * (Sᴿ - Sᴸ)
            @. G = 0.5 * (fy(Sᴮ) + fy(Sᵀ)) - 0.5 * max(λy(Sᴮ), λy(Sᵀ)) * (Sᵀ - Sᴮ)

            # CFL
            dt = 0.99 / maximum(λx.(S) ./ dx .+ λy.(S) ./ dy)

            # conservative update
            @. S[2:end-1, 2:end-1] -= dt * (
                (F[2:end, 2:end-1] - F[1:end-1, 2:end-1]) / dx +
                (G[2:end-1, 2:end] - G[2:end-1, 1:end-1]) / dy
            )

            # source terms
            for i in 2:nx-1, j in 2:ny-1
                h  = S[i,j][1]
                hu = S[i,j][2] - dt * g * h * dzdx[i,j]
                hv = S[i,j][3] - dt * g * h * dzdy[i,j]
                S[i,j] = SVector(h, hu, hv)
            end

            # reflective BCs
            for j in 1:ny
                S[1, j]   = SVector(S[2, j][1],   -S[2, j][2],    S[2, j][3])
                S[end, j] = SVector(S[end-1, j][1], -S[end-1, j][2], S[end-1, j][3])
            end

            for i in 1:nx
                S[i, 1]   = SVector(S[i, 2][1],    S[i, 2][2],   -S[i, 2][3])
                S[i, end] = SVector(S[i, end-1][1], S[i, end-1][2], -S[i, end-1][3])
            end

            # positivity fix
            for i in 1:nx, j in 1:ny
                if S[i,j][1] <= 0
                    S[i,j] = SVector(1e-8, S[i,j][2], S[i,j][3])
                end
            end

            if it % nvis == 0
                h_obs[]  = getindex.(S, 1)
                hu_obs[] = getindex.(S, 2)
                γ_obs[]  = h_obs[] .+ z
                recordframe!(io)
            end

            #progession info
            percent = 100 * it / nt
            print("\rProgress: $(round(percent,digits=1)) %")
            flush(stdout)
            
        end
    end

    return nothing
end

swe2d_topography()