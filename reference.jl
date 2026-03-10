using CairoMakie, Printf, StaticArrays

# gravity
const g = 1.0

# flux function
# f_h  = hu
# f_hu = u + g*h^2/2
f(S) = SA[S[2], S[2]^2/S[1]+0.5*g*S[1]^2]

# characteristic speed magnitude
# λ = u ± √gh
λ(S) = abs(S[2] / S[1]) + sqrt(g * S[1])

@views function swe1d()
    # physics
    lx = 10.0
    # numerics
    nx = 250
    nt = 2nx
    nvis = 5
    # preprocessing
    dx = lx / (nx - 1)
    xs = LinRange(-lx / 2, lx / 2, nx)
    # state vector: h, hu
    S = zeros(SVector{2}, nx)
    # left and right states
    Sᴸ = zeros(SVector{2}, nx - 1)
    Sᴿ = zeros(SVector{2}, nx - 1)
    # numerical flux
    F = zeros(SVector{2}, nx - 1)
    # initial conditions
    @. S = SVector(0.1exp(-xs^2)+0.1, 0)
    # visualisation
    fig = Figure()
    ax = (Axis(fig[1, 1]; xlabel="x", ylabel="h"),
          Axis(fig[2, 1]; xlabel="x", ylabel="hu"))
    lines!(ax[1], xs, getindex.(S, 1))
    lines!(ax[2], xs, getindex.(S, 2))
    plt = (lines!(ax[1], xs, getindex.(S, 1)),
           lines!(ax[2], xs, getindex.(S, 2)))
    display(fig)
    # time-stepping loop
    for it in 1:nt
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
        # update plots
        if it % nvis == 0
            plt[1][2] = getindex.(S, 1)
            plt[2][2] = getindex.(S, 2)
            display(fig)
        end
    end
    return
end

swe1d()