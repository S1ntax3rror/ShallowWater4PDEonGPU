using CairoMakie, Printf, StaticArrays

include("../solvers.jl")

# gravity
const g = 1.0

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
    # time-stepping loop
    record(fig, "swe1d.gif", 1:nt; framerate=30) do it

        solve(S, Sᴸ, Sᴿ, F, dx)
        # update plots
        if it % nvis == 0
            plt[1][2] = getindex.(S, 1)
            plt[2][2] = getindex.(S, 2)
        end
    end
    return
end

swe1d()