using CairoMakie, Printf, StaticArrays, LinearAlgebra, Roots



# gravity
const g = 1.0

function dam_break_exact(xs, t, hL, hR)
    @assert hL > hR > 0
    @assert t > 0

    # equation for hstar
    f(h) = 2*(sqrt(g*h) - sqrt(g*hL)) +
           (h - hR) * sqrt(0.5*g*(1/h + 1/hR))

    # solve for hstar in (hR, hL)
    hstar = find_zero(f, (hR, hL))
    ustar = 2*(sqrt(g*hL) - sqrt(g*hstar))

    cL    = sqrt(g*hL)
    cstar = sqrt(g*hstar)

    xiA = -cL
    xiB = ustar - cstar
    s   = hstar*ustar / (hstar - hR)

    h  = similar(xs, Float64)
    hu = similar(xs, Float64)

    for i in eachindex(xs)
        ξ = xs[i] / t

        if ξ < xiA
            h[i]  = hL
            hu[i] = 0.0

        elseif ξ <= xiB
            uξ = (2/3) * (ξ + cL)
            hξ = (1/(9g)) * (2cL - ξ)^2
            h[i]  = hξ
            hu[i] = hξ * uξ

        elseif ξ < s
            h[i]  = hstar
            hu[i] = hstar * ustar

        else
            h[i]  = hR
            hu[i] = 0.0
        end
    end

    return h, hu
end



# flux function for S = (h, hu)
f(S) = SA[S[2], S[2]^2 / S[1] + 0.5 * g * S[1]^2]

# characteristic speed magnitude
λ(S) = abs(S[2] / S[1]) + sqrt(g * S[1])

@views function swe1d_topography(nx; do_visualize=true)
    # physics
    lx = 10.0

    # numerics
    nt = round(nx/2 - 0.01*nx)
    nvis = 5
    time = 0.0

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
    z = zeros(nx)

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
    hL = γ0 +0.1
    hR = γ0 

    h0 = ifelse.(xs .< 0.0, hL, hR)

    @assert minimum(h0) > 0 "Initial water depth became non-positive."

    @. S = SVector(h0, 0.0)

    # ------------------------------------------------------------------
    # visualization
    # top: h and bottom z
    # bottom: hu
    # ------------------------------------------------------------------
    if do_visualize

        h_obs  = Observable(getindex.(S,1))
        hu_obs = Observable(getindex.(S,2))
        #γ_obs  = Observable(h_obs[] .+ z)

        h_obs_ex  = Observable(getindex.(S,1))
        hu_obs_ex = Observable(getindex.(S,2))

        fig = Figure(size = (900,700))

        ax1 = Axis(fig[1,1], xlabel="x", ylabel="height")
        ax2 = Axis(fig[2,1], xlabel="x", ylabel="hu")

        #lines!(ax1, xs, z, label="bottom z")
        lines!(ax1, xs, h_obs, label="depth h")
        lines!(ax1, xs, h_obs_ex, label="exact depth h")
        #lines!(ax1, xs, γ_obs, label="free surface")

        axislegend(ax1)

        lines!(ax2, xs, hu_obs)
        lines!(ax2, xs, hu_obs_ex)

        record(fig, "docs/dam_break.mp4"; fps=20) do io

            h_obs[]  = getindex.(S,1)
            hu_obs[] = getindex.(S,2)

            h_obs_ex[] = getindex.(S,1)
            hu_obs_ex[] = getindex.(S,2)


            # γ_obs[]  = h_obs[] .+ z

            #display(fig)
            recordframe!(io)
            for it in 1:nt
                # reconstruction step (piecewise constant)
                @. Sᴸ = S[1:end-1]
                @. Sᴿ = S[2:end]

                # Rusanov flux
                @. F = 0.5 * (f(Sᴸ) + f(Sᴿ)) - 0.5 * max(λ(Sᴸ), λ(Sᴿ)) * (Sᴿ - Sᴸ)

                # CFL time step
                dt = 0.99 * dx / maximum(λ.(S))

                # time update
                time += dt

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
                    
                    h_ex, hu_ex = dam_break_exact(xs, time, hL, hR)

                    h_obs_ex[] = h_ex
                    hu_obs_ex[] = hu_ex

                    #display(fig)
                    recordframe!(io)
                end
            end
        end

    else

        for it in 1:nt
            # reconstruction step (piecewise constant)
            @. Sᴸ = S[1:end-1]
            @. Sᴿ = S[2:end]

            # Rusanov flux
            @. F = 0.5 * (f(Sᴸ) + f(Sᴿ)) - 0.5 * max(λ(Sᴸ), λ(Sᴿ)) * (Sᴿ - Sᴸ)

            # CFL time step
            dt = 0.99 * dx / maximum(λ.(S))

            # time update
            time += dt

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
        end
    end

    # ------------------------------------------------------------------
    # error arrays 
    # ------------------------------------------------------------------
    h_error = similar(xs)
    hu_error = similar(xs)

    

    # ------------------------------------------------------------------
    # compute final error
    # ------------------------------------------------------------------
    h_ex, hu_ex = dam_break_exact(xs, time, hL, hR)
    for i in eachindex(xs)
        h_error[i] = abs(S[i][1] - h_ex[i])
        hu_error[i] = abs(S[i][2] - hu_ex[i])
    end
    total_h_error = sum(h_error)*dx 
    total_hu_error = sum(hu_error)*dx
    @printf("Total h error at final time: %.6f\n", total_h_error)
    @printf("Total hu error at final time: %.6f\n", total_hu_error)

    return total_h_error, total_hu_error
end

@views function benchmark_dam_break(;do_viz =false)
    # run the simulation for different grid resolutions and record the errors
    nx_values = [50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]

    errors_h = zeros(length(nx_values))
    errors_hu = zeros(length(nx_values))

    for (i, nx) in enumerate(nx_values)
        @printf("Running simulation with nx = %d...\n", nx)
        err_h, err_hu = swe1d_topography(nx, do_visualize=do_viz)
        errors_h[i] = err_h
        errors_hu[i] = err_hu
    end

    # plot the errors
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="nx", ylabel="Error", xscale=log, yscale=log)
    scatter!(ax, nx_values, errors_h, label="h Error")
    scatter!(ax, nx_values, errors_hu, label="hu Error")

    # fit a line to the error data to estimate convergence rate
    log_nx = log.(nx_values)

    log_errors_h = log.(errors_h)
    log_errors_hu = log.(errors_hu)
    A = [ones(length(log_nx)) log_nx]   # design matrix
    coeffs_h = A \ log_errors_h
    coeffs_hu = A \ log_errors_hu
    slope_h = coeffs_h[2]
    slope_hu = coeffs_hu[2]
    rate_h = -slope_h
    rate_hu = -slope_hu
    @printf("Estimated convergence rate for h: %.2f\n", rate_h)
    @printf("Estimated convergence rate for hu: %.2f\n", rate_hu)

    # add the fitted line to the plot
    fitted_errors_h = exp.(A * coeffs_h)
    fitted_errors_hu = exp.(A * coeffs_hu)
    lines!(ax, nx_values, fitted_errors_h, label=@sprintf("convergence rate: p = %.2f", rate_h))
    lines!(ax, nx_values, fitted_errors_hu, label=@sprintf("convergence rate: p = %.2f", rate_hu))
    axislegend(ax)
    save("docs/dam_break_error.png", fig)   
    
    show(fig)

    return nothing
end


benchmark_dam_break()