# 
using CairoMakie   # for visualisation
using StaticArrays # for small fixed-size state vectors

const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
import ParallelStencil: @reset_parallel_stencil
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=false)
else
    @init_parallel_stencil(Threads, Float64, 2, inbounds=false)
    @info "threads" Threads.nthreads()
end
using Printf

@inline avx_comp(hv1, hv2, h, ix, iy) = 0.5 * (hv1[ix, iy] * hv2[ix, iy] / h[ix, iy] + hv1[ix+1, iy] * hv2[ix+1, iy] / h[ix+1, iy])
@inline avy_comp(hv1, hv2, h, ix, iy) = 0.5 * (hv1[ix, iy] * hv2[ix, iy] / h[ix, iy] + hv1[ix, iy+1] * hv2[ix, iy+1] / h[ix, iy+1])
@inline avx_simp(h, ix, iy) = 0.5 * (h[ix, iy]*h[ix, iy] + h[ix+1, iy]*h[ix+1, iy])
@inline avy_simp(h, ix, iy) = 0.5 * (h[ix, iy]*h[ix, iy] + h[ix, iy+1]*h[ix, iy+1])
@inline dxa(h, ix, iy) = h[ix+1, iy] - h[ix, iy]
@inline dya(h, ix, iy) = h[ix, iy+1] - h[ix, iy]

@inline dxb(h, ix, iy) = h[ix, iy] - h[ix-1, iy]
@inline dyb(h, ix, iy) = h[ix, iy] - h[ix, iy-1]


const g = 1.0

#--------------------------------------------------------------------------------
# Multithread maximum speed for CFL condition
#--------------------------------------------------------------------------------

@views function dt_multithread(max_speed_x, max_speed_y, _dx, _dy, n)
    nthreads = Threads.nthreads()

    # get an array of maximum speeds for each thread
    max_speeds_x = zeros(nthreads)
    max_speeds_y = zeros(nthreads)

    Threads.@threads for i in 1:n
        thread_id = Threads.threadid()
        max_speeds_x[thread_id] = max(max_speeds_x[thread_id], maximum(max_speed_x[:, i]))
        max_speeds_y[thread_id] = max(max_speeds_y[thread_id], maximum(max_speed_y[i, :]))
    end
    return 0.99 / (maximum(max_speeds_x) * _dx + maximum(max_speeds_y) * _dy)

end

#--------------------------------------------------------------------------------
# Kernel definitions for parallel execution
#--------------------------------------------------------------------------------

# max speed in x and y directions for Rusanov flux
@parallel_indices (ix, iy) function compute_maxspeed!(max_speed_x, max_speed_y, h, hu, hv, g)
    nx, ny = size(h)
    if (ix <= nx - 1 && iy <= ny) max_speed_x[ix, iy] = max(abs(hu[ix, iy] / h[ix, iy]) + sqrt(g * h[ix, iy]), abs(hu[ix+1, iy] / h[ix+1, iy]) + sqrt(g * h[ix+1, iy])) end
    if (ix <= nx && iy <= ny - 1) max_speed_y[ix, iy] = max(abs(hv[ix, iy] / h[ix, iy]) + sqrt(g * h[ix, iy]), abs(hv[ix, iy+1] / h[ix, iy+1]) + sqrt(g * h[ix, iy+1])) end
    return nothing
end

# Rusanov fluxes computed component-wise 
@parallel function compute_first_flux!(F₁,G₁, hu, hv, h, max_speed_x, max_speed_y)
    @all(F₁) = @av_xa(hu) - 0.5 * @all(max_speed_x) * @d_xa(h)
    @all(G₁) = @av_ya(hv) - 0.5 * @all(max_speed_y) * @d_ya(h)
    return nothing
end

@parallel_indices (ix, iy) function compute_f2_g3_flux!(F₂,G₃, hu, hv, h, g, max_speed_x, max_speed_y)
    nx, ny = size(h)
    if (ix <= nx - 1 && iy <= ny) F₂[ix, iy] = avx_comp(hu, hu, h, ix, iy) + 0.5 *g * avx_simp(h, ix, iy) - 0.5 * max_speed_x[ix, iy] * dxa(hu, ix, iy) end
    if (ix <= nx && iy <= ny - 1) G₃[ix, iy] = avy_comp(hv, hv, h, ix, iy) + 0.5 *g * avy_simp(h, ix, iy) - 0.5 * max_speed_y[ix, iy] * dya(hv, ix, iy) end
    return nothing
end


@parallel_indices (ix, iy) function compute_f3_g2_flux!(F₃,G₂, hu, hv, h, max_speed_x, max_speed_y)
    nx, ny = size(h)
    if (ix <= nx - 1 && iy <= ny) F₃[ix, iy] = avx_comp(hu, hv, h, ix, iy) - 0.5 * max_speed_x[ix, iy] * dxa(hv, ix, iy) end
    if (ix <= nx && iy <= ny - 1) G₂[ix, iy] = avy_comp(hv, hu, h, ix, iy) - 0.5 * max_speed_y[ix, iy] * dya(hu, ix, iy) end
    return nothing
end

# update of height and momentum fields component-wise
@parallel_indices (ix, iy) function update_height!(h, F₁, G₁, dt, _dx, _dy)
    nx, ny = size(h)
    if (2 <= ix <= nx-1 && 2 <= iy <= ny-1) h[ix, iy] -= dt * ( dxb(F₁, ix, iy) * _dx + dyb(G₁, ix, iy) * _dy) end
    return nothing
end

@parallel_indices (ix, iy) function update_momentum_with_source!( hu, hv, h, F₂, F₃, G₂, G₃, dzdx, dzdy, g, dt, _dx, _dy)
    nx, ny = size(h)
    if (2 <= ix <= nx-1 && 2 <= iy <= ny-1)
        hu[ix, iy] -= dt * (dxb(F₂, ix, iy) * _dx + dyb(G₂, ix, iy) * _dy + g * h[ix, iy] * dzdx[ix, iy])
        hv[ix, iy] -= dt * (dxb(F₃, ix, iy) * _dx + dyb(G₃, ix, iy) * _dy + g * h[ix, iy] * dzdy[ix, iy])
    end
    return nothing
end

# absorbing boundary conditions component-wise
# left and right boundaries
@parallel_indices (iy) function sides_bc!(h, hu, hv, g, dt, _dx)
    ny = size(h, 2)
    if iy <= ny
        cL = (abs(hu[1, iy] / h[1, iy]) + sqrt(g * h[1, iy])) * dt * _dx
        αL = (cL - 1) / (cL + 1)

        h1  = h[2, iy]      + αL * (h[2, iy]      - h[1, iy])
        hu1 = hu[2, iy]     + αL * (hu[2, iy]     - hu[1, iy])
        hv1 = hv[2, iy]     + αL * (hv[2, iy]     - hv[1, iy])

        cR = (abs(hu[end, iy] / h[end, iy]) + sqrt(g * h[end, iy])) * dt * _dx
        αR = (cR - 1) / (cR + 1)

        hR  = h[end-1, iy]  + αR * (h[end-1, iy]  - h[end, iy])
        huR = hu[end-1, iy] + αR * (hu[end-1, iy] - hu[end, iy])
        hvR = hv[end-1, iy] + αR * (hv[end-1, iy] - hv[end, iy])

        h[1, iy]    = h1
        hu[1, iy]   = hu1
        hv[1, iy]   = hv1

        h[end, iy]  = hR
        hu[end, iy] = huR
        hv[end, iy] = hvR
    end
    return nothing
end

# top and bottom boundaries
@parallel_indices (ix) function top_bottom_bc!(h, hu, hv, g, dt, _dy)
    nx = size(h, 1)
    if ix <= nx
        cB = (abs(hv[ix, 1] / h[ix, 1]) + sqrt(g * h[ix, 1])) * dt * _dy
        αB = (cB - 1) / (cB + 1)

        hB  = h[ix, 2]      + αB * (h[ix, 2]      - h[ix, 1])
        huB = hu[ix, 2]     + αB * (hu[ix, 2]     - hu[ix, 1])
        hvB = hv[ix, 2]     + αB * (hv[ix, 2]     - hv[ix, 1])

        cT = (abs(hv[ix, end] / h[ix, end]) + sqrt(g * h[ix, end])) * dt * _dy
        αT = (cT - 1) / (cT + 1)

        hT  = h[ix, end-1]  + αT * (h[ix, end-1]  - h[ix, end])
        huT = hu[ix, end-1] + αT * (hu[ix, end-1] - hu[ix, end])
        hvT = hv[ix, end-1] + αT * (hv[ix, end-1] - hv[ix, end])

        h[ix, 1]    = hB
        hu[ix, 1]   = huB
        hv[ix, 1]   = hvB

        h[ix, end]  = hT
        hu[ix, end] = huT
        hv[ix, end] = hvT
    end
    return nothing
end

#sponge effect near boundaries
@parallel function sponge_layer!(hu, hv, σ)
    @all(hu) = @all(hu) * (1 - @all(σ))
    @all(hv) = @all(hv) * (1 - @all(σ))
    return nothing
end

# positivity fix for height field
@parallel function positivity_fix!(h, hmin)
    @all(h) = max(@all(h), hmin)
    return nothing
end



# -----------------------------------------------------------------------------
# 2D shallow water with topography
# -----------------------------------------------------------------------------

@views function swe2d_topography()
    # physics
    lx = 50.0
    ly = 50.0

    # numerics
    nx   = 1000
    ny   = 1000
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
    F₁ = zeros(nx - 1, ny)
    F₂ = zeros(nx - 1, ny)
    F₃ = zeros(nx - 1, ny)


    G₁ = zeros(nx, ny - 1)
    G₂ = zeros(nx, ny - 1)
    G₃ = zeros(nx, ny - 1)

    # max speed matrices for Rusanov flux
    max_speed_x = zeros(nx - 1, ny)
    max_speed_y = zeros(nx, ny - 1)


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
    #  MAIN LOOP - Time stepping (calling GPU kernels)
    # -------------------------------------------------------------------------

    record(fig, "docs/swe2d_topo_absorbing_xpu.mp4"; fps = 20) do io
        for it in 1:nt

            # compute max speeds for Rusanov fluxes
            @parallel compute_maxspeed!(max_speed_x, max_speed_y, h, hu, hv, g)

            # Rusanov fluxes 
            @parallel compute_first_flux!(F₁,G₁, hu, hv, h, max_speed_x, max_speed_y)
            @parallel compute_f2_g3_flux!(F₂,G₃, hu, hv, h, g, max_speed_x, max_speed_y)
            @parallel compute_f3_g2_flux!(F₃,G₂, hu, hv, h, max_speed_x, max_speed_y)

            
            # CFL condition for stability
            if !USE_GPU
                dt = dt_multithread(max_speed_x, max_speed_y, _dx, _dy, ny)
            else
                dt = 0.99 / (maximum(max_speed_x) * _dx + maximum(max_speed_y) * _dy)
            end

            # update states 
            @parallel update_height!(h, F₁, G₁, dt, _dx, _dy)
            @parallel update_momentum_with_source!(hu, hv, h, F₂, F₃, G₂, G₃, dzdx, dzdy, g, dt, _dx, _dy)

            # Boundary conditions
            @parallel sides_bc!(h, hu, hv, g, dt, _dx)
            @parallel top_bottom_bc!(h, hu, hv, g, dt, _dy)

            # simple sponge layer near boundaries
            @parallel sponge_layer!(hu, hv, σ)

            # positivity fix
            @parallel positivity_fix!(h, hmin)

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