using GLMakie
using StaticArrays
using Random

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
@inline avx_simp(h, ix, iy) = 0.5 * (h[ix, iy] * h[ix, iy] + h[ix+1, iy] * h[ix+1, iy])
@inline avy_simp(h, ix, iy) = 0.5 * (h[ix, iy] * h[ix, iy] + h[ix, iy+1] * h[ix, iy+1])
@inline dxa(h, ix, iy) = h[ix+1, iy] - h[ix, iy]
@inline dya(h, ix, iy) = h[ix, iy+1] - h[ix, iy]

@inline dxb(h, ix, iy) = h[ix, iy] - h[ix-1, iy]
@inline dyb(h, ix, iy) = h[ix, iy] - h[ix, iy-1]

const g = 1.0

@views function dt_multithread(max_speed_x, max_speed_y, _dx, _dy, n)
    nthreads = Threads.nthreads()
    max_speeds_x = zeros(nthreads)
    max_speeds_y = zeros(nthreads)

    Threads.@threads for i in 1:n
        tid = Threads.threadid()
        max_speeds_x[tid] = max(max_speeds_x[tid], maximum(max_speed_x[:, i]))
        max_speeds_y[tid] = max(max_speeds_y[tid], maximum(max_speed_y[i, :]))
    end

    return 0.99 / (maximum(max_speeds_x) * _dx + maximum(max_speeds_y) * _dy)
end

# -----------------------------------------------------------------------------
# Kernels
# -----------------------------------------------------------------------------

@parallel_indices (ix, iy) function compute_maxspeed!(max_speed_x, max_speed_y, h, hu, hv, g)
    nx, ny = size(h)
    if (ix <= nx - 1 && iy <= ny)
        max_speed_x[ix, iy] = max(
            abs(hu[ix, iy] / h[ix, iy]) + sqrt(g * h[ix, iy]),
            abs(hu[ix+1, iy] / h[ix+1, iy]) + sqrt(g * h[ix+1, iy])
        )
    end
    if (ix <= nx && iy <= ny - 1)
        max_speed_y[ix, iy] = max(
            abs(hv[ix, iy] / h[ix, iy]) + sqrt(g * h[ix, iy]),
            abs(hv[ix, iy+1] / h[ix, iy+1]) + sqrt(g * h[ix, iy+1])
        )
    end
    return nothing
end

@parallel function compute_first_flux!(F₁, G₁, hu, hv, h, max_speed_x, max_speed_y)
    @all(F₁) = @av_xa(hu) - 0.5 * @all(max_speed_x) * @d_xa(h)
    @all(G₁) = @av_ya(hv) - 0.5 * @all(max_speed_y) * @d_ya(h)
    return nothing
end

@parallel_indices (ix, iy) function compute_f2_g3_flux!(F₂, G₃, hu, hv, h, g, max_speed_x, max_speed_y)
    nx, ny = size(h)
    if (ix <= nx - 1 && iy <= ny)
        F₂[ix, iy] = avx_comp(hu, hu, h, ix, iy) + 0.5 * g * avx_simp(h, ix, iy) - 0.5 * max_speed_x[ix, iy] * dxa(hu, ix, iy)
    end
    if (ix <= nx && iy <= ny - 1)
        G₃[ix, iy] = avy_comp(hv, hv, h, ix, iy) + 0.5 * g * avy_simp(h, ix, iy) - 0.5 * max_speed_y[ix, iy] * dya(hv, ix, iy)
    end
    return nothing
end

@parallel_indices (ix, iy) function compute_f3_g2_flux!(F₃, G₂, hu, hv, h, max_speed_x, max_speed_y)
    nx, ny = size(h)
    if (ix <= nx - 1 && iy <= ny)
        F₃[ix, iy] = avx_comp(hu, hv, h, ix, iy) - 0.5 * max_speed_x[ix, iy] * dxa(hv, ix, iy)
    end
    if (ix <= nx && iy <= ny - 1)
        G₂[ix, iy] = avy_comp(hv, hu, h, ix, iy) - 0.5 * max_speed_y[ix, iy] * dya(hu, ix, iy)
    end
    return nothing
end

@parallel_indices (ix, iy) function update_height!(h, F₁, G₁, dt, _dx, _dy)
    nx, ny = size(h)
    if (2 <= ix <= nx-1 && 2 <= iy <= ny-1)
        h[ix, iy] -= dt * (dxb(F₁, ix, iy) * _dx + dyb(G₁, ix, iy) * _dy)
    end
    return nothing
end

@parallel_indices (ix, iy) function update_momentum_with_source!(hu, hv, h, F₂, F₃, G₂, G₃, dzdx, dzdy, g, dt, _dx, _dy)
    nx, ny = size(h)
    if (2 <= ix <= nx-1 && 2 <= iy <= ny-1)
        hu[ix, iy] -= dt * (dxb(F₂, ix, iy) * _dx + dyb(G₂, ix, iy) * _dy + g * h[ix, iy] * dzdx[ix, iy])
        hv[ix, iy] -= dt * (dxb(F₃, ix, iy) * _dx + dyb(G₃, ix, iy) * _dy + g * h[ix, iy] * dzdy[ix, iy])
    end
    return nothing
end

@parallel_indices (iy) function sides_bc!(h, hu, hv, g, dt, _dx)
    ny = size(h, 2)
    if iy <= ny
        cL = (abs(hu[1, iy] / h[1, iy]) + sqrt(g * h[1, iy])) * dt * _dx
        αL = (cL - 1) / (cL + 1)

        h1  = h[2, iy]  + αL * (h[2, iy]  - h[1, iy])
        hu1 = hu[2, iy] + αL * (hu[2, iy] - hu[1, iy])
        hv1 = hv[2, iy] + αL * (hv[2, iy] - hv[1, iy])

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

@parallel_indices (ix) function top_bottom_bc!(h, hu, hv, g, dt, _dy)
    nx = size(h, 1)
    if ix <= nx
        cB = (abs(hv[ix, 1] / h[ix, 1]) + sqrt(g * h[ix, 1])) * dt * _dy
        αB = (cB - 1) / (cB + 1)

        hB  = h[ix, 2]  + αB * (h[ix, 2]  - h[ix, 1])
        huB = hu[ix, 2] + αB * (hu[ix, 2] - hu[ix, 1])
        hvB = hv[ix, 2] + αB * (hv[ix, 2] - hv[ix, 1])

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

@parallel function sponge_layer!(hu, hv, σ)
    @all(hu) = @all(hu) * (1 - @all(σ))
    @all(hv) = @all(hv) * (1 - @all(σ))
    return nothing
end

@parallel function positivity_fix!(h, hmin)
    @all(h) = max(@all(h), hmin)
    return nothing
end


struct Island
    x0::Float64
    y0::Float64
    zmax::Float64
    rflat::Float64
    redge::Float64
end

function background_bumps(xs, ys; nhills=40, amp_range=(0.01, 0.06),
                          sigma_range=(1.5, 4.0), seed=nothing)

    if seed !== nothing
        Random.seed!(seed)
    end

    X = [x for x in xs, y in ys]
    Y = [y for x in xs, y in ys]

    Z = zeros(length(xs), length(ys))

    # Domain limits
    xmin, xmax = minimum(xs), maximum(xs)
    ymin, ymax = minimum(ys), maximum(ys)

    for _ in 1:nhills
        # Random hill center
        x0 = rand() * (xmax - xmin) + xmin
        y0 = rand() * (ymax - ymin) + ymin

        # Random shallow amplitude
        A = rand() * (amp_range[2] - amp_range[1]) + amp_range[1]

        # Random width
        σ = rand() * (sigma_range[2] - sigma_range[1]) + sigma_range[1]

        Z .+= A .* exp.(-((X .- x0).^2 .+ (Y .- y0).^2) ./ (2σ^2))
    end

    return Z
end

function add_island!(z, xs, ys, isl::Island)
    for i in eachindex(xs), j in eachindex(ys)
        x = xs[i]
        y = ys[j]
        r = sqrt((x - isl.x0)^2 + (y - isl.y0)^2)

        if r <= isl.rflat
            z[i, j] += isl.zmax
        elseif r <= isl.redge
            s = (r - isl.rflat) / (isl.redge - isl.rflat)
            z[i, j] += isl.zmax * 0.5 * (1 + cos(pi * s))
        end
    end
    return z
end

function build_topography(xs, ys; islands=Island[], background=nothing)
    z = zeros(length(xs), length(ys))

    for isl in islands
        add_island!(z, xs, ys, isl)
    end

    if background !== nothing
        z .+= background(xs, ys)
    end

    return z
end
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@views function swe2d_topography_frames(; outdir = "frames")
    # physics
    lx = 50.0
    ly = 50.0

    # numerics
    nx   = 400
    ny   = 400
    nt   = Int(2 * nx)
    nvis = 5

    dx = lx / (nx - 1)
    dy = ly / (ny - 1)

    _dx  = 1.0 / dx
    _dy  = 1.0 / dy
    _2dx = 1.0 / (2 * dx)
    _2dy = 1.0 / (2 * dy)

    xs = LinRange(-lx / 2, lx / 2, nx)
    ys = LinRange(-ly / 2, ly / 2, ny)

    # state
    h  = zeros(nx, ny)
    hu = zeros(nx, ny)
    hv = zeros(nx, ny)

    # fluxes
    F₁ = zeros(nx - 1, ny)
    F₂ = zeros(nx - 1, ny)
    F₃ = zeros(nx - 1, ny)

    G₁ = zeros(nx, ny - 1)
    G₂ = zeros(nx, ny - 1)
    G₃ = zeros(nx, ny - 1)

    max_speed_x = zeros(nx - 1, ny)
    max_speed_y = zeros(nx, ny - 1)

    # -------------------------------------------------------------------------
    # topography
    # -------------------------------------------------------------------------
    islands = [
    Island(-10.0,  0.0, 0.12, 3.0, 4.5),   # above free surface if η≈0.10 outside
    Island(  9.0,  6.0, 0.105,5.0, 6.5),   # submerged bump
    Island(  5.0, -8.0, 0.12, 2.5, 4.0),
    Island( 15.0, -3.0, 0.11, 2.0, 7.0),
    Island(-12.0,  8.0, 0.11, 3.0, 5.0),
    Island(-15.0,-13.0, 0.12, 4.5, 6.0)   # clearly emergent   
    ]

    z = build_topography(xs, ys; islands=islands, background= (xs, ys) -> background_bumps(xs, ys, seed=42))

    dzdx = zeros(nx, ny)
    dzdy = zeros(nx, ny)

    dzdx[2:end-1, :] .= (z[3:end, :] .- z[1:end-2, :]) .* _2dx
    dzdx[1, :]       .= dzdx[2, :]
    dzdx[end, :]     .= dzdx[end-1, :]

    dzdy[:, 2:end-1] .= (z[:, 3:end] .- z[:, 1:end-2]) .* _2dy
    dzdy[:, 1]       .= dzdy[:, 2]
    dzdy[:, end]     .= dzdy[:, end-1]

    # -------------------------------------------------------------------------
    # sponge layer
    # -------------------------------------------------------------------------

    d = zeros(nx, ny)
    σ = zeros(nx, ny)

    for i in 1:nx, j in 1:ny
        di = min(i - 1, nx - i)
        dj = min(j - 1, ny - j)
        d[i, j] = min(di, dj)
    end

    layers  = 20
    _layers = 1.0 / layers
    σmax    = 0.15

    for i in 1:nx, j in 1:ny
        if d[i, j] < layers
            σ[i, j] = σmax * (1 - d[i, j] * _layers)
        else
            σ[i, j] = 0.0
        end
    end

    # # -------------------------------------------------------------------------
    # # initial condition
    # # -------------------------------------------------------------------------

    # h_in  = 0.20
    # h_out = 0.10
    # r0    = 2.5
    # hmin  = 1e-6

    # η0 = [((x^2 + y^2) < r0^2) ? h_in : h_out for x in xs, y in ys]
    # h .= max.(hmin, η0 .- z)

    # -------------------------------------------------------------------------
    # initial condition
    # -------------------------------------------------------------------------

    h_out = 0.10          # background free-surface level
    a     = 0.08          # wave amplitude
    y0    = 20.0          # crest location
    σy    = 3.0           # wave width
    hmin  = 1e-6

    # free surface eta(x,y) = eta(y), homogeneous in x
    η0 = [h_out + a * exp(-((y - y0)^2) / (2 * σy^2)) for x in xs, y in ys]

    # water depth
    h .= max.(hmin, η0 .- z)

    # ------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # visualization
    # -------------------------------------------------------------------------

    mkpath(outdir)

    vertical_exaggeration = 6.0
    hmin_plot = 1e-3

    z_plot = vertical_exaggeration .* z

    # terrain color as full matrix, not a single Symbol
    terrain_color = fill(RGBf(0.82, 0.82, 0.82), size(z))

    η_water_plot0  = vertical_exaggeration .* (h .+ z)
    η_water_color0 = h .+ z

    η_water_plot0[h .<= hmin_plot]  .= NaN
    η_water_color0[h .<= hmin_plot] .= NaN

    η_water_plot  = Observable(η_water_plot0)
    η_water_color = Observable(η_water_color0)

    fig = Figure(size = (1200, 900))
    ax = Axis3(
        fig[1, 1],
        xlabel = "x",
        ylabel = "y",
        zlabel = "height",
        aspect = (1, 1, 0.25),
        azimuth = -1.1 - π/2,
        elevation = 0.45,
        perspectiveness = 0.35
    )

    # gray terrain / islands
    surface!(ax, xs, ys, z_plot;
        color = terrain_color,
        shading = true
    )

    # water only
    water = surface!(ax, xs, ys, η_water_plot;
        color = η_water_color,
        colormap = :turbo,
        colorrange = (0.05, 0.15),
        shading = true
    )

    Colorbar(fig[1, 2], water, label = "free surface")
    display(fig)

    frame_id = Ref(0)

    function save_frame!()
        frame_id[] += 1
        fname = joinpath(outdir, @sprintf("frame_%06d.png", frame_id[]))
        save(fname, fig)
    end

    save_frame!()

    # -------------------------------------------------------------------------
    # main loop
    # -------------------------------------------------------------------------

    for it in 1:nt
        @parallel compute_maxspeed!(max_speed_x, max_speed_y, h, hu, hv, g)

        dt = if !USE_GPU
            dt_multithread(max_speed_x, max_speed_y, _dx, _dy, ny)
        else
            0.99 / (maximum(max_speed_x) * _dx + maximum(max_speed_y) * _dy)
        end

        @parallel compute_first_flux!(F₁, G₁, hu, hv, h, max_speed_x, max_speed_y)
        @parallel compute_f2_g3_flux!(F₂, G₃, hu, hv, h, g, max_speed_x, max_speed_y)
        @parallel compute_f3_g2_flux!(F₃, G₂, hu, hv, h, max_speed_x, max_speed_y)

        @parallel update_height!(h, F₁, G₁, dt, _dx, _dy)
        @parallel update_momentum_with_source!(hu, hv, h, F₂, F₃, G₂, G₃, dzdx, dzdy, g, dt, _dx, _dy)

        @parallel sides_bc!(h, hu, hv, g, dt, _dx)
        @parallel top_bottom_bc!(h, hu, hv, g, dt, _dy)

        @parallel sponge_layer!(hu, hv, σ)
        @parallel positivity_fix!(h, hmin)

        if it % nvis == 0
            ηtmp_plot  = vertical_exaggeration .* (h .+ z)
            ηtmp_color = h .+ z

            ηtmp_plot[h .<= hmin_plot]  .= NaN
            ηtmp_color[h .<= hmin_plot] .= NaN

            η_water_plot[]  = ηtmp_plot
            η_water_color[] = ηtmp_color

            save_frame!()
        end

        percent = 100 * it / nt
        print("\rProgress: $(round(percent, digits=1)) %")
        flush(stdout)
    end

    println("\nSaved $(frame_id[]) frames to: $(abspath(outdir))")
    return nothing
end

swe2d_topography_frames()