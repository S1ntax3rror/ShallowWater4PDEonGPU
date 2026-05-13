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

@parallel_indices (ix, iy) function compute_1st_2nd_and_3th_flux!(F₁, F₂, F₃, G₁, G₂, G₃, hu, hv, h, g, max_speed_x, max_speed_y)
    nx, ny = size(h)
    if (ix <= nx - 1 && iy <= ny)
        F₁[ix, iy] = 0.5 * (hu[ix, iy] + hu[ix+1, iy]) - 0.5 * max_speed_x[ix, iy] * dxa(h, ix, iy)
        F₂[ix, iy] = avx_comp(hu, hu, h, ix, iy) + 0.5 * g * avx_simp(h, ix, iy) - 0.5 * max_speed_x[ix, iy] * dxa(hu, ix, iy)
        F₃[ix, iy] = avx_comp(hu, hv, h, ix, iy) - 0.5 * max_speed_x[ix, iy] * dxa(hv, ix, iy)
    end
    if (ix <= nx && iy <= ny - 1)
        G₁[ix, iy] = 0.5 * (hv[ix, iy] + hv[ix, iy+1]) - 0.5 * max_speed_y[ix, iy] * dya(h, ix, iy)
        G₂[ix, iy] = avy_comp(hv, hu, h, ix, iy) - 0.5 * max_speed_y[ix, iy] * dya(hu, ix, iy)
        G₃[ix, iy] = avy_comp(hv, hv, h, ix, iy) + 0.5 * g * avy_simp(h, ix, iy) - 0.5 * max_speed_y[ix, iy] * dya(hv, ix, iy)
    end
    return nothing
end



@parallel_indices (ix, iy) function update_height_momentum!(h, hu, hv, F₁, G₁, F₂, F₃, G₂, G₃, dzdx, dzdy, g, dt, _dx, _dy)
    nx, ny = size(h)

    if (2 <= ix <= nx-1 && 2 <= iy <= ny-1)
        hu[ix, iy] -= dt * (dxb(F₂, ix, iy) * _dx + dyb(G₂, ix, iy) * _dy + g * h[ix, iy] * dzdx[ix, iy])
        hv[ix, iy] -= dt * (dxb(F₃, ix, iy) * _dx + dyb(G₃, ix, iy) * _dy + g * h[ix, iy] * dzdy[ix, iy])
        h[ix, iy] -= dt * (dxb(F₁, ix, iy) * _dx + dyb(G₁, ix, iy) * _dy)
    end
    return nothing
end


@parallel_indices (ix, iy) function all_bc!(h, hu, hv, g, dt, _dx, _dy)
    nx, ny = size(h)

    # Left boundary (ix=1)
    if ix == 1 && iy <= ny
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

    # Right boundary (ix=nx)
    if ix == nx && iy <= ny
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

    # Bottom boundary (iy=1)
    if iy == 1 && ix <= nx
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

    # Top boundary (iy=ny)
    if iy == ny && ix <= nx
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

function check_bc_preserves_eta(h, z, η0, ix_roi, iy_roi; tol=1e-8)
    """ Check if BC (eta = h + z) = eta0 """
    eta_roi = h[ix_roi, iy_roi] .+ z[ix_roi, iy_roi]

    top = eta_roi[:, end]
    bottom = eta_roi[:, 1]
    left = eta_roi[1, :]
    right = eta_roi[end, :]

    bvals = vcat(vec(top), vec(bottom), vec(left), vec(right))
    maxdev = maximum(abs.(bvals .- η0))
    @info "BC eta_roi max deviation" maxdev
    return maxdev <= tol
end


struct Island
    x0::Float64
    y0::Float64
    zmax::Float64
    rflat::Float64
    redge::Float64
end

function background_bumps(xs, ys; nhills=40, amp_range=(0.01, 0.03),
                          sigma_range=(1.5, 4.0), seed=nothing)

    if seed !== nothing
        Random.seed!(seed)
    end

    X = [x for x in xs, y in ys]
    Y = [y for x in xs, y in ys]

    Z = @zeros(length(xs), length(ys))

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
    z = @zeros(length(xs), length(ys))

    for isl in islands
        add_island!(z, xs, ys, isl)
    end

    if background !== nothing
        z .+= background(xs, ys)
    end

    return z
end

function load_topography_data(domain_expansion_factor, nx_aoi_ext, ny_aoi_ext)
    base_file = "data/tsunamiOku/D112-94-50m.txt"
    wave_file = "data/tsunamiOku/I112-94-50m-17a.txt"

    nx_aoi, ny_aoi = 112, 94 # ENSURE THIS MATCHES THE DATA FILES

    # Read as string, split by whitespace, and parse to Float64
    read_values(filename) = parse.(Float64, split(read(filename, String)))

    z_vec = read_values(base_file)
    η0_vec = zeros(nx_aoi, ny_aoi)

    if length(z_vec) != nx_aoi * ny_aoi || length(η0_vec) != nx_aoi * ny_aoi
        error("Data files do not match expected dimensions.")
    end

    # Reshape to 2D arrays
    z_inner_orig = reshape(z_vec, nx_aoi, ny_aoi)
    η0_inner_orig = reshape(η0_vec, nx_aoi, ny_aoi)

    z_inner = zeros(nx_aoi_ext, ny_aoi_ext)
    η0_inner = zeros(nx_aoi_ext, ny_aoi_ext)
    
    # Bilinear interpolation to expand the inner grid to the extended grid
    for i in 1:nx_aoi_ext
        for j in 1:ny_aoi_ext
            # Map the new output index continuously to the original input grid
            x = 1 + (i - 1) * (nx_aoi - 1) / (nx_aoi_ext - 1)
            y = 1 + (j - 1) * (ny_aoi - 1) / (ny_aoi_ext - 1)
            
            # Get integer bounds for interpolation
            x1, y1 = floor(Int, x), floor(Int, y)
            x2, y2 = min(x1 + 1, nx_aoi), min(y1 + 1, ny_aoi)
            
            # Calculate weights
            wx = x - x1
            wy = y - y1
            
            # Interpolate z
            z_inner[i, j] = (1 - wx) * (1 - wy) * z_inner_orig[x1, y1] + 
                                 wx  * (1 - wy) * z_inner_orig[x2, y1] + 
                            (1 - wx) * wy  * z_inner_orig[x1, y2] + 
                                 wx  * wy  * z_inner_orig[x2, y2]
                            
            # Interpolate η0
            η0_inner[i, j] = (1 - wx) * (1 - wy) * η0_inner_orig[x1, y1] + 
                                  wx  * (1 - wy) * η0_inner_orig[x2, y1] + 
                             (1 - wx) * wy  * η0_inner_orig[x1, y2] + 
                                  wx  * wy  * η0_inner_orig[x2, y2]
        end
    end

    # Calculate expanded dimensions 
    nx = round(Int, domain_expansion_factor * nx_aoi_ext)
    ny = round(Int, domain_expansion_factor * ny_aoi_ext)

    pad_x = round(Int, (nx - nx_aoi_ext) / 2)
    pad_y = round(Int, (ny - ny_aoi_ext) / 2)

    z_expanded = zeros(nx, ny)
    η0_expanded = zeros(nx, ny)

    # Pad the arrays
    for i in 1:nx
        for j in 1:ny
            # Clamp to the closest valid index of the inner grid
            orig_i = clamp(i - pad_x, 1, nx_aoi_ext)
            orig_j = clamp(j - pad_y, 1, ny_aoi_ext)
            
            # Stretch the edge bathymetry outwards
            z_expanded[i, j] = z_inner[orig_i, orig_j]
            
            # Check if the current cell is inside the ROI
            in_roi = (1 <= i - pad_x <= nx_aoi_ext) && (1 <= j - pad_y <= ny_aoi_ext)
            
            # If in ROI, load the wave. If in padding, use resting sea level (0.0)
            η0_expanded[i, j] = in_roi ? η0_inner[i - pad_x, j - pad_y] : 0.0
        end
    end

    # Cast to ParallelStencil arrays
    z = Data.Array(z_expanded)
    η0 = Data.Array(η0_expanded)

    return z, η0
end



# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@views function swe2d_topography_frames(nx_aoi, ny_aoi; outdir = "frames", do_viz = true)
    # physics and numerics
    lx_aoi = 50.0 # aoi = area of interest
    ly_aoi = 50.0
    #nx_aoi = 125
    #ny_aoi = 125

    # Multiply domain size to allow for sponge layer and BCs
    domain_expansion_factor = 3

    lx = domain_expansion_factor * lx_aoi
    ly = domain_expansion_factor * ly_aoi
    nx = round(Int, domain_expansion_factor * nx_aoi)
    ny = round(Int, domain_expansion_factor * ny_aoi)

    nt   = Int(2 * nx_aoi)
    nvis = 5

    dx = lx / (nx - 1)
    dy = ly / (ny - 1)

    _dx  = 1.0 / dx
    _dy  = 1.0 / dy
    _2dx = 1.0 / (2 * dx)
    _2dy = 1.0 / (2 * dy)
    
    xs = LinRange(-lx / 2, lx / 2, nx)
    ys = LinRange(-ly / 2, ly / 2, ny)

    # ROI indices for visualization
    pad_x = round(Int, (nx - nx_aoi) / 2)
    pad_y = round(Int, (ny - ny_aoi) / 2)

    ix_roi = (pad_x + 1):(pad_x + nx_aoi)
    iy_roi = (pad_y + 1):(pad_y + ny_aoi)

    xs_roi = xs[ix_roi]
    ys_roi = ys[iy_roi]

    # state
    h  = @zeros(nx, ny)
    hu = @zeros(nx, ny)
    hv = @zeros(nx, ny)

    # fluxes
    F₁ = @zeros(nx - 1, ny)
    F₂ = @zeros(nx - 1, ny)
    F₃ = @zeros(nx - 1, ny)

    G₁ = @zeros(nx, ny - 1)
    G₂ = @zeros(nx, ny - 1)
    G₃ = @zeros(nx, ny - 1)

    max_speed_x = @zeros(nx - 1, ny)
    max_speed_y = @zeros(nx, ny - 1)

    

    z, η0 = load_topography_data(domain_expansion_factor, nx_aoi, ny_aoi)

    η0_const = maximum(z) + 0.1
    h .= η0_const .- z

    η0 .= η0_const

    hmin  = 1e-6
    h .= max.(hmin, η0 .- z)

    dzdx = @zeros(nx, ny)
    dzdy = @zeros(nx, ny)

    dzdx[2:end-1, :] .= (z[3:end, :] .- z[1:end-2, :]) .* _2dx
    dzdx[1, :]       .= dzdx[2, :]
    dzdx[end, :]     .= dzdx[end-1, :]

    dzdy[:, 2:end-1] .= (z[:, 3:end] .- z[:, 1:end-2]) .* _2dy
    dzdy[:, 1]       .= dzdy[:, 2]
    dzdy[:, end]     .= dzdy[:, end-1]

    # -------------------------------------------------------------------------
    # sponge layer
    # -------------------------------------------------------------------------

    # backend arrays
    d = @zeros(nx, ny)
    σ = @zeros(nx, ny)

    layers  = 20
    _layers = 1.0 / layers
    σmax    = 0.15

    # 1D index vectors on the backend
    ix = Data.Array(reshape(collect(1:nx), nx, 1))   # nx × 1
    iy = Data.Array(reshape(collect(1:ny), 1, ny))   # 1 × ny

    # distance to nearest boundary in each direction
    di = min.(ix .- 1, nx .- ix)
    dj = min.(iy .- 1, ny .- iy)

    # full 2D distance field by broadcasting
    d .= min.(di, dj)

    # damping profile
    σ .= ifelse.(d .< layers,
                σmax .* (1 .- d .* _layers),
                zero(eltype(σ)))

    if do_viz
        mkpath(outdir)

        vertical_exaggeration = 6.0
        hmin_plot = 1e-3
        
        z_slice = z[ix_roi, iy_roi]
        z_plot = vertical_exaggeration .* z_slice

        # terrain color as full matrix, not a single Symbol
        terrain_color = fill(RGBf(0.82, 0.82, 0.82), size(z_plot))

        h_slice = h[ix_roi, iy_roi]

        η_water_plot0  = vertical_exaggeration .* (h_slice .+ z_slice)
        η_water_color0 = h_slice .+ z_slice

        η_water_plot0[h_slice .<= hmin_plot]  .= NaN
        η_water_color0[h_slice .<= hmin_plot] .= NaN

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
        surface!(ax, xs_roi, ys_roi, z_plot;
            color = terrain_color,
            shading = true
        )

        # water only
        water = surface!(ax, xs_roi, ys_roi, η_water_plot;
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
    end

    # -------------------------------------------------------------------------
    # main loop
    # -------------------------------------------------------------------------

    initial_err = maximum(abs.(h[ix_roi, iy_roi] .+ z[ix_roi, iy_roi] .- η0_const))
    println("initial lake-at-rest error = ", initial_err)

    for it in 1:nt
        @parallel compute_maxspeed!(max_speed_x, max_speed_y, h, hu, hv, g)

        dt = if !USE_GPU
            dt_multithread(max_speed_x, max_speed_y, _dx, _dy, ny)
        else
            0.99 / (maximum(max_speed_x) * _dx + maximum(max_speed_y) * _dy)
        end

        @parallel compute_1st_2nd_and_3th_flux!(F₁, F₂, F₃, G₁, G₂, G₃, hu, hv, h, g, max_speed_x, max_speed_y)

        @parallel update_height_momentum!(h, hu, hv, F₁, G₁, F₂, F₃, G₂, G₃, dzdx, dzdy, g, dt, _dx, _dy)

        @parallel all_bc!(h, hu, hv, g, dt, _dx, _dy)

        @parallel sponge_layer!(hu, hv, σ)
        @parallel positivity_fix!(h, hmin)

        if it % nvis == 0
            if do_viz
                h_slice = h[ix_roi, iy_roi]
                z_slice = z[ix_roi, iy_roi]

                ηtmp_plot  = vertical_exaggeration .* (h_slice .+ z_slice)
                ηtmp_color = h_slice .+ z_slice

                ηtmp_plot[h_slice .<= hmin_plot]  .= NaN
                ηtmp_color[h_slice .<= hmin_plot] .= NaN

                η_water_plot[]  = ηtmp_plot
                η_water_color[] = ηtmp_color

                save_frame!()
            end

        end

        percent = 100 * it / nt
        print("\rProgress: $(round(percent, digits=1)) %")
        flush(stdout)
    end

    h_slice = h[ix_roi, iy_roi]
    z_slice = z[ix_roi, iy_roi]

    η_0 = zeros(nx_aoi, ny_aoi)


    # Calculate global errors
    max_err = maximum(abs.(η_0 .- (h_slice .+ z_slice)))
    max_η0_val = h_slice isa AbstractArray ? maximum(abs.(h_slice)) : abs(h_slice)
    rel_err = max_err / max_η0_val
    println("relative error: ", rel_err)

    # print average height
    avg_height = mean(h_slice)
    println("average height in ROI: ", avg_height)

    if do_viz
        println("\nSaved $(frame_id[]) frames to: $(abspath(outdir))")
    end
    return rel_err
end

# error benchmark

# resolutions
resolutions = [25, 50, 100, 125, 250, 500, 1000]

errors = Float64[]

for res in resolutions
    println("\nRunning benchmark for resolution: $(res)x$(res)")
    error = swe2d_topography_frames(res, res; outdir = "frames_$(res)x$(res)", do_viz = false)
    push!(errors, error)
end

# plot errors

using Plots

plot(resolutions, errors, xscale=:log10, yscale=:log10, marker=:o, xlabel="Resolution (nx=ny)", ylabel="Relative Error", title="Error Benchmark for 2D SWE with Topography", legend=false)
savefig("docs/swe2d_topography_error_benchmark.png")


swe2d_topography_frames()