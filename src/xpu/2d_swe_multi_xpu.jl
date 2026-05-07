using GLMakie
using StaticArrays
using Random

using ImplicitGlobalGrid
import MPI

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

min_g(A) = (min_l = minimum(A); MPI.Allreduce(min_l, MPI.MIN, MPI.COMM_WORLD))
max_g(A) = (max_l = maximum(A); MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD))

@views function dt_multithread(max_speed_x, max_speed_y, _dx, _dy, n)
    nthreads = Threads.nthreads()
    max_speeds_x = zeros(nthreads)
    max_speeds_y = zeros(nthreads)

    Threads.@threads for i in 1:n
        tid = Threads.threadid()
        max_speeds_x[tid] = max(max_speeds_x[tid], maximum(max_speed_x[:, i]))
        max_speeds_y[tid] = max(max_speeds_y[tid], maximum(max_speed_y[i, :]))
    end

    return 0.99 / (max_g(max_speeds_x) * _dx + max_g(max_speeds_y) * _dy)
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
        h[ix, iy] -= dt * (dxb(F₁, ix, iy) * _dx + dyb(G₁, ix, iy) * _dy)
    end

    if (2 <= ix <= nx-1 && 2 <= iy <= ny-1)
        hu[ix, iy] -= dt * (dxb(F₂, ix, iy) * _dx + dyb(G₂, ix, iy) * _dy + g * h[ix, iy] * dzdx[ix, iy])
        hv[ix, iy] -= dt * (dxb(F₃, ix, iy) * _dx + dyb(G₃, ix, iy) * _dy + g * h[ix, iy] * dzdy[ix, iy])
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
    xmin, xmax = min_g(xs), max_g(xs)
    ymin, ymax = min_g(ys), max_g(ys)

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
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@views function swe2d_topography_frames(; outdir = "frames", do_viz = true)
    # physics and numerics
    lx_aoi = 50.0 # aoi = area of interest
    ly_aoi = 50.0
    nx_aoi = 100
    ny_aoi = 100

    # Multiply domain size to allow for sponge layer and BCs
    domain_expansion_factor = 3

    lx = domain_expansion_factor * lx_aoi
    ly = domain_expansion_factor * ly_aoi
    nx = round(Int, domain_expansion_factor * nx_aoi)
    ny = round(Int, domain_expansion_factor * ny_aoi)
    me, dims = init_global_grid(nx, ny, 1; select_device = false)
    b_width     = (8, 8, 1)

    nt   = Int(2 * nx_aoi)
    nvis = 5

    dx = lx / (nx_g() - 1)
    dy = ly / (ny_g() - 1)

    _dx  = 1.0 / dx
    _dy  = 1.0 / dy
    _2dx = 1.0 / (2 * dx)
    _2dy = 1.0 / (2 * dy)
    
    # state
    h  = @zeros(nx, ny)
    hu = @zeros(nx, ny)
    hv = @zeros(nx, ny)

    xs = [x_g(ix, dx, h) - lx / 2 for ix in 1:nx]
    ys = [y_g(iy, dy, h) - ly / 2 for iy in 1:ny]

    # fluxes
    F₁ = @zeros(nx - 1, ny)
    F₂ = @zeros(nx - 1, ny)
    F₃ = @zeros(nx - 1, ny)

    G₁ = @zeros(nx, ny - 1)
    G₂ = @zeros(nx, ny - 1)
    G₃ = @zeros(nx, ny - 1)

    max_speed_x = @zeros(nx - 1, ny)
    max_speed_y = @zeros(nx, ny - 1)

    # -------------------------------------------------------------------------
    # topography
    # -------------------------------------------------------------------------
    # islands = [
    # Island(-10.0,  0.0, 0.12, 3.0, 4.5),   # above free surface if η≈0.10 outside
    # Island(  9.0,  6.0, 0.105,5.0, 6.5),   # submerged bump
    # Island(  5.0, -8.0, 0.12, 2.5, 4.0),
    # Island( 15.0, -3.0, 0.11, 2.0, 7.0),
    # Island(-12.0,  8.0, 0.11, 3.0, 5.0),
    # Island(-15.0,-13.0, 0.12, 4.5, 6.0)   # clearly emergent   
    # ]


    z = build_topography(xs, ys; islands=[], background= (xs, ys) -> background_bumps(xs, ys, seed=43))

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
    X = Data.Array(reshape(xs, nx, 1))   # nx × 1
    Y = Data.Array(reshape(ys, 1, ny))   # 1 × ny

    # distance to global boundary in terms of grid cells
    dist_x = min.(X .- (-lx/2), (lx/2) .- X) .* _dx
    dist_y = min.(Y .- (-ly/2), (ly/2) .- Y) .* _dy
    
    d .= min.(dist_x, dist_y)

    # damping profile
    σ .= ifelse.(d .< layers,
                σmax .* (1 .- d .* _layers),
                zero(eltype(σ)))

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
    # η0 = [h_out + a * exp(-((y - y0)^2) / (2 * σy^2)) for x in xs, y in ys]
    η0 = h_out

    # water depth
    h .= max.(hmin, η0 .- z)

    # -------------------------------------------------------------------------
    # visualization
    # -------------------------------------------------------------------------

    if do_viz
        mkpath(outdir)

        vertical_exaggeration = 6.0
        hmin_plot = 1e-3
        
        nx_v, ny_v = (nx - 2) * dims[1], (ny - 2) * dims[2]
        h_v   = zeros(nx_v, ny_v)
        z_v   = zeros(nx_v, ny_v)
        h_inn = zeros(nx - 2, ny - 2)
        z_inn = zeros(nx - 2, ny - 2)
        
        # Compute area of interes size relative to global grid
        nx_aoi_v = round(Int, nx_v / domain_expansion_factor)
        ny_aoi_v = round(Int, ny_v / domain_expansion_factor)
        
        pad_x_v  = round(Int, (nx_v - nx_aoi_v) / 2)
        pad_y_v  = round(Int, (ny_v - ny_aoi_v) / 2)
        ix_roi_v = (pad_x_v + 1):(pad_x_v + nx_aoi_v)
        iy_roi_v = (pad_y_v + 1):(pad_y_v + ny_aoi_v)
        

        xs_v = LinRange(-lx / 2 + dx, lx / 2 - dx, nx_v)
        ys_v = LinRange(-ly / 2 + dy, ly / 2 - dy, ny_v)
        xs_roi_v = xs_v[ix_roi_v]
        ys_roi_v = ys_v[iy_roi_v]

        h_inn .= Array(h)[2:end-1, 2:end-1]; gather!(h_inn, h_v)
        z_inn .= Array(z)[2:end-1, 2:end-1]; gather!(z_inn, z_v)

        if me == 0
            h_slice = h_v[ix_roi_v, iy_roi_v]
            z_slice = z_v[ix_roi_v, iy_roi_v]

            z_plot = vertical_exaggeration .* z_slice

            # terrain color as full matrix, not a single Symbol
            terrain_color = fill(RGBf(0.82, 0.82, 0.82), size(z_plot))

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
            surface!(ax, xs_roi_v, ys_roi_v, z_plot;
                color = terrain_color,
                shading = true
            )

            # water only
            water = surface!(ax, xs_roi_v, ys_roi_v, η_water_plot;
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
    end

    # -------------------------------------------------------------------------
    # main loop
    # -------------------------------------------------------------------------

    for it in 1:nt
        @parallel compute_maxspeed!(max_speed_x, max_speed_y, h, hu, hv, g)

        dt = if !USE_GPU
            dt_multithread(max_speed_x, max_speed_y, _dx, _dy, ny)
        else
            0.99 / (max_g(max_speed_x) * _dx + max_g(max_speed_y) * _dy)
        end
        
        @parallel compute_1st_2nd_and_3th_flux!(F₁, F₂, F₃, G₁, G₂, G₃, hu, hv, h, g, max_speed_x, max_speed_y)
        
        @parallel update_height_momentum!(h, hu, hv, F₁, G₁, F₂, F₃, G₂, G₃, dzdx, dzdy, g, dt, _dx, _dy)

        @hide_communication b_width begin
            @parallel all_bc!(h, hu, hv, g, dt, _dx, _dy)
            @parallel sponge_layer!(hu, hv, σ)
            @parallel positivity_fix!(h, hmin)
            
            update_halo!(h, hu, hv)
        end

        if do_viz && it % nvis == 0

            h_inn .= Array(h)[2:end-1, 2:end-1]; gather!(h_inn, h_v)
            z_inn .= Array(z)[2:end-1, 2:end-1]; gather!(z_inn, z_v)
            
            if me == 0
                h_slice = h_v[ix_roi_v, iy_roi_v]
                z_slice = z_v[ix_roi_v, iy_roi_v]

                ηtmp_plot  = vertical_exaggeration .* (h_slice .+ z_slice)
                ηtmp_color = h_slice .+ z_slice

                ηtmp_plot[h_slice .<= hmin_plot]  .= NaN
                ηtmp_color[h_slice .<= hmin_plot] .= NaN

                η_water_plot[]  = ηtmp_plot
                η_water_color[] = ηtmp_color

                save_frame!()
            end
        end
        
        if me == 0
            percent = 100 * it / nt
            print("\rProgress: $(round(percent, digits=1)) %")
            flush(stdout)
        end
    end
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    if me == 0
        # Check the boundaries of the ROI
        is_preserved = check_bc_preserves_eta(h_v, z_v, η0, ix_roi_v, iy_roi_v)
        
        # Total error inside the global ROI (v=visualization domain)
        h_roi_v = h_v[ix_roi_v, iy_roi_v]
        z_roi_v = z_v[ix_roi_v, iy_roi_v]
        max_err_roi = maximum(abs.(η0 .- (h_roi_v .+ z_roi_v)))
        
        println("\nValidation Results:")
        println("ROI boundaries preserved within tolerance? ", is_preserved)
        println("Maximum error across entire ROI: ", max_err_roi)

        max_err = maximum(η0.-(h_v+z_v))
        rel_err = max_err/η0
        print("relative error: ", rel_err)

        if do_viz
            println("\nSaved $(frame_id[]) frames to: $(abspath(outdir))")
        end
    end

    finalize_global_grid()
    return nothing
end

swe2d_topography_frames()