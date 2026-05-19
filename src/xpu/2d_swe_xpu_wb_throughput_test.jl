using Serialization
using DelimitedFiles
using BenchmarkTools

const HAS_MAKIE = try
    @eval using GLMakie
    true
catch
    @info "GLMakie not found. Falling back to array output."
    false
end

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

const h_eps = 1e-2
const nt_nx_multiplier = 4

@inline avx_comp(hv1, hv2, h, ix, iy) = 0.5 * (hv1[ix, iy] * hv2[ix, iy] / h[ix, iy] + hv1[ix+1, iy] * hv2[ix+1, iy] / h[ix+1, iy])
@inline avy_comp(hv1, hv2, h, ix, iy) = 0.5 * (hv1[ix, iy] * hv2[ix, iy] / h[ix, iy] + hv1[ix, iy+1] * hv2[ix, iy+1] / h[ix, iy+1])
@inline avx_simp(h, ix, iy) = 0.5 * (h[ix, iy] * h[ix, iy] + h[ix+1, iy] * h[ix+1, iy])
@inline avy_simp(h, ix, iy) = 0.5 * (h[ix, iy] * h[ix, iy] + h[ix, iy+1] * h[ix, iy+1])
@inline dxa(h, ix, iy) = h[ix+1, iy] - h[ix, iy]
@inline dya(h, ix, iy) = h[ix, iy+1] - h[ix, iy]

@inline dxb(h, ix, iy) = h[ix, iy] - h[ix-1, iy]
@inline dyb(h, ix, iy) = h[ix, iy] - h[ix, iy-1]

@inline eta(h, z, ix, iy) = h[ix, iy] + z[ix, iy]

@inline zx_face(z, ix, iy) = 0.5 * (z[ix, iy] + z[ix+1, iy])
@inline zy_face(z, ix, iy) = 0.5 * (z[ix, iy] + z[ix, iy+1])

@inline hx_L(h, z, ix, iy) =
    max(0.0, eta(h, z, ix, iy) - zx_face(z, ix, iy))

@inline hx_R(h, z, ix, iy) =
    max(0.0, eta(h, z, ix+1, iy) - zx_face(z, ix, iy))

@inline hy_L(h, z, ix, iy) =
    max(0.0, eta(h, z, ix, iy) - zy_face(z, ix, iy))

@inline hy_R(h, z, ix, iy) =
    max(0.0, eta(h, z, ix, iy+1) - zy_face(z, ix, iy))

@inline function desing_velocity(hval, qval, vel_eps)
    if hval <= 0.0
        return 0.0
    end

    return sqrt(2.0) * hval * qval /
           sqrt(hval^4 + max(hval^4, vel_eps))
end

@inline vel_u(h, hu, ix, iy, vel_eps) =
    desing_velocity(h[ix, iy], hu[ix, iy], vel_eps)

@inline vel_v(h, hv, ix, iy, vel_eps) =
    desing_velocity(h[ix, iy], hv[ix, iy], vel_eps)

@inline bc_speed_x(h, hu, ix, iy, g) =
    h[ix, iy] > h_eps ?
        abs(hu[ix, iy] / h[ix, iy]) + sqrt(g * h[ix, iy]) :
        0.0

@inline bc_speed_y(h, hv, ix, iy, g) =
    h[ix, iy] > h_eps ?
        abs(hv[ix, iy] / h[ix, iy]) + sqrt(g * h[ix, iy]) :
        0.0

const g = 1.0


@parallel_indices (ix, iy) function compute_maxspeed!(
    max_speed_x, max_speed_y,
    h, hu, hv, z, g, vel_eps
)
    nx, ny = size(h)

    if ix <= nx - 1 && iy <= ny
        hL = hx_L(h, z, ix, iy)
        hR = hx_R(h, z, ix, iy)

        uL = vel_u(h, hu, ix, iy, vel_eps)
        uR = vel_u(h, hu, ix+1, iy, vel_eps)

        max_speed_x[ix, iy] = max(
            abs(uL) + sqrt(g * hL),
            abs(uR) + sqrt(g * hR)
        )
    end

    if ix <= nx && iy <= ny - 1
        hL = hy_L(h, z, ix, iy)
        hR = hy_R(h, z, ix, iy)

        vL = vel_v(h, hv, ix, iy, vel_eps)
        vR = vel_v(h, hv, ix, iy+1, vel_eps)

        max_speed_y[ix, iy] = max(
            abs(vL) + sqrt(g * hL),
            abs(vR) + sqrt(g * hR)
        )
    end

    return nothing
end

# -----------------------------------------------------------------------------
# Kernels
# -----------------------------------------------------------------------------

@parallel_indices (ix, iy) function compute_draining_timestep!(
    dt_drain,
    F₁, G₁,
    h,
    dt,
    _dx, _dy
)
    nx, ny = size(h)

    if 2 <= ix <= nx-1 && 2 <= iy <= ny-1
        out_x =
            max(F₁[ix, iy], 0.0) +
            max(-F₁[ix-1, iy], 0.0)

        out_y =
            max(G₁[ix, iy], 0.0) +
            max(-G₁[ix, iy-1], 0.0)

        drain_rate = out_x * _dx + out_y * _dy

        if drain_rate > 0.0
            dt_drain[ix, iy] = min(dt, h[ix, iy] / drain_rate)
        else
            dt_drain[ix, iy] = dt
        end
    end

    return nothing
end

@parallel_indices (ix, iy) function compute_effective_flux_timesteps!(
    dtFx, dtGy,
    dt_drain,
    F₁, G₁,
    dt
)
    nxm1, ny = size(F₁)
    nx, nym1 = size(G₁)

    # x-faces
    if ix <= nxm1 && iy <= ny
        if F₁[ix, iy] > 0.0
            dtFx[ix, iy] = min(dt, dt_drain[ix, iy])
        elseif F₁[ix, iy] < 0.0
            dtFx[ix, iy] = min(dt, dt_drain[ix+1, iy])
        else
            dtFx[ix, iy] = dt
        end
    end

    # y-faces
    if ix <= nx && iy <= nym1
        if G₁[ix, iy] > 0.0
            dtGy[ix, iy] = min(dt, dt_drain[ix, iy])
        elseif G₁[ix, iy] < 0.0
            dtGy[ix, iy] = min(dt, dt_drain[ix, iy+1])
        else
            dtGy[ix, iy] = dt
        end
    end

    return nothing
end

@parallel_indices (ix, iy) function compute_1st_2nd_and_3th_flux!(
    F₁, F₂, F₃,
    G₁, G₂, G₃,
    hu, hv, h, z, g,
    max_speed_x, max_speed_y,
    vel_eps
)
    nx, ny = size(h)

    # -------------------------------------------------------------------------
    # x-direction fluxes
    # -------------------------------------------------------------------------
    if ix <= nx - 1 && iy <= ny
        hL = hx_L(h, z, ix, iy)
        hR = hx_R(h, z, ix, iy)

        ηL = eta(h, z, ix, iy)
        ηR = eta(h, z, ix+1, iy)

        uL = vel_u(h, hu, ix, iy, vel_eps)
        uR = vel_u(h, hu, ix+1, iy, vel_eps)

        vL = vel_v(h, hv, ix, iy, vel_eps)
        vR = vel_v(h, hv, ix+1, iy, vel_eps)

        # Reconstruct momenta consistently:
        # momentum = reconstructed depth × cell velocity
        huL = hL * uL
        huR = hR * uR
        hvL = hL * vL
        hvR = hR * vR

        ax = max_speed_x[ix, iy]

        # Mass / free-surface flux
        F₁[ix, iy] =
            0.5 * (huL + huR) -
            0.5 * ax * (ηR - ηL)

        # x-momentum flux
        F₂[ix, iy] =
            0.5 * (
                huL * uL + 0.5 * g * hL^2 +
                huR * uR + 0.5 * g * hR^2
            ) -
            0.5 * ax * (huR - huL)

        # y-momentum transported in x
        F₃[ix, iy] =
            0.5 * (
                huL * vL +
                huR * vR
            ) -
            0.5 * ax * (hvR - hvL)
    end

    # -------------------------------------------------------------------------
    # y-direction fluxes
    # -------------------------------------------------------------------------
    if ix <= nx && iy <= ny - 1
        hL = hy_L(h, z, ix, iy)
        hR = hy_R(h, z, ix, iy)

        ηL = eta(h, z, ix, iy)
        ηR = eta(h, z, ix, iy+1)

        uL = vel_u(h, hu, ix, iy, vel_eps)
        uR = vel_u(h, hu, ix, iy+1, vel_eps)

        vL = vel_v(h, hv, ix, iy, vel_eps)
        vR = vel_v(h, hv, ix, iy+1, vel_eps)

        huL = hL * uL
        huR = hR * uR
        hvL = hL * vL
        hvR = hR * vR

        ay = max_speed_y[ix, iy]

        # Mass / free-surface flux
        G₁[ix, iy] =
            0.5 * (hvL + hvR) -
            0.5 * ay * (ηR - ηL)

        # x-momentum transported in y
        G₂[ix, iy] =
            0.5 * (
                hvL * uL +
                hvR * uR
            ) -
            0.5 * ay * (huR - huL)

        # y-momentum flux
        G₃[ix, iy] =
            0.5 * (
                hvL * vL + 0.5 * g * hL^2 +
                hvR * vR + 0.5 * g * hR^2
            ) -
            0.5 * ay * (hvR - hvL)
    end

    return nothing
end

@parallel_indices (ix, iy) function update_height_momentum!(
    h, hu, hv,
    F₁, G₁, F₂, F₃, G₂, G₃, dtFx, dtGy,
    z, g, dt, _dx, _dy
)
    nx, ny = size(h)

    if 2 <= ix <= nx-1 && 2 <= iy <= ny-1
        ηC = eta(h, z, ix, iy)

        # ---------------------------------------------------------------------
        # x-source term
        # ---------------------------------------------------------------------
        zE = 0.5 * (z[ix, iy] + z[ix+1, iy])
        zW = 0.5 * (z[ix-1, iy] + z[ix, iy])

        hE = max(0.0, ηC - zE)
        hW = max(0.0, ηC - zW)

        hsrc_x = 0.5 * (hE + hW)
        dzdx_face = (zE - zW) * _dx

        # ---------------------------------------------------------------------
        # y-source term
        # ---------------------------------------------------------------------
        zN = 0.5 * (z[ix, iy] + z[ix, iy+1])
        zS = 0.5 * (z[ix, iy-1] + z[ix, iy])

        hN = max(0.0, ηC - zN)
        hS = max(0.0, ηC - zS)

        hsrc_y = 0.5 * (hN + hS)
        dzdy_face = (zN - zS) * _dy

        # ---------------------------------------------------------------------
        # Momentum updates first
        # ---------------------------------------------------------------------
        hu[ix, iy] -= dt * (
            dxb(F₂, ix, iy) * _dx +
            dyb(G₂, ix, iy) * _dy +
            g * hsrc_x * dzdx_face
        )

        hv[ix, iy] -= dt * (
            dxb(F₃, ix, iy) * _dx +
            dyb(G₃, ix, iy) * _dy +
            g * hsrc_y * dzdy_face
        )

        # ---------------------------------------------------------------------
        # Water-depth update
        # Since z is stationary, h_t = η_t
        # ---------------------------------------------------------------------
        h[ix, iy] -= (
            (F₁[ix, iy] * dtFx[ix, iy] -
            F₁[ix-1, iy] * dtFx[ix-1, iy]) * _dx
            +
            (G₁[ix, iy] * dtGy[ix, iy] -
            G₁[ix, iy-1] * dtGy[ix, iy-1]) * _dy
        )

    end

    return nothing
end

@parallel_indices (iy) function left_right_bc!(h, hu, hv, g, dt, _dx)
    nx, ny = size(h)

    # Left boundary (ix=1)
    cL = bc_speed_x(h, hu, 1, iy, g) * dt * _dx
    αL = (cL - 1) / (cL + 1)

    h1  = max(0.0, h[2, iy] + αL * (h[2, iy] - h[1, iy]))
    hu1 = hu[2, iy] + αL * (hu[2, iy] - hu[1, iy])
    hv1 = hv[2, iy] + αL * (hv[2, iy] - hv[1, iy])

    if h1 <= h_eps
        hu1 = 0.0
        hv1 = 0.0
    end

    cR = bc_speed_x(h, hu, nx, iy, g) * dt * _dx
    αR = (cR - 1) / (cR + 1)

    hR  = max(0.0, h[end-1, iy]  + αR * (h[end-1, iy]  - h[end, iy]))
    huR = hu[end-1, iy] + αR * (hu[end-1, iy] - hu[end, iy])
    hvR = hv[end-1, iy] + αR * (hv[end-1, iy] - hv[end, iy])

    if hR <= h_eps
        huR = 0.0
        hvR = 0.0
    end

    h[1, iy]    = h1
    hu[1, iy]   = hu1
    hv[1, iy]   = hv1

    h[end, iy]  = hR
    hu[end, iy] = huR
    hv[end, iy] = hvR
    return nothing
end

@parallel_indices (ix) function bottom_top_bc!(h, hu, hv, g, dt, _dy)
    nx, ny = size(h)

    # Bottom boundary (iy=1)
    cB = bc_speed_y(h, hv, ix, 1, g) * dt * _dy
    αB = (cB - 1) / (cB + 1)

    hB  = max(0.0, h[ix, 2]  + αB * (h[ix, 2]  - h[ix, 1]))
    huB = hu[ix, 2] + αB * (hu[ix, 2] - hu[ix, 1])
    hvB = hv[ix, 2] + αB * (hv[ix, 2] - hv[ix, 1])

    if hB <= h_eps
        huB = 0.0
        hvB = 0.0
    end

    cT = bc_speed_y(h, hv, ix, ny, g) * dt * _dy
    αT = (cT - 1) / (cT + 1)

    hT  = max(0.0, h[ix, end-1]  + αT * (h[ix, end-1]  - h[ix, end]))
    huT = hu[ix, end-1] + αT * (hu[ix, end-1] - hu[ix, end])
    hvT = hv[ix, end-1] + αT * (hv[ix, end-1] - hv[ix, end])

    if hT <= h_eps
        huT = 0.0
        hvT = 0.0
    end

    h[ix, 1]    = hB
    hu[ix, 1]   = huB
    hv[ix, 1]   = hvB

    h[ix, end]  = hT
    hu[ix, end] = huT
    hv[ix, end] = hvT
    return nothing
end

@parallel function sponge_layer!(hu, hv, σ)
    @all(hu) = @all(hu) * (1 - @all(σ))
    @all(hv) = @all(hv) * (1 - @all(σ))
    return nothing
end

@parallel_indices (ix, iy) function dry_cell_fix!(h, hu, hv, h_eps)
    nx, ny = size(h)

    if ix <= nx && iy <= ny
        if !isfinite(h[ix, iy]) || h[ix, iy] <= h_eps
            h[ix, iy]  = 0.0
            hu[ix, iy] = 0.0
            hv[ix, iy] = 0.0
        elseif !isfinite(hu[ix, iy]) || !isfinite(hv[ix, iy])
            hu[ix, iy] = 0.0
            hv[ix, iy] = 0.0
        end
    end

    return nothing
end

function benchmark_swe_loop!(nt, h, hu, hv, F₁, F₂, F₃, G₁, G₂, G₃, dtFx, dtGy, dt_drain, max_speed_x, max_speed_y, z, σ, g, vel_eps, _dx, _dy)
    for it in 1:nt
        @parallel compute_maxspeed!(max_speed_x, max_speed_y, h, hu, hv, z, g, vel_eps)
        dt = 0.99 / (maximum(max_speed_x) * _dx + maximum(max_speed_y) * _dy)
        
        @parallel compute_1st_2nd_and_3th_flux!(F₁, F₂, F₃, G₁, G₂, G₃, hu, hv, h, z, g, max_speed_x, max_speed_y, vel_eps)
        @parallel compute_draining_timestep!(dt_drain, F₁, G₁, h, dt, _dx, _dy)
        @parallel compute_effective_flux_timesteps!(dtFx, dtGy, dt_drain, F₁, G₁, dt)
        @parallel update_height_momentum!(h, hu, hv, F₁, G₁, F₂, F₃, G₂, G₃, dtFx, dtGy, z, g, dt, _dx, _dy)
        @parallel dry_cell_fix!(h, hu, hv, 1e-2)
        @parallel left_right_bc!(h, hu, hv, g, dt, _dx)
        @parallel bottom_top_bc!(h, hu, hv, g, dt, _dy)
        @parallel sponge_layer!(hu, hv, σ)
        @parallel dry_cell_fix!(h, hu, hv, 1e-2)
    end
    
    @static if USE_GPU
        CUDA.synchronize()
    end
    return nothing
end



# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
@views function swe2d_topography_frames(; nt=0, nx_aoi=125, ny_aoi=125, domain_expansion_factor=1, outdir = "frames", do_viz = true, force_array_output=false, perf_test=false, debug_roi=false)
    # physics and numerics
    lx_aoi = 154 * 50 # aoi = area of interest
    ly_aoi = 124 * 50 

    lx = domain_expansion_factor * lx_aoi
    ly = domain_expansion_factor * ly_aoi
    nx = round(Int, domain_expansion_factor * nx_aoi)
    ny = round(Int, domain_expansion_factor * ny_aoi)

    if nt == 0
        nt = Int(nt_nx_multiplier * nx_aoi)
    end

    nvis = 5

    dx = lx / (nx - 1)
    dy = ly / (ny - 1)

    vel_eps = min(dx, dy)^4

    _dx  = 1.0 / dx
    _dy  = 1.0 / dy
    
    xs = LinRange(-lx / 2, lx / 2, nx)
    ys = LinRange(-ly / 2, ly / 2, ny)

    # ROI indices for visualization
    pad_x = round(Int, (nx - nx_aoi) / 2)
    pad_y = round(Int, (ny - ny_aoi) / 2)

    if debug_roi
    ix_roi = 1:nx
    iy_roi = 1:ny
    else
        ix_roi = (pad_x + 1):(pad_x + nx_aoi)
        iy_roi = (pad_y + 1):(pad_y + ny_aoi)
    end

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

    #-------------------------------------------------------------------------
    # Initial condition topography + wave data
    #-------------------------------------------------------------------------
    
    z_cpu  = zeros(nx, ny)
    η0_cpu = zeros(nx, ny)

    # Physics parameters for performance test
    h_base  = 15.0   # Base water depth
    A_spike = 5.0    # Height of the Gaussian bump
    σ_spike = 0.5 # Width of the bump
    for i in 1:nx
        for j in 1:ny
            x = xs[i]
            y = ys[j]
            
            # Gaussian bump centered at global (0,0) + bumps at each local center 
            η0_cpu[i, j] = h_base + A_spike * exp(-(x^2 + y^2) / (2 * σ_spike^2))
        end
    end

    z  = Data.Array(z_cpu)
    η0 = Data.Array(η0_cpu)

    hmin  = 1e-2
    h .= max.(0.0, η0 .- z)

    dt_drain = @zeros(nx, ny)

    dtFx = @zeros(nx - 1, ny)
    dtGy = @zeros(nx, ny - 1)

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

    # # -------------------------------------------------------------------------
    # # initial condition
    # # -------------------------------------------------------------------------

    time = 0.0

    # -------------------------------------------------------------------------
    # main loop
    # -------------------------------------------------------------------------

    # Setup the benchmark function
    println("--- Performance Analysis ($nx x $ny grid, $nt iterations) ---")
        
    # Warmup
    benchmark_swe_loop!(10, h, hu, hv, F₁, F₂, F₃, G₁, G₂, G₃, dtFx, dtGy, dt_drain, max_speed_x, max_speed_y, z, σ, g, vel_eps, _dx, _dy)
    
    # Measure execution time
    println("Benchmarking main loop...")
    t_total = @belapsed benchmark_swe_loop!($nt, $h, $hu, $hv, $F₁, $F₂, $F₃, $G₁, $G₂, $G₃, $dtFx, $dtGy, $dt_drain, $max_speed_x, $max_speed_y, $z, $σ, $g, $vel_eps, $_dx, $_dy)
    
    # Calculate metrics
    t_it = t_total / nt  # Time per iteration
    
    # A_eff = (2 * D_u + 1 * D_k) * nx * ny * sizeof(Float64) / 1e9 (to get GB)
    # D_u = h, hu, hv, F1-3, G1-3, dtFx, dtGy, dt_drain, max_speed_x, max_speed_y
    # D_k = z, 
    A_eff = (2 * 14 + 1) * (nx * ny) * sizeof(Float64) / 1e9 
    
    T_eff = A_eff / t_it
    
    println("Avg Time per iteration (t_it): ", round(t_it, digits=5), " s")
    println("Effective Memory Access (A_eff): ", round(A_eff, digits=6), " GB")
    println("Effective Memory Throughput (T_eff): ", round(T_eff, digits=2), " GB/s")
    
    T_peak = 4000.0 # GH200
    println("Percentage of Peak: ", round((T_eff / T_peak) * 100, digits=2), "%")

    return nothing
end

for dim in [100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000]

    swe2d_topography_frames(;
        outdir = "docs/frames/frames_topography",
        do_viz = true,
        force_array_output = true,
        nt = 50, nx_aoi=dim, ny_aoi=dim
    )
end


# # Performance test:
# for n in [100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
#     for trial in 1:3
#         println("\nRunning performance test at resolution: $n x $n (trial $trial)")
#         println("nt: ", Int(2000))
#         @time swe2d_topography_frames(nt=2000, nx_aoi=n, ny_aoi=n, domain_expansion_factor=1, 
#                                     do_viz=false, force_array_output=true, perf_test=true, debug_roi=false)
#     end
# end

# # error benchmark

# resolutions = [25, 50, 125, 250, 500]
# errors = Float64[]

# for res in resolutions
#     println("\nRunning simulation at resolution: $res x $res")
#     err = swe2d_topography_frames(res, res; outdir = "frames_topography_$(res)x$(res)", do_viz = false)
#     push!(errors, err)
# end

# # Plot errors
# using Plots

# Plots.plot(resolutions, errors, marker=:o, xscale=:log10, yscale=:log10,
#     xlabel="Resolution (nx = ny)", ylabel="Steady-state L∞ Error",
#     title="Error vs Resolution for Swe2D fully-wet topography test case",
#     legend=false)
# savefig("docs/error_convergence_full_wet.png")