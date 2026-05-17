using Serialization

const HAS_MAKIE = try
    @eval using GLMakie
    true
catch
    @info "GLMakie not found. Falling back to array output."
    false
end

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

nt_nx_multiplier = 2 # no const on purpose
const h_eps = 1e-2

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

min_g(A) = (min_l = minimum(A); MPI.Allreduce(min_l, MPI.MIN, MPI.COMM_WORLD))
max_g(A) = (max_l = maximum(A); MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD))

# -----------------------------------------------------------------------------
# Kernels
# -----------------------------------------------------------------------------

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

@parallel_indices (iy) function left_bc!(h, hu, hv, g, dt, _dx)
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

    h[1, iy]    = h1
    hu[1, iy]   = hu1
    hv[1, iy]   = hv1

    return nothing
end

@parallel_indices (iy) function right_bc!(h, hu, hv, g, dt, _dx)
    nx, ny = size(h)

    # Right boundary (ix=nx)
    cR = bc_speed_x(h, hu, nx, iy, g) * dt * _dx
    αR = (cR - 1) / (cR + 1)

    hR  = max(0.0, h[end-1, iy]  + αR * (h[end-1, iy]  - h[end, iy]))
    huR = hu[end-1, iy] + αR * (hu[end-1, iy] - hu[end, iy])
    hvR = hv[end-1, iy] + αR * (hv[end-1, iy] - hv[end, iy])

    if hR <= h_eps
        huR = 0.0
        hvR = 0.0
    end

    h[end, iy]  = hR
    hu[end, iy] = huR
    hv[end, iy] = hvR
    return nothing
end

@parallel_indices (ix) function bottom_bc!(h, hu, hv, g, dt, _dy)
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

    h[ix, 1]    = hB
    hu[ix, 1]   = huB
    hv[ix, 1]   = hvB
    return nothing
end

@parallel_indices (ix) function top_bc!(h, hu, hv, g, dt, _dy)
    nx, ny = size(h)

    # Top boundary (iy=ny)
    cT = bc_speed_y(h, hv, ix, ny, g) * dt * _dy
    αT = (cT - 1) / (cT + 1)

    hT  = max(0.0, h[ix, end-1]  + αT * (h[ix, end-1]  - h[ix, end]))
    huT = hu[ix, end-1] + αT * (hu[ix, end-1] - hu[ix, end])
    hvT = hv[ix, end-1] + αT * (hv[ix, end-1] - hv[ix, end])

    if hT <= h_eps
        huT = 0.0
        hvT = 0.0
    end

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

    X = Data.Array(X_cpu)
    Y = Data.Array(Y_cpu)

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

function load_topography_data(nx_aoi_ext, ny_aoi_ext, nx_local, ny_local, me, coords, comm_cart)
    # Use the true global dimensions dictated by ImplicitGlobalGrid
    NX = nx_g()
    NY = ny_g()

    # Allocate full global arrays on ALL ranks (Required for MPI.Bcast!)
    z_global  = zeros(NX, NY)
    η0_global = zeros(NX, NY)

    # Only Rank 0 builds the full global data
    if me == 0
        base_file = "data/tsunamiOku/D112-94-50m.txt"
        wave_file = "data/tsunamiOku/I112-94-50m-17a.txt"

        nx_aoi, ny_aoi = 112, 94 # ENSURE THIS MATCHES THE DATA FILES

        # Read as string, split by whitespace, and parse to Float64
        read_values(filename) = parse.(Float64, split(read(filename, String)))

        z_vec = read_values(base_file)
        η0_vec = read_values(wave_file)

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

        # Center the interpolated AOI inside the True Global Domain
        pad_x = round(Int, (NX - nx_aoi_ext) / 2)
        pad_y = round(Int, (NY - ny_aoi_ext) / 2)

        # Pad the arrays
        for i in 1:NX
            for j in 1:NY
                # Clamp to the closest valid index of the inner grid
                orig_i = clamp(i - pad_x, 1, nx_aoi_ext)
                orig_j = clamp(j - pad_y, 1, ny_aoi_ext)
                
                # Stretch the edge bathymetry outwards
                z_global[i, j] = z_inner[orig_i, orig_j]
                
                # Stretch the wave data outwards 
                η0_global[i, j] = η0_inner[orig_i, orig_j]
            end
        end
    end

    # Broadcast the completed global arrays from Rank 0 to all other ranks
    MPI.Bcast!(z_global, 0, comm_cart)
    MPI.Bcast!(η0_global, 0, comm_cart)

    # Each rank mathematically extracts its local slice (including halos!)
    z_local = zeros(nx_local, ny_local)
    η0_local = zeros(nx_local, ny_local)

    for ix in 1:nx_local
        for iy in 1:ny_local
            # Map local index to global index based on MPI Cartesian coordinates
            IX = coords[1] * (nx_local - 2) + ix
            IY = coords[2] * (ny_local - 2) + iy
            
            z_local[ix, iy] = z_global[IX, IY]
            η0_local[ix, iy] = η0_global[IX, IY]
        end
    end

    # Cast to ParallelStencil arrays
    return Data.Array(z_local), Data.Array(η0_local)
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@views function swe2d_topography_frames(nx_aoi, ny_aoi;
            nt=0,
            outdir = "frames", 
            do_viz = true, 
            force_array_output = false, 
            debug_roi=false,
            print_error_metrics=true,
            gpu_test_memory_restriction_workound=false,
            domain_expansion_factor=3.0)
    # physics and numerics
    lx_aoi = 50.0 # aoi = area of interest
    ly_aoi = 50.0

    # Multiply domain size to allow for sponge layer and BCs
    if gpu_test_memory_restriction_workound
        domain_expansion_factor = 1.0
    end
    lx = domain_expansion_factor * lx_aoi
    ly = domain_expansion_factor * ly_aoi
    nx = round(Int, domain_expansion_factor * nx_aoi)
    ny = round(Int, domain_expansion_factor * ny_aoi)

    # Define desired global size
    nx_global = round(Int, domain_expansion_factor * nx_aoi)
    ny_global = round(Int, domain_expansion_factor * ny_aoi)

    println("Global domain size (including halos): ", nx_global, " x ", ny_global)

    # get num ranks
    if !MPI.Initialized()
        MPI.Init()
    end
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)

    # Get ideal 2D topology
    dims_mpi = [0, 0]
    MPI.Dims_create!(nprocs, dims_mpi)

    println("MPI topology: ", dims_mpi[1], " x ", dims_mpi[2])

    # Compute local chunk sizes (+2 for halos)
    nx = round(Int, (nx_global - 2) / dims_mpi[1]) + 2
    ny = round(Int, (ny_global - 2) / dims_mpi[2]) + 2

    # Init global grid and get local grid info
    me, dims, nprocs, coords, comm_cart = init_global_grid(
        nx, ny, 1; 
        init_MPI = false, 
        select_device = false
    )

    neighbors_x = MPI.Cart_shift(comm_cart, 0, 1) 
    neighbors_y = MPI.Cart_shift(comm_cart, 1, 1)
    
    b_width     = (8, 8, 0)

    if nt==0
        nt   = Int(nt_nx_multiplier * nx_aoi)
    end

    println("Local domain size (including halos): ", nx, " x ", ny)
    println("Time steps: ", nt)

    nvis = 5

    dx = lx / (nx_g() - 1)
    dy = ly / (ny_g() - 1)

    vel_eps = min(dx, dy)^4

    _dx  = 1.0 / dx
    _dy  = 1.0 / dy

    # ROI indices for visualization
    pad_x = round(Int, (nx_g() - nx_aoi) / 2)
    pad_y = round(Int, (ny_g() - ny_aoi) / 2)

    if debug_roi
    ix_roi = 1:nx_g()
    iy_roi = 1:ny_g()
    else
        ix_roi = (pad_x + 1):(pad_x + nx_aoi)
        iy_roi = (pad_y + 1):(pad_y + ny_aoi)
    end
    
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

    ## DEFAULT TOPOGRAPHY: Load from file and interpolate to the grid
    if !gpu_test_memory_restriction_workound
        z, η0 = load_topography_data(nx_aoi, ny_aoi, nx, ny, me, coords, comm_cart)
    else
        # -------------------------------------------------------------------------
        # topography (Analytical for Scaling Study)
        # -------------------------------------------------------------------------
    
        z_local  = zeros(nx, ny)
        η0_local = zeros(nx, ny)

        # Physics parameters for the analytical test
        h_base  = 15.0   # Base water depth
        A_spike = 5.0    # Height of the Gaussian bump
        σ_spike = max(lx, ly) / 100 # Width of the bump (scales with domain)

        for i in 1:nx
            for j in 1:ny
                x = xs[i]
                y = ys[j]
                
                # Flat bathymetry
                z_local[i, j] = 0.0
                
                # Gaussian bump centered at global (0,0) + bumps at each local center 
                η0_local[i, j] = h_base + A_spike * exp(-(x^2 + y^2) / (2 * σ_spike^2)) +
                                        A_spike * exp(-(((x - xs[div(nx,2)])^2 + (y - ys[div(ny,2)])^2)) / (2 * σ_spike^2))
            end
        end

        # Cast to ParallelStencil arrays (GPU or CPU depending on @init_parallel_stencil)
        z  = Data.Array(z_local)
        η0 = Data.Array(η0_local)

    end

    # wet/dry sea level steady state
    # η0 .= 0

    # # add a gaussian bump to the initial condition to generate some wave activity
    # x_c     = -10.0    # X center of the spike
    # y_c     = -20.0    # Y center of the spike
    # σ_spike = 2.5    # Width of the spike (standard deviation)
    # A_spike = 30.0    # Amplitude of the drop/spike
    # for i in eachindex(xs), j in eachindex(ys)
    #     x = xs[i]
    #     y = ys[j]
    #     η0[i, j] += A_spike * exp(-((x - x_c)^2 + (y - y_c)^2) / (2 * σ_spike^2))
    # end

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
    
    time = 0.0

    # h_in  = 0.20
    # h_out = 0.10
    # r0    = 2.5
    # hmin  = 1e-6

    # η0 = [((x^2 + y^2) < r0^2) ? h_in : h_out for x in xs, y in ys]
    # h .= max.(hmin, η0 .- z)


    # -------------------------------------------------------------------------
    # visualization
    # -------------------------------------------------------------------------

    if do_viz
        use_makie = HAS_MAKIE && !force_array_output
        print("Using visualization: ", use_makie ? "Makie" : "Array output")
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

            if use_makie

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
                    colorrange = (-10, 20),
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
            else
                frame_id = Ref(0)
                @info "Saving arrays to $outdir"
                function save_array!()
                    frame_id[] += 1
                    # Save as a standard Julia serialized file
                    fname = joinpath(outdir, @sprintf("array_frame_%06d.jls", frame_id[]))
                    # Storing a NamedTuple containing the ROI arrays
                    serialize(fname, (h=Array(convert.(Float32, h_slice)),)) 
                end
                function save_array_with_z!()
                    frame_id[] += 1
                    # Save as a standard Julia serialized file
                    fname = joinpath(outdir, @sprintf("array_frame_%06d.jls", frame_id[]))
                    # Storing a NamedTuple containing the ROI arrays
                    serialize(fname, (h=Array(convert.(Float32, h_slice)), z=Array(convert.(Float32, z_slice))))
                end
                save_array_with_z!()
            end
        end
    end

    # -------------------------------------------------------------------------
    # main loop
    # -------------------------------------------------------------------------

    for it in 1:nt
        @parallel compute_maxspeed!(max_speed_x, max_speed_y, h, hu, hv, z, g, vel_eps)

        dt =  0.99 / (max_g(max_speed_x) * _dx + max_g(max_speed_y) * _dy)
        time += dt

        if !isfinite(dt)
            error("Non-finite dt at iteration $it: dt=$dt, max_sx=$(maximum(max_speed_x)), max_sy=$(maximum(max_speed_y))")
        end

        @parallel compute_1st_2nd_and_3th_flux!(
            F₁, F₂, F₃,
            G₁, G₂, G₃,
            hu, hv, h, z, g,
            max_speed_x, max_speed_y, vel_eps
        )

        @parallel compute_draining_timestep!(
            dt_drain,
            F₁, G₁,
            h,
            dt,
            _dx, _dy
        )
        update_halo!(dt_drain)

        @parallel compute_effective_flux_timesteps!(
            dtFx, dtGy,
            dt_drain,
            F₁, G₁,
            dt
        )
        
        @hide_communication b_width begin
            @parallel update_height_momentum!(
                h, hu, hv,
                F₁, G₁, F₂, F₃, G₂, G₃, dtFx, dtGy,
                z, g, dt, _dx, _dy
            )           
            update_halo!(h, hu, hv)
        end

        @parallel dry_cell_fix!(h, hu, hv, hmin)

        # Apply BCs only on ranks that border the global domain boundaries
        if neighbors_x[1] == MPI.PROC_NULL
            @parallel (1:ny) left_bc!(h, hu, hv, g, dt, _dx)
        end
        if neighbors_x[2] == MPI.PROC_NULL
            @parallel (1:ny) right_bc!(h, hu, hv, g, dt, _dx)
        end
        if neighbors_y[1] == MPI.PROC_NULL
            @parallel (1:nx) bottom_bc!(h, hu, hv, g, dt, _dy)
        end
        if neighbors_y[2] == MPI.PROC_NULL
            @parallel (1:nx) top_bc!(h, hu, hv, g, dt, _dy)
        end
        
        # Would need to only apply sponge layer on ranks that border the global domain boundaries
        # As multiGPU runs super fast we can affort to simply make the domain huge 
        # such that the sponge layer is not required
        # @parallel sponge_layer!(hu, hv, σ)
        
        @parallel dry_cell_fix!(h, hu, hv, hmin)

        if do_viz && it % nvis == 0

            h_inn .= Array(h)[2:end-1, 2:end-1]; gather!(h_inn, h_v)
            z_inn .= Array(z)[2:end-1, 2:end-1]; gather!(z_inn, z_v)
            
            if me == 0
                h_slice = h_v[ix_roi_v, iy_roi_v]
                z_slice = z_v[ix_roi_v, iy_roi_v]
                
                if use_makie
                    ηtmp_plot  = vertical_exaggeration .* (h_slice .+ z_slice)
                    ηtmp_color = h_slice .+ z_slice

                    ηtmp_plot[h_slice .<= hmin_plot]  .= NaN
                    ηtmp_color[h_slice .<= hmin_plot] .= NaN

                    η_water_plot[]  = ηtmp_plot
                    η_water_color[] = ηtmp_color

                    save_frame!()
                else
                    save_array!()
                end
            end
        end
        
        if me == 0 && !gpu_test_memory_restriction_workound
            percent = 100 * it / nt
            print("\rProgress: $(round(percent, digits=1)) %")
            flush(stdout)
        end
    end
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------


    η = h .+ z

    # Compare against the initial free surface η0
    err = abs.(η .- η0)

    # Wet-cell mask:
    # use cells that were initially wet and are still meaningfully wet
    wet_mask = (η0 .- z .> h_eps) .& (h .> h_eps)

    local_nwet = sum(wet_mask)
    nwet = MPI.Allreduce(local_nwet, MPI.SUM, comm_cart)
    
    local_err_max = local_nwet > 0 ? maximum(err[wet_mask]) : 0.0
    local_scale_max = local_nwet > 0 ? maximum(abs.(η0[wet_mask])) : 0.0

    Linf_abs = MPI.Allreduce(local_err_max, MPI.MAX, comm_cart)
    η0_scale = MPI.Allreduce(local_scale_max, MPI.MAX, comm_cart)

    if me == 0 && print_error_metrics
        if nwet > 0
            # A sensible relative L∞ error:
            # normalize by the largest initial free-surface magnitude on wet cells
            Linf_rel = η0_scale > 0 ? Linf_abs / η0_scale : Linf_abs

            println("wet cells used: ", nwet)
            println("steady-state L∞ absolute error on wet cells: ", Linf_abs)
            println("steady-state L∞ relative error on wet cells: ", Linf_rel)
        else
            println("No wet cells found for steady-state error evaluation.")
            Linf_abs = NaN
            Linf_rel = NaN
        end
    end
    if me == 0
        # print time
        println("\nSimulation completed.")
        println("Total simulation time: $(round(time, digits=2)) seconds")

        if do_viz
            println("\nSaved $(frame_id[]) frames to: $(abspath(outdir))")
        end
    end

    finalize_global_grid()
    return nothing
end

input_nx = 125
input_ny = 125
num_repetitions = 1

for i in 1:length(ARGS)
    if ARGS[i] == "--nx"
        global input_nx = parse(Int, ARGS[i+1])
    elseif ARGS[i] == "--ny"
        global input_ny = parse(Int, ARGS[i+1])
    elseif ARGS[i] == "--num_repetitions"
        global num_repetitions = parse(Int, ARGS[i+1])
    elseif ARGS[i] == "--dt_multiplier"
        global nt_nx_multiplier = parse(Float64, ARGS[i+1])
    end
end

for rep in 1:num_repetitions
    println(" ------------------------------- ")
    println("Repetition $rep of $num_repetitions")
    @time swe2d_topography_frames(input_nx, input_ny; nt=2000,
        outdir = "docs/frames/frames_topography_multi",
        do_viz = true,
        force_array_output = false,
        print_error_metrics = false,
        gpu_test_memory_restriction_workound = true,
        domain_expansion_factor = 3.0
    )
end