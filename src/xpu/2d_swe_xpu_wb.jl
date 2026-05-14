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
const nt_nx_multiplier = 2

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



# @parallel_indices (ix, iy) function update_height_momentum!(h, hu, hv, F₁, G₁, F₂, F₃, G₂, G₃, dzdx, dzdy, g, dt, _dx, _dy)
#     nx, ny = size(h)
#     if (2 <= ix <= nx-1 && 2 <= iy <= ny-1)
#         hu[ix, iy] -= dt * (dxb(F₂, ix, iy) * _dx + dyb(G₂, ix, iy) * _dy + g * h[ix, iy] * dzdx[ix, iy])
#         hv[ix, iy] -= dt * (dxb(F₃, ix, iy) * _dx + dyb(G₃, ix, iy) * _dy + g * h[ix, iy] * dzdy[ix, iy])
#         h[ix, iy] -= dt * (dxb(F₁, ix, iy) * _dx + dyb(G₁, ix, iy) * _dy)
#     end
#     return nothing
# end

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


@parallel_indices (ix, iy) function all_bc!(h, hu, hv, g, dt, _dx, _dy)
    nx, ny = size(h)

    # Left boundary (ix=1)
    if ix == 1 && iy <= ny
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
    end

    # Right boundary (ix=nx)
    if ix == nx && iy <= ny
    
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
    end

    # Bottom boundary (iy=1)
    if iy == 1 && ix <= nx
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

    end

    # Top boundary (iy=ny)
    if iy == ny && ix <= nx

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
    end
    return nothing
end

@parallel function sponge_layer!(hu, hv, σ)
    @all(hu) = @all(hu) * (1 - @all(σ))
    @all(hv) = @all(hv) * (1 - @all(σ))
    return nothing
end

# @parallel_indices (ix, iy) function dry_cell_fix!(h, hu, hv, h_eps)
#     nx, ny = size(h)

#     if ix <= nx && iy <= ny
#         if h[ix, iy] < h_eps
#             h[ix, iy]  = 0.0
#             hu[ix, iy] = 0.0
#             hv[ix, iy] = 0.0
#         end
#     end

#     return nothing
# end

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

@parallel_indices (ix_out, iy_out) function downsample_kernel!(out_arr, in_arr, start_x, step_x, start_y, step_y)
    nx_out, ny_out = size(out_arr)
    
    if ix_out <= nx_out && iy_out <= ny_out
        ix_in = start_x + (ix_out - 1) * step_x
        iy_in = start_y + (iy_out - 1) * step_y
        out_arr[ix_out, iy_out] = in_arr[ix_in, iy_in]
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
            # η0_expanded[i, j] = in_roi ? η0_inner[i - pad_x, j - pad_y] : 0.0

            # Alternatively, stretch the wave data outwards -> Huge watermass just like in a tsunami scenario
            η0_expanded[i, j] = η0_inner[orig_i, orig_j]
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

@views function swe2d_topography_frames(nx_aoi, ny_aoi; outdir = "frames", do_viz = true, force_array_output=false, debug_roi=false, max_output_res="HD")
    # physics and numerics
    lx_aoi = 50.0 # aoi = area of interest
    ly_aoi = 50.0

    # Multiply domain size to allow for sponge layer and BCs
    domain_expansion_factor = 3

    # Determine max output res
    if max_output_res == "SD"
        max_output_res_x = 640
        max_output_res_y = 480
    elseif max_output_res == "HD"
        max_output_res_x = 1280
        max_output_res_y = 720
    elseif max_output_res == "FULL HD"
        max_output_res_x = 1920
        max_output_res_y = 1080
    elseif max_output_res == "2K"
        max_output_res_x = 2560
        max_output_res_y = 1440
    elseif max_output_res == "4K"
        max_output_res_x = 3840
        max_output_res_y = 2160
    elseif max_output_res == "8K"
        max_output_res_x = 7680
        max_output_res_y = 4320
    else
        error("Unsupported max_output_res: $max_output_res. Use 'HD' or '4K'.")
    end

    lx = domain_expansion_factor * lx_aoi
    ly = domain_expansion_factor * ly_aoi
    nx = round(Int, domain_expansion_factor * nx_aoi)
    ny = round(Int, domain_expansion_factor * ny_aoi)

    nt   = Int(nt_nx_multiplier * nx_aoi)
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

    xs_roi = xs[ix_roi]
    ys_roi = ys[iy_roi]

    # Calculate the striding steps to hit target resolution
    step_x = max(1, floor(Int, length(ix_roi) / max_output_res_x))
    step_y = max(1, floor(Int, length(iy_roi) / max_output_res_y))

    # Create the strided ranges
    ix_roi_low = ix_roi[1:step_x:end]
    iy_roi_low = iy_roi[1:step_y:end]
    nx_low = length(ix_roi_low)
    ny_low = length(iy_roi_low)
    start_x = first(ix_roi_low)
    start_y = first(iy_roi_low)

    # Allocate low-res arrays for visualization output
    h_low_gpu = @zeros(nx_low, ny_low)
    z_low_gpu = @zeros(nx_low, ny_low)

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


    # z = build_topography(xs, ys; islands=[], background= (xs, ys) -> background_bumps(xs, ys, seed=43))


    # -------------------------------------------------------------------------
    # Thacker's Bowl with Gaussian Spike
    # -------------------------------------------------------------------------
    # h0 = 0.1    # Water depth at center
    # a  = 10.0   # Distance to zero elevation
    # B  = 0.05   # Amplitude of the slosh
    # ω  = sqrt(2 * g * h0) / a # Angular frequency

    # Gaussian Spike Parameters
    # A_spike = 0.3   # Amplitude of the drop/spike
    # x_c     = 3.0    # X center of the spike
    # y_c     = 3.0    # Y center of the spike
    # σ_spike = 1.5    # Width of the spike (standard deviation)

    # # Parabolic Topography
    # z = [h0 * ((x^2 + y^2) / a^2) for x in xs, y in ys]

    # # Tilted Planar Free Surface + Gaussian Spike
    # η0 = [h0 - h0 * ((x^2 + y^2) / a^2) + B * x + 
    #       A_spike * exp(-((x - x_c)^2 + (y - y_c)^2) / (2 * σ_spike^2)) 
    #       for x in xs, y in ys]

    #-------------------------------------------------------------------------
    # Initial condition from real topography + wave data
    #-------------------------------------------------------------------------
    

    z, η0 = load_topography_data(domain_expansion_factor, nx_aoi, ny_aoi)

    # wet/dry sea level steady state
    # η0 .= 0

    # add a gaussian bump to the initial condition to generate some wave activity
    x_c     = -10.0    # X center of the spike
    y_c     = -20.0    # Y center of the spike
    σ_spike = 2.5    # Width of the spike (standard deviation)
    A_spike = 30.0    # Amplitude of the drop/spike
    for i in eachindex(xs), j in eachindex(ys)
        x = xs[i]
        y = ys[j]
        η0[i, j] += A_spike * exp(-((x - x_c)^2 + (y - y_c)^2) / (2 * σ_spike^2))
    end


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

    # h_in  = 0.20
    # h_out = 0.10
    # r0    = 2.5
    # hmin  = 1e-6

    # η0 = [((x^2 + y^2) < r0^2) ? h_in : h_out for x in xs, y in ys]
    # h .= max.(hmin, η0 .- z)

    # -------------------------------------------------------------------------
    # initial condition
    # -------------------------------------------------------------------------

    # h_out = 0.10          # background free-surface level
    # a     = 0.08          # wave amplitude
    # y0    = 20.0          # crest location
    # σy    = 3.0           # wave width
    # hmin  = 1e-6

    # # free surface eta(x,y) = eta(y), homogeneous in x
    # # η0 = [h_out + a * exp(-((y - y0)^2) / (2 * σy^2)) for x in xs, y in ys]
    # η0 = h_out

    # # water depth
    # h .= max.(hmin, η0 .- z)
    

    # -------------------------------------------------------------------------
    # visualization
    # -------------------------------------------------------------------------

    if do_viz
        use_makie = HAS_MAKIE && !force_array_output
        print("Using visualization: ", use_makie ? "Makie" : "Array output")
        mkpath(outdir)

        @parallel downsample_kernel!(h_low_gpu, h, start_x, step_x, start_y, step_y)
        @parallel downsample_kernel!(z_low_gpu, z, start_x, step_x, start_y, step_y)
        h_slice = Array(h_low_gpu)
        z_slice = Array(z_low_gpu)

        if use_makie
            vertical_exaggeration = 6.0
            hmin_plot = 1e-3
            
            z_plot = vertical_exaggeration .* z_slice

            # terrain color as full matrix, not a single Symbol
            terrain_color = fill(RGBf(0.82, 0.82, 0.82), size(z_plot))

            η_water_plot0  = vertical_exaggeration .* (h_slice .+ z_slice)
            η_water_color0 = h_slice .+ z_slice

            η_water_plot0[h_slice .<= hmin_plot]  .= NaN
            η_water_color0[h_slice .<= hmin_plot] .= NaN

            η_water_plot  = Observable(η_water_plot0)
            η_water_color = Observable(η_water_color0)

            fig = Figure(size = (max_output_res_x, max_output_res_y))
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
                serialize(fname, (h=convert.(Float32, h_slice), z=convert.(Float32, z_slice)))
            end
            save_array!()
        end
    end

    # -------------------------------------------------------------------------
    # main loop
    # -------------------------------------------------------------------------

    for it in 1:nt
        @parallel compute_maxspeed!(max_speed_x, max_speed_y, h, hu, hv, z, g, vel_eps)


        dt =  0.99 / (maximum(max_speed_x) * _dx + maximum(max_speed_y) * _dy)
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

        @parallel compute_effective_flux_timesteps!(
            dtFx, dtGy,
            dt_drain,
            F₁, G₁,
            dt
        )


        @parallel update_height_momentum!(
            h, hu, hv,
            F₁, G₁, F₂, F₃, G₂, G₃, dtFx, dtGy,
            z, g, dt, _dx, _dy
        )

        @parallel dry_cell_fix!(h, hu, hv, hmin)

        @parallel all_bc!(h, hu, hv, g, dt, _dx, _dy)

        @parallel sponge_layer!(hu, hv, σ)
        @parallel dry_cell_fix!(h, hu, hv, hmin)

        if it % nvis == 0
            if do_viz
                @parallel downsample_kernel!(h_low_gpu, h, start_x, step_x, start_y, step_y)
                @parallel downsample_kernel!(z_low_gpu, z, start_x, step_x, start_y, step_y)
                h_slice = Array(h_low_gpu)
                z_slice = Array(z_low_gpu)

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

        percent = 100 * it / nt
        print("\rProgress: $(round(percent, digits=1)) %")
        flush(stdout)
    end

    # -------------------------------------------------------------------------
    # Steady-state error on wet cells onlyabsorbing boundary conditions for the numerical simulation of waves
    # -------------------------------------------------------------------------

    η = h .+ z

    # Compare against the initial free surface η0
    err = abs.(η .- η0)

    # Wet-cell mask:
    # use cells that were initially wet and are still meaningfully wet
    wet_mask = (η0 .- z .> h_eps) .& (h .> h_eps)

    nwet = sum(wet_mask)

    if nwet > 0
        Linf_abs = maximum(err[wet_mask])

        # A sensible relative L∞ error:
        # normalize by the largest initial free-surface magnitude on wet cells
        η0_scale = maximum(abs.(η0[wet_mask]))

        Linf_rel = η0_scale > 0 ? Linf_abs / η0_scale : Linf_abs

        println("wet cells used: ", nwet)
        println("steady-state L∞ absolute error on wet cells: ", Linf_abs)
        println("steady-state L∞ relative error on wet cells: ", Linf_rel)
    else
        println("No wet cells found for steady-state error evaluation.")
        Linf_abs = NaN
        Linf_rel = NaN
    end
    # print time
    println("Total simulation time: $(round(time, digits=2)) seconds")

    if do_viz
        println("\nSaved $(frame_id[]) frames to: $(abspath(outdir))")
    end
    return Linf_abs
end

swe2d_topography_frames(250, 250;
    outdir = "docs/frames/frames_topography",
    do_viz = true,
    force_array_output = true
)

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