using Serialization
using GLMakie

GLMakie.activate!()

# =======================================================
# 1. PATHS AND FILE INPUT
# =======================================================

script_dir = @__DIR__

const FRAME_DIR = normpath(joinpath(script_dir, "..", "..", "docs", "frames", "frames_topography"))
const OUT_DIR   = normpath(joinpath(script_dir, "..", "..", "docs", "multi_xpu_ray"))

mkpath(OUT_DIR)

# Matches:
# array_frame_000001.jls
# array_frame_000002.jls
# ...
const FRAME_REGEX = r"array_frame_\d+\.jls$"


# =======================================================
# 2. GENERAL RENDERING KNOBS
# =======================================================

# Render every kth grid cell:
# 1 = full resolution
# 2 = faster, half resolution
# 4 = much faster, coarser
const STRIDE = 1

# Vertical visual scale:
# Increase if the water/terrain appears too flat.
# This scales both terrain and water in the rendered scene.
const HEIGHT_SCALE = 0.35f0


# =======================================================
# 3. CAMERA KNOBS
# =======================================================

CAMERA_AZIMUTH = 0.70f0 * π + π / 2
const CAMERA_ELEVATION = 0.11f0 * π
const CAMERA_PERSPECTIVE = 1.0f0

# Visual vertical compression of the 3D plot.
# Smaller third value = flatter scene.
const AXIS_ASPECT = (1, 1, 0.20)

# Background sky color
const SKY_COLOR = RGBf(0.78, 0.88, 1.0)


# =======================================================
# 4. TERRAIN COLOR KNOBS
# =======================================================

# Terrain colors from low elevation to high elevation.
# Higher terrain becomes brighter.
const TERRAIN_LOW  = RGBf(0.48, 0.33, 0.18)
const TERRAIN_MID1 = RGBf(0.72, 0.52, 0.30)
const TERRAIN_MID2 = RGBf(0.90, 0.74, 0.48)
const TERRAIN_HIGH = RGBf(1.00, 0.96, 0.82)

# =======================================================
# 5. WATER TRANSPARENCY AND HIGHLIGHT KNOBS
# =======================================================

# Water opacity:
# shallow water = more transparent
# deep water    = more opaque
WATER_ALPHA_SHALLOW = 0.25f0
WATER_ALPHA_DEEP    = 0.75f0

# Subtle slope-based highlight.
# Increase if you want the water surface to shimmer more.
const WATER_HIGHLIGHT_GAIN = 0.12f0
const WATER_HIGHLIGHT_MAX  = 0.05f0


# =======================================================
# 6. THESIS-INSPIRED OPTICAL KNOBS
# =======================================================

# Air-water refractive indices for Schlick Fresnel approximation
const REFRACTIVE_INDEX_AIR   = 1.0f0
const REFRACTIVE_INDEX_WATER = 1.333f0

# Beer–Lambert absorption coefficients.
# Larger red absorption makes deeper water visually shift toward blue/green.
const WATER_ABSORB_R = 0.17435f0
const WATER_ABSORB_G = 0.03046f0
const WATER_ABSORB_B = 0.05129f0

# Strength of depth absorption.
# Increase if deep water is not dark/blue enough.
# Decrease if bottom terrain disappears too quickly.
const WATER_DEPTH_ABSORB_SCALE = 36.0f0

# Reflected sky colors
const SKY_HORIZON = RGBf(0.82, 0.90, 1.00)
const SKY_ZENITH  = RGBf(0.45, 0.66, 0.95)

# Fake specular sun highlight
const SUN_DIRECTION = (0.35f0, 0.25f0, 0.90f0)
const SUN_GLINT_STRENGTH = 0.16f0
const SUN_GLINT_SHININESS = 100f0

# Approximate viewing direction for Fresnel reflection.
# Tune this only if the reflected look becomes visually strange.
const VIEW_DIRECTION = (0.10f0, -0.92f0, 0.38f0)


# =======================================================
# 7. WATER DEPTH COLOR KNOBS
# =======================================================

# Depth-dependent water tint:
# shallow -> cyan
# medium  -> turquoise
# deep    -> dark blue
const WATER_TINT_SHALLOW = RGBf(0.22, 0.92, 0.92)
const WATER_TINT_MID     = RGBf(0.10, 0.78, 0.80)
const WATER_TINT_DEEP    = RGBf(0.01, 0.06, 0.24)

# How strongly water tint replaces the terrain color.
# Final interpolation weight:
# WATER_TINT_BASE_WEIGHT + WATER_TINT_DEPTH_WEIGHT * normalized_depth
const WATER_TINT_BASE_WEIGHT  = 0.45f0
const WATER_TINT_DEPTH_WEIGHT = 0.70f0


# =======================================================
# 8. DATA INTERPRETATION
# =======================================================

# h = water column height
# z = bathymetry / ground elevation
#
# Therefore:
# terrain surface = z
# water surface   = z + h

water_surface(h, z) = Float32.(h .+ z) .* HEIGHT_SCALE
bathymetry_surface(z) = Float32.(z) .* HEIGHT_SCALE


# =======================================================
# 9. FILE HANDLING
# =======================================================

function sorted_frame_paths(dir::AbstractString)
    files = filter(readdir(dir; join=true)) do f
        occursin(FRAME_REGEX, basename(f))
    end

    return sort(files)
end

function read_frame_topo(path::AbstractString)
    data = deserialize(path)

    h = data.h
    z = data.z

    return h, z
end

function read_frame(path::AbstractString)
    data = deserialize(path)

    h = data.h

    return h
end


# =======================================================
# 10. ARRAY HELPERS
# =======================================================

function downsample(A::AbstractMatrix, stride::Int)
    return A[1:stride:end, 1:stride:end]
end

function grid_xy(A::AbstractMatrix)
    nx, ny = size(A)

    x = range(0f0, 1f0, length=nx)
    y = range(0f0, 1f0, length=ny)

    return x, y
end


# =======================================================
# 11. COLOR AND VECTOR HELPERS
# =======================================================

lerp(a, b, t) = (1 - t) * a + t * b

function lerp_rgb(c1::RGBf, c2::RGBf, t)
    RGBf(
        lerp(c1.r, c2.r, t),
        lerp(c1.g, c2.g, t),
        lerp(c1.b, c2.b, t),
    )
end

function dot3(a, b)
    return a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
end

function norm3(a)
    return sqrt(dot3(a, a))
end

function normalize3(a)
    n = max(norm3(a), eps(Float32))
    return (a[1] / n, a[2] / n, a[3] / n)
end

function add_rgb(c::RGBf, x::Float32)
    return RGBf(
        clamp(c.r + x, 0f0, 1f0),
        clamp(c.g + x, 0f0, 1f0),
        clamp(c.b + x, 0f0, 1f0),
    )
end

function mul_rgb(c1::RGBf, c2::RGBf)
    return RGBf(
        clamp(c1.r * c2.r, 0f0, 1f0),
        clamp(c1.g * c2.g, 0f0, 1f0),
        clamp(c1.b * c2.b, 0f0, 1f0),
    )
end


# =======================================================
# 12. TERRAIN COLORING
# =======================================================

function make_terrain_colors(bed::AbstractMatrix{<:Real})
    lo, hi = extrema(bed)
    rng = max(hi - lo, eps(Float32))

    C = Matrix{RGBAf}(undef, size(bed)...)

    for I in eachindex(bed)
        t = clamp((Float32(bed[I]) - lo) / rng, 0f0, 1f0)

        rgb =
            t < 0.35f0 ? lerp_rgb(TERRAIN_LOW, TERRAIN_MID1, t / 0.35f0) :
            t < 0.70f0 ? lerp_rgb(TERRAIN_MID1, TERRAIN_MID2, (t - 0.35f0) / 0.35f0) :
                         lerp_rgb(TERRAIN_MID2, TERRAIN_HIGH, (t - 0.70f0) / 0.30f0)

        C[I] = RGBAf(rgb.r, rgb.g, rgb.b, 1.0f0)
    end

    return C
end


# =======================================================
# 13. WATER GEOMETRY HELPERS
# =======================================================

function slope_magnitude(A::AbstractMatrix{<:Real})
    nx, ny = size(A)
    S = zeros(Float32, nx, ny)

    for i in 2:nx-1, j in 2:ny-1
        dx = 0.5f0 * (A[i+1, j] - A[i-1, j])
        dy = 0.5f0 * (A[i, j+1] - A[i, j-1])
        S[i, j] = sqrt(Float32(dx^2 + dy^2))
    end

    S[1, :]   .= S[2, :]
    S[end, :] .= S[end-1, :]
    S[:, 1]   .= S[:, 2]
    S[:, end] .= S[:, end-1]

    return S
end

function surface_normals(A::AbstractMatrix{<:Real})
    nx, ny = size(A)
    N = Matrix{NTuple{3, Float32}}(undef, nx, ny)

    dx_grid = 1f0 / max(nx - 1, 1)
    dy_grid = 1f0 / max(ny - 1, 1)

    for i in 1:nx, j in 1:ny
        im = max(i - 1, 1)
        ip = min(i + 1, nx)
        jm = max(j - 1, 1)
        jp = min(j + 1, ny)

        dzdx = Float32(A[ip, j] - A[im, j]) / Float32((ip - im) * dx_grid)
        dzdy = Float32(A[i, jp] - A[i, jm]) / Float32((jp - jm) * dy_grid)

        N[i, j] = normalize3((-dzdx, -dzdy, 1f0))
    end

    return N
end


# =======================================================
# 14. THESIS-INSPIRED WATER OPTICS
# =======================================================

function schlick_fresnel(
    cosθ::Float32;
    n1::Float32=REFRACTIVE_INDEX_AIR,
    n2::Float32=REFRACTIVE_INDEX_WATER,
)
    r0 = ((n1 - n2) / (n1 + n2))^2
    return clamp(r0 + (1f0 - r0) * (1f0 - cosθ)^5, 0f0, 1f0)
end

function beer_lambert_transmittance(depth::Float32)
    d = max(depth * WATER_DEPTH_ABSORB_SCALE, 0f0)

    return RGBf(
        exp(-WATER_ABSORB_R * d),
        exp(-WATER_ABSORB_G * d),
        exp(-WATER_ABSORB_B * d),
    )
end

function make_water_colors(
    water::AbstractMatrix{<:Real},
    bed::AbstractMatrix{<:Real},
    terrain_colors::AbstractMatrix{RGBAf},
)
    depth = max.(Float32.(water .- bed), 0f0)
    slope = slope_magnitude(Float32.(water))
    normals = surface_normals(Float32.(water))

    dlo, dhi = extrema(depth)
    drng = max(dhi - dlo, eps(Float32))
    smax = max(maximum(slope), eps(Float32))

    view_dir = normalize3(VIEW_DIRECTION)
    sun_dir = normalize3(SUN_DIRECTION)

    half_vec = normalize3((
        view_dir[1] + sun_dir[1],
        view_dir[2] + sun_dir[2],
        view_dir[3] + sun_dir[3],
    ))

    C = Matrix{RGBAf}(undef, size(water)...)

    for I in eachindex(water)
        d = depth[I]
        td_linear = clamp((d - dlo) / drng, 0f0, 1f0)
        td = sqrt(td_linear)
        n = normals[I]

        # -------------------------------------------------------
        # 1. Terrain visible through water
        # -------------------------------------------------------

        terrain_rgba = terrain_colors[I]
        terrain_rgb = RGBf(
            terrain_rgba.r,
            terrain_rgba.g,
            terrain_rgba.b,
        )

        transmittance = beer_lambert_transmittance(d)
        bottom_seen = mul_rgb(terrain_rgb, transmittance)

        # -------------------------------------------------------
        # 2. Depth-based water tint
        # -------------------------------------------------------

        water_tint =
            td < 0.45f0 ?
                lerp_rgb(WATER_TINT_SHALLOW, WATER_TINT_MID, td / 0.45f0) :
                lerp_rgb(WATER_TINT_MID, WATER_TINT_DEEP, (td - 0.45f0) / 0.55f0)

        tint_weight = WATER_TINT_BASE_WEIGHT + WATER_TINT_DEPTH_WEIGHT * td

        refracted_col = lerp_rgb(
            bottom_seen,
            water_tint,
            tint_weight,
        )

        # -------------------------------------------------------
        # 3. Fresnel-style reflected sky
        # -------------------------------------------------------

        cosθ = clamp(abs(dot3(n, view_dir)), 0f0, 1f0)
        fresnel = schlick_fresnel(cosθ)

        fresnel_visual = clamp(
            0.10f0 + 4.5f0 * fresnel,
            0f0,
            0.85f0,
        )

        sky_mix = clamp(n[3], 0f0, 1f0)
        reflected_sky = lerp_rgb(SKY_HORIZON, SKY_ZENITH, sky_mix)

        # Stronger light / dark contrast across the disturbance
        light_amount = clamp(dot3(n, sun_dir), 0f0, 1f0)
        shadow_factor = 0.30f0 + 0.70f0 * light_amount

        # Bigger, more visible specular highlight
        sun_amount = clamp(dot3(n, half_vec), 0f0, 1f0)^SUN_GLINT_SHININESS
        sun_glint = SUN_GLINT_STRENGTH * sun_amount

        slope_factor = clamp(slope[I] / smax, 0f0, 1f0)
        slope_highlight = clamp(
            WATER_HIGHLIGHT_GAIN * slope_factor,
            0f0,
            WATER_HIGHLIGHT_MAX,
        )

        rgb = lerp_rgb(refracted_col, reflected_sky, fresnel_visual)

        rgb = RGBf(
            clamp(rgb.r * shadow_factor, 0f0, 1f0),
            clamp(rgb.g * shadow_factor, 0f0, 1f0),
            clamp(rgb.b * shadow_factor, 0f0, 1f0),
        )

        rgb = add_rgb(rgb, Float32(sun_glint + slope_highlight))

        alpha = lerp(WATER_ALPHA_SHALLOW, WATER_ALPHA_DEEP, td)

        C[I] = RGBAf(rgb.r, rgb.g, rgb.b, alpha)
    end

    return C
end

function center_crop(A::AbstractMatrix, n::Int)
    nx, ny = size(A)

    i0 = fld(nx - n, 2) + 1
    j0 = fld(ny - n, 2) + 1

    return A[i0:i0+n-1, j0:j0+n-1]
end


# =======================================================
# 15. SCENE CONSTRUCTION
# =======================================================

function make_scene(h_raw, z_raw; title="SWE water surface")
    h = downsample(Float32.(h_raw), STRIDE)
    z = downsample(Float32.(z_raw), STRIDE)

    bed = bathymetry_surface(z)
    water = water_surface(h, z)

    x, y = grid_xy(water)

    terrain_colors = make_terrain_colors(bed)
    water_colors   = make_water_colors(water, bed, terrain_colors)

    fig = Figure(
        size=(1400, 900),
        fontsize=18,
        backgroundcolor=SKY_COLOR,
    )

    ax = Axis3(
        fig[1, 1],
        title=title,
        xlabel="x",
        ylabel="y",
        zlabel="height",
        azimuth=CAMERA_AZIMUTH,
        elevation=CAMERA_ELEVATION,
        perspectiveness=CAMERA_PERSPECTIVE,
        aspect=AXIS_ASPECT,
        backgroundcolor=SKY_COLOR,
    )

    hidedecorations!(ax)
    hidespines!(ax)

    # Ground / bathymetry
    surface!(
        ax,
        x, y, bed;
        color=terrain_colors,
        shading=true,
    )

    # Water surface
    surface!(
        ax,
        x, y, water;
        color=water_colors,
        transparency=true,
        shading=true,
    )

    return fig
end


# =======================================================
# 16. RENDER ALL FRAMES
# =======================================================

function render_frames()
    paths = sorted_frame_paths(FRAME_DIR)

    if isempty(paths)
        error("No frames found in $(FRAME_DIR) matching $(FRAME_REGEX)")
    end

    @info "Found $(length(paths)) frames"
    z = nothing
    for (i, path) in enumerate(paths)
        if i == 1
            h, z_i = read_frame_topo(path)
            z = z_i
        else
            h = read_frame(path)
        end
        @info "Plotting $(basename(path))" frame=i size_h=size(h) size_z=size(z)

        fig = make_scene(h, z; title="SWE frame $(i)")

        outpath = joinpath(OUT_DIR, "swe_water_$(lpad(i, 5, '0')).png")
        save(outpath, fig)

        @info "Saved $outpath"
    end
end

render_frames()