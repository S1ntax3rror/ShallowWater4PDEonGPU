#!/usr/bin/env julia

using Serialization
using Printf
using FileIO
using Colors
using RPRMakie
using RadeonProRender
using GeometryBasics: Vec3f

const RPR = RadeonProRender

# ------------------------------------------------------------
# Activate backend
# ------------------------------------------------------------
RPRMakie.activate!(
    iterations = 64,
    plugin = RPR.Tahoe,
    resource = RPR.RPR_CREATION_FLAGS_ENABLE_CPU,
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
script_dir = @__DIR__
frames_dir = normpath(joinpath(script_dir, "..", "..", "frames"))
plot_dir   = normpath(joinpath(script_dir, "..", "..", "docs", "multi_xpu_ray"))
mkpath(plot_dir)

@info "Reading frames from $frames_dir"
@info "Saving plots to $plot_dir"

frame_files = sort(filter(f -> endswith(f, ".jls"), readdir(frames_dir)))
isempty(frame_files) && error("No .jls files found in $frames_dir")


# ------------------------------------------------------------
# Test peak on free surface (first frame only)
# ------------------------------------------------------------
add_test_peak = true
peak_amp      = 0.03f0     # amplitude in h-units
peak_sigma    = 2.5f0      # width in x/y units
peak_x0       = 0.0f0      # center x-position
peak_y0       = 0.0f0      # center y-position

# ------------------------------------------------------------
# Grid / domain
# ------------------------------------------------------------
lx_aoi = 50.0
ly_aoi = 50.0
nx_aoi = 99
ny_aoi = 99

xs = LinRange(-lx_aoi / 2, lx_aoi / 2, nx_aoi)
ys = LinRange(-ly_aoi / 2, ly_aoi / 2, ny_aoi)

# ------------------------------------------------------------
# Rendering knobs
# ------------------------------------------------------------
vertical_exaggeration = 25.0       # stronger terrain relief
hmin_plot             = 0.01f0     # hide very shallow water
water_offset          = 0.08f0     # lift water slightly above terrain

# camera in spherical-style parameters
camera_distance   = 95.0f0
camera_azimuth_deg   = 235.0f0     # rotate around z-axis
camera_elevation_deg = 28.0f0      # angle above xy-plane
camera_lookat        = Vec3f(0, 0, 2)
camera_fov_deg       = 20.0f0

# lighting
env_intensity    = 0.35            # lower than before -> less washed out
sun_radiance     = 14000f0
sun_position     = Vec3f(-60, -70, 65)
sun_radius       = 120.0f0

# ------------------------------------------------------------
# Helper: camera from azimuth/elevation
# ------------------------------------------------------------
function set_camera_azimuth_elevation!(scene;
    distance::Float32,
    azimuth_deg::Float32,
    elevation_deg::Float32,
    lookat::Vec3f,
    fov_deg::Float32
)
    az = deg2rad(azimuth_deg)
    el = deg2rad(elevation_deg)

    eye = Vec3f(
        lookat[1] + distance * cos(el) * cos(az),
        lookat[2] + distance * cos(el) * sin(az),
        lookat[3] + distance * sin(el)
    )

    cam = cameracontrols(scene)
    cam.eyeposition[] = eye
    cam.lookat[]      = lookat
    cam.upvector[]    = Vec3f(0, 0, 1)
    cam.fov[]         = fov_deg
    update_cam!(scene, cam)
    return nothing
end

function normalize01(A)
    amin, amax = extrema(A)
    if isapprox(amin, amax)
        return fill(0.5f0, size(A))
    end
    return Float32.((A .- amin) ./ (amax - amin))
end

function terrain_rgb_from_height(z_plot)
    zn = normalize01(z_plot)

    return map(zn) do v
        if v < 0.35f0
            # low terrain: dark green/brown
            t = v / 0.35f0
            RGBf(
                0.22f0 + 0.18f0 * t,
                0.30f0 + 0.20f0 * t,
                0.18f0 + 0.10f0 * t
            )
        elseif v < 0.70f0
            # mid terrain: brown/sand
            t = (v - 0.35f0) / 0.35f0
            RGBf(
                0.40f0 + 0.32f0 * t,
                0.50f0 + 0.20f0 * t,
                0.28f0 + 0.18f0 * t
            )
        else
            # high terrain: light rock
            t = (v - 0.70f0) / 0.30f0
            RGBf(
                0.72f0 + 0.18f0 * t,
                0.70f0 + 0.18f0 * t,
                0.62f0 + 0.22f0 * t
            )
        end
    end
end

function water_rgb_from_depth(h, hmin_plot)
    valid = h[h .> hmin_plot]

    if isempty(valid)
        return fill(RGBf(0.10f0, 0.25f0, 0.38f0), size(h))
    end

    hmin, hmax = extrema(valid)
    hn = if isapprox(hmin, hmax)
        fill(0.5f0, size(h))
    else
        Float32.((h .- hmin) ./ (hmax - hmin))
    end

    return map(hn) do v
        v = clamp(v, 0f0, 1f0)

        # darker blue for deeper water, softer cyan for shallow
        RGBf(
            0.08f0 + 0.22f0 * v,
            0.22f0 + 0.35f0 * v,
            0.35f0 + 0.45f0 * v
        )
    end
end

function gaussian_peak(xs, ys, x0, y0, amp, sigma)
    X = reshape(Float32.(collect(xs)), :, 1)
    Y = reshape(Float32.(collect(ys)), 1, :)
    return amp .* exp.(-((X .- x0).^2 .+ (Y .- y0).^2) ./ (2f0 * sigma^2))
end

# ------------------------------------------------------------
# Loop over frames
# ------------------------------------------------------------
for (i, file) in enumerate(frame_files)

    frame_path = joinpath(frames_dir, file)
    data = deserialize(frame_path)

    h = Float32.(data.h)
    z = Float32.(data.z)

    h_vis = copy(h)

    if add_test_peak && i == 1
        peak = gaussian_peak(xs, ys, peak_x0, peak_y0, peak_amp, peak_sigma)
        h_vis .+= peak
        @info "Added test peak to first frame" maximum_peak=maximum(peak)
    end

    @info "Plotting $file" frame=i size_h=size(h) size_z=size(z)

    # --------------------------------------------------------
    # Build terrain and water geometry
    # --------------------------------------------------------
    z_plot = vertical_exaggeration .* z

    η_water_plot = vertical_exaggeration .* (h_vis .+ z) .+ water_offset
    η_water_plot[h_vis .<= hmin_plot] .= NaN32

    # depth for coloring the water
    h_plot = copy(h_vis)
    h_plot[h .<= hmin_plot] .= NaN32

    terrain_color = terrain_rgb_from_height(z_plot)
    water_color = water_rgb_from_depth(h_vis, hmin_plot)

    # --------------------------------------------------------
    # Lights
    # --------------------------------------------------------
    lights = [
        EnvironmentLight(env_intensity, load(RPR.assetpath("studio026.exr"))),
        PointLight(
            RGBf(sun_radiance, sun_radiance, sun_radiance * 0.95f0),
            sun_position,
            sun_radius
        )
    ]

    # --------------------------------------------------------
    # Figure / scene
    # --------------------------------------------------------
    fig = Figure(size = (1400, 950))

    ax = LScene(
        fig[1, 1];
        show_axis = false,
        scenekw = (; lights = lights)
    )

    # --------------------------------------------------------
    # Terrain
    # --------------------------------------------------------
    surface!(
        ax,
        xs, ys, z_plot;
        color = terrain_color,
        shading = true,
        diffuse = Vec3f(0.70),
        specular = 0.03,
    )
    # --------------------------------------------------------
    # Water surface
    # --------------------------------------------------------
    # This is the TOP water layer.
    # We make it darker, less diffuse, more specular.
    surface!(
        ax,
        xs, ys, η_water_plot;
        color = water_color,
        shading = true,
        diffuse = Vec3f(0.03),
        specular = 2.8,
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------
    set_camera_azimuth_elevation!(
        ax.scene;
        distance = camera_distance,
        azimuth_deg = camera_azimuth_deg,
        elevation_deg = camera_elevation_deg,
        lookat = camera_lookat,
        fov_deg = camera_fov_deg
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    outname = joinpath(plot_dir, @sprintf("h_frame_%06d.png", i))
    save(outname, ax.scene)
end

@info "Done plotting $(length(frame_files)) frames."