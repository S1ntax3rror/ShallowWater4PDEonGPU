#!/usr/bin/env julia

using Serialization
using Printf
using GLMakie

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

# Directory of this script, e.g. project/src/viz
script_dir = @__DIR__

# Change this path to wherever your .jls files are,
# relative to src/viz/
frames_dir = normpath(joinpath(script_dir, "..", "..", "frames"))

# Output directory for plots
plot_dir = normpath(joinpath(script_dir, "..", "..", "docs", "multi_xpu"))
mkpath(plot_dir)

@info "Reading frames from $frames_dir"
@info "Saving plots to $plot_dir"

# ------------------------------------------------------------
# Collect frame files
# ------------------------------------------------------------

frame_files = sort(filter(f -> endswith(f, ".jls"), readdir(frames_dir)))

if isempty(frame_files)
    error("No .jls files found in $frames_dir")
end

# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------

# Choose backend if needed
# gr()

# Color limits can be fixed manually for consistent animation frames
# h_clims = (0.0, 1.0)
# z_clims = (0.0, 1.0)

# physics and numerics
lx_aoi = 50.0 # aoi = area of interest
ly_aoi = 50.0
nx_aoi = 99
ny_aoi = 99
xs = LinRange(-lx_aoi / 2, lx_aoi / 2, nx_aoi)
ys = LinRange(-ly_aoi / 2, ly_aoi / 2, ny_aoi)



# ------------------------------------------------------------
# Loop over frames
# ------------------------------------------------------------

for (i, file) in enumerate(frame_files)

    frame_path = joinpath(frames_dir, file)

    data = deserialize(frame_path)

    # Your saved NamedTuple has:
    # data.h
    # data.z
    h = data.h
    z = data.z

    @info "Plotting $file" frame=i size_h=size(h) size_z=size(z)

    # --------------------------------------------------------
    # Example 1: plot h as heatmap
    # --------------------------------------------------------

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


    save(joinpath(plot_dir, @sprintf("h_frame_%06d.png", i)), fig)
end

@info "Done plotting $(length(frame_files)) frames."