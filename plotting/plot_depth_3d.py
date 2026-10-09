"""Render an oblique terrain and depth-coloured free surface as a scientific MP4."""
import argparse
import os
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource, PowerNorm
from matplotlib.collections import PolyCollection
import numpy as np
from scipy.ndimage import binary_dilation


class ProjectedSurface(PolyCollection):
    def do_3d_projection(self):
        return 0.0


def project(vertices, matrix):
    points = np.concatenate((vertices, np.ones((*vertices.shape[:2], 1))), axis=-1)
    projected = points @ matrix.T
    return projected[:, :, :3] / projected[:, :, 3:]


def block_mean(field, step):
    ny, nx = field.shape
    ny, nx = ny // step * step, nx // step * step
    return field[:ny, :nx].reshape(ny // step, step, nx // step, step).mean(axis=(1, 3))


def corner_mean(field):
    padded = np.pad(field, 1, mode="edge")
    return .25 * (padded[:-1, :-1] + padded[:-1, 1:] + padded[1:, 1:] + padded[1:, :-1])


def quads(points):
    return np.stack((points[:-1, :-1], points[:-1, 1:],
                     points[1:, 1:], points[1:, :-1]), axis=-2)


def terrain_mesh(bed, cfg, spacing, water_spacing, wet_region):
    step = max(1, round(water_spacing / cfg["dx"]))
    z = block_mean(bed, step)
    ratio = max(1, round(spacing / (step * cfg["dx"])))
    ny, nx = (np.array(z.shape) // ratio * ratio).astype(int)
    z = z[:ny, :nx]
    cell = step * cfg["dx"]
    x = np.arange(nx + 1) * cell / 1000
    y = np.arange(ny + 1) * cell / 1000
    X, Y = np.meshgrid(x, y)
    points = np.stack((X, Y, corner_mean(z)), axis=-1)
    region = block_mean(wet_region, step)[:ny, :nx] > 0
    refine = block_mean(region, ratio) > 0
    refine = binary_dilation(refine, iterations=1)
    fine = np.repeat(np.repeat(refine, ratio, axis=0), ratio, axis=1)
    vertices = np.concatenate((quads(points)[fine], quads(points[::ratio, ::ratio])[~refine]))
    light = LightSource(azdeg=315, altdeg=45).hillshade(z, dx=cell, dy=-cell)
    shade = .58 + .27 * np.concatenate((light[fine], block_mean(light, ratio)[~refine]))
    colours = np.column_stack((shade, shade, shade, np.ones_like(shade)))
    return vertices, colours


def water_mesh(depth, bed, cfg, spacing, cmap, norm):
    step = max(1, round(spacing / cfg["dx"]))
    mean_depth = block_mean(depth, step)
    wet = depth > .05
    fraction = block_mean(wet, step)
    surface = corner_mean(block_mean(np.where(wet, bed + depth, 0), step)) / np.maximum(corner_mean(fraction), 1e-20)
    iy, ix = np.nonzero(mean_depth > .05)
    cell = step * cfg["dx"] / 1000
    x, y = (ix + .5) * cell, (iy + .5) * cell
    vertices = np.empty((len(ix), 4, 3))
    for corner, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        vertices[:, corner, 0] = x + sx * cell / 2
        vertices[:, corner, 1] = y + sy * cell / 2
        vertices[:, corner, 2] = surface[iy + (sy + 1) // 2, ix + (sx + 1) // 2]
    return vertices, cmap(norm(mean_depth[iy, ix]))


def render(case_dir, frames_dir, output, fps=20, terrain_spacing=50,
           water_spacing=10, elevation=45, azimuth=-25, exaggeration=1.5, preview_time=None):
    cfg = tomllib.loads((case_dir / "case.toml").read_text())
    shape = (cfg["ny"], cfg["nx"])
    before = np.fromfile(case_dir / "bed_before.bin", dtype="<f8").reshape(shape)
    after = np.fromfile(case_dir / "bed_after.bin", dtype="<f8").reshape(shape)
    maximum_depth = frames_dir / "maximum_depth.bin"
    wet_region = (np.fromfile(maximum_depth, dtype="<f4").reshape(shape) > .05
                  if maximum_depth.exists() else before < cfg["water_level"])
    mesh_before = terrain_mesh(before, cfg, terrain_spacing, water_spacing, wet_region)
    mesh_after = terrain_mesh(after, cfg, terrain_spacing, water_spacing, wet_region)
    logs = np.atleast_1d(np.genfromtxt(frames_dir / "diagnostics.csv", delimiter=",", names=True))
    fig = plt.figure(figsize=(12.8, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    fig.subplots_adjust(left=.02, right=.88, bottom=.075, top=.9)
    collection = ProjectedSurface([], edgecolors="none", antialiased=False)
    ax.add_collection(collection)
    lx, ly = cfg["nx"] * cfg["dx"] / 1000, cfg["ny"] * cfg["dy"] / 1000
    zmin, zmax = float(min(before.min(), after.min())), float(before.max())
    ax.set(xlim=(0, lx), ylim=(0, ly), zlim=(zmin - 30, zmax + 40),
           xlabel=f"Easting from {cfg['x_min'] / 1000:g} km (km)",
           ylabel=f"Northing from {cfg['y_min'] / 1000:g} km (km)", zlabel="Elevation (m, LN02)")
    ax.set_box_aspect((lx, ly, (zmax - zmin) / 1000 * exaggeration))
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_proj_type("ortho")
    ax.tick_params(labelsize=9)
    ax.set_zlabel("Elevation (m, LN02)", labelpad=12)
    for location in cfg.get("locations", []):
        x = (location["easting"] - cfg["x_min"]) / 1000
        y = (location["northing"] - cfg["y_min"]) / 1000
        ix = int(x * 1000 / cfg["dx"])
        iy = int(y * 1000 / cfg["dy"])
        if 0 <= ix < cfg["nx"] and 0 <= iy < cfg["ny"]:
            z = before[iy, ix] + 25
            ax.scatter([x], [y], [z], color="black", marker="+", s=20, depthshade=False, zorder=5)
            ax.text(x, y, z + 30, location["name"], fontsize=9, zorder=6)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 1))
        axis._axinfo["grid"]["color"] = (.75, .75, .75, .4)
    cmap, norm = plt.get_cmap("turbo"), PowerNorm(.5, vmin=0, vmax=80)
    colourbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=.7, pad=.035, fraction=.025)
    colourbar.set_ticks([0, 1, 5, 10, 20, 40, 60, 80])
    colourbar.set_label("Water depth (m; square-root colour scale)")
    ax.set_anchor("C")
    fig.suptitle(cfg["name"], fontsize=17, y=.98)
    status = fig.text(.5, .938, "", ha="center", fontsize=12)
    fig.text(.5, .025, f"{cfg['dx']:g} m simulation grid · vertical exaggeration {exaggeration:g}× · "
             "idealized lake bed · terrain © swisstopo", ha="center", fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    projection = ax.get_proj()
    projected_before = project(mesh_before[0], projection)
    projected_after = project(mesh_after[0], projection)

    def draw(row):
        removed = row["time_s"] >= cfg["breach_time"] - 1e-8
        bed = after if removed else before
        terrain_colours = (mesh_after if removed else mesh_before)[1]
        terrain_projection = projected_after if removed else projected_before
        depth = np.fromfile(frames_dir / f"depth_{int(row['frame']):05d}.bin", dtype="<f4").reshape(shape)
        water_vertices, water_colours = water_mesh(depth, bed, cfg, water_spacing, cmap, norm)
        # Cache the fixed terrain projection and sort all faces with vectorized operations.
        projected = np.concatenate((terrain_projection, project(water_vertices, projection)))
        order = np.argsort(projected[:, :, 2].mean(axis=1))[::-1]
        collection.set_verts(projected[order, :, :2])
        collection.set_facecolor(np.concatenate((terrain_colours, water_colours))[order])
        state = f"dam removed at {cfg['breach_time']:g} s" if removed else "dam intact"
        status.set_text(f"Simulation time {row['time_s']:6.1f} s · {state} · "
                        f"reservoir {row['reservoir_volume_m3'] / 1e6:.2f} million m³")
        fig.canvas.draw()

    if preview_time is not None:
        draw(logs[np.argmin(abs(logs["time_s"] - preview_time))])
        fig.savefig(output.with_suffix(".png"))
        plt.close(fig)
        return
    width, height = fig.canvas.get_width_height()
    process = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
                                "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                                "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], stdin=subprocess.PIPE)
    previews = {0: "initial", int(np.argmin(abs(logs["time_s"] - 100))): "flood", len(logs) - 1: "final"}
    try:
        for i, row in enumerate(logs):
            draw(row)
            pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy().tobytes()
            process.stdin.write(pixels)
            if i in (0, len(logs) - 1):
                for _ in range(fps):
                    process.stdin.write(pixels)
            if i in previews:
                fig.savefig(output.with_name(output.stem + "_" + previews[i] + ".png"))
            if i % 40 == 0:
                print(f"Rendered {i + 1}/{len(logs)} frames", flush=True)
    finally:
        process.stdin.close()
        result = process.wait()
        plt.close(fig)
    if result:
        raise RuntimeError(f"ffmpeg exited with status {result}")
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=ROOT / "cache/swiss_dam/cleuson_5m")
    parser.add_argument("--frames", type=Path, default=ROOT / "cache/swiss_dam/frames_5m")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/animations/cleuson_dam_break_5m_3d.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--terrain-spacing", type=float, default=50)
    parser.add_argument("--water-spacing", type=float, default=10)
    parser.add_argument("--elevation", type=float, default=45)
    parser.add_argument("--azimuth", type=float, default=-25)
    parser.add_argument("--exaggeration", type=float, default=1.5)
    parser.add_argument("--preview-time", type=float)
    args = parser.parse_args()
    render(args.case, args.frames, args.output, args.fps, args.terrain_spacing,
           args.water_spacing, args.elevation, args.azimuth, args.exaggeration, args.preview_time)
