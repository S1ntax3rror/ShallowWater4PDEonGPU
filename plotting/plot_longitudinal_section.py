"""Plot the free-surface elevation along a fixed polyline through a dam-break case."""
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
from matplotlib.colors import LightSource, PowerNorm
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np


def sample_path(points, spacing):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if spacing <= 0 or not np.isfinite(spacing) or np.any(lengths <= 0):
        raise ValueError("Sample spacing and segment lengths must be positive")
    coordinates, distances = [], []
    offset = 0.0
    for start, end, length in zip(points[:-1], points[1:], lengths):
        fraction = np.arange(int(np.ceil(length / spacing))) / np.ceil(length / spacing)
        coordinates.append(start + fraction[:, None] * (end - start))
        distances.append(offset + fraction * length)
        offset += length
    return np.vstack((*coordinates, points[-1:])), np.concatenate((*distances, [offset]))


def render(case_dir, frames_dir, path, output, fps=20, times=(0, 35, 45, 60, 100, 200)):
    cfg = tomllib.loads((case_dir / "case.toml").read_text())
    shape = (cfg["ny"], cfg["nx"])
    points = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    coordinates, distance = sample_path(points, min(cfg["dx"], cfg["dy"]))
    indices = np.vstack(((coordinates[:, 1] - cfg["y_min"]) / cfg["dy"] - .5,
                         (coordinates[:, 0] - cfg["x_min"]) / cfg["dx"] - .5))
    if (not np.all(np.isfinite(indices)) or np.any(indices < 0)
            or np.any(indices[0] > shape[0] - 1) or np.any(indices[1] > shape[1] - 1)):
        raise ValueError("Section must lie within the case's cell-centre bounds")

    lower = np.floor(indices).astype(int)
    fy, fx = indices - lower
    rows = np.minimum(lower[0] + np.array([[0], [0], [1], [1]]), shape[0] - 1)
    cols = np.minimum(lower[1] + np.array([[0], [1], [0], [1]]), shape[1] - 1)
    weights = np.vstack(((1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx))

    def read_cells(file, dtype="<f8"):
        field = np.memmap(file, dtype=dtype, mode="r", shape=shape)
        return field[rows, cols]

    before_cells = read_cells(case_dir / "bed_before.bin")
    after_cells = read_cells(case_dir / "bed_after.bin")
    before = (weights * before_cells).sum(axis=0)
    after = (weights * after_cells).sum(axis=0)
    dam = (weights * read_cells(case_dir / "dam_mask.bin")).sum(axis=0) > .5
    if not dam.any():
        raise ValueError("Section must cross the dam footprint")
    distance -= distance[dam].mean()
    logs = np.atleast_1d(np.genfromtxt(frames_dir / "diagnostics.csv", delimiter=",", names=True))
    depth_cells = np.stack([read_cells(frames_dir / f"depth_{int(row['frame']):05d}.bin", "<f4") for row in logs])
    if not np.all(np.isfinite(depth_cells)) or np.any(depth_cells < 0):
        raise ValueError("Section contains invalid water depths")
    depths = (weights * depth_cells).sum(axis=1)
    removed = logs["time_s"] >= cfg["breach_time"] - 1e-8
    bed = np.where(removed[:, None], after, before)
    # Dry bank/crest elevations must not enter interpolation of the water surface.
    wet_weights = weights * (depth_cells >= .05)
    wet_fraction = wet_weights.sum(axis=1)
    cell_bed = np.where(removed[:, None, None], after_cells, before_cells)
    surface = np.divide((wet_weights * (cell_bed + depth_cells)).sum(axis=1), wet_fraction,
                        out=np.full_like(depths, np.nan), where=(wet_fraction >= .5) & (depths >= .05))
    surface = np.where(surface >= bed, surface, np.nan)
    changed = np.abs(before - after) > .1
    zoom = np.abs(distance) <= 200
    ymin, ymax = min(after.min(), before.min()) - 15, max(before.max(), cfg["water_level"]) + 20
    zmin = after[zoom].min() - 10
    output.parent.mkdir(parents=True, exist_ok=True)

    def figure(animated=False):
        fig = plt.figure(figsize=(13.2, 7.8), dpi=120)
        grid = fig.add_gridspec(2, 2, width_ratios=(2.1, 1), height_ratios=(1, 1),
                               left=.065, right=.97, bottom=.11, top=.865, hspace=.48, wspace=.23)
        axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])]
        map_ax = fig.add_subplot(grid[:, 1])
        scales = (1000, 1)
        for ax, scale in zip(axes, scales):
            ax.axvline(0, color="crimson", ls=":", lw=1)
            ax.plot(distance / scale, np.where(changed, before, np.nan),
                    color="crimson", ls="--", lw=1, label="Intact dam (reference)")
            ax.set_ylabel("Elevation (m, LN02)")
            ax.grid(alpha=.2)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        axes[0].set(xlim=(distance[0] / 1000, distance[-1] / 1000), ylim=(ymin, ymax),
                    xlabel="Distance along section from dam (km; upstream < 0)",
                    title="A–B: reservoir → dam → outlet")
        axes[1].set(xlim=(-200, 200), ylim=(zmin, ymax),
                    xlabel="Distance along section from dam (m)", title="Detail around the breach")

        west, south = points.min(axis=0) - 200
        east, north = points.max(axis=0) + 200
        ix0 = max(0, int((west - cfg["x_min"]) / cfg["dx"]))
        ix1 = min(cfg["nx"], int(np.ceil((east - cfg["x_min"]) / cfg["dx"])))
        iy0 = max(0, int((south - cfg["y_min"]) / cfg["dy"]))
        iy1 = min(cfg["ny"], int(np.ceil((north - cfg["y_min"]) / cfg["dy"])))
        window = (slice(iy0, iy1), slice(ix0, ix1))
        terrain = np.memmap(case_dir / "terrain.bin", dtype="<f8", mode="r", shape=shape)[window]
        lake = np.memmap(case_dir / "h_initial.bin", dtype="<f8", mode="r", shape=shape)[window]
        extent = ((cfg["x_min"] + ix0 * cfg["dx"]) / 1000,
                  (cfg["x_min"] + ix1 * cfg["dx"]) / 1000,
                  (cfg["y_min"] + iy0 * cfg["dy"]) / 1000,
                  (cfg["y_min"] + iy1 * cfg["dy"]) / 1000)
        shade = LightSource(315, 45).hillshade(terrain, dx=cfg["dx"], dy=-cfg["dy"])
        map_ax.imshow(shade, cmap="gray", vmin=-.5, vmax=1.5, extent=extent, origin="lower")
        map_water = map_ax.imshow(np.ma.masked_less(lake, .05),
                                 cmap="viridis" if animated else "Blues",
                                 norm=PowerNorm(.5 if animated else 1, vmin=0, vmax=80),
                                 extent=extent, origin="lower", alpha=1 if animated else .8)
        if animated:
            colourbar = fig.colorbar(map_water, ax=map_ax, orientation="horizontal",
                                    fraction=.035, pad=.12, shrink=.9)
            colourbar.set_ticks([0, 1, 5, 10, 20, 40, 60, 80])
            colourbar.ax.tick_params(labelsize=8)
            colourbar.set_label("Water depth (m; square-root scale)", fontsize=9)
        map_ax.plot(points[:, 0] / 1000, points[:, 1] / 1000, color="crimson", lw=1.5)
        for label, point in zip(("A", "B"), points[[0, -1]]):
            map_ax.annotate(label, point / 1000, xytext=(6, 0), textcoords="offset points",
                            color="crimson", weight="bold", fontsize=12)
        anchor = np.array([np.interp(0, distance, coordinates[:, i]) for i in (0, 1)]) / 1000
        map_ax.plot(*anchor, "o", color="black", ms=4)
        map_ax.annotate("Dam / 0 m", anchor, xytext=(8, 6), textcoords="offset points", fontsize=9)
        map_ax.set(xlabel="LV95 easting (km)", ylabel="LV95 northing (km)",
                   title="Water depth\nSimulation time 0.0 s" if animated else "Section location\n(initial lake)",
                   xlim=extent[:2], ylim=extent[2:])
        for axis in (map_ax.xaxis, map_ax.yaxis):
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            axis.set_major_formatter(formatter)
        map_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        map_ax.tick_params(labelsize=9)
        fig.suptitle("Cleuson dam break — longitudinal section", fontsize=17, y=.97)
        status = fig.text(.37, .925, "", ha="center", fontsize=12)
        fig.text(.5, .025, f"{cfg['dx']:g} m grid · fixed curved section · free surface η = z + h · "
                 "idealized lake bed · terrain © swisstopo", ha="center", fontsize=10)
        return fig, axes, scales, status, map_ax, map_water, window

    selected = np.unique([np.argmin(abs(logs["time_s"] - time)) for time in times])
    fig, axes, scales, status, map_ax, map_water, window = figure()
    colours = plt.get_cmap("viridis")(np.linspace(.05, .9, len(selected)))
    for ax, scale in zip(axes, scales):
        ax.fill_between(distance / scale, ymin, after, color=".82", label="Bed after removal")
        ax.plot(distance / scale, after, color=".3", lw=1)
        for i, colour in zip(selected, colours):
            ax.plot(distance / scale, surface[i], color=colour, lw=1.7, label=f"{logs[i]['time_s']:g} s")
    axes[0].legend(ncol=4, fontsize=8, loc="lower left")
    status.set_text(f"Free-surface profiles; dam removed at {cfg['breach_time']:g} s")
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)

    fig, axes, scales, status, map_ax, map_water, window = figure(animated=True)
    lines, fills, bed_lines, bed_fills = [], [], [], []
    for ax, scale in zip(axes, scales):
        ax.plot(distance / scale, surface[0], color=".4", ls="--", lw=1, label="Initial free surface")
        bed_lines.append(ax.plot(distance / scale, before, color=".25", lw=1, label="Bed")[0])
        bed_fills.append((ax.fill_between(distance / scale, ymin, before, color=".82"),
                          ax.fill_between(distance / scale, ymin, after, color=".82", visible=False)))
        lines.append(ax.plot([], [], color="#0072b2", lw=2, label="Current free surface")[0])
        fills.append(None)
    axes[0].legend(ncol=2, fontsize=8, loc="lower left")
    width, height = fig.canvas.get_width_height()
    process = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
                                "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                                "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], stdin=subprocess.PIPE)
    previews = {0: "initial", int(np.argmin(abs(logs["time_s"] - 100))): "flood", len(logs) - 1: "final"}
    try:
        for i, row in enumerate(logs):
            depth_map = np.memmap(frames_dir / f"depth_{int(row['frame']):05d}.bin",
                                  dtype="<f4", mode="r", shape=shape)[window]
            map_water.set_data(np.ma.masked_less(depth_map, .05))
            map_ax.set_title(f"Water depth\nSimulation time {row['time_s']:.1f} s")
            for j, (ax, scale) in enumerate(zip(axes, scales)):
                bed_lines[j].set_ydata(bed[i])
                bed_fills[j][0].set_visible(not removed[i])
                bed_fills[j][1].set_visible(removed[i])
                lines[j].set_data(distance / scale, surface[i])
                if fills[j] is not None:
                    fills[j].remove()
                fills[j] = ax.fill_between(distance / scale, bed[i], surface[i],
                                           where=np.isfinite(surface[i]), color="#56b4e9", alpha=.45)
            state = f"dam removed at {cfg['breach_time']:g} s" if removed[i] else "dam intact"
            status.set_text(f"Simulation time {row['time_s']:6.1f} s · {state}")
            fig.canvas.draw()
            pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy().tobytes()
            process.stdin.write(pixels)
            if i in (0, len(logs) - 1):
                for _ in range(fps):
                    process.stdin.write(pixels)
            if i in previews:
                fig.savefig(output.with_name(output.stem + "_" + previews[i] + ".png"))
            if i % 80 == 0:
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
    parser.add_argument("--path", type=Path, default=ROOT / "data/cleuson/section_path.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/animations/cleuson_longitudinal_section.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--times", type=float, nargs="+", default=[0, 35, 45, 60, 100, 200])
    args = parser.parse_args()
    render(args.case, args.frames, args.path, args.output, args.fps, args.times)
