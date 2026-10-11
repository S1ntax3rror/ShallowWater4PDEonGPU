"""Render depth snapshots from a case.toml grid as a scientific MP4 map."""
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
from matplotlib.ticker import ScalarFormatter
import numpy as np


def crop_grid(cfg, bounds):
    if bounds is None:
        return cfg, (slice(None), slice(None))
    west, south, east, north = bounds
    dx, dy = cfg["dx"], cfg["dy"]
    if (not np.all(np.isfinite(bounds)) or west >= east or south >= north
            or west < cfg["x_min"] or south < cfg["y_min"]
            or east > cfg["x_min"] + cfg["nx"] * dx
            or north > cfg["y_min"] + cfg["ny"] * dy):
        raise ValueError("Bounds must define a finite rectangle within the case domain")
    ix0 = int(np.floor((west - cfg["x_min"]) / dx))
    ix1 = int(np.ceil((east - cfg["x_min"]) / dx))
    iy0 = int(np.floor((south - cfg["y_min"]) / dy))
    iy1 = int(np.ceil((north - cfg["y_min"]) / dy))
    cropped = dict(cfg, nx=ix1 - ix0, ny=iy1 - iy0,
                   x_min=cfg["x_min"] + ix0 * dx, y_min=cfg["y_min"] + iy0 * dy)
    return cropped, (slice(iy0, iy1), slice(ix0, ix1))


def render(case_dir, frames_dir, output, fps=20, bounds=None, preview_time=None):
    cfg = tomllib.loads((case_dir / "case.toml").read_text())
    full_shape = (cfg["ny"], cfg["nx"])
    cfg, window = crop_grid(cfg, bounds)
    nx, ny = cfg["nx"], cfg["ny"]
    shape = (ny, nx)
    dx, dy = cfg["dx"], cfg["dy"]
    terrain = np.fromfile(case_dir / "terrain.bin", dtype="<f8").reshape(full_shape)[window]
    extent = [cfg["x_min"] / 1000, (cfg["x_min"] + nx * dx) / 1000,
              cfg["y_min"] / 1000, (cfg["y_min"] + ny * dy) / 1000]
    x = cfg["x_min"] / 1000 + (np.arange(nx) + .5) * dx / 1000
    y = cfg["y_min"] / 1000 + (np.arange(ny) + .5) * dy / 1000
    logs = np.atleast_1d(np.genfromtxt(frames_dir / "diagnostics.csv", delimiter=",", names=True))
    fig, ax = plt.subplots(figsize=(9, 10), dpi=120)
    fig.subplots_adjust(left=.13, right=.83, bottom=.095, top=.9)
    # Binary rows increase northward; LightSource normally expects image rows downward.
    relief = LightSource(azdeg=315, altdeg=45).hillshade(terrain, vert_exag=1, dx=dx, dy=-dy)
    ax.imshow(relief, cmap="gray", vmin=-.5, vmax=1.5, origin="lower", extent=extent)
    contours = ax.contour(x, y, terrain, levels=np.arange(1400, 3401, 200),
                          colors=".35", linewidths=.35, alpha=.45)
    ax.clabel(contours, inline=True, fontsize=6, fmt="%d")
    image = ax.imshow(np.ma.masked_all(shape), cmap="viridis", norm=PowerNorm(.5, vmin=0, vmax=80),
                      interpolation="nearest", origin="lower", extent=extent)
    colorbar = fig.colorbar(image, ax=ax, fraction=.045, pad=.06, shrink=.7)
    colorbar.set_ticks([0, 1, 5, 10, 20, 40, 60, 80])
    colorbar.set_label("Water depth (m; square-root colour scale)")
    dam_contour = None
    if (case_dir / "dam_mask.bin").exists():
        dam = np.fromfile(case_dir / "dam_mask.bin", dtype="<f8").reshape(full_shape)[window]
        dam_contour = ax.contour(x, y, dam, [.5], colors="crimson", linewidths=1.3)
    for location in cfg.get("locations", []):
        point = (location["easting"] / 1000, location["northing"] / 1000)
        if not (extent[0] <= point[0] <= extent[1] and extent[2] <= point[1] <= extent[3]):
            continue
        if "label_easting" in location:
            label = (location["label_easting"] / 1000, location["label_northing"] / 1000)
            ax.annotate(location["name"], point, xytext=label, fontsize=9,
                        arrowprops={"arrowstyle": "-", "color": "black"})
        else:
            ax.plot(*point, "+", color="black", markersize=7)
            ax.annotate(location["name"], point, xytext=(10, 3), textcoords="offset points", fontsize=9)
    ax.set(xlabel="LV95 easting (km)", ylabel="LV95 northing (km)",
           xlim=extent[:2], ylim=extent[2:])
    for axis in (ax.xaxis, ax.yaxis):
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
    fig.suptitle(cfg["name"], fontsize=16, y=.975)
    title = ax.set_title("", fontsize=11, pad=12)
    status = fig.text(.5, .935, "", ha="center", fontsize=11)
    footer = f"{dx:g} m grid"
    for key in ("bed_description", "terrain_attribution"):
        if key in cfg:
            footer += " · " + cfg[key]
    fig.text(.5, .025, footer, ha="center", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)

    def draw(row):
        depth = np.fromfile(frames_dir / f"depth_{int(row['frame']):05d}.bin", dtype="<f4").reshape(full_shape)[window]
        image.set_data(np.ma.masked_less(depth, .05))
        time = row["time_s"]
        removed = time >= cfg["breach_time"] - 1e-8
        if dam_contour is not None:
            dam_contour.set_visible(not removed)
        status.set_text(f"Simulation time: {time:6.1f} s — " +
                        (f"dam removed at {cfg['breach_time']:.0f} s" if removed else "dam intact"))
        title.set_text(f"Reservoir water remaining: {row['reservoir_volume_m3'] / 1e6:.2f} million m³")
        fig.canvas.draw()

    if preview_time is not None:
        draw(logs[np.argmin(abs(logs["time_s"] - preview_time))])
        fig.savefig(output.with_suffix(".png"))
        plt.close(fig)
        return
    width, height = fig.canvas.get_width_height()
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    previews = {0: "initial", int(np.argmin(abs(logs["time_s"] - 150))): "flood", len(logs) - 1: "final"}
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
    parser.add_argument("--case", type=Path, default=ROOT / "data/cleuson")
    parser.add_argument("--frames", type=Path, default=ROOT / "cache/swiss_dam/frames")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/animations/cleuson_dam_break.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--bounds", type=float, nargs=4, metavar=("E_MIN", "N_MIN", "E_MAX", "N_MAX"),
                        help="Crop to projected case coordinates in metres")
    parser.add_argument("--preview-time", type=float, help="Save a PNG at the nearest snapshot time")
    args = parser.parse_args()
    render(args.case, args.frames, args.output, args.fps, args.bounds, args.preview_time)
