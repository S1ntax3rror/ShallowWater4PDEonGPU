"""Download swissALTI3D and construct the idealized Cleuson dam-break case."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.merge import merge
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "data/cleuson"
CACHE = ROOT / "cache/swiss_dam"
BOUNDS = (2589000, 1105000, 2592500, 1112500)
DX = 25.0
LEVEL = 2186.0  # FOEN lake inventory, LN02 elevation.
END_TIME = 600.0
SNAPSHOT_INTERVAL = 1.5
STAC = "https://data.geo.admin.ch/api/stac/v1/collections/ch.swisstopo.swissalti3d/items"
LAKE_URL = ("https://api3.geo.admin.ch/rest/services/ech/MapServer/identify?"
            "geometryType=esriGeometryPoint&geometry=2591104,1106271&sr=2056&"
            "tolerance=0&layers=all:ch.bafu.vec25-seen&returnGeometry=true&geometryFormat=geojson")


def get(url):
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def download_terrain():
    CASE.mkdir(parents=True, exist_ok=True)
    (CACHE / "tiles").mkdir(parents=True, exist_ok=True)
    catalog = CACHE / "catalog.json"
    if catalog.exists():
        items = json.loads(catalog.read_text())
    else:
        items = []
        url = STAC + "?bbox=7.27,46.09,7.35,46.17&limit=100"
        while url:
            page = get(url).json()
            items.extend(page["features"])
            url = next((link["href"] for link in page["links"] if link["rel"] == "next"), None)
        catalog.write_text(json.dumps(items))
    selected = {}
    for item in items:
        tile = item["id"].split("_")[-1]
        east, north = map(int, tile.split("-"))
        if not (east * 1000 < BOUNDS[2] and (east + 1) * 1000 > BOUNDS[0]
                and north * 1000 < BOUNDS[3] and (north + 1) * 1000 > BOUNDS[1]):
            continue
        if tile not in selected or item["properties"]["datetime"] > selected[tile]["properties"]["datetime"]:
            selected[tile] = item

    def fetch(item):
        name, asset = next((name, asset) for name, asset in item["assets"].items()
                           if name.endswith("_2_2056_5728.tif"))
        path = CACHE / "tiles" / name
        if not path.exists():
            payload = get(asset["href"]).content
            checksum = asset.get("file:checksum", "")
            if checksum.startswith("1220"):
                assert hashlib.sha256(payload).hexdigest() == checksum[4:].lower(), name
            path.write_bytes(payload)
        print(name, flush=True)
        return path, {"item": item["id"], "url": asset["href"],
                      "datetime": item["properties"]["datetime"],
                      "checksum": asset.get("file:checksum")}

    fetched = list(ThreadPoolExecutor(max_workers=4).map(fetch, selected.values()))
    datasets = [rasterio.open(path) for path, _ in fetched]
    try:
        dem, transform = merge(datasets, bounds=BOUNDS, res=DX, resampling=Resampling.average)
    finally:
        for dataset in datasets:
            dataset.close()
    assert np.all(np.isfinite(dem)) and np.min(dem) > 0
    terrain = dem[0, ::-1].astype("<f8")
    terrain.tofile(CASE / "terrain.bin")
    lake_path = CASE / "lake.geojson"
    if not lake_path.exists():
        reference = ROOT / "data/cleuson/lake.geojson"
        lake = json.loads(reference.read_text()) if reference.exists() else get(LAKE_URL).json()["results"][0]
        lake_path.write_text(json.dumps(lake, indent=2))
    lake = json.loads(lake_path.read_text())
    sources = {"terrain": "https://www.swisstopo.admin.ch/en/height-model-swissalti3d",
               "lake_inventory": LAKE_URL,
               "dam": "https://www.grande-dixence.ch/en/the-complex/dams/cleuson-82/",
               "crs": "EPSG:2056 (LV95); heights LN02", "tiles": [meta for _, meta in fetched]}
    (CASE / "sources.json").write_text(json.dumps(sources, indent=2))
    mask = rasterize([(lake["geometry"], 1)], out_shape=dem.shape[1:], transform=transform).astype(bool)[::-1]
    return terrain, mask


def prepare_case(terrain, lake):
    ny, nx = terrain.shape
    x = BOUNDS[0] + (np.arange(nx) + .5) * DX
    y = BOUNDS[1] + (np.arange(ny) + .5) * DX
    X, Y = np.meshgrid(x, y)
    # The straight northwestern lake shoreline follows the dam crest.
    a = np.array([2590780., 1106690.])
    b = np.array([2591150., 1106860.])
    tangent = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-tangent[1], tangent[0]])  # downstream: northwest
    along = (X - a[0]) * tangent[0] + (Y - a[1]) * tangent[1]
    across = (X - a[0]) * normal[0] + (Y - a[1]) * normal[1]
    length = np.linalg.norm(b - a)
    dam = (along > -50) & (along < length + 50) & (np.abs(across) < 50)

    depth_shape = 1 - np.exp(-ndimage.distance_transform_edt(lake) * DX / 65)
    # Calibrate the synthetic bathymetry to the inventory's 20 million m³.
    scale = 20e6 / (depth_shape.sum() * DX**2)
    bed = terrain.copy()
    bed[lake] = LEVEL - scale * depth_shape[lake]
    before = bed.copy()
    before[dam] = np.maximum(before[dam], LEVEL + 2)

    # Connected low terrain defines the initial lake, with a horizontal surface.
    components, _ = ndimage.label(before < LEVEL)
    seed = np.unravel_index(np.argmax(depth_shape), lake.shape)
    reservoir = components == components[seed]
    assert not reservoir[0].any() and not reservoir[-1].any()
    assert not reservoir[:, 0].any() and not reservoir[:, -1].any()
    h = np.where(reservoir, np.maximum(0, LEVEL - before), 0)

    # Remove the full dam footprint down to a smooth foundation channel.
    # Its transverse profile connects the lake bed to the valley floor; banks stay high.
    channel = LEVEL - 72 + 72 * ((along - length / 2) / (length / 2 + 50))**2
    after = bed.copy()
    after[dam] = np.minimum(after[dam], channel[dam])
    for name, field in (("bed_before", before), ("bed_after", after), ("h_initial", h),
                        ("reservoir_mask", reservoir.astype(float)), ("dam_mask", dam.astype(float))):
        field.astype("<f8").tofile(CASE / (name + ".bin"))
    config = f'''name = "Cleuson dam break"
nx = {nx}
ny = {ny}
dx = {DX}
dy = {DX}
x_min = {BOUNDS[0]}
y_min = {BOUNDS[1]}
water_level = {LEVEL}
gravity = 9.81
manning = 0.025
velocity_depth_scale = 0.001
breach_time = 30.0
end_time = {END_TIME}
snapshot_interval = {SNAPSHOT_INTERVAL}
cfl = 0.45
bed_description = "idealized lake bed"
terrain_attribution = "terrain © swisstopo"

[[locations]]
name = "Cleuson dam"
easting = 2590980.0
northing = 1106790.0
label_easting = 2591550.0
label_northing = 1107100.0

[[locations]]
name = "Siviez"
easting = 2590724.0
northing = 1109231.0
'''
    (CASE / "case.toml").write_text(config)
    volume = h[1:-1, 1:-1].sum() * DX**2
    print(f"Grid {nx} × {ny}, initial volume {volume:.0f} m³, maximum depth {h.max():.2f} m", flush=True)
    print(f"Synthetic lake-bed scale {scale:.2f} m; dam cells {dam.sum()}", flush=True)
    return before, h, dam


def plot_setup(terrain, before, h, dam):
    extent = [BOUNDS[0] / 1000, BOUNDS[2] / 1000, BOUNDS[1] / 1000, BOUNDS[3] / 1000]
    fig, ax = plt.subplots(figsize=(8, 10), constrained_layout=True)
    ax.imshow(terrain, origin="lower", extent=extent, cmap="gist_earth", vmin=1400, vmax=3200)
    water = ax.imshow(np.ma.masked_where(h <= 0, h), origin="lower", extent=extent, cmap="Blues", vmin=0, vmax=80)
    ax.contour(dam, [.5], origin="lower", extent=extent, colors="red", linewidths=1)
    ax.scatter([2590.724], [1109.231], marker="+", color="black")
    ax.annotate("Siviez", (2590.724, 1109.231), xytext=(8, 5), textcoords="offset points")
    ax.set(xlabel="LV95 easting (km)", ylabel="LV95 northing (km)",
           title="Cleuson: terrain, idealized reservoir and dam footprint")
    fig.colorbar(water, ax=ax, label="Initial water depth (m)", shrink=.7)
    fig.savefig(CASE / "setup.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", type=float, default=DX, help="Cell width in metres")
    parser.add_argument("--output", type=Path, default=CASE, help="Prepared case directory")
    parser.add_argument("--end-time", type=float, default=END_TIME, help="Simulation duration in seconds")
    parser.add_argument("--snapshot-interval", type=float, default=SNAPSHOT_INTERVAL, help="Seconds between depth snapshots")
    parser.add_argument("--terrain-only", action="store_true")
    args = parser.parse_args()
    if args.dx <= 0 or not np.isfinite(args.dx):
        parser.error("--dx must be positive and finite")
    if not np.isfinite(args.end_time) or args.end_time <= 30:
        parser.error("--end-time must be greater than the 30 s breach time")
    if not np.isfinite(args.snapshot_interval) or args.snapshot_interval <= 0:
        parser.error("--snapshot-interval must be positive and finite")
    for length in (BOUNDS[2] - BOUNDS[0], BOUNDS[3] - BOUNDS[1]):
        if not np.isclose(length / args.dx, round(length / args.dx)):
            parser.error("--dx must divide both domain lengths")
    DX, CASE = args.dx, args.output
    END_TIME, SNAPSHOT_INTERVAL = args.end_time, args.snapshot_interval
    terrain, lake = download_terrain()
    if args.terrain_only:
        plot_setup(terrain, terrain, lake.astype(float), np.zeros_like(lake))
    else:
        before, h, dam = prepare_case(terrain, lake)
        plot_setup(terrain, before, h, dam)
