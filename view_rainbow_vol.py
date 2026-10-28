#!/usr/bin/env python3
"""Open and display a Rainbow 5 .vol radar moment with wradlib.

Usage:
    python3 view_rainbow_vol.py /path/to/2025030707191200dBZ.vol
    python3 view_rainbow_vol.py /path/to/file.vol --sweep 0 --save ppi.png

This standalone viewer takes one moment file. For multi-moment quality control,
use Radar Analyze in the web app and select matching moment files together.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wradlib.io as wio

RADAR_LON, RADAR_LAT = 112.760422, -7.410306
REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("WINDSHEAR_DATA_DIR", REPO_ROOT / "data")).expanduser()
RADAR_DATA_DIR = Path(os.environ.get(
    "TRAINING_TL_DATA", DATA_ROOT / "radar" / "2025-03-07"
)).expanduser()
KKOP_GEOJSON = Path(os.environ.get(
    "KKOP_GEOJSON", DATA_ROOT / "KKOP" / "kkop.geojson"
)).expanduser()


def as_list(value):
    return value if isinstance(value, list) else [value] if value is not None else []


def attr(obj, name, default=None):
    return obj.get("@" + name, obj.get(name, default))


def load_volume(path: Path):
    data = wio.read_rainbow(str(path))
    volume = data.get("volume", data)
    scan = volume.get("scan", {})
    slices = as_list(scan.get("slice"))
    if not slices:
        raise ValueError("No elevation sweeps found in this Rainbow volume")
    return volume, slices


def decode_sweep(sweep):
    slicedata = sweep.get("slicedata", {})
    rawmeta = slicedata.get("rawdata", {})
    raw = np.asarray(rawmeta.get("data"))
    if raw.ndim != 2:
        raise ValueError("Expected a 2-D decoded rawdata array")

    depth = int(attr(rawmeta, "depth", 8))
    minimum = float(attr(rawmeta, "min", 0.0))
    maximum = float(attr(rawmeta, "max", 255.0))
    values = minimum + raw.astype(np.float32) * (maximum - minimum) / (2**depth)

    rayinfo = slicedata.get("rayinfo", {})
    az_raw = np.asarray(rayinfo.get("data")).squeeze()
    if az_raw.size != raw.shape[0]:
        raise ValueError("Azimuth ray count doesn't match rawdata")
    nrays = float(attr(rayinfo, "rays", raw.shape[0]))
    azimuth = (az_raw.astype(float) * nrays / (2**float(attr(rayinfo, "depth", 16))) * float(attr(sweep, "anglestep", 1.0))) % 360

    # Rainbow provides range step in km; plotted as km from radar.
    range_step = float(attr(sweep, "rangestep", 0.252))
    ranges = np.arange(raw.shape[1], dtype=float) * range_step
    moment = str(attr(rawmeta, "type", "Radar value"))
    elevation = float(sweep.get("posangle", np.nan))
    return values, azimuth, ranges, moment, elevation


def display(path: Path, sweep_number: int | None = None, save: Path | None = None, snr_min: float = 7.0):
    volume, slices = load_volume(path)
    if sweep_number is not None:
        if not 0 <= sweep_number < len(slices):
            raise ValueError(f"Sweep index out of range: choose 0 through {len(slices)-1}")
        selected = [(sweep_number, slices[sweep_number])]
    else:
        selected = list(enumerate(slices))

    print(f"File: {path}")
    print(f"Volume time: {volume.get('@datetime', 'not present in header')}")
    print(f"Sweeps: {len(slices)}")
    print("Note: one .vol file contains one radar moment/product; no cross-moment QC is applied.")

    n = len(selected)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    last_mesh = None
    for ax, (index, sweep) in zip(axes.flat, selected):
        values, azimuth, ranges, moment, elevation = decode_sweep(sweep)
        values, filter_text = apply_filters(values, moment, snr_min)
        # Convert native gate-cell edges to WGS84 and render every gate.
        az_edges = circular_edges(np.deg2rad(azimuth))
        range_step = float(attr(sweep, "rangestep", 0.252))
        range_edges = np.maximum(0.0, np.arange(values.shape[1] + 1, dtype=float) * range_step - range_step / 2)
        lat_edges, lon_edges = polar_edges_to_latlon(az_edges, range_edges)
        vmin, vmax = (0, 75) if moment == "dBZ" else ((0, 6) if moment == "ZDR" else (None, None))
        last_mesh = ax.pcolormesh(lon_edges, lat_edges, values, shading="flat", cmap="turbo", rasterized=True, linewidth=0, vmin=vmin, vmax=vmax)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°; negative = °S)")
        ax.set_aspect(1 / np.cos(np.deg2rad(RADAR_LAT)))
        add_map_context(ax)
        ax.set_title(f"Sweep {index}: {moment} · elevation {elevation:.2f}°\n{values.shape[0]} rays × {values.shape[1]} gates")
        ax.plot(RADAR_LON, RADAR_LAT, marker="^", color="black", markeredgecolor="white", markersize=9, zorder=15)
        ax.annotate("SBY radar", (RADAR_LON, RADAR_LAT), xytext=(6, 6), textcoords="offset points", color="black", fontsize=8, weight="bold", zorder=16)
        finite = values[np.isfinite(values)]
        if finite.size:
            print(f"  sweep {index}: elevation={elevation:.2f} deg, moment={moment}, filter={filter_text}, shape={values.shape}, valid={finite.size}, min={np.min(finite):.2f}, p05={np.percentile(finite,5):.2f}, median={np.median(finite):.2f}, p95={np.percentile(finite,95):.2f}, max={np.max(finite):.2f}")
    for ax in axes.flat[len(selected):]:
        ax.set_visible(False)
    if last_mesh is not None:
        fig.colorbar(last_mesh, ax=[a for a in axes.flat[:len(selected)]], shrink=0.78, pad=0.08, label="Rainbow moment value (units from file metadata)")
    fig.suptitle(f"Rainbow weather radar · {path.name}", fontsize=14)
    fig.subplots_adjust(top=0.90, right=0.88, hspace=0.32, wspace=0.25)
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=160, bbox_inches="tight")
        print(f"Saved display: {save}")
    else:
        plt.show()


def add_map_context(ax):
    """Plot geographic context, including the Juanda KKOP overlay."""
    bounds = (112.1, 113.25, -8.15, -6.8)
    cache = Path(os.environ.get(
        "WINDSHEAR_MAP_CACHE_DIR", DATA_ROOT / "map_cache"
    )).expanduser()
    try:
        ocean_path = cache / "ne_10m_ocean/ne_10m_ocean.shp"
        if ocean_path.exists():
            for segment in shapefile_segments(ocean_path, bounds):
                ax.plot(segment[:, 0], segment[:, 1], color="#08799c", linewidth=1.15,
                        alpha=0.95, solid_capstyle="round", zorder=8)
            ax.text(112.60, -7.02, "Java Sea", color="#155b78", fontsize=10,
                    weight="bold", ha="center", zorder=20)
    except Exception as exc:
        print(f"Map note: coastline layer unavailable: {exc}")
    try:
        admin_path = cache / "gadm41_IDN/gadm41_IDN_1.shp"
        if admin_path.exists():
            for segment in shapefile_segments(admin_path, bounds):
                ax.plot(segment[:, 0], segment[:, 1], color="#202b37", linewidth=1.0,
                        alpha=0.95, solid_capstyle="round", zorder=9)
            ax.text(112.91, -7.52, "East Java", color="#3f382f", fontsize=10,
                    weight="bold", ha="center", zorder=20)
    except Exception as exc:
        print(f"Map note: administrative boundaries unavailable: {exc}")
    try:
        with KKOP_GEOJSON.open(encoding="utf-8") as source:
            collection = json.load(source)
        drew_kkop = False
        for feature in collection.get("features", []):
            geometry = feature.get("geometry") or {}
            kind, coordinates = geometry.get("type"), geometry.get("coordinates", [])
            if kind == "LineString":
                lines = [coordinates]
            elif kind == "MultiLineString":
                lines = coordinates
            else:
                continue
            for line in lines:
                points = np.asarray(line, dtype=float)
                if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2:
                    continue
                # Outline ensures the designated boundary stays readable over
                # both radar returns and the underlying basemap.
                ax.plot(points[:, 0], points[:, 1], color="white", linewidth=3.0,
                        alpha=0.95, zorder=11)
                ax.plot(points[:, 0], points[:, 1], color="#ff3b30", linewidth=1.7,
                        linestyle="--", alpha=1.0,
                        label="KKOP Juanda" if not drew_kkop else None, zorder=12)
                drew_kkop = True
        if drew_kkop:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    except Exception as exc:
        print(f"Map note: KKOP overlay unavailable: {exc}")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.grid(True, color="#46525b", alpha=0.35, linewidth=0.5, zorder=2)


def circular_edges(azimuth_rad):
    """Create continuous azimuth edges, including the 360°/0° wrap."""
    az = np.unwrap(np.asarray(azimuth_rad, dtype=float))
    edges = np.empty(az.size + 1, dtype=float)
    edges[1:-1] = (az[:-1] + az[1:]) / 2
    edges[0] = az[0] - (az[1] - az[0]) / 2
    edges[-1] = az[-1] + (az[-1] - az[-2]) / 2
    return edges


def apply_filters(values, moment, snr_min):
    """Apply the requested range to the matching single-moment product."""
    data = np.asarray(values, dtype=float).copy()
    valid = np.isfinite(data)
    if moment == "SNR":
        valid &= data >= snr_min
        label = f"SNR >= {snr_min:g} dB"
    elif moment == "dBZ":
        valid &= (data >= 0) & (data <= 75)
        label = "0 <= dBZ <= 75"
    elif moment == "RhoHV":
        valid &= data > 0.85
        label = "RhoHV > 0.85"
    elif moment == "ZDR":
        valid &= (data >= 0) & (data <= 6)
        label = "0 <= ZDR <= 6 dB"
    else:
        label = f"no single-moment filter for {moment}"
    data[~valid] = np.nan
    return data, label


def polar_edges_to_latlon(azimuth_edges, range_edges_km):
    """Spherical forward geodesic from the radar origin for polar cell edges."""
    earth_radius_km = 6371.0088
    az = azimuth_edges[:, None]
    distance = np.asarray(range_edges_km)[None, :] / earth_radius_km
    lat1, lon1 = np.deg2rad(RADAR_LAT), np.deg2rad(RADAR_LON)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(distance) + np.cos(lat1) * np.sin(distance) * np.cos(az))
    lon2 = lon1 + np.arctan2(np.sin(az) * np.sin(distance) * np.cos(lat1),
                             np.cos(distance) - np.sin(lat1) * np.sin(lat2))
    return np.rad2deg(lat2), np.rad2deg(lon2)


def shapefile_segments(path: Path, bounds):
    """Read ESRI polyline/polygon parts without optional GIS packages."""
    data = path.read_bytes()
    if len(data) < 100:
        return
    pos = 100
    xmin, xmax, ymin, ymax = bounds
    while pos + 8 <= len(data):
        content_bytes = struct.unpack_from(">i", data, pos + 4)[0] * 2
        pos += 8
        end = pos + content_bytes
        if end > len(data) or content_bytes < 44:
            break
        shape_type = struct.unpack_from("<i", data, pos)[0]
        if shape_type not in (3, 5, 13, 15, 23, 25):
            pos = end
            continue
        box = struct.unpack_from("<4d", data, pos + 4)
        num_parts, num_points = struct.unpack_from("<2i", data, pos + 36)
        # Shapefile record bounds are (xmin, ymin, xmax, ymax). Skip only
        # records wholly outside the map; the old comparison rejected any
        # global polygon whose bounding box extended beyond the view.
        if box[0] > xmax or box[2] < xmin or box[1] > ymax or box[3] < ymin:
            pos = end
            continue
        parts_offset = pos + 44
        point_offset = parts_offset + num_parts * 4
        parts = struct.unpack_from("<%di" % num_parts, data, parts_offset) if num_parts else ()
        coords = np.frombuffer(data, dtype="<f8", count=num_points * 2, offset=point_offset).reshape(-1, 2)
        cuts = list(parts) + [num_points]
        for first, last in zip(cuts[:-1], cuts[1:]):
            segment = coords[first:last]
            if len(segment) > 1:
                yield segment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vol_file", type=Path, nargs="?", help="Rainbow .vol file; omit to choose a file in a dialog")
    parser.add_argument("--sweep", type=int, help="Display one zero-based sweep; default displays all sweeps")
    parser.add_argument("--save", type=Path, help="Save figure to an image instead of opening an interactive window")
    parser.add_argument("--snr-min", type=float, choices=(2, 3, 4, 5, 6, 7), default=7, help="SNR display threshold (2-7 dB; default 7)")
    args = parser.parse_args()
    vol_file = args.vol_file
    if vol_file is None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            chosen = filedialog.askopenfilename(
                title="Choose a Rainbow weather radar volume",
                initialdir=str(RADAR_DATA_DIR.parent),
                filetypes=[("Rainbow radar volumes", "*.vol"), ("All files", "*.*")],
            )
            root.destroy()
        except Exception as exc:
            parser.error(f"Could not open file picker ({exc}); supply the .vol path as an argument")
        if not chosen:
            print("No file selected.")
            return
        vol_file = Path(chosen)
    if vol_file.suffix.lower() != ".vol":
        parser.error("input file must have a .vol extension")
    if not vol_file.is_file():
        parser.error(f"file not found: {vol_file}")
    display(vol_file, args.sweep, args.save, args.snr_min)


if __name__ == "__main__":
    main()
