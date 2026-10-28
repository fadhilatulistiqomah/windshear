#!/usr/bin/env python3
"""
02_Himawari9_AMV_CTV699.py

Dense-grid Himawari-9 cloud-motion / AMV-like analysis for the CTV699 case.

IMPORTANT
---------
This script produces a DENSE CLOUD-MOTION VECTOR (CMV) field, AMV-like in
workflow, but it is NOT the official JMA AMV product. Official AMV requires
cloud-feature tracking, height assignment and operational quality control.
Here the vector is derived from image motion. Without a validated cloud-height
assignment, do not interpret the vector as an independent wind measurement at
every pixel.

Data
----
Himawari-9 AHI full-disk NetCDF R21 files containing:
  B03 = albedo_03 (0.64 um)
  B07 = tbb_07   (3.9 um)
  B13 = tbb_13   (10.4 um)

The script reads only the small spatial subset needed for the CTV699 case;
it does NOT load the ~4 GB global array into memory.

Case
----
Date   : 2025-03-07
Time   : 07:00-08:00 UTC
Domain : 112.0-113.5 E, 6.5-8.0 S

Outputs
-------
For each central time (07:10 ... 07:50 UTC):
  * dense CSV with one row per grid point
  * dense NetCDF with u/v, speed, direction, correlation and QC
  * speed PNG with dense vectors overlaid
  * direction PNG
  * QC-diagnostics PNG (good / acceptable / rejected / not trackable)
  * summary CSV
Maps use pale backgrounds, translucent data shading, and show accepted vectors only.

Dependencies
------------
numpy, pandas, xarray, scipy, matplotlib
Optional: geopandas, shapely, cartopy (not required)

The map overlay is designed to work with the user's Training_TL paths.
"""

from __future__ import annotations

import math
import re
import warnings
import io
import base64
import contextlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    cv2 = None
    HAVE_CV2 = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# USER CONFIGURATION
# =============================================================================
DATE = "20250307"
START_UTC = "0700"
END_UTC = "0800"
INTERVAL_MIN = 10

INPUT_DIR = Path(
    "/Users/aliwardhana/Downloads/Training_TL/00_data/Himawari_9/20250307"
)
OUTPUT_DIR = Path(
    "/Users/aliwardhana/Downloads/Training_TL/01_output/Himawari_9/CTV699_20250307_Dense_AMV"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CTV699 / Juanda domain
LON_MIN, LON_MAX = 112.0, 113.5
LAT_MIN, LAT_MAX = -8.0, -6.5

# Dense-grid spacing. 1 = every native subset pixel; 2 = every second pixel.
# The map is still dense, while the default avoids excessive computation.
GRID_STRIDE = 1

# Local block matching parameters, in pixels.
# Small window tracks fine texture; large window is more stable in deep cloud.
SMALL_HALF_WINDOW = 4       # 9 x 9 target window
LARGE_HALF_WINDOW = 7       # 15 x 15 target window
SEARCH_RADIUS = 10          # +/- 10 pixels per 10 min
MIN_CORRELATION = 0.55
GOOD_CORRELATION = 0.70
MAX_VECTOR_DISAGREEMENT_MS = 8.0
MAX_SPEED_MS = 42.0
MAX_DISPLACEMENT_KM = 25.0
MIN_TEXTURE_STD_K = 0.55  # slightly relaxed to reduce avoidable gaps near Juanda

# Cloud mask. B13 colder than this is considered cloud for this product.
# Warm cloud / surface pixels remain invalid unless texture + motion support it.
CLOUD_TBB_MAX_K = 280.0  # broader cloud mask; QC still controls vector acceptance
DEEP_CLOUD_TBB_MAX_K = 250.0

# Three-channel support. B13 is the primary tracking field.
USE_B03_SUPPORT = True
USE_B07_SUPPORT = True

# Plot vector density. This only controls plotting, not CSV/NetCDF density.
QUIVER_STRIDE = 4

# Radar / Juanda approximate plotting reference
JUANDA_LAT = -7.3797
JUANDA_LON = 112.7873
RWY10_HEADING_DEG = 99.3
RWY28_HEADING_DEG = 279.3

# Optional local GIS files. If missing, the script continues.
KKOP_GEOJSON = Path(
    "/Users/aliwardhana/Downloads/Training_TL/00_data/KKOP/kkop.geojson"
)
MAP_CACHE_DIR = Path(
    "/Users/aliwardhana/Downloads/Training_TL/01_output/Radar/_map_cache"
)
COASTLINE_SHP = MAP_CACHE_DIR / "ne_10m_ocean" / "ne_10m_ocean.shp"
GADM1_SHP = MAP_CACHE_DIR / "gadm41_IDN" / "gadm41_IDN_1.shp"
GADM2_SHP = MAP_CACHE_DIR / "gadm41_IDN" / "gadm41_IDN_2.shp"

# =============================================================================
# FILE / TIME HELPERS
# =============================================================================
FILE_RE = re.compile(
    r"NC_H09_(\d{8})_(\d{4})_R21_FLDK\.07001_06001(?:_ROI_[^/]*)?\.nc$"
)


def timestamp_from_filename(path: Path) -> datetime | None:
    m = FILE_RE.search(path.name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M").replace(
        tzinfo=timezone.utc
    )


def requested_times() -> list[datetime]:
    start = datetime.strptime(DATE + START_UTC, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    end = datetime.strptime(DATE + END_UTC, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    out = []
    t = start
    while t <= end:
        out.append(t)
        t = t + pd.Timedelta(minutes=INTERVAL_MIN)
    return out


def discover_files() -> dict[datetime, Path]:
    result: dict[datetime, Path] = {}
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    for p in sorted(INPUT_DIR.glob("NC_H09_*.nc")):
        ts = timestamp_from_filename(p)
        if ts is not None:
            result[ts] = p

    return result


# =============================================================================
# HIMAWARI READING
# =============================================================================

def _slice_latitude(coord: xr.DataArray, lo: float, hi: float) -> slice:
    vals = np.asarray(coord.values)
    ascending = bool(vals[0] < vals[-1])
    return slice(lo, hi) if ascending else slice(hi, lo)


def read_subset(path: Path) -> dict[str, np.ndarray]:
    """Read only B03/B07/B13 in the CTV699 domain."""
    print(f"    Reading subset: {path.name}")
    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        lat_slice = _slice_latitude(ds["latitude"], LAT_MIN, LAT_MAX)
        sub = ds[["tbb_13", "tbb_07", "albedo_03"]].sel(
            latitude=lat_slice,
            longitude=slice(LON_MIN, LON_MAX),
        )

        # Load only the small subset, then close the 4-GB source dataset.
        sub = sub.load()
        lat = np.asarray(sub["latitude"].values, dtype=float)
        lon = np.asarray(sub["longitude"].values, dtype=float)
        b13 = np.asarray(sub["tbb_13"].values, dtype=np.float32)
        b07 = np.asarray(sub["tbb_07"].values, dtype=np.float32)
        b03 = np.asarray(sub["albedo_03"].values, dtype=np.float32)
    finally:
        ds.close()

    # Ensure 2-D arrays are [lat, lon].
    b13 = np.squeeze(b13)
    b07 = np.squeeze(b07)
    b03 = np.squeeze(b03)

    if b13.ndim != 2:
        raise RuntimeError(f"Unexpected B13 dimensions in {path.name}: {b13.shape}")

    # Remove impossible values.
    b13[~np.isfinite(b13)] = np.nan
    b07[~np.isfinite(b07)] = np.nan
    b03[~np.isfinite(b03)] = np.nan

    return {"lat": lat, "lon": lon, "b13": b13, "b07": b07, "b03": b03}


# =============================================================================
# IMAGE PRE-PROCESSING
# =============================================================================

def robust_normalize(a: np.ndarray, low=2.0, high=98.0) -> np.ndarray:
    finite = np.isfinite(a)
    out = np.zeros_like(a, dtype=np.float32)
    if not np.any(finite):
        return out
    p1, p2 = np.nanpercentile(a[finite], [low, high])
    if not np.isfinite(p1) or not np.isfinite(p2) or p2 <= p1:
        return out
    out = (a - p1) / (p2 - p1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def make_tracking_image(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return primary B13 texture field and cloud/texture mask."""
    b13 = data["b13"]
    b07 = data["b07"]
    b03 = data["b03"]

    # For B13, colder cloud = larger tracking value.
    b13_cold = np.clip((275.0 - b13) / 60.0, 0.0, 1.0)
    b13_cold = np.where(np.isfinite(b13), b13_cold, np.nan)

    # Light texture enhancement. The primary field remains B13.
    # B03/B07 are supporting channels, not independent wind observations.
    grad_y, grad_x = np.gradient(np.nan_to_num(b13, nan=np.nanmedian(b13)))
    gradient_mag = np.hypot(grad_x, grad_y)
    texture = uniform_filter(np.nan_to_num(b13, nan=np.nanmedian(b13)), size=5)
    texture2 = uniform_filter(
        np.nan_to_num(b13 * b13, nan=np.nanmedian(b13) ** 2), size=5
    )
    local_std = np.sqrt(np.maximum(texture2 - texture * texture, 0.0))

    # Small amount of multi-channel support where valid.
    feature = b13_cold.copy()
    if USE_B07_SUPPORT:
        b07n = robust_normalize(b07)
        feature = 0.85 * feature + 0.15 * b07n
    if USE_B03_SUPPORT:
        b03n = robust_normalize(b03)
        feature = 0.90 * feature + 0.10 * b03n

    cloud_mask = (
        np.isfinite(b13)
        & (b13 <= CLOUD_TBB_MAX_K)
        & (local_std >= MIN_TEXTURE_STD_K)
    )

    return feature.astype(np.float32), cloud_mask


# =============================================================================
# DENSE LOCAL CORRELATION
# =============================================================================

def _local_ncc_best(
    image1: np.ndarray,
    image2: np.ndarray,
    y: int,
    x: int,
    half: int,
    radius: int,
) -> tuple[float, float, float]:
    """Find best displacement of one target window in image2.

    Returns (dy_pixels, dx_pixels, ncc).

    The implementation is deliberately dependency-light. It computes normalized
    cross-correlation for each integer displacement around the target.
    """
    h, w = image1.shape
    if y - half < 0 or y + half >= h or x - half < 0 or x + half >= w:
        return np.nan, np.nan, np.nan

    template = image1[y - half : y + half + 1, x - half : x + half + 1]
    if not np.all(np.isfinite(template)):
        return np.nan, np.nan, np.nan

    t = template.astype(np.float64)
    tmean = t.mean()
    t0 = t - tmean
    tnorm = math.sqrt(float(np.sum(t0 * t0)))
    if tnorm <= 1e-8:
        return np.nan, np.nan, np.nan

    best = (-np.inf, np.nan, np.nan)

    for dy in range(-radius, radius + 1):
        yy = y + dy
        if yy - half < 0 or yy + half >= h:
            continue
        for dx in range(-radius, radius + 1):
            xx = x + dx
            if xx - half < 0 or xx + half >= w:
                continue

            patch = image2[yy - half : yy + half + 1, xx - half : xx + half + 1]
            if not np.all(np.isfinite(patch)):
                continue

            p = patch.astype(np.float64)
            p0 = p - p.mean()
            pnorm = math.sqrt(float(np.sum(p0 * p0)))
            if pnorm <= 1e-8:
                continue

            corr = float(np.sum(t0 * p0) / (tnorm * pnorm))
            if corr > best[0]:
                best = (corr, dy, dx)

    if not np.isfinite(best[0]):
        return np.nan, np.nan, np.nan

    return float(best[1]), float(best[2]), float(best[0])


def dense_match(
    image1: np.ndarray,
    image2: np.ndarray,
    cloud_mask1: np.ndarray,
    cloud_mask2: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    half_window: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense displacement in pixels: dy, dx, local NCC quality.

    OpenCV Farneback is used when available because it provides a true dense
    optical-flow estimate at every pixel and is dramatically faster than
    running an independent block search at every grid point. A dependency-light
    NCC fallback is retained for environments without OpenCV.
    """
    ny, nx = image1.shape

    def prep(a):
        med = np.nanmedian(a) if np.any(np.isfinite(a)) else 0.0
        z = np.nan_to_num(a, nan=med).astype(np.float32)
        lo, hi = np.nanpercentile(z, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((z - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)

    if HAVE_CV2:
        a = prep(image1)
        b = prep(image2)

        flows = []
        for winsize in (15, 25):
            flow = cv2.calcOpticalFlowFarneback(
                a, b, None,
                pyr_scale=0.5, levels=3, winsize=winsize,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            flows.append(flow.astype(np.float32))

        flow = np.nanmean(np.stack(flows), axis=0)
        dx_out = flow[..., 0]
        dy_out = flow[..., 1]

        # Validate the predicted displacement using local normalized
        # cross-correlation after remapping image2 to the predicted positions.
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
        map_x = xx + dx_out
        map_y = yy + dy_out
        b_warp = cv2.remap(
            np.nan_to_num(image2, nan=float(np.nanmedian(image2))).astype(np.float32),
            map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        a0 = np.nan_to_num(image1, nan=float(np.nanmedian(image1))).astype(np.float32)
        w = max(2 * half_window + 1, 5)
        ma = uniform_filter(a0, size=w, mode="nearest")
        mb = uniform_filter(b_warp, size=w, mode="nearest")
        maa = uniform_filter(a0 * a0, size=w, mode="nearest")
        mbb = uniform_filter(b_warp * b_warp, size=w, mode="nearest")
        mab = uniform_filter(a0 * b_warp, size=w, mode="nearest")
        cov = mab - ma * mb
        va = np.maximum(maa - ma * ma, 0.0)
        vb = np.maximum(mbb - mb * mb, 0.0)
        corr_out = cov / np.sqrt(va * vb + 1e-12)
        corr_out = np.clip(corr_out, -1.0, 1.0).astype(np.float32)

        # Dense field is meaningful only where the source and destination are
        # cloud/texture pixels.
        yy_i = np.rint(np.arange(ny)[:, None] + dy_out).astype(int)
        xx_i = np.rint(np.arange(nx)[None, :] + dx_out).astype(int)
        yy_i = np.clip(yy_i, 0, ny - 1)
        xx_i = np.clip(xx_i, 0, nx - 1)
        dest_cloud = cloud_mask2[yy_i, xx_i]
        source_cloud = cloud_mask1
        valid = source_cloud & dest_cloud & np.isfinite(dx_out) & np.isfinite(dy_out)
        dx_out = np.where(valid, dx_out, np.nan).astype(np.float32)
        dy_out = np.where(valid, dy_out, np.nan).astype(np.float32)
        corr_out = np.where(valid, corr_out, np.nan).astype(np.float32)
        return dy_out, dx_out, corr_out

    # Fallback: sparse block matching. It is intentionally conservative and
    # still writes vectors on the configured dense grid where matching succeeds.
    dy_out = np.full((ny, nx), np.nan, dtype=np.float32)
    dx_out = np.full((ny, nx), np.nan, dtype=np.float32)
    corr_out = np.full((ny, nx), np.nan, dtype=np.float32)
    for y in range(half_window + radius, ny - half_window - radius, GRID_STRIDE):
        for x in range(half_window + radius, nx - half_window - radius, GRID_STRIDE):
            if not cloud_mask1[y, x]:
                continue
            dy, dx, corr = _local_ncc_best(image1, image2, y, x, half_window, radius)
            if not np.isfinite(corr) or corr < MIN_CORRELATION:
                continue
            yy = int(round(y + dy))
            xx = int(round(x + dx))
            if 0 <= yy < ny and 0 <= xx < nx and cloud_mask2[yy, xx]:
                dy_out[y, x] = dy
                dx_out[y, x] = dx
                corr_out[y, x] = corr
    return dy_out, dx_out, corr_out


# =============================================================================
# GEOMETRY / VECTOR CONVERSION
# =============================================================================

def grid_spacing_m(lat: np.ndarray, lon: np.ndarray) -> tuple[float, float]:
    """Approximate native grid spacing in metres."""
    dlat = float(np.nanmedian(np.abs(np.diff(lat))))
    dlon = float(np.nanmedian(np.abs(np.diff(lon))))
    lat0 = float(np.nanmean(lat))
    dy = dlat * 111_320.0
    dx = dlon * 111_320.0 * math.cos(math.radians(lat0))
    return dy, dx


def displacement_to_uv(
    dy_pix: np.ndarray,
    dx_pix: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    interval_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dy_m_per_px, dx_m_per_px = grid_spacing_m(lat, lon)
    u = dx_pix * dx_m_per_px / interval_seconds
    # Latitude may be descending. A positive pixel dy means movement toward the
    # next array row, which is south when latitude is descending.
    lat_desc = lat[0] > lat[-1]
    north_sign = -1.0 if lat_desc else 1.0
    v = dy_pix * dy_m_per_px * north_sign / interval_seconds

    speed = np.hypot(u, v)
    direction_to = (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0
    direction_from = (direction_to + 180.0) % 360.0
    return u, v, speed, direction_to, direction_from


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# =============================================================================
# THREE-IMAGE / QC PROCESSING
# =============================================================================

def process_triplet(
    t_prev: datetime,
    t_mid: datetime,
    t_next: datetime,
    data_prev: dict,
    data_mid: dict,
    data_next: dict,
) -> dict[str, np.ndarray]:
    print(f"\n  Dense tracking {t_prev:%H:%M} -> {t_mid:%H:%M} -> {t_next:%H:%M} UTC")

    f_prev, m_prev = make_tracking_image(data_prev)
    f_mid, m_mid = make_tracking_image(data_mid)
    f_next, m_next = make_tracking_image(data_next)

    lat = data_mid["lat"]
    lon = data_mid["lon"]
    interval_s = INTERVAL_MIN * 60.0

    print("    Forward small-window matching...")
    dy_f_s, dx_f_s, c_f_s = dense_match(
        f_mid, f_next, m_mid, m_next, lat, lon,
        SMALL_HALF_WINDOW, SEARCH_RADIUS,
    )
    print("    Backward small-window matching...")
    dy_b_s, dx_b_s, c_b_s = dense_match(
        f_prev, f_mid, m_prev, m_mid, lat, lon,
        SMALL_HALF_WINDOW, SEARCH_RADIUS,
    )

    print("    Forward large-window matching...")
    dy_f_l, dx_f_l, c_f_l = dense_match(
        f_mid, f_next, m_mid, m_next, lat, lon,
        LARGE_HALF_WINDOW, SEARCH_RADIUS,
    )
    print("    Backward large-window matching...")
    dy_b_l, dx_b_l, c_b_l = dense_match(
        f_prev, f_mid, m_prev, m_mid, lat, lon,
        LARGE_HALF_WINDOW, SEARCH_RADIUS,
    )

    # Average small/large estimates when available.
    def combine(a, b):
        return np.nanmean(np.stack([a, b]), axis=0)

    dy_f = combine(dy_f_s, dy_f_l)
    dx_f = combine(dx_f_s, dx_f_l)
    c_f = np.nanmean(np.stack([c_f_s, c_f_l]), axis=0)

    dy_b = combine(dy_b_s, dy_b_l)
    dx_b = combine(dx_b_s, dx_b_l)
    c_b = np.nanmean(np.stack([c_b_s, c_b_l]), axis=0)

    # Forward estimate is MID -> NEXT.
    u_f, v_f, speed_f, dir_to_f, dir_from_f = displacement_to_uv(
        dy_f, dx_f, lat, lon, interval_s
    )
    # Backward matching is PREV -> MID, which is also the central-time velocity.
    u_b, v_b, speed_b, dir_to_b, dir_from_b = displacement_to_uv(
        dy_b, dx_b, lat, lon, interval_s
    )

    # Central estimate = mean of forward and backward motion.
    u = np.nanmean(np.stack([u_f, u_b]), axis=0)
    v = np.nanmean(np.stack([v_f, v_b]), axis=0)
    speed = np.hypot(u, v)
    direction_to = (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0
    direction_from = (direction_to + 180.0) % 360.0

    disagreement = np.hypot(u_f - u_b, v_f - v_b)
    corr = np.nanmean(np.stack([c_f, c_b]), axis=0)

    # Native grid displacement / speed limits.
    displacement_km = speed * interval_s / 1000.0
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    dist_airport = haversine_km(lat2d, lon2d, JUANDA_LAT, JUANDA_LON)

    # Use B13 texture and cloud criteria for final validity.
    b13 = data_mid["b13"]
    local_mean = uniform_filter(np.nan_to_num(b13, nan=np.nanmedian(b13)), size=5)
    local_sq = uniform_filter(
        np.nan_to_num(b13 * b13, nan=np.nanmedian(b13) ** 2), size=5
    )
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0))

    valid = (
        np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(corr)
        & np.isfinite(disagreement)
        & (corr >= MIN_CORRELATION)
        & (speed <= MAX_SPEED_MS)
        & (displacement_km <= MAX_DISPLACEMENT_KM)
        & (disagreement <= MAX_VECTOR_DISAGREEMENT_MS)
        & (b13 <= CLOUD_TBB_MAX_K)
        & (local_std >= MIN_TEXTURE_STD_K)
    )

    quality = np.zeros_like(speed, dtype=np.int8)
    quality[valid & (corr >= GOOD_CORRELATION) & (disagreement <= 4.0)] = 2
    quality[valid & (quality == 0)] = 1

    # Diagnostic classes separate missing input/texture from candidate vectors
    # rejected by motion QC. This is diagnostic only; it does not invent vectors.
    tracking_attempted = (
        np.isfinite(b13)
        & (b13 <= CLOUD_TBB_MAX_K)
        & (local_std >= MIN_TEXTURE_STD_K)
    )
    qc_rejected = tracking_attempted & ~valid
    not_trackable = ~tracking_attempted

    # Deep-convective flag is informative, not a quality requirement.
    deep_flag = np.isfinite(b13) & (b13 <= DEEP_CLOUD_TBB_MAX_K)

    return {
        "lat": lat,
        "lon": lon,
        "u": u.astype(np.float32),
        "v": v.astype(np.float32),
        "speed": speed.astype(np.float32),
        "direction_to": direction_to.astype(np.float32),
        "direction_from": direction_from.astype(np.float32),
        "corr": corr.astype(np.float32),
        "disagreement": disagreement.astype(np.float32),
        "displacement_km": displacement_km.astype(np.float32),
        "quality": quality,
        "tracking_attempted": tracking_attempted,
        "qc_rejected": qc_rejected,
        "not_trackable": not_trackable,
        "deep_flag": deep_flag,
        "tbb13": b13.astype(np.float32),
        "tbb07": data_mid["b07"].astype(np.float32),
        "b03": data_mid["b03"].astype(np.float32),
        "distance_km": dist_airport.astype(np.float32),
    }


# =============================================================================
# GIS / MAP HELPERS
# =============================================================================

def add_circle(ax, radius_km, **kwargs):
    theta = np.linspace(0, 2 * np.pi, 361)
    lat = JUANDA_LAT + (radius_km / 111.32) * np.sin(theta)
    lon = JUANDA_LON + (radius_km / (111.32 * np.cos(np.radians(JUANDA_LAT)))) * np.cos(theta)
    ax.plot(lon, lat, **kwargs)


def add_gis(ax):
    """Draw subtle coastline/admin boundaries above the raster layer."""
    try:
        import geopandas as gpd
    except Exception:
        return

    # The Natural Earth file named ne_10m_ocean is a polygon dataset, not a
    # coastline-only line file. Plot only its boundary here: filling it can
    # obscure the satellite raster and create a black/gray sea background.
    if COASTLINE_SHP.exists():
        try:
            ocean = gpd.read_file(COASTLINE_SHP).to_crs("EPSG:4326")
            ocean.boundary.plot(ax=ax, color="#52616B", linewidth=0.65, zorder=5)
        except Exception as exc:
            print(f"    Coastline warning: {exc}")

    for path, color, width, alpha in [
        (GADM1_SHP, "#626B70", 0.55, 0.85),
        (GADM2_SHP, "#899196", 0.30, 0.65),
    ]:
        if path.exists():
            try:
                g = gpd.read_file(path).to_crs("EPSG:4326")
                g.boundary.plot(ax=ax, color=color, linewidth=width, alpha=alpha, zorder=5)
            except Exception as exc:
                print(f"    GIS warning: {path.name}: {exc}")

    if KKOP_GEOJSON.exists():
        try:
            g = gpd.read_file(KKOP_GEOJSON).to_crs("EPSG:4326")
            g.boundary.plot(ax=ax, color="#A000A8", linewidth=1.0, alpha=0.95, zorder=7)
        except Exception as exc:
            print(f"    KKOP warning: {exc}")


def add_juanda(ax):
    ax.scatter(
        [JUANDA_LON], [JUANDA_LAT], s=45, marker="*", c="yellow",
        edgecolors="black", linewidths=0.8, zorder=10,
    )
    ax.text(
        JUANDA_LON + 0.02, JUANDA_LAT + 0.02,
        "Juanda (SUB)", fontsize=9, weight="bold", zorder=11,
    )

    # Approximate runway line for visual orientation only.
    length_km = 1.2
    for heading, label, offset in [
        (RWY10_HEADING_DEG, "RWY 10", 0.0),
        (RWY28_HEADING_DEG, "RWY 28", 0.0),
    ]:
        th = math.radians(heading)
        dlat = (length_km / 2) * math.cos(th) / 111.32
        dlon = (length_km / 2) * math.sin(th) / (111.32 * math.cos(math.radians(JUANDA_LAT)))
        ax.plot(
            [JUANDA_LON - dlon, JUANDA_LON + dlon],
            [JUANDA_LAT - dlat, JUANDA_LAT + dlat],
            color="black", linewidth=3.0, solid_capstyle="butt", zorder=9,
        )

    ax.text(JUANDA_LON + 0.02, JUANDA_LAT - 0.035, "10", fontsize=9, weight="bold")
    ax.text(JUANDA_LON - 0.045, JUANDA_LAT + 0.015, "28", fontsize=9, weight="bold")


def setup_ax(ax):
    """Set a light, neutral geographic background before raster plotting."""
    ax.set_facecolor("#DCEAF2")  # pale blue sea / uncovered map area
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°S)")
    yt = ax.get_yticks()
    ax.set_yticklabels([f"{abs(y):.1f}°S" for y in yt])
    ax.grid(True, color="#FFFFFF", alpha=0.65, linewidth=0.6, zorder=1)


def add_map_overlays(ax):
    """Add geographic/airport context above the semi-transparent raster."""
    add_gis(ax)
    add_circle(ax, 25, color="#A000A8", linestyle="--", linewidth=0.85, alpha=0.8, zorder=6)
    add_circle(ax, 50, color="#A000A8", linestyle=":", linewidth=0.85, alpha=0.8, zorder=6)
    add_juanda(ax)


# =============================================================================
# OUTPUTS
# =============================================================================

def save_csv(result: dict, t_mid: datetime):
    lat2d, lon2d = np.meshgrid(result["lat"], result["lon"], indexing="ij")
    ny, nx = lat2d.shape
    yy, xx = np.indices((ny, nx))

    df = pd.DataFrame({
        "time_utc": t_mid.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "row": yy.ravel(),
        "col": xx.ravel(),
        "latitude": lat2d.ravel(),
        "longitude": lon2d.ravel(),
        "tbb_13_K": result["tbb13"].ravel(),
        "tbb_07_K": result["tbb07"].ravel(),
        "albedo_03": result["b03"].ravel(),
        "u_mps": result["u"].ravel(),
        "v_mps": result["v"].ravel(),
        "speed_mps": result["speed"].ravel(),
        "motion_to_deg": result["direction_to"].ravel(),
        "motion_from_deg": result["direction_from"].ravel(),
        "correlation": result["corr"].ravel(),
        "forward_backward_disagreement_mps": result["disagreement"].ravel(),
        "displacement_km_10min": result["displacement_km"].ravel(),
        "distance_from_juanda_km": result["distance_km"].ravel(),
        "deep_convection_flag": result["deep_flag"].ravel().astype(int),
        "quality": result["quality"].ravel(),
        "tracking_attempted": result["tracking_attempted"].ravel().astype(int),
        "qc_rejected": result["qc_rejected"].ravel().astype(int),
        "not_trackable": result["not_trackable"].ravel().astype(int),
    })
    path = OUTPUT_DIR / f"CTV699_Dense_CMV_{t_mid:%Y%m%d_%H%M}UTC.csv"
    df.to_csv(path, index=False, float_format="%.4f")
    return path


def save_netcdf(result: dict, t_mid: datetime):
    ds = xr.Dataset(
        data_vars={
            "tbb_13_K": (("latitude", "longitude"), result["tbb13"]),
            "tbb_07_K": (("latitude", "longitude"), result["tbb07"]),
            "albedo_03": (("latitude", "longitude"), result["b03"]),
            "u_mps": (("latitude", "longitude"), result["u"]),
            "v_mps": (("latitude", "longitude"), result["v"]),
            "speed_mps": (("latitude", "longitude"), result["speed"]),
            "motion_to_deg": (("latitude", "longitude"), result["direction_to"]),
            "motion_from_deg": (("latitude", "longitude"), result["direction_from"]),
            "correlation": (("latitude", "longitude"), result["corr"]),
            "forward_backward_disagreement_mps": (("latitude", "longitude"), result["disagreement"]),
            "displacement_km_10min": (("latitude", "longitude"), result["displacement_km"]),
            "distance_from_juanda_km": (("latitude", "longitude"), result["distance_km"]),
            "deep_convection_flag": (("latitude", "longitude"), result["deep_flag"].astype(np.int8)),
            "quality": (("latitude", "longitude"), result["quality"]),
            "tracking_attempted": (("latitude", "longitude"), result["tracking_attempted"].astype(np.int8)),
            "qc_rejected": (("latitude", "longitude"), result["qc_rejected"].astype(np.int8)),
            "not_trackable": (("latitude", "longitude"), result["not_trackable"].astype(np.int8)),
        },
        coords={"latitude": result["lat"], "longitude": result["lon"]},
        attrs={
            "title": "CTV699 Dense Himawari-9 Cloud Motion Vector / AMV-like Field",
            "case": "CTV699 Juanda 2025-03-07",
            "time_utc": t_mid.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "warning": "Not official JMA AMV; no operational cloud-height assignment or JMA QC.",
            "primary_channel": "B13 tbb_13, 10.4 um",
            "support_channels": "B03 albedo_03, B07 tbb_07",
            "direction_convention": "motion_to_deg is direction toward; motion_from_deg is direction from",
        },
    )
    path = OUTPUT_DIR / f"CTV699_Dense_CMV_{t_mid:%Y%m%d_%H%M}UTC.nc"
    ds.to_netcdf(path)
    return path


def plot_speed(result: dict, t_mid: datetime):
    lat = result["lat"]
    lon = result["lon"]
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    fig, ax = plt.subplots(figsize=(11, 9))
    setup_ax(ax)
    # Hide rejected/invalid vectors in the primary product; do not paint them
    # as valid low-speed pixels. A little transparency keeps the geography visible.
    valid = (result["quality"] > 0) & np.isfinite(result["speed"])
    speed_plot = np.ma.masked_where(~valid, result["speed"])
    im = ax.pcolormesh(
        lon2d, lat2d, speed_plot, shading="auto",
        cmap="turbo", vmin=0, vmax=25, alpha=0.78, zorder=2,
    )
    add_map_overlays(ax)
    cb = fig.colorbar(im, ax=ax, pad=0.02, extend="max")
    cb.set_label("Cloud-motion speed (m s$^{-1}$)")

    good = result["quality"] > 0
    sample = good.copy()
    sample[:, :] = False
    sample[::QUIVER_STRIDE, ::QUIVER_STRIDE] = True
    sample &= good

    ax.quiver(
        lon2d[sample], lat2d[sample],
        result["u"][sample], result["v"][sample],
        color="#202830", scale=420, width=0.0022,
        headwidth=3.5, headlength=4.5, headaxislength=4.0,
        zorder=8,
    )

    ax.set_title(
        f"CTV699 — Himawari-9 Dense Cloud Motion\n"
        f"{t_mid:%Y-%m-%d %H:%M} UTC | 10-min motion | B13 primary",
        fontsize=13,
    )
    fig.text(
        0.5, 0.015,
        "Masked: rejected vectors. CMV is cloud motion, not a direct surface-wind measurement.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    path = OUTPUT_DIR / f"CTV699_Dense_CMV_Speed_{t_mid:%Y%m%d_%H%M}UTC.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_direction(result: dict, t_mid: datetime):
    lat = result["lat"]
    lon = result["lon"]
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    fig, ax = plt.subplots(figsize=(11, 9))
    setup_ax(ax)
    valid = (result["quality"] > 0) & np.isfinite(result["direction_to"])
    direction_plot = np.ma.masked_where(~valid, result["direction_to"])
    im = ax.pcolormesh(
        lon2d, lat2d, direction_plot, shading="auto",
        cmap="hsv", vmin=0, vmax=360, alpha=0.78, zorder=2,
    )
    add_map_overlays(ax)
    cb = fig.colorbar(im, ax=ax, pad=0.02, ticks=np.arange(0, 361, 45))
    cb.set_label("Cloud-motion direction TO (°)")

    good = result["quality"] > 0
    sample = np.zeros_like(good, dtype=bool)
    sample[::QUIVER_STRIDE, ::QUIVER_STRIDE] = True
    sample &= good
    ax.quiver(
        lon2d[sample], lat2d[sample],
        result["u"][sample], result["v"][sample],
        color="#202830", scale=420, width=0.0022,
        headwidth=3.5, headlength=4.5, headaxislength=4.0,
        zorder=8,
    )

    ax.set_title(
        f"CTV699 — Himawari-9 Dense Cloud Motion Direction\n"
        f"{t_mid:%Y-%m-%d %H:%M} UTC | Direction is TOWARD",
        fontsize=13,
    )
    fig.text(
        0.5, 0.015,
        "Direction is motion TOWARD. Height assignment is not performed; do not treat as runway wind.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    path = OUTPUT_DIR / f"CTV699_Dense_CMV_Direction_{t_mid:%Y%m%d_%H%M}UTC.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_qc_diagnostics(result: dict, t_mid: datetime):
    """Show valid vectors, QC-rejected candidates, and untrackable pixels separately."""
    lat2d, lon2d = np.meshgrid(result["lat"], result["lon"], indexing="ij")
    fig, ax = plt.subplots(figsize=(11, 9))
    setup_ax(ax)

    # 0 = not trackable (no cloud/texture candidate), 1 = rejected,
    # 2 = acceptable, 3 = good. The classes are categorical, not wind speed.
    cls = np.zeros(result["quality"].shape, dtype=np.float32)
    cls[result["qc_rejected"]] = 1
    cls[result["quality"] == 1] = 2
    cls[result["quality"] == 2] = 3
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#DCEAF2", "#D95F02", "#4C78A8", "#2CA25F"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    im = ax.pcolormesh(
        lon2d, lat2d, np.ma.masked_invalid(cls), shading="auto",
        cmap=cmap, norm=norm, alpha=0.85, zorder=2,
    )
    add_map_overlays(ax)

    good = result["quality"] == 2
    sample = np.zeros_like(good, dtype=bool)
    sample[::QUIVER_STRIDE, ::QUIVER_STRIDE] = True
    sample &= good
    ax.quiver(
        lon2d[sample], lat2d[sample], result["u"][sample], result["v"][sample],
        color="#17202A", scale=420, width=0.0022, zorder=9,
    )
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#DCEAF2", edgecolor="#667780", label="Not trackable: cloud/texture mask failed"),
        Patch(facecolor="#D95F02", label="Candidate rejected by tracking/QC"),
        Patch(facecolor="#4C78A8", label="Acceptable vector (quality=1)"),
        Patch(facecolor="#2CA25F", label="Good vector (quality=2)"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.92)
    ax.set_title(
        f"CTV699 — Himawari-9 Dense CMV QC Diagnostics\n"
        f"{t_mid:%Y-%m-%d %H:%M} UTC | no gap filling/interpolation", fontsize=13
    )
    fig.text(0.5, 0.015,
             "Empty areas are classified, not interpolated. CMV is cloud motion, not runway wind.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    path = OUTPUT_DIR / f"CTV699_Dense_CMV_QC_{t_mid:%Y%m%d_%H%M}UTC.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("HIMAWARI-9 DENSE GRID CMV / AMV-LIKE ANALYSIS — CTV699")
    print("=" * 78)
    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Time  : {DATE} {START_UTC}-{END_UTC} UTC")
    print(f"Domain: {LON_MIN}-{LON_MAX} E, {abs(LAT_MAX)}-{abs(LAT_MIN)} S")
    print("NOTE  : This is dense cloud motion, not official JMA AMV.")

    times = requested_times()
    files = discover_files()
    missing = [t for t in times if t not in files]
    if missing:
        print("\nERROR: missing Himawari files:")
        for t in missing:
            print(f"  {t:%Y-%m-%d %H:%M} UTC")
        print("\nExpected filenames like:")
        print("  NC_H09_20250307_0700_R21_FLDK.07001_06001.nc")
        raise SystemExit(1)

    print(f"\nFound {len(times)} requested files:")
    for t in times:
        print(f"  {t:%H:%M} UTC -> {files[t].name}")

    # Load all subsets. Each is only the small CTV699 domain.
    data = {}
    for i, t in enumerate(times, 1):
        print(f"\n[{i}/{len(times)}] {t:%H:%M} UTC")
        data[t] = read_subset(files[t])
        print(
            f"    subset shape={data[t]['b13'].shape}, "
            f"lat={data[t]['lat'][0]:.3f}..{data[t]['lat'][-1]:.3f}, "
            f"lon={data[t]['lon'][0]:.3f}..{data[t]['lon'][-1]:.3f}"
        )

    summary_rows = []

    # Central times only: 07:10 ... 07:50.
    for i in range(1, len(times) - 1):
        t_prev, t_mid, t_next = times[i - 1], times[i], times[i + 1]
        result = process_triplet(
            t_prev, t_mid, t_next,
            data[t_prev], data[t_mid], data[t_next],
        )

        csv_path = save_csv(result, t_mid)
        nc_path = save_netcdf(result, t_mid)
        png_speed = plot_speed(result, t_mid)
        png_dir = plot_direction(result, t_mid)
        png_qc = plot_qc_diagnostics(result, t_mid)

        q = result["quality"]
        valid = q > 0
        good = q == 2
        deep = result["deep_flag"] & valid
        near25 = valid & (result["distance_km"] <= 25.0)
        near50 = valid & (result["distance_km"] <= 50.0)

        summary_rows.append({
            "time_utc": t_mid.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "valid_vectors": int(valid.sum()),
            "good_vectors": int(good.sum()),
            "tracking_attempted_pixels": int(result["tracking_attempted"].sum()),
            "qc_rejected_candidates": int(result["qc_rejected"].sum()),
            "not_trackable_pixels": int(result["not_trackable"].sum()),
            "deep_cloud_valid_vectors": int(deep.sum()),
            "within_25km_juanda": int(near25.sum()),
            "within_50km_juanda": int(near50.sum()),
            "mean_speed_valid_mps": float(np.nanmean(result["speed"][valid])) if np.any(valid) else np.nan,
            "max_speed_valid_mps": float(np.nanmax(result["speed"][valid])) if np.any(valid) else np.nan,
            "mean_correlation": float(np.nanmean(result["corr"][valid])) if np.any(valid) else np.nan,
            "max_disagreement_mps": float(np.nanmax(result["disagreement"][valid])) if np.any(valid) else np.nan,
            "csv": str(csv_path),
            "netcdf": str(nc_path),
            "speed_png": str(png_speed),
            "direction_png": str(png_dir),
            "qc_diagnostics_png": str(png_qc),
        })

        print(
            f"    QC: valid={valid.sum()} good={good.sum()} "
            f"deep={deep.sum()} <=25km={near25.sum()} <=50km={near50.sum()}"
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "CTV699_Dense_CMV_Summary_20250307.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 78)
    print("COMPLETE")
    print(f"Summary: {summary_path}")
    print(f"Output : {OUTPUT_DIR}")
    print("=" * 78)


def process_uploaded_sequence(files, output_dir=None):
    """Run the CTV699 dense three-scene CMV algorithm on uploaded NetCDFs.

    ``files`` is [(filename, bytes), ...] in any order. All scenes must form a
    consecutive 10-minute sequence with matching grids and cover the CTV699
    domain. Output files are written only when an output directory is given.
    """
    global OUTPUT_DIR
    if len(files) < 3:
        raise ValueError("Dense CTV699 motion needs at least three consecutive Himawari files, 10 minutes apart.")
    if not HAVE_CV2:
        raise RuntimeError("Dense CTV699 optical flow requires OpenCV. Install it with: pip install opencv-python-headless")
    parsed = []
    import tempfile
    temp_paths = []
    try:
        for name, content in files:
            timestamp = timestamp_from_filename(Path(name))
            if timestamp is None:
                raise ValueError(f"Could not read timestamp from {Path(name).name}; expected NC_H09_YYYYMMDD_HHMM.")
            handle = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
            handle.write(content)
            handle.close()
            path = Path(handle.name)
            temp_paths.append(path)
            parsed.append((timestamp, path))
        parsed.sort(key=lambda item:item[0])
        times = [item[0] for item in parsed]
        gaps = [(b-a).total_seconds()/60 for a,b in zip(times,times[1:])]
        if any(gap != INTERVAL_MIN for gap in gaps):
            raise ValueError("Dense CTV699 motion requires scenes at exact 10-minute intervals with no gaps.")
        data = {timestamp:read_subset(path) for timestamp,path in parsed}
        old_output = OUTPUT_DIR
        if output_dir:
            OUTPUT_DIR = Path(output_dir).expanduser().resolve()
            OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
        products=[]
        try:
            for index in range(1,len(times)-1):
                previous,mid,next_time=times[index-1:index+2]
                result=process_triplet(previous,mid,next_time,data[previous],data[mid],data[next_time])
                csv_path=save_csv(result,mid)
                nc_path=save_netcdf(result,mid)
                speed_path=plot_speed(result,mid)
                direction_path=plot_direction(result,mid)
                qc_path=plot_qc_diagnostics(result,mid)
                q=result["quality"]; valid=q>0; speeds=result["speed"][valid]
                summary={"time_utc":mid.strftime("%Y-%m-%dT%H:%M:%SZ"),"valid_vectors":int(valid.sum()),
                         "good_vectors":int((q==2).sum()),"deep_cloud_vectors":int((result["deep_flag"]&valid).sum()),
                         "mean_speed_m_s":float(np.nanmean(speeds)) if speeds.size else None,
                         "max_speed_m_s":float(np.nanmax(speeds)) if speeds.size else None}
                summary_path=OUTPUT_DIR/f"CTV699_Dense_CMV_Summary_{times[0]:%Y%m%d_%H%M}_{times[-1]:%H%M}.csv"
                pd.DataFrame([summary]).to_csv(summary_path,index=False)
                def image(path):
                    return base64.b64encode(Path(path).read_bytes()).decode("ascii")
                products.append({**summary,"files":{"csv":csv_path.name,"netcdf":nc_path.name,"speed_png":speed_path.name,
                    "direction_png":direction_path.name,"qc_png":qc_path.name,"summary_csv":summary_path.name},
                    "speed_png_base64":image(speed_path),"direction_png_base64":image(direction_path),
                    "qc_png_base64":image(qc_path)})
        finally:
            OUTPUT_DIR=old_output
        return {"files":[Path(name).name for name,_ in files],"scenes":len(times),"central_times":len(products),
                "products":products,"output_dir":str(Path(output_dir).expanduser().resolve()) if output_dir else None,
                "caveat":"Research-only dense cloud-motion-vector approximation adapted from the supplied CTV699 script. These are apparent cloud-feature motions, not official JMA AMVs or validated winds; no cloud-height assignment, parallax correction, full JMA QC or sonde validation. Not for operational decisions."}
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
