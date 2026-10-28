"""Research summary of local Rainbow SBY case volumes; advisory, not alerting."""
from __future__ import annotations

import math
import re
import base64
import contextlib
import io
from pathlib import Path

import numpy as np
import wradlib.io as wio

DEFAULT_DATA = Path("/Users/aliwardhana/Downloads/Training_TL/00_data/Radar/2025-03-07")
RADAR_LON, RADAR_LAT = 112.760422, -7.410306
RWY10_LON, RWY10_LAT = 112.773492, -7.377669
RWY28_LON, RWY28_LAT = 112.800269, -7.382014
TARGET_ELEVATIONS = (0.4, 0.8, 1.8)


def as_list(item):
    return item if isinstance(item, list) else ([item] if item is not None else [])


def val(meta, key, default=None):
    return meta.get("@" + key, meta.get(key, default))


def decode(path: Path, target_elevation: float):
    volume = wio.read_rainbow(str(path)).get("volume", {})
    scan = volume.get("scan", {})
    slices = as_list(scan.get("slice"))
    if not slices:
        raise ValueError("No scan slices")
    sweep = min(slices, key=lambda s: abs(float(s.get("posangle", 99)) - target_elevation))
    sd = sweep.get("slicedata", {})
    rawmeta = sd.get("rawdata", {})
    raw = np.asarray(rawmeta.get("data"))
    if raw.ndim != 2:
        raise ValueError("Expected 2D Rainbow rawdata")
    depth = int(val(rawmeta, "depth", 8))
    low, high = float(val(rawmeta, "min", 0)), float(val(rawmeta, "max", 255))
    data = low + raw.astype(np.float32) * (high - low) / (2 ** depth)
    rayinfo = sd.get("rayinfo", {})
    azraw = np.asarray(rayinfo.get("data")).squeeze()
    rays = float(val(rayinfo, "rays", len(azraw)))
    azdepth = float(val(rayinfo, "depth", 16))
    anglestep = float(val(sweep, "anglestep", 1.0))
    azimuth = (azraw.astype(float) * rays / (2 ** azdepth) * anglestep) % 360
    bins = raw.shape[1]
    range_step = float(val(sweep, "rangestep", 0.252))
    ranges = np.arange(bins, dtype=float) * range_step
    return {"values": data, "azimuth": azimuth, "ranges": ranges,
            "elevation": float(sweep.get("posangle", target_elevation)),
            "n_rays": int(raw.shape[0]), "n_gates": bins}


def process_uploaded_volume(filename: str, content: bytes, snr_min: float = 7.0):
    """Backward-compatible single-file processing wrapper."""
    return process_uploaded_volumes([(filename, content)], snr_min=snr_min)


def decode_uploaded(content: bytes):
    volume = wio.read_rainbow(io.BytesIO(content)).get("volume", {})
    slices = as_list(volume.get("scan", {}).get("slice"))
    if not slices:
        raise ValueError("No Rainbow scan sweeps found")
    decoded = []
    for index, sweep in enumerate(slices):
        sd, rawmeta = sweep.get("slicedata", {}), sweep.get("slicedata", {}).get("rawdata", {})
        raw = np.asarray(rawmeta.get("data"))
        if raw.ndim != 2:
            continue
        depth = int(val(rawmeta, "depth", 8))
        minimum, maximum = float(val(rawmeta, "min", 0)), float(val(rawmeta, "max", 255))
        values = minimum + raw.astype(np.float32) * (maximum-minimum) / (2**depth)
        rayinfo = sd.get("rayinfo", {})
        azraw = np.asarray(rayinfo.get("data")).squeeze()
        nrays = float(val(rayinfo, "rays", len(azraw)))
        azimuth = (azraw.astype(float) * nrays / (2**float(val(rayinfo, "depth", 16))) * float(val(sweep, "anglestep", 1.0))) % 360
        decoded.append({"index": index, "elevation": float(sweep.get("posangle", np.nan)),
                        "range_step": float(val(sweep, "rangestep", .252)),
                        "azimuth": azimuth, "values": values,
                        "moment": str(val(rawmeta, "type", volume.get("@type", "unknown")))})
    return decoded


def process_uploaded_volumes(volumes, snr_min: float = 7.0, display_moment: str = "dBZ", thresholds=None, attenuation_method: str = "none"):
    """Co-register matching Rainbow moments and render selected field with shared QC."""
    attenuation_method = str(attenuation_method).lower()
    if attenuation_method not in {"none", "linear_phidp", "zphi"}:
        raise ValueError("Choose No Atten. Corr, Linear PhiDP, or ZPHI")
    limits = {"snr_min": snr_min, "snr_max": 75.0,
              "dbz_min": 0.0, "dbz_max": 75.0, "rhohv_min": 0.85,
              "zdr_min": 0.0, "zdr_max": 6.0, "v_min": -25.0,
              "v_max": 25.0, "w_min": 0.0, "w_max": 4.8,
              "kdp_min": -5.0, "kdp_max": 15.0}
    if thresholds:
        limits.update({k: float(v) for k, v in thresholds.items() if k in limits})
    for low, high in (("snr_min", "snr_max"), ("dbz_min", "dbz_max"), ("zdr_min", "zdr_max"),
                      ("v_min", "v_max"), ("w_min", "w_max"),
                      ("kdp_min", "kdp_max")):
        if limits[low] >= limits[high]:
            raise ValueError(f"{low} must be less than {high}")
    if not 0 <= limits["rhohv_min"] <= 1:
        raise ValueError("RhoHV minimum must be between 0 and 1")
    by_moment = {}
    filenames = []
    for filename, content in volumes:
        sweeps = decode_uploaded(content)
        filenames.append(filename)
        for sweep in sweeps:
            moment = sweep["moment"]
            by_moment.setdefault(moment, {})[round(sweep["elevation"], 2)] = (filename, sweep)
    available = sorted(by_moment)
    if attenuation_method != "none":
        if "dBuZ" not in by_moment:
            raise ValueError("No uncorrected dBuZ .vol file was selected. Select the dBuZ input and matching PhiDP file.")
        if "PhiDP" not in by_moment:
            raise ValueError("PhiDP input is required for attenuation correction. Select dBuZ and matching PhiDP .vol files together.")
        # Attenuation correction always starts with the raw/un-corrected
        # reflectivity product, independent of the ordinary display dropdown.
        display_moment = "dBuZ"
    source_moment = "PhiDP" if display_moment == "KDP" else display_moment
    if source_moment not in by_moment:
        source_moment = next((name for name in ("dBZ", "dBuZ") if name in by_moment), available[0])
        display_moment = source_moment
    selected = by_moment[source_moment]
    results = []
    for elev, (filename, sweep) in sorted(selected.items()):
        fields = {display_moment: sweep["values"]}
        for moment, sweeps in by_moment.items():
            match = min(sweeps, key=lambda e: abs(e-elev))
            other = sweeps[match][1]
            if abs(match-elev) <= 0.06 and other["values"].shape == sweep["values"].shape and abs(other["range_step"]-sweep["range_step"]) < 1e-5:
                fields[moment] = other["values"]
        shape = sweep["values"].shape
        mask = np.ones(shape, dtype=bool)
        applied = []
        if "SNR" in fields:
            mask &= np.isfinite(fields["SNR"]) & (fields["SNR"] >= limits["snr_min"]) & (fields["SNR"] <= limits["snr_max"])
            applied.append(f"{limits['snr_min']:g} <= SNR <= {limits['snr_max']:g} dB")
        reflectivity_moment = next((name for name in ("dBZ", "dBuZ") if name in fields), None)
        if reflectivity_moment:
            mask &= np.isfinite(fields[reflectivity_moment]) & (fields[reflectivity_moment] >= limits["dbz_min"]) & (fields[reflectivity_moment] <= limits["dbz_max"])
            applied.append(f"{limits['dbz_min']:g} <= dBZ <= {limits['dbz_max']:g}")
        if "RhoHV" in fields:
            mask &= np.isfinite(fields["RhoHV"]) & (fields["RhoHV"] > limits["rhohv_min"])
            applied.append(f"RhoHV > {limits['rhohv_min']:g}")
        if "ZDR" in fields:
            mask &= np.isfinite(fields["ZDR"]) & (fields["ZDR"] >= limits["zdr_min"]) & (fields["ZDR"] <= limits["zdr_max"])
            applied.append(f"{limits['zdr_min']:g} <= ZDR <= {limits['zdr_max']:g} dB")
        if "V" in fields:
            mask &= np.isfinite(fields["V"]) & (fields["V"] >= limits["v_min"]) & (fields["V"] <= limits["v_max"])
            applied.append(f"{limits['v_min']:g} <= V <= {limits['v_max']:g} m/s")
        if "W" in fields:
            mask &= np.isfinite(fields["W"]) & (fields["W"] >= limits["w_min"]) & (fields["W"] <= limits["w_max"])
            applied.append(f"{limits['w_min']:g} <= W <= {limits['w_max']:g} m/s")
        kdp = None
        attenuation_info = {"method": attenuation_method, "applied": False,
                            "input_moment": reflectivity_moment,
                            "note": "Select dBuZ (uncorrected reflectivity) and include matching PhiDP from the same scan."}
        if "PhiDP" in fields:
            # Unwrap and smooth each ray in range before differentiating.
            # Seven 0.252-km gates gives ~1.5-km smoothing, still research-only.
            phi = np.rad2deg(np.unwrap(np.deg2rad(fields["PhiDP"]), axis=1))
            window = 7
            padded = np.pad(phi, ((0, 0), (window // 2, window // 2)), mode="edge")
            smooth_phi = np.apply_along_axis(lambda row: np.convolve(row, np.ones(window) / window, mode="valid"), 1, padded)
            kdp = 0.5 * np.gradient(smooth_phi, sweep["range_step"], axis=1)
            fields["KDP"] = kdp
            applied.append("KDP estimated as 0.5 dPhiDP/dr after 7-gate (~1.5-km) smoothing")
        if display_moment in ("dBZ", "dBuZ") and attenuation_method != "none":
            if "PhiDP" not in fields:
                raise ValueError(f"{attenuation_method} attenuation correction requires both {display_moment} and a matching PhiDP .vol file; select them together.")
            else:
                corrected, attenuation_info = correct_reflectivity_attenuation(
                    fields[display_moment], fields["PhiDP"], sweep["range_step"], attenuation_method,
                    quality_mask=mask,
                )
                fields[f"{display_moment}_corrected"] = corrected
                attenuation_info["input_moment"] = display_moment
                display_values = corrected.astype(float).copy()
                applied.append(f"{attenuation_info['label']} applied to dBZ")
                attenuation_info["applied"] = True
                attenuation_info.pop("label", None)
        elif display_moment not in ("dBZ", "dBuZ"):
            attenuation_info["note"] = f"Not applied to displayed {display_moment}; reflectivity attenuation correction is defined here only for dBZ."
        if display_moment == "KDP":
            display_values = fields["KDP"].astype(float).copy()
            mask &= np.isfinite(fields["KDP"]) & (fields["KDP"] >= limits["kdp_min"]) & (fields["KDP"] <= limits["kdp_max"])
            applied.append(f"{limits['kdp_min']:g} <= KDP <= {limits['kdp_max']:g} deg/km")
        elif not (display_moment in ("dBZ", "dBuZ") and attenuation_method != "none" and attenuation_info["applied"]):
            display_values = fields[display_moment].astype(float).copy()
        display_values[~mask] = np.nan
        if display_moment == "KDP":
            view_min, view_max = limits["kdp_min"], limits["kdp_max"]
        elif display_moment in ("dBZ", "dBuZ") and attenuation_info["applied"]:
            corrected_finite = display_values[np.isfinite(display_values)]
            view_min, view_max = 0.0, max(75.0, float(np.ceil(np.percentile(corrected_finite, 98)))) if corrected_finite.size else 75.0
        else:
            view_min, view_max = display_limits(display_moment, float(np.nanmin(sweep["values"])), float(np.nanmax(sweep["values"])))
        correction_label = (" · " + attenuation_method.upper() + " corrected dBuZ" if attenuation_method != "none" and attenuation_info["applied"] else "")
        image = render_geographic_sweep(sweep["azimuth"], sweep["range_step"], display_values, view_min, view_max, filename + correction_label, sweep["index"], elev)
        finite = display_values[np.isfinite(display_values)]
        stats = {"valid_gates": int(finite.size), "filtered_gates": int(mask.size-mask.sum()),
                 "min": round(float(np.min(finite)), 2) if finite.size else None,
                 "p05": round(float(np.percentile(finite, 5)), 2) if finite.size else None,
                 "median": round(float(np.median(finite)), 2) if finite.size else None,
                 "p95": round(float(np.percentile(finite, 95)), 2) if finite.size else None,
                 "max": round(float(np.max(finite)), 2) if finite.size else None}
        if kdp is not None:
            k = kdp[mask & np.isfinite(kdp)]
            stats["KDP_deg_per_km"] = {"valid_gates": int(k.size), "p95": round(float(np.percentile(k, 95)), 2) if k.size else None, "max": round(float(k.max()), 2) if k.size else None}
        results.append({"index": sweep["index"], "elevation_deg": elev, "moment": display_moment,
                        "shape": list(shape), "range_step_km": sweep["range_step"], "range_max_km": round(shape[1]*sweep["range_step"], 2),
                        "available_moments_for_sweep": sorted(fields), "filters": applied, "attenuation_correction": attenuation_info, "statistics": stats,
                        "preview": {"min": view_min, "max": view_max, "coordinate_system": "WGS84 longitude/latitude", "png_base64": image}})
    snr_min = limits["snr_min"]
    return {"filename": filenames, "sweep_count": len(results), "available_moments": available,
            "display_moment": display_moment, "snr_min_db_setting": snr_min, "sweeps": results,
            "thresholds": limits,
            "attenuation_method": attenuation_method,
            "caveat": "QC fields are aligned only when sweep elevation, array shape and gate spacing match. Files must represent the same scan time. C-band attenuation settings are provisional: Linear PhiDP uses an assumed alpha=0.08 dB/degree, converts one-way PIA to reflectivity correction as 2*alpha*delta-PhiDP, and caps correction at 30 dB. ZPHI uses wradlib with alpha=0.08, b=0.75 and 2-km phase interval. Validate alpha, calibration, phase filtering, and correction against this radar and local precipitation before scientific or operational use. V and W bounds are initial display/QC limits; verify against instrument configuration. KDP is a noisy finite-difference estimate from range-unwrapped PhiDP."}


def correct_reflectivity_attenuation(dbz, phidp, range_step_km, method, quality_mask=None):
    """Provisional C-band reflectivity correction using PhiDP as input."""
    z = np.asarray(dbz, dtype=float)
    phi = np.asarray(phidp, dtype=float)
    if z.shape != phi.shape:
        raise ValueError("dBZ and PhiDP fields must have matching gate geometry")
    good_quality = np.isfinite(z) & np.isfinite(phi)
    if quality_mask is not None:
        quality_mask = np.asarray(quality_mask, dtype=bool)
        if quality_mask.shape != z.shape:
            raise ValueError("Attenuation QC mask must match reflectivity/PhiDP gate geometry")
        good_quality &= quality_mask
    if method == "linear_phidp":
        # Provisional C-band coefficient. Effective alpha depends on radar,
        # wavelength, hydrometeors and rain regime; users must validate it.
        coefficient = 0.08  # one-way PIA dB per degree differential phase
        from scipy.ndimage import median_filter
        correction = np.full_like(z, np.nan, dtype=float)
        for ray in range(z.shape[0]):
            starts = np.flatnonzero(good_quality[ray] & ~np.r_[False, good_quality[ray, :-1]])
            ends = np.flatnonzero(good_quality[ray] & ~np.r_[good_quality[ray, 1:], False]) + 1
            for start, end in zip(starts, ends):
                if end - start < 9:
                    continue
                segment = median_filter(phi[ray, start:end], size=9, mode="nearest")
                unwrapped = np.rad2deg(np.unwrap(np.deg2rad(segment)))
                monotonic = np.maximum.accumulate(unwrapped)
                gradient = np.gradient(monotonic, range_step_km)
                for sub_start, sub_end in _contiguous_runs(np.abs(gradient) <= 40.0):
                    if sub_end - sub_start < 9:
                        continue
                    phase_delta = np.maximum(0.0, monotonic[sub_start:sub_end] - monotonic[sub_start])
                    correction[ray, start + sub_start:start + sub_end] = np.minimum(2.0 * coefficient * phase_delta, 30.0)
        valid_correction = np.isfinite(correction)
        corrected = np.full_like(z, np.nan, dtype=float)
        corrected[valid_correction] = z[valid_correction] + correction[valid_correction]
        return corrected, {"method": method, "label": "Linear PhiDP (provisional C-band alpha=0.08)",
                                "radar_band": "C", "alpha_db_per_degree": coefficient,
                                "two_way_reflectivity_correction": True, "correction_cap_db": 30.0,
                                "qc_gates_used": int(valid_correction.sum()),
                                "note": "Provisional C-band alpha=0.08 dB/degree. Only selected QC-valid gates are used; phase is median-filtered within contiguous segments, and gradients above 40 deg/km break integration. Applies 2*one-way PIA; correction capped at 30 dB. Validate alpha for this radar and precipitation regime."}
    if method == "zphi":
        try:
            import xarray as xr
            from wradlib.atten import specific_attenuation_zphi
            from scipy.ndimage import median_filter
            range_m = np.arange(z.shape[1], dtype=float) * range_step_km * 1000.0
            corrected = np.full_like(z, np.nan, dtype=float)
            correction_cap_db = 30.0
            for ray in range(z.shape[0]):
                finite = good_quality[ray]
                if finite.sum() < 10:
                    continue
                unwrapped = np.full(phi.shape[1], np.nan, dtype=float)
                for phase_start, phase_end in _contiguous_runs(finite):
                    if phase_end - phase_start < 9:
                        continue
                    filtered = median_filter(phi[ray, phase_start:phase_end], size=9, mode="nearest")
                    unwrapped[phase_start:phase_end] = np.rad2deg(np.unwrap(np.deg2rad(filtered)))
                monotonic = np.full(phi.shape[1], np.nan, dtype=float)
                for phase_start, phase_end in _contiguous_runs(np.isfinite(unwrapped)):
                    monotonic[phase_start:phase_end] = np.maximum.accumulate(unwrapped[phase_start:phase_end])
                # Large phase gradients are usually phase noise/alias jumps in
                # this unvalidated input. Reject them and split into contiguous
                # range segments so the solver never bridges missing gates.
                gradient = np.gradient(monotonic, range_step_km)
                usable = finite & np.isfinite(gradient) & (np.abs(gradient) <= 40.0)
                starts = np.flatnonzero(usable & ~np.r_[False, usable[:-1]])
                ends = np.flatnonzero(usable & ~np.r_[usable[1:], False]) + 1
                for start, end in zip(starts, ends):
                    if end - start < 10:
                        continue
                    ray_dbz = z[ray, start:end]
                    ray_phase = monotonic[start:end]
                    # wradlib's internal phase-interval lookup expects exact
                    # coordinate hits; use a zero-based integer-meter range
                    # coordinate for each contiguous segment.
                    ray_range = np.arange(end - start, dtype=np.int64) * int(round(range_step_km * 1000.0))
                    da_phi = xr.DataArray(ray_phase, dims=("range",), coords={"range": ray_range}, name="PHIDP")
                    da_dbz = xr.DataArray(ray_dbz, dims=("range",), coords={"range": ray_range}, name="DBZH")
                    specific = specific_attenuation_zphi(da_phi, da_dbz, alpha=0.08, b=0.75, rng=2000.0).values
                    delta_r_km = np.diff(ray_range, prepend=ray_range[0]) / 1000.0
                    pia = np.cumsum(np.nan_to_num(specific, nan=0.0) * delta_r_km)
                    correction = np.minimum(2.0 * pia, correction_cap_db)
                    corrected[ray, start:end] = ray_dbz + correction
            return corrected, {"method": method, "label": "ZPHI (provisional C-band; alpha=0.08, b=0.75)",
                               "radar_band": "C", "alpha": 0.08, "b": 0.75, "phase_interval_km": 2.0,
                               "phase_median_filter_gates": 9, "maximum_abs_phase_gradient_deg_per_km": 40.0,
                               "correction_cap_db": correction_cap_db,
                               "qc_gates_used": int(np.isfinite(corrected).sum()),
                               "note": "Provisional C-band settings. Only selected QC-valid gates are used; PhiDP is median-filtered within contiguous valid segments, gradients above 40 deg/km are rejected, correction restarts after invalid gaps and is capped at 30 dB. Validate all settings for this radar and precipitation regime."}
        except Exception as exc:
            raise ValueError("ZPHI correction failed: " + str(exc)) from exc
    raise ValueError("Unknown attenuation correction method")


def _contiguous_runs(mask):
    """Yield half-open intervals for contiguous True values in a 1-D mask."""
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
    return zip(starts, ends)


def display_limits(moment, source_min, source_max):
    if moment == "dBZ":
        return 0.0, 75.0
    if moment == "ZDR":
        return 0.0, 6.0
    return source_min, source_max


def apply_moment_filters(data, moment, snr_min):
    """Filter invalid gates only for the product contained in this .vol."""
    values = np.asarray(data, dtype=np.float32).copy()
    valid = np.isfinite(values)
    if moment == "SNR":
        valid &= values >= snr_min
        applied = f"SNR >= {snr_min:.1f} dB"
    elif moment == "dBZ":
        valid &= (values >= 0.0) & (values <= 75.0)
        applied = "0 <= dBZ <= 75"
    elif moment == "RhoHV":
        valid &= values > 0.85
        applied = "RhoHV > 0.85"
    elif moment == "ZDR":
        valid &= (values >= 0.0) & (values <= 6.0)
        applied = "0 <= ZDR <= 6 dB"
    else:
        applied = "No parameter-specific filter (moment %s)" % moment
    values[~valid] = np.nan
    return values, valid, applied


def render_geographic_sweep(azimuth_deg, range_step_km, values, vmin, vmax, filename, sweep_index, elevation):
    """Render the uploaded sweep with geographic axes and cached map context."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from view_rainbow_vol import add_map_context, RADAR_LON as SITE_LON, RADAR_LAT as SITE_LAT

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.12, top=0.88)
    azimuth_edges = circular_edges(np.asarray(azimuth_deg, dtype=float))
    ranges = np.arange(values.shape[1], dtype=float) * range_step_km
    range_edges = np.maximum(0.0, np.arange(values.shape[1] + 1, dtype=float) * range_step_km - range_step_km / 2)
    lat_edges, lon_edges = polar_edges_to_latlon(azimuth_edges, range_edges)
    # C is (ray, gate), and edge grids are (ray+1, gate+1), so every native
    # Rainbow gate is drawn as its actual cell instead of a sparse point marker.
    mesh = ax.pcolormesh(lon_edges, lat_edges, values, shading="flat", cmap="turbo",
                         vmin=vmin, vmax=vmax, linewidth=0, rasterized=True, zorder=3)
    add_map_context(ax)
    ax.plot(SITE_LON, SITE_LAT, marker="^", color="black", markeredgecolor="white",
            markersize=9, zorder=15)
    ax.annotate("SBY radar", (SITE_LON, SITE_LAT), xytext=(6, 6),
                textcoords="offset points", color="black", fontsize=8,
                weight="bold", zorder=16)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°; negative = °S)")
    ax.set_aspect(1 / np.cos(np.deg2rad(RADAR_LAT)))
    ax.set_title(f"{filename} · sweep {sweep_index} · {elevation:.2f}°")
    fig.colorbar(mesh, ax=ax, label="Rainbow moment value", shrink=0.8)
    output = io.BytesIO()
    with contextlib.redirect_stdout(io.StringIO()):
        fig.savefig(output, format="png", dpi=115, facecolor="white")
    plt.close(fig)
    return base64.b64encode(output.getvalue()).decode("ascii")


def circular_edges(azimuth_deg):
    """Return continuous azimuth boundaries for rays, including north wrap."""
    az = np.unwrap(np.deg2rad(azimuth_deg))
    if az.size < 2:
        raise ValueError("At least two radar rays are required")
    edges = np.empty(az.size + 1, dtype=float)
    edges[1:-1] = (az[:-1] + az[1:]) / 2
    edges[0] = az[0] - (az[1] - az[0]) / 2
    edges[-1] = az[-1] + (az[-1] - az[-2]) / 2
    return edges


def polar_edges_to_latlon(azimuth_edges, range_edges_km):
    """Spherical forward-geodesic transform of polar cell corners to WGS84."""
    az = azimuth_edges[:, None]
    distance = range_edges_km[None, :] / 6371.0088
    lat1 = np.deg2rad(RADAR_LAT)
    lon1 = np.deg2rad(RADAR_LON)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(distance) + np.cos(lat1) * np.sin(distance) * np.cos(az))
    lon2 = lon1 + np.arctan2(np.sin(az) * np.sin(distance) * np.cos(lat1),
                             np.cos(distance) - np.sin(lat1) * np.sin(lat2))
    return np.rad2deg(lat2), np.rad2deg(lon2)


def polar_xy(azimuth, ranges):
    az = np.deg2rad(azimuth)[:, None]
    r = ranges[None, :]
    return r * np.sin(az), r * np.cos(az)


def local_xy(lon, lat):
    lat0 = math.radians(RADAR_LAT)
    return (math.radians(lon - RADAR_LON) * 6371 * math.cos(lat0),
            math.radians(lat - RADAR_LAT) * 6371)


def percentile_summary(arr):
    a = np.asarray(arr)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "min": round(float(np.min(a)), 2),
            "p05": round(float(np.percentile(a, 5)), 2),
            "median": round(float(np.median(a)), 2),
            "p95": round(float(np.percentile(a, 95)), 2),
            "max": round(float(np.max(a)), 2)}


def analyze_case(data_dir=DEFAULT_DATA, start="20250307070000", end="20250307080000"):
    """Pair field files by scan stamp; summarize low-elevation final corridor."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Radar data folder not found: {data_dir}")
    index = {}
    for f in data_dir.glob("*.vol"):
        m = re.match(r"(\d{14})(.+)\.vol$", f.name)
        # Names are YYYYMMDDhhmmss00<product>.vol; the trailing 00 is centiseconds.
        if m and start[:14] <= m.group(1) <= end[:14]:
            product = m.group(2)
            if product.startswith("00"):
                product = product[2:]
            index.setdefault(m.group(1), {})[product] = f
    stamps = sorted(t for t, fields in index.items() if any(k.startswith("dBZ") for k in fields))
    if not stamps:
        raise FileNotFoundError(f"No timestamped dBZ volumes in {data_dir} for requested window")

    # Corridor is 0-10 km outward of THR10 on the extended final, +/-1.5 km cross-track.
    tx, ty = local_xy(RWY10_LON, RWY10_LAT)
    bearing = math.radians(279.3)
    ux, uy = math.sin(bearing), math.cos(bearing)
    rows = []
    errors = []
    for stamp in stamps:
        for elev in TARGET_ELEVATIONS:
            loaded = {}
            for field, paths in ((k, v) for k, v in index[stamp].items()):
                # Normalize variants to their base moment. Prefer standard products.
                base = next((candidate for candidate in ("dBZ", "V", "W", "ZDR", "RhoHV", "SNR", "SQI", "PhiDP", "dBuZ") if field == candidate), None)
                if base and base not in loaded:
                    try:
                        loaded[base] = decode(paths, elev)
                    except Exception as exc:
                        errors.append(f"{paths.name} {elev} deg: {exc}")
            z = loaded.get("dBZ")
            if z is None:
                continue
            x, y = polar_xy(z["azimuth"], z["ranges"])
            dx, dy = x - tx, y - ty
            along = dx * ux + dy * uy
            cross = -dx * uy + dy * ux
            corridor = (along >= 0) & (along <= 10) & (np.abs(cross) <= 1.5)
            qc = corridor & np.isfinite(z["values"])
            if "SNR" in loaded:
                qc &= np.isfinite(loaded["SNR"]["values"]) & (loaded["SNR"]["values"] >= 15)
            if "SQI" in loaded:
                qc &= np.isfinite(loaded["SQI"]["values"]) & (loaded["SQI"]["values"] >= 0.5)
            if "RhoHV" in loaded:
                qc &= np.isfinite(loaded["RhoHV"]["values"]) & (loaded["RhoHV"]["values"] >= 0.7)
            row = {"timestamp_utc": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[8:10]}:{stamp[10:12]}:{stamp[12:14]}Z",
                   "target_elevation_deg": elev, "actual_elevation_deg": z["elevation"],
                   "corridor_km_from_threshold": [0, 10], "corridor_half_width_km": 1.5,
                   "quality_mask_fields_available": [f for f in ("SNR", "SQI", "RhoHV") if f in loaded],
                   "quality_gates": int(qc.sum()), "candidate_gates": int(corridor.sum()),
                   "reflectivity_dBZ": percentile_summary(np.where(qc, z["values"], np.nan)),
                   "gates_ge_35dBZ": int(np.count_nonzero(qc & (z["values"] >= 35))),
                   "gates_ge_45dBZ": int(np.count_nonzero(qc & (z["values"] >= 45))),
                   "available_moments": sorted(loaded)}
            for field in ("V", "W", "ZDR", "RhoHV", "SNR", "SQI", "PhiDP", "dBuZ"):
                if field in loaded:
                    arr = loaded[field]["values"]
                    use = qc & np.isfinite(arr)
                    row[field] = percentile_summary(np.where(use, arr, np.nan))
            if "V" in loaded:
                v = loaded["V"]["values"]
                valid = qc & np.isfinite(v)
                row["radial_velocity_p95_minus_p05_mps"] = round(float(np.percentile(v[valid], 95) - np.percentile(v[valid], 5)), 2) if valid.sum() >= 10 else None
                row["radial_velocity_note"] = "Radar-radial variation only; not runway headwind change. Percentile spread is not a spatial microburst-couplet test."
            rows.append(row)
    if not rows:
        raise RuntimeError("No decodable dBZ sweep summaries produced")
    return {"case": "CTV699 / Juanda historical research case", "data_dir": str(data_dir),
            "time_window_utc": [start, end], "scan_count": len(stamps), "summary_count": len(rows),
            "threshold": "RWY10", "method": "Rainbow native min/max/depth scaling; nearest low elevation; 0-10 km extended final corridor, +/-1.5 km; available SNR>=15, SQI>=0.5, RhoHV>=0.7 masks.",
            "limitations": ["This is a research screening summary, not an operational alert or validated decision.", "Separate moments may have timing/geometry mismatches; check reported available_moments and quality_mask_fields_available.", "If optional quality moments are absent, the gates are not equivalently quality controlled.", "Radial velocity statistics are not runway-aligned wind measurements; no automatic microburst classification is made.", "Vertical persistence must be assessed across the elevation summaries and scans; no motion forecast is calculated."],
            "rows": rows, "decode_warnings": errors[:40]}
