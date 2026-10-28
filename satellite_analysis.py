"""Local plotting for georeferenced Himawari NetCDF files."""
from __future__ import annotations

import base64
import contextlib
import datetime as dt
import io
import re
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from view_rainbow_vol import add_map_context

JUANDA_REGION = (112.1, 113.25, -8.15, -6.8)


def process_satellite_file(filename: str, content: bytes, variable: str = ""):
    """Crop a Himawari equal lat/lon NetCDF to the Juanda map and plot a field."""
    with Dataset("satellite-upload.nc", mode="r", memory=content) as dataset:
        if "latitude" not in dataset.variables or "longitude" not in dataset.variables:
            raise ValueError("Expected one-dimensional latitude and longitude coordinate variables.")
        latitude = np.asarray(dataset.variables["latitude"][:], dtype=float)
        longitude = np.asarray(dataset.variables["longitude"][:], dtype=float)
        lat_indices = np.flatnonzero((latitude >= -8.15) & (latitude <= -6.8))
        lon_indices = np.flatnonzero((longitude >= 112.1) & (longitude <= 113.25))
        if not lat_indices.size or not lon_indices.size:
            raise ValueError("The file does not cover the Juanda map area (112.1–113.25°E, 6.8–8.15°S).")

        candidates = []
        for name, item in dataset.variables.items():
            if (name in ("latitude", "longitude", "band_id", "start_time", "end_time", "geometry_parameters")
                    or len(item.dimensions) != 2 or set(item.dimensions) != {"latitude", "longitude"}):
                continue
            candidates.append(name)
        if not candidates:
            raise ValueError("No two-dimensional latitude/longitude data fields were found.")
        if variable not in candidates:
            variable = "tbb_13" if "tbb_13" in candidates else candidates[0]

        data_var = dataset.variables[variable]
        lat_dim = data_var.dimensions.index("latitude")
        lon_dim = data_var.dimensions.index("longitude")
        y0, y1 = int(lat_indices.min()), int(lat_indices.max()) + 1
        x0, x1 = int(lon_indices.min()), int(lon_indices.max()) + 1
        section = [slice(None), slice(None)]
        section[lat_dim], section[lon_dim] = slice(y0, y1), slice(x0, x1)
        raw = np.ma.asarray(data_var[tuple(section)])
        if lat_dim == 1:
            raw = raw.T
        field = np.asarray(raw.filled(np.nan), dtype=np.float32)
        lat = latitude[y0:y1]
        lon = longitude[x0:x1]

        # Keep upload response images manageable while retaining a local view.
        stride = max(1, int(np.ceil(max(field.shape) / 1400)))
        field, lat, lon = field[::stride, ::stride], lat[::stride], lon[::stride]
        units = str(getattr(data_var, "units", ""))
        long_name = str(getattr(data_var, "long_name", variable))
        title = str(getattr(dataset, "title", "Himawari satellite"))
        start = np.asarray(dataset.variables["start_time"][:]).reshape(-1) if "start_time" in dataset.variables else []
        time_value = float(start[0]) if len(start) else None

    valid = np.isfinite(field)
    if not valid.any():
        raise ValueError(f"Field {variable} has no valid pixels in the Juanda crop.")
    lo, hi = np.percentile(field[valid], [2, 98])
    if lo == hi:
        lo, hi = float(np.min(field[valid])), float(np.max(field[valid]) + 1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.12, top=0.88)
    mesh = ax.pcolormesh(lon, lat, field, shading="auto", cmap="turbo", vmin=lo, vmax=hi,
                         rasterized=True, zorder=3)
    add_map_context(ax)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°; negative = °S)")
    ax.set_title(f"{Path(filename).name} · {variable} · {long_name}")
    fig.colorbar(mesh, ax=ax, label=f"{long_name} ({units})" if units else long_name, shrink=0.8)
    output = io.BytesIO()
    with contextlib.redirect_stdout(io.StringIO()):
        fig.savefig(output, format="png", dpi=115, facecolor="white")
    plt.close(fig)

    return {
        "filename": Path(filename).name,
        "title": title,
        "variable": variable,
        "long_name": long_name,
        "units": units,
        "available_variables": candidates,
        "region": {"west": round(float(lon.min()), 4), "east": round(float(lon.max()), 4),
                   "south": round(float(lat.min()), 4), "north": round(float(lat.max()), 4)},
        "grid_shape": list(field.shape),
        "stride": stride,
        "statistics": {"valid_pixels": int(valid.sum()), "min": round(float(field[valid].min()), 4),
                       "p02": round(float(np.percentile(field[valid], 2)), 4),
                       "median": round(float(np.median(field[valid])), 4),
                       "p98": round(float(np.percentile(field[valid], 98)), 4),
                       "max": round(float(field[valid].max()), 4)},
        "preview": {"png_base64": base64.b64encode(output.getvalue()).decode("ascii"),
                    "min": float(lo), "max": float(hi), "coordinate_system": "WGS84 longitude/latitude"},
        "caveat": "This is a geographic visualization and field summary of the selected NetCDF variable, not a validated cloud-top, storm-growth, or windshear retrieval. Check the variable meaning, units, calibration, and observation time before interpretation.",
    }


def process_motion_files(files, band="all"):
    """Process the same two scenes for Bands 3, 7, and 13 automatically."""
    bands = ("03", "07", "13") if str(band).lower() in ("all", "auto", "") else (str(band).zfill(2),)
    products = [_process_motion_band(files, selected_band) for selected_band in bands]
    return {
        "bands": products,
        "files": products[0]["files"],
        "time_delta_minutes": products[0]["time_delta_minutes"],
        "caveat": "Each panel shows band-specific image-feature displacement. These are not complete or validated AMV winds: no cloud-height assignment, radiative-transfer/first-guess profile, parallax correction, full JMA quality index, or sonde validation is applied. In particular, Band 7 (3.9 µm) is not the paper's 7.0 µm water-vapor channel (Band 10). Not for operational use.",
    }


def process_convective_cooling_files(files):
    """Summarize Band-13 cooling, cold-cloud areas, and core-centroid evolution."""
    from scipy.ndimage import uniform_filter
    if len(files) < 2:
        raise ValueError("Convective cooling analysis needs at least two Himawari scenes.")
    scenes=[]
    for filename,content in files:
        timestamp=_filename_time(filename)
        if timestamp is None:
            raise ValueError(f"Could not read observation time from {Path(filename).name}.")
        with Dataset(Path(filename).name,mode="r",memory=content) as ds:
            if "tbb_13" not in ds.variables:
                raise ValueError(f"{Path(filename).name} has no Band 13 field (tbb_13).")
            lat=np.asarray(ds.variables["latitude"][:],dtype=float)
            lon=np.asarray(ds.variables["longitude"][:],dtype=float)
            y=np.flatnonzero((lat>=-8.0)&(lat<=-6.5));x=np.flatnonzero((lon>=112.0)&(lon<=113.5))
            if not y.size or not x.size:
                raise ValueError(f"{Path(filename).name} does not cover the CTV699/Juanda area (112–113.5°E, 8–6.5°S).")
            var=ds.variables["tbb_13"]
            iy,ix=var.dimensions.index("latitude"),var.dimensions.index("longitude")
            selection=[slice(None),slice(None)];selection[iy]=slice(int(y[0]),int(y[-1])+1);selection[ix]=slice(int(x[0]),int(x[-1])+1)
            var.set_auto_maskandscale(False);raw=np.ma.asarray(var[tuple(selection)])
            if iy==1:raw=raw.T
            field=np.asarray(raw,dtype=np.float32)
            fill=getattr(var,"_FillValue",None)
            if fill is not None:field[field==float(fill)]=np.nan
            field=field*float(getattr(var,"scale_factor",1.0))+float(getattr(var,"add_offset",0.0))
            field[(field<150)|(field>330)]=np.nan
            scenes.append({"name":Path(filename).name,"time":timestamp,"lat":lat[y[0]:y[-1]+1],"lon":lon[x[0]:x[-1]+1],"tbb":field})
    scenes.sort(key=lambda x:x["time"])
    ref=scenes[0]
    for scene in scenes[1:]:
        if scene["tbb"].shape!=ref["tbb"].shape or not np.allclose(scene["lat"],ref["lat"]) or not np.allclose(scene["lon"],ref["lon"]):
            raise ValueError("All selected scenes must share the same latitude/longitude grid and regional coverage.")
    lat,lon=ref["lat"],ref["lon"]
    dlat=abs(float(np.median(np.diff(lat))));dlon=abs(float(np.median(np.diff(lon))))
    pixel_area=(111.32*dlat)*(111.32*np.cos(np.deg2rad(lat))*dlon)[:,None]
    stats=[];centroids=[];smoothed=[]
    for scene in scenes:
        tbb=scene["tbb"];valid=np.isfinite(tbb)
        if not valid.any():raise ValueError(f"No valid Band 13 pixels in {scene['name']}.")
        cold240=valid&(tbb<240);cold220=valid&(tbb<220)
        area240=float(np.sum(pixel_area*np.where(cold240,1,0)));area220=float(np.sum(pixel_area*np.where(cold220,1,0)))
        lat2d,lon2d=np.meshgrid(lat,lon,indexing="ij")
        if cold220.any():
            weights=np.broadcast_to(pixel_area,tbb.shape)[cold220]
            centroid_lat=float(np.average(lat2d[cold220],weights=weights));centroid_lon=float(np.average(lon2d[cold220],weights=weights))
        else:centroid_lat=centroid_lon=None
        stats.append({"time_utc":scene["time"].strftime("%Y-%m-%d %H:%M UTC"),"minimum_tbb_K":float(np.nanmin(tbb)),"p05_tbb_K":float(np.nanpercentile(tbb,5)),"mean_tbb_K":float(np.nanmean(tbb)),"area_tbb_below_240_km2":area240,"area_tbb_below_220_km2":area220})
        centroids.append({"time_utc":scene["time"].strftime("%Y-%m-%d %H:%M UTC"),"latitude":centroid_lat,"longitude":centroid_lon})
        data=np.nan_to_num(tbb,nan=0.0);weights0=valid.astype(float)
        numerator=uniform_filter(data,size=5,mode="nearest");denominator=uniform_filter(weights0,size=5,mode="nearest")
        smoothed.append(np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0))
    cooling=[]
    for index in range(len(scenes)-1):
        minutes=(scenes[index+1]["time"]-scenes[index]["time"]).total_seconds()/60
        if minutes<=0:raise ValueError("Scene timestamps must be strictly increasing.")
        rate=(smoothed[index+1]-smoothed[index])/(minutes/60)
        cooling.append({"interval":f"{scenes[index]['time']:%H:%M}–{scenes[index+1]['time']:%H:%M} UTC","rate":rate})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    fig,axes=plt.subplots(1,2,figsize=(12,5.4))
    times=[s["time"] for s in scenes]
    axes[0].plot(times,[v["area_tbb_below_240_km2"] for v in stats],marker="o",label="TBB < 240 K")
    axes[0].plot(times,[v["area_tbb_below_220_km2"] for v in stats],marker="o",label="TBB < 220 K")
    axes[0].set_title("Cold-cloud area");axes[0].set_ylabel("Area (km²)");axes[0].grid(alpha=.3);axes[0].legend();fig.autofmt_xdate()
    axes[1].plot(times,[v["minimum_tbb_K"] for v in stats],marker="o",label="Minimum")
    axes[1].plot(times,[v["p05_tbb_K"] for v in stats],marker="o",label="5th percentile")
    axes[1].plot(times,[v["mean_tbb_K"] for v in stats],marker="o",label="Mean")
    axes[1].set_title("Band 13 cloud-top brightness temperature");axes[1].set_ylabel("Temperature (K)");axes[1].grid(alpha=.3);axes[1].legend()
    summary_png=_figure_base64(fig);plt.close(fig)
    ncols=min(3,len(cooling));nrows=int(np.ceil(len(cooling)/ncols));fig,axes=plt.subplots(nrows,ncols,figsize=(5*ncols,4*nrows),squeeze=False,constrained_layout=True)
    vmax=max(1.0,max(float(np.nanpercentile(np.abs(c["rate"]),98)) for c in cooling if np.isfinite(c["rate"]).any()))
    mesh=None
    for ax,item in zip(axes.flat,cooling):
        mesh=ax.pcolormesh(lon,lat,item["rate"],cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax),shading="auto")
        ax.set_title(item["interval"]);ax.set_xlabel("Longitude (°E)");ax.set_ylabel("Latitude (°; negative = °S)");ax.scatter([112.7873],[-7.38],marker="x",c="black")
    for ax in list(axes.flat)[len(cooling):]:ax.remove()
    fig.colorbar(mesh,ax=list(axes.flat)[:len(cooling)],label="Smoothed apparent TBB change rate (K/hour); negative = cooling",shrink=.85)
    cooling_png=_figure_base64(fig);plt.close(fig)
    return {"files":[s["name"] for s in scenes],"scene_count":len(scenes),"interval_count":len(cooling),"statistics":stats,"cold_core_centroids":centroids,
            "summary_png_base64":summary_png,"cooling_png_base64":cooling_png,
            "caveat":"Band-13 brightness-temperature changes and cold-cloud areas are satellite indicators, not direct updraft measurements. Cooling is apparent change, affected by cloud motion, parallax, sampling, retrieval and scene mismatch. Thresholds are simple diagnostics; these products do not establish convective severity or windshear and are not for operational decisions."}


def _figure_base64(fig):
    output=io.BytesIO()
    with contextlib.redirect_stdout(io.StringIO()):fig.savefig(output,format="png",dpi=110,facecolor="white",bbox_inches="tight")
    return base64.b64encode(output.getvalue()).decode("ascii")


def _process_motion_band(files, band="13"):
    """Estimate cloud-pattern displacement between sequential Himawari fields.

    Uses local normalized cross-correlation. The result is image feature motion,
    not a cloud-height-corrected operational Atmospheric Motion Vector product.
    """
    if len(files) != 2:
        raise ValueError("Select exactly two Himawari files separated by 10 minutes.")
    if not re.fullmatch(r"(?:03|07|13)", str(band)):
        raise ValueError("Choose Himawari band 3, 7, or 13.")
    band = str(band).zfill(2)
    variable = ("albedo_" if band == "03" else "tbb_") + band

    frames = []
    for filename, content in files:
        with Dataset(filename, mode="r", memory=content) as ds:
            if variable not in ds.variables:
                raise ValueError(f"{filename} has no {variable} field for Band {int(band)}.")
            lat = np.asarray(ds.variables["latitude"][:], dtype=float)
            lon = np.asarray(ds.variables["longitude"][:], dtype=float)
            x = np.flatnonzero((lon >= JUANDA_REGION[0]) & (lon <= JUANDA_REGION[1]))
            y = np.flatnonzero((lat >= JUANDA_REGION[2]) & (lat <= JUANDA_REGION[3]))
            if not x.size or not y.size:
                raise ValueError(f"{filename} does not cover Juanda region.")
            var = ds.variables[variable]
            iy, ix = var.dimensions.index("latitude"), var.dimensions.index("longitude")
            sl = [slice(None), slice(None)]
            sl[iy] = slice(int(y[0]), int(y[-1])+1)
            sl[ix] = slice(int(x[0]), int(x[-1])+1)
            var.set_auto_maskandscale(False)
            raw = np.ma.asarray(var[tuple(sl)])
            if iy == 1:
                raw = raw.T
            frame = np.asarray(raw, dtype=np.float32)
            fill = getattr(var, "_FillValue", None)
            if fill is not None:
                frame[frame == float(fill)] = np.nan
            # Decoding follows netCDF scale/offset metadata and valid range.
            scale = float(getattr(var, "scale_factor", 1.0))
            offset = float(getattr(var, "add_offset", 0.0))
            frame = frame * scale + offset
            frame[~np.isfinite(frame)] = np.nan
            # Normalize day/night/albedo or brightness-temperature data without
            # changing the spatial structure used for matching.
            frames.append((Path(filename).name, frame, lat[y[0]:y[-1]+1], lon[x[0]:x[-1]+1],
                           _filename_time(filename)))

    frames.sort(key=lambda x: (x[4] is None, x[4] or dt.datetime.max, x[0]))
    if frames[0][4] is None or frames[1][4] is None:
        raise ValueError("Could not read observation times from both filenames; expected NC_H09_YYYYMMDD_HHMM files.")
    selected_interval = (frames[1][4] - frames[0][4]).total_seconds() / 60
    if selected_interval != 10:
        raise ValueError(f"The two files must be exactly 10 minutes apart; selected interval is {selected_interval:g} minutes.")
    base_name, base, lat, lon, base_time = frames[0]
    for name, other, other_lat, other_lon, timestamp in frames[1:]:
        if other.shape != base.shape or not np.allclose(other_lat, lat) or not np.allclose(other_lon, lon):
            raise ValueError("Selected files do not share the same latitude/longitude grid and regional crop.")
    # Downsample to about 250x250 for correlation calculations and a concise plot.
    stride = max(1, int(np.ceil(max(base.shape) / 260)))
    frames = [(name, frame[::stride, ::stride]) for name, frame, *_ in frames]
    lat, lon = lat[::stride], lon[::stride]
    if len(lat) < 12 or len(lon) < 12:
        raise ValueError("Regional crop is too small for a stable feature-motion estimate.")

    reference_name, reference = frames[0]
    latest_name, latest = frames[-1]
    time_delta_minutes = None
    t0, t1 = _filename_time(reference_name), _filename_time(latest_name)
    if t0 and t1:
        time_delta_minutes = (t1 - t0).total_seconds() / 60
    if time_delta_minutes != 10:
        raise ValueError("Selected Himawari files must be exactly 10 minutes apart.")

    # Paper-inspired tracking: four normalized-correlation surfaces (7x7 and
    # 33x33 targets, forward and backward), reverse the backward surfaces,
    # average by direction, then average forward/backward for the final vector.
    # Target interval is greater than the smaller 7-pixel box to limit
    # correlated neighboring targets, following the paper's grid rationale.
    half_search = max(3, min(6, min(reference.shape) // 10))
    centers = _tracking_centers(reference.shape, half_search, min_spacing=9)
    matched = []
    for cy, cx in centers:
        forward_surfaces, backward_surfaces = [], []
        for side in (7, 33):
            forward_surfaces.append(_correlation_surface(reference, latest, cy, cx, side, half_search))
            backward_surfaces.append(_correlation_surface(latest, reference, cy, cx, side, half_search))
        if any(s is None for s in forward_surfaces + backward_surfaces):
            continue
        forward = np.mean(forward_surfaces, axis=0)
        backward = np.mean([np.flip(s, axis=(0, 1)) for s in backward_surfaces], axis=0)
        final_surface = (forward + backward) / 2
        fy, fx = np.unravel_index(np.argmax(forward), forward.shape)
        by, bx = np.unravel_index(np.argmax(backward), backward.shape)
        py, px = np.unravel_index(np.argmax(final_surface), final_surface.shape)
        dx, dy = px - half_search, py - half_search
        forward_dx, forward_dy = fx - half_search, fy - half_search
        backward_dx, backward_dy = bx - half_search, by - half_search
        forward_backward_error = float(np.hypot(forward_dx-backward_dx, forward_dy-backward_dy))
        peak = float(final_surface[py, px])
        # The paper's operational threshold is 0.75. Report that quality gate
        # explicitly; retain lower-confidence vectors as experimental preview
        # only, rather than presenting them as accepted operational AMVs.
        if np.hypot(dx, dy) < 0.75 or np.hypot(dx, dy) > half_search * 1.5:
            continue
        matched.append((cy, cx, dx, dy, peak, forward_backward_error,
                        peak >= 0.75 and forward_backward_error <= 2.0))
    if not matched:
        raise ValueError("No moving features matched in the Juanda crop. Try another band or a wider time separation.")

    # Grid coordinates in regular Himawari lat/lon data. Eastward longitude
    # speed is corrected by cos(latitude) for local physical distance.
    vectors = []
    for cy, cx, dx, dy, quality, fb_error, passes_qc in matched:
        east_km = dx * (lon[1] - lon[0]) * 111.32 * np.cos(np.deg2rad(lat[cy]))
        north_km = dy * (lat[1] - lat[0]) * 111.32
        u = east_km / (time_delta_minutes * 60) * 1000
        v = north_km / (time_delta_minutes * 60) * 1000
        vectors.append((lon[cx], lat[cy], u, v, quality, fb_error, passes_qc))
    vectors = np.asarray(vectors)
    speeds = np.hypot(vectors[:, 2], vectors[:, 3])
    accepted = vectors[:, 6].astype(bool)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.12, top=0.88)
    # The base image remains visible as cloud context, while the color raster
    # represents the tracked apparent speed (not brightness temperature).
    ax.pcolormesh(lon, lat, latest, shading="auto", cmap="gray", zorder=2, alpha=0.35)
    speed_grid = np.full((len(lat), len(lon)), np.nan, dtype=float)
    if len(vectors) >= 3:
        from scipy.interpolate import griddata
        grid_lon, grid_lat = np.meshgrid(lon, lat)
        speed_grid = griddata(vectors[:, :2], speeds, (grid_lon, grid_lat), method="linear")
    speed_mesh = ax.pcolormesh(lon, lat, speed_grid, shading="auto", cmap="turbo",
                               vmin=float(np.nanmin(speeds)), vmax=float(np.nanmax(speeds)) if np.nanmax(speeds) > np.nanmin(speeds) else float(np.nanmin(speeds) + 1e-6),
                               zorder=3, alpha=0.88)
    add_map_context(ax)
    # Convert m/s to map degrees and exaggerate displacement for legibility.
    # Arrow orientation is physical; color encodes apparent feature speed.
    display_scale = 8.0
    arrow_lon = vectors[:, 2] * (time_delta_minutes * 60) / (111320.0 * np.cos(np.deg2rad(vectors[:, 1]))) * display_scale
    arrow_lat = vectors[:, 3] * (time_delta_minutes * 60) / 111320.0 * display_scale
    quiv = ax.quiver(vectors[:, 0], vectors[:, 1], arrow_lon, arrow_lat, speeds,
                     cmap="turbo", norm=speed_mesh.norm, angles="xy", scale_units="xy", scale=1,
                     width=0.0045, headwidth=4.5, headlength=5.5, headaxislength=4.5,
                     pivot="middle", zorder=13)
    if accepted.any():
        ax.scatter(vectors[accepted, 0], vectors[accepted, 1], s=16, facecolors="none",
                   edgecolors="#16a34a", linewidths=1.2, label="passes paper-inspired screening", zorder=14)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.text(0.01, 0.99, "Arrow direction = tracked motion · arrow color = speed\nArrow length enlarged 8× for visibility",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "#777", "boxstyle": "round,pad=0.35"},
            zorder=20)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°; negative = °S)")
    ax.set_title(f"Paper-inspired Himawari Band {int(band)} apparent feature motion · 10-minute interval")
    fig.colorbar(speed_mesh, ax=ax, label="Interpolated apparent feature speed (m/s)", shrink=0.8)
    output = io.BytesIO()
    with contextlib.redirect_stdout(io.StringIO()):
        fig.savefig(output, format="png", dpi=115, facecolor="white")
    plt.close(fig)
    return {
        "band": int(band), "variable": variable,
        "files": [x[0] for x in frames], "time_delta_minutes": time_delta_minutes,
        "matched_vectors": len(vectors), "vectors_passing_screening": int(accepted.sum()),
        "correlation_qc_threshold": 0.75, "forward_backward_error_limit_pixels": 2.0,
        "statistics": {"median_speed_m_s": round(float(np.median(speeds)), 2),
                       "p95_speed_m_s": round(float(np.percentile(speeds, 95)), 2),
                       "median_correlation": round(float(np.median(vectors[:, 4])), 3),
                       "median_forward_backward_error_pixels": round(float(np.median(vectors[:, 5])), 2)},
        "preview": {"png_base64": base64.b64encode(output.getvalue()).decode("ascii")},
        "caveat": "Research-only, paper-inspired tracking approximation: averages 7x7 and 33x33 forward/backward correlation surfaces and applies a 0.75 correlation plus 2-pixel forward/backward consistency screen. This is not a reproduction of JMA AMV: no optimal-estimation cloud-height assignment, radiative-transfer/first-guess wind profile, parallax correction, full JMA quality index, or sonde validation is implemented. Himawari Band 7 is 3.9 µm, not the paper's 7.0 µm water-vapor channel (Band 10). Vectors are apparent image-feature motion, not surface wind or runway windshear, and are not for operational use.",
    }


def _tracking_centers(shape, half_search, min_spacing):
    # Large 33x33 targets require a 16-pixel border in addition to search.
    margin = 16 + half_search
    ys = list(range(margin, shape[0] - margin, min_spacing))
    xs = list(range(margin, shape[1] - margin, min_spacing))
    return [(y, x) for y in ys for x in xs]


def _correlation_surface(reference, target, cy, cx, side, search):
    """Pearson correlation at integer displacements for one target box size."""
    radius = side // 2
    template = reference[cy-radius:cy+radius+1, cx-radius:cx+radius+1].astype(float)
    surface = np.full((2*search+1, 2*search+1), -1.0, dtype=float)
    if template.shape != (side, side) or not np.isfinite(template).all():
        return None
    template -= template.mean()
    template_norm = np.sqrt(np.sum(template * template))
    if template_norm <= 1e-8:
        return None
    for iy, dy in enumerate(range(-search, search+1)):
        for ix, dx in enumerate(range(-search, search+1)):
            candidate = target[cy+dy-radius:cy+dy+radius+1, cx+dx-radius:cx+dx+radius+1].astype(float)
            if candidate.shape != (side, side) or not np.isfinite(candidate).all():
                continue
            candidate -= candidate.mean()
            norm = np.sqrt(np.sum(candidate * candidate)) * template_norm
            if norm > 1e-8:
                surface[iy, ix] = float(np.sum(template * candidate) / norm)
    return surface


def _filename_time(filename):
    match = re.search(r"H09_(\d{8})_(\d{4})", filename)
    if not match:
        return None
    try:
        return dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M")
    except ValueError:
        return None
