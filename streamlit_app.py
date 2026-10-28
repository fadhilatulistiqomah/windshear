#!/usr/bin/env python3
"""Streamlit interface for local Juanda radar, satellite and upper-air tools."""
from __future__ import annotations

import io
import json
import os
import csv
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Windshear Agent · Juanda", page_icon="🌦️", layout="wide")

ASSETS = Path(__file__).parent / "assets"
HEADER_IMAGE_NAMES = ("header.jpg", "header.jpeg", "header.png", "header.webp")


def accept(*extensions):
    """Streamlit matches upload extensions case-sensitively, so allow both cases."""
    return [form for ext in extensions for form in (ext, ext.upper())]


VOL_TYPES = accept("vol")
NC_TYPES = accept("nc", "nc4", "cdf")
CSV_TYPES = accept("csv")
OBS_TYPES = accept("csv", "txt", "metar", "speci")
SHEET_TYPES = accept("xls", "xlsx", "csv")


def header_image_path():
    """Return the first available header photo in assets/, if any."""
    for name in HEADER_IMAGE_NAMES:
        candidate = ASSETS / name
        if candidate.is_file():
            return candidate
    return None


header_text, header_photo = st.columns([2, 1], vertical_alignment="top")
with header_text:
    st.markdown(
        '<div class="header-block">'
        '<p style="font-size:1.3rem;font-weight:800;color:#1d4ed8;line-height:1.25;margin:0 0 0.45rem">'
        "Dr. Ali Wardhana | Innovation and Modeling of Aviation Meteorology (IMAM)"
        "</p>"
        '<h1 style="margin:0;padding:0">Read the signal.<br>Respect the uncertainty</h1>'
        '<p style="font-size:1.3rem;line-height:1.55;margin:0.5rem 0 0;opacity:0.78">'
        "&quot;Observe and Forecast with Precision, Fly with Decision.&quot; &mdash; "
        "Highlights the bridge between meteorology data and decisive flight-deck actions."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )
with header_photo:
    photo = header_image_path()
    if photo is not None:
        st.image(str(photo), width=300)
    else:
        st.caption(f"Add a header photo as `assets/{HEADER_IMAGE_NAMES[0]}` to display it here.")
st.warning("Advisory and research only. Derived radar and satellite products are provisional, are not validated operational warnings, and must not be used alone for aviation decisions.")


def uploaded_pairs(files):
    return [(f.name, f.getvalue()) for f in files]


def show_b64_image(value, caption=None):
    import base64
    st.image(base64.b64decode(value), caption=caption, use_container_width=True)


def radar_scan_moment(filename):
    """Return (scan timestamp, moment) from Rainbow names like 2025030700011100dBuZ.vol."""
    import re
    match = re.match(r"^(\d{14})\d*([A-Za-z]+)\.vol$", Path(filename).name, re.IGNORECASE)
    return (match.group(1), match.group(2).lower()) if match else (None, None)


def validate_attenuation_inputs(volumes):
    """Check the uploaded filenames before decoding; True means proceed."""
    parsed = [(name, *radar_scan_moment(name)) for name, _ in volumes]
    parsed = [(name, stamp, moment) for name, stamp, moment in parsed if stamp]
    if not parsed:
        return True  # Unrecognised naming: let the decoder decide.
    phidp = [(name, stamp) for name, stamp, moment in parsed if moment == "phidp"]
    dbu = [(name, stamp) for name, stamp, moment in parsed if moment == "dbuz"]
    if not phidp:
        st.error("Attenuation correction needs a PhiDP volume. Add the file ending in `PhiDP.vol` from the same scan in the **PhiDP input** box.")
        return False
    if not dbu:
        st.error("Attenuation correction needs the uncorrected reflectivity. Add the file ending in `dBuZ.vol` to the moment files.")
        return False
    if not {stamp for _, stamp in phidp} & {stamp for _, stamp in dbu}:
        labels = lambda items: ", ".join(f"`{Path(name).name}`" for name, _ in items[:3])
        st.error(
            "These files cannot be paired because the scan times differ:\n\n"
            f"- dBuZ: {labels(dbu)}\n"
            f"- PhiDP: {labels(phidp)}\n\n"
            "Select the dBuZ and PhiDP volumes with the same scan timestamp."
        )
        return False
    return True


def radar_tab():
    from radar_analysis import process_uploaded_volumes
    st.subheader("Rainbow radar volumes")
    st.caption("Select matching moment files from the same scan. Quality masks only use matching fields present in the uploaded files.")
    files = st.file_uploader("Rainbow moment files (.vol)", type=VOL_TYPES, accept_multiple_files=True, key="radar_files")
    with st.expander("Quality-control thresholds", expanded=False):
        c1, c2, c3 = st.columns(3)
        thresholds = {
            "snr_min": c1.number_input("SNR minimum (dB)", value=7.0),
            "snr_max": c1.number_input("SNR maximum (dB)", value=75.0),
            "dbz_min": c2.number_input("Reflectivity minimum (dBZ)", value=0.0),
            "dbz_max": c2.number_input("Reflectivity maximum (dBZ)", value=75.0),
            "rhohv_min": c3.number_input("RhoHV minimum", min_value=0.0, max_value=1.0, value=0.85),
            "zdr_min": c1.number_input("ZDR minimum (dB)", value=0.0),
            "zdr_max": c1.number_input("ZDR maximum (dB)", value=6.0),
            "v_min": c2.number_input("V minimum (m/s)", value=-25.0),
            "v_max": c2.number_input("V maximum (m/s)", value=25.0),
            "w_min": c3.number_input("W minimum (m/s)", value=0.0),
            "w_max": c3.number_input("W maximum (m/s)", value=4.8),
            "kdp_min": c3.number_input("KDP minimum (deg/km)", value=-5.0),
            "kdp_max": c3.number_input("KDP maximum (deg/km)", value=15.0),
        }
    a, b = st.columns(2)
    moment = a.selectbox("Display moment", ["dBZ", "dBuZ", "V", "W", "RhoHV", "ZDR", "PhiDP", "KDP", "SNR", "SQI"])
    attenuation = b.selectbox("C-band attenuation correction", ["none", "linear_phidp", "zphi"], format_func=lambda x: {"none": "None", "linear_phidp": "Linear PhiDP", "zphi": "ZPHI"}[x])
    phidp = None
    if attenuation != "none":
        st.caption("Correction always uses the uncorrected **dBuZ** product plus the matching **PhiDP** volume from the same scan. Include the dBuZ file above and the PhiDP file here.")
        phidp = st.file_uploader("PhiDP input for attenuation correction (.vol)", type=VOL_TYPES, key="radar_phidp")
    st.caption("Attenuation parameters are provisional radar-specific assumptions; validate before scientific interpretation. KDP is a noisy finite-difference estimate.")
    if st.button("Process radar volumes", type="primary", key="run_radar"):
        if not files and phidp is None:
            st.error("Choose at least one .vol file.")
        else:
            volumes = uploaded_pairs(files or [])
            if phidp is not None:
                # Avoid sending the same volume twice if it was also selected above.
                existing = {(name, len(content)) for name, content in volumes}
                if (phidp.name, len(phidp.getvalue())) not in existing:
                    volumes.append((phidp.name, phidp.getvalue()))
            if attenuation != "none" and not validate_attenuation_inputs(volumes):
                return
            try:
                with st.spinner("Decoding, aligning and QC-filtering moments…"):
                    result = process_uploaded_volumes(volumes, display_moment=moment, thresholds=thresholds, attenuation_method=attenuation)
                st.info(f"Moments: {', '.join(result['available_moments'])} · Display: {result['display_moment']} · attenuation: {result['attenuation_method']}")
                st.warning(result["caveat"])
                for sweep in result["sweeps"]:
                    st.markdown(f"**Sweep {sweep['index'] + 1} · elevation {sweep['elevation_deg']:.2f}° · {sweep['range_max_km']} km**")
                    show_b64_image(sweep["preview"]["png_base64"])
                    st.json(sweep["statistics"])
            except Exception as exc:
                message = str(exc)
                st.error(f"Radar processing failed: {message}")
                if "PhiDP input is required" in message:
                    st.info("Add the file ending in `PhiDP.vol` from the same scan in the **PhiDP input** box, and keep the matching `dBuZ.vol` selected above.")


def satellite_tab():
    from satellite_analysis import process_satellite_file, process_motion_files, process_convective_cooling_files
    from dense_amv_ctv699 import process_uploaded_sequence
    from himawari_download import download_himawari
    st.subheader("Himawari satellite")
    st.caption("Upload limit is 800 MB per file, set in `.streamlit/config.toml`. Full-disk R21 scenes are often ~600 MB, so prefer the regional ROI crops produced by the downloader below—they are much faster to upload and use far less memory.")
    with st.expander("Download from JAXA P-Tree", expanded=False):
        st.caption("Credentials are used for this request only. Files are written on the server, not downloaded to your browser.")
        username = st.text_input("JAXA username", key="jaxa_user")
        password = st.text_input("JAXA password", type="password", key="jaxa_password")
        c1, c2 = st.columns(2)
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
        start = c1.datetime_input("Start time (UTC)", value=now_utc)
        end = c2.datetime_input("End time (UTC)", value=now_utc)
        c1, c2 = st.columns(2)
        interval = c1.selectbox("Interval (minutes)", [10, 20, 30, 60])
        output_dir = c2.text_input("Server output folder", value="./Himawari_9")
        c1, c2 = st.columns(2)
        south = c1.number_input("South latitude", value=-8.15, format="%.4f")
        north = c2.number_input("North latitude", value=-6.8, format="%.4f")
        c1, c2 = st.columns(2)
        west = c1.number_input("West longitude", value=112.1, format="%.4f")
        east = c2.number_input("East longitude", value=113.25, format="%.4f")
        if st.button("Download and crop scenes", key="download_hima"):
            progress = st.progress(0, text="Starting download…")
            try:
                def update(p):
                    progress.progress(min(100, int(p.get("percent", 0))), text=f"{p.get('stage', '')} · {p.get('current_time', '')}")
                result = download_himawari(username, password, start.strftime("%Y%m%d%H%M"), end.strftime("%Y%m%d%H%M"), [[south, west], [south, east], [north, east], [north, west]], output_dir, interval, update)
                st.success(f"Saved {len(result['cropped_downloaded'])} cropped scene(s) to {result['output_dir']}")
                st.json({k: v for k, v in result.items() if k not in ("full_disk_files",)})
            except Exception as exc:
                st.error(f"Download failed: {exc}")
            finally:
                # Streamlit widget values live only for this browser session;
                # avoid logging or persisting submitted credentials elsewhere.
                pass

    st.markdown("#### Plot one NetCDF field")
    single = st.file_uploader("Himawari NetCDF (.nc/.nc4/.cdf)", type=NC_TYPES, key="sat_single")
    variable = st.text_input("Variable name (optional; auto-select Band 13 where available)")
    if st.button("Plot satellite field", key="plot_sat"):
        if not single:
            st.error("Choose a NetCDF file first.")
        else:
            try:
                with st.spinner("Reading and plotting Juanda-area field…"):
                    out = process_satellite_file(single.name, single.getvalue(), variable)
                st.caption(f"{out['filename']} · {out['variable']} · {out['long_name']} · {out['units']} · grid {out['grid_shape']}")
                show_b64_image(out["preview"]["png_base64"])
                st.json(out["statistics"])
                st.warning(out["caveat"])
                st.session_state["sat_variables"] = out["available_variables"]
            except Exception as exc:
                st.error(f"Satellite plotting failed: {exc}")

    st.markdown("#### Two-scene feature-motion approximation")
    c1, c2 = st.columns(2)
    earlier = c1.file_uploader("Earlier scene", type=NC_TYPES, key="motion_1")
    later = c2.file_uploader("Later scene (10 minutes after)", type=NC_TYPES, key="motion_2")
    if st.button("Estimate motion for Bands 3, 7 and 13", key="run_motion"):
        if not earlier or not later:
            st.error("Choose both scenes.")
        else:
            try:
                with st.spinner("Tracking image features…"):
                    result = process_motion_files(uploaded_pairs([earlier, later]))
                st.warning(result["caveat"])
                for band in result["bands"]:
                    st.markdown(f"**Band {band['band']} · {band['variable']} · {band['time_delta_minutes']} minute interval**")
                    show_b64_image(band["preview"]["png_base64"])
                    st.json({k: v for k, v in band.items() if k not in ("preview", "caveat")})
                    st.caption(band["caveat"])
            except Exception as exc:
                st.error(f"Motion estimate failed: {exc}")

    st.markdown("#### Dense cloud-motion vectors")
    dense = st.file_uploader("Select at least 3 consecutive scenes, exactly 10 minutes apart", type=NC_TYPES, accept_multiple_files=True, key="dense_files")
    if st.button("Process dense motion sequence", key="run_dense"):
        if len(dense or []) < 3: st.error("Select at least three scenes.")
        else:
            try:
                with st.spinner("Calculating dense cloud motion…"):
                    result = process_uploaded_sequence(uploaded_pairs(dense))
                st.warning(result["caveat"])
                for product in result["products"]:
                    st.markdown(f"**{product['time_utc']} · {product['valid_vectors']} valid vectors · {product['good_vectors']} good**")
                    for key in ("speed_png_base64", "direction_png_base64", "qc_png_base64"):
                        show_b64_image(product[key], key.replace("_png_base64", "").replace("_", " "))
            except Exception as exc: st.error(f"Dense motion failed: {exc}")

    st.markdown("#### Convective cooling / cold-cloud evolution")
    cooling_files = st.file_uploader("Select 2–15 scenes with matching coverage", type=NC_TYPES, accept_multiple_files=True, key="cooling_files")
    if st.button("Process cooling sequence", key="run_cooling"):
        if len(cooling_files or []) < 2: st.error("Select at least two scenes.")
        else:
            try:
                with st.spinner("Calculating Band 13 diagnostics…"):
                    result = process_convective_cooling_files(uploaded_pairs(cooling_files))
                show_b64_image(result["summary_png_base64"], "Cold-cloud evolution")
                show_b64_image(result["cooling_png_base64"], "Apparent brightness-temperature change")
                st.dataframe(pd.DataFrame(result["statistics"]), use_container_width=True)
                st.dataframe(pd.DataFrame(result["cold_core_centroids"]), use_container_width=True)
                st.warning(result["caveat"])
            except Exception as exc: st.error(f"Cooling analysis failed: {exc}")


def sounding_panel(prefix):
    st.caption("CSV profile plots run locally on the server. Height is measured when present, otherwise estimated from pressure using a standard atmosphere.")
    file = st.file_uploader("Radiosonde / pilot-balloon CSV", type=CSV_TYPES, key=f"{prefix}_sounding_csv")
    if file:
        try:
            def normalize(value): return "".join(ch for ch in str(value).lower() if ch.isalnum())
            # Sounding exports may contain metadata and diagnostics with a
            # different field count from the actual profile. csv.reader keeps
            # those ragged records intact; pandas' default C parser rejects them.
            rows = list(csv.reader(io.StringIO(file.getvalue().decode("utf-8-sig", errors="replace"))))
            special_header = next((i for i, row in enumerate(rows) if any("obstime" in normalize(v) for v in row) and any("press0" in normalize(v) for v in row)), None)
            if special_header is not None:
                header = [v.strip() for v in rows[special_header]]
                records = rows[special_header + 1:]
            else:
                header_index = next((i for i, row in enumerate(rows) if row and not row[0].lstrip().startswith("#")), None)
                if header_index is None: raise ValueError("CSV is empty or contains only comments.")
                header = [v.strip() for v in rows[header_index]]
                records = rows[header_index + 1:]
            width = len(header)
            records = [row[:width] + [None] * max(0, width - len(row)) for row in records if any(cell.strip() for cell in row)]
            frame = pd.DataFrame(records, columns=header)
            normalized = {normalize(c): c for c in frame.columns}
            def col(*names): return next((normalized[n] for n in names if n in normalized), None)
            pcol, tcol = col("pressure", "pressurehpa", "press", "press0", "pmb"), col("temperature", "temperaturec", "temp", "temp0", "tdegc")
            if pcol is None or tcol is None: raise ValueError("Could not identify pressure and temperature columns in this CSV format.")
            dcol, rhcol = col("dewpoint", "dewpointc", "dewpt", "td"), col("relativehumidity", "humidity", "rh", "humi0")
            zcol, wscol, wdcol = col("height", "heightm", "altitude", "altitudem"), col("windspeed", "windspeedms", "ws"), col("winddirection", "winddirectiondeg", "wd")
            profile = pd.DataFrame({"p": pd.to_numeric(frame[pcol], errors="coerce"), "T": pd.to_numeric(frame[tcol], errors="coerce")})
            profile["Td"] = pd.to_numeric(frame[dcol], errors="coerce") if dcol else np.nan
            profile["RH"] = pd.to_numeric(frame[rhcol], errors="coerce") if rhcol else np.nan
            if dcol is None and rhcol:
                t, rh = profile["T"], profile["RH"]
                valid = rh.gt(0) & rh.le(100)
                gamma = np.log(rh[valid] / 100) + 17.625 * t[valid] / (243.04 + t[valid])
                profile.loc[valid, "Td"] = 243.04 * gamma / (17.625 - gamma)
            profile["z"] = pd.to_numeric(frame[zcol], errors="coerce") if zcol else np.nan
            profile["ws"] = pd.to_numeric(frame[wscol], errors="coerce") if wscol else np.nan
            profile["wd"] = pd.to_numeric(frame[wdcol], errors="coerce") if wdcol else np.nan
            profile = profile[(profile["p"] > 0) & profile["p"].le(1100) & profile["T"].between(-100, 60)].sort_values("p", ascending=False)
            if len(profile) < 5: raise ValueError(f"Only {len(profile)} valid pressure levels found.")
            profile["height_km"] = profile.z / 1000 if profile.z.notna().any() else 44330 * (1 - (profile.p / 1013.25) ** 0.1903) / 1000
            st.caption(f"{file.name} · {len(profile)} valid levels")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_yscale("log"); ax.set_ylim(min(1050, profile.p.max()), max(10, profile.p.min()))
            ax.set_ylabel("Pressure (hPa)"); ax.set_xlabel("Temperature (°C; skewed x-coordinate)"); ax.grid(True, which="both", alpha=.3)
            for temp_col, color, label in (("T", "tab:red", "Temperature"), ("Td", "tab:green", "Dewpoint")):
                good = profile[temp_col].notna()
                ax.plot(profile.loc[good, temp_col] + 35 * np.log(1000 / profile.loc[good, "p"]), profile.loc[good, "p"], color=color, label=label)
            ax.legend(loc="best"); ax.set_title("Skew-T Log-P (temperature axis skewed 35°C per natural-log pressure unit)")
            ax2 = ax.twinx(); ticks = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
            ticks = [p for p in ticks if profile.p.min() <= p <= profile.p.max()]
            ax2.set_yscale("log"); ax2.set_ylim(ax.get_ylim()); ax2.set_yticks(ticks); ax2.set_yticklabels([f"{np.interp(p, profile.p[::-1], profile.height_km[::-1]):.1f}" for p in ticks]); ax2.set_ylabel("Height (km; approximate if not measured)")
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            charts = st.columns(3)
            for axis, value, title, color, limits in zip(charts, ["ws", "wd", "RH"], ["Wind speed (m/s)", "Wind direction (°)", "Relative humidity (%)"], ["tab:blue", "purple", "teal"], [(0, None), (0, 360), (0, 100)]):
                fig, ax = plt.subplots(figsize=(4.4, 5.5)); data = profile.dropna(subset=[value]);
                if len(data): ax.plot(data[value], data.height_km, color=color); ax.set_xlim(*limits) if limits[1] is not None else None
                ax.set_xlabel(title); ax.set_ylabel("Height (km)"); ax.grid(alpha=.3); axis.pyplot(fig, use_container_width=True); plt.close(fig)
            with st.expander("Parsed sounding table"): st.dataframe(profile, use_container_width=True)
        except Exception as exc: st.error(f"Could not plot sounding: {exc}")
    for label, key in (("Wind profiler CSV", f"{prefix}_profiler_csv"), ("LIDAR CSV", f"{prefix}_lidar_csv")):
        f = st.file_uploader(label, type=CSV_TYPES, key=key)
        if f:
            try: st.dataframe(pd.read_csv(io.BytesIO(f.getvalue())), use_container_width=True)
            except Exception as exc: st.error(f"Could not read {label}: {exc}")


METAR_PATTERNS = {
    "wind": r"\b(?:\d{3}|VRB)\d{2,3}(?:G\d{2,3})?(?:KT|MPS|KMH)\b",
    "visibility": r"\b(?:\d{4}|CAVOK)\b",
    "temperature_dewpoint_C": r"\bM?\d{2}/M?\d{2}\b",
    "qnh_hPa": r"\bQ\d{4}\b",
}


def parse_metar(report):
    """Extract the common groups from one raw METAR/SPECI string; keep it unmodified."""
    import re
    text = str(report).strip()
    parsed = {"raw": text}
    for field, pattern in METAR_PATTERNS.items():
        found = re.findall(pattern, text)
        parsed[field] = found[0] if found else ""
    return parsed


def show_observation_table(frame):
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.caption("Parsing is partial. Compare against the original report before using any field.")


def surface_tab():
    st.subheader("Surface Analyze")
    st.caption("Surface observations only: METAR/SPECI reports and LLWAS data. Radiosonde, wind profiler and LIDAR sources are in Upper Air Analyze. METAR parsing is partial; LLWAS is displayed as supplied.")
    st.markdown("##### METAR / SPECI reports")
    metar_file = st.file_uploader("METAR/SPECI file (.csv or .txt)", type=OBS_TYPES, key="surface_metar")
    if metar_file:
        try:
            text = metar_file.getvalue().decode("utf-8-sig", errors="replace")
            if metar_file.name.lower().endswith(".csv"):
                source = pd.read_csv(io.StringIO(text), dtype=object, encoding="utf-8-sig")
                column = st.selectbox("Column containing the raw report", list(source.columns), key="surface_metar_column")
                reports = source[column].dropna().astype(str).tolist()
                st.dataframe(source, use_container_width=True, hide_index=True)
            else:
                reports = [line.strip() for line in text.splitlines() if line.strip()]
            parsed = pd.DataFrame([parse_metar(report) for report in reports])
            if parsed.empty:
                st.info("No report lines found in this file.")
            else:
                st.markdown(f"**{len(parsed)} report(s) parsed**")
                show_observation_table(parsed)
        except Exception as exc:
            st.error(f"Could not read METAR/SPECI file: {exc}")

    st.markdown("##### LLWAS observations")
    st.caption("Low-level windshear alerting system data. Threshold/mid/far and tower winds should be fused with radar and pilot reports before any interpretation.")
    llwas_file = st.file_uploader("LLWAS CSV", type=CSV_TYPES, key="surface_llwas")
    if llwas_file:
        try:
            frame = pd.read_csv(io.BytesIO(llwas_file.getvalue()), encoding="utf-8-sig")
            if frame.empty:
                st.info("The LLWAS file has no data rows.")
            else:
                show_observation_table(frame)
                numeric = frame.select_dtypes("number")
                if not numeric.empty:
                    st.markdown("**Numeric summary**")
                    st.dataframe(numeric.describe().T, use_container_width=True)
                st.caption("Column meanings vary by LLWAS export; confirm sensor, height and units before use.")
        except Exception as exc:
            st.error(f"Could not read LLWAS CSV: {exc}")


def upper_air_tab():
    st.subheader("Upper Air Analyze")
    st.caption("Skew-T sounding plots and upper-air CSV previews.")
    sounding_panel("upper")


def pilot_tab():
    st.subheader("Pilot report date lookup")
    file = st.file_uploader("Pilot report file (.xls, .xlsx, .csv)", type=SHEET_TYPES, key="pilot_file")
    if not file: return
    try:
        if Path(file.name).suffix.lower() == ".csv": frame = pd.read_csv(io.BytesIO(file.getvalue()), dtype=object, encoding="utf-8-sig")
        else: frame = pd.read_excel(io.BytesIO(file.getvalue()), engine="xlrd" if file.name.lower().endswith(".xls") else "openpyxl", dtype=object)
        if frame.empty: st.info("No data rows in file."); return
        date_column = st.selectbox("Report date column", list(frame.columns))
        accident_date = st.date_input("Accident date", value=date.today())
        if st.button("Show reports for selected date", type="primary"):
            dates = pd.to_datetime(frame[date_column], errors="coerce", dayfirst=False).dt.date
            matches = frame.loc[dates == accident_date]
            st.success(f"{len(matches)} matching row(s) in {file.name} for {accident_date.isoformat()}.")
            if matches.empty: st.info("No rows matched the selected date.")
            else: st.dataframe(matches, use_container_width=True, hide_index=True)
    except Exception as exc: st.error(f"Could not read pilot report: {exc}")


tabs = st.tabs(["Radar Analyze", "Satellite Analyze", "Surface Analyze", "Upper Air Analyze", "Pilot Report"])
with tabs[0]: radar_tab()
with tabs[1]: satellite_tab()
with tabs[2]: surface_tab()
with tabs[3]: upper_air_tab()
with tabs[4]: pilot_tab()

st.caption("Derived radar attenuation and satellite motion/cooling displays are provisional research products only—not validated operational decisions.")
