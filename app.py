#!/usr/bin/env python3
"""Local aviation-windshear briefing assistant (LLM-backed, advisory only)."""
import json
import base64
import csv
import io
import os
import re
import threading
import uuid
import zipfile
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).parent
PORT = int(os.environ.get("PORT", "8080"))
HIMAWARI_JOBS = {}
HIMAWARI_JOBS_LOCK = threading.Lock()

SYSTEM_PROMPT = """You are an aviation-weather training and decision-support assistant focused on low-level windshear near Juanda (WARR/SUB), especially RWY10/28. Never issue or cancel an operational alert, clearance, or go-around instruction. Clearly label your response advisory and require confirmation by qualified meteorologists and ATC.

Use these local training notes as context: windshear criterion is sustained headwind/tailwind or vertical change >=15 kt (7.7 m/s) in lowest 500 m / within 10 km of threshold. Microburst signature: differential radial velocity >=10 m/s over <=4 km, with outflow/couplet, precipitation and supporting turbulence/vertical evidence; do not diagnose from one variable. High reflectivity alone is not an alert: demand vertical continuity at >=2 elevations, RhoHV >0.9, plausible ZDR, velocity/width continuity and motion; otherwise consider clutter/AP. For this Juanda radar, quality masks in the training notes are SNR >=15 dB, SQI >=0.5, RhoHV >=0.7. Vr is radial, not runway headwind: the beam crosses the runway at ~70-80 degrees and headwind shear may be nearly invisible. Isolated large Vr jumps may be aliased; require spatial continuity. X-band attenuation and clutter can degrade data. Fuse radar with AWS/LLWAS at threshold/mid/far, tower wind and pilot reports. State uncertainty, missing data and recency limits. Never invent measurements or claim a forecast without a time sequence.

Uploaded-data rules: distinguish parser-derived observations from visual interpretation and from user text. METAR/SPECI parsing is partial; preserve original report and flag unknown groups. PIREPs and radiosonde text/CSV are evidence only if fields, units, station and time can be identified. NPZ radar transects are summaries, not raw radar volumes. Satellite image interpretation is qualitative and may only be attempted if the selected model supports image input; never claim quantitative satellite-derived winds or cloud-top heights from an ordinary image. Do not combine mismatched times/locations as if simultaneous. Explain unsupported formats, parse failures and stale/missing sources.

Respond in concise sections: Assessment (no evidence / watch / concerning signature, not an operational alert), Evidence, Limitations / missing checks, Suggested briefing wording (explicitly draft/advisory). Use only submitted observations; don't assume absent data are normal."""


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/himawari-download/"):
            job_id = self.path.rsplit("/", 1)[-1]
            with HIMAWARI_JOBS_LOCK:
                job = HIMAWARI_JOBS.get(job_id)
                snapshot = dict(job) if job else None
            if snapshot is None:
                self.send_json(404, {"error": "Download job not found."})
            else:
                self.send_json(200, snapshot)
            return
        if self.path == "/api/status":
            self.send_json(200, {"configured": bool(os.environ.get("OPENAI_API_KEY")), "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini")})
            return
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        body = (ROOT / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/api/pilot-reports":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 100 * 1024 * 1024:
                    self.send_json(413, {"error": "Pilot report file must be no larger than 100 MB."})
                    return
                msg = BytesParser(policy=policy.default).parsebytes(
                    b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + self.rfile.read(length)
                )
                part = next((p for p in msg.iter_parts() if p.get_filename()), None)
                if part is None:
                    self.send_json(400, {"error": "Choose a pilot report .xls, .xlsx, or .csv file."})
                    return
                filename = Path(part.get_filename()).name
                suffix = Path(filename).suffix.lower()
                content = part.get_payload(decode=True) or b""
                if suffix == ".csv":
                    text = content.decode("utf-8-sig", errors="replace")
                    reader = csv.DictReader(io.StringIO(text))
                    columns = reader.fieldnames or []
                    records = list(reader)
                elif suffix in (".xls", ".xlsx"):
                    import pandas as pd
                    engine = "xlrd" if suffix == ".xls" else "openpyxl"
                    frame = pd.read_excel(io.BytesIO(content), engine=engine, dtype=object)
                    columns = [str(x) for x in frame.columns]
                    records = frame.where(pd.notna(frame), None).to_dict(orient="records")
                    records = [{str(k): (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()} for row in records]
                else:
                    self.send_json(400, {"error": "Pilot reports must be .xls, .xlsx, or .csv."})
                    return
                if len(records) > 100000:
                    self.send_json(400, {"error": "Pilot report contains more than 100,000 rows; split the file and retry."})
                    return
                self.send_json(200, {"filename": filename, "columns": columns, "records": records, "row_count": len(records)})
            except ImportError as exc:
                self.send_json(400, {"error": "Excel reader dependency is missing. Install project requirements, including xlrd for .xls files."})
            except Exception as exc:
                self.send_json(400, {"error": "Could not read pilot report: " + str(exc)})
            return
        if self.path == "/api/himawari-download":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 64 * 1024:
                    self.send_json(413, {"error": "Himawari download request is too large."})
                    return
                data = json.loads(self.rfile.read(length) or b"{}")
                from himawari_download import download_himawari
                job_id = uuid.uuid4().hex
                with HIMAWARI_JOBS_LOCK:
                    HIMAWARI_JOBS[job_id] = {"status": "running", "progress": {"completed": 0, "total": 0, "percent": 0, "current_time": "", "stage": "Starting"}}

                def update_progress(progress):
                    with HIMAWARI_JOBS_LOCK:
                        if job_id in HIMAWARI_JOBS:
                            HIMAWARI_JOBS[job_id]["progress"] = progress

                def run_download():
                    try:
                        result = download_himawari(
                            username=str(data.get("username", "")),
                            password=str(data.get("password", "")),
                            start=str(data.get("start", "")),
                            end=str(data.get("end", "")),
                            corners=data.get("corners", []),
                            output_dir=str(data.get("output_dir", "")),
                            interval_minutes=int(data.get("interval_minutes", 10)),
                            progress_callback=update_progress,
                        )
                        with HIMAWARI_JOBS_LOCK:
                            HIMAWARI_JOBS[job_id].update(status="complete", result=result)
                    except Exception as exc:
                        message = str(exc)
                        for secret in (str(data.get("username", "")), str(data.get("password", ""))):
                            if secret:
                                message = message.replace(secret, "[redacted]")
                        with HIMAWARI_JOBS_LOCK:
                            HIMAWARI_JOBS[job_id].update(status="failed", error=message)
                    finally:
                        data.pop("password", None)

                threading.Thread(target=run_download, daemon=True).start()
                self.send_json(202, {"job_id": job_id, "status": "running"})
            except Exception as exc:
                # Never echo submitted credentials in errors/log output.
                self.send_json(400, {"error": "Himawari download failed: " + str(exc)})
            return
        if self.path == "/api/satellite-motion":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1500 * 1024 * 1024:
                    self.send_json(413, {"error": "Selected NetCDF files must total no more than 1.5 GB."})
                    return
                msg = BytesParser(policy=policy.default).parsebytes(
                    b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + self.rfile.read(length)
                )
                parts = list(msg.iter_parts())
                fileparts = [p for p in parts if p.get_filename()]
                if len(fileparts) != 2:
                    self.send_json(400, {"error": "Select exactly two Himawari NetCDF files, 10 minutes apart."})
                    return
                if any(not Path(p.get_filename()).name.lower().endswith((".nc", ".nc4", ".cdf")) for p in fileparts):
                    self.send_json(400, {"error": "Motion approximation accepts NetCDF (.nc/.nc4/.cdf) files."})
                    return
                band_part = next((p for p in parts if p.get_param("name", header="content-disposition") == "band"), None)
                band = (band_part.get_payload(decode=True) or b"all").decode().strip() if band_part else "all"
                from satellite_analysis import process_motion_files
                files = [(Path(p.get_filename()).name, p.get_payload(decode=True) or b"") for p in fileparts]
                self.send_json(200, process_motion_files(files, band))
            except Exception as exc:
                self.send_json(400, {"error": "Could not estimate satellite feature motion: " + str(exc)})
            return
        if self.path in ("/api/satellite-dense-amv", "/api/satellite-cooling"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                max_upload_bytes = (4 * 1024 * 1024 * 1024) if self.path == "/api/satellite-dense-amv" else (1500 * 1024 * 1024)
                if length > max_upload_bytes:
                    limit_label = "4 GB" if self.path == "/api/satellite-dense-amv" else "1.5 GB"
                    self.send_json(413, {"error": f"Selected NetCDF files must total no more than {limit_label}. Regional crops use much less upload bandwidth and memory."})
                    return
                msg = BytesParser(policy=policy.default).parsebytes(
                    b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + self.rfile.read(length)
                )
                fileparts = [part for part in msg.iter_parts() if part.get_filename()]
                minimum = 3 if self.path == "/api/satellite-dense-amv" else 2
                if len(fileparts) < minimum or len(fileparts) > 15:
                    self.send_json(400, {"error": f"Select between {minimum} and 15 Himawari NetCDF files."})
                    return
                if any(not Path(part.get_filename()).name.lower().endswith((".nc", ".nc4", ".cdf")) for part in fileparts):
                    self.send_json(400, {"error": "Satellite sequence analysis accepts NetCDF (.nc/.nc4/.cdf) files."})
                    return
                files = [(Path(part.get_filename()).name, part.get_payload(decode=True) or b"") for part in fileparts]
                from satellite_analysis import process_convective_cooling_files
                if self.path == "/api/satellite-cooling":
                    self.send_json(200, process_convective_cooling_files(files))
                else:
                    from dense_amv_ctv699 import process_uploaded_sequence
                    self.send_json(200, process_uploaded_sequence(files))
            except Exception as exc:
                label = "convective cooling" if self.path == "/api/satellite-cooling" else "dense AMV"
                self.send_json(400, {"error": f"Could not process Himawari {label} sequence: " + str(exc)})
            return
        if self.path == "/api/satellite-file":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 800 * 1024 * 1024:
                    self.send_json(413, {"error": "A Himawari NetCDF file may be up to 800 MB."})
                    return
                msg = BytesParser(policy=policy.default).parsebytes(
                    b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + self.rfile.read(length)
                )
                parts = list(msg.iter_parts())
                filepart = next((part for part in parts if part.get_filename()), None)
                if filepart is None:
                    self.send_json(400, {"error": "Select a Himawari .nc file first."})
                    return
                filename = Path(filepart.get_filename()).name
                if not filename.lower().endswith((".nc", ".nc4", ".cdf")):
                    self.send_json(400, {"error": "Satellite processing accepts NetCDF (.nc/.nc4/.cdf) files."})
                    return
                variable_part = next((part for part in parts if part.get_param("name", header="content-disposition") == "variable"), None)
                variable = (variable_part.get_payload(decode=True) or b"").decode().strip() if variable_part else ""
                from satellite_analysis import process_satellite_file
                result = process_satellite_file(filename, filepart.get_payload(decode=True) or b"", variable)
                self.send_json(200, result)
            except Exception as exc:
                self.send_json(400, {"error": "Could not process satellite NetCDF: " + str(exc)})
            return
        if self.path == "/api/radar-file":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 100 * 1024 * 1024:
                    self.send_json(413, {"error": "Selected radar files must total no more than 100 MB."})
                    return
                msg = BytesParser(policy=policy.default).parsebytes(
                    b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + self.rfile.read(length)
                )
                parts = list(msg.iter_parts())
                fileparts = [part for part in parts if part.get_filename()]
                if not fileparts:
                    self.send_json(400, {"error": "Select a Rainbow .vol file first."})
                    return
                if len(fileparts) > 12:
                    self.send_json(400, {"error": "Select no more than 12 moment files."})
                    return
                if any(not Path(part.get_filename()).name.lower().endswith(".vol") for part in fileparts):
                    self.send_json(400, {"error": "Radar processing accepts Rainbow .vol files."})
                    return
                from radar_analysis import process_uploaded_volumes
                settings_part = next((part for part in parts if part.get_param("name", header="content-disposition") == "snr_min_db"), None)
                snr_min = float((settings_part.get_payload(decode=True) or b"7").decode().strip() or "7") if settings_part else 7.0
                moment_part = next((part for part in parts if part.get_param("name", header="content-disposition") == "display_moment"), None)
                display_moment = (moment_part.get_payload(decode=True) or b"dBZ").decode().strip() if moment_part else "dBZ"
                attenuation_part = next((part for part in parts if part.get_param("name", header="content-disposition") == "attenuation_method"), None)
                attenuation_method = (attenuation_part.get_payload(decode=True) or b"none").decode().strip() if attenuation_part else "none"
                settings = {}
                for part in parts:
                    name = part.get_param("name", header="content-disposition")
                    if name and name.startswith("threshold_"):
                        settings[name[len("threshold_"):]] = float((part.get_payload(decode=True) or b"").decode().strip())
                volumes = [(Path(part.get_filename()).name, part.get_payload(decode=True) or b"") for part in fileparts]
                result = process_uploaded_volumes(volumes, snr_min=snr_min, display_moment=display_moment, thresholds=settings, attenuation_method=attenuation_method)
                self.send_json(200, result)
            except Exception as exc:
                self.send_json(400, {"error": "Could not process Rainbow .vol: " + str(exc)})
            return
        if self.path == "/api/radar":
            try:
                from radar_analysis import analyze_case
                length = int(self.headers.get("Content-Length", "0"))
                if length > 4096:
                    self.send_json(413, {"error": "Radar analysis request is too large."})
                    return
                data = json.loads(self.rfile.read(length) or b"{}")
                result = analyze_case(
                    data.get("data_dir", os.environ.get("TRAINING_TL_DATA", "/Users/aliwardhana/Downloads/Training_TL/00_data/Radar/2025-03-07")),
                    data.get("start", "20250307070000"),
                    data.get("end", "20250307080000"),
                )
                self.send_json(200, result)
            except Exception as exc:
                self.send_json(500, {"error": "Local Rainbow radar analysis failed: " + str(exc)})
            return
        if self.path != "/api/analyze":
            self.send_error(404)
            return
        if not os.environ.get("OPENAI_API_KEY"):
            self.send_json(503, {"error": "Model analysis is unavailable because no provider API key is configured."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 100 * 1024 * 1024:
                self.send_json(413, {"error": "Request too large (maximum 100 MB)."})
                return
            raw = self.rfile.read(length)
            if "multipart/form-data" in self.headers.get("Content-Type", ""):
                msg = BytesParser(policy=policy.default).parsebytes(
                    b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + raw
                )
                obs, files = "", []
                for part in msg.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    content = part.get_payload(decode=True) or b""
                    filename = part.get_filename()
                    if name == "observations":
                        obs = content.decode("utf-8", errors="replace").strip()
                    elif filename and content:
                        files.append((Path(filename).name, part.get_content_type(), content))
                extraction, images = [], []
                for filename, mime, content in files[:12]:
                    extraction.append(extract_file(filename, mime, content))
                    if mime.startswith("image/") and len(content) <= 8 * 1024 * 1024:
                        images.append((mime, base64.b64encode(content).decode("ascii")))
                obs += "\n\nUPLOADED FILE EXTRACTION (parser output; verify against originals):\n" + json.dumps(extraction, ensure_ascii=False)
            else:
                data = json.loads(raw)
                obs = str(data.get("observations", "")).strip()
                images = []
            if not obs:
                self.send_json(400, {"error": "Enter observations before analysis."})
                return
            base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            user_content = obs
            if images:
                user_content = [{"type": "text", "text": obs}]
                user_content.extend({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, b64), "detail": "low"}} for mime, b64 in images[:4])
            payload = json.dumps({
                "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
                "temperature": 0.2,
            }).encode()
            req = Request(base + "/chat/completions", data=payload, headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"})
            with urlopen(req, timeout=60) as response:
                result = json.loads(response.read())
            self.send_json(200, {"analysis": result["choices"][0]["message"]["content"]})
        except (ValueError, KeyError, IndexError) as exc:
            self.send_json(400, {"error": "Could not read the request or model response: " + str(exc)})
        except HTTPError as exc:
            detail = exc.read(2000).decode(errors="replace")
            self.send_json(502, {"error": "Model provider error (HTTP %s): %s" % (exc.code, detail)})
        except (URLError, TimeoutError) as exc:
            self.send_json(502, {"error": "Could not reach the model provider: " + str(exc)})

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))


def extract_file(filename, mime, content):
    """Conservative local extraction; unsupported formats are reported, not guessed."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        text = content.decode("utf-8-sig", errors="replace")
        rows = list(csv.DictReader(io.StringIO(text)))
        return {"file": filename, "type": "CSV table", "columns": list(rows[0].keys()) if rows else [], "row_count": len(rows), "sample_rows": rows[:30]}
    if suffix in (".metar", ".speci", ".txt", ".taf") or mime.startswith("text/"):
        text = content.decode("utf-8", errors="replace")[:12000]
        reports = re.findall(r"(?:METAR|SPECI)\s+[^\n]+", text, re.I)
        if not reports:
            reports = [line.strip() for line in text.splitlines() if line.strip()][:20]
        parsed = []
        for report in reports[:20]:
            parsed.append({"raw": report[:500], "wind": re.findall(r"\b(?:\d{3}|VRB)\d{2,3}(?:G\d{2,3})?(?:KT|MPS|KMH)\b", report), "visibility": re.findall(r"\b(?:\d{4}|CAVOK)\b", report), "temperature_dewpoint_C": re.findall(r"\bM?\d{2}/M?\d{2}\b", report), "pressure_QNH_hPa": re.findall(r"\bQ\d{4}\b", report)})
        kind = "METAR/SPECI-like reports" if reports and any(re.search(r"METAR|SPECI", r, re.I) for r in reports) else "text (unclassified; may contain PIREP or radiosonde data)"
        return {"file": filename, "type": kind, "reports": parsed}
    if suffix == ".npz":
        try:
            import numpy as np
            with np.load(io.BytesIO(content), allow_pickle=False) as data:
                arrays = {}
                for key in data.files:
                    arr = np.asarray(data[key])
                    if arr.dtype.kind in "iuf" and arr.size < 250000:
                        finite = arr[np.isfinite(arr)]
                        arrays[key] = {"shape": list(arr.shape), "finite_count": int(finite.size), "min": float(finite.min()) if finite.size else None, "max": float(finite.max()) if finite.size else None}
                    else:
                        arrays[key] = {"shape": list(arr.shape), "type": str(arr.dtype)}
                return {"file": filename, "type": "numeric NPZ summary (not raw-volume decoding)", "arrays": arrays}
        except Exception as exc:
            return {"file": filename, "error": "Could not parse NPZ: " + str(exc)}
    if suffix == ".vol":
        try:
            import wradlib.io as wio
            volume = wio.read_rainbow(io.BytesIO(content))
            root = volume.get("volume", volume)
            scan = root.get("scan", {}) if isinstance(root, dict) else {}
            slices = scan.get("slice", []) if isinstance(scan, dict) else []
            if isinstance(slices, dict):
                slices = [slices]
            sweeps = []
            for index, sweep in enumerate(slices):
                slicedata = sweep.get("slicedata", {}) if isinstance(sweep, dict) else {}
                moments = {}
                if isinstance(slicedata, dict):
                    for moment, node in slicedata.items():
                        if not isinstance(node, dict):
                            continue
                        values = node.get("data", node.get("rawdata"))
                        if values is None:
                            continue
                        try:
                            import numpy as np
                            array = np.asarray(values)
                            moments[moment] = {"shape": list(array.shape), "dtype": str(array.dtype), "scale": node.get("@scale"), "offset": node.get("@offset"), "unit": node.get("@unit")}
                        except Exception:
                            moments[moment] = {"metadata": "present; shape unavailable", "scale": node.get("@scale"), "offset": node.get("@offset")}
                sweeps.append({"index": index, "elevation_deg": sweep.get("posangle"), "moments": moments})
            return {"file": filename, "type": "Rainbow .vol metadata / moment inventory via wradlib", "sweep_count": len(sweeps), "sweeps": sweeps, "calibrated_measurements": False, "note": "No project-specific Rainbow scaling/quality-control or georeferenced windshear calculation has been applied. Treat these as format inventory only."}
        except Exception as exc:
            return {"file": filename, "error": "Rainbow .vol read failed; ensure wradlib is installed and file is supported: " + str(exc)}
    if mime.startswith("image/"):
        return {"file": filename, "type": "image; attached for qualitative model review if supported", "bytes": len(content)}
    return {"file": filename, "type": "unsupported format", "mime": mime, "bytes": len(content)}


if __name__ == "__main__":
    print("Windshear Agent running at http://localhost:%d" % PORT)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
