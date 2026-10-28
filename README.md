# Juanda Windshear Agent

A local prototype for drafting advisory windshear assessments from manually entered observations and uploaded files. It uses an OpenAI-compatible chat-completions API; it does **not** provide live alerts.

## Run with Streamlit

Requires Python 3.9+ and the packages in `requirements.txt`:

```sh
pip install -r requirements.txt
streamlit run streamlit_app.py
```

To deploy on Streamlit Community Cloud, connect this GitHub repository and set `streamlit_app.py` as the app's main file.

The app's upload workflows do not depend on files from a developer's computer. Optional local batch-analysis data defaults to folders under this repository: `data/radar/2025-03-07`, `data/himawari/20250307`, `data/KKOP/kkop.geojson`, and `data/map_cache`. Add the corresponding files to the repository if you want them bundled with a deployment, or set `WINDSHEAR_DATA_DIR`, `TRAINING_TL_DATA`, `HIMAWARI_INPUT_DIR`, `KKOP_GEOJSON`, and `WINDSHEAR_MAP_CACHE_DIR` to server-side paths. Generated batch outputs default to `outputs/`; set `WINDSHEAR_OUTPUT_DIR` to change that location. Uploaded dense-motion results are temporary unless an output folder is explicitly provided.

The separate prototype HTTP server remains available with `python3 app.py` on port 8080. For its OpenAI-compatible model integration, configure `OPENAI_API_KEY` and, optionally, `OPENAI_BASE_URL` and `OPENAI_MODEL` through the server environment.

## Current upload support (prototype)

- METAR/SPECI-like text: partial extraction of wind, visibility, temperature/dewpoint and QNH groups; the original report is retained as context.
- CSV: column names, row count and first 30 rows are passed as context (may include sensitive or irrelevant data; inspect before upload).
- Rainbow `.vol`: uses `wradlib` to inspect sweeps and moment metadata. This prototype does not yet apply the project-specific Rainbow scaling, quality masks, geo-referencing, or windshear calculations, so the result is only a file/moment inventory—not measured radar evidence.
- Local case analysis: reads timestamped files under `data/radar/2025-03-07` by default (or `TRAINING_TL_DATA`), pairs available Rainbow moment files, applies Rainbow min/max/depth scaling and summarizes the 0–10 km extended RWY10 final corridor at 0.4°, 0.8°, and 1.8°. It applies SNR/SQI/RhoHV masks only when those matching moment files are available. Radial-velocity spread is explicitly not interpreted as runway headwind or as a standalone microburst decision.
- NPZ: small numeric arrays are reduced to shape/range summaries; this is not a substitute for decoding the source radar volume.
- Images (PNG/JPEG/WebP/TIFF): attached for qualitative review if the chosen model supports vision. Do not expect scientific Himawari retrievals, calibrated cloud-top temperatures or wind derivation from images.
- Text files: passed as text with light METAR/SPECI-like group extraction. PIREP and radiosonde interpretation is not yet a validated parser.

Uploads are processed in memory and sent to the configured model provider. Up to 100 MB per request and 12 files are accepted; at most four images (8 MB each) are forwarded. Use sanitized, non-sensitive test data. Large posts are still subject to memory limits of this simple development server.

## Standalone Rainbow `.vol` viewer

For reliable direct display without the web API, run the `wradlib` viewer:

```sh
python3 view_rainbow_vol.py "/path/to/2025030707191200dBZ.vol"
python3 view_rainbow_vol.py "/path/to/file.vol" --sweep 0 --save ppi.png
```

By default it displays all elevation sweeps with `matplotlib`; `--sweep` selects one zero-based sweep and `--save` writes an image. It prints a measurement summary per sweep. One Rainbow `.vol` file is one moment/product: the viewer does not combine moments, perform quality control or make a windshear decision.

## Next validated-data work

Before using this for decisions, apply and test the site's Rainbow moment scaling and QC using representative `.vol` samples; implement georeferencing and deterministic feature calculations; add radiosonde/PIREP schemas, geospatial/time alignment, provider vision checks, and a labeled case dataset with meteorologist review. Himawari products should be ingested as calibrated scientific data through an authoritative source—not inferred from arbitrary screenshots. Do not produce an automatic operational decision or alert.

## Safety and scope

- Advisory/training aid only; never an operational alert, clearance or go-around recommendation.
- Uses thresholds and Juanda-specific caveats from `../rwy10_lecture/README_lecture.md`.
- No live-data ingestion, authentication, or persistence. Keep it bound to localhost; do not expose this prototype to a network.
- Review every output against current approved procedures and qualified meteorological/ATC judgment.
