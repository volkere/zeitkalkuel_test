
# Photo Metadata Suite

Detect faces, estimate age & gender, and extract GPS location (EXIF) from digitized photos.  
Includes a **CLI** and a **Streamlit multi-page UI** (Enroll + Annotate).

## Features
- Face detection & embeddings (InsightFace `buffalo_l`).
- Age & gender estimation (approximate).
- Known-person matching via embeddings database (`embeddings.pkl`).
- EXIF GPS extraction with optional reverse geocoding to human-readable address.
- Streamlit UI with drag & drop, bounding boxes, JSON export.
- CLI for batch processing.

> ⚠️ **Use responsibly:** Face analysis and attribute inference can be biased and regulated. Ensure you have the right to process the images and comply with local laws (see `docs/PRIVACY.md`).

---

## Quickstart (UI)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Quickstart (CLI)
```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# (optional) build embeddings from a labeled gallery
python -m app.main enroll --gallery ./gallery --db embeddings.pkl

# annotate a folder
python -m app.main annotate --input ./photos --out output.json --recursive --reverse-geocode
```

## Repo layout
```
app/                  # Python package (engine, CLI)
pages/                # Streamlit pages (Enroll, Annotate)
streamlit_app.py      # Streamlit entry
requirements.txt      # runtime deps
pyproject.toml        # package metadata + console script
docs/                 # documentation
.github/workflows/    # CI (lint/build)
```

## Install as a package (optional)
```bash
pip install -e .
# now the CLI is available as:
photo-meta annotate --input photos --out output.json --recursive
```

## License
MIT — see `LICENSE`.
