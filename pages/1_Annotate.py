
import io, json, tempfile
from typing import List, Dict, Any
import streamlit as st
import numpy as np
from PIL import Image
import cv2

from app.face_recognizer import FaceEngine, GalleryDB
from app.location import extract_exif_gps, reverse_geocode

st.title("🖼️ Annotate: Fotos analysieren")
st.caption("Gesichter, Alter/Geschlecht, EXIF-GPS; optional Namensmatching mit embeddings.pkl.")

with st.sidebar:
    det = st.slider("Detector size", 320, 1024, 640, 64, key="det_annot")
    do_reverse = st.checkbox("Reverse geocode GPS (Internet)", value=False)
    threshold = st.slider("Identity threshold (cosine)", 0.3, 0.9, 0.55, 0.01)
    gallery_file = st.file_uploader("Embeddings DB (embeddings.pkl)", type=["pkl"], key="db_upload")
    files = st.file_uploader("Bilder hochladen", type=["jpg","jpeg","png","bmp","webp","tif","tiff"], accept_multiple_files=True)

if "engine_annot" not in st.session_state or st.session_state.get("det_annot_state") != det:
    st.session_state["engine_annot"] = FaceEngine(det_size=(det, det))
    st.session_state["det_annot_state"] = det

db = None
if gallery_file is not None:
    import pickle
    try:
        db = GalleryDB()
        db.people = pickle.load(gallery_file)
        st.success(f"Embeddings geladen: {len(db.people)} Personen.")
    except Exception as e:
        st.error(f"Fehler beim Laden der Embeddings: {e}")

def draw_boxes(img_bgr, persons):
    img = img_bgr.copy()
    for p in persons:
        x1,y1,x2,y2 = map(int, p["bbox"])
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        label = []
        if p.get("name"):
            sim = f" ({p['similarity']:.2f})" if p.get("similarity") is not None else ""
            label.append(p["name"] + sim)
        if p.get("gender"):
            label.append(p["gender"])
        if p.get("age") is not None:
            label.append(str(p["age"]))
        txt = " | ".join(label) if label else f"{p.get('prob', 1.0):.2f}"
        cv2.putText(img, txt, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
    return img

results: List[Dict[str, Any]] = []

if files:
    for up in files:
        data = up.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        faces = st.session_state["engine_annot"].analyze(img_bgr)
        persons = []
        for f in faces:
            name, sim = (None, None)
            if db:
                n, s = db.match(f["embedding"], threshold=threshold)
                name, sim = (n, s)
            persons.append({
                "bbox": f["bbox"],
                "prob": f["prob"],
                "name": name,
                "similarity": sim,
                "age": f["age"],
                "gender": f["gender"]
            })

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            image.save(tmp.name, format="JPEG")
            loc = extract_exif_gps(tmp.name)
        addr = reverse_geocode(loc["lat"], loc["lon"]) if (loc and do_reverse) else None

        record = {
            "image": up.name,
            "location": {**loc, "address": addr} if loc else None,
            "persons": persons
        }
        results.append(record)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(up.name)
            st.image(image, caption="Original", use_column_width=True)
        with col2:
            boxed = draw_boxes(img_bgr, persons)
            st.image(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB), caption="Detections", use_column_width=True)
        st.json(record)
        st.divider()

    st.download_button("⬇️ Download results JSON",
                       data=json.dumps(results, ensure_ascii=False, indent=2),
                       file_name="results.json",
                       mime="application/json")
else:
    st.info("Bilder in der Sidebar hochladen, um zu starten.")
