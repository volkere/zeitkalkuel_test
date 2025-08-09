
from __future__ import annotations
import pickle, os, glob
from typing import Dict, List
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from .utils import cosine_similarity

class FaceEngine:
    def __init__(self, det_size=(640,640)):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=-1, det_size=det_size)

    def analyze(self, img_bgr):
        faces = self.app.get(img_bgr)
        results = []
        for f in faces:
            box = f.bbox.astype(int).tolist()
            prob = float(getattr(f, "det_score", 1.0))
            gender = getattr(f, "gender", None)
            gender_str = "male" if gender == 0 else ("female" if gender == 1 else None)
            age = int(getattr(f, "age", -1)) if getattr(f, "age", None) is not None else None
            emb = f.embedding.astype(np.float32)
            results.append({
                "bbox": box,
                "prob": prob,
                "embedding": emb,
                "age": age if age and age >= 0 else None,
                "gender": gender_str
            })
        return results

class GalleryDB:
    def __init__(self):
        self.people: Dict[str, List[np.ndarray]] = {}

    def add(self, name: str, embedding: np.ndarray):
        self.people.setdefault(name, []).append(embedding.astype(np.float32))

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.people, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> 'GalleryDB':
        db = GalleryDB()
        with open(path, "rb") as f:
            db.people = pickle.load(f)
        return db

    def match(self, embedding: np.ndarray, threshold: float = 0.55):
        best_name, best_sim = None, -1.0
        for name, embs in self.people.items():
            sims = [cosine_similarity(embedding, e) for e in embs]
            if sims:
                sim = float(np.mean(sims))
                if sim > best_sim:
                    best_sim, best_name = sim, name
        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim

def build_gallery_from_folder(gallery_dir: str, det_size=(640,640)) -> 'GalleryDB':
    engine = FaceEngine(det_size=det_size)
    db = GalleryDB()
    exts = (".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff")
    for person in sorted(os.listdir(gallery_dir)):
        person_dir = os.path.join(gallery_dir, person)
        if not os.path.isdir(person_dir):
            continue
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(person_dir, f"*{ext}")))
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            faces = engine.analyze(img)
            if not faces:
                continue
            faces.sort(key=lambda f: (f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1]), reverse=True)
            db.add(person, faces[0]["embedding"])
    return db
