
from __future__ import annotations
import numpy as np
from typing import Optional

def dms_to_dd(dms, ref) -> Optional[float]:
    if not dms or not ref:
        return None
    try:
        deg = dms[0][0] / dms[0][1]
        minutes = dms[1][0] / dms[1][1]
        seconds = dms[2][0] / dms[2][1]
        dd = deg + minutes/60.0 + seconds/3600.0
        if ref in ['S','W']:
            dd = -dd
        return dd
    except Exception:
        return None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))
