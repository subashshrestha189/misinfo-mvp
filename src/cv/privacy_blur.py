# src/cv/privacy_blur.py
import cv2
import numpy as np
from typing import List
from .face_detect import FaceBox

def blur_faces(bgr_img: np.ndarray, boxes: List[FaceBox], blur_strength: int = 35) -> np.ndarray:
    if bgr_img is None or bgr_img.size == 0 or not boxes:
        return bgr_img

    if blur_strength < 3:
        blur_strength = 3
    if blur_strength % 2 == 0:
        blur_strength += 1

    out = bgr_img.copy()
    h, w = out.shape[:2]

    for b in boxes:
        x1 = max(0, min(w - 1, b.x1))
        y1 = max(0, min(h - 1, b.y1))
        x2 = max(0, min(w, b.x2))
        y2 = max(0, min(h, b.y2))

        if x2 <= x1 or y2 <= y1:
            continue

        roi = out[y1:y2, x1:x2]
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)

    return out
