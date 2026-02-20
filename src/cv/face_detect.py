import numpy as np
import cv2
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class FaceBox:
    # pixel coordinates
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


class FaceDetector:
    """
    MediaPipe face detector wrapper.
    Returns bounding boxes in pixel coords.
    """
    def __init__(self, min_confidence: float = 0.6, model_selection: int = 0):
        self._mp_face = mp.solutions.face_detection
        self._detector = self._mp_face.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence
        )

    def detect(self, bgr_img: np.ndarray) -> List[FaceBox]:
        if bgr_img is None or bgr_img.size == 0:
            return []

        h, w = bgr_img.shape[:2]
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        result = self._detector.process(rgb)

        boxes: List[FaceBox] = []
        if not result.detections:
            return boxes

        for det in result.detections:
            score = float(det.score[0]) if det.score else 0.0
            bbox = det.location_data.relative_bounding_box

            x1 = int(max(0, bbox.xmin * w))
            y1 = int(max(0, bbox.ymin * h))
            x2 = int(min(w, (bbox.xmin + bbox.width) * w))
            y2 = int(min(h, (bbox.ymin + bbox.height) * h))

            # sanity clamp
            if x2 > x1 and y2 > y1:
                boxes.append(FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, score=score))

        # Sort: strongest first
        boxes.sort(key=lambda b: b.score, reverse=True)
        return boxes
