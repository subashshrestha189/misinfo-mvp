import numpy as np
import cv2
from typing import Dict, Any, List
from .face_detect import FaceDetector, FaceBox


def blur_score_laplacian(bgr_img: np.ndarray) -> float:
    """
    Blur detection proxy:
    Higher variance = sharper.
    We'll convert to a normalized "blur risk" later.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(var)


def compression_proxy_score(bgr_img: np.ndarray) -> float:
    """
    Proxy for compression artifacts:
    Compares original image to a recompressed version and measures difference.
    Higher difference can indicate heavy compression / artifacts.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 35]  # intentionally low quality
    ok, enc = cv2.imencode(".jpg", bgr_img, encode_param)
    if not ok:
        return 0.0
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(bgr_img, dec)
    # normalize by pixel intensity
    score = float(np.mean(diff) / 255.0)
    return score


def edge_density_score(bgr_img: np.ndarray) -> float:
    """
    Heuristic: avatars/logos often have sharp edges or very flat regions.
    We measure edge density to capture "logo-like" structure.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    density = float(np.mean(edges > 0))  # 0..1
    return density


def color_variance_score(bgr_img: np.ndarray) -> float:
    """
    Low color variance can indicate blank/default/simple avatar.
    """
    # convert to HSV for stable measure
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # variance across channels averaged
    var = float(np.mean([np.var(hsv[..., c]) for c in range(3)]))
    # normalize roughly (empirical)
    return float(min(1.0, var / 5000.0))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute_profile_image_risk(
    bgr_img: np.ndarray,
    face_detector: FaceDetector,
) -> Dict[str, Any]:
    """
    Returns:
      - profile_image_risk_score: 0..1
      - risk_level: low/medium/high
      - signals: explainable evidence
    """
    h, w = bgr_img.shape[:2]

    # --- Face detection ---
    boxes: List[FaceBox] = face_detector.detect(bgr_img)
    num_faces = len(boxes)
    face_detected = num_faces > 0

    # face area ratio (largest face)
    face_area_ratio = 0.0
    if boxes:
        b = boxes[0]
        face_area = (b.x2 - b.x1) * (b.y2 - b.y1)
        face_area_ratio = float(face_area / (w * h))

    # --- Quality signals ---
    lap_var = blur_score_laplacian(bgr_img)  # high => sharp
    # Convert to blur risk 0..1 (heuristic thresholds)
    # typical: <80 very blurry, >300 sharp (depends on resolution)
    blur_risk = clamp01((200.0 - lap_var) / 200.0)

    comp_proxy = compression_proxy_score(bgr_img)  # 0..~0.2
    compression_risk = clamp01(comp_proxy / 0.12)

    # --- Avatar/logo heuristics ---
    edges = edge_density_score(bgr_img)  # 0..1
    # both extremely low edges (blank) OR extremely high edges (logo) can be suspicious
    edge_risk = clamp01(abs(edges - 0.10) / 0.10)  # centered around 0.10 as "normal-ish"

    color_var = color_variance_score(bgr_img)  # 0..1
    # low color variance => higher risk
    color_risk = clamp01(1.0 - color_var)

    # Resolution risk
    resolution_risk = 0.0
    min_side = min(h, w)
    if min_side < 96:
        resolution_risk = 1.0
    elif min_side < 128:
        resolution_risk = 0.6
    elif min_side < 160:
        resolution_risk = 0.3

    # Multi-face risk (group photo not necessarily bad, but higher uncertainty)
    multi_face_risk = 0.0
    if num_faces >= 3:
        multi_face_risk = 0.6
    elif num_faces == 2:
        multi_face_risk = 0.3

    # No-face risk: bots often use non-face imagery
    no_face_risk = 1.0 if not face_detected else 0.0

    # --- Weighted score (explainable) ---
    # Keep it transparent and adjustable.
    score = (
        0.35 * no_face_risk +
        0.15 * resolution_risk +
        0.15 * blur_risk +
        0.10 * compression_risk +
        0.10 * edge_risk +
        0.10 * color_risk +
        0.05 * multi_face_risk
    )
    score = clamp01(score)

    if score < 0.33:
        level = "low"
    elif score < 0.66:
        level = "medium"
    else:
        level = "high"

    signals = {
        "face_detected": face_detected,
        "num_faces": num_faces,
        "face_area_ratio": round(face_area_ratio, 4),
        "resolution": {"width": w, "height": h, "min_side": min_side},
        "laplacian_variance": round(lap_var, 2),
        "blur_risk": round(blur_risk, 3),
        "compression_proxy": round(comp_proxy, 3),
        "compression_risk": round(compression_risk, 3),
        "edge_density": round(edges, 4),
        "edge_risk": round(edge_risk, 3),
        "color_variance_norm": round(color_var, 3),
        "color_risk": round(color_risk, 3),
        "resolution_risk": round(resolution_risk, 3),
        "multi_face_risk": round(multi_face_risk, 3),
        "no_face_risk": round(no_face_risk, 3),
    }

    notes = []
    if not face_detected:
        notes.append("No face detected (common in logos/default avatars).")
    if resolution_risk >= 0.6:
        notes.append("Low resolution image; higher uncertainty.")
    if blur_risk >= 0.6:
        notes.append("Image appears blurry; face analysis quality reduced.")
    if compression_risk >= 0.6:
        notes.append("Image shows signs of heavy compression artifacts.")
    if num_faces >= 2:
        notes.append("Multiple faces detected; profile ownership uncertain.")

    return {
        "profile_image_risk_score": round(score, 3),
        "risk_level": level,
        "signals": signals,
        "notes": notes
    }
