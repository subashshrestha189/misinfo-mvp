import numpy as np
import cv2


def decode_image_from_bytes(data: bytes) -> np.ndarray:
    """
    Safely decode image bytes into OpenCV BGR image.
    Supports jpg/png.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid JPG/PNG.")
    return img


def read_image_bytes(data: bytes) -> np.ndarray:
    """
    Backward-compatible helper used by API endpoints.
    """
    return decode_image_from_bytes(data)


def encode_png_bytes(bgr_img: np.ndarray) -> bytes:
    """
    Encode a BGR OpenCV image to PNG bytes for HTTP responses.
    """
    if bgr_img is None or getattr(bgr_img, "size", 0) == 0:
        raise ValueError("Cannot encode an empty image.")

    ok, enc = cv2.imencode(".png", bgr_img)
    if not ok:
        raise ValueError("Failed to encode image as PNG.")
    return enc.tobytes()


def resize_max(bgr_img: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = bgr_img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr_img
    scale = max_side / m
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
