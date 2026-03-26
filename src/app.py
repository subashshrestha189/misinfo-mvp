# src/app.py
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from src.ensemble import combine_scores, compute_heuristic_score
from src.twibot_features import build_features

try:
    from src.cv.face_detect import FaceDetector
    from src.cv.io_utils import decode_image_from_bytes, encode_png_bytes, read_image_bytes, resize_max
    from src.cv.privacy_blur import blur_faces
    from src.cv.profile_risk import compute_profile_image_risk

    CV_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    FaceDetector = None
    decode_image_from_bytes = None
    resize_max = None
    read_image_bytes = None
    encode_png_bytes = None
    compute_profile_image_risk = None
    blur_faces = None
    CV_IMPORT_ERROR = exc


# ------------- CONFIG PATHS -----------------
BASE_DIR = Path(__file__).resolve().parent.parent
BOT_MODEL_DIR = BASE_DIR / "models" / "bot_tuned"

# ---- BOT THRESHOLDS (tuned) ----
BOT_THRESHOLD = 0.30
HIGH_RISK = 0.60

# ------------- FASTAPI APP ------------------
app = FastAPI(
    title="Bot & Profile Risk Detection API",
    description="Detects bots and profile image risk using ML models",
    version="1.0",
)

# Dev: allow all origins. Prod: restrict to ["https://*.x.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ------------- LOAD MODELS ------------------
print("Loading models...")

# --- Bot model ---
bot_model = None
bot_schema = None
bot_model_load_error: Optional[Exception] = None

try:
    bot_model = joblib.load(BOT_MODEL_DIR / "twibot_rf_calibrated.joblib")
    bot_schema = joblib.load(BOT_MODEL_DIR / "feature_schema.joblib")
except Exception as exc:
    bot_model_load_error = exc

# --- CV profile risk detector ---
face_detector = None
cv_model_load_error: Optional[Exception] = None

if CV_IMPORT_ERROR is None:
    try:
        face_detector = FaceDetector(min_confidence=0.6, model_selection=0)
    except Exception as exc:
        cv_model_load_error = exc
else:
    cv_model_load_error = CV_IMPORT_ERROR


# ------------- SCHEMAS ------------------
class UserInput(BaseModel):
    followers_count: float
    following_count: float
    tweet_count: float
    listed_count: float
    account_age_days: float
    has_profile_image: int
    has_description: int
    verified: int
    has_location: int
    has_url: int


# ------------- INTERNAL HELPERS ------------------
def _dependency_error_detail(service: str, exc: Optional[Exception]) -> str:
    detail = f"{service} is unavailable. Install required dependencies and model files."
    if exc is not None:
        detail = f"{detail} Root cause: {type(exc).__name__}: {exc}"
    return detail


def _ensure_bot_model_ready() -> None:
    if bot_model is None or bot_schema is None:
        raise HTTPException(
            status_code=503,
            detail=_dependency_error_detail("Bot model", bot_model_load_error),
        )


def _ensure_cv_ready() -> None:
    if (
        face_detector is None
        or decode_image_from_bytes is None
        or resize_max is None
        or read_image_bytes is None
        or encode_png_bytes is None
        or compute_profile_image_risk is None
        or blur_faces is None
    ):
        raise HTTPException(
            status_code=503,
            detail=_dependency_error_detail("CV module", cv_model_load_error),
        )


def analyze_user_internal(user_dict: dict) -> dict:
    """Run bot model on a single user metadata dict and return a structured dict."""
    _ensure_bot_model_ready()
    df = pd.DataFrame([user_dict])
    x_features = build_features(df)
    x_features = x_features.reindex(columns=bot_schema["feature_list"], fill_value=0)

    prob = float(bot_model.predict_proba(x_features)[:, 1][0])
    is_bot = bool(prob >= BOT_THRESHOLD)

    if prob >= HIGH_RISK:
        risk_level = "High"
    elif prob >= BOT_THRESHOLD:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "is_bot": is_bot,
        "bot_probability": round(prob, 3),
        "risk_level": risk_level,
        "threshold_used": BOT_THRESHOLD,
    }


# ------------- ROUTES ------------------
@app.get("/")
def home():
    return {"message": "Bot & Profile Risk Detection API is running!"}


@app.post("/analyze/user")
def analyze_user(data: UserInput):
    """Bot detection endpoint."""
    if data.followers_count < 0 or data.following_count < 0 or data.tweet_count < 0:
        raise HTTPException(status_code=400, detail="Counts (followers, following, tweets) cannot be negative.")
    if data.account_age_days <= 0:
        raise HTTPException(status_code=400, detail="Account age must be a positive number of days.")

    user_dict = data.model_dump()
    user_result = analyze_user_internal(user_dict)
    heur_score = compute_heuristic_score(user=user_dict)
    ensemble = combine_scores(
        bot_probability=user_result.get("bot_probability", 0.0),
        heuristic_score=heur_score,
    )

    return {
        "user": user_result,
        "heuristics": {"heuristic_score": heur_score},
        "ensemble": ensemble,
    }


@app.post("/analyze/profile-image")
async def analyze_profile_image(file: UploadFile = File(...)):
    """
    Profile Image Risk Scoring.
    Upload an image and return a risk score + explainable signals.
    """
    _ensure_cv_ready()
    try:
        data = await file.read()
        bgr = decode_image_from_bytes(data)
        bgr = resize_max(bgr, max_side=512)

        result = compute_profile_image_risk(bgr, face_detector)
        result["filename"] = file.filename
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Profile image analysis failed: {exc}")


@app.post("/privacy/blur-on-demand")
async def blur_on_demand(file: UploadFile = File(...), blur_strength: int = 35):
    """
    Upload profile image -> compute risk -> blur faces only if risk is medium/high.
    Returns PNG image bytes + risk headers.
    """
    _ensure_cv_ready()
    try:
        content = await file.read()
        img = read_image_bytes(content)
        img = resize_max(img, max_side=512)

        risk_result = compute_profile_image_risk(img, face_detector)
        risk_level = risk_result["risk_level"]
        risk_score = risk_result["profile_image_risk_score"]
        boxes = risk_result["boxes"]

        privacy_applied = risk_level in ["medium", "high"] and len(boxes) > 0
        out_img = blur_faces(img, boxes, blur_strength=blur_strength) if privacy_applied else img

        png_bytes = encode_png_bytes(out_img)
        headers = {
            "X-Risk-Score": str(risk_score),
            "X-Risk-Level": risk_level,
            "X-Privacy-Applied": str(privacy_applied),
        }
        return Response(content=png_bytes, media_type="image/png", headers=headers)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Blur-on-demand failed: {exc}")
