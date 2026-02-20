# src/app.py
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from src.ensemble import combine_scores, compute_heuristic_score
from src.twibot_features import build_features

try:
    import torch
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

    FAKENEWS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    torch = None
    DistilBertTokenizerFast = None
    DistilBertForSequenceClassification = None
    FAKENEWS_IMPORT_ERROR = exc

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
FAKENEWS_MODEL_DIR = BASE_DIR / "models" / "fake_news_bert_fast"
ENCODER_PATH = BASE_DIR / "models" / "fake_news_baseline" / "label_encoder.joblib"

# ---- BOT THRESHOLDS (tuned) ----
BOT_THRESHOLD = 0.30
HIGH_RISK = 0.60
MEDIUM_RISK = 0.30

# ------------- FASTAPI APP ------------------
app = FastAPI(
    title="Misinformation Detection API",
    description="Detects bots and fake news using ML models",
    version="1.0",
)

# ------------- LOAD MODELS ------------------
print("Loading models...")

# --- Fake-news model ---
tokenizer = None
fake_model = None
label_encoder = None
fake_model_load_error: Optional[Exception] = None

if FAKENEWS_IMPORT_ERROR is None:
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(FAKENEWS_MODEL_DIR)
        fake_model = DistilBertForSequenceClassification.from_pretrained(FAKENEWS_MODEL_DIR)
        label_encoder = joblib.load(ENCODER_PATH)
        fake_model.eval()
    except Exception as exc:
        fake_model_load_error = exc
else:
    fake_model_load_error = FAKENEWS_IMPORT_ERROR

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
class ArticleInput(BaseModel):
    text: str


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


class FullInput(BaseModel):
    text: str
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


def _ensure_fake_news_ready() -> None:
    if tokenizer is None or fake_model is None or label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail=_dependency_error_detail("Fake-news model", fake_model_load_error),
        )


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


def analyze_article_internal(text: str) -> dict:
    """Run fake-news model on a single text and return a structured dict."""
    _ensure_fake_news_ready()
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    with torch.no_grad():
        outputs = fake_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]
    pred_id = probs.argmax()
    label = label_encoder.inverse_transform([pred_id])[0]
    confidence = float(probs[pred_id])
    probabilities = {label_encoder.classes_[i]: float(p) for i, p in enumerate(probs)}

    return {
        "predicted_label": label,
        "confidence": confidence,
        "probabilities": probabilities,
    }


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
    elif prob >= MEDIUM_RISK:
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
    return {"message": "Misinformation Detection API is running!"}


@app.post("/analyze/article")
def analyze_article(data: ArticleInput):
    """Fake-news analysis endpoint."""
    return analyze_article_internal(data.text)


@app.post("/analyze/user")
def analyze_user(data: UserInput):
    """Bot detection endpoint."""
    if data.followers_count < 0 or data.following_count < 0 or data.tweet_count < 0:
        raise HTTPException(status_code=400, detail="Counts (followers, following, tweets) cannot be negative.")
    if data.account_age_days <= 0:
        raise HTTPException(status_code=400, detail="Account age must be a positive number of days.")

    return analyze_user_internal(data.dict())


@app.post("/analyze/full")
def analyze_full(data: FullInput):
    """
    Combined endpoint:
    - analyzes article text with BERT
    - analyzes user with RF bot model
    - computes overall trust_score using ensemble weights + heuristics
    """
    article_result = analyze_article_internal(data.text)

    user_payload = {
        "followers_count": data.followers_count,
        "following_count": data.following_count,
        "tweet_count": data.tweet_count,
        "listed_count": data.listed_count,
        "account_age_days": data.account_age_days,
        "has_profile_image": data.has_profile_image,
        "has_description": data.has_description,
        "verified": data.verified,
        "has_location": data.has_location,
        "has_url": data.has_url,
    }

    if user_payload["followers_count"] < 0 or user_payload["following_count"] < 0 or user_payload["tweet_count"] < 0:
        raise HTTPException(status_code=400, detail="Counts (followers, following, tweets) cannot be negative.")
    if user_payload["account_age_days"] <= 0:
        raise HTTPException(status_code=400, detail="Account age must be a positive number of days.")

    user_result = analyze_user_internal(user_payload)
    heur_score = compute_heuristic_score(
        user=user_payload,
        article_label=article_result.get("predicted_label"),
        article_confidence=article_result.get("confidence"),
    )
    ensemble = combine_scores(
        content_confidence=article_result.get("confidence", 0.0),
        bot_probability=user_result.get("bot_probability", 0.0),
        heuristic_score=heur_score,
    )

    return {
        "article": article_result,
        "user": user_result,
        "heuristics": {"heuristic_score": heur_score},
        "ensemble": ensemble,
    }


@app.post("/analyze/profile-image")
async def analyze_profile_image(file: UploadFile = File(...)):
    """
    Feature 2: Profile Image Risk Scoring
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

        risk_result = compute_profile_image_risk(img, face_detector)
        risk_level = risk_result["risk_level"]
        risk_score = risk_result["profile_image_risk_score"]

        boxes = face_detector.detect(img)
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
