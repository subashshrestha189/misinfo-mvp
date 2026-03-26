# src/bot_infer.py
import joblib
import pandas as pd
from pathlib import Path
from src.twibot_features import build_features

MODEL_DIR = Path("models/bot_tuned")
model = joblib.load(MODEL_DIR / "twibot_rf_calibrated.joblib")
schema = joblib.load(MODEL_DIR / "feature_schema.joblib")

def predict_user(user_row: dict):
    df = pd.DataFrame([user_row])
    X = build_features(df)
    # ensure column order matches training
    X = X.reindex(columns=schema["feature_list"], fill_value=0)
    prob = float(model.predict_proba(X)[:,1][0])
    lbl = int(prob >= 0.30)
    return {"is_bot": lbl, "bot_probability": prob}

if __name__ == "__main__":
    # example
    sample = {
        "followers_count": 120, "following_count": 200, "tweet_count": 1500, "listed_count": 10,
        "account_age_days": 800, "has_profile_image": 1, "has_description": 1,
        "verified": 0, "has_location": 1, "has_url": 0
    }
    print(predict_user(sample))
