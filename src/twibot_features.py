# src/twibot_features.py
import numpy as np
import pandas as pd

NUM_BASE = [
    "followers_count","following_count","tweet_count","listed_count",
    "favourites_count","account_age_days"
]
BIN_BASE = [
    "has_profile_image","default_profile","has_description",
    "verified","has_location","has_url"
]

def _safe_ratio(a, b):
    b = np.where(b==0, 1, b)
    return a / b

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure base columns exist
    for col in NUM_BASE + BIN_BASE:
        if col not in df.columns:
            df[col] = 0

    age = np.where(df["account_age_days"] <= 0, 1, df["account_age_days"])

    # Basic rates
    df["followers_following_ratio"] = _safe_ratio(df["followers_count"], df["following_count"])
    df["tweets_per_day"]      = _safe_ratio(df["tweet_count"],      age)
    df["favourites_per_day"]  = _safe_ratio(df["favourites_count"], age)

    # Log transforms (stabilize heavy tails)
    for c in ["followers_count","following_count","tweet_count","listed_count",
              "favourites_count","account_age_days"]:
        df[f"log1p_{c}"] = np.log1p(df[c].clip(lower=0))

    # Interaction / quality signals
    df["ff_log_ratio"] = _safe_ratio(np.log1p(df["followers_count"]), np.log1p(df["following_count"]))
    df["listed_per_1k_followers"]     = 1000.0 * _safe_ratio(df["listed_count"],      (df["followers_count"]+1))
    df["tweets_per_1k_followers"]     = 1000.0 * _safe_ratio(df["tweet_count"],       (df["followers_count"]+1))
    df["favourites_per_1k_followers"] = 1000.0 * _safe_ratio(df["favourites_count"],  (df["followers_count"]+1))

    # Clip extreme ratios to reduce outlier impact
    for c in ["followers_following_ratio","tweets_per_day","favourites_per_day",
              "ff_log_ratio","listed_per_1k_followers","tweets_per_1k_followers",
              "favourites_per_1k_followers"]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1e6)

    # Final feature list (numeric + binaries)
    feat_cols = (
        NUM_BASE
        + ["followers_following_ratio","tweets_per_day","favourites_per_day",
           "ff_log_ratio","listed_per_1k_followers","tweets_per_1k_followers",
           "favourites_per_1k_followers"]
        + [f"log1p_{c}" for c in ["followers_count","following_count","tweet_count",
                                   "listed_count","favourites_count","account_age_days"]]
        + BIN_BASE
    )

    return df[feat_cols]

def feature_list():
    # central source of truth for downstream scripts
    return build_features(pd.DataFrame(columns=NUM_BASE+BIN_BASE)).columns.tolist()
