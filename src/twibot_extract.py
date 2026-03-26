# src/twibot_extract.py
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

# Path to the folder where you saved user.json / label.csv / split.csv
TWIBOT_ROOT = Path.home() / "Documents" / "TwiBot-22"

USER_JSON = TWIBOT_ROOT / "user.json"
LABEL_CSV = TWIBOT_ROOT / "label.csv"
SPLIT_CSV = TWIBOT_ROOT / "split.csv"   # optional but we'll use it

OUT_DIR = Path("data") / "twibot"
OUT_RAW = OUT_DIR / "users_raw.csv"
OUT_FINAL = OUT_DIR / "users.csv"


def parse_created_at(created_at: str) -> int:
    """Convert Twitter created_at string to account_age_days."""
    if not created_at:
        return 0
    # Most Twitter data uses this format; if it fails we return 0.
    try:
        dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
    except Exception:
        try:
            dt = datetime.fromisoformat(created_at)
        except Exception:
            return 0
    now = datetime.now(timezone.utc)
    return max((now - dt).days, 0)


def main():
    if not USER_JSON.exists():
        raise SystemExit(f"USER_JSON not found at {USER_JSON}")
    if not LABEL_CSV.exists():
        raise SystemExit(f"LABEL_CSV not found at {LABEL_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading users from {USER_JSON} ...")
    users = []

    # Load the whole JSON array once
    with USER_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure we have a list
    if isinstance(data, dict):
        data = [data]

    for i, obj in enumerate(data, start=1):
        # Some TwiBot formats wrap the user dict in a list, e.g. [id, user_dict]
        if isinstance(obj, list):
            if len(obj) >= 2 and isinstance(obj[1], dict):
                obj = obj[1]
            elif obj and isinstance(obj[0], dict):
                obj = obj[0]
            else:
                print(f"Skipping unexpected list entry at index {i}: {type(obj)}")
                continue

        if not isinstance(obj, dict):
            print(f"Skipping non-dict entry at index {i}: {type(obj)}")
            continue

        # ----- same feature extraction as before -----
        user_id = obj.get("id")
        profile = obj.get("profile", obj)

        followers_count = profile.get("followers_count", 0)
        following_count = profile.get("friends_count", 0)   # may be friends_count
        tweet_count     = profile.get("statuses_count", profile.get("tweet_count", 0))
        listed_count    = profile.get("listed_count", 0)

        description       = (profile.get("description") or "").strip()
        profile_image_url = (profile.get("profile_image_url") or "").strip()
        verified          = bool(profile.get("verified", False))
        location          = (profile.get("location") or "").strip()
        url               = (profile.get("url") or "").strip()
        created_at        = profile.get("created_at", "")

        account_age_days = parse_created_at(created_at)

        users.append({
            "id": user_id,
            "followers_count": followers_count,
            "following_count": following_count,
            "tweet_count": tweet_count,
            "listed_count": listed_count,
            "account_age_days": account_age_days,
            "has_profile_image": int(bool(profile_image_url)),
            "has_description": int(description != ""),
            "verified": int(verified),
            "has_location": int(location != ""),
            "has_url": int(url != ""),
        })

        if i % 50000 == 0:
            print(f"  processed {i} users...")


    df_users = pd.DataFrame(users)
    print(f"Loaded {len(df_users)} users from JSON.")
    df_users.to_csv(OUT_RAW, index=False)
    print(f"Saved raw user table to {OUT_RAW}")

    print(f"Reading labels from {LABEL_CSV} ...")
    df_labels = pd.read_csv(LABEL_CSV)

    if "id" not in df_labels.columns:
        raise SystemExit(f"'id' column not found in {LABEL_CSV}")
    if "label" not in df_labels.columns:
        raise SystemExit(f"'label' column not found in {LABEL_CSV}")

    # Optional: restrict to TRAIN users only using split.csv
    if SPLIT_CSV.exists():
        print(f"Reading split info from {SPLIT_CSV} ...")
        df_split = pd.read_csv(SPLIT_CSV)
        # Usually split.csv has columns like id, split (train/val/test)
        if {"id", "split"}.issubset(df_split.columns):
            train_ids = df_split[df_split["split"] == "train"]["id"]
            df_labels = df_labels[df_labels["id"].isin(train_ids)]
            print(f"Filtered labels to TRAIN split: {len(df_labels)} rows")
        else:
            print("split.csv format unknown; using all labels.")

    df = df_users.merge(df_labels[["id", "label"]], on="id", how="inner")
    print(f"After join, have {len(df)} labeled users.")

    # Inspect unique labels to choose mapping
    uniques = sorted(df["label"].unique())
    print("Unique labels:", uniques)

    # Common TwiBot-20 labels: 'human', 'bot', 'organization', 'other'
    # For this project we keep only human vs bot and drop others.
    if set(["human", "bot"]).issubset(set(uniques)):
        df = df[df["label"].isin(["human", "bot"])].reset_index(drop=True)
        df["label"] = df["label"].map({"human": 0, "bot": 1})
    else:
        # If labels are already 0/1 or different, adapt here.
        try:
            df["label"] = df["label"].astype(int)
        except Exception:
            raise SystemExit(f"Please adjust label mapping for labels: {uniques}")

    df.to_csv(OUT_FINAL, index=False)
    print(f"✅ Saved final users file to {OUT_FINAL} (rows={len(df)})")


if __name__ == "__main__":
    main()
