# src/bot_train_twibot.py
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.twibot_features import build_features, feature_list

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "twibot" / "users.csv"
OUT_DIR = BASE_DIR / "models" / "bot_tuned"


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)

    # Separate features / target
    y = df["label"].astype(int)
    X_raw = df.drop(columns=["label", "id"], errors="ignore")

    # Build engineered features
    print("Building features ...")
    X = build_features(X_raw)
    cols = feature_list()
    # make sure columns are in consistent order
    X = X.reindex(columns=cols, fill_value=0)

    print("Final feature shape:", X.shape)

    # Basic train/holdout split (80/20) – stratified
    from sklearn.model_selection import train_test_split

    # ---- Balanced class subsample to fix the 11.8:1 human/bot imbalance ----
    # The original dataset has far more humans than bots. A plain random subsample
    # keeps that ratio, causing the model to predict "human" for everything
    # (0% bot recall). Instead we cap each class separately at a 2:1 ratio.
    bot_idx   = y[y == 1].index
    human_idx = y[y == 0].index

    n_bots   = min(30_000, len(bot_idx))
    n_humans = min(60_000, len(human_idx))   # at most 2:1

    rng = np.random.default_rng(42)
    sampled_bots   = rng.choice(bot_idx,   size=n_bots,   replace=False)
    sampled_humans = rng.choice(human_idx, size=n_humans, replace=False)
    balanced_idx   = np.concatenate([sampled_bots, sampled_humans])
    rng.shuffle(balanced_idx)

    X_small = X.loc[balanced_idx].reset_index(drop=True)
    y_small = y.loc[balanced_idx].reset_index(drop=True)
    print(f"Balanced sample — bots: {n_bots}, humans: {n_humans}, total: {len(X_small)}")

    # Train/holdout split on the *smaller* set
    X_train, X_test, y_train, y_test = train_test_split(
        X_small,
        y_small,
        test_size=0.2,
        random_state=42,
        stratify=y_small,
    )
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)


    # Define base RF and hyperparameter search space
    rf = RandomForestClassifier(
        n_estimators=200,                 # a bit smaller
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    param_dist = {
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 0.5],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print("Running RandomizedSearchCV on subset ...")
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=8,            # fewer candidates
        scoring="f1",
        n_jobs=1,            # IMPORTANT: no multi-process copies
        cv=cv,
        random_state=42,
        verbose=1,
    )


    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    print("Best CV score (F1):", search.best_score_)

    best_rf = search.best_estimator_

    # Calibrate probabilities on the holdout set
    print("Calibrating probabilities ...")
    calibrated = CalibratedClassifierCV(best_rf, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)

    # Evaluate on holdout
    print("Evaluating on holdout set ...")
    y_proba = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))

    try:
        auc = roc_auc_score(y_test, y_proba)
        print("ROC AUC:", auc)
    except Exception as e:
        print("Could not compute AUC:", e)

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Prepare output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = OUT_DIR / "twibot_rf_calibrated.joblib"
    dump(calibrated, model_path)
    print(f"Saved calibrated model to {model_path}")

    # Save feature schema
    schema = {
        "feature_list": cols,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "best_params": search.best_params_,
    }
    schema_path = OUT_DIR / "feature_schema.joblib"
    dump(schema, schema_path)
    print(f"Saved feature schema to {schema_path}")

    # Save a human-readable summary.json
    summary = {
        "n_samples": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "class_balance": {
            "0_human": int((y == 0).sum()),
            "1_bot": int((y == 1).sum()),
        },
        "best_params": search.best_params_,
        "test_metrics": {
            "report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else None,
        },
    }
    summary_path = OUT_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
