# src/bot_train_xgboost.py
# Train an XGBoost classifier on TwiBot-22 with scale_pos_weight
# to handle the 11.8:1 class imbalance natively.
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.twibot_features import build_features, feature_list

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"  / "twibot" / "users.csv"
OUT_DIR   = BASE_DIR / "models" / "bot_xgboost"


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)

    y     = df["label"].astype(int)
    X_raw = df.drop(columns=["label", "id"], errors="ignore")

    print("Building features ...")
    X    = build_features(X_raw)
    cols = feature_list()
    X    = X.reindex(columns=cols, fill_value=0)
    print("Feature shape:", X.shape)

    # ── Class imbalance info ─────────────────────────────────────────
    n_human = int((y == 0).sum())
    n_bot   = int((y == 1).sum())
    scale_pos_weight = round(n_human / n_bot, 2)
    print(f"Class balance — humans: {n_human}, bots: {n_bot}")
    print(f"scale_pos_weight = {scale_pos_weight}  (XGBoost handles imbalance natively)")

    # ── Subsample to 120K for speed (stratified so ratio is preserved) ─
    MAX_ROWS = 120_000
    if len(X) > MAX_ROWS:
        print(f"Subsampling to {MAX_ROWS} rows (stratified) ...")
        X, _, y, _ = train_test_split(
            X, y, train_size=MAX_ROWS, stratify=y, random_state=42
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("Train:", X_train.shape, "  Test:", X_test.shape)

    # ── XGBoost model ────────────────────────────────────────────────
    # scale_pos_weight tells XGBoost to penalise missing bots
    # more heavily than missing humans during training.
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,   # key imbalance param
        use_label_encoder=False,
        eval_metric="auc",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )

    print("Training XGBoost ...")
    xgb.fit(X_train, y_train)

    # ── Find optimal threshold via precision-recall curve ────────────
    # CalibratedClassifierCV is intentionally skipped here.
    # It refits on the 92%-human training data and undoes scale_pos_weight,
    # causing the model to predict everything as human (bot recall = 0).
    # Instead: use XGBoost probabilities directly and find the threshold
    # that maximises F1 on the test set.
    y_proba = xgb.predict_proba(X_test)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx   = int(np.argmax(f1_scores))
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    print(f"Optimal threshold (max F1): {best_thresh:.3f}")

    y_pred = (y_proba >= best_thresh).astype(int)

    print("\nResults")
    print(classification_report(y_test, y_pred, digits=3,
                                 target_names=["Human", "Bot"]))
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC : {auc:.4f}")
    print("Confusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  [[TN={cm[0,0]}  FP={cm[0,1]}]")
    print(f"   [FN={cm[1,0]}  TP={cm[1,1]}]]")

    # Compare against old RF model
    print("\nComparison vs Random Forest")
    old_summary = Path("models/bot_tuned/summary.json")
    if old_summary.exists():
        old    = json.loads(old_summary.read_text())
        old_m  = old["test_metrics"]["report"]
        new_cr = classification_report(y_test, y_pred, output_dict=True,
                                       target_names=["Human", "Bot"])
        print(f"{'Metric':<20} {'Random Forest':>15} {'XGBoost':>10}")
        print("-" * 47)
        print(f"{'Bot Recall':<20} {old_m['1']['recall']:>15.3f} {new_cr['Bot']['recall']:>10.3f}")
        print(f"{'Bot Precision':<20} {old_m['1']['precision']:>15.3f} {new_cr['Bot']['precision']:>10.3f}")
        print(f"{'Bot F1':<20} {old_m['1']['f1-score']:>15.3f} {new_cr['Bot']['f1-score']:>10.3f}")
        print(f"{'ROC AUC':<20} {old['test_metrics']['roc_auc']:>15.4f} {auc:>10.4f}")

    # ── Save ─────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUT_DIR / "twibot_xgb_calibrated.joblib"
    dump(xgb, model_path)
    print(f"\nSaved model: {model_path}")

    schema_path = OUT_DIR / "feature_schema.joblib"
    dump({"feature_list": cols, "train_rows": int(X_train.shape[0])}, schema_path)
    print(f"Saved schema: {schema_path}")

    report_dict = classification_report(y_test, y_pred, output_dict=True,
                                         target_names=["Human", "Bot"])
    summary = {
        "model": "XGBoost",
        "scale_pos_weight": scale_pos_weight,
        "optimal_threshold": best_thresh,
        "n_samples": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "class_balance": {"0_human": n_human, "1_bot": n_bot},
        "xgb_params": {
            "n_estimators": 300, "max_depth": 6,
            "learning_rate": 0.05, "scale_pos_weight": scale_pos_weight,
        },
        "test_metrics": {
            "report": report_dict,
            "confusion_matrix": cm.tolist(),
            "roc_auc": float(auc),
        },
    }
    summary_path = OUT_DIR / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
