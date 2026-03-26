# src/bot_train_cv_calibrated.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from twibot_features import build_features, feature_list

MODEL_DIR = Path("models/bot_tuned")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load data ----
df = pd.read_csv("data/twibot/users.csv")
assert "label" in df.columns, "users.csv must include binary 'label' column (1=bot, 0=human)."

y = df["label"].astype(int).values
X = build_features(df)
feat_names = X.columns.tolist()

# --- Diagnose label counts ---
vc = pd.Series(y).value_counts().sort_index()
print("Label counts:", vc.to_dict())
min_count = int(vc.min())
n_classes = int(vc.size)

# -------- Train/Holdout split (handle tiny minority class) --------
use_stratify = (n_classes > 1) and (min_count >= 2)
if not use_stratify:
    print("[WARN] Not enough samples in at least one class for a stratified split. "
          "Proceeding without stratify; holdout may be single-class.")

X_tr, X_ho, y_tr, y_ho = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if use_stratify else None
)

n_tr = len(y_tr)
uniq_tr, counts_tr = np.unique(y_tr, return_counts=True)
min_count_tr = counts_tr.min()
print(f"[INFO] Train size={n_tr}, class counts (train)={dict(zip(uniq_tr, counts_tr))}")

# ---- Preprocessor ----
num_cols = [c for c in feat_names if c not in []]
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
pipe = Pipeline([("pre", pre), ("rf", rf)])

# -------- Choose CV safely (or skip search if too tiny) --------
from sklearn.model_selection import KFold, StratifiedKFold

def make_cv():
    # want stratified only if each class can appear in every fold
    if (len(uniq_tr) > 1) and (min_count_tr >= 2):
        # folds cannot exceed the smallest class count nor the train size
        n_splits = max(2, min(5, min(min_count_tr, n_tr)))
        print(f"[INFO] Using StratifiedKFold with n_splits={n_splits}")
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        # non-stratified fallback; n_splits must be <= train size
        n_splits = max(2, min(5, n_tr))
        print(f"[WARN] Using non-stratified KFold with n_splits={n_splits} "
              f"(min class count in train={min_count_tr}).")
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)

too_tiny_for_search = (n_tr < 3) or (len(uniq_tr) < 2)
if too_tiny_for_search:
    print("[WARN] Train set is too small or single-class; skipping hyperparameter search.")
    best = pipe.fit(X_tr, y_tr)
else:
    param_distributions = {
        "rf__n_estimators": randint(200, 600),
        "rf__max_depth": randint(4, 20),
        "rf__min_samples_split": randint(2, 20),
        "rf__min_samples_leaf": randint(1, 10),
        "rf__max_features": uniform(0.3, 0.7),
    }
    cv = make_cv()
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=min(25, max(1, n_tr)),  # cap iterations on tiny data
        scoring="f1",                  # OK; may be undefined on single-class folds, but we handle tiny case above
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    print("\nBest params:", getattr(search, "best_params_", {}))

# ---- Probability calibration (skip if impossibly small) ----
if (len(uniq_tr) > 1) and (min_count_tr >= 2) and (n_tr >= 10):
    cal_folds = max(2, min(3, min_count_tr, n_tr))
    print(f"[INFO] Calibrating with cv={cal_folds}")
    calibrated = CalibratedClassifierCV(base_estimator=best, cv=cal_folds,
                                        method="isotonic" if n_tr >= 100 else "sigmoid")
    calibrated.fit(X_tr, y_tr)
else:
    print("[WARN] Skipping calibration (train too small or single-class).")
    calibrated = best



# ---- Evaluate on holdout ----
probs = calibrated.predict_proba(X_ho)[:,1]
preds = (probs >= 0.5).astype(int)

print("\nHOLDOUT REPORT")
print(classification_report(y_ho, preds, digits=3))

if len(np.unique(y_ho))>1 and len(np.unique(preds))>1:
    try:
        auc = roc_auc_score(y_ho, probs)
        ap = average_precision_score(y_ho, probs)
    except Exception:
        auc, ap = float("nan"), float("nan")
else:
    auc, ap = float("nan"), float("nan")

brier = brier_score_loss(y_ho, probs)

print(f"AUC:   {auc:.3f}")
print(f"AP:    {ap:.3f}")
print(f"Brier: {brier:.4f}")

# ---- Save artifacts ----
joblib.dump(calibrated, MODEL_DIR / "twibot_rf_calibrated.joblib")
joblib.dump({"feature_list": feat_names}, MODEL_DIR / "feature_schema.joblib")

summary = {
    "best_params": search.best_params_ if not too_tiny_for_search else {},
    "n_features": len(feat_names),
    "holdout_metrics": {"auc": auc, "average_precision": ap, "brier": brier},
}
Path(MODEL_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nSaved model → {MODEL_DIR/'twibot_rf_calibrated.joblib'}")
print(f"Saved schema → {MODEL_DIR/'feature_schema.joblib'}")
print(f"Saved summary → {MODEL_DIR/'summary.json'}")
