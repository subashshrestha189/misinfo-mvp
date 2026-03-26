# src/bot_permutation_importance.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.inspection import permutation_importance
from src.twibot_features import build_features, feature_list

MODEL_DIR = Path("models/bot_tuned")
model = joblib.load(MODEL_DIR / "twibot_rf_calibrated.joblib")

df = pd.read_csv("data/twibot/users.csv")

# Must use a balanced sample — the full 700K CSV is 92% human.
# Running permutation_importance on imbalanced data causes all importances
# to collapse to 0.0 because accuracy doesn't change when features are
# shuffled (the model can always "win" by predicting the majority class).
rng = np.random.default_rng(42)
bot_idx   = df.index[df["label"] == 1].tolist()
human_idx = df.index[df["label"] == 0].tolist()
n = min(5_000, len(bot_idx))

sampled_idx = np.concatenate([
    rng.choice(bot_idx,   size=n, replace=False),
    rng.choice(human_idx, size=n, replace=False),
])
df_bal = df.loc[sampled_idx].reset_index(drop=True)
print(f"Balanced sample — bots: {n}, humans: {n}, total: {len(df_bal)}")

y = df_bal["label"].astype(int).values
X = build_features(df_bal)
cols = feature_list()
X = X.reindex(columns=cols, fill_value=0)

print("Running permutation importance (scoring=roc_auc, n_repeats=10) ...")
r = permutation_importance(
    model, X, y,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
    scoring="roc_auc",   # AUC is meaningful on balanced data; accuracy is not
)

imp = pd.DataFrame({
    "feature":          cols,
    "importance_mean":  r.importances_mean,
    "importance_std":   r.importances_std,
}).sort_values("importance_mean", ascending=False)

out_csv = MODEL_DIR / "permutation_importance.csv"
imp.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}\n")
print(imp.to_string(index=False))
