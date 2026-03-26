import pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from src.utils_io import ensure_dir

DATA = Path("data/twibot/users.csv")
MODEL_DIR = ensure_dir("models/bot_baseline")

NUM_COLS = [
    "followers_count","following_count","tweet_count","listed_count",
    "account_age_days","followers_following_ratio","tweets_per_day"
]
BIN_COLS = ["has_profile_image","has_description","verified","has_location","has_url"]
LABEL = "label"

df = pd.read_csv(DATA)

missing = set(NUM_COLS + BIN_COLS + [LABEL]) - set(df.columns)
if missing:
    raise SystemExit(f"CSV missing columns: {missing}")

# drop rows without label
df = df.dropna(subset=[LABEL]).copy()

X = df[NUM_COLS + BIN_COLS]
y = df[LABEL].astype(int)

# note: with 3 rows, stratify might fail, so we guard it
if len(df) > 10 and y.nunique() > 1:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

pre = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("bin", "passthrough", BIN_COLS),
])

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("pre", pre),
    ("rf", rf)
])

pipe.fit(X_tr, y_tr)

# with tiny data this won’t be meaningful, but it proves pipeline works
y_pred = pipe.predict(X_te)
print(classification_report(y_te, y_pred, digits=3))

# save model + feature schema
joblib.dump(pipe, MODEL_DIR / "twibot_rf.joblib")
joblib.dump({"NUM_COLS": NUM_COLS, "BIN_COLS": BIN_COLS}, MODEL_DIR / "feature_schema.joblib")

print(f"✅ Saved bot model to {MODEL_DIR/'twibot_rf.joblib'}")
