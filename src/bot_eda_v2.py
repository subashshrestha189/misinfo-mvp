# src/bot_eda_v2.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.twibot_features import build_features

df = pd.read_csv("data/twibot/users.csv")
assert "label" in df.columns, "CSV must include 'label' (1=bot, 0=human)."

print("Rows:", len(df))
print(df["label"].value_counts(dropna=False))

X = build_features(df)
print("Feature columns:", list(X.columns))
print(X.describe(percentiles=[.1,.5,.9]).T)

# Plot class balance
sns.countplot(x=df["label"])
plt.title("Class balance (0=human, 1=bot)")
plt.show()
