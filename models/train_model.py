"""
train_model.py
──────────────
Trains a RandomForestClassifier on fatigue_data.csv and saves the model.

Run order:
  1. python data/generate_dataset.py     ← create initial dataset
  2. python models/train_model.py        ← train and save model
  3. python main.py                      ← run detection with trained model

Re-run this script every ~20 real sessions to improve accuracy with your
own collected data.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "fatigue_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fatigue_model.pkl")

# ─── Load Data ────────────────────────────────────────────────────────────────
print("Loading dataset...")
data = pd.read_csv(DATA_PATH)
print(f"  Total samples : {len(data)}")
print(f"  Class counts  :\n{data['label'].value_counts().sort_index().to_string()}")

# ─── Features ─────────────────────────────────────────────────────────────────
FEATURES = ["blink_total", "yawn_total", "eye_closure_events",
            "blink_rate", "duration", "fatigue_score"]

X = data[FEATURES]
y = data["label"]

# ─── Train / Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── Model ────────────────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=4,
    class_weight="balanced",
    random_state=42,
)

print("\nTraining model...")
model.fit(X_train, y_train)

# ─── Evaluation ───────────────────────────────────────────────────────────────
y_pred    = model.predict(X_test)
accuracy  = model.score(X_test, y_test)

print(f"\nModel Accuracy : {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["Low", "Moderate", "High"]))

# Feature importance
print("Feature Importances:")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    print(f"  {feat:<25} {imp:.4f}")

# ─── Save Model ───────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\n✅ Model saved to: {MODEL_PATH}")