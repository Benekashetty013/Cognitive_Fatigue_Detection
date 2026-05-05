"""
fatigue_model.py
────────────────
Loads the trained RandomForest model and exposes predict_fatigue().

If no trained model is found (fatigue_model.pkl), a rule-based fallback
is used so the app still runs before any training has been done.

Feature order (must match train_model.py)
─────────────────────────────────────────
  [blink_total, yawn_total, eye_closure_events,
   blink_rate, duration, fatigue_score]
"""

import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "fatigue_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
    print("[fatigue_model] ✅ Trained model loaded successfully.")
except Exception:
    model = None
    MODEL_LOADED = False
    print("[fatigue_model] ⚠  No trained model found — using rule-based fallback.")


def predict_fatigue(features: dict) -> int:
    """
    Predict fatigue level from a feature dictionary.

    Parameters
    ----------
    features : dict returned by fuse_features()

    Returns
    -------
    int : 0 = Low Fatigue, 1 = Moderate Fatigue, 2 = High Fatigue
    """
    if MODEL_LOADED:
        x = [[
            features["blink_total"],
            features["yawn_total"],
            features["eye_closure_events"],
            features["blink_rate"],
            features["duration"],
            features["fatigue_score"],
        ]]
        return int(model.predict(x)[0])

    # ── Rule-based fallback (used before first model is trained) ──────────────
    score = features.get("fatigue_score", 0)

    if score < 5:
        return 0   # Low fatigue
    elif score < 12:
        return 1   # Moderate fatigue
    else:
        return 2   # High fatigue