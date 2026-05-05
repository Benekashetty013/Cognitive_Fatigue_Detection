# 🧠 Fatigue Detection System — Complete Project Guide

## 📋 Table of Contents
1. [Project Overview](#overview)
2. [Code Analysis — Bugs & Issues Found](#bugs)
3. [Correct Project Structure](#structure)
4. [Feature Engineering](#features)
5. [Dataset Creation](#dataset)
6. [Model Training](#model)
7. [Step-by-Step Build Order](#steps)
8. [Files You Must Change](#changes)

---

## 1. Project Overview <a name="overview"></a>

This app detects human fatigue in real time via webcam by:
- Counting **blinks** (via Eye Aspect Ratio)
- Counting **yawns** (via Mouth Aspect Ratio)
- Measuring **microsleep duration** (prolonged eye closure)
- Computing **blink rate** (blinks per minute)
- Computing a **fatigue score** from all features

### Fatigue Levels & Recommendations

| Level | Label | Recommendation |
|-------|-------|----------------|
| 🟢 Low Fatigue | `0` | Drink water, breathe deeply |
| 🟡 Moderate Fatigue | `1` | 10-min walk, wash face |
| 🔴 High Fatigue | `2` | 20-min power nap, cold face wash, 5-min walk outside |

---

## 2. Code Analysis — Bugs & Issues Found <a name="bugs"></a>

### ❌ Bug 1 — Wrong `time` import in `main.py`
```python
# WRONG (line 1)
from time import time
# Then later uses:
time.sleep(delay)   # ← This crashes! `time` is a function here, not a module

# FIX
import time
```

### ❌ Bug 2 — All import paths are broken
`main.py` imports from `utils.camera`, `src.vision.*`, `src.behavior.*` etc. — but **all your files sit flat** in one folder with no subfolders. Every import will fail with `ModuleNotFoundError`.

```python
# WRONG (current)
from utils.camera import start_camera
from src.vision.face_detector import detect_face_landmarks

# FIX — after reorganizing (see structure below)
# All imports must match the actual folder structure
```

### ❌ Bug 3 — Wrong recommendations in `main.py`
```python
# CURRENT — label 0 means "Not Fatigued" (shows nothing)
# REQUIRED — label 0 = Low Fatigue → recommend water + breathing
```

### ⚠️ Issue 4 — Only 4 features; no fatigue score
`feature_fusion.py` returns raw counts. The model needs a computed **fatigue score** and **blink rate** for meaningful predictions.

### ⚠️ Issue 5 — No dataset exists yet
`train_model.py` reads `data/fatigue_data.csv` but that file doesn't exist. You must generate it before training.

### ⚠️ Issue 6 — `euclidean()` defined twice
Both `blink_detector.py` and `yawn_detector.py` define the same `euclidean()` function. Move it to a shared utility.

---

## 3. Correct Project Structure <a name="structure"></a>

```
fatigue_detection/
│
├── main.py                        ← Entry point (UPDATED)
├── requirements.txt               ← Dependencies
│
├── src/
│   ├── __init__.py
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── face_detector.py       ← Detects face landmarks (unchanged)
│   │   ├── blink_detector.py      ← EAR calculation (unchanged)
│   │   └── yawn_detector.py       ← MAR calculation (unchanged)
│   │
│   ├── behavior/
│   │   ├── __init__.py
│   │   └── session_analyzer.py    ← Session timer (unchanged)
│   │
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── feature_fusion.py      ← UPDATED: adds blink_rate + fatigue_score
│   │
│   └── model/
│       ├── __init__.py
│       └── fatigue_model.py       ← UPDATED: improved rule-based fallback
│
├── utils/
│   ├── __init__.py
│   ├── camera.py                  ← Camera stream (unchanged)
│   └── geometry.py                ← NEW: shared euclidean() function
│
├── data/
│   ├── __init__.py
│   ├── fatigue_logger.py          ← Logs session data (unchanged)
│   ├── generate_dataset.py        ← NEW: synthetic dataset generator
│   └── fatigue_data.csv           ← Generated after running generator
│
└── models/
    ├── train_model.py             ← UPDATED: uses 6 features
    └── fatigue_model.pkl          ← Generated after training
```

---

## 4. Feature Engineering <a name="features"></a>

The model uses **6 features** (upgraded from 4):

| Feature | How It's Computed | Why It Matters |
|---------|-------------------|----------------|
| `blink_total` | Count of complete blinks | Too few = staring/drowsy |
| `yawn_total` | Count of confirmed yawns | Direct fatigue indicator |
| `eye_closure_events` | Prolonged eye closures (>3 frames) | Microsleep indicator |
| `blink_rate` | blinks_per_minute = blink_total / elapsed_minutes | Normalized across session lengths |
| `duration` | Session length in minutes | Context for other features |
| `fatigue_score` | Weighted formula (see below) | Composite signal |

### Fatigue Score Formula
```python
fatigue_score = (yawn_total * 2.5) + (eye_closure_events * 3.0) + max(0, 15 - blink_rate)
```
- Yawns are weighted highest (strongest signal)
- Microsleep events are critical
- Very low blink rate (< 15/min) adds to score

---

## 5. Dataset Creation <a name="dataset"></a>

Run `data/generate_dataset.py` **before** training. It creates synthetic but realistic samples:

```
Label 0 (Low):      blink=16-20/min, yawn=0-1, closure=0-1
Label 1 (Moderate): blink=10-15/min, yawn=2-4, closure=2-4
Label 2 (High):     blink=4-9/min,   yawn=5+,  closure=5+
```

After collecting real sessions, **real data will replace synthetic data** automatically (the logger appends to the same CSV).

---

## 6. Model Training <a name="model"></a>

After generating the dataset:
```bash
python models/train_model.py
```
This trains a **RandomForestClassifier** and saves `models/fatigue_model.pkl`.

Every time you run a session and provide feedback, the CSV grows. Re-run training periodically to improve accuracy.

---

## 7. Step-by-Step Build Order <a name="steps"></a>

```
PHASE 1 — Setup
  Step 1: Create the folder structure above
  Step 2: Copy each file into its correct folder
  Step 3: Create all __init__.py files (can be empty)
  Step 4: pip install -r requirements.txt

PHASE 2 — Data
  Step 5: Run python data/generate_dataset.py
           → Creates data/fatigue_data.csv with 300 samples

PHASE 3 — Model
  Step 6: Run python models/train_model.py
           → Creates models/fatigue_model.pkl

PHASE 4 — Run
  Step 7: Run python main.py
           → Camera opens, session starts, fatigue is detected live

PHASE 5 — Improve
  Step 8: After each session, answer the feedback prompt
           → Your real data gets added to the CSV
  Step 9: Re-run train_model.py every ~20 sessions
           → Model improves with your real-world data
```

---

## 8. Files You Must Change <a name="changes"></a>

| File | What to Change |
|------|----------------|
| `main.py` | Fix `import time`, fix import paths, update recommendations |
| `src/fusion/feature_fusion.py` | Add `blink_rate` and `fatigue_score` to returned features |
| `src/model/fatigue_model.py` | Update to use 6 features, improve rule-based fallback |
| `models/train_model.py` | Use 6 features, save to `models/` folder |
| **NEW** `data/generate_dataset.py` | Create synthetic dataset |
| **NEW** `utils/geometry.py` | Shared `euclidean()` function |

All corrected versions of these files are provided alongside this guide.
