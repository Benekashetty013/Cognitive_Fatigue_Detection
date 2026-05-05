"""
generate_dataset.py
───────────────────
Generates a synthetic but realistic fatigue dataset (300 samples).

Key fix: eye_closure_events represents MICROSLEEP events (eyes closed 0.67s+),
NOT regular blinks. A healthy person has 0-1 microsleep events per session.
eye_closure_events must never equal blink_total.

Feature ranges (per label)
───────────────────────────
Label 0 — Low Fatigue      (alert, working fine)
  blink_rate     : 15–22 blinks/min   (healthy range)
  yawn_total     : 0–1
  eye_closures   : 0–1               (almost no microsleep)

Label 1 — Moderate Fatigue (drowsy, losing focus)
  blink_rate     : 8–14 blinks/min   (slowing down)
  yawn_total     : 2–5
  eye_closures   : 1–3               (occasional microsleep)

Label 2 — High Fatigue     (severely drowsy)
  blink_rate     : 3–7 blinks/min    (very slow)
  yawn_total     : 6–12
  eye_closures   : 4–10              (frequent microsleep)
"""

import csv
import os
import random
import time

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "fatigue_data.csv")
SAMPLES_PER_CLASS = 100


def fatigue_score(yawn_total, eye_closure_events, blink_rate):
    return (yawn_total * 2.5) + (eye_closure_events * 3.0) + max(0.0, 15.0 - blink_rate)


def generate_sample(label: int, rng: random.Random) -> dict:
    duration = round(rng.uniform(5, 60), 1)

    if label == 0:
        blink_rate         = rng.uniform(15, 22)
        yawn_total         = rng.randint(0, 1)
        eye_closure_events = rng.randint(0, 1)

    elif label == 1:
        blink_rate         = rng.uniform(8, 14)
        yawn_total         = rng.randint(2, 5)
        eye_closure_events = rng.randint(1, 3)

    else:
        blink_rate         = rng.uniform(3, 7)
        yawn_total         = rng.randint(6, 12)
        eye_closure_events = rng.randint(4, 10)

    blink_total = max(1, int(blink_rate * duration))
    score = fatigue_score(yawn_total, eye_closure_events, blink_rate)

    return {
        "timestamp"          : round(time.time() + rng.uniform(-86400, 0), 2),
        "blink_total"        : blink_total,
        "yawn_total"         : yawn_total,
        "eye_closure_events" : eye_closure_events,
        "blink_rate"         : round(blink_rate, 3),
        "duration"           : duration,
        "fatigue_score"      : round(score, 3),
        "label"              : label,
    }


def main():
    rng = random.Random(42)
    rows = []
    for label in (0, 1, 2):
        for _ in range(SAMPLES_PER_CLASS):
            rows.append(generate_sample(label, rng))
    rng.shuffle(rows)

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)

    fieldnames = [
        "timestamp", "blink_total", "yawn_total", "eye_closure_events",
        "blink_rate", "duration", "fatigue_score", "label"
    ]

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset generated: {OUTPUT_FILE}")
    print(f"  Total samples : {len(rows)}")
    print(f"  Per class     : {SAMPLES_PER_CLASS}")
    print()
    print("Sanity check — eye_closure_events should NOT equal blink_total:")
    same = sum(1 for r in rows if r['eye_closure_events'] == r['blink_total'])
    print(f"  Rows where closure == blink_total : {same}  (should be ~0)")


if __name__ == "__main__":
    main()