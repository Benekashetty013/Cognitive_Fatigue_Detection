"""
feature_fusion.py
─────────────────
Combines raw detection counts into a 6-feature vector for the fatigue model.

Features
--------
blink_total       : total complete blinks in session
yawn_total        : total confirmed yawns in session
eye_closure_events: prolonged eye closures (microsleep events)
blink_rate        : blinks per minute (normalized speed indicator)
duration          : planned session length in minutes
fatigue_score     : weighted composite score (primary signal)

Fatigue Score Formula
---------------------
  score = (yawn_total * 2.5)
        + (eye_closure_events * 3.0)
        + max(0, 15 - blink_rate)

  Interpretation:
  - More yawns      → higher score  (strong fatigue signal)
  - More closures   → higher score  (microsleep = critical)
  - Blink rate < 15 → higher score  (drowsy staring)
"""


def fuse_features(blink_total: int,
                  yawn_total: int,
                  eye_closure_events: int,
                  elapsed_minutes: float,
                  duration: float) -> dict:
    """
    Parameters
    ----------
    blink_total        : number of blinks detected so far
    yawn_total         : number of yawns detected so far
    eye_closure_events : number of prolonged eye closures
    elapsed_minutes    : time elapsed in the current session (minutes)
    duration           : total planned session duration (minutes)

    Returns
    -------
    dict with keys: blink_total, yawn_total, eye_closure_events,
                    blink_rate, duration, fatigue_score
    """
    elapsed_minutes = max(elapsed_minutes, 0.01)   # guard against zero division

    # Blinks per minute — normalises across short and long sessions
    blink_rate = blink_total / elapsed_minutes

    # Composite fatigue score
    # Yawns and eye-closures carry the most weight; low blink-rate adds penalty
    fatigue_score = (
        (yawn_total         * 2.5) +
        (eye_closure_events * 3.0) +
        max(0.0, 15.0 - blink_rate)
    )

    return {
        "blink_total"       : blink_total,
        "yawn_total"        : yawn_total,
        "eye_closure_events": eye_closure_events,
        "blink_rate"        : round(blink_rate, 3),
        "duration"          : duration,
        "fatigue_score"     : round(fatigue_score, 3),
    }