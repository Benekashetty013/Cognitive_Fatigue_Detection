"""
fatigue_logger.py
─────────────────
Appends a completed session's feature vector and label to fatigue_data.csv.
Creates the file with headers if it doesn't exist yet.
"""

import csv
import os
import time


class FatigueLogger:

    FIELDNAMES = [
        "timestamp", "blink_total", "yawn_total", "eye_closure_events",
        "blink_rate", "duration", "fatigue_score", "label"
    ]

    def __init__(self, filepath: str = None):
        if filepath is None:
            base = os.path.dirname(__file__)
            filepath = os.path.join(base, "fatigue_data.csv")

        self.file = filepath
        self._ensure_file()

    def _ensure_file(self):
        """Create the CSV with headers if it doesn't exist."""
        if not os.path.exists(self.file):
            os.makedirs(os.path.dirname(self.file), exist_ok=True)
            with open(self.file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def log(self,
            blink_total: int,
            yawn_total: int,
            eye_closure_events: int,
            duration: float,
            label: int,
            blink_rate: float = 0.0,
            fatigue_score: float = 0.0):
        """
        Append one session record to the CSV.

        Parameters
        ----------
        blink_total        : total blinks in session
        yawn_total         : total yawns in session
        eye_closure_events : total prolonged eye-closure events
        duration           : session duration in minutes
        label              : fatigue label (0=Low, 1=Moderate, 2=High)
        blink_rate         : blinks per minute (optional, computed if 0)
        fatigue_score      : composite score (optional)
        """
        if blink_rate == 0.0 and duration > 0:
            blink_rate = round(blink_total / duration, 3)

        row = {
            "timestamp"          : round(time.time(), 2),
            "blink_total"        : blink_total,
            "yawn_total"         : yawn_total,
            "eye_closure_events" : eye_closure_events,
            "blink_rate"         : blink_rate,
            "duration"           : duration,
            "fatigue_score"      : fatigue_score,
            "label"              : label,
        }

        with open(self.file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(row)