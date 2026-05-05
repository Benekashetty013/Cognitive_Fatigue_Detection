"""
geometry.py
───────────
Shared geometric utility functions used across the vision modules.
Centralises euclidean() so it isn't duplicated in blink_detector and yawn_detector.
"""

import numpy as np


def euclidean(p1, p2) -> float:
    """Euclidean distance between two (x, y) points."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))