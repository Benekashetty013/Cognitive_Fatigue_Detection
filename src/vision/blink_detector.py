from collections import deque
from dataclasses import dataclass, field

from utils.geometry import euclidean

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(landmarks, eye):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye]

    v1 = euclidean(p2, p6)
    v2 = euclidean(p3, p5)
    h = euclidean(p1, p4)

    if h == 0:
        return 0.0

    return float((v1 + v2) / (2 * h))


@dataclass
class AdaptiveBlinkTracker:
    """
    Tracks blink and prolonged eye closures with a person-specific EAR baseline.

    Small-eye users often have naturally lower EAR values. Instead of a single
    fixed threshold, we learn a running "eyes open" baseline and derive a
    closure threshold from it. Timing is also based on estimated FPS so blink
    and microsleep detection stays realistic across cameras.
    """

    baseline_ear: float | None = None
    smoothed_ear: float | None = None
    close_frame_count: int = 0
    blink_total: int = 0
    eye_closure_events: int = 0
    recent_open_ears: deque = field(default_factory=lambda: deque(maxlen=60))

    def update(self, ear: float, fps: float) -> dict:
        ear = float(max(ear, 0.0))
        fps = max(float(fps), 1.0)

        if self.smoothed_ear is None:
            self.smoothed_ear = ear
        else:
            self.smoothed_ear = (0.65 * self.smoothed_ear) + (0.35 * ear)

        threshold = self.current_threshold()
        is_closed = self.smoothed_ear < threshold

        if not is_closed and self.smoothed_ear > threshold * 1.08:
            self._update_open_baseline(self.smoothed_ear)
            threshold = self.current_threshold()

        blink_min_frames = max(2, int(round(0.10 * fps)))
        closure_min_frames = max(blink_min_frames + 2, int(round(0.60 * fps)))

        if is_closed:
            self.close_frame_count += 1
        else:
            if blink_min_frames <= self.close_frame_count < closure_min_frames:
                self.blink_total += 1
            elif self.close_frame_count >= closure_min_frames:
                self.eye_closure_events += 1
            self.close_frame_count = 0

        return {
            "ear": round(ear, 4),
            "smoothed_ear": round(self.smoothed_ear, 4),
            "ear_threshold": round(threshold, 4),
            "is_closed": is_closed,
            "blink_total": self.blink_total,
            "eye_closure_events": self.eye_closure_events,
            "blink_min_frames": blink_min_frames,
            "closure_min_frames": closure_min_frames,
        }

    def finalize(self, fps: float) -> dict:
        fps = max(float(fps), 1.0)
        blink_min_frames = max(2, int(round(0.10 * fps)))
        closure_min_frames = max(blink_min_frames + 2, int(round(0.60 * fps)))

        if blink_min_frames <= self.close_frame_count < closure_min_frames:
            self.blink_total += 1
        elif self.close_frame_count >= closure_min_frames:
            self.eye_closure_events += 1

        self.close_frame_count = 0
        return {
            "blink_total": self.blink_total,
            "eye_closure_events": self.eye_closure_events,
        }

    def current_threshold(self) -> float:
        if self.baseline_ear is None:
            return 0.18

        # Adaptive threshold:
        # - lower for naturally small eyes
        # - capped for large-eye users to avoid false closures
        return max(0.11, min(0.23, self.baseline_ear * 0.73))

    def _update_open_baseline(self, ear: float) -> None:
        self.recent_open_ears.append(float(ear))
        sorted_ears = sorted(self.recent_open_ears)
        top_window = sorted_ears[max(0, len(sorted_ears) // 2):]
        candidate = sum(top_window) / max(len(top_window), 1)

        if self.baseline_ear is None:
            self.baseline_ear = candidate
        else:
            self.baseline_ear = (0.85 * self.baseline_ear) + (0.15 * candidate)
