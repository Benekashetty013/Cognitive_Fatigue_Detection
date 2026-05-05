from dataclasses import dataclass

from utils.geometry import euclidean

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
NOSE_TIP = 1
CHIN = 152
FACE_LEFT = 234
FACE_RIGHT = 454


def mouth_aspect_ratio(landmarks):
    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]

    vertical_gap = euclidean(top, bottom)
    mouth_width = euclidean(left, right)

    if mouth_width == 0:
        return 0.0

    return float(vertical_gap / mouth_width)


def mouth_open_ratio(landmarks):
    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]
    nose = landmarks[NOSE_TIP]
    chin = landmarks[CHIN]

    vertical_gap = euclidean(top, bottom)
    lower_face_height = euclidean(nose, chin)

    if lower_face_height == 0:
        return 0.0

    return float(vertical_gap / lower_face_height)


def jaw_drop_ratio(landmarks):
    nose = landmarks[NOSE_TIP]
    chin = landmarks[CHIN]
    face_left = landmarks[FACE_LEFT]
    face_right = landmarks[FACE_RIGHT]

    lower_face_height = euclidean(nose, chin)
    face_width = euclidean(face_left, face_right)

    if face_width == 0:
        return 0.0

    return float(lower_face_height / face_width)


@dataclass
class AdaptiveYawnTracker:
    """
    Tracks yawns with frame-rate-aware timing and multiple mouth signals.

    We keep the requested MAR threshold at 0.5, but also consider shorter-span
    yawns and partially covered yawns through two extra signals:
    - lip-gap relative to lower-face height
    - jaw drop relative to face width
    """

    yawn_frame_count: int = 0
    yawn_total: int = 0
    covered_yawn_total: int = 0
    covered_yawn_candidate_in_window: bool = False

    def update(self, landmarks, fps: float) -> dict:
        fps = max(float(fps), 1.0)

        mar = mouth_aspect_ratio(landmarks)
        open_ratio = mouth_open_ratio(landmarks)
        jaw_ratio = jaw_drop_ratio(landmarks)

        yawn_min_frames = max(6, int(round(0.40 * fps)))

        is_primary_yawn = (mar >= 0.50) or (open_ratio >= 0.24)
        is_covered_yawn = (jaw_ratio >= 0.66) and (open_ratio >= 0.16)
        is_yawning = is_primary_yawn or is_covered_yawn

        if is_yawning:
            self.yawn_frame_count += 1
            self.covered_yawn_candidate_in_window = (
                self.covered_yawn_candidate_in_window or is_covered_yawn
            )
        else:
            if self.yawn_frame_count >= yawn_min_frames:
                self.yawn_total += 1
                if self.covered_yawn_candidate_in_window:
                    self.covered_yawn_total += 1
            self.yawn_frame_count = 0
            self.covered_yawn_candidate_in_window = False

        return {
            "mar": round(mar, 4),
            "mouth_open_ratio": round(open_ratio, 4),
            "jaw_drop_ratio": round(jaw_ratio, 4),
            "is_yawning": is_yawning,
            "is_covered_yawn_candidate": is_covered_yawn,
            "yawn_total": self.yawn_total,
            "covered_yawn_total": self.covered_yawn_total,
            "yawn_min_frames": yawn_min_frames,
        }

    def finalize(self, fps: float) -> dict:
        fps = max(float(fps), 1.0)
        yawn_min_frames = max(6, int(round(0.40 * fps)))

        if self.yawn_frame_count >= yawn_min_frames:
            self.yawn_total += 1
            if self.covered_yawn_candidate_in_window:
                self.covered_yawn_total += 1

        self.yawn_frame_count = 0
        self.covered_yawn_candidate_in_window = False
        return {
            "yawn_total": self.yawn_total,
            "covered_yawn_total": self.covered_yawn_total,
        }
