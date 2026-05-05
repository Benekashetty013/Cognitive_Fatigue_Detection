import time
from collections import deque

import cv2
import mediapipe as mp

from utils.camera import start_camera
from src.vision.face_detector import detect_face_landmarks
from src.vision.blink_detector import (
    AdaptiveBlinkTracker,
    LEFT_EYE,
    RIGHT_EYE,
    eye_aspect_ratio,
)
from src.vision.yawn_detector import AdaptiveYawnTracker
from src.behavior.session_analyzer import SessionAnalyzer
from src.fusion.feature_fusion import fuse_features
from src.model.fatigue_model import predict_fatigue
from data.fatigue_logger import FatigueLogger

mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def rolling_fps(frame_times: deque) -> float:
    if len(frame_times) < 2:
        return 30.0

    elapsed = frame_times[-1] - frame_times[0]
    if elapsed <= 0:
        return 30.0

    return (len(frame_times) - 1) / elapsed


# Session setup
while True:
    try:
        duration = float(input("Enter session duration (in minutes): "))
        if duration <= 0:
            raise ValueError
        break
    except ValueError:
        print("Please enter a valid positive number.")

session = SessionAnalyzer(duration)
logger = FatigueLogger()

# Detection state
blink_tracker = AdaptiveBlinkTracker()
yawn_tracker = AdaptiveYawnTracker()
frame_times = deque(maxlen=30)

blink_total = 0
yawn_total = 0
eye_closure_events = 0
covered_yawn_total = 0

features = None
fatigue_level = 0
latest_metrics = {
    "ear": 0.0,
    "ear_threshold": 0.18,
    "mar": 0.0,
    "mouth_open_ratio": 0.0,
    "jaw_drop_ratio": 0.0,
    "fps": 30.0,
}

# Detection loop
for frame in start_camera():
    frame_times.append(time.perf_counter())
    fps = rolling_fps(frame_times)
    latest_metrics["fps"] = fps

    if not session.is_active():
        break

    landmarks = detect_face_landmarks(frame)

    if landmarks:
        for face in landmarks:
            h, w, _ = frame.shape
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in face.landmark]

            ear = (
                eye_aspect_ratio(coords, LEFT_EYE) +
                eye_aspect_ratio(coords, RIGHT_EYE)
            ) / 2.0

            blink_update = blink_tracker.update(ear, fps)
            yawn_update = yawn_tracker.update(coords, fps)

            blink_total = blink_update["blink_total"]
            eye_closure_events = blink_update["eye_closure_events"]
            yawn_total = yawn_update["yawn_total"]
            covered_yawn_total = yawn_update["covered_yawn_total"]

            latest_metrics.update({
                "ear": blink_update["smoothed_ear"],
                "ear_threshold": blink_update["ear_threshold"],
                "mar": yawn_update["mar"],
                "mouth_open_ratio": yawn_update["mouth_open_ratio"],
                "jaw_drop_ratio": yawn_update["jaw_drop_ratio"],
            })

            elapsed_minutes = (duration * 60 - session.time_left()) / 60
            elapsed_minutes = max(elapsed_minutes, 0.01)

            features = fuse_features(
                blink_total,
                yawn_total,
                eye_closure_events,
                elapsed_minutes,
                duration,
            )
            fatigue_level = predict_fatigue(features)

            level_colors = {0: (0, 220, 80), 1: (0, 200, 255), 2: (0, 60, 255)}
            level_labels = {0: "Low Fatigue", 1: "Moderate Fatigue", 2: "High Fatigue"}

            color = level_colors.get(fatigue_level, (255, 255, 255))
            label = level_labels.get(fatigue_level, "Unknown")

            cv2.rectangle(frame, (18, 14), (520, 118), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Fatigue: {label}",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"Blinks: {blink_total}  Yawns: {yawn_total}  Closures: {eye_closure_events}",
                (30, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (220, 220, 220),
                1,
            )
            cv2.putText(
                frame,
                (
                    f"EAR: {latest_metrics['ear']:.3f}/{latest_metrics['ear_threshold']:.3f}  "
                    f"MAR: {latest_metrics['mar']:.3f}  FPS: {fps:.1f}"
                ),
                (30, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (180, 180, 180),
                1,
            )

            mp_draw.draw_landmarks(
                frame,
                face,
                mp_face_mesh.FACEMESH_TESSELATION,
                mp_draw.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                mp_draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
            )

    secs_left = int(session.time_left())
    cv2.putText(
        frame,
        f"Time left: {secs_left}s",
        (30, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (150, 150, 150),
        1,
    )

    cv2.imshow("Fatigue Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

blink_finalize = blink_tracker.finalize(latest_metrics["fps"])
yawn_finalize = yawn_tracker.finalize(latest_metrics["fps"])
blink_total = blink_finalize["blink_total"]
eye_closure_events = blink_finalize["eye_closure_events"]
yawn_total = yawn_finalize["yawn_total"]
covered_yawn_total = yawn_finalize["covered_yawn_total"]

if features is not None:
    elapsed_minutes = (duration * 60 - session.time_left()) / 60
    elapsed_minutes = max(elapsed_minutes, 0.01)
    features = fuse_features(
        blink_total,
        yawn_total,
        eye_closure_events,
        elapsed_minutes,
        duration,
    )

print("\n" + "=" * 40)
print("         SESSION RESULT")
print("=" * 40)
print(f"  Blinks detected        : {blink_total}")
print(f"  Yawns detected         : {yawn_total}")
print(f"  Eye closure events     : {eye_closure_events}")
print(f"  Covered yawn signals   : {covered_yawn_total}")
print("=" * 40)

if features is None:
    print("\nNo face data collected. Cannot predict fatigue.")
    print("Exiting in 5 seconds...")
    time.sleep(5)
    raise SystemExit(0)

fatigue_level = predict_fatigue(features)

if fatigue_level == 0:
    print("\nLOW FATIGUE")
    print("-" * 35)
    print("  You're doing okay, but stay mindful.")
    print()
    print("  Recommendations:")
    print("     - Drink a glass of water")
    print("     - Take 5 deep slow breaths")
    print("     - Roll your shoulders back")
elif fatigue_level == 1:
    print("\nMODERATE FATIGUE")
    print("-" * 35)
    print("  Your focus is dropping. Take a break.")
    print()
    print("  Recommendations:")
    print("     - Take a 10-minute walk outside")
    print("     - Wash your face with cold water")
    print("     - Drink water and have a light snack")
else:
    print("\nHIGH FATIGUE")
    print("-" * 35)
    print("  Stop working. Rest is essential now.")
    print()
    print("  Recommendations:")
    print("     - Take a 20-minute power nap")
    print("     - Wash your face with cold water")
    print("     - Take a 5-minute walk outside")
    print("     - Resume work only after you feel fresh")

print()
feedback = input("Was this prediction correct? (y/n): ").strip().lower()

if feedback == "n":
    while True:
        try:
            label = int(input("Enter correct label - 0 (Low), 1 (Moderate), 2 (High): "))
            if label in (0, 1, 2):
                break
            print("Please enter 0, 1, or 2.")
        except ValueError:
            print("Invalid input.")
else:
    label = fatigue_level

logger.log(
    blink_total,
    yawn_total,
    eye_closure_events,
    duration,
    label,
    blink_rate=features.get("blink_rate", 0.0),
    fatigue_score=features.get("fatigue_score", 0.0),
)

print("\nSession data saved. Thank you!")
