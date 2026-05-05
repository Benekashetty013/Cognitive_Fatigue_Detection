# Cognitive Fatigue Detection

This repository contains a cognitive fatigue detection project with two parts:

- A Python computer-vision and machine-learning pipeline for fatigue detection.
- An Android application project built with Kotlin and Jetpack Compose.

## Project Contents

- `main.py` - Python entry point for webcam-based fatigue detection.
- `src/`, `utils/`, `data/`, `models/` - Python pipeline modules, dataset utilities, and trained model files.
- `app/`, `gradle/`, `gradlew.bat`, `settings.gradle.kts` - Android application source and Gradle project files.
- `fatigue_model.onnx`, `fatigue_model.tflite`, `mobile_assets/` - exported model/mobile assets.
- `apk/FatigueApp-debug.apk` - debug APK built for sharing/testing.
- `PROJECT_GUIDE.md` - detailed project guide.
- `PROJECT_PROMPTS.md` - collected visible prompts for the project/session.

## Python Setup

```bash
pip install -r requirements.txt
python main.py
```

## Android Build

```bash
.\gradlew.bat :app:assembleDebug
```

The generated debug APK is copied to:

```text
apk/FatigueApp-debug.apk
```
