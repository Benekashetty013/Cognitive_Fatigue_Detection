"""
export_android_assets.py
------------------------
Syncs the Android runtime assets for the real FatigueApp project.

Why this exists
---------------
This repository owns the fatigue prediction pipeline and the exported
`fatigue_model.tflite`, but the Android app also needs a separate
MediaPipe face landmark model bundle:

* `fatigue_model.tflite`        -> your fatigue classifier
* `face_landmarker.task`        -> pretrained MediaPipe face landmark detector

The face landmarker is not produced by this training pipeline. It must be
downloaded as an official pretrained MediaPipe model bundle, then copied to
the Android app assets folder alongside the fatigue model.

Usage
-----
    python export_android_assets.py

Optional:
    python export_android_assets.py --force-download
    python export_android_assets.py --app-dir "C:\\path\\to\\FatigueApp"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_APP_DIR = Path(r"C:\Users\Admin\AndroidStudioProjects\FatigueApp")
ANDROID_ASSETS_REL = Path("app/src/main/assets")
CACHED_ASSETS_DIR = PROJECT_ROOT / "mobile_assets"
FATIGUE_MODEL_PATH = PROJECT_ROOT / "fatigue_model.tflite"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_NAME = "face_landmarker.task"
MANIFEST_NAME = "asset_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def download_face_landmarker(cache_path: Path, force_download: bool) -> Path:
    if cache_path.exists() and not force_download:
        print(f"[cache] Using existing {cache_path}")
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] Fetching official MediaPipe model from:\n  {FACE_LANDMARKER_URL}")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, cache_path)

    if not cache_path.exists() or cache_path.stat().st_size == 0:
        raise RuntimeError("Downloaded face_landmarker.task is missing or empty.")

    print(f"[download] Saved to {cache_path}")
    return cache_path


def write_manifest(manifest_path: Path, fatigue_asset: Path, face_asset: Path, app_dir: Path) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(PROJECT_ROOT),
        "target_app_dir": str(app_dir),
        "assets": {
            "fatigue_model.tflite": {
                "source": str(FATIGUE_MODEL_PATH),
                "target": str(fatigue_asset),
                "size_bytes": fatigue_asset.stat().st_size,
                "sha256": sha256(fatigue_asset),
                "origin": "local fatigue pipeline export",
            },
            FACE_LANDMARKER_NAME: {
                "source": FACE_LANDMARKER_URL,
                "target": str(face_asset),
                "size_bytes": face_asset.stat().st_size,
                "sha256": sha256(face_asset),
                "origin": "official MediaPipe pretrained model bundle",
            },
        },
        "notes": [
            "fatigue_model.tflite is the fatigue classifier exported from this repository.",
            "face_landmarker.task is a separate pretrained MediaPipe dependency and is not trained by this repository.",
            "Both assets must be present in the Android app assets folder for the full app experience.",
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Android runtime assets for FatigueApp.")
    parser.add_argument(
        "--app-dir",
        default=str(DEFAULT_APP_DIR),
        help="Root folder of the Android FatigueApp project.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the official face_landmarker.task even if it is cached locally.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    app_dir = Path(args.app_dir).resolve()
    android_assets_dir = app_dir / ANDROID_ASSETS_REL
    fatigue_asset_path = android_assets_dir / FATIGUE_MODEL_PATH.name
    face_cache_path = CACHED_ASSETS_DIR / FACE_LANDMARKER_NAME
    face_asset_path = android_assets_dir / FACE_LANDMARKER_NAME
    manifest_path = android_assets_dir / MANIFEST_NAME

    if not FATIGUE_MODEL_PATH.exists():
        print(f"[error] Missing fatigue model export: {FATIGUE_MODEL_PATH}")
        print("Run convert_to_tflite.py first if you need to regenerate fatigue_model.tflite.")
        return 1

    if not app_dir.exists():
        print(f"[error] Android app directory not found: {app_dir}")
        return 1

    print(f"[sync] Android app: {app_dir}")
    print(f"[sync] Assets dir : {android_assets_dir}")

    try:
        face_cache = download_face_landmarker(face_cache_path, force_download=args.force_download)
    except Exception as exc:
        print(f"[error] Could not fetch face_landmarker.task: {exc}")
        return 1

    copy_file(FATIGUE_MODEL_PATH, fatigue_asset_path)
    copy_file(face_cache, face_asset_path)
    write_manifest(manifest_path, fatigue_asset_path, face_asset_path, app_dir)

    print("[done] Export complete.")
    print(f"  fatigue model  -> {fatigue_asset_path}")
    print(f"  face landmarker-> {face_asset_path}")
    print(f"  manifest       -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
