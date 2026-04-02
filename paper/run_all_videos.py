"""Run TRIBE v2 inference on all sample videos and save results for comparison."""

import sys, os, time
from pathlib import Path

# Fix: remove project root from sys.path so the tribev2/ directory
# doesn't shadow the installed tribev2 package
_project_root = str(Path(__file__).resolve().parent.parent)
sys.path = [p for p in sys.path if p != _project_root]

# Ensure ffmpeg/uvx on PATH (MUST happen before any tribev2 imports)
_extra_paths = [
    Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin",
    Path.home() / "AppData/Local/Microsoft/WinGet/Packages/astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe",
    Path.home() / "AppData/Local/Programs/Python/Python311/Scripts",
]
for _extra in _extra_paths:
    if _extra.exists():
        os.environ["PATH"] = str(_extra) + os.pathsep + os.environ.get("PATH", "")

# Verify tools are accessible
import shutil
print(f"uvx found: {shutil.which('uvx')}")
print(f"ffmpeg found: {shutil.which('ffmpeg')}")

import numpy as np
import torch
from tribev2 import TribeModel

PROJECT = Path(__file__).resolve().parent.parent
COMPARE_DIR = PROJECT / "paper" / "comparison_data"
COMPARE_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    ("BusinessEdLeilaHarmozi.mp4", "Business Education (Leila Hormozi)"),
    ("ElonAI.mp4", "Tech/AI News (Elon)"),
    ("PerfumeUGCInterview.mp4", "UGC Product Interview (Perfume)"),
    ("sanitaryPadProductDemo.mp4", "Product Demo (Sanitary Pad)"),
    ("viralJapaneseIceCutter.mp4", "Viral Satisfying (Ice Cutter)"),
]


def main():
    # Load model
    config_update = None
    if not torch.cuda.is_available():
        config_update = {
            "data.text_feature.device": "cpu",
            "data.audio_feature.device": "cpu",
            "data.video_feature.image.device": "cpu",
            "data.image_feature.image.device": "cpu",
        }
    print("Loading TRIBE v2 model...")
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=str(PROJECT / "cache"),
        config_update=config_update,
    )

    # Load Yeo labels
    labels_path = PROJECT / "results" / "yeo7_labels.npy"
    if labels_path.exists():
        yeo_labels = np.load(labels_path).ravel()
    else:
        print("ERROR: Yeo labels not found. Run the app first.")
        return

    results = {}
    for filename, label in VIDEOS:
        video_path = PROJECT / filename
        out_path = COMPARE_DIR / f"{Path(filename).stem}_preds.npy"
        tc_path = COMPARE_DIR / f"{Path(filename).stem}_tc.npy"

        if tc_path.exists():
            print(f"[CACHED] {label}: {tc_path}")
            results[label] = np.load(tc_path)
            continue

        if not video_path.exists():
            print(f"[SKIP] {filename} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {label} ({filename})")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            events = model.get_events_dataframe(video_path=str(video_path))
            preds, segments = model.predict(events=events)
            np.save(out_path, preds)

            # Compute network timecourses
            tc = np.zeros((preds.shape[0], 7))
            for i in range(7):
                mask = yeo_labels == (i + 1)
                if mask.any():
                    tc[:, i] = preds[:, mask].mean(axis=1)
            np.save(tc_path, tc)
            results[label] = tc

            elapsed = time.time() - t0
            print(f"Done: {preds.shape[0]} timesteps, {elapsed:.0f}s")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for label, tc in results.items():
        mean_abs = np.abs(tc).mean(axis=0)
        dominant = int(np.argmax(mean_abs))
        names = ["Visual", "Somatomotor", "Dorsal Attn", "Ventral Attn",
                 "Limbic", "Frontoparietal", "Default Mode"]
        print(f"  {label}: {tc.shape[0]} TRs, dominant={names[dominant]}")


if __name__ == "__main__":
    main()
