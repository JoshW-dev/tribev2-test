"""
TRIBE v2 inference script.
Accepts a video, audio, or text file and outputs predicted brain responses.
"""

import sys
from pathlib import Path

# The cloned repo directory "tribev2/" shadows the installed package.
# Remove the script's parent directory from sys.path so the real package is found.
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _script_dir]

import argparse
import numpy as np
from tribev2 import TribeModel


def main():
    parser = argparse.ArgumentParser(description="Run TRIBE v2 inference")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--audio", type=str, help="Path to input audio file")
    parser.add_argument("--text", type=str, help="Path to input text file")
    parser.add_argument("--cache", type=str, default="./cache", help="Model cache dir")
    parser.add_argument("--output", type=str, default="predictions.npy", help="Output .npy file")
    args = parser.parse_args()

    if not any([args.video, args.audio, args.text]):
        parser.error("Provide at least one of --video, --audio, or --text")

    # The upstream config hardcodes device: cuda for all feature extractors.
    # Override to cpu when CUDA is not available (e.g. Mac).
    import torch
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
        "facebook/tribev2", cache_folder=args.cache, config_update=config_update
    )

    event_kwargs = {}
    if args.video:
        event_kwargs["video_path"] = args.video
        print(f"Processing video: {args.video}")
    if args.audio:
        event_kwargs["audio_path"] = args.audio
        print(f"Processing audio: {args.audio}")
    if args.text:
        event_kwargs["text_path"] = args.text
        print(f"Processing text: {args.text}")

    print("Extracting events...")
    events = model.get_events_dataframe(**event_kwargs)
    print(f"Events shape: {events.shape}")
    print(f"Event columns: {list(events.columns)}")

    print("Running prediction...")
    preds, segments = model.predict(events=events)

    print(f"\nResults:")
    print(f"  preds shape: {preds.shape}  (n_timesteps x n_vertices)")
    print(f"  segments type: {type(segments)}")
    if hasattr(segments, "__len__"):
        print(f"  segments length: {len(segments)}")
        if len(segments) > 0:
            print(f"  first 3 segments: {segments[:3]}")

    np.save(args.output, preds)
    print(f"\nPredictions saved to {args.output}")


if __name__ == "__main__":
    main()
