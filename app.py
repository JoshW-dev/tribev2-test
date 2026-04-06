"""
TRIBE v2 Gradio UI — Select a video (or audio/text), run inference,
and view predicted brain activation maps with synced playback.
"""

import os
import sys
from pathlib import Path

# Fix: the cloned "tribev2/" directory shadows the installed package.
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _script_dir]

# Ensure ffmpeg/uvx are on PATH (winget install locations)
for _extra in [
    Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin",
    Path.home() / "AppData/Local/Microsoft/WinGet/Packages/astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe",
]:
    if _extra.exists() and str(_extra) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(_extra) + os.pathsep + os.environ.get("PATH", "")

import json
import shutil
import tempfile
import time

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from nilearn import datasets, plotting, surface

# ── Global state (loaded once at startup) ────────────────────────────
MODEL = None
FSAVERAGE = None
_MESH_CACHE = {}   # pre-computed mesh geometry for plotly
_LAST_PREDS = None  # cache predictions for timestep slider
_LAST_VIDEO = None  # cache video path for synced playback
_YEO_LABELS = None  # (20484,) int array mapping vertices to Yeo networks
_NETWORK_TIMECOURSES = None  # (n_timesteps, 7) mean activation per network

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ── Yeo 2011 7-Network Atlas ────────────────────────────────────────
# Canonical network names, colors, and cognitive interpretations
# Reference: Yeo et al. 2011, J Neurophysiol, doi:10.1152/jn.00338.2011

YEO_NETWORK_NAMES = [
    "Visual",
    "Somatomotor",
    "Dorsal Attention",
    "Ventral Attention",
    "Limbic",
    "Frontoparietal",
    "Default Mode",
]

# Canonical colors from Yeo 2011 publication
YEO_NETWORK_COLORS = [
    "#781286",  # Visual — purple
    "#4682B4",  # Somatomotor — steel blue
    "#00760E",  # Dorsal Attention — green
    "#C43AFA",  # Ventral Attention — violet
    "#DCF8A4",  # Limbic — cream/yellow-green
    "#E69422",  # Frontoparietal — orange
    "#CD3E4E",  # Default Mode — red
]

COGNITIVE_INTERPRETATIONS = {
    0: ("Visual processing", "Predicted activity in visual cortex — estimated engagement with visual features of the stimulus (color, shape, motion, faces, scenes)."),
    1: ("Somatomotor processing", "Predicted activity in sensorimotor cortex — estimated processing of bodily sensations, tactile information, or motor representations."),
    2: ("Directed attention", "Predicted activity in dorsal attention network — estimated top-down spatial attention and voluntary focus on specific stimulus features."),
    3: ("Salience detection", "Predicted activity in ventral attention/salience network — estimated detection of behaviorally relevant or unexpected stimulus events."),
    4: ("Emotional/limbic response", "Predicted activity in limbic network — estimated emotional or affective processing, including reward and memory-related responses."),
    5: ("Cognitive control", "Predicted activity in frontoparietal control network — estimated higher-order reasoning, decision-making, or working memory engagement."),
    6: ("Internal/reflective processing", "Predicted activity in default mode network — estimated self-referential thought, semantic processing, or narrative comprehension."),
}


# ── Results persistence ──────────────────────────────────────────────
def extract_video_frames(video_path: str, n_timesteps: int, tr: float = 1.0) -> list[str]:
    """Extract one frame per TR from the video and save to results/frames/."""
    import subprocess as sp
    frames_dir = RESULTS_DIR / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Get video duration with ffprobe
    try:
        probe = sp.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True
        )
        video_duration = float(probe.stdout.strip())
    except Exception:
        video_duration = n_timesteps * tr

    frame_paths = []
    for t in range(n_timesteps):
        # Each TR maps to a time in the video
        # TRIBE shifts predictions 5s back for hemodynamic lag
        video_time = min(t * tr, video_duration - 0.1)
        out_path = frames_dir / f"frame_{t:04d}.jpg"
        if not out_path.exists():
            sp.run(
                ["ffmpeg", "-y", "-ss", str(video_time), "-i", video_path,
                 "-frames:v", "1", "-q:v", "2", str(out_path)],
                capture_output=True,
            )
        frame_paths.append(str(out_path) if out_path.exists() else None)

    return frame_paths


_FRAME_PATHS: list[str] = []  # cached video frame paths for slider
_BRAIN_FRAMES: list[str] = []  # pre-rendered brain PNGs per timestep


def prerender_brain_frames(preds: np.ndarray) -> list[str]:
    """Pre-render a brain activation PNG for every timestep + mean."""
    brain_dir = RESULTS_DIR / "brain_frames"
    brain_dir.mkdir(parents=True, exist_ok=True)

    n_timesteps, n_vertices = preds.shape
    lh = n_vertices // 2
    fs = FSAVERAGE
    abs_max = float(max(abs(preds.min()), abs(preds.max())))

    paths = []
    # Index -1 → mean, 0..n-1 → individual TRs
    all_data = [("mean", preds.mean(axis=0))] + [
        (f"{t:04d}", preds[t]) for t in range(n_timesteps)
    ]

    for label, data in all_data:
        out_path = brain_dir / f"brain_{label}.png"
        if out_path.exists():
            paths.append(str(out_path))
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5),
                                 subplot_kw={"projection": "3d"})
        fig.patch.set_facecolor("white")
        for ax, hemi in zip(axes, ["left", "right"]):
            hemi_data = data[:lh] if hemi == "left" else data[lh:]
            plotting.plot_surf_stat_map(
                fs[f"pial_{hemi}"], hemi_data, hemi=hemi, view="lateral",
                colorbar=False, bg_map=fs[f"sulc_{hemi}"],
                axes=ax, threshold=0.01, vmax=abs_max,
            )
            ax.set_facecolor("white")

        title = "Mean activation" if label == "mean" else f"TR {int(label)}"
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(str(out_path))

    print(f"Pre-rendered {len(paths)} brain frames to {brain_dir}")
    return paths


def render_brain_animation(preds: np.ndarray,
                           fps: int = 24,
                           tr: float = 1.0,
                           progress_cb=None) -> str | None:
    """Render a rotating 3D brain video using PyVista GPU acceleration.

    Each TR lasts `tr` seconds in the output video so the animation
    matches the duration of the original input.  At `fps` frames per
    second, each TR gets ``int(fps * tr)`` rendered frames.
    """
    frames_per_tr = max(1, int(fps * tr))
    import subprocess as sp
    import pyvista as pv
    pv.OFF_SCREEN = True

    anim_dir = RESULTS_DIR / "animation_frames"
    anim_dir.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "brain_animation.mp4"

    n_timesteps, n_vertices = preds.shape
    abs_max = float(max(abs(preds.min()), abs(preds.max())))

    # Build PyVista mesh (combined hemispheres, same as _MESH_CACHE)
    fs = FSAVERAGE
    lh_data = surface.load_surf_mesh(fs["pial_left"])
    rh_data = surface.load_surf_mesh(fs["pial_right"])
    if isinstance(lh_data, tuple):
        lc, lf = lh_data
        rc, rf = rh_data
    else:
        lc, lf = np.asarray(lh_data[0]), np.asarray(lh_data[1])
        rc, rf = np.asarray(rh_data[0]), np.asarray(rh_data[1])

    rc2 = rc.copy()
    rc2[:, 0] += 2  # same gap as _build_mesh_cache
    n_lh = len(lc)
    coords = np.vstack([lc, rc2])
    faces_l = np.hstack([np.full((len(lf), 1), 3), lf]).astype(np.int64)
    faces_r = np.hstack([np.full((len(rf), 1), 3), rf + n_lh]).astype(np.int64)
    faces = np.vstack([faces_l, faces_r])
    pv_mesh = pv.PolyData(coords, faces)

    # Compute the brain center for the camera focal point
    center = coords.mean(axis=0)

    total_frames = n_timesteps * frames_per_tr
    azim_step = 360.0 / total_frames

    # Camera orbit radius — distance from center
    cam_radius = 250.0

    frame_idx = 0
    t0 = time.time()
    for t in range(n_timesteps):
        pv_mesh["activation"] = preds[t].astype(np.float32)
        for f in range(frames_per_tr):
            frame_path = anim_dir / f"anim_{frame_idx:05d}.png"
            if not frame_path.exists():
                pl = pv.Plotter(off_screen=True, window_size=(1280, 720))
                pl.set_background([0.05, 0.05, 0.1])  # dark blue-black
                pl.add_mesh(
                    pv_mesh, scalars="activation", cmap="RdBu_r",
                    clim=[-abs_max, abs_max],
                    smooth_shading=True, show_scalar_bar=True,
                    scalar_bar_args=dict(
                        title="Activation", color="white",
                        width=0.35, height=0.05,
                        position_x=0.32, position_y=0.02,
                    ),
                )

                # Orbit camera around the brain center (start from front)
                angle_rad = np.radians(frame_idx * azim_step)
                cam_x = center[0] + cam_radius * np.sin(angle_rad)
                cam_y = center[1] - cam_radius * np.cos(angle_rad)
                cam_z = center[2] + cam_radius * 0.25  # slight top-down

                pl.camera.position = (cam_x, cam_y, cam_z)
                pl.camera.focal_point = tuple(center)
                pl.camera.up = (0, 0, 1)

                # Timestep label
                pl.add_text(
                    f"TR {t}  |  {t}s",
                    position="upper_left", font_size=14, color="white",
                )
                pl.screenshot(str(frame_path))
                pl.close()
            frame_idx += 1

        if progress_cb:
            elapsed = time.time() - t0
            eta = elapsed / (t + 1) * (n_timesteps - t - 1)
            progress_cb(
                (t + 1) / n_timesteps,
                desc=f"Rendering TR {t+1}/{n_timesteps} "
                     f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
            )

    # Stitch frames into MP4 with ffmpeg
    sp.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(anim_dir / "anim_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        str(out_path),
    ], capture_output=True)

    # Clean up PNGs
    for f in anim_dir.glob("anim_*.png"):
        f.unlink()
    if anim_dir.exists():
        anim_dir.rmdir()

    if out_path.exists():
        print(f"Brain animation saved to {out_path} ({frame_idx} frames)")
        return str(out_path)
    return None


def generate_animation(progress=gr.Progress()):
    """Gradio callback for the Generate Animation button."""
    if _LAST_PREDS is None:
        return None, None, "\u26a0\ufe0f Run inference first."

    input_vid = _LAST_VIDEO  # show input video alongside

    # Check if already rendered
    cached = RESULTS_DIR / "brain_animation.mp4"
    if cached.exists():
        return input_vid, str(cached), "\u2705 Animation loaded from cache."

    try:
        n_ts = _LAST_PREDS.shape[0]
        progress(0, desc=f"Rendering {n_ts}s brain animation at 24fps ({n_ts * 24} frames)...")
        video_path = render_brain_animation(
            _LAST_PREDS, fps=24, tr=1.0,
            progress_cb=progress,
        )
        if video_path:
            return input_vid, video_path, "\u2705 Animation complete! Play both videos to compare."
        return input_vid, None, "\u274c Failed to generate animation \u2014 ffmpeg may have failed."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return input_vid, None, f"\u274c **Error:** `{type(e).__name__}: {e}`"


def save_results(preds: np.ndarray, input_desc: str, n_events: int,
                 elapsed: float, mean_img: str, ts_img: str,
                 video_path: str | None = None) -> None:
    """Persist predictions + metadata + rendered images to disk."""
    global _FRAME_PATHS, _BRAIN_FRAMES
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / "predictions.npy", preds)

    # Pre-render brain frames for smooth playback
    print("Pre-rendering brain frames for smooth playback...")
    _BRAIN_FRAMES = prerender_brain_frames(preds)

    # Copy rendered images into results dir
    for src, dst_name in [(mean_img, "mean_map.png"), (ts_img, "timestep_map.png")]:
        if src and Path(src).exists():
            shutil.copy2(src, RESULTS_DIR / dst_name)

    # Copy video and extract frames
    if video_path and Path(video_path).exists():
        suffix = Path(video_path).suffix
        saved_video = RESULTS_DIR / f"input_video{suffix}"
        shutil.copy2(video_path, saved_video)
        print("Extracting video frames for synced playback...")
        _FRAME_PATHS = extract_video_frames(str(saved_video), preds.shape[0])
        print(f"Extracted {len(_FRAME_PATHS)} frames.")

    meta = {
        "input_desc": input_desc,
        "n_events": n_events,
        "n_timesteps": int(preds.shape[0]),
        "n_vertices": int(preds.shape[1]),
        "activation_min": float(preds.min()),
        "activation_max": float(preds.max()),
        "elapsed": elapsed,
        "video_file": f"input_video{Path(video_path).suffix}" if video_path else None,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (RESULTS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Results saved to {RESULTS_DIR}")


def load_cached_results() -> dict | None:
    """Load previously saved results from disk. Returns None if not found."""
    global _LAST_PREDS, _LAST_VIDEO, _FRAME_PATHS, _BRAIN_FRAMES
    npy_path = RESULTS_DIR / "predictions.npy"
    meta_path = RESULTS_DIR / "metadata.json"
    if not npy_path.exists() or not meta_path.exists():
        return None

    try:
        preds = np.load(npy_path)
        meta = json.loads(meta_path.read_text())
        _LAST_PREDS = preds

        # Restore video path if saved
        video_path = None
        if meta.get("video_file"):
            vpath = RESULTS_DIR / meta["video_file"]
            if vpath.exists():
                _LAST_VIDEO = str(vpath)
                video_path = str(vpath)

        # Restore or re-extract video frames
        frames_dir = RESULTS_DIR / "frames"
        if frames_dir.exists() and any(frames_dir.glob("frame_*.jpg")):
            _FRAME_PATHS = sorted([str(f) for f in frames_dir.glob("frame_*.jpg")])
        elif video_path:
            print("Re-extracting video frames...")
            _FRAME_PATHS = extract_video_frames(video_path, preds.shape[0])
        else:
            _FRAME_PATHS = []

        # Restore or re-render brain frames
        brain_dir = RESULTS_DIR / "brain_frames"
        if brain_dir.exists() and any(brain_dir.glob("brain_*.png")):
            _BRAIN_FRAMES = sorted([str(f) for f in brain_dir.glob("brain_*.png")])
        else:
            print("Pre-rendering brain frames...")
            _BRAIN_FRAMES = prerender_brain_frames(preds)

        # Restore rendered images
        mean_img = str(RESULTS_DIR / "mean_map.png") if (RESULTS_DIR / "mean_map.png").exists() else None
        ts_img = str(RESULTS_DIR / "timestep_map.png") if (RESULTS_DIR / "timestep_map.png").exists() else None

        # Initial brain frame (mean = index 0 in _BRAIN_FRAMES)
        init_brain_img = _BRAIN_FRAMES[0] if _BRAIN_FRAMES else None

        # First video frame for initial display
        init_frame = _FRAME_PATHS[0] if _FRAME_PATHS else None

        summary_text = (
            f"### Restored results  *(from {meta.get('saved_at', 'unknown')})*\n\n"
            f"| | |\n|---|---|\n"
            f"| **Input** | {meta['input_desc']} |\n"
            f"| **Events** | {meta['n_events']} extracted |\n"
            f"| **Predictions** | {meta['n_timesteps']} timesteps \u00d7 "
            f"{meta['n_vertices']:,} vertices |\n"
            f"| **Activation range** | [{meta['activation_min']:.4f}, "
            f"{meta['activation_max']:.4f}] |\n"
            f"| **Original runtime** | {meta['elapsed']:.0f}s |\n\n"
            f"*Loaded from cache \u2014 no recomputation needed. "
            f"Use the **Timestep** slider or **\u25b6 Play** to watch synced playback.*"
        )

        print(f"Restored cached results: {meta['n_timesteps']} timesteps from {meta.get('saved_at')}")
        return {
            "preds": preds,
            "mean_img": mean_img,
            "ts_img": ts_img,
            "init_brain_img": init_brain_img,
            "summary": summary_text,
            "meta": meta,
            "init_frame": init_frame,
            "video_path": video_path,
        }
    except Exception as e:
        print(f"Failed to load cached results: {e}")
        return None


def load_model():
    global MODEL, FSAVERAGE
    from tribev2 import TribeModel

    config_update = None
    if not torch.cuda.is_available():
        config_update = {
            "data.text_feature.device": "cpu",
            "data.audio_feature.device": "cpu",
            "data.video_feature.image.device": "cpu",
            "data.image_feature.image.device": "cpu",
            "data.num_workers": 2,
        }
    MODEL = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder="./cache", config_update=config_update
    )
    FSAVERAGE = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    # Pre-load and cache the combined mesh for the interactive viewer
    _build_mesh_cache()

    # Load Yeo 7-network atlas for cognitive analysis
    _load_yeo_labels()


def _build_mesh_cache():
    """Load both hemispheres, offset the right, and cache for plotly."""
    global _MESH_CACHE
    fs = FSAVERAGE

    lh_data = surface.load_surf_mesh(fs["pial_left"])
    rh_data = surface.load_surf_mesh(fs["pial_right"])

    # Handle tuple vs object return types across nilearn versions
    if isinstance(lh_data, tuple):
        lh_coords, lh_faces = lh_data
        rh_coords, rh_faces = rh_data
    else:
        lh_coords, lh_faces = np.asarray(lh_data[0]), np.asarray(lh_data[1])
        rh_coords, rh_faces = np.asarray(rh_data[0]), np.asarray(rh_data[1])

    # Offset right hemisphere just enough so they sit together like a real brain
    rh_shifted = rh_coords.copy()
    rh_shifted[:, 0] += 2

    n_lh = len(lh_coords)
    coords = np.vstack([lh_coords, rh_shifted])
    faces = np.vstack([lh_faces, rh_faces + n_lh])

    _MESH_CACHE = {
        "x": np.round(coords[:, 0], 2).tolist(),
        "y": np.round(coords[:, 1], 2).tolist(),
        "z": np.round(coords[:, 2], 2).tolist(),
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
    }


# ── Yeo Atlas Loading ────────────────────────────────────────────────
def _load_yeo_labels() -> np.ndarray:
    """Load or compute per-vertex Yeo 7-network labels on fsaverage5."""
    global _YEO_LABELS
    cache_path = RESULTS_DIR / "yeo7_labels.npy"

    if cache_path.exists():
        _YEO_LABELS = np.load(cache_path).astype(int).ravel()
        print(f"Loaded cached Yeo labels: {(_YEO_LABELS > 0).sum()} labelled vertices")
        return _YEO_LABELS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Projecting Yeo 2011 7-network atlas to fsaverage5...")
    yeo_atlas = datasets.fetch_atlas_yeo_2011()

    # Project volumetric atlas to surface per hemisphere
    from nilearn.surface import vol_to_surf
    labels_parts = []
    for hemi in ["left", "right"]:
        proj = vol_to_surf(
            yeo_atlas["maps"],
            FSAVERAGE[f"pial_{hemi}"],
            interpolation="nearest_most_frequent",
        )
        labels_parts.append(np.round(proj).astype(int))

    _YEO_LABELS = np.concatenate(labels_parts).ravel()  # ensure (20484,)
    np.save(cache_path, _YEO_LABELS)
    print(f"Yeo labels computed and cached: {(_YEO_LABELS > 0).sum()} labelled vertices")
    return _YEO_LABELS


# ── Network Analysis ────────────────────────────────────────────────
def compute_network_timecourses(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute mean activation per Yeo network per timestep.

    Returns (n_timesteps, 7) array. Network index 0 = Visual, ..., 6 = Default Mode.
    """
    n_timesteps = preds.shape[0]
    timecourses = np.zeros((n_timesteps, 7))
    for net_idx in range(7):
        mask = labels == (net_idx + 1)  # Yeo labels are 1-indexed; 0 = unassigned
        if mask.any():
            timecourses[:, net_idx] = preds[:, mask].mean(axis=1)
    return timecourses


def find_robust_peaks(timecourse: np.ndarray, n_peaks: int = 3,
                      boundary_trim: int = 4, min_distance: int = 3) -> np.ndarray:
    """Find peaks in a timecourse with detrending and boundary trimming.

    Addresses end-of-video bias caused by:
    - Linear drift in model predictions (removed via detrending)
    - Conv1d zero-padding edge artifacts (removed via boundary trimming)
    - Adjacent timesteps dominating top-N (mitigated via minimum distance)

    Args:
        timecourse: 1D array of network activation values.
        n_peaks: Number of peaks to return.
        boundary_trim: Number of timesteps to exclude from each end
                       (matches TemporalSmoothing kernel_size//2 = 4).
        min_distance: Minimum distance between returned peaks.

    Returns:
        Array of peak timestep indices (in original timecourse coordinates),
        sorted by descending magnitude.
    """
    from scipy.signal import detrend, find_peaks as scipy_find_peaks

    n = len(timecourse)
    if n == 0:
        return np.array([], dtype=int)

    # Clamp boundary trim so we don't eliminate the whole signal
    trim = min(boundary_trim, max(n // 4, 1))

    # 1. Detrend: remove linear drift that biases toward start/end
    detrended = detrend(timecourse, type='linear')

    # 2. Z-score: normalize so peaks are relative to video's own distribution
    std = detrended.std()
    if std > 0:
        z = detrended / std
    else:
        z = detrended

    # 3. Trim boundaries to avoid Conv1d zero-padding artifacts
    interior = np.abs(z[trim:n - trim])

    # 4. Use scipy find_peaks with prominence for robust detection
    scipy_peaks, properties = scipy_find_peaks(interior, distance=min_distance,
                                                prominence=0.1)

    if len(scipy_peaks) >= n_peaks:
        # Sort by prominence (most prominent first)
        top_idx = np.argsort(properties['prominences'])[::-1][:n_peaks]
        peak_indices = scipy_peaks[top_idx] + trim
    else:
        # Fallback: top-N by absolute z-scored magnitude with spacing
        ranked = np.argsort(interior)[::-1]
        selected = []
        for idx in ranked:
            original_idx = idx + trim
            if all(abs(original_idx - s) >= min_distance for s in selected):
                selected.append(original_idx)
            if len(selected) == n_peaks:
                break
        peak_indices = np.array(selected, dtype=int)

    # Sort by magnitude (descending)
    if len(peak_indices) > 0:
        magnitudes = np.abs(z[peak_indices])
        peak_indices = peak_indices[np.argsort(magnitudes)[::-1]]

    return peak_indices


def interpret_cognitive_state(timecourses: np.ndarray) -> list[dict]:
    """Derive cognitive state interpretations from network activations.

    Returns one dict per timestep with dominant network and interpretation.
    """
    results = []
    for t in range(timecourses.shape[0]):
        activations = timecourses[t]
        # Use absolute values for dominance (both + and - activations are meaningful)
        abs_act = np.abs(activations)
        ranked = np.argsort(abs_act)[::-1]
        dominant = ranked[0]
        secondary = ranked[1]

        short_label, long_desc = COGNITIVE_INTERPRETATIONS[dominant]

        # Detect compound state: if top-2 are within 20% of each other
        if abs_act[dominant] > 0 and abs_act[secondary] / abs_act[dominant] > 0.8:
            sec_label, _ = COGNITIVE_INTERPRETATIONS[secondary]
            short_label = f"{short_label} + {sec_label}"

        results.append({
            "timestep": t,
            "dominant_network": YEO_NETWORK_NAMES[dominant],
            "dominant_index": int(dominant),
            "dominant_activation": float(activations[dominant]),
            "interpretation": short_label,
            "description": long_desc,
            "secondary_network": YEO_NETWORK_NAMES[secondary],
            "all_activations": activations.tolist(),
        })
    return results


# ── Cognitive Visualizations ─────────────────────────────────────────
def plot_network_timecourses(timecourses: np.ndarray) -> go.Figure:
    """Plotly line chart: 7 network activations over time."""
    fig = go.Figure()
    for i in range(7):
        fig.add_trace(go.Scatter(
            x=list(range(timecourses.shape[0])),
            y=timecourses[:, i],
            mode="lines",
            name=YEO_NETWORK_NAMES[i],
            line=dict(color=YEO_NETWORK_COLORS[i], width=2.5),
        ))
    fig.update_layout(
        title=dict(text="Predicted Network Activation Over Time", x=0.5),
        xaxis_title="Time (TR = 1 second)",
        yaxis_title="Mean Predicted Activation",
        height=400,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5,
        ),
        margin=dict(l=60, r=20, t=50, b=100),
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        annotations=[dict(
            text="Based on TRIBE v2 predicted brain activations (Yeo 2011 7-network parcellation)",
            xref="paper", yref="paper", x=0.5, y=-0.5,
            showarrow=False, font=dict(size=10, color="gray"),
        )],
    )
    return fig


def plot_network_bar(timecourses: np.ndarray, timestep: int = -1) -> go.Figure:
    """Plotly bar chart: network activations for a single timestep."""
    if timestep < 0:
        values = timecourses.mean(axis=0)
        title = "Mean Network Activation (all TRs)"
    else:
        values = timecourses[min(timestep, len(timecourses) - 1)]
        title = f"Network Activation at TR {timestep}"

    fig = go.Figure(data=[go.Bar(
        x=YEO_NETWORK_NAMES,
        y=values,
        marker_color=YEO_NETWORK_COLORS,
    )])
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=13)),
        yaxis_title="Predicted Activation",
        height=350,
        margin=dict(l=50, r=20, t=50, b=80),
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        xaxis_tickangle=-30,
    )
    return fig


def generate_cognitive_summary(timecourses: np.ndarray,
                               interpretations: list[dict]) -> str:
    """Markdown summary of cognitive analysis results."""
    n_ts = timecourses.shape[0]

    # Peak activations per network (using robust peak detection)
    peak_rows = []
    for i in range(7):
        robust = find_robust_peaks(timecourses[:, i], n_peaks=1)
        peak_tr = int(robust[0]) if len(robust) > 0 else int(np.argmax(np.abs(timecourses[:, i])))
        peak_val = timecourses[peak_tr, i]
        peak_rows.append(
            f"| {YEO_NETWORK_NAMES[i]} | TR {peak_tr} | {peak_val:+.4f} |"
        )
    peaks_table = "\n".join(peak_rows)

    # Interpretation timeline (up to 20 rows)
    timeline_rows = []
    for interp in interpretations[:20]:
        t = interp["timestep"]
        dom = interp["dominant_network"]
        label = interp["interpretation"]
        act = interp["dominant_activation"]
        timeline_rows.append(f"| TR {t} | {dom} | {act:+.4f} | {label} |")
    if len(interpretations) > 20:
        timeline_rows.append(f"| ... | *{len(interpretations) - 20} more rows* | | |")
    timeline_table = "\n".join(timeline_rows)

    # Overall dominant network
    mean_abs = np.abs(timecourses).mean(axis=0)
    overall_dom = int(np.argmax(mean_abs))
    overall_name = YEO_NETWORK_NAMES[overall_dom]
    _, overall_desc = COGNITIVE_INTERPRETATIONS[overall_dom]

    return f"""### Cognitive Analysis Results

> **Note:** These results are based on *predicted* brain responses from TRIBE v2,
> not measured fMRI data. Interpretations reflect estimated neural activity patterns
> associated with the Yeo 2011 7-network parcellation
> ([Yeo et al., 2011](https://doi.org/10.1152/jn.00338.2011)).

#### Overall Dominant Network: **{overall_name}**
{overall_desc}

#### Peak Network Activations
| Network | Peak TR | Activation |
|---------|---------|------------|
{peaks_table}

#### Cognitive State Timeline
| Time | Dominant Network | Activation | Interpretation |
|------|-----------------|------------|----------------|
{timeline_table}
"""


def generate_narrative_abstract(timecourses: np.ndarray,
                                interpretations: list[dict],
                                input_desc: str = "stimulus") -> str:
    """Generate a plain-English narrative abstract of the brain response analysis.

    This creates a readable summary that a non-neuroscientist can understand,
    describing what the brain is predicted to be doing over time.
    """
    n_ts = timecourses.shape[0]
    mean_act = timecourses.mean(axis=0)
    mean_abs = np.abs(timecourses).mean(axis=0)

    # Rank networks by overall engagement
    ranked = np.argsort(mean_abs)[::-1]
    top3 = ranked[:3]

    # Find temporal phases — group consecutive timesteps with same dominant network
    phases = []
    current_dom = interpretations[0]["dominant_index"]
    phase_start = 0
    for t in range(1, n_ts):
        dom = interpretations[t]["dominant_index"]
        if dom != current_dom or t == n_ts - 1:
            end_t = t if dom != current_dom else t + 1
            dur = end_t - phase_start
            phases.append({
                "start": phase_start,
                "end": end_t - 1,
                "duration": dur,
                "network": YEO_NETWORK_NAMES[current_dom],
                "network_idx": current_dom,
                "label": COGNITIVE_INTERPRETATIONS[current_dom][0],
            })
            current_dom = dom
            phase_start = t

    # Compute engagement percentages
    total_engagement = mean_abs.sum()
    pct = (mean_abs / total_engagement * 100) if total_engagement > 0 else np.zeros(7)

    # Find moments of peak activity
    peak_t = int(np.argmax(np.abs(timecourses).max(axis=1)))
    peak_net = int(np.argmax(np.abs(timecourses[peak_t])))

    # Find most variable network (most dynamic)
    variability = timecourses.std(axis=0)
    most_dynamic = int(np.argmax(variability))

    # Build the narrative
    lines = []
    lines.append("## Brain Response Abstract")
    lines.append("")
    lines.append(
        f"This analysis examines predicted brain network responses to "
        f"**{input_desc}** across **{n_ts} seconds** of stimulus, using "
        f"Meta's TRIBE v2 brain encoding model mapped onto the Yeo 2011 "
        f"7-network cortical parcellation."
    )
    lines.append("")

    # Overall profile
    lines.append("### Overall Neural Profile")
    lines.append("")
    top_names = [YEO_NETWORK_NAMES[i] for i in top3]
    top_pcts = [pct[i] for i in top3]
    lines.append(
        f"The predicted brain response is dominated by the "
        f"**{top_names[0]}** network ({top_pcts[0]:.0f}% of total engagement), "
        f"followed by **{top_names[1]}** ({top_pcts[1]:.0f}%) and "
        f"**{top_names[2]}** ({top_pcts[2]:.0f}%). "
    )

    # Interpret what this profile means
    profile_interp = []
    if 0 in top3[:2]:  # Visual in top 2
        profile_interp.append(
            "strong visual processing, suggesting the stimulus contains "
            "rich visual content that engages early and higher visual areas"
        )
    if 4 in top3[:2]:  # Limbic in top 2
        profile_interp.append(
            "notable limbic/emotional engagement, suggesting the content "
            "evokes affective or reward-related responses"
        )
    if 6 in top3[:2]:  # Default mode in top 2
        profile_interp.append(
            "significant default mode network activity, suggesting engagement "
            "of semantic processing, narrative comprehension, or self-referential thought"
        )
    if 2 in top3[:2]:  # Dorsal attention in top 2
        profile_interp.append(
            "strong directed attention, suggesting the viewer is actively "
            "tracking or focusing on specific elements in the stimulus"
        )
    if 3 in top3[:2]:  # Ventral attention in top 2
        profile_interp.append(
            "elevated salience detection, suggesting the stimulus contains "
            "unexpected or behaviorally relevant events"
        )
    if 5 in top3[:2]:  # Frontoparietal in top 2
        profile_interp.append(
            "frontoparietal control network engagement, suggesting "
            "higher-order cognitive processing or decision-making"
        )
    if 1 in top3[:2]:  # Somatomotor in top 2
        profile_interp.append(
            "somatomotor activation, suggesting processing of bodily "
            "sensations, movement, or tactile information in the stimulus"
        )

    if profile_interp:
        lines.append("This profile indicates " + "; and ".join(profile_interp) + ".")
    lines.append("")

    # Temporal dynamics
    lines.append("### Temporal Dynamics")
    lines.append("")
    if len(phases) <= 3:
        lines.append(
            f"The brain response is relatively **stable** across the stimulus, "
            f"with {len(phases)} distinct phase(s):"
        )
    else:
        lines.append(
            f"The brain response shows **dynamic shifts** across the stimulus, "
            f"with {len(phases)} distinct phases:"
        )
    lines.append("")

    for i, phase in enumerate(phases[:8]):
        time_range = f"{phase['start']}–{phase['end']}s" if phase['duration'] > 1 else f"{phase['start']}s"
        lines.append(
            f"- **{time_range}**: {phase['label']} "
            f"(*{phase['network']}* network dominant)"
        )
    if len(phases) > 8:
        lines.append(f"- *(plus {len(phases) - 8} additional transitions)*")
    lines.append("")

    # Peak moment
    lines.append("### Key Moments")
    lines.append("")
    peak_label = COGNITIVE_INTERPRETATIONS[peak_net][0]
    lines.append(
        f"The **strongest predicted brain response** occurs at "
        f"**TR {peak_t}** ({peak_t} seconds), driven by the "
        f"**{YEO_NETWORK_NAMES[peak_net]}** network ({peak_label}). "
        f"The most **temporally variable** network is "
        f"**{YEO_NETWORK_NAMES[most_dynamic]}**, showing the greatest "
        f"fluctuation in predicted activity across the stimulus duration."
    )
    lines.append("")

    # Engagement breakdown
    lines.append("### Network Engagement Breakdown")
    lines.append("")
    lines.append("| Network | Engagement | Mean Activation |")
    lines.append("|---------|-----------|-----------------|")
    for i in ranked:
        bar_len = int(pct[i] / 5)  # rough visual bar
        bar = "\u2588" * bar_len
        lines.append(
            f"| {YEO_NETWORK_NAMES[i]} | {bar} {pct[i]:.1f}% | {mean_act[i]:+.4f} |"
        )
    lines.append("")

    # Caveats
    lines.append("---")
    lines.append(
        "*This abstract was generated from TRIBE v2 predicted brain responses, "
        "not measured fMRI data. Network labels follow the Yeo 2011 7-network "
        "parcellation. Activation values reflect predicted BOLD signal magnitude, "
        "where positive values indicate above-baseline activity and negative values "
        "indicate below-baseline activity. Individual variability in brain responses "
        "is not captured by this model, which predicts the average brain's response "
        "to the given stimulus.*"
    )

    return "\n".join(lines)


def _build_ai_prompt(timecourses: np.ndarray, interpretations: list[dict],
                     input_desc: str) -> str:
    """Build the data context to send to Claude for interpretation."""
    n_ts = timecourses.shape[0]
    mean_act = timecourses.mean(axis=0)
    mean_abs = np.abs(timecourses).mean(axis=0)
    ranked = np.argsort(mean_abs)[::-1]
    total = mean_abs.sum()
    pct = (mean_abs / total * 100) if total > 0 else np.zeros(7)

    # Network engagement summary
    engagement = "\n".join([
        f"  - {YEO_NETWORK_NAMES[i]}: {pct[i]:.1f}% engagement, mean activation {mean_act[i]:+.4f}"
        for i in ranked
    ])

    # Temporal timeline
    timeline = "\n".join([
        f"  - TR {interp['timestep']} ({interp['timestep']}s): "
        f"Dominant = {interp['dominant_network']} ({interp['dominant_activation']:+.4f}), "
        f"Secondary = {interp['secondary_network']}, "
        f"Interpretation = {interp['interpretation']}"
        for interp in interpretations
    ])

    # Peak moments per network
    peaks = "\n".join([
        f"  - {YEO_NETWORK_NAMES[i]}: peak at TR {int(np.argmax(np.abs(timecourses[:, i])))} "
        f"(activation {timecourses[int(np.argmax(np.abs(timecourses[:, i]))), i]:+.4f})"
        for i in range(7)
    ])

    return f"""You are a neuroscience communicator. Analyze the following predicted brain response data and write a clear, insightful abstract that a non-neuroscientist can understand. Be specific about what the brain activity patterns suggest about how a person would process this stimulus.

INPUT: {input_desc}
DURATION: {n_ts} seconds ({n_ts} TRs at 1 second each)

BRAIN NETWORK ENGAGEMENT (ranked by overall activity):
{engagement}

TEMPORAL TIMELINE (second-by-second dominant brain network):
{timeline}

PEAK ACTIVATIONS PER NETWORK:
{peaks}

NETWORK REFERENCE:
- Visual: processes visual features (color, shape, motion, faces, scenes)
- Somatomotor: bodily sensations, motor representations
- Dorsal Attention: voluntary focused attention, spatial tracking
- Ventral Attention: salience detection, surprising/important stimuli
- Limbic: emotion, affect, reward, memory encoding
- Frontoparietal: cognitive control, reasoning, working memory
- Default Mode: narrative comprehension, self-referential thought, semantic processing

Write an abstract (3-4 paragraphs) that:
1. Opens with a one-sentence summary of the overall brain response character
2. Describes the temporal arc — how the brain's processing evolves over time, and what this suggests about the viewer's experience
3. Highlights the most interesting or notable patterns (e.g., sudden shifts, sustained engagement, co-activation of networks)
4. Ends with a brief caveat that these are model-predicted responses, not measured fMRI

Use natural language. Avoid jargon where possible. When you do use network names, briefly explain what they do. Be specific — reference actual timepoints and values. Write as if for an educated general audience."""


def on_generate_abstract(progress=gr.Progress()):
    """Call Claude API to generate a plain-English brain response abstract."""
    if _NETWORK_TIMECOURSES is None or _LAST_PREDS is None:
        return "\u26a0\ufe0f Run inference first to generate brain predictions."

    # Get input description from metadata
    meta_path = RESULTS_DIR / "metadata.json"
    input_desc = "the stimulus"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            input_desc = meta.get("input_desc", "the stimulus")
        except Exception:
            pass

    interps = interpret_cognitive_state(_NETWORK_TIMECOURSES)
    prompt = _build_ai_prompt(_NETWORK_TIMECOURSES, interps, input_desc)

    # Try Claude API first
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            # Check for .env file
            env_path = Path(__file__).resolve().parent / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("ANTHROPIC_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")

        if not api_key:
            return (
                "\u26a0\ufe0f **Anthropic API key not found.** "
                "Set `ANTHROPIC_API_KEY` in your environment or create a `.env` file "
                "in the project root with:\n\n"
                "```\nANTHROPIC_API_KEY=sk-ant-...\n```\n\n"
                "Then restart the server."
            )

        progress(0.2, desc="Sending brain data to Claude...")
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        progress(1.0, desc="Abstract generated!")

        abstract_text = response.content[0].text
        return f"## \U0001f9e0 AI Brain Response Abstract\n\n{abstract_text}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Fall back to template-based narrative
        fallback = generate_narrative_abstract(_NETWORK_TIMECOURSES, interps, input_desc)
        return (
            f"> \u26a0\ufe0f Claude API call failed (`{type(e).__name__}: {e}`). "
            f"Showing template-based analysis instead.\n\n{fallback}"
        )


def plot_yeo_parcellation() -> str:
    """Render and cache a Yeo 7-network atlas visualization on the brain."""
    cache_path = RESULTS_DIR / "yeo_parcellation.png"
    if cache_path.exists():
        return str(cache_path)

    from matplotlib.colors import ListedColormap

    # Build colormap: index 0 = gray (unassigned), 1-7 = network colors
    cmap_colors = ["#888888"] + YEO_NETWORK_COLORS
    cmap = ListedColormap(cmap_colors)

    fig_mpl, axes = plt.subplots(1, 2, figsize=(12, 5),
                                  subplot_kw={"projection": "3d"})
    fig_mpl.patch.set_facecolor("white")
    lh = 10242

    for ax, hemi in zip(axes, ["left", "right"]):
        labels = _YEO_LABELS[:lh] if hemi == "left" else _YEO_LABELS[lh:]
        plotting.plot_surf_roi(
            FSAVERAGE[f"pial_{hemi}"],
            roi_map=labels,
            hemi=hemi, view="lateral",
            bg_map=FSAVERAGE[f"sulc_{hemi}"],
            axes=ax, cmap=cmap,
            vmin=0, vmax=7,
        )
        ax.set_facecolor("white")
        ax.set_title(f"{hemi.capitalize()} Hemisphere",
                     fontsize=13, fontweight="bold")

    # Legend
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=YEO_NETWORK_COLORS[i], label=YEO_NETWORK_NAMES[i])
        for i in range(7)
    ]
    fig_mpl.legend(handles=legend_patches, loc="lower center", ncol=4,
                   fontsize=10, frameon=False)
    fig_mpl.suptitle("Yeo 2011 7-Network Parcellation", fontsize=14,
                     fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig_mpl.savefig(cache_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_mpl)
    print(f"Yeo parcellation saved to {cache_path}")
    return str(cache_path)


# ── Visualization helpers ────────────────────────────────────────────
def _get_cmap():
    """Return the nilearn 'cold_hot' cmap, falling back to RdBu_r."""
    try:
        return plt.colormaps["cold_hot"]
    except (KeyError, AttributeError):
        return plt.colormaps.get("RdBu_r", plt.cm.RdBu_r)


def render_brain_map(preds: np.ndarray) -> tuple[str, str]:
    """Improved static brain maps -> (mean_map_path, timestep_montage_path)."""
    n_timesteps, n_vertices = preds.shape
    lh = n_vertices // 2
    fs = FSAVERAGE
    mean_pred = preds.mean(axis=0)
    vmax = max(abs(float(mean_pred.min())), abs(float(mean_pred.max())))
    cmap = _get_cmap()

    # ── Mean activation map (4 views) ────────────────────────────────
    fig, axes = plt.subplots(
        1, 4, figsize=(24, 6), subplot_kw={"projection": "3d"}
    )
    fig.patch.set_facecolor("white")

    for ax, (hemi, view) in zip(
        axes,
        [("left", "lateral"), ("left", "medial"),
         ("right", "lateral"), ("right", "medial")],
    ):
        data = mean_pred[:lh] if hemi == "left" else mean_pred[lh:]
        plotting.plot_surf_stat_map(
            fs[f"pial_{hemi}"], data, hemi=hemi, view=view,
            colorbar=False, bg_map=fs[f"sulc_{hemi}"],
            axes=ax, threshold=0.01, vmax=vmax,
        )
        ax.set_title(
            f"{hemi.capitalize()} {view.capitalize()}",
            fontsize=14, fontweight="bold", pad=10,
        )
        ax.set_facecolor("white")

    # Shared horizontal colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes, orientation="horizontal",
        fraction=0.04, pad=0.08, shrink=0.5, aspect=40,
    )
    cbar.set_label("Predicted Activation", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        f"Mean Predicted Brain Response  \u00b7  {n_timesteps} TRs",
        fontsize=16, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    mean_path = tempfile.mktemp(suffix=".png")
    fig.savefig(mean_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ── Per-timestep montage (up to 8) ───────────────────────────────
    n_show = min(8, n_timesteps)
    ts_vmax = max(abs(float(preds[:n_show].min())), abs(float(preds[:n_show].max())))

    fig2, axes2 = plt.subplots(
        n_show, 2, figsize=(12, 3.5 * n_show),
        subplot_kw={"projection": "3d"},
    )
    fig2.patch.set_facecolor("white")
    if n_show == 1:
        axes2 = axes2.reshape(1, -1)

    for t in range(n_show):
        for col, hemi in enumerate(["left", "right"]):
            data = preds[t, :lh] if hemi == "left" else preds[t, lh:]
            plotting.plot_surf_stat_map(
                fs[f"pial_{hemi}"], data, hemi=hemi, view="lateral",
                colorbar=False, bg_map=fs[f"sulc_{hemi}"],
                axes=axes2[t, col], threshold=0.01, vmax=ts_vmax,
            )
            axes2[t, col].set_facecolor("white")
            if t == 0:
                axes2[t, col].set_title(
                    f"{hemi.capitalize()} Hemisphere",
                    fontsize=13, fontweight="bold", pad=8,
                )
        # Timestep label on the left
        axes2[t, 0].text2D(
            -0.12, 0.5, f"TR {t}",
            transform=axes2[t, 0].transAxes,
            fontsize=13, fontweight="bold", va="center", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8e8e8", edgecolor="none"),
        )

    # Shared colorbar
    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-ts_vmax, ts_vmax))
    sm2.set_array([])
    cbar2 = fig2.colorbar(
        sm2, ax=axes2, orientation="horizontal",
        fraction=0.03, pad=0.04, shrink=0.5,
    )
    cbar2.set_label("Predicted Activation", fontsize=11, fontweight="bold")

    fig2.suptitle(
        "Per-Timestep Brain Predictions",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=[0.06, 0.03, 1, 0.98])

    ts_path = tempfile.mktemp(suffix=".png")
    fig2.savefig(ts_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    return mean_path, ts_path


def render_interactive_brain(preds: np.ndarray, timestep: int = -1) -> go.Figure:
    """Interactive 3D brain for a single timestep (or mean if timestep == -1)."""
    n_timesteps, n_vertices = preds.shape
    abs_max = float(max(abs(preds.min()), abs(preds.max())))

    if timestep < 0:
        intensity = np.round(preds.mean(axis=0), 5).tolist()
        title_text = f"Mean activation ({n_timesteps} TRs)"
    else:
        intensity = np.round(preds[timestep], 5).tolist()
        title_text = f"TR {timestep} / {n_timesteps - 1}"

    mc = _MESH_CACHE

    mesh = go.Mesh3d(
        x=mc["x"], y=mc["y"], z=mc["z"],
        i=mc["i"], j=mc["j"], k=mc["k"],
        intensity=intensity,
        intensitymode="vertex",
        colorscale="RdBu_r",
        cmin=-abs_max, cmax=abs_max,
        colorbar=dict(
            title=dict(text="Activation", font=dict(size=14)),
            thickness=18, len=0.6, x=1.02,
            tickfont=dict(size=11),
        ),
        lighting=dict(ambient=0.55, diffuse=0.7, specular=0.15, roughness=0.5),
        lightposition=dict(x=100, y=200, z=0),
        flatshading=False,
        hoverinfo="skip",
    )

    fig = go.Figure(data=[mesh])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            aspectmode="data",
            bgcolor="#fafafa",
            camera=dict(
                eye=dict(x=0, y=-2.2, z=0.5),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        title=dict(text=title_text, x=0.5, font=dict(size=15)),
        height=550,
        margin=dict(l=0, r=30, t=40, b=10),
        paper_bgcolor="white",
    )

    return fig


# ── Progress helper ─────────────────────────────────────────────────
def _status_md(steps: list[tuple[str, str]], elapsed: float | None = None) -> str:
    """Build a markdown status block from a list of (label, state) tuples."""
    icons = {"pending": "\u23f3", "running": "\u25b6\ufe0f", "done": "\u2705", "error": "\u274c"}
    lines = ["### Progress\n"]
    for label, state in steps:
        lines.append(f"{icons.get(state, '')}  **{label}** \u2014 *{state}*")
    if elapsed is not None:
        lines.append(f"\n*Elapsed: {elapsed:.0f}s*")
    return "\n\n".join(lines)


# ── Timestep slider callback ────────────────────────────────────────
def on_timestep_change(ts: int):
    """Show the pre-rendered brain image and matching video frame."""
    ts = int(ts)

    # Brain frames: index 0 = mean, 1..n = TR 0..n-1
    brain_idx = 0 if ts < 0 else ts + 1
    if _BRAIN_FRAMES and brain_idx < len(_BRAIN_FRAMES):
        brain_img = _BRAIN_FRAMES[brain_idx]
    else:
        brain_img = None

    # Video frame (ts=-1 shows first frame)
    frame_idx = max(0, ts)
    if _FRAME_PATHS and frame_idx < len(_FRAME_PATHS):
        frame_img = _FRAME_PATHS[frame_idx]
    else:
        frame_img = None

    return brain_img, frame_img


def on_play_tick(is_playing: bool, current_ts: int, max_ts: int):
    """Auto-advance the slider by 1 step when playing."""
    if not is_playing or _LAST_PREDS is None:
        return current_ts
    next_ts = current_ts + 1
    if next_ts > max_ts:
        next_ts = 0  # loop back
    return next_ts


# ── Inference entry point ────────────────────────────────────────────
def run_inference(video, audio, text, progress=gr.Progress(track_tqdm=True)):
    global _LAST_PREDS, _LAST_VIDEO

    if MODEL is None:
        yield {summary_output: "\u26a0\ufe0f Model not loaded yet \u2014 wait for startup."}
        return

    kwargs = {}
    input_desc = ""
    _LAST_VIDEO = None
    if video is not None:
        kwargs["video_path"] = video
        input_desc = f"Video: {Path(video).name}"
        _LAST_VIDEO = video
    elif audio is not None:
        kwargs["audio_path"] = audio
        input_desc = f"Audio: {Path(audio).name}"
    elif text is not None and text.strip():
        txt_path = tempfile.mktemp(suffix=".txt")
        Path(txt_path).write_text(text)
        kwargs["text_path"] = txt_path
        input_desc = f"Text: {text[:80]}..."
    else:
        yield {summary_output: "\u26a0\ufe0f Please provide a video, audio file, or text."}
        return

    t0 = time.time()
    steps = [
        ("Extracting events", "pending"),
        ("Preparing features (text, audio, video)", "pending"),
        ("Running brain prediction", "pending"),
        ("Rendering visualizations", "pending"),
        ("Computing cognitive analysis", "pending"),
    ]

    try:
        # Step 1: Extract events
        steps[0] = ("Extracting events", "running")
        yield {summary_output: _status_md(steps, time.time() - t0)}
        events = MODEL.get_events_dataframe(**kwargs)
        steps[0] = (f"Extracting events \u2014 {events.shape[0]} found", "done")

        # Step 2: Prepare features (this is the slow part)
        steps[1] = ("Preparing features (text, audio, video)", "running")
        yield {summary_output: _status_md(steps, time.time() - t0)}
        loader = MODEL.data.get_loaders(events=events, split_to_build="all")["all"]
        steps[1] = ("Preparing features (text, audio, video)", "done")

        # Step 3: Run prediction
        steps[2] = ("Running brain prediction", "running")
        yield {summary_output: _status_md(steps, time.time() - t0)}

        from einops import rearrange as _rearrange
        model = MODEL._model
        preds_list, all_segments = [], []
        n_samples, n_kept = 0, 0
        total_batches = len(loader)
        with torch.inference_mode():
            for batch_idx, batch in enumerate(loader):
                steps[2] = (
                    f"Running brain prediction \u2014 batch {batch_idx + 1}/{total_batches}",
                    "running",
                )
                yield {summary_output: _status_md(steps, time.time() - t0)}
                batch = batch.to(model.device)
                batch_segments = []
                for segment in batch.segments:
                    for t in np.arange(0, segment.duration - 1e-2, MODEL.data.TR):
                        batch_segments.append(
                            segment.copy(offset=t, duration=MODEL.data.TR)
                        )
                if MODEL.remove_empty_segments:
                    keep = np.array([len(s.ns_events) > 0 for s in batch_segments])
                else:
                    keep = np.ones(len(batch_segments), dtype=bool)
                n_kept += keep.sum()
                n_samples += len(batch_segments)
                batch_segments = [s for i, s in enumerate(batch_segments) if keep[i]]
                y_pred = model(batch).detach().cpu().numpy()
                y_pred = _rearrange(y_pred, "b d t -> (b t) d")[keep]
                preds_list.append(y_pred)
                all_segments.extend(batch_segments)

        preds = np.concatenate(preds_list)
        _LAST_PREDS = preds
        steps[2] = (
            f"Running brain prediction \u2014 {preds.shape[0]} timesteps predicted",
            "done",
        )

        # Step 4: Render visualizations
        steps[3] = ("Rendering visualizations", "running")
        yield {summary_output: _status_md(steps, time.time() - t0)}

        mean_img, ts_img = render_brain_map(preds)
        steps[3] = ("Rendering visualizations", "done")

        # Persist results + pre-render brain frames for smooth playback
        elapsed_render = time.time() - t0
        save_results(
            preds, input_desc, events.shape[0], elapsed_render,
            mean_img, ts_img, video_path=_LAST_VIDEO,
        )

        # Step 5: Cognitive analysis
        steps[4] = ("Computing cognitive analysis", "running")
        yield {summary_output: _status_md(steps, time.time() - t0)}

        cog_tc_plot_val = None
        cog_bar_plot_val = None
        cog_summary_val = ""
        cog_yeo_img_val = None
        cog_slider_update = gr.Slider()
        if _YEO_LABELS is not None:
            _NETWORK_TIMECOURSES = compute_network_timecourses(preds, _YEO_LABELS)
            np.save(RESULTS_DIR / "network_timecourses.npy", _NETWORK_TIMECOURSES)
            interps = interpret_cognitive_state(_NETWORK_TIMECOURSES)
            cog_tc_plot_val = plot_network_timecourses(_NETWORK_TIMECOURSES)
            cog_bar_plot_val = plot_network_bar(_NETWORK_TIMECOURSES, -1)
            cog_summary_val = generate_cognitive_summary(_NETWORK_TIMECOURSES, interps)
            cog_yeo_img_val = plot_yeo_parcellation()
            cog_slider_update = gr.Slider(
                minimum=-1, maximum=preds.shape[0] - 1, value=-1,
                step=1, interactive=True,
                label=f"Timestep (-1 = mean, 0\u2013{preds.shape[0]-1} = TRs)",
            )
        steps[4] = ("Computing cognitive analysis", "done")

        # Initial brain image = mean (index 0)
        init_brain = _BRAIN_FRAMES[0] if _BRAIN_FRAMES else None

        elapsed = time.time() - t0
        summary_text = (
            f"### Results  *(completed in {elapsed:.0f}s)*\n\n"
            f"| | |\n|---|---|\n"
            f"| **Input** | {input_desc} |\n"
            f"| **Events** | {events.shape[0]} extracted |\n"
            f"| **Predictions** | {preds.shape[0]} timesteps \u00d7 "
            f"{preds.shape[1]:,} vertices |\n"
            f"| **Activation range** | [{preds.min():.4f}, {preds.max():.4f}] |\n"
            f"| **Saved to** | `{RESULTS_DIR}` |\n\n"
            f"*Results cached \u2014 will auto-restore on server restart. "
            f"Check the **Cognitive Analysis** tab for network-level insights.*"
        )
        yield {
            mean_map: mean_img,
            timestep_map: ts_img,
            brain_image: init_brain,
            npy_download: str((RESULTS_DIR / "predictions.npy").resolve()),
            summary_output: summary_text,
            ts_slider: gr.Slider(
                minimum=-1, maximum=preds.shape[0] - 1, value=-1,
                step=1, interactive=True,
                label=f"Timestep (-1 = mean, 0\u2013{preds.shape[0]-1} = TRs)",
            ),
            network_tc_plot: cog_tc_plot_val,
            network_bar_plot: cog_bar_plot_val,
            cognitive_summary: cog_summary_val,
            yeo_brain_img: cog_yeo_img_val,
            cognitive_ts_slider: cog_slider_update,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        for i, (label, state) in enumerate(steps):
            if state == "running":
                steps[i] = (label, "error")
        yield {
            summary_output: _status_md(steps, time.time() - t0) + f"\n\n\u274c **Error:** `{type(e).__name__}: {e}`",
        }


def load_npy(npy_file):
    """Visualize a previously saved predictions .npy file."""
    global _LAST_PREDS, _BRAIN_FRAMES, _NETWORK_TIMECOURSES
    empty = (None,) * 11  # match all_outputs length
    if npy_file is None:
        return (*empty[:4], "\u26a0\ufe0f Please upload a .npy file.", *empty[5:])

    try:
        preds = np.load(npy_file)
        if preds.ndim != 2 or preds.shape[1] != 20484:
            return (*empty[:4],
                    f"\u274c **Bad shape:** expected (n_timesteps, 20484), got {preds.shape}",
                    *empty[5:])

        # Need mesh cache for the interactive viewer
        if not _MESH_CACHE:
            global FSAVERAGE
            FSAVERAGE = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
            _build_mesh_cache()

        _LAST_PREDS = preds
        mean_img, ts_img = render_brain_map(preds)
        _BRAIN_FRAMES = prerender_brain_frames(preds)
        init_brain = _BRAIN_FRAMES[0] if _BRAIN_FRAMES else None

        # Cognitive analysis
        cog_tc = cog_bar = cog_summary = cog_yeo = None
        cog_slider = gr.Slider()
        if _YEO_LABELS is not None:
            _NETWORK_TIMECOURSES = compute_network_timecourses(preds, _YEO_LABELS)
            interps = interpret_cognitive_state(_NETWORK_TIMECOURSES)
            cog_tc = plot_network_timecourses(_NETWORK_TIMECOURSES)
            cog_bar = plot_network_bar(_NETWORK_TIMECOURSES, -1)
            cog_summary = generate_cognitive_summary(_NETWORK_TIMECOURSES, interps)
            cog_yeo = plot_yeo_parcellation()
            cog_slider = gr.Slider(
                minimum=-1, maximum=preds.shape[0] - 1, value=-1,
                step=1, interactive=True,
                label=f"Timestep (-1 = mean, 0\u2013{preds.shape[0]-1} = TRs)",
            )

        summary = (
            f"### Results (loaded from file)\n\n"
            f"| | |\n|---|---|\n"
            f"| **Source** | {Path(npy_file).name} |\n"
            f"| **Predictions** | {preds.shape[0]} timesteps \u00d7 "
            f"{preds.shape[1]:,} vertices |\n"
            f"| **Activation range** | [{preds.min():.4f}, {preds.max():.4f}] |"
        )
        slider_update = gr.Slider(
            minimum=-1, maximum=preds.shape[0] - 1, value=-1,
            step=1, interactive=True,
            label=f"Timestep (-1 = mean, 0\u2013{preds.shape[0]-1} = TRs)",
        )
        # Return matches all_outputs order: mean_map, timestep_map, brain_image,
        # npy_download, summary_output, ts_slider,
        # network_tc_plot, network_bar_plot, cognitive_summary,
        # yeo_brain_img, cognitive_ts_slider
        return (mean_img, ts_img, init_brain, npy_file, summary, slider_update,
                cog_tc, cog_bar, cog_summary, cog_yeo, cog_slider)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (None, None, None, None,
                f"\u274c **Error:** `{type(e).__name__}: {e}`",
                gr.Slider(), None, None, "", None, gr.Slider())


# ── Gradio UI ────────────────────────────────────────────────────────
CUSTOM_CSS = ""


def build_ui(cached: dict | None = None):
    """Build the Gradio UI with clean layout."""
    # Register paper figures as static paths so Gradio serves them
    figures_dir = str((Path(__file__).resolve().parent / "paper" / "figures"))
    gr.set_static_paths([figures_dir])

    # ── Pre-compute initial values from cache ────────────────────────
    init_mean = cached["mean_img"] if cached else None
    init_ts = cached["ts_img"] if cached else None
    init_brain_img = cached.get("init_brain_img") if cached else None
    init_brain_3d = render_interactive_brain(cached["preds"]) if (cached and _MESH_CACHE) else None
    init_summary = cached["summary"] if cached else ""
    init_npy = str(RESULTS_DIR / "predictions.npy") if (cached and (RESULTS_DIR / "predictions.npy").exists()) else None
    init_frame = cached.get("init_frame") if cached else None
    init_input_video = cached.get("video_path") if cached else None
    init_anim_video = str(RESULTS_DIR / "brain_animation.mp4") if (RESULTS_DIR / "brain_animation.mp4").exists() else None

    # Cognitive analysis
    init_tc_plot = None
    init_bar_plot = None
    init_cog_summary = ""
    init_yeo_img = None
    if cached and _YEO_LABELS is not None:
        global _NETWORK_TIMECOURSES
        tc_cache = RESULTS_DIR / "network_timecourses.npy"
        if tc_cache.exists():
            _NETWORK_TIMECOURSES = np.load(tc_cache)
        else:
            _NETWORK_TIMECOURSES = compute_network_timecourses(cached["preds"], _YEO_LABELS)
            np.save(tc_cache, _NETWORK_TIMECOURSES)
        interps = interpret_cognitive_state(_NETWORK_TIMECOURSES)
        init_tc_plot = plot_network_timecourses(_NETWORK_TIMECOURSES)
        init_bar_plot = plot_network_bar(_NETWORK_TIMECOURSES, -1)
        init_cog_summary = generate_cognitive_summary(_NETWORK_TIMECOURSES, interps)
        init_yeo_img = plot_yeo_parcellation()

    has_data = cached is not None
    n_ts = cached["preds"].shape[0] if cached else 0
    slider_max = max(n_ts - 1, 0)
    status_text = f"\u2705 Results loaded \u00b7 {n_ts} timesteps" if has_data else "No results yet \u2014 upload input and run inference"

    # ── Build UI ─────────────────────────────────────────────────────
    with gr.Blocks(
        title="TRIBE v2 Brain Encoder",
        theme=gr.themes.Soft(),
    ) as demo:
        is_playing = gr.State(False)

        gr.Markdown(
            "# \U0001f9e0 TRIBE v2 Brain Encoder\n"
            "Predict brain responses from video, audio, or text "
            "using Meta's brain encoding foundation model."
        )

        # ── Input Section (compact top bar) ─────────────────────
        with gr.Accordion("\U0001f4e4  Upload Input", open=not has_data):
            with gr.Row(equal_height=True):
                video_input = gr.Video(label="Video", sources=["upload"], scale=2)
                audio_input = gr.Audio(label="Audio", type="filepath",
                                       sources=["upload"], scale=1)
                text_input = gr.Textbox(label="Text", lines=3, scale=2,
                                        placeholder="Or type text here...")
                with gr.Column(scale=1, min_width=140):
                    run_btn = gr.Button("\u25b6  Run Inference",
                                       variant="primary", size="lg")
                    gr.Examples(
                        examples=[["A person walks through a quiet forest."],
                                  ["Bears crossing a mountain road."]],
                        inputs=[text_input], label="Try:",
                    )
            with gr.Accordion("\U0001f4c2  Load .npy file", open=False):
                with gr.Row():
                    npy_input = gr.File(label="Upload .npy", file_types=[".npy"], scale=2)
                    load_btn = gr.Button("\U0001f4c2  Load", variant="secondary", scale=1)

        # ── Global Playback Bar ─────────────────────────────────
        with gr.Group(elem_classes=["playback-bar"]):
            with gr.Row():
                play_btn = gr.Button("\u25b6", variant="secondary",
                                     scale=0, min_width=50)
                pause_btn = gr.Button("\u23f8", variant="secondary",
                                      scale=0, min_width=50)
                ts_slider = gr.Slider(
                    minimum=-1, maximum=slider_max, value=-1, step=1,
                    label=f"Timestep  \u00b7  -1 = mean  \u00b7  0\u2013{slider_max} = seconds" if has_data else "Timestep (run inference first)",
                    interactive=has_data, scale=6,
                )

        # ── Main Content Tabs ───────────────────────────────────
        with gr.Tabs():

            # ── Tab 1: Synced Viewer ────────────────────────────
            with gr.TabItem("\U0001f3ac  Viewer"):
                with gr.Row(equal_height=True):
                    video_frame = gr.Image(label="Video Frame",
                                           value=init_frame, height=400)
                    brain_image = gr.Image(label="Predicted Brain Activity",
                                           value=init_brain_img, height=400)

            # ── Tab 2: 3D Brain ─────────────────────────────────
            with gr.TabItem("\U0001f9e0  3D Brain"):
                gr.Markdown("*Drag to rotate \u00b7 Scroll to zoom \u00b7 "
                            "Use the button below to load a specific timestep*")
                interactive_plot = gr.Plot(label="Interactive 3D Brain",
                                            value=init_brain_3d)
                with gr.Row():
                    interactive_ts = gr.Slider(
                        minimum=-1, maximum=slider_max, value=-1, step=1,
                        label="Timestep for 3D view", interactive=has_data, scale=4,
                    )
                    render_3d_btn = gr.Button("\U0001f504  Render", variant="secondary",
                                              scale=1, min_width=100)
                gr.Markdown("---")
                gr.Markdown("**Rotation Video** \u2014 generate a smooth rotating animation")
                with gr.Row():
                    anim_btn = gr.Button("\U0001f3ac  Generate Rotation Video",
                                         variant="primary")
                    anim_status = gr.Markdown("")
                with gr.Row():
                    anim_input_video = gr.Video(label="Input Video",
                                                 value=init_input_video)
                    anim_video = gr.Video(label="Brain Animation",
                                           value=init_anim_video)

            # ── Tab 3: Analysis ─────────────────────────────────
            with gr.TabItem("\U0001f4ca  Analysis"):
                gr.Markdown(
                    "> Network activations from the "
                    "[Yeo 2011 7-network parcellation]"
                    "(https://doi.org/10.1152/jn.00338.2011) "
                    "mapped onto TRIBE v2 predicted brain responses."
                )
                network_tc_plot = gr.Plot(label="Network Activation Over Time",
                                          value=init_tc_plot)
                with gr.Row():
                    with gr.Column(scale=1):
                        network_bar_plot = gr.Plot(label="Network Activation (current timestep)",
                                                    value=init_bar_plot)
                    with gr.Column(scale=1):
                        yeo_brain_img = gr.Image(label="Yeo 7-Network Atlas",
                                                  value=init_yeo_img)
                with gr.Row():
                    mean_map = gr.Image(label="Mean Activation Map",
                                        value=init_mean)
                    timestep_map = gr.Image(label="Per-Timestep Montage",
                                             value=init_ts)

            # ── Tab 4: Report ───────────────────────────────────
            with gr.TabItem("\U0001f4cb  Report"):
                summary_output = gr.Markdown(value=init_summary)
                cognitive_summary = gr.Markdown(value=init_cog_summary)
                gr.Markdown("---")
                with gr.Row():
                    abstract_btn = gr.Button(
                        "\U0001f9e0  Generate AI Abstract",
                        variant="primary", size="lg", scale=1,
                    )
                    npy_download = gr.File(
                        label="Download predictions (.npy)",
                        interactive=False, value=init_npy, scale=2,
                    )
                abstract_output = gr.Markdown(
                    value="*Click **Generate AI Abstract** for a plain-English "
                    "interpretation powered by Claude.*",
                )

            # ── Tab 5: Paper Viewer ─────────────────────────────
            with gr.TabItem("\U0001f4c4  Paper"):
                gr.Markdown("Preview markdown files from the `paper/` directory.")
                paper_dir = Path(__file__).resolve().parent / "paper"
                md_files = sorted(paper_dir.glob("*.md")) if paper_dir.exists() else []
                file_choices = [f.name for f in md_files] if md_files else ["(no .md files found)"]

                with gr.Row():
                    paper_dropdown = gr.Dropdown(
                        choices=file_choices,
                        value=file_choices[0] if md_files else None,
                        label="Select file", scale=3,
                    )
                    paper_refresh_btn = gr.Button("\U0001f504 Refresh", scale=1)
                    paper_load_btn = gr.Button("Load", variant="primary", scale=1)

                paper_viewer = gr.HTML(
                    value="<p><em>Select a file and click Load to preview.</em></p>",
                )

        # ── Hidden slider alias for cognitive bar (synced to global) ─
        # We reuse ts_slider for everything; cognitive_ts_slider is a
        # hidden dummy that run_inference can still target.
        cognitive_ts_slider = gr.Slider(visible=False, minimum=-1,
                                         maximum=slider_max, value=-1)

        # ── Timer for auto-play ──────────────────────────────────
        play_timer = gr.Timer(value=1.0, active=False)

        def start_playing():
            return True, gr.Timer(active=True)

        def stop_playing():
            return False, gr.Timer(active=False)

        def tick_forward(playing, current_ts):
            if not playing or _LAST_PREDS is None:
                return current_ts, gr.Timer(active=False)
            max_ts = _LAST_PREDS.shape[0] - 1
            next_ts = current_ts + 1
            if next_ts > max_ts:
                return max_ts, gr.Timer(active=False)
            return next_ts, gr.Timer(active=True)

        play_btn.click(fn=start_playing, outputs=[is_playing, play_timer])
        pause_btn.click(fn=stop_playing, outputs=[is_playing, play_timer])
        play_timer.tick(fn=tick_forward, inputs=[is_playing, ts_slider],
                        outputs=[ts_slider, play_timer])

        # ── Event wiring ─────────────────────────────────────────
        all_outputs = [
            mean_map, timestep_map, brain_image,
            npy_download, summary_output, ts_slider,
            network_tc_plot, network_bar_plot, cognitive_summary,
            yeo_brain_img, cognitive_ts_slider,
        ]

        run_btn.click(fn=run_inference,
                      inputs=[video_input, audio_input, text_input],
                      outputs=all_outputs)
        load_btn.click(fn=load_npy, inputs=[npy_input], outputs=all_outputs)

        # Global slider — FAST updates only (no Plotly re-render)
        def on_global_slider(ts):
            ts = int(ts)
            # Brain frame (pre-rendered PNG — instant)
            brain_idx = 0 if ts < 0 else ts + 1
            brain_img = _BRAIN_FRAMES[brain_idx] if (_BRAIN_FRAMES and brain_idx < len(_BRAIN_FRAMES)) else None
            # Video frame (cached JPG — instant)
            frame_idx = max(0, ts)
            frame_img = _FRAME_PATHS[frame_idx] if (_FRAME_PATHS and frame_idx < len(_FRAME_PATHS)) else None
            # Bar chart (lightweight plotly — fast)
            bar = plot_network_bar(_NETWORK_TIMECOURSES, ts) if _NETWORK_TIMECOURSES is not None else None
            return brain_img, frame_img, bar

        ts_slider.change(fn=on_global_slider, inputs=[ts_slider],
                         outputs=[brain_image, video_frame, network_bar_plot])

        # 3D brain — render on button click (not auto, avoids lag)
        def on_render_3d(ts):
            if _LAST_PREDS is None:
                return None
            return render_interactive_brain(_LAST_PREDS, timestep=int(ts))

        render_3d_btn.click(fn=on_render_3d, inputs=[interactive_ts],
                            outputs=[interactive_plot])

        # Abstract
        abstract_btn.click(fn=on_generate_abstract, outputs=[abstract_output])

        # Rotation animation
        anim_btn.click(fn=generate_animation,
                       outputs=[anim_input_video, anim_video, anim_status])

        # Paper viewer
        def load_paper(filename):
            if not filename:
                return "<p><em>No file selected.</em></p>"
            import re, base64, markdown
            paper_dir = Path(__file__).resolve().parent / "paper"
            p = paper_dir / filename
            if not p.exists():
                return f"<p><em>File not found: {filename}</em></p>"
            md_text = p.read_text(encoding="utf-8")

            # Step 1: Replace markdown images with base64 HTML img tags
            def replace_img(match):
                alt = match.group(1)
                rel_path = match.group(2)
                img_path = paper_dir / rel_path
                if img_path.exists():
                    b64 = base64.b64encode(img_path.read_bytes()).decode()
                    ext = img_path.suffix.lower().lstrip(".")
                    mime = {"png": "image/png", "jpg": "image/jpeg",
                            "jpeg": "image/jpeg"}.get(ext, "image/png")
                    return (
                        f'<div style="text-align:center;margin:1.5em 0;">'
                        f'<img src="data:{mime};base64,{b64}" '
                        f'alt="{alt}" style="max-width:100%;border-radius:8px;">'
                        f'<p style="font-size:0.85em;color:#888;margin-top:0.5em;">'
                        f'<em>{alt}</em></p></div>'
                    )
                return match.group(0)

            md_text = re.sub(
                r'!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|gif))\)',
                replace_img, md_text
            )

            # Step 2: Convert LaTeX math to MathML (server-side, no JS needed)
            import latex2mathml.converter

            math_blocks = []
            def save_math_block(match):
                math_blocks.append(match.group(0))
                return f"MATHPLACEHOLDER{len(math_blocks)-1}END"
            # Block math $$...$$ first
            md_text = re.sub(r'\$\$(.+?)\$\$', save_math_block, md_text, flags=re.DOTALL)
            # Inline math $...$ (not $$)
            md_text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', save_math_block, md_text)

            # Step 3: Convert markdown to HTML
            html = markdown.markdown(
                md_text,
                extensions=["tables", "fenced_code", "toc", "footnotes"],
            )

            # Step 4: Restore math as rendered MathML
            for i, math_str in enumerate(math_blocks):
                is_block = math_str.startswith("$$")
                latex = math_str.strip("$").strip()
                try:
                    mathml = latex2mathml.converter.convert(latex)
                    if is_block:
                        # Force display mode
                        mathml = mathml.replace('display="inline"', 'display="block"')
                        replacement = f'<div style="text-align:center;margin:1em 0;overflow-x:auto;">{mathml}</div>'
                    else:
                        replacement = mathml
                except Exception:
                    # Fallback: show raw LaTeX in monospace
                    replacement = f'<code>{latex}</code>'
                html = html.replace(f"MATHPLACEHOLDER{i}END", replacement)

            # Step 5: Wrap in styled container
            styled = (
                '<div style="max-width:900px;margin:0 auto;padding:2em;'
                'font-family:Georgia,serif;line-height:1.7;color:#333;">'
                '<style>'
                'h1{font-size:1.8em;border-bottom:2px solid #333;padding-bottom:0.3em;}'
                'h2{font-size:1.4em;border-bottom:1px solid #ccc;padding-bottom:0.2em;margin-top:2em;}'
                'h3{font-size:1.15em;margin-top:1.5em;}'
                'table{border-collapse:collapse;width:100%;margin:1em 0;}'
                'th,td{border:1px solid #ddd;padding:8px 12px;text-align:left;}'
                'th{background:#f5f5f5;font-weight:bold;}'
                'blockquote{border-left:4px solid #6366f1;margin:1em 0;padding:0.5em 1em;background:#f8f8ff;}'
                'code{background:#f0f0f0;padding:2px 6px;border-radius:3px;font-size:0.9em;}'
                'pre{background:#f0f0f0;padding:1em;border-radius:6px;overflow-x:auto;}'
                '</style>'
                f'{html}'
                '</div>'
            )
            return styled

        def refresh_paper_list():
            paper_dir = Path(__file__).resolve().parent / "paper"
            files = sorted(f.name for f in paper_dir.glob("*.md")) if paper_dir.exists() else []
            return gr.Dropdown(choices=files if files else ["(none)"],
                               value=files[0] if files else None)

        paper_load_btn.click(fn=load_paper, inputs=[paper_dropdown],
                             outputs=[paper_viewer])
        paper_refresh_btn.click(fn=refresh_paper_list, outputs=[paper_dropdown])

    return demo, {
        "mean_map": mean_map,
        "timestep_map": timestep_map,
        "brain_image": brain_image,
        "npy_download": npy_download,
        "summary_output": summary_output,
        "ts_slider": ts_slider,
        "video_frame": video_frame,
        "network_tc_plot": network_tc_plot,
        "network_bar_plot": network_bar_plot,
        "cognitive_summary": cognitive_summary,
        "yeo_brain_img": yeo_brain_img,
        "cognitive_ts_slider": cognitive_ts_slider,
    }


if __name__ == "__main__":
    print("Loading TRIBE v2 model (this may take a minute)...")
    load_model()
    print("Model loaded!")

    cached = load_cached_results()
    if cached:
        print(f"Restored {cached['meta']['n_timesteps']} timesteps from cache.")
    else:
        print("No cached results found. Run inference to generate them.")

    print("Starting Gradio server...")
    demo, components = build_ui(cached=cached)

    # Expose component refs at module level for run_inference's dict-yield
    mean_map = components["mean_map"]
    timestep_map = components["timestep_map"]
    brain_image = components["brain_image"]
    npy_download = components["npy_download"]
    summary_output = components["summary_output"]
    ts_slider = components["ts_slider"]
    video_frame = components["video_frame"]
    network_tc_plot = components["network_tc_plot"]
    network_bar_plot = components["network_bar_plot"]
    cognitive_summary = components["cognitive_summary"]
    yeo_brain_img = components["yeo_brain_img"]
    cognitive_ts_slider = components["cognitive_ts_slider"]

    demo.launch(
        server_port=7860, share=False,
        allowed_paths=[str(Path(__file__).resolve().parent / "paper" / "figures")],
    )
