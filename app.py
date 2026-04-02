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

RESULTS_DIR = Path(__file__).resolve().parent / "results"


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

        elapsed = time.time() - t0

        # Persist results + pre-render brain frames for smooth playback
        save_results(
            preds, input_desc, events.shape[0], elapsed,
            mean_img, ts_img, video_path=_LAST_VIDEO,
        )

        # Initial brain image = mean (index 0)
        init_brain = _BRAIN_FRAMES[0] if _BRAIN_FRAMES else None

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
            f"Use the **Timestep** slider or **\u25b6 Play** for synced playback.*"
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
    global _LAST_PREDS, _BRAIN_FRAMES
    if npy_file is None:
        return None, None, None, None, "\u26a0\ufe0f Please upload a .npy file.", gr.Slider()

    try:
        preds = np.load(npy_file)
        if preds.ndim != 2 or preds.shape[1] != 20484:
            return None, None, None, None, (
                f"\u274c **Bad shape:** expected (n_timesteps, 20484), "
                f"got {preds.shape}"
            ), gr.Slider()

        # Need mesh cache for the interactive viewer
        if not _MESH_CACHE:
            global FSAVERAGE
            FSAVERAGE = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
            _build_mesh_cache()

        _LAST_PREDS = preds
        mean_img, ts_img = render_brain_map(preds)
        _BRAIN_FRAMES = prerender_brain_frames(preds)
        init_brain = _BRAIN_FRAMES[0] if _BRAIN_FRAMES else None

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
        return mean_img, ts_img, init_brain, npy_file, summary, slider_update

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, None, f"\u274c **Error:** `{type(e).__name__}: {e}`", gr.Slider()


# ── Gradio UI ────────────────────────────────────────────────────────
def build_ui(cached: dict | None = None):
    """Build the Gradio UI, optionally pre-populated with cached results."""

    # Pre-compute initial values from cache
    init_mean = cached["mean_img"] if cached else None
    init_ts = cached["ts_img"] if cached else None
    init_brain_img = cached.get("init_brain_img") if cached else None
    init_brain_3d = render_interactive_brain(cached["preds"]) if (cached and _MESH_CACHE) else None
    init_summary = cached["summary"] if cached else "*Run inference or load a .npy file to see results.*"
    init_npy = str(RESULTS_DIR / "predictions.npy") if (cached and (RESULTS_DIR / "predictions.npy").exists()) else None
    init_frame = cached.get("init_frame") if cached else None
    init_input_video = cached.get("video_path") if cached else None
    init_anim_video = str(RESULTS_DIR / "brain_animation.mp4") if (RESULTS_DIR / "brain_animation.mp4").exists() else None

    if cached:
        n_ts = cached["preds"].shape[0]
        init_slider_max = n_ts - 1
        init_slider_interactive = True
    else:
        init_slider_max = 0
        init_slider_interactive = False

    with gr.Blocks(
        title="TRIBE v2 Brain Encoder",
    ) as demo:
        # Hidden state for play/pause
        is_playing = gr.State(False)

        gr.Markdown(
            "# \U0001f9e0 TRIBE v2 \u2014 Brain Encoding Demo\n\n"
            "Upload a **video**, **audio file**, or enter **text** to predict "
            "brain responses on the fsaverage5 cortical mesh (20,484 vertices)."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                video_input = gr.Video(label="Video", sources=["upload"])
                audio_input = gr.Audio(
                    label="Audio", type="filepath", sources=["upload"]
                )
                text_input = gr.Textbox(
                    label="Text",
                    placeholder="Or type/paste text here...",
                    lines=3,
                )
                run_btn = gr.Button(
                    "\u25b6  Run Inference", variant="primary", size="lg"
                )

                gr.Markdown("---\n**Or load previous results:**")
                npy_input = gr.File(
                    label="Load .npy predictions",
                    file_types=[".npy"],
                )
                load_btn = gr.Button(
                    "\U0001f4c2  Load & Visualize", variant="secondary", size="lg"
                )
                npy_download = gr.File(
                    label="Download predictions (.npy)", interactive=False,
                    value=init_npy,
                )
                summary_output = gr.Markdown(label="Summary", value=init_summary)

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("\U0001f9e0 Synced Brain Viewer"):
                        with gr.Row():
                            video_frame = gr.Image(
                                label="Video Frame",
                                value=init_frame,
                                height=350,
                            )
                            brain_image = gr.Image(
                                label="Brain Activity",
                                value=init_brain_img,
                                height=350,
                            )
                        with gr.Row():
                            play_btn = gr.Button(
                                "\u25b6 Play", variant="secondary", scale=1,
                            )
                            pause_btn = gr.Button(
                                "\u23f8 Pause", variant="secondary", scale=1,
                            )
                        slider_label = f"Timestep (-1 = mean, 0\u2013{init_slider_max} = TRs)" if cached else "Timestep (-1 = mean)"
                        ts_slider = gr.Slider(
                            minimum=-1, maximum=init_slider_max, value=-1, step=1,
                            label=slider_label,
                            interactive=init_slider_interactive,
                        )
                    with gr.TabItem("\U0001f4ca Mean Activation"):
                        mean_map = gr.Image(
                            label="Mean Brain Activation",
                            value=init_mean,
                        )
                    with gr.TabItem("\u23f1\ufe0f Per-Timestep"):
                        timestep_map = gr.Image(
                            label="Per-Timestep Activations",
                            value=init_ts,
                        )
                    with gr.TabItem("\U0001f9e0 Interactive 3D"):
                        gr.Markdown(
                            "*Drag to rotate \u00b7 Scroll to zoom \u00b7 "
                            "Use the timestep dropdown to change activation.*"
                        )
                        interactive_plot = gr.Plot(
                            label="Interactive 3D Brain",
                            value=init_brain_3d,
                        )
                        interactive_ts = gr.Slider(
                            minimum=-1,
                            maximum=init_slider_max,
                            value=-1, step=1,
                            label="Timestep (-1 = mean)",
                            interactive=init_slider_interactive,
                        )
                    with gr.TabItem("\U0001f3ac 3D Rotation Video"):
                        gr.Markdown(
                            "*Generate a rotating brain animation that matches "
                            "the input video duration. Play both side-by-side to "
                            "see what the brain is doing at each moment.*"
                        )
                        anim_btn = gr.Button(
                            "\U0001f3ac  Generate Rotating Animation",
                            variant="primary", size="lg",
                        )
                        anim_status = gr.Markdown("")
                        with gr.Row():
                            anim_input_video = gr.Video(
                                label="Input Video",
                                value=init_input_video,
                            )
                            anim_video = gr.Video(
                                label="Brain Animation",
                                value=init_anim_video,
                            )

        # ── Timer for auto-play ──────────────────────────────────────
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
                # Stop at the end
                return max_ts, gr.Timer(active=False)
            return next_ts, gr.Timer(active=True)

        play_btn.click(
            fn=start_playing,
            outputs=[is_playing, play_timer],
        )
        pause_btn.click(
            fn=stop_playing,
            outputs=[is_playing, play_timer],
        )
        play_timer.tick(
            fn=tick_forward,
            inputs=[is_playing, ts_slider],
            outputs=[ts_slider, play_timer],
        )

        # ── Wire up events ───────────────────────────────────────────
        all_outputs = [
            mean_map, timestep_map, brain_image,
            npy_download, summary_output, ts_slider,
        ]

        run_btn.click(
            fn=run_inference,
            inputs=[video_input, audio_input, text_input],
            outputs=all_outputs,
        )
        load_btn.click(
            fn=load_npy,
            inputs=[npy_input],
            outputs=all_outputs,
        )

        # Timestep slider updates brain + video frame (instant image swap)
        ts_slider.change(
            fn=on_timestep_change,
            inputs=[ts_slider],
            outputs=[brain_image, video_frame],
        )

        # Interactive 3D slider re-renders Plotly figure
        def on_interactive_ts(ts):
            if _LAST_PREDS is None:
                return None
            return render_interactive_brain(_LAST_PREDS, timestep=int(ts))

        interactive_ts.change(
            fn=on_interactive_ts,
            inputs=[interactive_ts],
            outputs=[interactive_plot],
        )

        # Generate 3D rotation animation
        anim_btn.click(
            fn=generate_animation,
            outputs=[anim_input_video, anim_video, anim_status],
        )

        gr.Examples(
            examples=[
                [None, None, "A person walks through a quiet forest. Birds are singing in the trees above."],
                [None, None, "The crowd erupts in cheers as the home team scores the winning goal."],
            ],
            inputs=[video_input, audio_input, text_input],
        )

    return demo, {
        "mean_map": mean_map,
        "timestep_map": timestep_map,
        "brain_image": brain_image,
        "npy_download": npy_download,
        "summary_output": summary_output,
        "ts_slider": ts_slider,
        "video_frame": video_frame,
    }


if __name__ == "__main__":
    print("Loading TRIBE v2 model (this may take a minute)...")
    load_model()
    print("Model loaded!")

    # Try to restore cached results
    cached = load_cached_results()
    if cached:
        print(f"Restored {cached['meta']['n_timesteps']} timesteps from cache.")
    else:
        print("No cached results found. Run inference to generate them.")

    print("Starting Gradio server...")
    demo, components = build_ui(cached=cached)

    # Expose component refs at module level so run_inference's dict-yield works
    mean_map = components["mean_map"]
    timestep_map = components["timestep_map"]
    brain_image = components["brain_image"]
    npy_download = components["npy_download"]
    summary_output = components["summary_output"]
    ts_slider = components["ts_slider"]
    video_frame = components["video_frame"]

    demo.launch(server_port=7860, share=False)
