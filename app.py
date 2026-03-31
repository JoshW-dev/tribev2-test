"""
TRIBE v2 Gradio UI — Select a video (or audio/text), run inference,
and view predicted brain activation maps.
"""

import sys
from pathlib import Path

# Fix: the cloned "tribev2/" directory shadows the installed package.
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _script_dir]

import tempfile

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

    # Offset right hemisphere so both are visible side-by-side
    rh_shifted = rh_coords.copy()
    rh_shifted[:, 0] += 55

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
    """Improved static brain maps → (mean_map_path, timestep_montage_path)."""
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
        f"Mean Predicted Brain Response  ·  {n_timesteps} TRs",
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


def render_interactive_brain(preds: np.ndarray) -> go.Figure:
    """Interactive 3D brain with play/pause animation over timesteps."""
    n_timesteps, n_vertices = preds.shape
    abs_max = float(max(abs(preds.min()), abs(preds.max())))
    mean_intensity = np.round(preds.mean(axis=0), 5).tolist()
    mc = _MESH_CACHE

    # ── Base trace (mean activation) ─────────────────────────────────
    base = go.Mesh3d(
        x=mc["x"], y=mc["y"], z=mc["z"],
        i=mc["i"], j=mc["j"], k=mc["k"],
        intensity=mean_intensity,
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

    fig = go.Figure(data=[base])

    # ── Animation frames (full mesh per frame for plotly 6 compat) ──
    frames = []
    for t in range(n_timesteps):
        frames.append(go.Frame(
            data=[go.Mesh3d(
                x=mc["x"], y=mc["y"], z=mc["z"],
                i=mc["i"], j=mc["j"], k=mc["k"],
                intensity=np.round(preds[t], 5).tolist(),
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
            )],
            name=str(t),
        ))
    fig.frames = frames

    # ── Slider + Play/Pause controls ─────────────────────────────────
    steps = [
        dict(
            args=[[str(t)], dict(
                frame=dict(duration=0, redraw=True),
                mode="immediate",
                transition=dict(duration=0),
            )],
            label=str(t),
            method="animate",
        )
        for t in range(n_timesteps)
    ]

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
        updatemenus=[dict(
            type="buttons",
            showactive=True,
            y=-0.06, x=0.5, xanchor="center",
            direction="left",
            pad=dict(r=10),
            font=dict(size=13),
            buttons=[
                dict(
                    label="\u25b6  Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=600, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=250, easing="cubic-in-out"),
                    )],
                ),
                dict(
                    label="\u23f8  Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(
                prefix="Timestep: TR ",
                visible=True,
                font=dict(size=14, color="#333"),
            ),
            transition=dict(duration=200, easing="cubic-in-out"),
            x=0.06, len=0.88, xanchor="left", y=-0.01,
            pad=dict(b=10),
            steps=steps,
        )],
        height=650,
        margin=dict(l=0, r=30, t=10, b=100),
        paper_bgcolor="white",
    )

    return fig


# ── Inference entry point ────────────────────────────────────────────
def run_inference(video, audio, text):
    if MODEL is None:
        return None, None, None, None, "\u26a0\ufe0f Model not loaded yet — wait for startup."

    kwargs = {}
    input_desc = ""
    if video is not None:
        kwargs["video_path"] = video
        input_desc = f"Video: {Path(video).name}"
    elif audio is not None:
        kwargs["audio_path"] = audio
        input_desc = f"Audio: {Path(audio).name}"
    elif text is not None and text.strip():
        txt_path = tempfile.mktemp(suffix=".txt")
        Path(txt_path).write_text(text)
        kwargs["text_path"] = txt_path
        input_desc = f"Text: {text[:80]}..."
    else:
        return None, None, None, None, "\u26a0\ufe0f Please provide a video, audio file, or text."

    try:
        events = MODEL.get_events_dataframe(**kwargs)
        preds, segments = MODEL.predict(events=events)

        # Save predictions to disk
        npy_path = Path("predictions.npy")
        np.save(npy_path, preds)

        mean_img, ts_img = render_brain_map(preds)
        brain_fig = render_interactive_brain(preds)

        summary = (
            f"### Results\n\n"
            f"| | |\n|---|---|\n"
            f"| **Input** | {input_desc} |\n"
            f"| **Events** | {events.shape[0]} extracted |\n"
            f"| **Predictions** | {preds.shape[0]} timesteps \u00d7 "
            f"{preds.shape[1]:,} vertices |\n"
            f"| **Activation range** | [{preds.min():.4f}, {preds.max():.4f}] |\n"
            f"| **Saved to** | `{npy_path.resolve()}` |"
        )
        return mean_img, ts_img, brain_fig, str(npy_path.resolve()), summary

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, None, f"\u274c **Error:** {e}"


def load_npy(npy_file):
    """Visualize a previously saved predictions .npy file."""
    if npy_file is None:
        return None, None, None, None, "\u26a0\ufe0f Please upload a .npy file."

    try:
        preds = np.load(npy_file)
        if preds.ndim != 2 or preds.shape[1] != 20484:
            return None, None, None, None, (
                f"\u274c **Bad shape:** expected (n_timesteps, 20484), "
                f"got {preds.shape}"
            )

        # Need mesh cache for the interactive viewer
        if not _MESH_CACHE:
            global FSAVERAGE
            FSAVERAGE = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
            _build_mesh_cache()

        mean_img, ts_img = render_brain_map(preds)
        brain_fig = render_interactive_brain(preds)

        summary = (
            f"### Results (loaded from file)\n\n"
            f"| | |\n|---|---|\n"
            f"| **Source** | {Path(npy_file).name} |\n"
            f"| **Predictions** | {preds.shape[0]} timesteps \u00d7 "
            f"{preds.shape[1]:,} vertices |\n"
            f"| **Activation range** | [{preds.min():.4f}, {preds.max():.4f}] |"
        )
        return mean_img, ts_img, brain_fig, npy_file, summary

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, None, f"\u274c **Error:** {e}"


# ── Gradio UI ────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        title="TRIBE v2 Brain Encoder",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# \U0001f9e0 TRIBE v2 — Brain Encoding Demo\n\n"
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
                )
                summary_output = gr.Markdown(label="Summary")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("\U0001f9e0 Interactive 3D Brain"):
                        gr.Markdown(
                            "*Drag to rotate \u00b7 Scroll to zoom \u00b7 "
                            "Use the slider or \u25b6 Play to animate through timesteps*"
                        )
                        interactive_plot = gr.Plot(label="Interactive Brain Viewer")
                    with gr.TabItem("\U0001f4ca Mean Activation"):
                        mean_map = gr.Image(label="Mean Brain Activation")
                    with gr.TabItem("\u23f1\ufe0f Per-Timestep"):
                        timestep_map = gr.Image(label="Per-Timestep Activations")

        run_btn.click(
            fn=run_inference,
            inputs=[video_input, audio_input, text_input],
            outputs=[mean_map, timestep_map, interactive_plot, npy_download, summary_output],
        )
        load_btn.click(
            fn=load_npy,
            inputs=[npy_input],
            outputs=[mean_map, timestep_map, interactive_plot, npy_download, summary_output],
        )

        gr.Examples(
            examples=[
                [None, None, "A person walks through a quiet forest. Birds are singing in the trees above."],
                [None, None, "The crowd erupts in cheers as the home team scores the winning goal."],
            ],
            inputs=[video_input, audio_input, text_input],
        )

    return demo


if __name__ == "__main__":
    print("Loading TRIBE v2 model (this may take a minute)...")
    load_model()
    print("Model loaded! Starting Gradio server...")
    demo = build_ui()
    demo.launch(server_port=7860, share=False)
