"""Visualize TRIBE v2 predictions on the fsaverage5 cortical surface."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving to file

import matplotlib.pyplot as plt
import numpy as np
from nilearn import datasets, plotting, surface


def main():
    npy_path = sys.argv[1] if len(sys.argv) > 1 else "predictions.npy"
    preds = np.load(npy_path)
    print(f"Loaded {npy_path}: shape {preds.shape}")

    n_timesteps, n_vertices = preds.shape
    # fsaverage5 has 10242 vertices per hemisphere = 20484 total
    assert n_vertices == 20484, f"Expected 20484 vertices, got {n_vertices}"

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    lh_verts = n_vertices // 2  # 10242

    # Plot the mean activation across all timesteps
    mean_pred = preds.mean(axis=0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5),
                             subplot_kw={"projection": "3d"})

    views = [
        ("left", "lateral"),
        ("left", "medial"),
        ("right", "lateral"),
        ("right", "medial"),
    ]

    for ax, (hemi, view) in zip(axes, views):
        mesh = fsaverage[f"pial_{hemi}"]
        data = mean_pred[:lh_verts] if hemi == "left" else mean_pred[lh_verts:]
        plotting.plot_surf_stat_map(
            mesh, data, hemi=hemi, view=view,
            colorbar=False, bg_map=fsaverage[f"sulc_{hemi}"],
            axes=ax, threshold=0.01,
        )
        ax.set_title(f"{hemi} {view}", fontsize=12)

    fig.suptitle(
        f"TRIBE v2 — Mean predicted brain response ({n_timesteps} timesteps)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    out_path = Path(npy_path).with_suffix(".png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved brain map to {out_path}")

    # Also save per-timestep montage (first 8 timesteps)
    n_show = min(8, n_timesteps)
    fig2, axes2 = plt.subplots(n_show, 2, figsize=(10, 3 * n_show),
                                subplot_kw={"projection": "3d"})
    if n_show == 1:
        axes2 = axes2.reshape(1, -1)

    for t in range(n_show):
        for col, hemi in enumerate(["left", "right"]):
            mesh = fsaverage[f"pial_{hemi}"]
            data = preds[t, :lh_verts] if hemi == "left" else preds[t, lh_verts:]
            plotting.plot_surf_stat_map(
                mesh, data, hemi=hemi, view="lateral",
                colorbar=False, bg_map=fsaverage[f"sulc_{hemi}"],
                axes=axes2[t, col], threshold=0.01,
            )
            if t == 0:
                axes2[t, col].set_title(f"{hemi} hemisphere", fontsize=11)
        axes2[t, 0].set_ylabel(f"t={t}", fontsize=10, labelpad=40)

    fig2.suptitle("TRIBE v2 — Per-timestep predictions", fontsize=14, y=1.01)
    plt.tight_layout()

    montage_path = Path(npy_path).with_name("predictions_timesteps.png")
    fig2.savefig(montage_path, dpi=150, bbox_inches="tight")
    print(f"Saved timestep montage to {montage_path}")


if __name__ == "__main__":
    main()
