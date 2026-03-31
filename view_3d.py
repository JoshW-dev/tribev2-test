"""
Interactive 3D brain mesh viewer for TRIBE v2 predictions.
Opens in your browser with a rotatable/zoomable cortical surface.
"""

import sys
from pathlib import Path

import numpy as np
from nilearn import datasets, plotting


def main():
    npy_path = sys.argv[1] if len(sys.argv) > 1 else "predictions.npy"
    preds = np.load(npy_path)
    print(f"Loaded {npy_path}: shape {preds.shape}")

    n_timesteps, n_vertices = preds.shape
    lh = n_vertices // 2  # 10242 per hemisphere

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    # Use mean activation across timesteps
    mean_pred = preds.mean(axis=0)

    # --- Left hemisphere interactive 3D ---
    print("Generating interactive 3D viewer (left hemisphere)...")
    view_lh = plotting.view_surf(
        fsaverage["pial_left"],
        mean_pred[:lh],
        bg_map=fsaverage["sulc_left"],
        threshold=0.01,
        title="TRIBE v2 — Left Hemisphere",
        colorbar=True,
        symmetric_cmap=True,
    )
    lh_path = Path(npy_path).with_name("brain_3d_left.html")
    view_lh.save_as_html(str(lh_path))
    print(f"  Saved: {lh_path}")

    # --- Right hemisphere interactive 3D ---
    print("Generating interactive 3D viewer (right hemisphere)...")
    view_rh = plotting.view_surf(
        fsaverage["pial_right"],
        mean_pred[lh:],
        bg_map=fsaverage["sulc_right"],
        threshold=0.01,
        title="TRIBE v2 — Right Hemisphere",
        colorbar=True,
        symmetric_cmap=True,
    )
    rh_path = Path(npy_path).with_name("brain_3d_right.html")
    view_rh.save_as_html(str(rh_path))
    print(f"  Saved: {rh_path}")

    # --- Open in browser ---
    import webbrowser
    webbrowser.open(f"file://{lh_path.resolve()}")
    webbrowser.open(f"file://{rh_path.resolve()}")
    print("\nOpened in browser — click and drag to rotate, scroll to zoom.")


if __name__ == "__main__":
    main()
