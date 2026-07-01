"""Render 4 distinct cortical surface PNGs for the scalefactor-next landing page.

Each image shows fsaverage5 lateral views (L + R hemisphere side by side) with a
real-looking activation overlay. Variety across the 4 panels comes from:
  - using both available timesteps (TR 0 and TR 1) of `predictions.npy`
  - rotating between lateral and medial views

The output is intentionally chrome-free (no titles, no colorbar, no axis labels,
transparent background) so the React layer can compose its own annotations.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from nilearn import datasets, plotting

OUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREDS = np.load("predictions.npy")  # (T, 20484)
T, V = PREDS.shape
HEMI_V = V // 2  # 10242

fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")


def render(name: str, data: np.ndarray, view: str = "lateral", *, vmax: float | None = None,
           threshold: float = 0.06, cmap: str = "cold_hot"):
    """Render L + R hemisphere side by side at 4:3 panel aspect."""
    fig, axes = plt.subplots(
        1, 2, figsize=(8, 6),
        subplot_kw={"projection": "3d"},
        facecolor="#050505",
    )
    for ax, hemi in zip(axes, ("left", "right")):
        ax.set_facecolor("#050505")
        mesh = fsaverage[f"pial_{hemi}"]
        d = data[:HEMI_V] if hemi == "left" else data[HEMI_V:]
        plotting.plot_surf_stat_map(
            mesh, d,
            hemi=hemi, view=view,
            colorbar=False,
            bg_map=fsaverage[f"sulc_{hemi}"],
            bg_on_data=True,
            threshold=threshold,
            vmax=vmax,
            cmap=cmap,
            axes=ax,
        )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=-0.05)
    out = OUT_DIR / f"{name}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.05,
                facecolor="#050505", transparent=False)
    plt.close(fig)
    print(f"  wrote {out}")


# Build 4 distinct activation patterns from the 2 real timesteps.
# Pair-color hints come from the landing component's regionColor field but the
# rendered colormap is the standard cold-hot scientific palette.

t0 = PREDS[0]
t1 = PREDS[1] if T > 1 else PREDS[0]

# Pair 1 — viral hook: high-magnitude moment (lateral view, full punch)
render("viral_hook", t0, view="lateral", vmax=0.55, threshold=0.08)

# Pair 2 — emotional arc: medial view shows midline DMN territory
render("emotional_arc", t0 * 0.85 + t1 * 0.15, view="medial", vmax=0.45, threshold=0.06)

# Pair 3 — tactic list: lateral view of the second timestep, slightly amped
render("tactic_list", t1 * 1.4, view="lateral", vmax=0.55, threshold=0.08)

# Pair 4 — flat response: dampened activation, expanded vmax so what little
# response exists looks faint rather than absent. Conveys "physically watching,
# cognitively gone" without making the brain look broken.
flat = t1 * 0.55
render("flat_response", flat, view="lateral", vmax=0.85, threshold=0.05)

print("Done.")
