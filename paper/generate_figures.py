"""Generate all paper figures from cached TRIBE v2 results."""

import sys
from pathlib import Path

# Fix tribev2 shadow
_script_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _script_dir)
sys.path = [p for p in sys.path if p != _script_dir or p == sys.path[0]]

import json
import numpy as np
from scipy.signal import detrend, find_peaks as scipy_find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

# Load cached data
preds = np.load(RESULTS / "predictions.npy")
labels = np.load(RESULTS / "yeo7_labels.npy").ravel()
meta = json.loads((RESULTS / "metadata.json").read_text())

YEO_NAMES = [
    "Visual", "Somatomotor", "Dorsal Attention",
    "Ventral Attention", "Limbic", "Frontoparietal", "Default Mode",
]
YEO_COLORS = [
    "#781286", "#4682B4", "#00760E", "#C43AFA",
    "#DCF8A4", "#E69422", "#CD3E4E",
]

def find_robust_peaks(timecourse, n_peaks=3, start_trim=1, end_trim=4, min_distance=3):
    """Find peaks with detrending and asymmetric boundary trimming.

    Start trim is minimal (1 frame) to preserve opening hooks.
    End trim is larger (4 frames) to avoid drift + conv1d edge artifacts.
    """
    n = len(timecourse)
    if n == 0:
        return np.array([], dtype=int)
    st = min(start_trim, max(n // 4, 1))
    et = min(end_trim, max(n // 4, 1))
    detrended = detrend(timecourse, type='linear')
    std = detrended.std()
    z = detrended / std if std > 0 else detrended
    interior = np.abs(z[st:n - et])
    peaks, props = scipy_find_peaks(interior, distance=min_distance, prominence=0.1)
    if len(peaks) >= n_peaks:
        top_idx = np.argsort(props['prominences'])[::-1][:n_peaks]
        peak_indices = peaks[top_idx] + st
    else:
        ranked = np.argsort(interior)[::-1]
        selected = []
        for idx in ranked:
            orig = idx + st
            if all(abs(orig - s) >= min_distance for s in selected):
                selected.append(orig)
            if len(selected) == n_peaks:
                break
        peak_indices = np.array(selected, dtype=int)
    if len(peak_indices) > 0:
        magnitudes = np.abs(z[peak_indices])
        peak_indices = peak_indices[np.argsort(magnitudes)[::-1]]
    return peak_indices


n_ts = preds.shape[0]

# Compute network timecourses
tc = np.zeros((n_ts, 7))
for i in range(7):
    mask = labels == (i + 1)
    if mask.any():
        tc[:, i] = preds[:, mask].mean(axis=1)


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: Network Timecourse Overview
# ═══════════════════════════════════════════════════════════════════
print("Generating Figure 1: Network Timecourse...")

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("white")

for i in range(7):
    ax.plot(range(n_ts), tc[:, i], color=YEO_COLORS[i],
            linewidth=2.5, label=YEO_NAMES[i], alpha=0.9)

# Annotate peak moment (robust: detrended, boundary-trimmed)
all_net_max = np.abs(tc).max(axis=1)
peak_t_robust = find_robust_peaks(all_net_max, n_peaks=1)
peak_t = int(peak_t_robust[0]) if len(peak_t_robust) > 0 else int(np.argmax(all_net_max))
peak_net = int(np.argmax(np.abs(tc[peak_t])))
ax.axvline(peak_t, color="gray", linestyle="--", alpha=0.5)
ax.annotate(f"Peak: {YEO_NAMES[peak_net]}\n(TR {peak_t})",
            xy=(peak_t, tc[peak_t, peak_net]),
            xytext=(peak_t + 1.5, tc[peak_t, peak_net] + 0.1),
            fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="gray"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.9))

ax.set_xlabel("Time (seconds)", fontsize=13, fontweight="bold")
ax.set_ylabel("Predicted Activation", fontsize=13, fontweight="bold")
ax.set_title("Predicted Brain Network Activation Over Time",
             fontsize=15, fontweight="bold", pad=15)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9,
          ncol=2, borderaxespad=0.5)
ax.set_xlim(0, n_ts - 1)
ax.grid(True, alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(FIGURES / "fig1_network_timecourse.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  -> {FIGURES / 'fig1_network_timecourse.png'}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Ventral Attention Peaks with Video Frames
# ═══════════════════════════════════════════════════════════════════
print("Generating Figure 2: Ventral Attention Peaks...")

van_idx = 3  # Ventral Attention is network index 3
van_tc = tc[:, van_idx]
# Find top 3 peak timesteps (robust: detrended, boundary-trimmed, min spacing)
peaks = find_robust_peaks(van_tc, n_peaks=3)

fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor("white")
gs = GridSpec(2, 3, figure=fig, height_ratios=[2, 1], hspace=0.4, wspace=0.3)

# Top: full timecourse with peaks marked
ax_tc = fig.add_subplot(gs[0, :])
ax_tc.plot(range(n_ts), van_tc, color=YEO_COLORS[van_idx],
           linewidth=2.5, label="Ventral Attention (Salience)")
ax_tc.fill_between(range(n_ts), van_tc, alpha=0.15, color=YEO_COLORS[van_idx])
for i, p in enumerate(peaks):
    ax_tc.axvline(p, color="red", linestyle="--", alpha=0.6)
    ax_tc.annotate(f"Peak {i+1}\nTR {p}", xy=(p, van_tc[p]),
                   xytext=(p + 0.8, van_tc[p] + 0.05),
                   fontsize=9, fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color="red"))
ax_tc.set_xlabel("Time (seconds)", fontsize=12)
ax_tc.set_ylabel("Predicted Activation", fontsize=12)
ax_tc.set_title("Ventral Attention (Salience) Network — Peak Detection",
                fontsize=14, fontweight="bold")
ax_tc.legend(fontsize=10)
ax_tc.grid(True, alpha=0.2)
ax_tc.spines["top"].set_visible(False)
ax_tc.spines["right"].set_visible(False)

# Bottom: video frames at peak moments
frames_dir = RESULTS / "frames"
for i, p in enumerate(peaks):
    ax_frame = fig.add_subplot(gs[1, i])
    frame_path = frames_dir / f"frame_{p:04d}.jpg"
    if frame_path.exists():
        img = plt.imread(str(frame_path))
        ax_frame.imshow(img)
    ax_frame.set_title(f"TR {p} (Peak {i+1})", fontsize=11, fontweight="bold")
    ax_frame.axis("off")

fig.suptitle("Hook/Surprise Detection: Moments of Peak Salience",
             fontsize=15, fontweight="bold", y=1.02)
fig.savefig(FIGURES / "fig2_ventral_attention_peaks.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  -> {FIGURES / 'fig2_ventral_attention_peaks.png'}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: Radar Chart — Neural Profile
# ═══════════════════════════════════════════════════════════════════
print("Generating Figure 3: Neural Profile Radar...")

mean_abs = np.abs(tc).mean(axis=0)
total = mean_abs.sum()
pct = mean_abs / total * 100

angles = np.linspace(0, 2 * np.pi, 7, endpoint=False).tolist()
angles += angles[:1]  # close the polygon
values = pct.tolist() + [pct[0]]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("white")

ax.fill(angles, values, alpha=0.2, color="#6366f1")
ax.plot(angles, values, linewidth=2.5, color="#6366f1")

# Colored dots at each vertex
for i in range(7):
    ax.scatter(angles[i], values[i], color=YEO_COLORS[i],
               s=120, zorder=5, edgecolors="white", linewidth=2)
    # Label with percentage
    label_offset = values[i] + max(values) * 0.08
    ax.text(angles[i], label_offset, f"{pct[i]:.1f}%",
            ha="center", va="center", fontsize=10, fontweight="bold")

ax.set_xticks(angles[:-1])
ax.set_xticklabels(YEO_NAMES, fontsize=10, fontweight="bold")
ax.set_title("Neural Engagement Profile\n(Mean Absolute Activation)",
             fontsize=14, fontweight="bold", pad=30)
ax.set_ylim(0, max(values) * 1.2)
ax.yaxis.set_visible(False)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(FIGURES / "fig3_radar_neural_profile.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  -> {FIGURES / 'fig3_radar_neural_profile.png'}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: Method Comparison Matrix
# ═══════════════════════════════════════════════════════════════════
print("Generating Figure 5: Method Comparison Matrix...")

methods = ["A/B Testing", "Analytics", "AI Scoring\nTools",
           "Neuromarketing\n(EEG/fMRI)", "Neural Content\nIntelligence"]
dimensions = [
    "Pre-publication",
    "Cost per test",
    "Scalability",
    "Speed",
    "Neural grounding",
    "Emotional insight",
    "Attention prediction",
    "CTA optimization",
    "Scientific basis",
]

# Scores: 0=poor, 1=fair, 2=good, 3=excellent
scores = np.array([
    # A/B   Analytics  AI     Neuro  NCI
    [0,     0,         2,     3,     3],  # Pre-publication
    [1,     3,         3,     0,     3],  # Cost per test
    [1,     3,         3,     0,     3],  # Scalability
    [0,     2,         3,     0,     2],  # Speed
    [0,     0,         0,     3,     2],  # Neural grounding
    [0,     0,         1,     3,     2],  # Emotional insight
    [1,     1,         1,     3,     2],  # Attention prediction
    [2,     1,         1,     2,     2],  # CTA optimization
    [2,     2,         1,     3,     2],  # Scientific basis
])

cmap = matplotlib.colors.ListedColormap(["#ef4444", "#f59e0b", "#22c55e", "#3b82f6"])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("white")

im = ax.imshow(scores, cmap=cmap, norm=norm, aspect="auto")

ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=11, fontweight="bold")
ax.set_yticks(range(len(dimensions)))
ax.set_yticklabels(dimensions, fontsize=11)

# Add score labels
labels_map = {0: "Poor", 1: "Fair", 2: "Good", 3: "Excellent"}
for i in range(len(dimensions)):
    for j in range(len(methods)):
        text_color = "white" if scores[i, j] in [0, 3] else "black"
        ax.text(j, i, labels_map[scores[i, j]], ha="center", va="center",
                fontsize=9, fontweight="bold", color=text_color)

# Highlight NCI column
rect = matplotlib.patches.FancyBboxPatch(
    (3.5, -0.5), 1, len(dimensions), linewidth=3,
    edgecolor="#6366f1", facecolor="none", boxstyle="round,pad=0.1")
ax.add_patch(rect)

ax.set_title("Content Optimization Method Comparison",
             fontsize=15, fontweight="bold", pad=15)

# Legend
legend_patches = [
    mpatches.Patch(color="#ef4444", label="Poor"),
    mpatches.Patch(color="#f59e0b", label="Fair"),
    mpatches.Patch(color="#22c55e", label="Good"),
    mpatches.Patch(color="#3b82f6", label="Excellent"),
]
ax.legend(handles=legend_patches, loc="lower center",
          bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10, frameon=False)

fig.tight_layout()
fig.savefig(FIGURES / "fig5_method_comparison.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  -> {FIGURES / 'fig5_method_comparison.png'}")


# ═══════════════════════════════════════════════════════════════════
# BONUS: Yeo Parcellation (copy from results)
# ═══════════════════════════════════════════════════════════════════
import shutil
yeo_src = RESULTS / "yeo_parcellation.png"
if yeo_src.exists():
    shutil.copy2(yeo_src, FIGURES / "yeo_parcellation.png")
    print(f"  -> Copied yeo_parcellation.png")

mean_src = RESULTS / "mean_map.png"
if mean_src.exists():
    shutil.copy2(mean_src, FIGURES / "mean_activation_map.png")
    print(f"  -> Copied mean_activation_map.png")

ts_src = RESULTS / "timestep_map.png"
if ts_src.exists():
    shutil.copy2(ts_src, FIGURES / "per_timestep_montage.png")
    print(f"  -> Copied per_timestep_montage.png")

anim_src = RESULTS / "brain_animation.mp4"
if anim_src.exists():
    shutil.copy2(anim_src, FIGURES / "brain_rotation_animation.mp4")
    print(f"  -> Copied brain_rotation_animation.mp4")

print("\nAll figures generated!")
print(f"Output directory: {FIGURES}")
