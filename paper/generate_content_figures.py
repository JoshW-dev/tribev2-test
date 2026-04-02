"""Generate all paper figures using the real content videos (not bears)."""

import sys
from pathlib import Path

_project = str(Path(__file__).resolve().parent.parent)
sys.path = [p for p in sys.path if p != _project]

import json
import subprocess as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import os
for _extra in [
    Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin",
]:
    if _extra.exists():
        os.environ["PATH"] = str(_extra) + os.pathsep + os.environ.get("PATH", "")

PROJECT = Path(__file__).resolve().parent.parent
COMPARE = PROJECT / "paper" / "comparison_data"
FIGURES = PROJECT / "paper" / "figures"
FIGURES.mkdir(exist_ok=True)

YEO_NAMES = ["Visual", "Somatomotor", "Dorsal Attention",
             "Ventral Attention", "Limbic", "Frontoparietal", "Default Mode"]
YEO_COLORS = ["#781286", "#4682B4", "#00760E", "#C43AFA",
              "#DCF8A4", "#E69422", "#CD3E4E"]

VIDEOS = {
    "BusinessEdLeilaHarmozi": ("Business Education (Leila Hormozi)", "#6366f1", "Talking head / educational"),
    "ElonAI": ("Tech News (Elon AI)", "#22c55e", "News / commentary"),
    "PerfumeUGCInterview": ("UGC Interview (Perfume)", "#f59e0b", "UGC product review"),
    "sanitaryPadProductDemo": ("Product Demo", "#ef4444", "Product demonstration"),
    "viralJapaneseIceCutter": ("Viral Satisfying (Ice Cutter)", "#06b6d4", "Viral / satisfying content"),
}


def extract_thumbnail(video_path, out_path, time_sec=1.0):
    """Extract a single frame from a video."""
    if out_path.exists():
        return
    sp.run(["ffmpeg", "-y", "-ss", str(time_sec), "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2", str(out_path)], capture_output=True)


def extract_key_frames(video_path, out_dir, timestamps):
    """Extract frames at specific timestamps."""
    out_dir.mkdir(exist_ok=True)
    paths = []
    for t in timestamps:
        out = out_dir / f"frame_{t:04.1f}s.jpg"
        if not out.exists():
            sp.run(["ffmpeg", "-y", "-ss", str(t), "-i", str(video_path),
                    "-frames:v", "1", "-q:v", "2", str(out)], capture_output=True)
        paths.append(str(out) if out.exists() else None)
    return paths


# ═══════════════════════════════════════════════════════════════════
# Extract thumbnails for all videos
# ═══════════════════════════════════════════════════════════════════
print("Extracting video thumbnails...")
for stem in VIDEOS:
    video_path = PROJECT / f"{stem}.mp4"
    if video_path.exists():
        extract_thumbnail(video_path, FIGURES / f"thumb_{stem}.jpg", time_sec=2.0)
        print(f"  -> thumb_{stem}.jpg")


# ═══════════════════════════════════════════════════════════════════
# Load all timecourse data
# ═══════════════════════════════════════════════════════════════════
all_tc = {}
for stem, (label, color, desc) in VIDEOS.items():
    tc_path = COMPARE / f"{stem}_tc.npy"
    if tc_path.exists():
        all_tc[stem] = np.load(tc_path)
        print(f"Loaded {stem}: {all_tc[stem].shape}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE: Per-video deep analysis (timecourse + peaks + thumbnail)
# One figure per video showing the full story
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating per-video analysis figures...")

for stem, (label, color, desc) in VIDEOS.items():
    if stem not in all_tc:
        continue
    tc = all_tc[stem]
    n_ts = tc.shape[0]
    video_path = PROJECT / f"{stem}.mp4"

    # Find peaks for ventral attention (hook/surprise)
    van_tc = tc[:, 3]  # Ventral Attention
    peaks = np.argsort(np.abs(van_tc))[-3:][::-1]

    # Extract frames at peak moments
    peak_frames = extract_key_frames(
        video_path, FIGURES / f"peaks_{stem}",
        [float(p) for p in peaks]
    )

    # Extract a few evenly-spaced frames for the overview
    overview_times = np.linspace(1, n_ts - 1, 5).astype(int)
    overview_frames = extract_key_frames(
        video_path, FIGURES / f"overview_{stem}",
        [float(t) for t in overview_times]
    )

    # ── Create the analysis figure ──
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("white")

    # Top row: video frames overview
    gs = GridSpec(4, 5, figure=fig, hspace=0.4, wspace=0.3,
                  height_ratios=[1, 2, 2, 1.5])

    for i, (t, fpath) in enumerate(zip(overview_times, overview_frames)):
        ax = fig.add_subplot(gs[0, i])
        if fpath and Path(fpath).exists():
            img = plt.imread(fpath)
            ax.imshow(img)
        ax.set_title(f"{t}s", fontsize=9, fontweight="bold")
        ax.axis("off")

    # Row 2: Full 7-network timecourse
    ax_tc = fig.add_subplot(gs[1, :])
    for i in range(7):
        ax_tc.plot(range(n_ts), tc[:, i], color=YEO_COLORS[i],
                   linewidth=2, label=YEO_NAMES[i], alpha=0.85)
    ax_tc.set_xlabel("Time (seconds)", fontsize=11)
    ax_tc.set_ylabel("Predicted Activation", fontsize=11)
    ax_tc.set_title("Brain Network Activation Over Time", fontsize=13, fontweight="bold")
    ax_tc.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    ax_tc.set_xlim(0, n_ts - 1)
    ax_tc.grid(True, alpha=0.2)
    ax_tc.spines["top"].set_visible(False)
    ax_tc.spines["right"].set_visible(False)

    # Row 3: Ventral attention peaks + peak frames
    ax_van = fig.add_subplot(gs[2, :3])
    ax_van.plot(range(n_ts), van_tc, color=YEO_COLORS[3], linewidth=2.5)
    ax_van.fill_between(range(n_ts), van_tc, alpha=0.15, color=YEO_COLORS[3])
    for i, p in enumerate(peaks):
        ax_van.axvline(p, color="red", linestyle="--", alpha=0.5)
        ax_van.annotate(f"Peak {i+1}\n{p}s", xy=(p, van_tc[p]),
                        xytext=(p + 1, van_tc[p] + 0.02),
                        fontsize=8, arrowprops=dict(arrowstyle="->", color="red"))
    ax_van.set_title("Salience/Hook Detection (Ventral Attention)", fontsize=11, fontweight="bold")
    ax_van.set_xlabel("Time (seconds)", fontsize=10)
    ax_van.grid(True, alpha=0.2)
    ax_van.spines["top"].set_visible(False)
    ax_van.spines["right"].set_visible(False)

    # Peak frames
    for i, (p, fpath) in enumerate(zip(peaks, peak_frames)):
        ax_f = fig.add_subplot(gs[2, 3 + i]) if i < 2 else None
        if ax_f and fpath and Path(fpath).exists():
            img = plt.imread(fpath)
            ax_f.imshow(img)
            ax_f.set_title(f"Peak {i+1} ({p}s)", fontsize=9, fontweight="bold")
            ax_f.axis("off")

    # Row 4: Radar chart
    ax_radar = fig.add_subplot(gs[3, :2], polar=True)
    mean_abs = np.abs(tc).mean(axis=0)
    total = mean_abs.sum()
    pct = mean_abs / total * 100
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False).tolist()
    angles += angles[:1]
    values = pct.tolist() + [pct[0]]
    ax_radar.fill(angles, values, alpha=0.2, color=color)
    ax_radar.plot(angles, values, linewidth=2.5, color=color)
    for j in range(7):
        ax_radar.scatter(angles[j], values[j], color=YEO_COLORS[j], s=60, zorder=5)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels([n[:8] for n in YEO_NAMES], fontsize=7)
    ax_radar.set_title("Neural Profile", fontsize=10, fontweight="bold", pad=15)
    ax_radar.yaxis.set_visible(False)

    # Row 4: Key stats
    ax_stats = fig.add_subplot(gs[3, 2:])
    ax_stats.axis("off")
    dominant = YEO_NAMES[np.argmax(pct)]
    secondary = YEO_NAMES[np.argsort(pct)[-2]]
    peak_t = int(np.argmax(np.abs(tc).max(axis=1)))
    peak_net = YEO_NAMES[int(np.argmax(np.abs(tc[peak_t])))]

    stats_text = (
        f"Content Type: {desc}\n"
        f"Duration: {n_ts} seconds\n\n"
        f"Dominant Network: {dominant} ({pct[np.argmax(pct)]:.1f}%)\n"
        f"Secondary Network: {secondary} ({pct[np.argsort(pct)[-2]]:.1f}%)\n"
        f"Peak Brain Response: {peak_t}s ({peak_net})\n\n"
        f"Hook Moments (Salience Peaks): {', '.join(f'{p}s' for p in peaks)}\n"
    )
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=11, verticalalignment="top", fontfamily="monospace",
                  bbox=dict(boxstyle="round", facecolor="#f8f8f8", edgecolor="#ddd"))

    fig.suptitle(f"Neural Content Analysis: {label}",
                 fontsize=16, fontweight="bold", y=0.98)

    out_path = FIGURES / f"analysis_{stem}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path.name}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE: Side-by-side comparison dashboard
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating comparison dashboard...")

fig, axes = plt.subplots(2, 5, figsize=(22, 8),
                          gridspec_kw={"height_ratios": [1, 2]})
fig.patch.set_facecolor("white")

for idx, (stem, (label, color, desc)) in enumerate(VIDEOS.items()):
    if stem not in all_tc:
        continue
    tc = all_tc[stem]

    # Top: thumbnail
    thumb = FIGURES / f"thumb_{stem}.jpg"
    if thumb.exists():
        img = plt.imread(str(thumb))
        axes[0, idx].imshow(img)
    axes[0, idx].set_title(label.split("(")[0].strip(), fontsize=10, fontweight="bold")
    axes[0, idx].axis("off")

    # Bottom: mini timecourse
    for i in range(7):
        axes[1, idx].plot(range(tc.shape[0]), tc[:, i],
                         color=YEO_COLORS[i], linewidth=1.5, alpha=0.8)
    axes[1, idx].set_xlabel("Time (s)", fontsize=8)
    if idx == 0:
        axes[1, idx].set_ylabel("Activation", fontsize=8)
    axes[1, idx].tick_params(labelsize=7)
    axes[1, idx].grid(True, alpha=0.2)
    axes[1, idx].spines["top"].set_visible(False)
    axes[1, idx].spines["right"].set_visible(False)

    # Dominant label
    mean_abs = np.abs(tc).mean(axis=0)
    dom = YEO_NAMES[np.argmax(mean_abs)]
    axes[1, idx].set_title(f"Dominant: {dom}", fontsize=8, color="gray")

fig.suptitle("Neural Content Analysis: 5 Content Types Compared",
             fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIGURES / "comparison_dashboard.png", dpi=180,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  -> comparison_dashboard.png")


# ═══════════════════════════════════════════════════════════════════
# Print actionable insights summary
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CONTENT ANALYSIS INSIGHTS")
print("=" * 70)

for stem, (label, color, desc) in VIDEOS.items():
    if stem not in all_tc:
        continue
    tc = all_tc[stem]
    mean_abs = np.abs(tc).mean(axis=0)
    total = mean_abs.sum()
    pct = mean_abs / total * 100
    dom_idx = np.argmax(pct)
    van_peaks = np.argsort(np.abs(tc[:, 3]))[-3:][::-1]

    print(f"\n{label} ({desc}):")
    print(f"  Duration: {tc.shape[0]}s")
    print(f"  Dominant: {YEO_NAMES[dom_idx]} ({pct[dom_idx]:.1f}%)")
    print(f"  Profile: " + " | ".join(f"{YEO_NAMES[i][:6]}:{pct[i]:.0f}%" for i in range(7)))
    print(f"  Hook moments (salience peaks): {', '.join(f'{p}s' for p in van_peaks)}")

    # Business insight
    if dom_idx == 0:  # Visual
        print(f"  Insight: Visually-driven engagement. Optimize for visual quality, motion, color.")
    elif dom_idx == 1:  # Somatomotor
        print(f"  Insight: Speech/embodied processing dominant. Voice and delivery matter most.")
    elif dom_idx == 2:  # Dorsal Attention
        print(f"  Insight: Strong focused attention. Viewers are tracking something specific.")
    elif dom_idx == 3:  # Ventral Attention
        print(f"  Insight: High salience/surprise. Content has strong hooks and pattern breaks.")
    elif dom_idx == 4:  # Limbic
        print(f"  Insight: Emotional engagement dominant. Content evokes strong affect.")
    elif dom_idx == 5:  # Frontoparietal
        print(f"  Insight: Cognitive processing. Complex/persuasive content requiring thought.")
    elif dom_idx == 6:  # Default Mode
        print(f"  Insight: Narrative engagement. Story/self-referential processing active.")

print("\nAll figures generated!")
