"""Generate all paper figures using the real content videos (not bears)."""

import sys
from pathlib import Path

_project = str(Path(__file__).resolve().parent.parent)
sys.path = [p for p in sys.path if p != _project]

import json
import subprocess as sp
import numpy as np
from scipy.signal import detrend, find_peaks as scipy_find_peaks
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
    peaks = find_robust_peaks(van_tc, n_peaks=3)

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
    # Compute the salience magnitude signal (what peak detection actually uses)
    van_detrended = detrend(van_tc, type='linear')
    van_std = van_detrended.std()
    van_z = van_detrended / van_std if van_std > 0 else van_detrended
    van_salience = np.abs(van_z)  # Absolute z-score = salience magnitude
    # Plot salience magnitude as baseline
    ax_van.plot(range(n_ts), van_salience, color=YEO_COLORS[3], linewidth=1.8,
                alpha=0.5, zorder=2)
    ax_van.fill_between(range(n_ts), van_salience, alpha=0.08, color=YEO_COLORS[3])
    # Threshold line — median salience as "baseline attention"
    median_sal = np.median(van_salience)
    ax_van.axhline(median_sal, color="gray", linewidth=1, linestyle=":",
                   alpha=0.6, zorder=1)
    ax_van.text(n_ts - 1, median_sal, " baseline", va="bottom", ha="right",
                fontsize=7, color="gray", alpha=0.7)
    # Highlight peak regions with colored spans and bold markers
    peak_colors_list = ["#FF4444", "#FF8800", "#FFBB00"]
    peak_labels = ["\u2605 Strongest Hook", "\u2605 2nd Hook", "\u2605 3rd Hook"]
    for i, p in enumerate(sorted(peaks[:3])):
        rank = list(peaks).index(p)
        pc = peak_colors_list[min(rank, 2)]
        # Highlight region around peak
        span_lo = max(0, p - 1)
        span_hi = min(n_ts - 1, p + 1)
        ax_van.axvspan(span_lo, span_hi, alpha=0.18, color=pc, zorder=1)
        # Bold marker at peak
        ax_van.scatter([p], [van_salience[p]], color=pc, s=120, zorder=5,
                       edgecolors="white", linewidths=1.5)
        # Label
        y_offset = van_salience[p] + (ax_van.get_ylim()[1] - ax_van.get_ylim()[0]) * 0.02 if ax_van.get_ylim()[1] > 0 else van_salience[p] + 0.1
        ax_van.annotate(f"{peak_labels[rank]}\n{p}s",
                        xy=(p, van_salience[p]),
                        xytext=(p + 1.5, van_salience[p] + 0.3),
                        fontsize=8, fontweight="bold", color=pc,
                        arrowprops=dict(arrowstyle="-|>", color=pc, lw=1.5),
                        zorder=6)
    # Mark boundary trim zones (asymmetric: 1 at start, 4 at end)
    et = min(4, max(n_ts // 4, 1))
    ax_van.axvspan(n_ts - et, n_ts, alpha=0.10, color="#cccccc", zorder=0)
    ax_van.text(n_ts - et / 2, 0, "edge\ntrimmed", ha="center", va="bottom",
                fontsize=6, color="gray", alpha=0.6)
    # Opening hook indicator — check if ANY network fires strongly in the opening
    # Scale opening window to video length: ~10% of duration, min 2s, max 6s
    opening_threshold = max(2, min(6, round(n_ts * 0.10)))
    opening_end = min(opening_threshold + 1, n_ts)
    # For each network: is its peak absolute activation in the opening above its video mean?
    has_opening_hook = False
    for net_i in range(tc.shape[1]):
        net_abs = np.abs(tc[:, net_i])
        opening_max = net_abs[:opening_end].max()
        video_mean = net_abs.mean()
        if opening_max > video_mean * 1.3:  # 30% above mean = strong opening for this network
            has_opening_hook = True
            break
    y_top = van_salience.max() * 1.15  # position badge above signal
    if has_opening_hook:
        # Green banner for strong opening hook
        ax_van.axvspan(0, opening_threshold, alpha=0.10, color="#22c55e", zorder=0)
        ax_van.text(opening_threshold / 2, y_top,
                    "\u2713 Strong Opening Hook",
                    ha="center", va="top", fontsize=8, fontweight="bold",
                    color="#16a34a",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#dcfce7",
                              edgecolor="#22c55e", alpha=0.9),
                    zorder=7)
    else:
        # Gray note for no opening hook
        ax_van.axvspan(0, opening_threshold, alpha=0.05, color="#94a3b8", zorder=0)
        ax_van.text(opening_threshold / 2, y_top,
                    "Slow build opening",
                    ha="center", va="top", fontsize=7,
                    color="#94a3b8", style="italic",
                    zorder=7)
    ax_van.set_title("Hook Moment Detection — Where Attention Spikes",
                     fontsize=11, fontweight="bold")
    ax_van.text(0.5, 1.0, "Higher = stronger attention capture. Peaks mark the best moments to use as hooks.",
                transform=ax_van.transAxes, fontsize=7.5, color="gray",
                ha="center", va="top", style="italic")
    ax_van.set_xlabel("Time (seconds)", fontsize=10)
    ax_van.set_ylabel("Attention Capture Strength", fontsize=9)
    ax_van.set_xlim(0, n_ts - 1)
    ax_van.set_ylim(bottom=0)
    ax_van.grid(True, alpha=0.15)
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
    # Find peak brain response across all networks (robust)
    all_net_max = np.abs(tc).max(axis=1)
    peak_t_robust = find_robust_peaks(all_net_max, n_peaks=1)
    peak_t = int(peak_t_robust[0]) if len(peak_t_robust) > 0 else int(np.argmax(all_net_max))
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
    van_peaks = find_robust_peaks(tc[:, 3], n_peaks=3)

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
