# Peak Detection Fix — Desktop Next Steps

## Problem

Peak moment detection (salience/hook moments) is biased toward the end of videos. 4 of 5 test videos show peaks clustering in the final few seconds:

| Video | Duration | Old Peaks | Issue |
|---|---|---|---|
| Leila Hormozi | 49s | 48s, 47s, 45s | All 3 in last 4s |
| Elon AI | 60s | 59s, 58s, 29s | 2 of 3 in last 2s |
| Perfume UGC | 35s | 34s, 33s, 0s | 2 of 3 in last 2s |
| Sanitary Pad | 25s | 24s, 17s, 21s | Borderline |
| Ice Cutter | 48s | 21s, 19s, 14s | OK (mid-video) |

## Root Cause

The old peak detection (`np.argsort(np.abs(signal))[-3:][::-1]`) uses raw absolute values with no correction for:

1. **Linear drift** — transformer positional embeddings and model dynamics produce a slow ramp in predicted activations over time. Raw argmax picks up this drift as "peaks."
2. **Conv1d zero-padding edge artifacts** — the `TemporalSmoothing` layer (kernel_size=9, padding=4) distorts the first/last 4 timesteps.
3. **No minimum spacing** — adjacent timesteps (e.g., 47s, 48s, 49s) can all appear as separate "peaks" when they're really one event.

## What Was Fixed (already committed)

A new `find_robust_peaks()` function replaces raw argmax in three files:

- `app.py` — canonical implementation + `generate_cognitive_summary()` updated
- `paper/generate_content_figures.py` — per-video analysis figures + insights summary
- `paper/generate_figures.py` — Figure 1 (overall peak) + Figure 2 (ventral attention peaks)

The function applies:
1. `scipy.signal.detrend` — removes linear drift
2. Z-scoring — normalizes peaks relative to the video's own distribution
3. Boundary trimming — excludes first/last 4 timesteps
4. `scipy.signal.find_peaks` with prominence — finds true local maxima
5. Minimum distance of 3s between peaks — prevents clustering

No model re-inference needed. All fixes are post-processing on cached `_tc.npy` arrays.

## Steps to Run on Desktop

### 1. Pull latest code

```bash
cd /path/to/tribev2-test
git pull origin main
```

### 2. Sanity check on one video

Run the comparison script on the worst offender (Leila Hormozi):

```bash
python paper/test_peak_fix.py paper/comparison_data/BusinessEdLeilaHarmozi_tc.npy
```

This prints old vs new peaks, drift diagnostics, and flags any networks that shifted. Confirm:
- New ventral attention peaks are **not** all in the last 4 seconds
- Drift diagnostic shows a ratio > 1.5 for at least some networks (confirming the ramp exists)

### 3. Run on all videos

If the sanity check looks good, run the remaining videos:

```bash
python paper/test_peak_fix.py paper/comparison_data/ElonAI_tc.npy
python paper/test_peak_fix.py paper/comparison_data/PerfumeUGCInterview_tc.npy
python paper/test_peak_fix.py paper/comparison_data/sanitaryPadProductDemo_tc.npy
python paper/test_peak_fix.py paper/comparison_data/viralJapaneseIceCutter_tc.npy
```

### 4. Regenerate figures

```bash
python paper/generate_content_figures.py
python paper/generate_figures.py
```

This overwrites the PNGs in `paper/figures/` with corrected peak annotations.

### 5. Update the paper

After confirming the new peak values, update `paper/neural_content_intelligence.md`:
- Table 2 peak timestamps
- Per-video analysis sections (hook moments)
- Summary table at end of Section 6

### 6. Delete the test script

```bash
rm paper/test_peak_fix.py
```
