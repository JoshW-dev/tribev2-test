"""Quick sanity check: old vs new peak detection on a single _tc.npy file.

Usage:
    python paper/test_peak_fix.py paper/comparison_data/BusinessEdLeilaHarmozi_tc.npy
    python paper/test_peak_fix.py path/to/any_tc.npy
"""

import sys
import numpy as np
from scipy.signal import detrend, find_peaks as scipy_find_peaks


def find_robust_peaks(timecourse, n_peaks=3, boundary_trim=4, min_distance=3):
    n = len(timecourse)
    if n == 0:
        return np.array([], dtype=int)
    trim = min(boundary_trim, max(n // 4, 1))
    detrended = detrend(timecourse, type='linear')
    std = detrended.std()
    z = detrended / std if std > 0 else detrended
    interior = np.abs(z[trim:n - trim])
    peaks, props = scipy_find_peaks(interior, distance=min_distance, prominence=0.1)
    if len(peaks) >= n_peaks:
        top_idx = np.argsort(props['prominences'])[::-1][:n_peaks]
        peak_indices = peaks[top_idx] + trim
    else:
        ranked = np.argsort(interior)[::-1]
        selected = []
        for idx in ranked:
            orig = idx + trim
            if all(abs(orig - s) >= min_distance for s in selected):
                selected.append(orig)
            if len(selected) == n_peaks:
                break
        peak_indices = np.array(selected, dtype=int)
    if len(peak_indices) > 0:
        magnitudes = np.abs(z[peak_indices])
        peak_indices = peak_indices[np.argsort(magnitudes)[::-1]]
    return peak_indices


def old_peaks(tc_col, n=3):
    return np.argsort(np.abs(tc_col))[-n:][::-1]


YEO = ["Visual", "Somatomotor", "Dorsal Attn", "Ventral Attn",
       "Limbic", "Frontoparietal", "Default Mode"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    tc = np.load(sys.argv[1])
    n_ts = tc.shape[0]
    print(f"Loaded: {sys.argv[1]}  shape={tc.shape}  ({n_ts} seconds)\n")

    # --- Ventral Attention (the main hook/salience network) ---
    van = tc[:, 3]
    op = old_peaks(van)
    np_ = find_robust_peaks(van)

    print("=" * 60)
    print("VENTRAL ATTENTION (salience/hook detection)")
    print("=" * 60)
    print(f"  Old peaks (raw argmax):  {', '.join(f'{p}s' for p in op)}")
    print(f"  New peaks (robust):      {', '.join(f'{p}s' for p in np_)}")
    print()

    end_zone = int(n_ts * 0.9)
    old_end = sum(1 for p in op if p >= end_zone)
    new_end = sum(1 for p in np_ if p >= end_zone)
    print(f"  Peaks in final 10% (>={end_zone}s):  old={old_end}/3  new={new_end}/3")
    print()

    # --- All 7 networks ---
    print("=" * 60)
    print("ALL NETWORKS — per-network peak comparison")
    print("=" * 60)
    print(f"  {'Network':<16} {'Old peak':>10} {'New peak':>10}  {'Moved?'}")
    print(f"  {'-'*16} {'-'*10} {'-'*10}  {'-'*6}")
    for i in range(7):
        o = int(old_peaks(tc[:, i], n=1)[0])
        n = find_robust_peaks(tc[:, i], n_peaks=1)
        n = int(n[0]) if len(n) > 0 else o
        flag = "<-- FIXED" if abs(o - n) > 3 else ""
        print(f"  {YEO[i]:<16} {f'{o}s':>10} {f'{n}s':>10}  {flag}")

    # --- Drift diagnostic ---
    print()
    print("=" * 60)
    print("DRIFT DIAGNOSTIC")
    print("=" * 60)
    first_q = tc[:n_ts // 4]
    last_q = tc[3 * n_ts // 4:]
    for i in range(7):
        mean_first = np.abs(first_q[:, i]).mean()
        mean_last = np.abs(last_q[:, i]).mean()
        ratio = mean_last / mean_first if mean_first > 0 else float('inf')
        flag = " <-- DRIFT" if ratio > 1.5 else ""
        print(f"  {YEO[i]:<16}  first-quarter mean={mean_first:.4f}  "
              f"last-quarter mean={mean_last:.4f}  ratio={ratio:.2f}{flag}")
