"""
ECT Batch Seizure Detection Runner — v4
=========================================
Walks through all patient folders under BASE_DIR, finds every CSV file,
runs the full detection pipeline on each, and writes a text log summary.

Features: Standard + Wavelet Decomposition (D1-D4, A4) + HRV LF/HF ratio

Folder structure expected:
  BASE_DIR/
    SZPCRC11004/
      2ASZPCRC11004.csv
      2BSZPCRC11004.csv
      ...
    SZPCRC11005/
      2ASZPCRC11005.csv
      ...

Output:
  BASE_DIR/batch_summary.log   — one entry per file with full summary
  BASE_DIR/batch_results.csv   — one row per file, machine-readable

Usage:
  python ect_batch_runner.py
"""

import os
import sys
import traceback
import time as time_module
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.interpolate import interp1d
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
# CONFIGURATION  ← only thing you need to change
# ══════════════════════════════════════════════════════
BASE_DIR = r"C:\Users\AREEN PATIL\Desktop\self\ECT_EEG_Work\csv_8ch_512Hz\csv_8ch_512Hz"
LOG_FILE = os.path.join(BASE_DIR, "batch_summary.log")
CSV_FILE = os.path.join(BASE_DIR, "batch_results.csv")

WINDOW_S  = 2.0
STEP_S    = 0.5
N_EEG     = 2
HRV_WIN_S = 30.0
FS_RR     = 4.0

# ══════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════
def infer_fs(time_col):
    dt = float(np.median(np.diff(time_col[:2000])))
    return round(1.0 / dt)

def bandpass(sig, lo, hi, fs, order=4):
    sos = sp_signal.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sp_signal.sosfiltfilt(sos, sig)

def notch_filter(sig, freq=50, fs=512, Q=30):
    b, a = sp_signal.iirnotch(freq, Q, fs)
    return sp_signal.filtfilt(b, a, sig)

def lowpass(sig, hi, fs, order=4):
    sos = sp_signal.butter(order, hi, btype="low", fs=fs, output="sos")
    return sp_signal.sosfiltfilt(sos, sig)

def line_length(x):
    return float(np.sum(np.abs(np.diff(x))))

def zero_crossings(x):
    return int(np.sum(np.diff(np.sign(x)) != 0))

def clip_frac(x, thresh=0.95):
    amax = np.max(np.abs(x))
    return 0.0 if amax == 0 else float(np.mean(np.abs(x) > thresh * amax))

def band_power(x, lo, hi, fs):
    f, psd = sp_signal.welch(x, fs=fs, nperseg=min(len(x), 256))
    idx = (f >= lo) & (f <= hi)
    fn  = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(fn(psd[idx], f[idx]))

def hjorth(x):
    d1, d2 = np.diff(x), np.diff(np.diff(x))
    act  = float(np.var(x))
    mob  = float(np.sqrt(np.var(d1) / act)) if act > 0 else 0.0
    comp = float(np.sqrt(np.var(d2) / np.var(d1)) / mob
                 if mob > 0 and np.var(d1) > 0 else 0.0)
    return act, mob, comp

def detect_rpeaks_full(ecg_sig, fs):
    """Adaptive-threshold Pan-Tompkins R-peak detector."""
    sos  = sp_signal.butter(4, [5, 15], btype="band", fs=fs, output="sos")
    filt = sp_signal.sosfiltfilt(sos, ecg_sig)
    pos_vals = filt[filt > 0]
    thresh   = np.percentile(pos_vals, 60) if len(pos_vals) > 0 else 0.5 * np.max(filt)
    peaks, _ = sp_signal.find_peaks(filt, height=thresh, distance=int(0.3 * fs))
    return peaks

def wavelet_decompose(x, fs, n_levels=4):
    """
    4-level DWT approximation via cascaded Butterworth filters.
    Returns [D1, D2, D3, D4, A4]
    At 512 Hz: D1=128-256Hz, D2=64-128Hz, D3=32-64Hz, D4=16-32Hz, A4=0-16Hz
    """
    details = []
    sig     = x.copy()
    cur_fs  = fs
    for _ in range(n_levels):
        cutoff = cur_fs / 4.0
        if cutoff <= 0.5:
            break
        lo = max(cutoff, 0.5)
        sos_hi = sp_signal.butter(4, lo, btype="high", fs=cur_fs, output="sos")
        sos_lo = sp_signal.butter(4, lo, btype="low",  fs=cur_fs, output="sos")
        details.append(sp_signal.sosfiltfilt(sos_hi, sig))
        sig    = sp_signal.sosfiltfilt(sos_lo, sig)[::2]
        cur_fs = cur_fs / 2.0
    details.append(sig)
    return details   # [D1, D2, D3, D4, A4]

WAVELET_LABELS = ["D1_128-256Hz", "D2_64-128Hz", "D3_32-64Hz",
                  "D4_16-32Hz",   "A4_0-16Hz"]

def wavelet_features(coeffs):
    feats = {}
    for c, lbl in zip(coeffs, WAVELET_LABELS):
        energy  = float(np.sum(c**2))
        p       = c**2 / (energy + 1e-12)
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        mav     = float(np.mean(np.abs(c)))
        feats[f"wt_{lbl}_energy"]  = energy
        feats[f"wt_{lbl}_entropy"] = entropy
        feats[f"wt_{lbl}_mav"]     = mav
    return feats

# ══════════════════════════════════════════════════════
# CORE PIPELINE
# ══════════════════════════════════════════════════════
def run_pipeline(filepath):
    raw  = pd.read_csv(filepath)
    time = raw["Time_s"].values
    FS   = infer_fs(time)

    # ── Baseline correction ──────────────────────────
    ecg_bc   = raw["Ch1"].values.astype(float) - raw["Ch5"].values.astype(float)
    eeg1_bc  = raw["Ch2"].values.astype(float) - raw["Ch6"].values.astype(float)
    eeg2_bc  = raw["Ch3"].values.astype(float) - raw["Ch7"].values.astype(float)
    motor_bc = raw["Ch4"].values.astype(float) - raw["Ch8"].values.astype(float)
    eeg_bc   = np.stack([eeg1_bc, eeg2_bc], axis=1)

    # ── Filtering ────────────────────────────────────
    eeg_filt = np.zeros_like(eeg_bc)
    for c in range(N_EEG):
        eeg_filt[:, c] = notch_filter(bandpass(eeg_bc[:, c], 0.5, 70, FS), fs=FS)
    ecg_filt   = bandpass(ecg_bc, 0.5, 40, FS)
    motor_filt = lowpass(np.abs(motor_bc), 10, FS)

    # ── Full-recording R-peaks for HRV ───────────────
    all_rpeaks   = detect_rpeaks_full(ecg_filt, FS)
    rpeak_times  = time[all_rpeaks]
    rr_intervals = np.diff(rpeak_times)
    rr_times     = rpeak_times[1:]

    def compute_lf_hf(t_mid):
        t_start = t_mid - HRV_WIN_S
        mask    = (rr_times >= t_start) & (rr_times <= t_mid)
        if mask.sum() < 8:
            return np.nan, np.nan, np.nan
        rr_t = rr_times[mask]
        rr_v = rr_intervals[mask]
        t_uni = np.arange(rr_t[0], rr_t[-1], 1.0 / FS_RR)
        if len(t_uni) < 16:
            return np.nan, np.nan, np.nan
        rr_rs = interp1d(rr_t, rr_v, kind="linear",
                         bounds_error=False, fill_value="extrapolate")(t_uni)
        nperseg = min(len(rr_rs), int(FS_RR * HRV_WIN_S / 2))
        f, psd  = sp_signal.welch(rr_rs, fs=FS_RR, nperseg=nperseg)
        fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        lf = fn(psd[(f >= 0.04) & (f <= 0.15)], f[(f >= 0.04) & (f <= 0.15)])
        hf = fn(psd[(f >= 0.15) & (f <= 0.40)], f[(f >= 0.15) & (f <= 0.40)])
        return lf, hf, (lf / hf if hf > 0 else np.nan)

    # ── Feature extraction ───────────────────────────
    win_samp  = int(WINDOW_S * FS)
    step_samp = int(STEP_S   * FS)
    n_steps   = (len(time) - win_samp) // step_samp + 1

    records = []
    for i in range(n_steps):
        s, e = i * step_samp, i * step_samp + win_samp
        if e > len(time):
            break

        t_mid     = (time[s] + time[e - 1]) / 2.0
        seg_eeg   = eeg_filt[s:e, :]
        seg_ecg   = ecg_filt[s:e]
        seg_motor = motor_filt[s:e]

        # EEG time-domain
        rms_v = [float(np.sqrt(np.mean(seg_eeg[:, c]**2))) for c in range(N_EEG)]
        ll_v  = [line_length(seg_eeg[:, c])                for c in range(N_EEG)]
        p2p_v = [float(np.ptp(seg_eeg[:, c]))              for c in range(N_EEG)]

        # Band powers
        delta = np.mean([band_power(seg_eeg[:, c], 1,  4,  FS) for c in range(N_EEG)])
        theta = np.mean([band_power(seg_eeg[:, c], 4,  8,  FS) for c in range(N_EEG)])
        alpha = np.mean([band_power(seg_eeg[:, c], 8,  13, FS) for c in range(N_EEG)])
        beta  = np.mean([band_power(seg_eeg[:, c], 13, 30, FS) for c in range(N_EEG)])

        # Hjorth
        hj = [hjorth(seg_eeg[:, c]) for c in range(N_EEG)]

        # Wavelet (mean across 2 EEG channels)
        wt_feats = {}
        for c in range(N_EEG):
            coeffs = wavelet_decompose(seg_eeg[:, c], FS, n_levels=4)
            for k, v in wavelet_features(coeffs).items():
                wt_feats[k] = wt_feats.get(k, 0) + v / N_EEG

        # ECG
        local_peaks = all_rpeaks[(all_rpeaks >= s) & (all_rpeaks < e)]
        hr_mean = (60.0 / float(np.mean(np.diff(local_peaks) / FS))
                   if len(local_peaks) >= 2 else np.nan)
        lf, hf, lf_hf = compute_lf_hf(t_mid)

        rec = {
            "t_mid_s":   t_mid,
            "t_start_s": time[s],
            "t_stop_s":  time[e - 1],
            "eeg_rms":   np.mean(rms_v),
            "eeg_std":   np.mean([np.std(seg_eeg[:, c]) for c in range(N_EEG)]),
            "eeg_ll":    np.mean(ll_v),
            "eeg_p2p":   np.mean(p2p_v),
            "eeg_zc":    np.mean([zero_crossings(seg_eeg[:, c]) for c in range(N_EEG)]),
            "eeg_clip":  np.mean([clip_frac(seg_eeg[:, c])      for c in range(N_EEG)]),
            "eeg_ch_agree": float(np.std(rms_v)),
            "delta": delta, "theta": theta, "alpha": alpha, "beta": beta,
            "hj_act":  np.mean([h[0] for h in hj]),
            "hj_mob":  np.mean([h[1] for h in hj]),
            "hj_comp": np.mean([h[2] for h in hj]),
            "ecg_rms": float(np.sqrt(np.mean(seg_ecg**2))),
            "ecg_ll":  line_length(seg_ecg),
            "ecg_p2p": float(np.ptp(seg_ecg)),
            "hr_mean": hr_mean,
            "hrv_lf":    lf,
            "hrv_hf":    hf,
            "hrv_lf_hf": lf_hf,
            "motor_rms": float(np.sqrt(np.mean(seg_motor**2))),
            "motor_p2p": float(np.ptp(seg_motor)),
        }
        rec.update(wt_feats)
        records.append(rec)

    feat_df = pd.DataFrame(records)

    # ── Z-score vs baseline (first 60 s) ─────────────
    FEAT_COLS = [c for c in feat_df.columns
                 if c not in ("t_start_s", "t_stop_s", "t_mid_s")]
    bl      = feat_df.loc[feat_df["t_mid_s"] < 60, FEAT_COLS]
    bl_mean = bl.mean()
    bl_std  = bl.std().replace(0, 1)
    z_df    = (feat_df[FEAT_COLS] - bl_mean) / bl_std
    z_df.columns = ["z_" + c for c in FEAT_COLS]
    feat_df = pd.concat([feat_df, z_df], axis=1)
    feat_df.ffill(inplace=True)
    feat_df.fillna(0, inplace=True)

    # ── Composite seizure score ───────────────────────
    weights = {
        "z_eeg_rms":   1.0, "z_eeg_ll":  1.0, "z_eeg_p2p":  1.0,
        "z_delta":     0.5, "z_theta":   0.5, "z_hj_act":   0.8,
        "z_ecg_ll":    0.4, "z_ecg_p2p": 0.4, "z_motor_rms": 0.3,
        "z_wt_D4_16-32Hz_energy": 0.4,
        "z_wt_A4_0-16Hz_energy":  0.4,
        "z_hrv_lf_hf": 0.3,
    }
    feat_df["seizure_score"] = sum(
        feat_df[c] * w for c, w in weights.items() if c in feat_df.columns
    )

    # ── Unsupervised ML ───────────────────────────────
    Z_COLS = [c for c in feat_df.columns if c.startswith("z_")]
    X = feat_df[Z_COLS].values

    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso.fit(X)
    feat_df["iso_flag"] = iso.predict(X) == -1

    gmm = GaussianMixture(n_components=3, covariance_type="full",
                          n_init=5, random_state=42)
    gmm.fit(X)
    feat_df["gmm_cluster"] = gmm.predict(X)
    ictal_cl = feat_df.groupby("gmm_cluster")["seizure_score"].mean().idxmax()
    feat_df["gmm_ictal"] = feat_df["gmm_cluster"] == ictal_cl

    thresh = feat_df["seizure_score"].mean() + 2.0 * feat_df["seizure_score"].std()
    feat_df["score_flag"] = feat_df["seizure_score"] > thresh

    # ── Ensemble vote ─────────────────────────────────
    feat_df["vote"] = (feat_df["iso_flag"].astype(int)
                     + feat_df["gmm_ictal"].astype(int)
                     + feat_df["score_flag"].astype(int))
    feat_df["seizure_flag"] = feat_df["vote"] >= 2

    detected = feat_df[feat_df["seizure_flag"]]
    if len(detected):
        onset  = float(detected["t_start_s"].min())
        offset = float(detected["t_stop_s"].max())
        dur    = offset - onset
        ictal_rms = detected["eeg_rms"].mean()
        post_mask = feat_df["t_mid_s"] > offset + 2
        psi = None
        if post_mask.sum() > 5:
            post_rms = feat_df.loc[post_mask, "eeg_rms"].head(10).mean()
            psi = max(0.0, (1.0 - post_rms / ictal_rms) * 100.0)
    else:
        onset = offset = dur = psi = None

    return {
        "fs":                 FS,
        "duration_s":         float(time[-1]),
        "n_windows":          len(feat_df),
        "n_rpeaks":           len(all_rpeaks),
        "avg_hr_bpm":         round(60.0 / float(np.mean(rr_intervals)), 1) if len(rr_intervals) > 0 else None,
        "iso_flags":          int(feat_df["iso_flag"].sum()),
        "gmm_flags":          int(feat_df["gmm_ictal"].sum()),
        "score_flags":        int(feat_df["score_flag"].sum()),
        "ensemble_flags":     int(feat_df["seizure_flag"].sum()),
        "seizure_detected":   len(detected) > 0,
        "onset_s":            onset,
        "offset_s":           offset,
        "duration_seizure_s": dur,
        "psi_pct":            psi,
    }

# ══════════════════════════════════════════════════════
# BATCH RUNNER
# ══════════════════════════════════════════════════════
def find_all_csvs(base_dir):
    found = []
    for root, dirs, files in os.walk(base_dir):
        dirs.sort()
        for f in sorted(files):
            if f.lower().endswith(".csv"):
                patient = os.path.basename(root)
                found.append((patient, f, os.path.join(root, f)))
    return found

def format_summary(patient, filename, result, elapsed):
    lines = [
        "─" * 60,
        f"Patient : {patient}",
        f"File    : {filename}",
        f"Runtime : {elapsed:.1f} s",
        f"Fs      : {result['fs']} Hz",
        f"Duration: {result['duration_s']:.1f} s",
        f"Windows : {result['n_windows']}",
        f"R-peaks : {result['n_rpeaks']}  (avg HR: {result['avg_hr_bpm']} bpm)",
        f"Iso Forest flags : {result['iso_flags']}",
        f"GMM flags        : {result['gmm_flags']}",
        f"Score flags      : {result['score_flags']}",
        f"Ensemble flags   : {result['ensemble_flags']}",
    ]
    if result["seizure_detected"]:
        lines += [
            f"Seizure DETECTED ✓",
            f"  Onset    : {result['onset_s']:.1f} s",
            f"  Offset   : {result['offset_s']:.1f} s",
            f"  Duration : {result['duration_seizure_s']:.1f} s",
            f"  PSI      : {str(round(result['psi_pct'], 1)) + '%' if result['psi_pct'] is not None else 'N/A'}",
        ]
    else:
        lines.append("Seizure NOT detected")
    return "\n".join(lines)

def format_error(patient, filename, error_msg, tb, elapsed):
    return "\n".join([
        "─" * 60,
        f"Patient : {patient}",
        f"File    : {filename}",
        f"Runtime : {elapsed:.1f} s",
        f"Status  : ERROR",
        f"Error   : {error_msg}",
        f"Traceback:\n{tb}",
    ])

def main():
    all_csvs = find_all_csvs(BASE_DIR)
    total    = len(all_csvs)

    if total == 0:
        print(f"No CSV files found under {BASE_DIR}")
        sys.exit(1)

    print(f"Found {total} CSV files across all patient folders.")
    print(f"Log → {LOG_FILE}")
    print(f"CSV → {CSV_FILE}")
    print("=" * 60)

    log_lines = [
        "ECT Batch Detection Run — v4 (Wavelet + LF/HF)",
        f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Base dir: {BASE_DIR}",
        f"Total files: {total}",
        "=" * 60,
    ]
    csv_rows = []

    for idx, (patient, filename, filepath) in enumerate(all_csvs, 1):
        print(f"[{idx:3d}/{total}]  {patient}/{filename} ...", end=" ", flush=True)
        t0 = time_module.time()

        try:
            result  = run_pipeline(filepath)
            elapsed = time_module.time() - t0
            print(f"done in {elapsed:.1f}s  |  "
                  f"{'SEIZURE' if result['seizure_detected'] else 'no seizure'}  |  "
                  f"HR={result['avg_hr_bpm']} bpm")
            log_lines.append(format_summary(patient, filename, result, elapsed))

            csv_rows.append({
                "patient":          patient,
                "filename":         filename,
                "filepath":         filepath,
                "status":           "OK",
                "runtime_s":        round(elapsed, 2),
                "fs_hz":            result["fs"],
                "recording_dur_s":  result["duration_s"],
                "n_windows":        result["n_windows"],
                "n_rpeaks":         result["n_rpeaks"],
                "avg_hr_bpm":       result["avg_hr_bpm"],
                "iso_flags":        result["iso_flags"],
                "gmm_flags":        result["gmm_flags"],
                "score_flags":      result["score_flags"],
                "ensemble_flags":   result["ensemble_flags"],
                "seizure_detected": result["seizure_detected"],
                "onset_s":          result["onset_s"],
                "offset_s":         result["offset_s"],
                "seizure_dur_s":    result["duration_seizure_s"],
                "psi_pct":          result["psi_pct"],
                "error":            "",
            })

        except Exception as ex:
            elapsed   = time_module.time() - t0
            error_msg = f"{type(ex).__name__}: {ex}"
            tb        = traceback.format_exc()
            print(f"ERROR — {error_msg}")
            log_lines.append(format_error(patient, filename, error_msg, tb, elapsed))

            csv_rows.append({
                "patient": patient, "filename": filename, "filepath": filepath,
                "status": "ERROR", "runtime_s": round(elapsed, 2),
                "fs_hz": "", "recording_dur_s": "", "n_windows": "",
                "n_rpeaks": "", "avg_hr_bpm": "",
                "iso_flags": "", "gmm_flags": "", "score_flags": "",
                "ensemble_flags": "", "seizure_detected": "",
                "onset_s": "", "offset_s": "", "seizure_dur_s": "",
                "psi_pct": "", "error": error_msg,
            })

    # ── Write log ──────────────────────────────────────
    n_ok   = sum(1 for r in csv_rows if r["status"] == "OK")
    n_err  = sum(1 for r in csv_rows if r["status"] == "ERROR")
    n_seiz = sum(1 for r in csv_rows if r.get("seizure_detected") is True)

    footer = "\n".join([
        "=" * 60,
        f"Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total    : {total}",
        f"Success  : {n_ok}",
        f"Errors   : {n_err}",
        f"Seizures : {n_seiz} / {n_ok} successful files",
    ])
    log_lines.append(footer)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    pd.DataFrame(csv_rows).to_csv(CSV_FILE, index=False)

    print("=" * 60)
    print(f"Done.  Success={n_ok}  Errors={n_err}  Seizures={n_seiz}/{n_ok}")
    print(f"Log → {LOG_FILE}")
    print(f"CSV → {CSV_FILE}")

if __name__ == "__main__":
    main()
