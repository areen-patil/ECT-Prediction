"""
ECT Batch Seizure Detection Runner
====================================
Walks through all patient folders under BASE_DIR, finds every CSV file,
runs the full detection pipeline on each, and writes a text log summary.

Folder structure expected:
  BASE_DIR/
    SZPCRC11004/
      2ASZPCRC11004.csv
      2BSZPCRC11004.csv
      ...
    SZPCRC11005/
      2ASZPCRC11005.csv
      ...
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

WINDOW_S = 2.0
STEP_S   = 0.5
N_EEG    = 2

# ══════════════════════════════════════════════════════
# HELPER FUNCTIONS  (same as single-file pipeline)
# ══════════════════════════════════════════════════════
def infer_fs(time_col):
    dt = float(np.median(np.diff(time_col[:2000])))
    return round(1.0 / dt), dt

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
    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(fn(psd[idx], f[idx]))

def hjorth(x):
    d1, d2 = np.diff(x), np.diff(np.diff(x))
    act  = float(np.var(x))
    mob  = float(np.sqrt(np.var(d1) / act)) if act > 0 else 0.0
    comp = float(np.sqrt(np.var(d2) / np.var(d1)) / mob
                 if mob > 0 and np.var(d1) > 0 else 0.0)
    return act, mob, comp

def r_peaks(ecg_seg, fs):
    sos = sp_signal.butter(4, [5, 15], btype="band", fs=fs, output="sos")
    filt = sp_signal.sosfiltfilt(sos, ecg_seg)
    thresh = 0.5 * np.max(filt)
    peaks, _ = sp_signal.find_peaks(filt, height=thresh, distance=int(0.3 * fs))
    return peaks

# ══════════════════════════════════════════════════════
# CORE PIPELINE  (returns a dict of summary values)
# ══════════════════════════════════════════════════════
def run_pipeline(filepath):
    raw  = pd.read_csv(filepath)
    time = raw["Time_s"].values
    FS, _ = infer_fs(time)

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
    ecg_filt   = bandpass(ecg_bc,    0.5, 40, FS)
    motor_filt = lowpass(np.abs(motor_bc), 10, FS)

    # ── Feature extraction ───────────────────────────
    win_samp  = int(WINDOW_S * FS)
    step_samp = int(STEP_S   * FS)
    n_steps   = (len(time) - win_samp) // step_samp + 1

    records = []
    for i in range(n_steps):
        s = i * step_samp
        e = s + win_samp
        if e > len(time):
            break

        seg_eeg   = eeg_filt[s:e, :]
        seg_ecg   = ecg_filt[s:e]
        seg_motor = motor_filt[s:e]

        t_mid = (time[s] + time[e - 1]) / 2.0

        rms_v  = [float(np.sqrt(np.mean(seg_eeg[:, c]**2))) for c in range(N_EEG)]
        ll_v   = [line_length(seg_eeg[:, c])                for c in range(N_EEG)]
        p2p_v  = [float(np.ptp(seg_eeg[:, c]))              for c in range(N_EEG)]
        zc_v   = [zero_crossings(seg_eeg[:, c])             for c in range(N_EEG)]
        clip_v = [clip_frac(seg_eeg[:, c])                  for c in range(N_EEG)]

        delta = np.mean([band_power(seg_eeg[:, c], 1,  4,  FS) for c in range(N_EEG)])
        theta = np.mean([band_power(seg_eeg[:, c], 4,  8,  FS) for c in range(N_EEG)])
        alpha = np.mean([band_power(seg_eeg[:, c], 8,  13, FS) for c in range(N_EEG)])
        beta  = np.mean([band_power(seg_eeg[:, c], 13, 30, FS) for c in range(N_EEG)])

        hj = [hjorth(seg_eeg[:, c]) for c in range(N_EEG)]

        ecg_ll  = line_length(seg_ecg)
        ecg_p2p = float(np.ptp(seg_ecg))
        peaks   = r_peaks(seg_ecg, FS)
        hr_mean = (60.0 / float(np.mean(np.diff(peaks) / FS))
                   if len(peaks) >= 2 else np.nan)

        records.append({
            "t_mid_s":   t_mid,
            "t_start_s": time[s],
            "t_stop_s":  time[e - 1],
            "eeg_rms":   np.mean(rms_v),
            "eeg_ll":    np.mean(ll_v),
            "eeg_p2p":   np.mean(p2p_v),
            "eeg_zc":    np.mean(zc_v),
            "eeg_clip":  np.mean(clip_v),
            "eeg_ch_agree": float(np.std(rms_v)),
            "delta": delta, "theta": theta, "alpha": alpha, "beta": beta,
            "hj_act":  np.mean([h[0] for h in hj]),
            "hj_mob":  np.mean([h[1] for h in hj]),
            "hj_comp": np.mean([h[2] for h in hj]),
            "ecg_rms": float(np.sqrt(np.mean(seg_ecg**2))),
            "ecg_ll":  ecg_ll,
            "ecg_p2p": ecg_p2p,
            "hr_mean": hr_mean,
            "motor_rms": float(np.sqrt(np.mean(seg_motor**2))),
            "motor_p2p": float(np.ptp(seg_motor)),
        })

    feat_df = pd.DataFrame(records)

    # ── Z-score vs baseline (first 60 s) ─────────────
    FEAT_COLS = [
        "eeg_rms", "eeg_ll", "eeg_p2p", "eeg_zc", "eeg_clip", "eeg_ch_agree",
        "delta", "theta", "alpha", "beta",
        "hj_act", "hj_mob", "hj_comp",
        "ecg_rms", "ecg_ll", "ecg_p2p", "hr_mean",
        "motor_rms", "motor_p2p",
    ]
    bl = feat_df.loc[feat_df["t_mid_s"] < 60, FEAT_COLS]
    bl_mean = bl.mean()
    bl_std  = bl.std().replace(0, 1)
    z_df    = (feat_df[FEAT_COLS] - bl_mean) / bl_std
    z_df.columns = ["z_" + c for c in FEAT_COLS]
    feat_df = pd.concat([feat_df, z_df], axis=1)
    feat_df.ffill(inplace=True)
    feat_df.fillna(0, inplace=True)

    # ── Composite seizure score ───────────────────────
    weights = {
        "z_eeg_rms": 1.0, "z_eeg_ll": 1.0, "z_eeg_p2p": 1.0,
        "z_delta":   0.5, "z_theta":  0.5,
        "z_hj_act":  0.8, "z_ecg_ll": 0.4,
        "z_ecg_p2p": 0.4, "z_motor_rms": 0.3,
    }
    feat_df["seizure_score"] = sum(feat_df[c] * w for c, w in weights.items())

    # ── Unsupervised ML ───────────────────────────────
    Z_COLS = [c for c in feat_df.columns if c.startswith("z_")]
    X = feat_df[Z_COLS].values

    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso.fit(X)
    feat_df["iso_flag"] = iso.predict(X) == -1

    gmm = GaussianMixture(n_components=3, covariance_type="full", n_init=5, random_state=42)
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
        # PSI
        ictal_rms = detected["eeg_rms"].mean()
        post_mask = feat_df["t_mid_s"] > offset + 2
        psi = None
        if post_mask.sum() > 5:
            post_rms = feat_df.loc[post_mask, "eeg_rms"].head(10).mean()
            psi = max(0.0, (1.0 - post_rms / ictal_rms) * 100.0)
    else:
        onset = offset = dur = psi = None

    return {
        "fs":              FS,
        "duration_s":      float(time[-1]),
        "n_windows":       len(feat_df),
        "iso_flags":       int(feat_df["iso_flag"].sum()),
        "gmm_flags":       int(feat_df["gmm_ictal"].sum()),
        "score_flags":     int(feat_df["score_flag"].sum()),
        "ensemble_flags":  int(feat_df["seizure_flag"].sum()),
        "seizure_detected": len(detected) > 0,
        "onset_s":         onset,
        "offset_s":        offset,
        "duration_seizure_s": dur,
        "psi_pct":         psi,
    }

# ══════════════════════════════════════════════════════
# BATCH RUNNER
# ══════════════════════════════════════════════════════
def find_all_csvs(base_dir):
    """Walk base_dir and return list of (patient_folder, filename, full_path)."""
    found = []
    for root, dirs, files in os.walk(base_dir):
        dirs.sort()   # consistent ordering
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
            f"  PSI : {str(round(result['psi_pct'], 1)) + '%' if result['psi_pct'] is not None else 'N/A'}",
        ]
    else:
        lines.append("Seizure NOT detected")
    return "\n".join(lines)

def format_error(patient, filename, error_msg, elapsed):
    return "\n".join([
        "─" * 60,
        f"Patient : {patient}",
        f"File    : {filename}",
        f"Runtime : {elapsed:.1f} s",
        f"Status  : ERROR",
        f"Error   : {error_msg}",
    ])

def main():
    all_csvs = find_all_csvs(BASE_DIR)
    total    = len(all_csvs)

    if total == 0:
        print(f"No CSV files found under {BASE_DIR}")
        sys.exit(1)

    print(f"Found {total} CSV files across all patient folders.")
    print(f"Log  → {LOG_FILE}")
    print(f"CSV  → {CSV_FILE}")
    print("=" * 60)

    log_lines  = [
        f"ECT Batch Detection Run",
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
            status  = "OK"
            summary = format_summary(patient, filename, result, elapsed)
            print(f"done in {elapsed:.1f}s  |  "
                  f"{'SEIZURE' if result['seizure_detected'] else 'no seizure'}")
            log_lines.append(summary)

            csv_rows.append({
                "patient":          patient,
                "filename":         filename,
                "filepath":         filepath,
                "status":           status,
                "runtime_s":        round(elapsed, 2),
                "fs_hz":            result["fs"],
                "recording_dur_s":  result["duration_s"],
                "n_windows":        result["n_windows"],
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
            elapsed    = time_module.time() - t0
            error_msg  = f"{type(ex).__name__}: {ex}"
            tb         = traceback.format_exc()
            print(f"ERROR — {error_msg}")
            log_lines.append(format_error(patient, filename, error_msg, elapsed))
            log_lines.append(f"Traceback:\n{tb}")

            csv_rows.append({
                "patient": patient, "filename": filename, "filepath": filepath,
                "status": "ERROR", "runtime_s": round(elapsed, 2),
                "fs_hz": "", "recording_dur_s": "", "n_windows": "",
                "iso_flags": "", "gmm_flags": "", "score_flags": "",
                "ensemble_flags": "", "seizure_detected": "",
                "onset_s": "", "offset_s": "", "seizure_dur_s": "",
                "psi_pct": "", "error": error_msg,
            })

    # ── Write log ──────────────────────────────────────
    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_ok      = sum(1 for r in csv_rows if r["status"] == "OK")
    n_err     = sum(1 for r in csv_rows if r["status"] == "ERROR")
    n_seiz    = sum(1 for r in csv_rows if r.get("seizure_detected") is True)

    footer = "\n".join([
        "=" * 60,
        f"Finished : {finished_at}",
        f"Total    : {total}",
        f"Success  : {n_ok}",
        f"Errors   : {n_err}",
        f"Seizures detected: {n_seiz} / {n_ok} successful files",
    ])
    log_lines.append(footer)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # ── Write CSV ──────────────────────────────────────
    pd.DataFrame(csv_rows).to_csv(CSV_FILE, index=False)

    print("=" * 60)
    print(f"Done.  Success={n_ok}  Errors={n_err}  Seizures={n_seiz}/{n_ok}")
    print(f"Log  → {LOG_FILE}")
    print(f"CSV  → {CSV_FILE}")

if __name__ == "__main__":
    main()
