"""
ECT Seizure Detection Pipeline — v4
=====================================
Patient : 2ASZPCRC11004
Fs      : inferred from Time_s column

Channel mapping
───────────────
  Ch1 = ECG        Ch5 = baseline(ECG)
  Ch2 = EEG1       Ch6 = baseline(EEG1)
  Ch3 = EEG2       Ch7 = baseline(EEG2)
  Ch4 = Motor      Ch8 = baseline(Motor)

New features added in v4
─────────────────────────
  EEG — Wavelet decomposition (4-level DWT via cascaded Butterworth):
    D1: 128–256 Hz  D2: 64–128 Hz  D3: 32–64 Hz  D4: 16–32 Hz  A4: 0–16 Hz
    Per level → energy, wavelet entropy, mean absolute value

  ECG — LF/HF HRV ratio:
    RR intervals resampled to 4 Hz tachogram over a 30s rolling window
    LF: 0.04–0.15 Hz  |  HF: 0.15–0.4 Hz  |  LF/HF ratio
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as sp_signal
from scipy.interpolate import interp1d
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
# 0. LOAD & INFER Fs
# ══════════════════════════════════════════════════════
print("Loading data...")
raw = pd.read_csv("/mnt/user-data/uploads/2ASZPCRC11004.csv")

dt = float(np.median(np.diff(raw["Time_s"].values[:2000])))
FS = round(1.0 / dt)
print(f"  Inferred Fs = {FS} Hz  |  Duration: {raw['Time_s'].values[-1]:.1f}s  |  Shape: {raw.shape}")

time  = raw["Time_s"].values
N_EEG = 2

WINDOW_S  = 2.0
STEP_S    = 0.5
HRV_WIN_S = 30.0   # rolling window for LF/HF computation
FS_RR     = 4.0    # resample RR tachogram to 4 Hz (HRV standard)

# ══════════════════════════════════════════════════════
# 1. BASELINE CORRECTION
# ══════════════════════════════════════════════════════
print("Applying baseline correction...")

ecg_bc   = raw["Ch1"].values.astype(float) - raw["Ch5"].values.astype(float)
eeg1_bc  = raw["Ch2"].values.astype(float) - raw["Ch6"].values.astype(float)
eeg2_bc  = raw["Ch3"].values.astype(float) - raw["Ch7"].values.astype(float)
motor_bc = raw["Ch4"].values.astype(float) - raw["Ch8"].values.astype(float)
eeg_bc   = np.stack([eeg1_bc, eeg2_bc], axis=1)

# ══════════════════════════════════════════════════════
# 2. FILTERING
# ══════════════════════════════════════════════════════
print("Filtering...")

def bandpass(sig, lo, hi, fs, order=4):
    sos = sp_signal.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sp_signal.sosfiltfilt(sos, sig)

def notch_filter(sig, freq=50, fs=512, Q=30):
    b, a = sp_signal.iirnotch(freq, Q, fs)
    return sp_signal.filtfilt(b, a, sig)

def lowpass(sig, hi, fs, order=4):
    sos = sp_signal.butter(order, hi, btype="low", fs=fs, output="sos")
    return sp_signal.sosfiltfilt(sos, sig)

eeg_filt = np.zeros_like(eeg_bc)
for c in range(N_EEG):
    eeg_filt[:, c] = notch_filter(bandpass(eeg_bc[:, c], 0.5, 70, FS), fs=FS)

ecg_filt   = bandpass(ecg_bc, 0.5, 40, FS)
motor_filt = lowpass(np.abs(motor_bc), 10, FS)

# ══════════════════════════════════════════════════════
# 3. PRECOMPUTE R-PEAKS (for LF/HF across full recording)
# ══════════════════════════════════════════════════════
print("Detecting R-peaks across full ECG for HRV...")

def detect_rpeaks_full(ecg_sig, fs):
    sos  = sp_signal.butter(4, [5, 15], btype="band", fs=fs, output="sos")
    filt = sp_signal.sosfiltfilt(sos, ecg_sig)
    # Adaptive threshold: 60th percentile of positive values
    # avoids global max being skewed by seizure artefacts
    pos_vals = filt[filt > 0]
    thresh   = np.percentile(pos_vals, 60) if len(pos_vals) > 0 else 0.5 * np.max(filt)
    peaks, _ = sp_signal.find_peaks(filt, height=thresh, distance=int(0.3 * fs))
    return peaks

all_rpeaks = detect_rpeaks_full(ecg_filt, FS)
rpeak_times = time[all_rpeaks]              # timestamps of each R-peak
rr_intervals = np.diff(rpeak_times)         # RR in seconds
rr_times     = rpeak_times[1:]              # time of each RR interval

print(f"  {len(all_rpeaks)} R-peaks detected  |  "
      f"Avg HR: {60/np.mean(rr_intervals):.0f} bpm")

def compute_lf_hf(t_mid, hrv_win=HRV_WIN_S, fs_rr=FS_RR):
    """
    Compute LF/HF ratio using RR intervals in a rolling window
    ending at t_mid. Returns (lf_power, hf_power, lf_hf_ratio).
    """
    t_end   = t_mid
    t_start = t_end - hrv_win
    mask    = (rr_times >= t_start) & (rr_times <= t_end)

    if mask.sum() < 8:   # need at least 8 RR intervals
        return np.nan, np.nan, np.nan

    rr_t = rr_times[mask]
    rr_v = rr_intervals[mask]

    # Resample to uniform grid at fs_rr Hz
    t_uniform = np.arange(rr_t[0], rr_t[-1], 1.0 / fs_rr)
    if len(t_uniform) < 16:
        return np.nan, np.nan, np.nan

    interp    = interp1d(rr_t, rr_v, kind="linear", bounds_error=False,
                         fill_value="extrapolate")
    rr_resampled = interp(t_uniform)

    # Welch PSD
    nperseg = min(len(rr_resampled), int(fs_rr * hrv_win / 2))
    f, psd  = sp_signal.welch(rr_resampled, fs=fs_rr, nperseg=nperseg)

    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    lf_idx = (f >= 0.04) & (f <= 0.15)
    hf_idx = (f >= 0.15) & (f <= 0.40)

    lf_pwr = fn(psd[lf_idx], f[lf_idx]) if lf_idx.sum() > 0 else np.nan
    hf_pwr = fn(psd[hf_idx], f[hf_idx]) if hf_idx.sum() > 0 else np.nan
    ratio  = lf_pwr / hf_pwr if (hf_pwr and hf_pwr > 0) else np.nan

    return lf_pwr, hf_pwr, ratio

# ══════════════════════════════════════════════════════
# 4. WAVELET DECOMPOSITION (4-level DWT via cascaded filters)
# ══════════════════════════════════════════════════════
# At 512 Hz, Nyquist = 256 Hz
# Level 1 detail (D1): 128–256 Hz  ← high-gamma / artefact
# Level 2 detail (D2):  64–128 Hz  ← gamma
# Level 3 detail (D3):  32– 64 Hz  ← low-gamma / high-beta
# Level 4 detail (D4):  16– 32 Hz  ← beta
# Level 4 approx (A4):   0– 16 Hz  ← delta + theta + alpha

def wavelet_decompose(x, fs, n_levels=4):
    """
    Approximate 4-level DWT using cascaded Butterworth filters.
    Returns list: [D1, D2, D3, D4, A4]
    """
    details = []
    sig = x.copy()
    cur_fs = fs
    for lvl in range(n_levels):
        cutoff = cur_fs / 4.0    # half of current Nyquist
        if cutoff <= 0.5:        # can't filter below 0.5 Hz reliably
            break
        hi  = min(cutoff * 2.0 - 0.5, cur_fs / 2.0 - 1.0)
        lo  = max(cutoff, 0.5)
        # detail = highpass
        sos_hi = sp_signal.butter(4, lo, btype="high", fs=cur_fs, output="sos")
        detail = sp_signal.sosfiltfilt(sos_hi, sig)
        details.append(detail)
        # approx = lowpass, downsample by 2
        sos_lo = sp_signal.butter(4, lo, btype="low", fs=cur_fs, output="sos")
        sig    = sp_signal.sosfiltfilt(sos_lo, sig)[::2]
        cur_fs = cur_fs / 2.0

    details.append(sig)   # final approximation
    return details         # [D1, D2, D3, D4, A4]

WAVELET_LABELS = ["D1_128-256Hz", "D2_64-128Hz", "D3_32-64Hz",
                  "D4_16-32Hz",   "A4_0-16Hz"]

def wavelet_features(coeffs):
    """Per-level: energy, wavelet entropy, mean absolute value."""
    feats = {}
    for i, (c, lbl) in enumerate(zip(coeffs, WAVELET_LABELS)):
        energy  = float(np.sum(c**2))
        p       = c**2 / (energy + 1e-12)
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        mav     = float(np.mean(np.abs(c)))
        feats[f"wt_{lbl}_energy"]  = energy
        feats[f"wt_{lbl}_entropy"] = entropy
        feats[f"wt_{lbl}_mav"]     = mav
    return feats

# ══════════════════════════════════════════════════════
# 5. STANDARD HELPER FUNCTIONS
# ══════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════
# 6. FEATURE EXTRACTION
# ══════════════════════════════════════════════════════
print("Extracting features (wavelet + LF/HF added)...")

win_samp  = int(WINDOW_S * FS)
step_samp = int(STEP_S   * FS)
n_steps   = (len(time) - win_samp) // step_samp + 1

records = []
for i in range(n_steps):
    s, e = i * step_samp, i * step_samp + win_samp
    if e > len(time):
        break

    t_start  = time[s]
    t_stop   = time[e - 1]
    t_mid    = (t_start + t_stop) / 2.0
    seg_eeg  = eeg_filt[s:e, :]
    seg_ecg  = ecg_filt[s:e]
    seg_motor = motor_filt[s:e]

    # ── Standard EEG features ────────────────────────
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
    hj_act  = np.mean([h[0] for h in hj])
    hj_mob  = np.mean([h[1] for h in hj])
    hj_comp = np.mean([h[2] for h in hj])

    # ── NEW: Wavelet decomposition (mean across 2 EEG channels) ──
    wt_feats_combined = {}
    for c in range(N_EEG):
        coeffs = wavelet_decompose(seg_eeg[:, c], FS, n_levels=4)
        wf = wavelet_features(coeffs)
        for k, v in wf.items():
            wt_feats_combined[k] = wt_feats_combined.get(k, 0) + v / N_EEG

    # ── Standard ECG features ─────────────────────────
    ecg_rms = float(np.sqrt(np.mean(seg_ecg**2)))
    ecg_ll  = line_length(seg_ecg)
    ecg_p2p = float(np.ptp(seg_ecg))

    # Instantaneous HR from local R-peaks
    local_peaks = all_rpeaks[(all_rpeaks >= s) & (all_rpeaks < e)]
    if len(local_peaks) >= 2:
        rr_local = np.diff(local_peaks) / FS
        hr_mean  = 60.0 / float(np.mean(rr_local))
    else:
        hr_mean = np.nan

    # ── NEW: LF/HF HRV ratio (30s rolling window) ────
    lf_pwr, hf_pwr, lf_hf = compute_lf_hf(t_mid)

    # ── Motor ─────────────────────────────────────────
    motor_rms = float(np.sqrt(np.mean(seg_motor**2)))
    motor_p2p = float(np.ptp(seg_motor))

    rec = {
        "t_start_s": t_start, "t_stop_s": t_stop, "t_mid_s": t_mid,
        # EEG time-domain
        "eeg_rms":       np.mean(rms_v),
        "eeg_std":       np.mean([np.std(seg_eeg[:, c]) for c in range(N_EEG)]),
        "eeg_ll":        np.mean(ll_v),
        "eeg_p2p":       np.mean(p2p_v),
        "eeg_zc":        np.mean(zc_v),
        "eeg_clip":      np.mean(clip_v),
        "eeg_ch_agree":  float(np.std(rms_v)),
        # Band powers
        "delta": delta, "theta": theta, "alpha": alpha, "beta": beta,
        # Hjorth
        "hj_act": hj_act, "hj_mob": hj_mob, "hj_comp": hj_comp,
        # ECG
        "ecg_rms": ecg_rms, "ecg_ll": ecg_ll, "ecg_p2p": ecg_p2p,
        "hr_mean": hr_mean,
        # HRV — LF/HF
        "hrv_lf":    lf_pwr,
        "hrv_hf":    hf_pwr,
        "hrv_lf_hf": lf_hf,
        # Motor
        "motor_rms": motor_rms, "motor_p2p": motor_p2p,
    }
    rec.update(wt_feats_combined)   # add wavelet features
    records.append(rec)

feat_df = pd.DataFrame(records)
print(f"  {len(feat_df)} windows  |  {len(feat_df.columns)} features per window")

# ══════════════════════════════════════════════════════
# 7. Z-SCORE vs BASELINE (first 60 s)
# ══════════════════════════════════════════════════════
print("Z-score normalising (baseline = first 60 s)...")

FEAT_COLS = [c for c in feat_df.columns
             if c not in ("t_start_s", "t_stop_s", "t_mid_s")]

bl      = feat_df.loc[feat_df["t_mid_s"] < 60, FEAT_COLS]
bl_mean = bl.mean()
bl_std  = bl.std().replace(0, 1)

z_df = (feat_df[FEAT_COLS] - bl_mean) / bl_std
z_df.columns = ["z_" + c for c in FEAT_COLS]
feat_df = pd.concat([feat_df, z_df], axis=1)
feat_df.ffill(inplace=True)
feat_df.fillna(0, inplace=True)

# ══════════════════════════════════════════════════════
# 8. COMPOSITE SEIZURE SCORE
# ══════════════════════════════════════════════════════
weights = {
    "z_eeg_rms":   1.0,
    "z_eeg_ll":    1.0,
    "z_eeg_p2p":   1.0,
    "z_delta":     0.5,
    "z_theta":     0.5,
    "z_hj_act":    0.8,
    "z_ecg_ll":    0.4,
    "z_ecg_p2p":   0.4,
    "z_motor_rms": 0.3,
    # Wavelet — delta/theta range (D4+A4) energy rises during seizure
    "z_wt_D4_16-32Hz_energy":  0.4,
    "z_wt_A4_0-16Hz_energy":   0.4,
    # LF/HF — autonomic shift during seizure
    "z_hrv_lf_hf": 0.3,
}
feat_df["seizure_score"] = sum(
    feat_df[c] * w for c, w in weights.items() if c in feat_df.columns
)

# ══════════════════════════════════════════════════════
# 9. UNSUPERVISED ML
# ══════════════════════════════════════════════════════
print("Running unsupervised ML...")
Z_COLS = [c for c in feat_df.columns if c.startswith("z_")]
X = feat_df[Z_COLS].values

iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
iso.fit(X)
feat_df["iso_score"] = -iso.score_samples(X)
feat_df["iso_flag"]  = iso.predict(X) == -1

gmm = GaussianMixture(n_components=3, covariance_type="full", n_init=5, random_state=42)
gmm.fit(X)
feat_df["gmm_cluster"] = gmm.predict(X)
ictal_cl = feat_df.groupby("gmm_cluster")["seizure_score"].mean().idxmax()
feat_df["gmm_ictal"] = feat_df["gmm_cluster"] == ictal_cl

thresh = feat_df["seizure_score"].mean() + 2.0 * feat_df["seizure_score"].std()
feat_df["score_flag"] = feat_df["seizure_score"] > thresh

# ══════════════════════════════════════════════════════
# 10. ENSEMBLE VOTE
# ══════════════════════════════════════════════════════
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
    print(f"\n  ✓ Seizure  onset={onset:.1f}s  offset={offset:.1f}s  duration={dur:.1f}s"
          + (f"  PSI≈{psi:.1f}%" if psi else ""))
else:
    onset = offset = dur = psi = None
    print("\n  ⚠  No seizure detected.")

# ══════════════════════════════════════════════════════
# 11. SAVE CSV
# ══════════════════════════════════════════════════════
out_csv = "/mnt/user-data/outputs/2A_extracted_features_v4.csv"
feat_df.to_csv(out_csv, index=False)
print(f"\nFeatures saved → {out_csv}  ({len(feat_df.columns)} columns)")

# ══════════════════════════════════════════════════════
# 12. VISUALISATION
# ══════════════════════════════════════════════════════
print("Generating plots...")

t      = feat_df["t_mid_s"].values
flag_t = feat_df.loc[feat_df["seizure_flag"], "t_mid_s"].values
ds     = 10

def shade(ax):
    if len(flag_t):
        ax.axvspan(flag_t[0], flag_t[-1], color="#EF5350", alpha=0.10,
                   label="Detected seizure", zorder=0)

BLUE   = "#2196F3"; GREEN  = "#4CAF50"; RED    = "#EF5350"
ORANGE = "#FF9800"; PURPLE = "#9C27B0"

fig = plt.figure(figsize=(20, 30))
gs  = gridspec.GridSpec(8, 2, figure=fig, hspace=0.55, wspace=0.35)

# 1 — Raw signals
ax1 = fig.add_subplot(gs[0, :])
off = 300
ax1.plot(time[::ds], (eeg1_bc - eeg1_bc.mean())[::ds] + off, color=BLUE,  lw=0.4, label="EEG1")
ax1.plot(time[::ds], (eeg2_bc - eeg2_bc.mean())[::ds] - off, color=GREEN, lw=0.4, label="EEG2")
ax1r = ax1.twinx()
ax1r.plot(time[::ds], (ecg_bc - ecg_bc.mean())[::ds], color=RED, lw=0.3, alpha=0.45, label="ECG")
shade(ax1)
ax1.set_title("Baseline-Corrected Signals  —  EEG1 (blue) | EEG2 (green) | ECG (red)", fontsize=10)
ax1.set_xlabel("Time (s)"); ax1.set_ylabel("EEG"); ax1r.set_ylabel("ECG", color=RED)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax1r.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc="upper right", fontsize=8)

# 2 — Band powers
ax2 = fig.add_subplot(gs[1, 0])
for band, col in [("delta", PURPLE), ("theta", BLUE), ("alpha", GREEN), ("beta", ORANGE)]:
    ax2.plot(t, np.log1p(feat_df[band]), lw=1.0, alpha=0.85, color=col, label=band)
shade(ax2)
ax2.set_title("Log Band Powers (EEG)"); ax2.set_xlabel("Time (s)"); ax2.set_ylabel("log(Power+1)")
ax2.legend(fontsize=8)

# 3 — Hjorth
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t, feat_df["z_hj_act"],  color="#E91E63", lw=1.0, label="Activity (z)")
ax3.plot(t, feat_df["z_hj_mob"],  color="#00BCD4", lw=1.0, label="Mobility (z)")
ax3.plot(t, feat_df["z_hj_comp"], color="#8BC34A", lw=1.0, label="Complexity (z)")
shade(ax3)
ax3.set_title("Hjorth Parameters (z-scored)"); ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Z-score")
ax3.legend(fontsize=8)

# 4 — Wavelet energy per level
ax4 = fig.add_subplot(gs[2, :])
wt_cols_energy = [c for c in feat_df.columns if "wt_" in c and "_energy" in c]
wt_colors = [RED, ORANGE, GREEN, BLUE, PURPLE]
for col, col_c in zip(wt_cols_energy, wt_colors):
    label = col.replace("wt_","").replace("_energy","")
    ax4.plot(t, np.log1p(feat_df[col]), lw=1.0, alpha=0.8, color=col_c, label=label)
shade(ax4)
ax4.set_title("Wavelet Decomposition — Log Energy per Level (EEG)  |  D1=high-freq → A4=low-freq")
ax4.set_xlabel("Time (s)"); ax4.set_ylabel("log(Energy+1)"); ax4.legend(fontsize=8, ncol=5)

# 5 — Wavelet entropy per level
ax5 = fig.add_subplot(gs[3, 0])
wt_cols_entropy = [c for c in feat_df.columns if "wt_" in c and "_entropy" in c]
for col, col_c in zip(wt_cols_entropy, wt_colors):
    label = col.replace("wt_","").replace("_entropy","")
    ax5.plot(t, feat_df[col], lw=0.9, alpha=0.8, color=col_c, label=label)
shade(ax5)
ax5.set_title("Wavelet Entropy per Level"); ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Entropy")
ax5.legend(fontsize=7)

# 6 — LF/HF ratio
ax6 = fig.add_subplot(gs[3, 1])
lf_hf_vals = feat_df["hrv_lf_hf"].replace(0, np.nan)
ax6.plot(t, lf_hf_vals, color="#F44336", lw=1.2, label="LF/HF ratio")
ax6r = ax6.twinx()
ax6r.plot(t, feat_df["hrv_lf"].replace(0, np.nan), color=BLUE,   lw=0.8, alpha=0.6, label="LF power")
ax6r.plot(t, feat_df["hrv_hf"].replace(0, np.nan), color=GREEN,  lw=0.8, alpha=0.6, label="HF power")
if onset:
    ax6.axvline(onset,  color="darkorange", ls="--", lw=1.5, label=f"Onset {onset:.0f}s")
    ax6.axvline(offset, color="purple",     ls="--", lw=1.5, label=f"Offset {offset:.0f}s")
shade(ax6)
ax6.set_title("HRV LF/HF Ratio  (30s rolling window)"); ax6.set_xlabel("Time (s)")
ax6.set_ylabel("LF/HF", color=RED); ax6r.set_ylabel("Power", color=BLUE)
h1, l1 = ax6.get_legend_handles_labels(); h2, l2 = ax6r.get_legend_handles_labels()
ax6.legend(h1+h2, l1+l2, fontsize=7, loc="upper left")

# 7 — Heart rate
ax7 = fig.add_subplot(gs[4, 0])
hr = feat_df["hr_mean"].replace(0, np.nan)
ax7.plot(t, hr, color=RED, lw=1.2)
shade(ax7)
ax7.set_title("Heart Rate (from ECG)"); ax7.set_xlabel("Time (s)"); ax7.set_ylabel("BPM")

# 8 — Motor
ax8 = fig.add_subplot(gs[4, 1])
ax8.plot(time[::ds], motor_filt[::ds], color=ORANGE, lw=0.8, label="Motor envelope")
shade(ax8)
ax8.set_title("Motor Channel (envelope)"); ax8.set_xlabel("Time (s)"); ax8.set_ylabel("ADC units")

# 9 — Seizure score
ax9 = fig.add_subplot(gs[5, :])
ax9.plot(t, feat_df["seizure_score"], color="#3F51B5", lw=1.5, label="Composite score")
ax9.axhline(thresh, color=RED, ls="--", lw=1.2, label=f"Threshold μ+2σ = {thresh:.1f}")
shade(ax9)
ax9.set_title("Composite Seizure Score"); ax9.set_xlabel("Time (s)")
ax9.set_ylabel("Score"); ax9.legend(fontsize=9)

# 10 — Isolation Forest
ax10 = fig.add_subplot(gs[6, 0])
ax10.plot(t, feat_df["iso_score"], color="#795548", lw=0.9)
ax10.fill_between(t, 0, feat_df["iso_score"],
                  where=feat_df["iso_flag"].values, color=RED, alpha=0.35, label="Anomaly")
shade(ax10)
ax10.set_title("Isolation Forest Anomaly Score"); ax10.set_xlabel("Time (s)")
ax10.set_ylabel("Score"); ax10.legend(fontsize=8)

# 11 — GMM
ax11 = fig.add_subplot(gs[6, 1])
cmap = {0: GREEN, 1: ORANGE, 2: RED}
for cl in range(3):
    m   = (feat_df["gmm_cluster"] == cl).values
    lbl = "Ictal (GMM)" if cl == ictal_cl else f"State {cl}"
    ax11.scatter(t[m], feat_df.loc[m, "seizure_score"],
                 c=cmap[cl], s=8, alpha=0.55, label=lbl)
shade(ax11)
ax11.set_title("GMM Clustering (3 states)"); ax11.set_xlabel("Time (s)")
ax11.set_ylabel("Seizure Score"); ax11.legend(fontsize=8)

# 12 — Ensemble vote
ax12 = fig.add_subplot(gs[7, :])
bar_c = [RED if v >= 2 else "#90CAF9" for v in feat_df["vote"]]
ax12.bar(t, feat_df["vote"], width=STEP_S * 0.9, color=bar_c, alpha=0.85)
ax12.axhline(2, color=RED, ls="--", lw=1.2, label="Seizure (≥2/3 votes)")
ax12.set_title("Ensemble Vote  —  Isolation Forest + GMM + Score Threshold  |  RED = Seizure")
ax12.set_xlabel("Time (s)"); ax12.set_ylabel("Votes"); ax12.set_yticks([0,1,2,3])
ax12.legend(fontsize=9)

title = (f"ECT Seizure Detection v4 — 2ASZPCRC11004    Fs={FS} Hz\n"
         f"Features: Standard + Wavelet Decomposition (D1–D4, A4) + HRV LF/HF")
if onset:
    title += f"\nDetected: {onset:.0f}s – {offset:.0f}s  (duration {dur:.0f}s)"
    if psi:
        title += f"  |  PSI ≈ {psi:.0f}%"
plt.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

fig.savefig("/mnt/user-data/outputs/ect_seizure_detection_v4.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved → /mnt/user-data/outputs/ect_seizure_detection_v4.png")

# ══════════════════════════════════════════════════════
# 13. SUMMARY
# ══════════════════════════════════════════════════════
print("\n" + "═"*55)
print("SUMMARY — 2ASZPCRC11004  (v4: wavelet + LF/HF)")
print("═"*55)
print(f"Fs (inferred)    : {FS} Hz")
print(f"Duration         : {time[-1]:.1f} s")
print(f"Windows          : {len(feat_df)}")
print(f"Features/window  : {len(FEAT_COLS)}")
print(f"Iso Forest flags : {feat_df['iso_flag'].sum()}")
print(f"GMM ictal flags  : {feat_df['gmm_ictal'].sum()}")
print(f"Score flags      : {feat_df['score_flag'].sum()}")
print(f"Ensemble flags   : {feat_df['seizure_flag'].sum()}")
if onset:
    print(f"Seizure onset    : {onset:.1f} s")
    print(f"Seizure offset   : {offset:.1f} s")
    print(f"Duration         : {dur:.1f} s")
    if psi:
        print(f"PSI (approx)     : {psi:.1f}%")
print("═"*55)
