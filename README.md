# ECT EEG Seizure Detection Pipeline

> **Unsupervised, end-to-end seizure onset and offset detection from raw multi-channel ECT EEG recordings — no labels, no annotations, no hardcoded assumptions.**

Built in collaboration with the **National Institute of Mental Health and Neurosciences (NIMHANS), Bengaluru**.

---

## Overview

Electroconvulsive Therapy (ECT) is one of the most effective treatments for severe, treatment-resistant psychiatric conditions. The quality of the induced seizure — its onset, duration, and degree of postictal suppression — is directly linked to therapeutic outcome.

In standard clinical practice, seizure onset and termination are marked **manually** by a clinician watching the EEG trace. This is slow, subjective, and impossible to scale for retrospective studies.

This project builds a fully automated pipeline that detects seizure onset and offset from raw 8-channel ECT EEG/ECG recordings, without any labelled training data.

---

## Dataset

| Property | Value |
|---|---|
| Source | NIMHANS, Bengaluru (clinical, not public) |
| Total recordings | 96 ECT session CSVs |
| Channels | 8 (Ch1=ECG, Ch2=EEG1, Ch3=EEG2, Ch4=Motor, Ch5–Ch8=Baseline refs) |
| Sampling rate | 512 Hz (inferred from data, not hardcoded) |
| Typical recording duration | 80 – 650 s (avg 279.9 s) |
| Labels / annotations | None — fully unsupervised |

> **Note:** The dataset is not publicly available due to patient privacy and ethics restrictions. The pipeline is designed to run on any similarly structured ECT EEG CSV.

---

## Pipeline Architecture

The pipeline runs in **9 sequential stages**, from raw CSV to structured seizure detection output:

```
Raw CSV
  │
  ▼
[1] Load & Infer Fs          — Sampling frequency inferred from Time_s column (median diff)
  │
  ▼
[2] Baseline Correction      — Subtract paired reference channels (Ch2−Ch6, Ch3−Ch7, etc.)
  │
  ▼
[3] Digital Filtering        — EEG: 0.5–70 Hz bandpass + 50 Hz notch
                               ECG: 0.5–40 Hz bandpass
                               Motor: full-wave rectify + 10 Hz lowpass
  │
  ▼
[4] Full R-Peak Detection    — Adaptive threshold (60th percentile) for robust HRV
  │
  ▼
[5] Feature Extraction       — 39 features over 2s windows, 0.5s step
                               EEG time-domain, band powers, Hjorth, wavelet DWT,
                               Sample Entropy, ECG, LF/HF HRV, motor envelope
  │
  ▼
[6] Z-Score Normalisation    — Against each patient's own first 60s baseline
  │
  ▼
[7] Composite Seizure Score  — Weighted sum of 13 z-scored features (SampEn inverted)
  │
  ▼
[8] Unsupervised ML          — Isolation Forest + GMM (3 states) + score threshold
                               run in parallel
  │
  ▼
[9] Ensemble Vote + Refinement — Majority vote (≥2/3 methods)
                                 SampEn minimum used to refine seizure offset
                                 (Yoo et al. 2012)
  │
  ▼
Seizure onset, offset, duration, PSI, HR — logged per file
```

---

## Features Extracted (39 total)

| Group | Features | Count |
|---|---|---|
| EEG time-domain | RMS, Std, Line Length, Peak-to-Peak, Zero Crossing Rate, Clip Fraction, Inter-Channel Agreement | 7 |
| EEG spectral bands | Delta, Theta, Alpha, Beta (Welch PSD) | 4 |
| Hjorth parameters | Activity, Mobility, Complexity | 3 |
| Sample Entropy | SampEn (m=2, r=0.2σ, downsampled 4×) | 1 |
| Wavelet DWT (4-level) | Energy, Entropy, MAV × 5 sub-bands (D1–D4, A4) | 15 |
| ECG time-domain | RMS, Line Length, Peak-to-Peak, Mean HR | 4 |
| HRV frequency domain | LF power, HF power, LF/HF ratio | 3 |
| Motor envelope | RMS, Peak-to-Peak | 2 |

---

## Unsupervised ML Methods

Three independent detectors run in parallel. A window is flagged as ictal only if **at least 2 of 3 agree** (majority vote):

| Method | Approach |
|---|---|
| **Isolation Forest** | Anomaly detection on 39-dim feature space. `contamination=0.05`, `n_estimators=200` |
| **Gaussian Mixture Model** | 3-component GMM (baseline / transitional / ictal). Ictal cluster auto-identified by highest mean composite score |
| **Score Threshold** | Flag windows where composite seizure score > μ + 2σ of the recording |

---

## Key Engineering Decisions

**Why unsupervised?**
No expert-annotated onset/offset times exist for any of the 96 recordings. Supervised learning was not an option.

**Why per-patient Z-scoring?**
Absolute EEG amplitude varies significantly between patients. Normalising against each patient's own 60-second pre-stimulus baseline makes the pipeline self-calibrating.

**Why the SampEn offset refinement?**
Based directly on Yoo et al. (2012), who showed that Sample Entropy reaches its minimum value at the exact seizure termination boundary in ECT EEG. The pipeline uses this as a dedicated offset refinement step after the ensemble vote.

**Why an adaptive R-peak threshold?**
A static `0.5 × max(signal)` threshold anchored to the loudest point in the recording — which occurs during the seizure — made the threshold too high for baseline heartbeats, detecting only 5 R-peaks across an 8-minute recording (~1 bpm). Switching to the **60th percentile of positive signal values** gave a threshold tied to typical signal amplitude, correctly detecting 840 R-peaks at a physiologically plausible 97 bpm.

---

## Batch Results (96 files)

| Metric | Value |
|---|---|
| Total files processed | 96 |
| Successful runs | 94 |
| Errors | 2 |
| Seizures detected | 93 / 94 (98.9%) |
| Average recording duration | 279.9 s |
| Average seizure duration | 63.8 s |
| Sessions with seizure ≤ 90s | 63 / 93 (67.7%) |
| Mean PSI | 70.4% (clinical threshold: ≥80%) |

---

## Outputs

**Per recording:**
- `ect_seizure_detection_v5.png` — 13-panel diagnostic plot (raw signals, band powers, wavelet energy, SampEn, LF/HF, composite score, all ML flags, onset/offset markers)
- `*_extracted_features_v4.csv` — 89-column feature table (all raw features, z-scores, ML flags, composite score)

**Batch:**
- `batch_summary.log` — Human-readable per-file results (onset, offset, duration, PSI, HR)
- `batch_results.csv` — Machine-readable results table, one row per file
- `batch_summary_plots.png` — 6-panel summary dashboard

---

## Scripts

| Script | Purpose |
|---|---|
| `ect_seizure_detection_v4.py` | Main single-file pipeline — load → features → ML → ensemble → output |
| `ect_batch_runner_v4.py` | Batch runner for all 96 files. Writes log + CSV. Resilient to per-file errors |
| `ect_log_stats.py` | Parses `batch_summary.log`, prints console stats, saves summary dashboard plot |

---

## Requirements

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

> **Note:** `pywt` (PyWavelets) is recommended for a proper orthogonal DWT filter bank. The current implementation uses cascaded 4th-order Butterworth filters as a drop-in approximation (pywt was unavailable in the original compute environment).

---

## Usage

**Single file:**
```bash
python ect_seizure_detection_v4.py
# Edit BASE_PATH and FILENAME at the top of the script
```

**Full batch:**
```bash
python ect_batch_runner_v4.py
# Edit BASE_DIR at the top of the script to point to your patient folder hierarchy
```

**Log analysis:**
```bash
python ect_log_stats.py
# Edit LOG_FILE path at the top of the script
```

---

## Limitations

- **No ground truth validation.** No expert-annotated onset/offset times exist. Detection quality is assessed by clinical plausibility (durations, PSI), not precision/recall.
- **Short detections may be false positives.** Seizure durations < 10s are likely stimulus artefacts. A minimum duration filter is recommended.
- **Fixed contamination parameter.** Isolation Forest uses `contamination=0.05` globally. Patients with unusually long seizures (where ictal windows may be 40–60% of the recording) will have their detections capped at 5%, producing artificially short reported durations.
- **Heuristic composite score weights.** Feature weights were chosen from clinical literature, not learned from data. A supervised approach with labelled data would likely improve performance.

---

## References

1. **Yoo et al. (2012).** Automatic detection of seizure termination during ECT using sample entropy of the EEG. *Psychiatry Research, 195*(1–2), 76–82.
2. **Hjorth B. (1970).** EEG analysis based on time domain properties. *Electroencephalography and Clinical Neurophysiology, 29*(3), 306–310.
3. **Pan & Tompkins (1985).** A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering, 32*(3), 230–236.

---

## Acknowledgements

This work was conducted in collaboration with **NIMHANS, Bengaluru**, under the supervision of **Prof. Sakshi Arora**. Data collection and clinical context were provided by the NIMHANS psychiatry and neurophysiology teams.

---

*IIIT Bangalore · Areen Patil · 2025*
