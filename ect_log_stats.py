"""
ECT Batch Log Stats Parser + Visual Summary
=============================================
Parses batch_summary.log, prints console stats,
and saves a summary dashboard plot.

Usage:
    python ect_log_stats.py
"""

import os
import re
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════
LOG_FILE  = r"C:\Users\AREEN PATIL\Desktop\self\ECT_EEG_Work\csv_8ch_512Hz\csv_8ch_512Hz\batch_summary.log"
PLOT_FILE = os.path.join(os.path.dirname(LOG_FILE), "batch_summary_plots.png")

# ══════════════════════════════════════════════════════
# 1. PARSE LOG
# ══════════════════════════════════════════════════════
with open(LOG_FILE, "r", encoding="utf-8") as f:
    content = f.read()

blocks = content.split("─" * 60)

rows          = []
total_files   = 0

for block in blocks:
    patient  = re.search(r"Patient\s*:\s*(\S+)", block)
    filename = re.search(r"File\s*:\s*(\S+)", block)
    if not patient or not filename:
        continue

    total_files += 1
    errored  = "Status  : ERROR" in block
    detected = "Seizure DETECTED" in block

    rec_dur  = re.search(r"Duration:\s*([\d.]+)\s*s",      block)   # no space = recording
    seiz_dur = re.search(r"Duration\s+:\s*([\d.]+)\s*s",   block)   # space    = seizure
    onset    = re.search(r"Onset\s+:\s*([\d.]+)\s*s",      block)
    offset   = re.search(r"Offset\s+:\s*([\d.]+)\s*s",     block)
    psi      = re.search(r"PSI\s+:\s*([\d.]+)",             block)
    runtime  = re.search(r"Runtime\s*:\s*([\d.]+)",         block)

    rows.append({
        "patient":  patient.group(1),
        "file":     filename.group(1),
        "errored":  errored,
        "detected": detected,
        "rec_dur":  float(rec_dur.group(1))  if rec_dur                   else None,
        "seiz_dur": float(seiz_dur.group(1)) if (detected and seiz_dur)   else None,
        "onset":    float(onset.group(1))    if onset                     else None,
        "offset":   float(offset.group(1))   if offset                    else None,
        "psi":      float(psi.group(1))      if psi                       else None,
        "runtime":  float(runtime.group(1))  if runtime                   else None,
    })

# ── Derived lists ──────────────────────────────────────
successful    = [r for r in rows if not r["errored"]]
detected_rows = [r for r in successful if r["detected"]]
no_seiz_rows  = [r for r in successful if not r["detected"]]
error_rows    = [r for r in rows if r["errored"]]

rec_durs  = [r["rec_dur"]  for r in successful    if r["rec_dur"]  is not None]
seiz_durs = [r["seiz_dur"] for r in detected_rows if r["seiz_dur"] is not None]
onsets    = [r["onset"]    for r in detected_rows if r["onset"]    is not None]
psis      = [r["psi"]      for r in detected_rows if r["psi"]      is not None]

n_errors   = len(error_rows)
n_success  = total_files - n_errors
n_seizures = len(detected_rows)
n_no_seiz  = len(no_seiz_rows)

# Per-patient aggregation
patient_data = defaultdict(lambda: {"total": 0, "seizure": 0, "seiz_durs": []})
for r in successful:
    patient_data[r["patient"]]["total"] += 1
    if r["detected"]:
        patient_data[r["patient"]]["seizure"] += 1
        if r["seiz_dur"]:
            patient_data[r["patient"]]["seiz_durs"].append(r["seiz_dur"])
patients = sorted(patient_data.keys())

# ══════════════════════════════════════════════════════
# 2. CONSOLE STATS
# ══════════════════════════════════════════════════════
print("=" * 55)
print("ECT BATCH SUMMARY STATS")
print("=" * 55)
print(f"Total files             : {total_files}")
print(f"  Successful runs       : {n_success}  (= seizures + no seizure)")
print(f"    Seizures detected   : {n_seizures}")
print(f"    No seizure detected : {n_no_seiz}")
print(f"  Errors                : {n_errors}")
print(f"  Check: {n_seizures} + {n_no_seiz} + {n_errors} = {n_seizures + n_no_seiz + n_errors}  "
      f"{'✓' if n_seizures + n_no_seiz + n_errors == total_files else '✗ MISMATCH'}")

print()
print("── Recording Duration (successful runs) ────────")
if rec_durs:
    print(f"  Average : {np.mean(rec_durs):.1f} s  ({np.mean(rec_durs)/60:.2f} min)")
    print(f"  Min     : {min(rec_durs):.1f} s")
    print(f"  Max     : {max(rec_durs):.1f} s")

print()
print("── Seizure Duration ────────────────────────────")
if seiz_durs:
    print(f"  Average : {np.mean(seiz_durs):.1f} s  ({np.mean(seiz_durs)/60:.2f} min)")
    print(f"  Min     : {min(seiz_durs):.1f} s")
    print(f"  Max     : {max(seiz_durs):.1f} s")

print()
print("── Seizure Duration Distribution ───────────────")
brackets = [(0, 30), (30, 60), (60, 90), (90, 120),
            (120, 180), (180, 300), (300, 999)]
for lo, hi in brackets:
    count = sum(1 for d in seiz_durs if lo < d <= hi)
    bar   = "█" * count
    label = f"{hi}" if hi < 999 else "∞"
    print(f"  {lo:4d}–{label:>4}s : {count:3d}  {bar}")

print()
print("── Sessions with seizure duration ≤ 90s ────────")
under_90 = [r for r in detected_rows if r["seiz_dur"] and r["seiz_dur"] <= 90]
print(f"  Count : {len(under_90)}")
for r in sorted(under_90, key=lambda x: x["seiz_dur"]):
    print(f"  {r['patient']:20s}  {r['file']:30s}  {r['seiz_dur']:.1f}s")

if no_seiz_rows:
    print()
    print("── Files with no seizure detected ──────────────")
    for r in no_seiz_rows:
        print(f"  {r['patient']}/{r['file']}")

if error_rows:
    print()
    print("── Errored files ───────────────────────────────")
    for r in error_rows:
        print(f"  {r['patient']}/{r['file']}")

print("=" * 55)

# ══════════════════════════════════════════════════════
# 3. PLOTS
# ══════════════════════════════════════════════════════
print(f"\nGenerating plots → {PLOT_FILE}")

BLUE   = "#2196F3"
GREEN  = "#4CAF50"
RED    = "#EF5350"
ORANGE = "#FF9800"
PURPLE = "#9C27B0"
GREY   = "#9E9E9E"

fig = plt.figure(figsize=(20, 22))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

# ── Panel 1: Pie — outcome breakdown ──────────────────
ax1 = fig.add_subplot(gs[0, 0])
sizes  = [n_seizures, n_no_seiz, n_errors]
labels = [f"Seizure detected\n({n_seizures})",
          f"No seizure\n({n_no_seiz})",
          f"Error\n({n_errors})"]
colors = [RED, GREEN, GREY]
wedges, texts, autotexts = ax1.pie(
    sizes, labels=labels, colors=colors,
    autopct="%1.1f%%", startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2))
for at in autotexts:
    at.set_fontsize(9)
ax1.set_title(f"Session Outcome\n({total_files} total files)",
              fontweight="bold", fontsize=11)

# ── Panel 2: Histogram — seizure duration ─────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(seiz_durs, bins=20, color=RED, edgecolor="white", linewidth=0.8, alpha=0.85)
ax2.axvline(np.mean(seiz_durs), color="black", ls="--", lw=1.5,
            label=f"Mean: {np.mean(seiz_durs):.1f}s")
ax2.axvline(90, color=ORANGE, ls=":", lw=1.8, label="90s threshold")
ax2.set_title("Seizure Duration Distribution", fontweight="bold", fontsize=11)
ax2.set_xlabel("Duration (s)"); ax2.set_ylabel("Count")
ax2.legend(fontsize=9)

# ── Panel 3: Histogram — recording duration ───────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(rec_durs, bins=20, color=BLUE, edgecolor="white", linewidth=0.8, alpha=0.85)
ax3.axvline(np.mean(rec_durs), color="black", ls="--", lw=1.5,
            label=f"Mean: {np.mean(rec_durs):.1f}s")
ax3.set_title("Recording Duration Distribution", fontweight="bold", fontsize=11)
ax3.set_xlabel("Duration (s)"); ax3.set_ylabel("Count")
ax3.legend(fontsize=9)

# ── Panel 4: Scatter — onset vs seizure duration ──────
ax4 = fig.add_subplot(gs[1, 0:2])
onset_arr = np.array([r["onset"]    for r in detected_rows if r["onset"] and r["seiz_dur"]])
sdur_arr  = np.array([r["seiz_dur"] for r in detected_rows if r["onset"] and r["seiz_dur"]])
sc = ax4.scatter(onset_arr, sdur_arr, c=sdur_arr, cmap="RdYlGn_r",
                 s=60, alpha=0.75, edgecolors="white", linewidth=0.5)
plt.colorbar(sc, ax=ax4, label="Seizure duration (s)")
ax4.axhline(90, color=ORANGE, ls=":", lw=1.8, label="90s threshold")
ax4.set_title("Seizure Onset Time vs Seizure Duration",
              fontweight="bold", fontsize=11)
ax4.set_xlabel("Onset time (s)"); ax4.set_ylabel("Seizure duration (s)")
ax4.legend(fontsize=9)

# ── Panel 5: PSI distribution ─────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
if psis:
    ax5.hist(psis, bins=15, color=PURPLE, edgecolor="white", linewidth=0.8, alpha=0.85)
    ax5.axvline(np.mean(psis), color="black", ls="--", lw=1.5,
                label=f"Mean: {np.mean(psis):.1f}%")
    ax5.axvline(80, color=GREEN, ls=":", lw=1.8, label="PSI≥80 (adequate)")
    ax5.legend(fontsize=9)
ax5.set_title(f"PSI Distribution\n({len(psis)} sessions with PSI)",
              fontweight="bold", fontsize=11)
ax5.set_xlabel("PSI (%)"); ax5.set_ylabel("Count")

# ── Panel 6: Per-patient stacked bar + avg seiz dur ───
ax6 = fig.add_subplot(gs[2, :])
x        = np.arange(len(patients))
totals   = [patient_data[p]["total"]   for p in patients]
seizures = [patient_data[p]["seizure"] for p in patients]
no_seizs = [t - s for t, s in zip(totals, seizures)]

ax6.bar(x, seizures, color=RED,   alpha=0.85, label="Seizure detected", width=0.6)
ax6.bar(x, no_seizs, color=GREEN, alpha=0.85, label="No seizure",
        bottom=seizures, width=0.6)

# avg seizure duration per patient as overlay line
avg_seiz_per_patient = [
    np.mean(patient_data[p]["seiz_durs"]) if patient_data[p]["seiz_durs"] else 0
    for p in patients
]
ax6r = ax6.twinx()
ax6r.plot(x, avg_seiz_per_patient, color="black", marker="o",
          ms=5, lw=1.5, label="Avg seizure duration (s)")
ax6r.set_ylabel("Avg seizure duration (s)", fontsize=10)

ax6.set_xticks(x)
ax6.set_xticklabels([p.replace("SZPCRC", "") for p in patients],
                    rotation=45, ha="right", fontsize=9)
ax6.set_title("Per-Patient: Sessions & Detection Rate  |  Line = Avg Seizure Duration",
              fontweight="bold", fontsize=11)
ax6.set_xlabel("Patient ID"); ax6.set_ylabel("Number of sessions")
h1, l1 = ax6.get_legend_handles_labels()
h2, l2 = ax6r.get_legend_handles_labels()
ax6.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")

plt.suptitle(
    f"ECT Batch Detection — Summary Dashboard\n"
    f"{total_files} files  |  {n_seizures} seizures detected  |  "
    f"Avg seizure: {np.mean(seiz_durs):.1f}s  |  "
    f"Avg recording: {np.mean(rec_durs):.1f}s",
    fontsize=13, fontweight="bold", y=1.01)

fig.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved → {PLOT_FILE}")
