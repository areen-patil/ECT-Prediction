import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EEG_COLUMNS = ["Ch1", "Ch2", "Ch3"]
AUX_COLUMN = "Ch4"


def infer_fs(df: pd.DataFrame) -> float:
    if "Time_s" not in df.columns or len(df) < 2:
        raise ValueError("CSV must contain a Time_s column with at least two rows.")
    dt = float(df["Time_s"].iloc[1] - df["Time_s"].iloc[0])
    if dt <= 0:
        raise ValueError("Time_s must be strictly increasing.")
    return 1.0 / dt


def robust_zscore(series: pd.Series) -> pd.Series:
    median = float(series.median())
    mad = float((series - median).abs().median())
    scale = 1.4826 * mad if mad > 1e-9 else max(float(series.std(ddof=0)), 1.0)
    return (series - median) / scale


def extract_window_features(df: pd.DataFrame, window_s: float, step_s: float) -> pd.DataFrame:
    fs = infer_fs(df)
    eeg_cols = [col for col in EEG_COLUMNS if col in df.columns]
    if len(eeg_cols) < 2:
        raise ValueError("Expected at least two EEG channels among Ch1, Ch2, Ch3.")
    if AUX_COLUMN not in df.columns:
        raise ValueError("Expected auxiliary channel Ch4.")

    window = max(int(round(window_s * fs)), 8)
    step = max(int(round(step_s * fs)), 1)

    eeg = df[eeg_cols].astype(float).to_numpy()
    eeg_centered = eeg - np.median(eeg, axis=0, keepdims=True)

    aux = df[AUX_COLUMN].astype(float).to_numpy()
    aux_centered = aux - np.median(aux)

    eeg_clip_max = float(np.max(eeg))
    aux_clip_max = float(np.max(aux))

    rows = []
    for start in range(0, len(df) - window + 1, step):
        stop = start + window

        eeg_win = eeg_centered[start:stop]
        eeg_raw = eeg[start:stop]
        aux_win = aux_centered[start:stop]
        aux_raw = aux[start:stop]

        t_start = float(df["Time_s"].iloc[start])
        t_stop = float(df["Time_s"].iloc[stop - 1])
        t_mid = (t_start + t_stop) / 2.0

        eeg_diff = np.diff(eeg_win, axis=0)
        aux_diff = np.diff(aux_win)

        per_ch_rms = np.sqrt(np.mean(eeg_win ** 2, axis=0))
        per_ch_std = np.std(eeg_win, axis=0)
        per_ch_line_length = np.sum(np.abs(eeg_diff), axis=0)
        per_ch_p2p = np.ptp(eeg_raw, axis=0)
        per_ch_zero_cross = np.sum(np.diff(np.signbit(eeg_win), axis=0) != 0, axis=0)
        per_ch_clip_frac = np.mean(eeg_raw >= eeg_clip_max, axis=0)

        aux_rms = float(np.sqrt(np.mean(aux_win ** 2)))
        aux_std = float(np.std(aux_win))
        aux_line_length = float(np.sum(np.abs(aux_diff)))
        aux_p2p = float(np.ptp(aux_raw))
        aux_clip_frac = float(np.mean(aux_raw >= aux_clip_max))

        rows.append(
            {
                "t_start_s": t_start,
                "t_stop_s": t_stop,
                "t_mid_s": t_mid,
                "eeg_rms_mean": float(np.mean(per_ch_rms)),
                "eeg_std_mean": float(np.mean(per_ch_std)),
                "eeg_line_length_mean": float(np.mean(per_ch_line_length)),
                "eeg_peak_to_peak_mean": float(np.mean(per_ch_p2p)),
                "eeg_zero_cross_mean": float(np.mean(per_ch_zero_cross)),
                "eeg_clip_frac_mean": float(np.mean(per_ch_clip_frac)),
                "eeg_channel_agreement_std": float(np.std(per_ch_line_length)),
                "aux_rms": aux_rms,
                "aux_std": aux_std,
                "aux_line_length": aux_line_length,
                "aux_peak_to_peak": aux_p2p,
                "aux_clip_frac": aux_clip_frac,
            }
        )

    features = pd.DataFrame(rows)

    features["z_eeg_rms"] = robust_zscore(features["eeg_rms_mean"])
    features["z_eeg_line_length"] = robust_zscore(features["eeg_line_length_mean"])
    features["z_eeg_peak_to_peak"] = robust_zscore(features["eeg_peak_to_peak_mean"])
    features["z_aux_line_length"] = robust_zscore(features["aux_line_length"])
    features["z_aux_peak_to_peak"] = robust_zscore(features["aux_peak_to_peak"])
    features["z_clip_penalty"] = robust_zscore(features["eeg_clip_frac_mean"] + features["aux_clip_frac"])

    features["seizure_score"] = (
        0.40 * features["z_eeg_line_length"]
        + 0.25 * features["z_eeg_rms"]
        + 0.20 * features["z_eeg_peak_to_peak"]
        + 0.10 * features["z_aux_line_length"]
        + 0.05 * features["z_aux_peak_to_peak"]
        - 0.20 * features["z_clip_penalty"]
    )

    features["seizure_flag"] = features["seizure_score"] >= 3.0
    return features


def summarize_candidates(features: pd.DataFrame) -> list[dict]:
    flagged = features.index[features["seizure_flag"]].tolist()

    if not flagged:
        top = features.nlargest(5, "seizure_score")
        return [
            {
                "start_s": float(row["t_start_s"]),
                "stop_s": float(row["t_stop_s"]),
                "peak_score": float(row["seizure_score"]),
                "n_windows": 1,
            }
            for _, row in top.iterrows()
        ]

    groups = []
    current = [flagged[0]]

    for idx in flagged[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]

    groups.append(current)

    summaries = []
    for group in groups:
        chunk = features.iloc[group]
        summaries.append(
            {
                "start_s": float(chunk["t_start_s"].iloc[0]),
                "stop_s": float(chunk["t_stop_s"].iloc[-1]),
                "peak_score": float(chunk["seizure_score"].max()),
                "n_windows": int(len(chunk)),
            }
        )

    summaries.sort(key=lambda item: item["peak_score"], reverse=True)
    return summaries


def process_file(csv_path: Path, output_dir: Path, window_s: float, step_s: float) -> None:
    df = pd.read_csv(csv_path)
    features = extract_window_features(df, window_s=window_s, step_s=step_s)

    output_dir.mkdir(parents=True, exist_ok=True)

    feature_csv = output_dir / f"{csv_path.stem}_features.csv"
    summary_csv = output_dir / f"{csv_path.stem}_candidates.csv"

    features.to_csv(feature_csv, index=False)
    pd.DataFrame(summarize_candidates(features)).to_csv(summary_csv, index=False)

    print(f"\nFile: {csv_path}")
    print(f"Saved features: {feature_csv}")
    print(f"Saved candidates: {summary_csv}")

    candidates = summarize_candidates(features)
    print("Top seizure candidates:")
    for item in candidates[:5]:
        print(
            f"  {item['start_s']:.1f}s to {item['stop_s']:.1f}s | "
            f"peak_score={item['peak_score']:.2f} | windows={item['n_windows']}"
        )


def collect_csvs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*.csv") if path.name.lower() != "data_loader.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract windowed features and heuristic seizure candidates from ECT CSV files."
    )
    parser.add_argument("input", help="Path to a CSV file or directory containing CSV files.")
    parser.add_argument(
        "--output-dir",
        default="analysis/feature_exports",
        help="Directory where feature CSVs will be written.",
    )
    parser.add_argument("--window-s", type=float, default=2.0, help="Window length in seconds.")
    parser.add_argument("--step-s", type=float, default=0.5, help="Window hop in seconds.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    csv_files = collect_csvs(input_path)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {input_path}")

    for csv_path in csv_files:
        process_file(csv_path, output_dir=output_dir, window_s=args.window_s, step_s=args.step_s)


if __name__ == "__main__":
    main()
