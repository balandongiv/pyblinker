from __future__ import annotations

import os
import warnings
import zipfile

import mne
import numpy as np
import pandas as pd


def restructure_blink_dataframe(
    df: pd.DataFrame,
    sampling_rate: float,
    frame_col: str = "adjusted_frame",
    label_col: str = "LabelName",
) -> pd.DataFrame:
    """Restructure a CVAT-style blink dataframe into one row per blink."""

    df = df.copy()
    df = df.sort_values(frame_col).reset_index(drop=True)

    def _blink_type(label: str) -> str:
        parts = str(label).split("_")
        return "_".join(parts[:2]) if len(parts) >= 2 else str(label)

    def _phase(label: str) -> str:
        return str(label).split("_")[-1]

    df["blink_type"] = df[label_col].apply(_blink_type)
    df["phase"] = df[label_col].apply(_phase)

    events: list[dict[str, object]] = []

    def finalize_event(ev: dict[str, object]) -> None:
        orig_s, orig_m, orig_e = ev["start"], ev["min"], ev["end"]
        missing_flags = {
            "start": orig_s is None,
            "min": orig_m is None,
            "end": orig_e is None,
        }
        missing_count = sum(missing_flags.values())

        if missing_count >= 2:
            warnings.warn(
                f"Skipping blink for type {ev['blink_type']} – "
                f"too many missing phases (start={orig_s}, min={orig_m}, end={orig_e})."
            )
            return

        s, m, e = orig_s, orig_m, orig_e
        remark = ""
        remark_code = 0

        if missing_count == 0:
            remark = "complete start/min/end; no imputation"
        elif missing_flags["start"]:
            s = m if m is not None else e
            remark_code = 1
            remark = "start imputed from min"
        elif missing_flags["min"]:
            m = s
            remark_code = 2
            remark = "min imputed from start"
        elif missing_flags["end"]:
            e = m
            remark_code = 3
            remark = "end imputed from min"

        duration_frames = int(e) - int(s)
        duration_seconds = duration_frames / float(sampling_rate)
        events.append(
            {
                "blink_type": ev["blink_type"],
                "start": int(s),
                "min": int(m),
                "end": int(e),
                "duration_frames": duration_frames,
                "duration_seconds": float(duration_seconds),
                "remark_code": int(remark_code),
                "remark": remark,
            }
        )

    current: dict[str, object] | None = None
    for _, row in df.iterrows():
        btype = row["blink_type"]
        phase = row["phase"]
        frame = row[frame_col]

        if current is None:
            current = {"blink_type": btype, "start": None, "min": None, "end": None}

        if (
            btype != current["blink_type"]
            or (
                current["start"] is not None
                and current["min"] is not None
                and current["end"] is not None
            )
        ):
            finalize_event(current)
            current = {"blink_type": btype, "start": None, "min": None, "end": None}

        if current[phase] is not None:
            finalize_event(current)
            current = {"blink_type": btype, "start": None, "min": None, "end": None}

        current[phase] = frame

    if current is not None and any(current[p] is not None for p in ("start", "min", "end")):
        finalize_event(current)

    return pd.DataFrame(events)


def load_ground_truth(csv_path: str, constant_shift: int, sampling_rate: float) -> pd.DataFrame:
    """Load ground truth and compute time in seconds."""

    df_gt = pd.read_csv(csv_path)
    df_gt["framenumber"] = df_gt["ImageID"].str.extract(r"(\d+)").astype(int)
    df_gt["adjusted_frame"] = df_gt["framenumber"] - constant_shift
    df_gt["seconds"] = df_gt["adjusted_frame"] / sampling_rate
    return df_gt


def unzip_file(zip_path: str, extract_to: str) -> None:
    """Unzip a file to the specified directory."""

    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted: {zip_path} -> {extract_to}")
    else:
        print(f"File not found: {zip_path}")


def load_actual_annotations(eog_path: str, preload: bool = True) -> pd.DataFrame:
    """Load the actual annotations from an MNE .fif file."""

    raw = mne.io.read_raw_fif(eog_path, preload=preload)
    return pd.DataFrame(raw.annotations)


def filter_min_labels(df_gt: pd.DataFrame) -> pd.DataFrame:
    """Filter ground truth to keep only rows where LabelName ends with '_min'."""

    return df_gt[df_gt["LabelName"].str.endswith("_min")].copy()


def match_ground_truth_to_annotations(
    df_gtruth: pd.DataFrame, df_model: pd.DataFrame
) -> pd.DataFrame:
    """Match ground-truth frames to annotation intervals."""

    df_model = df_model.copy()
    if "model_index" not in df_model.columns:
        df_model["model_index"] = df_model.index

    used = np.zeros(len(df_model), dtype=bool)
    starts = df_model["frame_start"].values
    ends = df_model["frame_end"].values
    model_records = df_model.to_dict("records")
    matched_annotations = []

    for gt_time in df_gtruth["adjusted_frame"]:
        mask = (starts <= gt_time) & (ends >= gt_time)
        candidate_indices = np.flatnonzero(mask)
        match = None
        for idx in candidate_indices:
            if not used[idx]:
                used[idx] = True
                match = model_records[idx]
                break
        if match is None:
            match = {col: None for col in df_model.columns}
        matched_annotations.append(match)

    df_matches = pd.DataFrame(matched_annotations)
    return pd.concat([df_gtruth.reset_index(drop=True), df_matches.reset_index(drop=True)], axis=1)
