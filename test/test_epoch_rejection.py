from __future__ import annotations

import numpy as np
import mne

from pyblinker.utils.epoch_rejection import detect_bad_epochs_peak_to_peak


def _inject_epoch_artifacts(
    epochs_data: np.ndarray,
    *,
    frac_bad: float = 0.35,
    seg_len_s: tuple[float, float] = (0.5, 2.0),
    amplitude_mult: tuple[float, float] = (20.0, 45.0),
    random_state: int = 42,
    sfreq: float,
) -> tuple[np.ndarray, dict]:
    """Inject synthetic artifacts into random epochs of EEG-E8 data."""

    rng = np.random.default_rng(random_state)
    data = epochs_data.copy()
    n_epochs, n_times = data.shape

    robust_scale = 1.4826 * np.median(np.abs(data - np.median(data)))
    if robust_scale <= 0:
        robust_scale = float(np.std(data) + 1e-12)

    n_bad = max(1, int(round(frac_bad * n_epochs)))
    bad_epochs = np.sort(rng.choice(n_epochs, size=n_bad, replace=False))

    records: list[dict] = []
    min_len = max(1, int(seg_len_s[0] * sfreq))
    max_len = max(min_len + 1, int(seg_len_s[1] * sfreq))

    for ep_idx in bad_epochs:
        seg_len = int(rng.integers(min_len, min(max_len, n_times - 1)))
        start = int(rng.integers(0, n_times - seg_len))
        stop = start + seg_len
        amp = float(rng.uniform(*amplitude_mult) * robust_scale)
        sign = float(rng.choice([-1.0, 1.0]))
        artifact_cycle = ["burst", "step", "drift", "flatline"]
        art_type = artifact_cycle[int(ep_idx) % len(artifact_cycle)]

        if art_type == "burst":
            burst = rng.standard_normal(seg_len) * np.hanning(seg_len)
            data[ep_idx, start:stop] += sign * amp * burst
        elif art_type == "step":
            data[ep_idx, start:stop] += sign * amp
        elif art_type == "drift":
            data[ep_idx, start:stop] += sign * amp * np.linspace(0.0, 1.0, seg_len)
        else:  # flatline
            fill_value = float(sign * amp)
            data[ep_idx, start:stop] = fill_value

        records.append(
            {
                "epoch": int(ep_idx),
                "channel": "EEG-E8",
                "start": int(start),
                "stop": int(stop),
                "type": art_type,
                "amplitude": amp,
            }
        )

    gt = {"bad_epoch_idx": bad_epochs, "records": records}
    return data, gt


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def test_detect_bad_epochs_peak_to_peak_on_corrupted_eeg_e8() -> None:
    raw = mne.io.read_raw_fif("test/test_files/ear_eog_raw.fif", preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])
    raw.pick(["EEG-E8"])

    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=30.0,
        preload=True,
        reject_by_annotation=False,
        verbose=False,
    )
    clean_data = epochs.get_data(copy=True)[:, 0, :]

    corrupted_data, gt = _inject_epoch_artifacts(clean_data, sfreq=sfreq, random_state=123)

    flattened = corrupted_data.reshape(-1)
    result = detect_bad_epochs_peak_to_peak(
        flattened,
        sfreq,
        epoch_duration_s=30.0,
        n_splits=3,
        n_candidates=41,
        random_state=123,
    )

    n_epochs = corrupted_data.shape[0]
    y_true = np.zeros(n_epochs, dtype=int)
    y_true[gt["bad_epoch_idx"]] = 1

    y_pred = np.zeros(n_epochs, dtype=int)
    y_pred[result.bad_epoch_indices] = 1

    precision, recall, f1 = _classification_metrics(y_true, y_pred)

    assert precision >= 0.50
    assert recall >= 0.50
    assert f1 >= 0.50
    assert len(gt["records"]) == len(gt["bad_epoch_idx"])
