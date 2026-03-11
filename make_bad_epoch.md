
## Good experimental setup
this experiment only valid for the EEG channel "EEG-E8"
Use the data from
test/test_files/ear_eog_raw.fif
and when comparing, compare with the output from
annotations, channel, _good, _df, _fig_data, _selected = detector.get_blink()
but you divide it into a 30 seconds epochs since now 
the annotations is output from long continous signal

to create a create a corrupted copy, you can follow the
step below

From a long recording:

1. cut into fixed-length epochs
2. keep a clean copy
3. create a corrupted copy
4. randomly choose some epochs to corrupt
5. inside each chosen epoch, corrupt only a random time segment
6. store:

    * which epochs were corrupted
    * which channels were corrupted
    * start/end samples of the artifact

That gives you exact ground truth.

## In MNE

For 30-second epochs:

```python
import mne

epochs = mne.make_fixed_length_epochs(
    raw,
    duration=30.0,
    overlap=0.0,
    preload=True
)
```

---

# A practical artifact injector

This function corrupts a random subset of epochs and returns the labels.

```python
import numpy as np
import mne

def inject_mock_artifacts(
    epochs,
    frac_bad=0.2,
    max_bad_channels=4,
    seg_len_s=(0.2, 3.0),
    amplitude_mult=(6.0, 15.0),
    artifact_types=("burst", "step", "drift", "flat"),
    picks="eeg",
    random_state=0,
):
    """
    Inject random artificial artifacts into a subset of epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs (must be preload=True).
    frac_bad : float
        Fraction of epochs to corrupt.
    max_bad_channels : int
        Maximum number of channels corrupted in one epoch.
    seg_len_s : tuple(float, float)
        Min/max artifact duration in seconds.
    amplitude_mult : tuple(float, float)
        Artifact amplitude as a multiple of robust channel scale.
    artifact_types : tuple[str]
        Types of artifacts to inject: 'burst', 'step', 'drift', 'flat'.
    picks : str | list
        Channels to consider for corruption, e.g. 'eeg'.
    random_state : int
        RNG seed.

    Returns
    -------
    epochs_corrupt : mne.Epochs
        Corrupted copy of epochs.
    info : dict
        Ground-truth info:
        - bad_epoch_idx
        - bad_channel_mask
        - segment_mask
        - records
    """
    rng = np.random.default_rng(random_state)

    epochs_corrupt = epochs.copy()
    data = epochs_corrupt.get_data(copy=True)   # shape: (n_epochs, n_channels, n_times)

    sfreq = epochs.info["sfreq"]
    n_epochs, n_channels, n_times = data.shape

    pick_idx = mne.pick_types(
        epochs.info,
        eeg=(picks == "eeg"),
        meg=(picks == "meg"),
        eog=(picks == "eog"),
        ecg=(picks == "ecg"),
        exclude="bads",
    ) if isinstance(picks, str) else np.array(picks, dtype=int)

    if len(pick_idx) == 0:
        raise ValueError("No channels selected for corruption.")

    # Robust per-channel scale from all epochs/timepoints
    x = data[:, pick_idx, :].transpose(1, 0, 2).reshape(len(pick_idx), -1)
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1)
    robust_sigma = 1.4826 * mad
    robust_sigma[robust_sigma == 0] = np.std(x, axis=1)[robust_sigma == 0] + 1e-12

    n_bad_epochs = max(1, int(round(frac_bad * n_epochs)))
    bad_epoch_idx = np.sort(rng.choice(n_epochs, size=n_bad_epochs, replace=False))

    bad_channel_mask = np.zeros((n_epochs, n_channels), dtype=bool)
    segment_mask = np.zeros((n_epochs, n_times), dtype=bool)
    records = []

    for ep in bad_epoch_idx:
        n_bad_ch = rng.integers(1, min(max_bad_channels, len(pick_idx)) + 1)
        ep_bad_picks = rng.choice(pick_idx, size=n_bad_ch, replace=False)

        seg_len = rng.integers(
            int(seg_len_s[0] * sfreq),
            max(int(seg_len_s[1] * sfreq), int(seg_len_s[0] * sfreq) + 1)
        )
        seg_len = min(seg_len, n_times - 1)

        start = rng.integers(0, n_times - seg_len)
        stop = start + seg_len
        segment_mask[ep, start:stop] = True

        for ch in ep_bad_picks:
            local_idx = np.where(pick_idx == ch)[0][0]
            sigma = robust_sigma[local_idx]
            amp = rng.uniform(*amplitude_mult) * sigma
            sign = rng.choice([-1.0, 1.0])
            art_type = rng.choice(artifact_types)

            if art_type == "burst":
                # Broadband noisy burst with taper
                burst = rng.standard_normal(seg_len)
                burst *= np.hanning(seg_len)
                burst = sign * amp * burst

            elif art_type == "step":
                # Electrode pop / step-like shift
                burst = np.ones(seg_len) * sign * amp

            elif art_type == "drift":
                # Slow ramp
                burst = sign * amp * np.linspace(0, 1, seg_len)

            elif art_type == "flat":
                # Flatline / dropout
                fill_value = np.median(data[:, ch, :])
                data[ep, ch, start:stop] = fill_value
                bad_channel_mask[ep, ch] = True
                records.append({
                    "epoch": int(ep),
                    "channel": int(ch),
                    "start": int(start),
                    "stop": int(stop),
                    "type": art_type,
                    "amplitude": float(amp),
                })
                continue

            else:
                raise ValueError(f"Unknown artifact type: {art_type}")

            data[ep, ch, start:stop] += burst
            bad_channel_mask[ep, ch] = True

            records.append({
                "epoch": int(ep),
                "channel": int(ch),
                "start": int(start),
                "stop": int(stop),
                "type": art_type,
                "amplitude": float(amp),
            })

    epochs_corrupt._data = data

    info = {
        "bad_epoch_idx": bad_epoch_idx,
        "bad_channel_mask": bad_channel_mask,
        "segment_mask": segment_mask,
        "records": records,
    }
    return epochs_corrupt, info
```

---

# Example usage

```python
epochs = mne.make_fixed_length_epochs(raw, duration=30.0, preload=True)

epochs_corrupt, gt = inject_mock_artifacts(
    epochs,
    frac_bad=0.25,
    max_bad_channels=3,
    seg_len_s=(0.5, 2.0),
    amplitude_mult=(8.0, 20.0),
    artifact_types=("burst", "step", "drift", "flat"),
    picks="eeg",
    random_state=42,
)

print("Bad epochs:", gt["bad_epoch_idx"])
print("Number of corrupted epochs:", len(gt["bad_epoch_idx"]))
print("First record:", gt["records"][0])
```

---

# If you want blink-like corruption specifically

For blink-threshold experiments, random noise is okay, but **blink-shaped artifacts** are better.

Typical blink simulation:

* strongest on frontal channels
* transient bump
* duration around 200–500 ms
* same event appears across several frontal channels with different amplitudes

Example signal shape:

```python
def gaussian_blink(seg_len, amp, sign=1.0):
    x = np.linspace(-2.5, 2.5, seg_len)
    return sign * amp * np.exp(-0.5 * x**2)
```

Then inject that into channels like `Fp1, Fp2, Fz, AF3, AF4` with spatially decaying amplitude.

That is better if your downstream goal is estimating blink threshold from MAD.

---

# How to make the benchmark meaningful

Do not use only one artifact strength. Make 3 regimes:

## Mild

* 1 bad channel
* 0.2 to 0.5 s segment
* amplitude 4 to 8 × robust sigma

## Moderate

* 2 to 4 bad channels
* 0.5 to 2 s segment
* amplitude 8 to 15 × robust sigma

## Severe

* many channels or long segment
* 1 to 5 s
* amplitude 15 to 30 × robust sigma

That tells you where your method starts to fail.

---

# Very important caution

If you inject artifacts into already dirty epochs, your ground truth becomes ambiguous.

Better workflow:

1. select a relatively clean subset first
2. copy it
3. inject synthetic artifacts only into the copy
4. compare detected bad epochs against known injected ones

---

# What to evaluate

After running your rejection method, compare:

* predicted bad epochs vs injected bad epochs
* precision
* recall
* F1 score

Example:

```python
from sklearn.metrics import precision_score, recall_score, f1_score

y_true = np.zeros(len(epochs), dtype=int)
y_true[gt["bad_epoch_idx"]] = 1

# example: predicted_bad_idx from your method
y_pred = np.zeros(len(epochs), dtype=int)
y_pred[predicted_bad_idx] = 1

print("precision:", precision_score(y_true, y_pred))
print("recall:", recall_score(y_true, y_pred))
print("f1:", f1_score(y_true, y_pred))
```

---

# Best match to your earlier autoreject question

Since autoreject is based on large peak-to-peak excursions, the most relevant synthetic corruptions are:

* step / electrode pop
* noisy burst
* drift
* flatline
* blink-shaped transient

Those are much more informative than adding tiny white noise everywhere.

