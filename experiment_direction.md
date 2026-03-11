


The possible adaptation is:

1. Treat each 30-second epoch as a “trial”.
2. Compute a per-epoch artifact score from peak-to-peak amplitude.
3. Learn the rejection threshold from the data by cross-validation, instead of choosing it manually.
4. Mark epochs above that learned threshold as bad.
5. Use only the remaining good epochs to compute the MAD for blink-threshold estimation.

The key point from the paper is not just “use peak-to-peak”; it is **learn the peak-to-peak cutoff automatically** by choosing the threshold that makes the average of retained training epochs look most like the robust center of held-out epochs.

## What to copy from the paper

### Option A: use the paper’s global version

This is the simplest and probably the best match if your only goal is:

* get a list of bad epochs
* do not repair channels
* keep only good epochs for later MAD computation

For each epoch (i) and channel (j), compute peak-to-peak amplitude:

[
A_{ij} = \max(x_{ij}) - \min(x_{ij})
]

Then define one epoch-level score:

[
S_i = \max_j A_{ij}
]

So an epoch is bad if **any channel** has unusually large peak-to-peak amplitude.

Then learn the threshold (\tau) by cross-validation exactly like the paper:

* split epochs into K folds
* for each candidate threshold (\tau):

    * in the training fold, keep epochs with (S_i \le \tau)
    * compute the **mean** of those kept training epochs
    * in the validation fold, compute the **median** across all validation epochs
    * score the threshold by RMSE (or Frobenius norm) between training mean and validation median
* choose the (\tau) with lowest average CV error

Finally:

[
\text{bad epochs} = { i : S_i > \tau^* }
]

That gives you the list you want.

## Why this works

The paper’s logic is:

* if (\tau) is too small, you reject too many epochs, so the training average becomes unstable
* if (\tau) is too large, you keep artifact-heavy epochs, so the training average gets contaminated
* the best threshold is the one where the retained training data best matches the robust central tendency of unseen data

That is the whole “automatic rejection” idea.

## For your use case, this is probably enough

Because you only want **a list of good/bad epochs before MAD**, I would start with the **global** approach, not the full local autoreject procedure.

The full local method is more useful when you want to:

* identify bad channels inside each epoch
* interpolate a few bad channels
* reject only epochs with too many bad channels

But if your downstream step is “only good epochs will be used”, then global thresholding is much simpler and more transparent.

---

# Better practical variant for EEG/MNE

For real EEG, one bad channel can make many otherwise okay epochs look bad. So a more robust adaptation is the paper’s **local-to-epoch** logic without interpolation:

## Option B: local thresholds, then reject epochs by bad-channel count

This is often better than global max peak-to-peak.

### Step 1: learn one threshold per channel

For each channel (j), learn its own threshold (\tau_j), because channel amplitudes differ.

Then define:

[
C_{ij} =
\begin{cases}
1 & \text{if } A_{ij} > \tau_j \
0 & \text{otherwise}
\end{cases}
]

So (C_{ij}=1) means channel (j) is bad in epoch (i).

### Step 2: convert channel-level badness into epoch-level badness

Count how many channels are bad in each epoch:

[
B_i = \sum_j C_{ij}
]

Then reject epoch (i) if:

[
B_i \ge \kappa
]

where (\kappa) is the maximum number of bad channels allowed.

So the final bad-epoch list is:

[
\text{bad epochs} = { i : B_i \ge \kappa }
]

This is closer to the paper’s local autoreject idea.

## Why this may be better for you

Because otherwise a single chronically noisy channel can make lots of epochs fail.

For MAD estimation, you usually want epochs that are globally usable, not epochs ruined by one electrode only. So this rule is often more sensible:

* a few bad channels → epoch can still count as good
* many bad channels → reject epoch

If you do not want interpolation, just skip the repair step and use the reject/not-reject decision only.

---

# My recommendation

For your problem, I would use this hierarchy:

## Recommended version

Use **local thresholds + bad-channel count**, but **no interpolation**.

That means:

1. Epoch into 30s windows
2. For each epoch and channel, compute peak-to-peak
3. Learn channel-specific thresholds (\tau_j)
4. Mark bad channels per epoch
5. Reject epoch if number of bad channels exceeds (\kappa)
6. Use only surviving epochs for MAD blink-threshold estimation

This keeps the paper’s adaptive idea, but simplifies the output to exactly what you need: a list of bad epochs.

---

# Important caveat with 30-second epochs

This is the biggest issue.

The paper was mostly demonstrated on shorter event-related epochs. With 30-second epochs:

* peak-to-peak is much more likely to catch at least one transient
* one blink, movement, or drift within 30 seconds can make the whole epoch look bad
* rejection may become too aggressive

So the method still works, but the definition of “bad epoch” becomes harsher.

## Better workaround

Use short windows internally, even if your final unit is 30 seconds.

Example:

* split each 30-second epoch into 5-second subwindows
* run the same peak-to-peak screening on subwindows
* declare a 30-second epoch bad if too many subwindows are bad

This is often more stable than using one peak-to-peak number over the full 30 seconds.

For example:

* 6 subwindows per 30-second epoch
* mark subwindow bad if it exceeds learned threshold
* mark the 30-second epoch bad if at least 2 of 6 subwindows are bad

That is not exactly the paper, but it preserves its spirit and is usually better for long resting/state epochs.

---

# How to implement it in MNE terms

Assume `epochs` is shape:

* `n_epochs x n_channels x n_times`

### Global version

For each epoch:

```python
ptp = epochs.get_data().ptp(axis=2)   # shape: (n_epochs, n_channels)
epoch_score = ptp.max(axis=1)         # one score per epoch
```

Then learn `tau_star` by CV and reject:

```python
bad_idx = np.where(epoch_score > tau_star)[0]
good_idx = np.where(epoch_score <= tau_star)[0]
```

### Local version

```python
ptp = epochs.get_data().ptp(axis=2)   # (n_epochs, n_channels)
# thresholds_per_channel -> shape (n_channels,)
bad_channel_mask = ptp > thresholds_per_channel[None, :]
n_bad_channels = bad_channel_mask.sum(axis=1)
bad_idx = np.where(n_bad_channels >= kappa)[0]
good_idx = np.where(n_bad_channels < kappa)[0]
```

That `bad_idx` is the list you want.

---

# What the CV objective should be in your setting

You do not need to change the paper’s objective much.

For a candidate threshold:

* keep “good” training epochs
* average them
* compare with the median of validation epochs

Concretely:

```python
train_mean = X_train[good_train].mean(axis=0)
val_median = np.median(X_val, axis=0)
error = np.sqrt(np.mean((train_mean - val_median) ** 2))
```

Average that over folds. Choose the threshold with lowest error.

For local thresholds, do that per channel, or use the `autoreject` package if available.

---

# Very practical simplification

You do not actually need to re-implement the whole paper from scratch if your goal is only the bad-epoch list.

Two reasonable paths:

## Path 1: use `autoreject` directly

If your data are standard MNE EEG epochs and channel positions are available, `autoreject` already implements the paper.

You can fit it, then extract which epochs were considered bad / repaired.

## Path 2: implement only the global CV threshold

This is much easier and may be enough for your MAD preprocessing step.

If you want something simple, defensible, and paper-aligned, this is the minimum viable version.

---

# Decision rule I would use

If your real question is “which 30-second epochs are safe to use for estimating blink MAD?”, then I would define bad epochs as those meeting either of these:

1. **Too many channels exceed learned peak-to-peak thresholds**
2. **Frontal/EOG channels exceed a stricter blink-related threshold**

So combine a general artifact screen with a blink-specific screen.

That prevents your MAD estimate from being polluted by:

* large motion artifacts
* electrode pops
* extreme drift
* obvious blink-heavy segments

---

# Bottom line

The paper’s idea can be turned into a bad-epoch list like this:

## Simplest paper-faithful answer

* Treat each 30-second epoch as a trial
* Compute per-epoch max peak-to-peak across channels
* Learn the optimal threshold by cross-validation using training mean vs validation median
* Reject epochs whose score exceeds the learned threshold

## Better practical answer

* Learn **channel-specific** peak-to-peak thresholds
* Count bad channels in each epoch
* Reject the epoch if the count exceeds (\kappa)
* Skip interpolation if all you need is the bad-epoch list

## Best for 30-second data

* Apply the same logic on shorter subwindows inside each 30-second epoch
* Then promote subwindow decisions to an epoch decision

So the key output is:

[
\text{bad epoch list} = {i \mid \text{artifact score of epoch } i \text{ exceeds learned criterion}}
]

