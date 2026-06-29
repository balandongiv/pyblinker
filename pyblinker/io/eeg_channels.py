"""EEG I/O helpers for loading channel configuration and raw FIF data."""

from __future__ import annotations

from pathlib import Path

import mne
import yaml


def load_brain_region_map(yaml_path: Path) -> dict[str, list]:
    """Return ``{region_name: [channel_entry, ...]}`` from a brain-region YAML.

    Channel entries are returned verbatim (they may be integers, e.g. EGI
    HydroCel indices such as ``13``, or strings, e.g. ``"Fp1"``).  Use
    :func:`resolve_channel_names` to map them onto a recording's actual channel
    names.
    """
    with yaml_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return {region: list(channels) for region, channels in config["eeg_regions"].items()}


def load_brain_region_channels(yaml_path: Path) -> list[str]:
    """Return a flat list of channel entries from a brain-region YAML config.

    Entries preserve their YAML type (int or str).  Callers that need names that
    match a specific recording should pass the result through
    :func:`resolve_channel_names`.
    """
    channels: list = []
    for region_channels in load_brain_region_map(yaml_path).values():
        channels.extend(region_channels)
    return channels


def resolve_channel_names(
    entries: list,
    available_ch_names: list[str],
) -> list[str]:
    """Map brain-region YAML entries onto a recording's actual channel names.

    Brain-region configs are written in dataset-native conventions, which do not
    always match the channel labels stored in the FIF file:

    * EGI HydroCel montages (Raja) store channels as ``E1 … E128`` but the YAML
      lists bare integer indices (``13``).  These are matched by prefixing ``E``.
    * 10-20 montages (Cao2018) store upper-case labels (``FP1``) while the YAML
      may use conventional mixed case (``Fp1``).  These are matched
      case-insensitively.

    Already-correct entries (e.g. ``"E22"``) match directly.  The returned list
    contains the *actual* channel names, in YAML order, de-duplicated.  Entries
    that match nothing are silently skipped.
    """
    available = list(available_ch_names)
    direct = set(available)
    upper_map: dict[str, str] = {}
    for ch in available:
        upper_map.setdefault(ch.upper(), ch)

    resolved: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        token = str(entry).strip()
        match: str | None = None
        for candidate in (token, f"E{token}"):
            if candidate in direct:
                match = candidate
                break
            if candidate.upper() in upper_map:
                match = upper_map[candidate.upper()]
                break
        if match is not None and match not in seen:
            resolved.append(match)
            seen.add(match)
    return resolved


def load_raw_with_brain_channels(
    fif_path: Path,
    brain_channels: list,
) -> mne.io.BaseRaw:
    """Load a FIF file and retain only the resolved brain-region channels."""
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose="ERROR")
    available = resolve_channel_names(brain_channels, raw.ch_names)
    if not available:
        raise ValueError(
            f"None of the {len(brain_channels)} brain-region entries matched any "
            f"channel in {Path(fif_path).name}. First entries: {brain_channels[:5]}; "
            f"recording channels: {raw.ch_names[:5]}…"
        )
    raw.pick(available)
    return raw


__all__ = [
    "load_brain_region_map",
    "load_brain_region_channels",
    "resolve_channel_names",
    "load_raw_with_brain_channels",
]
