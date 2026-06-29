"""I/O helpers for loading EEG recordings and channel configuration."""

from .eeg_channels import (
    load_brain_region_channels,
    load_brain_region_map,
    load_raw_with_brain_channels,
    resolve_channel_names,
)

__all__ = [
    "load_brain_region_channels",
    "load_brain_region_map",
    "load_raw_with_brain_channels",
    "resolve_channel_names",
]
