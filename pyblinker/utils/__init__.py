"""Utility functions for pyblinker."""

from .annotation_utils import create_annotation
from .channel_utils import (
    normalize_picks,
    pick_ear_channels_from_info,
    pick_ear_channels_from_raw,
    require_channels,
)
from .epoch_utils import (
    slice_into_mini_raws,
    slice_raw_into_epochs,
    slice_raw_into_mne_epochs,
    slice_raw_to_segments,
)
from .feature_dataframe import to_epoch_indexed
from .io_utils import prepare_refined_segments, save_epoch_raws
from .metadata_utils import onset_entry_to_blinks
from .modality import infer_modality
from .refinement_utils import (
    plot_refined_blinks,
    refine_blinks_from_epochs,
    refine_ear_extrema_and_threshold_stub,
    refine_local_maximum_stub,
    slice_raw_into_mne_epochs_refine_annot,
)
from .report_utils import add_blink_plots_to_report, generate_epoch_report
from .statistics_utils import (
    calculate_good_ratio,
    calculate_within_range,
    get_blink_statistic,
    get_good_blink_mask,
    get_max_blink,
)
from .velocity_utils import average_velocity

__all__ = [
    "create_annotation",
    "normalize_picks",
    "pick_ear_channels_from_info",
    "pick_ear_channels_from_raw",
    "require_channels",
    "slice_raw_to_segments",
    "slice_raw_into_mne_epochs",
    "slice_raw_into_epochs",
    "slice_into_mini_raws",
    "save_epoch_raws",
    "prepare_refined_segments",
    "to_epoch_indexed",
    "generate_epoch_report",
    "add_blink_plots_to_report",
    "refine_ear_extrema_and_threshold_stub",
    "refine_local_maximum_stub",
    "refine_blinks_from_epochs",
    "plot_refined_blinks",
    "slice_raw_into_mne_epochs_refine_annot",
    "onset_entry_to_blinks",
    "calculate_within_range",
    "calculate_good_ratio",
    "get_blink_statistic",
    "get_good_blink_mask",
    "get_max_blink",
    "average_velocity",
    "infer_modality",
]
