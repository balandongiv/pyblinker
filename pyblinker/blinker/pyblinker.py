from collections.abc import Mapping

from pyblinker.logging import get_logger

from . import default_setting
from .legacy_eeglab_filter import legacy_blinker_bandpass
from ..utils.annotation_utils import create_annotation
from ..viz.viz_pd import viz_complete_blink_prop
from ..pipeline_steps import (
    process_channel_data as core_process_channel_data,
    process_all_channels as core_process_all_channels,
    select_representative_channel as core_select_representative_channel,
    get_representative_blink_data as core_get_representative_blink_data,
    get_blink as core_get_blink,
)


logger = get_logger(__name__)


class BlinkDetector:
    @staticmethod
    def _build_detector_params(blink_params, blink_param_overrides):
        """Merge constructor blink-parameter overrides with canonical defaults."""

        if blink_params is not None and not isinstance(blink_params, Mapping):
            raise TypeError("blink_params must be a mapping of parameter overrides.")

        merged_overrides = {}
        if blink_params is not None:
            merged_overrides.update(blink_params)
        merged_overrides.update(blink_param_overrides)

        try:
            return default_setting.build_blink_params(merged_overrides)
        except KeyError as exc:
            raise TypeError(str(exc)) from exc

    def __init__(
        self,
        raw_data,
        visualize=False,
        annot_label=None,
        filter_bad=False,
        filter_low=1.0,
        filter_high=20.0,
        resample_rate=30,
        n_jobs=1,
        use_multiprocessing=False,
        pick_types_options=None,
        pipeline="legacy",
        blink_params=None,
        **blink_param_overrides,
    ):
        """
        Initialize the BlinkDetector.

        Parameters:
            raw_data: The raw EEG data.
            visualize (bool): Whether to generate visualization data.
            annot_label (str): Annotation label for blink events.
            filter_bad (bool): Whether to filter out bad blinks.
            filter_low (float): Low frequency filter cutoff.
            filter_high (float): High frequency filter cutoff.
            resample_rate (int): New sampling rate for the data.
            n_jobs (int): Number of jobs to use for processing.
            use_multiprocessing (bool): Whether to use multiprocessing.
            pick_types_options (dict): Dictionary of channel type options to pass
                                       to raw_data.pick_types, e.g. {'eeg': True, 'eog': True}.
            blink_params (dict): Optional dictionary of overrides for values in
                                 ``pyblinker.blinker.default_setting.DEFAULT_PARAMS``.
            **blink_param_overrides: Optional direct keyword overrides for values in
                                     ``DEFAULT_PARAMS``. These take precedence over
                                     ``blink_params`` when both are provided. For
                                     example, ``std_threshold=2.0`` or
                                     ``z_thresholds=np.array(...)``.
        """
        self.filter_bad = filter_bad
        self.raw_data = raw_data
        self.viz_data = visualize
        self.annot_label = annot_label
        self.sfreq = self.raw_data.info["sfreq"]
        self.params = self._build_detector_params(blink_params, blink_param_overrides)
        self.params["sfreq"] = float(self.sfreq)
        self.channel_list = self.raw_data.ch_names
        self.all_data_info = []  # To store processed blink data per channel
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.resample_rate = resample_rate
        self.n_jobs = n_jobs
        self.use_multiprocessing = use_multiprocessing
        self.all_data = []
        self.pipeline = pipeline
        # Default to picking only EEG if none is provided
        self.pick_types_options = (
            pick_types_options if pick_types_options is not None else {"eeg": True}
        )

    def prepare_raw_signal(self):
        """
        Preprocess raw signal:
          - pick channel types
          - filter
          - resample
        """
        logger.info(
            "Preparing raw signal: picking channels, filtering, and resampling."
        )
        self.raw_data.load_data()
        self.raw_data.pick_types(**self.pick_types_options)
        self.channel_list = list(self.raw_data.ch_names)

        filtered = legacy_blinker_bandpass(
            self.raw_data.get_data(),
            sfreq=float(self.raw_data.info["sfreq"]),
            low_cutoff_hz=float(self.filter_low),
            high_cutoff_hz=float(self.filter_high),
        )
        self.raw_data._data = filtered

        if float(self.raw_data.info["sfreq"]) != float(self.resample_rate):
            self.raw_data.resample(self.resample_rate, n_jobs=self.n_jobs)

        self.sfreq = float(self.raw_data.info["sfreq"])
        self.params["sfreq"] = self.sfreq
        logger.info(
            "Signal prepared with legacy Blinker bandpass and resample rate: %s Hz",
            self.sfreq,
        )
        return self.raw_data

    def process_channel_data(self, channel, verbose=True):
        """Process data for a single channel."""
        core_process_channel_data(self, channel, verbose)

    @staticmethod
    def filter_point(ch, all_data_info):
        """Helper to extract data information for a specific channel."""
        return list(filter(lambda data: data["ch"] == ch, all_data_info))[0]

    def filter_bad_blink(self, df):
        """Optionally filter out bad blinks."""
        if self.filter_bad:
            df = df[df["blink_quality"] == "Good"]
        return df

    def generate_viz(self, data, df):
        """Generate visualization for each blink if visualization is enabled."""
        fig_data = [
            viz_complete_blink_prop(data, row, self.sfreq) for _, row in df.iterrows()
        ]
        return fig_data

    def process_all_channels(self):
        """Process all channels available in the raw data."""
        core_process_all_channels(self)

    def select_representative_channel(self):
        """Select the best representative channel based on blink statistics."""
        return core_select_representative_channel(self)

    def get_representative_blink_data(self, ch_selected):
        """Retrieve blink data from the selected representative channel."""
        return core_get_representative_blink_data(self, ch_selected)

    def create_annotations(self, df):
        """Create annotations based on the blink data."""
        annot_description = self.annot_label if self.annot_label else "eye_blink"
        return create_annotation(df, self.sfreq, annot_description)

    def get_blink(self):
        """
        Run the complete blink detection pipeline:
            - Prepare raw signal
            - Process all channels
            - Select representative channel
            - Create annotations
            - Generate visualizations (optional)
        """
        logger.info(f"Starting blink detection pipeline (strategy: {self.pipeline}).")

        if self.pipeline == "mne":
            from .mne_pipeline import get_mne_blink
            # For MNE pipeline, we can pass None to let it auto-detect EOG, or default to the first picked channel
            channel = self.channel_list[0] if self.channel_list else None
            return get_mne_blink(self, channel=channel)

        return core_get_blink(self)
