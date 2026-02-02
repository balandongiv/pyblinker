"""
We try to immitate the behavior of the first stage of the blinker
In the first major part, under the pop_blinker, as shown in step0_pop_blinker.m,
we have call function name
[blinks, params] = extractBlinksEEG(EEG, params);
and it will output parameter such as
numberBlinks,numberGoodBlinks, blinkAmpRatio, cutoff,bestMedian,bestRobustStd,blinkPositions.
To avoid change the MATLAB code,and we also want to validate (all except fitBliks as we already have a dedicated file to validate it)

   +--------------------------------------------------------------+
        |        Step 3A: extractBlinks(...) (Candidate Selection)      |
        |                 [Loop over signalData(k)]                     |
        |                                                              |
        |   ✓ fitBlinks per candidate                                   |
        |   ✓ Compute blinkAmpRatio / goodRatio / numberGoodBlinks      |
        |   ✓ Filter by blinkAmpRange                                   |
        |   ✓ Filter by minGoodBlinks                                   |
        |   ✓ Apply goodRatioThreshold (may set usedSign=-1)            |
        |   ✓ Pick max(numberGoodBlinks) -> final used signal

so to do this, we may need to run to whole process, and compare with the MATLAB .mat output.

However, to avoid the need for downsampling, which mau cause other disprepancy, we will try to run the process without downsampling first. Meaning, we will skip
The following
if nargin < 2
    params = struct();
end

[params, errors] = checkBlinkerDefaults(params, getBlinkerDefaults(EEG));
if ~isempty(errors)
    error('extractBlinks:BadParameters', ['|' sprintf('%s|', errors{:})]);
end

%% Extract the candidate signals
if params.verbose
    fprintf('Extracting candidate signals...\n');
end
[candidateSignals, signalType, signalNumbers, ...
                signalLabels, params] = getCandidateSignals(EEG, params);
params.signalNumbers = signalNumbers;
params.signalLabels = signalLabels;
if params.verbose
    fprintf('Extracting blinks from the candidate signals... be patient....\n');
end

and directly do everything that
[blinks, params] = extractBlinks(candidateSignals, signalType, params);

The

"""

import logging
import unittest
from pathlib import Path

import mne
import numpy as np

from pyblinker.blinker import default_setting
from pyblinker.blinker.fit_blink import FitBlinks
from pyblinker.blinker.get_blink_positions import get_blink_position
from pyblinker.utils.statistics_utils import get_blink_statistic
from pyblinker.utils.statistics_utils import get_good_blink_mask
# pyblinker/blinker/default_setting.py
from pyblinker.blinker.default_setting import DEFAULT_PARAMS
from scipy.io import loadmat
# -----------------------------------------------------------------------------
# Logger configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def assert_close(name, ref, got, rtol=1e-6, atol=1e-8):
	"""Simple MATLAB-vs-Python assertion with tolerance (scalars or arrays)."""
	np.testing.assert_allclose(
		np.asarray(ref),
		np.asarray(got),
		rtol=rtol,
		atol=atol,
		equal_nan=True,
		err_msg=f"{name} mismatch: ref={ref} got={got}",
		)
# -----------------------------------------------------------------------------
# Test class
# -----------------------------------------------------------------------------
class TestFitBlinks(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		"""
		Load test data, run FitBlinks, and load MATLAB reference output.
		"""
		base_path = Path(__file__).resolve().parents[1] / "migration_files"
		fif_path = Path("test/test_files/ear_eog_raw.fif")
		mat_path = base_path / "step5_data_output_extract_blinks_rpb.mat"
		assert mat_path.exists(), f"Missing MATLAB file: {mat_path}"
		mat_data = loadmat(
			mat_path,
			squeeze_me=True,
			simplify_cells=True,
			struct_as_record=False,
			)
		# numberGoodBlinks_ref = mat_data['signalData']["numberGoodBlinks"]
		# blinkAmpRatio_ref = mat_data['signalData']["blinkAmpRatio"]
		# cutoff_ref = mat_data['signalData']["cutoff"]
		# bestMedian_ref = mat_data['signalData']["bestMedian"]
		# bestRobustStd_ref = mat_data['signalData']["bestRobustStd"]
		# goodRatio_ref = mat_data['signalData']["goodRatio"]
		# blinkPositions_ref = mat_data['signalData']["blinkPositions"]
		sig = mat_data["signalData"]

		cls.numberGoodBlinks_ref = sig["numberGoodBlinks"]
		cls.blinkAmpRatio_ref = sig["blinkAmpRatio"]
		cls.cutoff_ref = sig["cutoff"]
		cls.bestMedian_ref = sig["bestMedian"]
		cls.bestRobustStd_ref = sig["bestRobustStd"]
		cls.goodRatio_ref = sig["goodRatio"]
		cls.blinkPositions_ref = sig["blinkPositions"]

	# ---------------------------------------------------------------------
		# Load raw FIF data
		# ---------------------------------------------------------------------
		raw = mne.io.read_raw_fif(
			fif_path, preload=True, verbose="ERROR"
			)

		ch_name = "EEG-E8"
		if ch_name not in raw.ch_names:
			# Case-insensitive fallback
			ch_map = {c.lower(): c for c in raw.ch_names}
			ch_name = ch_map.get(ch_name.lower(), ch_name)

		raw = raw.copy().pick_channels([ch_name])

		if int(round(raw.info.get("sfreq", 100))) != 100:
			raw.resample(100)

		blink_comp = raw.get_data()[0].astype(np.float64)

		# ---------------------------------------------------------------------
		# Detect blink positions
		# ---------------------------------------------------------------------
		params = dict(
			min_event_len=0.05,
			std_threshold=1.5,
			sfreq=100,
			)

		params_default = default_setting.DEFAULT_PARAMS.copy()

		df_positions = get_blink_position(
			params,
			blink_component=blink_comp,
			ch="No_channel",
			progress_bar=False,
			)

		# ---------------------------------------------------------------------
		# Run FitBlinks
		# ---------------------------------------------------------------------
		fitblinks = FitBlinks(
			candidate_signal=blink_comp,
			df=df_positions,
			params=params_default,
			)
		fitblinks.dprocess()

		# df_output = fitblinks.frame_blinks
		df = fitblinks.frame_blinks

		# STEP 3: Extract blink statistics extractBlinkProperties.m
		# Calculate an amplitude criterion (frames in blink to those out) and Now calculate the cutoff ratios -- use default for the values. now, it seem we have issue with this get_blink_statistic
		blink_stats = get_blink_statistic(
			df,
			params_default["z_thresholds"],
			signal=blink_comp,
			)
		cls.number_good_blinks_py = blink_stats["number_good_blinks"]
		cls.blink_amp_ratio_py = blink_stats["blink_amp_ratio"]
		cls.cutoff_py = blink_stats["cutoff"]
		cls.best_median_py = blink_stats["best_median"]
		cls.best_robust_std_py = blink_stats["best_robust_std"]
		cls.good_ratio_py = blink_stats["good_ratio"]
		H=1

		# blink_stats["ch"] = channel
		# There is a step for << Reduce the number of candidate signals based on the blink amp ratios >>, but we move it to channel selection step.
		#
		# # STEP 4: Get good blink mask extractBlinkProperties.m
		_, df = get_good_blink_mask(
			df,
			blink_stats["best_median"],
			blink_stats["best_robust_std"],
			params_default["z_thresholds"],
			)

	def test_match_matlab_outputs(self):
		"""
		Compare the key MATLAB outputs with Python outputs using tolerance.
		"""
		assert_close("numberGoodBlinks", self.numberGoodBlinks_ref, self.number_good_blinks_py)
		assert_close("blinkAmpRatio", self.blinkAmpRatio_ref, self.blink_amp_ratio_py)
		assert_close("cutoff", self.cutoff_ref, self.cutoff_py)
		assert_close("bestMedian", self.bestMedian_ref, self.best_median_py)
		assert_close("bestRobustStd", self.bestRobustStd_ref, self.best_robust_std_py)
		assert_close("goodRatio", self.goodRatio_ref, self.good_ratio_py)

		# If you later compute python blink positions, you can add:
		# assert_close("blinkPositions", self.blinkPositions_ref, self.blink_positions_py, rtol=0, atol=0)

	def test_reach_setupclass(self):
		# This ensures unittest collects/runs the class (and thus setUpClass)
		self.assertTrue(True)
if __name__ == "__main__":
	unittest.main()
