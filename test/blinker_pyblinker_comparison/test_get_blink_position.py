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

import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pyblinker.blinker import default_setting
from pyblinker.blinker.get_blink_positions import get_blink_position
from scipy.io import loadmat
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
        # fif_path=r'C:\Users\balan\IdeaProjects\pyblinker\test\test_files\ear_eog_raw.fif'
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


        blink_positions_ref = np.array(sig["blinkPositions"]).squeeze()
        if blink_positions_ref.ndim == 1:
            blink_positions_ref = blink_positions_ref.reshape(2, 1)
        if blink_positions_ref.shape[0] != 2 and blink_positions_ref.shape[1] == 2:
            blink_positions_ref = blink_positions_ref.T
        cls.blink_positions_ref = blink_positions_ref.astype(np.int64) - 1

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


        params_default = default_setting.DEFAULT_PARAMS.copy()

        df_positions = get_blink_position(
            params_default,
            blink_component=blink_comp,
            ch="No_channel",
            progress_bar=False,
        )
        cls.blink_positions_py = df_positions

        # ---------------------------------------------------------------------
        # Run FitBlinks
        # ---------------------------------------------------------------------


    def test_match_matlab_outputs(self):
        """
        Compare the key MATLAB outputs with Python outputs using tolerance.
        """

        # If you later compute python blink positions, you can add:
        df_mat = pd.DataFrame(
            {
                "start_blink": self.blink_positions_ref[0, :].astype(np.int64),
                "end_blink": self.blink_positions_ref[1, :].astype(np.int64),
            }
        )
        df_py = self.blink_positions_py.astype(
            {"start_blink": np.int64, "end_blink": np.int64}, errors="ignore"
        )
        pd.testing.assert_frame_equal(
            df_py.reset_index(drop=True),
            df_mat.reset_index(drop=True),
            check_dtype=False,
        )

    def test_reach_setupclass(self):
        # This ensures unittest collects/runs the class (and thus setUpClass)
        self.assertTrue(True)
if __name__ == "__main__":
    unittest.main()
#
