import pickle
import unittest

import numpy as np
import pandas as pd
from pathlib import Path

from pyblinker.blinker.get_blink_positions import get_blink_position
from unit_test.blinker_migration.pyblinker.utils.update_pkl_variables import RENAME_MAP, rename_keys

class TestGetBlinkPosition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load debug data once for all tests
        base_path = Path(__file__).resolve().parents[1] / "test_files"
        with (base_path / 'file_test_blink_position.pkl').open('rb') as f:
            cls.debug_data = pickle.load(f)
        cls.debug_data['params'] = rename_keys(cls.debug_data['params'], RENAME_MAP)

    def test_blink_detection(self):
        params = self.debug_data['params']
        blink_component = self.debug_data['blink_component']
        ch = self.debug_data['ch']
        expected_output = self.debug_data['output']
        expected_output = expected_output.rename(columns={'startBlinks': 'start_blink',
                                                          'endBlinks': 'end_blink'})

        # Run the function
        result = get_blink_position(
            params=params,
            blink_component=blink_component,
            ch=ch,
            progress_bar=False,
        )

        # Check that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check that the result has the same columns
        self.assertListEqual(list(result.columns), ['start_blink', 'end_blink'])

        # Check that the values are the same (both start and end)
        np.testing.assert_array_equal(result['start_blink'].values, expected_output['start_blink'].values)
        np.testing.assert_array_equal(result['end_blink'].values, expected_output['end_blink'].values)

if __name__ == '__main__':
    unittest.main()
