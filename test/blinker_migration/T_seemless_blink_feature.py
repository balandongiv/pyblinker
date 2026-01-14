"""
We will modify the blink properties coding so that it can work seamlessly with code under
pyblinker/blink_features/morphology and pyblinker/blink_features/kinematics
This is to reduce the code redundancy and improve maintainability.
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_PATH = REPO_ROOT / "tutorial" / "blinkprops_case01.pkl"

from pyblinker.blink_features.waveform_features.extract_blink_properties import (  # noqa: E402
    BlinkProperties,
)
from pyblinker.utils.blinker_feature import (  # noqa: E402
    load_blinkprops_pickle,
    replay_and_assert_blinkprops,
)

fx = load_blinkprops_pickle(DATA_PATH)
replay_and_assert_blinkprops(fx, BlinkProperties)
