Here is a more detailed and polished project description:

**Project Description**

The goal of this project is to adapt the automatic artifact rejection approach proposed in the *autoreject* paper to a specific application: identifying poor-quality 30-second signal epochs before performing MAD-based blink threshold estimation.

We assume the input is a long continuous signal that will be segmented into consecutive 30-second epochs for analysis. In an ideal setting, all samples in the recording would be suitable for blink detection, and blink positions could be extracted directly using the existing implementation in `pyblinker/blinker/get_blink_positions.py`. In practice, however, the signal often contains substantial artifacts, such as motion noise, saturation, abrupt amplitude shifts, or other transient disturbances. These contaminated segments can distort downstream statistics and reduce the reliability of blink detection.

To address this, we propose introducing an automatic epoch rejection stage before MAD estimation. The key idea is borrowed from the *autoreject* framework, whose code is available at the provided repository. In the original method, rejection thresholds are learned automatically from the data using peak-to-peak amplitude as a quality metric. For this project, that idea will be adapted to the 30-second epoch structure of our signal.

More specifically, each 30-second epoch will be evaluated based on its peak-to-peak amplitude. Epochs with abnormally large peak-to-peak values will be treated as likely artifact-contaminated and excluded from further analysis. The remaining epochs, considered sufficiently clean, will then be passed to the MAD estimation step.

This preprocessing step is important because MAD-based threshold estimation is sensitive to contamination by extreme values. If heavily corrupted epochs are included, the estimated blink threshold may become inflated or biased, leading to poorer blink detection performance. By automatically filtering out bad epochs first, the MAD estimation can be based only on cleaner data, making the final blink threshold more robust and representative of true blink activity rather than artifacts.

In summary, the project aims to build an automatic quality-control layer for blink analysis by:

1. segmenting the continuous signal into 30-second epochs,
2. computing a peak-to-peak amplitude measure for each epoch,
3. learning or defining an automatic rejection threshold inspired by *autoreject*,
4. discarding highly contaminated epochs, and
5. using only the retained epochs for robust MAD-based blink threshold estimation.

The link to the *autoreject* codebase
https://github.com/autoreject/autoreject



