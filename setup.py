from setuptools import setup

setup(
    name="pyblinker",
    version="0.4.2",
    description="An open-source alternative to MATLAB BLINKER, integrated with MNE-Python, supporting multimodal biosignal research with flexible workflows, reproducible metrics, and enhanced accessibility.",
    author="rpb",
    packages=["pyblinker", "pyblinker.utilities", "pyblinker.vislab", "pyblinker.viz"],
    install_requires=["seaborn", "mne"],
)
