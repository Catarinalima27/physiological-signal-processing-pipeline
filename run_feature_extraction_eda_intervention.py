import os

from src.eda.load_signal import load_eda_data
from src.eda.features import extract_eda_features
from pipelines.intervention_pipeline import run_intervention_pipeline


# Create results directory if it does not exist
os.makedirs("results", exist_ok=True)


def load_eda_data_no_trim(filepath, sampling_rate, show_signal=False):
    """
    Load EDA signal without removing recording edges.

    Required for intervention analysis where the complete recording
    timeline is used to identify pre-, during-, and post-intervention intervals.
    """
    return load_eda_data(
        filepath,
        sampling_rate,
        trim_edges=False,
        show_signal=show_signal
    )


# Run EDA intervention analysis pipeline
# Extracts features from predefined intervention intervals
run_intervention_pipeline(
    load_path = r"F:/",
    load_data=load_eda_data_no_trim,
    extract_features=extract_eda_features,
    pickle_path="Students_recording_metadata.pkl",
    interval_and_group_information_path="Intervention_interval_and_group_information.py",
    output_excel="results/database_intervention_eda.xlsx",
    sampling_rate=100
)