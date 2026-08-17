import os

from src.ecg.load_signal import load_ecg_data
from src.ecg.features import extract_kubios_features
from pipelines.intervention_pipeline import run_intervention_pipeline


# Create results directory if it does not exist
os.makedirs("results", exist_ok=True)


def extract_ecg_no_quality(signal, sampling_rate):
    """
    Extract ECG features without applying signal quality filtering.
    
    Used for intervention analysis where feature extraction is performed
    on all available signal segments.
    """
    return extract_kubios_features(
        signal,
        sampling_rate,
        check_quality=False
    )


def load_ecg_data_no_trim(filepath, sampling_rate, show_signal=False):
    """
    Load ECG signal without removing recording edges.

    Required for intervention analysis where the complete recording
    timeline is used to define pre-, during-, and post-intervention intervals.
    """
    return load_ecg_data(
        filepath,
        sampling_rate,
        trim_edges=False,
        show_signal=show_signal
    )


# Run ECG intervention analysis pipeline
# Extracts features from predefined intervention intervals
run_intervention_pipeline(
    load_path = r"F:/",
    load_data=load_ecg_data_no_trim,
    extract_features=extract_ecg_no_quality,
    pickle_path="Students_recording_metadata.pkl",
    interval_and_group_information_path="Intervention_interval_and_group_information.py",
    output_excel="results/database_intervention_ecg.xlsx",
    sampling_rate=100
)