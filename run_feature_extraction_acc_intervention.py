import os

from src.acc.load_signal import load_acc_data
from src.acc.features import extract_acc_features
from pipelines.intervention_pipeline import run_intervention_pipeline


# Create results directory if it does not exist
os.makedirs("results", exist_ok=True)


def load_acc_data_no_trim(filepath, sampling_rate, show_signal=False):
    """
    Load ACC signal without removing the beginning and end of the recording.

    Used for intervention analysis where the full recording timeline
    is required to identify intervention-specific intervals.
    """
    return load_acc_data(
        filepath,
        sampling_rate,
        trim_edges=False,
        show_signal=show_signal
    )


# Run ACC intervention analysis pipeline
# Extracts features from predefined intervention intervals
run_intervention_pipeline(
    load_path = r"F:/",
    load_data=load_acc_data_no_trim,
    extract_features=extract_acc_features,
    pickle_path="Students_recording_metadata.pkl",
    interval_and_group_information_path="Intervention_interval_and_group_information.py",
    output_excel="results/database_intervention_acc.xlsx",
    sampling_rate=100
)