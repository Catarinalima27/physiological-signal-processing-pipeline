import os

from src.ecg.load_signal import load_ecg_data
from src.ecg.features import extract_kubios_features
from pipelines.all_phases_pipeline import run_all_phases_pipeline


# Create results directory if it does not exist
os.makedirs("results", exist_ok=True)


# Run ECG feature extraction pipeline across all study phases
# (Pre-intervention D1/D2, Intervention, Post-intervention, Follow-up)
run_all_phases_pipeline(
    load_path= r"F:/",
    load_data=load_ecg_data,
    extract_features=extract_kubios_features,
    pickle_path="Students_recording_metadata.pkl",
    group_information_path="all_phases_group_information.py",
    output_excel="results/database_all_phases_ecg.xlsx",
    sampling_rate=100,
    check_quality=True
)