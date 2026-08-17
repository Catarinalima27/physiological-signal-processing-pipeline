import os

from src.acc.load_signal import load_acc_data
from src.acc.features import extract_acc_features
from pipelines.all_phases_pipeline import run_all_phases_pipeline


# Create results directory if it does not exist
os.makedirs("results", exist_ok=True)


# Run ACC feature extraction pipeline across all study phases
# (Pre-intervention D1/D2, Intervention, Post-intervention, Follow-up)
run_all_phases_pipeline(
    load_path = r"F:/",
    load_data=load_acc_data,
    extract_features=extract_acc_features,
    pickle_path="Students_recording_metadata.pkl",
    group_information_path="all_phases_group_information.py",
    output_excel="results/database_all_phases_acc.xlsx",
    sampling_rate=100
)