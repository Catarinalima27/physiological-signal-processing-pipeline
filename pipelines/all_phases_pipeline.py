import pickle
import pandas as pd
import os
import json

def run_all_phases_pipeline(
    load_path,
    load_data,
    extract_features,
    pickle_path,
    group_information_path,
    output_excel="results.xlsx",
    sampling_rate=100,
    check_quality=True
):
    """
    Main pipeline:
    1. Loads the participant database from a pickle file.
    2. Iterates through every participant.
    3. Extracts intervention intervals.
    4. Computes physiological features for each interval.
    5. Stores participant metadata and extracted features.
    6. Exports the final dataset to an Excel file.

    Parameters
    ----------
    load_path : str
        Path to the directory containing the physiological recordings.

    load_data : callable
        Function responsible for loading a physiological signal.

    extract_features : callable
        Function that extracts features from the loaded signal.

    pickle_path : str
        Path to the participant database (.pkl).

    interval_and_group_information_path : str
        Path to the file containing the intervention interval information.

    output_excel : str, optional
        Output Excel filename.

    sampling_rate : int, optional
        Signal sampling frequency (Hz).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing participant metadata and extracted features.
    """

    with open(group_information_path, "r") as f:
        intervention_map = json.load(f)

    # Load participant database
    with open(pickle_path, "rb") as f:
        students_data = pickle.load(f)

    all_results = []

    # Iterate through all participant groups
    for group in students_data:

        for student in group["Participants"]:

            subject_id = student.get("Student Code", "unknown")
            class_id = student.get("Class", "N/A")

            # Assign intervention group based on class ID
            try:
                intervention = intervention_map[int(class_id)]
            except:
                intervention = "Unknown"

            print(f"\n=========================")
            print(f"👤 Processing subject: {subject_id}")

            # Store participant metadata
            result = {
                "subject": subject_id,
                "Class": class_id,
                "Intervention": intervention,
                "Sex": student.get("Sex", "N/A"),
                "Date": student.get("Date", "N/A")
            }

            # Map each study phase to its corresponding recording
            phase_map = {
                "pre_D1": os.path.join(load_path, student.get("Path Pre D1")) if student.get("Path Pre D1") else None,
                "pre_D2": os.path.join(load_path, student.get("Path Pre D2")) if student.get("Path Pre D2") else None,
                "Int": os.path.join(load_path, student.get("Path Intervention")) if student.get("Path Intervention") else None,
                "Pos": os.path.join(load_path, student.get("Path Pos Intervention")) if student.get("Path Pos Intervention") else None,
                "Fol": os.path.join(load_path, student.get("Path Fol Intervention")) if student.get("Path Fol Intervention") else None,
            }

            # Process each available study phase
            for phase, filepath in phase_map.items():

                if not filepath:
                    print(f"⚠️ Missing {phase}")
                    continue

                print(f"➡️ {phase}: {filepath}")

                signal = load_data(
                    filepath,
                    sampling_rate
                )

                # Skip invalid or empty recordings
                if signal is None or len(signal) == 0:
                    print("❌ Invalid signal")
                    continue

                # Extract physiological features
                try:
                    metrics = extract_features(
                        signal,
                        sampling_rate
                    )
                except Exception:
                    continue

                # Store extracted features
                for col, value in metrics.items():

                    if isinstance(value, (list, tuple)):
                        value = str(value)

                    result[f"{phase}_{col}"] = value

            all_results.append(result)

    # Build the final results table
    df = pd.DataFrame(all_results)

    # Keep participant metadata before feature columns
    metadata_cols = [
        "subject",
        "Class",
        "Intervention",
        "Sex",
        "Date"
    ]

    other_cols = [
        c for c in df.columns
        if c not in metadata_cols
    ]

    df = df[metadata_cols + other_cols]

    # Export results
    df.to_excel(
        output_excel,
        index=False
    )

    print(f"\n✅ Saved: {output_excel}")

    return df