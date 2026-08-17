import numpy as np
import pandas as pd
import pickle
import re
import os
import importlib.util

def time_to_ms(time_str, recording_start):
    """
    Convert a timestamp into milliseconds relative to recording start.
    """
    h, m = map(int, time_str.split(":"))
    hs, ms = map(int, recording_start.split(":"))

    return ((h * 60 + m) - (hs * 60 + ms)) * 60 * 1000



def build_intervals(filepath, signal_duration_sec, df_intervals):
    """
    Define pre-, intervention-, and post-intervention intervals.

    For intervention recordings:
        - Pre: 15 minutes before intervention
        - During: intervention period
        - Post: 15 minutes after intervention

    For control recordings:
        - Three equally positioned intervals are extracted around the
          middle of the recording.
    """

    signal_duration_ms = signal_duration_sec * 1000

    # Remove recordings shorter than the minimum required duration
    if signal_duration_sec / 60 < 15:
        return None


    # Extract class identifier from filename
    match = re.search(r"T(\d+)", filepath, re.IGNORECASE)

    if not match:
        return None

    class_num = int(match.group(1)) - 1


    inicio = df_intervals["Inicio"].iloc[class_num]
    fim = df_intervals["Fim"].iloc[class_num]
    recording_start = df_intervals["RecordingStart"].iloc[class_num]


    # Short recordings use 5-minute windows, longer recordings use 15-minute windows
    if 15 < signal_duration_sec / 60 < 45:
        interval_ms = 5 * 60 * 1000
    else:
        interval_ms = 15 * 60 * 1000


    # Control group: divide recording into three representative segments
    if str(inicio).lower().startswith("control"):

        start_2 = max(0, (signal_duration_ms / 2) - (interval_ms / 2))
        end_2 = min(signal_duration_ms, start_2 + interval_ms)

        start_1 = max(0, start_2 - interval_ms)
        end_1 = start_1 + interval_ms

        start_3 = end_2
        end_3 = min(signal_duration_ms, start_3 + interval_ms)

        intervals = [
            (start_1, end_1),
            (start_2, end_2),
            (start_3, end_3)
        ]

    else:
        # Intervention group: extract before, during and after periods
        start_ms = time_to_ms(inicio, recording_start)
        duration_ms = time_to_ms(fim, recording_start) - start_ms

        shift_ms = 15 * 60 * 1000

        intervals = [
            (max(0, start_ms - shift_ms), start_ms),
            (start_ms, start_ms + duration_ms),
            (start_ms + duration_ms,
             min(signal_duration_ms, start_ms + duration_ms + shift_ms))
        ]


    return [
        (max(0, start), min(end, signal_duration_ms))
        for start, end in intervals
    ]



def run_intervention_pipeline(
    load_path,
    load_data,
    extract_features,
    pickle_path,
    interval_and_group_information_path,
    output_excel="results.xlsx",
    sampling_rate=100
):
    """
    Main pipeline:
    1. Loads the participant database from a pickle file.
    2. Iterates through every participant.
    3. Extracts intervention intervals
    4. Computes physiological features for each interval
    5. Stores participant metadata and extracted features.
    6. Exports the final dataset to an Excel file.
    """

    spec = importlib.util.spec_from_file_location(
        "interval_and_group_information",
        interval_and_group_information_path
    )

    interval_and_group_information = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(interval_and_group_information)

    df_intervals = interval_and_group_information.df_intervals    

    with open(pickle_path, "rb") as f:
        students_data = pickle.load(f)

    all_results = []

    for group in students_data:

        for student in group["Participants"]:

            subject_id = student.get("Student Code", "unknown")
            class_id = int(student.get("Class", -1))

            print(f"\nProcessing subject: {subject_id}")

            relative_path = student.get("Path Intervention")
            filepath = os.path.join(load_path, relative_path) if relative_path else None


            # Retrieve intervention group information
            if 1 <= class_id <= len(df_intervals):
                intervention = df_intervals.loc[class_id - 1, "Group"]
            else:
                intervention = "Unknown"


            result = {
                "subject": subject_id,
                "Class": class_id,
                "Intervention": intervention,
                "Sex": student.get("Sex", "N/A"),
                "Date": student.get("Date", "N/A")
            }


            # Keep participant information even if signal is unavailable
            if not filepath:
                all_results.append(result)
                continue


            signal = load_data(filepath, sampling_rate, show_signal=False)


            if signal is None or len(signal) == 0:
                all_results.append(result)
                continue


            duration_sec = signal.shape[0] / sampling_rate


            try:
                intervals = build_intervals(filepath, duration_sec)

            except Exception as e:
                print(f"Interval error: {e}")
                all_results.append(result)
                continue


            if intervals is None:
                all_results.append(result)
                continue



            # Extract features from each defined interval
            for i, (start_ms, end_ms) in enumerate(intervals, start=1):

                start_idx = int((start_ms / 1000) * sampling_rate)
                end_idx = int((end_ms / 1000) * sampling_rate)

                segment = signal[start_idx:end_idx]


                if len(segment) == 0:
                    continue


                try:
                    metrics = extract_features(segment, sampling_rate)

                except Exception as e:
                    print(f"Feature extraction error: {e}")
                    continue


                # Store metrics with interval identifier
                for col, value in metrics.items():

                    if isinstance(value, (list, tuple)):
                        value = str(value)

                    result[f"Int{i}_{col}"] = value


            all_results.append(result)



    df = pd.DataFrame(all_results)


    # Keep participant metadata before extracted features
    metadata_cols = [
        "subject",
        "Class",
        "Intervention",
        "Sex",
        "Date"
    ]

    metric_cols = [
        c for c in df.columns
        if c not in metadata_cols
    ]

    df = df[metadata_cols + metric_cols]


    df.to_excel(output_excel, index=False)

    print(f"Saved results: {output_excel}")


    return df