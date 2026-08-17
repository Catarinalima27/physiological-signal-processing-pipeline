import pandas as pd
import re
import ast


def cut_into_segments(signal, sampling_rate=100, segment_minutes=5):
    """
    Cut the ECG signal into fixed-length segments (default 5 minutes).

    Args:
        signal (np.array): ECG signal array
        sampling_rate (int): Samples per second
        segment_minutes (int): Segment length in minutes

    Returns:
        List[np.array]: List of ECG segments
    """
    segment_length = segment_minutes * 60 * sampling_rate
    segments = []
    for i in range(0, len(signal), segment_length):
        segments.append(signal[i:i+segment_length])
    return segments


def time_to_ms(time_str, recording_start):
    """
    Convert a timestamp into milliseconds relative to recording start.
    """
    h, m = map(int, time_str.split(":"))
    hs, ms = map(int, recording_start.split(":"))

    return ((h * 60 + m) - (hs * 60 + ms)) * 60 * 1000



def get_given_interval(filepath, intervals_file="data/intervals.xlsx"):
    """
    Identify intervention/control recordings and return intervention timing.

    Returns:
        - ("control") for control recordings
        - (start_ms, end_ms) for intervention recordings
    """

    # Load intervention timing information
    df_intervals = pd.read_excel(intervals_file)
    df_intervals.columns = df_intervals.columns.str.strip()


    # Read metadata stored in the first line of the recording file
    try:
        with open(filepath, "r", encoding="latin-1") as f:
            first_line = f.readline().strip()

    except Exception as e:
        print(f"Failed to read file {filepath}: {e}")
        return "control"


    # Files without metadata are considered control recordings
    if not first_line.startswith("#{") or not first_line.endswith("}"):
        return "control"


    try:
        meta_dict = ast.literal_eval(first_line[1:])

    except Exception as e:
        print(f"Failed to parse metadata from {filepath}: {e}")
        return "control"


    # Extract class identifier (e.g., T1 -> 1)
    num_class_str = meta_dict.get("Num_Class")

    if num_class_str is None:
        return "control"


    try:
        num_class = int(num_class_str.replace("T", ""))

    except:
        return "control"



    # Classes corresponding to the control group
    control_classes = [3, 6, 13, 15]

    if num_class in control_classes:
        return "control"



    # Retrieve intervention timing information
    row = df_intervals[df_intervals["Class"] == num_class]

    if row.empty:
        print(f"No interval found for class {num_class}")
        return "control"


    group = row["Group"].values[0]
    inicio = row["Inicio"].values[0]
    fim = row["Fim"].values[0]
    recording_start = row["RecordingStart"].values[0]


    if str(group).lower() == "control":
        return "control"


    # Convert intervention timestamps into milliseconds
    start_ms = time_to_ms(inicio, recording_start)
    end_ms = time_to_ms(fim, recording_start)

    return start_ms, end_ms



def get_control_intervals(signal_length, sampling_rate=100):
    """
    Divide control recordings into three equally sized intervals.

    Returns:
        List of (start_ms, end_ms) intervals
    """

    total_ms = signal_length / sampling_rate * 1000

    # Control recordings are divided into three temporal regions
    segment_ms = total_ms / 3

    intervals = []
    for i in range(3):
        start_ms = i * segment_ms
        end_ms = (i + 1) * segment_ms
        intervals.append((start_ms, end_ms))

    return intervals