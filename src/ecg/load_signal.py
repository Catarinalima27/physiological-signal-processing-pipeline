import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_ecg_data(filepath, sampling_rate, trim_edges=True, show_signal=False):
    """
    Load ECG data from a raw signal file.

    Parameters
    ----------
    filepath : str
        Path to the raw ECG file.
    sampling_rate : int
        Sampling frequency of the signal (Hz).
    trim_edges : bool
        Whether to remove the beginning and end of the recording.
    show_signal : bool
        Whether to display the ECG signal plot.

    Returns
    -------
    np.ndarray or None
        ECG signal array.
    """

    # Check if input file exists
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        # Load raw file and extract ECG channel
        data = pd.read_csv(
            filepath,
            header=2,
            sep=r'\s+',
            engine='python',
            encoding='latin-1'
        )

        ecg_data = data.iloc[:, 2].dropna().astype(float).values

        # Remove initial and final recording segments if required
        if trim_edges:

            if len(ecg_data) < 30 * 60 * sampling_rate:
                trim = sampling_rate * 300   # 5 minutes
            else:
                trim = sampling_rate * 600   # 10 minutes

            ecg_data = ecg_data[trim:-trim]

    except Exception as e:
        raise ValueError(f"Error processing {filepath}: {e}")

    # Check minimum recording duration
    duration = len(ecg_data) / sampling_rate

    if duration < 300:
        print(f"Too short: {duration:.2f}s")
        return None

    # Optional signal visualization
    if show_signal:

        time = np.arange(len(ecg_data)) / sampling_rate

        plt.figure(figsize=(12, 4))
        plt.plot(time, ecg_data, linewidth=0.8)
        plt.title("ECG Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (g)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return np.array(ecg_data)