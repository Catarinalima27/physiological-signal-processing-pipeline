import os
import numpy as np
import pandas as pd

import os

import matplotlib.pyplot as plt


def load_acc_data(filepath, sampling_rate, trim_edges=True, show_signal=False):
    """
    Load accelerometer data from a raw signal file.

    Parameters
    ----------
    filepath : str
        Path to the raw accelerometer file.
    sampling_rate : int
        Sampling frequency of the signal (Hz).
    trim_edges : bool
        Whether to remove the beginning and end of the recording.
    show_signal : bool
        Whether to display the ACC signal plot.

    Returns
    -------
    np.ndarray or None
        Accelerometer signal with three axes (X, Y, Z).
    """

    # Check if input file exists
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        # Load raw file and extract accelerometer channels
        data = pd.read_csv(
            filepath,
            header=2,
            sep=r"\s+",
            engine="python",
            encoding="latin-1"
        )

        # Extract ACC X, Y and Z channels
        accx = data.iloc[:, 3].dropna().astype(float).values
        accy = data.iloc[:, 4].dropna().astype(float).values
        accz = data.iloc[:, 5].dropna().astype(float).values

        # Ensure all axes have the same length
        min_len = min(len(accx), len(accy), len(accz))

        accx = accx[:min_len]
        accy = accy[:min_len]
        accz = accz[:min_len]

        # Remove initial and final recording segments if required
        if trim_edges:

            if min_len < 30 * 60 * sampling_rate:
                trim = sampling_rate * 300      # 5 minutes
            else:
                trim = sampling_rate * 600      # 10 minutes

            # Safety check before trimming
            if 2 * trim < len(accx):
                accx = accx[trim:-trim]
                accy = accy[trim:-trim]
                accz = accz[trim:-trim]
            else:
                print("⚠️ Skipping trimming: signal too short")

        # Combine accelerometer axes into a single array
        acc_data = np.column_stack((accx, accy, accz))

    except Exception as e:
        raise ValueError(f"Error processing {filepath}: {e}")

    # Check minimum recording duration
    duration = len(acc_data) / sampling_rate

    if duration < 90:
        print(f"Too short: {duration:.2f}s")
        return None

    # Optional signal visualization
    if show_signal:

        time = np.arange(len(acc_data)) / sampling_rate

        fig, axs = plt.subplots(
            3,
            1,
            figsize=(12, 7),
            sharex=True
        )

        axs[0].plot(time, acc_data[:, 0], linewidth=0.8)
        axs[0].set_title("ACC X")
        axs[0].set_ylabel("Amplitude (g)")

        axs[1].plot(time, acc_data[:, 1], linewidth=0.8)
        axs[1].set_title("ACC Y")
        axs[1].set_ylabel("Amplitude (g)")

        axs[2].plot(time, acc_data[:, 2], linewidth=0.8)
        axs[2].set_title("ACC Z")
        axs[2].set_ylabel("Amplitude (g)")
        axs[2].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()

    return acc_data