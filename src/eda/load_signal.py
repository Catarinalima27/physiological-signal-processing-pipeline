import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_eda_data(filepath, sampling_rate, trim_edges=True, show_signal=False):
    """
    Load EDA signal from a raw text file.

    Parameters
    ----------
    filepath : str
        Path to the signal file.
    sampling_rate : int
        Sampling frequency in Hz.
    trim_edges : bool
        Whether to remove initial and final resting periods.
    show_signal : bool
        Whether to plot the loaded signal.

    Returns
    -------
    np.ndarray or None
        Preprocessed EDA signal.
    """

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        # Load raw data and extract EDA channel
        data = pd.read_csv(
            filepath,
            header=2,
            sep=r"\s+",
            engine="python",
            encoding="latin-1"
        )

        eda_data = data.iloc[:, 1].dropna().astype(float).values

        # Remove initial and final segments to reduce recording artifacts
        if trim_edges:

            if len(eda_data) < 30 * 60 * sampling_rate:
                trim = sampling_rate * 300   # 5 minutes
            else:
                trim = sampling_rate * 600   # 10 minutes

            eda_data = eda_data[trim:-trim]

    except Exception as e:
        raise ValueError(f"Error processing {filepath}: {e}")

    # Validate minimum signal duration
    duration = len(eda_data) / sampling_rate

    if duration < 90:
        print(f"⚠️ Signal too short: {duration:.2f}s")
        return None

    # Optional visualization
    if show_signal:

        time = np.arange(len(eda_data)) / sampling_rate

        plt.figure(figsize=(12, 4))
        plt.plot(time, eda_data, linewidth=0.8)
        plt.title("EDA Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (µS)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return np.asarray(eda_data)