import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(data, lowcut=0.25, highcut=20.0, fs=100.0, order=4):
    """Apply a Butterworth bandpass filter to a 3-axis ACC signal."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)


def preprocess_acc(signal, sampling_rate=100.0):
    """Clean and prepare raw accelerometer data."""
    # Ensure numpy array
    signal = np.array(signal, dtype=float)

    # Remove mean per axis (remove gravity/static offset)
    signal = signal - np.mean(signal, axis=0)

    # Bandpass filter to remove drift and noise
    clean_signal = bandpass_filter(signal, lowcut=0.25, highcut=20.0, fs=sampling_rate)
    return clean_signal
