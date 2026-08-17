import numpy as np


def trim_signal(signal, sampling_rate=100):
    """
    Remove initial and final portions of a signal.

    Trimming rule:
        - Recordings shorter than 30 minutes: remove 5 minutes from each side
        - Recordings longer than 30 minutes: remove 10 minutes from each side

    Returns:
        Trimmed signal or None if trimming is not possible.
    """

    if signal is None:
        return None

    signal = np.asarray(signal, dtype=float)

    if len(signal) == 0:
        return None


    # Define trimming duration based on recording length
    if len(signal) < 30 * 60 * sampling_rate:
        trim_seconds = 300   # 5 minutes
    else:
        trim_seconds = 600   # 10 minutes


    trim_samples = int(trim_seconds * sampling_rate)


    # Avoid removing the entire recording
    if len(signal) <= 2 * trim_samples:
        print("⚠️ Signal too short for trimming.")
        return None


    return signal[trim_samples:-trim_samples]