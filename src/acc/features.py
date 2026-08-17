import numpy as np

from biosppy.signals import acc
from src.acc.preprocessing import preprocess_acc


def extract_acc_features(acc_signal, sampling_rate=100):
    """
    Extract time-domain features from an accelerometer signal.

    The pipeline includes signal preprocessing, activity-related feature
    extraction, and final aggregation into summary metrics.

    Parameters
    ----------
    acc_signal : np.array
        Raw accelerometer signal with three axes (X, Y, Z).
    sampling_rate : int
        Sampling frequency in Hz.

    Returns
    -------
    dict
        Dictionary containing aggregated accelerometer features.
    """

    # Preprocess accelerometer signal (filtering and normalization)
    clean_signal = preprocess_acc(acc_signal, sampling_rate)


    # Extract standard accelerometer time-domain features using BioSPPy
    # VM: vector magnitude; SMA: signal magnitude area
    vm, sma = acc.time_domain_feature_extractor(
        signal=clean_signal
    )


    # Compute global RMS as a measure of signal intensity
    rms = np.sqrt(np.mean(clean_signal ** 2))


    # Calculate jerk (rate of change of acceleration), related to movement intensity
    jerk = np.linalg.norm(
        np.gradient(clean_signal, axis=0) * sampling_rate,
        axis=1
    )


    # Compute activity index and interpolate to match signal length
    ts_ai, ai = acc.activity_index(
        signal=clean_signal,
        sampling_rate=sampling_rate
    )

    ai = np.interp(
        np.arange(len(clean_signal)) / sampling_rate,
        ts_ai,
        ai
    )


    # Calculate peak-to-peak variation for each accelerometer axis
    ptp_x = np.ptp(clean_signal[:, 0])
    ptp_y = np.ptp(clean_signal[:, 1])
    ptp_z = np.ptp(clean_signal[:, 2])


    # Create binary activity indicator based on jerk threshold
    activity_flag = (jerk > 0.05).astype(int)


    # Store extracted features before aggregation
    features = {
        "VM": vm,
        "RMS": rms,
        "SMA": sma,
        "Jerk": jerk
    }


    # Aggregate time-series features into representative summary values
    final_features = {}

    for name, value in features.items():

        if isinstance(value, np.ndarray):
            final_features[f"{name}_mean"] = np.nanmean(value)
        else:
            final_features[f"{name}_mean"] = float(value)


    return final_features