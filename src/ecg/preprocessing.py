import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import interp1d, CubicSpline
from scipy.linalg import toeplitz, inv
from biosppy.signals import ecg


# -------------------------
# ECG FILTERING
# -------------------------
def filter_signal(signal, sampling_rate=100):
    """Filter the raw ECG signal using BioSPPy's default ECG preprocessing."""
    out = ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=False)
    filtered = out['filtered']
    return filtered


# -------------------------
# THRESHOLD-BASED RR CORRECTION
# -------------------------
def threshold_based_correction(rri, threshold=0.25):
    """
    Correct RR interval artifacts using median filtering and cubic interpolation.
    """

    rri = np.array(rri, dtype=float)
    threshold *= 1000

    rri_filt = medfilt(rri, 5)
    artifacts = np.abs(rri - rri_filt) > threshold

    if np.any(artifacts):
        valid = np.where(~artifacts)[0]
        invalid = np.where(artifacts)[0]

        if len(valid) > 3:
            f = interp1d(valid, rri[valid], kind='cubic', fill_value="extrapolate")
            rri[invalid] = f(invalid)

    return rri


# -------------------------
# SMOOTHNESS PRIORS DETRENDING
# -------------------------
def smoothness_priors_detrend(rri, lambda_smooth=500):
    """
    Remove slow trends from the RR interval series using smoothness priors.
    """

    rri = np.array(rri, dtype=float)
    N = len(rri)

    D = toeplitz([1, -2, 1] + [0] * (N - 3))
    I = np.eye(N)

    H = I - inv(I + lambda_smooth * (D.T @ D))
    return H @ rri


# -------------------------
# AUTOMATIC RR CORRECTION
# -------------------------
def automatic_correction(
    rri,
    threshold=300,
    percent_threshold=0.25,
    range_min=480,
    range_max=1091,
    rolling_window=5):
    """
    Corrects artifacts in an RRI sequence using a hybrid approach, suitable for children.

    Parameters
    ----------
    rri : array
        RR intervals (ms).
    threshold : int
        Absolute deviation threshold (ms). Default: 150.
    percent_threshold : float
        Percentage deviation threshold (fraction). Default: 0.25.
    range_min : int
        Minimum allowable RR interval (ms). Default: 480.
    range_max : int
        Maximum allowable RR interval (ms). Default: 1091.
    rolling_window : int
        Rolling window size for dynamic thresholding. Default: 90.

    Returns
    -------
    rri_corrected : array
        Corrected RR intervals.
    """
    
    if rri is None or len(rri) <= 4:
        raise ValueError("RRI array must contain more than 4 values.")

    rri = np.array(rri, dtype=float)
    dRR = np.diff(rri)

    # Compute adaptive thresholds
    Th_list = []
    for i in range(len(dRR)):
        start = max(0, i - rolling_window // 2)
        end = min(len(dRR), i + rolling_window // 2)
        window = np.abs(dRR[start:end])

        if len(window) >= 5:
            th_value = 5.2 * np.nanpercentile(window, 25)
        else:
            th_value = 0

        Th_list.append(th_value)

    Th_list = np.array(Th_list)

    # Compute local RR median
    window_size = 10

    local_medians = np.array([
        np.median(rri[max(0, i - window_size): min(len(rri), i + window_size)])
        for i in range(len(rri))
    ])

    # Detect artifacts
    artifacts = (
        (np.abs(rri - local_medians) > threshold) |
        (np.abs(rri - local_medians) / (local_medians + 1e-8) > percent_threshold) |
        (rri < range_min) |
        (rri > range_max)
    )

    # Additional detection based on dRR patterns
    for i in range(2, len(dRR) - 2):

        Th = Th_list[i] * 1.2 if i < len(Th_list) else Th_list[-1] * 1.2
        avg_dRR = np.mean(dRR[i - 1:i + 2])
        median_rr = np.median(rri[max(0, i - 10): i + 10])

        if ((dRR[i - 1] < -Th and dRR[i] > Th) or
            (dRR[i - 1] > Th and dRR[i] < -Th)):

            if np.abs(avg_dRR) > 0.8 * Th and np.abs(avg_dRR) > 0.15 * median_rr:
                artifacts[i] = True

    # Interpolate detected artifacts
    valid_indices = np.where(~artifacts)[0]
    invalid_indices = np.where(artifacts)[0]

    if len(valid_indices) > 3:

        spline_func = CubicSpline(valid_indices, rri[valid_indices], bc_type='natural')
        rri[invalid_indices] = spline_func(invalid_indices)

    else:

        print("⚠️ Not enough valid points for cubic spline interpolation. Using linear interpolation.")

        interp_func = interp1d(valid_indices, rri[valid_indices],
                               kind='linear',
                               fill_value="extrapolate")

        rri[invalid_indices] = interp_func(invalid_indices)

    # Limit corrected RR intervals to physiological range
    rri = np.clip(rri, range_min, range_max)

    #print(f"✅ Corrected {np.sum(artifacts)} artifacts in RRI sequence of length {len(rri)}.")

    return rri


# -------------------------
# RR CORRECTION WRAPPER
# -------------------------
def correct_rri(rri, method="automatic", threshold=0.25):
    """
    Apply the selected RR interval correction method.
    """

    if method == "threshold":
        return threshold_based_correction(rri, threshold)

    elif method == "automatic":
        return automatic_correction(rri)

    else:
        raise ValueError("Invalid method. Choose 'threshold' or 'automatic'.")