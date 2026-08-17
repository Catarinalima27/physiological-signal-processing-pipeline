import numpy as np
import numpy as np
from scipy.signal import medfilt
from scipy.integrate import trapezoid
from biosppy.signals.tools import filter_signal

# -------------------------
# Saturation thresholds
# -------------------------
LOW_THRESH = 1.5          # µS (negative saturation)
HIGH_THRESH = 24.7        # µS (positive saturation)
RANGE_THRESH = 0.07       # µS (robust global flatline)
ANGLE_MAX_DEG = 5         # degrees
MIN_SEGMENT_SEC = 60      # minimum plateau duration (seconds)
MIN_RUN_MINUTES = 5       # minimum consecutive saturated duration (minutes)

import scipy.ndimage as ndi


# -------------------------
# Startup flat detection
# -------------------------
def detect_startup_flat(
    eda,
    fs,
    flat_window_sec=10,
    flat_std_thresh=0.01,
    near_zero_thresh=0.1,
    min_duration_sec=30
):
    window = int(flat_window_sec * fs)
    min_len = int(min_duration_sec * fs)

    rolling_std = np.array([
        np.std(eda[max(0, i-window):i+1])
        for i in range(len(eda))
    ])

    flat_mask = (
        (rolling_std < flat_std_thresh) &
        (eda < near_zero_thresh)
    )

    labels, n = ndi.label(flat_mask)
    startup_flat = np.zeros_like(flat_mask, dtype=bool)

    for i in range(1, n + 1):
        idx = np.where(labels == i)[0]
        if len(idx) >= min_len:
            startup_flat[idx] = True

    cutoff = np.where(startup_flat)[0][-1] + 1 if startup_flat.any() else 0
    return cutoff, startup_flat


# -------------------------
# Flatline saturation detection
# -------------------------
def detect_flatline_saturation(
    eda,
    fs,
    window_sec=5,
    step_sec=1,
    unique_thresh=2
):
    n = len(eda)
    window = int(window_sec * fs)
    step = int(step_sec * fs)

    mask = np.zeros(n, dtype=bool)

    for start in range(0, n - window, step):
        seg = np.round(eda[start:start + window], 6)

        if len(np.unique(seg)) <= unique_thresh:
            mask[start:start + window] = True

    return mask


# -------------------------
# Extract contiguous segments
# -------------------------
def extract_segments(mask):
    segments = []
    start = None

    for i, v in enumerate(mask):

        if v and start is None:
            start = i

        elif not v and start is not None:
            segments.append((start, i))
            start = None

    if start is not None:
        segments.append((start, len(mask)))

    return segments


# -------------------------
# Validate negative saturation
# -------------------------
def validate_negative_saturation_segment(
    segment,
    fs,
    window_min=5,
    step_min=1,
    band_us=0.02,
    min_valid_windows=3
):
    window = int(window_min * 60 * fs)
    step = int(step_min * 60 * fs)

    valid = 0

    for start in range(0, len(segment) - window + 1, step):

        w = segment[start:start + window]

        low = np.nanpercentile(w, 5)
        high = np.nanpercentile(w, 95)
        core = w[(w >= low) & (w <= high)]

        if len(core) == 0:
            continue

        if (np.nanmax(core) - np.nanmin(core)) <= band_us:
            valid += 1

            if valid >= min_valid_windows:
                return True

    return False


# -------------------------
# Compute saturation metrics
# -------------------------
def compute_saturation(signal, sampling_rate):

    signal = np.asarray(signal, dtype=float)
    n = len(signal)

    # Detect startup flat regions
    cutoff, startup_flat = detect_startup_flat(signal, sampling_rate)

    eda = signal

    # -------------------------
    # Global saturation check
    # -------------------------
    low_p = np.nanpercentile(eda, 5)
    high_p = np.nanpercentile(eda, 95)

    core = eda[(eda >= low_p) & (eda <= high_p)]

    if len(core) > 0:
        robust_range = np.nanmax(core) - np.nanmin(core)
    else:
        robust_range = 0.0

    if robust_range < RANGE_THRESH:

        sat_mask = np.ones(n, dtype=bool)

        return (
            100.0,
            ">75%",
            sat_mask,
            np.full_like(signal, np.nan),
            sat_mask.copy(),
            np.zeros(n, dtype=bool)
        )

    # -------------------------
    # Positive saturation
    # -------------------------
    high_sat_mask = signal >= HIGH_THRESH

    # -------------------------
    # Negative saturation
    # -------------------------
    flat_mask = detect_flatline_saturation(eda, sampling_rate)
    segments = extract_segments(flat_mask)

    low_sat_post = np.zeros_like(eda, dtype=bool)

    for s, e in segments:

        seg = eda[s:e]

        if np.all(seg <= LOW_THRESH):

            if validate_negative_saturation_segment(seg, sampling_rate):
                low_sat_post[s:e] = True

    # -------------------------
    # Combine saturation masks
    # -------------------------
    low_sat_mask = np.zeros(n, dtype=bool)
    low_sat_mask = low_sat_post

    low_sat_mask[startup_flat] = True

    sat_mask = high_sat_mask | low_sat_mask
    percent = 100.0 * np.sum(sat_mask) / n

    # -------------------------
    # Assign saturation category
    # -------------------------
    if percent < 25:
        category = "<25%"
    elif percent <= 75:
        category = "25–75%"
    else:
        category = ">75%"

    clean_signal = signal.copy()
    clean_signal[sat_mask] = np.nan

    return (
        percent,
        category,
        sat_mask,
        clean_signal,
        low_sat_mask,
        high_sat_mask
    )