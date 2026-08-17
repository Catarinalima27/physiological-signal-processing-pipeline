import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import biosppy.signals.tools as st
from src.eda.saturation import compute_saturation

import numpy as np
import matplotlib.pyplot as plt
import biosppy.signals.tools as st
from src.eda.saturation import compute_saturation


def clean_eda(signal, sampling_rate=100.0, lowpass_cutoff=1.5, smooth_window=0.5,
              baseline_minutes=10, show_plot=False):
    """
    Clean the raw EDA signal and separate tonic and phasic components.

    Returns:
        tonic_preserved (np.ndarray): Clean tonic signal.
        phasic_ready (np.ndarray): Baseline-corrected phasic signal.
        sat_pct (float): Percentage of saturated samples.
        category (str): Saturation quality category.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import biosppy.signals.tools as st

    # Check for missing input
    if signal is None:
        print("❌ No EDA signal provided.")
        return None, None, 0, "invalid"

    # Handle signals passed as tuples
    if isinstance(signal, tuple):
        signal = signal[0]

    signal = np.asarray(signal)

    # Convert object arrays to numeric arrays when possible
    if signal.dtype == object:
        try:
            signal = np.asarray(signal.tolist()[0])
        except Exception:
            print("❌ Cannot parse object-type signal.")
            return None, None, 0, "invalid"

    # Keep only one signal dimension
    if signal.ndim > 1:
        signal = signal[:, 0]

    signal = signal.astype(float)

    # -------------------------
    # Low-pass filtering
    # -------------------------
    filtered, _, _ = st.filter_signal(
        signal=signal,
        ftype="butter",
        band="lowpass",
        order=4,
        frequency=lowpass_cutoff,
        sampling_rate=sampling_rate
    )

    # -------------------------
    # Signal smoothing
    # -------------------------
    smooth_size = max(3, int(smooth_window * sampling_rate))

    clean_signal, _ = st.smoother(
        signal=filtered,
        kernel="boxzen",
        size=smooth_size,
        mirror=True,
        check_quality=True
    )

    clean_signal = np.asarray(clean_signal).flatten()

    sat_pct, category, sat_mask, _, low_sat_mask, high_sat_mask = \
        compute_saturation(signal, sampling_rate)

    # Replace highly saturated signals with NaNs
    if sat_pct > 75:
        print(f"❌ EDA highly saturated — {sat_pct:.2f}%")
        clean_signal = np.full_like(signal, np.nan)

    tonic_preserved = clean_signal.copy()

    # -------------------------
    # Baseline correction
    # -------------------------
    baseline_samples = min(
        int(baseline_minutes * 60 * sampling_rate),
        len(clean_signal)
    )

    T0 = np.nanmean(clean_signal[:baseline_samples])
    phasic_ready = clean_signal - T0

    # -------------------------
    # Optional visualization
    # -------------------------
    if show_plot:

        t = np.arange(len(signal)) / sampling_rate

        plt.figure(figsize=(12, 4))
        plt.plot(t, signal, color="lightgray", label="Raw EDA")

        plt.scatter(t[high_sat_mask], signal[high_sat_mask],
                    color="red", s=2, label="High saturation")

        plt.scatter(t[low_sat_mask], signal[low_sat_mask],
                    color="blue", s=2, label="Low saturation")

        plt.plot(t, tonic_preserved, color="black", label="Tonic")
        plt.plot(t, phasic_ready, color="green", label="Phasic")

        plt.legend()
        plt.grid(alpha=0.3)
        plt.title(f"Saturation {sat_pct:.2f}% | {category}")
        plt.tight_layout()
        plt.show()

    return tonic_preserved, phasic_ready, sat_pct, category


def scl_sd_10s(tonic_signal, sampling_rate):
    """
    Compute the mean standard deviation of the tonic signal
    using consecutive 10-second windows.
    """

    window = int(10 * sampling_rate)
    n = len(tonic_signal)
    sds = []

    for i in range(0, n - window, window):

        seg = tonic_signal[i:i+window]

        if np.all(np.isnan(seg)):
            continue

        sds.append(np.nanstd(seg))

    if len(sds) == 0:
        return np.nan

    return np.nanmean(sds)