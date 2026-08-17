import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals import eda

from src.eda.cleaning import scl_sd_10s, clean_eda


def extract_eda_features(
    signal=None,
    sampling_rate=100.0,
    baseline_minutes=10,
    show=False,
    method_events="emotiphai",
    method_decomp="smoother"
):
    """
    Extract time-domain EDA features.

    Parameters
    ----------
    signal : array-like
        Raw EDA signal.
    sampling_rate : float
        Sampling frequency in Hz.
    baseline_minutes : int
        Duration used for baseline calculation.
    show : bool
        Display signal decomposition plot.
    method_events : str
        SCR event detection method.
    method_decomp : str
        SCR decomposition method.

    Returns
    -------
    dict
        Extracted EDA features.
    """

    # Handle signals returned together with metadata
    if isinstance(signal, tuple):
        signal = signal[0]

    # Clean signal and separate tonic/phasic components
    tonic, phasic, sat_pct, sat_label = clean_eda(
        signal,
        sampling_rate
    )

    # Return empty features if signal quality is insufficient
    if (
        tonic is None or
        len(tonic) == 0 or
        np.all(np.isnan(tonic))
    ):
        return {
            "scl_mean": np.nan,
            "scl_sd10_mean": np.nan,
            "delta_scl": np.nan,
            "scr_amplitude_mean": np.nan,
            "scr_risetime_mean": np.nan,
            "nSCR": 0,
            "NSSCR_per_min": 0,
            "lykken_scl_normalized": np.nan
        }

    # -------------------------
    # Tonic features (SCL)
    # -------------------------

    scl_mean = np.nanmean(tonic)

    baseline_samples = min(
        int(baseline_minutes * 60 * sampling_rate),
        len(tonic)
    )

    scl_baseline = np.nanmean(
        tonic[:baseline_samples]
    )

    delta_scl = scl_mean - scl_baseline

    scl_sd10 = scl_sd_10s(
        tonic,
        sampling_rate
    )

    # -------------------------
    # Phasic features (SCR)
    # -------------------------

    try:
        events = eda.eda_events(
            signal=phasic,
            sampling_rate=sampling_rate,
            method=method_events
        )

        _, peaks, amplitudes, risetimes, *_ = events

        duration_min = (
            len(phasic) /
            sampling_rate /
            60
        )

        nSCR = len(peaks)

        NSSCR_per_min = (
            nSCR / duration_min
            if duration_min > 0
            else 0
        )

        scr_amplitude = (
            np.nanmean(amplitudes)
            if len(amplitudes) > 0
            else np.nan
        )

        scr_risetime = (
            np.nanmean(risetimes)
            if len(risetimes) > 0
            else np.nan
        )

    except Exception:

        scr_amplitude = np.nan
        scr_risetime = np.nan
        nSCR = 0
        NSSCR_per_min = 0

    # -------------------------
    # Lykken SCL normalization
    # -------------------------

    scl_min = np.nanmin(tonic)
    scl_max = np.nanmax(tonic)

    scl_range = scl_max - scl_min

    if scl_range > 0:
        lykken_scl = (
            scl_mean - scl_min
        ) / scl_range

    else:
        lykken_scl = np.nan

    # Final feature dictionary
    features = {
        "scl_mean": scl_mean,
        "scl_sd10_mean": scl_sd10,
        "delta_scl": delta_scl,
        "scr_amplitude_mean": scr_amplitude,
        "scr_risetime_mean": scr_risetime,
        "nSCR": nSCR,
        "NSSCR_per_min": NSSCR_per_min,
        "lykken_scl_normalized": lykken_scl
    }

    if show:
        plt.figure(figsize=(12, 4))
        plt.plot(signal, label="Raw")
        plt.plot(tonic, label="Tonic (SCL)")
        plt.plot(phasic, label="Phasic (SCR)")
        plt.legend()
        plt.title("EDA decomposition")
        plt.show()

    return features