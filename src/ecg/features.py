from biosppy.signals import ecg, hrv
import numpy as np
import io
import warnings

from contextlib import redirect_stdout
from src.ecg.preprocessing import correct_rri, filter_signal
from src.ecg.quality import process_ecg_signal


def extract_kubios_features(signal, sampling_rate=100, check_quality=True):
    """
    Extract HRV features from an ECG signal.

    Pipeline:
    1. Filter raw ECG signal.
    2. Optionally assess signal quality using SQI.
    3. Detect and correct R-peaks.
    4. Compute and correct RR intervals.
    5. Extract time-domain, frequency-domain, and non-linear HRV features.

    Args:
        signal (np.array): ECG signal.
        sampling_rate (int): Sampling frequency in Hz.
        check_quality (bool): Apply ECG quality filtering before extraction.

    Returns:
        dict: Extracted HRV features.
    """

    # Preprocess ECG signal
    filtered = filter_signal(signal, sampling_rate=sampling_rate)

    # Optional signal quality assessment
    if check_quality:
        quality_result = process_ecg_signal(
            filtered,
            sampling_rate=sampling_rate
        )
        good_signal = np.asarray(quality_result["good_quality"])
    else:
        good_signal = filtered

    if len(good_signal) < 10:
        raise ValueError("Signal empty after SQI filtering")

    # Detect R-peaks using Hamilton ECG segmentation algorithm
    r_peaks = ecg.hamilton_segmenter(
        good_signal,
        sampling_rate=sampling_rate
    )
    r_peaks_array = r_peaks[0]

    # Correct detected R-peak positions
    r_peaks_corr = ecg.correct_rpeaks(
        signal=good_signal,
        rpeaks=r_peaks_array,
        sampling_rate=sampling_rate,
        tol=0.01
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Convert R-peaks into RR intervals (milliseconds)
        rr_intervals = hrv.compute_rri(
            r_peaks_corr,
            sampling_rate=sampling_rate,
            show=False
        )

    # Correct RR interval artifacts
    corrected_rri_threshold = correct_rri(
        rr_intervals,
        method="threshold",
        threshold=0.15
    )

    rr_int_fil_corr = correct_rri(
        rr_intervals,
        method="automatic"
    )

    # Keep physiologically valid RR intervals
    rr_int_fil_corr = rr_int_fil_corr[
        (rr_int_fil_corr >= 450) &
        (rr_int_fil_corr <= 950)
    ]

    if rr_int_fil_corr is None or len(rr_int_fil_corr) < 10:
        raise ValueError("RR intervals too short or invalid for feature extraction.")


    with redirect_stdout(io.StringIO()):

        # Extract HRV features from different domains
        time_hrv = hrv.hrv_timedomain(
            rri=rr_int_fil_corr,
            duration=None,
            show=False
        )

        frequency_hrv = hrv.hrv_frequencydomain(
            rri=rr_int_fil_corr,
            duration=None,
            show=False
        )

        non_linear_hrv = hrv.hrv_nonlinear(
            rri=rr_int_fil_corr,
            duration=None,
            show=False
        )

    # Selected HRV metrics used in analysis
    list_hrv_features = [
        'hr_mean',
        'rmssd',
        'pnn50',
        'sdnn',
        'lf_pwr',
        'hf_pwr',
        'lf_hf',
        'sd1',
        'sd2'
    ]

    # Combine HRV outputs from all domains
    combined = {}
    combined.update(time_hrv.as_dict())
    combined.update(frequency_hrv.as_dict())
    combined.update(non_linear_hrv.as_dict())

    # Return only required metrics
    extracted_dict = {
        key: combined.get(key, np.nan)
        for key in list_hrv_features
    }

    return extracted_dict