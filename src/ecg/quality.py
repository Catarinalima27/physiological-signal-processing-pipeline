import numpy as np
from biosppy import quality

def signal_quality_check(
    segment,
    sampling_rate=100,
    threshold=0.9,
    bit=10,
    fisher=True,
    f_thr=0.01,
    nseg=1024,
    num_spectrum=[5, 20],
    dem_spectrum=None,
    mode_fsqi="simple"
):
    """
    Evaluate ECG segment quality using BioSPPy SQI metrics.

    Parameters:
        segment (np.array): ECG signal segment.
        sampling_rate (int): Sampling frequency in Hz.

    Returns:
        tuple: Level3 and kSQI quality scores.
    """

    # Level3 quality assessment
    level3_quality = quality.quality_ecg(
        segment=segment,
        methods=["Level3"],
        sampling_rate=sampling_rate,
        fisher=fisher,
        f_thr=f_thr,
        threshold=threshold,
        bit=bit,
        nseg=nseg,
        num_spectrum=num_spectrum,
        dem_spectrum=dem_spectrum,
        mode_fsqi=mode_fsqi
    )


    # kSQI quality assessment
    ksqi_quality = quality.quality_ecg(
        segment=segment,
        methods=["kSQI"],
        sampling_rate=sampling_rate,
        fisher=fisher,
        f_thr=f_thr,
        threshold=threshold,
        bit=bit,
        nseg=nseg,
        num_spectrum=num_spectrum,
        dem_spectrum=dem_spectrum,
        mode_fsqi=mode_fsqi
    )


    return level3_quality, ksqi_quality



def process_ecg_signal(signal, sampling_rate=100, ksqi_threshold=5):
    """
    Divide ECG signal into segments and classify quality.

    Parameters:
        signal (np.array): Filtered ECG signal.
        sampling_rate (int): Sampling frequency in Hz.
        ksqi_threshold (float): Minimum kSQI value for valid signal.

    Returns:
        dict: ECG segments classified by quality level.
    """

    # Split signal into 5-minute segments
    segment_length = 5 * 60 * sampling_rate

    signal = signal[~np.isnan(signal)]

    num_segments = len(signal) // segment_length


    good_quality_segments = []
    bad_quality_segments = []

    sqi_0_segments = []
    sqi_5_segments = []
    sqi_10_segments = []


    # Evaluate quality of each segment
    for i in range(num_segments):

        segment = signal[
            i * segment_length:(i + 1) * segment_length
        ]

        level3, ksqi = signal_quality_check(
            segment,
            sampling_rate
        )


        # Extract quality scores
        try:
            level3 = level3[0]
            ksqi = ksqi[0]

        except (TypeError, IndexError):
            level3 = 0
            ksqi = 0


        # Store Level3 quality categories
        if level3 == 0:
            sqi_0_segments.append(segment)

        elif level3 == 5:
            sqi_5_segments.append(segment)

        elif level3 == 10:
            sqi_10_segments.append(segment)


        # Classify segment quality
        if level3 > 0 or ksqi >= ksqi_threshold:
            good_quality_segments.append(segment)

        else:
            bad_quality_segments.append(segment)


    good_quality_segments = np.concatenate(good_quality_segments).tolist()

    return {
        'sqi_0': sqi_0_segments,
        'sqi_5': sqi_5_segments,
        'sqi_10': sqi_10_segments,
        'good_quality': good_quality_segments,
        'bad_quality': bad_quality_segments
    }