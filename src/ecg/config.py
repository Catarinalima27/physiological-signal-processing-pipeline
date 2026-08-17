# ECG signal processing parameters

# Sampling frequency of the physiological signals (Hz)
sampling_rate = 100

# Minimum quality threshold used for signal quality assessment (kSQI)
ksqi_threshold = 5

# Physiological limits for valid RR intervals (milliseconds)
# Used for artifact detection and correction
RR_MIN = 450
RR_MAX = 950

# Length of ECG segments used for feature extraction (minutes)
segment_minutes = 5