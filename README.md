# Physiological Signal Analysis Pipeline

A Python-based processing pipeline for physiological signal analysis, including:

- Electrocardiography (ECG)
- Electrodermal Activity (EDA)
- Accelerometry (ACC)

The pipeline was developed to process physiological recordings collected during a longitudinal intervention study and provides automated preprocessing, quality assessment, feature extraction, and visualization.

---
## Pipeline Overview

The repository contains two complementary processing pipelines developed to support the **evaluation of physiological changes associated with an intervention**. The pipelines provide both a longitudinal perspective across the study and a more detailed analysis of physiological responses surrounding the intervention session.

### 1. Longitudinal Pipeline

The longitudinal pipeline processes the **complete physiological recordings** collected across different stages of the study:

* **Pre-intervention phases**
* **Intervention day**
* **Post-intervention phase**
* **Follow-up phase**

For each recording, ECG, EDA, and accelerometer signals are preprocessed and the corresponding physiological features are calculated across the complete recording.

The resulting features can subsequently be compared across study stages to **characterize physiological changes over the course of the study and evaluate changes associated with the intervention**.

### 2. Intervention-Day Pipeline

The intervention-day pipeline provides a more detailed analysis of the **short-term physiological response to the intervention session**. The intervention-day recording is divided into three analysis periods:

* **Pre-intervention:** period immediately before the intervention
* **During intervention:** intervention period
* **Post-intervention:** period immediately after the intervention

The same physiological signals and feature-extraction procedures are applied to each period, enabling the analysis of **short-term changes in physiological measures before, during, and after the intervention session**.

The two pipelines therefore provide complementary perspectives: the **longitudinal pipeline** captures physiological changes across different stages of the study, while the **intervention-day pipeline** focuses on the immediate physiological response surrounding the intervention.

---

## Features

## ECG
- Signal loading
- Signal filtering
- Signal visualization
- Signal quality assessment
- R-peak detection
- RR interval extraction
- Heart Rate Variability (HRV) metric extraction

## EDA
- Signal loading
- Signal filtering
- Signal visualization
- Signal quality assessment
- Phasic and tonic decomposition
- Skin Conductance Response (SCR) detection
- EDA feature extraction

## ACC
- Signal loading
- Signal filtering
- Signal visualization
- Activity feature extraction

---

## Repository Structure

```
.
.
├── src/
│   ├── acc/                  # Accelerometry processing modules
│   ├── ecg/                  # ECG processing modules
│   └── eda/                  # Electrodermal activity processing modules
│
├── pipelines/                # Feature extraction pipelines
├── signal_cutting/           # Signal segmentation utilities
├── examples/                 # Example input files and metadata template
│
├── run_feature_extraction_acc_all_phases.py
├── run_feature_extraction_acc_intervention.py
├── run_feature_extraction_ecg_all_phases.py
├── run_feature_extraction_ecg_intervention.py
├── run_feature_extraction_eda_all_phases.py
├── run_feature_extraction_eda_intervention.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Requirements

Python 3.13

Install the required packages using

```bash
pip install -r requirements.txt
```

Main dependencies:

- numpy
- scipy
- pandas
- matplotlib
- biosppy
- openpyxl
- peakutils

---

## Expected Input Format

Each recording should be stored as a text file with the following format:

```text
# {'sampling_rate':100, ...}
#MAC_address:XXXXXXXXXXXX
#T(ms) EDA(uS) ECG(mV) ACCx(g) ACCy(g) ACCz(g)

0.0
10.0
20.0
...
```

The expected columns are

| Column | Description |
|---------|-------------|
| T(ms) | Time (milliseconds) |
| EDA(uS) | Electrodermal activity |
| ECG(mV) | Electrocardiogram |
| ACCx(g) | Accelerometer X axis |
| ACCy(g) | Accelerometer Y axis |
| ACCz(g) | Accelerometer Z axis |

Sampling frequency:

```
100 Hz
```

---

## Data Availability

The original physiological recordings, participant metadata, intervention timing information, and intervention group information are **not included** in this repository due to privacy, ethical, and data protection restrictions.

To facilitate reproducibility and demonstrate the expected input structure, the repository includes example and template files:

- An example physiological recording (`.txt`) illustrating the expected input signal format.
- A template participant metadata file (`.json`) demonstrating the structure expected by the processing pipeline.
- An example intervention interval information file (`.py`) illustrating the structure used to define intervention intervals.
- An example intervention group information file (`.py`) illustrating the mapping structure between class IDs and intervention groups.

The example and template files contain fictional or illustrative information and do not contain real participant data, intervention timings, or class assignments.

Users wishing to apply the pipeline to their own dataset should organize their recordings and supporting information according to the structures illustrated in these files.
---

## Output

Depending on the selected modules, the pipeline can generate:

- HRV metrics
- ECG quality metrics
- EDA metrics
- Accelerometer features
- Original signal plots
- Databases containing extracted physiological metrics

---

## Related Project

The associated research publications will be linked here as they become publicly available.

---

## Citation

*Citation will be updated upon publication of the associated methods paper.*

---

## License

MIT License

---

## Contact

Catarina Soares Lima (catarinasoareslima@gmail.com)

Biomedical Engineer

For questions or suggestions, please open an Issue on GitHub.
