###2021 인공지능 온라인 경진대회 / 기계시설물
 
# Outlier Detection in Induction Motors Using Multivariate Kernel Density Estimation (MKDE)

This repository contains the implementation of the research presented in the paper:

**"Outlier Detection Based on Multivariate Kernel Density Estimation in Induction Motors"**  
Authors: Youngjun Sim, Jungyu Choi, Bobae Kim, Sungbin Im  
Affiliation: Soongsil University

## Abstract

The increasing adoption of the Industrial Internet of Things (IIoT) has led to a surge in data generated by industrial devices, making predictive maintenance a critical application for smart factories. This repository provides the implementation of a novel outlier detection method for induction motors using **Multivariate Kernel Density Estimation (MKDE)**. The method leverages unlabeled current signal data from three-phase induction motors to detect anomalies without requiring predefined probability distribution functions or explicit correlations.

Key results:
- Achieved 98.93% accuracy on test data using MKDE.
- Demonstrated robustness in detecting anomalies even with data imbalance.

## Features

- **Multivariate Kernel Density Estimation (MKDE):** A non-parametric statistical method to estimate joint probability density functions.
- **FFT-based Feature Extraction:** Extraction of frequency-domain features such as fundamental frequency and Total Harmonic Distortion (THD) from current signals.
- **Threshold Optimization:** Silverman's rule of thumb used for bandwidth adjustment and optimal threshold calculation.

## Methodology

1. **Dataset:**
   - Current signal data from three-phase induction motors provided by [AI Hub](https://aihub.or.kr).
   - 11,948 total samples, with 10,000 normal samples and 1,948 anomalous samples.
   - Train/test split: 5,974 samples each.

2. **Feature Extraction:**
   - Performed FFT on current signals to extract:
     - Three fundamental frequencies (120 Hz from R, T, S phases).
     - Total Harmonic Distortion (THD).
   - Resulting in six features per sample.

3. **MKDE Model:**
   - Estimated joint probability density function (JPDF) using Gaussian kernel functions.
   - Bandwidth selection via Silverman’s rule of thumb.
   - Anomaly detection based on probability thresholds.

4. **Evaluation:**
   - Accuracy: **98.93%** on test data.
