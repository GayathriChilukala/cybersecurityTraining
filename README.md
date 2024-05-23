# Deep Learning Models for Cybersecurity Applications

This repository contains implementations and evaluations of Inception and ResNet models for cybersecurity tasks, specifically designed for detecting anomalies in sensor data.

## Models Implemented

### Inception Model

The Inception model is designed using a series of inception modules with residual connections for improved performance in anomaly detection tasks.

#### Performance

- Normal Accuracy: 98.89%
- Load Change Accuracy: 94.04%
- Attack Accuracy: 97.68%

#### Training Details

- **Batch Size:** 4
- **Epochs:** 50
- **Learning Rate:** 0.0001

### ResNet Model

The ResNet model utilizes residual blocks to enable deeper networks without degradation in performance.


#### Performance

- Normal Accuracy: 94.85%
- Load Change Accuracy: 87.58%
- Attack Accuracy: 91.01%

#### Training Details

- **Batch Size:** 8
- **Epochs:** 25
- **Learning Rate:** 0.00001

# Python UDP Flood

This is a DoS attack program to flood servers using UDP or TCP packets. You can set the IP, port, and the number of packets to send per connection. This tool is created for educational purposes and to demonstrate network security concepts.

## Features

- Supports both UDP and TCP flooding.
- Customizable target IP, port, packet count, and thread count.
- Multi-threaded to maximize the flooding effect.

# Time-Series Forecasting with CNN

This project focuses on forecasting daily power consumption using a Convolutional Neural Network (CNN). The model is built using the 'Household Power Consumption' dataset from the UCI Machine Learning Repository, which contains daily power consumption data in kilowatts from 2006 to 2010.

## Business Problem

**Objective:** Predict the next 7 days of daily power consumption based on a specified number of prior days.

## Data

- **Dataset:** Household Power Consumption
- **Units:** Kilowatts
- **Frequency:** Daily
- **Time Range:** 2006 to 2010

## Strategy

1. **Time-sequence Forecasting: Autoregression**
   - Predict future power consumption using past consumption data.
   - Example: Predict the next week’s consumption based on the previous week.

2. **Model: Convolutional Neural Network (CNN)**
   - Low-bias model suitable for capturing non-linear relationships.
   - Implemented using Keras.

## Model Evaluation

- **Metric:** Root Mean Square Error (RMSE)
  - RMSE measures the average magnitude of errors in predictions.
  - It is expressed in the same units as the data (kilowatts), making it straightforward to interpret.
  - Lower RMSE values indicate better model performance.




