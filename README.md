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


# Dog and Cat Classification using CNN

This project focuses on classifying images of dogs and cats using a Convolutional Neural Network (CNN) implemented with Keras. The model architecture includes several convolutional layers followed by max-pooling layers, and finally dense layers to perform the classification.

## Model Architecture

The CNN model is designed as follows:

1. **Convolutional Layer 1:**
   - 32 filters, each of size 3x3.
   - ReLU activation function.
   - Input shape is defined by the image height, width, and 3 color channels (RGB).

2. **Max-Pooling Layer 1:**
   - Pool size of 3x3 to reduce the spatial dimensions of the output from the previous layer.

3. **Convolutional Layer 2:**
   - 32 filters, each of size 3x3.
   - ReLU activation function.

4. **Max-Pooling Layer 2:**
   - Pool size of 2x2 to further reduce the spatial dimensions of the output.

5. **Flatten Layer:**
   - Flattens the multi-dimensional output from the previous layer into a single dimension, preparing it for the fully connected layers.

6. **Dense Layer 1:**
   - 100 units.
   - ReLU activation function.
   - Followed by a Dropout layer with a rate of 0.5 to prevent overfitting by randomly setting half of the input units to 0 at each update during training.

7. **Output Layer:**
   - 1 unit.
   - Sigmoid activation function to produce a probability value for binary classification (dog or cat).

8. **Compilation:**
   - Loss function: Binary Cross-Entropy, suitable for binary classification tasks.
   - Optimizer: RMSprop, which is effective for this type of neural network.
   - Metrics: Accuracy, to monitor the performance of the model during training and testing.

By using this architecture, the model learns to identify and classify images of dogs and cats effectively.

## Data

- **Dataset:** Images of dogs and cats.
- **Image Dimensions:** Customizable through `IMAGE_HEIGHT` and `IMAGE_WIDTH`.
- **Format:** RGB images.

# Time-Series Forecasting Using CNN for Power Consumption
This tutorial was modified from the time-series-forecasting-CNN repository.

## Business Problem
Given some number of prior days of total daily power consumption, the goal is to predict the next standard week of daily power consumption.
## Data

**Dataset:**  'Household Power Consumption' from the UCI Machine Learning Repository
**Units:**  Kilowatts
**Frequency:** Daily
**Time Range:** 2006 to 2010
## Strategy
**Time-Sequence Forecasting: Autoregression**
	•	Predict a forecast for y number of days into the future based on x number of days up to the current day (e.g., predict the next week from this week).
**Convolutional Neural Network**
	•	Use a low-bias model that can learn non-linear relationships.
	•	Implemented in Keras.
## Model Evaluation
	•	Evaluate each forecast day individually.
	•	Use RMSE (Root Mean Squared Error) as the metric, in kilowatts.
## Prepare Data
**Download Data**
	•	Download the 'Household Power Consumption' dataset from the UCI Machine Learning Repository.
 
**Clean Data**
	•	Perform necessary data cleaning steps to handle missing values and outliers.
 
**Downsample Data**
	•	Downsample the data to a daily frequency if needed.
 
# Microgrid Energy Optimization Using Q-Learning
This repository contains the MATLAB code for optimizing a simplified microgrid energy setup using a single Q-learning agent. The code is designed for academic training and was initially supported by the NSF CyberTraining project.

## Code Explanation
The main components of the code are as follows:

1. **Initialization and Setup**
Clears the workspace, initializes battery parameters, and defines the state space.

2. **Q-Table Definition**
Initializes the Q-table for storing Q-values.

3. **Parameters**
Defines the initial parameters for the Q-learning algorithm.

4. **Main Loop**
Runs multiple simulations to train the Q-learning agent.

5. **Iteration Loop**
Iterates over training episodes and updates the Q-table based on the agent's actions and received rewards.

6. **Action Selection**
Chooses actions using an epsilon-greedy strategy to balance exploration and exploitation.

7. **Q-Table Update**
Updates the Q-table based on the obtained rewards and future state estimates.

8. **Results and Plotting**
Calculates and plots the results to visualize the learning process and the battery state of charge (SOC) over time.

# WEC-Sim Simulation Project
This repository contains MATLAB scripts for simulating a wave energy converter (WEC) using WEC-Sim (Wave Energy Converter SIMulator).

## Project Overview
This project simulates the performance of a WEC using different wave conditions. The simulation helps evaluate the behavior and efficiency of the WEC.

## Script Breakdown
### Simulation Data
Initializes the simulation class and sets simulation parameters such as start time, end time, solver type, and time step.

### Wave Information
Configures the wave conditions, such as wave type, height, and period.
### Body Data
Defines the physical properties and hydrodynamic data for the WEC components.
### PTO and Constraint Parameters
Sets up the Power Take-Off (PTO) components and constraints, which control the energy conversion process.

# Hardware Trojan Detection
## Overview
This project focuses on detecting hardware Trojans using machine learning techniques. Hardware Trojans are malicious alterations to a circuit that can cause incorrect behavior or leak sensitive information. Detecting such Trojans is crucial for ensuring the security and reliability of hardware systems.

## Data
The data consists of two main parts:

### DFG3 Dataset:

DFG3/metadata.tsv: Metadata file for DFG3 dataset.
DFG3/vectors.tsv: Vector file for DFG3 dataset.
AST3 Dataset:

AST3/metadata.tsv: Metadata file for AST3 dataset.
AST3/vectors.tsv: Vector file for AST3 dataset.

## Results
### AST Dataset
We performed cross-validation using StratifiedKFold with 4 splits. Here are the accuracy scores for each fold and the overall accuracy:

Fold 0: 0.6666666666666666

Fold 1: 0.8333333333333334

Fold 2: 0.5

Fold 3: 0.8

Overall Accuracy: 0.7

### DFG Dataset
Similarly, we performed cross-validation using StratifiedKFold with 4 splits for the DFG dataset. Here are the accuracy scores for each fold and the overall accuracy:

Fold 0: 0.8333333333333334

Fold 1: 0.8333333333333334

Fold 2: 0.8333333333333334

Fold 3: 1.0

Overall Accuracy: 0.875

# IEEE 39-Bus System Incidence Matrix
This repository contains a MATLAB script for generating the incidence matrix for the IEEE 39-Bus system, also known as the New England power system. The incidence matrix is used to represent the connectivity of the buses and lines within the system.

## Introduction
The IEEE 39-Bus system is a standard test case used in power system studies. It consists of 39 buses, 46 transmission lines, and 10 generators. The system is widely used for testing power system algorithms, especially those related to stability and reliability analysis.

## Installation
To use this script, you need to have MATLAB installed on your system. There are no additional dependencies required.

## Details
The script creates a 39x39 incidence matrix A for the IEEE 39-Bus system. The matrix is initialized to zeros and then populated with values representing the connections between the buses. The values for lines and transformers are assigned based on the connectivity in the system. Finally, the matrix is symmetrized, and an identity matrix is added to account for self-connections.

## Results
After running the script, the incidence matrix A will be generated and stored in the MATLAB workspace. This matrix can be used for further analysis and simulation of the IEEE 39-Bus system.
