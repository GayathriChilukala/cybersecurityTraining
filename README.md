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

# IEEE 14 Bus Case Analysis with Graph Signal Processing
## Overview
This project involves the analysis of the IEEE 14 bus case using Graph Signal Processing (GSP) techniques. The main objectives include power flow calculations, graph construction, graph Fourier Transform (GFT), and detection of false data injection attacks.

## Usage
1)Load the IEEE 14 Bus Case:
The script loads the IEEE 14 bus case using MATPOWER.

2)Power Flow Calculations:
Run power flow calculations using MATPOWER.

3)Building Graph:
The bus coordinates and edges are used to build a graph.

4)Graph Signal Processing:

-> Compute the Graph Laplacian

-> Perform Eigen-decomposition

-> Compute GFT of the original and corrupted signals

-> Plotting the results

5)False Data Injection Attack:
Introduce false data at a specific bus and analyze the effect.

6)Detection of Attack:

-> Apply a high-pass filter

-> Calculate the amount of high-frequency component

-> Determine if an attack occurred based on a threshold

## Results
The script generates plots showing the original and corrupted graph signals, their GFTs, and the detection of the false data injection attack.

# Graph Convolutional Network for IEEE 14 Bus Case
## Overview
This project demonstrates the application of Graph Convolutional Networks (GCNs) on the IEEE 14 bus case using the Deep Graph Library (DGL) and PyTorch. The main objectives are to build a GCN model, train it on a dataset of node features and labels, and evaluate its performance on a test set.

## Usage
Step 1: Load necessary packages
Step 2: Create graph from list of edges
Step 3: Visualize the graph
Step 4: Load dataset
Step 5: Add node features and labels to the graph
Step 6: Define a Graph Convolutional Network (GCN)
Step 7: Train-Test split
Step 8: Training loop
Step 9: Visualize iterations
Step 10: Test model performance

## Results
The results indicate that the GCN model effectively learned the node classifications in the IEEE 14 bus system, with a high training accuracy and reasonable testing performance. The visualizations helped in understanding the model's predictions and performance across different epochs.

# IoT Cloud Security: AWS S3 Example

This repository contains an exercise designed to evaluate the security of IoT devices that utilize AWS S3 for cloud storage. The exercise demonstrates how hard-coded AWS credentials can expose sensitive data and outlines steps for penetration testing using AWS CLI.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Hands-on Experiment](#hands-on-experiment)
  - [Step 1: Install AWS CLI](#step-1-install-aws-cli)
  - [Step 2: Configure AWS CLI with Provided Credentials](#step-2-configure-aws-cli-with-provided-credentials)
  - [Step 3: List S3 Buckets and Files](#step-3-list-s3-buckets-and-files)
  - [Step 4: Download Files from S3 Bucket](#step-4-download-files-from-s3-bucket)
- [Security Best Practices](#security-best-practices)
- [Contributing](#contributing)
- [License](#license)

## Overview
This exercise simulates a scenario where an IP camera stores screenshots and videos on AWS S3, using hard-coded access keys. As a security penetration tester, your task is to assess the security of this setup by using the provided AWS credentials to access the stored data.

## Prerequisites
- AWS CLI installed on your machine.
- Basic knowledge of command-line operations.
- A testing environment (e.g., Kali Linux) is recommended for penetration testing exercises.

## Setup

### Step 1: Install AWS CLI
Follow the official AWS CLI installation guide: [AWS CLI Installation Guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

For Kali Linux, use the command:
```bash
sudo apt install awscli
```

### Step 2: Configure AWS CLI with Provided Credentials
Use the following command to configure AWS CLI:
```bash
aws configure
```
Enter the following details when prompted:
- **IAM Access Key ID:** 
- **IAM Secret Access Key:** 
- Leave the default region name and output format as they are.

## Hands-on Experiment

### Step 3: List S3 Buckets and Files
List all S3 buckets associated with the account:
```bash
aws s3 ls
```
To list files in a specific bucket:
```bash
aws s3 ls s3://<bucket-name>/
```

### Step 4: Download Files from S3 Bucket
To download a specific file (e.g., a screenshot from an IP camera):
```bash
aws s3 cp s3://ip-camera-screenshot/alice@test.com/1717008630973.jpg 1717008630973.jpg
```

## Security Best Practices
This exercise highlights the risks of using hard-coded AWS credentials. To enhance security, consider the following best practices:
- Use IAM roles and policies to grant necessary permissions.
- Avoid embedding credentials in code. Use environment variables or AWS Secrets Manager.
- Regularly rotate access keys.
- Implement strong access controls and encryption for sensitive data.


