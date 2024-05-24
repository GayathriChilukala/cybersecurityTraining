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

# Microgrid Energy Optimization Using Q-Learning
This repository contains the MATLAB code for optimizing a simplified microgrid energy setup using a single Q-learning agent. The code is designed for academic training and was initially supported by the NSF CyberTraining project.

## Code Explanation
The main components of the code are as follows:

Initialization and Setup
Clears the workspace, initializes battery parameters, and defines the state space.

Q-Table Definition
Initializes the Q-table for storing Q-values.

Parameters
Defines the initial parameters for the Q-learning algorithm.

Main Loop
Runs multiple simulations to train the Q-learning agent.

Iteration Loop
Iterates over training episodes and updates the Q-table based on the agent's actions and received rewards.

Action Selection
Chooses actions using an epsilon-greedy strategy to balance exploration and exploitation.

Q-Table Update
Updates the Q-table based on the obtained rewards and future state estimates.

Results and Plotting
Calculates and plots the results to visualize the learning process and the battery state of charge (SOC) over time.

For a detailed explanation of the code, refer to the comments within the script.

