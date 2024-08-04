Introduction
This repository contains code for a neural network designed to work with the MERL BRDF dataset. 
The neural network architecture, training process, and data processing steps are detailed below.

Data Preprocessing
The MERL BRDF dataset is loaded and organized based on incident and exit angles. 
To ensure diverse learning, the dataset is randomized, mitigating any bias.

Neural Network Architecture
The neural network comprises four layers: an input layer for incident and exit angle vectors, two hidden layers, and an output layer with three neurons for RGB intensities. The neuron counts for the hidden layers are appropriately chosen to balance complexity and learning capacity.

Forward Propagation
Forward propagation involves computing weighted sums and applying activation functions at each layer. 
This process leads to predictions for red, green, and blue intensities.

Backpropagation
Backpropagation is utilized to compute the error between predicted and actual values. 
Gradient descent is employed to iteratively update weights, aiming to minimize the error.
 This process repeats for multiple epochs to refine the network's predictions.

Error Monitoring
Training errors per epoch are visualized using Matplotlib to observe the model's learning curve. 
Testing errors are evaluated to monitor the model's performance on unseen data, ensuring generalizability.

Network Weights
The final learned weights are saved into a text file for potential future use or deployment.

Repository Structure
data/: Contains the MERL BRDF dataset.
src/: Includes the Python code for data preprocessing, neural network architecture, forward and backpropagation, error monitoring, and weight saving.
Usage
Data Preprocessing: Run preprocess_data.py to load and organize the BRDF dataset.
Neural Network Training: Execute train_neural_network.py to train the neural network using the processed data.
Error Monitoring: Run plot_errors.py to visualize the training errors per epoch.
Weight Saving: The learned weights are automatically saved after training in a saved_weights.txt file.
Dependencies
Ensure you have the following dependencies installed:

NumPy
Matplotlib
Other necessary libraries for Python