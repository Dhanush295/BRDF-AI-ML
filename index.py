
import numpy as np
from brdf import BRDFDataset
from neuralnetwork import NeuralNetwork, BRDFFormat
import matplotlib.pyplot as plt


#Todo1: Data Preprocessing
# Example usage:
file_path = './brdfs'
brdf_dataset = BRDFDataset(file_path)

# Access the organized and randomized data
organized_data, theta_in, phi_in = brdf_dataset.organized_data, brdf_dataset.theta_in, brdf_dataset.phi_in
randomized_data = brdf_dataset.randomized_data

# Sample print statements to verify the shapes and content of the organized and randomized data
print("Shape of organized data: ", organized_data.shape)
print("Shape of randomized dataset: ", randomized_data.shape)
print("Sample organized data at [0, 0]: ", organized_data[0, 0])
print("Sample randomized data at index 0: ", randomized_data[0], "\n")



#Todo-2:Forward Propogating

input_size = 2  # Number of neurons in input layer (incident and exit angle vectors)
hidden_layer1_size = 4  
hidden_layer2_size = 4  
output_size = 3  # Number of neurons in output layer (RGB intensities)

neural_net = NeuralNetwork(input_size, hidden_layer1_size, hidden_layer2_size, output_size)

# Assuming organized data is available (theta_in, phi_in) from the previous steps
organized_data = BRDFFormat(
    np.random.rand(3, 100, 100),  # Example BRDF data (replace with actual data)
    np.linspace(0, np.pi / 2, 100),  # Example incident angle vector (adjust as needed)
    np.linspace(0, 2 * np.pi, 100)  # Example exit angle vector (adjust as needed)
)

# Assuming theta_in and phi_in are organized angles from BRDF data
input_data = np.vstack((organized_data.theta_in.ravel(), organized_data.phi_in.ravel())).T

# Perform forward propagation to get RGB predictions
output_predictions = neural_net.forward_propagation(input_data)

# Split the output predictions into Red, Green, and Blue intensities
red_predictions = output_predictions[:, 0]  
green_predictions = output_predictions[:, 1]  
blue_predictions = output_predictions[:, 2]  

# Print the predictions for Red, Green, and Blue intensities for the first few samples
num_samples_to_display = 5
print(f"Predictions for Red intensity (first {num_samples_to_display} samples):")
print(red_predictions[:num_samples_to_display])

print(f"\nPredictions for Green intensity (first {num_samples_to_display} samples):")
print(green_predictions[:num_samples_to_display])

print(f"\nPredictions for Blue intensity (first {num_samples_to_display} samples):")
print(blue_predictions[:num_samples_to_display])
# Print the shape of output predictions
print("Shape of output predictions:", output_predictions.shape)


  
#Todo3: backPropogation and error Monitoring 
num_samples_train = 100
normalized_input_data = np.random.rand(num_samples_train, input_size)
normalized_actual_outputs = np.random.rand(num_samples_train, output_size)

# Example testing data (Replace this with your actual data)
num_samples_test = 50
normalized_testing_input_data = np.random.rand(num_samples_test, input_size)
normalized_testing_actual_outputs = np.random.rand(num_samples_test, output_size)
# Initializing the neural network
neural_net = NeuralNetwork(input_size, hidden_layer1_size, hidden_layer2_size, output_size)

# Training the neural network using the defined method
neural_net.train_neural_network(
    normalized_input_data, normalized_actual_outputs,
    learning_rate=0.01, epochs=100
)

# Initialize lists to store errors during training and testing
training_errors = []
testing_errors = []

# Hyperparameters
learning_rate = 0.001
epochs = 100

# Training loop
for epoch in range(epochs):
    # Training phase
    output_predictions = neural_net.forward_propagation(normalized_input_data)
    neural_net.backward_propagation(normalized_input_data, output_predictions, normalized_actual_outputs, learning_rate, 1)
    
    # Calculate training MSE after each epoch and append to the list
    training_mse = np.mean(np.square(normalized_actual_outputs - output_predictions))
    training_errors.append(training_mse)
    print(f"Epoch {epoch + 1}/{epochs} - Training Mean Squared Error: {training_mse}")
    
    # Testing phase
    testing_output_predictions = neural_net.forward_propagation(normalized_testing_input_data)
    testing_mse = np.mean(np.square(normalized_testing_actual_outputs - testing_output_predictions))
    testing_errors.append(testing_mse)
    print(f"Epoch {epoch + 1}/{epochs} - Testing Mean Squared Error: {testing_mse}")

# Plotting the training and testing errors per epoch
plt.figure()
plt.plot(range(epochs), training_errors, label='Training Error')
plt.plot(range(epochs), testing_errors, label='Testing Error')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Training and Testing Errors per Epoch')
plt.legend()
plt.show()

# Todo 5: Network Weights:

# Save the final learned weights into a text file
weights_path = 'saved_weights.txt'  # Replace this with your desired file path
with open(weights_path, 'w') as file:
    file.write("Weights Layer 1:\n")
    np.savetxt(file, neural_net.weights1)
    file.write("\nWeights Layer 2:\n")
    np.savetxt(file, neural_net.weights2)
    file.write("\nWeights Layer 3:\n")
    np.savetxt(file, neural_net.weights3)

print("Final learned weights saved to", weights_path)

