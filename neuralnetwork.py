import numpy as np
from sklearn.model_selection import train_test_split

class BRDFFormat:
    def __init__(self, data, theta_in, phi_in):
        self.data = data  # Placeholder for your BRDF data
        self.theta_in = theta_in  # Incident angle vector
        self.phi_in = phi_in  # Exit angle vector

class NeuralNetwork:
    def __init__(self, input_size, hidden_layer1_size, hidden_layer2_size, output_size):
        self.weights1 = np.random.rand(input_size, hidden_layer1_size)
        self.weights2 = np.random.rand(hidden_layer1_size, hidden_layer2_size)
        self.weights3 = np.random.rand(hidden_layer2_size, output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # def tanh(self, x):
    #     return np.tanh(x)

    
    def train_neural_network(self, input_data, actual_outputs, learning_rate=0.01, epochs=100):
        # Splitting data into training and testing sets 
        input_train, input_test, outputs_train, outputs_test = train_test_split(
            input_data, actual_outputs, test_size=0.2, random_state=42
        )

        # Normalize training and testing data
        normalized_input_train = (input_train - np.mean(input_train)) / np.std(input_train)
        normalized_input_test = (input_test - np.mean(input_test)) / np.std(input_test)
        normalized_outputs_train = (outputs_train - np.mean(outputs_train)) / np.std(outputs_train)
        normalized_outputs_test = (outputs_test - np.mean(outputs_test)) / np.std(outputs_test)

        # Training loop
        for epoch in range(epochs):
            # Forward propagation
            output_predictions = self.forward_propagation(normalized_input_train)

            # Backward propagation
            self.backward_propagation(
                normalized_input_train, output_predictions, normalized_outputs_train, learning_rate, 1
            )

            # Calculate training MSE after each epoch and append to the list
            training_mse = np.mean(np.square(normalized_outputs_train - output_predictions))
            print(f"Epoch {epoch + 1}/{epochs} - Training Mean Squared Error: {training_mse}")

            # Testing phase
            testing_output_predictions = self.forward_propagation(normalized_input_test)
            testing_mse = np.mean(np.square(normalized_outputs_test - testing_output_predictions))
            print(f"Epoch {epoch + 1}/{epochs} - Testing Mean Squared Error: {testing_mse}")

        return self
    

    def forward_propagation(self, input_data):
        # Input layer to hidden layer 1
        self.hidden_layer1_output = self.sigmoid(np.dot(input_data, self.weights1))

        # Hidden layer 1 to hidden layer 2
        self.hidden_layer2_output = self.sigmoid(np.dot(self.hidden_layer1_output, self.weights2))

        # Hidden layer 2 to output layer
        output_predictions = np.dot(self.hidden_layer2_output, self.weights3)
        return output_predictions
    
    def backward_propagation(self, input_data, output_predictions, actual_outputs, learning_rate, epochs):
        for epoch in range(epochs):
            # Calculate the error using a suitable loss function (e.g., mean squared error)
            error = actual_outputs - output_predictions

            # Compute gradients for the output layer
            output_gradients = error

            # Adjust weights between hidden layer 2 and output layer
            self.weights3 += learning_rate * np.dot(self.hidden_layer2_output.T, output_gradients)

            # Compute gradients for hidden layer 2
            hidden_layer2_gradients = np.dot(output_gradients, self.weights3.T) * (self.hidden_layer2_output * (1 - self.hidden_layer2_output))

            # Adjust weights between hidden layer 1 and hidden layer 2
            self.weights2 += learning_rate * np.dot(self.hidden_layer1_output.T, hidden_layer2_gradients)

            # Compute gradients for hidden layer 1
            hidden_layer1_gradients = np.dot(hidden_layer2_gradients, self.weights2.T) * (self.hidden_layer1_output * (1 - self.hidden_layer1_output))

            # Adjust weights between input layer and hidden layer 1
            self.weights1 += learning_rate * np.dot(input_data.T, hidden_layer1_gradients)

            # Perform forward propagation again with updated weights
            output_predictions = self.forward_propagation(input_data)

            # Calculate and print the mean squared error for each epoch
            mse = np.mean(np.square(actual_outputs - output_predictions))
            print(f"Epoch {epoch + 1}/{epochs} - Mean Squared Error: {mse}")
