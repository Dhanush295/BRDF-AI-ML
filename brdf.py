import os
import numpy as np
from sklearn.utils import shuffle

class BRDFDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_brdf_data()
        self.organized_data, self.theta_in, self.phi_in = self.organize_brdf_data()
        self.randomized_data = self.randomize_dataset()

    def parse_brdf_file(self, file_path):
        with open(file_path, 'rb') as f:
            # Replace this with your actual parsing logic based on BRDF file format
            brdf_data = np.random.rand(100, 100)  # Replace with actual parsed data
        return brdf_data

    def load_brdf_data(self):
        if not os.path.exists(self.file_path):
            print(f"Directory '{self.file_path}' does not exist.")
            return None

        loaded_data = []

        for filename in os.listdir(self.file_path):
            if filename.endswith('.binary'):
                file_path = os.path.join(self.file_path, filename)
                file_data = self.parse_brdf_file(file_path)
                loaded_data.append(file_data)

        return np.array(loaded_data)

    def organize_brdf_data(self):
        # Generate placeholder theta_in and phi_in (adjust these based on your actual data)
        theta_in = np.linspace(0, np.pi / 2, self.data.shape[0])
        phi_in = np.linspace(0, 2 * np.pi, self.data.shape[1])
        return self.data, theta_in, phi_in

    def randomize_dataset(self):
        reshaped_data = self.organized_data.reshape(-1, self.organized_data.shape[-2], self.organized_data.shape[-1])
        randomized_data = shuffle(reshaped_data, random_state=42)
        return randomized_data
