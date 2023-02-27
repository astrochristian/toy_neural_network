import neural_network
import numpy as np
from mnist import MNIST

# Load handwriting data
mndata = MNIST('./data')

# Get images and labels
images, labels = mndata.load_training()

N_images   = len(images)
image_size = len(images[0])

# Convert images to numpy matrix
images_matrix = np.asarray(images)

# Generate truth matrix
labels_matrix = np.zeros((N_images, 10))

for i, label in enumerate(labels):
    labels_matrix[i, label] = 1.0

# Define network architecture
architecture = np.asarray([image_size, 16, 16, 10]) # Two layer network with 16 neurons each

# Train network
neural_network.train_network(images_matrix, labels_matrix, architecture)