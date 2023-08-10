import neural_network
import numpy as np
from mnist import MNIST

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

# Load handwriting data
mndata = MNIST('./data')

# Get images and labels
images, labels = mndata.load_training()

N_images = len(images)
image_size = len(images[0])

# Convert images to numpy matrix
images_matrix = np.asarray(images)

# Scale between 0 and 1
images_matrix = images_matrix / np.max(images_matrix)

# Generate truth matrix
labels_matrix = np.zeros((N_images, 10))

for i, label in enumerate(labels):
    labels_matrix[i, label] = 1.0

# Define network architecture
# Two layer network with 16 neurons each
architecture = np.asarray([image_size, 16, 16, 10])

# Train network
neural_network.train_network(
    images_matrix, labels_matrix, neural_network.sigmoid(), 0.001, architecture)
