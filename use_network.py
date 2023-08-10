import neural_network
import numpy as np
import time
from mnist import MNIST
from termcolor import colored

# Load handwriting data
mndata = MNIST('./data')

# Get images and labels
images, labels = mndata.load_testing()

# Loop through test images
for i in range(10):
    input_array = np.asarray(images[i])

    print(mndata.display(images[i]), "\n")

    input_array = input_array / np.max(input_array)

    result = neural_network.run_network(
        input_array, neural_network.sigmoid())[-1]
    total = np.sum(result)

    for j in range(10):
        norm_val = result[j]/total

        if j == np.argmax(result):
            print(colored("%i: %f" % (j, norm_val), "green"))
        else:
            print("%i: %f" % (j, norm_val))

    print("\n\n")

    time.sleep(1)
