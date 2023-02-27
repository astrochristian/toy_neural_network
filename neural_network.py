import numpy as np
import scipy.optimize

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def network(weights, input_array, architecture):
    # Run network on input
    N_layers = len(architecture) - 1

    # Store input in layer variable
    previous_layer = input_array

    # Iniatialise end index of weights array
    end_idx = 0

    # Loop through layers
    for i in range(N_layers):
        # Size of layers
        previous_layer_size = len(previous_layer)
        current_layer_size  = architecture[i+1]
        
        # Get start index of weights array
        start_idx = end_idx

        # Calculate end index of weights array
        end_idx = start_idx + previous_layer_size * current_layer_size + current_layer_size

        # Construct weights matrix
        weights_matrix = np.reshape(
            weights[start_idx:end_idx - current_layer_size],
            (current_layer_size, previous_layer_size)
        )

        # Get biases
        biases_array = weights[end_idx - current_layer_size:end_idx]   

        # Calculate current layer
        current_layer = sigmoid(weights_matrix.dot(previous_layer) + biases_array)

        # Set current layer as previous layer
        previous_layer = current_layer

    # Return final layer
    return current_layer


def cost_function(weights, input_matrix, truth_matrix, architecture):
    # Calculate the cost function of the network
    total_cost = 0

    # Get number of inputs
    N_inputs = input_matrix.shape[0]

    # Loop through inputs
    for i in range(N_inputs):
        # Get arrays
        input_array = input_matrix[i]
        truth_array = truth_matrix[i]

        # Get output of network
        output = network(weights, input_array, architecture)

        # Calculate cost
        cost = np.average((output - truth_array)**2)

        # Add to total cost
        total_cost += cost

    # Return total cost
    return cost

def train_network(input_matrix, truth_matrix, architecture, file_path=""):
    # Train network on input data
    print("Training network...")

    # Generate random weights and biases
    N_weights_and_biases = np.sum(architecture[:-1] * architecture[1:]) + np.sum(architecture[1:]) 

    init_weights = np.random.uniform(low=-1.0, high=1.0, size=N_weights_and_biases)

    # Gradient descent to get optimal parameters
    print("Minimising cost...")
    res = scipy.optimize.minimize(cost_function, init_weights, args=(input_matrix, truth_matrix, architecture))

    # Get optimal weights
    print("Cost minimised"))
    opt_weights = res.x

    # Format file path
    if len(file_path) > 0:
        if file_path[-1] != "/":
            file_path += "/"

    # Write weights to a file
    print("Saving trained weights to file")
    np.save(file_path + "trained_weights.npy", opt_weights)
    np.save(file_path + "architecture.npy", architecture)

def run_network(input_array, file_path=""):
    # Run trained network on an input array
    # Format file path
    if len(file_path) > 0:
        if file_path[-1] != "/":
            file_path += "/"

    # Load saved weights
    try:
        trained_weights = np.load(file_path + "trained_weights.npy")
        architecture    = np.load(file_path + "architecture.npy")

    except FileNotFoundError as e:
        raise FileNotFoundError(str(e) + ". Please use neural_network.train_network() before running the network.")

    # Get result of network
    result = network(trained_weights, input_array, architecture)    

    # Return result
    return result