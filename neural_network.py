import numpy as np
import scipy.optimize


class sigmoid:
    def __init__(self):
        pass

    def f(self, x):
        return 1/(1 + np.exp(-x))

    def derivative(self, x):
        return self.f(x) * (1 - self.f(x))


class tanh:
    def __init__(self):
        pass

    def f(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - self.f(x)**2


class relu:
    def __init__(self):
        pass

    def f(self, x):
        x[x <= 0] = 0
        return x

    def derivative(self, x):
        g = np.zeros_like(x)
        g[x > 0] = 1
        return g


def network(weights, input_array, activation, architecture):
    # Run network on input
    N_layers = len(architecture) - 1

    # Store input in layer variable
    previous_layer = input_array

    # Iniatialise end index of weights array
    end_idx = 0

    # Initialise layers array
    layers = [input_array]

    # Loop through layers
    for i in range(N_layers):
        # Size of layers
        previous_layer_size = architecture[i]
        current_layer_size = architecture[i+1]

        # Get start index of weights array
        start_idx = end_idx

        # Calculate end index of weights array
        end_idx = start_idx + previous_layer_size * \
            current_layer_size + current_layer_size

        # Construct weights matrix
        weights_matrix = np.reshape(
            weights[start_idx:end_idx - current_layer_size],
            (current_layer_size, previous_layer_size)
        )

        # Get biases
        biases_array = weights[end_idx - current_layer_size:end_idx]

        # Calculate current layer
        current_layer = activation.f(weights_matrix.dot(
            previous_layer) + biases_array)

        # Add layer to array
        layers.append(current_layer)

        # Set current layer as previous layer
        previous_layer = current_layer

    # Return final layer
    return layers


def loss_function(weights, input_matrix, truth_matrix, activation, architecture):
    # Calculate the loss function of the network
    total_loss = 0.0

    # Get number of inputs
    N_inputs = input_matrix.shape[0]

    # Loop through inputs
    for i in range(N_inputs):
        # Get arrays
        input_array = input_matrix[i]
        truth_array = truth_matrix[i]

        # Get output of network
        output = network(weights, input_array, activation, architecture)

        # Calculate loss
        loss = np.average((output[-1] - truth_array)**2)

        # Add to total loss
        total_loss += loss

    # Return total loss
    return total_loss / N_inputs


def train_network(input_matrix, truth_matrix, activation, learning_rate, architecture, file_path=""):
    # Train network on input data
    print("Training network...")

    # Generate random weights and biases
    N_inputs = input_matrix.shape[0]
    N_layers = len(architecture) - 1

    N_weights_and_biases = np.sum(
        architecture[:-1] * architecture[1:]) + np.sum(architecture[1:])

    weights = np.random.uniform(
        low=-1.0, high=1.0, size=N_weights_and_biases)

    # Gradient descent to get optimal parameters
    epoch = 0

    while epoch < 1000:
        try:
            # Calculate current loss
            loss = loss_function(weights, input_matrix,
                                 truth_matrix, activation, architecture)

            print("Epoch: %i" % epoch)
            print("Loss:  %f\n" % loss)

            epoch += 1

            weights_change = np.zeros_like(weights)

            # Loop through training inputs
            for mu in range(N_inputs):
                # Do forward pass of network
                layers = network(
                    weights, input_matrix[mu], activation, architecture)

                # Initialise end index of weights array
                start_idx = len(weights)

                # Backpropagate through layers
                for layer_i in range(N_layers)[::-1]:
                    # Index of layer
                    n = layer_i + 1

                    # Size of layers
                    previous_layer_size = architecture[n-1]
                    current_layer_size = architecture[n]

                    # Get start index of weights array
                    end_idx = start_idx

                    # Calculate end index of weights array
                    start_idx = end_idx - previous_layer_size * \
                        current_layer_size - current_layer_size

                    # Weights and biases
                    layer_weights = weights[start_idx:end_idx -
                                            current_layer_size]

                    biases_array = weights[end_idx -
                                           current_layer_size:end_idx]

                    # Construct weights matrix
                    current_weights_matrix = np.reshape(
                        layer_weights,
                        (current_layer_size, previous_layer_size)
                    )

                    # Calculate current layer value
                    current_layer = layers[n]
                    previous_layer = layers[n-1]

                    # Output layer
                    if n == N_layers:
                        delta = (truth_matrix[mu] - current_layer) * activation.derivative(
                            current_weights_matrix.dot(previous_layer) + biases_array)

                        # Update weights
                        weights_change[start_idx:end_idx -
                                       current_layer_size] += np.tile(delta * current_layer, previous_layer_size)

                        # Update biases
                        weights_change[end_idx -
                                       current_layer_size:end_idx] += delta

                    # Other layers
                    else:
                        delta = np.sum(delta[:, None] * next_weights_matrix * activation.derivative(
                            current_weights_matrix.dot(previous_layer) + biases_array), axis=0)

                        # Update weights
                        weights_change[start_idx:end_idx -
                                       current_layer_size] += np.tile(delta * current_layer, previous_layer_size)

                        # Update biases
                        weights_change[end_idx -
                                       current_layer_size:end_idx] += delta

                    next_weights_matrix = current_weights_matrix

            # Change weights
            weights += learning_rate * weights_change

        except KeyboardInterrupt:
            print("Keyboard Interrupt. Ending training.")
            break

    # Format file path
    if len(file_path) > 0:
        if file_path[-1] != "/":
            file_path += "/"

    # Write weights to a file
    print("Saving trained weights to file")
    np.save(file_path + "trained_weights.npy", weights)
    np.save(file_path + "architecture.npy", architecture)


def run_network(input_array, activation, file_path=""):
    # Run trained network on an input array
    # Format file path
    if len(file_path) > 0:
        if file_path[-1] != "/":
            file_path += "/"

    # Load saved weights
    try:
        trained_weights = np.load(file_path + "trained_weights.npy")
        architecture = np.load(file_path + "architecture.npy")

    except FileNotFoundError as e:
        raise FileNotFoundError(str(
            e) + ". Please use neural_network.train_network() before running the network.")

    # Get result of network
    result = network(trained_weights, input_array, activation, architecture)

    # Return result
    return result
