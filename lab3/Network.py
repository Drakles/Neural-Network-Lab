import random
import numpy as np


def sigmoid(input):
    return 1.0 / (1.0 + np.exp(-input))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network:
    def __init__(self, size):
        self.size = size
        self.layers_number = len(size)

        self.biases = [np.random.randn(y, 1) for y in size[1:]]

        wages_y_dimensions = size[1:]
        wages_x_dimensions = size[:-1]

        self.weights = [np.random.randn(y, x) for y, x in zip(wages_y_dimensions, wages_x_dimensions)]

    def feedforward(self, x):
        # output as activation
        output = x
        for b, w in zip(self.biases, self.weights):
            output = sigmoid(np.dot(w, output) + b)
        return output

    # returns tuple of bias and weights updates, which represents gradient of cost function
    def backprop(self, x, y):
        # lists of biases changes
        bias_updates = [np.zeros(b.shape) for b in self.biases]

        # lists of wages changes
        weights_updates = [np.zeros(w.shape) for w in self.weights]

        # feedforward to store all z's and all activations from all layers

        # list of activations of each layers, starting from last output layer
        activations = [x]

        # list of all z's, where from formula z = (activation * wages) + biases)
        z_list = []

        # start from last activation
        activation = activations[0]

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            z_list.append(z)

            # update activation
            activation = sigmoid(z)

            activations.append(activation)

        # now backpropagate

        # calculate change
        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(z_list[-1])

        # update biases and wages updates list starting from the last element - as we start from last layer
        bias_updates[-1] = delta
        # from formula from book
        weights_updates[-1] = np.dot(delta, activations[-2].transpose())

        # continue for all the rest of layers
        for layer_number in range(2, self.layers_number):
            # get last calculated z
            z = z_list[-layer_number]

            previous_weigths = self.weights[-layer_number + 1]

            delta = np.dot(previous_weigths.transpose(), delta) * sigmoid_derivative(z)

            # add to biases and weights updates
            bias_updates[-layer_number] = delta
            weights_updates[-layer_number] = np.dot(delta, activations[-layer_number - 1].transpose())

        return bias_updates, weights_updates

    def update_weights_biases(self, mini_batch, learning_rate):
        biases_updates = [np.zeros(b.shape) for b in self.biases]
        weights_updates = [np.zeros(w.shape) for w in self.weights]

        for input, expected_result in mini_batch:
            delta_bias, delta_weigth = self.backprop(input, expected_result)

            # update biases updates list
            biases_updates = [old_bias + bias_update for old_bias, bias_update in zip(biases_updates, delta_bias)]

            # update weight updates list
            weights_updates = [old_weight + weight_update for old_weight, weight_update in
                               zip(weights_updates, delta_weigth)]

        # update weigth of network
        self.weights = [w - learning_rate * weigth_update for w, weigth_update in zip(self.weights, weights_updates)]

        # update biases of network
        self.biases = [b - learning_rate * bias_update for b, bias_update in zip(self.biases, biases_updates)]

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate, test_data):
        training_data = list(training_data)
        training_size = len(training_data)

        test_data = list(test_data)
        test_size = len(test_data)

        for epoch_number in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, training_size, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_weights_biases(mini_batch, learning_rate)
            if test_data:
                print("epoch: " + str(epoch_number) + " efficency:" + str((self.evaluate(test_data) / test_size)*100.0))

    def cost_derivative(self, activation, expected):
        return activation - expected

    def __str__(self):
        return "Network: " + "size: " + str(self.size) + " layers number: " + str(self.layers_number)
