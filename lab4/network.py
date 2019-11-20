import copy
import random
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(signal):
    return 1.0 / (1.0 + np.exp(-signal))


# dericative of a cost function
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def plot_graph(train_result_per_epoch, test_result_per_epoch, epochs, batch_size, neurons_number, weigth_min,
               weigth_max):
    plt.plot(epochs, train_result_per_epoch, label='validation data')
    plt.plot(epochs, test_result_per_epoch, label='test data')
    plt.xlabel('number of epoch')
    plt.ylabel('accuracy in %')
    plt.legend()
    # plt.title('Batch size: ' + str(batch_size) + " ,number of neurons in hidden layer: " + str(neurons_number))
    plt.title('Momentum')

    plt.savefig("results/{0} - momentum.png".format(str(neurons_number)))
    plt.show()


def create_mini_batches(mini_batch_size, training_data, training_size):
    return [training_data[i:i + mini_batch_size] for i in range(0, training_size, mini_batch_size)]


def cost_derivative(activation, expected):
    return activation - expected


class Network:
    def __init__(self, size, min_weigth, max_weigth):
        self.size = size
        self.layers_number = len(size)

        self.biases = [np.random.randn(y, 1) for y in size[1:]]

        self.min_weigth = min_weigth
        self.max_weigth = max_weigth

        wages_y_dimensions = size[1:]
        wages_x_dimensions = size[:-1]


        #first method initialize randomly from ranfe of min_weight to max_weight
        self.weights = [np.random.uniform(low=min_weigth, high=max_weigth, size=(y, x)) for y, x in
                        zip(wages_y_dimensions, wages_x_dimensions)]

        #second method initialize using a Gaussian distribution with mean 0
        #and standard deviation 1 over the square root of the number of
        #weights connecting to the same neuron

        # self.weights = [np.random.randn(y, x) / np.sqrt(x)
        #                 for x, y in zip(self.sizes[:-1], self.sizes[1:])]

        self.momentum_previous_weigth_updates = [np.zeros(shape=(y, x)) for y, x in
                                                 zip(wages_y_dimensions, wages_x_dimensions)]
        self.momentum_previous_biases_updates = [np.zeros(shape=(y, 1)) for y in size[1:]]

        self.adagrad_accumulated_squared_weight_updates = self.momentum_previous_weigth_updates
        self.adagrad_accumulated_squared_bias_updates = self.momentum_previous_biases_updates

        self.adadelta_last_weight_E_delta_averages = self.momentum_previous_weigth_updates
        self.adadelta_last_weight_E_g_averages = self.momentum_previous_weigth_updates

        self.adadelta_last_bias_E_delta_averages = self.momentum_previous_biases_updates
        self.adadelta_last_bias_E_g_averages = self.momentum_previous_biases_updates

        self.m_weight = self.momentum_previous_weigth_updates
        self.v_weight = self.momentum_previous_weigth_updates

        self.m_bias = self.momentum_previous_biases_updates
        self.v_bias = self.momentum_previous_biases_updates

    def feedforward(self, x):
        # start from input
        output = x
        for b, w in zip(self.biases, self.weights):
            # update output for each layer
            previous_activation = output
            output = sigmoid(np.dot(w, previous_activation) + b)
        return output

    # returns tuple of bias and weights updates, which represents gradient of cost function
    def backpropagation(self, x, y):
        # lists of biases changes initialized with 0
        bias_updates = [np.zeros(b.shape) for b in self.biases]

        # lists of wages changes initialized with 0
        weights_updates = [np.zeros(w.shape) for w in self.weights]

        # first feedforward to store all z's and all activations from all layers

        # list of activations of each layers, starting from last output layer
        activations = [x]

        # list of all z's, where from formula z = (activation * wages) + biases)
        z_list = []

        # start from last activation
        activation = activations[0]

        for w, b in zip(self.weights, self.biases):
            # calculate z - total activation
            z = np.dot(w, activation) + b
            z_list.append(z)

            # update activation
            activation = sigmoid(z)

            activations.append(activation)

        # now backpropagate

        # calculate change -> calculate cost derivative for the last activation and last z
        delta = cost_derivative(activations[-1], y) * sigmoid_derivative(z_list[-1])

        # update the last biases and wages updates list starting from the last element - as we start from last layer
        bias_updates[-1] = delta

        # update the last weights, based on theirs activations
        weights_updates[-1] = np.dot(delta, activations[-2].transpose())

        # continue for all the rest of layers
        for layer_number in range(2, self.layers_number):
            # get last calculated z - total activation
            z = z_list[-layer_number]

            previous_weigths = self.weights[-layer_number + 1]

            # calculate the delta
            delta = np.dot(previous_weigths.transpose(), delta) * sigmoid_derivative(z)

            # add to biases and weights updates as before
            bias_updates[-layer_number] = delta
            weights_updates[-layer_number] = np.dot(delta, activations[-layer_number - 1].transpose())

        return bias_updates, weights_updates

    def update_weights_biases(self, mini_batch, learning_rate):
        biases_updates = [np.zeros(b.shape) for b in self.biases]
        weights_updates = [np.zeros(w.shape) for w in self.weights]

        for input, expected_result in mini_batch:
            delta_bias, delta_weigth = self.backpropagation(input, expected_result)

            # update biases updates list
            biases_updates = [old_bias + bias_update for old_bias, bias_update in zip(biases_updates, delta_bias)]

            # update weight updates list
            weights_updates = [old_weight + weight_update for old_weight, weight_update in
                               zip(weights_updates, delta_weigth)]

        # update current weigth of network
        self.weights = [w - learning_rate * weigth_update for w, weigth_update in zip(self.weights, weights_updates)]

        # update current biases of network
        self.biases = [b - learning_rate * bias_update for b, bias_update in zip(self.biases, biases_updates)]

    def update_weights_biases_with_momentum(self, mini_batch, learning_rate, momentum_factor):
        biases_updates = [np.zeros(b.shape) for b in self.biases]
        weights_updates = [np.zeros(w.shape) for w in self.weights]

        for input, expected_result in mini_batch:
            delta_bias, delta_weigth = self.backpropagation(input, expected_result)

            # update biases updates list
            biases_updates = [old_bias + bias_update for old_bias, bias_update in zip(biases_updates, delta_bias)]

            # update weight updates list
            weights_updates = [old_weight + weight_update for old_weight, weight_update in
                               zip(weights_updates, delta_weigth)]

        # update current weigth of network
        self.weights = [w - learning_rate * weigth_update + (momentum_factor * previous_weigth) for w, weigth_update,
                                                                                                    previous_weigth in
                        zip(self.weights, weights_updates, self.momentum_previous_weigth_updates)]

        # update current biases of network
        self.biases = [b - learning_rate * bias_update + (momentum_factor * previous_bias) for b, bias_update,
                                                                                               previous_bias in
                       zip(self.biases, biases_updates, self.momentum_previous_biases_updates)]

        # assign previous updates
        self.momentum_previous_biases_updates = biases_updates
        self.momentum_previous_weigth_updates = weights_updates

    def update_weights_biases_with_adagrad(self, mini_batch, learning_rate, epsilon):
        biases_updates = [np.zeros(b.shape) for b in self.biases]
        weights_updates = [np.zeros(w.shape) for w in self.weights]

        for input, expected_result in mini_batch:
            delta_bias, delta_weigth = self.backpropagation(input, expected_result)

            # update biases updates list
            biases_updates = [old_bias + bias_update for old_bias, bias_update in zip(biases_updates, delta_bias)]

            # update weight updates list
            weights_updates = [old_weight + weight_update for old_weight, weight_update in
                               zip(weights_updates, delta_weigth)]

        # update current weight of network
        corrected_w = []
        for w, weigth_update, acc_G in zip(self.weights, weights_updates,
                                           self.adagrad_accumulated_squared_weight_updates):
            corrected_w.append(w - ((learning_rate * weigth_update) / np.sqrt(np.add(acc_G, epsilon))))

        self.weights = corrected_w

        # update current biases of network
        corrected_b = []
        for b, bias_update, acc_G in zip(
                self.biases, biases_updates, self.adagrad_accumulated_squared_bias_updates):
            corrected_b.append(b - ((learning_rate * bias_update) / np.sqrt(np.add(acc_G, epsilon))))

        self.biases = corrected_b

        # update accumulated
        self.adagrad_accumulated_squared_weight_updates = np.add(self.adagrad_accumulated_squared_weight_updates,
                                                                 (np.power(weights_updates, 2)))
        self.adagrad_accumulated_squared_bias_updates = np.add(self.adagrad_accumulated_squared_bias_updates,
                                                               np.power(biases_updates, 2))

    def update_weights_biases_with_adadelta(self, mini_batch, y=0.9, epsilon=1e-4):
        biases_updates = [np.zeros(b.shape) for b in self.biases]
        weights_updates = [np.zeros(w.shape) for w in self.weights]

        for input, expected_result in mini_batch:
            delta_bias, delta_weigth = self.backpropagation(input, expected_result)

            # update biases updates list
            biases_updates = [old_bias + bias_update for old_bias, bias_update in zip(biases_updates, delta_bias)]

            # update weight updates list
            weights_updates = [old_weight + weight_update for old_weight, weight_update in
                               zip(weights_updates, delta_weigth)]

        # update current weight of network
        corrected_weights = []
        corrected_last_E_delta_averages = []
        corrected_last_E_g_averages = []
        for w, weight_update, last_E_delta, last_E_g in zip(self.weights, weights_updates,
                                                            self.adadelta_last_weight_E_delta_averages,
                                                            self.adadelta_last_weight_E_g_averages):
            delta_weight = self.adadelta(corrected_last_E_delta_averages, corrected_last_E_g_averages, epsilon,
                                         last_E_delta, last_E_g, weight_update, y)

            corrected_weights.append(w - delta_weight)

        self.weights = corrected_weights
        self.adadelta_last_weight_E_delta_averages = corrected_last_E_delta_averages
        self.adadelta_last_weight_E_g_averages = corrected_last_E_g_averages

        # update current biases of network
        corrected_biases = []
        corrected_last_E_delta_averages = []
        corrected_last_E_g_averages = []
        for b, bias_update, last_E_delta, last_E_g in zip(self.biases, biases_updates,
                                                          self.adadelta_last_bias_E_delta_averages,
                                                          self.adadelta_last_bias_E_g_averages):
            delta_bias = self.adadelta(corrected_last_E_delta_averages, corrected_last_E_g_averages, epsilon,
                                       last_E_delta, last_E_g, bias_update, y)

            corrected_biases.append(b - delta_bias)

        self.biases = corrected_biases
        self.adadelta_last_bias_E_delta_averages = corrected_last_E_delta_averages
        self.adadelta_last_bias_E_g_averages = corrected_last_E_g_averages

    def adadelta(self, corrected_last_E_delta_averages, corrected_last_E_g_averages, epsilon, last_E_delta, last_E_g,
                 update, y):
        e_delta = y * last_E_delta + (1 - y) * np.power(update, 2)

        corrected_last_E_delta_averages.append(e_delta)

        rms_last_delta = np.sqrt(np.add(e_delta, epsilon))

        e_g = y * last_E_g + (1 - y) * np.power(update, 2)

        corrected_last_E_g_averages.append(e_g)

        rms_update = np.sqrt(np.add(e_g, epsilon))

        delta_update = (rms_last_delta * update) / rms_update

        return delta_update

    def update_weights_biases_with_adam(self, mini_batch, learning_rate, t, epsilon):
        biases_updates = [np.zeros(b.shape) for b in self.biases]
        weights_updates = [np.zeros(w.shape) for w in self.weights]

        for input, expected_result in mini_batch:
            delta_bias, delta_weigth = self.backpropagation(input, expected_result)

            # update biases updates list
            biases_updates = [old_bias + bias_update for old_bias, bias_update in zip(biases_updates, delta_bias)]

            # update weight updates list
            weights_updates = [old_weight + weight_update for old_weight, weight_update in
                               zip(weights_updates, delta_weigth)]

        # g = compute_gradient(x, y)
        # m = beta_1 * m + (1 - beta_1) * g
        # v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
        # m_hat = m / (1 - np.power(beta_1, t))
        # v_hat = v / (1 - np.power(beta_2, t))
        # w = w - step_size * m_hat / (np.sqrt(v_hat) + epsilon)

        beta_1 = 0.9
        beta_2 = 0.999

        # update current weight of network
        weight_corrected = []
        m_weight_corrected = []
        v_weight_corrected = []
        for w, weigth_update, m, v in zip(self.weights, weights_updates, self.m_weight, self.v_bias):
            result, m_weight, v_weight = self.adam(beta_1, beta_2, epsilon, t, weigth_update, m, v)

            m_weight_corrected.append(m_weight)
            v_weight_corrected.append(v_weight)
            weight_corrected.append(w - learning_rate * result)

        self.weights = weight_corrected
        self.m_weight = m_weight_corrected
        self.v_weight = v_weight_corrected

        # update current biases of network
        corrected_biases = []
        m_bias_corrected = []
        v_bias_corrected = []
        for b, bias_update, m, v in zip(self.biases, biases_updates, self.m_bias, self.v_bias):
            result, m_bias, v_bias = self.adam(beta_1, beta_2, epsilon, t, bias_update, m, v)

            corrected_biases.append(b - learning_rate * result)
            m_bias_corrected.append(m_bias)
            v_bias_corrected.append(v_bias)

        self.biases = corrected_biases
        self.m_bias = m_bias_corrected
        self.v_bias = v_bias_corrected

    def adam(self, beta_1, beta_2, epsilon, t, update, old_m, old_v):
        m = beta_1 * old_m + (1 - beta_1) * update
        v = beta_2 * old_v + (1 - beta_2) * np.power(update, 2)
        m_hat = m / (1 - np.power(beta_1, t))
        v_hat = v / (1 - np.power(beta_2, t))
        result = m_hat / (np.sqrt(v_hat) + epsilon)
        return result, m, v

    def evaluate(self, test_data):
        # using argmax -> the highest value on given neuron is the response from neural net
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        total_result = 0
        for response_from_net, expected_response in test_results:
            total_result += int(response_from_net == expected_response)

        return total_result

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate, test_data,
                                    org_tr_data):
        training_data = list(training_data)

        org_tr_data_length = len(list(copy.deepcopy(org_tr_data)))
        test_data_length = len(list(copy.deepcopy(test_data)))

        train_result_per_epoch = []
        test_result_per_epoch = []
        epoch_history = []

        for epoch_number in range(epochs):
            # randomly shuffle training data
            random.shuffle(training_data)
            # create mini batch with given size
            mini_batches = create_mini_batches(mini_batch_size, training_data, len(training_data))

            for t, mini_batch in enumerate(mini_batches):
                # self.update_weights_biases(mini_batch, learning_rate)
                # self.update_weights_biases_with_momentum(mini_batch, learning_rate, 0.001)
                # self.update_weights_biases_with_adagrad(mini_batch, learning_rate, 1e-4)
                # self.update_weights_biases_with_adadelta(mini_batch, 0.9, 1e-4)
                self.update_weights_biases_with_adam(mini_batch, learning_rate, t+1, 1e-4)
            if test_data:
                print("epoch: " + str(epoch_number) + " efficency:"
                      + str((self.evaluate(copy.deepcopy(test_data)) / test_data_length) * 100.0))

            train_result_per_epoch.append((self.evaluate(copy.deepcopy(org_tr_data)) * 1.0) / org_tr_data_length *
                                          100.0)
            test_result_per_epoch.append((self.evaluate(copy.deepcopy(test_data)) * 1.0) / test_data_length * 100.0)
            epoch_history.append(epoch_number)

        plot_graph(train_result_per_epoch, test_result_per_epoch, epoch_history, mini_batch_size, self.size[1],
                   self.min_weigth, self.max_weigth)

    def __str__(self):
        return "Network: " + "size: " + str(self.size) + " layers number: " + str(self.layers_number)
