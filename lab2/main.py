import random
import numpy as np

from lab2.adaline import Adaline


def error_rate(current_output, expected_result):
    # using mean squared error rate
    return np.square(np.subtract(expected_result, current_output)).mean()


def update_weights(error, wages, input_signal, learning_rate):
    print("before update: " + str(wages))
    for i, wage in enumerate(wages):
        wages[i] = wage + learning_rate * error * input_signal[i]
    print("after updates: " + str(wages))


def learn(adaline, inputs, wages, learning_rate):
    for input_expected_result_pair in inputs:
        input_signal = input_expected_result_pair[0]
        current_output = adaline.compute(input_signal, wages)

        error = error_rate(current_output, input_expected_result_pair[1])

        update_weights(error, wages, input_signal, learning_rate)
        adaline.update_bias(error, learning_rate)


if __name__ == '__main__':
    theta = 0.5
    adaline = Adaline(0.1, theta)
    learning_rate = 0.1

    # input array with pair of input data as list and expected result from lab2
    inputs = np.array([[np.array([0, 0]), 0], [np.array([0, 1]), 1], [np.array([1, 0]), 1], [np.array([1, 1]), 1]])
    wages = np.array([random.uniform(-0.5, 0.5) for x in range(len(inputs[0][0]))])

    for x in range(100):
        print("epoch " + str(x))
        # print("wages before: " + str(wages) + " bias: " + str(adaline.bias))
        learn(adaline, inputs, wages, learning_rate)
        # print("wages after: " + str(wages) + " bias: " + str(adaline.bias))

    print("final wages:" + str(wages))
    print("final bias: " + str(adaline.bias))
    for i, input_signal in enumerate(inputs):
        print("input:" + str(input_signal[0]) + " expected result: " + str(
            input_signal[1]) + " response from adaline: " + str(adaline.compute(input_signal[0], wages)))

    repeat_number = 100
    range_random_wages = [1.0, 0.8, 0.5, 0.2, 0.1, 0.01]
    learning_rates = [0.5, 0.25, 0.1, 0.01, 0.001]
