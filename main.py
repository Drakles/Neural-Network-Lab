from perceptron import Perceptron
import numpy as np


def error_rate(current_output, expected_result):
    return expected_result - current_output


def update_weigth(error, wages, input_signal, learning_rate):
    for i, wage in enumerate(wages):
        wages[i] = wage + learning_rate * error * input_signal[i]


def learn(perceptron, input_signals, wages, expected_results, learning_rate):
    for i, input_signal in enumerate(input_signals):
        current_output = perceptron.compute(input_signal, wages)
        expected_result = expected_results[i]
        error = error_rate(current_output, expected_result)

        update_weigth(error, wages, input_signal, learning_rate)


if __name__ == '__main__':
    bias = 0.5
    learning_rate = 0.1

    perceptron = Perceptron(bias)

    input_signals = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_results = np.array([0, 1, 1, 1])
    wages = np.array([0.1, 0.1])

    epochs = 100

    for epoch in range(epochs):
        learn(perceptron, input_signals, wages, expected_results, learning_rate)

    print("wages:")
    for wage in wages:
        print(wage)

    for i, input_signal in enumerate(input_signals):
        print("input:" + str(input_signal[0]) + "," + str(
            input_signal[1]) + " expected result: " + str(
            expected_results[i]) + " response from perceptron: " + str(
            perceptron.compute(input_signal, wages)))
