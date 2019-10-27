import random

import numpy as np

from lab1.perceptronBipolar import PerceptronBipolar
from lab1.perceptronUnipolar import PerceptronUnipolar


def error_rate(current_output, expected_result):
    return expected_result - current_output


def update_weights(error, wages, input_signal, learning_rate):
    for i, wage in enumerate(wages):
        wages[i] = wage + learning_rate * error * input_signal[i]


def learn(perceptron, inputs, wages, learning_rate):
    for input_expected_result_pair in inputs:
        input_signal = input_expected_result_pair[0]
        current_output = perceptron.compute(input_signal, wages)

        error = error_rate(current_output, input_expected_result_pair[1])

        update_weights(error, wages, input_signal, learning_rate)


if __name__ == '__main__':
    theta = 0.5
    perceptrons = [PerceptronUnipolar(theta), PerceptronBipolar(theta)]
    # input array with pair of input data as list and expected result from perceptron
    inputs_unipolar_bipolar = [
        np.array([[np.array([0, 0]), 0], [np.array([0, 1]), 1], [np.array([1, 0]), 1], [np.array([1, 1]), 1]]),
        np.array([[np.array([-1, -1]), -1], [np.array([-1, 1]), 1], [np.array([1, -1]), 1], [np.array([1, 1]), 1]])]

    repeat_number = 100
    range_random_wages = [1.0, 0.8, 0.5, 0.2, 0.1]
    learning_rates = [0.5, 0.25, 0.1, 0.01]

    for perceptron, inputs in zip(perceptrons, inputs_unipolar_bipolar):
        for learning_rate in learning_rates:
            for range_random in range_random_wages:
                epoch_sum = 0

                for n in range(repeat_number):
                    wages = np.array([random.uniform(-range_random, range_random) for x in range(len(inputs[0][0]))])

                    epochs = 0
                    are_wages_changed = True

                    while are_wages_changed:
                        old_wages = wages.copy()
                        learn(perceptron, inputs, wages, learning_rate)
                        epochs += 1

                        if np.array_equal(old_wages, wages):
                            are_wages_changed = False

                    epoch_sum += epochs
                    # print("final wages:" + str(wages))
                    # for input_signal,expected_response in inputs:
                    #     print("input:" + str(input_signal) + " expected result: " + str(expected_response) + " response from perceptron: " + str(
                    #         perceptron.compute(input_signal, wages)))

                print(str(perceptron) + " wages range: " + str(-range_random) + "," + str(range_random) + " "
                                                                                                          "learning_rate:  "
                      + str(learning_rate) + " avg epoch number: " + str(epoch_sum / repeat_number))
            print("\n")
        print("\n")
