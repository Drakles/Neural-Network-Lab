import numpy as np


class Adaline:
    def __init__(self, bias, theta):
        self.bias = bias
        self.theta = theta

    def compute(self, inputs, wages):
        result = np.sum(inputs * wages) + self.bias
        if result > self.theta:
            return 1.0
        return -1.0

    def update_bias(self, error, learning_rate):
        self.bias = self.bias + learning_rate * error

    def total_stimulation(self, input_signal, wages):
        return np.sum(np.append(input_signal * wages, self.bias))

    def __str__(self):
        return "Adaline"
