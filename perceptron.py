import numpy as np


class Perceptron:
    def __init__(self, bias):
        self.bias = bias

    def compute(self, inputs, wages):
        result = np.sum(inputs * wages)
        if result > self.bias:
            return result
        return 0.0
