import numpy as np

from abstractPerceptron import AbstractPerceptron


class PerceptronBipolar(AbstractPerceptron):

    def compute(self, inputs, wages):
        result = np.sum(inputs * wages)
        if result > self.bias:
            return 1.0
        return -1.0

    def __str__(self):
        return "Perceptron bipolar"
