import numpy as np

from lab1.abstractPerceptron import AbstractPerceptron


class PerceptronBipolar(AbstractPerceptron):

    def compute(self, inputs, wages):
        result = np.sum(inputs * wages)
        if result > self.theta:
            return 1.0
        return -1.0

    def __str__(self):
        return "Perceptron bipolar"
