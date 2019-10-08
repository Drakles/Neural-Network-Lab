import numpy as np

from abstractPerceptron import AbstractPerceptron


class PerceptronUnipolar(AbstractPerceptron):

    def compute(self, inputs, wages):
        result = np.sum(inputs * wages)
        if result > self.bias:
            return 1
        return 0.0

    def __str__(self):
        return "Perceptron unipolar"
