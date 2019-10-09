import numpy as np

from lab1.abstractPerceptron import AbstractPerceptron


class PerceptronUnipolar(AbstractPerceptron):

    def compute(self, inputs, wages):
        result = np.sum(inputs * wages)
        if result > self.theta:
            return 1
        return 0.0

    def __str__(self):
        return "Perceptron unipolar"
