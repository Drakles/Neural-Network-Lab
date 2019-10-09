class AbstractPerceptron:
    def __init__(self, theta):
        self.theta = theta

    def compute(self, inputs, wages):
        raise NotImplementedError("The method not implemented")

    def __str__(self):
        raise NotImplementedError("The method not implemented")
