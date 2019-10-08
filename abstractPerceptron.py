class AbstractPerceptron:
    def __init__(self, bias):
        self.bias = bias

    def compute(self, inputs, wages):
        raise NotImplementedError("The method not implemented")

    def __str__(self):
        raise NotImplementedError("The method not implemented")
