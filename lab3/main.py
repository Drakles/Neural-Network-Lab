import pickle
import gzip
import numpy as np

from lab3.Network import Network


def load_data():
    f = gzip.open('data/mnist.pkl.gz', 'rb', )
    tr_d, va_d, te_d = pickle.load(f, encoding='latin1')
    f.close()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [reshape_expected_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return training_data, validation_data, test_data


def reshape_expected_result(index):
    # convert to 10 number vector with 1 at appropiate index of a correct response and 0 elsewhere
    result = np.zeros((10, 1))
    result[index] = 1.0
    return result


if __name__ == '__main__':
    training_set, validation_set, test_set = load_data()
    network = Network([28 * 28, 16, 10])

    network.stochastic_gradient_descent(training_set, 30, 5, 0.1, test_set)
