import pickle
import gzip
import numpy as np
import copy
import math

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


# convert to 10 number vector with 1 at appropiate index of a correct response and 0 elsewhere
def reshape_expected_result(index):
    # initialize with 0
    result = np.zeros((10, 1))
    result[index] = 1.0
    return result


if __name__ == '__main__':
    (training_set, validation_set, test_set) = load_data()
    data = (training_set, validation_set, test_set)

    mini_batches_sizes = [1, 2, 5, 10, 20]
    #zmienic sobie na zapamietywanie jaka efektywosc po danej epoce

    #dodac bufor do przeuczenia dla 3 np epok, jak polepszylo na testowym a pogorszylo na walidaycjnym
    epoch_numbers = [3, 5, 10, 20]
    learning_rates = [0.001, 0.01, 0.1, 1, 2]
    layers_configurations = [[28 * 28, 4, 10], [28 * 28, 16, 10], [28 * 28, 20, 10], [28 * 28, 20, 16, 10]]

    # mini_batches_sizes = [5]
    # epoch_numbers = [5]
    # learning_rates = [0.01]

    repeat_number = 3

    for learning_rate in learning_rates:
        for epoch_number in epoch_numbers:
            for mini_batch_size in mini_batches_sizes:
                result = []
                for i in range(repeat_number):
                    (tr_set, val_set, tes_set) = copy.deepcopy(data)

                    network = Network([28 * 28, 16, 10])
                    network.stochastic_gradient_descent(tr_set, epoch_number, mini_batch_size, learning_rate, None)

                    test_list = list(tes_set)
                    test_list_length = len(test_list)
                    result.append(network.evaluate(test_list) / test_list_length*100.0)

                print("avg efficeincy: " + str(sum(result) / (len(result))))
                print("mini_batch_size: " + str(mini_batch_size))
                print("epochs number: " + str(epoch_number))
                print("learning rate: " + str(learning_rate))
                print('')
