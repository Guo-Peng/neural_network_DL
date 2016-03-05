# -*- coding: UTF-8 -*-
import numpy as np
import random


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        """return the output of the network as a is the input"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, train_data, epoch, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(train_data)
        for x in xrange(epoch):
            random.shuffle(train_data)
            mini_batches = [train_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mimi_batch in mini_batches:
                self.update_mini_batch(mimi_batch, eta)
            if test_data:
                print 'Epoch {0}: {1} / {2}'.format(x + 1, self.evaluate(test_data), n_test)
            else:
                print 'Epoch {0} complete'.format(x + 1)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            deta_nabla_b, deta_nabla_w = self.back_prop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, deta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, deta_nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]

    def back_prop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            zs.append(np.dot(w, activations[-1]) + b)
            activations.append(sigmoid(zs[-1]))

        deta = [(activations[-1] - y) * sigmoid_prime(zs[-1])]
        nabla_b[-1] = deta[-1]
        nabla_w[-1] = np.dot(deta[-1], activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            deta.append(np.dot(self.weights[-l + 1].transpose(), deta[-1]) * sigmoid_prime(zs[-l]))
            nabla_b[-l] = deta[-1]
            nabla_w[-l] = np.dot(deta[-1], activations[-l - 1].transpose())
        return nabla_b, nabla_w

    # def backprop(self, x, y):
    #     """Return a tuple ``(nabla_b, nabla_w)`` representing the
    #     gradient for the cost function C_x.  ``nabla_b`` and
    #     ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    #     to ``self.biases`` and ``self.weights``."""
    #     nabla_b = [np.zeros(b.shape) for b in self.biases]
    #     nabla_w = [np.zeros(w.shape) for w in self.weights]
    #     # feedforward
    #     activation = x
    #     activations = [x]  # list to store all the activations, layer by layer
    #     zs = []  # list to store all the z vectors, layer by layer
    #     for b, w in zip(self.biases, self.weights):
    #         z = np.dot(w, activation) + b
    #         zs.append(z)
    #         activation = sigmoid(z)
    #         activations.append(activation)
    #     # backward pass
    #     delta = self.cost_derivative(activations[-1], y) * \
    #             sigmoid_prime(zs[-1])
    #     nabla_b[-1] = delta
    #     nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    #     # Note that the variable l in the loop below is used a little
    #     # differently to the notation in Chapter 2 of the book.  Here,
    #     # l = 1 means the last layer of neurons, l = 2 is the
    #     # second-last layer, and so on.  It's a renumbering of the
    #     # scheme in the book, used here to take advantage of the fact
    #     # that Python can use negative indices in lists.
    #     for l in xrange(2, self.num_layers):
    #         z = zs[-l]
    #         sp = sigmoid_prime(z)
    #         delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
    #         nabla_b[-l] = delta
    #         nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
    #     return nabla_b, nabla_w

    def evaluate(self, test_data):

        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return output_activations - y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


if __name__ == '__main__':
    l = np.array([1, 2, 3, 4])
    print
