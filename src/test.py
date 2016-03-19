# -*- coding: UTF-8 -*-
# from network import Network
# from mnist_loader import load_data_wrapper
#
# train_data, validation_data, test_data = load_data_wrapper()
# net = Network([784, 30, 10])
# net.SGD(train_data, 30, 10, 3.0, test_data=test_data)

# import network2
# from mnist_loader import load_data_wrapper
#
# train_data, validation_data, test_data = load_data_wrapper()
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(train_data, 30, 10, 0.5, evaluation_data=test_data,
#         monitor_evaluation_accuracy=True)

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10
net = Network([
    FullyConnectedLayer(n_in=784, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

net.SGD(training_data, 60, mini_batch_size, 0.1,
        validation_data, test_data)
