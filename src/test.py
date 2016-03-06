# -*- coding: UTF-8 -*-
# from network import Network
# from mnist_loader import load_data_wrapper
#
# train_data, validation_data, test_data = load_data_wrapper()
# net = Network([784, 30, 10])
# net.SGD(train_data, 30, 10, 3.0, test_data=test_data)

import network2
from mnist_loader import load_data_wrapper

train_data, validation_data, test_data = load_data_wrapper()
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(train_data, 30, 10, 0.5, evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
