#!/usr/bin/python3
import sys

import numpy as np

from keras import backend as K
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import Sequential
from keras.utils import to_categorical


class NDL:
    """Naive Discriminative Learning network, i.e. parallel perceptrons with linear activation function.
    Parameters
    ----------
    config:         kerasndl configuration object
                    configuration of network structure
    init_weights:   [np.array]
                    list with array of weights to initialize network with to continue learning.
    """

    def __init__(self, config, init_weights, events):
        self.config = config
        self.batches = self.batch_generator(events)

        # structure
        self.network = Sequential()
        self.network.add(Dense(config.num_outputs, input_dim = config.num_inputs, kernel_initializer = config.init, activation = config.activation, use_bias = config.bias))

        # compile
        # self.network.compile(loss = self.summed_squared_error, optimizer = SGD(lr = config.learning_rate))

        # compile
        self.network.compile(loss = "mean_squared_error", optimizer = SGD(lr = config.learning_rate))

        # set initial weights
        # (must be a list containing the numpy array with weights)
        if init_weights:
            self.network.set_weights(init_weights)


    def batch_generator(self, events):
        """Generator object tht yields n-hot encoded training samples.
        """
        for cues, outcomes in events:
            cues = to_categorical(cues, num_classes = self.config.num_inputs)
            cues = np.expand_dims(sum(cues), 0)
            outcomes = to_categorical(outcomes, num_classes = self.config.num_outputs)
            outcomes = np.expand_dims(sum(outcomes), 0)
            yield (cues, outcomes)


    def learn(self, num_events):
        """Learn from the next samples.
        Parameters
        ----------
        num_events: int
                    number of events to learn
        """
        for i in range(num_events):
            cues, outcomes = next(self.batches)
            self.network.train_on_batch(cues, outcomes)
            sys.stderr.write(f"\rLearnt {i} of {num_events}.")
            sys.stderr.flush()


    def get_weights(self, cues, outcomes):
        """Returns weights from and to provided indices of cues and outcomes.
        """
        return self.network.get_weights()[0][cues,:][:,outcomes]


    @staticmethod
    def summed_squared_error(y_true, y_pred):
        return K.sum(K.square(y_pred - y_true))/2

    @property
    def weights(self):
        return self.network.get_weights()[0]

    @property
    def learning_rate(self):
        return self.config.learning_rate