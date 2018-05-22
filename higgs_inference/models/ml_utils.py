################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import numpy as np

from keras.layers import Dense, Dropout
from keras.callbacks import Callback


################################################################################
# Helper functions to build fully connected hidden layers
################################################################################

def stack_layers(layers):

    """ Helper function for Keras models.  """

    def f(x):
        for k in range(len(layers)):
            x = layers[k](x)
        return x

    return f


def build_hidden_layers(n,
                        hidden_layer_size=100,
                        activation='tanh',
                        dropout_prob=0.0):
    """ Helper function for Keras models.  """

    r = []
    for k in range(n):
        if dropout_prob > 0.:
            s = stack_layers([
                Dropout(dropout_prob),
                Dense(hidden_layer_size, activation=activation)
            ])
        else:
            s = stack_layers([Dense(hidden_layer_size, activation=activation)])
        r.append(s)
    return stack_layers(r)


################################################################################
# Callback to save metrics after every few batches
################################################################################

class DetailedHistory(Callback):

    """ Keras Callback that saves metric every n batches rather than just every epoch.  """

    def __init__(self, detailed_history, n_batches_per_entry=100):
        super(DetailedHistory, self).__init__()
        self.detailed_history = detailed_history
        self.buffer = {}
        self.n_batches_in_buffer = 0
        self.n_batches_per_entry = n_batches_per_entry

    def on_train_begin(self, logs={}):
        self.buffer = {}
        self.n_batches_in_buffer = 0

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.buffer.setdefault(k, []).append(v)
        self.n_batches_in_buffer += 1

        if self.n_batches_in_buffer >= self.n_batches_per_entry:
            for k, v in self.buffer.iteritems():
                self.detailed_history.setdefault(k, []).append(np.mean(v))
            self.buffer = {}
            self.n_batches_in_buffer = 0

    def on_epoch_end(self, epoch, logs={}):
        if self.n_batches_in_buffer > 0:
            for k, v in self.buffer.iteritems():
                self.detailed_history.setdefault(k, []).append(np.mean(v))
            self.buffer = {}
            self.n_batches_in_buffer = 0
