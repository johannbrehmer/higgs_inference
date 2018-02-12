################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

from keras.layers import Dense, Dropout
from keras.callbacks import Callback


################################################################################
# Helper functions to build fully connected hidden layers
################################################################################

def stack_layers(layers):
    def f(x):
        for k in range(len(layers)):
            x = layers[k](x)
        return x

    return f


def build_hidden_layers(n,
                        hidden_layer_size=100,
                        activation='tanh',
                        dropout_prob=0.0):
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
# Callback to save metrics after every batch
################################################################################

class DetailedHistory(Callback):

    def __init__(self, detailed_history):
        super(DetailedHistory).__init__()
        self.detailed_history = detailed_history

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.detailed_history.setdefault(k, []).append(v)
