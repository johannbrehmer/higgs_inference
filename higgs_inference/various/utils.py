from __future__ import absolute_import, division, print_function

from keras.layers import Dense, Dropout


def s_from_r(r):
    return 1./(1. + r)


def r_from_s(s, epsilon=1.e-3):
    return (1. - s + epsilon)/(s + epsilon)


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
