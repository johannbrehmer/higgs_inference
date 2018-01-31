################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import numpy as np

from keras.layers import Dense, Dropout

from higgs_inference import settings


################################################################################
# Translation between optimal decision function s and likelihood ratio r
################################################################################

def s_from_r(r):
    return np.clip(1. / (1. + r), 0., 1.)


def r_from_s(s, epsilon=1.e-3):
    return np.clip((1. - s + epsilon) / (s + epsilon), epsilon, None)


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
# Formatting numbers as output strings
################################################################################

def format_number(number,
                  precision=2,
                  trailing_zeros=True,
                  fix_minus_zero=True,
                  latex_math_mode=False,
                  emphasize=False):
    if precision == 0:
        temp = str(int(round(number, precision)))
    elif trailing_zeros:
        temp = ('{:.' + str(precision) + 'f}').format(round(number, precision))
    else:
        temp = str(round(number, precision))
    if fix_minus_zero and len(temp) > 0:
        if temp[0] == '-' and float(temp) == 0.:
            temp = temp[1:]
    if latex_math_mode:
        if emphasize:
            temp = '$\mathbf{' + temp + '}$'
        else:
            temp = '$' + temp + '$'
    elif emphasize:
        temp = r'\emph{' + temp + r'}'
    return temp



################################################################################
# Decide if two a given Neyman toy experiment should be evaluated at a given theta
################################################################################

def decide_toy_evaluation(theta_hypothesis, theta_evaluation, distance_threshold=0.4):

    if theta_evaluation == theta_hypothesis:
        return True

    if theta_evaluation in settings.pbp_training_thetas:
        return True

    delta_theta = np.linalg.norm(settings.thetas[theta_hypothesis] - settings.thetas[theta_evaluation])

    return (delta_theta <= distance_threshold)