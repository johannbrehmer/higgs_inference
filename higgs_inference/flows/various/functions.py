import autograd.numpy as np


def s_from_r(r):
    return np.clip(1. / (1. + r), 0., 1.)


def r_from_s(s, epsilon=1.e-6):
    return np.clip((1. - s + epsilon) / (s + epsilon), epsilon, None)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))
