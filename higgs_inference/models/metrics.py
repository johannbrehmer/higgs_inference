################################################################################
# Imports
################################################################################

# import logging

# import tensorflow as tf
from keras import backend as K
from keras import losses

from higgs_inference import settings


################################################################################
# Normal metrics
################################################################################

def full_cross_entropy(y_true, y_pred):
    return losses.binary_crossentropy(y_true[:, 0], y_pred[:, 0])


def full_modified_cross_entropy(y_true, y_pred):
    r = K.exp(K.clip(y_true[:, 1], -settings.log_r_clip_value, settings.log_r_clip_value))
    s = K.clip(1. / (1. + r), 0., 1.)

    return losses.binary_crossentropy(s, y_pred[:, 0])


def full_mse_log_r(y_true, y_pred):
    return losses.mean_squared_error(y_true[:, 1], y_pred[:, 1])


def full_mse_score(y_true, y_pred):
    return losses.mean_squared_error(y_true[:, 2:settings.n_params + 2], y_pred[:, 2:settings.n_params + 2])


def full_mae_log_r(y_true, y_pred):
    return losses.mean_absolute_error(y_true[:, 1], y_pred[:, 1])


def full_mae_score(y_true, y_pred):
    return losses.mean_absolute_error(y_true[:, 2:settings.n_params + 2], y_pred[:, 2:settings.n_params + 2])


################################################################################
# Metrics ignoring top and bottom event of each batch
################################################################################

def trimmed_cross_entropy(y_true, y_pred):
    # Calculate cross entropies
    cross_entropies = losses.binary_crossentropy(y_true[:, 0], y_pred[:, 0])

    cross_entropies = cross_entropies - K.max(cross_entropies) / settings.batch_size - K.min(
        cross_entropies) / settings.batch_size

    return cross_entropies


def trimmed_mse_log_r(y_true, y_pred):
    # Calculate MSE
    mse = losses.mean_squared_error(y_true[:, 1], y_pred[:, 1])

    mse = mse - K.max(mse) / settings.batch_size - K.min(mse) / settings.batch_size

    # Mean
    return mse


def trimmed_mse_score(y_true, y_pred):
    # Calculate cross entropies
    mse = losses.mean_squared_error(y_true[:, 2:settings.n_params + 2], y_pred[:, 2:settings.n_params + 2])

    mse = mse - K.max(mse) / settings.batch_size - K.min(mse) / settings.batch_size

    # Mean
    return mse
