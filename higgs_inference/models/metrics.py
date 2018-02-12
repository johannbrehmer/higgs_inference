################################################################################
# Imports
################################################################################

import tensorflow as tf

from higgs_inference import settings
from higgs_inference.models.loss_functions import loss_function_carl, loss_function_ratio_regression, \
    loss_function_score


################################################################################
# Metrics ignoring top and bottom 5% per batch
################################################################################

def trimmed_cross_entropy(y_true, y_pred):
    # Calculate cross entropies
    cross_entropies = loss_function_carl(y_true, y_pred)

    # Trim at bottom and then at top
    _, top_indices = tf.nn.top_k(cross_entropies, settings.trim_mean_absolute)
    _, bottom_indices = tf.nn.top_k(- cross_entropies, settings.trim_mean_absolute)

    cross_entropies[top_indices] = 0.
    cross_entropies[bottom_indices] = 0.

    # Mean
    return cross_entropies


def trimmed_mse_log_r(y_true, y_pred):
    # Calculate MSE
    mse = loss_function_ratio_regression(y_true, y_pred)

    # Set loss for top and bottom indices to zero
    _, top_indices = tf.nn.top_k(mse, settings.trim_mean_absolute)
    _, bottom_indices = tf.nn.top_k(- mse, settings.trim_mean_absolute)
    mse[top_indices] = 0.
    mse[bottom_indices] = 0.

    # Mean
    return mse


def trimmed_mse_score(y_true, y_pred):
    # Calculate cross entropies
    mse = loss_function_score(y_true, y_pred)

    # Set loss for top and bottom indices to zero
    _, top_indices = tf.nn.top_k(mse, settings.trim_mean_absolute)
    _, bottom_indices = tf.nn.top_k(- mse, settings.trim_mean_absolute)
    mse[top_indices] = 0.
    mse[bottom_indices] = 0.

    # Mean
    return mse
