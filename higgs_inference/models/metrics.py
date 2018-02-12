################################################################################
# Imports
################################################################################

import keras.backend as K
import tensorflow as tf

from higgs_inference import settings


################################################################################
# Metrics ignoring 5% outliers
################################################################################

def trimmed_cross_entropy(y_true, y_pred):

    # Number of samples to be trimmed at each end
    n_samples = y_true.shape[0]
    n_trim = int(round(settings.trim_mean_fraction * n_samples), 0)

    # Calculate cross entropies
    cross_entropies = K.mean(K.binary_crossentropy(y_true[:, 0], y_pred[:, 0]), axis=-1)

    # Trim at bottom and then at top
    _, top_indices = tf.nn.top_k(cross_entropies, n_trim)
    _, bottom_indices = tf.nn.top_k(- cross_entropies, n_trim)
    cross_entropies[top_indices] = 0.
    cross_entropies[bottom_indices] = 0.

    # Mean
    return cross_entropies


def trimmed_mse_log_r(y_true, y_pred):

    # Number of samples to be trimmed at each end
    n_samples = y_true.shape[0]
    n_trim = int(round(settings.trim_mean_fraction * n_samples), 0)

    # Calculate cross entropies
    mse = K.mean(K.square(y_true[:, 1] - y_pred[:, 1]), axis=-1)

    # Set loss for top and bottom indices to zero
    _, top_indices = tf.nn.top_k(mse, n_trim)
    _, bottom_indices = tf.nn.top_k(- mse, n_trim)
    mse[top_indices] = 0.
    mse[bottom_indices] = 0.

    # Mean
    return mse


def trimmed_mse_score(y_true, y_pred):

    # Number of samples to be trimmed at each end
    n_samples = y_true.shape[0]
    n_trim = int(round(settings.trim_mean_fraction * n_samples), 0)

    # Calculate cross entropies
    mse = K.sum(K.square(y_true[:, 2:settings.n_params + 2] - y_pred[:, 2:settings.n_params + 2]), axis=-1)

    # Set loss for top and bottom indices to zero
    _, top_indices = tf.nn.top_k(mse, n_trim)
    _, bottom_indices = tf.nn.top_k(- mse, n_trim)
    mse[top_indices] = 0.
    mse[bottom_indices] = 0.

    # Mean
    return mse
