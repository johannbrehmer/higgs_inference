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
    cross_entropies = K.binary_crossentropy(y_true[:, 0], y_pred[:, 0])

    # Trim at bottom and then at top
    cross_entropies, _ = tf.nn.top_k(cross_entropies, n_samples - n_trim)
    cross_entropies, _ = - tf.nn.top_k(- cross_entropies, n_samples - 2 * n_trim)

    # Mean
    return K.mean(cross_entropies, axis=-1)


def trimmed_mse_log_r(y_true, y_pred):

    # Number of samples to be trimmed at each end
    n_samples = y_true.shape[0]
    n_trim = int(round(settings.trim_mean_fraction * n_samples), 0)

    # Calculate cross entropies
    mse = K.square(y_true[:, 1] - y_pred[:, 1])

    # Trim at bottom and then at top
    mse, _ = tf.nn.top_k(mse, n_samples - n_trim)
    mse, _ = - tf.nn.top_k(- mse, n_samples - 2 * n_trim)

    # Mean
    return K.mean(mse, axis=-1)


def trimmed_mse_score(y_true, y_pred):

    # Number of samples to be trimmed at each end
    n_samples = y_true.shape[0]
    n_trim = int(round(settings.trim_mean_fraction * n_samples), 0)

    # Calculate cross entropies
    mse = K.square(y_true[:, 2:settings.n_params + 2] - y_pred[:, 2:settings.n_params + 2])

    # Trim at bottom and then at top
    mse, _ = tf.nn.top_k(mse, n_samples - n_trim)
    mse, _ = - tf.nn.top_k(- mse, n_samples - 2 * n_trim)

    # Mean
    return K.mean(mse, axis=-1)
