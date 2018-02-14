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

    # Trim at bottom and then at top
    # if cross_entropies.shape is not None:
    #    if cross_entropies.shape.ndims >= 1:
    #        if cross_entropies.shape[0].value is not None:
    # _, top_indices = tf.nn.top_k(cross_entropies, 2)
    # _, bottom_indices = tf.nn.top_k(- cross_entropies, 2)
    # cross_entropies[top_indices] = 0.
    # cross_entropies[bottom_indices] = 0.

    #            logging.debug('CE values: %s %s %s', cross_entropies, top_indices, bottom_indices)

    #        logging.debug('CE shapes: %s %s %s', cross_entropies.shape, cross_entropies.shape.ndims,
    #                      cross_entropies.shape[0].value)

    #    else:
    #        logging.debug('CE shapes: %s %s', cross_entropies.shape, cross_entropies.shape.ndims)
    # else:
    #    logging.debug('CE shapes: %s', cross_entropies.shape)

    return cross_entropies


def trimmed_mse_log_r(y_true, y_pred):
    # Calculate MSE
    mse = losses.mean_squared_error(y_true[:, 1], y_pred[:, 1])

    mse = mse - K.max(mse) / settings.batch_size - K.min(mse) / settings.batch_size

    # # Set loss for top and bottom indices to zero
    # if mse.shape is not None:
    #     if mse.shape.ndims >= 1:
    #         if mse.shape[0].value is not None:
    #             _, top_indices = tf.nn.top_k(mse, settings.trim_mean_absolute)
    #             _, bottom_indices = tf.nn.top_k(- mse, settings.trim_mean_absolute)
    #             mse[top_indices] = 0.
    #             mse[bottom_indices] = 0.

    # Mean
    return mse


def trimmed_mse_score(y_true, y_pred):
    # Calculate cross entropies
    mse = losses.mean_squared_error(y_true[:, 2:settings.n_params + 2], y_pred[:, 2:settings.n_params + 2])

    mse = mse - K.max(mse) / settings.batch_size - K.min(mse) / settings.batch_size

    # # Set loss for top and bottom indices to zero
    # if mse.shape is not None:
    #     if mse.shape.ndims >= 1:
    #         if mse.shape[0].value is not None:
    #             _, top_indices = tf.nn.top_k(mse, settings.trim_mean_absolute)
    #             _, bottom_indices = tf.nn.top_k(- mse, settings.trim_mean_absolute)
    #             mse[top_indices] = 0.
    #             mse[bottom_indices] = 0.

    # Mean
    return mse
