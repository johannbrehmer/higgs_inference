################################################################################
# Imports
################################################################################

from keras import losses

from higgs_inference import settings


################################################################################
# Loss functions
################################################################################

def loss_function_carl(y_true, y_pred):
    return losses.binary_crossentropy(y_true[:, 0], y_pred[:, 0])


def loss_function_regression(y_true, y_pred):
    return losses.mean_squared_error(y_true[:, 0], y_pred[:, 0])


def loss_function_score(y_true, y_pred):
    return losses.mean_squared_error(y_true[:, 1:settings.n_params + 1], y_pred[:, 1:settings.n_params + 1])


def loss_function_combinedregression(y_true, y_pred, alpha=0.005):
    return loss_function_regression(y_true, y_pred) + alpha * loss_function_score(y_true, y_pred)


def loss_function_combined(y_true, y_pred, alpha=0.1):
    return loss_function_carl(y_true, y_pred) + alpha * loss_function_score(y_true, y_pred)


def loss_function_score_regression(y_true, y_pred):
    return losses.mean_squared_error(y_true[:, :], y_pred[:, :])
