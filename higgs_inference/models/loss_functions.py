################################################################################
# Imports
################################################################################

from keras import losses
from keras import backend as K

from higgs_inference import settings


################################################################################
# Loss functions
################################################################################

def loss_function_carl(y_true, y_pred):

    """
    Cross entropy loss function.

    :param y_true: ndarray with shape (n_samples, 2 + n_parameters) where the first column are the y_i (0 if from
                   numerator, 1 if from denominator), the second column are the true log r(x, z | theta0, theta1), and the
                   remaining columns are the true t(x, z | theta0).
    :param y_pred: ndarray with shape (n_samples, 2 + n_parameters) where the first column is the classifier decision
                   function \hat{s}_i, the second column are the estimated log \hat{r}(x | theta0, theta1), and the
                   remaining columns are the estimated \hat{t}(x | theta0).
    :return: Binary cross entropy.
    """

    return losses.binary_crossentropy(y_true[:, 0], y_pred[:, 0])


def loss_function_ratio_regression(y_true, y_pred):

    """
    ROLR loss function.

    :param y_true: ndarray with shape (n_samples, 2 + n_parameters) where the first column are the y_i (0 if from
                   numerator, 1 if from denominator), the second column are the true log r(x, z | theta0, theta1), and the
                   remaining columns are the true t(x, z | theta0).
    :param y_pred: ndarray with shape (n_samples, 2 + n_parameters) where the first column is the classifier decision
                   function \hat{s}_i, the second column are the estimated log \hat{r}(x | theta0, theta1), and the
                   remaining columns are the estimated \hat{t}(x | theta0).
    :return: Squared error on r (for the y=1 samples) plus squared error on 1/r (for the y=0 samples).
    """

    r_loss = losses.mean_squared_error(K.exp(K.clip(y_true[:, 1],
                                                    -settings.log_r_clip_value, settings.log_r_clip_value)),
                                       K.exp(K.clip(y_pred[:, 1],
                                                    -settings.log_r_clip_value, settings.log_r_clip_value)))
    inverse_r_loss = losses.mean_squared_error(K.exp(-K.clip(y_true[:, 1],
                                                             -settings.log_r_clip_value,
                                                             settings.log_r_clip_value)),
                                               K.exp(-K.clip(y_pred[:, 1],
                                                             -settings.log_r_clip_value,
                                                             settings.log_r_clip_value)))

    return y_true[:, 0] * r_loss + (1. - y_true[:, 0]) * inverse_r_loss


def loss_function_score(y_true, y_pred):

    """
    Score term in the loss function for CASCAL and RASCAL

    :param y_true: ndarray with shape (n_samples, 2 + n_parameters) where the first column are the y_i (0 if from
                   numerator, 1 if from denominator), the second column are the true log r(x, z | theta0, theta1), and the
                   remaining columns are the true t(x, z | theta0).
    :param y_pred: ndarray with shape (n_samples, 2 + n_parameters) where the first column is the classifier decision
                   function \hat{s}_i, the second column are the estimated log \hat{r}(x | theta0, theta1), and the
                   remaining columns are the estimated \hat{t}(x | theta0).
    :return: Squared error on t (for the y=0 samples).
    """

    score_loss = losses.mean_squared_error(y_true[:, 2:settings.n_params + 2], y_pred[:, 2:settings.n_params + 2])

    return (1. - y_true[:, 0]) * score_loss


def loss_function_combined(y_true, y_pred, alpha=5.):

    """
    CASCAL loss function.

    :param y_true: ndarray with shape (n_samples, 2 + n_parameters) where the first column are the y_i (0 if from
                   numerator, 1 if from denominator), the second column are the true log r(x, z | theta0, theta1), and the
                   remaining columns are the true t(x, z | theta0).
    :param y_pred: ndarray with shape (n_samples, 2 + n_parameters) where the first column is the classifier decision
                   function \hat{s}_i, the second column are the estimated log \hat{r}(x | theta0, theta1), and the
                   remaining columns are the estimated \hat{t}(x | theta0).
    :param alpha: Hyperparameter that weights the score term in the loss function.
    :return: Combined CASCAL loss.
    """

    return loss_function_carl(y_true, y_pred) + alpha * loss_function_score(y_true, y_pred)


def loss_function_combinedregression(y_true, y_pred, alpha=100.):

    """
    RASCAL loss function.

    :param y_true: ndarray with shape (n_samples, 2 + n_parameters) where the first column are the y_i (0 if from
                   numerator, 1 if from denominator), the second column are the true log r(x, z | theta0, theta1), and the
                   remaining columns are the true t(x, z | theta0).
    :param y_pred: ndarray with shape (n_samples, 2 + n_parameters) where the first column is the classifier decision
                   function \hat{s}_i, the second column are the estimated log \hat{r}(x | theta0, theta1), and the
                   remaining columns are the estimated \hat{t}(x | theta0).
    :param alpha: Hyperparameter that weights the score term in the loss function.
    :return: Combined RASCAL loss.
    """

    return loss_function_ratio_regression(y_true, y_pred) + alpha * loss_function_score(y_true, y_pred)


def loss_function_score_regression(y_true, y_pred):

    """
    SALLY and SALLINO loss function.

    :param y_true: ndarray with shape (n_samples, n_parameters) with the true t(x, z | theta_score).
    :param y_pred: ndarray with shape (n_samples, n_parameters) with the estimated \hat{t}(x | theta_score).
    :return: Squared error on t.
    """

    return losses.mean_squared_error(y_true[:, :], y_pred[:, :])


def loss_function_modified_crossentropy(y_true, y_pred):

    """
    Cross entropy loss function, where instead of y = 0 or 1 we use s(x,z)

    :param y_true: ndarray with shape (n_samples, 2 + n_parameters) where the first column are the y_i (0 if from
                   numerator, 1 if from denominator), the second column are the true log r(x, z | theta0, theta1), and the
                   remaining columns are the true t(x, z | theta0).
    :param y_pred: ndarray with shape (n_samples, 2 + n_parameters) where the first column is the classifier decision
                   function \hat{s}_i, the second column are the estimated log \hat{r}(x | theta0, theta1), and the
                   remaining columns are the estimated \hat{t}(x | theta0).
    :return: Binary cross entropy
    """

    r = K.exp(K.clip(y_true[:, 1], -settings.log_r_clip_value, settings.log_r_clip_value))
    s = K.clip(1. / (1. + r), 0., 1.)

    return losses.binary_crossentropy(s, y_pred[:, 0])


def loss_function_combined_modified_crossentropy(y_true, y_pred, alpha=100.):

    """
    Modified cross entropy plus derived score.

    :param y_true: ndarray with shape (n_samples, 2 + n_parameters) where the first column are the y_i (0 if from
                   numerator, 1 if from denominator), the second column are the true log r(x, z | theta0, theta1), and the
                   remaining columns are the true t(x, z | theta0).
    :param y_pred: ndarray with shape (n_samples, 2 + n_parameters) where the first column is the classifier decision
                   function \hat{s}_i, the second column are the estimated log \hat{r}(x | theta0, theta1), and the
                   remaining columns are the estimated \hat{t}(x | theta0).
    :param alpha: Hyperparameter that weights the score term in the loss function.
    :return: Combined RASCAL loss.
    """

    return loss_function_modified_crossentropy(y_true, y_pred) + alpha * loss_function_score(y_true, y_pred)