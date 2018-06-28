################################################################################
# Imports
################################################################################

from keras.models import Model
from keras.layers import Input, Dense, Lambda, Concatenate, ActivityRegularization
from keras import optimizers
import keras.backend as K

from higgs_inference import settings
from higgs_inference.models.ml_utils import build_hidden_layers
from higgs_inference.models.loss_functions import loss_function_carl, loss_function_combined, \
    loss_function_combinedregression, loss_function_ratio_regression, loss_function_score, \
    loss_function_modified_crossentropy, loss_function_combined_modified_crossentropy
from higgs_inference.models.metrics import full_cross_entropy, full_mse_log_r, full_mse_score
from higgs_inference.models.metrics import full_mae_log_r, full_mae_score
from higgs_inference.models.metrics import trimmed_cross_entropy, trimmed_mse_log_r, trimmed_mse_score
from higgs_inference.models.metrics import full_modified_cross_entropy

metrics = [full_cross_entropy, trimmed_cross_entropy,
           full_modified_cross_entropy,
           full_mse_log_r, trimmed_mse_log_r,
           full_mae_log_r,
           full_mse_score, trimmed_mse_score,
           full_mae_score]


################################################################################
# ROLR
################################################################################

def make_regressor(n_hidden_layers=3,
                   hidden_layer_size=100,
                   activation='tanh',
                   dropout_prob=0.0,
                   learning_rate=1.e-3):
    """
    Builds a Keras model for the parameterized version of the ROLR technique.

    :param n_hidden_layers: Number of hidden layers.
    :param hidden_layer_size: Number of units in each hidden layer.
    :param activation: Activation function.
    :param dropout_prob: Dropout probability.
    :param learning_rate: Initial learning rate.
    :return: Keras model.
    """

    # Inputs
    input_layer = Input(shape=(settings.n_thetas_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                            hidden_layer_size=hidden_layer_size,
                                            activation=activation,
                                            dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)
    log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)

    # Translate to s
    r_hat_layer = Lambda(lambda x: K.exp(x))(log_r_hat_layer)
    s_hat_layer = Lambda(lambda x: 1. / (1. + x))(r_hat_layer)

    # Score
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0],
                            output_shape=(settings.n_thetas_features,))([log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer, score_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=loss_function_ratio_regression,
                  metrics=metrics,
                  optimizer=optimizers.Adam(lr=learning_rate, clipnorm=10.))

    return model


################################################################################
# RASCAL
################################################################################

def make_combined_regressor(n_hidden_layers=3,
                            hidden_layer_size=100,
                            activation='tanh',
                            dropout_prob=0.0,
                            alpha=100.,
                            learning_rate=1.e-3):
    """
    Builds a Keras model for the parameterized version of the RASCAL technique.

    :param n_hidden_layers: Number of hidden layers.
    :param hidden_layer_size: Number of units in each hidden layer.
    :param activation: Activation function.
    :param dropout_prob: Dropout probability.
    :param alpha: RASCAL hyperparameter that weights the score term in the loss.
    :param learning_rate: Initial learning rate.
    :return: Keras model.
    """

    # Inputs
    input_layer = Input(shape=(settings.n_thetas_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                            hidden_layer_size=hidden_layer_size,
                                            activation=activation,
                                            dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)
    log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)

    # Translate to s
    r_hat_layer = Lambda(lambda x: K.exp(x))(log_r_hat_layer)
    s_hat_layer = Lambda(lambda x: 1. / (1. + x))(r_hat_layer)

    # Score
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0],
                            output_shape=(settings.n_thetas_features,))([log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer, score_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=lambda x, y: loss_function_combinedregression(x, y, alpha=alpha),
                  metrics=metrics,
                  optimizer=optimizers.Adam(lr=learning_rate, clipnorm=10.))

    return model


################################################################################
# CARL
################################################################################

def make_classifier_carl(n_hidden_layers=3,
                         hidden_layer_size=100,
                         activation='tanh',
                         dropout_prob=0.0,
                         learn_log_r=False,
                         learning_rate=1.e-3):
    """
    Builds a Keras model for the parameterized version of the CARL technique.

    :param n_hidden_layers: Number of hidden layers.
    :param hidden_layer_size: Number of units in each hidden layer.
    :param activation: Activation function.
    :param dropout_prob: Dropout probability.
    :param learn_log_r: Fully connected network represents log r rather than s.
    :param learning_rate: Initial learning rate.
    :return: Keras model.
    """

    # Inputs
    input_layer = Input(shape=(settings.n_thetas_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                            hidden_layer_size=hidden_layer_size,
                                            activation=activation,
                                            dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)

    if learn_log_r:
        log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)
        r_hat_layer = Lambda(lambda x: K.exp(x))(log_r_hat_layer)
        s_hat_layer = Lambda(lambda x: 1. / (1. + x))(r_hat_layer)

    else:
        s_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)
        r_hat_layer = Lambda(lambda x: (1. - x) / x)(s_hat_layer)
        log_r_hat_layer = Lambda(lambda x: K.log(x))(r_hat_layer)

    # Score
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0],
                            output_shape=(settings.n_thetas_features,))([log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer, score_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=loss_function_carl,
                  metrics=metrics,
                  optimizer=optimizers.Adam(lr=learning_rate, clipnorm=1.))

    return model


################################################################################
# Score only models
################################################################################

def make_classifier_score(n_hidden_layers=3,
                          hidden_layer_size=100,
                          activation='tanh',
                          dropout_prob=0.0,
                          learn_log_r=False,
                          l2_regularization=0.001,
                          learning_rate=1.e-3):
    """
    Builds a Keras model for the parameterized version of an unnamed technique that only uses the score.

    :param n_hidden_layers: Number of hidden layers.
    :param hidden_layer_size: Number of units in each hidden layer.
    :param activation: Activation function.
    :param dropout_prob: Dropout probability.
    :param learn_log_r: Fully connected network represents log r rather than s.
    :param l2_regularization: Regularization parameter.
    :param learning_rate: Initial learning rate.
    :return: Keras model.
    """

    # Inputs
    input_layer = Input(shape=(settings.n_thetas_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                            hidden_layer_size=hidden_layer_size,
                                            activation=activation,
                                            dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)

    if learn_log_r:
        log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)
        r_hat_layer = Lambda(lambda x: K.exp(x))(log_r_hat_layer)
        s_hat_layer = Lambda(lambda x: 1. / (1. + x))(r_hat_layer)

    else:
        s_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)
        r_hat_layer = Lambda(lambda x: (1. - x) / x)(s_hat_layer)
        log_r_hat_layer = Lambda(lambda x: K.log(x))(r_hat_layer)

    # Score
    regularizer_layer = ActivityRegularization(l1=0., l2=l2_regularization)(log_r_hat_layer)
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0],
                            output_shape=(settings.n_thetas_features,))([regularizer_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer, score_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=loss_function_score,
                  metrics=metrics,
                  optimizer=optimizers.Adam(lr=learning_rate, clipnorm=1.))

    return model


################################################################################
# CASCAL
################################################################################

def make_classifier_combined(n_hidden_layers=3,
                             hidden_layer_size=100,
                             activation='tanh',
                             dropout_prob=0.0,
                             learn_log_r=False,
                             alpha=5.,
                             learning_rate=1.e-3):
    """
    Builds a Keras model for the parameterized version of the CASCAL technique.

    :param n_hidden_layers: Number of hidden layers.
    :param hidden_layer_size: Number of units in each hidden layer.
    :param activation: Activation function.
    :param dropout_prob: Dropout probability.
    :param learn_log_r: Fully connected network represents log r rather than s.
    :param alpha: RASCAL hyperparameter that weights the score term in the loss.
    :param learning_rate: Initial learning rate.
    :return: Keras model.
    """

    # Inputs
    input_layer = Input(shape=(settings.n_thetas_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                            hidden_layer_size=hidden_layer_size,
                                            activation=activation,
                                            dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)

    if learn_log_r:
        log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)
        r_hat_layer = Lambda(lambda x: K.exp(x))(log_r_hat_layer)
        s_hat_layer = Lambda(lambda x: 1. / (1. + x))(r_hat_layer)

    else:
        s_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)
        r_hat_layer = Lambda(lambda x: (1. - x) / x)(s_hat_layer)
        log_r_hat_layer = Lambda(lambda x: K.log(x))(r_hat_layer)

    # Score
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0],
                            output_shape=(settings.n_thetas_features,))([log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer, score_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=lambda x, y: loss_function_combined(x, y, alpha=alpha),
                  metrics=metrics,
                  optimizer=optimizers.Adam(lr=learning_rate, clipnorm=1.))

    return model


################################################################################
# Modified cross entropy
################################################################################

def make_modified_xe_model(n_hidden_layers=3,
                           hidden_layer_size=100,
                           activation='tanh',
                           dropout_prob=0.0,
                           learning_rate=1.e-3):
    """
    Builds a Keras model for the parameterized version of the modified XE technique.

    :param n_hidden_layers: Number of hidden layers.
    :param hidden_layer_size: Number of units in each hidden layer.
    :param activation: Activation function.
    :param dropout_prob: Dropout probability.
    :param learning_rate: Initial learning rate.
    :return: Keras model.
    """

    # Inputs
    input_layer = Input(shape=(settings.n_thetas_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                            hidden_layer_size=hidden_layer_size,
                                            activation=activation,
                                            dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)
    log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)

    # Translate to s
    r_hat_layer = Lambda(lambda x: K.exp(x))(log_r_hat_layer)
    s_hat_layer = Lambda(lambda x: 1. / (1. + x))(r_hat_layer)

    # Score
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0],
                            output_shape=(settings.n_thetas_features,))([log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer, score_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=lambda x, y: loss_function_modified_crossentropy(x, y),
                  metrics=metrics,
                  optimizer=optimizers.Adam(lr=learning_rate, clipnorm=10.))

    return model


################################################################################
# Modified cross entropy + score
################################################################################

def make_combined_modified_xe_model(n_hidden_layers=3,
                                    hidden_layer_size=100,
                                    activation='tanh',
                                    dropout_prob=0.0,
                                    alpha=100.,
                                    learning_rate=1.e-3):
    """
    Builds a Keras model for the modified XE + score technique.

    :param n_hidden_layers: Number of hidden layers.
    :param hidden_layer_size: Number of units in each hidden layer.
    :param activation: Activation function.
    :param dropout_prob: Dropout probability.
    :param alpha: RASCAL hyperparameter that weights the score term in the loss.
    :param learning_rate: Initial learning rate.
    :return: Keras model.
    """

    # Inputs
    input_layer = Input(shape=(settings.n_thetas_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                            hidden_layer_size=hidden_layer_size,
                                            activation=activation,
                                            dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)
    log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)

    # Translate to s
    r_hat_layer = Lambda(lambda x: K.exp(x))(log_r_hat_layer)
    s_hat_layer = Lambda(lambda x: 1. / (1. + x))(r_hat_layer)

    # Score
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0],
                            output_shape=(settings.n_thetas_features,))([log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer, score_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=lambda x, y: loss_function_combined_modified_crossentropy(x, y, alpha=alpha),
                  metrics=metrics,
                  optimizer=optimizers.Adam(lr=learning_rate, clipnorm=10.))

    return model
