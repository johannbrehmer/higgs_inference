################################################################################
# Imports
################################################################################

from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras import losses, optimizers
import keras.backend as K

from ..various.utils import build_hidden_layers

################################################################################
# Parameters
################################################################################

n_features = 42


################################################################################
# Loss functions
################################################################################

def loss_function_carl(y_true, y_pred):
    return losses.binary_crossentropy(y_true[:, 0], y_pred[:, 0])


def loss_function_regression(y_true, y_pred):
    return losses.mean_squared_error(y_true[:, 0], y_pred[:, 0])


################################################################################
# Regression
################################################################################

def make_regressor(n_hidden_layers=3,
                   hidden_layer_size=100,
                   activation='tanh',
                   dropout_prob=0.0):
    # Inputs
    input_layer = Input(shape=(n_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                            hidden_layer_size=hidden_layer_size,
                                            activation=activation,
                                            dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)
    log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)

    model = Model(inputs=[input_layer], outputs=[log_r_hat_layer])

    # Compile model
    model.compile(loss=loss_function_regression,
                  optimizer=optimizers.Adam(clipnorm=1.))

    return model


################################################################################
# carl
################################################################################

def make_classifier(n_hidden_layers=3,
                    hidden_layer_size=100,
                    activation='tanh',
                    dropout_prob=0.0,
                    learn_log_r=False):
    # Inputs
    input_layer = Input(shape=(n_features,))

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
        s_hat_layer = Lambda(lambda x: 1. / (1. + r_hat_layer))(log_r_hat_layer)

    else:
        s_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)

    # Combine outputs
    model = Model(inputs=[input_layer], outputs=[s_hat_layer])

    # Compile model
    model.compile(loss=loss_function_carl,
                  optimizer=optimizers.Adam(clipnorm=1.))

    return model
