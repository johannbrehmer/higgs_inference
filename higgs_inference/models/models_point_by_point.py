################################################################################
# Imports
################################################################################

from keras.models import Model
from keras.layers import Input, Dense, Lambda, Concatenate
from keras import optimizers
import keras.backend as K

from higgs_inference import settings
from higgs_inference.models.ml_utils import build_hidden_layers
from higgs_inference.models.loss_functions import loss_function_carl, loss_function_ratio_regression
from higgs_inference.models.metrics import full_cross_entropy, full_mse_log_r, full_mae_log_r
from higgs_inference.models.metrics import trimmed_cross_entropy, trimmed_mse_log_r

metrics = [full_cross_entropy, trimmed_cross_entropy,
           full_mse_log_r, trimmed_mse_log_r, full_mae_log_r]


################################################################################
# Regression
################################################################################

def make_regressor(n_hidden_layers=3,
                   hidden_layer_size=100,
                   activation='tanh',
                   dropout_prob=0.0):
    # Inputs
    input_layer = Input(shape=(settings.n_features,))

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

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=loss_function_ratio_regression,
                  metrics=metrics,
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
    input_layer = Input(shape=(settings.n_features,))

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
        r_hat_layer = Lambda(lambda x: (1. - x) / x)(s_hat_layer)
        log_r_hat_layer = Lambda(lambda x: K.log(x))(r_hat_layer)

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=loss_function_carl,
                  metrics=metrics,
                  optimizer=optimizers.Adam(clipnorm=1.))

    return model
