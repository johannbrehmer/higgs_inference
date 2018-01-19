################################################################################
# Imports
################################################################################

import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Concatenate, Multiply, Reshape, ActivityRegularization, \
    Activation
from keras import losses, optimizers
import keras.backend as K

################################################################################
# Parameters
################################################################################

n_params = 2
n_features = 42
n_thetas_features = n_features + n_params


################################################################################
# Helper functions
################################################################################

def stack_layer(layers):
    def f(x):
        for k in range(len(layers)):
            x = layers[k](x)
        return x

    return f


def hidden_layers(n,
                  hidden_layer_size=100,
                  activation='tanh',
                  dropout_prob=0.0):
    r = []
    for k in range(n):
        if dropout_prob > 0.:
            s = stack_layer([
                Dropout(dropout_prob),
                Dense(hidden_layer_size, activation=activation)
            ])
        else:
            s = stack_layer([Dense(hidden_layer_size, activation=activation)])
        r.append(s)
    return stack_layer(r)


################################################################################
# Loss functions
################################################################################

def loss_function_scoreregression(y_true, y_pred):
    return losses.mean_squared_error(y_true[:, :n_params], y_pred[:, :n_params])


################################################################################
# Score regression model
################################################################################

def make_regressor(n_hidden_layers=3,
                    hidden_layer_size=100,
                    activation='tanh',
                    dropout_prob=0.0):
    # Inputs
    input_layer = Input(shape=(42,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = hidden_layers(n_hidden_layers - 1,
                                      hidden_layer_size=hidden_layer_size,
                                      activation=activation,
                                      dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)
    score_layer = Dense(2, activation='linear')(hidden_layer)

    # gradients with respect to x... there must be a nicer way to do this?
    gradient0_layer = Lambda(lambda x: K.gradients(x[0][0], x[1])[0],
                            output_shape=(n_features,))([score_layer, input_layer])
    gradient1_layer = Lambda(lambda x: K.gradients(x[0][1], x[1])[0],
                            output_shape=(n_features,))([score_layer, input_layer])

    # Combine outputs
    output_layer = Concatenate()([score_layer, gradient0_layer, gradient1_layer])

    # Combine outputs
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=loss_function_score_regression,
                  optimizer=optimizers.Adam(clipnorm=1.))

    return model