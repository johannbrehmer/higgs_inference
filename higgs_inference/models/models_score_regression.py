################################################################################
# Imports
################################################################################

from keras.models import Model
from keras.layers import Input, Dense
from keras import optimizers

from higgs_inference.various.utils import build_hidden_layers
from higgs_inference.models.loss_functions import loss_function_score_regression


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
        hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                            hidden_layer_size=hidden_layer_size,
                                            activation=activation,
                                            dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)
    score_layer = Dense(2, activation='linear')(hidden_layer)

    # Combine outputs
    model = Model(inputs=[input_layer], outputs=[score_layer])

    # Compile model
    model.compile(loss=loss_function_score_regression,
                  optimizer=optimizers.Adam(clipnorm=1.))

    return model
