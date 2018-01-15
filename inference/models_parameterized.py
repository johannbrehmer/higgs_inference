#! /usr/bin/env python


################################################################################
# Imports
################################################################################

import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Concatenate, Multiply, Reshape, ActivityRegularization
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
# Morphing
################################################################################

# Morphing (original)
# sigma_component = np.load('../data/morphing/component_xsec.npy')[1:] # Ignore background component
# component_sample = np.load('../data/morphing/component_sample.npy')[1:] # Ignore background component
# sigma_sample = np.linalg.inv(component_sample).dot(sigma_component)
# n_samples = 15

# Morphing (optimized)
sample_component = np.load('../data/morphing/components_fakebasis2.npy')[:,1:] # Ignore background component
component_sample = np.linalg.inv(sample_component)
sigma_sample = np.load('../data/morphing/fakebasis2_xsecs.npy')
sigma_component = component_sample.dot(sigma_sample)
n_samples = 15

def _generate_wtilde_layer(input_layer):
    wtilde_component_layers = [Lambda(lambda t: 1. + 0. * t[:, 0])(input_layer),
                               Lambda(lambda t: t[:, 1])(input_layer),
                               Lambda(lambda t: t[:, 1] * t[:, 1])(input_layer),
                               Lambda(lambda t: t[:, 1] * t[:, 1] * t[:, 1])(input_layer),
                               Lambda(lambda t: t[:, 1] * t[:, 1] * t[:, 1] * t[:, 1])(input_layer),
                               Lambda(lambda t: t[:, 0])(input_layer),
                               Lambda(lambda t: t[:, 0] * t[:, 1])(input_layer),
                               Lambda(lambda t: t[:, 0] * t[:, 1] * t[:, 1])(input_layer),
                               Lambda(lambda t: t[:, 0] * t[:, 1] * t[:, 1] * t[:, 1])(input_layer),
                               Lambda(lambda t: t[:, 0] * t[:, 0])(input_layer),
                               Lambda(lambda t: t[:, 0] * t[:, 0] * t[:, 1])(input_layer),
                               Lambda(lambda t: t[:, 0] * t[:, 0] * t[:, 1] * t[:, 1])(input_layer),
                               Lambda(lambda t: t[:, 0] * t[:, 0] * t[:, 0])(input_layer),
                               Lambda(lambda t: t[:, 0] * t[:, 0] * t[:, 0] * t[:, 1])(input_layer),
                               Lambda(lambda t: t[:, 0] * t[:, 0] * t[:, 0] * t[:, 0])(input_layer)]
    wtilde_component_reshaped_layers = [Reshape((1,))(layer) for layer in wtilde_component_layers]
    wtilde_component_layer = Concatenate()(wtilde_component_reshaped_layers)

    component_sample_var = K.variable(component_sample)
    wtilde_layer = Lambda(lambda x: K.dot(x, component_sample_var))(wtilde_component_layer)
    return wtilde_layer


def _generate_wi_layer(wtilde_layer):
    sigma_sample_var = K.variable(sigma_sample)
    sigma_wtilde_layer = Lambda(lambda w: w * sigma_sample_var)(wtilde_layer)

    wi_denom_layer = Lambda(lambda sw: 1. / K.sum(sw, axis=1))(sigma_wtilde_layer)
    wi_denoms_layer = Concatenate()([Reshape((1,))(wi_denom_layer) for i in range(n_samples)])

    wi_layer = Multiply()([sigma_wtilde_layer, wi_denoms_layer])
    return wi_layer



################################################################################
# Loss functions
################################################################################

def loss_function_carl(y_true, y_pred):
    return losses.binary_crossentropy(y_true[:, 0], y_pred[:, 0])

def loss_function_regression(y_true, y_pred):
    return losses.mean_squared_error(y_true[:, 0], y_pred[:, 0])

def loss_function_score(y_true, y_pred):
    return losses.mean_squared_error(y_true[:, 1:n_params+1], y_pred[:, 1:n_params+1])

def loss_function_combinedregression(y_true, y_pred, alpha=0.005):
    return loss_function_regression(y_true, y_pred) + alpha * loss_function_score(y_true, y_pred)

def loss_function_combined(y_true, y_pred, alpha=0.1):
    return loss_function_carl(y_true, y_pred) + alpha * loss_function_score(y_true, y_pred)




################################################################################
# Regression
################################################################################

def make_regressor(n_hidden_layers=3,
                   hidden_layer_size=100,
                   activation='tanh',
                   dropout_prob=0.0):

    # Inputs
    input_layer = Input(shape=(n_thetas_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = hidden_layers(n_hidden_layers - 1,
                                      hidden_layer_size=hidden_layer_size,
                                      activation=activation,
                                      dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)
    log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)

    # Score
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0],
                            output_shape=(n_thetas_features,))([log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -n_params:], output_shape=(n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([log_r_hat_layer, score_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=loss_function_regression,
                  metrics=[loss_function_score],
                  optimizer=optimizers.Adam(clipnorm=1.))

    return model



def make_regressor_morphingaware(n_hidden_layers=2,
                                 hidden_layer_size=100,
                                 activation='tanh',
                                 dropout_prob=0.0,
                                 epsilon=1.e-4):
    # Inputs
    input_layer = Input(shape=(n_thetas_features,))
    x_layer = Lambda(lambda x: x[:, :n_features], output_shape=(n_features,))(input_layer)
    theta_layer = Lambda(lambda x: x[:, -n_params:], output_shape=(n_params,))(input_layer)

    # Morphing weights
    wtilde_layer = _generate_wtilde_layer(theta_layer)
    wi_layer = _generate_wi_layer(wtilde_layer)

    # Ratio estimators for each component
    ri_hat_layers = []
    for i in range(n_samples):
        hidden_layer = Dense(hidden_layer_size, activation=activation)(x_layer)
        if n_hidden_layers > 1:
            hidden_layer_ = hidden_layers(n_hidden_layers - 1,
                                          hidden_layer_size=hidden_layer_size,
                                          activation=activation,
                                          dropout_prob=dropout_prob)
            hidden_layer = hidden_layer_(hidden_layer)
        si_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)
        ri_hat_layers.append(Reshape((1,))(Lambda(lambda x: (1. - x) / (x + epsilon))(si_hat_layer)))
    ri_hat_layer = Concatenate()(ri_hat_layers)

    # Combine, clip, transform to \hat{s}
    wi_ri_hat_layer = Multiply()([wi_layer, ri_hat_layer])
    r_hat_layer = Reshape((1,))(Lambda(lambda x: K.sum(x, axis=1))(wi_ri_hat_layer))

    # Score
    log_r_hat_layer = Lambda(lambda x: K.log(x + epsilon))(r_hat_layer)
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0], output_shape=(n_thetas_features,))(
        [log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -n_params:], output_shape=(n_params,))(gradient_layer)

    # Combine outputs
    # output_layer = Concatenate()([log_r_hat_layer, score_layer])
    output_layer = Concatenate()([log_r_hat_layer, score_layer, wi_layer, ri_hat_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=loss_function_regression,
                  metrics=[loss_function_score],
                  optimizer=optimizers.Adam(clipnorm=1.))

    return model



################################################################################
# Regression + score
################################################################################

def make_combined_regressor(n_hidden_layers=3,
                            hidden_layer_size=100,
                            activation='tanh',
                            dropout_prob=0.0,
                            alpha=0.005):
    # Inputs
    input_layer = Input(shape=(n_thetas_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = hidden_layers(n_hidden_layers - 1,
                                      hidden_layer_size=hidden_layer_size,
                                      activation=activation,
                                      dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)
    log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)

    # Score
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0],
                            output_shape=(n_thetas_features,))([log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -n_params:], output_shape=(n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([log_r_hat_layer, score_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=lambda x, y: loss_function_combinedregression(x, y, alpha=alpha),
                  metrics=[loss_function_regression, loss_function_score],
                  optimizer=optimizers.Adam(clipnorm=1.))

    return model



def make_combined_regressor_morphingaware(n_hidden_layers=2,
                                          hidden_layer_size=100,
                                          activation='tanh',
                                          dropout_prob=0.0,
                                          alpha=0.005,
                                          epsilon=1.e-4):
    # Inputs
    input_layer = Input(shape=(n_thetas_features,))
    x_layer = Lambda(lambda x: x[:, :n_features], output_shape=(n_features,))(input_layer)
    theta_layer = Lambda(lambda x: x[:, -n_params:], output_shape=(n_params,))(input_layer)

    # Morphing weights
    wtilde_layer = _generate_wtilde_layer(theta_layer)
    wi_layer = _generate_wi_layer(wtilde_layer)

    # Ratio estimators for each component
    ri_hat_layers = []
    for i in range(n_samples):
        hidden_layer = Dense(hidden_layer_size, activation=activation)(x_layer)
        if n_hidden_layers > 1:
            hidden_layer_ = hidden_layers(n_hidden_layers - 1,
                                          hidden_layer_size=hidden_layer_size,
                                          activation=activation,
                                          dropout_prob=dropout_prob)
            hidden_layer = hidden_layer_(hidden_layer)
        si_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)
        ri_hat_layers.append(Reshape((1,))(Lambda(lambda x: (1. - x) / (x + epsilon))(si_hat_layer)))
    ri_hat_layer = Concatenate()(ri_hat_layers)

    # Combine, clip, transform to \hat{s}
    wi_ri_hat_layer = Multiply()([wi_layer, ri_hat_layer])
    r_hat_layer = Reshape((1,))(Lambda(lambda x: K.sum(x, axis=1))(wi_ri_hat_layer))

    # Score
    log_r_hat_layer = Lambda(lambda x: K.log(x + epsilon))(r_hat_layer)
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0], output_shape=(n_thetas_features,))(
        [log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -n_params:], output_shape=(n_params,))(gradient_layer)

    # Combine outputs
    # output_layer = Concatenate()([log_r_hat_layer, score_layer])
    output_layer = Concatenate()([log_r_hat_layer, score_layer, wi_layer, ri_hat_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=lambda x, y: loss_function_combinedregression(x, y, alpha=alpha),
                  metrics=[loss_function_regression, loss_function_score],
                  optimizer=optimizers.Adam(clipnorm=1.))

    return model



################################################################################
# carl
################################################################################

def make_classifier_carl(n_hidden_layers=3,
                                  hidden_layer_size=100,
                                  activation='tanh',
                                  dropout_prob=0.0,
                                  learn_log_r=False):
    # Inputs
    input_layer = Input(shape=(n_thetas_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = hidden_layers(n_hidden_layers - 1,
                                      hidden_layer_size=hidden_layer_size,
                                      activation=activation,
                                      dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)

    if learn_log_r:
        log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)
        r_hat_layer = Lambda(lambda x: K.exp(x))(log_r_hat_layer)
        s_hat_layer = Lambda(lambda x: 1./(1. + r_hat_layer))(log_r_hat_layer)

    else:
        s_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)
        r_hat_layer = Lambda(lambda x: (1. - x) / x)(s_hat_layer)
        log_r_hat_layer = Lambda(lambda x: K.log(x))(r_hat_layer)

    # Score
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0],
                            output_shape=(n_thetas_features,))([log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -n_params:], output_shape=(n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, score_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=loss_function_carl,
                  metrics=[loss_function_score],
                  optimizer=optimizers.Adam(clipnorm=1.))

    return model


def make_classifier_carl_morphingaware(n_hidden_layers=2,
                                       hidden_layer_size=100,
                                       activation='tanh',
                                       dropout_prob=0.0,
                                       learn_log_r=False,
                                       epsilon=1.e-4):
    # Inputs
    input_layer = Input(shape=(n_thetas_features,))
    x_layer = Lambda(lambda x: x[:, :n_features], output_shape=(n_features,))(input_layer)
    theta_layer = Lambda(lambda x: x[:, -n_params:], output_shape=(n_params,))(input_layer)

    # Morphing weights
    wtilde_layer = _generate_wtilde_layer(theta_layer)
    wi_layer = _generate_wi_layer(wtilde_layer)

    # Ratio estimators for each component
    ri_hat_layers = []
    for i in range(n_samples):
        hidden_layer = Dense(hidden_layer_size, activation=activation)(x_layer)
        if n_hidden_layers > 1:
            hidden_layer_ = hidden_layers(n_hidden_layers - 1,
                                          hidden_layer_size=hidden_layer_size,
                                          activation=activation,
                                          dropout_prob=dropout_prob)
            hidden_layer = hidden_layer_(hidden_layer)

        if learn_log_r:
            log_ri_hat_layer = Dense(1, activation='linear')(hidden_layer)
            ri_hat_layers.append(Lambda(lambda x: K.exp(x))(log_ri_hat_layer))
        else:
            si_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)
            ri_hat_layers.append(Reshape((1,))(Lambda(lambda x: (1. - x) / (x + epsilon))(si_hat_layer)))

    ri_hat_layer = Concatenate()(ri_hat_layers)

    # Combine, clip, transform to \hat{s}
    wi_ri_hat_layer = Multiply()([wi_layer, ri_hat_layer])
    r_hat_layer = Reshape((1,))(Lambda(lambda x: K.sum(x, axis=1))(wi_ri_hat_layer))
    s_hat_layer = Lambda(lambda x: 1. / (1. + x))(r_hat_layer)

    # Score
    log_r_hat_layer = Lambda(lambda x: K.log(x + epsilon))(r_hat_layer)
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0], output_shape=(n_thetas_features,))(
        [log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -n_params:], output_shape=(n_params,))(gradient_layer)

    # Combine outputs
    # output_layer = Concatenate()([s_hat_layer, score_layer])
    output_layer = Concatenate()([s_hat_layer, score_layer, wi_layer, ri_hat_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=loss_function_carl,
                  metrics=[loss_function_score],
                  optimizer=optimizers.Adam(clipnorm=1.))

    return model



################################################################################
# Score only
################################################################################

def make_classifier_score(n_hidden_layers=3,
                          hidden_layer_size=100,
                          activation='tanh',
                          dropout_prob=0.0,
                          learn_log_r=False,
                          l2_regularization=0.001):
    # Inputs
    input_layer = Input(shape=(n_thetas_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = hidden_layers(n_hidden_layers - 1,
                                      hidden_layer_size=hidden_layer_size,
                                      activation=activation,
                                      dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)

    if learn_log_r:
        log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)
        r_hat_layer = Lambda(lambda x: K.exp(x))(log_r_hat_layer)
        s_hat_layer = Lambda(lambda x: 1./(1. + r_hat_layer))(log_r_hat_layer)

    else:
        s_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)
        r_hat_layer = Lambda(lambda x: (1. - x) / x)(s_hat_layer)
        log_r_hat_layer = Lambda(lambda x: K.log(x))(r_hat_layer)

    # Score
    regularizer_layer = ActivityRegularization(l1=0.,l2=l2_regularization)(log_r_hat_layer)
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0],
                            output_shape=(n_thetas_features,))([regularizer_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -n_params:], output_shape=(n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, score_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=loss_function_score,
                  metrics=[loss_function_carl],
                  optimizer=optimizers.Adam(clipnorm=1.))

    return model


def make_classifier_score_morphingaware(n_hidden_layers=2,
                                           hidden_layer_size=100,
                                           activation='tanh',
                                           dropout_prob=0.0,
                                           l2_regularization=0.001,
                                           learn_log_r=False,
                                           epsilon=1.e-4):
    # Inputs
    input_layer = Input(shape=(n_thetas_features,))
    x_layer = Lambda(lambda x: x[:, :n_features], output_shape=(n_features,))(input_layer)
    theta_layer = Lambda(lambda x: x[:, -n_params:], output_shape=(n_params,))(input_layer)

    # Morphing weights
    wtilde_layer = _generate_wtilde_layer(theta_layer)
    wi_layer = _generate_wi_layer(wtilde_layer)

    # Ratio estimators for each component
    ri_hat_layers = []
    for i in range(n_samples):
        hidden_layer = Dense(hidden_layer_size, activation=activation)(x_layer)
        if n_hidden_layers > 1:
            hidden_layer_ = hidden_layers(n_hidden_layers - 1,
                                          hidden_layer_size=hidden_layer_size,
                                          activation=activation,
                                          dropout_prob=dropout_prob)
            hidden_layer = hidden_layer_(hidden_layer)

        if learn_log_r:
            log_ri_hat_layer = Dense(1, activation='linear')(hidden_layer)
            ri_hat_layers.append(Lambda(lambda x: K.exp(x))(log_ri_hat_layer))
        else:
            si_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)
            ri_hat_layers.append(Reshape((1,))(Lambda(lambda x: (1. - x) / (x + epsilon))(si_hat_layer)))

    ri_hat_layer = Concatenate()(ri_hat_layers)

    # Combine, clip, transform to \hat{s}
    wi_ri_hat_layer = Multiply()([wi_layer, ri_hat_layer])
    r_hat_layer = Reshape((1,))(Lambda(lambda x: K.sum(x, axis=1))(wi_ri_hat_layer))
    s_hat_layer = Lambda(lambda x: 1. / (1. + x))(r_hat_layer)

    # Score
    log_r_hat_layer = Lambda(lambda x: K.log(x + epsilon))(r_hat_layer)
    regularizer_layer = ActivityRegularization(l1=0.,l2=l2_regularization)(log_r_hat_layer)
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0], output_shape=(n_thetas_features,))(
        [regularizer_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -n_params:], output_shape=(n_params,))(gradient_layer)

    # Combine outputs
    # output_layer = Concatenate()([s_hat_layer, score_layer])
    output_layer = Concatenate()([s_hat_layer, score_layer, wi_layer, ri_hat_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=loss_function_score,
                  metrics=[loss_function_carl],
                  optimizer=optimizers.Adam(clipnorm=1.))

    return model



################################################################################
# carl + score
################################################################################

def make_classifier_combined(n_hidden_layers=3,
                             hidden_layer_size=100,
                             activation='tanh',
                             dropout_prob=0.0,
                             learn_log_r=False,
                             alpha=0.1):
    # Inputs
    input_layer = Input(shape=(n_thetas_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = hidden_layers(n_hidden_layers - 1,
                                      hidden_layer_size=hidden_layer_size,
                                      activation=activation,
                                      dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)

    if learn_log_r:
        log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)
        r_hat_layer = Lambda(lambda x: K.exp(x))(log_r_hat_layer)
        s_hat_layer = Lambda(lambda x: 1./(1. + r_hat_layer))(log_r_hat_layer)

    else:
        s_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)
        r_hat_layer = Lambda(lambda x: (1. - x) / x)(s_hat_layer)
        log_r_hat_layer = Lambda(lambda x: K.log(x))(r_hat_layer)

    # Score
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0],
                            output_shape=(n_thetas_features,))([log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -n_params:], output_shape=(n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, score_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=lambda x, y: loss_function_combined(x, y, alpha=alpha),
                  metrics=[loss_function_carl, loss_function_score],
                  optimizer=optimizers.Adam(clipnorm=1.))

    return model


def make_classifier_combined_morphingaware(n_hidden_layers=2,
                                           hidden_layer_size=100,
                                           activation='tanh',
                                           dropout_prob=0.0,
                                           alpha=0.1,
                                           learn_log_r=False,
                                           epsilon=1.e-4):
    # Inputs
    input_layer = Input(shape=(n_thetas_features,))
    x_layer = Lambda(lambda x: x[:, :n_features], output_shape=(n_features,))(input_layer)
    theta_layer = Lambda(lambda x: x[:, -n_params:], output_shape=(n_params,))(input_layer)

    # Morphing weights
    wtilde_layer = _generate_wtilde_layer(theta_layer)
    wi_layer = _generate_wi_layer(wtilde_layer)

    # Ratio estimators for each component
    ri_hat_layers = []
    for i in range(n_samples):
        hidden_layer = Dense(hidden_layer_size, activation=activation)(x_layer)
        if n_hidden_layers > 1:
            hidden_layer_ = hidden_layers(n_hidden_layers - 1,
                                          hidden_layer_size=hidden_layer_size,
                                          activation=activation,
                                          dropout_prob=dropout_prob)
            hidden_layer = hidden_layer_(hidden_layer)

        if learn_log_r:
            log_ri_hat_layer = Dense(1, activation='linear')(hidden_layer)
            ri_hat_layers.append(Lambda(lambda x: K.exp(x))(log_ri_hat_layer))
        else:
            si_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)
            ri_hat_layers.append(Reshape((1,))(Lambda(lambda x: (1. - x) / (x + epsilon))(si_hat_layer)))

    ri_hat_layer = Concatenate()(ri_hat_layers)

    # Combine, clip, transform to \hat{s}
    wi_ri_hat_layer = Multiply()([wi_layer, ri_hat_layer])
    r_hat_layer = Reshape((1,))(Lambda(lambda x: K.sum(x, axis=1))(wi_ri_hat_layer))
    s_hat_layer = Lambda(lambda x: 1. / (1. + x))(r_hat_layer)

    # Score
    log_r_hat_layer = Lambda(lambda x: K.log(x + epsilon))(r_hat_layer)
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0], output_shape=(n_thetas_features,))(
        [log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -n_params:], output_shape=(n_params,))(gradient_layer)

    # Combine outputs
    # output_layer = Concatenate()([s_hat_layer, score_layer])
    output_layer = Concatenate()([s_hat_layer, score_layer, wi_layer, ri_hat_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=lambda x, y: loss_function_combined(x, y, alpha=alpha),
                  metrics=[loss_function_carl, loss_function_score],
                  optimizer=optimizers.Adam(clipnorm=1.))

    return model
