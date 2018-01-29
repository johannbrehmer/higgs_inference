################################################################################
# Imports
################################################################################

import numpy as np

from keras.layers import Lambda, Concatenate, Multiply, Reshape
import keras.backend as K

from higgs_inference import settings

################################################################################
# Morphing
################################################################################

# Load morphing data
sample_component = np.load(settings.base_dir + '/data/morphing/components_fakebasis2.npy')[:,
                   1:]  # Ignore background component
component_sample = np.linalg.inv(sample_component)
sigma_sample = np.load(settings.base_dir + '/data/morphing/fakebasis2_xsecs.npy')
sigma_component = component_sample.dot(sigma_sample)


def generate_wtilde_layer(input_layer):
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


def generate_wi_layer(wtilde_layer):
    sigma_sample_var = K.variable(sigma_sample)
    sigma_wtilde_layer = Lambda(lambda w: w * sigma_sample_var)(wtilde_layer)

    wi_denom_layer = Lambda(lambda sw: 1. / K.sum(sw, axis=1))(sigma_wtilde_layer)
    wi_denoms_layer = Concatenate()([Reshape((1,))(wi_denom_layer) for i in range(settings.n_morphing_samples)])

    wi_layer = Multiply()([sigma_wtilde_layer, wi_denoms_layer])
    return wi_layer
