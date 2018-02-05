################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern

from higgs_inference import settings


################################################################################
# Translation between optimal decision function s and likelihood ratio r
################################################################################

def s_from_r(r):
    return np.clip(1. / (1. + r), 0., 1.)


def r_from_s(s, epsilon=1.e-3):
    return np.clip((1. - s + epsilon) / (s + epsilon), epsilon, None)


################################################################################
# Formatting numbers as output strings
################################################################################

def format_number(number,
                  precision=2,
                  trailing_zeros=True,
                  fix_minus_zero=True,
                  latex_math_mode=False,
                  emphasize=False):
    if precision == 0:
        temp = str(int(round(number, precision)))
    elif trailing_zeros:
        temp = ('{:.' + str(precision) + 'f}').format(round(number, precision))
    else:
        temp = str(round(number, precision))
    if fix_minus_zero and len(temp) > 0:
        if temp[0] == '-' and float(temp) == 0.:
            temp = temp[1:]
    if latex_math_mode:
        if emphasize:
            temp = '$\mathbf{' + temp + '}$'
        else:
            temp = '$' + temp + '$'
    elif emphasize:
        temp = r'\emph{' + temp + r'}'
    return temp


################################################################################
# Decide if two a given Neyman toy experiment should be evaluated at a given theta
################################################################################

def decide_toy_evaluation(theta_hypothesis, theta_evaluation, distance_threshold=0.4):
    if theta_evaluation == theta_hypothesis:
        return True

    if theta_evaluation in settings.pbp_training_thetas:
        return True

    delta_theta = np.linalg.norm(settings.thetas[theta_hypothesis] - settings.thetas[theta_evaluation])

    return (delta_theta <= distance_threshold)



################################################################################
# Interpolation
################################################################################

def interpolate(thetas, z_thetas,
                xx, yy,
                method='linear',
                z_uncertainties_thetas=None,
                subtract_min=False):

    if method=='cubic':

        interpolator = CloughTocher2DInterpolator(thetas[:], z_thetas)

        zz = interpolator(np.dstack((xx.flatten(), yy.flatten())))
        zi = zz.reshape(xx.shape)

    elif method=='gp':

        if z_uncertainties_thetas is not None:
            gp = GaussianProcessRegressor(normalize_y=True,
                                          kernel=C(1.0) * Matern(1.0,nu=0.5),
                                          n_restarts_optimizer=10, alpha=z_uncertainties_thetas)
        else:
            gp = GaussianProcessRegressor(normalize_y=True,
                                          kernel=C(1.0) * Matern(1.0,nu=0.5),
                                          # kernel=C(1.0, (1.e-9, 1.e9)) + C(1.0, (1.e-9, 1.e9)) * Matern(1.0, nu=0.5),
                                          n_restarts_optimizer=10, alpha=1.e-6)

        gp.fit(thetas[:], z_thetas[:])

        zz, _ = gp.predict(np.c_[xx.ravel(), yy.ravel()], return_std=True)
        zi = zz.reshape(xx.shape)

    elif method=='linear':
        interpolator = LinearNDInterpolator(thetas[:], z_thetas)
        zz = interpolator(np.dstack((xx.flatten(), yy.flatten())))
        zi = zz.reshape(xx.shape)

    else:
        raise ValueError

    mle = np.unravel_index(zi.argmin(), zi.shape)

    if subtract_min:
        zi -= zi[mle]

    return zi, mle