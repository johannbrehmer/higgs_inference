################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
from scipy.stats import trim_mean, ncx2, chi2
from sklearn.metrics import mean_squared_error

try:
    from skopt.learning import GaussianProcessRegressor
    from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
except ImportError:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from higgs_inference import settings


################################################################################
# Translation between optimal decision function s and likelihood ratio r
################################################################################

def s_from_r(r):
    return np.clip(1. / (1. + r), 0., 1.)


def r_from_s(s, epsilon=1.e-6):
    return np.clip((1. - s + epsilon) / (s + epsilon), epsilon, None)


################################################################################
# Normal and trimmed mean squared error
################################################################################

def calculate_mean_squared_error(y_true, y_pred, trim='auto'):
    trim_ = trim
    if trim_ == 'auto':
        trim_ = settings.trim_mean_fraction

    if trim_ > 0.:
        return trim_mean((y_true - y_pred) ** 2, trim_)

    return mean_squared_error(y_true, y_pred)


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

# def decide_toy_evaluation(theta_hypothesis, theta_evaluation, distance_threshold=0.5): # distance_threshold = 0.3 in old results
#     if theta_evaluation == theta_hypothesis:
#         return True
#
#     if theta_evaluation in settings.pbp_training_thetas:
#         return True
#
#     delta_theta = np.linalg.norm(settings.thetas[theta_hypothesis] - settings.thetas[theta_evaluation])
#
#     return (delta_theta <= distance_threshold)


################################################################################
# Interpolation
################################################################################

def interpolate(thetas, z_thetas,
                xx, yy,
                method='linear',
                z_uncertainties_thetas=None,
                matern_exponent=0.5,
                length_scale_min=0.001,
                length_scale_default=1.,
                length_scale_max=1000.,
                noise_level=0.001,
                subtract_min=False):
    if method == 'cubic':

        interpolator = CloughTocher2DInterpolator(thetas[:], z_thetas)

        zz = interpolator(np.dstack((xx.flatten(), yy.flatten())))
        zi = zz.reshape(xx.shape)

    elif method == 'gp':

        if z_uncertainties_thetas is not None:
            gp = GaussianProcessRegressor(normalize_y=True,
                                          kernel=ConstantKernel(1.0, (1.e-9, 1.e9))
                                                 * Matern(length_scale=[length_scale_default],
                                                          length_scale_bounds=[(length_scale_min, length_scale_max)],
                                                          nu=matern_exponent)
                                                 + WhiteKernel(noise_level),
                                          n_restarts_optimizer=10,
                                          alpha=z_uncertainties_thetas)
        else:
            gp = GaussianProcessRegressor(normalize_y=True,
                                          kernel=ConstantKernel(1.0, (1.e-9, 1.e9))
                                                 * Matern(length_scale=length_scale_default,
                                                          length_scale_bounds=(length_scale_min, length_scale_max),
                                                          nu=matern_exponent)
                                                 + WhiteKernel(noise_level),
                                          n_restarts_optimizer=10)

        gp.fit(thetas[:], z_thetas[:])

        zz, _ = gp.predict(np.c_[xx.ravel(), yy.ravel()], return_std=True)
        zi = zz.reshape(xx.shape)

    elif method == 'linear':
        interpolator = LinearNDInterpolator(thetas[:], z_thetas)
        zz = interpolator(np.dstack((xx.flatten(), yy.flatten())))
        zi = zz.reshape(xx.shape)

    else:
        raise ValueError

    mle = np.unravel_index(zi.argmin(), zi.shape)

    if subtract_min:
        zi -= zi[mle]

    return zi, mle


################################################################################
# Interpolation
################################################################################

def asymptotic_p_value(asimov_q, use_median_rather_than_asimov=False):
    if use_median_rather_than_asimov:
        median_q = ncx2.ppf(0.5, df=2, nc=max(0., asimov_q))
        p_value = chi2.sf(median_q, df=2)
    else:
        p_value = chi2.sf(asimov_q, df=2)
    return p_value


################################################################################
# Weighted quantile
################################################################################

def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to np.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: np.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with np.percentile.
    :return: np.array with computed quantiles.
    """
    values = np.array(values, dtype=np.float64)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight, dtype=np.float64)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with np.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)
