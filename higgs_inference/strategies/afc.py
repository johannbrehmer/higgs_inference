################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import logging
import numpy as np

from scipy.interpolate import LinearNDInterpolator
from sklearn.preprocessing import StandardScaler

from higgs_inference import settings
from higgs_inference.various.utils import format_number


def count_acceptances(hypothesis_statistics,
                      observed_statistics,
                      epsilon):

    """ Counts how often the observed_statistics are in epsilon-balls with radius epsilon around the
    hypothesis_statistics. """

    counter = 0

    for hypo in hypothesis_statistics:
        for obs in observed_statistics:
            if np.sum((hypo - obs)**2) <= epsilon**2:
                counter += 1

    return counter


def afc_inference(statistics='x',
                  indices_X=None,
                  epsilon=None,
                  options=''):

    """
    Approximates the likelihood through Approximate Frequentist Inference, a frequentist twist on classic rejection ABC.

    :param statistics: Defines which summary statistics is used to decide upon acceptance or rejection of events.
                       Currently the only option is 'x', which bases the acceptance decision on an epsilon ball in
                       (re-scaled) feature space.
    :param indices_X: If statistics is 'x', this defines which of the features are used in the distance calculation. If
                      None, a default selection of 15 variables is used.
    :param epsilon: This float > 0 defines the size of the epsilon-ball. A smaller value makes the inference more
                    precise, but requires more data, especially when using high-dimensional statistics. If no epsilon
                    is given, the algorithm uses 0.05^(1/n_dim), where n_dim is the number of dimensions of the
                    statistics space, for instance the length of indices_X.
    :param options: Further options in a list of strings or string.
    """

    logging.info('Starting AFC inference')

    ################################################################################
    # Settings
    ################################################################################

    if statistics is not 'x':
        raise NotImplementedError

    # TODO: Implement AFC with estimated score or classifier output as summary statistics

    if indices_X is None:
        indices_X = []

    statistics_dimensionality = len(indices_X)

    filename_addition = '_' + statistics

    if epsilon is None:
        epsilon = 0.05 ** (1. / statistics_dimensionality)
    else:
        filename_addition = filename_addition + '_epsilon_' + format_number(epsilon, 2)

    denom1_mode = ('denom1' in options)
    theta1 = settings.theta1_default
    input_filename_addition = ''
    if denom1_mode:
        input_filename_addition = '_denom1'
        filename_addition += '_denom1'
        theta1 = settings.theta1_alternative

    data_dir = settings.base_dir + '/data'
    results_dir = settings.base_dir + '/results/point_by_point'
    neyman_dir = settings.neyman_dir + '/point_by_point'

    logging.info('Settings:')
    if statistics == 'x':
        logging.info('  Statistics:              x %s', x_indices)
    else:
        logging.info('  Statistics:              %s', statistics)
    logging.info('  Epsilon:                 %s', epsilon)

    ################################################################################
    # Data
    ################################################################################

    thetas = np.load(data_dir + '/thetas/thetas_parameterized.npy')
    n_thetas = len(thetas)

    X_test = np.load(settings.unweighted_events_dir + '/X_test' + input_filename_addition + '.npy')
    X_neyman_observed = np.load(settings.unweighted_events_dir + '/X_neyman_observed.npy')

    n_events_test = X_test.shape[0]
    assert n_thetas == r_test.shape[0]

    ################################################################################
    # AFC
    ################################################################################

    expected_llr = []

    # Loop over the hypothesis thetas
    for i, t in enumerate(settings.pbp_training_thetas):

        logging.info('Starting theta %s/%s: number %s (%s)', i + 1, len(settings.pbp_training_thetas), t, thetas[t])

        # Load data
        X_hypothesis = np.load(
            settings.unweighted_events_dir + '/X_train_point_by_point_' + str(t) + input_filename_addition + '.npy')
        y_hypothesis = np.load(
            settings.unweighted_events_dir + '/y_train_point_by_point_' + str(t) + input_filename_addition + '.npy')

        # Scale data
        scaler = StandardScaler()
        scaler.fit(np.array(X_hypothesis, dtype=np.float64))
        X_hypothesis_transformed = scaler.transform(X_hypothesis)
        X_test_transformed = scaler.transform(X_test)
        X_neyman_observed_transformed = scaler.transform(
            X_neyman_observed.reshape((-1, X_neyman_observed.shape[2])))

        # Construct summary statistics
        if statistics == 'x':
            summary_statistics_hypothesis = X_hypothesis_transformed[indices_X]
            summary_statistics_test = X_test_transformed[indices_X]
        else:
            raise NotImplementedError

        # Calculate acceptance rate for nominator and denominator theta
        accepted_num = count_acceptances(summary_statistics_hypothesis[y_hypothesis == 0], summary_statistics_test)
        accepted_den = count_acceptances(summary_statistics_hypothesis[y_hypothesis == 1], summary_statistics_test)

        expected_llr.append(- 2. * settings.n_expected_events / n_events_test * np.sum(np.log(this_r)))

        # For some benchmark thetas, save r for each phase-space point
        if t == settings.theta_benchmark_nottrained:
            np.save(results_dir + '/r_nottrained_' + algorithm + filename_addition + '.npy', this_r)
        elif t == settings.theta_benchmark_trained:
            np.save(results_dir + '/r_trained_' + algorithm + filename_addition + '.npy', this_r)

        # Neyman construction: evaluate observed sample (raw)
        log_r_neyman_observed = regr.predict(X_neyman_observed_transformed)
        llr_neyman_observed = -2. * np.sum(log_r_neyman_observed.reshape((-1, settings.n_expected_events)), axis=1)
        np.save(neyman_dir + '/neyman_llr_observed_' + algorithm + '_' + str(t) + filename_addition + '.npy',
                llr_neyman_observed)

        # Neyman construction: loop over distribution samples generated from different thetas
        llr_neyman_distributions = []
        for tt in range(n_thetas):
            # Neyman construction: load distribution sample
            X_neyman_distribution = np.load(
                settings.unweighted_events_dir + '/X_neyman_distribution_' + str(tt) + '.npy')
            X_neyman_distribution_transformed = scaler.transform(
                X_neyman_distribution.reshape((-1, X_neyman_distribution.shape[2])))

            # Neyman construction: evaluate distribution sample (raw)
            log_r_neyman_distribution = regr.predict(X_neyman_distribution_transformed)
            llr_neyman_distributions.append(
                -2. * np.sum(log_r_neyman_distribution.reshape((-1, settings.n_expected_events)), axis=1))

        llr_neyman_distributions = np.asarray(llr_neyman_distributions)
        np.save(neyman_dir + '/neyman_llr_distribution_' + algorithm + '_' + str(t) + filename_addition + '.npy',
                llr_neyman_distributions)

    expected_llr = np.asarray(expected_llr)

    logging.info('Interpolation')

    interpolator = LinearNDInterpolator(thetas[settings.pbp_training_thetas], expected_llr)
    expected_llr_all = interpolator(thetas)
    # gp = GaussianProcessRegressor(normalize_y=True,
    #                              kernel=C(1.0) * Matern(1.0, nu=0.5), n_restarts_optimizer=10)
    # gp.fit(thetas[settings.pbp_training_thetas], expected_llr)
    # expected_llr_all = gp.predict(thetas)
    np.save(results_dir + '/llr_' + algorithm + filename_addition + '.npy', expected_llr_all)
