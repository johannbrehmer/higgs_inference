################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import logging
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern

from higgs_inference import settings
from higgs_inference.various.utils import decide_toy_evaluation, s_from_r


def truth_inference(do_neyman=False,
                    options=''):
    """ Extracts the true likelihood ratios for the evaluation samples. """

    logging.info('Starting truth calculation')

    ################################################################################
    # Settings
    ################################################################################

    denom1_mode = ('denom1' in options)

    filename_addition = ''
    input_filename_addition = ''
    if denom1_mode:
        input_filename_addition = '_denom1'
        filename_addition += '_denom1'

    neyman_dir = settings.neyman_dir + '/truth'
    results_dir = settings.base_dir + '/results/truth'

    ################################################################################
    # Data
    ################################################################################

    scores_test = np.load(settings.unweighted_events_dir + '/scores_test' + input_filename_addition + '.npy')
    r_test = np.load(settings.unweighted_events_dir + '/r_test' + input_filename_addition + '.npy')
    r_roam = np.load(settings.unweighted_events_dir + '/r_roam' + input_filename_addition + '.npy')
    r_neyman_observed = np.load(settings.unweighted_events_dir + '/r_neyman_observed.npy')

    # To calculate cross entropy on train set
    s_train = s_from_r(np.load(settings.unweighted_events_dir + '/r_train' + input_filename_addition + '.npy'))
    y_train = np.load(settings.unweighted_events_dir + '/y_train' + input_filename_addition + '.npy')

    n_events_test = r_test.shape[1]
    assert settings.n_thetas == r_test.shape[0]

    xi = np.linspace(-1.0, 1.0, settings.n_thetas_roam)
    yi = np.linspace(-1.0, 1.0, settings.n_thetas_roam)
    xx, yy = np.meshgrid(xi, yi)

    ################################################################################
    # Evaluate truth likelihood ratios
    ################################################################################

    logging.info('Starting evaluation')
    expected_llr_truth = []

    for t, theta in enumerate(settings.thetas):
        log_r = np.log(r_test[t, :])
        expected_llr_truth.append(
            - 2. * float(settings.n_expected_events) / float(n_events_test) * np.sum(log_r))

    r_nottrained_truth = np.copy(r_test[settings.theta_benchmark_nottrained, :])
    r_trained_truth = np.copy(r_test[settings.theta_benchmark_trained, :])
    scores_trained_truth = np.copy(scores_test[settings.theta_benchmark_trained, :])
    scores_nottrained_truth = np.copy(scores_test[settings.theta_benchmark_nottrained, :])

    np.save(results_dir + '/r_nottrained_truth' + filename_addition + '.npy', r_nottrained_truth)
    np.save(results_dir + '/r_trained_truth' + filename_addition + '.npy', r_trained_truth)
    np.save(results_dir + '/scores_trained_truth' + filename_addition + '.npy', scores_trained_truth)
    np.save(results_dir + '/scores_nottrained_truth' + filename_addition + '.npy', scores_nottrained_truth)
    np.save(results_dir + '/llr_truth' + filename_addition + '.npy', expected_llr_truth)

    logging.info('Starting roaming')
    gp = GaussianProcessRegressor(normalize_y=True,
                                  kernel=C(1.0) * Matern(1.0, nu=0.5), n_restarts_optimizer=10)
    gp.fit(settings.thetas[:], np.log(r_roam))
    r_roam_truth = np.exp(gp.predict(np.c_[xx.ravel(), yy.ravel()])).T
    np.save(results_dir + '/r_roam_truth' + filename_addition + '.npy', r_roam_truth)

    # Calculate cross entropy on training sample (for comparison to carl loss)
    logging.info('Calculating cross-entropy on train set')
    s_train = np.clip(s_train, settings.epsilon, 1. - settings.epsilon)
    cross_entropies = y_train * np.log(s_train) + (1. - y_train) * np.log(1. - s_train)
    logging.debug('Cross entropy terms: shape %s, NaN %s, min %s, max %s, median %s, mean %s,\nvalues $s',
                  cross_entropies.shape, np.sum(np.isnan(cross_entropies)),
                  np.nanmin(cross_entropies), np.nanmax(cross_entropies),
                  np.median(cross_entropies), np.mean(cross_entropies), cross_entropies)
    cross_entropy_train = np.mean(y_train * np.log(s_train) + (1. - y_train) * np.log(1. - s_train))
    logging.info('Train set true cross-entropy: %s', cross_entropy_train)
    np.save(results_dir + '/cross_entropy_truth_train.npy', np.asarray([cross_entropy_train]))

    if do_neyman:
        logging.info('Starting evaluation of Neyman experiments')
        for t in range(settings.n_thetas):

            # Observed
            llr_neyman_observed = -2. * np.sum(np.log(r_neyman_observed[t]), axis=1)
            np.save(neyman_dir + '/neyman_llr_observed_truth_' + str(t) + filename_addition + '.npy', llr_neyman_observed)

            # Hypothesis distributions
            llr_neyman_distributions = []
            for tt in range(settings.n_thetas):

                # Only evaluate certain combinations of thetas to save computation time
                if not decide_toy_evaluation(tt, t):
                    placeholder = np.empty(settings.n_neyman_distribution_experiments)
                    placeholder[:] = np.nan
                    llr_neyman_distributions.append(placeholder)
                    continue

                r_neyman_distribution = np.load(
                    settings.unweighted_events_dir + '/r_neyman_distribution_' + str(tt) + '.npy')
                llr_neyman_distributions.append(-2. * np.sum(np.log(r_neyman_distribution[t]), axis=1))

            llr_neyman_distributions = np.asarray(llr_neyman_distributions)
            np.save(neyman_dir + '/neyman_llr_distribution_truth_' + str(t) + filename_addition + '.npy',
                    llr_neyman_distributions)