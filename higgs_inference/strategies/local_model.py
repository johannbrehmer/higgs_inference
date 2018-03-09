################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import logging
import numpy as np

from higgs_inference import settings


def local_model_truth_inference(do_neyman=False,
                                options=''):
    """ Extracts the likelihood ratios in the local model based on the true SM scores. """

    # TODO: Calibration for the local model (calculating Z(theta))

    logging.info('Starting local model calculation based on true score')

    ################################################################################
    # Settings
    ################################################################################

    denom1_mode = ('denom1' in options)

    theta1 = settings.theta1_default
    filename_addition = ''
    input_filename_addition = ''
    if denom1_mode:
        input_filename_addition = '_denom1'
        filename_addition += '_denom1'
        theta1 = settings.theta1_alternative

    neyman_dir = settings.neyman_dir + '/truth'
    results_dir = settings.base_dir + '/results/truth'

    ################################################################################
    # Data
    ################################################################################

    scores_test = np.load(settings.unweighted_events_dir + '/scores_test' + input_filename_addition + '.npy')[
        settings.theta_score_regression]
    scores_neyman_observed = np.load(settings.unweighted_events_dir + '/scores_neyman_observed.npy')

    logging.debug('Scores test: shape %s, content\n%s', scores_test.shape, scores_test)
    logging.debug('Scores Neyman observed: shape %s, content\n%s',
                  scores_neyman_observed.shape, scores_neyman_observed)

    n_events_test = scores_test.shape[0]

    ################################################################################
    # Evaluate truth likelihood ratios
    ################################################################################

    logging.info('Starting evaluation')
    expected_llr = []

    for t, theta in enumerate(settings.thetas):

        # Contract estimated scores with delta theta
        delta_theta = theta - settings.thetas[theta1]

        # Evaluation
        tt_test = scores_test.dot(delta_theta)

        expected_score = float(settings.n_expected_events) / float(n_events_test) * np.sum(scores_test, axis=0)

        expected_llr.append(
            - 2. * float(settings.n_expected_events) / float(n_events_test) * np.sum(tt_test))

        logging.debug('Theta = %s, expected score = %s, expected t.theta = %s', theta, expected_score, expected_llr[-1])

        # For some benchmark thetas, save r for each phase-space point
        if t == settings.theta_benchmark_nottrained:
            np.save(results_dir + '/r_nottrained_localmodel' + filename_addition + '.npy', np.exp(tt_test))

        elif t == settings.theta_benchmark_trained:
            np.save(results_dir + '/r_trained_localmodel' + filename_addition + '.npy', np.exp(tt_test))

    # Save expected LLR
    expected_llr = np.asarray(expected_llr)
    np.save(results_dir + '/llr_localmodel' + filename_addition + '.npy', expected_llr)

    # # Neyman construction
    # if do_neyman:
    #     logging.info('Starting evaluation of Neyman experiments')
    #     for t, theta in enumerate(settings.thetas):
    #
    #         # Contract estimated scores with delta theta
    #         delta_theta = theta - settings.thetas[theta1]
    #
    #         # Neyman construction: evaluate observed sample (raw)
    #         tt_neyman_observed = scores_neyman_observed.dot(delta_theta)
    #         llr_raw_neyman_observed = -2. * np.sum(tt_neyman_observed, axis=1)
    #         np.save(neyman_dir + '/neyman_llr_observed_localmodel_' + str(t) + filename_addition + '.npy',
    #                 llr_raw_neyman_observed)
    #
    #         # Neyman construction: loop over distribution samples generated from different thetas
    #         llr_neyman_distributions = []
    #         for tt in range(settings.n_thetas):
    #
    #             # Only evaluate certain combinations of thetas to save computation time
    #             if not decide_toy_evaluation(tt, t):
    #                 placeholder = np.empty(settings.n_neyman_null_experiments)
    #                 placeholder[:] = np.nan
    #                 llr_neyman_distributions.append(placeholder)
    #                 continue
    #
    #             # Neyman construction: load distribution scores
    #             scores_neyman_distribution = np.load(
    #                 settings.unweighted_events_dir + '/scores_neyman_distribution_' + str(tt) + '.npy')
    #             tt_neyman_distribution = scores_neyman_distribution.dot(delta_theta)
    #             llr_neyman_distributions.append(-2. * np.sum(tt_neyman_distribution, axis=1))
    #
    #         llr_neyman_distributions = np.asarray(llr_neyman_distributions)
    #         np.save(neyman_dir + '/neyman_llr_distribution_localmodel_' + str(t) + filename_addition + '.npy',
    #                 llr_neyman_distributions)
