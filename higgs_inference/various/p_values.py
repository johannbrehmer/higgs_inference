################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import logging
import numpy as np

from higgs_inference import settings


def calculate_median_p_value(llr_distribution, llr_observed):
    """ Calculates the median p-value given a set of observed LLR values and a distribution of LLR values given the
    hypothesis to test """

    distribution = np.sort(llr_distribution.flatten())

    p_values_left = 1. - np.searchsorted(distribution, llr_observed, side='left').astype('float') / len(distribution)
    p_values_right = 1. - np.searchsorted(distribution, llr_observed, side='right').astype('float') / len(distribution)
    p_values = 0.5 * (p_values_left + p_values_right)

    return np.median(p_values)


def subtract_mle(filename, folder, theta_sm=0):
    """ For a given filename and folder, takes the log likelihood ratios with respect to some arbitrary denominator
    theta and subtracts the log likelihood ratios of the maximum likelihood estimators. """

    logging.info('Subtracting MLE for ' + folder + ' ' + filename)

    # Settings
    neyman_dir = settings.neyman_dir + '/' + folder
    result_dir = settings.base_dir + '/results/' + folder
    n_thetas = settings.n_thetas

    # Load log likelihood ratios
    llr_distributions = []
    llr_observeds = []

    files_found = 0
    files_not_found = 0

    for t in range(n_thetas):
        try:
            entry = np.load(neyman_dir + '/neyman_llr_distribution_' + filename + '_' + str(t) + '.npy')
            assert entry.shape == (settings.n_thetas, settings.n_neyman_distribution_experiments)
            llr_distributions.append(entry)
            files_found += 1

        except (IOError, AssertionError):
            placeholder = np.empty((settings.n_thetas, settings.n_neyman_distribution_experiments))
            placeholder[:,:] = np.nan
            llr_distributions.append(placeholder)
            files_not_found += 1

        try:
            llr_observeds.append(
                np.load(neyman_dir + '/neyman_llr_observed_' + filename + '_' + str(t) + '.npy'))
            files_found += 1
        except IOError:
            placeholder = np.empty(settings.n_neyman_observed_experiments)
            placeholder[:] = np.nan
            llr_observeds.append(placeholder)
            files_not_found += 1

    logging.debug("Found %s files, didn't find %s files", files_found, files_not_found)

    llr_distributions = np.asarray(llr_distributions)  # Shape: (n_thetas_eval, n_thetas_assumed_true, n_experiments)
    llr_observeds = np.asarray(llr_observeds)  # Shape: (n_thetas_eval, n_experiments)

    # Find MLE
    theta_mle_distribution = np.nanargmin(llr_distributions, axis=0)  # Shape: (n_thetas_assumed_true, n_experiments)
    theta_mle_observed = np.nanargmin(llr_observeds, axis=0)  # Shape: (n_experiments,)
    # logging.debug('MLE thetas: %s, %s', theta_mle_distribution, theta_mle_observed)

    # Subtract MLE
    llr_compared_to_mle_distributions = np.zeros( (llr_distributions.shape[1], llr_distributions.shape[2]) )
    for t_true in range(llr_distributions.shape[1]):
        for exp in range(llr_distributions.shape[2]):
            llr_compared_to_mle_distributions[t_true, exp] = (llr_distributions[t_true, t_true, exp]
                                                                      - llr_distributions[theta_mle_distribution[
                                                                                              t_true, exp], t_true, exp])

    llr_compared_to_mle_observeds = np.zeros_like(llr_observeds)
    for t_eval in range(llr_observeds.shape[0]):
        for exp in range(llr_observeds.shape[1]):
            llr_compared_to_mle_observeds[t_eval, exp] = (llr_observeds[t_eval, exp]
                                                          - llr_observeds[theta_mle_observed[exp], exp])

    # Subtract true
    # llr_compared_to_true_distributions = np.zeros_like(llr_distributions)
    # for t_eval in range(llr_distributions.shape[0]):
    #     for t_true in range(llr_distributions.shape[1]):
    #         for exp in range(llr_distributions.shape[2]):
    #             llr_compared_to_true_distributions[t_eval, t_true, exp] = (llr_distributions[t_eval, t_true, exp]
    #                                                                        - llr_distributions[t_true, t_true, exp])

    # llr_compared_to_true_observeds = np.zeros_like(llr_observeds)
    # for t_eval in range(llr_observeds.shape[0]):
    #     for exp in range(llr_observeds.shape[1]):
    #         llr_compared_to_true_observeds[t_eval, exp] = (llr_observeds[t_eval, exp]
    #                                                        - llr_observeds[theta_sm, exp])

    # Save results
    np.save(result_dir + '/neyman_llr_vs_mle_distributions_' + filename + '.npy',
            llr_compared_to_mle_distributions)
    np.save(result_dir + '/neyman_llr_vs_mle_observeds_' + filename + '.npy',
            llr_compared_to_mle_observeds)
    # np.save(result_dir + '/neyman_llr_vs_true_distributions_' + filename + '.npy',
    #         llr_compared_to_true_distributions)
    # np.save(result_dir + '/neyman_llr_vs_true_observeds_' + filename + '.npy',
    #         llr_compared_to_true_observeds)


def calculate_CL(filename, folder):
    """ Steers the calculation of all p-values for a given filename and folder. """

    # Preprocessing
    try:
        subtract_mle(filename, folder)
    except ValueError:
        logging.warning('Error in MLE subtraction, skipping set')
        return

    logging.info('Calculating p-values for ' + folder + ' ' + filename)

    # Settings
    result_dir = settings.base_dir + '/results/' + folder
    n_thetas = 1017

    # Load LLR
    llr_compared_to_mle_distributions = np.load(result_dir + '/neyman_llr_vs_mle_distributions_' + filename + '.npy')
    llr_compared_to_mle_observeds = np.load(result_dir + '/neyman_llr_vs_mle_observeds_' + filename + '.npy')
    #llr_compared_to_true_distributions = np.load(result_dir + '/neyman_llr_vs_true_distributions_' + filename + '.npy')
    #llr_compared_to_true_observeds = np.load(result_dir + '/neyman_llr_vs_true_observeds_' + filename + '.npy')

    # Calculate expected p values
    p_values_mle = np.zeros(n_thetas)
    #p_values_true = np.zeros(n_thetas)
    for t in range(n_thetas):
        p_values_mle[t] = calculate_median_p_value(llr_compared_to_mle_distributions[t, :],
                                                   llr_compared_to_mle_observeds[t, :])
        #p_values_true[t] = calculate_median_p_value(llr_compared_to_true_distributions[t, :],
        #                                            llr_compared_to_true_observeds[t, :])

    np.save(result_dir + '/p_values_' + filename + '.npy', p_values_mle)
    #np.save(result_dir + '/p_values_ratio_vs_true_' + filename + '.npy', p_values_true)


def calculate_all_CL():
    """ Starts the p-value calculation for all inference strategies."""

    logging.info('Starting p-value calculation')

    calculate_CL('truth', 'truth')

    calculate_CL('carl', 'point_by_point')
    calculate_CL('carl_calibrated', 'point_by_point')
    calculate_CL('regression', 'point_by_point')

    calculate_CL('carl', 'parameterized')
    calculate_CL('carl_calibrated', 'parameterized')
    calculate_CL('score', 'parameterized')
    calculate_CL('score_calibrated', 'parameterized')
    calculate_CL('combined', 'parameterized')
    calculate_CL('combined_calibrated', 'parameterized')
    calculate_CL('regression', 'parameterized')
    calculate_CL('combinedregression', 'parameterized')

    calculate_CL('scoreregression', 'score_regression')
    calculate_CL('scoreregression_calibrated', 'score_regression')
