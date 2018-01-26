################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import logging
import numpy as np


def calculate_median_p_value(llr_distribution, llr_observed):
    """ Calculates the median p-value given a set of observed LLR values and a distribution of LLR values given the
    hypothesis to test """

    distribution = np.sort(llr_distribution.flatten())
    p_values = 1. - np.searchsorted(distribution, llr_observed).astype('float') / len(distribution)
    return np.median(p_values)


def calculate_CL(filename, folder):
    # Settings
    neyman_dir = '../results/' + folder + '/neyman'
    output_dir = '../results/' + folder
    n_thetas = 1017
    theta_sm = 0

    # Load log likelihood ratios
    llr_distributions = []
    llr_observeds = []

    for t in range(n_thetas):
        try:
            llr_distributions.append(
                np.load(neyman_dir + '/neyman_llr_distribution_' + filename + '_' + str(t) + '.npy'))
        except IOError:
            llr_distributions.append(np.asarray([np.nan] * 10000))

        try:
            llr_observeds.append(
                np.load(neyman_dir + '/neyman_llr_observed_' + filename + '_' + str(t) + '.npy'))
        except IOError:
            llr_observeds.append(np.asarray([np.nan] * 10000))

    llr_distributions = np.asarray(llr_distributions).T  # Shape: (n_experiments, n_thetas)
    llr_observeds = np.asarray(llr_observeds).T  # Shape: (n_experiments, n_thetas)

    # Find MLE
    theta_mle = np.nanargmin(llr_observeds, axis=0)

    # Subtract MLE
    llr_compared_to_mle_distributions = np.zeros_like(llr_distributions)
    for exp in range(llr_distributions.shape[0]):
        llr_compared_to_mle_distributions[exp, :] = llr_distributions[exp, :] - llr_distributions[exp, theta_mle[exp]]

    llr_compared_to_mle_observeds = np.zeros_like(llr_observeds)
    for exp in range(llr_observeds.shape[0]):
        llr_compared_to_mle_observeds[exp, :] = llr_observeds[exp, :] - llr_observeds[exp, theta_mle[exp]]

    llr_compared_to_mle_distributions = llr_compared_to_mle_distributions.T  # Shape: (n_thetas, n_experiments)
    llr_compared_to_mle_observeds = llr_compared_to_mle_observeds.T  # Shape: (n_thetas, n_experiments)

    # Subtract SM
    llr_compared_to_sm_distributions = np.zeros_like(llr_distributions)
    for exp in range(llr_distributions.shape[0]):
        llr_compared_to_sm_distributions[exp, :] = llr_distributions[exp, :] - llr_distributions[exp, theta_sm]

    llr_compared_to_sm_observeds = np.zeros_like(llr_observeds)
    for exp in range(llr_observeds.shape[0]):
        llr_compared_to_sm_observeds[exp, :] = llr_observeds[exp, :] - llr_observeds[exp, theta_sm]

    llr_compared_to_sm_distributions = llr_compared_to_sm_distributions.T  # Shape: (n_thetas, n_experiments)
    llr_compared_to_sm_observeds = llr_compared_to_sm_observeds.T  # Shape: (n_thetas, n_experiments)

    # Calculate expected p values
    p_values_mle = np.zeros(n_thetas)
    p_values_sm = np.zeros(n_thetas)
    for t in range(n_thetas):
        p_values_mle[t] = calculate_median_p_value(llr_compared_to_mle_distributions[t, :],
                                                   llr_compared_to_mle_observeds[t, :])
        p_values_sm[t] = calculate_median_p_value(llr_compared_to_sm_distributions[t, :],
                                                  llr_compared_to_sm_observeds[t, :])

    np.save(output_dir + '/p_values_' + filename + '.npy', p_values_mle)
    np.save(output_dir + '/p_values_ratiotosm_' + filename + '.npy', p_values_sm)


def calculate_all_CL():

    logging.info('Starting p-value calculation')

    # calculate_CL('carl', 'point_by_point')
    # calculate_CL('carl_calibrated', 'point_by_point')
    # calculate_CL('regression', 'point_by_point')

    # calculate_CL('carl', 'parameterized')
    # calculate_CL('carl_calibrated', 'parameterized')
    # calculate_CL('score', 'parameterized')
    # calculate_CL('score_calibrated', 'parameterized')
    # calculate_CL('combined', 'parameterized')
    # calculate_CL('combined_calibrated', 'parameterized')
    # calculate_CL('regression', 'parameterized')
    # calculate_CL('combinedregression', 'parameterized')

    calculate_CL('scoreregression', 'score_regression')
    calculate_CL('scoreregression_calibrated', 'score_regression')
