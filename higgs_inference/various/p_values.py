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
    p_values_left = 1. - np.searchsorted(distribution, llr_observed, side='left').astype('float') / len(distribution)
    p_values_right = 1. - np.searchsorted(distribution, llr_observed, side='right').astype('float') / len(distribution)
    p_values = 0.5 * (p_values_left + p_values_right)
    return np.median(p_values)


def subtract_mle(filename, folder, theta_sm=0):

    logging.info('Subtracting MLE for ' + folder + ' ' + filename)

    # Settings
    neyman_dir = '../results/' + folder + '/neyman'
    n_thetas = 1017

    # Load log likelihood ratios
    llr_distributions = []
    llr_observeds = []

    for t in range(n_thetas):
        try:
            llr_distributions.append(
                np.load(neyman_dir + '/neyman_llr_distribution_' + filename + '_' + str(t) + '.npy'))
        except IOError:
            #logging.debug('File ' + neyman_dir + '/neyman_llr_distribution_' + filename + '_' + str(t) + '.npy'
            #                + ' not found')
            llr_distributions.append(np.asarray([np.nan] * 1000))

        try:
            llr_observeds.append(
                np.load(neyman_dir + '/neyman_llr_observed_' + filename + '_' + str(t) + '.npy'))
        except IOError:
            #logging.debug('File ' + neyman_dir + '/neyman_llr_observed_' + filename + '_' + str(t) + '.npy'
            #                + ' not found')
            llr_observeds.append(np.asarray([np.nan] * 101))

    llr_distributions = np.asarray(llr_distributions).T  # Shape: (n_experiments, n_thetas)
    llr_observeds = np.asarray(llr_observeds).T  # Shape: (n_experiments, n_thetas)

    # Find MLE
    theta_mle_distribution = np.nanargmin(llr_distributions, axis=1)
    theta_mle_observed = np.nanargmin(llr_observeds, axis=1)
    logging.debug('MLE thetas: %s, %s', theta_mle_distribution, theta_mle_observed)

    # Subtract MLE
    llr_compared_to_mle_distributions = np.zeros_like(llr_distributions)
    for exp in range(llr_distributions.shape[0]):
        llr_compared_to_mle_distributions[exp, :] = llr_distributions[exp, :] - llr_distributions[exp, theta_mle_distribution[exp]]

    llr_compared_to_mle_observeds = np.zeros_like(llr_observeds)
    for exp in range(llr_observeds.shape[0]):
        llr_compared_to_mle_observeds[exp, :] = llr_observeds[exp, :] - llr_observeds[exp, theta_mle_observed[exp]]

    llr_compared_to_mle_distributions = llr_compared_to_mle_distributions.T  # Shape: (n_thetas, n_experiments)
    llr_compared_to_mle_observeds = llr_compared_to_mle_observeds.T  # Shape: (n_thetas, n_experiments)

    # Subtract true
    llr_compared_to_sm_distributions = np.zeros_like(llr_distributions)
    for exp in range(llr_distributions.shape[0]):
        llr_compared_to_sm_distributions[exp, :] = llr_distributions[exp, :] - llr_distributions[exp, theta_sm]

    llr_compared_to_sm_observeds = np.zeros_like(llr_observeds)
    for exp in range(llr_observeds.shape[0]):
        llr_compared_to_sm_observeds[exp, :] = llr_observeds[exp, :] - llr_observeds[exp, theta_sm]

    llr_compared_to_sm_distributions = llr_compared_to_sm_distributions.T  # Shape: (n_thetas, n_experiments)
    llr_compared_to_sm_observeds = llr_compared_to_sm_observeds.T  # Shape: (n_thetas, n_experiments)

    # Save results
    np.save(neyman_dir + '/neyman_llr_vs_mle_distributions_' + filename + '.npy',
            llr_compared_to_mle_distributions)
    np.save(neyman_dir + '/neyman_llr_vs_mle_observeds_' + filename + '.npy',
            llr_compared_to_mle_observeds)
    np.save(neyman_dir + '/neyman_llr_vs_true_distributions_' + filename + '.npy',
            llr_compared_to_sm_distributions)
    np.save(neyman_dir + '/neyman_llr_vs_true_observeds_' + filename + '.npy',
            llr_compared_to_sm_observeds)


def calculate_CL(filename, folder):

    # Preprocessing
    subtract_mle(filename, folder)

    logging.info('Calculating p-values for ' + folder + ' ' + filename)

    # Settings
    neyman_dir = '../results/' + folder + '/neyman'
    output_dir = '../results/'
    n_thetas = 1017

    # Load LLR
    llr_compared_to_mle_distributions = np.load(neyman_dir + '/neyman_llr_vs_mle_distributions_' + filename + '.npy')
    llr_compared_to_mle_observeds = np.load(neyman_dir + '/neyman_llr_vs_mle_observeds_' + filename + '.npy')
    llr_compared_to_sm_distributions = np.load(neyman_dir + '/neyman_llr_vs_sm_distributions_' + filename + '.npy')
    llr_compared_to_sm_observeds = np.load(neyman_dir + '/neyman_llr_vs_sm_observeds_' + filename + '.npy')

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

    calculate_CL('carl', 'point_by_point')
    calculate_CL('carl_calibrated', 'point_by_point')
    calculate_CL('regression', 'point_by_point')

    # calculate_CL('carl', 'parameterized')
    # calculate_CL('carl_calibrated', 'parameterized')
    # calculate_CL('score', 'parameterized')
    # calculate_CL('score_calibrated', 'parameterized')
    # calculate_CL('combined', 'parameterized')
    # calculate_CL('combined_calibrated', 'parameterized')
    calculate_CL('regression', 'parameterized')
    calculate_CL('combinedregression', 'parameterized')

    calculate_CL('scoreregression', 'score_regression')
    calculate_CL('scoreregression_calibrated', 'score_regression')
