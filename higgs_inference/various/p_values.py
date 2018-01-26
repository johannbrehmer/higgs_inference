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

    # Go!
    median_p_values = []

    try:
        for t in range(n_thetas):
            llr_distribution = np.load(neyman_dir + '/neyman_llr_distribution_' + filename + '_' + str(t) + '.npy')
            llr_observed = np.load(neyman_dir + '/neyman_llr_observed_' + filename + '_' + str(t) + '.npy')

            median_p_values.append(calculate_median_p_value(llr_distribution, llr_observed))

        median_p_values = np.array(median_p_values)

        np.save(output_dir + '/p_values_' + filename + '.npy', median_p_values)

    except IOError:
        logging.warning('Neyman files %s in folder %s not found', filename, folder)


def calculate_all_CL():
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
