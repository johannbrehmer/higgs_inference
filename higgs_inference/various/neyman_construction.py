################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import math

from higgs_inference import settings
from higgs_inference.various.utils import weighted_quantile


def calculate_self_convolutions(x, convolutions, xmin, xmax, nbins):
    norm = nbins / (xmax - xmin)

    histo, _ = np.histogram(x, bins=nbins, range=(xmin, xmax))
    histo = histo * norm / np.sum(histo)

    convolved_histo = histo
    for c in range(convolutions):
        convolved_histo = np.convolve(convolved_histo, histo, mode='same')
        convolved_histo = convolved_histo * norm / np.sum(convolved_histo)
    return convolved_histo


def neyman_construction(test_statistics_null, test_statistics_alternate, n_self_convolutions=0):
    null = test_statistics_null
    alternate = test_statistics_alternate

    if not (np.all(np.isfinite(null)) and np.all(np.isfinite(alternate))):
        return None, [None] * 3, [None] * 3, None, None, None

    # Self convolutions: require histogram
    if n_self_convolutions > 0:
        xmin = settings.neyman_convolution_min
        xmax = settings.neyman_convolution_max
        nbins = settings.neyman_convolution_bins
        xvals = np.linspace(xmin + 0.5 * (xmax - xmin) / nbins, xmax - 0.5 * (xmax - xmin) / nbins, nbins)

        # Calculate self-convolutions
        null_histo = calculate_self_convolutions(null, n_self_convolutions, xmin, xmax, nbins)
        alternate_histo = calculate_self_convolutions(alternate, n_self_convolutions, xmin, xmax, nbins)

        # Calculate alternate median
        q_median = weighted_quantile(xvals, 0.5, alternate_histo)

        # Calculate null boundaries for given CLs
        q_cuts = []
        for cl in settings.confidence_levels:
            q_cuts.append(weighted_quantile(xvals, cl, null_histo))
        q_cut_uncertainties = np.ones_like(q_cuts) * 0.5 * (xmax - xmin) / nbins

        # Calculate p-values
        q_median_histo_index = int(math.floor((q_median - xmin) / (xmax - xmin) * nbins))
        q_median_histo_index = max(q_median_histo_index, 0)
        q_median_histo_index = min(q_median_histo_index, nbins - 1)

        p_value = ((np.sum(alternate_histo[q_median_histo_index:]) - 0.5 * alternate_histo[q_median_histo_index])
                   / np.sum(alternate_histo[:]))

    else:
        # Sort
        null = np.sort(null.flatten())

        # Calculate p-values
        p_values_left = 1. - np.searchsorted(null, alternate, side='left').astype('float') / len(null)
        p_values_right = 1. - np.searchsorted(null, alternate, side='right').astype('float') / len(null)
        p_value = np.median(0.5 * (p_values_left + p_values_right))

        # Calculate alternate median
        q_median = np.median(alternate)

        # Calculate null boundaries for given CLs
        q_cuts = []
        q_cut_uncertainties = []
        for cl in settings.confidence_levels:
            q_cut_index = int(cl * len(test_statistics_null)) - 1
            q_cuts.append((null[q_cut_index] + null[q_cut_index + 1]) / 2)
            q_cut_uncertainties.append((null[q_cut_index + 1] - null[q_cut_index]) / 2)

        null_histo = None
        alternate_histo = None

    return p_value, np.asarray(q_cuts), np.asarray(q_cut_uncertainties), q_median, null_histo, alternate_histo


def calculate_observed_test_statistics(test_statistics_alternate, n_observed_events, n_experiments):

    observed_q = []
    for i in range(n_experiments):
        observed_q.append(np.sum(test_statistics_alternate[i*n_observed_events:(i+1)*n_observed_events]))

    return np.asarray(observed_q)


# def subtract_mle(filename, filename_suffix, folder, neyman2_mode=False):
#     """ For a given filename and folder, takes the log likelihood ratios with respect to some arbitrary denominator
#     theta and subtracts the log likelihood ratios of the maximum likelihood estimators. """
#
#     logging.info('Subtracting MLE for ' + folder + ' ' + filename + filename_suffix)
#
#     # Settings
#     neyman_dir = settings.neyman_dir + '/' + folder
#     result_dir = settings.base_dir + '/results/' + folder
#     n_thetas = settings.n_thetas
#
#     n_expected_events_neyman = settings.n_expected_events_neyman
#     n_neyman_null_experiments = settings.n_neyman_null_experiments
#     n_neyman_alternate_experiments = settings.n_neyman_alternate_experiments
#     neyman_filename = 'neyman'
#     if neyman2_mode:
#         neyman_filename = 'neyman2'
#         n_expected_events_neyman = settings.n_expected_events_neyman2
#         n_neyman_null_experiments = settings.n_neyman2_null_experiments
#         n_neyman_alternate_experiments = settings.n_neyman2_alternate_experiments
#
#     # Load log likelihood ratios
#     llr_nulls = []
#     llr_alternates = []
#
#     files_found = 0
#     files_not_found = 0
#     files_wrong_shape = 0
#
#     for t in range(n_thetas):
#         try:
#             entry = np.load(neyman_dir + '/' + neyman_filename + '_llr_null_' + filename + '_' + str(
#                 t) + filename_suffix + '.npy')
#             assert entry.shape == (settings.n_thetas, n_neyman_null_experiments)
#             llr_nulls.append(entry)
#             files_found += 1
#
#         except IOError as err:
#             logging.debug("Error loading file: %s", err)
#             # logging.debug("Didn't find file %s", neyman_dir + '/neyman_llr_null_' + filename + '_' + str(t) + '.npy')
#
#             placeholder = np.empty((settings.n_thetas, n_neyman_null_experiments))
#             placeholder[:, :] = np.nan
#             llr_nulls.append(placeholder)
#
#             files_not_found += 1
#
#         except AssertionError as err:
#             logging.warning("File %s has wrong shape %s",
#                             neyman_dir + '/' + neyman_filename + '_llr_null_' + filename + '_' + str(
#                                 t) + filename_suffix + '.npy', entry.shape)
#
#             placeholder = np.empty((settings.n_thetas, n_neyman_null_experiments))
#             placeholder[:, :] = np.nan
#             llr_nulls.append(placeholder)
#
#             files_wrong_shape += 1
#
#         try:
#             llr_alternates.append(
#                 np.load(neyman_dir + '/' + neyman_filename + '_llr_alternate_' + filename + '_' + str(
#                     t) + filename_suffix + '.npy'))
#             files_found += 1
#
#         except IOError as err:
#             logging.debug("Error loading file: %s", err)
#             # logging.debug("Didn't find file %s", neyman_dir + '/neyman_llr_alternate_' + filename + '_' + str(t) + '.npy')
#
#             placeholder = np.empty(n_neyman_alternate_experiments)
#             placeholder[:] = np.nan
#             llr_alternates.append(placeholder)
#
#             files_not_found += 1
#
#     logging.debug("Found %s files, didn't find %s files, found %s files with wrong shape", files_found, files_not_found,
#                   files_wrong_shape)
#
#     llr_nulls = np.asarray(llr_nulls)  # Shape: (n_thetas_eval, n_thetas_assumed_true, n_experiments)
#     llr_alternates = np.asarray(llr_alternates)  # Shape: (n_thetas_eval, n_experiments)
#
#     # Check that some results exist
#     insufficient_data_nulls = np.any(np.all(np.invert(np.isfinite(llr_nulls)), axis=0))
#     insufficient_data_alternate = np.any(np.all(np.invert(np.isfinite(llr_alternates)), axis=0))
#     if insufficient_data_nulls:
#         logging.warning("Insufficient data to find MLEs for null")
#         logging.debug("NaNs nulls:\n%s", np.all(np.invert(np.isfinite(llr_nulls)), axis=0))
#         logging.debug("NaNs alternate:\n%s", np.all(np.invert(np.isfinite(llr_alternates)), axis=0))
#     if insufficient_data_alternate:
#         logging.warning("Insufficient data to find MLEs for alternate")
#         logging.debug("NaNs nulls:\n%s", np.all(np.invert(np.isfinite(llr_nulls)), axis=0))
#         logging.debug("NaNs alternate:\n%s", np.all(np.invert(np.isfinite(llr_alternates)), axis=0))
#     if insufficient_data_alternate or insufficient_data_nulls:
#         raise ValueError
#
#     # Find MLE
#     theta_mle_null = np.nanargmin(llr_nulls, axis=0)  # Shape: (n_thetas_assumed_true, n_experiments)
#     theta_mle_alternate = np.nanargmin(llr_alternates, axis=0)  # Shape: (n_experiments,)
#
#     # Subtract MLE
#     llr_compared_to_mle_nulls = np.zeros((llr_nulls.shape[1], llr_nulls.shape[2]))
#     for t_true in range(llr_nulls.shape[1]):
#         for exp in range(llr_nulls.shape[2]):
#             llr_compared_to_mle_nulls[t_true, exp] = (llr_nulls[t_true, t_true, exp]
#                                                               - llr_nulls[
#                                                                   theta_mle_null[t_true, exp], t_true, exp])
#
#     llr_compared_to_mle_alternates = np.zeros_like(llr_alternates)
#     for t_eval in range(llr_alternates.shape[0]):
#         for exp in range(llr_alternates.shape[1]):
#             llr_compared_to_mle_alternates[t_eval, exp] = (llr_alternates[t_eval, exp]
#                                                           - llr_alternates[theta_mle_alternate[exp], exp])
#
#     # Save results
#     np.save(result_dir + '/' + neyman_filename + '_llr_vs_mle_nulls_' + filename + filename_suffix + '.npy',
#             llr_compared_to_mle_nulls)
#     np.save(result_dir + '/' + neyman_filename + '_llr_vs_mle_alternates_' + filename + filename_suffix + '.npy',
#             llr_compared_to_mle_alternates)


def subtract_sm(filename, folder, neyman_set=1):
    """ For a given filename and folder, takes the log likelihood ratios with respect to some arbitrary denominator
    theta and subtracts the log likelihood ratios of the estimators. """

    logging.info('Subtracting SM for ' + folder + ' ' + filename)

    # Settings
    t_alternate = settings.theta_observed
    neyman_dir = settings.neyman_dir + '/' + folder
    n_thetas = settings.n_thetas

    n_neyman_null_experiments = settings.n_neyman_null_experiments
    n_neyman_alternate_experiments = settings.n_neyman_alternate_experiments
    neyman_filename = 'neyman'
    if neyman_set == 2:
        neyman_filename = 'neyman2'
        n_neyman_null_experiments = settings.n_neyman2_null_experiments
        n_neyman_alternate_experiments = settings.n_neyman2_alternate_experiments
    elif neyman_set == 3:
        neyman_filename = 'neyman3'
        n_neyman_null_experiments = settings.n_neyman3_null_experiments
        n_neyman_alternate_experiments = settings.n_neyman3_alternate_experiments

    # Load log likelihood ratios
    llr_nulls = []
    llr_alternates = []
    llr_nullsatalternate = []

    files_found = 0
    files_not_found = 0
    files_wrong_shape = 0

    for t in range(n_thetas):
        try:
            entry = np.load(neyman_dir + '/' + neyman_filename + '_llr_alternate_' + str(t) + '_' + filename + '.npy')
            assert entry.shape == (n_neyman_alternate_experiments,)
            llr_alternates.append(entry)
            files_found += 1
        except IOError as err:
            if files_not_found < 10:
                logging.debug("Error loading file: %s", err)
            placeholder = np.empty((n_neyman_alternate_experiments,))
            placeholder[:] = np.nan
            llr_alternates.append(placeholder)
            files_not_found += 1
        except AssertionError:
            logging.warning("File %s has wrong shape %s",
                            neyman_dir + '/' + neyman_filename + '_llr_alternate_' + str(t) + '_' + filename + '.npy',
                            entry.shape)
            placeholder = np.empty((n_neyman_alternate_experiments,))
            placeholder[:] = np.nan
            llr_alternates.append(placeholder)
            files_wrong_shape += 1

        try:
            entry = np.load(neyman_dir + '/' + neyman_filename + '_llr_null_' + str(t) + '_' + filename + '.npy')
            assert entry.shape == (n_neyman_null_experiments,)
            llr_nulls.append(entry)
            files_found += 1
        except IOError as err:
            if files_not_found < 10:
                logging.debug("Error loading file: %s", err)
            placeholder = np.empty((n_neyman_null_experiments,))
            placeholder[:] = np.nan
            llr_nulls.append(placeholder)
            files_not_found += 1
        except AssertionError:
            logging.warning("File %s has wrong shape %s",
                            neyman_dir + '/' + neyman_filename + '_llr_null_' + str(t) + '_' + filename + '.npy',
                            entry.shape)
            placeholder = np.empty((n_neyman_null_experiments,))
            placeholder[:] = np.nan
            llr_nulls.append(placeholder)
            files_wrong_shape += 1

        try:
            entry = np.load(
                neyman_dir + '/' + neyman_filename + '_llr_nullatalternate_' + str(t) + '_' + filename + '.npy')
            assert entry.shape == (n_neyman_null_experiments,)
            llr_nullsatalternate.append(entry)
            files_found += 1
        except IOError as err:
            if files_not_found < 10:
                logging.debug("Error loading file: %s", err)
            placeholder = np.empty((n_neyman_null_experiments,))
            placeholder[:] = np.nan
            llr_nullsatalternate.append(placeholder)
            files_not_found += 1
        except AssertionError:
            logging.warning("File %s has wrong shape %s",
                            neyman_dir + '/' + neyman_filename + '_llr_nullatalternate_' + str(
                                t) + '_' + filename + '.npy', entry.shape)
            placeholder = np.empty((n_neyman_null_experiments,))
            placeholder[:] = np.nan
            llr_nullsatalternate.append(placeholder)
            files_wrong_shape += 1

    logging.debug("Found %s files, didn't find %s files, found %s files with wrong shape", files_found, files_not_found,
                  files_wrong_shape)

    llr_nulls = np.asarray(llr_nulls)  # Shape: (n_thetas, n_experiments)
    llr_nullsatalternate = np.asarray(llr_nullsatalternate)  # Shape: (n_thetas, n_experiments)
    llr_alternates = np.asarray(llr_alternates)  # Shape: (n_thetas, n_experiments)

    # # Check that some results exist
    # insufficient_data_nulls = np.any(np.all(np.invert(np.isfinite(llr_nulls)), axis=0))
    # insufficient_data_alternate = np.any(np.all(np.invert(np.isfinite(llr_alternates)), axis=0))
    # if insufficient_data_nulls:
    #     logging.warning("Insufficient data to find MLEs for null")
    #     logging.debug("NaNs nulls:\n%s", np.all(np.invert(np.isfinite(llr_nulls)), axis=0))
    #     logging.debug("NaNs alternate:\n%s", np.all(np.invert(np.isfinite(llr_alternates)), axis=0))
    # if insufficient_data_alternate:
    #     logging.warning("Insufficient data to find MLEs for alternate")
    #     logging.debug("NaNs nulls:\n%s", np.all(np.invert(np.isfinite(llr_nulls)), axis=0))
    #     logging.debug("NaNs alternate:\n%s", np.all(np.invert(np.isfinite(llr_alternates)), axis=0))
    # if insufficient_data_alternate or insufficient_data_nulls:
    #     raise ValueError

    # Subtract SM values
    llr_vs_true_nulls = np.zeros_like(llr_nulls)
    for t in range(llr_nulls.shape[0]):
        for exp in range(llr_nulls.shape[1]):
            llr_vs_true_nulls[t, exp] = (llr_nulls[t, exp] - llr_nullsatalternate[t, exp])

    llr_vs_true_alternates = np.zeros_like(llr_alternates)
    for t in range(llr_alternates.shape[0]):
        for exp in range(llr_alternates.shape[1]):
            llr_vs_true_alternates[t, exp] = (llr_alternates[t, exp] - llr_alternates[t_alternate, exp])

    # Save results
    np.save(neyman_dir + '/' + neyman_filename + '_llr_vs_sm_nulls_' + filename + '.npy', llr_vs_true_nulls)
    np.save(neyman_dir + '/' + neyman_filename + '_llr_vs_sm_alternates_' + filename + '.npy', llr_vs_true_alternates)


def find_mle(filename, folder, neyman_set=1):
    logging.info('Finding MLE distribution under alternate for ' + folder + ' ' + filename)

    # Settings
    neyman_dir = settings.neyman_dir + '/' + folder
    result_dir = settings.base_dir + '/results/' + folder
    random_thetas = settings.thetas_train

    n_neyman_alternate_experiments = settings.n_neyman_alternate_experiments
    neyman_filename = 'neyman'
    if neyman_set == 2:
        neyman_filename = 'neyman2'
        n_neyman_alternate_experiments = settings.n_neyman2_alternate_experiments
    elif neyman_set == 3:
        neyman_filename = 'neyman3'
        n_neyman_alternate_experiments = settings.n_neyman3_alternate_experiments

    # Load log likelihood ratios
    llr_alternates = []

    files_found = 0
    files_not_found = 0
    files_wrong_shape = 0

    placeholder = np.empty((n_neyman_alternate_experiments,))
    placeholder[:] = np.nan

    for t in range(settings.n_thetas):
        if t in random_thetas:
            try:
                entry = np.load(
                    neyman_dir + '/' + neyman_filename + '_llr_alternate_' + str(t) + '_' + filename + '.npy')
                assert entry.shape == (n_neyman_alternate_experiments,)
                llr_alternates.append(entry)
                files_found += 1
            except IOError as err:
                if files_not_found < 10:
                    logging.debug("Error loading file: %s", err)
                llr_alternates.append(placeholder)
                files_not_found += 1
            except AssertionError:
                logging.warning("File %s has wrong shape %s",
                                neyman_dir + '/' + neyman_filename + '_llr_alternate_' + str(
                                    t) + '_' + filename + '.npy',
                                entry.shape)
                llr_alternates.append(placeholder)
                files_wrong_shape += 1
        else:
            llr_alternates.append(placeholder)

    logging.debug("Found %s files, didn't find %s files, found %s files with wrong shape", files_found, files_not_found,
                  files_wrong_shape)

    llr_alternates = np.asarray(llr_alternates)  # Shape: (n_thetas, n_experiments)

    # Find MLE
    theta_mle_alternate = np.nanargmin(llr_alternates, axis=0)  # Shape: (n_experiments,)

    # Save results
    np.save(result_dir + '/' + neyman_filename + '_mle_' + filename + '.npy', theta_mle_alternate)


def calculate_confidence_limits(filename, folder, neyman_set=1):
    """ Steers the calculation of all p-values for a given filename and folder. """

    # Find and save MLEs
    # try:
    #    find_mle(filename, folder, neyman_set)
    # except ValueError as err:
    #    logging.warning('Error in MLE determination: %s', err)

    # Preprocessing
    try:
        subtract_sm(filename, folder, neyman_set)
    except ValueError:
        logging.warning('Error in SM subtraction, skipping set')
        return

    logging.info('Neyman construction for ' + folder + '/' + filename)

    # Settings
    neyman_dir = settings.neyman_dir + '/' + folder
    result_dir = settings.base_dir + '/results/' + folder
    n_thetas = settings.n_thetas

    neyman_filename = 'neyman'
    n_self_convolutions = settings.n_convolutions_neyman
    if neyman_set == 2:
        neyman_filename = 'neyman2'
        n_self_convolutions = settings.n_convolutions_neyman2
    if neyman_set == 3:
        neyman_filename = 'neyman3'
        n_self_convolutions = settings.n_convolutions_neyman3

    # Load LLR
    llr_vs_true_nulls = np.load(neyman_dir + '/' + neyman_filename + '_llr_vs_sm_nulls_' + filename + '.npy')
    llr_vs_true_alternates = np.load(neyman_dir + '/' + neyman_filename + '_llr_vs_sm_alternates_' + filename + '.npy')

    # Quantities to calculate
    p_values = np.zeros(n_thetas)
    q_cut_values = np.zeros((n_thetas, len(settings.confidence_levels)))
    q_cut_uncertainties = np.zeros((n_thetas, len(settings.confidence_levels)))
    null_histos = np.zeros((n_thetas, settings.neyman_convolution_bins))
    alternate_histos = np.zeros((n_thetas, settings.neyman_convolution_bins))
    q_median_values = np.zeros(n_thetas)
    q_observed_values = np.zeros((n_thetas, 10))

    # Go!
    for t in range(n_thetas):
        results = neyman_construction(llr_vs_true_nulls[t, :], llr_vs_true_alternates[t, :], n_self_convolutions)
        (p_values[t], q_cut_values[t, :], q_cut_uncertainties[t, :], q_median_values[t], null_histos[t, :],
         alternate_histos[t, :]) = results

        q_observed_values[t, :] = calculate_observed_test_statistics(llr_vs_true_alternates[t, :],
                                                                     settings.n_expected_events, 10)

    np.save(result_dir + '/' + neyman_filename + '_pvalues_' + filename + '.npy', p_values)
    np.save(result_dir + '/' + neyman_filename + '_qcut_' + filename + '.npy', q_cut_values)
    np.save(result_dir + '/' + neyman_filename + '_qcut_uncertainties_' + filename + '.npy', q_cut_uncertainties)
    np.save(result_dir + '/' + neyman_filename + '_qmedian_' + filename + '.npy', q_median_values)
    np.save(result_dir + '/' + neyman_filename + '_qobserved_' + filename + '.npy', q_observed_values)


def start_cl_calculation(options=''):
    """ Starts the p-value calculation for all inference strategies."""

    neyman_set = 1
    if 'neyman2' in options:
        neyman_set = 2
    if 'neyman3' in options:
        neyman_set = 3

    logging.info('Starting p-value calculation')

    #calculate_confidence_limits('truth', 'truth', neyman_set)

    calculate_confidence_limits('histo_2d_asymmetricbinning', 'histo', neyman_set)
    #calculate_confidence_limits('scoreregression_rotatedscore_deep', 'score_regression', neyman_set)
    #calculate_confidence_limits('carl_calibrated_shallow', 'parameterized', neyman_set)
    #calculate_confidence_limits('combined_calibrated_deep', 'parameterized', neyman_set)
    #calculate_confidence_limits('regression_calibrated', 'parameterized', neyman_set)
    #calculate_confidence_limits('combinedregression_calibrated_deep', 'parameterized', neyman_set)

    calculate_confidence_limits('histo_2d_asymmetricbinning_smeared', 'histo', neyman_set)
    #calculate_confidence_limits('scoreregression_rotatedscore_deep_smeared', 'score_regression', neyman_set)
    #calculate_confidence_limits('carl_calibrated_shallow_smeared', 'parameterized', neyman_set)
    #calculate_confidence_limits('combined_calibrated_deep_smeared', 'parameterized', neyman_set)
    #calculate_confidence_limits('regression_calibrated_smeared', 'parameterized', neyman_set)
    #calculate_confidence_limits('combinedregression_calibrated_deep_smeared', 'parameterized', neyman_set)
