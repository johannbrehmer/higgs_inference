################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import logging
import time
import numpy as np

from scipy.interpolate import LinearNDInterpolator

from carl.learning.calibration import NDHistogramCalibrator

from higgs_inference import settings
from higgs_inference.various.utils import calculate_mean_squared_error, r_from_s


def histo_inference(indices_X=None,
                    binning='optimized',
                    use_smearing=False,
                    denominator=0,
                    do_neyman=False,
                    options=''):
    """
    Approximates the likelihood throungh Approximate Frequentist Inference, a frequentist twist on ABC
    and effectively the same as kernel density estimation in the summary statistics space.

    :param use_smearing:
    :param indices_X: Defines which of the features to histogram.
    :param binning:
    :param use_smearing:
    :param do_neyman:
    :param options: Further options in a list of strings or string.
    """

    logging.info('Starting histogram-based inference')

    ################################################################################
    # Settings
    ################################################################################

    if indices_X is None:
        indices_X = [1, 41]  # pT(j1), delta_phi(jj)

    histogram_dimensionality = len(indices_X)

    filename_addition = '_' + str(histogram_dimensionality) + 'd'

    bins = binning

    # Manually chosen histogram binning
    if binning == 'optimized':

        if histogram_dimensionality == 2 and indices_X == [1, 41]:
            bins_pt = np.concatenate((
                np.linspace(0., 100., 6),  # steps of 20 GeV
                [130., 160., 200., 250., 300., 400., 600., 1000., 14000.]
            ))
            bins_deltaphi = np.linspace(0., np.pi, 11)
            bins = (bins_pt, bins_deltaphi)

        elif histogram_dimensionality == 2 and indices_X == [41, 1]:
            bins_pt = np.concatenate((
                np.linspace(0., 100., 6),  # steps of 20 GeV
                [130., 160., 200., 250., 300., 400., 600., 1000., 14000.]
            ))
            bins_deltaphi = np.linspace(0., np.pi, 11)
            bins = (bins_deltaphi, bins_pt)

        elif histogram_dimensionality == 1 and indices_X == [1]:
            bins_pt = np.concatenate((
                np.linspace(0., 200., 21),  # steps of 10 GeV
                np.linspace(225., 400., 8),  # steps of 25 GeV
                [450., 500., 600., 800., 1000., 14000.]
            ))
            bins = (bins_pt,)
            filename_addition = '_ptj'

        elif histogram_dimensionality == 1 and indices_X == [41]:
            bins_deltaphi = np.linspace(0., np.pi, 21)
            bins = (bins_deltaphi,)
            filename_addition = '_deltaphi'

        else:
            raise ValueError(indices_X)

    input_X_prefix = ''
    if use_smearing:
        input_X_prefix = 'smeared_'
        filename_addition += '_smeared'

    new_sample_mode = ('new' in options)
    neyman2_mode = ('neyman2' in options)
    neyman3_mode = ('neyman3' in options)

    input_filename_addition = ''
    if denominator > 0:
        input_filename_addition = '_denom' + str(denominator)
        filename_addition += '_denom' + str(denominator)

    if new_sample_mode:
        filename_addition += '_new'

    n_expected_events_neyman = settings.n_expected_events_neyman
    neyman_filename = 'neyman'
    if neyman2_mode:
        neyman_filename = 'neyman2'
        n_expected_events_neyman = settings.n_expected_events_neyman2
    if neyman3_mode:
        neyman_filename = 'neyman3'
        n_expected_events_neyman = settings.n_expected_events_neyman3

    results_dir = settings.base_dir + '/results/histo'
    neyman_dir = settings.neyman_dir + '/histo'

    logging.info('Settings:')
    logging.info('  Statistics:              x %s', indices_X)
    logging.info('  Binning:                 %s', binning)

    ################################################################################
    # Data
    ################################################################################

    X_test = np.load(
        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_test' + input_filename_addition + '.npy')
    r_test = np.load(settings.unweighted_events_dir + '/r_test' + input_filename_addition + '.npy')

    if do_neyman:
        X_neyman_alternate = np.load(
            settings.unweighted_events_dir + '/' + input_X_prefix + 'X_' + neyman_filename + '_alternate.npy')
    n_events_test = X_test.shape[0]

    ################################################################################
    # Loop over theta
    ################################################################################

    expected_llr = []
    mse_log_r = []
    trimmed_mse_log_r = []
    mse_log_r_train = []
    cross_entropies_train = []
    eval_times = []

    # Loop over the hypothesis thetas
    for i, t in enumerate(settings.extended_pbp_training_thetas):

        logging.info('Starting theta %s/%s: number %s (%s)',
                     i + 1, len(settings.extended_pbp_training_thetas), t, settings.thetas[t])

        # Load data
        new_sample_prefix = '_new' if new_sample_mode else ''
        X_train = np.load(
            settings.unweighted_events_dir + '/' + input_X_prefix + 'X_train_point_by_point_' + str(
                t) + input_filename_addition + new_sample_prefix + '.npy')
        y_train = np.load(
            settings.unweighted_events_dir + '/y_train_point_by_point_' + str(
                t) + input_filename_addition + new_sample_prefix + '.npy')
        r_train = np.load(
            settings.unweighted_events_dir + '/r_train_point_by_point_' + str(
                t) + input_filename_addition + new_sample_prefix + '.npy')

        # Construct summary statistics
        summary_statistics_train = X_train[:, indices_X]
        summary_statistics_test = X_test[:, indices_X]

        ################################################################################
        # "Training"
        ################################################################################

        histogram = NDHistogramCalibrator(bins=bins, range=None)

        histogram.fit(summary_statistics_train,
                      y_train)

        ################################################################################
        # Evaluation
        ################################################################################

        # Evaluation
        time_before = time.time()
        s_hat_test = histogram.predict(summary_statistics_test)
        eval_times.append(time.time() - time_before)
        log_r_hat_test = np.log(r_from_s(s_hat_test))
        s_hat_train = histogram.predict(summary_statistics_train)
        log_r_hat_train = np.log(r_from_s(s_hat_train))

        # Extract numbers of interest
        expected_llr.append(- 2. * settings.n_expected_events / n_events_test * np.sum(log_r_hat_test))
        mse_log_r.append(calculate_mean_squared_error(np.log(r_test[t]), log_r_hat_test, 0.))
        trimmed_mse_log_r.append(calculate_mean_squared_error(np.log(r_test[t]), log_r_hat_test, 'auto'))

        # For some benchmark thetas, save r for each phase-space point
        if t == settings.theta_benchmark_nottrained:
            np.save(results_dir + '/r_nottrained_histo' + filename_addition + '.npy', np.exp(log_r_hat_test))
        elif t == settings.theta_benchmark_trained:
            np.save(results_dir + '/r_trained_histo' + filename_addition + '.npy', np.exp(log_r_hat_test))

        # Calculate cross-entropy
        mse_log_r_train.append(calculate_mean_squared_error(np.log(r_train), log_r_hat_train, 0.))
        cross_entropy_train = - (y_train * np.log(s_hat_train)
                                 + (1. - y_train) * np.log(1. - s_hat_train)).astype(np.float128)
        cross_entropy_train = np.mean(cross_entropy_train)
        cross_entropies_train.append(cross_entropy_train)

        ################################################################################
        # Neyman construction toys
        ################################################################################

        if do_neyman:
            # Neyman construction: prepare alternate data and construct summary statistics
            summary_statistics_neyman_alternate = X_neyman_alternate.reshape((-1, X_neyman_alternate.shape[2]))
            summary_statistics_neyman_alternate = summary_statistics_neyman_alternate[:, indices_X]

            # Neyman construction: evaluate alternate sample
            s_hat_neyman_alternate = histogram.predict(summary_statistics_neyman_alternate)
            log_r_hat_neyman_alternate = np.log(r_from_s(s_hat_neyman_alternate))
            log_r_hat_neyman_alternate = log_r_hat_neyman_alternate.reshape((-1, n_expected_events_neyman))

            llr_neyman_alternate = -2. * np.sum(log_r_hat_neyman_alternate, axis=1)
            np.save(
                neyman_dir + '/' + neyman_filename + '_llr_alternate_' + str(t) + '_histo' + filename_addition + '.npy',
                llr_neyman_alternate)

            # # Neyman construction: old null
            # llr_neyman_nulls = []
            # for tt in range(settings.n_thetas):
            #
            #     # Only evaluate certain combinations of thetas to save computation time
            #     if not decide_toy_evaluation(tt, t):
            #         placeholder = np.empty(n_neyman_null_experiments)
            #         placeholder[:] = np.nan
            #         llr_neyman_nulls.append(placeholder)
            #         continue
            #
            #     # Neyman construction: load null sample and construct summary statistics
            #     X_neyman_null = np.load(
            #         settings.unweighted_events_dir + '/' + input_X_prefix + 'X_' + neyman_filename + '_null_' + str(
            #             tt) + '.npy')
            #     summary_statistics_neyman_null = X_neyman_null.reshape(
            #         (-1, X_neyman_null.shape[2]))[:, indices_X]
            #
            #     # Evaluation
            #     s_hat_neyman_null = histogram.predict(summary_statistics_neyman_null)
            #     log_r_hat_neyman_null = np.log(r_from_s(s_hat_neyman_null))
            #     log_r_hat_neyman_null = log_r_hat_neyman_null.reshape((-1, n_expected_events_neyman))
            #
            #     llr_neyman_nulls.append(-2. * np.sum(log_r_hat_neyman_null, axis=1))
            #
            # llr_neyman_nulls = np.asarray(llr_neyman_nulls)
            # np.save(neyman_dir + '/' + neyman_filename + '_llr_null_histo' + '_' + str(
            #     t) + filename_addition + '.npy',
            #         llr_neyman_nulls)

            # Neyman construction: null
            X_neyman_null = np.load(
                settings.unweighted_events_dir + '/' + input_X_prefix + 'X_' + neyman_filename + '_null_' + str(
                    t) + '.npy')
            summary_statistics_neyman_null = X_neyman_null.reshape(
                (-1, X_neyman_null.shape[2]))[:, indices_X]

            # Evaluation
            s_hat_neyman_null = histogram.predict(summary_statistics_neyman_null)
            log_r_hat_neyman_null = np.log(r_from_s(s_hat_neyman_null))
            log_r_hat_neyman_null = log_r_hat_neyman_null.reshape((-1, n_expected_events_neyman))
            llr_neyman_null = -2. * np.sum(log_r_hat_neyman_null, axis=1)
            np.save(neyman_dir + '/' + neyman_filename + '_llr_null_' + str(
                t) + '_histo' + filename_addition + '.npy',
                    llr_neyman_null)

            # Neyman construction: null evaluated at alternative
            if t == settings.theta_observed:
                for tt in settings.extended_pbp_training_thetas:
                    X_neyman_null = np.load(
                        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_' + neyman_filename + '_null_' + str(
                            tt) + '.npy')
                    summary_statistics_neyman_null = X_neyman_null.reshape(
                        (-1, X_neyman_null.shape[2]))[:, indices_X]

                    # Evaluation
                    s_hat_neyman_null = histogram.predict(summary_statistics_neyman_null)
                    log_r_hat_neyman_null = np.log(r_from_s(s_hat_neyman_null))
                    log_r_hat_neyman_null = log_r_hat_neyman_null.reshape((-1, n_expected_events_neyman))
                    llr_neyman_null = -2. * np.sum(log_r_hat_neyman_null, axis=1)
                    np.save(neyman_dir + '/' + neyman_filename + '_llr_nullatalternate_' + str(
                        tt) + '_histo' + filename_addition + '.npy',
                            llr_neyman_null)

    # Evaluation times
    logging.info('Evaluation timing: median %s s, mean %s s', np.median(eval_times), np.mean(eval_times))

    # Interpolate and save evaluation results
    expected_llr = np.asarray(expected_llr)
    mse_log_r = np.asarray(mse_log_r)
    trimmed_mse_log_r = np.asarray(trimmed_mse_log_r)
    cross_entropies_train = np.asarray(cross_entropies_train)

    logging.info('Interpolation')

    interpolator = LinearNDInterpolator(settings.thetas[settings.extended_pbp_training_thetas], expected_llr)
    expected_llr_all = interpolator(settings.thetas)
    np.save(results_dir + '/llr_histo' + filename_addition + '.npy', expected_llr_all)

    interpolator = LinearNDInterpolator(settings.thetas[settings.extended_pbp_training_thetas], mse_log_r)
    mse_log_r_all = interpolator(settings.thetas)
    np.save(results_dir + '/mse_logr_histo' + filename_addition + '.npy',
            mse_log_r_all)

    interpolator = LinearNDInterpolator(settings.thetas[settings.extended_pbp_training_thetas], trimmed_mse_log_r)
    trimmed_mse_log_r_all = interpolator(settings.thetas)
    np.save(results_dir + '/trimmed_mse_logr_histo' + filename_addition + '.npy',
            trimmed_mse_log_r_all)

    interpolator = LinearNDInterpolator(settings.thetas[settings.extended_pbp_training_thetas], cross_entropies_train)
    cross_entropy_train_mean = np.mean(interpolator(settings.thetas[settings.thetas_train]))
    logging.info('Training cross-entropy: %s', cross_entropy_train_mean)
    np.save(results_dir + '/cross_entropy_train_histo' + filename_addition + '.npy',
            [cross_entropy_train_mean])

    interpolator = LinearNDInterpolator(settings.thetas[settings.extended_pbp_training_thetas], mse_log_r_train)
    mse_log_r_train_mean = np.mean(interpolator(settings.thetas[settings.thetas_train]))
    logging.info('Training MSE log r: %s', mse_log_r_train_mean)
    np.save(results_dir + '/mse_logr_train_histo' + filename_addition + '.npy',
            [cross_entropy_train_mean])
