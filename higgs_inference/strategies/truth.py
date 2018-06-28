################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import logging
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern

from carl.learning.calibration import IsotonicCalibrator

from higgs_inference import settings
from higgs_inference.various.utils import s_from_r, r_from_s


def truth_inference(do_neyman=False,
                    denominator=0,
                    options=''):

    """
    Ground truth for likelihood ratios.
    :param do_neyman: Switches on the evaluation of toy experiments for the Neyman construction.
    :param denominator: Which of five predefined denominator (reference) hypotheses to use.
    :param options: Further options in a list of strings or string.  'new' changes the samples. 'neyman2' and
                    'neyman3' change the Neyman construction sample.
    """

    logging.info('Starting truth calculation')

    ################################################################################
    # Settings
    ################################################################################

    new_sample_mode = ('new' in options)
    neyman2_mode = ('neyman2' in options)
    neyman3_mode = ('neyman3' in options)

    filename_addition = ''
    input_filename_addition = ''
    theta1 = settings.theta1_default
    if denominator > 0:
        input_filename_addition = '_denom' + str(denominator)
        filename_addition += '_denom' + str(denominator)
        theta1 = settings.theta1_alternatives[denominator - 1]

    if new_sample_mode:
        filename_addition += '_new'
        input_filename_addition += '_new'

    neyman_dir = settings.neyman_dir + '/truth'
    results_dir = settings.base_dir + '/results/truth'

    n_expected_events_neyman = settings.n_expected_events_neyman
    n_neyman_null_experiments = settings.n_neyman_null_experiments
    n_neyman_alternate_experiments = settings.n_neyman_alternate_experiments
    neyman_filename = 'neyman'
    if neyman2_mode:
        neyman_filename = 'neyman2'
        n_expected_events_neyman = settings.n_expected_events_neyman2
        n_neyman_null_experiments = settings.n_neyman2_null_experiments
        n_neyman_alternate_experiments = settings.n_neyman2_alternate_experiments
    if neyman3_mode:
        neyman_filename = 'neyman3'
        n_expected_events_neyman = settings.n_expected_events_neyman3
        n_neyman_null_experiments = settings.n_neyman3_null_experiments
        n_neyman_alternate_experiments = settings.n_neyman3_alternate_experiments

    logging.debug('NC settings: %s %s %s %s %s', neyman2_mode, neyman_filename, n_expected_events_neyman,
                  n_neyman_null_experiments, n_neyman_alternate_experiments)
    logging.debug('Diagnostics settings: %s %s', input_filename_addition, filename_addition)

    ################################################################################
    # Data
    ################################################################################

    scores_test = np.load(settings.unweighted_events_dir + '/scores_test' + input_filename_addition + '.npy')
    r_test = np.load(settings.unweighted_events_dir + '/r_test' + input_filename_addition + '.npy')
    r_roam = np.load(settings.unweighted_events_dir + '/r_roam' + input_filename_addition + '.npy')
    if do_neyman:
        r_neyman_alternate = np.load(settings.unweighted_events_dir + '/neyman/r_' + neyman_filename + '_alternate.npy')

    # To calculate cross entropy on train set
    r_train = np.load(settings.unweighted_events_dir + '/r_train' + input_filename_addition + '.npy')
    s_train = s_from_r(r_train)
    y_train = np.load(settings.unweighted_events_dir + '/y_train' + input_filename_addition + '.npy')

    # For calibrated truth results
    weights_calibration = np.load(settings.unweighted_events_dir + '/weights_calibration' + input_filename_addition
                                  + '.npy')

    # Recalibration
    weights_recalibration = np.load(settings.unweighted_events_dir + '/weights_recalibration' + input_filename_addition
                                    + '.npy')

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
    r_vs_sm_nottrained_truth = r_nottrained_truth / r_test[settings.theta_observed, :]
    r_vs_sm_trained_truth = r_trained_truth / r_test[settings.theta_observed, :]
    scores_trained_truth = np.copy(scores_test[settings.theta_benchmark_trained, :])
    scores_nottrained_truth = np.copy(scores_test[settings.theta_benchmark_nottrained, :])

    np.save(results_dir + '/r_nottrained_truth' + filename_addition + '.npy', r_nottrained_truth)
    np.save(results_dir + '/r_trained_truth' + filename_addition + '.npy', r_trained_truth)
    np.save(results_dir + '/r_vs_sm_nottrained_truth' + filename_addition + '.npy', r_vs_sm_nottrained_truth)
    np.save(results_dir + '/r_vs_sm_trained_truth' + filename_addition + '.npy', r_vs_sm_trained_truth)
    np.save(results_dir + '/scores_trained_truth' + filename_addition + '.npy', scores_trained_truth)
    np.save(results_dir + '/scores_nottrained_truth' + filename_addition + '.npy', scores_nottrained_truth)
    np.save(results_dir + '/llr_truth' + filename_addition + '.npy', expected_llr_truth)

    ################################################################################
    # Calibration (for fun)
    ################################################################################

    logging.info('Starting calibrated evaluation')
    expected_llr_calibrated = []

    for t, theta in enumerate(settings.thetas):

        if (t + 1) % 100 == 0:
            logging.info('Starting theta %s / %s', t + 1, settings.n_thetas)

        # Prepare data for calibration
        n_calibration_each = weights_calibration.shape[1]
        y_calibration = np.zeros(2 * n_calibration_each)
        y_calibration[n_calibration_each:] = 1.
        w_calibration = np.zeros(2 * n_calibration_each)
        w_calibration[:n_calibration_each] = weights_calibration[t]
        w_calibration[n_calibration_each:] = weights_calibration[theta1]
        s_calibration = weights_calibration[theta1, :] / (weights_calibration[t, :] + weights_calibration[theta1, :])
        s_calibration = np.concatenate([s_calibration, s_calibration])

        # Fit calibrator
        calibrator = IsotonicCalibrator()
        calibrator.fit(s_calibration, y_calibration, sample_weight=w_calibration)

        # Apply to test data
        s_test = s_from_r(r_test[t, :])
        s_test_calibrated = calibrator.predict(s_test)
        r_test_calibrated = r_from_s(s_test_calibrated)

        # Extract numbers of interest
        expected_llr_calibrated.append(
            - 2. * settings.n_expected_events / n_events_test * np.sum(np.log(r_test_calibrated)))

        # For benchmark theta, save more data
        if t == settings.theta_benchmark_nottrained:
            np.save(results_dir + '/r_nottrained_truth_calibrated' + filename_addition + '.npy', r_test_calibrated)
        elif t == settings.theta_benchmark_trained:
            np.save(results_dir + '/r_trained_truth_calibrated' + filename_addition + '.npy', r_test_calibrated)

    # Save evaluation results
    expected_llr_calibrated = np.asarray(expected_llr_calibrated)
    np.save(results_dir + '/llr_truth_calibrated' + filename_addition + '.npy',
            expected_llr_calibrated)

    ################################################################################
    # Recalibration
    ################################################################################

    recalibration_expected_r = np.mean(weights_recalibration, axis=1)
    np.save(results_dir + '/recalibration_expected_r_vs_sm_truth' + filename_addition + '.npy', recalibration_expected_r)

    ################################################################################
    # Roaming etc
    ################################################################################

    logging.info('Starting roaming')
    gp = GaussianProcessRegressor(normalize_y=True,
                                  kernel=C(1.0) * Matern(1.0, nu=0.5), n_restarts_optimizer=10)
    gp.fit(settings.thetas[:], np.log(r_roam))
    r_roam_truth = np.exp(gp.predict(np.c_[xx.ravel(), yy.ravel()])).T
    np.save(results_dir + '/r_roam_truth' + filename_addition + '.npy', r_roam_truth)
    np.save(results_dir + '/r_roam_thetas_truth' + filename_addition + '.npy', r_roam)

    # Calculate cross entropy on training sample (for comparison to carl loss)
    logging.info('Calculating cross-entropy on train set')
    s_train = np.clip(s_train, settings.epsilon, 1. - settings.epsilon)
    cross_entropy_train = - (y_train * np.log(s_train) + (1. - y_train) * np.log(1. - s_train)).astype(np.float64)
    cross_entropy_train = np.mean(cross_entropy_train)
    logging.info('Training cross-entropy: %s', cross_entropy_train)
    np.save(results_dir + '/cross_entropy_truth_train.npy', np.asarray([cross_entropy_train]))

    ################################################################################
    # Neyman construction
    ################################################################################

    if do_neyman:
        logging.info('Starting evaluation of Neyman experiments')
        for t in range(settings.n_thetas):
            # Alternate evaluated at null
            llr_neyman_alternate = -2. * np.sum(np.log(r_neyman_alternate[t]), axis=1)
            np.save(neyman_dir + '/' + neyman_filename + '_llr_alternate_' + str(
                t) + '_truth' + filename_addition + '.npy', llr_neyman_alternate)
            # Null evaluated at null
            r_neyman_null = np.load(
                settings.unweighted_events_dir + '/neyman/r_' + neyman_filename + '_null_' + str(t) + '.npy')
            llr_neyman_null = -2. * np.sum(np.log(r_neyman_null[1]), axis=1)
            np.save(neyman_dir + '/' + neyman_filename + '_llr_null_' + str(t) + '_truth' + filename_addition + '.npy',
                    llr_neyman_null)

            # Null evaluated at alternative
            llr_neyman_nullatalternative = -2. * np.sum(np.log(r_neyman_null[0]), axis=1)
            np.save(neyman_dir + '/' + neyman_filename + '_llr_nullatalternate_' + str(
                t) + '_truth' + filename_addition + '.npy', llr_neyman_nullatalternative)
