################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import logging
import numpy as np

from sklearn.preprocessing import StandardScaler

from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping

from carl.learning.calibration import HistogramCalibrator

from higgs_inference.models.models_score_regression import make_regressor
from higgs_inference.various.utils import r_from_s


################################################################################
# What do
################################################################################

def score_regression_inference(options=''):
    """
    Trains and evaluates one of the parameterized higgs_inference methods.

    :param options: Further options in a list of strings or string.
    """

    logging.info('Starting score regression inference')

    deep_mode = ('deep' in options)
    shallow_mode = ('shallow' in options)
    short_mode = ('short' in options)
    long_mode = ('long' in options)
    denom1_mode = ('denom1' in options)

    filename_addition = ''
    n_hidden_layers = 2
    if shallow_mode:
        n_hidden_layers = 1
        filename_addition += '_shallow'
    elif deep_mode:
        n_hidden_layers = 3
        filename_addition += '_deep'

    n_epochs = 20
    early_stopping = True
    if long_mode:
        n_epochs = 50
        filename_addition += '_long'
    elif short_mode:
        n_epochs = 1
        early_stopping = False
        filename_addition += '_short'

    theta1 = 708
    input_filename_addition = ''
    if denom1_mode:
        input_filename_addition = '_denom1'
        filename_addition += '_denom1'
        theta1 = 422

    data_dir = '../data'
    # unweighted_events_dir = '../data/unweighted_events'
    unweighted_events_dir = '/scratch/jb6504/higgs_inference/data/unweighted_events'
    results_dir = '../results/score_regression'
    neyman_dir = results_dir + '/neyman'

    logging.info('Options:')
    logging.info('  Number of epochs:         %s', n_epochs)
    logging.info('  Number of hidden layers:  %s', n_hidden_layers)

    ################################################################################
    # Data
    ################################################################################

    thetas = np.load(data_dir + '/thetas/thetas_parameterized.npy')

    n_thetas = len(thetas)
    # theta_expected = 0
    theta_benchmark_trained = 422
    theta_benchmark_nottrained = 9
    # theta_score = 0

    X_train = np.load(unweighted_events_dir + '/X_train_scoreregression' + input_filename_addition + '.npy')
    scores_train = np.load(unweighted_events_dir + '/scores_train_scoreregression' + input_filename_addition + '.npy')

    X_calibration = np.load(unweighted_events_dir + '/X_calibration' + input_filename_addition + '.npy')
    weights_calibration = np.load(unweighted_events_dir + '/weights_calibration' + input_filename_addition + '.npy')

    X_test = np.load(unweighted_events_dir + '/X_test' + input_filename_addition + '.npy')
    # scores_test = np.load(unweighted_events_dir + '/scores_test' + input_filename_addition + '.npy')
    r_test = np.load(unweighted_events_dir + '/r_test' + input_filename_addition + '.npy')
    X_neyman_observed = np.load(unweighted_events_dir + '/X_neyman_observed.npy')

    n_expected_events = 36
    n_events_test = X_test.shape[0]
    assert n_thetas == r_test.shape[0]

    scaler = StandardScaler()
    scaler.fit(np.array(X_train, dtype=np.float64))
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    X_calibration_transformed = scaler.transform(X_calibration)
    X_neyman_observed_transformed = scaler.transform(X_neyman_observed.reshape((-1, X_neyman_observed.shape[2])))

    ################################################################################
    # Score regression
    ################################################################################

    regr = KerasRegressor(lambda: make_regressor(n_hidden_layers=n_hidden_layers),
                          epochs=n_epochs, verbose=2, validation_split=0.1,
                          callbacks=[EarlyStopping(verbose=1, patience=3)])

    logging.info('Starting training of score regression')
    regr.fit(X_train_transformed, scores_train,
             callbacks=([EarlyStopping(verbose=1, patience=3)] if early_stopping else None))

    logging.info('Starting evaluation')
    that_calibration = regr.predict(X_calibration_transformed)
    that_test = regr.predict(X_test_transformed)

    logging.info('Starting density estimation')
    expected_llr = []

    for t, theta in enumerate(thetas):

        # Contract estimated scores with delta theta
        delta_theta = theta - theta1
        tthat_calibration = that_calibration.dot(delta_theta)

        # Weights for density estimation histograms
        tthat_calibration = np.hstack((tthat_calibration, tthat_calibration))
        y_calibration = np.hstack((np.zeros(that_calibration.shape[0]), np.ones(that_calibration.shape[0])))
        w_calibration = np.hstack((weights_calibration[t, ::], weights_calibration[theta1, ::]))

        # Calibration histograms
        calibrator = HistogramCalibrator(bins=500, independent_binning=False, variable_width=False,
                                         interpolation='quadratic')
        calibrator.fit(tthat_calibration, y_calibration, sample_weight=w_calibration)

        # Evaluation
        tthat_test = that_test.dot(delta_theta)
        r_hat_test = r_from_s(calibrator.predict(tthat_test.reshape((-1,))))
        expected_llr.append(- 2. * n_expected_events / n_events_test * np.sum(np.log(r_hat_test)))

        # For some benchmark thetas, save r for each phase-space point
        if t == theta_benchmark_nottrained:
            np.save(results_dir + '/r_nottrained_scoreregression' + filename_addition + '.npy', r_hat_test)
        elif t == theta_benchmark_trained:
            np.save(results_dir + '/r_trained_scoreregression' + filename_addition + '.npy', r_hat_test)

        # Neyman construction: evaluate observed sample (raw)
        tthat_neyman_observed = regr.predict(X_neyman_observed_transformed).dot(delta_theta)
        llr_neyman_observed = -2. * np.sum(tthat_neyman_observed.reshape((-1, n_expected_events)), axis=1)
        np.save(neyman_dir + '/neyman_llr_observed_scoreregression_' + str(t) + '.npy',
                llr_neyman_observed)

        # Neyman construction: evaluate observed sample (calibrated)
        r_neyman_observed = r_from_s(calibrator.predict(tthat_neyman_observed))
        llr_calibrated_neyman_observed = -2. * np.sum(np.log(r_neyman_observed).reshape((-1, n_expected_events)),
                                                      axis=1)
        np.save(neyman_dir + '/neyman_llr_observed_scoreregression_calibrated_' + str(t) + '.npy',
                llr_calibrated_neyman_observed)

        # Neyman construction: load distribution sample
        X_neyman_distribution = np.load(unweighted_events_dir + '/X_neyman_distribution_' + str(t) + '.npy')
        X_neyman_distribution_transformed = scaler.transform(
            X_neyman_distribution.reshape((-1, X_neyman_distribution.shape[2])))

        # Neyman construction: evaluate distribution sample (raw)
        tthat_neyman_distribution = regr.predict(X_neyman_distribution_transformed).dot(delta_theta)
        llr_neyman_distribution = -2. * np.sum(tthat_neyman_distribution.reshape((-1, n_expected_events)), axis=1)
        np.save(neyman_dir + '/neyman_llr_distribution_scoreregression_' + str(t) + '.npy',
                llr_neyman_distribution)

        # Neyman construction: evaluate distribution sample (calibrated)
        r_neyman_distribution = r_from_s(calibrator.predict(tthat_neyman_distribution))
        llr_calibrated_neyman_distribution = -2. * np.sum(
            np.log(r_neyman_distribution).reshape((-1, n_expected_events)), axis=1)
        np.save(neyman_dir + '/neyman_llr_distribution_scoreregression_calibrated_' + str(t) + '.npy',
                llr_calibrated_neyman_distribution)

    # Save expected LLR
    expected_llr = np.asarray(expected_llr)
    np.save(results_dir + '/llr_scoreregression' + filename_addition + '.npy', expected_llr)
