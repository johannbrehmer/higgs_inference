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

    logging.info('Starting score regression higgs_inference')

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
    unweighted_events_dir = '/scratch/jb6504/higgs_inference/data/unweighted_events'
    results_dir = '../results/parameterized'

    logging.info('Options:')
    logging.info('  Number of epochs:         %s', n_epochs)
    logging.info('  Number of hidden layers:  %s', n_hidden_layers)

    ################################################################################
    # Data
    ################################################################################

    thetas = np.load(data_dir + '/thetas/thetas_parameterized.npy')

    n_thetas = len(thetas)
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

    n_expected_events = 36
    n_events_test = X_test.shape[0]
    assert n_thetas == r_test.shape[0]

    # p values
    n_neyman_distribution_experiments = 1000000
    n_neyman_observed_experiments = 101

    scaler = StandardScaler()
    scaler.fit(np.array(X_train, dtype=np.float64))
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    X_calibration_transformed = scaler.transform(X_calibration)

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

    # Toy experiments for p values
    logging.info('Starting toy experiments for Neyman construction')
    that_neyman_observed_experiments = np.zeros((n_neyman_observed_experiments, n_expected_events, 2))
    for i in range(n_neyman_observed_experiments):
        indices = np.random.choice(X_test_transformed.shape[0], n_expected_events)
        that_neyman_observed_experiments[i, :, :] = regr.predict(X_test_transformed[indices])

    logging.info('Starting density estimation')
    expected_llr = []
    median_p_values = []

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

        # Toy experimemts for distribution of test statistics (Neyman construction)
        that_neyman_distribution_experiments = np.zeros((n_neyman_distribution_experiments, n_expected_events, 2))
        event_probabilities = np.copy(weights_calibration[t]).astype(np.float64)
        event_probabilities /= np.sum(event_probabilities)
        for i in range(n_neyman_distribution_experiments):
            indices = np.random.choice(X_calibration_transformed.shape[0], n_expected_events, p=event_probabilities)
            that_neyman_distribution_experiments[i, :, :] = regr.predict(X_calibration_transformed[indices])

        # Calculate distribution of test statistics
        tthat_neyman_distribution_experiments = that_neyman_distribution_experiments.dot(delta_theta)
        llr_neyman_distribution_experiments = np.zeros(n_neyman_distribution_experiments)
        for i in range(n_neyman_distribution_experiments):
            this_r = r_from_s(calibrator.predict(tthat_neyman_distribution_experiments[i]))
            llr_neyman_distribution_experiments[i] = -2. * np.sum(np.log(this_r))
        llr_neyman_distribution_experiments = np.sort(llr_neyman_distribution_experiments)

        # Calculate observed test statistics
        tthat_neyman_observed_experiments = that_neyman_observed_experiments.dot(delta_theta)
        llr_neyman_observed_experiments = np.zeros(n_neyman_observed_experiments)
        for i in range(n_neyman_observed_experiments):
            this_r = r_from_s(calibrator.predict(tthat_neyman_observed_experiments[i]))
            llr_neyman_observed_experiments[i] = -2. * np.sum(np.log(this_r))

        # Calculate p values and store median p value
        p_values = (1. - np.searchsorted(llr_neyman_distribution_experiments,
                                         llr_neyman_observed_experiments).astype('float')
                    / n_neyman_distribution_experiments)
        median_p_values.append(np.median(p_values))

        # For some benchmark thetas, save more information on Neyman construction
        if t == theta_benchmark_nottrained:
            np.save(results_dir + '/neyman_llr_distribution_nottrained_scoreregression' + filename_addition + '.npy',
                    llr_neyman_distribution_experiments)
            np.save(results_dir + '/neyman_llr_observed_nottrained_scoreregression' + filename_addition + '.npy',
                    llr_neyman_observed_experiments)
        elif t == theta_benchmark_trained:
            np.save(results_dir + '/neyman_llr_distribution_trained_scoreregression' + filename_addition + '.npy',
                    llr_neyman_distribution_experiments)
            np.save(results_dir + '/neyman_llr_observed_trained_scoreregression' + filename_addition + '.npy',
                    llr_neyman_observed_experiments)

    # Save expected LLR and median p values
    expected_llr = np.asarray(expected_llr)
    median_p_values = np.asarray(median_p_values)
    np.save(results_dir + '/llr_scoreregression' + filename_addition + '.npy', expected_llr)
    np.save(results_dir + '/p_values_scoreregression' + filename_addition + '.npy', median_p_values)
