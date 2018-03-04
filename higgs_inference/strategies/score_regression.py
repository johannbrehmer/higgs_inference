################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import logging
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, LearningRateScheduler

from carl.learning.calibration import HistogramCalibrator, NDHistogramCalibrator

from higgs_inference import settings
from higgs_inference.models.models_score_regression import make_regressor
from higgs_inference.various.utils import r_from_s, decide_toy_evaluation, calculate_mean_squared_error


def score_regression_inference(use_smearing=False,
                               do_neyman=False,
                               options=''):
    """
    Trains and evaluates one of the parameterized higgs_inference methods.

    :param use_smearing:
    :param do_neyman:
    :param options: Further options in a list of strings or string.
    """

    logging.info('Starting score regression inference')

    ################################################################################
    # Settings
    ################################################################################

    deep_mode = ('deep' in options)
    shallow_mode = ('shallow' in options)
    short_mode = ('short' in options)
    long_mode = ('long' in options)
    denom1_mode = ('denom1' in options)
    small_lr_mode = ('slowlearning' in options)
    large_lr_mode = ('fastlearning' in options)
    large_batch_mode = ('largebatch' in options)
    small_batch_mode = ('smallbatch' in options)
    constant_lr_mode = ('constantlr' in options)
    new_sample_mode = ('new' in options)

    filename_addition = ''

    learning_rate = settings.learning_rate_default
    if small_lr_mode:
        filename_addition += '_slowlearning'
        learning_rate = settings.learning_rate_small
    elif large_lr_mode:
        filename_addition += '_fastlearning'
        learning_rate = settings.learning_rate_large

    lr_decay = settings.learning_rate_decay
    if constant_lr_mode:
        lr_decay = 0.
        filename_addition += '_constantlr'

    batch_size = settings.batch_size_default
    if large_batch_mode:
        filename_addition += '_largebatch'
        batch_size = settings.batch_size_large
    elif small_batch_mode:
        filename_addition += '_smallbatch'
        batch_size = settings.batch_size_small
    settings.batch_size = batch_size

    n_hidden_layers = settings.n_hidden_layers_default
    if shallow_mode:
        n_hidden_layers = settings.n_hidden_layers_shallow
        filename_addition += '_shallow'
    elif deep_mode:
        n_hidden_layers = settings.n_hidden_layers_deep
        filename_addition += '_deep'

    n_epochs = settings.n_epochs_default
    early_stopping = True
    if long_mode:
        n_epochs = settings.n_epochs_long
        filename_addition += '_long'
    elif short_mode:
        n_epochs = settings.n_epochs_short
        early_stopping = False
        filename_addition += '_short'

    input_X_prefix = ''
    if use_smearing:
        input_X_prefix = 'smeared_'
        filename_addition += '_smeared'

    theta1 = settings.theta1_default
    input_filename_addition = ''
    if denom1_mode:
        input_filename_addition = '_denom1'
        filename_addition += '_denom1'
        theta1 = settings.theta1_alternative

    if new_sample_mode:
        filename_addition += '_new'

    results_dir = settings.base_dir + '/results/score_regression'
    neyman_dir = settings.neyman_dir + '/score_regression'

    logging.info('Options:')
    logging.info('  Number of hidden layers: %s', n_hidden_layers)
    logging.info('  Batch size:              %s', batch_size)
    logging.info('  Learning rate:           %s', learning_rate)
    logging.info('  Learning rate decay:     %s', lr_decay)
    logging.info('  Number of epochs:        %s', n_epochs)

    ################################################################################
    # Data
    ################################################################################

    # Load data
    if new_sample_mode:
        X_train = np.load(
            settings.unweighted_events_dir + '/' + input_X_prefix + 'X_train_scoreregression' + input_filename_addition + '_new.npy')
        scores_train = np.load(
            settings.unweighted_events_dir + '/scores_train_scoreregression' + input_filename_addition + '_new.npy')
    else:
        X_train = np.load(
            settings.unweighted_events_dir + '/' + input_X_prefix + 'X_train_scoreregression' + input_filename_addition + '.npy')
        scores_train = np.load(
            settings.unweighted_events_dir + '/scores_train_scoreregression' + input_filename_addition + '.npy')

    X_calibration = np.load(
        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_calibration' + input_filename_addition + '.npy')
    weights_calibration = np.load(
        settings.unweighted_events_dir + '/weights_calibration' + input_filename_addition + '.npy')

    X_test = np.load(
        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_test' + input_filename_addition + '.npy')
    r_test = np.load(settings.unweighted_events_dir + '/r_test' + input_filename_addition + '.npy')
    if do_neyman:
        X_neyman_observed = np.load(settings.unweighted_events_dir + '/' + input_X_prefix + 'X_neyman_observed.npy')

    # Shuffle training data
    X_train, scores_train = shuffle(X_train, scores_train, random_state=44)

    n_events_test = X_test.shape[0]
    assert settings.n_thetas == r_test.shape[0]

    # Normalize data
    scaler = StandardScaler()
    scaler.fit(np.array(X_train, dtype=np.float64))
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    X_calibration_transformed = scaler.transform(X_calibration)
    if do_neyman:
        X_neyman_observed_transformed = scaler.transform(X_neyman_observed.reshape((-1, X_neyman_observed.shape[2])))

    ################################################################################
    # Training
    ################################################################################

    regr = KerasRegressor(lambda: make_regressor(n_hidden_layers=n_hidden_layers,
                                                 learning_rate=learning_rate),
                          epochs=n_epochs, verbose=2, validation_split=settings.validation_split)

    logging.info('Starting training of score regression')

    # Callbacks
    callbacks = []
    if not constant_lr_mode:
        def lr_scheduler(epoch):
            return learning_rate * np.exp(- epoch * lr_decay)

        callbacks.append(LearningRateScheduler(lr_scheduler))
    if early_stopping:
        callbacks.append(EarlyStopping(verbose=1, patience=settings.early_stopping_patience))

    # Training
    regr.fit(X_train_transformed, scores_train, callbacks=callbacks, batch_size=batch_size)

    logging.info('Starting evaluation')
    that_calibration = regr.predict(X_calibration_transformed)
    that_test = regr.predict(X_test_transformed)

    ################################################################################
    # Evaluation and density estimation
    ################################################################################

    logging.info('Starting density estimation')
    expected_llr = []
    expected_llr_scoretheta = []
    expected_llr_score = []
    expected_llr_rotatedscore = []
    mse_log_r = []
    mse_log_r_scoretheta = []
    mse_log_r_score = []
    mse_log_r_rotatedscore = []
    trimmed_mse_log_r = []
    trimmed_mse_log_r_scoretheta = []
    trimmed_mse_log_r_score = []
    trimmed_mse_log_r_rotatedscore = []

    for t, theta in enumerate(settings.thetas):

        if (t + 1) % 100 == 0:
            logging.info('Starting theta %s / %s', t + 1, settings.n_thetas)

        # Delta_theta
        delta_theta = theta - settings.thetas[theta1]
        delta_theta_norm = np.linalg.norm(delta_theta)
        if delta_theta_norm > settings.epsilon:
            rotation_matrix = (np.array([[delta_theta[0], - delta_theta[1]], [delta_theta[1], delta_theta[0]]])
                               / np.linalg.norm(delta_theta))
        else:
            rotation_matrix = np.identity(2)

        # Prepare calibration data
        tthat_calibration = that_calibration.dot(delta_theta)
        that_rotated_calibration = that_calibration.dot(rotation_matrix)

        # Weights for density estimation histograms
        _that_calibration = np.vstack((that_calibration, that_calibration))
        _tthat_calibration = np.hstack((tthat_calibration, tthat_calibration))
        _that_rotated_calibration = np.vstack((that_rotated_calibration, that_rotated_calibration))
        y_calibration = np.hstack((np.zeros(that_calibration.shape[0]), np.ones(that_calibration.shape[0])))
        w_calibration = np.hstack((weights_calibration[t, ::], weights_calibration[theta1, ::]))

        # 1d density estimation with score * theta
        _bins = [np.concatenate(([-100000., -100., -70., -50., -40., -30., -25., -22.],
                                 np.linspace(-20., -11., 10),
                                 np.linspace(-10., -5.5, 10),
                                 np.linspace(-5., -2.2, 15),
                                 np.linspace(-2., -1.1, 10),
                                 np.linspace(-1., 1., 41),
                                 np.linspace(1.1, 2., 10),
                                 np.linspace(2.2, 5., 15),
                                 np.linspace(5.5, 10., 10),
                                 np.linspace(11., 20., 10),
                                 [22., 25., 30., 40., 50., 70., 100., 100000.]))]

        calibrator_scoretheta = HistogramCalibrator(bins=_bins, independent_binning=False, variable_width=False)
        calibrator_scoretheta.fit(_tthat_calibration, y_calibration, sample_weight=w_calibration)

        # 2d density estimation with score (fixed binning)
        _bins = np.concatenate(([-100000., -20., -15., -10., -8., -6.],
                                np.linspace(-5., -2.5, 6),
                                np.linspace(-2., -1.2, 5),
                                np.linspace(-1., 1., 21),
                                np.linspace(1.2, 2.0, 5),
                                np.linspace(2.5, 5., 6),
                                [6., 8., 10., 15., 20., 100000.]))
        _bins = (_bins, _bins)
        _range = (np.array((-100000., 100000.)), np.array((-100000., 100000.)))
        calibrator_score = NDHistogramCalibrator(bins=_bins, range=_range)
        calibrator_score.fit(_that_calibration,
                             y_calibration,
                             sample_weight=w_calibration)

        # 2d density estimation with score (dynamicically rotated)
        _bins_main = np.concatenate(([-100000., -20., -15., -10., -8., -6.],
                                     np.linspace(-5., -2.5, 6),
                                     np.linspace(-2., -1.2, 5),
                                     np.linspace(-1., 1., 21),
                                     np.linspace(1.2, 2.0, 5),
                                     np.linspace(2.5, 5., 6),
                                     [6., 8., 10., 15., 20., 100000.]))
        _bins_other = np.array(
            [-100000., -20., -10., -5., -3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 5., 10., 20., 100000.])
        _bins = (_bins_main, _bins_other)
        _range = (np.array((-100000., 100000.)), np.array((-100000., 100000.)))
        calibrator_rotatedscore = NDHistogramCalibrator(bins=_bins, range=_range)
        calibrator_rotatedscore.fit(_that_rotated_calibration,
                                    y_calibration,
                                    sample_weight=w_calibration)

        # Evaluation
        tthat_test = that_test.dot(delta_theta)
        that_rotated_test = that_test.dot(rotation_matrix)

        # Calibration
        r_hat_scoretheta_test = r_from_s(calibrator_scoretheta.predict(tthat_test.reshape((-1,))))
        r_hat_score_test = r_from_s(calibrator_score.predict(that_test))
        r_hat_rotatedscore_test = r_from_s(calibrator_rotatedscore.predict(that_rotated_test))

        # Extract relevant numnbers
        expected_llr.append(-2. * settings.n_expected_events / n_events_test * np.sum(tthat_test))
        expected_llr_scoretheta.append(- 2. * settings.n_expected_events / n_events_test
                                       * np.sum(np.log(r_hat_scoretheta_test)))
        expected_llr_score.append(- 2. * settings.n_expected_events / n_events_test
                                  * np.sum(np.log(r_hat_score_test)))
        expected_llr_rotatedscore.append(- 2. * settings.n_expected_events / n_events_test
                                         * np.sum(np.log(r_hat_rotatedscore_test)))

        mse_log_r.append(calculate_mean_squared_error(np.log(r_test[t]), tthat_test, 0.))
        mse_log_r_scoretheta.append(calculate_mean_squared_error(np.log(r_test[t]), np.log(r_hat_scoretheta_test), 0.))
        mse_log_r_score.append(calculate_mean_squared_error(np.log(r_test[t]), np.log(r_hat_score_test), 0.))
        mse_log_r_rotatedscore.append(
            calculate_mean_squared_error(np.log(r_test[t]), np.log(r_hat_rotatedscore_test), 0.))

        trimmed_mse_log_r.append(calculate_mean_squared_error(np.log(r_test[t]), tthat_test, 'auto'))
        trimmed_mse_log_r_score.append(
            calculate_mean_squared_error(np.log(r_test[t]), np.log(r_hat_score_test), 'auto'))
        trimmed_mse_log_r_scoretheta.append(
            calculate_mean_squared_error(np.log(r_test[t]), np.log(r_hat_scoretheta_test), 'auto'))
        trimmed_mse_log_r_rotatedscore.append(
            calculate_mean_squared_error(np.log(r_test[t]), np.log(r_hat_rotatedscore_test), 'auto'))

        # For some benchmark thetas, save r for each phase-space point
        if t == settings.theta_benchmark_nottrained:
            np.save(results_dir + '/r_nottrained_scoreregression' + filename_addition + '.npy',
                    tthat_test)
            np.save(results_dir + '/r_nottrained_scoreregression_scoretheta' + filename_addition + '.npy',
                    r_hat_scoretheta_test)
            np.save(results_dir + '/r_nottrained_scoreregression_score' + filename_addition + '.npy',
                    r_hat_score_test)
            np.save(results_dir + '/r_nottrained_scoreregression_rotatedscore' + filename_addition + '.npy',
                    r_hat_rotatedscore_test)

        elif t == settings.theta_benchmark_trained:
            np.save(results_dir + '/r_trained_scoreregression' + filename_addition + '.npy',
                    tthat_test)
            np.save(results_dir + '/r_trained_scoreregression_scoretheta' + filename_addition + '.npy',
                    r_hat_scoretheta_test)
            np.save(results_dir + '/r_trained_scoreregression_score' + filename_addition + '.npy',
                    r_hat_score_test)
            np.save(results_dir + '/r_trained_scoreregression_rotatedscore' + filename_addition + '.npy',
                    r_hat_rotatedscore_test)

        if do_neyman:
            # Neyman construction: evaluate observed sample (raw)
            that_neyman_observed = regr.predict(X_neyman_observed_transformed)
            tthat_neyman_observed = that_neyman_observed.dot(delta_theta)
            that_rotated_neyman_observed = that_neyman_observed.dot(rotation_matrix)

            llr_raw_neyman_observed = -2. * np.sum(tthat_neyman_observed.reshape((-1, settings.n_expected_events)),
                                                   axis=1)
            np.save(neyman_dir + '/neyman_llr_observed_scoreregression_' + str(t) + filename_addition + '.npy',
                    llr_raw_neyman_observed)

            # Neyman construction: evaluate observed sample (calibrated) -- score * theta calibration
            s_hat_neyman_observed = calibrator_scoretheta.predict(tthat_neyman_observed.reshape((-1,)))
            r_hat_neyman_observed = r_from_s(s_hat_neyman_observed)
            r_hat_neyman_observed = r_hat_neyman_observed.reshape((-1, settings.n_expected_events))
            llr_calibrated_neyman_observed = -2. * np.sum(np.log(r_hat_neyman_observed), axis=1)
            np.save(
                neyman_dir + '/neyman_llr_observed_scoreregression_scoretheta_' + str(t) + filename_addition + '.npy',
                llr_calibrated_neyman_observed)

            # Neyman construction: evaluate observed sample (calibrated) -- score calibration
            s_hat_neyman_observed = calibrator_score.predict(that_neyman_observed)
            r_hat_neyman_observed = r_from_s(s_hat_neyman_observed)
            r_hat_neyman_observed = r_hat_neyman_observed.reshape((-1, settings.n_expected_events))
            llr_calibrated_neyman_observed = -2. * np.sum(np.log(r_hat_neyman_observed), axis=1)
            np.save(
                neyman_dir + '/neyman_llr_observed_scoreregression_score_' + str(t) + filename_addition + '.npy',
                llr_calibrated_neyman_observed)

            # Neyman construction: evaluate observed sample (calibrated) -- rotated score claibration
            s_hat_neyman_observed = calibrator_rotatedscore.predict(that_rotated_neyman_observed)
            r_hat_neyman_observed = r_from_s(s_hat_neyman_observed)
            r_hat_neyman_observed = r_hat_neyman_observed.reshape((-1, settings.n_expected_events))
            llr_calibrated_neyman_observed = -2. * np.sum(np.log(r_hat_neyman_observed), axis=1)
            np.save(
                neyman_dir + '/neyman_llr_observed_scoreregression_rotatedscore_' + str(t) + filename_addition + '.npy',
                llr_calibrated_neyman_observed)

            # Neyman construction: loop over distribution samples generated from different thetas
            llr_neyman_distributions = []
            llr_neyman_distributions_scoretheta = []
            llr_neyman_distributions_score = []
            llr_neyman_distributions_rotatedscore = []

            for tt in range(settings.n_thetas):

                # Only evaluate certain combinations of thetas to save computation time
                if not decide_toy_evaluation(tt, t):
                    placeholder = np.empty(settings.n_neyman_distribution_experiments)
                    placeholder[:] = np.nan
                    llr_neyman_distributions.append(placeholder)
                    llr_neyman_distributions_scoretheta.append(placeholder)
                    llr_neyman_distributions_score.append(placeholder)
                    llr_neyman_distributions_rotatedscore.append(placeholder)
                    continue

                # Neyman construction: load distribution sample
                X_neyman_distribution = np.load(
                    settings.unweighted_events_dir + '/' + input_X_prefix + 'X_neyman_distribution_' + str(tt) + '.npy')
                X_neyman_distribution_transformed = scaler.transform(
                    X_neyman_distribution.reshape((-1, X_neyman_distribution.shape[2])))

                # Neyman construction: evaluate distribution sample (raw)
                that_neyman_distribution = regr.predict(X_neyman_distribution_transformed)
                tthat_neyman_distribution = that_neyman_distribution.dot(delta_theta)
                that_rotated_neyman_distribution = that_neyman_distribution.dot(rotation_matrix)

                llr_neyman_distributions.append(
                    -2. * np.sum(tthat_neyman_distribution.reshape((-1, settings.n_expected_events)), axis=1))

                # Neyman construction: evaluate distribution sample (score * theta calibration)
                s_hat_neyman_distribution = calibrator_scoretheta.predict(tthat_neyman_distribution.reshape((-1,)))
                r_hat_neyman_distribution = r_from_s(s_hat_neyman_distribution)
                r_hat_neyman_distribution = r_hat_neyman_distribution.reshape((-1, settings.n_expected_events))
                llr_neyman_distributions_scoretheta.append(-2. * np.sum(np.log(r_hat_neyman_distribution), axis=1))

                # Neyman construction: evaluate distribution sample (score calibration)
                s_hat_neyman_distribution = calibrator_score.predict(that_neyman_distribution)
                r_hat_neyman_distribution = r_from_s(s_hat_neyman_distribution)
                r_hat_neyman_distribution = r_hat_neyman_distribution.reshape((-1, settings.n_expected_events))
                llr_neyman_distributions_score.append(-2. * np.sum(np.log(r_hat_neyman_distribution), axis=1))

                # Neyman construction: evaluate distribution sample (rotated score calibration)
                s_hat_neyman_distribution = calibrator_rotatedscore.predict(that_rotated_neyman_distribution)
                r_hat_neyman_distribution = r_from_s(s_hat_neyman_distribution)
                r_hat_neyman_distribution = r_hat_neyman_distribution.reshape((-1, settings.n_expected_events))
                llr_neyman_distributions_rotatedscore.append(-2. * np.sum(np.log(r_hat_neyman_distribution), axis=1))

            llr_neyman_distributions = np.asarray(llr_neyman_distributions)
            llr_neyman_distributions_scoretheta = np.asarray(llr_neyman_distributions_scoretheta)
            llr_neyman_distributions_score = np.asarray(llr_neyman_distributions_score)
            llr_neyman_distributions_rotatedscore = np.asarray(llr_neyman_distributions_rotatedscore)

            np.save(neyman_dir + '/neyman_llr_distribution_scoreregression_' + str(t) + filename_addition + '.npy',
                    llr_neyman_distributions)
            np.save(
                neyman_dir + '/neyman_llr_distribution_scoreregression_scoretheta_' + str(
                    t) + filename_addition + '.npy',
                llr_neyman_distributions_scoretheta)
            np.save(
                neyman_dir + '/neyman_llr_distribution_scoreregression_score_' + str(t) + filename_addition + '.npy',
                llr_neyman_distributions_score)
            np.save(
                neyman_dir + '/neyman_llr_distribution_scoreregression_rotatedscore_' + str(
                    t) + filename_addition + '.npy',
                llr_neyman_distributions_rotatedscore)

    # Save expected LLR
    expected_llr = np.asarray(expected_llr)
    expected_llr_scoretheta = np.asarray(expected_llr_scoretheta)
    expected_llr_score = np.asarray(expected_llr_score)
    expected_llr_rotatedscore = np.asarray(expected_llr_rotatedscore)

    mse_log_r = np.asarray(mse_log_r)
    mse_log_r_scoretheta = np.asarray(mse_log_r_scoretheta)
    mse_log_r_score = np.asarray(mse_log_r_score)
    mse_log_r_rotatedscore = np.asarray(mse_log_r_rotatedscore)

    trimmed_mse_log_r = np.asarray(trimmed_mse_log_r)
    trimmed_mse_log_r_scoretheta = np.asarray(trimmed_mse_log_r_scoretheta)
    trimmed_mse_log_r_score = np.asarray(trimmed_mse_log_r_score)
    trimmed_mse_log_r_rotatedscore = np.asarray(trimmed_mse_log_r_rotatedscore)

    np.save(results_dir + '/llr_scoreregression' + filename_addition + '.npy', expected_llr)
    np.save(results_dir + '/llr_scoreregression_scoretheta' + filename_addition + '.npy', expected_llr_scoretheta)
    np.save(results_dir + '/llr_scoreregression_score' + filename_addition + '.npy', expected_llr_score)
    np.save(results_dir + '/llr_scoreregression_rotatedscore' + filename_addition + '.npy', expected_llr_rotatedscore)

    np.save(results_dir + '/mse_logr_scoreregression' + filename_addition + '.npy', mse_log_r)
    np.save(results_dir + '/mse_logr_scoreregression_scoretheta' + filename_addition + '.npy', mse_log_r_scoretheta)
    np.save(results_dir + '/mse_logr_scoreregression_score' + filename_addition + '.npy', mse_log_r_score)
    np.save(results_dir + '/mse_logr_scoreregression_rotatedscore' + filename_addition + '.npy', mse_log_r_rotatedscore)

    np.save(results_dir + '/trimmed_mse_logr_scoreregression' + filename_addition + '.npy', trimmed_mse_log_r)
    np.save(results_dir + '/trimmed_mse_logr_scoreregression_scoretheta' + filename_addition + '.npy',
            trimmed_mse_log_r_scoretheta)
    np.save(results_dir + '/trimmed_mse_logr_scoreregression_score' + filename_addition + '.npy',
            trimmed_mse_log_r_score)
    np.save(results_dir + '/trimmed_mse_logr_scoreregression_rotatedscore' + filename_addition + '.npy',
            trimmed_mse_log_r_rotatedscore)
