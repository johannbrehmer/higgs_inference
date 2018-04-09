################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import logging
import time
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, LearningRateScheduler

from carl.learning.calibration import HistogramCalibrator, NDHistogramCalibrator

from higgs_inference import settings
from higgs_inference.models.models_score_regression import make_regressor
from higgs_inference.various.utils import r_from_s, calculate_mean_squared_error


def score_regression_inference(use_smearing=False,
                               denominator=0,
                               do_neyman=False,
                               options=''):
    """
    Trains and evaluates one of the parameterized higgs_inference methods.

    :param denominator:
    :param use_smearing:
    :param do_neyman:
    :param options: Further options in a list of strings or string.
    """

    logging.info('Starting score regression inference')

    ################################################################################
    # Settings
    ################################################################################

    fixed_binning_mode = ('fixedbinning' in options)
    deep_mode = ('deep' in options)
    shallow_mode = ('shallow' in options)
    short_mode = ('short' in options)
    long_mode = ('long' in options)
    small_lr_mode = ('slowlearning' in options)
    large_lr_mode = ('fastlearning' in options)
    large_batch_mode = ('largebatch' in options)
    small_batch_mode = ('smallbatch' in options)
    constant_lr_mode = ('constantlr' in options)
    new_sample_mode = ('new' in options)
    neyman2_mode = ('neyman2' in options)
    neyman3_mode = ('neyman3' in options)

    filename_addition = ''

    if fixed_binning_mode:
        filename_addition += '_fixedbinning'

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
    if denominator > 0:
        input_filename_addition = '_denom' + str(denominator)
        filename_addition += '_denom' + str(denominator)
        theta1 = settings.theta1_alternatives[denominator - 1]

    if new_sample_mode:
        input_filename_addition = '_new'
        filename_addition += '_new'

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

    results_dir = settings.base_dir + '/results/score_regression'
    neyman_dir = settings.neyman_dir + '/score_regression'

    logging.info('Options:')
    logging.info('  Denominator theta:       denominator %s = theta %s = %s', denominator, theta1,
                 settings.thetas[theta1])
    logging.info('  Number of hidden layers: %s', n_hidden_layers)
    logging.info('  Batch size:              %s', batch_size)
    logging.info('  Learning rate:           %s', learning_rate)
    logging.info('  Learning rate decay:     %s', lr_decay)
    logging.info('  Number of epochs:        %s', n_epochs)
    if do_neyman:
        logging.info('  NC experiments:          (%s alternate + %s null) experiments with %s alternate events each',
                     n_neyman_alternate_experiments, n_neyman_null_experiments, n_expected_events_neyman)
    else:
        logging.info('  NC experiments:          False')

    ################################################################################
    # Data
    ################################################################################

    # Load data
    X_train = np.load(
        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_train_scoreregression' + input_filename_addition + '.npy')
    scores_train = np.load(
        settings.unweighted_events_dir + '/scores_train_scoreregression' + input_filename_addition + '.npy')

    X_calibration = np.load(
        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_calibration' + input_filename_addition + '.npy')
    weights_calibration = np.load(
        settings.unweighted_events_dir + '/weights_calibration' + input_filename_addition + '.npy')

    X_recalibration = np.load(
        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_recalibration' + '.npy')

    X_test = np.load(
        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_test' + input_filename_addition + '.npy')
    r_test = np.load(settings.unweighted_events_dir + '/r_test' + input_filename_addition + '.npy')
    X_illustration = np.load(
        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_illustration' + input_filename_addition + '.npy')
    if do_neyman:
        X_neyman_alternate = np.load(
            settings.unweighted_events_dir + '/neyman/' + input_X_prefix + 'X_' + neyman_filename + '_alternate.npy')

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
    X_recalibration_transformed = scaler.transform(X_recalibration)
    X_illustration_transformed = scaler.transform(X_illustration)
    if do_neyman:
        X_neyman_alternate_transformed = scaler.transform(X_neyman_alternate.reshape((-1, X_neyman_alternate.shape[2])))

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

    # Estimate local score on different samples
    logging.info('Starting evaluation')
    time_before = time.time()
    that_test = regr.predict(X_test_transformed)
    eval_time = time.time() - time_before

    that_calibration = regr.predict(X_calibration_transformed)
    that_recalibration = regr.predict(X_recalibration_transformed)
    that_illustration = regr.predict(X_illustration_transformed)

    logging.info('Score regression evaluation timing: %s s', eval_time)

    ################################################################################
    # Evaluation loop
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
    eval_times_scoretheta = []
    eval_times_score = []
    eval_times_rotatedscore = []
    expected_r_vs_sm_scoretheta = []
    expected_r_vs_sm_score = []
    expected_r_vs_sm_rotatedscore = []
    recalibration_expected_r_scoretheta = []
    recalibration_expected_r_score = []
    recalibration_expected_r_rotatedscore = []

    for t, theta in enumerate(settings.thetas):

        if (t + 1) % 100 == 0:
            logging.info('Starting theta %s / %s', t + 1, settings.n_thetas)

        ################################################################################
        # Calibration (density estimation)
        ################################################################################

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
        if fixed_binning_mode:
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
        else:
            _bins = [np.concatenate((
                [-100000.],
                np.linspace(np.percentile(_tthat_calibration, .1), np.percentile(_tthat_calibration, 2.5), 30)[:-1],
                np.linspace(np.percentile(_tthat_calibration, 2.5), np.percentile(_tthat_calibration, 97.5), 40)[:-1],
                np.linspace(np.percentile(_tthat_calibration, 97.5), np.percentile(_tthat_calibration, 99.9), 30),
                [100000.]
            ))]

        logging.debug('Score * theta binning: %s', _bins)

        calibrator_scoretheta = HistogramCalibrator(bins=_bins, independent_binning=False, variable_width=False)
        calibrator_scoretheta.fit(_tthat_calibration, y_calibration, sample_weight=w_calibration)

        # 2d density estimation with score (fixed binning)
        if fixed_binning_mode:
            _bins = np.concatenate(([-100000., -20., -15., -10., -8., -6.],
                                    np.linspace(-5., -2.5, 6),
                                    np.linspace(-2., -1.2, 5),
                                    np.linspace(-1., 1., 21),
                                    np.linspace(1.2, 2.0, 5),
                                    np.linspace(2.5, 5., 6),
                                    [6., 8., 10., 15., 20., 100000.]))
            _bins = (_bins, _bins)
        else:
            _bins0 = np.concatenate((
                [-100000.],
                np.linspace(np.percentile(_that_calibration[:, 0], .2), np.percentile(_that_calibration[:, 0], 2.5), 7)[
                :-1],
                np.linspace(np.percentile(_that_calibration[:, 0], 2.5), np.percentile(_that_calibration[:, 0], 97.5),
                            12)[:-1],
                np.linspace(np.percentile(_that_calibration[:, 0], 97.5), np.percentile(_that_calibration[:, 0], 99.8),
                            7),
                [100000.]
            ))
            _bins1 = np.concatenate((
                [-100000.],
                np.linspace(np.percentile(_that_calibration[:, 1], .2), np.percentile(_that_calibration[:, 1], 2.5), 7)[
                :-1],
                np.linspace(np.percentile(_that_calibration[:, 1], 2.5), np.percentile(_that_calibration[:, 1], 97.5),
                            12)[:-1],
                np.linspace(np.percentile(_that_calibration[:, 1], 97.5), np.percentile(_that_calibration[:, 1], 99.8),
                            7),
                [100000.]
            ))
            _bins = (_bins0, _bins1)

        _range = (np.array((-100000., 100000.)), np.array((-100000., 100000.)))

        calibrator_score = NDHistogramCalibrator(bins=_bins, range=_range)
        calibrator_score.fit(_that_calibration,
                             y_calibration,
                             sample_weight=w_calibration)

        logging.debug('Score binning: %s', _bins)

        # 2d density estimation with score (dynamicically rotated)
        if fixed_binning_mode:
            _bins_main = np.concatenate(([-100000., -20., -15., -10., -8., -6.],
                                         np.linspace(-5., -2.5, 6),
                                         np.linspace(-2., -1.2, 5),
                                         np.linspace(-1., 1., 21),
                                         np.linspace(1.2, 2.0, 5),
                                         np.linspace(2.5, 5., 6),
                                         [6., 8., 10., 15., 20., 100000.]))
            _bins_other = np.array(
                [-100000., -20., -10., -5., -3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 5., 10., 20., 100000.])
        else:
            _bins_main = np.concatenate((
                [-100000.],
                np.linspace(np.percentile(_that_rotated_calibration[:, 0], .1),
                            np.percentile(_that_rotated_calibration[:, 0], 2.5), 24)[:-1],
                np.linspace(np.percentile(_that_rotated_calibration[:, 0], 2.5),
                            np.percentile(_that_rotated_calibration[:, 0], 97.5), 32)[:-1],
                np.linspace(np.percentile(_that_rotated_calibration[:, 0], 97.5),
                            np.percentile(_that_rotated_calibration[:, 0], 99.9), 24),
                [100000.]
            ))
            _bins_other = np.concatenate((
                [-100000.],
                np.linspace(np.percentile(_that_rotated_calibration[:, 1], .3),
                            np.percentile(_that_rotated_calibration[:, 1], 2.5), 3)[:-1],
                np.linspace(np.percentile(_that_rotated_calibration[:, 1], 2.5),
                            np.percentile(_that_rotated_calibration[:, 1], 97.5), 5)[:-1],
                np.linspace(np.percentile(_that_rotated_calibration[:, 1], 97.5),
                            np.percentile(_that_rotated_calibration[:, 1], 99.7), 3),
                [100000.]
            ))

        _bins = (_bins_main, _bins_other)

        logging.debug('Dynamic score binning: %s', _bins)

        _range = (np.array((-100000., 100000.)), np.array((-100000., 100000.)))

        calibrator_rotatedscore = NDHistogramCalibrator(bins=_bins, range=_range)
        calibrator_rotatedscore.fit(_that_rotated_calibration,
                                    y_calibration,
                                    sample_weight=w_calibration)

        ################################################################################
        # Evaluation
        ################################################################################

        # Prepare evaluation data
        tthat_test = that_test.dot(delta_theta)
        that_rotated_test = that_test.dot(rotation_matrix)

        # Calibrated evaluation
        time_before = time.time()
        r_hat_scoretheta_test = r_from_s(calibrator_scoretheta.predict(tthat_test.reshape((-1,))))
        eval_times_scoretheta.append(time.time() - time_before)
        time_before = time.time()
        r_hat_score_test = r_from_s(calibrator_score.predict(that_test))
        eval_times_score.append(time.time() - time_before)
        time_before = time.time()
        r_hat_rotatedscore_test = r_from_s(calibrator_rotatedscore.predict(that_rotated_test))
        eval_times_rotatedscore.append(time.time() - time_before)

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

        if t == settings.theta_observed:
            r_sm_score = r_hat_score_test
            r_sm_scoretheta = r_hat_scoretheta_test
            r_sm_rotatedscore = r_hat_rotatedscore_test
        expected_r_vs_sm_score.append(np.mean(r_hat_score_test / r_sm_score))
        expected_r_vs_sm_scoretheta.append(np.mean(r_hat_scoretheta_test / r_sm_scoretheta))
        expected_r_vs_sm_rotatedscore.append(np.mean(r_hat_rotatedscore_test / r_sm_rotatedscore))

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

            np.save(results_dir + '/r_vs_sm_nottrained_scoreregression_scoretheta' + filename_addition + '.npy',
                    r_hat_scoretheta_test / r_sm_scoretheta)
            np.save(results_dir + '/r_vs_sm_nottrained_scoreregression_score' + filename_addition + '.npy',
                    r_hat_score_test / r_sm_score)
            np.save(results_dir + '/r_vs_sm_nottrained_scoreregression_rotatedscore' + filename_addition + '.npy',
                    r_hat_rotatedscore_test / r_sm_rotatedscore)

        elif t == settings.theta_benchmark_trained:
            np.save(results_dir + '/r_trained_scoreregression' + filename_addition + '.npy',
                    tthat_test)
            np.save(results_dir + '/r_trained_scoreregression_scoretheta' + filename_addition + '.npy',
                    r_hat_scoretheta_test)
            np.save(results_dir + '/r_trained_scoreregression_score' + filename_addition + '.npy',
                    r_hat_score_test)
            np.save(results_dir + '/r_trained_scoreregression_rotatedscore' + filename_addition + '.npy',
                    r_hat_rotatedscore_test)

            np.save(results_dir + '/r_vs_sm_trained_scoreregression_scoretheta' + filename_addition + '.npy',
                    r_hat_scoretheta_test / r_sm_scoretheta)
            np.save(results_dir + '/r_vs_sm_trained_scoreregression_score' + filename_addition + '.npy',
                    r_hat_score_test / r_sm_score)
            np.save(results_dir + '/r_vs_sm_trained_scoreregression_rotatedscore' + filename_addition + '.npy',
                    r_hat_rotatedscore_test / r_sm_rotatedscore)

        ################################################################################
        # Recalibration
        ################################################################################

        # Prepare recalibration data
        tthat_recalibration = that_recalibration.dot(delta_theta)
        that_rotated_recalibration = that_recalibration.dot(rotation_matrix)

        r_hat_recalibration_score = r_from_s(calibrator_score.predict(that_recalibration))
        r_hat_recalibration_scoretheta = r_from_s(calibrator_scoretheta.predict(tthat_recalibration.reshape((-1,))))
        r_hat_recalibration_rotatedscore = r_from_s(calibrator_rotatedscore.predict(that_rotated_recalibration))

        if t == settings.theta_observed:
            r_hat_recalibration_sm_score = r_hat_recalibration_score
            r_hat_recalibration_sm_scoretheta = r_hat_recalibration_scoretheta
            r_hat_recalibration_sm_rotatedscore = r_hat_recalibration_rotatedscore

        # Evaluate recalibration data
        recalibration_expected_r_score.append(np.mean(
            r_hat_recalibration_score / r_hat_recalibration_sm_score
        ))
        recalibration_expected_r_scoretheta.append(np.mean(
            r_hat_recalibration_scoretheta / r_hat_recalibration_sm_scoretheta
        ))
        recalibration_expected_r_rotatedscore.append(np.mean(
            r_hat_recalibration_rotatedscore / r_hat_recalibration_sm_rotatedscore
        ))

        ################################################################################
        # Illustration
        ################################################################################

        if t == settings.theta_benchmark_illustration:
            # Prepare illustration data
            tthat_illustration = that_illustration.dot(delta_theta)
            that_rotated_illustration = that_illustration.dot(rotation_matrix)

            r_hat_illustration_score = r_from_s(calibrator_score.predict(that_illustration))
            r_hat_illustration_scoretheta = r_from_s(calibrator_scoretheta.predict(tthat_illustration.reshape((-1,))))
            r_hat_illustration_rotatedscore = r_from_s(calibrator_rotatedscore.predict(that_rotated_illustration))

            np.save(results_dir + '/that_illustration_scoreregression' + filename_addition + '.npy',
                    that_illustration)
            np.save(results_dir + '/r_illustration_scoreregression_scoretheta' + filename_addition + '.npy',
                    r_hat_illustration_scoretheta)
            np.save(results_dir + '/r_illustration_scoreregression_score' + filename_addition + '.npy',
                    r_hat_illustration_score)
            np.save(results_dir + '/r_illustration_scoreregression_rotatedscore' + filename_addition + '.npy',
                    r_hat_illustration_rotatedscore)

        ################################################################################
        # Neyman construction toys
        ################################################################################

        if do_neyman:
            # Neyman construction: evaluate alternate sample (raw)
            that_neyman_alternate = regr.predict(X_neyman_alternate_transformed)
            tthat_neyman_alternate = that_neyman_alternate.dot(delta_theta)
            that_rotated_neyman_alternate = that_neyman_alternate.dot(rotation_matrix)
            np.save(neyman_dir + '/' + neyman_filename + '_that_alternate_' + str(
                t) + '_scoreregression' + filename_addition + '.npy', that_neyman_alternate)

            # No calibration (local model)
            llr_raw_neyman_alternate = -2. * np.sum(tthat_neyman_alternate.reshape((-1, n_expected_events_neyman)),
                                                    axis=1)
            np.save(neyman_dir + '/' + neyman_filename + '_llr_alternate_' + str(
                t) + '_scoreregression' + filename_addition + '.npy', llr_raw_neyman_alternate)

            # Neyman construction: evaluate alternate sample (calibrated) -- score * theta calibration
            s_hat_neyman_alternate = calibrator_scoretheta.predict(tthat_neyman_alternate.reshape((-1,)))
            r_hat_neyman_alternate = r_from_s(s_hat_neyman_alternate)
            r_hat_neyman_alternate = r_hat_neyman_alternate.reshape((-1, n_expected_events_neyman))
            llr_calibrated_neyman_alternate = -2. * np.sum(np.log(r_hat_neyman_alternate), axis=1)
            np.save(neyman_dir + '/' + neyman_filename + '_llr_alternate_' + str(
                t) + '_scoreregression_scoretheta' + filename_addition + '.npy', llr_calibrated_neyman_alternate)

            # Neyman construction: evaluate alternate sample (calibrated) -- score calibration
            s_hat_neyman_alternate = calibrator_score.predict(that_neyman_alternate)
            r_hat_neyman_alternate = r_from_s(s_hat_neyman_alternate)
            r_hat_neyman_alternate = r_hat_neyman_alternate.reshape((-1, n_expected_events_neyman))
            llr_calibrated_neyman_alternate = -2. * np.sum(np.log(r_hat_neyman_alternate), axis=1)
            np.save(neyman_dir + '/' + neyman_filename + '_llr_alternate_' + str(
                t) + '_scoreregression_score' + filename_addition + '.npy', llr_calibrated_neyman_alternate)

            # Neyman construction: evaluate alternate sample (calibrated) -- rotated score claibration
            s_hat_neyman_alternate = calibrator_rotatedscore.predict(that_rotated_neyman_alternate)
            r_hat_neyman_alternate = r_from_s(s_hat_neyman_alternate)
            r_hat_neyman_alternate = r_hat_neyman_alternate.reshape((-1, n_expected_events_neyman))
            llr_calibrated_neyman_alternate = -2. * np.sum(np.log(r_hat_neyman_alternate), axis=1)
            np.save(neyman_dir + '/' + neyman_filename + '_llr_alternate_' + str(
                t) + '_scoreregression_rotatedscore' + filename_addition + '.npy', llr_calibrated_neyman_alternate)

            # # Neyman construction: loop over null samples generated from different thetas (old)
            # llr_neyman_nulls = []
            # llr_neyman_nulls_scoretheta = []
            # llr_neyman_nulls_score = []
            # llr_neyman_nulls_rotatedscore = []
            #
            # for tt in range(settings.n_thetas):
            #
            #     # Only evaluate certain combinations of thetas to save computation time
            #     if not decide_toy_evaluation(tt, t):
            #         placeholder = np.empty(n_neyman_null_experiments)
            #         placeholder[:] = np.nan
            #         llr_neyman_nulls.append(placeholder)
            #         llr_neyman_nulls_scoretheta.append(placeholder)
            #         llr_neyman_nulls_score.append(placeholder)
            #         llr_neyman_nulls_rotatedscore.append(placeholder)
            #         continue
            #
            #     # Neyman construction: load null sample
            #     X_neyman_null = np.load(
            #         settings.unweighted_events_dir + '/' + input_X_prefix + 'X_' + neyman_filename + '_null_' + str(tt) + '.npy')
            #     X_neyman_null_transformed = scaler.transform(
            #         X_neyman_null.reshape((-1, X_neyman_null.shape[2])))
            #
            #     # Neyman construction: evaluate null sample (raw)
            #     that_neyman_null = regr.predict(X_neyman_null_transformed)
            #     tthat_neyman_null = that_neyman_null.dot(delta_theta)
            #     that_rotated_neyman_null = that_neyman_null.dot(rotation_matrix)
            #
            #     llr_neyman_nulls.append(
            #         -2. * np.sum(tthat_neyman_null.reshape((-1, n_expected_events_neyman)), axis=1))
            #
            #     # Neyman construction: evaluate null sample (score * theta calibration)
            #     s_hat_neyman_null = calibrator_scoretheta.predict(tthat_neyman_null.reshape((-1,)))
            #     r_hat_neyman_null = r_from_s(s_hat_neyman_null)
            #     r_hat_neyman_null = r_hat_neyman_null.reshape((-1, n_expected_events_neyman))
            #     llr_neyman_nulls_scoretheta.append(-2. * np.sum(np.log(r_hat_neyman_null), axis=1))
            #
            #     # Neyman construction: evaluate null sample (score calibration)
            #     s_hat_neyman_null = calibrator_score.predict(that_neyman_null)
            #     r_hat_neyman_null = r_from_s(s_hat_neyman_null)
            #     r_hat_neyman_null = r_hat_neyman_null.reshape((-1, n_expected_events_neyman))
            #     llr_neyman_nulls_score.append(-2. * np.sum(np.log(r_hat_neyman_null), axis=1))
            #
            #     # Neyman construction: evaluate null sample (rotated score calibration)
            #     s_hat_neyman_null = calibrator_rotatedscore.predict(that_rotated_neyman_null)
            #     r_hat_neyman_null = r_from_s(s_hat_neyman_null)
            #     r_hat_neyman_null = r_hat_neyman_null.reshape((-1, n_expected_events_neyman))
            #     llr_neyman_nulls_rotatedscore.append(-2. * np.sum(np.log(r_hat_neyman_null), axis=1))

            # Neyman construction: load null sample
            X_neyman_null = np.load(
                settings.unweighted_events_dir + '/neyman/' + input_X_prefix + 'X_' + neyman_filename + '_null_' + str(
                    t) + '.npy')
            X_neyman_null_transformed = scaler.transform(
                X_neyman_null.reshape((-1, X_neyman_null.shape[2])))

            # Neyman construction: evaluate null sample (raw)
            that_neyman_null = regr.predict(X_neyman_null_transformed)
            tthat_neyman_null = that_neyman_null.dot(delta_theta)
            that_rotated_neyman_null = that_neyman_null.dot(rotation_matrix)
            llr_neyman_null = -2. * np.sum(tthat_neyman_null.reshape((-1, n_expected_events_neyman)), axis=1)

            # Neyman construction: evaluate null sample (score * theta calibration)
            s_hat_neyman_null = calibrator_scoretheta.predict(tthat_neyman_null.reshape((-1,)))
            r_hat_neyman_null = r_from_s(s_hat_neyman_null)
            r_hat_neyman_null = r_hat_neyman_null.reshape((-1, n_expected_events_neyman))
            llr_neyman_null_scoretheta = -2. * np.sum(np.log(r_hat_neyman_null), axis=1)

            # Neyman construction: evaluate null sample (score calibration)
            s_hat_neyman_null = calibrator_score.predict(that_neyman_null)
            r_hat_neyman_null = r_from_s(s_hat_neyman_null)
            r_hat_neyman_null = r_hat_neyman_null.reshape((-1, n_expected_events_neyman))
            llr_neyman_null_score = -2. * np.sum(np.log(r_hat_neyman_null), axis=1)

            # Neyman construction: evaluate null sample (rotated score calibration)
            s_hat_neyman_null = calibrator_rotatedscore.predict(that_rotated_neyman_null)
            r_hat_neyman_null = r_from_s(s_hat_neyman_null)
            r_hat_neyman_null = r_hat_neyman_null.reshape((-1, n_expected_events_neyman))
            llr_neyman_null_rotatedscore = -2. * np.sum(np.log(r_hat_neyman_null), axis=1)

            np.save(neyman_dir + '/' + neyman_filename + '_that_null_' + str(
                t) + '_scoreregression' + filename_addition + '.npy', that_neyman_null)
            np.save(neyman_dir + '/' + neyman_filename + '_llr_null_' + str(
                t) + '_scoreregression' + filename_addition + '.npy', llr_neyman_null)
            np.save(neyman_dir + '/' + neyman_filename + '_llr_null_' + str(
                t) + '_scoreregression_scoretheta' + filename_addition + '.npy', llr_neyman_null_scoretheta)
            np.save(neyman_dir + '/' + neyman_filename + '_llr_null_' + str(
                t) + '_scoreregression_score' + filename_addition + '.npy', llr_neyman_null_score)
            np.save(neyman_dir + '/' + neyman_filename + '_llr_null_' + str(
                t) + '_scoreregression_rotatedscore' + filename_addition + '.npy', llr_neyman_null_rotatedscore)

            # Neyman construction: null data evaluated at alternative
            if t == settings.theta_observed:
                for tt in range(settings.n_thetas):
                    X_neyman_null = np.load(
                        settings.unweighted_events_dir + '/neyman/' + input_X_prefix + 'X_' + neyman_filename + '_null_' + str(
                            tt) + '.npy')
                    X_neyman_null_transformed = scaler.transform(
                        X_neyman_null.reshape((-1, X_neyman_null.shape[2])))

                    # Neyman construction: evaluate null sample (raw)
                    that_neyman_null = regr.predict(X_neyman_null_transformed)
                    tthat_neyman_null = that_neyman_null.dot(delta_theta)
                    that_rotated_neyman_null = that_neyman_null.dot(rotation_matrix)
                    llr_neyman_null = -2. * np.sum(tthat_neyman_null.reshape((-1, n_expected_events_neyman)), axis=1)

                    # Neyman construction: evaluate null sample (score * theta calibration)
                    s_hat_neyman_null = calibrator_scoretheta.predict(tthat_neyman_null.reshape((-1,)))
                    r_hat_neyman_null = r_from_s(s_hat_neyman_null)
                    r_hat_neyman_null = r_hat_neyman_null.reshape((-1, n_expected_events_neyman))
                    llr_neyman_null_scoretheta = -2. * np.sum(np.log(r_hat_neyman_null), axis=1)

                    # Neyman construction: evaluate null sample (score calibration)
                    s_hat_neyman_null = calibrator_score.predict(that_neyman_null)
                    r_hat_neyman_null = r_from_s(s_hat_neyman_null)
                    r_hat_neyman_null = r_hat_neyman_null.reshape((-1, n_expected_events_neyman))
                    llr_neyman_null_score = -2. * np.sum(np.log(r_hat_neyman_null), axis=1)

                    # Neyman construction: evaluate null sample (rotated score calibration)
                    s_hat_neyman_null = calibrator_rotatedscore.predict(that_rotated_neyman_null)
                    r_hat_neyman_null = r_from_s(s_hat_neyman_null)
                    r_hat_neyman_null = r_hat_neyman_null.reshape((-1, n_expected_events_neyman))
                    llr_neyman_null_rotatedscore = -2. * np.sum(np.log(r_hat_neyman_null), axis=1)

                    np.save(neyman_dir + '/' + neyman_filename + '_llr_nullatalternate_' + str(
                        tt) + '_scoreregression' + filename_addition + '.npy', llr_neyman_null)
                    np.save(neyman_dir + '/' + neyman_filename + '_llr_nullatalternate_' + str(
                        tt) + '_scoreregression_scoretheta' + filename_addition + '.npy', llr_neyman_null_scoretheta)
                    np.save(neyman_dir + '/' + neyman_filename + '_llr_nullatalternate_' + str(
                        tt) + '_scoreregression_score' + filename_addition + '.npy', llr_neyman_null_score)
                    np.save(neyman_dir + '/' + neyman_filename + '_llr_nullatalternate_' + str(
                        tt) + '_scoreregression_rotatedscore' + filename_addition + '.npy',
                            llr_neyman_null_rotatedscore)

    # Evaluation times
    logging.info('Score density estimation timing: median %s s, mean %s s',
                 np.median(eval_times_scoretheta), np.mean(eval_times_scoretheta))
    logging.info('Rotated score density estimation timing: median %s s, mean %s s',
                 np.median(eval_times_rotatedscore), np.mean(eval_times_rotatedscore))
    logging.info('Score times theta density estimation timing: median %s s, mean %s s',
                 np.median(eval_times_score), np.mean(eval_times_score))

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

    expected_r_vs_sm_score = np.asarray(expected_r_vs_sm_score)
    expected_r_vs_sm_scoretheta = np.asarray(expected_r_vs_sm_scoretheta)
    expected_r_vs_sm_rotatedscore = np.asarray(expected_r_vs_sm_rotatedscore)

    recalibration_expected_r_score = np.asarray(recalibration_expected_r_score)
    recalibration_expected_r_rotatedscore = np.asarray(recalibration_expected_r_rotatedscore)
    recalibration_expected_r_scoretheta = np.asarray(recalibration_expected_r_scoretheta)

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

    np.save(results_dir + '/expected_r_vs_sm_scoreregression_score' + filename_addition + '.npy',
            expected_r_vs_sm_score)
    np.save(results_dir + '/expected_r_vs_sm_scoreregression_scoretheta' + filename_addition + '.npy',
            expected_r_vs_sm_scoretheta)
    np.save(results_dir + '/expected_r_vs_sm_scoreregression_rotatedscore' + filename_addition + '.npy',
            expected_r_vs_sm_rotatedscore)

    np.save(results_dir + '/recalibration_expected_r_vs_sm_scoreregression_score' + filename_addition + '.npy',
            recalibration_expected_r_score)
    np.save(results_dir + '/recalibration_expected_r_vs_sm_scoreregression_scoretheta' + filename_addition + '.npy',
            recalibration_expected_r_scoretheta)
    np.save(results_dir + '/recalibration_expected_r_vs_sm_scoreregression_rotatedscore' + filename_addition + '.npy',
            recalibration_expected_r_rotatedscore)
