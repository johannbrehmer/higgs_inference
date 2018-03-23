################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import logging
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from scipy.interpolate import LinearNDInterpolator

from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, LearningRateScheduler

from carl.ratios import ClassifierScoreRatio
from carl.learning import CalibratedClassifierScoreCV

from higgs_inference import settings
from higgs_inference.various.utils import calculate_mean_squared_error
from higgs_inference.models.models_point_by_point import make_classifier, make_regressor
from higgs_inference.models.ml_utils import DetailedHistory


def point_by_point_inference(algorithm='carl',
                             use_smearing=False,
                             denominator=0,
                             do_neyman=False,
                             options=''):
    """
    Trains and evaluates one of the point-by-point higgs_inference methods.

    :param use_smearing:
    :param do_neyman:
    :param algorithm: Type of the algorithm used. Currently supported: 'carl' and 'regression'.
    :param options: Further options in a list of strings or string.
    """

    logging.info('Starting point-by-point inference')

    ################################################################################
    # Settings
    ################################################################################

    assert algorithm in ['carl', 'regression']

    debug_mode = ('debug' in options)
    learn_logr_mode = ('learns' not in options)
    short_mode = ('short' in options)
    long_mode = ('long' in options)
    deep_mode = ('deep' in options)
    shallow_mode = ('shallow' in options)
    small_lr_mode = ('slowlearning' in options)
    large_lr_mode = ('fastlearning' in options)
    large_batch_mode = ('largebatch' in options)
    small_batch_mode = ('smallbatch' in options)
    constant_lr_mode = ('constantlr' in options)

    filename_addition = ''

    if not learn_logr_mode:
        filename_addition += '_learns'

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
    if debug_mode:
        n_epochs = settings.n_epochs_short
        early_stopping = False
        filename_addition += '_debug'
    elif long_mode:
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

    results_dir = settings.base_dir + '/results/point_by_point'
    neyman_dir = settings.neyman_dir + '/point_by_point'

    logging.info('Main settings:')
    logging.info('  Algorithm:               %s', algorithm)
    logging.info('Options:')
    logging.info('  Number of hidden layers: %s', n_hidden_layers)
    logging.info('  Batch size:              %s', batch_size)
    logging.info('  Learning rate:           %s', learning_rate)
    logging.info('  Learning rate decay:     %s', lr_decay)
    logging.info('  Number of epochs:        %s', n_epochs)
    logging.info('  Denominator theta:       denominator %s = theta %s = %s', denominator, theta1,
                 settings.thetas[theta1])

    ################################################################################
    # Data
    ################################################################################

    X_calibration = np.load(
        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_calibration' + input_filename_addition + '.npy')
    weights_calibration = np.load(
        settings.unweighted_events_dir + '/weights_calibration' + input_filename_addition + '.npy')

    X_test = np.load(
        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_test' + input_filename_addition + '.npy')
    r_test = np.load(settings.unweighted_events_dir + '/r_test' + input_filename_addition + '.npy')
    X_neyman_observed = np.load(settings.unweighted_events_dir + '/neyman/' + input_X_prefix + 'X_neyman_observed.npy')

    n_events_test = X_test.shape[0]
    assert settings.n_thetas == r_test.shape[0]

    ################################################################################
    # Loop over thetas
    ################################################################################

    expected_llr = []
    expected_llr_calibrated = []
    mse_log_r = []
    mse_log_r_calibrated = []
    trimmed_mse_log_r = []
    trimmed_mse_log_r_calibrated = []

    # Loop over the 15 thetas
    for i, t in enumerate(settings.pbp_training_thetas):

        logging.info('Starting theta %s/%s: number %s (%s)', i + 1, len(settings.pbp_training_thetas), t,
                     settings.thetas[t])

        # Load data
        X_train = np.load(
            settings.unweighted_events_dir + '/point_by_point/' + input_X_prefix + 'X_train_point_by_point_' + str(
                t) + input_filename_addition + '.npy')
        r_train = np.load(
            settings.unweighted_events_dir + '/point_by_point/r_train_point_by_point_' + str(t) + input_filename_addition + '.npy')
        y_train = np.load(
            settings.unweighted_events_dir + '/point_by_point/y_train_point_by_point_' + str(t) + input_filename_addition + '.npy')

        # Shuffle training data
        X_train, y_train, r_train = shuffle(X_train, y_train, r_train, random_state=44)

        # Mash together
        y_logr_train = np.hstack((y_train.reshape(-1, 1), np.log(r_train).reshape((-1, 1))))
        assert np.all(np.isfinite(y_logr_train))

        # Scale data
        scaler = StandardScaler()
        scaler.fit(np.array(X_train, dtype=np.float64))
        X_train_transformed = scaler.transform(X_train)
        X_test_transformed = scaler.transform(X_test)
        X_calibration_transformed = scaler.transform(X_calibration)
        X_neyman_observed_transformed = scaler.transform(
            X_neyman_observed.reshape((-1, X_neyman_observed.shape[2])))

        ################################################################################
        # Training
        ################################################################################

        if algorithm == 'carl':
            regr = KerasRegressor(lambda: make_classifier(n_hidden_layers=n_hidden_layers,
                                                          learn_log_r=learn_logr_mode,
                                                          learning_rate=learning_rate),
                                  epochs=n_epochs, validation_split=settings.validation_split,
                                  verbose=2)

        elif algorithm == 'regression':
            regr = KerasRegressor(lambda: make_regressor(n_hidden_layers=n_hidden_layers,
                                                         learning_rate=learning_rate),
                                  epochs=n_epochs, validation_split=settings.validation_split,
                                  verbose=2)

        else:
            raise ValueError

        # Callbacks
        callbacks = []
        detailed_history = {}
        callbacks.append(DetailedHistory(detailed_history))
        if not constant_lr_mode:
            def lr_scheduler(epoch):
                return learning_rate * np.exp(- epoch * lr_decay)

            callbacks.append(LearningRateScheduler(lr_scheduler))
        if early_stopping:
            callbacks.append(EarlyStopping(verbose=1, patience=settings.early_stopping_patience))

        # Training
        history = regr.fit(X_train_transformed, y_logr_train,
                           callbacks=callbacks, batch_size=batch_size)

        # Save metrics
        def _save_metrics(key, filename):
            try:
                metrics = np.asarray([history.history[key], history.history['val_' + key]])
                np.save(results_dir + '/traininghistory_' + filename + '_' + algorithm + filename_addition + '.npy',
                        metrics)
            except KeyError:
                logging.warning('Key %s not found. Available keys: %s', key, list(history.history.keys()))

        if t == settings.theta_benchmark_nottrained:
            _save_metrics('loss', 'loss_nottrained')
            _save_metrics('full_cross_entropy', 'ce_nottrained')
            _save_metrics('full_mse_log_r', 'mse_logr_nottrained')
        elif t == settings.theta_benchmark_trained:
            _save_metrics('loss', 'loss_trained')
            _save_metrics('full_cross_entropy', 'ce_trained')
            _save_metrics('full_mse_log_r', 'mse_logr_trained')

        ################################################################################
        # Evaluation
        ################################################################################

        # carl wrapper
        # ratio = ClassifierScoreRatio(regr, prefit=True)

        # Evaluation
        prediction = regr.predict(X_test_transformed)
        this_log_r = prediction[:, 1]

        # Extract numbers of interest
        expected_llr.append(- 2. * settings.n_expected_events / n_events_test * np.sum(this_log_r))
        mse_log_r.append(calculate_mean_squared_error(np.log(r_test[t]), this_log_r, 0.))
        trimmed_mse_log_r.append(calculate_mean_squared_error(np.log(r_test[t]), this_log_r, 'auto'))

        # For benchmark thetas, save more info
        if t == settings.theta_benchmark_nottrained:
            np.save(results_dir + '/r_nottrained_' + algorithm + filename_addition + '.npy', np.exp(this_log_r))
        elif t == settings.theta_benchmark_trained:
            np.save(results_dir + '/r_trained_' + algorithm + filename_addition + '.npy', np.exp(this_log_r))

        ################################################################################
        # Calibration
        ################################################################################

        # Calibration
        n_calibration_each = X_calibration_transformed.shape[0]
        X_calibration_both = np.zeros((2 * n_calibration_each, X_calibration_transformed.shape[1]))
        X_calibration_both[:n_calibration_each] = X_calibration_transformed
        X_calibration_both[n_calibration_each:] = X_calibration_transformed
        y_calibration = np.zeros(2 * n_calibration_each)
        y_calibration[n_calibration_each:] = 1.
        w_calibration = np.zeros(2 * n_calibration_each)
        w_calibration[:n_calibration_each] = weights_calibration[t]
        w_calibration[n_calibration_each:] = weights_calibration[theta1]

        ratio_calibrated = ClassifierScoreRatio(
            CalibratedClassifierScoreCV(regr, cv='prefit', method='isotonic')
        )
        ratio_calibrated.fit(X_calibration_both, y_calibration, sample_weight=w_calibration)

        # Evaluation of calibrated classifier
        this_r, _ = ratio_calibrated.predict(X_test_transformed)

        # Extract numbers of interest
        expected_llr_calibrated.append(- 2. * settings.n_expected_events / n_events_test * np.sum(np.log(this_r)))
        mse_log_r_calibrated.append(calculate_mean_squared_error(np.log(r_test[t]), np.log(this_r), 0.))
        trimmed_mse_log_r_calibrated.append(calculate_mean_squared_error(np.log(r_test[t]), np.log(this_r), 'auto'))

        # For benchmark theta, save more data
        if t == settings.theta_benchmark_nottrained:
            np.save(results_dir + '/r_nottrained_' + algorithm + '_calibrated' + filename_addition + '.npy', this_r)

            # Save calibration histograms
            np.save(results_dir + '/calvalues_nottrained_' + algorithm + filename_addition + '.npy',
                    ratio_calibrated.classifier_.calibration_sample[:n_calibration_each])
        elif t == settings.theta_benchmark_trained:
            np.save(results_dir + '/r_trained_' + algorithm + '_calibrated' + filename_addition + '.npy', this_r)

            # Save calibration histograms
            np.save(results_dir + '/calvalues_trained_' + algorithm + filename_addition + '.npy',
                    ratio_calibrated.classifier_.calibration_sample[:n_calibration_each])

        ################################################################################
        # Neyman construction
        ################################################################################

        # if do_neyman:
        #     # Neyman construction: evaluate observed sample (raw)
        #     log_r_neyman_observed = regr.predict(X_neyman_observed_transformed)[:, 1]
        #     llr_neyman_observed = -2. * np.sum(log_r_neyman_observed.reshape((-1, settings.n_expected_events)),
        #                                        axis=1)
        #     np.save(neyman_dir + '/neyman_llr_observed_' + algorithm + '_' + str(t) + filename_addition + '.npy',
        #             llr_neyman_observed)
        #
        #     if do_neyman_calibrated:
        #         # Neyman construction: evaluate observed sample (calibrated)
        #         r_neyman_observed, _ = ratio_calibrated.predict(X_neyman_observed_transformed)
        #         llr_neyman_observed = -2. * np.sum(
        #             np.log(r_neyman_observed).reshape((-1, settings.n_expected_events)),
        #             axis=1)
        #         np.save(
        #             neyman_dir + '/neyman_llr_observed_' + algorithm + '_calibrated_' + str(
        #                 t) + filename_addition + '.npy',
        #             llr_neyman_observed)
        #
        #     # Neyman construction: loop over distribution samples generated from different thetas
        #     llr_neyman_distributions = []
        #     llr_neyman_distributions_calibrated = []
        #     for tt in range(settings.n_thetas):
        #
        #         # Only evaluate certain combinations of thetas to save computation time
        #         if not decide_toy_evaluation(tt, t):
        #             placeholder = np.empty(settings.n_neyman_null_experiments)
        #             placeholder[:] = np.nan
        #             llr_neyman_distributions.append(placeholder)
        #             continue
        #
        #         # Neyman construction: load distribution sample
        #         X_neyman_distribution = np.load(
        #             settings.unweighted_events_dir + '/' + input_X_prefix + 'X_neyman_distribution_' + str(
        #                 tt) + '.npy')
        #         X_neyman_distribution_transformed = scaler.transform(
        #             X_neyman_distribution.reshape((-1, X_neyman_distribution.shape[2])))
        #
        #         # Neyman construction: evaluate distribution sample (raw)
        #         log_r_neyman_distribution = regr.predict(X_neyman_distribution_transformed)[:, 1]
        #         llr_neyman_distributions.append(
        #             -2. * np.sum(log_r_neyman_distribution.reshape((-1, settings.n_expected_events)), axis=1))
        #
        #         if do_neyman_calibrated:
        #             # Neyman construction: evaluate distribution sample (calibrated)
        #             r_neyman_distribution, _ = ratio_calibrated.predict(X_neyman_distribution_transformed)
        #             llr_neyman_distributions_calibrated.append(
        #                 -2. * np.sum(np.log(r_neyman_distribution).reshape((-1, settings.n_expected_events)),
        #                              axis=1))
        #
        #     llr_neyman_distributions = np.asarray(llr_neyman_distributions)
        #     np.save(
        #         neyman_dir + '/neyman_llr_distribution_' + algorithm + '_' + str(t) + filename_addition + '.npy',
        #         llr_neyman_distributions)
        #
        #     if do_neyman_calibrated:
        #         llr_neyman_distributions_calibrated = np.asarray(llr_neyman_distributions_calibrated)
        #         np.save(neyman_dir + '/neyman_llr_distribution_' + algorithm + '_calibrated_' + str(t)
        #                 + filename_addition + '.npy',
        #                 llr_neyman_distributions_calibrated)

    # Interpolate and save evaluation results
    expected_llr = np.asarray(expected_llr)
    expected_llr_calibrated = np.asarray(expected_llr_calibrated)
    mse_log_r = np.asarray(mse_log_r)
    trimmed_mse_log_r = np.asarray(trimmed_mse_log_r)
    mse_log_r_calibrated = np.asarray(mse_log_r_calibrated)
    trimmed_mse_log_r_calibrated = np.asarray(trimmed_mse_log_r_calibrated)

    logging.info('Starting interpolation')

    interpolator = LinearNDInterpolator(settings.thetas[settings.pbp_training_thetas], expected_llr)
    expected_llr_all = interpolator(settings.thetas)
    # gp = GaussianProcessRegressor(normalize_y=True,
    #                              kernel=C(1.0) * Matern(1.0, nu=0.5), n_restarts_optimizer=10)
    # gp.fit(thetas[settings.pbp_training_thetas], expected_llr)
    # expected_llr_all = gp.predict(thetas)
    np.save(results_dir + '/llr_' + algorithm + filename_addition + '.npy', expected_llr_all)

    interpolator = LinearNDInterpolator(settings.thetas[settings.pbp_training_thetas], expected_llr_calibrated)
    expected_llr_calibrated_all = interpolator(settings.thetas)
    # gp = GaussianProcessRegressor(normalize_y=True,
    #                              kernel=C(1.0) * Matern(1.0, nu=0.5), n_restarts_optimizer=10)
    # gp.fit(thetas[settings.pbp_training_thetas], expected_llr_calibrated)
    # expected_llr_calibrated_all = gp.predict(thetas)
    np.save(results_dir + '/llr_' + algorithm + '_calibrated' + filename_addition + '.npy',
            expected_llr_calibrated_all)

    interpolator = LinearNDInterpolator(settings.thetas[settings.pbp_training_thetas], mse_log_r)
    mse_log_r_all = interpolator(settings.thetas)
    np.save(results_dir + '/mse_logr_' + algorithm + filename_addition + '.npy',
            mse_log_r_all)

    interpolator = LinearNDInterpolator(settings.thetas[settings.pbp_training_thetas], mse_log_r_calibrated)
    mse_log_r_calibrated_all = interpolator(settings.thetas)
    np.save(results_dir + '/mse_logr_' + algorithm + '_calibrated' + filename_addition + '.npy',
            mse_log_r_calibrated_all)

    interpolator = LinearNDInterpolator(settings.thetas[settings.pbp_training_thetas], trimmed_mse_log_r)
    trimmed_mse_log_r_all = interpolator(settings.thetas)
    np.save(results_dir + '/trimmed_mse_logr_' + algorithm + filename_addition + '.npy',
            trimmed_mse_log_r_all)

    interpolator = LinearNDInterpolator(settings.thetas[settings.pbp_training_thetas], trimmed_mse_log_r_calibrated)
    trimmed_mse_log_r_calibrated_all = interpolator(settings.thetas)
    np.save(results_dir + '/trimmed_mse_logr_' + algorithm + '_calibrated' + filename_addition + '.npy',
            trimmed_mse_log_r_calibrated_all)
