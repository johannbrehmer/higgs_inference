################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import math

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
from sklearn.utils import shuffle

from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, LearningRateScheduler

from carl.ratios import ClassifierScoreRatio
from carl.learning import CalibratedClassifierScoreCV

from higgs_inference import settings
from higgs_inference.various.utils import decide_toy_evaluation, format_number
from higgs_inference.models.models_parameterized import make_classifier_carl, make_classifier_carl_morphingaware
from higgs_inference.models.models_parameterized import make_classifier_score, make_classifier_score_morphingaware
from higgs_inference.models.models_parameterized import make_classifier_combined, make_classifier_combined_morphingaware
from higgs_inference.models.models_parameterized import make_regressor, make_regressor_morphingaware
from higgs_inference.models.models_parameterized import make_combined_regressor, make_combined_regressor_morphingaware
from higgs_inference.models.ml_utils import DetailedHistory


def parameterized_inference(algorithm='carl',  # 'carl', 'score', 'combined', 'regression', 'combinedregression'
                            morphing_aware=False,
                            training_sample='baseline',  # 'baseline', 'basis', 'random'
                            use_smearing=False,
                            alpha=None,
                            do_neyman=False,
                            do_neyman_calibrated=False,
                            options=''):  # all other options in a string

    """
    Trains and evaluates one of the parameterized higgs_inference methods.

    :param algorithm: Type of the algorithm used. Currently supported: 'carl', 'score', 'combined', 'regression', and
                      'combinedregression'.
    :param morphing_aware: bool that decides whether a morphing-aware or morphing-agnostic architecture is used.
    :param training_sample: Training sample. Can be 'baseline', 'basis', or 'random'.
    :param use_smearing:
    :param alpha: Factor that weights the score term in the if algorithm is 'combined' or 'combinedregression'.
    :param do_neyman: Switches on the evaluation of toy experiments for the Neyman construction for the raw algorithm.
    :param do_neyman_calibrated: Switches on the evaluation of toy experiments for the Neyman construction for the
                                 calibrated algorithm.
    :param options: Further options in a list of strings or string.
    """

    logging.info('Starting parameterized inference')

    ################################################################################
    # Settings
    ################################################################################

    assert algorithm in ['carl', 'score', 'combined', 'regression', 'combinedregression']
    assert training_sample in ['baseline', 'basis', 'random']

    random_theta_mode = training_sample == 'random'
    basis_theta_mode = training_sample == 'basis'

    learn_logr_mode = ('learns' not in options)
    denom1_mode = ('denom1' in options)
    short_mode = ('short' in options)
    long_mode = ('long' in options)
    deep_mode = ('deep' in options)
    shallow_mode = ('shallow' in options)
    debug_mode = ('debug' in options)
    factor_out_sm_in_aware_mode = morphing_aware and ('factorsm' in options)
    small_lr_mode = ('slowlearning' in options)
    large_lr_mode = ('fastlearning' in options)
    large_batch_mode = ('largebatch' in options)
    small_batch_mode = ('smallbatch' in options)
    constant_lr_mode = ('constantlr' in options)

    filename_addition = ''
    if morphing_aware:
        filename_addition = '_aware'

    if random_theta_mode:
        filename_addition += '_random'
    elif basis_theta_mode:
        filename_addition += '_basis'

    if not learn_logr_mode:
        filename_addition += '_learns'

    if factor_out_sm_in_aware_mode:
        filename_addition += '_factorsm'

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

    alpha_regression = settings.alpha_regression_default
    alpha_carl = settings.alpha_carl_default
    if alpha is not None:
        alpha_regression = alpha
        alpha_carl = alpha
        precision = int(max(- math.floor(np.log10(alpha)) + 1, 0))
        filename_addition += '_alpha_' + format_number(alpha, precision)

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
    if denom1_mode:
        input_filename_addition = '_denom1'
        filename_addition += '_denom1'
        theta1 = settings.theta1_alternative

    results_dir = settings.base_dir + '/results/parameterized'
    neyman_dir = settings.neyman_dir + '/parameterized'

    logging.info('Main settings:')
    logging.info('  Algorithm:                %s', algorithm)
    logging.info('  Morphing-aware:           %s', morphing_aware)
    logging.info('  Training sample:          %s', training_sample)
    logging.info('Options:')
    logging.info('  Number of hidden layers:  %s', n_hidden_layers)
    if algorithm == 'combined':
        logging.info('  alpha:                    %s', alpha_carl)
    elif algorithm == 'combinedregression':
        logging.info('  alpha:                    %s', alpha_regression)
    logging.info('  Batch size:               %s', batch_size)
    logging.info('  Learning rate:            %s', learning_rate)
    logging.info('  Learning rate decay:      %s', lr_decay)
    logging.info('  Number of epochs:         %s', n_epochs)
    logging.info('  Debug mode:               %s', debug_mode)

    ################################################################################
    # Data
    ################################################################################

    # Load data
    if random_theta_mode:
        X_train = np.load(
            settings.unweighted_events_dir + '/' + input_X_prefix + 'X_train_random' + input_filename_addition + '.npy')
        y_train = np.load(settings.unweighted_events_dir + '/y_train_random' + input_filename_addition + '.npy')
        scores_train = np.load(
            settings.unweighted_events_dir + '/scores_train_random' + input_filename_addition + '.npy')
        r_train = np.load(settings.unweighted_events_dir + '/r_train_random' + input_filename_addition + '.npy')
        theta0_train = np.load(
            settings.unweighted_events_dir + '/theta0_train_random' + input_filename_addition + '.npy')
    elif basis_theta_mode:
        X_train = np.load(
            settings.unweighted_events_dir + '/' + input_X_prefix + 'X_train_basis' + input_filename_addition + '.npy')
        y_train = np.load(settings.unweighted_events_dir + '/y_train_basis' + input_filename_addition + '.npy')
        scores_train = np.load(
            settings.unweighted_events_dir + '/scores_train_basis' + input_filename_addition + '.npy')
        r_train = np.load(settings.unweighted_events_dir + '/r_train_basis' + input_filename_addition + '.npy')
        theta0_train = np.load(
            settings.unweighted_events_dir + '/theta0_train_basis' + input_filename_addition + '.npy')
    else:
        X_train = np.load(
            settings.unweighted_events_dir + '/' + input_X_prefix + 'X_train' + input_filename_addition + '.npy')
        y_train = np.load(settings.unweighted_events_dir + '/y_train' + input_filename_addition + '.npy')
        scores_train = np.load(settings.unweighted_events_dir + '/scores_train' + input_filename_addition + '.npy')
        r_train = np.load(settings.unweighted_events_dir + '/r_train' + input_filename_addition + '.npy')
        theta0_train = np.load(settings.unweighted_events_dir + '/theta0_train' + input_filename_addition + '.npy')

    X_calibration = np.load(
        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_calibration' + input_filename_addition + '.npy')
    weights_calibration = np.load(
        settings.unweighted_events_dir + '/weights_calibration' + input_filename_addition + '.npy')

    X_test = np.load(
        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_test' + input_filename_addition + '.npy')
    r_test = np.load(settings.unweighted_events_dir + '/r_test' + input_filename_addition + '.npy')

    X_roam = np.load(
        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_roam' + input_filename_addition + '.npy')
    n_roaming = len(X_roam)

    if do_neyman:
        X_neyman_observed = np.load(settings.unweighted_events_dir + '/' + input_X_prefix + 'X_neyman_observed.npy')

    n_events_test = X_test.shape[0]
    assert settings.n_thetas == r_test.shape[0]

    # Shuffle training data
    X_train, y_train, scores_train, r_train, theta0_train = shuffle(X_train, y_train, scores_train, r_train,
                                                                    theta0_train, random_state=44)

    # Normalize data
    scaler = StandardScaler()
    scaler.fit(np.array(X_train, dtype=np.float64))
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    X_roam_transformed = scaler.transform(X_roam)
    X_calibration_transformed = scaler.transform(X_calibration)
    if do_neyman:
        X_neyman_observed_transformed = scaler.transform(
            X_neyman_observed.reshape((-1, X_neyman_observed.shape[2])))

    # Roaming data
    X_thetas_train = np.hstack((X_train_transformed, theta0_train))
    y_logr_score_train = np.hstack((y_train.reshape(-1, 1), np.log(r_train).reshape((-1, 1)), scores_train))
    xi = np.linspace(-1.0, 1.0, settings.n_thetas_roam)
    yi = np.linspace(-1.0, 1.0, settings.n_thetas_roam)
    xx, yy = np.meshgrid(xi, yi)
    thetas_roam = np.asarray((xx.flatten(), yy.flatten())).T
    X_thetas_roam = []
    for i in range(n_roaming):
        X_thetas_roam.append(np.zeros((settings.n_thetas_roam ** 2, X_roam_transformed.shape[1] + 2)))
        X_thetas_roam[-1][:, :-2] = X_roam_transformed[i, :]
        X_thetas_roam[-1][:, -2:] = thetas_roam

    if debug_mode:
        X_thetas_train = X_thetas_train[::100]
        y_logr_score_train = y_logr_score_train[::100]
        X_test_transformed = X_test[::100]
        X_calibration_transformed = X_calibration_transformed[::100]
        weights_calibration = weights_calibration[:, ::100]
        n_events_test = len(X_test_transformed)

    ################################################################################
    # Training
    ################################################################################

    if algorithm == 'carl':
        if morphing_aware:
            regr = KerasRegressor(lambda: make_classifier_carl_morphingaware(n_hidden_layers=n_hidden_layers,
                                                                             learn_log_r=learn_logr_mode,
                                                                             learning_rate=learning_rate),
                                  epochs=n_epochs, validation_split=settings.validation_split,
                                  verbose=2)
        else:
            regr = KerasRegressor(lambda: make_classifier_carl(n_hidden_layers=n_hidden_layers,
                                                               learn_log_r=learn_logr_mode,
                                                               learning_rate=learning_rate),
                                  epochs=n_epochs, validation_split=settings.validation_split,
                                  verbose=2)

    elif algorithm == 'score':
        if morphing_aware:
            regr = KerasRegressor(lambda: make_classifier_score_morphingaware(n_hidden_layers=n_hidden_layers,
                                                                              learn_log_r=learn_logr_mode,
                                                                              learning_rate=learning_rate),
                                  epochs=n_epochs, validation_split=settings.validation_split,
                                  verbose=2)
        else:
            regr = KerasRegressor(lambda: make_classifier_score(n_hidden_layers=n_hidden_layers,
                                                                learn_log_r=learn_logr_mode,
                                                                learning_rate=learning_rate),
                                  epochs=n_epochs, validation_split=settings.validation_split,
                                  verbose=2)

    elif algorithm == 'combined':
        if morphing_aware:
            regr = KerasRegressor(
                lambda: make_classifier_combined_morphingaware(n_hidden_layers=n_hidden_layers,
                                                               learn_log_r=learn_logr_mode,
                                                               alpha=alpha_carl,
                                                               learning_rate=learning_rate),
                epochs=n_epochs, validation_split=settings.validation_split,
                verbose=2)
        else:
            regr = KerasRegressor(lambda: make_classifier_combined(n_hidden_layers=n_hidden_layers,
                                                                   learn_log_r=learn_logr_mode,
                                                                   alpha=alpha_carl,
                                                                   learning_rate=learning_rate),
                                  epochs=n_epochs, validation_split=settings.validation_split,
                                  verbose=2)

    elif algorithm == 'regression':
        if morphing_aware:
            regr = KerasRegressor(lambda: make_regressor_morphingaware(n_hidden_layers=n_hidden_layers,
                                                                       factor_out_sm=factor_out_sm_in_aware_mode,
                                                                       learning_rate=learning_rate),
                                  epochs=n_epochs, validation_split=settings.validation_split,
                                  verbose=2)
        else:
            regr = KerasRegressor(lambda: make_regressor(n_hidden_layers=n_hidden_layers),
                                  epochs=n_epochs, validation_split=settings.validation_split,
                                  verbose=2)

    elif algorithm == 'combinedregression':
        if morphing_aware:
            regr = KerasRegressor(
                lambda: make_combined_regressor_morphingaware(n_hidden_layers=n_hidden_layers,
                                                              factor_out_sm=factor_out_sm_in_aware_mode,
                                                              alpha=alpha_regression,
                                                              learning_rate=learning_rate),
                epochs=n_epochs, validation_split=settings.validation_split,
                verbose=2)
        else:
            regr = KerasRegressor(lambda: make_combined_regressor(n_hidden_layers=n_hidden_layers,
                                                                  alpha=alpha_regression,
                                                                  learning_rate=learning_rate),
                                  epochs=n_epochs, validation_split=settings.validation_split,
                                  verbose=2)

    else:
        raise ValueError()

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
    logging.info('Starting training')
    history = regr.fit(X_thetas_train[::], y_logr_score_train[::], callbacks=callbacks, batch_size=batch_size)

    # Save metrics
    def _save_metrics(key, filename):
        try:
            metrics = np.asarray([history.history[key], history.history['val_' + key]])
            np.save(results_dir + '/traininghistory_' + filename + '_' + algorithm + filename_addition + '.npy',
                    metrics)
        except KeyError:
            logging.warning('Key %s not found in per-epoch history. Available keys: %s', key,
                            list(history.history.keys()))
        try:
            detailed_metrics = np.asarray(detailed_history[key])
            np.save(
                results_dir + '/traininghistory_100batches_' + filename + '_' + algorithm + filename_addition + '.npy',
                detailed_metrics)
        except KeyError:
            logging.warning('Key %s not found in per-batch history. Available keys: %s', key,
                            list(detailed_history.keys()))

    _save_metrics('loss', 'loss')
    _save_metrics('full_cross_entropy', 'ce')
    _save_metrics('full_mse_log_r', 'mse_logr')
    _save_metrics('full_mse_score', 'mse_scores')

    ################################################################################
    # Evaluation
    ################################################################################

    # carl ratio object
    # ratio = ClassifierScoreRatio(regr, prefit=True)

    logging.info('Starting evaluation')
    expected_llr = []

    for t, theta in enumerate(settings.thetas):

        # Prepare test data
        thetas0_array = np.zeros((X_test_transformed.shape[0], 2), dtype=X_test_transformed.dtype)
        thetas0_array[:, :] = theta
        X_thetas_test = np.hstack((X_test_transformed, thetas0_array))

        # Evaluation
        prediction = regr.predict(X_thetas_test)
        this_logr = prediction[:, 1]
        this_score = prediction[:, 2:4]
        if morphing_aware:
            this_wi = prediction[:, 4:19]
            this_ri = prediction[:, 19:]
            logging.debug('Morphing weights for theta %s (%s): %s', t, theta, this_wi[0])

        expected_llr.append(- 2. * settings.n_expected_events / n_events_test * np.sum(this_logr))

        # For benchmark thetas, save more info
        if t == settings.theta_benchmark_nottrained:
            np.save(results_dir + '/r_nottrained_' + algorithm + filename_addition + '.npy', np.exp(this_logr))
            np.save(results_dir + '/scores_nottrained_' + algorithm + filename_addition + '.npy', this_score)
            if morphing_aware:
                np.save(results_dir + '/morphing_ri_nottrained_' + algorithm + filename_addition + '.npy', this_ri)
                np.save(results_dir + '/morphing_wi_nottrained_' + algorithm + filename_addition + '.npy', this_wi)
        elif t == settings.theta_benchmark_trained:
            np.save(results_dir + '/r_trained_' + algorithm + filename_addition + '.npy', np.exp(this_logr))
            np.save(results_dir + '/scores_trained_' + algorithm + filename_addition + '.npy', this_score)
            if morphing_aware:
                np.save(results_dir + '/morphing_ri_trained_' + algorithm + filename_addition + '.npy', this_ri)
                np.save(results_dir + '/morphing_wi_trained_' + algorithm + filename_addition + '.npy', this_wi)

        if do_neyman:
            # Prepare observed data for Neyman construction
            thetas0_array = np.zeros((X_neyman_observed_transformed.shape[0], 2),
                                     dtype=X_neyman_observed_transformed.dtype)
            thetas0_array[:, :] = theta
            X_thetas_neyman_observed = np.hstack((X_neyman_observed_transformed, thetas0_array))

            # Neyman construction: evaluate observed sample (raw)
            log_r_neyman_observed = regr.predict(X_thetas_neyman_observed)[:, 1]
            llr_neyman_observed = -2. * np.sum(log_r_neyman_observed.reshape((-1, settings.n_expected_events)),
                                               axis=1)
            np.save(neyman_dir + '/neyman_llr_observed_' + algorithm + '_' + str(t) + filename_addition + '.npy',
                    llr_neyman_observed)

            # Neyman construction: loop over distribution samples generated from different thetas
            llr_neyman_distributions = []
            for tt in range(settings.n_thetas):

                # Only evaluate certain combinations of thetas to save computation time
                if not decide_toy_evaluation(tt, t):
                    placeholder = np.empty(settings.n_neyman_distribution_experiments)
                    placeholder[:] = np.nan
                    llr_neyman_distributions.append(placeholder)
                    continue

                # Neyman construction: load distribution sample
                X_neyman_distribution = np.load(
                    settings.unweighted_events_dir + '/' + input_X_prefix + 'X_neyman_distribution_' + str(
                        tt) + '.npy')
                X_neyman_distribution_transformed = scaler.transform(
                    X_neyman_distribution.reshape((-1, X_neyman_distribution.shape[2])))

                # Prepare distribution data for Neyman construction
                thetas0_array = np.zeros((X_neyman_distribution_transformed.shape[0], 2),
                                         dtype=X_neyman_distribution_transformed.dtype)
                thetas0_array[:, :] = settings.thetas[t]
                X_thetas_neyman_distribution = np.hstack((X_neyman_distribution_transformed, thetas0_array))

                # Neyman construction: evaluate distribution sample (raw)
                log_r_neyman_distribution = regr.predict(X_thetas_neyman_distribution)[:, 1]
                llr_neyman_distributions.append(
                    -2. * np.sum(log_r_neyman_distribution.reshape((-1, settings.n_expected_events)), axis=1))

            llr_neyman_distributions = np.asarray(llr_neyman_distributions)
            np.save(
                neyman_dir + '/neyman_llr_distribution_' + algorithm + '_' + str(t) + filename_addition + '.npy',
                llr_neyman_distributions)

    expected_llr = np.asarray(expected_llr)
    np.save(results_dir + '/llr_' + algorithm + filename_addition + '.npy', expected_llr)

    logging.info('Starting roaming')
    r_roam = []
    for i in range(n_roaming):
        prediction = regr.predict(X_thetas_roam[i])
        r_roam.append(np.exp(prediction[:, 1]))
    r_roam = np.asarray(r_roam)
    np.save(results_dir + '/r_roam_' + algorithm + filename_addition + '.npy', r_roam)

    ################################################################################
    # Calibration
    ################################################################################

    logging.info('Starting calibrated evaluation and roaming')
    expected_llr_calibrated = []
    r_roam_temp = np.zeros((settings.n_thetas, n_roaming))

    for t, theta in enumerate(settings.thetas):

        # Prepare data for calibration
        n_calibration_each = X_calibration_transformed.shape[0]
        thetas0_array = np.zeros((n_calibration_each, 2), dtype=X_calibration_transformed.dtype)
        thetas0_array[:, :] = settings.thetas[t]
        X_thetas_calibration = np.hstack((X_calibration_transformed, thetas0_array))
        X_thetas_calibration = np.vstack((X_thetas_calibration, X_thetas_calibration))
        y_calibration = np.zeros(2 * n_calibration_each)
        y_calibration[n_calibration_each:] = 1.
        w_calibration = np.zeros(2 * n_calibration_each)
        w_calibration[:n_calibration_each] = weights_calibration[t]
        w_calibration[n_calibration_each:] = weights_calibration[theta1]

        # Calibration
        ratio_calibrated = ClassifierScoreRatio(
            CalibratedClassifierScoreCV(regr, cv='prefit', method='isotonic')
        )
        ratio_calibrated.fit(X_thetas_calibration, y_calibration, sample_weight=w_calibration)

        # Prepare data
        thetas0_array = np.zeros((X_test_transformed.shape[0], 2), dtype=X_test_transformed.dtype)
        thetas0_array[:, :] = settings.thetas[t]
        X_thetas_test = np.hstack((X_test_transformed, thetas0_array))

        this_r, this_other = ratio_calibrated.predict(X_thetas_test)
        this_score = this_other[:, 1:3]

        expected_llr_calibrated.append(- 2. * settings.n_expected_events / n_events_test * np.sum(np.log(this_r)))

        # For benchmark theta, save more data
        if t == settings.theta_benchmark_nottrained:
            np.save(results_dir + '/scores_nottrained_' + algorithm + '_calibrated' + filename_addition + '.npy',
                    this_score)
            np.save(results_dir + '/r_nottrained_' + algorithm + '_calibrated' + filename_addition + '.npy', this_r)
            np.save(results_dir + '/calvalues_nottrained_' + algorithm + filename_addition + '.npy',
                    ratio_calibrated.classifier_.calibration_sample[:n_calibration_each])
            # np.save(results_dir + '/cal0histo_nottrained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator0.histogram_)
            # np.save(results_dir + '/cal0edges_nottrained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator0.edges_[0])
            # np.save(results_dir + '/cal1histo_nottrained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator1.histogram_)
            # np.save(results_dir + '/cal1edges_nottrained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator1.edges_[0])

        elif t == settings.theta_benchmark_trained:
            np.save(results_dir + '/scores_trained_' + algorithm + '_calibrated' + filename_addition + '.npy',
                    this_score)
            np.save(results_dir + '/r_trained_' + algorithm + '_calibrated' + filename_addition + '.npy', this_r)
            np.save(results_dir + '/calvalues_trained_' + algorithm + filename_addition + '.npy',
                    ratio_calibrated.classifier_.calibration_sample[:n_calibration_each])
            # np.save(results_dir + '/cal0histo_trained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator0.histogram_)
            # np.save(results_dir + '/cal0edges_trained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator0.edges_[0])
            # np.save(results_dir + '/cal1histo_trained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator1.histogram_)
            # np.save(results_dir + '/cal1edges_trained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator1.edges_[0])

        if do_neyman_calibrated:
            # Prepare observed data for Neyman construction
            thetas0_array = np.zeros((X_neyman_observed_transformed.shape[0], 2),
                                     dtype=X_neyman_observed_transformed.dtype)
            thetas0_array[:, :] = settings.thetas[t]
            X_thetas_neyman_observed = np.hstack((X_neyman_observed_transformed, thetas0_array))

            # Neyman construction: evaluate observed sample (raw)
            r_neyman_observed, _ = ratio_calibrated.predict(X_thetas_neyman_observed)
            llr_neyman_observed = -2. * np.sum(np.log(r_neyman_observed).reshape((-1, settings.n_expected_events)),
                                               axis=1)
            np.save(
                neyman_dir + '/neyman_llr_observed_' + algorithm + '_calibrated_' + str(
                    t) + filename_addition + '.npy',
                llr_neyman_observed)

            # Neyman construction: loop over distribution samples generated from different thetas
            llr_neyman_distributions = []
            for tt in range(settings.n_thetas):
                # Neyman construction: load distribution sample
                X_neyman_distribution = np.load(
                    settings.unweighted_events_dir + '/' + input_X_prefix + 'X_neyman_distribution_' + str(
                        tt) + '.npy')
                X_neyman_distribution_transformed = scaler.transform(
                    X_neyman_distribution.reshape((-1, X_neyman_distribution.shape[2])))

                # Prepare distribution data for Neyman construction
                thetas0_array = np.zeros((X_neyman_distribution_transformed.shape[0], 2),
                                         dtype=X_neyman_distribution_transformed.dtype)
                thetas0_array[:, :] = settings.thetas[t]
                X_thetas_neyman_distribution = np.hstack((X_neyman_distribution_transformed, thetas0_array))

                # Neyman construction: evaluate distribution sample (calibrated)
                r_neyman_distribution, _ = ratio_calibrated.predict(X_thetas_neyman_distribution)
                llr_neyman_distributions.append(
                    -2. * np.sum(np.log(r_neyman_distribution).reshape((-1, settings.n_expected_events)), axis=1))

            llr_neyman_distributions = np.asarray(llr_neyman_distributions)
            np.save(neyman_dir + '/neyman_llr_distribution_' + algorithm + '_calibrated_' + str(
                t) + filename_addition + '.npy',
                    llr_neyman_distributions)

        # Roaming
        thetas0_array = np.zeros((n_roaming, 2), dtype=X_roam_transformed.dtype)
        thetas0_array[:, :] = settings.thetas[t]
        X_thetas_roaming_temp = np.hstack((X_roam_transformed, thetas0_array))
        r_roam_temp[t, :], _ = ratio_calibrated.predict(X_thetas_roaming_temp)

    # Save LLR
    expected_llr_calibrated = np.asarray(expected_llr_calibrated)
    np.save(results_dir + '/llr_' + algorithm + '_calibrated' + filename_addition + '.npy',
            expected_llr_calibrated)

    logging.info('Interpolating calibrated roaming')
    gp = GaussianProcessRegressor(normalize_y=True,
                                  kernel=C(1.0) * Matern(1.0, nu=0.5), n_restarts_optimizer=10)
    gp.fit(settings.thetas[:], np.log(r_roam_temp))
    r_roam_calibrated = np.exp(gp.predict(np.c_[xx.ravel(), yy.ravel()])).T
    np.save(results_dir + '/r_roam_' + algorithm + '_calibrated' + filename_addition + '.npy', r_roam_calibrated)
