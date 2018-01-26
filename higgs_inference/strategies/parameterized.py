################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import logging
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern

from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping

from carl.ratios import ClassifierScoreRatio
from carl.learning import CalibratedClassifierScoreCV

from higgs_inference.models.models_parameterized import make_classifier_carl, make_classifier_carl_morphingaware
from higgs_inference.models.models_parameterized import make_classifier_score, make_classifier_score_morphingaware
from higgs_inference.models.models_parameterized import make_classifier_combined, make_classifier_combined_morphingaware
from higgs_inference.models.models_parameterized import make_regressor, make_regressor_morphingaware
from higgs_inference.models.models_parameterized import make_combined_regressor, make_combined_regressor_morphingaware


################################################################################
# What do
################################################################################

def parameterized_inference(algorithm='carl',  # 'carl', 'score', 'combined', 'regression', 'combinedregression'
                            morphing_aware=False,
                            training_sample='baseline',  # 'baseline', 'basis', 'random'
                            options=''):  # all other options in a string

    """
    Trains and evaluates one of the parameterized higgs_inference methods.

    :param algorithm: Type of the algorithm used. Currently supported: 'carl', 'score', 'combined', 'regression', and
    'combinedregression'.
    :param morphing_aware: bool that decides whether a morphing-aware or morphing-agnostic architecture is used.
    :param training_sample: Training sample. Can be 'baseline', 'basis', or 'random'.
    :param options: Further options in a list of strings or string.
    """

    logging.info('Starting parameterized inference ')

    assert algorithm in ['carl', 'score', 'combined', 'regression', 'combinedregression']
    assert training_sample in ['baseline', 'basis', 'random']

    random_theta_mode = training_sample == 'random'
    basis_theta_mode = training_sample == 'basis'

    learn_logr_mode = ('learnlogr' in options)
    denom1_mode = ('denom1' in options)
    short_mode = ('short' in options)
    long_mode = ('long' in options)
    small_alpha_mode = ('smallalpha' in options)
    deep_mode = ('deep' in options)
    shallow_mode = ('shallow' in options)
    debug_mode = ('debug' in options)

    filename_addition = ''
    if morphing_aware:
        filename_addition = '_aware'

    if random_theta_mode:
        filename_addition += '_random'
    elif basis_theta_mode:
        filename_addition += '_basis'

    if learn_logr_mode:
        filename_addition += '_learnlogr'

    alpha_regression = 0.011
    alpha_carl = 0.35
    if small_alpha_mode:
        alpha_regression = 0.005
        alpha_carl = 0.1
        filename_addition += '_smallalpha'

    n_hidden_layers = 2
    n_hidden_layers_aware = 2
    if shallow_mode:
        n_hidden_layers = 1
        n_hidden_layers_aware = 1
        filename_addition += '_shallow'
    elif deep_mode:
        n_hidden_layers = 3
        n_hidden_layers_aware = 3
        filename_addition += '_deep'

    n_epochs = 20
    early_stopping = True
    if debug_mode:
        n_epochs = 1
        early_stopping = False
        filename_addition += '_debug'
    elif long_mode:
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
    neyman_dir = results_dir + '/neyman'

    logging.info('Main settings:')
    logging.info('  Algorithm:                %s', algorithm)
    logging.info('  Morphing-aware:           %s', morphing_aware)
    logging.info('  Training sample:          %s', training_sample)
    logging.info('Options:')
    logging.info('  Number of epochs:         %s', n_epochs)
    logging.info('  Number of hidden layers:  %s', n_hidden_layers)
    logging.info('  alpha carl:               %s', alpha_carl)
    logging.info('  alpha regression:         %s', alpha_regression)
    logging.info('  Debug mode:               %s', debug_mode)

    ################################################################################
    # Data
    ################################################################################

    thetas = np.load(data_dir + '/thetas/thetas_parameterized.npy')

    n_thetas = len(thetas)
    theta_benchmark_trained = 422
    theta_benchmark_nottrained = 9

    if random_theta_mode:
        X_train = np.load(unweighted_events_dir + '/X_train_random' + input_filename_addition + '.npy')
        y_train = np.load(unweighted_events_dir + '/y_train_random' + input_filename_addition + '.npy')
        scores_train = np.load(unweighted_events_dir + '/scores_train_random' + input_filename_addition + '.npy')
        r_train = np.load(unweighted_events_dir + '/r_train_random' + input_filename_addition + '.npy')
        theta0_train = np.load(unweighted_events_dir + '/theta0_train_random' + input_filename_addition + '.npy')
    elif basis_theta_mode:
        X_train = np.load(unweighted_events_dir + '/X_train_basis' + input_filename_addition + '.npy')
        y_train = np.load(unweighted_events_dir + '/y_train_basis' + input_filename_addition + '.npy')
        scores_train = np.load(unweighted_events_dir + '/scores_train_basis' + input_filename_addition + '.npy')
        r_train = np.load(unweighted_events_dir + '/r_train_basis' + input_filename_addition + '.npy')
        theta0_train = np.load(unweighted_events_dir + '/theta0_train_basis' + input_filename_addition + '.npy')
    else:
        X_train = np.load(unweighted_events_dir + '/X_train' + input_filename_addition + '.npy')
        y_train = np.load(unweighted_events_dir + '/y_train' + input_filename_addition + '.npy')
        scores_train = np.load(unweighted_events_dir + '/scores_train' + input_filename_addition + '.npy')
        r_train = np.load(unweighted_events_dir + '/r_train' + input_filename_addition + '.npy')
        theta0_train = np.load(unweighted_events_dir + '/theta0_train' + input_filename_addition + '.npy')

    X_calibration = np.load(unweighted_events_dir + '/X_calibration' + input_filename_addition + '.npy')
    weights_calibration = np.load(unweighted_events_dir + '/weights_calibration' + input_filename_addition + '.npy')

    X_test = np.load(unweighted_events_dir + '/X_test' + input_filename_addition + '.npy')
    # scores_test = np.load(unweighted_events_dir + '/scores_test' + input_filename_addition + '.npy')
    r_test = np.load(unweighted_events_dir + '/r_test' + input_filename_addition + '.npy')

    X_roam = np.load(unweighted_events_dir + '/X_roam' + input_filename_addition + '.npy')
    # r_roam = np.load(unweighted_events_dir + '/r_roam' + input_filename_addition + '.npy')
    n_roaming = len(X_roam)

    X_neyman_observed = np.load(unweighted_events_dir + '/X_neyman_observed.npy')

    n_expected_events = 36
    n_events_test = X_test.shape[0]
    assert n_thetas == r_test.shape[0]

    scaler = StandardScaler()
    scaler.fit(np.array(X_train, dtype=np.float64))
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    X_roam_transformed = scaler.transform(X_roam)
    X_calibration_transformed = scaler.transform(X_calibration)
    X_neyman_observed_transformed = scaler.transform(
        X_neyman_observed.reshape((-1,X_neyman_observed.shape[2])))

    log_r_score_train = np.hstack(((np.log(np.clip(r_train, 1.e-3, 1.e3))).reshape(-1, 1), scores_train))
    X_thetas_train = np.hstack((X_train_transformed, theta0_train))
    y_score_train = np.hstack((y_train.reshape(-1, 1), scores_train))

    n_thetas_roam = 101
    xi = np.linspace(-1.0, 1.0, n_thetas_roam)
    yi = np.linspace(-1.0, 1.0, n_thetas_roam)
    xx, yy = np.meshgrid(xi, yi)
    thetas_roam = np.asarray((xx.flatten(), yy.flatten())).T
    X_thetas_roam = []
    for i in range(n_roaming):
        X_thetas_roam.append(np.zeros((n_thetas_roam ** 2, X_roam_transformed.shape[1] + 2)))
        X_thetas_roam[-1][:, :-2] = X_roam_transformed[i, :]
        X_thetas_roam[-1][:, -2:] = thetas_roam

    if debug_mode:
        X_thetas_train = X_thetas_train[::100]
        y_score_train = y_score_train[::100]
        log_r_score_train = log_r_score_train[::100]
        X_test_transformed = X_test[::100]
        X_calibration_transformed = X_calibration_transformed[::100]
        weights_calibration = weights_calibration[:, ::100]
        n_events_test = len(X_test_transformed)

    ################################################################################
    # Regression approaches
    ################################################################################

    if algorithm in ['regression', 'combinedregression']:

        if algorithm == 'regression':
            if morphing_aware:
                regr = KerasRegressor(lambda: make_regressor_morphingaware(n_hidden_layers=n_hidden_layers_aware),
                                      epochs=n_epochs, validation_split=0.1,
                                      verbose=2)
            else:
                regr = KerasRegressor(lambda: make_regressor(n_hidden_layers=n_hidden_layers),
                                      epochs=n_epochs, validation_split=0.1,
                                      verbose=2)
        else:
            if morphing_aware:
                regr = KerasRegressor(
                    lambda: make_combined_regressor_morphingaware(n_hidden_layers=n_hidden_layers_aware,
                                                                  alpha=alpha_regression),
                    epochs=n_epochs, validation_split=0.1,
                    verbose=2)
            else:
                regr = KerasRegressor(lambda: make_combined_regressor(n_hidden_layers=n_hidden_layers,
                                                                      alpha=alpha_regression),
                                      epochs=n_epochs, validation_split=0.1,
                                      verbose=2)

        logging.info('Starting training')
        regr.fit(X_thetas_train, log_r_score_train,
                 callbacks=([EarlyStopping(verbose=1, patience=3)] if early_stopping else None))

        logging.info('Starting evaluation')
        expected_llr = []

        for t, theta in enumerate(thetas):

            # Prepare test data
            thetas0_array = np.zeros((X_test_transformed.shape[0], 2), dtype=X_test_transformed.dtype)
            thetas0_array[:, :] = thetas[t]
            X_thetas_test = np.hstack((X_test_transformed, thetas0_array))

            # Evaluation
            prediction = regr.predict(X_thetas_test)
            this_r = np.exp(prediction[:, 0])
            this_score = prediction[:, 1:3]
            if morphing_aware:
                this_wi = prediction[:, 3:18]
                this_ri = prediction[:, 18:]
                logging.debug('Morphing weights for theta %s (%s): $s', t, theta, this_wi[0])

            expected_llr.append(- 2. * n_expected_events / n_events_test * np.sum(np.log(this_r)))

            # For benchmark thetas, save r for different phase-space points
            if t == theta_benchmark_nottrained:
                np.save(results_dir + '/r_nottrained_' + algorithm + filename_addition + '.npy', this_r)
                np.save(results_dir + '/scores_nottrained_' + algorithm + filename_addition + '.npy', this_score)
                if morphing_aware:
                    np.save(results_dir + '/morphing_ri_nottrained_' + algorithm + filename_addition + '.npy', this_ri)
                    np.save(results_dir + '/morphing_wi_nottrained_' + algorithm + filename_addition + '.npy', this_wi)
            elif t == theta_benchmark_trained:
                np.save(results_dir + '/r_trained_' + algorithm + filename_addition + '.npy', this_r)
                np.save(results_dir + '/scores_trained_' + algorithm + filename_addition + '.npy', this_score)
                if morphing_aware:
                    np.save(results_dir + '/morphing_ri_trained_' + algorithm + filename_addition + '.npy', this_ri)
                    np.save(results_dir + '/morphing_wi_trained_' + algorithm + filename_addition + '.npy', this_wi)

            # Prepare observed data for Neyman construction
            thetas0_array = np.zeros((X_neyman_observed_transformed.shape[0], 2), dtype=X_neyman_observed_transformed.dtype)
            thetas0_array[:, :] = thetas[t]
            X_thetas_neyman_observed = np.hstack((X_neyman_observed_transformed, thetas0_array))

            # Neyman construction: evaluate observed sample (raw)
            log_r_neyman_observed = regr.predict(X_thetas_neyman_observed)[:, 0]
            llr_neyman_observed = -2. * np.sum(log_r_neyman_observed.reshape((-1, 36)), axis=1)
            np.save(neyman_dir + '/neyman_llr_observed_' + algorithm + '_' + str(t) + '.npy',
                    llr_neyman_observed)

            # Neyman construction: load distribution sample
            X_neyman_distribution = np.load(unweighted_events_dir + '/X_neyman_distribution_' + str(t) + '.npy')
            X_neyman_distribution_transformed = scaler.transform(
                X_neyman_distribution.reshape((-1, X_neyman_distribution.shape[2])))

            # Prepare distribution data for Neyman construction
            thetas0_array = np.zeros((X_neyman_distribution_transformed.shape[0], 2),
                                     dtype=X_neyman_distribution_transformed.dtype)
            thetas0_array[:, :] = thetas[t]
            X_thetas_neyman_distribution = np.hstack((X_neyman_distribution_transformed, thetas0_array))

            # Neyman construction: evaluate distribution sample (raw)
            log_r_neyman_distribution = regr.predict(X_thetas_neyman_distribution)
            llr_neyman_distribution = -2. * np.sum(log_r_neyman_distribution.reshape((-1, 36)), axis=1)
            np.save(neyman_dir + '/neyman_llr_distribution_' + algorithm + '_' + str(t) + '.npy',
                    llr_neyman_distribution)

        # Save LLR
        expected_llr = np.asarray(expected_llr)
        np.save(results_dir + '/llr_' + algorithm + filename_addition + '.npy', expected_llr)

        logging.info('Starting roaming')
        r_roam = []
        for i in range(n_roaming):
            prediction = regr.predict(X_thetas_roam[i])
            r_roam.append(np.exp(prediction[:, 0]))
        r_roam = np.asarray(r_roam)
        np.save(results_dir + '/r_roam_' + algorithm + filename_addition + '.npy', r_roam)

    ################################################################################
    # Carl approaches
    ################################################################################

    else:

        logging.info('Starting training')
        if algorithm == 'carl':
            if morphing_aware:
                clf = KerasRegressor(lambda: make_classifier_carl_morphingaware(n_hidden_layers=n_hidden_layers_aware,
                                                                                learn_log_r=learn_logr_mode),
                                     epochs=n_epochs, validation_split=0.1,
                                     verbose=2)
            else:
                clf = KerasRegressor(lambda: make_classifier_carl(n_hidden_layers=n_hidden_layers,
                                                                  learn_log_r=learn_logr_mode),
                                     epochs=n_epochs, validation_split=0.1,
                                     verbose=2)

        elif algorithm == 'score':
            if morphing_aware:
                clf = KerasRegressor(lambda: make_classifier_score_morphingaware(n_hidden_layers=n_hidden_layers_aware,
                                                                                 learn_log_r=learn_logr_mode),
                                     epochs=n_epochs, validation_split=0.1,
                                     verbose=2)
            else:
                clf = KerasRegressor(lambda: make_classifier_score(n_hidden_layers=n_hidden_layers,
                                                                   learn_log_r=learn_logr_mode),
                                     epochs=n_epochs, validation_split=0.1,
                                     verbose=2)

        elif algorithm == 'combined':
            if morphing_aware:
                clf = KerasRegressor(
                    lambda: make_classifier_combined_morphingaware(n_hidden_layers=n_hidden_layers_aware,
                                                                   learn_log_r=learn_logr_mode,
                                                                   alpha=alpha_carl),
                    epochs=n_epochs, validation_split=0.1,
                    verbose=2)
            else:
                clf = KerasRegressor(lambda: make_classifier_combined(n_hidden_layers=n_hidden_layers,
                                                                      learn_log_r=learn_logr_mode,
                                                                      alpha=alpha_carl),
                                     epochs=n_epochs, validation_split=0.1,
                                     verbose=2)

        else:
            raise ValueError()

        # Fit
        clf.fit(X_thetas_train[::], y_score_train[::],
                callbacks=([EarlyStopping(verbose=1, patience=3)] if early_stopping else None))

        # carl ratio object
        ratio = ClassifierScoreRatio(clf, prefit=True)

        logging.info('Starting evaluation')
        expected_llr = []

        for t, theta in enumerate(thetas):

            # Prepare test data
            thetas0_array = np.zeros((X_test_transformed.shape[0], 2), dtype=X_test_transformed.dtype)
            thetas0_array[:, :] = thetas[t]
            X_thetas_test = np.hstack((X_test_transformed, thetas0_array))

            # Evaluation
            this_r, this_other = ratio.predict(X_thetas_test)
            this_score = this_other[:, :2]
            if morphing_aware:
                this_wi = this_other[:, 2:17]
                this_ri = this_other[:, 17:]
                logging.debug('Morphing weights for theta %s (%s): $s', t, theta, this_wi[0])

            expected_llr.append(- 2. * n_expected_events / n_events_test * np.sum(np.log(this_r)))

            # For benchmark thetas, save more info
            if t == theta_benchmark_nottrained:
                np.save(results_dir + '/r_nottrained_' + algorithm + filename_addition + '.npy', this_r)
                np.save(results_dir + '/scores_nottrained_' + algorithm + filename_addition + '.npy', this_score)
                if morphing_aware:
                    np.save(results_dir + '/morphing_ri_nottrained_' + algorithm + filename_addition + '.npy', this_ri)
                    np.save(results_dir + '/morphing_wi_nottrained_' + algorithm + filename_addition + '.npy', this_wi)
            elif t == theta_benchmark_trained:
                np.save(results_dir + '/r_trained_' + algorithm + filename_addition + '.npy', this_r)
                np.save(results_dir + '/scores_trained_' + algorithm + filename_addition + '.npy', this_score)
                if morphing_aware:
                    np.save(results_dir + '/morphing_ri_trained_' + algorithm + filename_addition + '.npy', this_ri)
                    np.save(results_dir + '/morphing_wi_trained_' + algorithm + filename_addition + '.npy', this_wi)

            # Prepare observed data for Neyman construction
            X_thetas_neyman_observed = X_neyman_observed_transformed.reshape(
                (-1, X_neyman_observed_transformed.shape[2]))
            thetas0_array = np.zeros((X_thetas_neyman_observed.shape[0], 2), dtype=X_thetas_neyman_observed.dtype)
            thetas0_array[:, :] = thetas[t]
            X_thetas_neyman_observed = np.hstack((X_thetas_neyman_observed, thetas0_array))

            # Neyman construction: evaluate observed sample (raw)
            r_neyman_observed, _ = ratio.predict(X_thetas_neyman_observed)
            llr_neyman_observed = -2. * np.sum(np.log(r_neyman_observed).reshape((-1, n_expected_events)), axis=1)
            np.save(neyman_dir + '/neyman_llr_observed_' + algorithm + '_' + str(t) + '.npy',
                    llr_neyman_observed)

            # Neyman construction: load distribution sample
            X_neyman_distribution = np.load(unweighted_events_dir + '/X_neyman_distribution_' + str(t) + '.npy')
            X_neyman_distribution_transformed = scaler.transform(
                X_neyman_distribution.reshape((-1,X_neyman_distribution.shape[2])))

            # Prepare distribution data for Neyman construction
            thetas0_array = np.zeros((X_neyman_distribution_transformed.shape[0], 2),
                                     dtype=X_neyman_distribution_transformed.dtype)
            thetas0_array[:, :] = thetas[t]
            X_thetas_neyman_distribution = np.hstack((X_neyman_distribution_transformed, thetas0_array))

            # Neyman construction: evaluate distribution sample (raw)
            r_neyman_distribution, _ = ratio.predict(X_thetas_neyman_distribution)
            llr_neyman_distribution = -2. * np.sum(np.log(r_neyman_distribution).reshape((-1, n_expected_events)), axis=1)
            np.save(neyman_dir + '/neyman_llr_distribution_' + algorithm + '_' + str(t) + '.npy',
                    llr_neyman_distribution)

        expected_llr = np.asarray(expected_llr)
        np.save(results_dir + '/llr_' + algorithm + filename_addition + '.npy', expected_llr)

        logging.info('Starting roaming')
        r_roam = []
        for i in range(n_roaming):
            prediction, _ = ratio.predict(X_thetas_roam[i])
            r_roam.append(prediction)
        r_roam = np.asarray(r_roam)
        np.save(results_dir + '/r_roam_' + algorithm + filename_addition + '.npy', r_roam)

        logging.info('Starting calibrated evaluation and roaming')
        expected_llr_calibrated = []
        r_roam_temp = np.zeros((n_thetas, n_roaming))

        for t, theta in enumerate(thetas):

            # Prepare data for calibration
            n_calibration_each = X_calibration_transformed.shape[0]
            thetas0_array = np.zeros((n_calibration_each, 2), dtype=X_calibration_transformed.dtype)
            thetas0_array[:, :] = thetas[t]
            X_thetas_calibration = np.hstack((X_calibration_transformed, thetas0_array))
            X_thetas_calibration = np.vstack((X_thetas_calibration, X_thetas_calibration))
            y_calibration = np.zeros(2 * n_calibration_each)
            y_calibration[n_calibration_each:] = 1.
            w_calibration = np.zeros(2 * n_calibration_each)
            w_calibration[:n_calibration_each] = weights_calibration[t]
            w_calibration[n_calibration_each:] = weights_calibration[theta1]

            # Calibration
            ratio_calibrated = ClassifierScoreRatio(
                CalibratedClassifierScoreCV(clf, cv='prefit', method='isotonic')
                # CalibratedClassifierScoreCV(clf, cv='prefit', bins=100, variable_width=False, independent_binning=False)
            )
            ratio_calibrated.fit(X_thetas_calibration, y_calibration, sample_weight=w_calibration)

            # Prepare data
            thetas0_array = np.zeros((X_test_transformed.shape[0], 2), dtype=X_test_transformed.dtype)
            thetas0_array[:, :] = thetas[t]
            X_thetas_test = np.hstack((X_test_transformed, thetas0_array))

            this_r, this_other = ratio_calibrated.predict(X_thetas_test)
            this_score = this_other[:, :2]

            expected_llr_calibrated.append(- 2. * n_expected_events / n_events_test * np.sum(np.log(this_r)))

            # For benchmark theta, save more data
            if t == theta_benchmark_nottrained:
                np.save(results_dir + '/scores_nottrained_' + algorithm + '_calibrated' + filename_addition + '.npy',
                        this_score)
                np.save(results_dir + '/r_nottrained_' + algorithm + '_calibrated' + filename_addition + '.npy', this_r)
                np.save(results_dir + '/calvalues_nottrained_' + algorithm + filename_addition + '.npy',
                        ratio_calibrated.classifier_.calibration_sample[:n_calibration_each])
                # np.save(results_dir + '/cal0histo_nottrained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator0.histogram_)
                # np.save(results_dir + '/cal0edges_nottrained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator0.edges_[0])
                # np.save(results_dir + '/cal1histo_nottrained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator1.histogram_)
                # np.save(results_dir + '/cal1edges_nottrained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator1.edges_[0])

            elif t == theta_benchmark_trained:
                np.save(results_dir + '/scores_trained_' + algorithm + '_calibrated' + filename_addition + '.npy',
                        this_score)
                np.save(results_dir + '/r_trained_' + algorithm + '_calibrated' + filename_addition + '.npy', this_r)
                np.save(results_dir + '/calvalues_trained_' + algorithm + filename_addition + '.npy',
                        ratio_calibrated.classifier_.calibration_sample[:n_calibration_each])
                # np.save(results_dir + '/cal0histo_trained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator0.histogram_)
                # np.save(results_dir + '/cal0edges_trained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator0.edges_[0])
                # np.save(results_dir + '/cal1histo_trained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator1.histogram_)
                # np.save(results_dir + '/cal1edges_trained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator1.edges_[0])

            # Prepare observed data for Neyman construction
            thetas0_array = np.zeros((X_neyman_observed_transformed.shape[0], 2), dtype=X_neyman_observed_transformed.dtype)
            thetas0_array[:, :] = thetas[t]
            X_thetas_neyman_observed = np.hstack((X_neyman_observed_transformed, thetas0_array))

            # Neyman construction: evaluate observed sample (raw)
            r_neyman_observed, _ = ratio_calibrated.predict(X_thetas_neyman_observed)
            llr_neyman_observed = -2. * np.sum(np.log(r_neyman_observed).reshape((-1, n_expected_events)), axis=1)
            np.save(neyman_dir + '/neyman_llr_observed_' + algorithm + '_calibrated_' + str(t) + '.npy',
                    llr_neyman_observed)

            # Neyman construction: load distribution sample
            X_neyman_distribution = np.load(unweighted_events_dir + '/X_neyman_distribution_' + str(t) + '.npy')
            X_neyman_distribution_transformed = scaler.transform(
                X_neyman_distribution.reshape((-1,X_neyman_distribution.shape[2])))

            # Prepare distribution data for Neyman construction
            thetas0_array = np.zeros((X_neyman_distribution_transformed.shape[0], 2),
                                     dtype=X_neyman_distribution_transformed.dtype)
            thetas0_array[:, :] = thetas[t]
            X_thetas_neyman_distribution = np.hstack((X_neyman_distribution_transformed, thetas0_array))

            # Neyman construction: evaluate distribution sample (raw)
            r_neyman_distribution, _ = ratio_calibrated.predict(X_thetas_neyman_distribution)
            llr_neyman_distribution = -2. * np.sum(np.log(r_neyman_distribution).reshape((-1, n_expected_events)), axis=1)
            np.save(neyman_dir + '/neyman_llr_distribution_' + algorithm + '_calibrated_' + str(t) + '.npy',
                    llr_neyman_distribution)

            # Roaming
            thetas0_array = np.zeros((n_roaming, 2), dtype=X_roam_transformed.dtype)
            thetas0_array[:, :] = thetas[t]
            X_thetas_roaming_temp = np.hstack((X_roam_transformed, thetas0_array))
            r_roam_temp[t, :], _ = ratio_calibrated.predict(X_thetas_roaming_temp)

        # Save LLR
        expected_llr_calibrated = np.asarray(expected_llr_calibrated)
        np.save(results_dir + '/llr_' + algorithm + '_calibrated' + filename_addition + '.npy',
                expected_llr_calibrated)

        logging.info('Interpolating calibrated roaming')
        gp = GaussianProcessRegressor(normalize_y=True,
                                      kernel=C(1.0) * Matern(1.0, nu=0.5), n_restarts_optimizer=10)
        gp.fit(thetas[:], np.log(r_roam_temp))
        r_roam_calibrated = np.exp(gp.predict(np.c_[xx.ravel(), yy.ravel()])).T
        np.save(results_dir + '/r_roam_' + algorithm + '_calibrated' + filename_addition + '.npy', r_roam_calibrated)
