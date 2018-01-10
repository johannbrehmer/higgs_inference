################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern

from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping

from carl.ratios import ClassifierScoreRatio
from carl.learning import CalibratedClassifierScoreCV

from models_parameterized import make_classifier_carl, make_classifier_carl_morphingaware
from models_parameterized import make_classifier_score, make_classifier_score_morphingaware
from models_parameterized import make_classifier_combined, make_classifier_combined_morphingaware
from models_parameterized import make_regressor, make_regressor_morphingaware
from models_parameterized import make_combined_regressor, make_combined_regressor_morphingaware



################################################################################
# What do
################################################################################

def parameterized_inference(algorithm='carl', #'carl', 'score', 'combined', 'regression', 'combinedregression'
                            morphing_aware=False,
                            training_sample='baseline', #'baseline', 'basis', 'random'
                            options=''): # all other options in a string

    """
    Trains and evaluates one of the parameterized inference methods.

    :param algorithm: Type of the algorithm used. Currently supported: 'carl', 'score', 'combined', 'regression', and
    'combinedregression'.
    :param morphing_aware: bool that decides whether a morphing-aware or morphing-agnostic architecture is used.
    :param training_sample: Training sample. Can be 'baseline', 'basis', or 'random'.
    :param options: Further options in a list of strings or string.
    """
    
    assert algorithm in ['carl', 'score', 'combined', 'regression', 'combinedregression']
    assert training_sample in ['baseline', 'basis', 'random']

    random_theta_mode = training_sample == 'random'
    basis_theta_mode = training_sample == 'basis'

    denom1_mode = ('denom1' in options)
    
    short_mode = ('short' in options)
    long_mode = ('long' in options)
    large_alpha_mode = ('largealpha' in options)
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
    
    alpha_regression = 0.005
    alpha_carl = 0.1
    if large_alpha_mode:
        alpha_regression = 0.008
        alpha_carl = 0.2
        filename_addition += '_largealpha'
    
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
    results_dir = '../results/parameterized'
    
    print('')
    print('Main settings:')
    print('  Algorithm:                ', algorithm)
    print('  Morphing-aware:           ', morphing_aware)
    print('  Training sample:          ', training_sample)
    print('')
    print('Options:')
    print('  Number of epochs:         ', n_epochs)
    print('  Number of hidden layers:  ', n_hidden_layers)
    print('  alpha carl:               ', alpha_carl)
    print('  alpha regression:         ', alpha_regression)
    print('  Debug mode:               ', debug_mode)
    
    
    
    ################################################################################
    # Data
    ################################################################################
    
    thetas = np.load(data_dir + '/thetas/thetas_parameterized.npy')
    
    n_thetas = len(thetas)
    theta_trained = 422
    theta_nottrained = 9
    
    if random_theta_mode:
        X_train = np.load(data_dir + '/unweighted_events/X_train_random' + input_filename_addition + '.npy')
        y_train = np.load(data_dir + '/unweighted_events/y_train_random' + input_filename_addition + '.npy')
        scores_train = np.load(data_dir + '/unweighted_events/scores_train_random' + input_filename_addition + '.npy')
        r_train = np.load(data_dir + '/unweighted_events/r_train_random' + input_filename_addition + '.npy')
        theta0_train = np.load(data_dir + '/unweighted_events/theta0_train_random' + input_filename_addition + '.npy')
    elif basis_theta_mode:
        X_train = np.load(data_dir + '/unweighted_events/X_train_basis' + input_filename_addition + '.npy')
        y_train = np.load(data_dir + '/unweighted_events/y_train_basis' + input_filename_addition + '.npy')
        scores_train = np.load(data_dir + '/unweighted_events/scores_train_basis' + input_filename_addition + '.npy')
        r_train = np.load(data_dir + '/unweighted_events/r_train_basis' + input_filename_addition + '.npy')
        theta0_train = np.load(data_dir + '/unweighted_events/theta0_train_basis' + input_filename_addition + '.npy')
    else:
        X_train = np.load(data_dir + '/unweighted_events/X_train' + input_filename_addition + '.npy')
        y_train = np.load(data_dir + '/unweighted_events/y_train' + input_filename_addition + '.npy')
        scores_train = np.load(data_dir + '/unweighted_events/scores_train' + input_filename_addition + '.npy')
        r_train = np.load(data_dir + '/unweighted_events/r_train' + input_filename_addition + '.npy')
        theta0_train = np.load(data_dir + '/unweighted_events/theta0_train' + input_filename_addition + '.npy')
    
    X_calibration = np.load(data_dir + '/unweighted_events/X_calibration' + input_filename_addition + '.npy')
    weights_calibration = np.load(data_dir + '/unweighted_events/weights_calibration' + input_filename_addition + '.npy')
    
    X_test = np.load(data_dir + '/unweighted_events/X_test' + input_filename_addition + '.npy')
    #scores_test = np.load(data_dir + '/unweighted_events/scores_test' + input_filename_addition + '.npy')
    r_test = np.load(data_dir + '/unweighted_events/r_test' + input_filename_addition + '.npy')
    
    X_roam = np.load(data_dir + '/unweighted_events/X_roam' + input_filename_addition + '.npy')
    #r_roam = np.load(data_dir + '/unweighted_events/r_roam' + input_filename_addition + '.npy')
    n_roaming = len(X_roam)
    
    n_observed = X_test.shape[0]
    assert n_thetas == r_test.shape[0]
    n_pseudoexperiments_series = 5
    n_pseudoexperiments_events = [10,30,100,300,1000]
    n_pseudoexperiments_repetitions = 1000
    
    scaler = StandardScaler()
    scaler.fit(np.array(X_train, dtype=np.float64))
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    X_roam_transformed = scaler.transform(X_roam)
    X_calibration_transformed = scaler.transform(X_calibration)
    
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
        weights_calibration = weights_calibration[:,::100]
        n_observed = len(X_test_transformed)
        n_pseudoexperiments_repetitions = 10
    
    
    ################################################################################
    # Regression approaches
    ################################################################################
    
    if algorithm=='regression':

        if algorithm=='regression':
            if morphing_aware:
                regr = KerasRegressor(lambda: make_regressor_morphingaware(n_hidden_layers=n_hidden_layers_aware),
                                      epochs=n_epochs, validation_split=0.142857,
                                      verbose=2)
            else:
                regr = KerasRegressor(lambda: make_regressor(n_hidden_layers=n_hidden_layers),
                                      epochs=n_epochs, validation_split=0.142857,
                                      verbose=2)
        else:
            if morphing_aware:
                regr = KerasRegressor(lambda: make_combined_regressor_morphingaware(n_hidden_layers=n_hidden_layers_aware,
                                                                                    alpha=alpha_regression),
                                      epochs=n_epochs, validation_split=0.142857,
                                      verbose=2)
            else:
                regr = KerasRegressor(lambda: make_combined_regressor(n_hidden_layers=n_hidden_layers,
                                                                      alpha=alpha_regression),
                                      epochs=n_epochs, validation_split=0.142857,
                                      verbose=2)

        print('')
        print('Training:')
        regr.fit(X_thetas_train, log_r_score_train,
                 callbacks=([EarlyStopping(verbose=1,patience=3)] if early_stopping else None))
    
        print('')
        print('Evaluation:')
        llr = []
        for t, theta in enumerate(thetas):
            thetas0_array = np.zeros((X_test_transformed.shape[0], 2), dtype=X_test_transformed.dtype)
            thetas0_array[:, :] = thetas[t]
            X_thetas_test = np.hstack((X_test_transformed, thetas0_array))
    
            prediction = regr.predict(X_thetas_test)
            this_r = np.exp(prediction[:, 0])
            this_score = prediction[:, 1:]
    
            llr.append(- 19.2 / float(n_observed) * np.sum(np.log(this_r)))
            expected_score = 1. / float(n_observed) * np.sum(this_score, axis=0)
    
            if t == theta_nottrained:
                r_nottrained = np.copy(this_r)
                scores_nottrained = np.copy(this_score)
            elif t == theta_trained:
                r_trained = np.copy(this_r)
                scores_trained = np.copy(this_score)
    
        llr = np.asarray(llr)
        np.save(results_dir + '/r_nottrained_' + algorithm + filename_addition + '.npy', r_nottrained)
        np.save(results_dir + '/r_trained_' + algorithm + filename_addition + '.npy', r_trained)
        np.save(results_dir + '/scores_trained_' + algorithm + filename_addition + '.npy', scores_trained)
        np.save(results_dir + '/scores_nottrained_' + algorithm + filename_addition + '.npy', scores_nottrained)
        np.save(results_dir + '/llr_' + algorithm + filename_addition + '.npy', llr)
    
        print('')
        print('Roaming:')
        r_roam = []
        for i in range(n_roaming):
            prediction = regr.predict(X_thetas_roam[i])
            r_roam.append(np.exp(prediction[:, 0]))
        r_roam = np.asarray(r_roam)
        np.save(results_dir + '/r_roam_' + algorithm + filename_addition + '.npy', r_roam)
    
        print('')
        print('Pseudo-experiments:')
        pseudoexperiments = np.zeros((n_thetas, n_pseudoexperiments_series, n_pseudoexperiments_repetitions))
        for i, n in enumerate(n_pseudoexperiments_events):
            for j in range(n_pseudoexperiments_repetitions):
                indices = np.random.choice(list(range(n_observed)), n)
                for t, theta in enumerate(thetas):
                    thetas0_array = np.zeros((n, 2), dtype=X_test_transformed.dtype)
                    thetas0_array[:, :] = thetas[t]
                    X_thetas_test = np.hstack((X_test_transformed[indices], thetas0_array))
                    prediction = regr.predict(X_thetas_test)
                    this_r = np.exp(prediction[:, 0])
                    pseudoexperiments[t, i, j] = (- 19.2 / float(n) * np.sum(np.log(this_r)))
    
        pseudoexperiments_variance = np.zeros((n_thetas, n_pseudoexperiments_series))
        for t in range(n_thetas):
            for i in range(n_pseudoexperiments_series):
                pseudoexperiments_variance[t, i] = np.var(pseudoexperiments[t, i, :])
        np.save(results_dir + '/pseudoexperiments_variance_' + algorithm + filename_addition + '.npy',
                pseudoexperiments_variance)
    


    ################################################################################
    # Carl approaches
    ################################################################################
    
    else:

        print('')
        print('Training:')
        if algorithm=='carl':
            if morphing_aware:
                clf = KerasRegressor(lambda: make_classifier_carl_morphingaware(n_hidden_layers=n_hidden_layers_aware),
                                     epochs=n_epochs, validation_split=0.142857,
                                     verbose=2)
            else:
                clf = KerasRegressor(lambda: make_classifier_carl(n_hidden_layers=n_hidden_layers),
                                     epochs=n_epochs, validation_split=0.142857,
                                     verbose=2)

        elif algorithm=='score':
            if morphing_aware:
                clf = KerasRegressor(lambda: make_classifier_score_morphingaware(n_hidden_layers=n_hidden_layers_aware),
                                     epochs=n_epochs, validation_split=0.142857,
                                     verbose=2)
            else:
                clf = KerasRegressor(lambda: make_classifier_score(n_hidden_layers=n_hidden_layers),
                                     epochs=n_epochs, validation_split=0.142857,
                                     verbose=2)

        elif algorithm=='combined':
            if morphing_aware:
                clf = KerasRegressor(
                    lambda: make_classifier_combined_morphingaware(n_hidden_layers=n_hidden_layers_aware,
                                                                   alpha=alpha_carl),
                    epochs=n_epochs, validation_split=0.142857,
                    verbose=2)
            else:
                clf = KerasRegressor(lambda: make_classifier_combined(n_hidden_layers=n_hidden_layers,
                                                                      alpha=alpha_carl),
                                     epochs=n_epochs, validation_split=0.142857,
                                     verbose=2)

        clf.fit(X_thetas_train[::], y_score_train[::],
                 callbacks=([EarlyStopping(verbose=1,patience=3)] if early_stopping else None))
    
        print('')
        print('Evaluation:')
        ratio = ClassifierScoreRatio(clf, prefit=True)
        llr = []
        for t, theta in enumerate(thetas):
            thetas0_array = np.zeros((X_test_transformed.shape[0], 2), dtype=X_test_transformed.dtype)
            thetas0_array[:, :] = thetas[t]
            X_thetas_test = np.hstack((X_test_transformed, thetas0_array))
            this_r, this_score = ratio.predict(X_thetas_test)
            llr.append(- 19.2 / float(n_observed) * np.sum(np.log(this_r)))
            expected_score = 1. / float(n_observed) * np.sum(this_score, axis=0)
            if t == theta_nottrained:
                r_nottrained = np.copy(this_r)
                scores_nottrained = np.copy(this_score)
            elif t == theta_trained:
                r_trained = np.copy(this_r)
                scores_trained = np.copy(this_score)
        llr = np.asarray(llr)
        np.save(results_dir + '/r_nottrained_' + algorithm + filename_addition + '.npy', r_nottrained)
        np.save(results_dir + '/r_trained_' + algorithm + filename_addition + '.npy', r_trained)
        np.save(results_dir + '/scores_trained_' + algorithm + filename_addition + '.npy', scores_trained)
        np.save(results_dir + '/scores_nottrained_' + algorithm + filename_addition + '.npy', scores_nottrained)
        np.save(results_dir + '/llr_' + algorithm + filename_addition + '.npy', llr)
    
        print('')
        print('Roaming:')
        r_roam = []
        for i in range(n_roaming):
            prediction, _ = ratio.predict(X_thetas_roam[i])
            r_roam.append(prediction)
        r_roam = np.asarray(r_roam)
        np.save(results_dir + '/r_roam_' + algorithm + filename_addition + '.npy', r_roam)
    
        print('')
        print('Pseudo-experiments:')
        # Pseudo-experiments for statistical uncertainty
        pseudoexperiments = np.zeros((n_thetas, n_pseudoexperiments_series, n_pseudoexperiments_repetitions))
        for i, n in enumerate(n_pseudoexperiments_events):
            for j in range(n_pseudoexperiments_repetitions):
                indices = np.random.choice(list(range(n_observed)), n)
                for t, theta in enumerate(thetas):
                    thetas0_array = np.zeros((n, 2), dtype=X_test_transformed.dtype)
                    thetas0_array[:, :] = thetas[t]
                    X_thetas_test = np.hstack((X_test_transformed[indices], thetas0_array))
                    this_r, this_score = ratio.predict(X_thetas_test)
                    pseudoexperiments[t,i,j] = (- 19.2 / float(n) * np.sum(np.log(this_r)))
        pseudoexperiments_variance = np.zeros((n_thetas, n_pseudoexperiments_series))
        for t in range(n_thetas):
            for i in range(n_pseudoexperiments_series):
                pseudoexperiments_variance[t,i] = np.var(pseudoexperiments[t,i,:])
        np.save(results_dir + '/pseudoexperiments_variance_' + algorithm + filename_addition + '.npy',
                pseudoexperiments_variance)
    
        print('')
        print('Calibrated evaluation:')
        llr_calibrated = []
        r_roam_temp = np.zeros((n_thetas, n_roaming))
        pseudoexperiments_calibrated = np.zeros((n_thetas, n_pseudoexperiments_series, n_pseudoexperiments_repetitions))
        # Pick indices for pseudo-experiments
        indices = [
            [np.random.choice(list(range(n_observed)), n) for j in range(n_pseudoexperiments_repetitions) ]
            for i, n in enumerate(n_pseudoexperiments_events)
        ]
    
        for t, theta in enumerate(thetas):
            # Calibration
            nc = X_calibration_transformed.shape[0]
            thetas0_array = np.zeros((nc, 2), dtype=X_calibration_transformed.dtype)
            thetas0_array[:, :] = thetas[t]
            X_thetas_calibration = np.hstack((X_calibration_transformed, thetas0_array))
            X_thetas_calibration = np.vstack((X_thetas_calibration, X_thetas_calibration))
            y_calibration = np.zeros(2 * nc)
            y_calibration[nc:] = 1.
            w_calibration = np.zeros(2 * nc)
            w_calibration[:nc] = weights_calibration[t]
            w_calibration[nc:] = weights_calibration[theta1]
    
            ratio_calibrated = ClassifierScoreRatio(
                CalibratedClassifierScoreCV(clf, cv='prefit', bins=50, independent_binning=False)
            )
            ratio_calibrated.fit(X_thetas_calibration, y_calibration, sample_weight=w_calibration)
    
            # Evaluation
            thetas0_array = np.zeros((X_test_transformed.shape[0], 2), dtype=X_test_transformed.dtype)
            thetas0_array[:, :] = thetas[t]
            X_thetas_test = np.hstack((X_test_transformed, thetas0_array))
    
            this_r, this_score = ratio_calibrated.predict(X_thetas_test)
            llr_calibrated.append(- 19.2 / float(n_observed) * np.sum(np.log(this_r)))
    
            if t == theta_nottrained:
                r_nottrained_calibrated = np.copy(this_r)
                scores_nottrained_calibrated = np.copy(this_score)

                np.save(results_dir + '/cal0histo_nottrained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator0.histogram_)
                np.save(results_dir + '/cal0edges_nottrained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator0.edges_[0])
                np.save(results_dir + '/cal1histo_nottrained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator1.histogram_)
                np.save(results_dir + '/cal1edges_nottrained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator1.edges_[0])

            elif t == theta_trained:
                r_trained_calibrated = np.copy(this_r)
                scores_trained_calibrated = np.copy(this_score)

                np.save(results_dir + '/cal0histo_trained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator0.histogram_)
                np.save(results_dir + '/cal0edges_trained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator0.edges_[0])
                np.save(results_dir + '/cal1histo_trained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator1.histogram_)
                np.save(results_dir + '/cal1edges_trained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator1.edges_[0])
    
            # Roaming
            thetas0_array = np.zeros((n_roaming, 2), dtype=X_roam_transformed.dtype)
            thetas0_array[:, :] = thetas[t]
            X_thetas_roaming_temp = np.hstack((X_roam_transformed, thetas0_array))
            r_roam_temp[t,:], _ = ratio_calibrated.predict(X_thetas_roaming_temp)
    
            # Pseudo-experiments
            for i, n in enumerate(n_pseudoexperiments_events):
                thetas0_array = np.zeros((n, 2), dtype=X_test_transformed.dtype)
                thetas0_array[:, :] = thetas[t]
                for j in range(n_pseudoexperiments_repetitions):
                    X_thetas_pseudoexp_test = np.hstack((X_test_transformed[indices[i][j]], thetas0_array))
                    this_r, _ = ratio.predict(X_thetas_pseudoexp_test)
                    pseudoexperiments_calibrated[t, i, j] = (- 19.2 / float(n) * np.sum(np.log(this_r)))
    
        llr_calibrated = np.asarray(llr_calibrated)
        np.save(results_dir + '/r_nottrained_' + algorithm + '_calibrated' + filename_addition + '.npy', r_nottrained_calibrated)
        np.save(results_dir + '/r_trained_' + algorithm + '_calibrated' + filename_addition + '.npy', r_trained_calibrated)
        np.save(results_dir + '/scores_trained_' + algorithm + '_calibrated' + filename_addition + '.npy', scores_trained_calibrated)
        np.save(results_dir + '/scores_nottrained_' + algorithm + '_calibrated' + filename_addition + '.npy', scores_nottrained_calibrated)
        np.save(results_dir + '/llr_' + algorithm + '_calibrated' + filename_addition + '.npy', llr_calibrated)
    
        print('')
        print('Calibrated pseudo-experiments:')
        pseudoexperiments_variance_calibrated = np.zeros((n_thetas, n_pseudoexperiments_series))
        for t in range(n_thetas):
            for i in range(n_pseudoexperiments_series):
                pseudoexperiments_variance_calibrated[t,i] = np.var(pseudoexperiments_calibrated[t,i,:])
        np.save(results_dir + '/pseudoexperiments_variance_' + algorithm + '_calibrated' + filename_addition + '.npy',
                pseudoexperiments_variance_calibrated)
    
        print('')
        print('Calibrated roaming:')
        gp = GaussianProcessRegressor(normalize_y=True,
                                      kernel=C(1.0) * Matern(1.0, nu=0.5), n_restarts_optimizer=10)
        gp.fit(thetas[:], np.log(r_roam_temp))
        r_roam_calibrated = np.exp(gp.predict(np.c_[xx.ravel(), yy.ravel()])).T
        np.save(results_dir + '/r_roam_' + algorithm + '_calibrated' + filename_addition + '.npy', r_roam_calibrated)
