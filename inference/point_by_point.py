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

from models_point_by_point import make_classifier, make_regressor



################################################################################
# What do
################################################################################

def point_by_point_inference(algorithm='carl',
                             options=''):

    """
    Trains and evaluates one of the point-by-point inference methods.

    :param algorithm: Type of the algorithm used. Currently supported: 'carl' and 'regression'.
    :param options: Further options in a list of strings or string.
    """

    assert algorithm in ['carl', 'regression']

    denom1_mode = ('denom1' in options)

    debug_mode = ('debug' in options)

    short_mode = ('short' in options)
    long_mode = ('long' in options)
    deep_mode = ('deep' in options)
    shallow_mode = ('shallow' in options)

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
    results_dir = '../results/point_by_point'

    print('')
    print('Main settings:')
    print('  Algorithm:                ', algorithm)
    print('')
    print('Options:')
    print('  Number of epochs:         ', n_epochs)
    print('  Number of hidden layers:  ', n_hidden_layers)



    ################################################################################
    # Data
    ################################################################################

    thetas = np.load(data_dir + '/thetas/thetas_parameterized.npy')
    n_thetas = len(thetas)
    theta_benchmark1 = 422
    theta_benchmark2 = 9
    training_thetas = [0, 13, 14, 15, 16, 9, 422, 956, 666, 802, 675, 839, 699, 820, 203, 291, 634, 371, 973, 742, 901, 181, 82, 937, 510, 919, 745, 588, 804, 963, 396, 62, 401, 925, 874, 770, 108, 179, 669, 758, 113, 587, 600, 975, 496, 66, 467, 412, 701, 986, 598, 810, 97, 18, 723, 159, 320, 301, 352, 159, 89, 421, 574, 923, 849, 299, 119, 167, 939, 402, 52, 787, 978, 41, 873, 533, 827, 304, 294, 760, 890, 539, 1000, 291, 740, 276, 679, 167, 125, 429, 149, 430, 720, 123, 908, 256, 777, 809, 269, 851]

    X_calibration = np.load(unweighted_events_dir + '/X_calibration' + input_filename_addition + '.npy')
    weights_calibration = np.load(
        unweighted_events_dir + '/weights_calibration' + input_filename_addition + '.npy')

    X_test = np.load(unweighted_events_dir + '/X_test' + input_filename_addition + '.npy')
    r_test = np.load(unweighted_events_dir + '/r_test' + input_filename_addition + '.npy')

    n_observed = X_test.shape[0]
    assert n_thetas == r_test.shape[0]



    ################################################################################
    # Regression approaches
    ################################################################################

    if algorithm == 'regression':

        llr = []

        # Loop over the 15 thetas
        print('')

        for t in training_thetas:

            print('Theta', t, thetas[t])

            # Load data
            X_train = np.load(unweighted_events_dir + '/X_train_point_by_point_' + str(t) + input_filename_addition + '.npy')
            r_train = np.load(unweighted_events_dir + '/r_train_point_by_point_' + str(t) + input_filename_addition + '.npy')

            # Scale data
            scaler = StandardScaler()
            scaler.fit(np.array(X_train, dtype=np.float64))
            X_train_transformed = scaler.transform(X_train)
            X_test_transformed = scaler.transform(X_test)

            regr = KerasRegressor(lambda: make_regressor(n_hidden_layers=n_hidden_layers),
                                  epochs=n_epochs, validation_split=0.142857,
                                  verbose=2)

            # Training
            regr.fit(X_train_transformed, np.log(r_train),
                     callbacks=([EarlyStopping(verbose=1, patience=3)] if early_stopping else None))

            # Evaluation
            prediction = regr.predict(X_test_transformed)
            this_r = np.exp(prediction[:])

            llr.append(- 19.2 / float(n_observed) * np.sum(np.log(this_r)))

            if t == theta_benchmark2:
                np.save(results_dir + '/r_nottrained_' + algorithm + filename_addition + '.npy', this_r)
            elif t == theta_benchmark1:
                np.save(results_dir + '/r_trained_' + algorithm + filename_addition + '.npy', this_r)

        llr = np.asarray(llr)

        print('')
        print('Interpolation')

        gp = GaussianProcessRegressor(normalize_y=True,
                                      kernel=C(1.0) * Matern(1.0, nu=0.5), n_restarts_optimizer=10)
        gp.fit(thetas[training_thetas], llr)
        llr_all = gp.predict(thetas)
        np.save(results_dir + '/llr_' + algorithm + filename_addition + '.npy', llr_all)



    ################################################################################
    # Carl approaches
    ################################################################################

    else:

        llr = []
        llr_calibrated = []

        # Loop over the 15 thetas
        print('')

        for t in training_thetas:

            print('Theta', t, thetas[t])

            # Load data
            X_train = np.load(unweighted_events_dir + '/X_train_point_by_point_' + str(t) + input_filename_addition + '.npy')
            y_train = np.load(unweighted_events_dir + '/y_train_point_by_point_' + str(t) + input_filename_addition + '.npy')

            # Scale data
            scaler = StandardScaler()
            scaler.fit(np.array(X_train, dtype=np.float64))
            X_train_transformed = scaler.transform(X_train)
            X_test_transformed = scaler.transform(X_test)
            X_calibration_transformed = scaler.transform(X_calibration)

            clf = KerasRegressor(lambda: make_classifier(n_hidden_layers=n_hidden_layers),
                                  epochs=n_epochs, validation_split=0.142857,
                                  verbose=2)

            # Training
            clf.fit(X_train_transformed, y_train,
                     callbacks=([EarlyStopping(verbose=1, patience=3)] if early_stopping else None))

            # carl wrapper
            ratio = ClassifierScoreRatio(clf, prefit=True)

            # Evaluation
            this_r, _ = ratio.predict(X_test_transformed)

            llr.append(- 19.2 / float(n_observed) * np.sum(np.log(this_r)))

            if t == theta_benchmark2:
                np.save(results_dir + '/r_nottrained_' + algorithm + filename_addition + '.npy', this_r)
            elif t == theta_benchmark1:
                np.save(results_dir + '/r_trained_' + algorithm + filename_addition + '.npy', this_r)

            # Calibration
            nc = X_calibration_transformed.shape[0]
            X_calibration_both = np.zeros((2 * nc, X_calibration_transformed.shape[1]))
            X_calibration_both[:nc] = X_calibration_transformed
            X_calibration_both[nc:] = X_calibration_transformed
            y_calibration = np.zeros(2 * nc)
            y_calibration[nc:] = 1.
            w_calibration = np.zeros(2 * nc)
            w_calibration[:nc] = weights_calibration[t]
            w_calibration[nc:] = weights_calibration[theta1]

            ratio_calibrated = ClassifierScoreRatio(
                CalibratedClassifierScoreCV(clf, cv='prefit', bins=50, independent_binning=False)
            )
            ratio_calibrated.fit(X_calibration_both, y_calibration, sample_weight=w_calibration)

            # Evaluation of calibrated classifier
            this_r, _ = ratio_calibrated.predict(X_test_transformed)
            llr_calibrated.append(- 19.2 / float(n_observed) * np.sum(np.log(this_r)))

            if t == theta_benchmark2:
                np.save(results_dir + '/r_nottrained_' + algorithm + '_calibrated' + filename_addition + '.npy', this_r)

                # Save calibration histograms
                np.save(results_dir + '/cal0histo_nottrained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator0.histogram_)
                np.save(results_dir + '/cal0edges_nottrained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator0.edges_[0])
                np.save(results_dir + '/cal1histo_nottrained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator1.histogram_)
                np.save(results_dir + '/cal1edges_nottrained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator1.edges_[0])
                
            elif t == theta_benchmark1:
                np.save(results_dir + '/r_trained_' + algorithm + '_calibrated' + filename_addition + '.npy', this_r)

                # Save calibration histograms
                np.save(results_dir + '/cal0histo_trained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator0.histogram_)
                np.save(results_dir + '/cal0edges_trained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator0.edges_[0])
                np.save(results_dir + '/cal1histo_trained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator1.histogram_)
                np.save(results_dir + '/cal1edges_trained_' + algorithm + filename_addition + '.npy', ratio_calibrated.classifier_.calibrators_[0].calibrator1.edges_[0])
                
        llr = np.asarray(llr)
        llr_calibrated = np.asarray(llr_calibrated)

        print('')
        print('Interpolation')

        gp = GaussianProcessRegressor(normalize_y=True,
                                      kernel=C(1.0) * Matern(1.0, nu=0.5), n_restarts_optimizer=10)
        gp.fit(thetas[training_thetas], llr)
        llr_all = gp.predict(thetas)

        np.save(results_dir + '/llr_' + algorithm + filename_addition + '.npy', llr_all)

        gp = GaussianProcessRegressor(normalize_y=True,
                                      kernel=C(1.0) * Matern(1.0, nu=0.5), n_restarts_optimizer=10)
        gp.fit(thetas[training_thetas], llr_calibrated)
        llr_calibrated_all = gp.predict(thetas)
        np.save(results_dir + '/llr_' + algorithm + '_calibrated' + filename_addition + '.npy', llr_calibrated_all)
