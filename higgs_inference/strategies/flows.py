################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import math
import time

from sklearn.preprocessing import StandardScaler
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
from sklearn.utils import shuffle

from higgs_inference import settings
from higgs_inference.various.utils import format_number, calculate_mean_squared_error
from higgs_inference.flows.inference.nde import MAFInference
from higgs_inference.flows.inference.scandal import SCANDALInference


def flow_inference(algorithm='maf',
                   training_sample='baseline',  # 'baseline', 'basis', 'random'
                   use_smearing=False,
                   denominator=0,
                   alpha=None,
                   training_sample_size=None,
                   do_neyman=False,
                   options=''):  # all other options in a string

    """
    Likelihood ratio estimation through parameterized or morphing-aware versions of CARL, CASCAL, ROLR, and RASCAL.

    :param algorithm: Inference strategy. 'maf' for MAF, 'scandal' for SCANDAL.
    :param training_sample: Training sample. Can be 'baseline', 'basis', or 'random'.
    :param use_smearing: Whether to use the training and evaluation sample with (simplified) detector simulation.
    :param denominator: Which of five predefined denominator (reference) hypotheses to use.
    :param alpha: Hyperparameter that multiplies score term in loss function for RASCAL and CASCAL. If None, default
                  values are used.
    :param training_sample_size: If not None, limits the training sample size to the given value.
    :param do_neyman: Switches on the evaluation of toy experiments for the Neyman construction.
    :param options: Further options in a list of strings or string. 'learns' changes the architecture such that the
                    fully connected networks represent s rather than log r. 'new' changes the samples. 'short' and
                    'long' change the number of epochs. 'deep' and 'shallow' use more or less hidden layers. 'factorsm'
                    changes the architecture in the morphing-aware mode such that the SM and the deviation from it are
                    modelled independently. 'slowlearning' and 'fastlearning' change the learning rate, while
                    'constantlr' turns off the learning rate decay. 'neyman2' and 'neyman3' change the Neyman
                    construction sample, and 'recalibration' activates the calculation of E[r] on a separate sample
                    for the expectation calibration step. 'debug' activates a debug mode with much smaller samples.
    """

    logging.info('Starting parameterized inference')

    ################################################################################
    # Settings
    ################################################################################

    assert algorithm in ['maf', 'scandal']
    assert training_sample in ['baseline', 'basis', 'random']

    random_theta_mode = training_sample == 'random'
    basis_theta_mode = training_sample == 'basis'

    new_sample_mode = ('new' in options)
    short_mode = ('short' in options)
    long_mode = ('long' in options)
    deep_mode = ('deep' in options)
    shallow_mode = ('shallow' in options)
    small_lr_mode = ('slowlearning' in options)
    large_lr_mode = ('fastlearning' in options)
    large_batch_mode = ('largebatch' in options)
    small_batch_mode = ('smallbatch' in options)
    constant_lr_mode = ('constantlr' in options)
    neyman2_mode = ('neyman2' in options)
    neyman3_mode = ('neyman3' in options)

    filename_addition = ''

    if random_theta_mode:
        filename_addition += '_random'
    elif basis_theta_mode:
        filename_addition += '_basis'

    learning_rate = settings.learning_rate_default
    if small_lr_mode:
        filename_addition += '_slowlearning'
        learning_rate = settings.learning_rate_small
    elif large_lr_mode:
        filename_addition += '_fastlearning'
        learning_rate = settings.learning_rate_large

    lr_decay = 0.01
    if constant_lr_mode:
        lr_decay = 1.
        filename_addition += '_constantlr'

    batch_size = settings.batch_size_default
    if large_batch_mode:
        filename_addition += '_largebatch'
        batch_size = settings.batch_size_large
    elif small_batch_mode:
        filename_addition += '_smallbatch'
        batch_size = settings.batch_size_small
    settings.batch_size = batch_size

    alpha_scandal = settings.alpha_scandal_default
    if alpha is not None:
        alpha_scandal = alpha
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
    early_stopping_patience = settings.early_stopping_patience
    if long_mode:
        n_epochs = settings.n_epochs_long
        filename_addition += '_long'
    elif short_mode:
        n_epochs = settings.n_epochs_short
        early_stopping = False
        filename_addition += '_short'

    if training_sample_size is not None:
        filename_addition += '_trainingsamplesize_' + str(training_sample_size)
        n_epoch_factor = int(len(settings.thetas_train) * (settings.n_events_baseline_num
                                                           + settings.n_events_baseline_den)
                             / training_sample_size)
        n_epochs *= n_epoch_factor
        early_stopping_patience *= n_epoch_factor

    input_X_prefix = ''
    if use_smearing:
        input_X_prefix = 'smeared_'
        filename_addition += '_smeared'

    th1 = settings.theta1_default
    input_filename_addition = ''
    if denominator > 0:
        input_filename_addition = '_denom' + str(denominator)
        filename_addition += '_denom' + str(denominator)
        th1 = settings.theta1_alternatives[denominator - 1]
    theta1 = settings.thetas[th1]

    if new_sample_mode:
        filename_addition += '_new'
        input_filename_addition += '_new'

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

    results_dir = settings.base_dir + '/results/parameterized'
    neyman_dir = settings.neyman_dir + '/parameterized'

    logging.info('Main settings:')
    logging.info('  Algorithm:                %s', algorithm)
    logging.info('  Training sample:          %s', training_sample)
    logging.info('  Denominator theta:        denominator %s = theta %s = %s', denominator, th1,
                 theta1)
    logging.info('Options:')
    logging.info('  Number of MADEs:    %s', n_hidden_layers)
    if algorithm == 'scandal':
        logging.info('  alpha:                    %s', alpha_scandal)
    elif algorithm == 'combinedregression':
        logging.info('  Batch size:               %s', batch_size)
    logging.info('  Learning rate:            %s', learning_rate)
    logging.info('  Learning rate decay:      %s', lr_decay)
    logging.info('  Number of epochs:         %s', n_epochs)
    logging.info('  Training samples:         %s', 'all' if training_sample_size is None else training_sample_size)
    if do_neyman:
        logging.info('  NC experiments:           (%s alternate + %s null) experiments with %s alternate events each',
                     n_neyman_alternate_experiments, n_neyman_null_experiments, n_expected_events_neyman)
    else:
        logging.info('  NC experiments:           False')

    ################################################################################
    # Data
    ################################################################################

    # Load data
    train_filename = '_train'
    if random_theta_mode:
        train_filename += '_random'
    elif basis_theta_mode:
        train_filename += '_basis'
    train_filename += input_filename_addition

    X_train = np.load(settings.unweighted_events_dir + '/' + input_X_prefix + 'X' + train_filename + '.npy')
    y_train = np.load(settings.unweighted_events_dir + '/y' + train_filename + '.npy')
    scores_train = np.load(settings.unweighted_events_dir + '/scores' + train_filename + '.npy')
    r_train = np.load(settings.unweighted_events_dir + '/r' + train_filename + '.npy')
    theta0_train = np.load(settings.unweighted_events_dir + '/theta0' + train_filename + '.npy')

    X_test = np.load(
        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_test' + input_filename_addition + '.npy')
    r_test = np.load(settings.unweighted_events_dir + '/r_test' + input_filename_addition + '.npy')

    X_illustration = np.load(
        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_illustration' + input_filename_addition + '.npy')

    if do_neyman:
        X_neyman_alternate = np.load(
            settings.unweighted_events_dir + '/neyman/' + input_X_prefix + 'X_' + neyman_filename + '_alternate.npy')

    n_events_test = X_test.shape[0]
    assert settings.n_thetas == r_test.shape[0]

    # Shuffle training data
    X_train, y_train, scores_train, r_train, theta0_train = shuffle(X_train, y_train, scores_train, r_train,
                                                                    theta0_train, random_state=44)

    # Limit training sample size
    if training_sample_size is not None:
        original_training_sample_size = X_train.shape[0]

        X_train = X_train[:training_sample_size]
        y_train = y_train[:training_sample_size]
        scores_train = scores_train[:training_sample_size]
        r_train = r_train[:training_sample_size]
        theta0_train = theta0_train[:training_sample_size]

        logging.info('Reduced training sample size from %s to %s (factor %s)', original_training_sample_size,
                     X_train.shape[0], n_epoch_factor)

    # Normalize data
    scaler = StandardScaler()
    scaler.fit(np.array(X_train, dtype=np.float64))
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    X_illustration_transformed = scaler.transform(X_illustration)
    if do_neyman:
        X_neyman_alternate_transformed = scaler.transform(X_neyman_alternate.reshape((-1, X_neyman_alternate.shape[2])))

    n_parameters = scores_train.shape[1]
    n_observables = X_train_transformed.shape[1]

    ################################################################################
    # Training
    ################################################################################

    # Inference object
    inference_type = SCANDALInference if algorithm == 'scandal' else MAFInference
    inference = inference_type(
        n_mades=n_hidden_layers,
        n_made_hidden_layers=1,
        n_made_units_per_layer=100,
        batch_norm=False,
        activation='tanh',
        n_parameters=n_parameters,
        n_observables=n_observables
    )

    # Training
    logging.info('Starting training')
    inference.fit(
        theta0_train, X_train_transformed,
        y_train, r_train, scores_train,
        n_epochs=n_epochs,
        batch_size=batch_size,
        trainer='adam',
        initial_learning_rate=learning_rate,
        final_learning_rate=learning_rate * lr_decay,
        alpha=alpha_scandal,
        validation_split=settings.validation_split,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience
    )

    ################################################################################
    # Raw evaluation loop
    ################################################################################

    logging.info('Starting evaluation')
    expected_llr = []
    mse_log_r = []
    trimmed_mse_log_r = []
    eval_times = []
    expected_r_vs_sm = []

    for t, theta in enumerate(settings.thetas):

        if (t + 1) % 100 == 0:
            logging.info('Starting theta %s / %s', t + 1, settings.n_thetas)

        ################################################################################
        # Evaluation
        ################################################################################

        # Evaluation
        time_before = time.time()
        this_log_r = inference.predict_ratio(
            x=X_test_transformed,
            theta0=theta,
            theta1=theta1,
            log=True
        )
        this_score = inference.predict_score(
            theta=theta,
            x=X_test_transformed
        )
        eval_times.append(time.time() - time_before)

        # Extract numbers of interest
        expected_llr.append(- 2. * settings.n_expected_events / n_events_test * np.sum(this_log_r))
        mse_log_r.append(calculate_mean_squared_error(np.log(r_test[t]), this_log_r, 0.))
        trimmed_mse_log_r.append(calculate_mean_squared_error(np.log(r_test[t]), this_log_r, 'auto'))

        if t == settings.theta_observed:
            r_sm = np.exp(this_log_r)
        expected_r_vs_sm.append(np.mean(np.exp(this_log_r) / r_sm))

        # For benchmark thetas, save more info
        if t == settings.theta_benchmark_nottrained:
            np.save(results_dir + '/r_nottrained_' + algorithm + filename_addition + '.npy', np.exp(this_log_r))
            np.save(results_dir + '/scores_nottrained_' + algorithm + filename_addition + '.npy', this_score)
            np.save(results_dir + '/r_vs_sm_nottrained_' + algorithm + filename_addition + '.npy',
                    np.exp(this_log_r) / r_sm)
        elif t == settings.theta_benchmark_trained:
            np.save(results_dir + '/r_trained_' + algorithm + filename_addition + '.npy', np.exp(this_log_r))
            np.save(results_dir + '/scores_trained_' + algorithm + filename_addition + '.npy', this_score)
            np.save(results_dir + '/r_vs_sm_trained_' + algorithm + filename_addition + '.npy',
                    np.exp(this_log_r) / r_sm)

        ################################################################################
        # Illustration
        ################################################################################

        if t == settings.theta_benchmark_illustration:
            # Evaluate illustration data
            r_hat_illustration = inference.predict_ratio(
                x=X_illustration_transformed,
                theta0=theta,
                theta1=theta1,
                log=False
            )

            np.save(results_dir + '/r_illustration_' + algorithm + filename_addition + '.npy', r_hat_illustration)

        ################################################################################
        # Neyman construction toys
        ################################################################################

        if do_neyman:

            # Neyman construction: evaluate alternate sample (raw)
            log_r_neyman_alternate = inference.predict_ratio(
                theta,
                theta1,
                X_neyman_alternate_transformed,
                log=True
            )
            llr_neyman_alternate = -2. * np.sum(log_r_neyman_alternate.reshape((-1, n_expected_events_neyman)),
                                                axis=1)
            np.save(neyman_dir + '/' + neyman_filename + '_llr_alternate_' + str(
                t) + '_' + algorithm + filename_addition + '.npy', llr_neyman_alternate)

            # NC: null
            X_neyman_null = np.load(
                settings.unweighted_events_dir + '/neyman/' + input_X_prefix + 'X_' + neyman_filename + '_null_' + str(
                    t) + '.npy')
            X_neyman_null_transformed = scaler.transform(X_neyman_null.reshape((-1, X_neyman_null.shape[2])))

            # Neyman construction: evaluate null sample (raw)
            log_r_neyman_null = inference.predict_ratio(
                theta,
                theta1,
                X_neyman_null_transformed,
                log=True
            )
            llr_neyman_null = -2. * np.sum(log_r_neyman_null.reshape((-1, n_expected_events_neyman)), axis=1)
            np.save(neyman_dir + '/' + neyman_filename + '_llr_null_' + str(
                t) + '_' + algorithm + filename_addition + '.npy', llr_neyman_null)

            # NC: null evaluated at alternate
            if t == settings.theta_observed:
                for tt in range(settings.n_thetas):
                    X_neyman_null = np.load(
                        settings.unweighted_events_dir + '/neyman/' + input_X_prefix + 'X_' + neyman_filename + '_null_'
                        + str(tt) + '.npy'
                    )
                    X_neyman_null_transformed = scaler.transform(
                        X_neyman_null.reshape((-1, X_neyman_null.shape[2])))

                    # Neyman construction: evaluate null sample (raw)
                    log_r_neyman_null = inference.predict_ratio(
                        theta,
                        theta1,
                        X_neyman_null_transformed,
                        log=True
                    )
                    llr_neyman_null = -2. * np.sum(log_r_neyman_null.reshape((-1, n_expected_events_neyman)), axis=1)
                    np.save(neyman_dir + '/' + neyman_filename + '_llr_nullatalternate_' + str(
                        tt) + '_' + algorithm + filename_addition + '.npy', llr_neyman_null)

    # Save evaluation results
    expected_llr = np.asarray(expected_llr)
    mse_log_r = np.asarray(mse_log_r)
    trimmed_mse_log_r = np.asarray(trimmed_mse_log_r)
    expected_r_vs_sm = np.asarray(expected_r_vs_sm)
    np.save(results_dir + '/llr_' + algorithm + filename_addition + '.npy', expected_llr)
    np.save(results_dir + '/mse_logr_' + algorithm + filename_addition + '.npy', mse_log_r)
    np.save(results_dir + '/trimmed_mse_logr_' + algorithm + filename_addition + '.npy', trimmed_mse_log_r)
    np.save(results_dir + '/expected_r_vs_sm_' + algorithm + filename_addition + '.npy',
            expected_r_vs_sm)

    # Evaluation times
    logging.info('Evaluation timing: median %s s, mean %s s', np.median(eval_times), np.mean(eval_times))
