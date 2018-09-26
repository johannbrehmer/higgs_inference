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
                   morphing_aware=False,
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
    :param morphing_aware: bool that decides whether a morphing-aware or morphing-agnostic parameterized architecture is
                           used.
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

    learn_logr_mode = ('learns' not in options)
    new_sample_mode = ('new' in options)
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
    neyman2_mode = ('neyman2' in options)
    neyman3_mode = ('neyman3' in options)
    # recalibration_mode = ('recalibration' in options)

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

    if training_sample_size is not None:
        filename_addition += '_trainingsamplesize_' + str(training_sample_size)
        n_epoch_factor = int(len(settings.thetas_train) * (settings.n_events_baseline_num
                                                           + settings.n_events_baseline_den)
                             / training_sample_size)
        n_epochs *= n_epoch_factor
        lr_decay /= float(n_epoch_factor)
        early_stopping_patience *= n_epoch_factor

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
    logging.info('  Morphing-aware:           %s', morphing_aware)
    logging.info('  Training sample:          %s', training_sample)
    logging.info('  Denominator theta:        denominator %s = theta %s = %s', denominator, theta1,
                 settings.thetas[theta1])
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
    logging.info('  Debug mode:               %s', debug_mode)

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
    # X_train_unshuffled = np.load(settings.unweighted_events_dir + '/' + input_X_prefix + 'X' + train_filename + '.npy')
    y_train = np.load(settings.unweighted_events_dir + '/y' + train_filename + '.npy')
    scores_train = np.load(settings.unweighted_events_dir + '/scores' + train_filename + '.npy')
    r_train = np.load(settings.unweighted_events_dir + '/r' + train_filename + '.npy')
    theta0_train = np.load(settings.unweighted_events_dir + '/theta0' + train_filename + '.npy')
    # theta0_train_unshuffled = np.load(settings.unweighted_events_dir + '/theta0' + train_filename + '.npy')

    # X_calibration = np.load(
    #    settings.unweighted_events_dir + '/' + input_X_prefix + 'X_calibration' + input_filename_addition + '.npy')
    # weights_calibration = np.load(
    #    settings.unweighted_events_dir + '/weights_calibration' + input_filename_addition + '.npy')

    # if recalibration_mode:
    #     X_recalibration = np.load(
    #         settings.unweighted_events_dir + '/' + input_X_prefix + 'X_recalibration' + '.npy')

    X_test = np.load(
        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_test' + input_filename_addition + '.npy')
    r_test = np.load(settings.unweighted_events_dir + '/r_test' + input_filename_addition + '.npy')

    # X_roam = np.load(
    #    settings.unweighted_events_dir + '/' + input_X_prefix + 'X_roam' + input_filename_addition + '.npy')
    # n_roaming = len(X_roam)

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
    # X_train_transformed_unshuffled = scaler.transform(X_train_unshuffled)
    X_test_transformed = scaler.transform(X_test)
    # X_roam_transformed = scaler.transform(X_roam)
    # X_calibration_transformed = scaler.transform(X_calibration)
    X_illustration_transformed = scaler.transform(X_illustration)
    # if recalibration_mode:
    #     X_recalibration_transformed = scaler.transform(X_recalibration)
    if do_neyman:
        X_neyman_alternate_transformed = scaler.transform(
            X_neyman_alternate.reshape((-1, X_neyman_alternate.shape[2])))

    # # Roaming data
    # X_thetas_train = np.hstack((X_train_transformed, theta0_train))
    # # X_thetas_train_unshuffled = np.hstack((X_train_transformed_unshuffled, theta0_train_unshuffled))
    # # y_logr_score_train = np.hstack((y_train.reshape(-1, 1), np.log(r_train).reshape((-1, 1)), scores_train))
    # xi = np.linspace(-1.0, 1.0, settings.n_thetas_roam)
    # yi = np.linspace(-1.0, 1.0, settings.n_thetas_roam)
    # xx, yy = np.meshgrid(xi, yi)
    # thetas_roam = np.asarray((xx.flatten(), yy.flatten())).T
    # X_thetas_roam = []
    # for i in range(n_roaming):
    #     X_thetas_roam.append(np.zeros((settings.n_thetas_roam ** 2, X_roam_transformed.shape[1] + 2)))
    #     X_thetas_roam[-1][:, :-2] = X_roam_transformed[i, :]
    #     X_thetas_roam[-1][:, -2:] = thetas_roam

    # if debug_mode:
    #     X_thetas_train = X_thetas_train[::100]
    #     y_logr_score_train = y_logr_score_train[::100]
    #     X_test_transformed = X_test[::100]
    #     r_test = r_test[:, ::100]
    #     X_calibration_transformed = X_calibration_transformed[::100]
    #     weights_calibration = weights_calibration[:, ::100]
    #     X_illustration_transformed = X_illustration_transformed[::100]
    #     if recalibration_mode:
    #         X_recalibration_transformed = X_recalibration_transformed[::100]
    #     n_events_test = len(X_test_transformed)

    n_parameters = scores_train.shape[1]
    n_observables = X_train_transformed.shape[1]

    ################################################################################
    # Training
    ################################################################################

    # Inference object
    inference_type = SCANDALInference if algorithm == 'scandal' else MAFInference
    inference = inference_type(
        algorithm,
        n_mades=n_hidden_layers,
        n_made_hidden_layers=1,
        n_made_units_per_layer=100,
        batch_norm=False,
        activation='tanh',
        n_parameters=n_parameters,
        n_observables=n_observables
    )

    # # Callbacks
    # callbacks = []
    # detailed_history = {}
    # callbacks.append(DetailedHistory(detailed_history))
    # if not constant_lr_mode:
    #     def lr_scheduler(epoch):
    #         return learning_rate * np.exp(- epoch * lr_decay)
    #
    #     callbacks.append(LearningRateScheduler(lr_scheduler))
    # if early_stopping:
    #     callbacks.append(EarlyStopping(verbose=1, patience=early_stopping_patience))

    # Training
    logging.info('Starting training')
    inference.fit(
        theta0_train, X_train_transformed,
        y_train, r_train, scores_train,
        n_epochs=n_epochs,
        batch_size=batch_size,
        trainer='adam',
        initial_learning_rate=learning_rate,
        final_learning_rate=0.01,
        alpha=alpha_scandal,
        validation_split=settings.validation_split,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience
    )

    # history = regr.fit(X_thetas_train[::], y_logr_score_train[::], callbacks=callbacks, batch_size=batch_size)

    # Save metrics
    # def _save_metrics(key, filename):
    #     try:
    #         metrics = np.asarray([history.history[key], history.history['val_' + key]])
    #         np.save(results_dir + '/traininghistory_' + filename + '_' + algorithm + filename_addition + '.npy',
    #                 metrics)
    #     except KeyError:
    #         logging.warning('Key %s not found in per-epoch history. Available keys: %s', key,
    #                         list(history.history.keys()))
    #     try:
    #         detailed_metrics = np.asarray(detailed_history[key])
    #         np.save(
    #             results_dir + '/traininghistory_100batches_' + filename + '_' + algorithm + filename_addition + '.npy',
    #             detailed_metrics)
    #     except KeyError:
    #         logging.warning('Key %s not found in per-batch history. Available keys: %s', key,
    #                         list(detailed_history.keys()))

    # _save_metrics('loss', 'loss')
    # _save_metrics('full_cross_entropy', 'ce')
    # _save_metrics('full_modified_cross_entropy', 'mce')
    # _save_metrics('full_mse_log_r', 'mse_logr')
    # _save_metrics('full_mse_score', 'mse_scores')

    # Evaluate rhat on training sample
    # r_hat_train = np.exp(regr.predict(X_thetas_train_unshuffled)[:, 1])
    # np.save(results_dir + '/r_train_' + algorithm + filename_addition + '.npy', r_hat_train)

    ################################################################################
    # Raw evaluation loop
    ################################################################################

    # carl wrapper
    # ratio = ClassifierScoreRatio(regr, prefit=True)

    logging.info('Starting evaluation')
    expected_llr = []
    mse_log_r = []
    trimmed_mse_log_r = []
    eval_times = []
    expected_r_vs_sm = []
    # if recalibration_mode:
    #     recalibration_expected_r = []

    for t, theta in enumerate(settings.thetas):

        if (t + 1) % 100 == 0:
            logging.info('Starting theta %s / %s', t + 1, settings.n_thetas)

        ################################################################################
        # Evaluation
        ################################################################################

        # Prepare test data
        thetas0_array = np.zeros((X_test_transformed.shape[0], 2), dtype=X_test_transformed.dtype)
        thetas0_array[:, :] = theta
        # X_thetas_test = np.hstack((X_test_transformed, thetas0_array))

        # Evaluation
        time_before = time.time()
        this_log_r = inference.predict_ratio(
            x=X_test_transformed,
            theta0=thetas0_array,
            theta1=settings.thetas[theta1],
            log=True
        )
        this_score = inference.predict_score(
            theta=thetas0_array,
            x=X_test_transformed
        )
        eval_times.append(time.time() - time_before)

        # if morphing_aware:
        #     this_wi = prediction[:, 4:19]
        #     this_ri = prediction[:, 19:]
        #     logging.debug('Morphing weights for theta %s (%s): %s', t, theta, this_wi[0])

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
            # if morphing_aware:
            #     np.save(results_dir + '/morphing_ri_nottrained_' + algorithm + filename_addition + '.npy', this_ri)
            #     np.save(results_dir + '/morphing_wi_nottrained_' + algorithm + filename_addition + '.npy', this_wi)
        elif t == settings.theta_benchmark_trained:
            np.save(results_dir + '/r_trained_' + algorithm + filename_addition + '.npy', np.exp(this_log_r))
            np.save(results_dir + '/scores_trained_' + algorithm + filename_addition + '.npy', this_score)
            np.save(results_dir + '/r_vs_sm_trained_' + algorithm + filename_addition + '.npy',
                    np.exp(this_log_r) / r_sm)
            # if morphing_aware:
            #     np.save(results_dir + '/morphing_ri_trained_' + algorithm + filename_addition + '.npy', this_ri)
            #     np.save(results_dir + '/morphing_wi_trained_' + algorithm + filename_addition + '.npy', this_wi)

        ################################################################################
        # Recalibration
        ################################################################################

        # if recalibration_mode:
        #     # Prepare data for recalibration
        #     thetas0_array = np.zeros((X_recalibration_transformed.shape[0], 2),
        #                              dtype=X_recalibration_transformed.dtype)
        #     thetas0_array[:, :] = settings.thetas[t]
        #     X_thetas_recalibration = np.hstack((X_recalibration_transformed, thetas0_array))
        #
        #     # Evaluate recalibration data
        #     prediction = regr.predict(X_thetas_recalibration)
        #     this_r = np.exp(prediction[:, 1])
        #     if t == settings.theta_observed:
        #         r_recalibration_sm = this_r
        #     recalibration_expected_r.append(np.mean(this_r / r_recalibration_sm))

        ################################################################################
        # Illustration
        ################################################################################

        if t == settings.theta_benchmark_illustration:
            # Prepare data for illustration
            thetas0_array = np.zeros((X_illustration_transformed.shape[0], 2),
                                     dtype=X_illustration_transformed.dtype)
            thetas0_array[:, :] = settings.thetas[t]
            # X_thetas_illustration = np.hstack((X_illustration_transformed, thetas0_array))

            # Evaluate illustration data
            r_hat_illustration = inference.predict_ratio(
                x=X_illustration_transformed,
                theta0=thetas0_array,
                theta1=settings.thetas[theta1],
                log=False
            )

            np.save(results_dir + '/r_illustration_' + algorithm + filename_addition + '.npy', r_hat_illustration)

        ################################################################################
        # Neyman construction toys
        ################################################################################

        if do_neyman:
            # Prepare alternate data for Neyman construction
            thetas0_array = np.zeros((X_neyman_alternate_transformed.shape[0], 2),
                                     dtype=X_neyman_alternate_transformed.dtype)
            thetas0_array[:, :] = theta
            # X_thetas_neyman_alternate = np.hstack((X_neyman_alternate_transformed, thetas0_array))

            # Neyman construction: evaluate alternate sample (raw)
            log_r_neyman_alternate = inference.predict_ratio(
                thetas0_array,
                settings.thetas[theta1],
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
            X_neyman_null_transformed = scaler.transform(
                X_neyman_null.reshape((-1, X_neyman_null.shape[2])))

            # Prepare null data for Neyman construction
            thetas0_array = np.zeros((X_neyman_null_transformed.shape[0], 2), dtype=X_neyman_null_transformed.dtype)
            thetas0_array[:, :] = settings.thetas[t]
            # X_thetas_neyman_null = np.hstack((X_neyman_null_transformed, thetas0_array))

            # Neyman construction: evaluate null sample (raw)
            log_r_neyman_null = inference.predict_ratio(
                thetas0_array,
                settings.thetas[theta1],
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
                        settings.unweighted_events_dir + '/neyman/' + input_X_prefix + 'X_' + neyman_filename + '_null_' + str(
                            tt) + '.npy')
                    X_neyman_null_transformed = scaler.transform(
                        X_neyman_null.reshape((-1, X_neyman_null.shape[2])))
                    # X_thetas_neyman_null = np.hstack((X_neyman_null_transformed, thetas0_array))

                    # Neyman construction: evaluate null sample (raw)
                    log_r_neyman_null = inference.predict_ratio(
                        thetas0_array,
                        settings.thetas[theta1],
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
    # if recalibration_mode:
    #     recalibration_expected_r = np.asarray(recalibration_expected_r)
    #     np.save(results_dir + '/recalibration_expected_r_vs_sm_' + algorithm + filename_addition + '.npy',
    #             recalibration_expected_r)

    # Evaluation times
    logging.info('Evaluation timing: median %s s, mean %s s', np.median(eval_times), np.mean(eval_times))

    # logging.info('Starting roaming')
    # r_roam = []
    # for i in range(n_roaming):
    #     prediction = regr.predict(X_thetas_roam[i])
    #     r_roam.append(np.exp(prediction[:, 1]))
    # r_roam = np.asarray(r_roam)
    # np.save(results_dir + '/r_roam_' + algorithm + filename_addition + '.npy', r_roam)

    ################################################################################
    # Calibrated evaluation loop
    ################################################################################

    # logging.info('Starting calibrated evaluation and roaming')
    # expected_llr_calibrated = []
    # mse_log_r_calibrated = []
    # trimmed_mse_log_r_calibrated = []
    # r_roam_temp = np.zeros((settings.n_thetas, n_roaming))
    # eval_times = []
    # expected_r_vs_sm = []
    # if recalibration_mode:
    #     recalibration_expected_r = []
    #
    # for t, theta in enumerate(settings.thetas):
    #
    #     if (t + 1) % 100 == 0:
    #         logging.info('Starting theta %s / %s', t + 1, settings.n_thetas)
    #
    #     ################################################################################
    #     # Calibration
    #     ################################################################################
    #
    #     # Prepare data for calibration
    #     n_calibration_each = X_calibration_transformed.shape[0]
    #     thetas0_array = np.zeros((n_calibration_each, 2), dtype=X_calibration_transformed.dtype)
    #     thetas0_array[:, :] = settings.thetas[t]
    #     X_thetas_calibration = np.hstack((X_calibration_transformed, thetas0_array))
    #     X_thetas_calibration = np.vstack((X_thetas_calibration, X_thetas_calibration))
    #     y_calibration = np.zeros(2 * n_calibration_each)
    #     y_calibration[n_calibration_each:] = 1.
    #     w_calibration = np.zeros(2 * n_calibration_each)
    #     w_calibration[:n_calibration_each] = weights_calibration[t]
    #     w_calibration[n_calibration_each:] = weights_calibration[theta1]
    #
    #     # Calibration
    #     ratio_calibrated = ClassifierScoreRatio(
    #         CalibratedClassifierScoreCV(regr, cv='prefit', method='isotonic')
    #     )
    #     ratio_calibrated.fit(X_thetas_calibration, y_calibration, sample_weight=w_calibration)
    #
    #     ################################################################################
    #     # Evaluation
    #     ################################################################################
    #
    #     # Prepare data
    #     thetas0_array = np.zeros((X_test_transformed.shape[0], 2), dtype=X_test_transformed.dtype)
    #     thetas0_array[:, :] = settings.thetas[t]
    #     X_thetas_test = np.hstack((X_test_transformed, thetas0_array))
    #
    #     time_before = time.time()
    #     this_r, this_other = ratio_calibrated.predict(X_thetas_test)
    #     eval_times.append(time.time() - time_before)
    #     this_score = this_other[:, 1:3]
    #
    #     # Extract numbers of interest
    #     expected_llr_calibrated.append(- 2. * settings.n_expected_events / n_events_test * np.sum(np.log(this_r)))
    #     mse_log_r_calibrated.append(calculate_mean_squared_error(np.log(r_test[t]), np.log(this_r), 0.))
    #     trimmed_mse_log_r_calibrated.append(calculate_mean_squared_error(np.log(r_test[t]), np.log(this_r), 'auto'))
    #
    #     if t == settings.theta_observed:
    #         r_sm = this_r
    #     expected_r_vs_sm.append(np.mean(this_r / r_sm))
    #
    #     # For benchmark theta, save more data
    #     if t == settings.theta_benchmark_nottrained:
    #         np.save(results_dir + '/scores_nottrained_' + algorithm + '_calibrated' + filename_addition + '.npy',
    #                 this_score)
    #         np.save(results_dir + '/r_nottrained_' + algorithm + '_calibrated' + filename_addition + '.npy', this_r)
    #         np.save(results_dir + '/r_vs_sm_nottrained_' + algorithm + '_calibrated' + filename_addition + '.npy',
    #                 this_r / r_sm)
    #         np.save(results_dir + '/calvalues_nottrained_' + algorithm + filename_addition + '.npy',
    #                 ratio_calibrated.classifier_.calibration_sample[:n_calibration_each])
    #
    #     elif t == settings.theta_benchmark_trained:
    #         np.save(results_dir + '/scores_trained_' + algorithm + '_calibrated' + filename_addition + '.npy',
    #                 this_score)
    #         np.save(results_dir + '/r_trained_' + algorithm + '_calibrated' + filename_addition + '.npy', this_r)
    #         np.save(results_dir + '/r_vs_sm_trained_' + algorithm + '_calibrated' + filename_addition + '.npy',
    #                 this_r / r_sm)
    #         np.save(results_dir + '/calvalues_trained_' + algorithm + filename_addition + '.npy',
    #                 ratio_calibrated.classifier_.calibration_sample[:n_calibration_each])
    #
    #     ################################################################################
    #     # Recalibration
    #     ################################################################################
    #
    #     if recalibration_mode:
    #         # Prepare data for recalibration
    #         thetas0_array = np.zeros((X_recalibration_transformed.shape[0], 2), dtype=X_recalibration_transformed.dtype)
    #         thetas0_array[:, :] = settings.thetas[t]
    #         X_thetas_recalibration = np.hstack((X_recalibration_transformed, thetas0_array))
    #
    #         # Evaluate recalibration data
    #         this_r, _ = ratio_calibrated.predict(X_thetas_recalibration)
    #         if t == settings.theta_observed:
    #             r_recalibration_sm = this_r
    #         recalibration_expected_r.append(np.mean(this_r / r_recalibration_sm))
    #
    #     ################################################################################
    #     # Illustration
    #     ################################################################################
    #
    #     if t == settings.theta_benchmark_illustration:
    #         # Prepare data for illustration
    #         thetas0_array = np.zeros((X_illustration_transformed.shape[0], 2),
    #                                  dtype=X_illustration_transformed.dtype)
    #         thetas0_array[:, :] = settings.thetas[t]
    #         X_thetas_illustration = np.hstack((X_illustration_transformed, thetas0_array))
    #
    #         # Evaluate illustration data
    #         r_hat_illustration, _ = ratio_calibrated.predict(X_thetas_illustration)
    #
    #         np.save(results_dir + '/r_illustration_' + algorithm + '_calibrated' + filename_addition + '.npy',
    #                 r_hat_illustration)
    #
    #     ################################################################################
    #     # Neyman construction toys
    #     ################################################################################
    #
    #     # Neyman construction
    #     if do_neyman:
    #         # Prepare alternate data for Neyman construction
    #         thetas0_array = np.zeros((X_neyman_alternate_transformed.shape[0], 2),
    #                                  dtype=X_neyman_alternate_transformed.dtype)
    #         thetas0_array[:, :] = settings.thetas[t]
    #         X_thetas_neyman_alternate = np.hstack((X_neyman_alternate_transformed, thetas0_array))
    #
    #         # Neyman construction: alternate (calibrated)
    #         r_neyman_alternate, _ = ratio_calibrated.predict(X_thetas_neyman_alternate)
    #         llr_neyman_alternate = -2. * np.sum(np.log(r_neyman_alternate).reshape((-1, n_expected_events_neyman)),
    #                                             axis=1)
    #         np.save(neyman_dir + '/' + neyman_filename + '_llr_alternate_' + str(
    #             t) + '_' + algorithm + '_calibrated' + filename_addition + '.npy', llr_neyman_alternate)
    #
    #         # Neyman construction: null
    #         X_neyman_null = np.load(
    #             settings.unweighted_events_dir + '/neyman/' + input_X_prefix + 'X_' + neyman_filename + '_null_' + str(
    #                 t) + '.npy')
    #         X_neyman_null_transformed = scaler.transform(
    #             X_neyman_null.reshape((-1, X_neyman_null.shape[2])))
    #
    #         # Prepare null data for Neyman construction
    #         thetas0_array = np.zeros((X_neyman_null_transformed.shape[0], 2),
    #                                  dtype=X_neyman_null_transformed.dtype)
    #         thetas0_array[:, :] = settings.thetas[t]
    #         X_thetas_neyman_null = np.hstack((X_neyman_null_transformed, thetas0_array))
    #
    #         # Neyman construction: evaluate null (calibrated)
    #         r_neyman_null, _ = ratio_calibrated.predict(X_thetas_neyman_null)
    #         llr_neyman_null = -2. * np.sum(
    #             np.log(r_neyman_null).reshape((-1, n_expected_events_neyman)), axis=1)
    #
    #         np.save(neyman_dir + '/' + neyman_filename + '_llr_null_' + str(
    #             t) + '_' + algorithm + '_calibrated' + filename_addition + '.npy', llr_neyman_null)
    #
    #         # NC: null evaluated at alternate
    #         if t == settings.theta_observed:
    #             for tt in range(settings.n_thetas):
    #                 X_neyman_null = np.load(
    #                     settings.unweighted_events_dir + '/neyman/' + input_X_prefix + 'X_' + neyman_filename + '_null_' + str(
    #                         tt) + '.npy')
    #                 X_neyman_null_transformed = scaler.transform(
    #                     X_neyman_null.reshape((-1, X_neyman_null.shape[2])))
    #                 X_thetas_neyman_null = np.hstack((X_neyman_null_transformed, thetas0_array))
    #
    #                 # Neyman construction: evaluate null sample (calibrated)
    #                 r_neyman_null, _ = ratio_calibrated.predict(X_thetas_neyman_null)
    #                 llr_neyman_null = -2. * np.sum(
    #                     np.log(r_neyman_null).reshape((-1, n_expected_events_neyman)), axis=1)
    #                 np.save(neyman_dir + '/' + neyman_filename + '_llr_nullatalternate_' + str(
    #                     tt) + '_' + algorithm + '_calibrated' + filename_addition + '.npy', llr_neyman_null)
    #
    #     # Roaming
    #     thetas0_array = np.zeros((n_roaming, 2), dtype=X_roam_transformed.dtype)
    #     thetas0_array[:, :] = settings.thetas[t]
    #     X_thetas_roaming_temp = np.hstack((X_roam_transformed, thetas0_array))
    #     r_roam_temp[t, :], _ = ratio_calibrated.predict(X_thetas_roaming_temp)
    #
    # # Save evaluation results
    # expected_llr_calibrated = np.asarray(expected_llr_calibrated)
    # mse_log_r_calibrated = np.asarray(mse_log_r_calibrated)
    # trimmed_mse_log_r_calibrated = np.asarray(trimmed_mse_log_r_calibrated)
    # expected_r_vs_sm = np.asarray(expected_r_vs_sm)
    # if recalibration_mode:
    #     recalibration_expected_r = np.asarray(recalibration_expected_r)
    # np.save(results_dir + '/llr_' + algorithm + '_calibrated' + filename_addition + '.npy',
    #         expected_llr_calibrated)
    # np.save(results_dir + '/mse_logr_' + algorithm + '_calibrated' + filename_addition + '.npy', mse_log_r_calibrated)
    # np.save(results_dir + '/trimmed_mse_logr_' + algorithm + '_calibrated' + filename_addition + '.npy',
    #         trimmed_mse_log_r_calibrated)
    # np.save(results_dir + '/expected_r_vs_sm_' + algorithm + '_calibrated' + filename_addition + '.npy',
    #         expected_r_vs_sm)
    # if recalibration_mode:
    #     recalibration_expected_r = np.asarray(recalibration_expected_r)
    #     np.save(
    #         results_dir + '/recalibration_expected_r_vs_sm_' + algorithm + '_calibrated' + filename_addition + '.npy',
    #         recalibration_expected_r)
    #
    # # Evaluation times
    # logging.info('Calibrated evaluation timing: median %s s, mean %s s', np.median(eval_times), np.mean(eval_times))
    #
    # logging.info('Interpolating calibrated roaming')
    # gp = GaussianProcessRegressor(normalize_y=True,
    #                               kernel=C(1.0) * Matern(1.0, nu=0.5), n_restarts_optimizer=10)
    # gp.fit(settings.thetas[:], np.log(r_roam_temp))
    # r_roam_calibrated = np.exp(gp.predict(np.c_[xx.ravel(), yy.ravel()])).T
    # np.save(results_dir + '/r_roam_' + algorithm + '_calibrated' + filename_addition + '.npy', r_roam_calibrated)
