#! /usr/bin/env python

################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

from os import sys, path
import argparse
import logging
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split

base_dir = path.abspath(path.join(path.dirname(__file__), '..'))
try:
    from higgs_inference import settings
except ImportError:
    if base_dir in sys.path:
        raise
    sys.path.append(base_dir)
    from higgs_inference import settings
settings.base_dir = base_dir

################################################################################
# What do?
################################################################################

# Set up logging
logging.basicConfig(format='%(asctime)s %(levelname)s    %(message)s', level=logging.DEBUG, datefmt='%d.%m.%Y %H:%M:%S')
logging.info('Welcome! How are you today?')

# Parse arguments
parser = argparse.ArgumentParser(description='Data preprocessing for Higgs inference experiments')

parser.add_argument("-t", "--train", action="store_true",
                    help="Generate baseline training sample")
parser.add_argument("-b", "--basis", action="store_true",
                    help="Generate morphing basis training sample")
parser.add_argument("-r", "--random", action="store_true",
                    help="Generate random theta training sample")
parser.add_argument("-p", "--pointbypoint", action="store_true",
                    help="Generate point-by-point training samples")
parser.add_argument("-s", "--scoreregression", action="store_true",
                    help="Generate score regression training sample")
parser.add_argument("-c", "--calibration", action="store_true",
                    help="Generate calibration sample")
parser.add_argument("--recalibration", action="store_true",
                    help="Generate recalibration sample")
parser.add_argument("-e", "--test", action="store_true",
                    help="Generate likelihood ratio evaluation sample")
parser.add_argument("-n", "--neyman", action="store_true",
                    help="Generate samples for Neyman construction")
parser.add_argument("--neyman2", action="store_true",
                    help="Generate samples for Neyman construction (alternative settings)")
parser.add_argument("--neyman3", action="store_true",
                    help="Generate samples for Neyman construction (alternative settings)")
parser.add_argument("-x", "--roam", action="store_true",
                    help="Generate roaming evaluation sample")
parser.add_argument("--alternativedenom1", action="store_true",
                    help="Use alternative denominator theta 1")
parser.add_argument("--alternativedenom2", action="store_true",
                    help="Use alternative denominator theta 2")
parser.add_argument("--alternativedenom3", action="store_true",
                    help="Use alternative denominator theta 3")
parser.add_argument("--alternativedenom4", action="store_true",
                    help="Use alternative denominator theta 4")
parser.add_argument("--new", action="store_true",
                    help="Generate alternative training set, in which events with NaNs are not removed")
parser.add_argument("--dry", action="store_true",
                    help="Don't save results")

args = parser.parse_args()

logging.info('Tasks:')
logging.info('  Baseline training:                 %s', args.train)
logging.info('  Random training:                   %s', args.random)
logging.info('  Morphing training:                 %s', args.basis)
logging.info('  Point-by-point training:           %s', args.pointbypoint)
logging.info('  Score regression training:         %s', args.scoreregression)
logging.info('  Calibration:                       %s', args.calibration)
logging.info('  Recalibration:                     %s', args.recalibration)
logging.info('  Likelihood ratio eval:             %s', args.test)
logging.info('  Neyman construction:               %s', args.neyman)
logging.info('  Neyman construction (alternative): %s', args.neyman2)
logging.info('  Neyman construction (alternative): %s', args.neyman3)
logging.info('  Roaming:                           %s', args.roam)
logging.info('Options:')
if args.alternativedenom1:
    logging.info('  Denominator:                       alternative 1')
elif args.alternativedenom2:
    logging.info('  Denominator:                       alternative 2')
elif args.alternativedenom3:
    logging.info('  Denominator:                       alternative 3')
elif args.alternativedenom4:
    logging.info('  Denominator:                       alternative 4')
else:
    logging.info('  Denominator:                       standard')
logging.info('  New samples (no NaN removal):      %s', args.new)
logging.info('  Dry run:                           %s', args.dry)

################################################################################
# Settings
################################################################################


data_dir = settings.base_dir + '/data'

thetas = np.load(data_dir + '/thetas/thetas_parameterized.npy')
n_thetas = len(thetas)

filename_addition = ''
theta1 = settings.theta1_default
if args.alternativedenom1:
    theta1 = settings.theta1_alternatives[0]
    filename_addition += '_denom1'
elif args.alternativedenom2:
    theta1 = settings.theta1_alternatives[1]
    filename_addition += '_denom2'
elif args.alternativedenom3:
    theta1 = settings.theta1_alternatives[2]
    filename_addition += '_denom3'
elif args.alternativedenom4:
    theta1 = settings.theta1_alternatives[3]
    filename_addition += '_denom4'

if args.new:
    filename_addition += '_new'

need_train_sample = args.train or args.random or args.basis or args.pointbypoint or args.scoreregression
need_calibration_sample = args.calibration or args.recalibration
need_test_sample = args.test or args.neyman or args.neyman2 or args.neyman3 or args.roam

################################################################################
# Data
################################################################################

subset_features = list(range(settings.n_features))

weighted_data = pd.read_csv(settings.weighted_events_dir + '/wbf_4l_supernew.dat', sep='\t', dtype=np.float32)

logging.info('Splitting...')

# Split: 70% train + evaluation, 10% calibration, 20% test
weighted_data_train, weighted_data_test = train_test_split(weighted_data,
                                                           test_size=0.3,
                                                           random_state=42)
weighted_data_test, weighted_data_calibrate = train_test_split(weighted_data_test,
                                                               test_size=0.3333333,
                                                               random_state=43)

n_events = weighted_data.shape[0]
n_events_train = weighted_data_train.shape[0]
n_events_calibrate = weighted_data_calibrate.shape[0]
n_events_test = weighted_data_test.shape[0]

logging.info('Number of events:')
logging.info('  All:    %s', n_events)
logging.info('  Training:    %s', n_events_train)
logging.info('  Calibration: %s', n_events_calibrate)
logging.info('  Evaluation:  %s', n_events_test)

# Clean up now already
del weighted_data
if not need_train_sample:
    del weighted_data_train
if not need_calibration_sample:
    del weighted_data_calibrate
if not need_test_sample:
    del weighted_data_test

gc.collect()

logging.info('Normalizing probabilities')

weights_train = []
if need_train_sample:
    for n in range(n_thetas):
        temp_weights = np.array(weighted_data_train['p_theta_' + str(n)])
        temp_weights *= 1. / sum(temp_weights)
        weights_train.append(temp_weights)
weights_train = np.asarray(weights_train)

weights_calibrate = []
if need_calibration_sample:
    for n in range(n_thetas):
        temp_weights = np.array(weighted_data_calibrate['p_theta_' + str(n)])
        temp_weights *= 1. / sum(temp_weights)
        weights_calibrate.append(temp_weights)
weights_calibrate = np.asarray(weights_calibrate)

weights_test = []
if need_test_sample:
    for n in range(n_thetas):
        temp_weights = np.array(weighted_data_test['p_theta_' + str(n)])
        temp_weights *= 1. / sum(temp_weights)
        weights_test.append(temp_weights)
weights_test = np.asarray(weights_test)

################################################################################
# Baseline training
################################################################################

if args.train:

    logging.info('Generating baseline training sample')


    def generate_data_train(theta0, theta1):
        indices_num = np.random.choice(list(range(n_events_train)), settings.n_events_baseline_num,
                                       p=weights_train[theta0])
        indices_den = np.random.choice(list(range(n_events_train)), settings.n_events_baseline_den,
                                       p=weights_train[theta1])

        X = np.vstack((
            np.array(weighted_data_train.iloc[indices_num, subset_features]),
            np.array(weighted_data_train.iloc[indices_den, subset_features])
        ))

        y = np.zeros(settings.n_events_baseline_num + settings.n_events_baseline_den)
        y[settings.n_events_baseline_num:] = 1.

        labels_scores = ["score_theta_" + str(theta0) + "_0", "score_theta_" + str(theta0) + "_1"]
        subset_scores = [weighted_data_train.columns.get_loc(x) for x in labels_scores]
        scores = np.vstack((
            np.array(weighted_data_train.iloc[indices_num, subset_scores]),
            np.array(weighted_data_train.iloc[indices_den, subset_scores])
        ))

        thetas0 = np.zeros((settings.n_events_baseline_num + settings.n_events_baseline_den, 2))
        thetas0[:] = thetas[theta0]
        thetas1 = np.zeros((settings.n_events_baseline_num + settings.n_events_baseline_den, 2))
        thetas1[:] = thetas[theta1]

        r = np.hstack((
            np.array(weights_train[theta0][indices_num] / weights_train[theta1][indices_num]),
            np.array(weights_train[theta0][indices_den] / weights_train[theta1][indices_den]),
        ))

        p0 = np.hstack((
            np.array(weights_train[theta0][indices_num]),
            np.array(weights_train[theta0][indices_den]),
        ))

        p1 = np.hstack((
            np.array(weights_train[theta1][indices_num]),
            np.array(weights_train[theta1][indices_den]),
        ))

        # Sanitization
        if args.new:
            r[(~np.isfinite(np.log(r))) & (r < 1.)] = 1. / settings.new_samples_nan_r
            r[~np.isfinite(np.log(r))] = settings.new_samples_nan_r
            scores[~ (np.isfinite(np.log(scores[:, 0]))
                      & np.isfinite(np.log(scores[:, 1]))), :] = settings.new_samples_nan_score

        cut = np.isfinite(np.log(r)) & np.isfinite(scores[:, 0]) & np.isfinite(scores[:, 1])
        return thetas0[cut], thetas1[cut], X[cut], y[cut], scores[cut], r[cut], p0[cut], p1[cut]


    for i, t in enumerate(settings.thetas_train):
        this_th0, this_th1, this_X, this_y, this_scores, this_r, this_p0, this_p1 = generate_data_train(t, theta1)

        logging.debug('Returned shapes: %s %s %s %s %s %s %s %s', this_th0.shape, this_th1.shape, this_X.shape,
                      this_y.shape, this_scores.shape, this_r.shape, this_p0.shape, this_p1.shape)

        if i > 0:
            th0 = np.vstack((th0, np.array(this_th0, dtype=np.float32)))
            th1 = np.vstack((th1, np.array(this_th1, dtype=np.float32)))
            X = np.vstack((X, np.array(this_X, dtype=np.float32)))
            y = np.hstack((y, np.array(this_y, dtype=np.float32)))
            scores = np.vstack((scores, np.array(this_scores, dtype=np.float32)))
            r = np.hstack((r, np.array(this_r, dtype=np.float32)))
            p0 = np.hstack((p0, np.array(this_p0, dtype=np.float32)))
            p1 = np.hstack((p1, np.array(this_p1, dtype=np.float32)))
        else:
            th0 = np.array(this_th0, dtype=np.float32)
            th1 = np.array(this_th1, dtype=np.float32)
            X = np.array(this_X, dtype=np.float32)
            y = np.array(this_y, dtype=np.float32)
            scores = np.array(this_scores, dtype=np.float32)
            r = np.array(this_r, dtype=np.float32)
            p0 = np.array(this_p0, dtype=np.float32)
            p1 = np.array(this_p1, dtype=np.float32)

    logging.debug('Combined shapes: %s %s %s %s %s %s %s %s', th0.shape, th1.shape, X.shape,
                  y.shape, scores.shape, r.shape, p0.shape, p1.shape)

    # Just to make sure
    cut = np.isfinite(np.log(r)) & np.isfinite(scores[:, 0]) & np.isfinite(scores[:, 1])
    th0 = th0[cut]
    th1 = th1[cut]
    X = X[cut]
    y = y[cut]
    scores = scores[cut]
    r = r[cut]
    p0 = p0[cut]
    p1 = p1[cut]

    logging.debug('Combined shapes after filter: %s %s %s %s %s %s %s %s', th0.shape, th1.shape, X.shape,
                  y.shape, scores.shape, r.shape, p0.shape, p1.shape)

    np.save(settings.unweighted_events_dir + '/theta0_train' + filename_addition + '.npy', th0)
    np.save(settings.unweighted_events_dir + '/theta1_train' + filename_addition + '.npy', th1)
    np.save(settings.unweighted_events_dir + '/X_train' + filename_addition + '.npy', X)
    np.save(settings.unweighted_events_dir + '/y_train' + filename_addition + '.npy', y)
    np.save(settings.unweighted_events_dir + '/scores_train' + filename_addition + '.npy', scores)
    np.save(settings.unweighted_events_dir + '/r_train' + filename_addition + '.npy', r)
    np.save(settings.unweighted_events_dir + '/p0_train' + filename_addition + '.npy', p0)
    np.save(settings.unweighted_events_dir + '/p1_train' + filename_addition + '.npy', p1)

################################################################################
# Basis training
################################################################################

if args.basis:

    logging.info('Generating morphing basis training sample')


    def generate_data_train_basis(theta0, theta1):
        indices_num = np.random.choice(list(range(n_events_train)), settings.n_events_basis_num,
                                       p=weights_train[theta0])
        indices_den = np.random.choice(list(range(n_events_train)), settings.n_events_basis_den,
                                       p=weights_train[theta1])

        X = np.vstack((
            np.array(weighted_data_train.iloc[indices_num, subset_features]),
            np.array(weighted_data_train.iloc[indices_den, subset_features])
        ))

        y = np.zeros(settings.n_events_basis_num + settings.n_events_basis_den)
        y[settings.n_events_basis_num:] = 1.

        labels_scores = ["score_theta_" + str(theta0) + "_0", "score_theta_" + str(theta0) + "_1"]
        subset_scores = [weighted_data_train.columns.get_loc(x) for x in labels_scores]
        scores = np.vstack((
            np.array(weighted_data_train.iloc[indices_num, subset_scores]),
            np.array(weighted_data_train.iloc[indices_den, subset_scores])
        ))

        thetas0 = np.zeros((settings.n_events_basis_num + settings.n_events_basis_den, 2))
        thetas0[:] = thetas[theta0]
        thetas1 = np.zeros((settings.n_events_basis_num + settings.n_events_basis_den, 2))
        thetas1[:] = thetas[theta1]

        r = np.hstack((
            np.array(weights_train[theta0][indices_num] / weights_train[theta1][indices_num]),
            np.array(weights_train[theta0][indices_den] / weights_train[theta1][indices_den]),
        ))

        p0 = np.hstack((
            np.array(weights_train[theta0][indices_num]),
            np.array(weights_train[theta0][indices_den]),
        ))

        p1 = np.hstack((
            np.array(weights_train[theta1][indices_num]),
            np.array(weights_train[theta1][indices_den]),
        ))

        # Sanitization
        if args.new:
            r[(~np.isfinite(np.log(r))) & (r < 1.)] = 1. / settings.new_samples_nan_r
            r[~np.isfinite(np.log(r))] = settings.new_samples_nan_r
            scores[~ (np.isfinite(np.log(scores[:, 0]))
                      & np.isfinite(np.log(scores[:, 1]))), :] = settings.new_samples_nan_score

        cut = np.isfinite(np.log(r)) & np.isfinite(scores[:, 0]) & np.isfinite(scores[:, 1])

        return thetas0[cut], thetas1[cut], X[cut], y[cut], scores[cut], r[cut], p0[cut], p1[
            cut]


    for i, t in enumerate(settings.thetas_morphing_basis):
        this_th0, this_th1, this_X, this_y, this_scores, this_r, this_p0, this_p1 = generate_data_train_basis(t, theta1)

        if i > 0:
            th0 = np.vstack((th0, np.array(this_th0, dtype=np.float32)))
            th1 = np.vstack((th1, np.array(this_th1, dtype=np.float32)))
            X = np.vstack((X, np.array(this_X, dtype=np.float32)))
            y = np.hstack((y, np.array(this_y, dtype=np.float32)))
            scores = np.vstack((scores, np.array(this_scores, dtype=np.float32)))
            r = np.hstack((r, np.array(this_r, dtype=np.float32)))
            p0 = np.hstack((p0, np.array(this_p0, dtype=np.float32)))
            p1 = np.hstack((p1, np.array(this_p1, dtype=np.float32)))
        else:
            th0 = np.array(this_th0, dtype=np.float32)
            th1 = np.array(this_th1, dtype=np.float32)
            X = np.array(this_X, dtype=np.float32)
            y = np.array(this_y, dtype=np.float32)
            scores = np.array(this_scores, dtype=np.float32)
            r = np.array(this_r, dtype=np.float32)
            p0 = np.array(this_p0, dtype=np.float32)
            p1 = np.array(this_p1, dtype=np.float32)

    # Just to make sure
    cut = np.isfinite(np.log(r)) & np.isfinite(scores[:, 0]) & np.isfinite(scores[:, 1])
    th0 = th0[cut]
    th1 = th1[cut]
    X = X[cut]
    y = y[cut]
    scores = scores[cut]
    r = r[cut]
    p0 = p0[cut]
    p1 = p1[cut]

    np.save(settings.unweighted_events_dir + '/theta0_train_basis' + filename_addition + '.npy', th0)
    np.save(settings.unweighted_events_dir + '/theta1_train_basis' + filename_addition + '.npy', th1)
    np.save(settings.unweighted_events_dir + '/X_train_basis' + filename_addition + '.npy', X)
    np.save(settings.unweighted_events_dir + '/y_train_basis' + filename_addition + '.npy', y)
    np.save(settings.unweighted_events_dir + '/scores_train_basis' + filename_addition + '.npy', scores)
    np.save(settings.unweighted_events_dir + '/r_train_basis' + filename_addition + '.npy', r)
    np.save(settings.unweighted_events_dir + '/p0_train_basis' + filename_addition + '.npy', p0)
    np.save(settings.unweighted_events_dir + '/p1_train_basis' + filename_addition + '.npy', p1)

################################################################################
# Point-by-point training
################################################################################

if args.pointbypoint:

    logging.info('Generating point-by-point training samples')


    def generate_data_train_point_by_point(theta0, theta1):
        indices_num = np.random.choice(list(range(n_events_train)), settings.n_events_n_point_by_point_num,
                                       p=weights_train[theta0])
        indices_den = np.random.choice(list(range(n_events_train)), settings.n_events_n_point_by_point_den,
                                       p=weights_train[theta1])

        X = np.vstack((
            np.array(weighted_data_train.iloc[indices_num, subset_features]),
            np.array(weighted_data_train.iloc[indices_den, subset_features])
        ))

        y = np.zeros(settings.n_events_n_point_by_point_num + settings.n_events_n_point_by_point_den)
        y[settings.n_events_n_point_by_point_num:] = 1.

        labels_scores = ["score_theta_" + str(theta0) + "_0", "score_theta_" + str(theta0) + "_1"]
        subset_scores = [weighted_data_train.columns.get_loc(x) for x in labels_scores]
        scores = np.vstack((
            np.array(weighted_data_train.iloc[indices_num, subset_scores]),
            np.array(weighted_data_train.iloc[indices_den, subset_scores])
        ))

        thetas0 = np.zeros((settings.n_events_n_point_by_point_num + settings.n_events_n_point_by_point_den, 2))
        thetas0[:] = thetas[theta0]
        thetas1 = np.zeros((settings.n_events_n_point_by_point_num + settings.n_events_n_point_by_point_den, 2))
        thetas1[:] = thetas[theta1]

        r = np.hstack((
            np.array(weights_train[theta0][indices_num] / weights_train[theta1][indices_num]),
            np.array(weights_train[theta0][indices_den] / weights_train[theta1][indices_den]),
        ))

        p0 = np.hstack((
            np.array(weights_train[theta0][indices_num]),
            np.array(weights_train[theta0][indices_den]),
        ))

        p1 = np.hstack((
            np.array(weights_train[theta1][indices_num]),
            np.array(weights_train[theta1][indices_den]),
        ))

        # Sanitization
        if args.new:
            r[(~np.isfinite(np.log(r))) & (r < 1.)] = 1. / settings.new_samples_nan_r
            r[~np.isfinite(np.log(r))] = settings.new_samples_nan_r
            scores[~ (np.isfinite(np.log(scores[:, 0]))
                      & np.isfinite(np.log(scores[:, 1]))), :] = settings.new_samples_nan_score

        cut = np.isfinite(np.log(r)) & np.isfinite(scores[:, 0]) & np.isfinite(scores[:, 1])

        return thetas0[cut], thetas1[cut], X[cut], y[cut], scores[cut], r[cut], p0[cut], p1[
            cut]


    for t in range(125, settings.n_thetas):  # enumerate(settings.extended_pbp_training_thetas):
        this_th0, this_th1, this_X, this_y, this_scores, this_r, this_p0, this_p1 = generate_data_train_point_by_point(
            t, theta1)

        np.save(settings.unweighted_events_dir + '/point_by_point/X_train_point_by_point_' + str(
            t) + filename_addition + '.npy', this_X)
        np.save(settings.unweighted_events_dir + '/point_by_point/y_train_point_by_point_' + str(
            t) + filename_addition + '.npy', this_y)
        np.save(settings.unweighted_events_dir + '/point_by_point/r_train_point_by_point_' + str(
            t) + filename_addition + '.npy', this_r)
        np.save(settings.unweighted_events_dir + '/point_by_point/p0_train_point_by_point_' + str(
            t) + filename_addition + '.npy', this_p0)
        np.save(settings.unweighted_events_dir + '/point_by_point/p1_train_point_by_point_' + str(
            t) + filename_addition + '.npy', this_p1)

        del this_th0, this_th1, this_X, this_y, this_scores, this_r, this_p0, this_p1

################################################################################
# Random training
################################################################################

if args.random:

    logging.info('Generating random theta training sample')


    def generate_random_data_train(randomtheta0, theta1):
        prob_num = (
                float(settings.n_events_randomtheta_num * n_events) / float(settings.n_randomthetas * n_events_train)
                * weighted_data_train['p_randomtheta_' + str(randomtheta0)])
        prob_den = (float(settings.n_events_randomtheta_den) / float(settings.n_randomthetas)
                    * weights_train[theta1])

        n_dice = int(max(max(prob_num), max(prob_den)) + 1.)
        assert n_dice < 200
        prob_num /= float(n_dice)
        prob_den /= float(n_dice)

        accepted_num = weighted_data_train[prob_num > np.random.rand(n_events_train)]
        accepted_den = weighted_data_train[prob_den > np.random.rand(n_events_train)]
        for i in range(n_dice - 1):
            accepted_num = pd.concat([accepted_num, weighted_data_train[prob_num > np.random.rand(n_events_train)]])
            accepted_den = pd.concat([accepted_den, weighted_data_train[prob_den > np.random.rand(n_events_train)]])

        X = np.vstack((accepted_num.iloc[:, subset_features],
                       accepted_den.iloc[:, subset_features]))

        y = np.zeros(len(X))
        y[len(accepted_num):] = 1.

        r = np.hstack((np.array(accepted_num['p_randomtheta_' + str(randomtheta0)]
                                / accepted_num['p_theta_' + str(theta1)]),
                       np.array(accepted_den['p_randomtheta_' + str(randomtheta0)]
                                / accepted_den['p_theta_' + str(theta1)])))

        subset_scores = [weighted_data_train.columns.get_loc(x)
                         for x in ['score_randomtheta_' + str(randomtheta0) + '_0',
                                   'score_randomtheta_' + str(randomtheta0) + '_1']]
        scores = np.vstack((
            np.array(accepted_num.iloc[:, subset_scores]),
            np.array(accepted_den.iloc[:, subset_scores])
        ))

        subset_randomthetas = [weighted_data_train.columns.get_loc(x)
                               for x in
                               ['randomtheta_' + str(randomtheta0) + '_0', 'randomtheta_' + str(randomtheta0) + '_1']]
        thetas0 = np.vstack((
            np.array(accepted_num.iloc[:, subset_randomthetas]),
            np.array(accepted_den.iloc[:, subset_randomthetas])
        ))
        thetas1 = np.zeros((len(X), 2))
        thetas1[:] = thetas[theta1]

        # Sanitization
        if args.new:
            r[(~np.isfinite(np.log(r))) & (r < 1.)] = 1. / settings.new_samples_nan_r
            r[~np.isfinite(np.log(r))] = settings.new_samples_nan_r
            scores[~ (np.isfinite(np.log(scores[:, 0]))
                      & np.isfinite(np.log(scores[:, 1]))), :] = settings.new_samples_nan_score

        cut = np.isfinite(np.log(r)) & np.isfinite(scores[:, 0]) & np.isfinite(scores[:, 1])

        return thetas0[cut], thetas1[cut], X[cut], y[cut], scores[cut], r[cut]


    for t in range(settings.n_randomthetas):
        this_th0, this_th1, this_X, this_y, this_scores, this_r = generate_random_data_train(t, theta1)

        if t > 0:
            th0 = np.vstack((th0, np.array(this_th0, dtype=np.float32)))
            th1 = np.vstack((th1, np.array(this_th1, dtype=np.float32)))
            X = np.vstack((X, np.array(this_X, dtype=np.float32)))
            y = np.hstack((y, np.array(this_y, dtype=np.float32)))
            scores = np.vstack((scores, np.array(this_scores, dtype=np.float32)))
            r = np.hstack((r, np.array(this_r, dtype=np.float32)))
        else:
            th0 = np.array(this_th0, dtype=np.float32)
            th1 = np.array(this_th1, dtype=np.float32)
            X = np.array(this_X, dtype=np.float32)
            y = np.array(this_y, dtype=np.float32)
            scores = np.array(this_scores, dtype=np.float32)
            r = np.array(this_r, dtype=np.float32)

    # Just to make sure
    cut = np.isfinite(np.log(r)) & np.isfinite(scores[:, 0]) & np.isfinite(scores[:, 1])
    th0 = th0[cut]
    th1 = th1[cut]
    X = X[cut]
    y = y[cut]
    scores = scores[cut]
    r = r[cut]

    np.save(settings.unweighted_events_dir + '/theta0_train_random' + filename_addition + '.npy', th0)
    np.save(settings.unweighted_events_dir + '/theta1_train_random' + filename_addition + '.npy', th1)
    np.save(settings.unweighted_events_dir + '/X_train_random' + filename_addition + '.npy', X)
    np.save(settings.unweighted_events_dir + '/y_train_random' + filename_addition + '.npy', y)
    np.save(settings.unweighted_events_dir + '/scores_train_random' + filename_addition + '.npy', scores)
    np.save(settings.unweighted_events_dir + '/r_train_random' + filename_addition + '.npy', r)

################################################################################
# Calibration
################################################################################

if args.calibration:

    logging.info('Generating calibration sample')


    def generate_data_calibration(theta_observed):
        indices = np.random.choice(list(range(n_events_calibrate)), settings.n_events_calibration,
                                   p=weights_calibrate[theta_observed])

        X = np.asarray(weighted_data_calibrate.iloc[indices, subset_features])

        r = np.zeros((n_thetas, settings.n_events_calibration))
        for t in range(n_thetas):
            r[t, :] = np.array(weights_calibrate[t][indices] / weights_calibrate[theta_observed][indices])

        # Sanitization
        if args.new:
            r = r.flatten()
            r[(~np.isfinite(np.log(r))) & (r < 1.)] = 1. / settings.new_samples_nan_r
            r[~np.isfinite(np.log(r))] = settings.new_samples_nan_r
            r = r.reshape((n_thetas, settings.n_events_calibration))

        cut = np.all(np.isfinite(np.log(r)), axis=0)

        return X[cut], r[:, cut]


    for i, t in enumerate(settings.thetas_train):
        this_X, this_weights = generate_data_calibration(t)

        if i > 0:
            X = np.vstack((X, np.array(this_X, dtype=np.float32)))
            weights = np.hstack((weights, np.array(this_weights, dtype=np.float32)))
        else:
            X = np.array(this_X, dtype=np.float32)
            weights = np.array(this_weights, dtype=np.float32)

    np.save(settings.unweighted_events_dir + '/X_calibration' + filename_addition + '.npy', X)
    np.save(settings.unweighted_events_dir + '/weights_calibration' + filename_addition + '.npy', weights)

################################################################################
# Recalibration
################################################################################

if args.recalibration:

    logging.info('Generating recalibration sample')


    def generate_data_recalibration(theta_observed):
        indices = np.random.choice(list(range(n_events_calibrate)), settings.n_events_recalibration,
                                   p=weights_calibrate[theta_observed])

        X = np.asarray(weighted_data_calibrate.iloc[indices, subset_features])

        r = np.zeros((n_thetas, settings.n_events_recalibration))
        for t in range(n_thetas):
            r[t, :] = np.array(weights_calibrate[t][indices] / weights_calibrate[theta_observed][indices])

        # Sanitization
        if args.new:
            r = r.flatten()
            r[(~np.isfinite(np.log(r))) & (r < 1.)] = 1. / settings.new_samples_nan_r
            r[~np.isfinite(np.log(r))] = settings.new_samples_nan_r
            r = r.reshape((n_thetas, settings.n_events_calibration))

        cut = np.all(np.isfinite(np.log(r)), axis=0)

        return X[cut], r[:, cut]


    X, weights = generate_data_recalibration(settings.theta_observed)

    np.save(settings.unweighted_events_dir + '/X_recalibration' + filename_addition + '.npy', X)
    np.save(settings.unweighted_events_dir + '/weights_recalibration' + filename_addition + '.npy', weights)

################################################################################
# Training sample for score regression
################################################################################

if args.scoreregression:
    logging.info('Generating training sample for score regression')


    def generate_data_score_regression(theta):
        indices = np.random.choice(list(range(n_events_train)), settings.n_events_score_regression,
                                   p=weights_train[theta])

        X = np.array(weighted_data_train.iloc[indices, subset_features])

        labels_scores = ["score_theta_" + str(theta) + "_0", "score_theta_" + str(theta) + "_1"]
        subset_scores = [weighted_data_train.columns.get_loc(x) for x in labels_scores]
        scores = np.array(weighted_data_train.iloc[indices, subset_scores])

        p = np.array(weights_train[theta][indices])

        # Sanitization
        if args.new:
            scores[~ (np.isfinite(np.log(scores[:, 0]))
                      & np.isfinite(np.log(scores[:, 1]))), :] = settings.new_samples_nan_score

        cut = np.isfinite(scores[:, 0]) & np.isfinite(scores[:, 1])

        return X[cut], scores[cut], p[cut]


    X, scores, p = generate_data_score_regression(settings.theta_score_regression)

    np.save(settings.unweighted_events_dir + '/X_train_scoreregression' + filename_addition + '.npy', X)
    np.save(settings.unweighted_events_dir + '/scores_train_scoreregression' + filename_addition + '.npy', scores)
    np.save(settings.unweighted_events_dir + '/p_train_scoreregression' + filename_addition + '.npy', p)

################################################################################
# Likelihood ratio evaluation
################################################################################

if args.test:

    logging.info('Generating likelihood ratio evaluation sample')


    def generate_data_test(theta_observed, theta1):
        indices = np.random.choice(list(range(n_events_test)), settings.n_events_test,
                                   p=weights_test[theta_observed])

        X = np.asarray(weighted_data_test.iloc[indices, subset_features])

        r = np.zeros((n_thetas, settings.n_events_test))
        for t in range(n_thetas):
            r[t, :] = np.array(weights_test[t][indices] / weights_test[theta1][indices])

        scores = np.zeros((n_thetas, settings.n_events_test, 2))
        for t in range(n_thetas):
            labels_scores = ["score_theta_" + str(t) + "_0", "score_theta_" + str(t) + "_1"]
            subset_scores = [weighted_data_test.columns.get_loc(x) for x in labels_scores]
            scores[t] = np.array(weighted_data_test.iloc[indices, subset_scores])

        p1 = np.array(weights_test[theta1][indices])

        # Sanitization
        if args.new:
            r = r.flatten()
            r[(~np.isfinite(np.log(r))) & (r < 1.)] = 1. / settings.new_samples_nan_r
            r[~np.isfinite(np.log(r))] = settings.new_samples_nan_r
            r = r.reshape((n_thetas, settings.n_events_test))

            scores = scores.reshape((-1, 2))
            scores[~ (np.isfinite(np.log(scores[:, 0]))
                      & np.isfinite(np.log(scores[:, 1]))), :] = settings.new_samples_nan_score
            scores = scores.reshape((n_thetas, settings.n_events_test, 2))

        cut = np.all(np.isfinite(np.log(r[:, :])) & np.isfinite(scores[:, :, 0]) & np.isfinite(scores[:, :, 1]),
                     axis=0)

        return X[cut], scores[:, cut, :], r[:, cut], p1[cut]


    X, scores, r, p1 = generate_data_test(settings.theta_observed, theta1)

    if not args.dry:
        np.save(settings.unweighted_events_dir + '/X_test' + filename_addition + '.npy', X)
        np.save(settings.unweighted_events_dir + '/scores_test' + filename_addition + '.npy', scores)
        np.save(settings.unweighted_events_dir + '/r_test' + filename_addition + '.npy', r)
        np.save(settings.unweighted_events_dir + '/p1_test' + filename_addition + '.npy', p1)

################################################################################
# Neyman construction
################################################################################

if args.neyman:
    logging.info('Generating Neyman construction samples')


    def generate_data_neyman(theta_observed, n_toy_experiments, theta1, theta_score, thetas_r=None):
        gc.collect()

        if thetas_r is None:
            thetas_r = list(range(settings.n_thetas))

        indices = np.random.choice(list(range(n_events_test)), n_toy_experiments * settings.n_expected_events_neyman,
                                   p=weights_test[theta_observed])

        # Check how many repeated entries we have
        unique, counts = np.unique(indices.flatten(), return_counts=True)
        counts = - np.sort(- counts)
        logging.debug('Repeated events: %s', counts[:5])

        X = np.asarray(weighted_data_test.iloc[indices, subset_features])

        r = np.zeros((len(thetas_r), n_toy_experiments * settings.n_expected_events_neyman))
        for i, t in enumerate(thetas_r):
            r[i, :] = np.array(weights_test[t][indices] / weights_test[theta1][indices])

        # Scores for score regression / local model
        labels_scores = ["score_theta_" + str(theta_score) + "_0", "score_theta_" + str(theta_score) + "_1"]
        subset_scores = [weighted_data_test.columns.get_loc(x) for x in labels_scores]
        scores = np.array(weighted_data_test.iloc[indices, subset_scores])

        # Sanitization
        if args.new:
            r = r.flatten()
            r[(~np.isfinite(np.log(r))) & (r < 1.)] = 1. / settings.new_samples_nan_r
            r[~np.isfinite(np.log(r))] = settings.new_samples_nan_r
            r = r.reshape((len(thetas_r), n_toy_experiments * settings.n_expected_events_neyman))

            scores[~ (np.isfinite(np.log(scores[:, 0]))
                      & np.isfinite(np.log(scores[:, 1]))), :] = settings.new_samples_nan_score

        # Reshape to experiments x expected events
        X = X.reshape((n_toy_experiments, settings.n_expected_events_neyman, -1))
        r = r.reshape((len(thetas_r), n_toy_experiments, settings.n_expected_events_neyman))
        scores = scores.reshape((n_toy_experiments, settings.n_expected_events_neyman, 2))

        return X, r, scores


    # Observed
    X, r, scores = generate_data_neyman(settings.theta_observed, settings.n_neyman_alternate_experiments, theta1,
                                        settings.theta_score_regression)

    logging.info('Generated %s toy experiments with %s events each for the alternate according to theta = %s',
                 X.shape[0], X.shape[1], thetas[settings.theta_observed])

    if not args.dry:
        np.save(settings.unweighted_events_dir + '/neyman/X_neyman_alternate' + filename_addition + '.npy', X)
        np.save(settings.unweighted_events_dir + '/neyman/r_neyman_alternate' + filename_addition + '.npy', r)
        np.save(settings.unweighted_events_dir + '/neyman/scores_neyman_alternate' + filename_addition + '.npy', scores)

    del X, r, scores

    # Distribution
    for t, theta in enumerate(thetas):
        X, r, scores = generate_data_neyman(t, settings.n_neyman_null_experiments, theta1,
                                            settings.theta_score_regression,
                                            thetas_r=[settings.theta_observed, t])

        logging.info('Generated %s toy experiments with %s events each for the null according to theta = %s',
                     X.shape[0], X.shape[1], theta)

        if not args.dry:
            np.save(settings.unweighted_events_dir + '/neyman/X_neyman_null_' + str(t) + filename_addition + '.npy', X)
            np.save(settings.unweighted_events_dir + '/neyman/r_neyman_null_' + str(t) + filename_addition + '.npy', r)
            np.save(
                settings.unweighted_events_dir + '/neyman/scores_neyman_null_' + str(t) + filename_addition + '.npy',
                scores)

        del X, r, scores

################################################################################
# Neyman construction, alternative version
################################################################################

if args.neyman2:
    logging.info('Generating Neyman2 samples')


    def generate_data_neyman2(theta_observed, n_toy_experiments, theta1, theta_score, thetas_r=None):
        gc.collect()

        if thetas_r is None:
            thetas_r = list(range(settings.n_thetas))

        indices = np.random.choice(list(range(n_events_test)), n_toy_experiments * settings.n_expected_events_neyman2,
                                   p=weights_test[theta_observed])

        # Check how many repeated entries we have
        unique, counts = np.unique(indices.flatten(), return_counts=True)
        counts = - np.sort(- counts)
        logging.debug('Repeated events: %s', counts[:5])

        X = np.asarray(weighted_data_test.iloc[indices, subset_features])

        r = np.zeros((len(thetas_r), n_toy_experiments * settings.n_expected_events_neyman2))
        for i, t in enumerate(thetas_r):
            r[i, :] = np.array(weights_test[t][indices] / weights_test[theta1][indices])

        # Scores for score regression / local model
        labels_scores = ["score_theta_" + str(theta_score) + "_0", "score_theta_" + str(theta_score) + "_1"]
        subset_scores = [weighted_data_test.columns.get_loc(x) for x in labels_scores]
        scores = np.array(weighted_data_test.iloc[indices, subset_scores])

        # Sanitization
        if args.new:
            r = r.flatten()
            r[(~np.isfinite(np.log(r))) & (r < 1.)] = 1. / settings.new_samples_nan_r
            r[~np.isfinite(np.log(r))] = settings.new_samples_nan_r
            r = r.reshape((len(thetas_r), settings.n_expected_events_neyman2))

            scores[~ (np.isfinite(np.log(scores[:, 0]))
                      & np.isfinite(np.log(scores[:, 1]))), :] = settings.new_samples_nan_score

        # Reshape to experiments x expected events
        X = X.reshape((n_toy_experiments, settings.n_expected_events_neyman2, -1))
        r = r.reshape((len(thetas_r), n_toy_experiments, settings.n_expected_events_neyman2))
        scores = scores.reshape((n_toy_experiments, settings.n_expected_events_neyman2, 2))

        return X, r, scores


    # Observed
    X, r, scores = generate_data_neyman2(settings.theta_observed, settings.n_neyman2_alternate_experiments, theta1,
                                         settings.theta_score_regression)

    logging.info('Generated %s toy experiments with %s events each for the alternate according to theta = %s',
                 X.shape[0], X.shape[1], thetas[settings.theta_observed])

    if not args.dry:
        np.save(settings.unweighted_events_dir + '/neyman/X_neyman2_alternate' + filename_addition + '.npy', X)
        np.save(settings.unweighted_events_dir + '/neyman/r_neyman2_alternate' + filename_addition + '.npy', r)
        np.save(settings.unweighted_events_dir + '/neyman/scores_neyman2_alternate' + filename_addition + '.npy',
                scores)

    del X, r, scores

    # Distribution
    for t, theta in enumerate(thetas):
        X, r, scores = generate_data_neyman2(t, settings.n_neyman2_null_experiments, theta1,
                                             settings.theta_score_regression,
                                             thetas_r=[settings.theta_observed, t])

        logging.info('Generated %s toy experiments with %s events each for the null according to theta = %s',
                     X.shape[0], X.shape[1], theta)

        if not args.dry:
            np.save(settings.unweighted_events_dir + '/neyman/X_neyman2_null_' + str(t) + filename_addition + '.npy', X)
            np.save(settings.unweighted_events_dir + '/neyman/r_neyman2_null_' + str(t) + filename_addition + '.npy', r)
            np.save(
                settings.unweighted_events_dir + '/neyman/scores_neyman2_null_' + str(t) + filename_addition + '.npy',
                scores)

        del X, r, scores

################################################################################
# Neyman construction, alternative version
################################################################################

if args.neyman3:
    logging.info('Generating Neyman3 samples')


    def generate_data_neyman3(theta_observed, n_toy_experiments, theta1, theta_score, thetas_r=None):
        gc.collect()

        if thetas_r is None:
            thetas_r = list(range(settings.n_thetas))

        indices = np.random.choice(list(range(n_events_test)), n_toy_experiments * settings.n_expected_events_neyman3,
                                   p=weights_test[theta_observed],
                                   replace=False)

        # Check how many repeated entries we have
        unique, counts = np.unique(indices.flatten(), return_counts=True)
        counts = - np.sort(- counts)
        logging.debug('Repeated events: %s', counts[:5])

        X = np.asarray(weighted_data_test.iloc[indices, subset_features])

        r = np.zeros((len(thetas_r), n_toy_experiments * settings.n_expected_events_neyman3))
        for i, t in enumerate(thetas_r):
            r[i, :] = np.array(weights_test[t][indices] / weights_test[theta1][indices])

        # Scores for score regression / local model
        labels_scores = ["score_theta_" + str(theta_score) + "_0", "score_theta_" + str(theta_score) + "_1"]
        subset_scores = [weighted_data_test.columns.get_loc(x) for x in labels_scores]
        scores = np.array(weighted_data_test.iloc[indices, subset_scores])

        # Sanitization
        if args.new:
            r = r.flatten()
            r[(~np.isfinite(np.log(r))) & (r < 1.)] = 1. / settings.new_samples_nan_r
            r[~np.isfinite(np.log(r))] = settings.new_samples_nan_r
            r = r.reshape((len(thetas_r), settings.n_expected_events_neyman3))

            scores[~ (np.isfinite(np.log(scores[:, 0]))
                      & np.isfinite(np.log(scores[:, 1]))), :] = settings.new_samples_nan_score

        # Reshape to experiments x expected events
        X = X.reshape((n_toy_experiments, settings.n_expected_events_neyman3, -1))
        r = r.reshape((len(thetas_r), n_toy_experiments, settings.n_expected_events_neyman3))
        scores = scores.reshape((n_toy_experiments, settings.n_expected_events_neyman3, 2))

        return X, r, scores


    # Observed
    X, r, scores = generate_data_neyman3(settings.theta_observed, settings.n_neyman3_alternate_experiments, theta1,
                                         settings.theta_score_regression)

    logging.info('Generated %s toy experiments with %s events each for the alternate according to theta = %s',
                 X.shape[0], X.shape[1], thetas[settings.theta_observed])

    if not args.dry:
        np.save(settings.unweighted_events_dir + '/neyman/X_neyman3_alternate' + filename_addition + '.npy', X)
        np.save(settings.unweighted_events_dir + '/neyman/r_neyman3_alternate' + filename_addition + '.npy', r)
        np.save(settings.unweighted_events_dir + '/neyman/scores_neyman3_alternate' + filename_addition + '.npy',
                scores)

    del X, r, scores

    # Distribution
    for t, theta in enumerate(thetas):
        X, r, scores = generate_data_neyman3(t, settings.n_neyman3_null_experiments, theta1,
                                             settings.theta_score_regression,
                                             thetas_r=[settings.theta_observed, t])

        logging.info('Generated %s toy experiments with %s events each for the null according to theta = %s',
                     X.shape[0], X.shape[1], theta)

        if not args.dry:
            np.save(settings.unweighted_events_dir + '/neyman/X_neyman3_null_' + str(t) + filename_addition + '.npy', X)
            np.save(settings.unweighted_events_dir + '/neyman/r_neyman3_null_' + str(t) + filename_addition + '.npy', r)
            np.save(
                settings.unweighted_events_dir + '/neyman/scores_neyman3_null_' + str(t) + filename_addition + '.npy',
                scores)

        del X, r, scores

################################################################################
# Roam data
################################################################################

if args.roam:

    logging.info('Generating roaming sample')


    def generate_data_roam(theta_observed, theta1):
        indices = np.random.choice(list(range(n_events_test)), settings.n_events_roam,
                                   p=weights_test[theta_observed])

        X = np.asarray(weighted_data_test.iloc[indices, subset_features])

        r = np.zeros((n_thetas, settings.n_events_roam))
        for t in range(n_thetas):
            r[t, :] = np.array(weights_test[t][indices] / weights_test[theta1][indices])

        return X, r


    X, r = generate_data_roam(settings.theta_observed, theta1)

    np.save(settings.unweighted_events_dir + '/X_roam' + filename_addition + '.npy', X)
    np.save(settings.unweighted_events_dir + '/r_roam' + filename_addition + '.npy', r)

logging.info('Done!')
