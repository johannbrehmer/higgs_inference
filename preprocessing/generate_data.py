#! /usr/bin/env python

################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

################################################################################
# What do?
################################################################################

# Set up logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt='%d.%m.%Y %H:%M:%S')
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
parser.add_argument("-e", "--test", action="store_true",
                    help="Generate likelihood ratio evaluation sample")
parser.add_argument("-n", "--neyman", action="store_true",
                    help="Generate samples for Neyman construction")
parser.add_argument("-x", "--roam", action="store_true",
                    help="Generate roaming evaluation sample")
parser.add_argument("--alternativedenom", action="store_true",
                    help="Use alternative denominator theta")
parser.add_argument("--debug", action="store_true",
                    help="Debug mode")

args = parser.parse_args()

args.train = ('train' in args)
args.basis = ('basis' in args)
args.random = ('random' in args)
args.pointbypoint = ('point-by-point' in args)
args.scoreregression = ('score-regression' in args)

args.calibration = ('calibration' in args)
args.test = ('test' in args)
args.neyman = ('neyman' in args)
args.roam = ('roam' in args)

args.alternativedenom = ('denom1' in args)
args.debug = ('debug' in args)

logging.info('Tasks:')
logging.info('  Baseline training:       %s', args.train)
logging.info('  Random training:         %s', args.random)
logging.info('  Morphing training:       %s', args.basis)
logging.info('  Point-by-point training: %s', args.pointbypoint)
logging.info('  Calibration:             %s', args.calibration)
logging.info('  Likelihood ratio eval:   %s', args.test)
logging.info('  Neyman construction:     %s', args.neyman)
logging.info('  Roaming:                 %s', args.roam)
logging.info('Options:')
if args.alternativedenom:
    logging.info('  Denominator: standard')
else:
    logging.info('  Denominator: alternative')
logging.info('  Debug mode:  %s', args.debug)

filename_addition = ''
if args.alternativedenom:
    filename_addition = '_denom1'
if args.debug:
    filename_addition += '_debug'

weighted_events_dir = '/scratch/jb6504/higgs_inference/data/events'
unweighted_events_dir = '/scratch/jb6504/higgs_inference/data/unweighted_events'
data_dir = '../data'

################################################################################
# Thetas
################################################################################

thetas = np.load(data_dir + '/thetas/thetas_parameterized.npy')

n_thetas = len(thetas)
n_randomthetas = 100
theta1 = 708
if args.alternativedenom:
    theta1 = 422
theta_test = 213
theta_observed = 0
theta_score = 0  # for local model
theta_score_regression = 0
thetas_train = list(range(17, 1017))
thetas_basis = [0, 101, 106, 902, 910,
                226, 373, 583, 747, 841,
                599, 709, 422, 367, 167]
thetas_point_by_point = [0, 13, 14, 15, 16, 9, 422, 956, 666, 802, 675, 839, 699, 820, 203, 291, 634, 371, 973, 742,
                         901, 181, 82, 937, 510, 919, 745, 588, 804, 963, 396, 62, 401, 925, 874, 770, 108, 179, 669,
                         758, 113, 587, 600, 975, 496, 66, 467, 412, 701, 986, 598, 810, 97, 18, 723, 159, 320, 301,
                         352, 159, 89, 421, 574, 923, 849, 299, 119, 167, 939, 402, 52, 787, 978, 41, 873, 533, 827,
                         304, 294, 760, 890, 539, 1000, 291, 740, 276, 679, 167, 125, 429, 149, 430, 720, 123, 908,
                         256, 777, 809, 269, 851]

thetas_test = list(range(17))

################################################################################
# Sample sizes
################################################################################

# Baseline training, per theta
n_num = 5000
n_den = 5000

# Basis training, per basis theta
n_basis_num = 333333
n_basis_den = 333333

# Random theta training, total
n_randomtheta_num = 5000000
n_randomtheta_den = 5000000

# PbP training, per PbP theta
n_point_by_point_num = 500000
n_point_by_point_den = 500000

# Score regression training, total
n_score_regression = 10000000

# Calibration, per theta
n_calibrate = 1000

# Neyman construction
n_expected_events = 36
n_neyman_distribution_experiments = 1000
n_neyman_observed_experiments = 101

# Evaluation, total
n_test = 50000

# Roaming, total
n_roam = 20

if args.debug:
    n_num = 500
    n_den = 500
    n_basis_num = 33333
    n_basis_den = 33333
    n_randomtheta_num = 500000
    n_randomtheta_den = 500000
    n_point_by_point_num = 5000
    n_point_by_point_den = 5000
    n_calibrate = 100
    n_neyman_distribution_experiments = 100
    n_neyman_observed_experiments = 11
    n_test = 5000
    n_roam = 2

################################################################################
# Data
################################################################################

subset_features = list(range(42))  # list(range(15))

if args.debug:
    weighted_data = pd.read_csv(weighted_events_dir + '/wbf_4l_supernew_excerpt.dat', sep='\t', dtype=np.float32)
    # weighted_data = pd.read_csv('../data/events/wbf_4l_supernew_excerpt.dat', sep='\t', dtype=np.float32)
else:
    weighted_data = pd.read_csv(weighted_events_dir + '/wbf_4l_supernew.dat', sep='\t', dtype=np.float32)

# # Check probabilities
# 
# logging.info('Sum of probabilities for regular thetas:')
# for t in range(n_thetas):
#     logging.info(t, thetas[t], np.sum(weighted_data['p_theta_' + str(t)]))
#
# 
# logging.info('Sum of probabilities for random thetas:')
# for t in range(n_randomthetas):
#     logging.info(t, np.sum(weighted_data['p_randomtheta_' + str(t)]))


logging.info('Splitting...')

# Split: 60% train, 10% evaluation, 10% calibration, 20% test
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

logging.info('Number of events (full, train, calibrate, test):')
logging.info(n_events, n_events_train, n_events_calibrate, n_events_test)

del weighted_data

logging.info('Calibrating probabilities...')

weights_train = []
for n in range(n_thetas):
    temp_weights = np.array(weighted_data_train['p_theta_' + str(n)])
    temp_weights *= 1. / sum(temp_weights)
    weights_train.append(temp_weights)
    # del weighted_data_train['p_theta_' + str(n)]
weights_train = np.asarray(weights_train)

weights_calibrate = []
for n in range(n_thetas):
    temp_weights = np.array(weighted_data_calibrate['p_theta_' + str(n)])
    temp_weights *= 1. / sum(temp_weights)
    weights_calibrate.append(temp_weights)
    # del weighted_data_calibrate['p_theta_' + str(n)]
weights_calibrate = np.asarray(weights_calibrate)

weights_test = []
for n in range(n_thetas):
    temp_weights = np.array(weighted_data_test['p_theta_' + str(n)])
    temp_weights *= 1. / sum(temp_weights)
    weights_test.append(temp_weights)
    # del weighted_data_test['p_theta_' + str(n)]
weights_test = np.asarray(weights_test)

################################################################################
# Baseline training
################################################################################

if args.train:

    logging.info('Generating baseline training sample...')


    def generate_data_train(theta0, theta1):
        indices_num = np.random.choice(list(range(n_events_train)), n_num, p=weights_train[theta0])
        indices_den = np.random.choice(list(range(n_events_train)), n_den, p=weights_train[theta1])

        X = np.vstack((
            np.array(weighted_data_train.iloc[indices_num, subset_features]),
            np.array(weighted_data_train.iloc[indices_den, subset_features])
        ))

        y = np.zeros(n_num + n_den)
        y[n_num:] = 1.

        labels_scores = ["score_theta_" + str(theta0) + "_0", "score_theta_" + str(theta0) + "_1"]
        subset_scores = [weighted_data_train.columns.get_loc(x) for x in labels_scores]
        scores = np.vstack((
            np.array(weighted_data_train.iloc[indices_num, subset_scores]),
            np.array(weighted_data_train.iloc[indices_den, subset_scores])
        ))

        thetas0 = np.zeros((n_num + n_den, 2))
        thetas0[:] = thetas[theta0]
        thetas1 = np.zeros((n_num + n_den, 2))
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

        p_score = np.hstack((
            np.array(weights_train[theta_score][indices_num]),
            np.array(weights_train[theta_score][indices_den]),
        ))

        # filter out bad events
        filter = (scores[:, 0] ** 2 + scores[:, 1] ** 2 < 2500.) & (np.log(r) ** 2 < 10000.)

        # print thetas0.shape, thetas1.shape, X.shape, y.shape, scores.shape, r.shape, p_score.shape

        return thetas0[filter], thetas1[filter], X[filter], y[filter], scores[filter], r[filter], p0[filter], p1[
            filter], p_score[filter]


    for i, t in enumerate(thetas_train):
        this_th0, this_th1, this_X, this_y, this_scores, this_r, this_p0, this_p1, this_p_score = generate_data_train(t,
                                                                                                                      theta1)
        logging.info(t, thetas[t], len(this_r))

        if i > 0:
            th0 = np.vstack((th0, np.array(this_th0, dtype=np.float16)))
            th1 = np.vstack((th1, np.array(this_th1, dtype=np.float16)))
            X = np.vstack((X, np.array(this_X, dtype=np.float16)))
            y = np.hstack((y, np.array(this_y, dtype=np.float16)))
            scores = np.vstack((scores, np.array(this_scores, dtype=np.float16)))
            r = np.hstack((r, np.array(this_r, dtype=np.float16)))
            p0 = np.hstack((r, np.array(this_p0, dtype=np.float16)))
            p1 = np.hstack((r, np.array(this_p1, dtype=np.float16)))
        else:
            th0 = np.array(this_th0, dtype=np.float16)
            th1 = np.array(this_th1, dtype=np.float16)
            X = np.array(this_X, dtype=np.float16)
            y = np.array(this_y, dtype=np.float16)
            scores = np.array(this_scores, dtype=np.float16)
            r = np.array(this_r, dtype=np.float16)
            p0 = np.array(this_p0, dtype=np.float16)
            p1 = np.array(this_p1, dtype=np.float16)

    np.save(unweighted_events_dir + '/theta0_train' + filename_addition + '.npy', th0)
    np.save(unweighted_events_dir + '/theta1_train' + filename_addition + '.npy', th1)
    np.save(unweighted_events_dir + '/X_train' + filename_addition + '.npy', X)
    np.save(unweighted_events_dir + '/y_train' + filename_addition + '.npy', y)
    np.save(unweighted_events_dir + '/scores_train' + filename_addition + '.npy', scores)
    np.save(unweighted_events_dir + '/r_train' + filename_addition + '.npy', r)
    np.save(unweighted_events_dir + '/p0_train' + filename_addition + '.npy', p0)
    np.save(unweighted_events_dir + '/p1_train' + filename_addition + '.npy', p1)

    logging.info('...done!')

################################################################################
# Basis training
################################################################################

if args.basis:

    logging.info('Generating basis-only training sample...')


    def generate_data_train_basis(theta0, theta1):
        indices_num = np.random.choice(list(range(n_events_train)), n_basis_num, p=weights_train[theta0])
        indices_den = np.random.choice(list(range(n_events_train)), n_basis_den, p=weights_train[theta1])

        X = np.vstack((
            np.array(weighted_data_train.iloc[indices_num, subset_features]),
            np.array(weighted_data_train.iloc[indices_den, subset_features])
        ))

        y = np.zeros(n_basis_num + n_basis_den)
        y[n_basis_num:] = 1.

        labels_scores = ["score_theta_" + str(theta0) + "_0", "score_theta_" + str(theta0) + "_1"]
        subset_scores = [weighted_data_train.columns.get_loc(x) for x in labels_scores]
        scores = np.vstack((
            np.array(weighted_data_train.iloc[indices_num, subset_scores]),
            np.array(weighted_data_train.iloc[indices_den, subset_scores])
        ))

        thetas0 = np.zeros((n_basis_num + n_basis_den, 2))
        thetas0[:] = thetas[theta0]
        thetas1 = np.zeros((n_basis_num + n_basis_den, 2))
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

        p_score = np.hstack((
            np.array(weights_train[theta_score][indices_num]),
            np.array(weights_train[theta_score][indices_den]),
        ))

        # print thetas0.shape, thetas1.shape, X.shape, y.shape, scores.shape, r.shape, p_score.shape

        # filter out bad events
        filter = (scores[:, 0] ** 2 + scores[:, 1] ** 2 < 2500.) & (np.log(r) ** 2 < 10000.)

        return thetas0[filter], thetas1[filter], X[filter], y[filter], scores[filter], r[filter], p0[filter], p1[
            filter], p_score[filter]


    for i, t in enumerate(thetas_basis):
        this_th0, this_th1, this_X, this_y, this_scores, this_r, this_p0, this_p1, \
            this_p_score = generate_data_train_basis(t, theta1)
        logging.info(t, thetas[t], len(this_r))

        if i > 0:
            th0 = np.vstack((th0, np.array(this_th0, dtype=np.float16)))
            th1 = np.vstack((th1, np.array(this_th1, dtype=np.float16)))
            X = np.vstack((X, np.array(this_X, dtype=np.float16)))
            y = np.hstack((y, np.array(this_y, dtype=np.float16)))
            scores = np.vstack((scores, np.array(this_scores, dtype=np.float16)))
            r = np.hstack((r, np.array(this_r, dtype=np.float16)))
            p0 = np.hstack((r, np.array(this_p0, dtype=np.float16)))
            p1 = np.hstack((r, np.array(this_p1, dtype=np.float16)))
        else:
            th0 = np.array(this_th0, dtype=np.float16)
            th1 = np.array(this_th1, dtype=np.float16)
            X = np.array(this_X, dtype=np.float16)
            y = np.array(this_y, dtype=np.float16)
            scores = np.array(this_scores, dtype=np.float16)
            r = np.array(this_r, dtype=np.float16)
            p0 = np.array(this_p0, dtype=np.float16)
            p1 = np.array(this_p1, dtype=np.float16)

    np.save(unweighted_events_dir + '/theta0_train_basis' + filename_addition + '.npy', th0)
    np.save(unweighted_events_dir + '/theta1_train_basis' + filename_addition + '.npy', th1)
    np.save(unweighted_events_dir + '/X_train_basis' + filename_addition + '.npy', X)
    np.save(unweighted_events_dir + '/y_train_basis' + filename_addition + '.npy', y)
    np.save(unweighted_events_dir + '/scores_train_basis' + filename_addition + '.npy', scores)
    np.save(unweighted_events_dir + '/r_train_basis' + filename_addition + '.npy', r)
    np.save(unweighted_events_dir + '/p0_train_basis' + filename_addition + '.npy', p0)
    np.save(unweighted_events_dir + '/p1_train_basis' + filename_addition + '.npy', p1)

    logging.info('...done!')

################################################################################
# Point-by-point training
################################################################################

if args.pointbypoint:

    logging.info('Generating point-by-point training samples...')


    def generate_data_train_point_by_point(theta0, theta1):
        indices_num = np.random.choice(list(range(n_events_train)), n_point_by_point_num, p=weights_train[theta0])
        indices_den = np.random.choice(list(range(n_events_train)), n_point_by_point_den, p=weights_train[theta1])

        X = np.vstack((
            np.array(weighted_data_train.iloc[indices_num, subset_features]),
            np.array(weighted_data_train.iloc[indices_den, subset_features])
        ))

        y = np.zeros(n_point_by_point_num + n_point_by_point_den)
        y[n_point_by_point_num:] = 1.

        labels_scores = ["score_theta_" + str(theta0) + "_0", "score_theta_" + str(theta0) + "_1"]
        subset_scores = [weighted_data_train.columns.get_loc(x) for x in labels_scores]
        scores = np.vstack((
            np.array(weighted_data_train.iloc[indices_num, subset_scores]),
            np.array(weighted_data_train.iloc[indices_den, subset_scores])
        ))

        thetas0 = np.zeros((n_point_by_point_num + n_point_by_point_den, 2))
        thetas0[:] = thetas[theta0]
        thetas1 = np.zeros((n_point_by_point_num + n_point_by_point_den, 2))
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

        p_score = np.hstack((
            np.array(weights_train[theta_score][indices_num]),
            np.array(weights_train[theta_score][indices_den]),
        ))

        # print thetas0.shape, thetas1.shape, X.shape, y.shape, scores.shape, r.shape, p_score.shape

        # filter out bad events
        filter = (scores[:, 0] ** 2 + scores[:, 1] ** 2 < 2500.) & (np.log(r) ** 2 < 10000.) & (np.isfinite(np.log(r)))

        return thetas0[filter], thetas1[filter], X[filter], y[filter], scores[filter], r[filter], p0[filter], p1[
            filter], p_score[filter]


    for i, t in enumerate(thetas_point_by_point):
        this_th0, this_th1, this_X, this_y, this_scores, this_r, this_p0, this_p1, \
            this_p_score = generate_data_train_point_by_point(t, theta1)
        logging.info(t, thetas[t], len(this_y))

        np.save(unweighted_events_dir + '/X_train_point_by_point_' + str(t) + filename_addition + '.npy', this_X)
        np.save(unweighted_events_dir + '/y_train_point_by_point_' + str(t) + filename_addition + '.npy', this_y)
        np.save(unweighted_events_dir + '/r_train_point_by_point_' + str(t) + filename_addition + '.npy', this_r)
        np.save(unweighted_events_dir + '/p0_train_point_by_point_' + str(t) + filename_addition + '.npy', this_p0)
        np.save(unweighted_events_dir + '/p1_train_point_by_point_' + str(t) + filename_addition + '.npy', this_p1)

        del this_th0, this_th1, this_X, this_y, this_scores, this_r, this_p0, this_p1, this_p_score

    logging.info('...done!')

################################################################################
# Random training
################################################################################

if args.random:

    logging.info('Generating random training sample...')


    def generate_random_data_train(randomtheta0, theta1):
        prob_num = (float(n_randomtheta_num * n_events) / float(n_randomthetas * n_events_train)
                    * weighted_data_train['p_randomtheta_' + str(randomtheta0)])
        prob_den = (float(n_randomtheta_den) / float(n_randomthetas)
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

        subset_scores_den = [weighted_data_train.columns.get_loc(x)
                             for x in ['score_theta_' + str(theta1) + '_0', 'score_theta_' + str(theta1) + '_1']]
        subset_scores_num = [weighted_data_train.columns.get_loc(x)
                             for x in ['score_randomtheta_' + str(randomtheta0) + '_0',
                                       'score_randomtheta_' + str(randomtheta0) + '_1']]
        scores = np.vstack((
            np.array(accepted_num.iloc[:, subset_scores_num]),
            np.array(accepted_den.iloc[:, subset_scores_den])
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

        logging.info(randomtheta0, '-', n_dice, 'throws,', len(accepted_num), 'num', len(accepted_den), 'den')

        # filter out bad events
        filter = (scores[:, 0] ** 2 + scores[:, 1] ** 2 < 2500.) & (np.log(r) ** 2 < 10000.)

        return thetas0[filter], thetas1[filter], X[filter], y[filter], scores[filter], r[filter]


    for t in range(n_randomthetas):
        this_th0, this_th1, this_X, this_y, this_scores, this_r = generate_random_data_train(t, theta1)

        if t > 0:
            th0 = np.vstack((th0, np.array(this_th0, dtype=np.float16)))
            th1 = np.vstack((th1, np.array(this_th1, dtype=np.float16)))
            X = np.vstack((X, np.array(this_X, dtype=np.float16)))
            y = np.hstack((y, np.array(this_y, dtype=np.float16)))
            scores = np.vstack((scores, np.array(this_scores, dtype=np.float16)))
            r = np.hstack((r, np.array(this_r, dtype=np.float16)))
        else:
            th0 = np.array(this_th0, dtype=np.float16)
            th1 = np.array(this_th1, dtype=np.float16)
            X = np.array(this_X, dtype=np.float16)
            y = np.array(this_y, dtype=np.float16)
            scores = np.array(this_scores, dtype=np.float16)
            r = np.array(this_r, dtype=np.float16)

    np.save(unweighted_events_dir + '/theta0_train_random' + filename_addition + '.npy', th0)
    np.save(unweighted_events_dir + '/theta1_train_random' + filename_addition + '.npy', th1)
    np.save(unweighted_events_dir + '/X_train_random' + filename_addition + '.npy', X)
    np.save(unweighted_events_dir + '/y_train_random' + filename_addition + '.npy', y)
    np.save(unweighted_events_dir + '/scores_train_random' + filename_addition + '.npy', scores)
    np.save(unweighted_events_dir + '/r_train_random' + filename_addition + '.npy', r)

    logging.info('...done!')

################################################################################
# Calibration
################################################################################

if args.calibration:

    logging.info('Generating calibration data...')


    def generate_data_calibration(theta_observed):
        indices = np.random.choice(list(range(n_events_train)), n_calibrate, p=weights_train[theta_observed])

        X = np.asarray(weighted_data_train.iloc[indices, subset_features])

        r = np.zeros((n_thetas, n_calibrate))
        for t in range(n_thetas):
            r[t, :] = np.array(weights_train[t][indices] / weights_train[theta_observed][indices])

        # filter out bad events
        filter = np.all(np.log(r) ** 2 < 10000., axis=0)

        return X[filter], r[:, filter]


    for i, t in enumerate(thetas_train):
        this_X, this_weights = generate_data_calibration(t)
        logging.info(t, thetas[t], this_X.shape[0])

        if i > 0:
            X = np.vstack((X, np.array(this_X, dtype=np.float16)))
            weights = np.hstack((weights, np.array(this_weights, dtype=np.float16)))
        else:
            X = np.array(this_X, dtype=np.float16)
            weights = np.array(this_weights, dtype=np.float16)

        np.save(unweighted_events_dir + '/X_calibration' + filename_addition + '.npy', X)
        np.save(unweighted_events_dir + '/weights_calibration' + filename_addition + '.npy', weights)

################################################################################
# Training sample for score regression
################################################################################

if args.scoreregression:
    logging.info('Generating training sample for score regression...')


    def generate_data_score_regression(theta):
        indices = np.random.choice(list(range(n_events_train)), n_score_regression, p=weights_train[theta])

        X = np.array(weighted_data_train.iloc[indices, subset_features])

        labels_scores = ["score_theta_" + str(theta) + "_0", "score_theta_" + str(theta) + "_1"]
        subset_scores = [weighted_data_train.columns.get_loc(x) for x in labels_scores]
        scores = np.array(weighted_data_train.iloc[indices, subset_scores])

        p = np.array(weights_train[theta][indices])

        # filter out bad events
        filter = (scores[:, 0] ** 2 + scores[:, 1] ** 2 < 2500.)

        return X[filter], scores[filter], p[filter]


    X, scores, p = generate_data_score_regression(theta_score_regression)

    np.save(unweighted_events_dir + '/X_train_scoreregression' + filename_addition + '.npy', X)
    np.save(unweighted_events_dir + '/scores_train_scoreregression' + filename_addition + '.npy', scores)
    np.save(unweighted_events_dir + '/p_train_scoreregression' + filename_addition + '.npy', p)

    logging.info('...done!')

################################################################################
# Likelihood ratio evaluation
################################################################################

if args.test:

    logging.info('Generating test sample...')


    def generate_data_test(theta_observed, theta1):
        indices = np.random.choice(list(range(n_events_test)), n_test, p=weights_test[theta_observed])

        X = np.asarray(weighted_data_test.iloc[indices, subset_features])

        r = np.zeros((n_thetas, n_test))
        for t in range(n_thetas):
            r[t, :] = np.array(weights_test[t][indices] / weights_test[theta1][indices])

        scores = np.zeros((n_thetas, n_test, 2))
        for t in range(n_thetas):
            labels_scores = ["score_theta_" + str(t) + "_0", "score_theta_" + str(t) + "_1"]
            subset_scores = [weighted_data_test.columns.get_loc(x) for x in labels_scores]
            scores[t] = np.array(weighted_data_test.iloc[indices, subset_scores])

        p1 = np.array(weights_test[theta1][indices])

        # logging.info(X.shape, scores.shape, r.shape, p1.shape)

        # filter out bad events
        filter = np.all((scores[:, :, 0] ** 2 + scores[:, :, 1] ** 2 < 2500.) & (np.log(r) ** 2 < 10000.), axis=0)

        return X[filter], scores[:, filter, :], r[:, filter], p1[filter]


    X, scores, r, p1 = generate_data_test(theta_observed, theta1)

    np.save(unweighted_events_dir + '/X_test' + filename_addition + '.npy', X)
    np.save(unweighted_events_dir + '/scores_test' + filename_addition + '.npy', scores)
    np.save(unweighted_events_dir + '/r_test' + filename_addition + '.npy', r)
    np.save(unweighted_events_dir + '/p1_test' + filename_addition + '.npy', p1)

    logging.info('...done!')

################################################################################
# Neyman construction
################################################################################

if args.neyman:

    logging.info('Generating Neyman construction samples...')


    def generate_data_neyman(theta_observed, n_toy_experiments):

        indices = np.random.choice(list(range(n_events_test)), n_toy_experiments * n_expected_events,
                                   p=weights_test[theta_observed])

        X = np.asarray(weighted_data_test.iloc[indices, subset_features])

        return X.reshape((n_toy_experiments, n_expected_events, -1))


    # Observed
    X = generate_data_neyman(theta_observed, n_neyman_observed_experiments)
    logging.info('Generated %s "observed" toy experiments with %s events each according to theta = %s',
                 X.shape[0], X.shape[1], thetas[theta_observed])
    np.save(unweighted_events_dir + '/X_neyman_observed' + filename_addition + '.npy', X)

    # Distribution
    for t, theta in enumerate(thetas):
        X = generate_data_neyman(t, n_neyman_distribution_experiments)
        logging.info('Generated %s "distribution" toy experiments with %s events each according to theta = %s',
                     X.shape[0], X.shape[1], theta)
        np.save(unweighted_events_dir + '/X_neyman_distribution_' + str(t) + filename_addition + '.npy', X)

    logging.info('...done!')

################################################################################
# Roam data
################################################################################

if args.roam:

    logging.info('Generating roaming sample...')


    def generate_data_roam(theta_observed, theta1):
        indices = np.random.choice(list(range(n_events_test)), n_roam, p=weights_test[theta_observed])

        X = np.asarray(weighted_data_test.iloc[indices, subset_features])

        r = np.zeros((n_thetas, n_roam))
        for t in range(n_thetas):
            r[t, :] = np.array(weights_test[t][indices] / weights_test[theta1][indices])

        return X, r


    X, r = generate_data_roam(theta_observed, theta1)

    np.save(unweighted_events_dir + '/X_roam' + filename_addition + '.npy', X)
    np.save(unweighted_events_dir + '/r_roam' + filename_addition + '.npy', r)

    logging.info('...done!')
