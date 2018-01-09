#! /usr/bin/env python

################################################################################
# Imports
################################################################################

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



################################################################################
# What do?
################################################################################

args = sys.argv[1:]

do_train = ('train' in args)
do_basis = ('basis' in args)
do_random = ('random' in args)
do_point_by_point = ('point-by-point' in args)
do_calibration = ('calibration' in args)
do_test = ('test' in args)
do_roam = ('roam' in args)

denom1_mode = ('denom1' in args)

print('')
print('Tasks:')
print('  Baseline training:      ', do_train)
print('  Random training:        ', do_random)
print('  Morphing training:      ', do_basis)
print('  Point-by-point training:', do_point_by_point)
print('  Calibration:            ', do_calibration)
print('  Evaluation:             ', do_test)
print('  Roaming:                ', do_roam)
print('')
print('Options:')
if denom1_mode:
    print('  Denominator: theta', 1)
else:
    print('  Denominator: theta', 0)

filename_addition = ''
if denom1_mode:
    filename_addition = '_denom1'

data_dir = '../data'



################################################################################
# Thetas etc
################################################################################

thetas = np.load(data_dir + '/thetas/thetas_parameterized.npy')

n_thetas = len(thetas)
n_randomthetas = 100
theta1 = 708
if denom1_mode:
    theta1 = 422
theta_test = 213
theta_observed = 0
theta_score = 0 # for local model
thetas_train = list(range(17,1017))
thetas_basis = [0, 101, 106, 902, 910,
                226, 373, 583, 747, 841,
                599, 709, 422, 367, 167]
thetas_point_by_point = [0, 13, 14, 15, 16, 9, 213, 647, 736, 643, 794, 727, 52, 86, 824, 538, 581, 828, 583, 549,
                         738, 872, 912, 935, 410, 209, 972, 706, 120, 113, 407, 281, 115, 20, 820, 172, 32, 546, 176,
                         974, 71, 831, 834, 730, 266, 621, 333, 742, 991, 647, 580, 124, 817, 720, 95, 142, 984, 777,
                         699, 472, 479, 206, 830, 287, 648, 393, 610, 257, 683, 425, 827, 484, 568, 601, 913, 39, 830,
                         973, 786, 468, 609, 430, 73, 89, 578, 850, 997, 176, 125, 277, 847, 867, 904, 584, 327, 423,
                         559, 351, 123, 903]
thetas_test = list(range(17))

n_num                =    5000 # per value of theta
n_den                =    5000 # per value of theta
n_basis_num          =  333333 # per basis thetas
n_basis_den          =  333333 # per basis thetas
n_randomtheta_num    = 5000000 # in total (expected)
n_randomtheta_den    = 5000000 # in total (expected)
n_point_by_point_num =   50000 # per theta
n_point_by_point_den =   50000 # per theta
n_calibrate          =   20000
n_observed           =   50000
n_roam               =      20

subset_features = list(range(42)) #list(range(15))



################################################################################
# Data
################################################################################

weighted_data = pd.read_csv('/scratch/jb6504/eft-data/wbf_4l_supernew.dat', sep='\t', dtype=np.float32)

# # Check probabilities
# print('')
# print('Sum of probabilities for regular thetas:')
# for t in range(n_thetas):
#     print(t, thetas[t], np.sum(weighted_data['p_theta_' + str(t)]))
#
# print('')
# print('Sum of probabilities for random thetas:')
# for t in range(n_randomthetas):
#     print(t, np.sum(weighted_data['p_randomtheta_' + str(t)]))

print('')
print('Splitting...')

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

print('')
print('Number of events (full, train, calibrate, test):')
print(n_events, n_events_train, n_events_calibrate, n_events_test)

del weighted_data

print('')
print('Calibrating probabilities...')

weights_train = []
for n in range(n_thetas):
    temp_weights = np.array(weighted_data_train['p_theta_' + str(n)])
    temp_weights *= 1. / sum(temp_weights)
    weights_train.append(temp_weights)
    #del weighted_data_train['p_theta_' + str(n)]
weights_train = np.asarray(weights_train)

weights_calibrate = []
for n in range(n_thetas):
    temp_weights = np.array(weighted_data_calibrate['p_theta_' + str(n)])
    temp_weights *= 1. / sum(temp_weights)
    weights_calibrate.append(temp_weights)
    #del weighted_data_calibrate['p_theta_' + str(n)]
weights_calibrate = np.asarray(weights_calibrate)

weights_test = []
for n in range(n_thetas):
    temp_weights = np.array(weighted_data_test['p_theta_' + str(n)])
    temp_weights *= 1. / sum(temp_weights)
    weights_test.append(temp_weights)
    #del weighted_data_test['p_theta_' + str(n)]
weights_test = np.asarray(weights_test)



################################################################################
# Baseline training
################################################################################

if do_train:

    print('')
    print('Generating baseline training sample...')

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
        filter = (scores[:,0]**2 + scores[:,1]**2 < 2500.) & (np.log(r)**2 < 10000.)

        # print thetas0.shape, thetas1.shape, X.shape, y.shape, scores.shape, r.shape, p_score.shape

        return thetas0[filter], thetas1[filter], X[filter], y[filter], scores[filter], r[filter], p0[filter], p1[filter], p_score[filter]

    for i, t in enumerate(thetas_train):
        this_th0, this_th1, this_X, this_y, this_scores, this_r, this_p0, this_p1, this_p_score = generate_data_train(t, theta1)
        print(t, thetas[t], len(this_r))

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

    np.save(data_dir + '/unweighted_events/theta0_train' + filename_addition + '.npy', th0)
    np.save(data_dir + '/unweighted_events/theta1_train' + filename_addition + '.npy', th1)
    np.save(data_dir + '/unweighted_events/X_train' + filename_addition + '.npy', X)
    np.save(data_dir + '/unweighted_events/y_train' + filename_addition + '.npy', y)
    np.save(data_dir + '/unweighted_events/scores_train' + filename_addition + '.npy', scores)
    np.save(data_dir + '/unweighted_events/r_train' + filename_addition + '.npy', r)
    np.save(data_dir + '/unweighted_events/p0_train' + filename_addition + '.npy', p0)
    np.save(data_dir + '/unweighted_events/p1_train' + filename_addition + '.npy', p1)

    print('...done!')



################################################################################
# Basis training
################################################################################

if do_basis:

    print('')
    print('Generating basis-only training sample...')

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
        filter = (scores[:,0]**2 + scores[:,1]**2 < 2500.) & (np.log(r)**2 < 10000.)

        return thetas0[filter], thetas1[filter], X[filter], y[filter], scores[filter], r[filter], p0[filter], p1[filter], p_score[filter]

    for i, t in enumerate(thetas_basis):
        this_th0, this_th1, this_X, this_y, this_scores, this_r, this_p0, this_p1,\
            this_p_score = generate_data_train_basis(t, theta1)
        print(t, thetas[t], len(this_r))

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

    np.save(data_dir + '/unweighted_events/theta0_train_basis' + filename_addition + '.npy', th0)
    np.save(data_dir + '/unweighted_events/theta1_train_basis' + filename_addition + '.npy', th1)
    np.save(data_dir + '/unweighted_events/X_train_basis' + filename_addition + '.npy', X)
    np.save(data_dir + '/unweighted_events/y_train_basis' + filename_addition + '.npy', y)
    np.save(data_dir + '/unweighted_events/scores_train_basis' + filename_addition + '.npy', scores)
    np.save(data_dir + '/unweighted_events/r_train_basis' + filename_addition + '.npy', r)
    np.save(data_dir + '/unweighted_events/p0_train_basis' + filename_addition + '.npy', p0)
    np.save(data_dir + '/unweighted_events/p1_train_basis' + filename_addition + '.npy', p1)

    print('...done!')



################################################################################
# Point-by-point training
################################################################################

if do_point_by_point:

    print('')
    print('Generating point-by-point training samples...')

    def generate_data_train_point_by_point(theta0, theta1):
        indices_num = np.random.choice(list(range(n_events_train)), n_point_by_point_num, p=weights_train[theta0])
        indices_den = np.random.choice(list(range(n_events_train)), n_point_by_point_den, p=weights_train[theta1])

        X = np.vstack((
            np.array(weighted_data_train.iloc[indices_num, subset_features]),
            np.array(weighted_data_train.iloc[indices_den, subset_features])
        ))

        y = np.zeros(n_point_by_point_num + n_point_by_point_den)
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
        filter = (scores[:,0]**2 + scores[:,1]**2 < 2500.) & (np.log(r)**2 < 10000.)

        return thetas0, thetas1, X, y, scores, r, p0, p1, p_score

    for i, t in enumerate(thetas_point_by_point):
        this_th0, this_th1, this_X, this_y, this_scores, this_r, this_p0, this_p1,\
            this_p_score = generate_data_train_basis(t, theta1)
        print(t, thetas[t])

        np.save(data_dir + '/unweighted_events/X_train_point_by_point_' + str(t) + filename_addition + '.npy', this_X)
        np.save(data_dir + '/unweighted_events/y_train_point_by_point_' + str(t) + filename_addition + '.npy', this_y)
        np.save(data_dir + '/unweighted_events/r_train_point_by_point_' + str(t) + filename_addition + '.npy', this_r)
        np.save(data_dir + '/unweighted_events/p0_train_point_by_point_' + str(t) + filename_addition + '.npy', this_p0)
        np.save(data_dir + '/unweighted_events/p1_train_point_by_point_' + str(t) + filename_addition + '.npy', this_p1)

    print('...done!')






################################################################################
# Random training
################################################################################

if do_random:

    print('')
    print('Generating random training sample...')

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

        print(randomtheta0, '-', n_dice, 'throws,', len(accepted_num), 'num', len(accepted_den), 'den')

        # filter out bad events
        filter = (scores[:,0]**2 + scores[:,1]**2 < 2500.) & (np.log(r)**2 < 10000.)

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

    np.save(data_dir + '/unweighted_events/theta0_train_random' + filename_addition + '.npy', th0)
    np.save(data_dir + '/unweighted_events/theta1_train_random' + filename_addition + '.npy', th1)
    np.save(data_dir + '/unweighted_events/X_train_random' + filename_addition + '.npy', X)
    np.save(data_dir + '/unweighted_events/y_train_random' + filename_addition + '.npy', y)
    np.save(data_dir + '/unweighted_events/scores_train_random' + filename_addition + '.npy', scores)
    np.save(data_dir + '/unweighted_events/r_train_random' + filename_addition + '.npy', r)

    print('...done!')



################################################################################
# Calibration
################################################################################

if do_calibration:

    print('')
    print('Generating calibration data...')


    def generate_data_calibration(theta_observed):
        indices = np.random.choice(list(range(n_events_train)), n_calibrate, p=weights_train[theta_observed])

        X = np.asarray(weighted_data_train.iloc[indices, subset_features])

        r = np.zeros((n_thetas, n_calibrate))
        for t in range(n_thetas):
            r[t, :] = np.array(weights_train[t][indices] / weights_train[theta_observed][indices])


        # filter out bad events
        filter = (np.log(r)**2 < 10000.)

        return X[filter], r[:,filter]


    X, weights = generate_data_calibration(theta_observed)

    np.save(data_dir + '/unweighted_events/X_calibration' + filename_addition + '.npy', X)
    np.save(data_dir + '/unweighted_events/weights_calibration' + filename_addition + '.npy', weights)




################################################################################
# Test data
################################################################################

if do_test:

    print('')
    print('Generating test sample...')


    def generate_data_test(theta_observed, theta1):
        indices = np.random.choice(list(range(n_events_test)), n_observed, p=weights_test[theta_observed])

        X = np.asarray(weighted_data_test.iloc[indices, subset_features])

        r = np.zeros((n_thetas, n_observed))
        for t in range(n_thetas):
            r[t, :] = np.array(weights_test[t][indices] / weights_test[theta1][indices])

        scores = np.zeros((n_thetas, n_observed, 2))
        for t in range(n_thetas):
            labels_scores = ["score_theta_" + str(t) + "_0", "score_theta_" + str(t) + "_1"]
            subset_scores = [weighted_data_test.columns.get_loc(x) for x in labels_scores]
            scores[t] = np.array(weighted_data_test.iloc[indices, subset_scores])

        p_score = np.array(weights_test[theta1][indices])

        print(X.shape, scores.shape, r.shape, p_score.shape)

        # filter out bad events
        filter = (scores[:,0]**2 + scores[:,1]**2 < 2500.) & (np.log(r)**2 < 10000.)

        return X[filter], scores[filter], r[:,filter], p1[filter]


    X, scores, r, p1 = generate_data_test(theta_observed, theta1)

    np.save(data_dir + '/unweighted_events/X_test' + filename_addition + '.npy', X)
    np.save(data_dir + '/unweighted_events/scores_test' + filename_addition + '.npy', scores)
    np.save(data_dir + '/unweighted_events/r_test' + filename_addition + '.npy', r)
    np.save(data_dir + '/unweighted_events/p1_test' + filename_addition + '.npy', p1)

    print('...done!')



################################################################################
# Roam data
################################################################################

if do_roam:
    print('')
    print('Generating roaming sample...')

    def generate_data_roam(theta_observed, theta1):
        indices = np.random.choice(list(range(n_events_test)), n_roam, p=weights_test[theta_observed])

        X = np.asarray(weighted_data_test.iloc[indices, subset_features])

        r = np.zeros((n_thetas, n_roam))
        for t in range(n_thetas):
            r[t, :] = np.array(weights_test[t][indices] / weights_test[theta1][indices])

        return X, r


    X, r = generate_data_roam(theta_observed, theta1)

    np.save(data_dir + '/unweighted_events/X_roam' + filename_addition + '.npy', X)
    np.save(data_dir + '/unweighted_events/r_roam' + filename_addition + '.npy', r)

    print('...done!')
