#! /usr/bin/env python

################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import logging
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern


################################################################################
# What do
################################################################################

def truth_inference(options=''):
    denom1_mode = ('denom1' in options)

    filename_addition = ''
    input_filename_addition = ''
    if denom1_mode:
        input_filename_addition = '_denom1'
        filename_addition += '_denom1'

    data_dir = '../data'
    unweighted_events_dir = '/scratch/jb6504/higgs_inference/data/unweighted_events'
    results_dir = '../results/truth'

    ################################################################################
    # Data
    ################################################################################

    thetas = np.load(data_dir + '/thetas/thetas_parameterized.npy')

    n_thetas = len(thetas)
    theta_trained = 422
    theta_nottrained = 9

    scores_test = np.load(unweighted_events_dir + '/scores_test' + input_filename_addition + '.npy')
    r_test = np.load(unweighted_events_dir + '/r_test' + input_filename_addition + '.npy')

    r_roam = np.load(unweighted_events_dir + '/r_roam' + input_filename_addition + '.npy')

    n_expected_events = 36
    n_events_test = r_test.shape[1]
    assert n_thetas == r_test.shape[0]
    # n_pseudoexperiments_series = 5
    # n_pseudoexperiments_events = [10, 30, 100, 300, 1000]
    # n_pseudoexperiments_repetitions = 1000

    n_thetas_roam = 101
    xi = np.linspace(-1.0, 1.0, n_thetas_roam)
    yi = np.linspace(-1.0, 1.0, n_thetas_roam)
    xx, yy = np.meshgrid(xi, yi)

    ################################################################################
    # Truth
    ################################################################################

    logging.info('Starting evaluation')
    expected_llr_truth = []
    for t, theta in enumerate(thetas):
        ratios = np.array(np.log(r_test[t, :]))
        expected_llr_truth.append(- 2. * float(n_expected_events) / float(n_events_test) * np.sum(ratios[np.isfinite(ratios)]))

    r_nottrained_truth = np.copy(r_test[theta_nottrained, :])
    r_trained_truth = np.copy(r_test[theta_trained, :])
    scores_trained_truth = np.copy(scores_test[theta_trained, :])
    scores_nottrained_truth = np.copy(scores_test[theta_nottrained, :])

    np.save(results_dir + '/r_nottrained_truth' + filename_addition + '.npy', r_nottrained_truth)
    np.save(results_dir + '/r_trained_truth' + filename_addition + '.npy', r_trained_truth)
    np.save(results_dir + '/scores_trained_truth' + filename_addition + '.npy', scores_trained_truth)
    np.save(results_dir + '/scores_nottrained_truth' + filename_addition + '.npy', scores_nottrained_truth)
    np.save(results_dir + '/llr_truth' + filename_addition + '.npy', expected_llr_truth)

    logging.info('Starting roaming')
    gp = GaussianProcessRegressor(normalize_y=True,
                                  kernel=C(1.0) * Matern(1.0, nu=0.5), n_restarts_optimizer=10)
    gp.fit(thetas[:], np.log(r_roam))
    r_roam_truth = np.exp(gp.predict(np.c_[xx.ravel(), yy.ravel()])).T
    np.save(results_dir + '/r_roam_truth' + filename_addition + '.npy', r_roam_truth)

    # logging.info('Starting pseudo-experiments')
    # pseudoexperiments = np.zeros((n_thetas, n_pseudoexperiments_series, n_pseudoexperiments_repetitions))
    # for i, n in enumerate(n_pseudoexperiments_events):
    #     for j in range(n_pseudoexperiments_repetitions):
    #         indices = np.random.choice(list(range(n_events_test)), n)
    #         for t, theta in enumerate(thetas):
    #             ratios = np.array(np.log(r_test[t, indices]))
    #             pseudoexperiments[t, i, j] = - n_expected_events / float(n) * np.sum(ratios[np.isfinite(ratios)])
    # pseudoexperiments_variance = np.zeros((n_thetas, n_pseudoexperiments_series))
    # for t in range(n_thetas):
    #     for i in range(n_pseudoexperiments_series):
    #         pseudoexperiments_variance[t, i] = np.var(pseudoexperiments[t, i, :])
    # np.save(results_dir + '/pseudoexperiments_variance_truth' + filename_addition + '.npy',
    #         pseudoexperiments_variance)
