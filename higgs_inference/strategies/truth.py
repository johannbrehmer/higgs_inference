#! /usr/bin/env python

################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import logging
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern

from higgs_inference import settings


################################################################################
# What do
################################################################################

def truth_inference(options=''):
    logging.info('Starting truth calculation')

    denom1_mode = ('denom1' in options)

    filename_addition = ''
    input_filename_addition = ''
    if denom1_mode:
        input_filename_addition = '_denom1'
        filename_addition += '_denom1'

    data_dir = settings.base_dir + '/data'
    neyman_dir = settings.neyman_dir + '/truth'
    results_dir = settings.base_dir + '/results/truth'

    ################################################################################
    # Data
    ################################################################################

    thetas = np.load(data_dir + '/thetas/thetas_parameterized.npy')
    n_thetas = len(thetas)

    scores_test = np.load(settings.unweighted_events_dir + '/scores_test' + input_filename_addition + '.npy')
    r_test = np.load(settings.unweighted_events_dir + '/r_test' + input_filename_addition + '.npy')
    r_roam = np.load(settings.unweighted_events_dir + '/r_roam' + input_filename_addition + '.npy')
    r_neyman_observed = np.load(settings.unweighted_events_dir + '/r_neyman_observed.npy')

    n_events_test = r_test.shape[1]
    assert n_thetas == r_test.shape[0]

    xi = np.linspace(-1.0, 1.0, settings.n_thetas_roam)
    yi = np.linspace(-1.0, 1.0, settings.n_thetas_roam)
    xx, yy = np.meshgrid(xi, yi)

    ################################################################################
    # Evaluate truth likelihood ratios
    ################################################################################

    logging.info('Starting evaluation')
    expected_llr_truth = []

    for t, theta in enumerate(thetas):
        ratios = np.array(np.log(r_test[t, :]))
        expected_llr_truth.append(
            - 2. * float(settings.n_expected_events) / float(n_events_test) * np.sum(ratios[np.isfinite(ratios)]))

    r_nottrained_truth = np.copy(r_test[settings.theta_benchmark_nottrained, :])
    r_trained_truth = np.copy(r_test[settings.theta_benchmark_trained, :])
    scores_trained_truth = np.copy(scores_test[settings.theta_benchmark_trained, :])
    scores_nottrained_truth = np.copy(scores_test[settings.theta_benchmark_nottrained, :])

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

    logging.info('Starting evaluation of Neyman experiments')
    for t in range(n_thetas):
        llr_neyman_observed = -2. * np.sum(np.log(r_neyman_observed[t]), axis=1)
        np.save(neyman_dir + '/neyman_observed_truth_' + str(t) + filename_addition + '.npy', llr_neyman_observed)

        llr_neyman_distributions = []
        for tt in range(n_thetas):
            r_neyman_distribution = np.load(
                settings.unweighted_events_dir + '/r_neyman_distribution_' + str(tt) + '.npy')
            llr_neyman_distributions.append(-2. * np.sum(np.log(r_neyman_distribution[t]), axis=1))

        llr_neyman_distributions = np.asarray(llr_neyman_distributions)
        np.save(neyman_dir + '/neyman_llr_distribution__truth_' + str(t) + filename_addition + '.npy',
                llr_neyman_distributions)
