################################################################################
# Imports
################################################################################

import numpy as np

################################################################################
# Directories
################################################################################

base_dir = '/home/jb6504/higgs_inference'  # Will be overridden at startup

# Large directories, on scratch on cluster
weighted_events_dir = '/scratch/jb6504/higgs_inference/data/events'
unweighted_events_dir = '/scratch/jb6504/higgs_inference/data/unweighted_events'
neyman_dir = '/scratch/jb6504/higgs_inference/results/neyman'

################################################################################
# Features and parameters
################################################################################

feature_labels = ['jet1_E', 'jet1_pt', 'jet1_eta', 'jet1_phi',
                  'jet2_E', 'jet2_pt', 'jet2_eta', 'jet2_phi',
                  'lepton1_E', 'lepton1_pt', 'lepton1_eta', 'lepton1_phi',
                  'lepton2_E', 'lepton2_pt', 'lepton2_eta', 'lepton2_phi',
                  'lepton3_E', 'lepton3_pt', 'lepton3_eta', 'lepton3_phi',
                  'lepton4_E', 'lepton4_pt', 'lepton4_eta', 'lepton4_phi',
                  'higgs_E', 'higgs_pt', 'higgs_eta', 'higgs_phi', 'higgs_m',
                  'Z1_E', 'Z1_pt', 'Z1_eta', 'Z1_phi', 'Z1_m',
                  'Z2_E', 'Z2_pt', 'Z2_eta', 'Z2_phi', 'Z2_m',
                  'm_jj', 'deltaeta_jj', 'deltaphi_jj']

n_params = 2
n_features = len(feature_labels)
n_thetas_features = n_params + n_features
n_morphing_samples = 15

epsilon = 1.e-3  # for various numerical accuracy issues

################################################################################
# Benchmark theta points
################################################################################

try:
    thetas = np.load(base_dir + '/data/thetas/thetas_parameterized.npy')
except IOError:
    base_dir = '../'
    thetas = np.load(base_dir + '/data/thetas/thetas_parameterized.npy')

n_thetas = len(thetas)
n_thetas_roam = 101
n_randomthetas = 100

# The following are indices of the thetas array defined above

theta_observed = 0
theta_score_regression = 0

theta1_default = 708
theta1_alternatives = [0, 602, 498, 202]

theta_benchmark_trained = 422
theta_benchmark_nottrained = 9
theta_benchmark_illustration = 9

thetas_train = list(range(17, n_thetas))
thetas_test = list(range(17))
thetas_morphing_basis = [0, 101, 106, 902, 910,
                         226, 373, 583, 747, 841,
                         599, 709, 422, 367, 167]
pbp_training_thetas = [0, 13, 14, 15, 16, 9, 422, 956, 666, 802, 675, 839, 699, 820, 203, 291, 634, 371, 973, 742, 901,
                       181, 82, 937, 510, 919, 745, 588, 804, 963, 396, 62, 401, 925, 874, 770, 108, 179, 669, 758, 113,
                       587, 600, 975, 496, 66, 467, 412, 701, 986, 598, 810, 97, 18, 723, 159, 320, 301, 352, 159, 89,
                       421, 574, 923, 849, 299, 119, 167, 939, 402, 52, 787, 978, 41, 873, 533, 827, 304, 294, 760, 890,
                       539, 1000, 291, 740, 276, 679, 167, 125, 429, 149, 430, 720, 123, 908, 256, 777, 809, 269, 851]
thetas_around_sm = list(range(1, 9))
extended_pbp_training_thetas = pbp_training_thetas + thetas_around_sm

################################################################################
# Evaluation
################################################################################

# Default setup
n_expected_events = 36
n_neyman_null_experiments = 10000
n_neyman_alternate_experiments = 1001
n_expected_events_neyman = 36
n_convolutions_neyman = 0

# Single-event distributions
n_expected_events_neyman2 = 1
n_neyman2_null_experiments = 10000
n_neyman2_alternate_experiments = 10000
n_convolutions_neyman2 = 35

# Removing duplicates because why not
n_expected_events_neyman3 = 1
n_neyman3_null_experiments = 10000
n_neyman3_alternate_experiments = 10000
n_convolutions_neyman3 = 35

# Convolution settings
neyman_convolution_min = -75.
neyman_convolution_max = 75.
neyman_convolution_bins = 15001
neyman_convolution_histo_edges = np.linspace(neyman_convolution_min, neyman_convolution_max,
                                             neyman_convolution_bins + 1)

# Confidence limit
confidence_levels = np.asarray([0.68, 0.95, 0.997])
q_threshold = - 2. * np.log(1. - confidence_levels)

################################################################################
# Network architecture and training
################################################################################

# Depth
n_hidden_layers_shallow = 2
n_hidden_layers_default = 3
n_hidden_layers_deep = 5

# Carl / regression + score: relative weight in loss function
alpha_regression_default = 100.
alpha_carl_default = 5.

# Training length (for full training samples, will be increased for reduced training samples)
n_epochs_short = 1
n_epochs_default = 50
n_epochs_long = 100
early_stopping_patience = 10
validation_split = 0.2

# Learning rate
learning_rate_default = 0.001
learning_rate_small = 0.0001
learning_rate_large = 0.01

# Learning rate decay
learning_rate_decay = - 1. / n_epochs_default * np.log(0.01)  # Exponential decay to 1% of original weight

# Batch size
batch_size = 128  # Will be overwritten at startup
batch_size_default = 128
batch_size_small = 64
batch_size_large = 256

################################################################################
# Size of unweighted event samples
################################################################################

# Baseline training, per theta
n_events_baseline_num = 5000
n_events_baseline_den = 5000

# Basis training, per basis theta
n_events_basis_num = 333333
n_events_basis_den = 333333

# Random theta training, total
n_events_randomtheta_num = 5000000
n_events_randomtheta_den = 5000000

# PbP training, per PbP theta
n_events_n_point_by_point_num = 250000
n_events_n_point_by_point_den = 250000

# Score regression training, total
n_events_score_regression = 10000000

# Calibration, per theta
n_events_calibration = 1000

# Recalibration, overall
n_events_recalibration = 1000000

# Evaluation, total
n_events_test = 50000

# Roaming, total
n_events_roam = 100

# Illustration, per nom and denom
n_events_illustration = 100000

################################################################################
# Smearing (approximate shower + detector simulation)
################################################################################

smearing_eta_phi = 0.1  # Original: 0.1
smearing_jet_energies = 0.5  # times sqrt(E). Original: 0.5. Not used in current version.
smearing_lepton_pt = 3.e-4  # times pT^2. Original: 3.e-4

################################################################################
# Instead of discarding events with NaNs, can replace them with the following placeholder numbers:
################################################################################

new_samples_nan_score = np.asarray([0., 0.])
new_samples_nan_r = 1.e9

################################################################################
# Metrics
################################################################################

trim_mean_fraction = 0.05  # Fraction of best-fit and worst-fit events thrown away in the trimmed mean

theta_prior = np.exp(-(thetas[:, 0] ** 2 + thetas[:, 1] ** 2) / (2. * 2. * 0.2 ** 2))
theta_prior[:17] = 0.
theta_prior /= np.sum(theta_prior)
