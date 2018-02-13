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
theta1_alternative = 422

theta_benchmark_trained = 422
theta_benchmark_nottrained = 9

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

################################################################################
# Evaluation
################################################################################

# Numnber of events
n_expected_events = 36

# Number of toy experiments
n_neyman_distribution_experiments = 500
n_neyman_observed_experiments = 51

################################################################################
# Network architecture and training
################################################################################

# Depth
n_hidden_layers_shallow = 1
n_hidden_layers_default = 2
n_hidden_layers_deep = 3

# Carl / regression + score: relative weight in loss function
alpha_regression_default = 0.02
alpha_carl_default = 0.35
alpha_regression_small = 0.01
alpha_carl_small = 0.1

# Training length
n_epochs_short = 1
n_epochs_default = 20
n_epochs_long = 50
early_stopping_patience = 5
validation_split = 0.2

# Learning rate
learning_rate_default = 0.001
learning_rate_small = 0.0001
learning_rate_large = 0.01

# Batch size
batch_size = 32  # Will be overwritten at startup
batch_size_default = 32
batch_size_small = 16
batch_size_large = 64

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
n_events_n_point_by_point_num = 500000  # Next run: reduce
n_events_n_point_by_point_den = 500000  # Next run: reduce

# Score regression training, total
n_events_score_regression = 10000000

# Calibration, per theta
n_events_calibration = 1000

# Evaluation, total
n_events_test = 50000

# Roaming, total
n_events_roam = 20

################################################################################
# Smearing (approximate shower + detector simulation)
################################################################################

smearing_eta_phi = 0.1  # Original: 0.1
smearing_jet_energies = 0.5  # times sqrt(E). Original: 0.5. Not used in current version.
smearing_lepton_pt = 3.e-4  # times pT^2. Original: 3.e-4

################################################################################
# Sanitization cuts to avoid large impact of a few extreme events.
################################################################################

max_score = 50.
max_logr = 100.

trim_mean_fraction = 0.05  # Fraction of best-fit and worst-fit events thrown away in the trimmed mean
trim_mean_absolute = 2  # For default batch size of 32
