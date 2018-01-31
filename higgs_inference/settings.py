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
# Parameters and features
################################################################################

n_params = 2
n_features = 42
n_thetas_features = n_params + n_features
n_morphing_samples = 15

################################################################################
# Benchmark theta points
################################################################################

try:
    thetas = np.load(base_dir + '/data/thetas/thetas_parameterized.npy')
except:
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
alpha_regression_default = 0.011
alpha_carl_default = 0.35
alpha_regression_small = 0.005
alpha_carl_small = 0.1

# Training length
n_epochs_short = 1
n_epochs_default = 20
n_epochs_long = 50
early_stopping_patience = 3
validation_split = 0.1


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
n_events_n_point_by_point_num = 500000
n_events_n_point_by_point_den = 500000

# Score regression training, total
n_events_score_regression = 10000000

# Calibration, per theta
n_events_calibration = 1000

# Evaluation, total
n_events_test = 50000

# Roaming, total
n_events_roam = 20
