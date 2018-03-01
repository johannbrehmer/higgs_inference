#!/bin/bash

cd /home/jb6504/higgs_inference/cluster_scripts

################################################################################
# Default settings
################################################################################

# sbatch run_truth.sh

sbatch run_carl.sh
sbatch run_combined.sh
sbatch run_regression.sh
sbatch run_combinedregression.sh

# sbatch run_carl_point_by_point.sh
# sbatch run_regression_point_by_point.sh

# sbatch run_scoreregression.sh

sbatch run_histo.sh
# sbatch run_afc.sh

# sleep 30


################################################################################
# Hyperparameter scans
################################################################################

# sbatch run_combined_tuning.sh
# sbatch run_combinedregression_tuning.sh

# sbatch run_carl_depth_experiments.sh
# sbatch run_combined_depth_experiments.sh
# sbatch run_regression_depth_experiments.sh
# sbatch run_combinedregression_depth_experiments.sh

# sbatch run_carl_learning_experiments.sh
# sbatch run_regression_learning_experiments.sh

sleep 30


################################################################################
# Smearing
################################################################################

# sbatch run_carl_smearing.sh
# sbatch run_combined_smearing.sh
# sbatch run_regression_smearing.sh
# sbatch run_combinedregression_smearing.sh

# sbatch run_carl_point_by_point_smearing.sh
# sbatch run_regression_point_by_point_smearing.sh

# sbatch run_scoreregression_smearing.sh

# sbatch run_afc_smearing.sh
# sbatch run_histo_smearing.sh

# sleep 30


################################################################################
# Physics-aware
################################################################################

# sbatch run_carl_aware.sh
# sbatch run_combined_aware.sh
# sbatch run_regression_aware.sh
# sbatch run_combinedregression_aware.sh

# sbatch run_carl_aware2.sh
# sbatch run_combined_aware2.sh
# sbatch run_regression_aware2.sh
# sbatch run_combinedregression_aware2.sh
