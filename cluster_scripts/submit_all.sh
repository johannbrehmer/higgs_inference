#!/bin/bash

cd /home/jb6504/higgs_inference/cluster_scripts

sbatch run_truth.sh
sbatch run_scoreregression.sh
sbatch run_afc.sh

sleep 30

sbatch run_carl_point_by_point.sh
sbatch run_regression_point_by_point.sh
sbatch run_carl.sh
sbatch run_combined.sh
sbatch run_regression.sh
sbatch run_combinedregression.sh

# sleep 30

# sbatch run_carl_aware.sh
# sbatch run_combined_aware.sh
# sbatch run_regression_aware.sh
# sbatch run_combinedregression_aware.sh
