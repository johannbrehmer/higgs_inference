#! /usr/bin/env python

################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

import argparse
import logging
from os import sys, path

base_dir = path.abspath(path.join(path.dirname(__file__), '..'))

try:
    from higgs_inference import settings
except ImportError:
    if base_dir in sys.path:
        raise
    sys.path.append(base_dir)
    from higgs_inference import settings

from higgs_inference.various.p_values import calculate_all_CL
from higgs_inference.strategies.truth import truth_inference

try:
    from higgs_inference.strategies.parameterized import parameterized_inference
    from higgs_inference.strategies.point_by_point import point_by_point_inference
    from higgs_inference.strategies.score_regression import score_regression_inference
    loaded_ml_strategies = True
except ImportError:
    loaded_ml_strategies = False

################################################################################
# Set up logging and parse arguments
################################################################################

settings.base_dir = base_dir

# Set up logging
logging.basicConfig(format='%(asctime)s %(levelname)s    %(message)s', level=logging.DEBUG, datefmt='%d.%m.%Y %H:%M:%S')
logging.info('Hi! How are you today?')

# Parse arguments
parser = argparse.ArgumentParser(description='Inference experiments for Higgs EFT measurements')

parser.add_argument('algorithm', help='Algorithm type. Options are "p" or "cl" for the calculation of p values '
                                      + 'through the Neyman construction; "truth", "carl", "score" (in the carl setup), '
                                      + '"combined" (carl + score), "regression", "combinedregression" '
                                      + '(regression + score), or "scoreregression" (regresses on the score and '
                                      + 'performs density estimation on theta.score.')
parser.add_argument("-pbp", "--pointbypoint", action="store_true",
                    help="Point-by-point rather than parameterized setup.")
parser.add_argument("-a", "--aware", action="store_true",
                    help="Physics-aware parameterized setup.")
parser.add_argument("-t", "--training", default='baseline', help='Training sample: "baseline", "basis", or "random".')
parser.add_argument("-o", "--options", nargs='+', default='', help="Further options to be passed on to the algorithm.")

args = parser.parse_args()

logging.info('Startup options:')
logging.info('  Algorithm:                     %s', args.algorithm)
logging.info('  Point by point:                %s', args.pointbypoint)
logging.info('  Morphing-aware mode:           %s', args.aware)
logging.info('  Training sample:               %s', args.training)
logging.info('  Other options:                 %s', args.options)
logging.info('  Base directory:                %s', settings.base_dir)
logging.info('  ML-based strategies available: %s', loaded_ml_strategies)

# Sanity checks
assert args.algorithm in ['p', 'cl', 'pvalues',
                          'truth', 'carl', 'score', 'combined', 'regression', 'combinedregression', 'scoreregression']
assert args.training in ['baseline', 'basis', 'random']

################################################################################
# Do something
################################################################################

# Start calculation
if args.algorithm in ['p', 'cl', 'pvalues']:
    calculate_all_CL()

elif args.algorithm == 'truth':
    truth_inference(options=args.options)

elif args.algorithm == 'scoreregression':
    assert loaded_ml_strategies
    score_regression_inference(options=args.options)

elif args.pointbypoint:
    assert loaded_ml_strategies
    point_by_point_inference(algorithm=args.algorithm,
                             options=args.options)

else:
    assert loaded_ml_strategies
    parameterized_inference(algorithm=args.algorithm,
                            morphing_aware=args.aware,
                            training_sample=args.training,
                            options=args.options)

logging.info("That's it -- have a great day!")
