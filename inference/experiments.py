#! /usr/bin/env python

from __future__ import absolute_import, division, print_function

import argparse
import logging

from .strategies.truth import truth_inference
from .strategies.parameterized import parameterized_inference
from .strategies.point_by_point import point_by_point_inference
from .strategies.score_regression import score_regression_inference

# Set up logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt='%d.%m.%Y %H:%M:%S')
logging.info('Welcome! How are you today?')

# Parse arguments
parser = argparse.ArgumentParser(description='Inference experiments for Higgs EFT measurements')

parser.add_argument('algorithm', help='Algorithm type. Options are "truth", "carl", "score" (in the carl setup), '
                                      + '"combined" (carl + score), "regression", "combinedregression" '
                                      + '(regression + score), or "scoreregression" (regresses on the score and'
                                      + 'performs density estimation on theta.score.')
parser.add_argument("-pbp", "--pointbypoint", action="store_true",
                    help="Point-by-point rather than parameterized setup.")
parser.add_argument("-a", "--aware", action="store_true",
                    help="Physics-aware parameterized setup.")
parser.add_argument("-t", "--training", default='baseline', help='Training sample: "baseline", "basis", or "random".')
parser.add_argument("-o", "--options", nargs='+', default='', help="Further options to be passed on to the algorithm.")

args = parser.parse_args()

logging.info('The algorithm of the day is: %s', args.algorithm)

# Sanity checks
assert args.algorithm in ['truth', 'carl', 'score', 'combined', 'regression', 'combinedregression', 'scoreregression']
assert args.training in ['baseline', 'basis', 'random']

# Start calculation
if args.algorithm == 'truth':
    truth_inference(options=args.options)

elif args.algorithm == 'scoreregression':
    score_regression_inference(options=args.options)

elif args.pointbypoint:
    point_by_point_inference(algorithm=args.algorithm,
                             options=args.options)

else:
    parameterized_inference(algorithm=args.algorithm,
                            morphing_aware=args.aware,
                            training_sample=args.training,
                            options=args.options)

logging.info("That's it -- have a great day!")
