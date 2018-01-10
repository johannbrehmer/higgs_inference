#! /usr/bin/env python

import argparse

from truth import truth_inference
from parameterized import parameterized_inference
from point_by_point import point_by_point_inference

# Parse arguments
parser = argparse.ArgumentParser(description='Inference experiments for Higgs EFT measurements')

parser.add_argument('algorithm', help='Algorithm type. Options are "truth", "carl", "score", '
                                      + '"combined" (carl + score), "regression", or "combinedregression" '
                                      + '(regression + score).')
parser.add_argument("-pbp", "--pointbypoint", action="store_true",
                    help="Point-by-point rather than parameterized setup.")
parser.add_argument("-a", "--aware", action="store_true",
                    help="Physics-aware parameterized setup.")
parser.add_argument("-t", "--training", default='baseline', help='Training sample: "baseline", "basis", or "random".')
parser.add_argument("-o", "--options", nargs='+', default='', help="Further options to be passed on to the algorithm.")

args = parser.parse_args()

# Sanity checks
assert args.algorithm in ['truth', 'carl', 'score', 'combined', 'regression', 'combinedregression']
assert args.training in ['baseline', 'basis', 'random']

if args.algorithm == 'truth':
    truth_inference(options=args.options)

elif args.pointbypoint:
    point_by_point_inference(algorithm=args.algorithm,
                             options=args.options)

else:
    parameterized_inference(algorithm=args.algorithm,
                            morphing_aware=args.aware,
                            training_sample=args.training,
                            options=args.options)
