#! /usr/bin/env python

################################################################################
# Imports
################################################################################

from __future__ import absolute_import, division, print_function

from os import sys, path
import argparse
import logging
import itertools
import numpy as np

from scipy.stats import norm#, crystalball

base_dir = path.abspath(path.join(path.dirname(__file__), '..'))
try:
    from higgs_inference import settings
except ImportError:
    if base_dir in sys.path:
        raise
    sys.path.append(base_dir)
    from higgs_inference import settings
settings.base_dir = base_dir


################################################################################
# Smearing functions
################################################################################

def smear(values, absolute=0., relative=0.):
    sigmas = absolute + relative * values
    return norm(loc=values, scale=sigmas).rvs(size=values.shape[0])


def smear_jet_energies(values, resolution_factor=0.5): #, beta=1.5, m=2.5):
    sigmas = resolution_factor * values ** 0.5
    return norm(loc=values, scale=sigmas).rvs(size=values.shape[0])
    # return crystalball(beta, m, loc=values, scale=sigmas).rvs(size=values.shape[0])


def smear_lepton_pt(values, resolution_factor=3.e-4):
    sigmas = resolution_factor * values ** 2.
    return norm(loc=values, scale=sigmas).rvs(size=values.shape[0])


################################################################################
# Conversion tools
################################################################################

def e_pt_eta_phi(canonical_momentum):
    e, px, py, pz = canonical_momentum.T

    pt = (px ** 2 + py ** 2) ** 0.5
    pabs = (px ** 2 + py ** 2 + pz ** 2) ** 0.5
    eta = np.arctanh(pz / pabs)
    phi = np.arctan2(py, px)

    return np.hstack((e.reshape((-1, 1)),
                      pt.reshape((-1, 1)),
                      eta.reshape((-1, 1)),
                      phi.reshape((-1, 1))))


def e_px_py_pz(momentum):
    e, pt, eta, phi = momentum.T

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    return np.hstack((e.reshape((-1, 1)),
                      px.reshape((-1, 1)),
                      py.reshape((-1, 1)),
                      pz.reshape((-1, 1))))


def sum_momenta(momenta):
    momentum_sum = np.zeros_like(momenta[0])
    for momentum in momenta:
        momentum_sum += e_px_py_pz(momentum)
    return e_pt_eta_phi(momentum_sum)


def calculate_m(momentum):
    e, pt, eta, phi = momentum.T

    pabs = pt * np.cosh(eta)
    m_squared = e ** 2 - pabs ** 2
    m_squared = np.clip(m_squared, 0., None)

    return np.sqrt(m_squared)


def calculate_pt_massless(momentum):
    e, _, eta, phi = momentum.T

    pt_massless = e / np.cosh(eta)
    pt_massless = np.clip(pt_massless, 0., None)

    return pt_massless


def calculate_e_massless(momentum):
    _, pt, eta, phi = momentum.T

    e_massless = pt * np.cosh(eta)
    e_massless = np.clip(e_massless, 0., None)

    return e_massless


################################################################################
# Main smearing function
################################################################################

def apply_smearing(filename, dry_run=False):
    logging.info('Applying smearing to sample %s', filename)

    # Load stuff
    try:
        X_true = np.load(settings.unweighted_events_dir + '/X_' + filename + '.npy')
    except IOError:
        logging.error('File %s not found', settings.unweighted_events_dir + '/X_' + filename + '.npy')
        return

    if X_true.shape[1] != 42:
        logging.error('File %s has wrong shape %s',
                      settings.unweighted_events_dir + '/X_' + filename + '.npy', X_true.shape)
        return

    init_number = -1.e9
    X_smeared = init_number * np.ones_like(X_true)

    # 0: jet1_E jet1_pt jet1_eta jet1_phi
    # 4: jet2_E jet2_pt jet2_eta jet2_phi
    # 8: lepton1_E lepton1_pt lepton1_eta lepton1_phi
    # 12: lepton2_E lepton2_pt lepton2_eta lepton2_phi
    # 16: lepton3_E lepton3_pt lepton3_eta lepton3_phi
    # 20: lepton4_E lepton4_pt lepton4_eta lepton4_phi
    # 24: higgs_E higgs_pt higgs_eta higgs_phi higgs_m
    # 29: Z1_E Z1_pt Z1_eta Z1_phi Z1_m
    # 34: Z2_E Z2_pt Z2_eta Z2_phi Z2_m
    # 39: m_jj deltaeta_jj deltaphi_jj

    # Jet energies
    indices = [0, 4]
    for i in indices:
        X_smeared[:, i] = smear_jet_energies(X_true[:, i])

    # Jet eta and phi
    indices = [2, 3, 6, 7]
    for i in indices:
        X_smeared[:, i] = smear(X_true[:, i], absolute=settings.smearing_eta_phi)

    # Derive jet pT, assuming zero jet mass (switch to smearing pT independently?)
    X_smeared[:, 1] = calculate_pt_massless(X_smeared[:, 0:4])
    X_smeared[:, 5] = calculate_pt_massless(X_smeared[:, 4:8])

    # Lepton momenta
    indices = [9, 13, 17, 21]
    for i in indices:
        X_smeared[:, i] = smear_lepton_pt(X_true[:, i])

    # Lepton eta and phi
    indices = [10, 11, 14, 15, 18, 19, 22, 23]
    X_smeared[:, indices] = X_true[:, indices]

    # Derive lepton energies, assuming zero lepton masses
    X_smeared[:, 8] = calculate_e_massless(X_smeared[:, 8:12])
    X_smeared[:, 12] = calculate_e_massless(X_smeared[:, 12:16])
    X_smeared[:, 16] = calculate_e_massless(X_smeared[:, 16:20])
    X_smeared[:, 20] = calculate_e_massless(X_smeared[:, 20:24])

    # Reconstruct Higgs momentum
    X_smeared[:, 24:28] = sum_momenta([X_smeared[:, 8:12],
                                       X_smeared[:, 12:16],
                                       X_smeared[:, 16:20],
                                       X_smeared[:, 20:24]])

    # Higgs mass
    X_smeared[:, 28] = calculate_m(X_smeared[:, 24:28])

    # Find pairings for Zs and reconstruct
    epsilon = 0.1
    found_pairing = np.full((X_true.shape[0],), False, dtype=bool)
    found_multiple_pairings = np.full((X_true.shape[0],), False, dtype=bool)
    for z1l1, z1l2, z2l1, z2l2 in itertools.permutations([0, 1, 2, 3]):
        candidate1 = sum_momenta([X_true[:, 8 + z1l1 * 4:12 + z1l1 * 4],
                                  X_true[:, 8 + z1l2 * 4:12 + z1l2 * 4]])
        candidate2 = sum_momenta([X_true[:, 8 + z2l1 * 4:12 + z2l1 * 4],
                                  X_true[:, 8 + z2l2 * 4:12 + z2l2 * 4]])

        match = (np.all(candidate1 - X_true[:, 29:33] < epsilon, axis=1)
                 & np.all(candidate2 - X_true[:, 34:38] < epsilon, axis=1))

        found_pairing = np.logical_or(found_pairing, match)
        found_multiple_pairings = np.logical_and(found_pairing, match)

        X_smeared[match, 29:33] = candidate1[match, :]
        X_smeared[match, 34:38] = candidate2[match, :]

    # Check we found pairings for each event
    if not np.all(found_pairing):
        logging.error('Could not find lepton pairing to reconstruct Z for %s events', np.sum(np.invert(found_pairing)))
    if np.any(found_multiple_pairings):
        logging.error('Found multiple lepton pairings to reconstruct Z for %s events', np.sum(found_multiple_pairings))

    # Z masses
    X_smeared[:, 33] = calculate_m(X_smeared[:, 29:33])
    X_smeared[:, 38] = calculate_m(X_smeared[:, 34:38])

    # Dijet observables
    X_smeared[:, 39] = calculate_m(sum_momenta([X_smeared[:, 0:4],
                                                X_smeared[:, 4:8]]))
    X_smeared[:, 40] = X_smeared[:, 2] - X_smeared[:, 6]
    X_smeared[:, 41] = X_smeared[:, 3] - X_smeared[:, 7]

    # Check for values that have not been replaces
    unchanged = (X_smeared < init_number + settings.epsilon) & (X_smeared > init_number - settings.epsilon)
    if np.sum(unchanged) > 0:
        logging.error('%s values of kinematic values have not been touched!', np.sum(unchanged))
        unchanged_columns = np.sum(unchanged, axis=0)
        for i, uc in enumerate(unchanged_columns):
            if uc > 0:
                logging.debug('  Feature %s: %s untouched values', i, uc)

    # Save result
    if not dry_run:
        np.save(settings.unweighted_events_dir + '/smeared_X_' + filename + '.npy', X_smeared)


################################################################################
# Parse arguments, set up logging
################################################################################

# Set up logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt='%d.%m.%Y %H:%M:%S')
logging.info('Welcome! How are you today?')

# Parse arguments
parser = argparse.ArgumentParser(description='Smearing data for Higgs inference experiments')

parser.add_argument("-t", "--train", action="store_true",
                    help="Smear baseline training sample")
parser.add_argument("-b", "--basis", action="store_true",
                    help="Smear morphing basis training sample")
parser.add_argument("-r", "--random", action="store_true",
                    help="Smear random theta training sample")
parser.add_argument("-p", "--pointbypoint", action="store_true",
                    help="Smear point-by-point training samples")
parser.add_argument("-s", "--scoreregression", action="store_true",
                    help="Smear score regression training sample")
parser.add_argument("-c", "--calibration", action="store_true",
                    help="Smear calibration sample")
parser.add_argument("-e", "--test", action="store_true",
                    help="Smear evaluation sample")
parser.add_argument("-n", "--neyman", action="store_true",
                    help="Smear samples for Neyman construction")
parser.add_argument("-x", "--roam", action="store_true",
                    help="Smear roaming evaluation sample")
parser.add_argument("--dry", action="store_true",
                    help="Don't save results")

args = parser.parse_args()

logging.info('Tasks:')
logging.info('  Baseline training:       %s', args.train)
logging.info('  Random training:         %s', args.random)
logging.info('  Morphing training:       %s', args.basis)
logging.info('  Point-by-point training: %s', args.pointbypoint)
logging.info('  Calibration:             %s', args.calibration)
logging.info('  Likelihood ratio eval:   %s', args.test)
logging.info('  Neyman construction:     %s', args.neyman)
logging.info('  Roaming:                 %s', args.roam)
logging.info('Options:')
logging.info('  Dry run:                 %s', args.dry)

################################################################################
# Go!
################################################################################

if args.train:
    apply_smearing('train', args.dry)

if args.basis:
    apply_smearing('train_basis', args.dry)

if args.pointbypoint:
    for t in settings.pbp_training_thetas:
        apply_smearing('train_point_by_point_' + str(t), args.dry)

if args.random:
    apply_smearing('train_random', args.dry)

if args.scoreregression:
    apply_smearing('train_scoreregression', args.dry)

if args.calibration:
    apply_smearing('calibration', args.dry)

if args.test:
    apply_smearing('train_test', args.dry)

if args.neyman:
    apply_smearing('neyman_observed', args.dry)
    for t in range(settings.n_thetas):
        apply_smearing('neyman_distribution_' + str(t), args.dry)

if args.roam:
    apply_smearing('roam', args.dry)

logging.info("That's it!")
