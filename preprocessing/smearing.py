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

from scipy.stats import norm  # , crystalball

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
# Debug output
################################################################################

def get_statistics(values):
    content = str(values)

    if values.ndim <= 1:
        mean = str(np.mean(values))
        val_range = str(np.nanmin(values)) + '...' + str(np.nanmax(values))
        content = content.replace('\n', '')

    else:
        mean = str(np.mean(values, axis=0))
        val_range = '[ '
        for vmin, vmax in zip(np.nanmin(values, axis=0), np.nanmax(values, axis=0)):
            if len(val_range) > 2:
                val_range += '  '
            val_range += str(vmin) + '...' + str(vmax)
        val_range += ' ]'
        content = '\n        ' + content.replace('\n', '\n        ')

    output = ('shape ' + str(values.shape)
              + ', ' + str(np.sum(np.invert(np.isfinite(values))))
              + ' NaNs, mean ' + mean
              + ', range ' + val_range
              + ', content ' + content)
    return output


################################################################################
# Sanitization functions
################################################################################

def sanitize_eta(values):
    values[np.invert(np.isfinite(values))] = 0.
    return np.clip(values, -12., 12.)


def sanitize_phi(values):
    values[np.invert(np.isfinite(values))] = 0.
    while np.sum(values < - np.pi) > 0:
        values[values < -np.pi] += 2. * np.pi
    while np.sum(values > np.pi) > 0:
        values[values > np.pi] -= 2. * np.pi
    return values


def sanitize_energies(values):
    values[np.invert(np.isfinite(values))] = 13000.
    return np.clip(values, 0., 13000.)


def sanitize_sigmas(values):
    values[np.invert(np.isfinite(values))] = 100.
    return np.clip(values, 0., 100.)


################################################################################
# Smearing functions
################################################################################

def smear_eta(values, absolute=0., relative=0.):
    values = sanitize_eta(values)

    sigmas = absolute + relative * values
    sigmas = sanitize_sigmas(sigmas)

    smeared_values = norm(loc=values, scale=sigmas).rvs(size=values.shape[0])
    smeared_values = sanitize_eta(smeared_values)

    # logging.debug('Eta smearing with absolute uncertainty %s and relative uncertainty %s', absolute, relative)
    # logging.debug('  Before: %s', get_statistics(values))
    # logging.debug('  After:  %s', get_statistics(smeared_values))

    return smeared_values


def smear_phi(values, absolute=0., relative=0.):
    values = sanitize_phi(values)

    sigmas = absolute + relative * values
    sigmas = sanitize_sigmas(sigmas)

    smeared_values = norm(loc=values, scale=sigmas).rvs(size=values.shape[0])
    smeared_values = sanitize_phi(smeared_values)

    # logging.debug('Eta smearing with absolute uncertainty %s and relative uncertainty %s', absolute, relative)
    # logging.debug('  Before: %s', get_statistics(values))
    # logging.debug('  After:  %s', get_statistics(smeared_values))

    return smeared_values


def smear_jet_energies(values, resolution_factor=0.5):  # , beta=1.5, m=2.5):

    values = sanitize_energies(values)

    sigmas = resolution_factor * values ** 0.5
    sigmas = sanitize_sigmas(sigmas)

    # smeared_values = crystalball(beta, m, loc=values, scale=sigmas).rvs(size=values.shape[0])
    smeared_values = norm(loc=values, scale=sigmas).rvs(size=values.shape[0])
    smeared_values = sanitize_energies(smeared_values)

    # logging.debug('Jet energy smearing with resolution %s', resolution_factor)
    # logging.debug('  Before: %s', get_statistics(values))
    # logging.debug('  After:  %s', get_statistics(smeared_values))

    return smeared_values


def smear_lepton_pt(values, resolution_factor=3.e-4):
    values = sanitize_energies(values)

    sigmas = resolution_factor * values ** 2.
    sigmas = sanitize_sigmas(sigmas)

    smeared_values = norm(loc=values, scale=sigmas).rvs(size=values.shape[0])
    smeared_values = sanitize_energies(smeared_values)

    # logging.debug('Lepton pT smearing with resolution %s', resolution_factor)
    # logging.debug('  Before: %s', get_statistics(values))
    # logging.debug('  After:  %s', get_statistics(smeared_values))

    return smeared_values


################################################################################
# Conversion tools
################################################################################

def e_pt_eta_phi(canonical_momentum):
    e, px, py, pz = canonical_momentum.T

    pt = sanitize_energies((px ** 2 + py ** 2) ** 0.5)
    pabs = (px ** 2 + py ** 2 + pz ** 2) ** 0.5
    pabs = np.clip(pabs, 1.e-9, 8000.)
    eta = sanitize_eta(np.arctanh(pz / pabs))
    phi = sanitize_phi(np.arctan2(py, px))

    output = np.hstack((e.reshape((-1, 1)),
                        pt.reshape((-1, 1)),
                        eta.reshape((-1, 1)),
                        phi.reshape((-1, 1))))

    # logging.debug('Conversion from E,px,py,pz to E,pT,eta,phi:')
    # logging.debug('  Input: %s', get_statistics(canonical_momentum))
    # logging.debug('  E:     %s', get_statistics(e))
    # logging.debug('  px:    %s', get_statistics(px))
    # logging.debug('  py:    %s', get_statistics(py))
    # logging.debug('  pz:    %s', get_statistics(pz))
    # logging.debug('  pt:    %s', get_statistics(pt))
    # logging.debug('  pabs:  %s', get_statistics(pabs))
    # logging.debug('  eta:   %s', get_statistics(eta))
    # logging.debug('  phi :  %s', get_statistics(phi))
    # logging.debug('  Output:%s', get_statistics(output))

    return output


def e_px_py_pz(momentum):
    e, pt, eta, phi = momentum.T

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    output = np.hstack((e.reshape((-1, 1)),
                        px.reshape((-1, 1)),
                        py.reshape((-1, 1)),
                        pz.reshape((-1, 1))))

    # logging.debug('Conversion from E,pT,eta,phi to E,px,py,pz:')
    # logging.debug('  Input: %s', get_statistics(momentum))
    # logging.debug('  E:     %s', get_statistics(e))
    # logging.debug('  pt:    %s', get_statistics(pt))
    # logging.debug('  eta:   %s', get_statistics(eta))
    # logging.debug('  phi :  %s', get_statistics(phi))
    # logging.debug('  px:    %s', get_statistics(px))
    # logging.debug('  py:    %s', get_statistics(py))
    # logging.debug('  pz:    %s', get_statistics(pz))
    # logging.debug('  Output:%s', get_statistics(output))

    return output


def sum_momenta(momenta):
    momentum_sum = np.zeros_like(momenta[0])
    for momentum in momenta:
        momentum_sum += e_px_py_pz(momentum)
    return e_pt_eta_phi(momentum_sum)


def calculate_m(momentum):
    e, pt, eta, phi = momentum.T

    e = sanitize_energies(e)
    eta = sanitize_eta(eta)
    pabs = sanitize_energies(pt * np.cosh(eta))
    m_squared = e ** 2 - pabs ** 2
    m_squared = np.clip(m_squared, 0., None)

    return np.sqrt(m_squared)


def calculate_pt_massless(momentum):
    e, _, eta, phi = momentum.T

    e = sanitize_energies(e)
    eta = sanitize_eta(eta)
    pt_massless = sanitize_energies(e / np.cosh(eta))

    return pt_massless


def calculate_e_massless(momentum):
    _, pt, eta, phi = momentum.T

    pt = sanitize_energies(pt)
    eta = sanitize_eta(eta)
    e_massless = sanitize_energies(pt * np.cosh(eta))

    return e_massless


################################################################################
# Main smearing function
################################################################################

def apply_smearing(filename, dry_run=False):
    logging.info('Applying smearing to sample %s', filename)

    # For debugging
    def print_statistics(label, index):
        logging.debug(label)
        logging.debug('  Before: %s', get_statistics(X_true[:, index]))
        logging.debug('  After:  %s', get_statistics(X_smeared[:, index]))

    # Load stuff
    try:
        X_true = np.load(settings.unweighted_events_dir + '/X_' + filename + '.npy').astype(np.float64)
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
        X_smeared[:, i] = smear_jet_energies(X_true[:, i], resolution_factor=settings.smearing_jet_energies)
    print_statistics('E(j1)', 0)
    print_statistics('E(j2)', 4)

    # Jet eta
    indices = [2, 6]
    for i in indices:
        X_smeared[:, i] = smear_eta(X_true[:, i], absolute=settings.smearing_eta_phi)
    print_statistics('eta(j1)', 2)
    print_statistics('eta(j2)', 6)

    # Jet phi
    indices = [3, 7]
    for i in indices:
        X_smeared[:, i] = smear_phi(X_true[:, i], absolute=settings.smearing_eta_phi)
    print_statistics('phi(j1)', 3)
    print_statistics('phi(j2)', 7)

    # Derive jet pT, assuming zero jet mass (switch to smearing pT independently?)
    X_smeared[:, 1] = calculate_pt_massless(X_smeared[:, 0:4])
    X_smeared[:, 5] = calculate_pt_massless(X_smeared[:, 4:8])
    print_statistics('pT(j1)', 1)
    print_statistics('pT(j2)', 5)

    # Lepton momenta
    indices = [9, 13, 17, 21]
    for i in indices:
        X_smeared[:, i] = smear_lepton_pt(X_true[:, i], resolution_factor=settings.smearing_lepton_pt)
    print_statistics('pT(l1)', 9)
    print_statistics('pT(l2)', 13)
    print_statistics('pT(l3)', 17)
    print_statistics('pT(l4)', 21)

    # Lepton eta
    indices = [10, 14, 18, 22]
    X_smeared[:, indices] = X_true[:, indices]
    print_statistics('eta(l1)', 10)
    print_statistics('eta(l2)', 14)
    print_statistics('eta(l3)', 18)
    print_statistics('eta(l4)', 22)

    # Lepton phi
    indices = [11, 15, 19, 23]
    X_smeared[:, indices] = X_true[:, indices]
    print_statistics('phi(l1)', 11)
    print_statistics('phi(l2)', 15)
    print_statistics('phi(l3)', 19)
    print_statistics('phi(l4)', 23)

    # Derive lepton energies, assuming zero lepton masses
    X_smeared[:, 8] = calculate_e_massless(X_smeared[:, 8:12])
    X_smeared[:, 12] = calculate_e_massless(X_smeared[:, 12:16])
    X_smeared[:, 16] = calculate_e_massless(X_smeared[:, 16:20])
    X_smeared[:, 20] = calculate_e_massless(X_smeared[:, 20:24])
    print_statistics('E(l1)', 8)
    print_statistics('E(l2)', 12)
    print_statistics('E(l3)', 16)
    print_statistics('E(l4)', 20)

    # Reconstruct Higgs momentum
    X_smeared[:, 24:28] = sum_momenta([X_smeared[:, 8:12],
                                       X_smeared[:, 12:16],
                                       X_smeared[:, 16:20],
                                       X_smeared[:, 20:24]])
    print_statistics('E(H)', 24)
    print_statistics('pT(H)', 25)
    print_statistics('eta(H)', 26)
    print_statistics('phi(H)', 27)

    # Higgs mass
    X_smeared[:, 28] = calculate_m(X_smeared[:, 24:28])
    print_statistics('m(H)', 28)

    # Find pairings for Zs and reconstruct
    logging.debug('Looking for Z reconstructions')

    best_distance = 1.e9 * np.ones((X_true.shape[0],))
    e_factor = 1. / 100.
    pt_factor = 1. / 100.
    phi_factor = 1. / 3.
    eta_factor = 1. / 2.

    for z1l1, z1l2, z2l1, z2l2 in itertools.permutations([0, 1, 2, 3]):

        # Avoid overcounting equivalent combinations
        if not (z1l1 > z1l2 and z2l1 > z2l2):
            continue

        # Construct candidates
        candidate1 = sum_momenta([X_true[:, 8 + z1l1 * 4:12 + z1l1 * 4],
                                  X_true[:, 8 + z1l2 * 4:12 + z1l2 * 4]])
        candidate2 = sum_momenta([X_true[:, 8 + z2l1 * 4:12 + z2l1 * 4],
                                  X_true[:, 8 + z2l2 * 4:12 + z2l2 * 4]])

        # logging.debug('True combination 1: %s', get_statistics(X_true[:,29:33]))
        # logging.debug('Candidate 1: %s', get_statistics(candidate1))
        # logging.debug('True combination 2: %s', get_statistics(X_true[:,34:38]))
        # logging.debug('Candidate 2: %s', get_statistics(candidate2))

        # See if they match
        distance = (
                e_factor * (candidate1[:, 0] - X_true[:, 29]) ** 2
                + pt_factor * (candidate1[:, 1] - X_true[:, 30]) ** 2
                + eta_factor * (candidate1[:, 2] - X_true[:, 31]) ** 2
                + phi_factor * (candidate1[:, 3] - X_true[:, 32]) ** 2
                + e_factor * (candidate2[:, 0] - X_true[:, 34]) ** 2
                + pt_factor * (candidate2[:, 1] - X_true[:, 35]) ** 2
                + eta_factor * (candidate2[:, 2] - X_true[:, 36]) ** 2
                + phi_factor * (candidate2[:, 3] - X_true[:, 37]) ** 2
        )
        match = distance < best_distance
        best_distance[match] = distance[match]

        # For these combinations, reconstruct smearing
        X_smeared[match, 29:33] = sum_momenta([X_smeared[match, 8 + z1l1 * 4:12 + z1l1 * 4],
                                               X_smeared[match, 8 + z1l2 * 4:12 + z1l2 * 4]])
        X_smeared[match, 34:38] = sum_momenta([X_smeared[match, 8 + z2l1 * 4:12 + z2l1 * 4],
                                               X_smeared[match, 8 + z2l2 * 4:12 + z2l2 * 4]])

        logging.debug('  Pairing %s  ->  %s / %s events improve', (z1l1, z1l2, z2l1, z2l2), np.sum(match),
                      X_true.shape[0])

    logging.debug('Reconstruction distance measure: %s', get_statistics(distance))

    print_statistics('E(Z1)', 29)
    print_statistics('pT(Z1)', 30)
    print_statistics('eta(Z1)', 31)
    print_statistics('phi(Z1)', 32)
    print_statistics('E(Z2)', 34)
    print_statistics('pT(Z2)', 35)
    print_statistics('eta(Z2)', 36)
    print_statistics('phi(Z2)', 37)

    # Z masses
    X_smeared[:, 33] = calculate_m(X_smeared[:, 29:33])
    X_smeared[:, 38] = calculate_m(X_smeared[:, 34:38])
    print_statistics('m(Z1)', 33)
    print_statistics('m(Z2)', 38)

    # Dijet observables
    X_smeared[:, 39] = sanitize_energies(calculate_m(sum_momenta([X_smeared[:, 0:4],
                                                                  X_smeared[:, 4:8]])))
    X_smeared[:, 40] = np.absolute(sanitize_eta(X_smeared[:, 2] - X_smeared[:, 6]))
    X_smeared[:, 41] = np.absolute(sanitize_phi(X_smeared[:, 3] - X_smeared[:, 7]))
    print_statistics('m(jj)', 39)
    print_statistics('delta eta(jj)', 40)
    print_statistics('delta phi(jj)', 41)

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
logging.basicConfig(format='%(asctime)s %(levelname)s    %(message)s', level=logging.DEBUG, datefmt='%d.%m.%Y %H:%M:%S')
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
