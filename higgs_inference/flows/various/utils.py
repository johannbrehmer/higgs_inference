import numpy as np
import autograd
import os
import logging
import sys
import inspect
import torch


def check_random_state(random_state, use_autograd=False):
    if random_state is None or isinstance(random_state, int):
        if use_autograd:
            return autograd.numpy.random.RandomState(random_state)
        return np.random.RandomState(random_state)
    else:
        return random_state


def general_init(debug=False):
    logging.basicConfig(format='%(asctime)s  %(message)s', datefmt='%H:%M')

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    logging.info('')
    logging.info('------------------------------------------------------------')
    logging.info('|                                                          |')
    logging.info('|  goldmine                                                |')
    logging.info('|                                                          |')
    logging.info('|              Experiments with simulator-based inference  |')
    logging.info('|                                                          |')
    logging.info('------------------------------------------------------------')
    logging.info('')

    logging.info('Hi! How are you today?')

    np.seterr(divide='ignore', invalid='ignore')

    np.set_printoptions(formatter={'float_kind': lambda x: "%.2f" % x})


def create_missing_folders(base_dir, simulator_name, inference_name=None):
    required_subfolders = ['thetas/' + simulator_name,
                           'samples/' + simulator_name]
    if inference_name is not None:
        required_subfolders += ['models/' + simulator_name + '/' + inference_name,
                                'results/' + simulator_name + '/' + inference_name]

    for subfolder in required_subfolders:
        folder = base_dir + '/goldmine/data/' + subfolder

        if not os.path.exists(folder):
            logging.info('Folder %s does not exist, will be created', folder)
            os.makedirs(folder)

        elif not os.path.isdir(folder):
            raise OSError('Path {} exists, but is no folder!'.format(folder))


def shuffle(*arrays):
    """ Shuffles multiple arrays simultaneously"""

    permutation = None
    n_samples = None
    shuffled_arrays = []

    for i, a in enumerate(arrays):
        if a is None:
            shuffled_arrays.append(a)
            continue

        if permutation is None:
            n_samples = a.shape[0]
            permutation = np.random.permutation(n_samples)

        if a.shape[0] != n_samples:
            logging.error('Mismatched shapes in shuffle:')
            for arr in arrays:
                if arr is None:
                    logging.info('  None')
                else:
                    logging.info('  Array with shape %s', arr.shape)
            raise RuntimeError('Mismatched shapes in shuffle')

        shuffled_a = a[permutation]
        shuffled_arrays.append(shuffled_a)

    return shuffled_arrays


def load_and_check(filename, warning_threshold=1.e9, replace_infs_with=np.exp(10.)):
    data = np.load(filename)

    n_nans = np.sum(np.isnan(data))
    n_infs = np.sum(np.isinf(data))
    n_finite = np.sum(np.isfinite(data))

    if n_nans + n_infs > 0:
        logging.warning('Warning: file %s contains %s NaNs and %s Infs, compared to %s finite numbers!',
                        filename, n_nans, n_infs, n_finite)

    if n_infs > 0 and replace_infs_with is not None:
        logging.info('Replacing %s  infinite values with %s', n_infs, replace_infs_with)
        data[np.isinf(data)] = replace_infs_with

    smallest = np.nanmin(data)
    largest = np.nanmax(data)

    if np.abs(smallest) > warning_threshold or np.abs(largest) > warning_threshold:
        logging.warning('Warning: file %s has some large numbers, rangin from %s to %s',
                        filename, smallest, largest)

    return data


def get_activation_function(activation_name):
    if activation_name == 'relu':
        return torch.relu
    elif activation_name == 'tanh':
        return torch.tanh
    elif activation_name == 'sigmoid':
        return torch.sigmoid
    else:
        raise ValueError('Activation function %s unknown', activation_name)


def discretize(data, discretization):
    for c in range(data.shape[1]):
        if discretization[c] is None or discretization[c] <= 0.:
            continue
        assert discretization[c] > 0.

        data[:, c] = np.round(data[:, c] / discretization[c], 0) * discretization[c]

    return data


def get_size(obj, seen=None):
    """ Recursively finds size of objects in bytes, from https://github.com/bosswissam/pysize/blob/master/pysize.py """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += get_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((get_size(v, seen) for v in obj.values()))
        size += sum((get_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum((get_size(i, seen) for i in obj))

    if hasattr(obj, '__slots__'):  # can have __slots__ with __dict__
        size += sum(get_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))

    return float(size)


def check_for_nans_in_parameters(model, check_gradients=True):
    for param in model.parameters():
        if torch.any(torch.isnan(param)):
            return True

        if check_gradients and torch.any(torch.isnan(param.grad)):
            return True

    return False
