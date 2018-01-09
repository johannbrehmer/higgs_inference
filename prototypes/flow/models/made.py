import numpy as np
import numpy.random as rng

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

dtype=np.float32


def create_degrees(n_inputs, n_hiddens, input_order, mode):
    """
    Generates a degree for each hidden and input unit. A unit with degree d can only receive input from units with
    degree less than d.
    :param n_inputs: the number of inputs
    :param n_hiddens: a list with the number of hidden units
    :param input_order: the order of the inputs; can be 'random', 'sequential', or an array of an explicit order
    :param mode: the strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
    :return: list of degrees
    """

    degrees = []

    # create degrees for inputs
    if isinstance(input_order, str):

        if input_order == 'random':
            degrees_0 = np.arange(1, n_inputs + 1)
            rng.shuffle(degrees_0)

        elif input_order == 'sequential':
            degrees_0 = np.arange(1, n_inputs + 1)

        else:
            raise ValueError('invalid input order')

    else:
        input_order = np.array(input_order)
        assert np.all(np.sort(input_order) == np.arange(1, n_inputs + 1)), 'invalid input order'
        degrees_0 = input_order
    degrees.append(degrees_0)

    # create degrees for hiddens
    if mode == 'random':
        for N in n_hiddens:
            min_prev_degree = min(np.min(degrees[-1]), n_inputs - 1)
            degrees_l = rng.randint(min_prev_degree, n_inputs, N)
            degrees.append(degrees_l)

    elif mode == 'sequential':
        for N in n_hiddens:
            degrees_l = np.arange(N) % max(1, n_inputs - 1) + min(1, n_inputs - 1)
            degrees.append(degrees_l)

    else:
        raise ValueError('invalid mode')

    return degrees


def create_masks(degrees):
    """
    Creates the binary masks that make the connectivity autoregressive.
    :param degrees: a list of degrees for every layer
    :return: list of all masks, as theano shared variables
    """

    Ms = []

    for l, (d0, d1) in enumerate(zip(degrees[:-1], degrees[1:])):
        M = d0[:, np.newaxis] <= d1
        M = Variable(torch.Tensor(M.astype(dtype)))  # ?
        Ms.append(M)

    Mmp = degrees[-1][:, np.newaxis] < degrees[0]
    Mmp = Variable(torch.Tensor(Mmp.astype(dtype)))

    return Ms, Mmp


def create_weights(n_inputs, n_hiddens, n_comps=None):
    """
    Creates all learnable weight matrices and bias vectors.
    :param n_inputs: the number of inputs
    :param n_hiddens: a list with the number of hidden units
    :param n_comps: number of gaussian components
    :return: weights and biases, as theano shared variables
    """

    Ws = []
    bs = []

    n_units = np.concatenate(([n_inputs], n_hiddens))

    for N0, N1 in zip(n_units[:-1], n_units[1:]):
        W = nn.Parameter(torch.Tensor((rng.randn(N0, N1) / np.sqrt(N0 + 1)).astype(dtype)))
        b = nn.Parameter(torch.Tensor(np.zeros(N1, dtype=dtype)))
        Ws.append(W)
        bs.append(b)

    if n_comps is None:
        Wm = nn.Parameter(torch.Tensor((rng.randn(n_units[-1], n_inputs)
                                        / np.sqrt(n_units[-1] + 1)).astype(dtype)))
        Wp = nn.Parameter(torch.Tensor((rng.randn(n_units[-1], n_inputs)
                                        / np.sqrt(n_units[-1] + 1)).astype(dtype)))
        bm = nn.Parameter(torch.Tensor(np.zeros(n_inputs, dtype=dtype)))
        bp = nn.Parameter(torch.Tensor(np.zeros(n_inputs, dtype=dtype)))

        return Ws, bs, Wm, bm, Wp, bp
    else:

        Wm = nn.Parameter(torch.Tensor((rng.randn(n_units[-1], n_inputs, n_comps)
                                        / np.sqrt(n_units[-1] + 1)).astype(dtype)))
        Wp = nn.Parameter(torch.Tensor((rng.randn(n_units[-1], n_inputs, n_comps)
                                        / np.sqrt(n_units[-1] + 1)).astype(dtype)))
        Wa = nn.Parameter(torch.Tensor((rng.randn(n_units[-1], n_inputs, n_comps)
                                        / np.sqrt(n_units[-1] + 1)).astype(dtype)))
        bm = nn.Parameter(torch.Tensor(rng.randn(n_inputs, n_comps).astype(dtype)))
        bp = nn.Parameter(torch.Tensor(rng.randn(n_inputs, n_comps).astype(dtype)))
        ba = nn.Parameter(torch.Tensor(rng.randn(n_inputs, n_comps).astype(dtype)))

        return Ws, bs, Wm, bm, Wp, bp, Wa, ba


class GaussianMADE(nn.Module):

    def __init__(self, n_inputs, n_hiddens, input_order='sequential',
                 mode='sequential'):  # , input=None):

        """
        Constructor.
        :param n_inputs: number of inputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param act_fun: name of activation function
        :param input_order: order of inputs
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        :param input: theano variable to serve as input; if None, a new variable is created
        """

        super(GaussianMADE, self).__init__()

        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.mode = mode

        # create network's parameters
        self.degrees = create_degrees(n_inputs, n_hiddens, input_order, mode)
        self.Ms, self.Mmp = create_masks(self.degrees)
        self.Ws, self.bs, self.Wm, self.bm, self.Wp, self.bp = create_weights(n_inputs, n_hiddens, None)
        self.input_order = self.degrees[0]

        # Output info
        self.m = None
        self.logp = None
        self.log_likelihood = None


    def forward(self, x):

        """ Transforms x into u = f^-1(x) """

        h = x

        # feedforward propagation
        for M, W, b in zip(self.Ms, self.Ws, self.bs):
            h = F.relu(F.linear(h, torch.t(M * W), b))

        # output means
        self.m = F.linear(h, torch.t(self.Mmp * self.Wm), self.bm)

        # output log precisions
        self.logp = F.linear(h, torch.t(self.Mmp * self.Wp), self.bp)

        # random numbers driving made
        u = torch.exp(0.5 * self.logp) * (x - self.m)

        # log likelihoods
        diff = torch.sum(u ** 2 - self.logp, dim=1)
        constant = float(self.n_inputs * np.log(2. * np.pi))
        self.log_likelihood = -0.5 * (constant + diff)

        return u


    def eval(self, x):

        """ Calculates log p(x) """

        u = self.forward(x)

        return self.log_likelihood


    def gen(self, n_samples=1, u=None):
        """
        Generate samples from made. Requires as many evaluations as number of inputs.
        :param n_samples: number of samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        x = np.zeros([n_samples, self.n_inputs], dtype=dtype)
        u = rng.randn(n_samples, self.n_inputs).astype(dtype) if u is None else u

        for i in range(1, self.n_inputs + 1):
            self.forward(Variable(torch.Tensor(x))) # Sets Gaussian parameters: self.m and self.logp
            m = self.m.data.numpy()
            logp = self.logp.data.numpy()

            idx = np.argwhere(self.input_order == i)[0, 0]
            x[:, idx] = m[:, idx] + np.exp(np.minimum(-0.5 * logp[:, idx], 10.0)) * u[:, idx]

        return x