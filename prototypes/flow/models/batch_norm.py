import numpy as np

import torch
from torch import Tensor
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter

dtype=np.float32


class BatchNorm(nn.Module):
    def __init__(self, n_units, alpha=0.1, eps=1.e-5):

        super(BatchNorm, self).__init__()

        self.n_units = n_units
        self.alpha = alpha
        self.eps = eps

        # parameters
        self.log_gamma = Parameter(Tensor(np.zeros(n_units, dtype=dtype)))
        self.beta = Parameter(Tensor(np.zeros(n_units, dtype=dtype)))

        # Running averages: will be created at first call of forward
        self.running_mean = None
        self.running_var = None

    def forward(self, x, fixed_params=False):

        """ Calculates u(x) """

        # minibatch statistics
        if fixed_params:
            self.mean = self.running_mean
            self.var = self.running_var
        else:
            self.mean = Variable(torch.zeros(self.n_units))  # torch.mean(x, dim=0)
            self.var = 2. * Variable(torch.ones(self.n_units))  # torch.mean((x - self.mean) ** 2, dim=0) + self.eps

            # keep track of running mean and var (for u -> x direction)
            if self.running_mean is None:
                self.running_mean = Variable(torch.zeros(self.n_units))
                self.running_var = Variable(torch.zeros(self.n_units))
                self.running_mean += self.mean
                self.running_var += self.var
            else:
                self.running_mean = (1. - self.alpha) * self.running_mean + self.alpha * self.mean
                self.running_var = (1. - self.alpha) * self.running_var + self.alpha * self.var

        # transformation
        x_hat = (x - self.mean) / torch.sqrt(self.var)

        u = torch.exp(self.log_gamma) * x_hat + self.beta

        return u

    def inverse(self, y):
        """
        Evaluates the inverse batch norm transformation for output y.
        NOTE: this calculation is done with numpy and not with theano.
        :param y: output as numpy array
        :return: input as numpy array
        """

        x_hat = (y - self.beta.data.numpy()) * np.exp(-self.log_gamma.data.numpy())
        x = np.sqrt(self.running_var.data.numpy()) * x_hat + self.running_mean.data.numpy()

        return x