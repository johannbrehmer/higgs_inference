import numpy as np
import logging
import torch
from torch import tensor

from higgs_inference.flows.inference.base import Inference
from higgs_inference.flows.ml.models.maf import ConditionalMaskedAutoregressiveFlow
from higgs_inference.flows.ml.trainer import train_model
from higgs_inference.flows.ml.losses import negative_log_likelihood
from higgs_inference.flows.various.utils import expand_array_2d


class MAFInference(Inference):
    """ Neural conditional density estimation with masked autoregressive flows. """

    def __init__(self, **params):
        super().__init__()

        filename = params.get('filename', None)

        if filename is None:
            # Parameters for new MAF
            n_parameters = params['n_parameters']
            n_observables = params['n_observables']
            n_mades = params.get('n_mades', 2)
            n_made_hidden_layers = params.get('n_made_hidden_layers', 2)
            n_made_units_per_layer = params.get('n_made_units_per_layer', 20)
            activation = params.get('activation', 'relu')
            batch_norm = params.get('batch_norm', False)

            logging.info('Initialized NDE (MAF) with the following settings:')
            logging.info('  Parameters:    %s', n_parameters)
            logging.info('  Observables:   %s', n_observables)
            logging.info('  MADEs:         %s', n_mades)
            logging.info('  Hidden layers: %s', n_made_hidden_layers)
            logging.info('  Units:         %s', n_made_units_per_layer)
            logging.info('  Activation:    %s', activation)
            logging.info('  Batch norm:    %s', batch_norm)

            # MAF
            self.maf = ConditionalMaskedAutoregressiveFlow(
                n_conditionals=n_parameters,
                n_inputs=n_observables,
                n_hiddens=tuple([n_made_units_per_layer] * n_made_hidden_layers),
                n_mades=n_mades,
                activation=activation,
                batch_norm=batch_norm,
                input_order='random',
                mode='sequential',
                alpha=0.1
            )

        else:
            self.maf = torch.load(filename + '.pt', map_location='cpu')

            logging.info('Loaded NDE (MAF) from file:')
            logging.info('  Filename:      %s', filename)
            logging.info('  Parameters:    %s', self.maf.n_conditionals)
            logging.info('  Observables:   %s', self.maf.n_inputs)
            logging.info('  MADEs:         %s', self.maf.n_mades)
            logging.info('  Hidden layers: %s', self.maf.n_hiddens)
            logging.info('  Activation:    %s', self.maf.activation)
            logging.info('  Batch norm:    %s', self.maf.batch_norm)

        # Have everything on CPU (unless training)
        self.device = torch.device("cpu")
        self.dtype = torch.float

    def requires_class_label(self):
        return False

    def requires_joint_ratio(self):
        return False

    def requires_joint_score(self):
        return False

    def fit(self,
            theta=None,
            x=None,
            y=None,
            r_xz=None,
            t_xz=None,
            theta1=None,
            batch_size=64,
            trainer='adam',
            initial_learning_rate=0.001,
            final_learning_rate=0.0001,
            n_epochs=50,
            validation_split=0.2,
            early_stopping=True,
            alpha=None,
            learning_curve_folder=None,
            learning_curve_filename=None,
            **params):
        """ Trains MAF """

        logging.info('Training NDE (MAF) with settings:')
        logging.info('  theta given:    %s', theta is not None)
        logging.info('  theta1 given:   %s', theta1 is not None)
        logging.info('  x given:        %s', x is not None)
        logging.info('  y given:        %s', y is not None)
        logging.info('  r_xz given:     %s', r_xz is not None)
        logging.info('  t_xz given:     %s', t_xz is not None)
        logging.info('  Samples:        %s', x.shape[0])
        logging.info('  Parameters:     %s', theta.shape[1])
        logging.info('  Obserables:     %s', x.shape[1])
        logging.info('  Batch size:     %s', batch_size)
        logging.info('  Optimizer:      %s', trainer)
        logging.info('  Learning rate:  %s initially, decaying to %s', initial_learning_rate, final_learning_rate)
        logging.info('  Valid. split:   %s', validation_split)
        logging.info('  Early stopping: %s', early_stopping)
        logging.info('  Epochs:         %s', n_epochs)

        train_model(
            model=self.maf,
            loss_functions=[negative_log_likelihood],
            thetas=theta,
            xs=x,
            ys=None,
            batch_size=batch_size,
            trainer=trainer,
            initial_learning_rate=initial_learning_rate,
            final_learning_rate=final_learning_rate,
            n_epochs=n_epochs,
            validation_split=validation_split,
            early_stopping=early_stopping,
            learning_curve_folder=learning_curve_folder,
            learning_curve_filename=learning_curve_filename
        )

    def save(self, filename):
        # Fix a bug in pyTorch, see https://github.com/pytorch/text/issues/350
        self.maf.to()

        # self.maf.to_args = None
        # self.maf.to_kwargs = None
        # for made in self.maf.mades:
        #     made.to_args = None
        #     made.to_kwargs = None

        torch.save(self.maf, filename + '.pt')

        self.maf.to(self.device, self.dtype)

    def predict_density(self, theta, x, log=False):
        # If just one theta given, broadcast to number of samples
        theta = expand_array_2d(theta, x.shape[0])

        self.maf = self.maf.to(self.device, self.dtype)
        theta_tensor = tensor(theta).to(self.device, self.dtype)
        x_tensor = tensor(x).to(self.device, self.dtype)

        _, log_likelihood = self.maf.log_likelihood(theta_tensor, x_tensor)
        log_likelihood = log_likelihood.detach().numpy()

        if log:
            return log_likelihood
        return np.exp(log_likelihood)

    def predict_ratio(self, theta0, theta1, x, log=False):
        # If just one theta given, broadcast to number of samples
        theta0 = expand_array_2d(theta0, x.shape[0])
        theta1 = expand_array_2d(theta1, x.shape[0])

        self.maf = self.maf.to(self.device, self.dtype)
        theta0_tensor = tensor(theta0).to(self.device, self.dtype)
        theta1_tensor = tensor(theta1).to(self.device, self.dtype)
        x_tensor = tensor(x).to(self.device, self.dtype)

        _, log_likelihood_theta0 = self.maf.log_likelihood(theta0_tensor, x_tensor)
        _, log_likelihood_theta1 = self.maf.log_likelihood(theta1_tensor, x_tensor)

        log_likelihood_theta0 = log_likelihood_theta0.detach().numpy()
        log_likelihood_theta1 = log_likelihood_theta1.detach().numpy()

        if log:
            return log_likelihood_theta0 - log_likelihood_theta1
        return np.exp(log_likelihood_theta0 - log_likelihood_theta1)

    def predict_score(self, theta, x):
        # If just one theta given, broadcast to number of samples
        theta = expand_array_2d(theta, x.shape[0])

        self.maf = self.maf.to(self.device, self.dtype)
        theta_tensor = tensor(theta).to(self.device, self.dtype)
        x_tensor = tensor(x).to(self.device, self.dtype)

        _, _, score = self.maf.log_likelihood_and_score(theta_tensor, x_tensor)

        score = score.detach().numpy()

        return score

    def generate_samples(self, theta):
        self.maf = self.maf.to(self.device, self.dtype)
        theta_tensor = tensor(theta).to(self.device, self.dtype)

        samples = self.maf.generate_samples(theta_tensor).detach().numpy()
        return samples
