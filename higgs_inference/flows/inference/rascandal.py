import logging

from higgs_inference.flows.inference.nde import MAFInference
from higgs_inference.flows.ml.trainer import train_model
from higgs_inference.flows.ml.losses import negative_log_likelihood, score_mse, ratio_mse


class RASCANDALInference(MAFInference):
    """ Neural conditional density estimation with masked autoregressive flows. """

    def requires_joint_score(self):
        return True

    def requires_class_label(self):
        return True

    def requires_joint_ratio(self):
        return True

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
            alpha=0.01,
            beta=0.01,
            learning_curve_folder=None,
            learning_curve_filename=None,
            **params):
        """ Trains RASCANDAL """

        logging.info('Training RASCANDAL (MAF + score + ratio) with settings:')
        logging.info('  alpha (score):  %s', alpha)
        logging.info('  beta (ratio):   %s', beta)
        logging.info('  theta given:    %s', theta is not None)
        logging.info('  theta1 given:   %s', theta1 is not None)
        logging.info('  x given:        %s', x is not None)
        logging.info('  y given:        %s', y is not None)
        logging.info('  r_xz given:     %s', r_xz is not None)
        logging.info('  t_xz given:     %s', t_xz is not None)
        logging.info('  theta1 given:   %s', theta1 is not None)
        logging.info('  Samples:        %s', x.shape[0])
        logging.info('  Parameters:     %s', theta.shape[1])
        logging.info('  Obserables:     %s', x.shape[1])
        logging.info('  Batch size:     %s', batch_size)
        logging.info('  Optimizer:      %s', trainer)
        logging.info('  Learning rate:  %s initially, decaying to %s', initial_learning_rate, final_learning_rate)
        logging.info('  Valid. split:   %s', validation_split)
        logging.info('  Early stopping: %s', early_stopping)
        logging.info('  Epochs:         %s', n_epochs)

        assert theta is not None
        assert x is not None
        assert y is not None
        assert r_xz is not None
        assert t_xz is not None
        assert theta1 is not None

        train_model(
            model=self.maf,
            loss_functions=[negative_log_likelihood, score_mse, ratio_mse],
            loss_weights=[1., alpha, beta],
            loss_labels=['nll', 'score', 'ratio'],
            thetas=theta,
            theta1=theta1,
            xs=x,
            ys=y,
            t_xzs=t_xz,
            r_xzs=r_xz,
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
