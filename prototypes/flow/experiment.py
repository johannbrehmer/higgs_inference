from __future__ import absolute_import, division, print_function
import logging
import os

class ExperimentHandler:
    def __init__(self, log_file, debug=False):
        """ Init. """

        # Set class variables
        self.debug = debug
        self.model = None
        self.base_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.data_dir = os.path.join(self.base_dir, "data/unweighted_events")
        self.model_dir = os.path.join(self.base_dir, "data/models_theano")
        self.result_dir = os.path.join(self.base_dir, "data/results")

        # Logger
        if self.debug:
            logging.basicConfig(level=log_level)
        else:
            logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.log_filename = os.path.join(self.log_dir, log_file)
        handler = logging.FileHandler(self.log_filename)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # First logging messages
        self.logger.info('Good morning!')
        self.logger.debug('Base dir: %s', self.base_dir)
        self.logger.debug('Log dir: %s', self.log_dir)
        self.logger.debug('Data dir: %s', self.data_dir)
        self.logger.debug('Model dir: %s', self.model_dir)
        self.logger.debug('Result dir: %s', self.result_dir)

    def train(self, architecture, parameters):
        """ Trains model. """

        self.architecture = architecture
        self.parameters = parameters

        raise NotImplementedError

    def save_model(self, name=None):
        """
        Save trained model to file. If the name is not explicitly given, construct it from the
        architecture and parameters.
        """

        if self.model is None:
            self.logger.error("No model trained or loaded")
            raise ValueError("No model trained or loaded")

        raise NotImplementedError

    def load_model(self, name=None, architecture=None, parameters=None):
        """
        Load trained model from file. If the name is not explicitly given, construct it from the
        architecture and parameters.
        """

        raise NotImplementedError

    def evaluate(self, name):
        """ Predict (log) densities on evaluation sample and save in data/results. """

        raise NotImplementedError
