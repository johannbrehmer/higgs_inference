class Inference:
    """ Base class for inference methods. """

    def __init__(self, **params):
        pass

    def requires_class_label(self):
        raise NotImplementedError()

    def requires_joint_ratio(self):
        raise NotImplementedError()

    def requires_joint_score(self):
        raise NotImplementedError()

    def fit(self, theta=None, x=None, y=None, r_xz=None, t_xz=None,  theta1=None,
            batch_size=64, initial_learning_rate=0.001, final_learning_rate=0.0001, n_epochs=50,
            early_stopping=True, **params):
        raise NotImplementedError()

    def save(self, filename):
        raise NotImplementedError()

    def predict_density(self, theta, x):
        raise NotImplementedError()

    def predict_ratio(self, theta0, theta1, x):
        raise NotImplementedError()

    def predict_score(self, theta, x):
        raise NotImplementedError()

    def generate_samples(self, theta):
        raise NotImplementedError()
