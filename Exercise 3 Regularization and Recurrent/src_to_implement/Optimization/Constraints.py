import numpy as np


class L2_Regularizer:
    """Enforces small weights."""
    def __init__(self, alpha):
        self.alpha = alpha      # The regularization weight


    def calculate_gradient(self, weights):
        """
        Calculates the (sub-)gradient on the weights needed for the optimizer.
        """
        return weights * self.alpha


    def norm(self, weights):
        """
        Calculates the norm enhanced loss.
        """
        return np.sum(np.square(weights) * self.alpha)


class L1_Regularizer:
    """Enforces sparse weights."""
    def __init__(self, alpha):
        self.alpha = alpha      # The regularization weight


    def calculate_gradient(self, weights):
        """
        Calculates the (sub-)gradient on the weights needed for the optimizer.
        """
        return np.sign(weights) * self.alpha


    def norm(self, weights):
        """
        Calculates the norm enhanced loss.
        """
        return np.sum(np.abs(weights) * self.alpha)