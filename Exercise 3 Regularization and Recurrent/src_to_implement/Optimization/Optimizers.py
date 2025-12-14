import numpy as np



class Sgd:
    """
    Basic Stochastic Gradient Descent (SGD) class.
    Updates weights based on the basic update scheme.
    """
    def __init__(self, learning_rate:float):
        self.learning_rate = learning_rate


    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Basic stochastic gradient descent weight update rule.
        :param weight_tensor: weight matrix of any layer.
        :param gradient_tensor: gradient of the loss function with respect to the weights.
        :return: the updated weight tensor.
        """
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum:
    """
    Stochastic Gradient Descent (SGD) weight updating scheme using momentum.
    """

    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None               # save the moment


    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Stochastic gradient descent weight update rule using momentum.
        :param weight_tensor: weight matrix of any layer.
        :param gradient_tensor: gradient of the loss function with respect to the weights.
        :return: the updated weight tensor.
        """
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)     # v must have the same shape as w

        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        return weight_tensor + self.v


class Adam:
    """
    Weight updates using the Adam algorithm.
    """

    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu            # aka beta_1
        self.rho = rho          # aka beta_2
        # Intermediate values:
        self.timestep = 0
        self.v = None           # 1st moment vector: mean
        self.r = None           # 2nd raw moment vector: uncentered variance
        self.eps = 1e-8


    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Weight updates using the Adam algorithm.
        See the paper "ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION" by Kingma & Ba (2015) for reference.
        :param weight_tensor: weight matrix of any layer.
        :param gradient_tensor: gradient of the loss function with respect to the weights.
        :return: the updated weight tensor.
        """
        # Initialization of moment vectors and values
        if self.timestep == 0:
            self.v = np.zeros_like(weight_tensor)
            self.r = np.zeros_like(weight_tensor)

        # Momentum calculation
        self.timestep += 1
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * np.square(gradient_tensor)

        # Bias correction for v and r:
        vhat = self.v / (1 - np.power(self.mu, self.timestep))
        rhat = self.r / (1 - np.power(self.rho, self.timestep))

        # Weight update:
        return weight_tensor - self.learning_rate * vhat / (np.sqrt(rhat) + self.eps)
