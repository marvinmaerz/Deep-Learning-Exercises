import numpy as np

class Constant:
    """
    Initializer class for constant weights and biases.
    Value can be set, with the default being 0.1.
    """
    def __init__(self, value = 0.1):
        self.value = value


    def initialize(self, weights_shape, fan_in=-1, fan_out=-1):
        return np.full(weights_shape, self.value)


class UniformRandom:
    """
    Initializer class for uniform random weights and biases in the half-open interval [0, 1).
    """
    def initialize(self, weights_shape, fan_in=-1, fan_out=-1):
        return np.random.default_rng().uniform(0, 1, weights_shape)


class Xavier:
    """
    Initializer class for Xavier weights and biases.
    Typically used for Sigmoid activations.
    """

    def initialize(self, weights_shape, fan_in, fan_out):
        mean = 0
        std = np.sqrt(2 / (fan_in + fan_out))
        return np.random.default_rng().normal(mean, std, weights_shape)


class He:
    """
    Initializer class for He weights and biases.
    Typically used for ReLU activations.
    """

    def initialize(self, weights_shape, fan_in, fan_out=-1):
        mean = 0
        std = np.sqrt(2 / fan_in)
        return np.random.default_rng().normal(mean, std, weights_shape)