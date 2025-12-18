import numpy as np
import copy
from Layers.Base import BaseLayer
import Layers.Helpers as Helpers


class BatchNormalization(BaseLayer):
    """
    Batch normalization regularization technique.
    """

    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels

        self.weights = None         # aka gamma
        self.bias = None            # aka beta
        self.initialize()
        self._gradient_weights = None
        self._gradient_bias = None

        self.mean_batch = None          # mean of first batch used for training
        self.variance_batch = None      # variance of first batch used for training
        self.mean_test = None              # moving average of mean during testing
        self.variance_test = None          # moving average of variance during testing

        self._optimizer = None
        self.bias_optimizer = None

        self.eps = 1e-11

        self.input_tensor = None
        self.normalized_input = None       # store for gradient w.r.t. weights in backward

    @property
    def gradient_weights(self):
        """Getter method for the gradient weights property."""
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        """Setter method for the gradient weights property."""
        self._gradient_weights = gradient_weights

    @property
    def gradient_bias(self):
        """Getter method for the gradient bias property."""
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        """Setter method for the gradient bias property."""
        self._gradient_bias = gradient_bias

    @property
    def optimizer(self):
        """Getter method for the optimizer property."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """Setter method for the optimizer property."""
        self._optimizer = optimizer
        self.bias_optimizer = copy.deepcopy(optimizer)


    def initialize(self, weights_initializer=None, bias_initializer=None):
        """
        Ignores assigned initializers and assigns weights to ones and the bias to zeros.
        This way they don't have an impact when beginning training.
        """
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)


    def forward(self, input_tensor):
        """
        Performs two steps:\n
        1. Normalization of the input tensor w.r.t. mean and variance.
        2. Calculating the linear combination of weights, normalized input and bias.
        :param input_tensor: Tensor of shape [b, c, y, (x)],
        where b = batch size, c = number of channels, y = height, and x = width (not present if input is a vector).
        """
        if self.mean_test is None:
            self.mean_test = np.mean(input_tensor, axis=0)
            self.variance_test = np.var(input_tensor, axis=0)

        self.mean_batch = np.mean(input_tensor, axis=0)
        self.variance_batch = np.var(input_tensor, axis=0)
        self.input_tensor = input_tensor

        # 1. Normalize input:
        if self.testing_phase:
            self.normalized_input = (input_tensor - self.mean_test) / np.sqrt(self.variance_test + self.eps)

        else:   # Training phase
            # Online estimation of training set mean & variance (moving average)
            decay = 0.8
            self.mean_test = decay * self.mean_test + (1 - decay) * self.mean_batch
            self.variance_test = decay * self.variance_test + (1 - decay) * self.variance_batch
            self.normalized_input = (input_tensor - self.mean_batch) / np.sqrt(self.variance_batch + self.eps)

        # 2. Return linear combination:
        return self.weights * self.normalized_input + self.bias


    def backward(self, error_tensor):
        self.gradient_weights = np.sum(error_tensor * self.normalized_input, axis=0)        # sum over batch dim
        self.gradient_bias = np.sum(error_tensor, axis=0)
        gradient_input = Helpers.compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean_batch, self.variance_batch, self.eps)

        # Update weights if optimizer is set
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return gradient_input


