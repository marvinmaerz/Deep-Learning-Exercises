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
        self.isConv = False
        self.conv_shape = None      # needed for reshaping

        self.weights = None         # aka gamma
        self.bias = None            # aka beta
        self.initialize()
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None
        self.bias_optimizer = None

        self.mean_batch = None          # mean of first batch used for training
        self.variance_batch = None      # variance of first batch used for training
        self.mean_test = None              # moving average of mean during testing
        self.variance_test = None          # moving average of variance during testing
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
        :param input_tensor: "Normal" case: tensor of shape [b, c],
        where b = batch size, c = number of image channels.
        Convolutional case: tensor of shape [b, h, m, n],
        where h = number of convolutional channels, m = height, and n = width.
        """
        # Reformat input tensor in convolutional case:
        self.isConv = len(input_tensor.shape) == 4
        self.input_tensor = input_tensor
        if self.isConv:
            # Reformat input tensor & calculate mean and variance from conv channels h
            self.input_tensor = self.reformat(input_tensor)

        if self.mean_test is None:
            self.mean_test = np.mean(self.input_tensor, axis=0)
            self.variance_test = np.var(self.input_tensor, axis=0)

        self.mean_batch = np.mean(self.input_tensor, axis=0)
        self.variance_batch = np.var(self.input_tensor, axis=0)


        # 1. Normalize input:
        if self.testing_phase:
            # Using fixed estimation of training set mean and variance
            self.normalized_input = (self.input_tensor - self.mean_test) / np.sqrt(self.variance_test + self.eps)

        else:
            # Training phase
            # Online estimation of training set mean & variance (moving average)
            decay = 0.8
            self.mean_test = decay * self.mean_test + (1 - decay) * self.mean_batch
            self.variance_test = decay * self.variance_test + (1 - decay) * self.variance_batch
            self.normalized_input = (self.input_tensor - self.mean_batch) / np.sqrt(self.variance_batch + self.eps)

        # 2. Return linear combination:
        output = self.weights * self.normalized_input + self.bias
        if self.isConv: output = self.reformat(output)
        return output


    def backward(self, error_tensor):
        # Reformat input tensor in convolutional case:
        error = error_tensor
        if self.isConv:
            error = self.reformat(error_tensor)
            
        self.gradient_weights = np.sum(error * self.normalized_input, axis=0)        # sum over batch dim
        self.gradient_bias = np.sum(error, axis=0)
        gradient_input = Helpers.compute_bn_gradients(error, self.input_tensor, self.weights, self.mean_batch, self.variance_batch, self.eps)

        # Update weights if optimizer is set
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        if self.isConv: gradient_input = self.reformat(gradient_input)
        return gradient_input


    def reformat(self, tensor):
        """
        Reshapes the tensor either from image-like into a vector (from 4 -> 2 dimensions),
        or inversely, from a vector to image-like (2 -> 4 dimensions).\n
        Dimension transformation: image-like (shape [b, h, m, n]) to vector (shape [b * m * n, h]) and vice versa.
        :param tensor: Tensor to be reshaped. Either image-like or vector.
        :return: Reshaped tensor.
        """
        is_img = len(tensor.shape) == 4
        if is_img:  # image -> vector
            b, h, m, n = tensor.shape
            self.conv_shape = tensor.shape  # store original shape
            reshaped = tensor.reshape((b, h, m * n)).transpose((0, 2, 1))   # now [b, m * n, h]
            reshaped = reshaped.reshape((b * m * n, h))

        else: # vector -> image
            b, h, m, n = self.conv_shape    # get original shape
            reshaped = tensor.reshape((b, m * n, h)).transpose((0, 2, 1))   # now [b, h, m * n]
            reshaped = reshaped.reshape((b, h, m, n))

        return reshaped




