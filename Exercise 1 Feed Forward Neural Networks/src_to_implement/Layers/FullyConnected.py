import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size

        rng = np.random.default_rng()
        self.weights = rng.random((input_size + 1, output_size))        # uniform distribution, input_size + 1 for bias allowance

        self.input = np.zeros(input_size)

        self._optimizer = None

        self._gradient_weights = None


    @property
    def optimizer(self):
        """Getter method for the optimizer property."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """Setter method for the optimizer property."""
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        """Getter method for the gradient weights property."""
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        """Setter method for the gradient weights property."""
        self._gradient_weights = gradient_weights



    def forward(self, input_tensor):
        """
        Yhat is the prediction of the model, or the input to the next layer of the network.
        :param input_tensor: is assumed to already be given in transposed form (Matrix X).
        :return: the linear operation X * W = Yhat.
        """
        # Take first dimension size of input_tensor as batch_size
        batch_size = input_tensor.shape[0]

        # Extend input_tensor with "1" entries to accommodate bias and allow for matrix multiplication
        input_tensor = np.hstack((input_tensor, np.ones((batch_size, 1))))
        self.input = input_tensor

        return input_tensor @ self.weights       # X^T @ W^T = Yhat^T


    def backward(self, error_tensor):
        """
        Calculates gradients w.r.t the input for backpropagation and the weights for gradient descent.
        If optimizer is set, updates the weights.
        :param error_tensor: from the preceding layer.
        :return: a tensor that serves as the error_tensor for the previous layer.
        """
        optimizer = self.optimizer

        gradient_input = error_tensor @ self.weights.T      # Gradient of L w.r.t. the input: for backpropagation into deeper layers
        gradient_input = gradient_input[:, :-1]             # Remove fake errors that were introduced by adding the bias column in the input
        gradient_weights = self.input.T @ error_tensor      # Gradient of L w.r.t. the weights: for weight updates in this layer
        self.gradient_weights = gradient_weights

        if optimizer is not None:
            self.weights = optimizer.calculate_update(self.weights, gradient_weights)   # update weights

        return gradient_input




# Test if inheritance works:
# fc = FullyConnected(0, 0)
# print(fc.trainable)       # should print 'False' if not overridden in the FC constructor
# print(fc.weights)