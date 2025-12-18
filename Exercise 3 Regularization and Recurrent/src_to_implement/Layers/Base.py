import numpy as np


class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.testing_phase = False          # False -> we are training; True -> we are testing
        self.weights = np.array([])


    def forward(self, input_tensor):
        """
        Calculates the forward pass of the layer.
        :param input_tensor: Tensor on which to calculate the forward pass.
        :return: Output tensor that serves input tensor for the next layer.
        """
        pass


    def backward(self, error_tensor):
        """
        Backward propagation of gradients to lower layers.
        :param error_tensor: Gradients of higher layer. L'(y).
        :return: Error tensor for the lower layers. L'(x) = L'(y) * y'(x).
        """
        pass
