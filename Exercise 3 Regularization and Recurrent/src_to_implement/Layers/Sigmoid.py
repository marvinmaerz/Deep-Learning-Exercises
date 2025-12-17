import numpy as np
from Layers.Base import BaseLayer

class Sigmoid(BaseLayer):
    """
    Sigmoid activation function.
    """
    def __init__(self):
        super().__init__()
        self.activations = 0


    def forward(self, input_tensor):
        """
        :param input_tensor: Input on which to apply the activation function.
        :return: Activations.
        """
        self.activations = 1.0 / (1.0 + np.exp(-input_tensor))
        return self.activations


    def backward(self, error_tensor):
        """
        :param error_tensor: Error to backpropagate after applying the derivative of the activation function.
        :return: Error tensor for lower layers.
        """
        return error_tensor * self.activations * (1.0 - self.activations)
