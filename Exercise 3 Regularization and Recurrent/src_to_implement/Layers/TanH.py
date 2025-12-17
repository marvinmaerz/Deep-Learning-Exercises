import numpy as np
from Layers.Base import BaseLayer

class TanH(BaseLayer):
    """
    TanH (Tangens Hyperbolicus) activation function.
    """
    def __init__(self):
        super().__init__()
        self.activations = 0


    def forward(self, input_tensor):
        """
        Applies the TanH activation function.
        :returns: tanh(input_tensor)
        """
        self.activations = np.tanh(input_tensor)
        return self.activations


    def backward(self, error_tensor):
        return error_tensor * (1.0 - np.square(self.activations))