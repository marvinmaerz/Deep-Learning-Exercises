import numpy as np
from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    """
    The "Rectified Linear Unit" (ReLU) activation function layer.
    """

    def __init__(self):
        super().__init__()
        self.input_tensor = None        # store for usage in the backward step


    def forward(self, input_tensor):
        """ReLU(x) = max(0, x).
        :return: a tensor that serves as the  input_tensor for the next layer, with the ReLU activation function applied."""
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)


    def backward(self, error_tensor):
        """
        :return: 0, iff the input tensor was zero or negative, else passes the error tensor to the next layer unchanged.
        """
        return np.where(self.input_tensor <= 0, 0, error_tensor)