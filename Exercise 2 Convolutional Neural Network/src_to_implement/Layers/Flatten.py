import numpy as np
from Layers.Base import BaseLayer

class Flatten(BaseLayer):
    """
    Reshapes the multi-dimensional input to a one-dimensional feature vector.
    Usually connects a convolutional or pooling layer with a fully connected layer.
    """
    def __init__(self):
        super().__init__()
        self.shape = None


    def forward(self, input_tensor):
        """
        Reshapes the multi-dimensional input to a one-dimensional feature vector.
        :param input_tensor: A vector of shape (batch_size, width, height, channels)
        :return: A vector of shape (batch_size, widht * height * channels)
        """
        self.shape = input_tensor.shape
        return np.reshape(input_tensor, (self.shape[0], np.prod(self.shape[1:])))


    def backward(self, error_tensor):
        return np.reshape(error_tensor, self.shape)
