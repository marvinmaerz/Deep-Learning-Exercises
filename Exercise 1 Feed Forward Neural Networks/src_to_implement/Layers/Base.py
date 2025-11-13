import numpy as np


class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.weights = np.array([])


    def forward(self, input_tensor):
        pass


    def backward(self, error_tensor):
        pass
