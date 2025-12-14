import numpy as np


class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.testing_phase = False          # False -> we are training; True -> we are testing
        self.weights = np.array([])


    def forward(self, input_tensor):
        pass


    def backward(self, error_tensor):
        pass
