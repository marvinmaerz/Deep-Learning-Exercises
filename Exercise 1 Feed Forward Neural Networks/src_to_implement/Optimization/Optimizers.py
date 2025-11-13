import numpy as np



class Sgd:
    """
    Basic Stochastic Gradient Descent (SGD) class.
    Updates weights based on the basic update scheme.
    """
    def __init__(self, learning_rate:float):
        self.learning_rate = learning_rate


    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Basic stochastic gradient descent weight update rule.
        :param weight_tensor: weight matrix of any layer.
        :param gradient_tensor: gradient of the loss function with respect to the weights.
        :return: the updated weight tensor.
        """
        return weight_tensor - self.learning_rate * gradient_tensor