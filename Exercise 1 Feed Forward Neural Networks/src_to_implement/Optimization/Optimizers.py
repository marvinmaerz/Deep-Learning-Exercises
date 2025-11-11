import numpy as np



# Basic Stochastic Gradient Descent (SGD) class.
# Updates weights based on the basic gradient update scheme.
class Sgd:
    def __init__(self, learning_rate:float):
        self.learning_rate = learning_rate

    # Basic sgd weight update rule.
    # Returns the updated weight tensor.
    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor