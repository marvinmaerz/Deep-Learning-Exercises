import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    """
    The SoftMax activation function layer.
    Transforms logits into a probability distribution.
    Typically used for classification tasks.
    """

    def __init__(self):
        super().__init__()
        self.output = None          # keep the softmax output for the backward pass, gradient calculation


    def forward(self, input_tensor):
        """
        Converts the logits input into a probability distribution.
        Each row of the batch is normalized to a probability distribution, and sums to 1.
        :param input_tensor: Logits tensor.
        :return: Normalized SoftMax probability distribution.
        """
        x_shifted = input_tensor - np.max(input_tensor)
        self.output = np.divide(np.exp(x_shifted), np.sum(np.exp(x_shifted), axis=1, keepdims=True))
        # Trick: np.sum(..., axis=1, keepdims=True)!
        # axis=1 let the sum compute row-wise
        # keepdims=True kept the shape compatible with the division

        # DEBUG statements:
        # print(f"x_shifted:\n {x_shifted}")
        # print(f"np.exp(x_shifted):\n {np.exp(x_shifted)}")      # this operation is alright!
        # print(f"np.sum(x_shifted):\n {np.sum(np.exp(x_shifted), axis=1, keepdims=True)}")
        # print("output shape: ", self.output.shape, "output: ")
        # print(self.output)
        # print("Rows should sum to 1.0. Actual values:\n ", np.sum(self.output, axis=1, keepdims=True))

        return self.output


    def backward(self, error_tensor):
        """
        Calculates the gradient (error_tensor) with respect to the input.
        :param error_tensor: The error from the succeeding layer.
        :return: The error_tensor for the previous layers.
        """
        gradient_input = self.output * (error_tensor - np.sum(error_tensor * self.output, axis=1, keepdims=True))
        return gradient_input