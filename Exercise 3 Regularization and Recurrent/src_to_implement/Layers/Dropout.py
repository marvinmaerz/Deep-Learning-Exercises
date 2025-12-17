import numpy as np
import copy
from Layers.Base import BaseLayer


class Dropout(BaseLayer):
    """Inverted dropout regularization technique."""

    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.indices = None                     # store indices of activations that were set to zero in forward for backward


    def forward(self, input_tensor):
        """
        Applies inverted dropout to the input tensor in the training phase.
        :param input_tensor: Activations from previous layer.
        :return: Activations multiplied by 1/probability and randomly set to zero with (1-probability).
        """
        # print("Input tensor shape: ", input_tensor.shape)
        # print("Input tensor element 0: \n", input_tensor[0])
        if self.testing_phase: return input_tensor

        output_tensor = copy.deepcopy(input_tensor)
        mask = np.random.rand(*input_tensor.shape) > (1 - self.probability)      # True in mask => keep activation, False => set to zero
        self.indices = np.where(mask == False)

        output_tensor[self.indices] = 0
        output_tensor *= (1 / self.probability)  # inverted dropout

        # print("Mask: \n", mask[0], " shape: ", mask.shape)
        # print("Indices:\n", self.indices, " shape: ", len(self.indices[0]))
        # print("Input tensor element 0 (after dropout): \n", output_tensor[0])
        return output_tensor


    def backward(self, error_tensor):
        """
        Applies inverted dropout to the error tensor,
        in the same way it was done for the activations in forward.
        :param error_tensor: Error tensor from higher layer.
        :return: Error multiplied by 1/probability and randomly set to zero with (1-probability).
        """
        if self.testing_phase: return error_tensor          # Technically not needed, since backpropagation (e.g. training) never happens on the test set

        output_tensor = copy.deepcopy(error_tensor)
        output_tensor[self.indices] = 0
        output_tensor *= (1 / self.probability)

        # print("Error tensor element 0: \n", error_tensor[0])
        # print("1/p = ", 1.0/self.probability)
        # print("Error tensor after dropout:\n", output_tensor[0])
        return output_tensor