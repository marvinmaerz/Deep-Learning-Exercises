import numpy as np
import copy
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid


class RNN(BaseLayer):
    """
    Elman Recurrent Neural Network cell.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc_layer = FullyConnected(input_size + hidden_size, hidden_size)   # hidden state: computes u_t = xtilde * Wh = fc_layer.forward(xtilde)
        self.fc_layer_output = FullyConnected(hidden_size, output_size)         # output: computes o_t = W_hy * h_t + b_y = fc_layer_output.forward(h_t)
        self.tanh_layer = TanH()
        self.sigmoid_layer = Sigmoid()

        self.hidden_state = np.zeros(self.hidden_size)
        self._memorize = False    # boolean state representing whether the RNN regards subsequent sequences as belonging to the same long sequence
        # self.hidden_state_memorized = None      # if memorize == True: stores the hidden state of the previous sequence       # todo: possibly remove

        # self._weights = None          # todo: check possible removal
        self.input_tensor = None  # saves input tensor for gradient calculation in the fc layer
        self._gradient_weights = None
        self._optimizer = None


    @property
    def memorize(self):
        """Getter for boolean memorize"""
        return self._memorize

    @memorize.setter
    def memorize(self, value:bool):
        """Setter for boolean memorize"""
        self._memorize = value

    @property
    def gradient_weights(self):
        """Getter for gradient weights"""
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        """Setter for gradient weights"""
        self._gradient_weights = value

    @property
    def weights(self):
        """Getter for FC layer weights (hidden state computation)."""
        return self.fc_layer.weights

    @weights.setter
    def weights(self, value):
        """Setter for weights"""
        if hasattr(self, "fc_layer"):
            self.fc_layer.weights = value
        self._weights = value

    @property
    def optimizer(self):
        """Getter for optimizer"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        """Setter for optimizer"""
        self._optimizer = value
        # also set the optimizer in the fc layer that computes new hidden state to update corresponding weights:
        self.fc_layer.optimizer = copy.deepcopy(value)
        self.fc_layer_output.optimizer = copy.deepcopy(value)


    def initialize(self, weights_initializer, bias_initializer):
        self.fc_layer.initialize(weights_initializer, bias_initializer)     # for the hidden state computation
        self.fc_layer_output.initialize(copy.deepcopy(weights_initializer), copy.deepcopy(bias_initializer))    # for the output computation


    def forward(self, input_tensor):
        # print("Input tensor shape: ", input_tensor.shape)
        # print("Hidden state shape: ", self.hidden_size)
        # print("Output tensor shape: ", self.output_size)

        timesteps = input_tensor.shape[0]
        self.input_tensor = input_tensor

        output_tensor = np.zeros((timesteps, self.output_size))

        if not self.memorize: self.hidden_state = np.zeros(self.hidden_size)

        for t in range(timesteps):     # Loop over time dimension
            # h_t = tanh(xtilde_t * W_h)        # W_h is weight matrix of the FC layer that calculates the hidden state
            # y_t = sigmoid(h_t * W_hy + b_y)   # Can also be expressed as a separate FC layer! => fc_layer_output.forward(h_t)
            xtilde = np.concat((input_tensor[t], self.hidden_state), axis=0)
            xtilde = xtilde.reshape(1, -1)                          # add "dummy" batch dimension of 1 for correct fc forward behavior
            u = self.fc_layer.forward(xtilde)                       # u_t = xtilde * W_h
            hidden_state = self.tanh_layer.forward(u)               # h_t = tanh(u_t)
            o = self.fc_layer_output.forward(hidden_state)          # o_t = h_t * W_hy
            output_tensor[t] = self.sigmoid_layer.forward(o)[0]     # y_t = sigmoid(o_t)
            self.hidden_state = hidden_state[0]                     # remove dummy batch dim again (also for output, see in line above)

            # print("x_t: \n", input_tensor[t])
            # print("h_t: \n", self.hidden_state)
            # print("Concatenation of x_t and h_t-1: \n", xtilde, " with shape: ", xtilde.shape)
            # print("u shape: ", u.shape)

        return output_tensor



