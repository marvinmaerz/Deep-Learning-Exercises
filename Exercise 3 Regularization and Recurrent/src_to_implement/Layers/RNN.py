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
        self.input_hidden = None        # this[t] is fc_layer.input_tensor at time step t, saved for gradient calculation in the fc layer
        self.input_output = None        # stored input tensors at all time steps of fc_layer_output, saved for gradient calculation
        self.tanh_activations = None
        self.sigmoid_activations = None

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
        # self._weights = value

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


    def calculate_regularization_loss(self):
        """
        Returns the regularization loss for the optimizers/regularizers (if given) of the two embedded FC layers.
        Sums both regularization losses.
        """
        reg_loss = 0
        if self.fc_layer.optimizer and self.fc_layer.optimizer.regularizer:
            reg_loss += self.fc_layer.optimizer.regularizer.norm(self.fc_layer.weights)
        if self.fc_layer_output.optimizer and self.fc_layer_output.optimizer.regularizer:
            reg_loss += self.fc_layer_output.optimizer.regularizer.norm(self.fc_layer_output.weights)
        return reg_loss


    def forward(self, input_tensor):
        # print("Input tensor shape: ", input_tensor.shape)
        # print("Hidden state shape: ", self.hidden_size)
        # print("Output tensor shape: ", self.output_size)

        timesteps = input_tensor.shape[0]

        output_tensor = np.zeros((timesteps, self.output_size))

        if not self.memorize: self.hidden_state = np.zeros(self.hidden_size)
        if self.input_hidden is None:
            self.input_hidden = np.zeros((timesteps, 1, self.input_size + self.hidden_size + 1))
            self.input_output = np.zeros((timesteps, 1, self.hidden_size + 1))
            self.tanh_activations = np.zeros((timesteps, self.hidden_size))
            self.sigmoid_activations = np.zeros((timesteps, self.output_size))

        for t in range(timesteps):     # Loop over samples in forward-time ("then to now")
            # Formulae: --------------
            # h_t = tanh(xtilde_t * W_h)        # W_h is weight matrix of the FC layer that calculates the hidden state
            # y_t = sigmoid(h_t * W_hy + b_y)   # Can also be expressed as a separate FC layer! => fc_layer_output.forward(h_t)
            # -------------------------
            xtilde = np.concat((input_tensor[t], self.hidden_state), axis=0)
            xtilde = xtilde.reshape(1, -1)                          # add "dummy" batch dimension of 1 for correct FC layer forward behavior
            u = self.fc_layer.forward(xtilde)                       # u_t = xtilde * W_h
            hidden_state = self.tanh_layer.forward(u)               # h_t = tanh(u_t)
            o = self.fc_layer_output.forward(hidden_state)          # o_t = h_t * W_hy
            output_tensor[t] = self.sigmoid_layer.forward(o)[0]     # y_t = sigmoid(o_t) & removing dummy batch dim again
            self.hidden_state = hidden_state[0]                     # remove dummy batch dim again
            # store inputs at time t for backprop through time:
            self.input_hidden[t] = self.fc_layer.input
            self.input_output[t] = self.fc_layer_output.input
            self.tanh_activations[t] = self.tanh_layer.activations
            self.sigmoid_activations[t] = self.sigmoid_layer.activations

            # print("x_t: \n", input_tensor[t])
            # print("h_t: \n", self.hidden_state)
            # print("Concatenation of x_t and h_t-1: \n", xtilde, " with shape: ", xtilde.shape)
            # print("u shape: ", u.shape)

        return output_tensor


    def backward(self, error_tensor):
        timesteps = error_tensor.shape[0]
        gradient_input = np.zeros((timesteps, self.input_size))
        gradient_hidden = np.zeros(self.hidden_size)            # Incoming hidden state gradient of previous timestep

        # For gradient weights:
        opt_hidden = self.fc_layer.optimizer
        opt_output = self.fc_layer_output.optimizer
        self.fc_layer.optimizer = None
        self.fc_layer_output.optimizer = None
        gradW_hidden = 0
        gradW_output = 0

        for t in range(timesteps - 1, -1, -1):       # go from t = T..0 (backwards)
            # Restore activations & inputs from time t (saved during forward):
            self.sigmoid_layer.activations = self.sigmoid_activations[t]
            self.tanh_layer.activations = self.tanh_activations[t]
            self.fc_layer_output.input = self.input_output[t]
            self.fc_layer.input = self.input_hidden[t]

            # Go through layers in reverse for backward:
            # First: Sigmoid
            sig_back = self.sigmoid_layer.backward(error_tensor[t]) # Nabla o_t
            sig_back = sig_back.reshape(1, -1)                      # add dummy batch dim for compatability with fc layers (2D)

            # Second: FC Output
            fc_out_back = self.fc_layer_output.backward(sig_back)   # error tensor y_t (orange box in slide 9)
            gradW_output += self.fc_layer_output.gradient_weights   # Nabla W_ht,y
            dh = fc_out_back + gradient_hidden                      # BP through time (adding the gradient from last time step because copying h_t in forward)

            # Third: TanH
            tanh_back = self.tanh_layer.backward(dh)                # Nabla u_t

            # Fourth: FC Hidden
            fc_hidden_back = self.fc_layer.backward(tanh_back)              # dxtilde = [dx_t ... dh_t] -> split into gradient_input and gradient_hidden
            gradient_input[t] = fc_hidden_back[0, 0: self.input_size]       # gradient_input[t] = dx_t
            gradient_hidden = fc_hidden_back[0, self.input_size:]           # gradient_hidden = dh_t ("through time") (blue box in slide 9)
            gradW_hidden += self.fc_layer.gradient_weights                  # dyhat_t/dW_h

        self.gradient_weights = gradW_hidden
        # Update weights in fc layers with accumulated gradient weights:
        if self.optimizer:
            self.fc_layer.optimizer = opt_hidden
            self.fc_layer_output.optimizer = opt_output
            self.fc_layer.weights = self.fc_layer.optimizer.calculate_update(self.fc_layer.weights, gradW_hidden)
            self.fc_layer_output.weights = self.fc_layer_output.optimizer.calculate_update(self.fc_layer_output.weights, gradW_output)

        return gradient_input




