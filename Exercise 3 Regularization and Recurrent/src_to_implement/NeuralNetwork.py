import numpy as np
import copy


class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = np.array([])        # contains the loss value for each iteration after calling train(iterations)
        self.layers = []                # holds the network architecture
        self.data_layer = None          # provides input data and labels, set from outside
        self.loss_layer = None          # reference to the special layer providing loss and prediction, last layer of the network, set from outside


    def forward(self):
        """
        Passes input from the data_layer through all layers of the network for a single learning iteration.
        :return: The output of the last layer, the loss layer, of the network. The loss is a scalar value.
        """
        input_tensor, label_tensor = self.data_layer.next()
        prediction = input_tensor
        for layer in self.layers:
            prediction = layer.forward(prediction)

        return self.loss_layer.forward(prediction, label_tensor), label_tensor


    def backward(self, label_tensor):
        """
        Performs one backpropagation and updating pass of the network for a single learning iteration.
        :param label_tensor: Returned from self.forward(), passed from self.train(iterations)
        """
        error_tensor = self.loss_layer.backward(label_tensor)
        for layer in self.layers[::-1]:                 # iterate through layers backwards
            error_tensor = layer.backward(error_tensor)
        #return error_tensor                            # output of the network backward function not important


    def append_layer(self, layer):
        """
        Adds a layer to the network architecture.
        If the layer is trainable, a deepcopy of the network's optimizer object is set for the given layer.
        :param layer: A neural network layer that extends BaseLayer, and is either trainable or not.
        """
        if layer.trainable:
            optimizer = copy.deepcopy(self.optimizer)
            layer.optimizer = optimizer
            layer.initialize(self.weights_initializer, self.bias_initializer)

        self.layers.append(layer)


    def train(self, iterations):
        """
        Trains the network architecture using the given number of iterations.
        Stores the loss for each iteration in self.loss.
        self.loss will always hold the loss values of the latest call to train(), since it is reset at the beginning of training.
        :param iterations: The number of iterations to train.
        """
        self.loss = np.zeros(iterations)        # reset the loss, allocate enough memory to store each iteration's loss value

        for i in range(iterations):
            loss, label_tensor = self.forward()
            self.loss[i] = loss
            self.backward(label_tensor)         # label_tensor comes from the forward function. This way I know that it matches the updates in backward


    def test(self, input_tensor):
        """
        Propagates the input tensor through the network and returns the prediction of the last layer.
        :param input_tensor: The data for which the prediction should be tested.
        :return: The prediction of the last layer.
        """
        prediction = input_tensor
        for layer in self.layers:
            prediction = layer.forward(prediction)

        return prediction

