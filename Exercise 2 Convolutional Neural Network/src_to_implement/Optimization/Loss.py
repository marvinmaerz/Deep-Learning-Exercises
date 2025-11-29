import numpy as np


class CrossEntropyLoss:
    """
    Cross entropy loss function, which is often used in classification tasks,
    typically in conjunction with the SoftMax (or Sigmoid) activation function.
    Not considered a layer.
    """

    def __init__(self):
        # store for the backward pass (gradient calculation)
        # don't store the loss value!
        self.prediction = None


    def forward(self, prediction_tensor, label_tensor):
        """
        Computes the cross entropy loss value accumulated over the batch.
        :param prediction_tensor: Typically from SoftMax or Sigmoid.
        :param label_tensor: The ground truth labels.
        :return: scalar loss value.
        """
        self.prediction = prediction_tensor
        eps = np.finfo(label_tensor.dtype).eps
        loss = np.sum(np.where(label_tensor == 1, -np.log(prediction_tensor + eps), 0))
        return loss


    def backward(self, label_tensor):
        """
        Computes the error_tensor for the previous layer.
        Backpropagation starts here.
        :param label_tensor: The ground truth labels.
        :return: The error_tensor which is back-propagated to the previous layer.
        """
        eps = np.finfo(label_tensor.dtype).eps
        error_tensor = - label_tensor / (self.prediction + eps)
        return error_tensor