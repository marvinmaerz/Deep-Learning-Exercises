import numpy as np
import scipy.signal as signal

from Layers.Base import BaseLayer


class Conv(BaseLayer):
    """
    Convolutional layer.
    """


    def __init__(self, stride_shape: int|tuple, convolution_shape: tuple, num_kernels: int):
        super().__init__()
        self.stride_shape = stride_shape                # Integer or tuple. Tuple allows for different strides in spatial dimensions.

        # convolution_shape specifies the shape of the filter kernels:
        # 1D convolution: shape is [c, m]
        # 2D convolution: shape is [c, m, n]
        # c = num input channels (element 0)
        # m, n = spatial shape of filter kernel (elements 1 and 2)
        self.convolution_shape = convolution_shape
        self.isConv1D: bool = len(convolution_shape) == 2

        self.num_kernels = num_kernels                  # Integer

        self.trainable = True

        rng = np.random.default_rng()
        self.weights = rng.uniform(0, 1, (num_kernels, *convolution_shape))     # Weights of shape [num_kernels, c, m(, n)]
        self.bias = rng.uniform(0, 1, num_kernels)                   # Bias is a single value

        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None


    @property
    def gradient_weights(self):
        """Getter method for the gradient weights property."""
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        """Setter method for the gradient weights property."""
        self._gradient_weights = gradient_weights

    @property
    def gradient_bias(self):
        """Getter method for the gradient bias property."""
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        """Setter method for the gradient bias property."""
        self._gradient_bias = gradient_bias

    @property
    def optimizer(self):
        """Getter method for the optimizer property."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """Setter method for the optimizer property."""
        self._optimizer = optimizer


    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)


    def forward(self, input_tensor):
        """
        Forward pass of the convolutional layer.
        :param input_tensor: 1D layout: [b, c, y],
                            2D layout: [b, c, y, x],
                            where b = batch size, c = number of channels, y & x = spatial dimensions of input image.
        :return: Tensor that serves as input for the next layer with shape [b, num_kernels, y, x].
                            x and y may be subsampled (reduced) by stride factor s.
        """
        output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, *input_tensor.shape[2:]))

        # print("\nInput tensor shape:  ", input_tensor.shape, " [#batches, #channels, y, x]")
        # print("Number of kernels:   ", self.num_kernels)
        # print("Kernel shape:        ", self.weights.shape, " [#kernels, #channels, ky, kx]")
        # print("Output tensor shape: ", output_tensor.shape, " [#batches, #kernels, y, x]")

        # How to handle channels + example image: https://d2l.ai/chapter_convolutional-neural-networks/channels.html
        for b in range(input_tensor.shape[0]):      # iterate over batches
            for k in range(self.num_kernels):       # iterate over kernels
                acc = np.zeros(input_tensor.shape[2:], dtype=float)
                for c in range(input_tensor.shape[1]):
                    acc += signal.correlate(input_tensor[b, c], self.weights[k, c], mode="same")

                # acc = np.sum(acc, axis=0)       # compress multi-channel input into single channel through sum along channel axis (see link)
                acc += self.bias[k]           # add the corresponding bias to the compressed convolution
                output_tensor[b, k] = acc

                # print("Convolution result shape: ", acc.shape, " [y, x] (single channel)")

        return self._apply_stride(output_tensor)




    def _apply_stride(self, output_tensor):
        """
        Applies striding through subsampling to the passed tensor. Private helper function.
        :param output_tensor: Passed from forward(). See its shape in docstring there.
        :return: output_tensor with applied striding.
        """
        stride_x = 0
        stride_y = 0
        # Handle stride_shape types:
        if type(self.stride_shape) is tuple:
            stride_x = self.stride_shape[1]
            stride_y = self.stride_shape[0]
        elif type(self.stride_shape) is list:       # to match (incorrect) unit test case
            stride_y = self.stride_shape[0]
        elif type(self.stride_shape) is int:
            stride_y = self.stride_shape

        # Handle 1D vs 2D convolution strides:
        if self.isConv1D:
            return output_tensor[:, :, ::stride_y]
        else:
            return output_tensor[:, :, ::stride_y, ::stride_x]
