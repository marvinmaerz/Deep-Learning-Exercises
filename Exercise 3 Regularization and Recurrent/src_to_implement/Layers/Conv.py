import numpy as np
import scipy.signal as signal
import copy

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
        self.channels = convolution_shape[0]
        self.isConv1D: bool = len(convolution_shape) == 2
        # Even kernel sizes => need asymmetric padding to achieve correct dimensions in backward pass
        self.even_ky: bool = convolution_shape[1] % 2 == 0
        self.even_kx: bool = convolution_shape[2] % 2 == 0 if not self.isConv1D else False

        self.num_kernels = num_kernels

        self.trainable = True

        rng = np.random.default_rng()
        self.weights = rng.uniform(0, 1, (num_kernels, *convolution_shape))     # Weights of shape [num_kernels, c, m(, n)]
        self.bias = rng.uniform(0, 1, num_kernels)                   # Bias is a single value
        self.input_tensor = None        # shape [b, c, y, x], where b is the number of images in a batch

        self._gradient_weights = np.empty((num_kernels, *convolution_shape))
        self._gradient_bias = np.empty(num_kernels)

        self._optimizer = None              # Optimizer for the weights
        self._bias_optimizer = None         # When self.optimizer is set, make two separate copies: one for weights, one for bias


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
        self._bias_optimizer = copy.deepcopy(optimizer)


    def initialize(self, weights_initializer, bias_initializer):
        """"Reinitializes the weights by using the provided initializer objects."""
        fan_in = np.prod(self.convolution_shape)
        fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)


    def calculate_regularization_loss(self):
        """Calculates the regularization loss for this layer if an optimizer/regularizer has been set."""
        reg_loss = 0
        if self.optimizer and self.optimizer.regularizer:
            reg_loss += self.optimizer.regularizer.norm(self.weights)
        return reg_loss



    def forward(self, input_tensor):
        """
        Forward pass of the convolutional layer.
        :param input_tensor: 1D layout: [b, c, y],
                            2D layout: [b, c, y, x],
                            where b = batch size, c = number of channels, y & x = spatial dimensions of input image.
        :return: Tensor that serves as input for the next layer with shape [b, num_kernels, y, x].
                            x and y may be subsampled (reduced) by stride factor s.
        """
        self.input_tensor = input_tensor
        output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, *input_tensor.shape[2:]))

        # print("\nInput tensor shape:  ", input_tensor.shape, " [#batches, #channels, y, x]")
        # print("Number of kernels:   ", self.num_kernels)
        # print("Kernel shape:        ", self.weights.shape, " [#kernels, #channels, ky, kx]")
        # print("Output tensor shape: ", output_tensor.shape, " [#batches, #kernels, y, x]")

        # How to handle channels + example image: https://d2l.ai/chapter_convolutional-neural-networks/channels.html
        for b in range(input_tensor.shape[0]):                          # iterate over batches
            for k in range(self.num_kernels):                           # iterate over kernels
                acc = np.zeros(input_tensor.shape[2:], dtype=float)
                for c in range(input_tensor.shape[1]):                  # iterate over channels for correct valid correlation
                    acc += signal.correlate(input_tensor[b, c], self.weights[k, c], mode="same")      # valid correlation: add 2D correlations over channels up
                acc += self.bias[k]                                     # add the corresponding bias to the compressed convolution
                output_tensor[b, k] = acc                               # could do it in-situ, but cleaner to read & understand this way

                # print("Correlation result shape: ", acc.shape, " [y, x] (single channel)")
        return self._apply_stride(output_tensor)


    def backward(self, error_tensor):
        """
        Backward pass of the convolutional layer.
        Updates the parameters using the optimizer (if set) and returns the error tensor
        with respect to the input to the previous layer.
        :param error_tensor: Of previous layer with shape [b, num_kernels, y, x].
        :return: Tensor that serves as error tensor for the next layer with shape [b, c, y, x],
                where b = batch size, c = number of input channels, y & x = spatial dimensions of input image.
        """
        batches = error_tensor.shape[0]

        if error_tensor.shape[2:] != self.input_tensor.shape[2:]:          # Upsample if stride was applied -> reverse it
            error_upsampled = np.zeros((batches, self.num_kernels, *self.input_tensor.shape[2:]))
            for b in range(batches):
                for k in range(self.num_kernels):
                    # error_upsampled[b, k] = skimage.transform.resize(error_tensor[b,k], self.input_tensor.shape[2:])
                    error_upsampled[b, k] = self._reverse_stride(error_tensor[b, k], self.input_tensor.shape[2:])
            error_tensor = error_upsampled

        # print("\nInput tensor shape: ", self.input_tensor.shape, " [b, c, y, x]")
        # print("Error tensor shape: ", error_tensor.shape, f"[b, num_kernels={self.num_kernels}, y, x]")
        # print("Gradient weights shape: ", self.gradient_weights.shape, f" [num_kernels={self.num_kernels}, c, n, m]")
        # print("Error tensor: \n", error_tensor[0,0])
        # print("Input tensor: \n", self.input_tensor[0,0])

        # Compute gradients w.r.t. the weights (kernels)
        input_padded = self._pad_tensor(self.input_tensor)
        self.gradient_weights = np.zeros((self.num_kernels, *self.convolution_shape))

        # print("Padded input tensor:\n", input_padded, " -> shape ", input_padded.shape)

        for b in range(error_tensor.shape[0]):                  # Loop over images in batch
            for h in range(self.num_kernels):              # Loop over channels of error tensor
                for s in range(self.input_tensor.shape[1]):     # Loop over channels of input tensor
                    self.gradient_weights[h, s] += signal.correlate(input_padded[b, s], error_tensor[b, h], mode="valid")

        self.gradient_bias = np.zeros(self.num_kernels)
        if self.isConv1D:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))
        else:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))          # resulting shape: (num_kernels, ) => biases for every kernel

        # Compute gradients w.r.t. the input (for lower layers)
        if self.isConv1D:
            kernels_new = np.transpose(self.weights, (1,0,2))
        else:
            kernels_new = np.transpose(self.weights, (1, 0, 2, 3))      # shape [c, num_kernels, n, m] instead of [num_kernels, c, n, m]
        gradient_input = np.empty((batches, *self.input_tensor.shape[1:]))            # shape [c, y, x]

        # print("Gradient input shape: ", gradient_input.shape)

        for b in range(batches):                    # Loop over images in batch
            for c in range(self.channels):          # Loop over channels of input tensor
                acc = np.zeros(error_tensor.shape[2:], dtype=float)
                for h in range(self.num_kernels):   # Loop and sum over channels of error tensor => valid across error channels
                    acc += signal.convolve(error_tensor[b, h], kernels_new[c, h], mode="same")      # Same conv along spatial dims
                gradient_input[b, c] = acc

        # Update weights if optimizer is set
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return gradient_input


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


    def _reverse_stride(self, error_tensor, shape):
        """
        Reverses the stride that was applied to the output tensor in the forward pass.
        Does so without interpolating.
        :param error_tensor: Tensor to be upsampled.
        :param shape: Shape to upsample to (y, (x)).
        :return: Upsampled tensor.
        """
        stride_x = 0
        stride_y = 0
        # Handle stride_shape types:
        if type(self.stride_shape) is tuple:
            stride_x = self.stride_shape[1]
            stride_y = self.stride_shape[0]
        elif type(self.stride_shape) is list:  # to match (incorrect) unit test case
            stride_y = self.stride_shape[0]
        elif type(self.stride_shape) is int:
            stride_y = self.stride_shape

        upsampled = np.zeros(shape)
        if self.isConv1D:
            upsampled[::stride_y] = error_tensor
        else:
            upsampled[::stride_y, ::stride_x] = error_tensor

        return upsampled


    def _pad_tensor(self, tensor):
        """
        Applies zero padding to the tensor, equaling to half the kernel size on each side.
        If kernel dimensions are even, applies asymmetric padding.
        Private helper function.
        :param tensor: Tensor to be padded, of shape [b, c, y, (x)].
        :return: Padded tensor, with accordingly increased shape.
        """
        y, x = 0, 0
        batches = tensor.shape[0]
        channels = tensor.shape[1]
        if self.isConv1D:
            y = tensor.shape[2]
        else:
            y, x = tensor.shape[2:]

        # Compute pad_width for all axes: ((before_y, after_y), (before_x, after_x))
        pad_width = np.floor(np.divide(self.convolution_shape[1:], 2)).astype(int)  # example pad width for kernel with spatial dims (3, 8): (1, 4)
        pad_width = np.repeat(pad_width, 2)
        # Asymmetric padding if kernel sizes are even
        if self.even_ky:
            pad_width[1] -= 1
        if not self.isConv1D:
            if self.even_kx:
                pad_width[3] -= 1

        # Allocate memory for padded tensor
        if self.isConv1D:
            padded_tensor = np.zeros((batches, channels, y + pad_width[0] + pad_width[1]))
        else:
            padded_tensor = np.zeros((batches, channels, y + pad_width[0] + pad_width[1],
                                      x + pad_width[2] + pad_width[3]))

        # Pad tensor
        for b in range(batches):
            for c in range(channels):
                if self.isConv1D:
                    pad = np.pad(tensor[b, c], (pad_width[0], pad_width[1]))
                else:
                    pad = np.pad(tensor[b, c], ((pad_width[0], pad_width[1]), (pad_width[2], pad_width[3])))
                padded_tensor[b, c] = pad

        return padded_tensor
