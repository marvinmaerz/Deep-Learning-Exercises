import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    """
    Max-Pooling layer.
    """

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape        # specifies how much the receptive field (the pooling kernels) is moved, shape [sy, sx]
        self.pooling_shape = pooling_shape      # specifies the size of the pooling receptive field, shape [py, px]

        self.indices = None                     # shape [b, c, out_y, out_x], where (out_y, out_x) is the output shape of the forward pass

        self.input_shape = None

        self.shape = None
        self.strides = None


    def forward(self, input_tensor):
        """
        Computes the output of the pooling layer.
        :param input_tensor: Tensor to which max pooling is applied.
        :return: Tensor with max pooling applied.
        """
        batches, channels, y, x = input_tensor.shape
        pool_y, pool_x = self.pooling_shape
        stride_y, stride_x = self.stride_shape
        self.input_shape = input_tensor.shape

        out_size = np.floor(([y, x] - np.array(self.pooling_shape)) / np.array(self.stride_shape)) + 1          # Valid pooling, so no padding to consider
        out_size = out_size.astype(int)
        output_tensor = np.zeros((batches, channels, *out_size))

        self.shape = (batches, channels, *out_size, pool_y, pool_x)
        self.strides = (*input_tensor.strides[0:2], stride_y * input_tensor.strides[2], stride_x * input_tensor.strides[3], *input_tensor.strides[2:4])
        slices = np.lib.stride_tricks.as_strided(input_tensor, shape=self.shape, strides=self.strides, writeable=False)  # returns slices (as views) of given pooling shape and stride from the input images
        # ^- Slices is of shape [b, c, out_y, out_x, pool_y, pool_x] => coordinate [out_y, out_x] specifies the slice we're in,
        # and [pool_y, pool_x] specifies the coordinates inside of each slice.
        # Example: slices[0,0,0,1,1,1] (with (2,2) pooling and stride) specifies the lower right corner of the second slice along the x-axis of the output

        output_tensor = np.max(slices, axis=(-2, -1))       # Maximize along the last axes of the slices, i.e. over axes=[slice_y, slice_x]
        self.indices = slices.reshape(batches, channels, *out_size, -1).argmax(axis=4)      # Flattened indices where each element of the window was largest, each index belongs to a slice

        # indices = np.unravel_index(indices, (2,2))
        # print("Input tensor:\n", input_tensor)
        # print("Slices: \n", slices, f" -> shape {slices.shape} [b, c, y, x, slice_y, slice_x], where each slice_y, slice_x represents a view of size (pool_y, pool_x) from image [b, c, y, x].")
        # print("Indices of maxima: ", self.indices)
        # print("Maximum index of second slice: ", np.unravel_index(indices[0,0,0,1], (2,2)))
        # print("Indices shape: ", self.indices.shape, " [b, c, out_y, out_x] => ever value determines where in the original window that computed the value [out_y, out_x] the maximum was.")
        # print("Output tensor:\n", output_tensor)

        return output_tensor


    def backward(self, error_tensor):
        """
        Computes the gradient of the pooling layer, given that the forward pass was max-pooling.
        :param error_tensor: Error tensor of previous layer.
        :return: Error tensor for lower layers.
        """
        batches, channels, out_y, out_x = error_tensor.shape

        output_tensor = np.zeros(self.input_shape)
        slices = np.lib.stride_tricks.as_strided(output_tensor, shape=self.shape, strides=self.strides, writeable=True) # Create slices in the same way it's done in the forward pass. Difference: writeable=True!
        # ^- Slices is of shape [b, c, out_y, out_x, pool_y, pool_x] => coordinate [out_y, out_x] specifies the slice we're in,
        # and [pool_y, pool_x] specifies the coordinates inside of each slice.
        # Example: slices[0,0,0,1,1,1] (with (2,2) pooling and stride) specifies the lower right corner of the second slice along the x-axis of the output
        # slices[0,0,0,1,1,1] = 100     # <= Setting a value in a slice also sets it in the output from which it was sliced

        # print(f"Slices of shape [{slices.shape}]: \n", slices)
        # print(f"Error tensor of shape [{error_tensor.shape}]: \n", error_tensor)
        # print(f"Saved index for slice[0,0,0,0]: ", np.unravel_index(self.indices[0,0,0,0], self.pooling_shape))

        for b in range(batches):
            for c in range(channels):
                for oy in range(out_y):
                    for ox in range(out_x):     # Same indexing for slices, indices and error_tensor
                        err = error_tensor[b, c, oy, ox]
                        idx_y, idx_x = np.unravel_index(self.indices[b,c,oy,ox], self.pooling_shape)
                        slices[b, c, oy, ox, idx_y, idx_x] += err

        # print(f"Output tensor of shape [{output_tensor.shape}]:\n", output_tensor)
        return output_tensor