import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size




# Test if inheritance works:
# fc = FullyConnected(0, 0)
# print(fc.trainable)       # should print 'False' if not overridden in the FC constructor