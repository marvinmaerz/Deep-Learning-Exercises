import numpy as np
import matplotlib.pyplot as plt
from numpy.conftest import dtype


class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.zeros((resolution, resolution))  # numpy array that stores the checkerboard


    def draw(self):
        assert(self.resolution % (2 * self.tile_size) == 0)
        runs = int(self.resolution / (self.tile_size * 2))
        black = np.zeros((self.tile_size,), dtype="b")
        white = np.ones((self.tile_size,), dtype="b")
        bw = np.concatenate((black, white))                     # length = tile_size * 2
        wb = np.concatenate((white, black))                     # length = tile_size * 2
        line_b = np.tile(bw, runs)                              # line that starts with black
        line_w = np.tile(wb, runs)                              # line that starts with white
        block_b = np.tile(line_b, self.tile_size)               # block that starts with black
        block_w = np.tile(line_w, self.tile_size)               # block that starts with white
        block_together = np.concatenate((block_b, block_w))
        self.output = np.tile(block_together, runs).reshape((self.resolution, self.resolution))
        return np.tile(block_together, runs).reshape((self.resolution, self.resolution))







    def show(self):
        pass
        plt.imshow(self.output, cmap='gray')  # gray_r: 1=black, 0=white
        plt.show()




class Circle:
    def __init__(self):
        return

    def draw(self):
        return

    def show(self):
        return