import numpy as np
import matplotlib.pyplot as plt
from numpy.conftest import dtype


class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.zeros((resolution, resolution), dtype="b")  # numpy array that stores the checkerboard


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
        plt.imshow(self.output, cmap='gray')  # gray: 0=black, 1=white
        plt.show()




class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = np.zeros((resolution, resolution), dtype="b")


    def draw(self):
        x = np.linspace(0, self.resolution, self.resolution)
        y = np.linspace(0, self.resolution, self.resolution)
        xx, yy = np.meshgrid(x, y)              # do meshgrid here to get xx's and yy's (grayscale gradients visually)
        self.output = self.circle_func(xx, yy)  # apply the circle condition to each of them (boolean return type)
        return self.circle_func(xx, yy)


    def show(self):
        plt.imshow(self.output, cmap='gray')  # gray: 0=black, 1=white
        plt.show()


    # Returns true, iff a given point lies inside the radius of a circle using the circle equation (pythagoras)
    def circle_func(self, x, y):
        return (x - self.position[0])**2 + (y - self.position[1])**2 <= self.radius ** 2



class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.zeros((resolution, resolution, 3))


    def draw(self):
        red = np.linspace(0, 1.0, self.resolution)
        green = np.linspace(0, 1.0, self.resolution)
        blue = np.linspace(1.0, 0.0, self.resolution)
        # Set colors. Each color has its own channel in the color dimension (3rd dimension, after the coordinates, which are left unchanged)
        self.output[:,:,0] = red                                    # red: left (0.0) -> right (1.0)
        self.output[:,:,1] = green.reshape((self.resolution,1))     # green: top (0.0) -> bottom (1.0)
        self.output[:,:,2] = blue                                   # blue: right (0.0) -> left (1.0)
        return np.copy(self.output)


    def show(self):
        plt.imshow(self.output)
        plt.show()