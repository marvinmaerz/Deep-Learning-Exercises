import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.images = []                                            # list of np arrays (training images)
        self.image_names = []                                       # corresponding file names of each array => to match with the label (matched indices)
        self.labels = json.loads(open(label_path, 'r').read())      # dict of labels (not in the same order as images and image_names, but that's not important here)
        files = os.listdir(file_path)
        for i in range(len(files)):
            self.images.append(np.load(file_path + files[i]))
            self.image_names.append(files[i].removesuffix(".npy"))

        # => images[i]: training data; image_names[i]: file_name for images[i]; labels[i]: label for file_name[i]
        # => Index i: same image, file name and label

        self.batch_size = batch_size
        self.image_size = image_size

        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        self.epoch = 0
        self.current_batch = 0
        self.batches_per_epoch = len(self.images) / self.batch_size
        self.indices = np.arange(len(self.images))          # shuffle changes this array # TODO: if shuffle=True, change this array


    # This function creates a batch of images and corresponding labels and returns them.
    # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
    # Note that your amount of total data might not be divisible without remainder with the batch_size.
    # Think about how to handle such cases
    def next(self):
        #TODO: add resize boolean option
        if self.current_batch == self.batches_per_epoch:         # increase epoch
            self.epoch += 1
            self.current_batch = 0
            # TODO: implement shuffling after an epoch

        images = []
        labels = []
        # get indices to access for this batch. If shuffle=False, this iterates over the images in-order
        batch_indices = self.indices[self.batch_size * self.current_batch : self.batch_size * (self.current_batch + 1)]

        # fill in batch_indices, if the last batch in the epoch was not filled completely => take images from the front
        remainder = self.batch_size - len(batch_indices)
        for i in range(remainder):
            batch_indices = np.append(batch_indices, i)

        for idx in batch_indices:
            images.append(np.copy(self.images[idx]))
            labels.append(self.labels[self.image_names[idx]])

        self.current_batch += 1

        return images, labels


    # this function takes a single image as an input and performs a random transformation
    # (mirroring and/or rotation) on it and outputs the transformed image
    def augment(self,img):
        #TODO: implement augmentation function: mirroring & rotating

        return img


    def current_epoch(self):
        return self.epoch


    # This function returns the class name for a specific input
    def class_name(self, x):
        return self.class_dict[x]


    # In order to verify that the generator creates batches as required, this functions calls next to get a
    # batch of images and labels and visualizes it.
    def show(self):
        images, labels = self.next()
        plt.figure(figsize=(10, 10))        # tip: use height 20 for big batches (60+ images), and 10 for around 20-30 images
        for i in range(len(images)):
            plt.subplot(int(len(images)/5),5,i+1)
            plt.axis('off')
            plt.title(self.class_name(labels[i]))
            plt.imshow(images[i])
        plt.tight_layout()
        plt.show()

