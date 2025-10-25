import os.path
import json
import scipy.misc
import skimage
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
        self.indices = np.arange(len(self.images))          # shuffle changes this array, since it dictates accesses to images and labels
        if self.shuffle: self.shuffle_indices()


    # This function creates a batch of images and corresponding labels and returns them.
    # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
    # Note that your amount of total data might not be divisible without remainder with the batch_size.
    # Think about how to handle such cases
    def next(self):
        images = np.empty((self.batch_size, self.image_size[0], self.image_size[1], 3))
        labels = np.empty(self.batch_size, dtype=int)

        if self.current_batch == self.batches_per_epoch:         # increase epoch
            self.epoch += 1
            self.current_batch = 0
            if self.shuffle: self.shuffle_indices()

        # Resizing images
        resize = self.image_size != self.images[0].shape      # resize = True, iff image_size not equal to the test data size
        if resize:
            for i in range(len(self.images)):
                self.images[i] = skimage.transform.resize(self.images[i], self.image_size)

        # get indices to access for this batch. If shuffle=False, this iterates over the images in-order, else accesses given indices for each batch (random order)
        batch_indices = self.indices[self.batch_size * self.current_batch: self.batch_size * (self.current_batch + 1)]

        # fill in batch_indices, if the last batch in the epoch was not filled completely => take images from the front
        remainder = self.batch_size - len(batch_indices)
        for i in range(remainder):
            batch_indices = np.append(batch_indices, i)

        i = 0   # fill images and labels from 0 -- batch_size, but access self.images via idx from batch_indices
        for idx in batch_indices:
            images[i] = self.augment(np.copy(self.images[idx]))
            labels[i] = self.labels[self.image_names[idx]]
            i += 1

        self.current_batch += 1

        return images, labels


    # this function takes a single image as an input and performs a random transformation
    # (mirroring and/or rotation) on it and outputs the transformed image
    def augment(self,img):
        if not self.rotation and not self.mirroring: return img
        rng = np.random.default_rng()
        augment = rng.integers(0, 1, endpoint=True)         # to augment or not to augment, that is the question
        if not augment: return img

        img_aug = None
        if self.rotation:
            degree = rng.choice([90, 180, 270])
            img_aug = skimage.transform.rotate(img, degree)

        if self.mirroring:
            direction = rng.integers(0, 1, endpoint=True)
            img_aug = np.fliplr(img) if direction == 0 else np.flipud(img)

        return img_aug


    # shuffles indices for the next epoch (helper method)
    def shuffle_indices(self):
        rng = np.random.default_rng()
        rng.shuffle(self.indices)


    def current_epoch(self):
        return self.epoch


    # This function returns the class name for a specific input
    def class_name(self, x):
        return self.class_dict[x]


    # In order to verify that the generator creates batches as required, this functions calls next to get a
    # batch of images and labels and visualizes it.
    def show(self):
        images, labels = self.next()
        plt.figure(figsize=(10, 10))        # tip: use height 20 (second value) for big batches (60+ images), and 10 for around 20-30 images
        for i in range(len(images)):
            plt.subplot(int(len(images)/5),5,i+1)
            plt.axis('off')
            plt.title(self.class_name(labels[i]))
            plt.imshow(images[i])
        plt.tight_layout()
        plt.show()

