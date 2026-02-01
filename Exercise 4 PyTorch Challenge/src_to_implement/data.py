from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode: str):
        """
        Arguments:
            data (pandas.dataframe): Holds the data found in "data.csv"
            mode (string): Flag with possible values "val" or "train"
        """
        self.data = data
        self.mode = mode

        # Perform data augmentation on training set, normalize
        self.train_transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),

            # Data augmentation transformations:
            tv.transforms.RandomResizedCrop(300),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomVerticalFlip(),
            tv.transforms.RandomRotation(10),
            tv.transforms.ColorJitter(brightness=0.1, contrast=0.1),

            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)
        ])

        # On validation set only normalize
        self.validation_transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)
        ])

        self.dirname = Path(__file__).parent.resolve()      # should be something like WindowsPath("C:/.../repository/<exercise folder>/src_to_implement")

        # Loading images into memory (only ~70 MB), hoping for faster read times at the cost of higher initial overhead.
        # Toggle via boolean below.
        # Results: test_normalization runtime using self.images in memory: ~6,5 seconds; with on demand loading: ~10,5 seconds
        # => ~40% faster access times
        self.preload_images = True
        if self.preload_images:
            self.images : dict[str, np.ndarray] = {}        # Example key-value pair: { "images/cell0001.png": [1 (cracked), 0 (inactive)] }
            for i in range(data.shape[0]):
                img_path = self.data["filename"][i]
                img_path_full = self.dirname.joinpath(img_path)
                img = imread(img_path_full)
                self.images[img_path] = img


    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, index):
        img_path, cracked, inactive = self.data.iloc[index]
        if self.preload_images:
            img = self.images[img_path]                                 # faster reading of pre-loaded images in working memory
        else:
            img_path_full = self.dirname.joinpath(img_path)
            img = imread(img_path_full)         # on demand reading from external memory (may be slower)
        img_rgb = gray2rgb(img)
        label = torch.tensor([cracked, inactive], dtype=torch.float32)
        if self.mode == "train":
            return self.train_transform(img_rgb), label
        elif self.mode == "val":
            return self.validation_transform(img_rgb), label
        else: return img_rgb, label

