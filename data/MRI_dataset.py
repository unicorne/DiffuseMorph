from torch.utils.data import Dataset
import data.util_2D as Util
import os
import numpy as np
from skimage import io

class MRIDataset(Dataset):
    """
    Custom dataset for loading 2D grayscale medical images for the DiffuseMorph model.
    """
    def __init__(self, dataroot, split='test'):
        self.split = split
        self.imageNum = []

        self.datapath = os.path.join(dataroot, split)
        # Assuming moving and fixed images are named 'moving_i.png' and 'fixed_i.png'
        num_images = len([name for name in os.listdir(self.datapath) if name.startswith('moving_')])

        for i in range(num_images):
            self.imageNum.append([f'moving_{i}.png', f'fixed_{i}.png'])

        self.data_len = len(self.imageNum)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        fileInfo = self.imageNum[index]
        moving_img_name, fixed_img_name = fileInfo[0], fileInfo[1]
        moving_img_path = os.path.join(self.datapath, moving_img_name)
        fixed_img_path = os.path.join(self.datapath, fixed_img_name)

        # Load images as grayscale and add a channel dimension
        moving_img = io.imread(moving_img_path, as_gray=True).astype(float)
        fixed_img = io.imread(fixed_img_path, as_gray=True).astype(float)

        moving_img = moving_img / 255.0
        fixed_img = fixed_img / 255.0
        moving_img = moving_img[:, :, np.newaxis]
        fixed_img = fixed_img[:, :, np.newaxis]

        # Apply transformations
        [moving_img, fixed_img] = Util.transform_augment([moving_img, fixed_img], split=self.split, min_max=(-1, 1))

        return {'M': moving_img, 'F': fixed_img, "Index": index}